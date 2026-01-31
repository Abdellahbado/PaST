from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from PaST.config import DataConfig
from PaST.data.sm_benchmark_data import RawInstance, generate_raw_instance
from PaST.solvers.alns_parallel import (
    ALNSAction,
    ALNSConfig,
    FullEval,
    Solution,
    _compute_epsilon,
    alns_apply_action,
    alns_state_features,
    build_initial_solution_cwp_spt,
    default_action_list,
)


def _ensure_scale_choices(cfg: DataConfig, scale: str) -> None:
    scale = str(scale).lower()
    if scale == "small":
        cfg.T_max_choices = [t for t in cfg.T_max_choices if int(t) <= 80]
    elif scale in {"mls", "medium"}:
        cfg.T_max_choices = [t for t in cfg.T_max_choices if 80 < int(t) <= 300]
    elif scale in {"vls", "large"}:
        cfg.T_max_choices = [t for t in cfg.T_max_choices if int(t) > 300]
    else:
        raise ValueError("scale must be one of: small, medium/mls, vls/large")


@dataclass
class StepInfo:
    action: ALNSAction
    accepted: bool
    feasible_cand: bool
    delta_energy: float
    cur_energy: float
    best_energy: float
    no_improve: int
    tau: float


class PMALNSEnv:
    """CPU environment: PPO controls ALNS destroy/repair selection (stage A1).

    - State: `alns_state_features` (currently 10 dims)
    - Action: discrete index into `default_action_list()`
    - Acceptance: simulated annealing (fixed rule for now; paper later adds it as action)
    - Reward: energy improvement if candidate is accepted, else 0; infeasible/fail penalized

    This environment is designed to be multiprocessing-friendly (pure Python + numpy).
    """

    def __init__(
        self,
        *,
        scale: str,
        seed: int,
        instance_id: int,
        slack_ratio: float,
        alns_cfg: ALNSConfig,
        top_k_cwp: int = 80,
        fail_penalty: float = 1.0,
        infeasible_penalty: float = 1.0,
    ):
        self.scale = str(scale)
        self.seed = int(seed)
        self.instance_id = int(instance_id)
        self.slack_ratio = float(slack_ratio)
        self.alns_cfg = alns_cfg
        self.top_k_cwp = int(top_k_cwp)
        self.fail_penalty = float(fail_penalty)
        self.infeasible_penalty = float(infeasible_penalty)

        self.action_list: List[ALNSAction] = default_action_list()

        self._rng = random.Random(self.seed)
        self._raw: Optional[RawInstance] = None
        self._epsilon: int = 0

        self._it = 0
        self._no_improve = 0
        self._tau = float(alns_cfg.sa_tau0)

        self._cur_sol: Optional[Solution] = None
        self._cur_ev: Optional[FullEval] = None
        self._best_sol: Optional[Solution] = None
        self._best_ev: Optional[FullEval] = None

    @property
    def obs_dim(self) -> int:
        return 10

    @property
    def action_dim(self) -> int:
        return len(self.action_list)

    def reset(self) -> np.ndarray:
        cfg = DataConfig()
        _ensure_scale_choices(cfg, self.scale)

        gen_rng = random.Random(int(self.seed) * 1_000_003 + int(self.instance_id) * 97)
        self._raw = generate_raw_instance(
            cfg, gen_rng, instance_id=int(self.instance_id)
        )
        self._epsilon = _compute_epsilon(self._raw, float(self.slack_ratio))

        self._it = 0
        self._no_improve = 0
        self._tau = float(self.alns_cfg.sa_tau0)

        cur_sol, cur_ev = build_initial_solution_cwp_spt(
            self._raw, self._epsilon, top_k=int(self.top_k_cwp)
        )
        self._cur_sol, self._cur_ev = cur_sol, cur_ev
        self._best_sol, self._best_ev = cur_sol, cur_ev

        return self._obs()

    def get_state_summary(self) -> Dict[str, float]:
        """Return a compact snapshot of the current search state.

        Useful for long HPC runs where you want to verify progress from logs.
        """

        if self._raw is None or self._cur_ev is None or self._best_ev is None:
            raise RuntimeError("Environment not reset yet")

        return {
            "cur_energy": float(self._cur_ev.total_energy),
            "best_energy": float(self._best_ev.total_energy),
            "epsilon": float(self._epsilon),
            "it": float(self._it),
            "no_improve": float(self._no_improve),
            "tau": float(self._tau),
        }

    def _obs(self) -> np.ndarray:
        assert self._raw is not None
        assert self._cur_sol is not None and self._cur_ev is not None
        assert self._best_ev is not None
        feats = alns_state_features(
            self._raw,
            self._cur_sol,
            self._cur_ev,
            self._best_ev,
            int(self._epsilon),
            int(self._it),
            int(self._no_improve),
            float(self._tau),
            self.alns_cfg,
        )
        return np.asarray(feats, dtype=np.float32)

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        assert self._raw is not None
        assert self._cur_sol is not None and self._cur_ev is not None
        assert self._best_sol is not None and self._best_ev is not None

        action_idx = int(max(0, min(self.action_dim - 1, int(action_idx))))
        action = self.action_list[action_idx]

        prev_energy = float(self._cur_ev.total_energy)

        step_exception = False
        try:
            cand_sol, cand_ev, _, _ = alns_apply_action(
                self._raw,
                int(self._epsilon),
                self._cur_sol,
                self._cur_ev,
                action,
                self.alns_cfg,
                self._rng,
            )
        except Exception:
            # In long PPO runs we want robustness: treat unexpected solver errors as
            # a failed repair attempt rather than crashing the entire training job.
            cand_sol, cand_ev = None, None
            step_exception = True

        reward = 0.0
        accepted = False
        feasible_cand = bool(cand_ev is not None and cand_ev.feasible)

        if cand_sol is None or cand_ev is None:
            # Repair failed
            self._no_improve += 1
            reward = -float(self.fail_penalty)
        elif not feasible_cand:
            # Candidate violates epsilon / DP infeasible
            self._no_improve += 1
            reward = -float(self.infeasible_penalty)
        else:
            cand_energy = float(cand_ev.total_energy)
            delta = float(cand_energy - prev_energy)
            if delta <= 0:
                accepted = True
            else:
                if self._tau > 1e-12:
                    prob = math.exp(-delta / float(self._tau))
                else:
                    prob = 0.0
                accepted = self._rng.random() < prob

            if accepted:
                self._cur_sol, self._cur_ev = cand_sol, cand_ev
                reward = -(float(self._cur_ev.total_energy) - prev_energy)

            improved_best = float(cand_energy) < float(self._best_ev.total_energy)
            if improved_best:
                self._best_sol, self._best_ev = cand_sol, cand_ev
                self._no_improve = 0
            else:
                self._no_improve += 1

        self._it += 1
        self._tau *= float(self.alns_cfg.sa_decay)

        done = bool(
            self._it >= int(self.alns_cfg.max_iters)
            or self._no_improve >= int(self.alns_cfg.no_improve_limit)
        )

        info = StepInfo(
            action=action,
            accepted=bool(accepted),
            feasible_cand=bool(feasible_cand),
            delta_energy=(
                (float(cand_ev.total_energy) - float(prev_energy))
                if (cand_ev is not None and feasible_cand)
                else float("nan")
            ),
            cur_energy=float(self._cur_ev.total_energy),
            best_energy=float(self._best_ev.total_energy),
            no_improve=int(self._no_improve),
            tau=float(self._tau),
        )

        return (
            self._obs(),
            float(reward),
            done,
            {"step": info.__dict__, "exception": bool(step_exception)},
        )

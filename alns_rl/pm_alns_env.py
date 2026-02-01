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
    build_action_list,
    default_action_list,
)


def _build_action_list_by_name(action_set: str) -> List[ALNSAction]:
    """Select a discrete action set for PPO.

    Rationale:
        - `destroy_all` is very likely to produce infeasible candidates when epsilon is tight.
        - `repair_random` increases expressiveness but can be noisy early; we keep it in
          the default `balanced` set.
    """

    name = str(action_set or "").strip().lower()
    if name in {"", "balanced"}:
        return build_action_list(
            destroy_ops=["random", "worst_machine", "longest"],
            repair_ops=["greedy", "random"],
            k_mults=[0.5, 1.0],
        )
    if name in {"safe", "feasible"}:
        return build_action_list(
            destroy_ops=["random", "worst_machine", "longest"],
            repair_ops=["greedy"],
            k_mults=[0.25, 0.5, 1.0],
        )
    if name in {"full", "default"}:
        return default_action_list()
    raise ValueError("action_set must be one of: balanced, safe, full")


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
        action_set: str = "balanced",
        top_k_cwp: int = 80,
        fail_penalty: float = 1.0,
        infeasible_penalty: float = 1.0,
        reward_scale: float = 1.0,
        reward_power: float = 1.0,
        best_improve_bonus: float = 0.0,
        reward_best_coef: float = 1.0,
        reward_accept_coef: float = 0.25,
        reject_penalty: float = 0.0,
        reject_worse_penalty_coef: float = 0.0,
        reject_worse_penalty_power: float = 1.0,
        # Paper-inspired additions (PPO-ALNS): termination control + terminal reward.
        min_stop_iters: int = 5,
        terminal_best_coef: float = 0.0,
        terminal_time_coef: float = 0.0,
    ):
        self.scale = str(scale)
        self.seed = int(seed)
        self.instance_id = int(instance_id)
        self.slack_ratio = float(slack_ratio)
        self.alns_cfg = alns_cfg
        self.action_set = str(action_set)
        self.top_k_cwp = int(top_k_cwp)
        self.fail_penalty = float(fail_penalty)
        self.infeasible_penalty = float(infeasible_penalty)

        self.reward_scale = float(reward_scale)
        self.reward_power = float(reward_power)
        self.best_improve_bonus = float(best_improve_bonus)
        self.reward_best_coef = float(reward_best_coef)
        self.reward_accept_coef = float(reward_accept_coef)

        # Optional shaping to discourage proposing candidates that SA will reject.
        # Keep these default-off; over-penalizing rejection can make the policy overly conservative.
        self.reject_penalty = float(reject_penalty)
        self.reject_worse_penalty_coef = float(reject_worse_penalty_coef)
        self.reject_worse_penalty_power = float(reject_worse_penalty_power)

        self.min_stop_iters = int(min_stop_iters)
        self.terminal_best_coef = float(terminal_best_coef)
        self.terminal_time_coef = float(terminal_time_coef)

        # Base discrete action list (destroy, k_mult, repair) used by ALNS core.
        self.action_list: List[ALNSAction] = _build_action_list_by_name(self.action_set)

        # Paper-style factorized action space:
        #   A1: destroy+intensity (destroy_name, k_mult)
        #   A2: repair operator (repair_name)
        #   A3: acceptance criterion (0=SA, 1=greedy)
        #   A4: termination decision (0=continue, 1=stop)
        destroy_pairs: List[Tuple[str, float]] = []
        repair_ops: List[str] = []
        for dname, km, rname in self.action_list:
            pair = (str(dname), float(km))
            if pair not in destroy_pairs:
                destroy_pairs.append(pair)
            r = str(rname)
            if r not in repair_ops:
                repair_ops.append(r)
        self.destroy_choices: List[Tuple[str, float]] = destroy_pairs
        self.repair_choices: List[str] = repair_ops

        self._rng = random.Random(self.seed)
        self._raw: Optional[RawInstance] = None
        self._epsilon: int = 0

        self._it = 0
        self._no_improve = 0
        self._tau = float(alns_cfg.sa_tau0)

        self._init_best_energy: float = float("nan")
        self._destroy_counts: Optional[np.ndarray] = None
        self._repair_counts: Optional[np.ndarray] = None

        self._cur_sol: Optional[Solution] = None
        self._cur_ev: Optional[FullEval] = None
        self._best_sol: Optional[Solution] = None
        self._best_ev: Optional[FullEval] = None

    @property
    def obs_dim(self) -> int:
        # Base ALNS features (10) + operator usage frequencies.
        return 10 + int(len(self.destroy_choices)) + int(len(self.repair_choices))

    @property
    def action_dim(self) -> int:
        # Backward-compatible single-action dimension (not used when training with
        # factorized action heads). Keep as product for sanity/debugging.
        return int(len(self.destroy_choices)) * int(len(self.repair_choices)) * 2 * 2

    @property
    def action_dims(self) -> List[int]:
        return [
            int(len(self.destroy_choices)),
            int(len(self.repair_choices)),
            2,  # acceptance criterion
            2,  # stop/continue
        ]

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

        self._init_best_energy = float(cur_ev.total_energy)
        self._destroy_counts = np.zeros(
            (int(len(self.destroy_choices)),), dtype=np.float32
        )
        self._repair_counts = np.zeros(
            (int(len(self.repair_choices)),), dtype=np.float32
        )

        return self._obs()

    def get_state_summary(self) -> Dict[str, float]:
        """Return a compact snapshot of the current search state.

        Useful for long HPC runs where you want to verify progress from logs.
        """

        if self._raw is None or self._cur_ev is None or self._best_ev is None:
            raise RuntimeError("Environment not reset yet")

        eps = int(self._epsilon)
        cur_ms = int(self._cur_ev.makespan)
        best_ms = int(self._best_ev.makespan)
        slack = int(eps - cur_ms)
        slack_norm = float(slack) / float(max(1, eps))

        return {
            "cur_energy": float(self._cur_ev.total_energy),
            "best_energy": float(self._best_ev.total_energy),
            "epsilon": float(self._epsilon),
            "cur_makespan": float(cur_ms),
            "best_makespan": float(best_ms),
            "slack": float(slack),
            "slack_norm": float(slack_norm),
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

        # Append operator usage frequencies (paper-style state components).
        # Frequencies are normalized by elapsed iterations to keep values in [0,1].
        it_denom = float(max(1, int(self._it)))
        if self._destroy_counts is None:
            destroy_freq = np.zeros((int(len(self.destroy_choices)),), dtype=np.float32)
        else:
            destroy_freq = (self._destroy_counts / it_denom).astype(np.float32)
        if self._repair_counts is None:
            repair_freq = np.zeros((int(len(self.repair_choices)),), dtype=np.float32)
        else:
            repair_freq = (self._repair_counts / it_denom).astype(np.float32)

        out = np.concatenate(
            [
                np.asarray(feats, dtype=np.float32),
                destroy_freq,
                repair_freq,
            ],
            axis=0,
        )
        return out.astype(np.float32)

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        assert self._raw is not None
        assert self._cur_sol is not None and self._cur_ev is not None
        assert self._best_sol is not None and self._best_ev is not None

        # Accept either legacy int action (index into action_list) or the new
        # factorized action (destroy_idx, repair_idx, accept_idx, stop_idx).
        destroy_idx: int
        repair_idx: int
        accept_idx: int
        stop_idx: int

        if isinstance(action, (list, tuple, np.ndarray)):
            a = list(action)
            if len(a) != 4:
                raise ValueError("Factorized action must have 4 components")
            destroy_idx = int(a[0])
            repair_idx = int(a[1])
            accept_idx = int(a[2])
            stop_idx = int(a[3])
        else:
            # Legacy: single categorical over the prebuilt action list.
            action_idx = int(action)
            action_idx = int(max(0, min(len(self.action_list) - 1, int(action_idx))))
            dname, km, rname = self.action_list[action_idx]
            # Map to factorized indices for consistent logging/state updates.
            destroy_idx = int(self.destroy_choices.index((str(dname), float(km))))
            repair_idx = int(self.repair_choices.index(str(rname)))
            accept_idx = 0
            stop_idx = 0

        destroy_idx = int(
            max(0, min(int(len(self.destroy_choices)) - 1, int(destroy_idx)))
        )
        repair_idx = int(
            max(0, min(int(len(self.repair_choices)) - 1, int(repair_idx)))
        )
        accept_idx = int(1 if int(accept_idx) != 0 else 0)
        stop_idx = int(1 if int(stop_idx) != 0 else 0)

        if self._destroy_counts is not None:
            self._destroy_counts[destroy_idx] += 1.0
        if self._repair_counts is not None:
            self._repair_counts[repair_idx] += 1.0

        dname, km = self.destroy_choices[destroy_idx]
        rname = self.repair_choices[repair_idx]
        action3 = (str(dname), float(km), str(rname))

        prev_energy = float(self._cur_ev.total_energy)
        old_best_energy = float(self._best_ev.total_energy)

        step_exception = False
        try:
            cand_sol, cand_ev, _, _ = alns_apply_action(
                self._raw,
                int(self._epsilon),
                self._cur_sol,
                self._cur_ev,
                action3,
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
        failure_mode = "ok"
        if step_exception:
            failure_mode = "exception"

        if cand_sol is None or cand_ev is None:
            # Repair failed
            self._no_improve += 1
            reward = -float(self.fail_penalty)
            if failure_mode == "ok":
                failure_mode = "repair_failed"
        elif not feasible_cand:
            # Candidate violates epsilon / DP infeasible
            self._no_improve += 1
            reward = -float(self.infeasible_penalty)
            if failure_mode == "ok":
                failure_mode = "infeasible"
        else:
            cand_energy = float(cand_ev.total_energy)
            delta = float(cand_energy - prev_energy)
            # Acceptance criterion action (paper-style):
            # 0 = SA (can accept worse), 1 = greedy (accept only if improves current).
            if accept_idx == 1:
                accepted = bool(delta <= 0.0)
            else:
                if delta <= 0.0:
                    accepted = True
                else:
                    if self._tau > 1e-12:
                        prob = math.exp(-delta / float(self._tau))
                    else:
                        prob = 0.0
                    accepted = self._rng.random() < prob

            if accepted:
                self._cur_sol, self._cur_ev = cand_sol, cand_ev
                # Reward accepted move improvement (can be negative if accepted worse).
                reward += float(self.reward_accept_coef) * (
                    float(prev_energy) - float(self._cur_ev.total_energy)
                )
            else:
                # Optional rejection penalties.
                # 1) Constant penalty for any feasible rejection.
                if float(self.reject_penalty) != 0.0:
                    reward -= float(self.reject_penalty)
                # 2) Penalize proposing *worse* candidates that get rejected.
                if float(self.reject_worse_penalty_coef) != 0.0 and float(delta) > 0.0:
                    p = float(self.reject_worse_penalty_power)
                    if not np.isfinite(p) or p <= 0.0:
                        p = 1.0
                    reward -= float(self.reject_worse_penalty_coef) * (
                        float(delta) ** float(p)
                    )

            # Track best-so-far across *all feasible candidates*, even if rejected.
            improved_best = float(cand_energy) < float(old_best_energy)
            if improved_best:
                self._best_sol, self._best_ev = cand_sol, cand_ev
                self._no_improve = 0
            else:
                self._no_improve += 1

            # Primary shaping: reward improvement in best-so-far energy.
            # This aligns training with evaluation, which measures final best energy.
            new_best_energy = float(self._best_ev.total_energy)
            best_delta = float(old_best_energy) - float(new_best_energy)
            if float(self.reward_best_coef) != 0.0 and best_delta != 0.0:
                reward += float(self.reward_best_coef) * float(best_delta)

            # Optional shaping: reward good proposals even if rejected.
            if improved_best and float(self.best_improve_bonus) != 0.0:
                reward += float(self.best_improve_bonus)

            # Optional shaping: apply a signed power transform and global scale.
            # Using reward_power in (0, 1) compresses large improvements and makes
            # small-but-consistent improvements relatively more salient.
            if reward != 0.0:
                p = float(self.reward_power)
                if p <= 0.0:
                    p = 1.0
                reward = math.copysign((abs(float(reward)) ** p), float(reward))
            reward *= float(self.reward_scale)

        self._it += 1
        self._tau *= float(self.alns_cfg.sa_decay)

        done = bool(
            self._it >= int(self.alns_cfg.max_iters)
            or self._no_improve >= int(self.alns_cfg.no_improve_limit)
        )
        if (
            stop_idx == 1
            and int(self._it) >= int(max(0, self.min_stop_iters))
            and (
                float(self.terminal_best_coef) != 0.0
                or float(self.terminal_time_coef) != 0.0
            )
        ):
            done = True

        # Optional terminal reward: align with the paper's idea of mixing
        # end-of-episode improvement with a time penalty.
        if done and (
            float(self.terminal_best_coef) != 0.0
            or float(self.terminal_time_coef) != 0.0
        ):
            init = float(self._init_best_energy)
            best = float(self._best_ev.total_energy)
            denom = float(max(1e-9, abs(init)))
            rel_improve = float((init - best) / denom)
            t_frac = float(self._it) / float(max(1, int(self.alns_cfg.max_iters)))
            reward += float(self.terminal_best_coef) * rel_improve
            reward -= float(self.terminal_time_coef) * t_frac

        info = StepInfo(
            action=action3,
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
            {
                "step": info.__dict__,
                "action_factor": {
                    "destroy_idx": int(destroy_idx),
                    "repair_idx": int(repair_idx),
                    "accept_idx": int(accept_idx),
                    "stop_idx": int(stop_idx),
                },
                "exception": bool(step_exception),
                "failure_mode": str(failure_mode),
            },
        )

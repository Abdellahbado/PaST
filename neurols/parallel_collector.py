"""Parallel episode collection for NeuroLS training.

Uses multiprocessing to run multiple episodes simultaneously on CPU,
maximising utilisation of available cores for env.step() (CPU-bound
due to DP-based move evaluation).

Architecture
────────────
  Main Process (GPU)
    • Owns policy_net, target_net, optimizer
    • Periodically exports model weights to workers
    • Trains on shared replay buffer

  Worker Processes (CPU)
    • Each creates own env + lightweight CPU model
    • Runs full episodes with epsilon-greedy policy
    • Returns transitions + metrics to main process
"""

from __future__ import annotations

import os
import time
import random as py_random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# ---------------------------------------------------------------------------
# Limit worker threads to avoid over-subscription on many-core machines.
# These are set *before* torch/numpy pick them up in the worker.
# ---------------------------------------------------------------------------
_WORKER_THREADS = int(os.environ.get("NEUROLS_WORKER_THREADS", "2"))


@dataclass
class EpisodeResult:
    """Result returned by a parallel episode-collection worker."""

    transitions: List[Tuple]  # [(state_feats, action, reward, next_feats, done), ...]
    metrics: Dict[str, Any]


# ── Worker function (runs in a child process) ────────────────────────────


def _collect_episode_worker(args: Tuple) -> EpisodeResult:
    """Collect one full episode using a CPU-only model.

    This function runs in a **separate process**.  It creates its own env and
    model, loads the latest weights, runs the episode, and returns all
    transitions + summary metrics.
    """
    (
        instance_data,  # RawInstance (pickle-safe dataclass)
        K,
        model_state_dict,  # OrderedDict of CPU tensors
        env_config_dict,  # kwargs for EnvConfig
        model_config,  # dict with architecture hyper-params
        epsilon,
        worker_seed,
    ) = args

    # -- limit per-worker parallelism --
    os.environ["OMP_NUM_THREADS"] = str(_WORKER_THREADS)
    os.environ["MKL_NUM_THREADS"] = str(_WORKER_THREADS)

    import torch

    torch.set_num_threads(_WORKER_THREADS)

    py_random.seed(worker_seed)
    np.random.seed(worker_seed % (2**31))

    # Imports here so they run inside the child process
    from PaST.neurols.env import NeuroLSEnv, EnvConfig
    from PaST.neurols.decoder import NeuroLSPolicy
    from PaST.neurols.action_space import get_action_space

    # ── create env ──
    env_config = EnvConfig(**env_config_dict)
    env = NeuroLSEnv(env_config)
    env.seed(worker_seed)

    # ── create lightweight CPU model ──
    action_space = get_action_space(model_config["action_space"])

    model = NeuroLSPolicy(
        d_job_in=model_config.get("d_job_in", 5),
        d_machine_in=model_config.get("d_machine_in", 5),
        d_state_in=model_config.get("d_state_in", 13),
        d_emb=model_config["d_emb"],
        n_actions=action_space.n_actions,
        n_layers_static=model_config["n_layers_static"],
        n_layers_dynamic=model_config["n_layers_dynamic"],
        use_iqn=model_config["use_iqn"],
        use_dueling=model_config["use_dueling"],
        price_mode=model_config["price_mode"],
        dropout=model_config["dropout"],
        graph_type=model_config.get("graph_type", "bipartite"),
    )
    model.load_state_dict(model_state_dict)
    model.eval()

    # ── run episode ──
    ep_start = time.time()
    state = env.reset(instance_data, K)
    features = env.get_state_features()

    transitions: List[Tuple] = []
    episode_reward = 0.0
    initial_cost = state.current_cost
    n_accepted = 0
    n_improved = 0
    n_operator_steps = 0
    step_count = 0

    # AAN_MC diagnostics (criterion usage + collapse/overlap indicators)
    mc_steps = 0
    mc_same_as_cost = 0
    mc_unique_sum = 0.0
    mc_valid_sum = 0.0
    crit_pick: Dict[str, int] = {}
    crit_accept: Dict[str, int] = {}

    while True:
        # epsilon-greedy on CPU
        if py_random.random() < epsilon:
            action = py_random.randrange(model.n_actions)
        else:
            with torch.no_grad():
                st = {k: torch.tensor(v) for k, v in features.items()}
                q = model(
                    job_features=st["job_features"].float(),
                    machine_features=st["machine_features"].float(),
                    state_features=st["state_features"].float(),
                    job_to_machine=st["job_to_machine"].long(),
                    static_edge_index=st["static_edge_index"].long(),
                    dynamic_edge_index=st["dynamic_edge_index"].long(),
                    price_features=st.get("price_per_hour", None),
                    machine_exposure=(
                        st["machine_exposure"].float()
                        if "machine_exposure" in st
                        else None
                    ),
                    period_features=(
                        st["period_features"].float()
                        if "period_features" in st
                        else None
                    ),
                    tripartite_edge_index=(
                        st["tripartite_edge_index"].long()
                        if "tripartite_edge_index" in st
                        else None
                    ),
                )
                action = q.argmax().item()

        next_state, reward, done, info = env.step(action)
        next_features = env.get_state_features()

        # Multi-criteria diagnostics
        try:
            decoded = action_space.decode_action(action)
        except Exception:
            decoded = None

        if decoded is not None and getattr(decoded, "criterion_id", None) is not None:
            mc_steps += 1
            cname = getattr(decoded.criterion_id, "name", str(decoded.criterion_id))
            crit_pick[cname] = crit_pick.get(cname, 0) + 1
            if info.get("accepted", False):
                crit_accept[cname] = crit_accept.get(cname, 0) + 1

            if info.get("mc_same_as_cost", False):
                mc_same_as_cost += 1
            if info.get("mc_unique_moves", None) is not None:
                mc_unique_sum += float(info.get("mc_unique_moves") or 0.0)
            if info.get("mc_valid_criteria", None) is not None:
                mc_valid_sum += float(info.get("mc_valid_criteria") or 0.0)

        if info.get("accepted", False):
            n_accepted += 1
        if info.get("improved", False):
            n_improved += 1
        if info.get("move_result") is not None or info.get("perturbed", False):
            n_operator_steps += 1

        transitions.append((features, action, reward, next_features, done))

        features = next_features
        state = next_state
        episode_reward += reward
        step_count += 1

        if done:
            break

    ep_time = time.time() - ep_start
    improvement = (initial_cost - state.best_cost) / initial_cost * 100
    accept_rate = n_accepted / max(1, n_operator_steps)

    cache_stats = env.get_cache_stats()

    mc_same_rate = mc_same_as_cost / max(1, mc_steps)
    mc_unique_avg = mc_unique_sum / max(1, mc_steps)
    mc_valid_avg = mc_valid_sum / max(1, mc_steps)

    crit_total = sum(crit_pick.values())
    crit_pick_frac = {k: (v / max(1, crit_total)) for k, v in crit_pick.items()}
    crit_accept_rate = {
        k: (crit_accept.get(k, 0) / max(1, crit_pick.get(k, 0)))
        for k in crit_pick.keys()
    }

    metrics: Dict[str, Any] = {
        "reward": episode_reward,
        "length": step_count,
        "improvement": improvement,
        "initial_cost": initial_cost,
        "best_cost": state.best_cost,
        "accept_rate": accept_rate,
        "improve_rate": n_improved / max(1, n_operator_steps),
        "ep_time": ep_time,
        "cache_hit_rate": cache_stats.get("hit_rate", 0.0),
        "cache_size": cache_stats.get("size", 0),
        # Multi-criteria diagnostics (present even if mc_steps=0)
        "mc_steps": mc_steps,
        "mc_step_frac": (mc_steps / max(1, step_count)),
        "mc_same_as_cost_rate": mc_same_rate,
        "mc_unique_moves_avg": mc_unique_avg,
        "mc_valid_criteria_avg": mc_valid_avg,
        "mc_crit_pick_frac": crit_pick_frac,
        "mc_crit_accept_rate": crit_accept_rate,
    }

    return EpisodeResult(transitions=transitions, metrics=metrics)


# ── Collector manager ─────────────────────────────────────────────────────


class ParallelCollector:
    """Manages a pool of worker processes for episode collection."""

    def __init__(self, n_workers: int = 8):
        self.n_workers = n_workers
        self._pool: Optional[ProcessPoolExecutor] = None

    def start(self):
        """Create the process pool (uses 'spawn' for CUDA safety)."""
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        self._pool = ProcessPoolExecutor(
            max_workers=self.n_workers,
            mp_context=ctx,
        )

    def stop(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def collect_episodes(
        self,
        instances_and_Ks: List[Tuple],
        model_state_dict: Dict,
        env_config_dict: Dict,
        model_config: Dict,
        epsilon: float,
        base_seed: int = 0,
    ) -> List[EpisodeResult]:
        """Collect N episodes in parallel.

        Args:
            instances_and_Ks: [(RawInstance, K), ...] one per worker
            model_state_dict: policy weights (CPU tensors)
            env_config_dict: EnvConfig as plain dict
            model_config: architecture hyperparams dict
            epsilon: exploration rate
            base_seed: each worker gets base_seed + i

        Returns:
            List of EpisodeResult
        """
        if self._pool is None:
            self.start()

        # Move weights to CPU once (shared across workers via pickle COW)
        cpu_state = {k: v.cpu() for k, v in model_state_dict.items()}

        worker_args = []
        for i, (instance, K) in enumerate(instances_and_Ks):
            worker_args.append(
                (
                    instance,
                    K,
                    cpu_state,
                    env_config_dict,
                    model_config,
                    epsilon,
                    base_seed + i,
                )
            )

        # map preserves order; chunksize=1 for even load distribution
        results = list(
            self._pool.map(_collect_episode_worker, worker_args, chunksize=1)
        )
        return results

    def __del__(self):
        self.stop()

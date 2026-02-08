#!/usr/bin/env python3
"""Task learnability test (oracle + probe), independent of RL training.

This script is *additive* (does not modify or replace the existing
`learnability_test.py`). Its goal is to answer:

- Does the environment define a learnable control problem?
- Given the provided observation tensors (state features), is there a
  predictable mapping to “good” actions?

Approach (per variant):
1) Collect states by running a short random policy.
2) For each collected state, compute an *oracle* best action via exhaustive
   one-step lookahead under deterministic acceptance.
3) Report action-value margin statistics (how much better the best action is
   than typical actions).
4) Train a small supervised probe to predict the oracle action from a fixed
   vectorization of the observation (quick sanity check of predictability).

Notes:
- The oracle uses deterministic acceptance to remove SA noise; this tests
  *task signal* (action consequences), not whether an RL variant learns.
- Acceptance bits may be less informative under deterministic oracle; the
  primary signal tested is operator/perturbation selection.

Usage:
  python -m PaST.scripts.task_learnability_test --variants AAN_zprice AAN_full

To run a more stable estimate, increase instances:
    python -m PaST.scripts.task_learnability_test --variants AAN_zprice --n-instances 50

Defaults are chosen to be reasonably fast on CPU.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


_VARIANT_TO_CONFIG = {
    "AA_zprice": "PaST/configs/neurols_AA_zprice.yaml",
    "AA_full": "PaST/configs/neurols_AA_full.yaml",
    "AAN_zprice": "PaST/configs/neurols_AAN_zprice.yaml",
    "AAN_full": "PaST/configs/neurols_AAN_full.yaml",
    "AANP_zprice": "PaST/configs/neurols_AANP_zprice.yaml",
    "AANP_full": "PaST/configs/neurols_AANP_full.yaml",
    "triA_zprice": "PaST/configs/neurols_triA_AANP_zprice.yaml",
    "triA_full": "PaST/configs/neurols_triA_AANP_full.yaml",
    "triB_full": "PaST/configs/neurols_triB_AANPD_full.yaml",
}


def _repo_root() -> Path:
    # This file lives at PaST/scripts/task_learnability_test.py
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping, got {type(data)}")
    return data


def _snapshot_env(env) -> Dict[str, Any]:
    """Snapshot mutable env state for exact restore after probing an action."""
    snap: Dict[str, Any] = {}
    snap["state"] = env.state.clone() if env.state is not None else None
    snap["current_eval"] = env._current_eval
    snap["temperature"] = env.temperature
    snap["step_count"] = env._step_count
    snap["no_improve_count"] = env._no_improve_count
    snap["best_cost_episode"] = env._best_cost_episode
    snap["initial_cost"] = env._initial_cost

    # Cache-related
    snap["solution_cache"] = dict(env._solution_cache)
    snap["cache_hits"] = env._cache_hits
    snap["cache_queries"] = env._cache_queries

    # Determinism flag
    snap["deterministic"] = getattr(env.config, "deterministic", False)

    # RNG states
    snap["py_random_state"] = random.getstate()
    try:
        snap["np_rng_state"] = copy.deepcopy(env._rng.bit_generator.state)
    except Exception:
        snap["np_rng_state"] = None

    return snap


def _restore_env(env, snap: Dict[str, Any]) -> None:
    env.state = snap["state"]
    env._current_eval = snap["current_eval"]
    env.temperature = snap["temperature"]
    env._step_count = snap["step_count"]
    env._no_improve_count = snap["no_improve_count"]
    env._best_cost_episode = snap["best_cost_episode"]
    env._initial_cost = snap["initial_cost"]

    env._solution_cache = dict(snap["solution_cache"])
    env._cache_hits = snap["cache_hits"]
    env._cache_queries = snap["cache_queries"]

    env.config.deterministic = snap["deterministic"]

    random.setstate(snap["py_random_state"])
    if snap.get("np_rng_state") is not None:
        env._rng.bit_generator.state = snap["np_rng_state"]


def _vectorize_features(features: Dict[str, np.ndarray]) -> np.ndarray:
    """Turn the variable-structured observation into a fixed-size vector.

    This intentionally uses simple pooling statistics so the probe answers:
    “Is there signal in the provided inputs at all?”

    Vector includes:
    - state_features (as-is)
    - mean/std of job_features over jobs
    - mean/std of machine_features over machines
    - mean/std over hours of price_per_hour (per-channel)
    - optionally: mean/std of machine_exposure (per-channel)
    - optionally: mean/std of period_features (per-channel)
    """

    vec_parts: List[np.ndarray] = []

    def _mean_std(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return np.zeros((0,), dtype=np.float32)
        if x.ndim == 1:
            return np.concatenate([x, np.zeros_like(x)], axis=0)
        mu = x.mean(axis=0)
        sd = x.std(axis=0)
        return np.concatenate([mu, sd], axis=0)

    vec_parts.append(np.asarray(features["state_features"], dtype=np.float32).ravel())

    vec_parts.append(_mean_std(features["job_features"]))
    vec_parts.append(_mean_std(features["machine_features"]))

    # Price per hour is constant per episode (20, 5)
    if "price_per_hour" in features:
        vec_parts.append(_mean_std(features["price_per_hour"]))

    if "machine_exposure" in features:
        vec_parts.append(_mean_std(features["machine_exposure"]))

    if "period_features" in features:
        vec_parts.append(_mean_std(features["period_features"]))

    return np.concatenate(vec_parts, axis=0).astype(np.float32)


def _oracle_one_step(
    env, *, rng: random.Random, verbose: bool = False
) -> Tuple[int, Dict[str, float], List[int]]:
    """Exhaustive one-step oracle (expected immediate reward).

    This oracle does NOT call env.step() for each action (too slow). Instead it
    estimates the *expected* 1-step reward under the environment's current
    reward shaping mode by:
    1) computing the cost of applying the generator once (operator/perturb/destroy)
    2) applying accept/reject semantics analytically (Metropolis / deterministic)
    3) computing the expected reward (incl. step penalty)
    """

    from PaST.neurols.perturbations import (
        PERTURBATION_BY_ID,
        PerturbationID,
        DESTROY_BY_ID,
    )
    from PaST.neurols.action_space import ActionType

    if env.state is None:
        raise RuntimeError("Env must be reset() before oracle evaluation")

    best_cost_before = float(env.state.best_cost)
    current_cost_before = float(env._current_eval.total_energy)
    denom = float(env._initial_cost) + float(env.config.reward_eps)

    def _accept_prob(*, accept_flag: bool, new_cost: float) -> float:
        if not np.isfinite(new_cost):
            return 0.0
        if new_cost < current_cost_before:
            return 1.0
        if not accept_flag:
            return 0.0
        if bool(env.config.deterministic):
            return 0.0
        if not bool(env.config.use_acceptance):
            return 1.0

        delta = float(new_cost) - float(current_cost_before)
        if not np.isfinite(delta):
            return 0.0
        if delta <= 0.0:
            return 1.0
        temp = float(getattr(env, "temperature", 0.0))
        if temp <= 0.0:
            return 0.0
        ratio = -delta / temp
        if not np.isfinite(ratio):
            return 0.0
        p = float(np.exp(ratio))
        return float(min(1.0, max(0.0, p)))

    def _expected_reward(*, accept_flag: bool, new_cost: Optional[float]) -> float:
        if new_cost is None:
            return float(-env.config.step_penalty)

        if not np.isfinite(float(new_cost)):
            return float(-env.config.step_penalty)

        p = _accept_prob(accept_flag=accept_flag, new_cost=float(new_cost))
        new_cost_f = float(new_cost)
        mode = str(env.config.reward_mode)

        if mode == "improvement":
            exp_improve = p * max(best_cost_before - new_cost_f, 0.0)
            reward = exp_improve * float(env.config.improvement_scale)
        elif mode == "normalized":
            exp_improve = p * (best_cost_before - min(best_cost_before, new_cost_f))
            reward = (exp_improve / denom) * float(env.config.improvement_scale)
        elif mode == "potential":
            exp_delta = p * (current_cost_before - new_cost_f)
            reward = exp_delta * float(env.config.improvement_scale)
        elif mode == "dense_best":
            exp_r_cur = p * (current_cost_before - new_cost_f) / denom
            exp_r_best = p * max(best_cost_before - new_cost_f, 0.0) / denom
            reward = (
                exp_r_cur + float(env.config.best_bonus_lambda) * exp_r_best
            ) * float(env.config.improvement_scale)
        else:
            raise ValueError(f"Unknown reward_mode: {mode}")

        reward -= float(env.config.step_penalty)
        if not np.isfinite(float(reward)):
            return float(-env.config.step_penalty)
        return float(reward)

    n_actions = int(env.action_space_size)

    gen_to_actions: Dict[Tuple[Any, Any, Any, Any], List[int]] = {}
    for a in range(n_actions):
        d = env.config.action_space.decode_action(a)
        gen_key = (d.action_type, d.operator_id, d.perturbation_id, d.destroy_id)
        gen_to_actions.setdefault(gen_key, []).append(a)

    def _stable_key(k: Tuple[Any, Any, Any, Any]) -> Tuple[int, int, int, int]:
        at, op, pert, dest = k
        return (
            int(at),
            int(op) if op is not None else -1,
            int(pert) if pert is not None else -1,
            int(dest) if dest is not None else -1,
        )

    gen_keys = sorted(gen_to_actions.keys(), key=_stable_key)
    gen_rewards = np.zeros((len(gen_keys),), dtype=np.float32)
    action_rewards = np.zeros((n_actions,), dtype=np.float32)

    accept_pairs = 0
    accept_better = 0
    reject_better = 0
    accept_tie = 0

    for gi, gen_key in enumerate(gen_keys):
        a_rep = gen_to_actions[gen_key][0]
        decoded = env.config.action_space.decode_action(a_rep)

        new_cost: Optional[float] = None

        if decoded.action_type == ActionType.OPERATOR:
            if decoded.operator_id is not None:
                result = env._apply_operator(decoded.operator_id)
            else:
                result = env._apply_best_operator()
            if result is not None:
                _move, c, _eval = result
                new_cost = float(c)

        elif decoded.action_type == ActionType.PERTURBATION:
            if (
                decoded.perturbation_id is None
                or decoded.perturbation_id == PerturbationID.NONE
            ):
                new_cost = None
            else:
                perturbation = PERTURBATION_BY_ID[decoded.perturbation_id]
                new_solution = perturbation.apply(
                    solution=env.state.solution,
                    evaluation=env._current_eval,
                    processing_times=env._processing_times,
                    ct=env._ct,
                    machine_energy_rates=env._machine_energy_rates,
                    K=env.K,
                )
                sol_key = new_solution.hash_key()
                cached = env._solution_cache.get(sol_key)
                if cached is not None:
                    new_cost = float(cached[0])
                else:
                    new_eval = env._evaluator.evaluate_solution(new_solution)
                    new_cost = float(new_eval.total_energy)

        elif decoded.action_type == ActionType.DESTROY:
            if decoded.destroy_id is None:
                new_cost = None
            else:
                destroy_op = DESTROY_BY_ID[decoded.destroy_id]
                new_solution = destroy_op.apply(
                    solution=env.state.solution,
                    processing_times=env._processing_times,
                    machine_energy_rates=env._machine_energy_rates,
                    K=env.K,
                )
                sol_key = new_solution.hash_key()
                cached = env._solution_cache.get(sol_key)
                if cached is not None:
                    new_cost = float(cached[0])
                else:
                    new_eval = env._evaluator.evaluate_solution(new_solution)
                    new_cost = float(new_eval.total_energy)

        accept_reward = None
        reject_reward = None
        for a in gen_to_actions[gen_key]:
            a_decoded = env.config.action_space.decode_action(a)
            r = _expected_reward(accept_flag=bool(a_decoded.accept), new_cost=new_cost)
            action_rewards[a] = r
            if bool(a_decoded.accept):
                accept_reward = r
            else:
                reject_reward = r

        if accept_reward is not None and reject_reward is not None:
            accept_pairs += 1
            if accept_reward > reject_reward:
                accept_better += 1
            elif accept_reward < reject_reward:
                reject_better += 1
            else:
                accept_tie += 1

        gen_rewards[gi] = float(np.max(action_rewards[gen_to_actions[gen_key]]))

    best_r = float(gen_rewards.max()) if gen_rewards.size else 0.0
    best_gen_idxs = np.flatnonzero(gen_rewards == best_r)
    if best_gen_idxs.size == 0:
        best_gen = 0
    elif best_gen_idxs.size == 1:
        best_gen = int(best_gen_idxs[0])
    else:
        best_gen = int(rng.choice(best_gen_idxs.tolist()))

    best_gen_set = [int(idx) for idx in best_gen_idxs.tolist()]

    sorted_gen_rewards = np.sort(gen_rewards) if gen_rewards.size else np.array([0.0])
    second_r = (
        float(sorted_gen_rewards[-2])
        if sorted_gen_rewards.size >= 2
        else float(sorted_gen_rewards[-1])
    )
    median_r = float(np.median(gen_rewards)) if gen_rewards.size else 0.0
    mean_r = float(np.mean(gen_rewards)) if gen_rewards.size else 0.0

    nonzero_frac = float(np.mean(gen_rewards > 0.0)) if gen_rewards.size else 0.0
    all_zero = float(1.0 if (best_r <= 0.0 and nonzero_frac == 0.0) else 0.0)
    best_is_zero = float(1.0 if best_r <= 0.0 else 0.0)
    tie_best_nonzero = float(1.0 if (best_gen_idxs.size > 1 and best_r > 0.0) else 0.0)

    best_a = float(action_rewards.max()) if action_rewards.size else 0.0
    best_action_idxs = np.flatnonzero(action_rewards == best_a)
    tie_best_action = float(1.0 if best_action_idxs.size > 1 else 0.0)
    nonzero_action_frac = (
        float(np.mean(action_rewards > 0.0)) if action_rewards.size else 0.0
    )

    if verbose and tie_best_nonzero > 0.5:
        print(f"\n[Tie Detected] Best Reward: {best_r:.6f}")
        for idx in best_gen_idxs:
            gk = gen_keys[idx]
            at_type = ActionType(gk[0])
            at_str = str(at_type.name)

            parts = []
            if at_type == ActionType.OPERATOR:
                OP_NAMES = {0: "SWAP", 1: "INSERT", 2: "MOVE", 3: "CROSS"}
                if gk[1] is not None and gk[1] != -1:
                    parts.append(OP_NAMES.get(gk[1], f"OP_{gk[1]}"))
                else:
                    parts.append("BEST_OP")
            elif at_type == ActionType.PERTURBATION:
                if gk[2] is not None and gk[2] != -1:
                    parts.append(PERTURBATION_BY_ID[gk[2]].__class__.__name__)
                else:
                    parts.append("NO_PERT")
            elif at_type == ActionType.DESTROY:
                if gk[3] is not None and gk[3] != -1:
                    parts.append(DESTROY_BY_ID[gk[3]].__class__.__name__)
                else:
                    parts.append("NO_DEST")

            details = " ".join(parts)
            print(f"  Gen {idx}: {at_str:12s} {details}")

    accept_pairs = max(1, int(accept_pairs))

    stats = {
        "best_reward": best_r,
        "second_best_reward": second_r,
        "margin_best_second": best_r - second_r,
        "margin_best_median": best_r - median_r,
        "adv_best_minus_mean": best_r - mean_r,
        "mean_reward": mean_r,
        "median_reward": median_r,
        "nonzero_reward_frac": nonzero_frac,
        "nonzero_action_frac": nonzero_action_frac,
        "n_generators": float(len(gen_keys)),
        "tie_best_count": float(best_gen_idxs.size),
        "tie_best": float(1.0 if best_gen_idxs.size > 1 else 0.0),
        "tie_best_action": tie_best_action,
        "tie_best_nonzero": tie_best_nonzero,
        "best_is_zero": best_is_zero,
        "all_zero": all_zero,
        "accept_better_rate": float(accept_better) / float(accept_pairs),
        "reject_better_rate": float(reject_better) / float(accept_pairs),
        "accept_tie_rate": float(accept_tie) / float(accept_pairs),
    }

    return best_gen, stats, best_gen_set


def _train_probe(
    X: np.ndarray,
    y: np.ndarray,
    n_actions: int,
    seed: int,
    device: str = "cpu",
    best_sets: Optional[Sequence[Sequence[int]]] = None,
) -> Dict[str, float]:
    """Train a small MLP probe to predict oracle action from pooled features."""

    if not _TORCH_AVAILABLE:
        return {"probe_enabled": 0.0}

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(0.8 * n)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32, device=device)
    y_train = torch.tensor(y[train_idx], dtype=torch.long, device=device)
    X_test = torch.tensor(X[test_idx], dtype=torch.float32, device=device)
    y_test = torch.tensor(y[test_idx], dtype=torch.long, device=device)

    d_in = X.shape[1]

    model = nn.Sequential(
        nn.Linear(d_in, 128),
        nn.ReLU(),
        nn.Linear(128, n_actions),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    batch_size = min(128, max(16, n_train // 4))
    n_epochs = 25

    for _epoch in range(n_epochs):
        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, batch_size):
            b = perm[start : start + batch_size]
            logits = model(X_train[b])
            loss = loss_fn(logits, y_train[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        pred = test_logits.argmax(dim=1)
        acc = (pred == y_test).float().mean().item() if y_test.numel() else 0.0
        ce = loss_fn(test_logits, y_test).item() if y_test.numel() else 0.0

    tie_acc = 0.0
    if best_sets is not None and y_test.numel():
        hits = 0
        total = int(y_test.numel())
        for i, pi in enumerate(pred.tolist()):
            if pi in set(best_sets[test_idx[i]]):
                hits += 1
        tie_acc = float(hits / max(1, total))

    return {
        "probe_enabled": 1.0,
        "probe_test_acc": float(acc),
        "probe_test_acc_tie": float(tie_acc),
        "probe_test_ce": float(ce),
        "n_train": float(n_train),
        "n_test": float(n - n_train),
    }


def _chance_accuracy(y: np.ndarray, n_actions: int) -> float:
    if y.size == 0:
        return 0.0
    counts = np.bincount(y, minlength=n_actions).astype(np.float64)
    return float(counts.max() / counts.sum())


def _chance_accuracy_tie(best_sets: Sequence[Sequence[int]], n_actions: int) -> float:
    """Tie-aware chance accuracy for a uniform-random predictor.

    If you pick a generator uniformly at random, probability of being correct
    under tie-aware scoring is |argmax_set| / n_actions.
    """

    if n_actions <= 0 or not best_sets:
        return 0.0
    sizes = [len(set(bs)) for bs in best_sets]
    return float(np.mean(np.asarray(sizes, dtype=np.float64) / float(n_actions)))


def run_variant(
    key: str,
    config_path: Path,
    *,
    seed: int,
    n_instances: int,
    states_per_instance: int,
    K_fixed: int,
    device: str,
    top_k: int,
    use_proxy: bool,
    step_penalty_override: Optional[float] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    from PaST.neurols.env import NeuroLSEnv, EnvConfig
    from PaST.neurols.action_space import get_action_space
    from PaST.data.sm_benchmark_data import generate_raw_instance
    from PaST.config import DataConfig

    cfg = _load_yaml(config_path)

    action_space_name = str(cfg.get("action_space", "AANP"))
    graph_type = str(cfg.get("graph_type", "bipartite"))
    price_mode = str(cfg.get("price_mode", "z_price"))

    # Use small fixed instance sizes for speed
    n_jobs_list = cfg.get("n_jobs_train", [20])
    n_machines_list = cfg.get("n_machines_train", [3])
    fixed_n = int(n_jobs_list[0])
    fixed_m = int(n_machines_list[0])

    step_penalty = (
        float(step_penalty_override)
        if step_penalty_override is not None
        else float(cfg.get("step_penalty", 0.0))
    )

    env_config = EnvConfig(
        max_steps=int(cfg.get("max_steps", 200)),
        stagnation_limit=int(cfg.get("stagnation_limit", 100)),
        action_space=get_action_space(action_space_name),
        reward_mode=str(cfg.get("reward_mode", "dense_best")),
        improvement_scale=float(cfg.get("improvement_scale", 1.0)),
        reward_eps=float(cfg.get("reward_eps", 1e-8)),
        best_bonus_lambda=float(cfg.get("best_bonus_lambda", 0.3)),
        step_penalty=step_penalty,
        top_k=int(top_k),
        use_proxy=bool(use_proxy),
        graph_type=graph_type,
        price_mode=price_mode,
    )

    env = NeuroLSEnv(env_config)
    env.seed(seed)

    data_config = DataConfig(sampling_mode="new_benchmark_grid")
    hours_per_day = int(getattr(data_config, "hours_per_day", 20))
    K = int(K_fixed)
    if K % hours_per_day != 0:
        K = (K // hours_per_day) * hours_per_day
        K = max(K, hours_per_day)
    D_days = int(K // hours_per_day)

    inst_rng = random.Random(seed)

    X_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    oracle_stats: List[Dict[str, float]] = []
    best_sets: List[List[int]] = []

    for instance_id in range(n_instances):
        instance = generate_raw_instance(
            config=data_config,
            rng=inst_rng,
            instance_id=instance_id,
            n=fixed_n,
            m=fixed_m,
            D_days=D_days,
        )

        env.reset(instance, K)

        # Collect a handful of states by random stepping
        for t in range(states_per_instance):
            features = env.get_state_features()

            # Oracle label from the *current* state.
            # Oracle label from the *current* state.
            best_action, stats, best_set = _oracle_one_step(
                env, rng=inst_rng, verbose=verbose
            )
            X_rows.append(_vectorize_features(features))
            y_rows.append(best_action)
            oracle_stats.append(stats)
            best_sets.append(best_set)

            # Step randomly to explore state space.
            a = int(inst_rng.randrange(env.action_space_size))
            _s, _r, done, _info = env.step(a)
            if done:
                break

    X = np.stack(X_rows, axis=0) if X_rows else np.zeros((0, 1), dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.int64)

    n_actions = int(env.action_space_size)

    # Oracle/probe are over *generator* labels (accept bit collapsed)
    if oracle_stats:
        n_generators = int(round(float(oracle_stats[0].get("n_generators", 0.0))))
    else:
        n_generators = 0
    n_generators = max(1, int(n_generators))
    chance = _chance_accuracy(y, n_generators)
    chance_tie = _chance_accuracy_tie(best_sets, n_generators)
    tie_set_size_mean = float(
        np.mean([len(set(bs)) for bs in best_sets]) if best_sets else 0.0
    )

    stats_mean = {
        k: float(np.mean([d[k] for d in oracle_stats])) if oracle_stats else 0.0
        for k in (
            "best_reward",
            "second_best_reward",
            "margin_best_second",
            "margin_best_median",
            "adv_best_minus_mean",
            "nonzero_reward_frac",
            "nonzero_action_frac",
            "mean_reward",
            "median_reward",
            "tie_best",
            "tie_best_count",
            "tie_best_action",
            "tie_best_nonzero",
            "best_is_zero",
            "all_zero",
            "accept_better_rate",
            "reject_better_rate",
            "accept_tie_rate",
            "n_generators",
        )
    }

    probe = _train_probe(
        X,
        y,
        n_actions=n_generators,
        seed=seed + 123,
        device=device,
        best_sets=best_sets,
    )

    return {
        "variant": key,
        "config": str(config_path),
        "action_space": action_space_name,
        "graph_type": graph_type,
        "price_mode": price_mode,
        "n_actions": n_actions,
        "n_generators": int(n_generators),
        "n_samples": int(y.size),
        "chance_acc": float(chance),
        "chance_acc_tie": float(chance_tie),
        "tie_set_size_mean": float(tie_set_size_mean),
        **stats_mean,
        **probe,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task learnability test (oracle + probe), independent of RL"
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=list(_VARIANT_TO_CONFIG.keys()),
        help=f"Variants to test. Default: all ({len(_VARIANT_TO_CONFIG)})",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-instances",
        type=int,
        default=32,
        help="Number of instances to sample per variant (higher = more stable, slower).",
    )
    parser.add_argument("--states-per-instance", type=int, default=20)
    parser.add_argument("--K", type=int, default=40)
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Max candidate moves per operator for the oracle (speed/faithfulness trade-off).",
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Disable proxy ranking in candidate generation (slower).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Probe device (cpu/cuda). Training itself is env-only.",
    )
    parser.add_argument(
        "--step-penalty",
        type=float,
        default=None,
        help="Override step_penalty for this run (e.g., 1e-4).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print details about tied generators.",
    )

    args = parser.parse_args()

    root = _repo_root()

    results: List[Dict[str, Any]] = []
    for key in args.variants:
        if key not in _VARIANT_TO_CONFIG:
            raise SystemExit(
                f"Unknown variant '{key}'. Available: {sorted(_VARIANT_TO_CONFIG.keys())}"
            )

        config_path = root / _VARIANT_TO_CONFIG[key]
        if not config_path.exists():
            raise SystemExit(f"Missing config: {config_path}")

        res = run_variant(
            key,
            config_path,
            seed=int(args.seed),
            n_instances=int(args.n_instances),
            states_per_instance=int(args.states_per_instance),
            K_fixed=int(args.K),
            device=str(args.device),
            top_k=int(args.top_k),
            use_proxy=not bool(args.no_proxy),
            step_penalty_override=args.step_penalty,
            verbose=bool(args.verbose),
        )
        results.append(res)

        # Short per-variant print (kept concise)
        probe_str = (
            f"probe_acc={res.get('probe_test_acc', 0.0):.3f}"
            if res.get("probe_enabled", 0.0) > 0
            else "probe=off"
        )
        probe_tie_str = (
            f"probe_tie={res.get('probe_test_acc_tie', 0.0):.3f}"
            if res.get("probe_enabled", 0.0) > 0
            else ""
        )
        tie_str = f"tie={res.get('tie_best', 0.0):.2%}"
        tie_action_str = f"tie_action={res.get('tie_best_action', 0.0):.2%}"
        zero_str = f"all0={res.get('all_zero', 0.0):.2%}"
        accept_str = (
            f"accept>reject={res.get('accept_better_rate', 0.0):.2%}"
            f"/tie={res.get('accept_tie_rate', 0.0):.2%}"
        )
        chance_tie_str = f"chance_tie={res.get('chance_acc_tie', 0.0):.3f}"
        tie_sz_str = f"tie_sz={res.get('tie_set_size_mean', 0.0):.2f}"
        print(
            f"[{key:9s}] samples={res['n_samples']:4d}  "
            f"gens={int(res.get('n_generators', 0)):3d}  "
            f"chance={res['chance_acc']:.3f}  {chance_tie_str}  {tie_sz_str}  "
            f"{probe_str}  {probe_tie_str}  {tie_str}  "
            f"{tie_action_str}  {zero_str}  {accept_str}  "
            f"best-mean={res['adv_best_minus_mean']:.4g}  "
            f"margin(best,2nd)={res['margin_best_second']:.4g}  "
            f"nonzero={res['nonzero_reward_frac']:.2%}"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"results": results}, f, indent=2)
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

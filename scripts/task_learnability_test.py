#!/usr/bin/env python3
"""Task learnability test (oracle + probe), independent of RL training.

This script is *additive* (does not modify or replace the existing
`learnability_test.py`). Its goal is to answer:

- Does the environment define a learnable control problem?
- Given the provided observation tensors (state features), is there a
  predictable mapping to “good” actions?

Approach (per variant):
1) Collect states by running a short random policy.
2) For each collected state, compute an *oracle* best generator via exhaustive
    one-step lookahead (expected 1-step reward under the env's acceptance semantics).
    Note: a 1-step oracle is intentionally myopic; use --oracle-horizon > 1 to
    evaluate delayed-payoff generators (perturb/destroy).
3) Report action-value margin statistics (how much better the best action is
   than typical actions).
4) Train a small supervised probe to predict the oracle action from a fixed
   vectorization of the observation (quick sanity check of predictability).

Notes:
- The 1-step oracle computes an *expected* immediate reward (analytical accept/reject),
    which removes sampling noise but is still a one-step, local signal.
- The multi-step oracle (when --oracle-horizon > 1) forces one generator step to be
    accepted, then performs a short deterministic greedy improvement rollout.
- Labels are over *generators* (accept/reject is collapsed by taking the best expected
    reward among that generator's accept/reject actions).

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


def _coeff_var(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if (not np.isfinite(mu)) or abs(mu) <= 1e-12:
        return 0.0
    v = sd / abs(mu)
    return float(v) if np.isfinite(v) else 0.0


def _compute_struct_metrics(
    env,
    *,
    hours_per_day: int = 20,
    cheap_quantile: float = 0.25,
) -> Dict[str, float]:
    """Compute interpretable structural metrics from the current env state.

    These are *diagnostic* metrics to see if a rollout is moving in the
    "right direction" structurally (cheap-time usage, spillover, balance),
    not a replacement for the true objective.
    """

    if env.state is None or env._current_eval is None:
        return {}

    ct = getattr(env, "_ct", None)
    processing_times = getattr(env, "_processing_times", None)
    solution = getattr(env.state, "solution", None)

    metrics: Dict[str, float] = {}
    try:
        metrics["cur_cost"] = float(env.state.current_cost)
        metrics["best_cost"] = float(env.state.best_cost)
    except Exception:
        pass

    try:
        per_m_energy = np.asarray(
            [float(me.energy) for me in env._current_eval.per_machine], dtype=np.float64
        )
        per_m_ms = np.asarray(
            [float(me.makespan) for me in env._current_eval.per_machine],
            dtype=np.float64,
        )
        metrics["load_cv_energy"] = _coeff_var(per_m_energy)
        metrics["load_cv_makespan"] = _coeff_var(per_m_ms)
    except Exception:
        pass

    try:
        if processing_times is not None and solution is not None:
            loads = solution.get_machine_loads(np.asarray(processing_times))
            metrics["load_cv_pt"] = _coeff_var(loads)
    except Exception:
        pass

    # Cross-day spillover: fraction of jobs whose execution interval crosses a day boundary.
    try:
        if (
            processing_times is not None
            and solution is not None
            and hours_per_day is not None
            and int(hours_per_day) > 0
        ):
            hpd = int(hours_per_day)
            n_jobs_total = 0
            n_cross = 0
            pt = np.asarray(processing_times, dtype=np.int64)
            for mi, seq in enumerate(solution.sequences):
                starts = getattr(env._current_eval.per_machine[mi], "start_times", [])
                if not seq or not starts:
                    continue
                n = min(len(seq), len(starts))
                for j_idx in range(n):
                    job = int(seq[j_idx])
                    st = int(starts[j_idx])
                    dur = int(pt[job])
                    if dur <= 0:
                        continue
                    end = st + dur - 1
                    n_jobs_total += 1
                    if (st // hpd) != (end // hpd):
                        n_cross += 1
            metrics["cross_day_job_frac"] = float(n_cross / max(1, n_jobs_total))
    except Exception:
        pass

    # Cheap-time usage: share of processing time executed in the cheapest quantile of slots.
    try:
        if ct is not None and processing_times is not None and solution is not None:
            ct_arr = np.asarray(ct, dtype=np.float64)
            if ct_arr.size:
                thr = float(np.quantile(ct_arr, float(cheap_quantile)))
                pt = np.asarray(processing_times, dtype=np.int64)
                cheap_t = 0.0
                total_t = 0.0
                price_sum = 0.0
                for mi, seq in enumerate(solution.sequences):
                    starts = getattr(
                        env._current_eval.per_machine[mi], "start_times", []
                    )
                    if not seq or not starts:
                        continue
                    n = min(len(seq), len(starts))
                    for j_idx in range(n):
                        job = int(seq[j_idx])
                        st = int(starts[j_idx])
                        dur = int(pt[job])
                        if dur <= 0:
                            continue
                        for t in range(st, st + dur):
                            if 0 <= t < int(ct_arr.size):
                                price = float(ct_arr[t])
                                total_t += 1.0
                                price_sum += price
                                if price <= thr:
                                    cheap_t += 1.0
                metrics["cheap_time_frac"] = float(cheap_t / max(1.0, total_t))
                metrics["mean_price_used"] = float(price_sum / max(1.0, total_t))
    except Exception:
        pass

    return metrics


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

    mode = str(env.config.reward_mode)
    use_exposure = (
        mode == "dense_best_exposure"
        and float(getattr(env.config, "exposure_bonus_lambda", 0.0)) != 0.0
    )

    phi_before = 0.0
    exp_denom = 1.0
    if use_exposure:
        phi_before = float(env._compute_peak_exposure_potential(env._current_eval))
        exp_denom = float(getattr(env, "_initial_exposure_potential", 1.0)) + float(
            getattr(env.config, "exposure_eps", 1e-8)
        )
        if (not np.isfinite(exp_denom)) or exp_denom <= 0.0:
            exp_denom = 1.0

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

    def _expected_reward(
        *, accept_flag: bool, new_cost: Optional[float], phi_after: Optional[float]
    ) -> float:
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
        elif mode == "dense_best_exposure":
            exp_r_cur = p * (current_cost_before - new_cost_f) / denom
            exp_r_best = p * max(best_cost_before - new_cost_f, 0.0) / denom
            base = (
                exp_r_cur + float(env.config.best_bonus_lambda) * exp_r_best
            ) * float(env.config.improvement_scale)
            if use_exposure and phi_after is not None and np.isfinite(float(phi_after)):
                r_exp = p * (float(phi_before) - float(phi_after)) / float(exp_denom)
                reward = base + float(env.config.exposure_bonus_lambda) * r_exp
            else:
                reward = base
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
        new_phi: Optional[float] = None

        if decoded.action_type == ActionType.OPERATOR:
            if decoded.operator_id is not None:
                result = env._apply_operator(decoded.operator_id)
            else:
                result = env._apply_best_operator()
            if result is not None:
                _move, c, move_eval = result
                new_cost = float(c)
                if use_exposure and move_eval is not None:
                    try:
                        # Reconstruct the per-machine schedule for the new solution
                        # using affected machine evals (mirrors env incremental update).
                        updated_per_machine = list(env._current_eval.per_machine)
                        if hasattr(move_eval, "affected_machine_evals"):
                            for mi, me in move_eval.affected_machine_evals.items():
                                updated_per_machine[int(mi)] = me
                        from PaST.neurols.move_evaluator import FullEvaluation

                        full_new = FullEvaluation(
                            total_energy=float(new_cost),
                            makespan=int(getattr(move_eval, "new_makespan", 0)),
                            feasible=bool(getattr(move_eval, "feasible", True)),
                            per_machine=updated_per_machine,
                            per_job_costs=None,
                        )
                        new_phi = float(env._compute_peak_exposure_potential(full_new))
                    except Exception:
                        new_phi = None

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
                    if use_exposure:
                        try:
                            new_phi = float(
                                env._compute_peak_exposure_potential(cached[1])
                            )
                        except Exception:
                            new_phi = None
                else:
                    new_eval = env._evaluator.evaluate_solution(new_solution)
                    new_cost = float(new_eval.total_energy)
                    if use_exposure:
                        try:
                            new_phi = float(
                                env._compute_peak_exposure_potential(new_eval)
                            )
                        except Exception:
                            new_phi = None

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
                    if use_exposure:
                        try:
                            new_phi = float(
                                env._compute_peak_exposure_potential(cached[1])
                            )
                        except Exception:
                            new_phi = None
                else:
                    new_eval = env._evaluator.evaluate_solution(new_solution)
                    new_cost = float(new_eval.total_energy)
                    if use_exposure:
                        try:
                            new_phi = float(
                                env._compute_peak_exposure_potential(new_eval)
                            )
                        except Exception:
                            new_phi = None

        accept_reward = None
        reject_reward = None
        for a in gen_to_actions[gen_key]:
            a_decoded = env.config.action_space.decode_action(a)
            r = _expected_reward(
                accept_flag=bool(a_decoded.accept), new_cost=new_cost, phi_after=new_phi
            )
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


def _oracle_multi_step(
    env,
    *,
    rng: random.Random,
    horizon: int,
    tie_eps: float = 1e-12,
    exposure_lambda: float = 0.0,
    hours_per_day: int = 20,
) -> Tuple[int, Dict[str, float], List[int]]:
    """Multi-step oracle label based on recovery after applying a generator.

    Motivation: a 1-step improvement oracle collapses to "reject" and ties in
    locally-stuck states. Here we score each generator by how much it improves
    best_cost after:
      1) forcing one generator step to be accepted (even if worsening)
      2) running greedy improvement-only operator steps for the remaining horizon

    Returns:
      (best_generator_index, stats, best_generator_set)
    """

    from PaST.neurols.action_space import ActionType
    from PaST.neurols.perturbations import PerturbationID

    if horizon <= 1:
        return _oracle_one_step(env, rng=rng, verbose=False)

    if env.state is None:
        raise RuntimeError("Env must be reset() before oracle evaluation")

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

    # Precompute accept-operator actions for greedy improvement steps.
    operator_accept_actions: List[int] = []
    for a in range(n_actions):
        d = env.config.action_space.decode_action(a)
        if bool(d.accept) and d.action_type == ActionType.OPERATOR:
            operator_accept_actions.append(int(a))

    best_cost_before = float(env.state.best_cost)
    current_cost_before = float(env.state.current_cost)
    struct_before = _compute_struct_metrics(env, hours_per_day=int(hours_per_day))
    phi_before = 0.0
    exp_denom = 1.0
    use_exp = float(exposure_lambda) != 0.0 and hasattr(
        env, "_compute_peak_exposure_potential"
    )
    if use_exp:
        try:
            phi_before = float(env._compute_peak_exposure_potential(env._current_eval))
            exp_denom = float(getattr(env, "_initial_exposure_potential", 1.0)) + float(
                getattr(env.config, "exposure_eps", 1e-8)
            )
            if (not np.isfinite(exp_denom)) or exp_denom <= 0.0:
                exp_denom = 1.0
        except Exception:
            use_exp = False
    denom = float(env._initial_cost) + float(env.config.reward_eps)
    if (not np.isfinite(denom)) or denom <= 0.0:
        denom = 1.0

    gen_scores = np.full((len(gen_keys),), -np.inf, dtype=np.float64)

    # Structural deltas (after rollout minus before). Not all metrics exist in all modes.
    struct_delta_cross = np.full((len(gen_keys),), np.nan, dtype=np.float64)
    struct_delta_cheap = np.full((len(gen_keys),), np.nan, dtype=np.float64)
    struct_delta_loadcv = np.full((len(gen_keys),), np.nan, dtype=np.float64)

    for gi, gen_key in enumerate(gen_keys):
        # Pick a representative ACCEPT action for this generator if available.
        candidates = gen_to_actions[gen_key]
        a_rep = None
        for a in candidates:
            if bool(env.config.action_space.decode_action(a).accept):
                a_rep = int(a)
                break
        if a_rep is None:
            a_rep = int(candidates[0])

        snap = _snapshot_env(env)

        # Force acceptance for the first step (even if worsening), so we can
        # evaluate basin quality rather than only immediate improvements.
        old_use_acc = bool(env.config.use_acceptance)
        old_det = bool(env.config.deterministic)
        try:
            env.config.use_acceptance = False
            env.config.deterministic = False
            _s, _r, _done, _info = env.step(a_rep)

            # Greedy improvement-only rollout using operator actions.
            env.config.use_acceptance = True
            env.config.deterministic = True

            for _ in range(int(horizon) - 1):
                if not operator_accept_actions:
                    break

                cur_cost = float(env.state.current_cost)
                best_a = None
                best_new_cost = cur_cost

                for a_op in operator_accept_actions:
                    d = env.config.action_space.decode_action(a_op)
                    if d.operator_id is None:
                        result = env._apply_best_operator()
                    else:
                        result = env._apply_operator(d.operator_id)
                    if result is None:
                        continue
                    _move, new_cost, _me = result
                    if float(new_cost) < float(best_new_cost):
                        best_new_cost = float(new_cost)
                        best_a = int(a_op)

                if best_a is None:
                    break
                if float(best_new_cost) >= float(cur_cost):
                    break

                _s, _r, done, _info = env.step(best_a)
                if done:
                    break

            # Score by dense_best-style improvement after horizon.
            # This is much less degenerate than best_cost-only, because it
            # distinguishes worsening vs improving trajectories.
            after_cur = float(env.state.current_cost)
            after_best = float(env.state.best_cost)

            struct_after = _compute_struct_metrics(
                env, hours_per_day=int(hours_per_day)
            )
            if (
                "cross_day_job_frac" in struct_before
                and "cross_day_job_frac" in struct_after
            ):
                struct_delta_cross[gi] = float(
                    struct_after["cross_day_job_frac"]
                    - struct_before["cross_day_job_frac"]
                )
            if "cheap_time_frac" in struct_before and "cheap_time_frac" in struct_after:
                struct_delta_cheap[gi] = float(
                    struct_after["cheap_time_frac"] - struct_before["cheap_time_frac"]
                )
            if "load_cv_pt" in struct_before and "load_cv_pt" in struct_after:
                struct_delta_loadcv[gi] = float(
                    struct_after["load_cv_pt"] - struct_before["load_cv_pt"]
                )

            r_cur = (current_cost_before - after_cur) / float(denom)
            r_best = max(best_cost_before - after_best, 0.0) / float(denom)
            score = float(
                r_cur + float(getattr(env.config, "best_bonus_lambda", 0.3)) * r_best
            )
            if use_exp:
                try:
                    phi_after = float(
                        env._compute_peak_exposure_potential(env._current_eval)
                    )
                    r_exp = (phi_before - phi_after) / float(exp_denom)
                    if np.isfinite(float(r_exp)):
                        score = float(score + float(exposure_lambda) * float(r_exp))
                except Exception:
                    pass
            if not np.isfinite(score):
                score = -np.inf
            gen_scores[gi] = float(score)
        finally:
            env.config.use_acceptance = old_use_acc
            env.config.deterministic = old_det
            _restore_env(env, snap)

    if gen_scores.size == 0:
        gen_scores = np.zeros((1,), dtype=np.float64)

    finite_mask = np.isfinite(gen_scores)
    if not bool(np.any(finite_mask)):
        # Degenerate case: every generator produced a non-finite score.
        # Fall back to all-zeros so downstream summary stats stay finite.
        gen_scores = np.zeros_like(gen_scores, dtype=np.float64)
        finite_mask = np.isfinite(gen_scores)

    # Use finite-only max for best selection.
    best_r = float(np.max(gen_scores[finite_mask]))
    best_gen_idxs = np.flatnonzero(
        finite_mask & (np.abs(gen_scores - best_r) <= float(tie_eps))
    )

    if best_gen_idxs.size == 0:
        best_gen = 0
    elif best_gen_idxs.size == 1:
        best_gen = int(best_gen_idxs[0])
    else:
        best_gen = int(rng.choice([int(x) for x in best_gen_idxs.tolist()]))

    best_gen_set = [int(idx) for idx in best_gen_idxs.tolist()]

    finite_scores = gen_scores[finite_mask]
    sorted_scores = np.sort(finite_scores) if finite_scores.size else np.array([0.0])
    second_r = (
        float(sorted_scores[-2])
        if sorted_scores.size >= 2
        else float(sorted_scores[-1])
    )
    median_r = float(np.median(finite_scores)) if finite_scores.size else 0.0
    mean_r = float(np.mean(finite_scores)) if finite_scores.size else 0.0
    nonzero_frac = float(np.mean(finite_scores > 0.0)) if finite_scores.size else 0.0

    stats = {
        "best_reward": float(best_r),
        "second_best_reward": float(second_r),
        "margin_best_second": float(best_r - second_r),
        "margin_best_median": float(best_r - median_r),
        "adv_best_minus_mean": float(best_r - mean_r),
        "mean_reward": float(mean_r),
        "median_reward": float(median_r),
        "nonzero_reward_frac": float(nonzero_frac),
        "nonzero_action_frac": float(0.0),
        "n_generators": float(len(gen_keys)),
        "tie_best_count": float(best_gen_idxs.size),
        "tie_best": float(1.0 if best_gen_idxs.size > 1 else 0.0),
        "tie_best_action": float(0.0),
        "tie_best_nonzero": float(
            1.0 if (best_gen_idxs.size > 1 and best_r > 0.0) else 0.0
        ),
        "best_is_zero": float(1.0 if best_r <= 0.0 else 0.0),
        "all_zero": float(1.0 if (best_r <= 0.0 and nonzero_frac == 0.0) else 0.0),
        "accept_better_rate": float(0.0),
        "reject_better_rate": float(0.0),
        "accept_tie_rate": float(0.0),
    }

    # Add interpretable structural deltas for the best generator and for typical generators.
    # Conventions:
    # - cross_day_job_frac: negative delta is better (fewer cross-day jobs)
    # - cheap_time_frac: positive delta is better (more time in cheap slots)
    # - load_cv_pt: negative delta is better (more balanced processing-time assignment)
    try:
        finite_cross = np.isfinite(struct_delta_cross)
        finite_cheap = np.isfinite(struct_delta_cheap)
        finite_load = np.isfinite(struct_delta_loadcv)

        if best_gen_idxs.size and bool(np.any(finite_cross)):
            stats["struct_delta_cross_day_best"] = float(
                struct_delta_cross[int(best_gen)]
            )
            stats["struct_delta_cross_day_mean"] = float(
                np.mean(struct_delta_cross[finite_cross])
            )
            stats["struct_delta_cross_day_median"] = float(
                np.median(struct_delta_cross[finite_cross])
            )

        if best_gen_idxs.size and bool(np.any(finite_cheap)):
            stats["struct_delta_cheap_time_best"] = float(
                struct_delta_cheap[int(best_gen)]
            )
            stats["struct_delta_cheap_time_mean"] = float(
                np.mean(struct_delta_cheap[finite_cheap])
            )
            stats["struct_delta_cheap_time_median"] = float(
                np.median(struct_delta_cheap[finite_cheap])
            )

        if best_gen_idxs.size and bool(np.any(finite_load)):
            stats["struct_delta_load_cv_pt_best"] = float(
                struct_delta_loadcv[int(best_gen)]
            )
            stats["struct_delta_load_cv_pt_mean"] = float(
                np.mean(struct_delta_loadcv[finite_load])
            )
            stats["struct_delta_load_cv_pt_median"] = float(
                np.median(struct_delta_loadcv[finite_load])
            )
    except Exception:
        pass

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
    proxy_mode: Optional[str] = None,
    reward_mode_override: Optional[str] = None,
    exposure_bonus_lambda_override: Optional[float] = None,
    step_penalty_override: Optional[float] = None,
    n_jobs_override: Optional[int] = None,
    n_machines_override: Optional[int] = None,
    collect_mode: str = "random_walk",
    disruptor_mode: str = "either",
    disrupt_steps: int = 1,
    oracle_horizon: int = 1,
    tie_eps: float = 1e-12,
    oracle_exposure_lambda: float = 0.0,
    force_topk: bool = False,
    n_jobs_topk_threshold_override: Optional[int] = None,
    max_moves_full_override: Optional[int] = None,
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

    reward_mode = (
        str(reward_mode_override)
        if reward_mode_override is not None
        else str(cfg.get("reward_mode", "dense_best"))
    )

    exposure_bonus_lambda = (
        float(exposure_bonus_lambda_override)
        if exposure_bonus_lambda_override is not None
        else float(cfg.get("exposure_bonus_lambda", 0.0))
    )

    exposure_eps = float(cfg.get("exposure_eps", 1e-8))

    proxy_mode_val = (
        str(proxy_mode)
        if proxy_mode is not None
        else str(cfg.get("proxy_mode", "load"))
    )

    # Use small fixed instance sizes for speed (override-able for medium/large)
    n_jobs_list = cfg.get("n_jobs_train", [20])
    n_machines_list = cfg.get("n_machines_train", [3])
    fixed_n = (
        int(n_jobs_override) if n_jobs_override is not None else int(n_jobs_list[0])
    )
    fixed_m = (
        int(n_machines_override)
        if n_machines_override is not None
        else int(n_machines_list[0])
    )

    step_penalty = (
        float(step_penalty_override)
        if step_penalty_override is not None
        else float(cfg.get("step_penalty", 0.0))
    )

    env_config = EnvConfig(
        max_steps=int(cfg.get("max_steps", 200)),
        stagnation_limit=int(cfg.get("stagnation_limit", 100)),
        action_space=get_action_space(action_space_name),
        reward_mode=reward_mode,
        improvement_scale=float(cfg.get("improvement_scale", 1.0)),
        reward_eps=float(cfg.get("reward_eps", 1e-8)),
        best_bonus_lambda=float(cfg.get("best_bonus_lambda", 0.3)),
        step_penalty=step_penalty,
        top_k=int(top_k),
        use_proxy=bool(use_proxy),
        proxy_mode=proxy_mode_val,
        exposure_bonus_lambda=float(exposure_bonus_lambda),
        exposure_eps=float(exposure_eps),
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

    # Guardrail: 1-step oracle is known to undervalue delayed-payoff generators.
    # If the action space includes perturb/destroy generators and horizon==1,
    # print a concise warning so results aren't misinterpreted as "not learnable".
    warn_myopic = bool(int(oracle_horizon) <= 1)
    warned_myopic = False

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

        if warn_myopic and not warned_myopic:
            try:
                from PaST.neurols.action_space import ActionType

                has_delayed = False
                for a in range(int(env.action_space_size)):
                    d = env.config.action_space.decode_action(int(a))
                    if d.action_type in (ActionType.PERTURBATION, ActionType.DESTROY):
                        has_delayed = True
                        break
                if has_delayed:
                    print(
                        f"[{key}] Note: --oracle-horizon=1 is myopic; perturb/destroy "
                        f"generators can look unpromising short-term. Consider "
                        f"--oracle-horizon 5-10 (and optionally --collect-mode perturb_then_label)."
                    )
            except Exception:
                pass
            warned_myopic = True

        # CandidateGen overrides (force top-k/proxy behavior when requested)
        try:
            if getattr(env, "_candidate_gen", None) is not None:
                if bool(force_topk):
                    env._candidate_gen.config.n_jobs_topk_threshold = 0
                    env._candidate_gen.config.max_moves_full = 0
                if n_jobs_topk_threshold_override is not None:
                    env._candidate_gen.config.n_jobs_topk_threshold = int(
                        n_jobs_topk_threshold_override
                    )
                if max_moves_full_override is not None:
                    env._candidate_gen.config.max_moves_full = int(
                        max_moves_full_override
                    )
        except Exception:
            pass

        # Precompute disruptor actions for perturb_then_label.
        from PaST.neurols.action_space import ActionType
        from PaST.neurols.perturbations import PerturbationID

        disruptor_actions: List[int] = []
        if str(collect_mode).lower() == "perturb_then_label":
            for a in range(int(env.action_space_size)):
                d = env.config.action_space.decode_action(int(a))
                if not bool(d.accept):
                    continue
                if d.action_type == ActionType.PERTURBATION:
                    if (
                        d.perturbation_id is None
                        or d.perturbation_id == PerturbationID.NONE
                    ):
                        continue
                    disruptor_actions.append(int(a))
                elif d.action_type == ActionType.DESTROY:
                    disruptor_actions.append(int(a))

        base_snap = _snapshot_env(env)

        for _t in range(states_per_instance):
            if str(collect_mode).lower() == "perturb_then_label" and disruptor_actions:
                # Restore to a consistent baseline, then apply a random disruptor with forced acceptance.
                _restore_env(env, base_snap)
                old_use_acc = bool(env.config.use_acceptance)
                old_det = bool(env.config.deterministic)
                try:
                    env.config.use_acceptance = False
                    env.config.deterministic = False
                    done = False
                    for _ in range(max(1, int(disrupt_steps))):
                        a_disrupt = int(inst_rng.choice(disruptor_actions))
                        _s, _r, done, _info = env.step(a_disrupt)
                        if done:
                            break
                    if done:
                        continue
                finally:
                    env.config.use_acceptance = old_use_acc
                    env.config.deterministic = old_det

            features = env.get_state_features()

            if int(oracle_horizon) > 1:
                best_action, stats, best_set = _oracle_multi_step(
                    env,
                    rng=inst_rng,
                    horizon=int(oracle_horizon),
                    tie_eps=float(tie_eps),
                    exposure_lambda=float(oracle_exposure_lambda),
                    hours_per_day=int(hours_per_day),
                )
            else:
                best_action, stats, best_set = _oracle_one_step(
                    env, rng=inst_rng, verbose=verbose
                )

            X_rows.append(_vectorize_features(features))
            y_rows.append(best_action)
            oracle_stats.append(stats)
            best_sets.append(best_set)

            if str(collect_mode).lower() == "random_walk":
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
        "--n-jobs",
        type=int,
        default=None,
        help="Override number of jobs (lets you test medium/large without editing YAML).",
    )
    parser.add_argument(
        "--n-machines",
        type=int,
        default=None,
        help="Override number of machines (lets you test medium/large without editing YAML).",
    )
    parser.add_argument(
        "--collect-mode",
        type=str,
        default="random_walk",
        choices=["random_walk", "perturb_then_label"],
        help="How to sample states: random SA walk vs perturb-then-label recovery states.",
    )
    parser.add_argument(
        "--disrupt-steps",
        type=int,
        default=1,
        help="Number of forced-accept disruptors to apply before labeling (perturb_then_label only).",
    )
    parser.add_argument(
        "--oracle-horizon",
        type=int,
        default=1,
        help=(
            "Oracle lookahead horizon. 1 = one-step expected reward (fast, myopic). "
            ">1 = multi-step recovery score (forces one generator accept + greedy operator rollout), "
            "better for delayed-payoff perturb/destroy generators."
        ),
    )
    parser.add_argument(
        "--oracle-exposure-lambda",
        type=float,
        default=0.0,
        help="Optional extra exposure-based term in the multi-step oracle score (0 disables).",
    )
    parser.add_argument(
        "--tie-eps",
        type=float,
        default=1e-12,
        help="Epsilon for considering generator scores tied (multi-step oracle).",
    )
    parser.add_argument(
        "--force-topk",
        action="store_true",
        help="Force top-k selection to activate (sets candidate_gen threshold/full cap aggressively).",
    )
    parser.add_argument(
        "--n-jobs-topk-threshold",
        type=int,
        default=None,
        help="Override candidate_gen n_jobs_topk_threshold (lower to activate proxy earlier).",
    )
    parser.add_argument(
        "--max-moves-full",
        type=int,
        default=None,
        help="Override candidate_gen max_moves_full (lower to activate top-k earlier).",
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Disable proxy ranking in candidate generation (slower).",
    )
    parser.add_argument(
        "--proxy-mode",
        type=str,
        default=None,
        choices=["load", "price_aware"],
        help="Proxy mode override (default: from variant config).",
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
        "--reward-mode",
        type=str,
        default=None,
        help="Reward mode override (e.g., dense_best, dense_best_exposure).",
    )
    parser.add_argument(
        "--exposure-bonus-lambda",
        type=float,
        default=None,
        help="Override exposure shaping coefficient (dense_best_exposure only).",
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
            proxy_mode=args.proxy_mode,
            reward_mode_override=args.reward_mode,
            exposure_bonus_lambda_override=args.exposure_bonus_lambda,
            step_penalty_override=args.step_penalty,
            n_jobs_override=args.n_jobs,
            n_machines_override=args.n_machines,
            collect_mode=str(args.collect_mode),
            disrupt_steps=int(args.disrupt_steps),
            oracle_horizon=int(args.oracle_horizon),
            tie_eps=float(args.tie_eps),
            oracle_exposure_lambda=float(args.oracle_exposure_lambda),
            force_topk=bool(args.force_topk),
            n_jobs_topk_threshold_override=args.n_jobs_topk_threshold,
            max_moves_full_override=args.max_moves_full,
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

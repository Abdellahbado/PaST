"""Statistical significance analysis for Q_SEQUENCE vs SPT/LPT (and Q-value sanity).

Goal
----
Help answer: is Q_Seq genuinely better than SPT/LPT, or is the observed mean
improvement just sampling noise?

This script:
1) Evaluates Q_Seq (Greedy+DP, optional SGBS+DP) and baselines (SPT+DP, LPT+DP)
   on the same instances.
2) Runs paired, nonparametric significance tests via bootstrap CIs and paired
   sign-flip permutation tests (no SciPy dependency).
3) Optionally probes whether the Q-values are informative by comparing predicted
   Q(s,a) against DP-computed costs for candidate actions at sampled states.

Outputs
-------
- CSV with per-instance energies (and ratios).
- JSON with summary statistics and p-values.

Example
-------
python -m PaST.cli.analysis.analyze_q_sequence_significance \
  --checkpoint PaST/runs_p100/ppo_q_seq/checkpoints/best.pt \
  --scale small --num_instances 256 --eval_seed 42 \
  --beta 4 --gamma 4 --device cuda \
  --qvalue_probe --qvalue_states_per_instance 3 --qvalue_max_actions 16
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from PaST.baselines_sequence_dp import dp_schedule_for_job_sequence, spt_lpt_with_dp
from PaST.config import VariantID, get_variant_config
from PaST.q_sequence_model import build_q_model, QSequenceNet
from PaST.cli.eval.run_eval_eas_ppo_short_base import (
    batch_from_episodes,
)
from PaST.cli.eval.run_eval_q_sequence import (
    greedy_decode_q_sequence,
    sgbs_q_sequence,
    _load_checkpoint as _load_q_ckpt,
    _extract_q_model_state,
    _slice_single_instance as _slice_single_instance_any,
)
from PaST.sm_benchmark_data import (
    generate_raw_instance,
    simulate_metaheuristic_assignment,
    make_single_machine_episode,
)
from PaST.sequence_env import GPUBatchSequenceEnv


# ----------------------------
# Utilities
# ----------------------------


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p

    p2 = Path.cwd() / path_str
    if p2.exists():
        return p2

    p3 = Path.cwd() / "PaST" / path_str
    if p3.exists():
        return p3

    parts = list(p.parts)
    if parts and parts[0] == "PaST":
        p4 = Path.cwd() / Path(*parts[1:])
        if p4.exists():
            return p4

    return p


def _ratios_from_arg(arg: str) -> List[float]:
    items = [x.strip() for x in arg.split(",") if x.strip()]
    out: List[float] = []
    for x in items:
        v = float(x)
        if v < 0.0 or v > 1.0:
            raise ValueError(f"ratio must be in [0,1], got {v}")
        out.append(v)
    if not out:
        raise ValueError("ratios list is empty")
    return out


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to write")

    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _finite(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x)


# ----------------------------
# Paired statistics (no SciPy)
# ----------------------------


def _bootstrap_mean_ci(
    diffs: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 5000,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean of diffs."""
    diffs = diffs[_finite(diffs)]
    n = diffs.size
    if n == 0:
        return float("nan"), float("nan")

    idx = rng.integers(0, n, size=(int(n_boot), n))
    means = diffs[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def _paired_sign_flip_pvalue(
    diffs: np.ndarray,
    rng: np.random.Generator,
    n_perm: int = 10000,
    alternative: str = "greater",
) -> float:
    """Paired permutation test using random sign flips.

    diffs = baseline - model (so >0 means model is better)

    alternative:
      - 'greater': test mean(diffs) > 0
      - 'two-sided': test |mean(diffs)| > 0
    """
    diffs = diffs[_finite(diffs)]
    n = diffs.size
    if n == 0:
        return float("nan")

    obs = float(diffs.mean())

    # Generate sign matrices: (n_perm, n) of +/-1.
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(int(n_perm), n))
    means = (signs * diffs.reshape(1, -1)).mean(axis=1)

    if alternative == "greater":
        # Under null, mean is symmetric around 0.
        p = (np.sum(means >= obs) + 1.0) / (means.size + 1.0)
        return float(p)

    if alternative == "two-sided":
        p = (np.sum(np.abs(means) >= abs(obs)) + 1.0) / (means.size + 1.0)
        return float(p)

    raise ValueError("alternative must be 'greater' or 'two-sided'")


def _summary_stats(model: np.ndarray, baseline: np.ndarray) -> Dict[str, Any]:
    model = np.asarray(model, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)

    mask = _finite(model) & _finite(baseline)
    model = model[mask]
    baseline = baseline[mask]

    diffs = baseline - model
    out: Dict[str, Any] = {
        "n": int(diffs.size),
        "mean_model": float(model.mean()) if diffs.size else float("nan"),
        "mean_baseline": float(baseline.mean()) if diffs.size else float("nan"),
        "mean_diff_baseline_minus_model": (
            float(diffs.mean()) if diffs.size else float("nan")
        ),
        "median_diff_baseline_minus_model": (
            float(np.median(diffs)) if diffs.size else float("nan")
        ),
        "win_rate_model_beats_baseline": (
            float(np.mean(diffs > 0)) if diffs.size else float("nan")
        ),
    }

    # Effect size (paired Cohen's d): mean(diff)/std(diff)
    if diffs.size >= 2:
        sd = float(diffs.std(ddof=1))
        out["cohens_d_paired"] = float(diffs.mean() / sd) if sd > 0 else float("inf")
    else:
        out["cohens_d_paired"] = float("nan")

    return out


# ----------------------------
# Rank correlation helpers
# ----------------------------


def _rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    """Rank data with average ranks for ties (1..n)."""
    x = np.asarray(x)
    n = x.size
    if n == 0:
        return x.astype(np.float64)

    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        # average rank for ties
        avg = 0.5 * ((i + 1) + (j + 1))
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    rx = _rankdata_average_ties(x)
    ry = _rankdata_average_ties(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denom == 0.0:
        return float("nan")
    return float((rx * ry).sum() / denom)


# ----------------------------
# Evaluation
# ----------------------------


def _generate_episodes_by_ratio(
    *,
    scale: str,
    num_instances: int,
    eval_seed: int,
    ratios: Sequence[float],
) -> List[List[Any]]:
    """Generate a fixed set of instances and derive episodes per ratio."""
    dummy_cfg = get_variant_config(VariantID.PPO_SHORT_BASE)
    if scale == "small":
        dummy_cfg.data.T_max_choices = [
            t for t in dummy_cfg.data.T_max_choices if int(t) <= 100
        ]
    elif scale == "medium":
        dummy_cfg.data.T_max_choices = [
            t for t in dummy_cfg.data.T_max_choices if 100 < int(t) <= 350
        ]
    else:
        dummy_cfg.data.T_max_choices = dummy_cfg.data.T_max_choices

    py_rng = random.Random(int(eval_seed))

    base_instances = []
    for _ in range(int(num_instances)):
        raw = generate_raw_instance(dummy_cfg.data, py_rng)
        assignments = simulate_metaheuristic_assignment(raw.n, raw.m, py_rng)
        non_empty = [idx for idx, a in enumerate(assignments) if len(a) > 0]
        m_idx = py_rng.choice(non_empty) if non_empty else 0
        base_instances.append(
            {"raw": raw, "m_idx": m_idx, "job_idxs": assignments[m_idx]}
        )

    episodes_by_ratio = []
    for r_idx, ratio in enumerate(ratios):
        eps = []
        for inst_idx, b in enumerate(base_instances):
            inst_seed = int(eval_seed) + int(inst_idx) + (int(r_idx) * 1000)
            rng_ep = random.Random(inst_seed)
            ep = make_single_machine_episode(
                b["raw"],
                b["m_idx"],
                b["job_idxs"],
                rng_ep,
                deadline_slack_ratio_min=float(ratio),
                deadline_slack_ratio_max=float(ratio),
            )
            eps.append(ep)
        episodes_by_ratio.append(eps)

    return episodes_by_ratio


def _load_q_model(
    checkpoint: Path,
    variant_id: VariantID,
    device: torch.device,
) -> QSequenceNet:
    var_cfg = get_variant_config(variant_id)
    model = build_q_model(var_cfg)

    ckpt = _load_q_ckpt(checkpoint, device)
    state = _extract_q_model_state(ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def evaluate_and_collect(
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    device = torch.device(args.device)
    ratios = _ratios_from_arg(args.ratios)

    # Evaluate Q_Seq
    q_variant = VariantID(args.variant_id)
    q_cfg = get_variant_config(q_variant)

    ckpt_path = _resolve_path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))

    print(f"Loading Q_Seq checkpoint: {ckpt_path}")
    model = _load_q_model(ckpt_path, q_variant, device)

    episodes_by_ratio = _generate_episodes_by_ratio(
        scale=args.scale,
        num_instances=args.num_instances,
        eval_seed=args.eval_seed,
        ratios=ratios,
    )

    rows: List[Dict[str, Any]] = []

    # Baselines use the batched episode representation directly.
    base_env_cfg = get_variant_config(VariantID.PPO_SHORT_BASE).env

    for ratio, episodes in zip(ratios, episodes_by_ratio):
        batch = batch_from_episodes(episodes, N_job_pad=int(q_cfg.env.N_job_pad))

        # Q_Seq greedy + DP
        t0 = time.time()
        q_greedy = greedy_decode_q_sequence(model, q_cfg, batch, device)
        t_q_greedy = time.time() - t0

        q_sgbs = None
        t_q_sgbs = None
        if args.sgbs:
            t0 = time.time()
            q_sgbs = sgbs_q_sequence(
                model, q_cfg, batch, device, beta=int(args.beta), gamma=int(args.gamma)
            )
            t_q_sgbs = time.time() - t0

        # Baselines (SPT/LPT + DP)
        t0 = time.time()
        spt_res = spt_lpt_with_dp(base_env_cfg, device, batch, which="spt")
        lpt_res = spt_lpt_with_dp(base_env_cfg, device, batch, which="lpt")
        t_base = time.time() - t0

        B = len(episodes)
        for i in range(B):
            row = {
                "instance_idx": int(i),
                "ratio": float(ratio),
                "q_seq_greedy_dp": float(q_greedy[i].total_energy),
                "spt_dp": float(spt_res[i].total_energy),
                "lpt_dp": float(lpt_res[i].total_energy),
                "time_q_greedy_s": float(t_q_greedy / max(1, B)),
                "time_baselines_s": float(t_base / max(1, B)),
            }
            if args.sgbs and q_sgbs is not None and t_q_sgbs is not None:
                row["q_seq_sgbs_dp"] = float(q_sgbs[i].total_energy)
                row["time_q_sgbs_s"] = float(t_q_sgbs / max(1, B))
            rows.append(row)

    # Compute stats
    stats: Dict[str, Any] = {
        "config": {
            "checkpoint": str(ckpt_path),
            "variant_id": args.variant_id,
            "scale": args.scale,
            "num_instances": int(args.num_instances),
            "eval_seed": int(args.eval_seed),
            "ratios": list(ratios),
            "sgbs": bool(args.sgbs),
            "beta": int(args.beta),
            "gamma": int(args.gamma),
        },
        "performance": {},
    }

    rng = np.random.default_rng(int(args.eval_seed) + 123)

    # Group by ratio and overall.
    def _extract(col: str, ratio_value: Optional[float]) -> np.ndarray:
        vals = [
            r[col]
            for r in rows
            if ratio_value is None or float(r["ratio"]) == float(ratio_value)
        ]
        return np.asarray(vals, dtype=np.float64)

    comparisons = [
        ("q_seq_greedy_dp", "spt_dp", "SPT+DP"),
        ("q_seq_greedy_dp", "lpt_dp", "LPT+DP"),
    ]
    if args.sgbs:
        comparisons.extend(
            [
                ("q_seq_sgbs_dp", "spt_dp", "SPT+DP"),
                ("q_seq_sgbs_dp", "lpt_dp", "LPT+DP"),
            ]
        )

    for ratio_value in [None] + list(ratios):
        ratio_key = "overall" if ratio_value is None else f"ratio_{ratio_value:.2f}"
        stats["performance"][ratio_key] = {}

        for model_col, base_col, base_name in comparisons:
            model_vals = _extract(model_col, ratio_value)
            base_vals = _extract(base_col, ratio_value)

            diffs = base_vals - model_vals
            out = _summary_stats(model_vals, base_vals)
            out["baseline"] = base_name
            out["model_metric"] = model_col

            # Sign-flip permutation test for paired mean difference.
            out["p_perm_mean_diff_greater"] = _paired_sign_flip_pvalue(
                diffs, rng, n_perm=int(args.n_perm), alternative="greater"
            )
            out["p_perm_mean_diff_two_sided"] = _paired_sign_flip_pvalue(
                diffs, rng, n_perm=int(args.n_perm), alternative="two-sided"
            )

            lo, hi = _bootstrap_mean_ci(
                diffs, rng, n_boot=int(args.n_boot), alpha=float(args.alpha)
            )
            out["mean_diff_ci"] = [lo, hi]

            stats["performance"][ratio_key][f"{model_col}_vs_{base_col}"] = out

    # Optional Q-value probe
    if args.qvalue_probe:
        stats["qvalue_probe"] = qvalue_probe(
            model=model,
            q_cfg=q_cfg,
            device=device,
            episodes_by_ratio=episodes_by_ratio,
            ratios=ratios,
            states_per_instance=int(args.qvalue_states_per_instance),
            max_actions=int(args.qvalue_max_actions),
            completion=str(args.qvalue_completion),
            seed=int(args.eval_seed) + 999,
            n_perm=int(args.qvalue_n_perm),
        )

    return rows, stats


# ----------------------------
# Q-value probe
# ----------------------------


def _complete_remaining_jobs(
    remaining: List[int],
    p_subset: np.ndarray,
    rng: random.Random,
    completion: str,
) -> List[int]:
    if completion == "spt":
        return sorted(remaining, key=lambda j: (int(p_subset[j]), int(j)))
    if completion == "lpt":
        return sorted(remaining, key=lambda j: (-int(p_subset[j]), int(j)))
    if completion == "random":
        rem = list(remaining)
        rng.shuffle(rem)
        return rem
    raise ValueError("completion must be one of: spt, lpt, random")


def _cost_by_model_completion(
    *,
    model: QSequenceNet,
    q_cfg,
    device: torch.device,
    single_np: Dict[str, Any],
    partial_actions: List[int],
    candidate_action: int,
) -> float:
    """Compute cost for a branch by completing greedily with the model.

    This defines a *policy-consistent* target: take (partial_actions + candidate_action),
    then follow greedy argmin-Q until the sequence is complete, and compute DP energy.
    """
    env2 = GPUBatchSequenceEnv(batch_size=1, env_config=q_cfg.env, device=device)
    obs2 = env2.reset(single_np)

    # Replay the prefix.
    for a in partial_actions:
        obs2, _r, done2, _info = env2.step(torch.tensor([int(a)], device=device))
        if bool(done2[0].item()):
            break

    # Take the branch action.
    obs2, _r, done2, _info = env2.step(
        torch.tensor([int(candidate_action)], device=device)
    )
    if bool(done2[0].item()):
        # Terminal reward contains -cost.
        return float((-_r[0]).item())

    # Complete greedily.
    last_cost: Optional[float] = None
    for _ in range(int(env2.N_job_pad) + 5):
        obs = obs2
        with torch.no_grad():
            jobs = obs["jobs"]
            periods = obs["periods"]
            ctx = obs["ctx"]
            if "action_mask" in obs:
                mask = obs["action_mask"] < 0.5
            else:
                mask = env2.job_available < 0.5

            q = model(jobs, periods, ctx, mask)
            q[0, mask[0]] = float("inf")
            a = int(q.argmin(dim=-1).item())

        obs2, r, done2, _info = env2.step(torch.tensor([a], device=device))
        if bool(done2[0].item()):
            last_cost = float((-r[0]).item())
            break

    if last_cost is None:
        # Should not happen; treat as infeasible.
        return float("inf")
    return last_cost


def qvalue_probe(
    *,
    model: QSequenceNet,
    q_cfg,
    device: torch.device,
    episodes_by_ratio: List[List[Any]],
    ratios: Sequence[float],
    states_per_instance: int,
    max_actions: int,
    completion: str,
    seed: int,
    n_perm: int,
) -> Dict[str, Any]:
    """Probe whether Q-values rank candidate actions consistently with DP costs.

    Notes on comparability
    ----------------------
    - The Spearman rho / top-1 metrics test *ranking* consistency.
    - To test whether Q-values are on the same *scale* as true costs, we also
      compute MAE/RMSE/bias on pooled (q_pred, true_cost) pairs, plus an affine
      calibration (true ≈ a*q + b) to measure whether Q-values are at least
      linearly comparable.
    """
    py_rng = random.Random(int(seed))

    all_rhos: List[float] = []
    all_top1: List[int] = []
    all_k: List[int] = []
    all_q_pairs: List[float] = []
    all_true_pairs: List[float] = []

    # Sample a small subset of ratios for speed if many are provided.
    ratio_indices = list(range(len(ratios)))

    for r_idx in ratio_indices:
        episodes = episodes_by_ratio[r_idx]
        # Limit instances in probe for speed.
        probe_instances = episodes[
            : int(min(len(episodes), max(1, states_per_instance * 32)))
        ]

        batch = batch_from_episodes(probe_instances, N_job_pad=int(q_cfg.env.N_job_pad))

        for inst_idx in range(len(probe_instances)):
            single_np = _slice_single_instance_any(batch, inst_idx)

            # Sequence env gives us the exact observation encoding used by Q_Seq.
            env = GPUBatchSequenceEnv(batch_size=1, env_config=q_cfg.env, device=device)
            obs = env.reset(single_np)

            n_jobs = int(single_np["n_jobs"][0])
            p_subset = single_np["p_subset"][0][:n_jobs]

            # Choose which steps to probe along the model's greedy trajectory.
            if n_jobs <= 1:
                continue
            target_steps = set(
                int(x)
                for x in py_rng.sample(
                    list(range(n_jobs)),
                    k=min(int(states_per_instance), int(n_jobs)),
                )
            )

            # Roll out greedily to reach these states.
            for step in range(n_jobs):
                # Probe at this state before taking the action.
                if step in target_steps:
                    with torch.no_grad():
                        jobs = obs["jobs"]
                        periods = obs["periods"]
                        ctx = obs["ctx"]
                        # action_mask: 1=valid
                        if "action_mask" in obs:
                            mask = obs["action_mask"] < 0.5
                        else:
                            mask = env.job_available < 0.5

                        q = model(jobs, periods, ctx, mask)
                        q_1d = q[0].detach().cpu().numpy()

                    valid = (~mask[0]).detach().cpu().numpy().astype(bool)
                    candidates = [j for j in range(n_jobs) if valid[j]]
                    if len(candidates) < 2:
                        break

                    if len(candidates) > int(max_actions):
                        candidates = py_rng.sample(candidates, int(max_actions))

                    # Predicted Q for candidates
                    q_pred = np.array(
                        [float(q_1d[j]) for j in candidates], dtype=np.float64
                    )

                    # True costs for candidate actions with a fixed completion rule.
                    # For completion='model', we define true cost as the cost after following
                    # the model's own greedy policy to the end (policy-consistent target).
                    true_costs = []
                    for a in candidates:
                        partial = (
                            env.job_sequences[0, : env.step_indices[0]]
                            .detach()
                            .cpu()
                            .tolist()
                        )
                        if completion == "model":
                            c = _cost_by_model_completion(
                                model=model,
                                q_cfg=q_cfg,
                                device=device,
                                single_np=single_np,
                                partial_actions=[int(x) for x in partial],
                                candidate_action=int(a),
                            )
                            true_costs.append(float(c))
                        else:
                            remaining = [j for j in candidates if j != a]
                            tail = _complete_remaining_jobs(
                                remaining, p_subset, py_rng, completion
                            )
                            full_seq = (
                                [int(j) for j in partial]
                                + [int(a)]
                                + [int(j) for j in tail]
                            )
                            res = dp_schedule_for_job_sequence(single_np, full_seq)
                            true_costs.append(float(res.total_energy))
                    true_costs = np.asarray(true_costs, dtype=np.float64)

                    # Collect absolute pairs for calibration metrics.
                    # (Filter non-finite to avoid contaminating stats.)
                    pair_mask = np.isfinite(q_pred) & np.isfinite(true_costs)
                    if bool(np.any(pair_mask)):
                        all_q_pairs.extend([float(x) for x in q_pred[pair_mask]])
                        all_true_pairs.extend([float(x) for x in true_costs[pair_mask]])

                    rho = _spearman_rho(q_pred, true_costs)
                    if np.isfinite(rho):
                        all_rhos.append(float(rho))
                        all_k.append(int(len(candidates)))

                        top1_pred = int(candidates[int(np.argmin(q_pred))])
                        top1_true = int(candidates[int(np.argmin(true_costs))])
                        all_top1.append(1 if top1_pred == top1_true else 0)

                # Take greedy action to move to next state.
                with torch.no_grad():
                    jobs = obs["jobs"]
                    periods = obs["periods"]
                    ctx = obs["ctx"]
                    if "action_mask" in obs:
                        mask = obs["action_mask"] < 0.5
                    else:
                        mask = env.job_available < 0.5

                    q = model(jobs, periods, ctx, mask)
                    q[0, mask[0]] = float("inf")
                    a = int(q.argmin(dim=-1).item())

                obs, _rew, done, _info = env.step(torch.tensor([a], device=device))
                if bool(done[0].item()):
                    break

    # Aggregate probe stats
    rhos = np.asarray(all_rhos, dtype=np.float64)
    top1 = np.asarray(all_top1, dtype=np.float64)
    ks = np.asarray(all_k, dtype=np.float64)

    out: Dict[str, Any] = {
        "n_states": int(rhos.size),
        "spearman_rho_mean": float(np.nanmean(rhos)) if rhos.size else float("nan"),
        "spearman_rho_median": float(np.nanmedian(rhos)) if rhos.size else float("nan"),
        "top1_accuracy": float(np.nanmean(top1)) if top1.size else float("nan"),
        "avg_num_actions": float(np.nanmean(ks)) if ks.size else float("nan"),
        "completion": completion,
    }

    # Absolute comparability / calibration metrics on pooled (q_pred, true_cost) pairs.
    qv = np.asarray(all_q_pairs, dtype=np.float64)
    tv = np.asarray(all_true_pairs, dtype=np.float64)
    if qv.size >= 2:
        err = qv - tv
        out["n_pairs"] = int(qv.size)
        out["mae"] = float(np.mean(np.abs(err)))
        out["rmse"] = float(np.sqrt(np.mean(err * err)))
        out["bias_mean_q_minus_true"] = float(np.mean(err))

        # Pearson correlation (scale-sensitive, unlike Spearman).
        q_std = float(np.std(qv))
        t_std = float(np.std(tv))
        if q_std > 0.0 and t_std > 0.0:
            out["pearson_r"] = float(np.corrcoef(qv, tv)[0, 1])
        else:
            out["pearson_r"] = float("nan")

        # Fit affine calibration: true ≈ a*q + b (least squares).
        q_mean = float(np.mean(qv))
        t_mean = float(np.mean(tv))
        q_var = float(np.mean((qv - q_mean) ** 2))
        if q_var > 0.0:
            cov = float(np.mean((qv - q_mean) * (tv - t_mean)))
            a = cov / q_var
            b = t_mean - a * q_mean
            tv_hat = a * qv + b
            err_cal = tv_hat - tv
            out["calib_affine_a"] = float(a)
            out["calib_affine_b"] = float(b)
            out["mae_affine_calibrated"] = float(np.mean(np.abs(err_cal)))
            out["rmse_affine_calibrated"] = float(np.sqrt(np.mean(err_cal * err_cal)))
        else:
            out["calib_affine_a"] = float("nan")
            out["calib_affine_b"] = float("nan")
            out["mae_affine_calibrated"] = float("nan")
            out["rmse_affine_calibrated"] = float("nan")
    else:
        out["n_pairs"] = int(qv.size)
        out["mae"] = float("nan")
        out["rmse"] = float("nan")
        out["bias_mean_q_minus_true"] = float("nan")
        out["pearson_r"] = float("nan")
        out["calib_affine_a"] = float("nan")
        out["calib_affine_b"] = float("nan")
        out["mae_affine_calibrated"] = float("nan")
        out["rmse_affine_calibrated"] = float("nan")

    # Permutation test: shuffle true costs within each state to create null for mean rho.
    if rhos.size >= 5 and n_perm > 0:
        obs_mean = float(np.nanmean(rhos))
        perm_rng = np.random.default_rng(int(seed) + 12345)

        # We don't store per-state vectors; approximate null by symmetric sign-flip on rhos.
        # This is conservative if the per-state rho distribution is roughly symmetric under null.
        # (Exact within-state shuffles would require storing all per-state cost vectors.)
        signs = perm_rng.choice(
            np.array([-1.0, 1.0], dtype=np.float64), size=(int(n_perm), rhos.size)
        )
        means = (signs * rhos.reshape(1, -1)).mean(axis=1)
        p = (np.sum(means >= obs_mean) + 1.0) / (means.size + 1.0)
        out["p_perm_mean_rho_greater"] = float(p)

        # Chance top-1 rate ~ mean(1/k)
        chance = float(np.mean(1.0 / np.maximum(ks, 1.0))) if ks.size else float("nan")
        out["top1_chance_mean_1_over_k"] = chance

    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--checkpoint",
        type=str,
        default="PaST/runs_p100/ppo_q_seq/checkpoints/best.pt",
        help="Q_Seq checkpoint path",
    )

    p.add_argument(
        "--variant_id",
        type=str,
        default=VariantID.Q_SEQUENCE.value,
        help="VariantID string for Q_Seq (e.g. 'q_sequence', 'q_sequence_cwe')",
    )

    p.add_argument(
        "--scale", type=str, default="small", choices=["small", "medium", "large"]
    )
    p.add_argument("--num_instances", type=int, default=256)
    p.add_argument("--eval_seed", type=int, default=42)
    p.add_argument(
        "--seeds",
        type=str,
        default=None,
        help=(
            "Optional comma-separated list of eval seeds. If provided, runs the full "
            "analysis for each seed and writes an aggregated summary. Example: 1,2,3,4,5"
        ),
    )
    p.add_argument(
        "--ratios",
        type=str,
        default="1.0",
        help="Comma-separated slack ratios in [0,1]. Default is 1.0 (single eval regime).",
    )

    p.add_argument("--sgbs", dest="sgbs", action="store_true", default=False)
    p.add_argument("--beta", type=int, default=4)
    p.add_argument("--gamma", type=int, default=4)

    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    p.add_argument("--out_dir", type=str, default="analysis_out")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n_boot", type=int, default=5000)
    p.add_argument("--n_perm", type=int, default=10000)

    # Q-value probe options
    p.add_argument("--qvalue_probe", action="store_true", default=False)
    p.add_argument("--qvalue_states_per_instance", type=int, default=3)
    p.add_argument("--qvalue_max_actions", type=int, default=16)
    p.add_argument(
        "--qvalue_completion",
        type=str,
        default="model",
        choices=["model", "spt", "lpt", "random"],
        help="How to define the 'true' target when probing Q-values. 'model' is policy-consistent.",
    )
    p.add_argument("--qvalue_n_perm", type=int, default=2000)

    return p


def main() -> None:
    args = build_parser().parse_args()

    out_dir = Path(args.out_dir)
    tag = "sgbs" if args.sgbs else "greedy"

    def _headline(stats_obj: Dict[str, Any]) -> None:
        try:
            overall = stats_obj["performance"]["overall"]
            for k, v in overall.items():
                print(
                    f"\n{k}: mean_diff={v['mean_diff_baseline_minus_model']:.6f} "
                    f"CI={v['mean_diff_ci']} p_perm={v['p_perm_mean_diff_greater']:.4g} "
                    f"win_rate={v['win_rate_model_beats_baseline']:.3f}"
                )
        except Exception:
            return

    # Multi-seed mode
    if args.seeds is not None and str(args.seeds).strip() != "":
        seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
        if not seeds:
            raise ValueError("--seeds was provided but empty")

        combined_rows: List[Dict[str, Any]] = []
        per_seed_summaries: List[Dict[str, Any]] = []

        for s in seeds:
            args_one = argparse.Namespace(**vars(args))
            args_one.eval_seed = int(s)

            print(f"\n=== Seed {s} ===")
            rows, stats = evaluate_and_collect(args_one)

            # Annotate rows with seed for combined CSV.
            for r in rows:
                rr = dict(r)
                rr["eval_seed"] = int(s)
                combined_rows.append(rr)

            out_csv = out_dir / f"qseq_significance_{args.scale}_seed{s}_{tag}.csv"
            out_json = out_dir / f"qseq_significance_{args.scale}_seed{s}_{tag}.json"
            _write_csv(out_csv, rows)
            _write_json(out_json, stats)

            print(f"Saved CSV: {out_csv}")
            print(f"Saved JSON: {out_json}")
            _headline(stats)

            # Extract a compact per-seed summary for aggregation.
            overall = stats.get("performance", {}).get("overall", {})
            for key, v in overall.items():
                per_seed_summaries.append(
                    {
                        "eval_seed": int(s),
                        "comparison": str(key),
                        "mean_diff": float(
                            v.get("mean_diff_baseline_minus_model", float("nan"))
                        ),
                        "p_perm_greater": float(
                            v.get("p_perm_mean_diff_greater", float("nan"))
                        ),
                        "win_rate": float(
                            v.get("win_rate_model_beats_baseline", float("nan"))
                        ),
                        "ci_lo": float(
                            v.get("mean_diff_ci", [float("nan"), float("nan")])[0]
                        ),
                        "ci_hi": float(
                            v.get("mean_diff_ci", [float("nan"), float("nan")])[1]
                        ),
                        "n": int(v.get("n", 0)),
                    }
                )

        # Aggregate across seeds per comparison.
        agg: Dict[str, Any] = {
            "config": {
                "checkpoint": str(_resolve_path(args.checkpoint)),
                "variant_id": args.variant_id,
                "scale": args.scale,
                "num_instances": int(args.num_instances),
                "ratios": _ratios_from_arg(args.ratios),
                "sgbs": bool(args.sgbs),
                "beta": int(args.beta),
                "gamma": int(args.gamma),
                "alpha": float(args.alpha),
                "seeds": seeds,
            },
            "per_seed": per_seed_summaries,
            "aggregate": {},
        }

        by_comp: Dict[str, List[Dict[str, Any]]] = {}
        for r in per_seed_summaries:
            by_comp.setdefault(r["comparison"], []).append(r)

        for comp, items in by_comp.items():
            mean_diffs = np.array([it["mean_diff"] for it in items], dtype=np.float64)
            ps = np.array([it["p_perm_greater"] for it in items], dtype=np.float64)
            wins = np.array([it["win_rate"] for it in items], dtype=np.float64)

            finite = np.isfinite(mean_diffs) & np.isfinite(ps)
            mean_diffs = mean_diffs[finite]
            ps = ps[finite]
            wins = wins[finite]

            agg["aggregate"][comp] = {
                "n_seeds": int(ps.size),
                "mean_of_mean_diffs": (
                    float(np.mean(mean_diffs)) if ps.size else float("nan")
                ),
                "median_of_mean_diffs": (
                    float(np.median(mean_diffs)) if ps.size else float("nan")
                ),
                "median_p": float(np.median(ps)) if ps.size else float("nan"),
                "min_p": float(np.min(ps)) if ps.size else float("nan"),
                "frac_p_below_alpha": (
                    float(np.mean(ps < float(args.alpha))) if ps.size else float("nan")
                ),
                "mean_win_rate": float(np.mean(wins)) if ps.size else float("nan"),
            }

        out_csv_all = out_dir / f"qseq_significance_{args.scale}_seeds_{tag}.csv"
        out_json_all = out_dir / f"qseq_significance_{args.scale}_seeds_{tag}.json"
        _write_csv(out_csv_all, combined_rows)
        _write_json(out_json_all, agg)

        print(f"\nSaved combined CSV: {out_csv_all}")
        print(f"Saved combined JSON: {out_json_all}")
        return

    # Single-seed mode
    rows, stats = evaluate_and_collect(args)

    out_csv = out_dir / f"qseq_significance_{args.scale}_seed{args.eval_seed}_{tag}.csv"
    out_json = (
        out_dir / f"qseq_significance_{args.scale}_seed{args.eval_seed}_{tag}.json"
    )

    _write_csv(out_csv, rows)
    _write_json(out_json, stats)

    print(f"\nSaved CSV: {out_csv}")
    print(f"Saved JSON: {out_json}")
    _headline(stats)


if __name__ == "__main__":
    main()

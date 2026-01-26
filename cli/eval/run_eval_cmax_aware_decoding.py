"""Evaluation script with Cmax-aware decoding for ppo_family variants.

This script implements a capacity-aware decoding rule that optimizes both
energy cost AND makespan (Cmax) by considering remaining cheap period capacity.

**Key Innovation:**
At each decoding step:
1. Calculate remaining work (sum of unscheduled job durations)
2. Calculate remaining cheap capacity (family 0 slots from t to T_limit)
3. If cheap_capacity < remaining_work:
   → Override family choice: use earliest feasible slot to minimize Cmax
4. Else:
   → Use best slot in chosen family (original best-start logic)

This prevents the "clustering at end" problem where jobs pile up in expensive
periods when cheap capacity is exhausted.

Example:
    python -m PaST.cli.eval.run_eval_cmax_aware_decoding \
        --checkpoint runs_p100/ppo_family_best/best.pt \
        --eval_seed 55 \
        --scale large \
        --num_instances 16 \
        --num_viz 4 \
        --slack_ratios "0.22"

Optional decoding:
    Use SGBS (simulation-guided beam search) to produce the action trace:
    python -m PaST.cli.eval.run_eval_cmax_aware_decoding \
        --checkpoint runs_p100/ppo_family_best/best.pt \
        --decoder sgbs --sgbs_beta 16 --sgbs_gamma 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
import yaml

os.environ.setdefault("MPLBACKEND", "Agg")

from PaST.baselines_sequence_dp import (
    DPResult,
    spt_lpt_with_dp,
    spt_sequence,
    lpt_sequence,
    dp_schedule_for_job_sequence,
)
from PaST.config import DataConfig, VariantID, get_variant_config
from PaST.past_sm_model import build_model
from PaST.sgbs import greedy_decode, sgbs, DecodeResult
from PaST.sm_benchmark_data import (
    generate_episode_batch,
    generate_raw_instance,
    simulate_metaheuristic_assignment,
    make_single_machine_episode,
    SingleMachineEpisode,
)
import random


# =============================================================================
# Helper Functions (from run_eval_family_q4_beststart_viz.py)
# =============================================================================


def batch_from_episodes(
    episodes: List[SingleMachineEpisode],
    N_job_pad: int = 50,
    K_period_pad: int = 250,
    T_max_pad: int = 500,
) -> Dict[str, np.ndarray]:
    """Manually batch a list of episodes."""
    batch_size = len(episodes)

    batch = {
        "p_subset": np.zeros((batch_size, N_job_pad), dtype=np.int32),
        "n_jobs": np.zeros((batch_size,), dtype=np.int32),
        "job_mask": np.zeros((batch_size, N_job_pad), dtype=np.float32),
        "T_max": np.zeros((batch_size,), dtype=np.int32),
        "T_limit": np.zeros((batch_size,), dtype=np.int32),
        "T_min": np.zeros((batch_size,), dtype=np.int32),
        "ct": np.zeros((batch_size, T_max_pad), dtype=np.int32),
        "Tk": np.zeros((batch_size, K_period_pad), dtype=np.int32),
        "ck": np.zeros((batch_size, K_period_pad), dtype=np.int32),
        "period_starts": np.zeros((batch_size, K_period_pad), dtype=np.int32),
        "K": np.zeros((batch_size,), dtype=np.int32),
        "period_mask": np.zeros((batch_size, K_period_pad), dtype=np.float32),
        "e_single": np.zeros((batch_size,), dtype=np.int32),
        "price_q": np.zeros((batch_size, 3), dtype=np.float32),
    }

    for i, episode in enumerate(episodes):
        n = episode.n_jobs
        k = min(episode.K, K_period_pad)
        t_max = min(episode.T_max, T_max_pad)
        n = min(n, N_job_pad)

        batch["n_jobs"][i] = n
        batch["K"][i] = k
        batch["T_max"][i] = t_max
        batch["T_limit"][i] = episode.T_limit
        batch["T_min"][i] = episode.T_min
        batch["e_single"][i] = episode.e_single

        if n > 0:
            batch["p_subset"][i, :n] = episode.p_subset[:n]
            batch["job_mask"][i, :n] = 1.0

        if k > 0:
            batch["Tk"][i, :k] = episode.Tk[:k]
            batch["ck"][i, :k] = episode.ck[:k]
            batch["period_starts"][i, :k] = episode.period_starts[:k]
            batch["period_mask"][i, :k] = 1.0

        if t_max > 0:
            batch["ct"][i, :t_max] = episode.ct[:t_max]
            ct_valid = episode.ct[:t_max]
            if len(ct_valid) > 0:
                q25, q50, q75 = np.quantile(ct_valid, [0.25, 0.5, 0.75])
                batch["price_q"][i] = [q25, q50, q75]

    return batch


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML at {path}")
    return data


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Invalid checkpoint format at {path}")
    return ckpt


def _extract_model_state(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    if "runner" in ckpt and isinstance(ckpt["runner"], dict):
        runner = ckpt["runner"]
        if "model" in runner and isinstance(runner["model"], dict):
            return runner["model"]
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    raise KeyError("Could not find model state in checkpoint")


def _resolve_run_dir(run_dir_arg: str) -> Path:
    raw = Path(run_dir_arg)
    if raw.exists():
        return raw
    pkg_root = Path(__file__).resolve().parent
    candidate = pkg_root / raw
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Run directory not found: {run_dir_arg}")


def _restrict_data_config(
    data: DataConfig,
    scale: Optional[str],
    T_max: Optional[int] = None,
) -> DataConfig:
    """Restrict data config to a specific instance scale."""
    cfg = replace(data)

    if T_max is not None:
        cfg.T_max_choices = [int(T_max)]
        return cfg

    if scale is None:
        return cfg

    s = scale.lower()
    if s == "small":
        cfg.T_max_choices = [t for t in cfg.T_max_choices if int(t) <= 100]
    elif s in ("medium", "mls"):
        cfg.T_max_choices = [t for t in cfg.T_max_choices if 100 < int(t) <= 350]
    elif s in ("large", "vls"):
        cfg.T_max_choices = [t for t in cfg.T_max_choices if int(t) > 350]
    else:
        raise ValueError(f"scale must be one of: small, medium, large (got {scale})")

    if not cfg.T_max_choices:
        raise ValueError(f"No T_max_choices match scale={scale}.")

    return cfg


def _slice_single_instance(batch_data: Dict[str, Any], index: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch_data.items():
        if isinstance(v, np.ndarray):
            out[k] = v[index : index + 1]
        elif torch.is_tensor(v):
            out[k] = v[index : index + 1]
        else:
            raise TypeError(f"Unsupported batch_data type for key={k}: {type(v)}")
    return out


def _sequence_schedule_to_bars(
    job_sequence: List[int],
    start_times: List[int],
    p_subset: np.ndarray,
) -> List[Dict[str, Any]]:
    """Convert sequence schedule to visualization bars."""
    bars = []
    for j, st in zip(job_sequence, start_times):
        job_id = int(j)
        start = int(st)
        p = int(p_subset[job_id])
        bars.append({"job_id": job_id, "start": start, "end": start + p})
    return bars


# =============================================================================
# CORE: Cmax-Aware Decoding Logic
# =============================================================================


def _action_trace_to_bars_cmax_aware(
    actions: List[int],
    env_config,
    p_subset: np.ndarray,
    ct: np.ndarray,
    e_single: int,
    T_limit: int,
    period_starts: np.ndarray,
    Tk: np.ndarray,
    ck: np.ndarray,
    K: int,
) -> Tuple[List[Dict[str, Any]], float, int]:
    """Convert action trace to bars with Cmax-aware decoding.

    CORRECTED LOGIC:
    - At each step, check if remaining cheap capacity >= remaining work
    - If NO (capacity exhausted): START IMMEDIATELY at time t to minimize Cmax
    - If YES (enough capacity): use best slot in chosen family (can afford to wait)

    The key insight is: when cheap capacity is exhausted, waiting for cheap slots
    just pushes everything later, increasing Cmax without reducing energy.

    Returns:
        bars: list of job bar dicts
        total_energy: total energy cost
        cmax: makespan (completion time of last job)
    """
    num_families = env_config.num_price_families

    # Compute price quantiles to determine slot families
    ct_valid = ct[:T_limit]
    if len(ct_valid) > 0:
        q25, q50, q75 = np.quantile(ct_valid, [0.25, 0.5, 0.75])
    else:
        q25 = q50 = q75 = 0

    # Assign each slot to a family (0=cheapest, 3=most expensive)
    slot_families = np.zeros(len(ct), dtype=np.int32)
    for u in range(len(ct)):
        price = ct[u]
        if price > q75:
            slot_families[u] = 3
        elif price > q50:
            slot_families[u] = 2
        elif price > q25:
            slot_families[u] = 1
        else:
            slot_families[u] = 0

    bars = []
    total_energy = 0.0
    t = 0
    cmax = 0

    # Pre-calculate cumulative sum for O(1) interval cost query
    ct_cumsum = np.zeros(len(ct) + 1, dtype=np.float64)
    ct_cumsum[1:] = np.cumsum(ct)

    def get_interval_cost(start, end):
        s = max(0, min(start, len(ct)))
        e = max(0, min(end, len(ct)))
        return ct_cumsum[e] - ct_cumsum[s]

    # Track remaining work
    n_jobs = len(p_subset)
    remaining_work = int(np.sum(p_subset))
    scheduled_jobs = set()

    for a in actions:
        job_id = int(a) // num_families
        family_id = int(a) % num_families

        # Validate job_id
        if job_id >= n_jobs:
            continue  # Skip invalid actions
        if job_id in scheduled_jobs:
            continue  # Skip already scheduled jobs

        p = int(p_subset[job_id])

        # ------------------------------------
        # CMAX-AWARE CAPACITY CHECK
        # ------------------------------------

        # Calculate remaining cheap capacity (family 0 slots from t to T_limit)
        cheap_capacity = sum(
            1
            for u in range(t, min(T_limit, len(slot_families)))
            if slot_families[u] == 0
        )

        # CRITICAL DECISION:
        # If cheap_capacity < remaining_work, we KNOW some jobs MUST use expensive slots.
        # Waiting for cheap slots only delays everything and increases Cmax.
        # Solution: START IMMEDIATELY to minimize Cmax.

        if cheap_capacity < remaining_work:
            # NOT ENOUGH CHEAP CAPACITY: Start at t immediately to minimize Cmax
            best_start = t
        else:
            # ENOUGH CHEAP CAPACITY: We can afford to wait for a good slot
            # Use best slot in chosen family (original logic)
            best_start = None
            best_cost = float("inf")

            for u in range(t, T_limit - p + 1):
                if slot_families[u] == family_id:
                    cost = get_interval_cost(u, u + p)
                    if cost < best_cost:
                        best_cost = cost
                        best_start = u

            if best_start is None:
                # No slot found in family: start immediately
                best_start = t

        # Ensure feasibility
        start = best_start
        end = start + p
        if end > T_limit:
            # Job doesn't fit - clip and warn
            end = T_limit
            start = max(t, end - p)

        energy = float(e_single) * float(get_interval_cost(start, end))
        total_energy += energy

        bars.append(
            {
                "job_id": job_id,
                "start": start,
                "end": end,
                "family_id": family_id,
                "energy": energy,
                "used_earliest": (cheap_capacity < remaining_work),
            }
        )

        t = end
        cmax = max(cmax, end)
        scheduled_jobs.add(job_id)
        remaining_work -= p

    # Report if not all jobs scheduled
    if len(scheduled_jobs) != n_jobs:
        print(f"WARNING: Scheduled {len(scheduled_jobs)}/{n_jobs} jobs")

    return bars, total_energy, cmax


def _action_trace_to_bars_baseline(
    actions: List[int],
    env_config,
    p_subset: np.ndarray,
    ct: np.ndarray,
    e_single: int,
    T_limit: int,
    period_starts: np.ndarray,
    Tk: np.ndarray,
    ck: np.ndarray,
    K: int,
) -> Tuple[List[Dict[str, Any]], float, int]:
    """Original best-start decoding (baseline for comparison).

    Returns:
        bars: list of job bar dicts
        total_energy: total energy cost
        cmax: makespan
    """
    num_families = env_config.num_price_families

    ct_valid = ct[:T_limit]
    if len(ct_valid) > 0:
        q25, q50, q75 = np.quantile(ct_valid, [0.25, 0.5, 0.75])
    else:
        q25 = q50 = q75 = 0

    slot_families = np.zeros(len(ct), dtype=np.int32)
    for u in range(len(ct)):
        price = ct[u]
        if price > q75:
            slot_families[u] = 3
        elif price > q50:
            slot_families[u] = 2
        elif price > q25:
            slot_families[u] = 1
        else:
            slot_families[u] = 0

    bars = []
    total_energy = 0.0
    t = 0
    cmax = 0

    ct_cumsum = np.zeros(len(ct) + 1, dtype=np.float64)
    ct_cumsum[1:] = np.cumsum(ct)

    def get_interval_cost(start, end):
        s = max(0, min(start, len(ct)))
        e = max(0, min(end, len(ct)))
        return ct_cumsum[e] - ct_cumsum[s]

    n_jobs = len(p_subset)
    remaining_work = int(np.sum(p_subset))
    scheduled_jobs = set()

    for a in actions:
        job_id = int(a) // num_families
        family_id = int(a) % num_families

        # Keep post-processing comparable to cmax-aware:
        # greedy_rollout records actions BEFORE env.step() repair, so traces can contain
        # repeated/unavailable jobs. Skip invalid/repeated selections here.
        if job_id >= n_jobs:
            continue
        if job_id in scheduled_jobs:
            continue

        p = int(p_subset[job_id])

        # Original best-start: find best slot in chosen family.
        # Use a tighter feasible window similar to env's completion feasibility:
        # we should not consider starts that make it impossible to fit the remaining work.
        best_start = None
        best_cost = float("inf")

        max_valid_start = T_limit - remaining_work
        max_valid_start = min(max_valid_start, T_limit - p)
        search_end = max(t, max_valid_start) + 1
        search_end = min(search_end, T_limit - p + 1)

        for u in range(t, search_end):
            if slot_families[u] == family_id:
                cost = get_interval_cost(u, u + p)
                if cost < best_cost:
                    best_cost = cost
                    best_start = u

        if best_start is None:
            best_start = t

        start = best_start
        end = start + p
        if end > T_limit:
            end = T_limit
            start = max(t, end - p)

        energy = float(e_single) * float(get_interval_cost(start, end))
        total_energy += energy

        bars.append(
            {"job_id": job_id, "start": start, "end": end, "family_id": family_id}
        )
        t = end
        cmax = max(cmax, end)
        scheduled_jobs.add(job_id)
        remaining_work -= p

    if len(scheduled_jobs) != n_jobs:
        print(f"WARNING (baseline): Scheduled {len(scheduled_jobs)}/{n_jobs} jobs")

    return bars, total_energy, cmax


# =============================================================================
# Visualization
# =============================================================================


def visualize_schedule(
    out_path: Path,
    instance_data: Dict[str, Any],
    schedules: Dict[str, Dict[str, Any]],
    title_suffix: str = "",
):
    """Create Gantt-style visualization of schedules."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib missing, skipping visualization")
        return

    n_jobs = instance_data["n_jobs"]
    T_limit = instance_data["T_limit"]
    Tk = instance_data["Tk"]
    ck = instance_data["ck"]
    period_starts = instance_data["period_starts"]
    K = instance_data["K"]

    # Get unique price levels for coloring
    unique_prices = sorted(set(int(ck[k]) for k in range(K)))
    price_to_color = {}
    cmap = plt.cm.RdYlGn_r
    for i, p in enumerate(unique_prices):
        price_to_color[p] = cmap(i / max(1, len(unique_prices) - 1))

    methods = list(schedules.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(
        n_methods + 1, 1, figsize=(14, 2 * (n_methods + 1)), sharex=True
    )
    if n_methods + 1 == 1:
        axes = [axes]

    # Price periods row
    ax0 = axes[0]
    for k in range(K):
        x0 = period_starts[k]
        width = Tk[k]
        price = int(ck[k])
        color = price_to_color.get(price, "gray")
        ax0.barh(
            0, width, left=x0, height=0.8, color=color, edgecolor="black", linewidth=0.5
        )
        if width > 3:
            ax0.text(
                x0 + width / 2, 0, f"p={price}", ha="center", va="center", fontsize=7
            )

    ax0.set_yticks([])
    ax0.set_ylabel("Prices")
    ax0.set_xlim(0, T_limit + 5)

    # Legend for prices
    handles = [
        mpatches.Patch(color=price_to_color[p], label=f"price={p}")
        for p in unique_prices
    ]
    ax0.legend(handles=handles, loc="upper right", fontsize=7, ncol=len(unique_prices))

    # Job colors
    job_colors = plt.cm.tab20(np.linspace(0, 1, max(n_jobs, 1)))

    # Schedule rows
    for i, method in enumerate(methods):
        ax = axes[i + 1]
        sched = schedules[method]
        bars = sched.get("bars", [])
        energy = sched.get("energy", 0)
        cmax = sched.get("cmax", 0)
        complete = sched.get("complete", True)

        for bar in bars:
            job_id = bar["job_id"]
            start = bar["start"]
            end = bar["end"]
            color = job_colors[job_id % len(job_colors)]
            ax.barh(
                0, end - start, left=start, height=0.8, color=color, edgecolor="black"
            )
            ax.text(
                start + (end - start) / 2,
                0,
                f"j{job_id}",
                ha="center",
                va="center",
                fontsize=7,
            )

        status = "" if complete else " [INCOMPLETE]"
        ax.set_ylabel(f"{method}\n(E={energy:.1f}, Cmax={cmax}){status}", fontsize=8)
        ax.set_yticks([])
        ax.axvline(T_limit, color="red", linestyle="--", linewidth=1, label="T_limit")

    axes[-1].set_xlabel("Time")
    fig.suptitle(
        f"Schedule Comparison | n_jobs={n_jobs} T_limit={T_limit}{title_suffix}",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# =============================================================================
# Main Evaluation Loop
# =============================================================================


def run_evaluation(
    args,
    variant_config,
    data_cfg,
    model,
    device,
    scale_str: str,
):
    """Run evaluation comparing Cmax-aware vs baseline decoding."""
    print("\n" + "=" * 70)
    decoder_name = str(getattr(args, "decoder", "greedy"))
    print(f"CMAX-AWARE DECODING EVALUATION | decoder={decoder_name}")
    print("=" * 70)

    out_dir = Path(args.out_dir) if args.out_dir else Path("test_results_cmax_aware")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(args.eval_seed)
    py_rng = random.Random(args.eval_seed)

    # Parse slack ratios
    if args.slack_ratios:
        ratios = [float(x) for x in args.slack_ratios.split(",")]
    else:
        ratios = [0.2, 0.3, 0.4, 0.5]

    results = []

    for ratio in ratios:
        print(f"\n--- Slack Ratio: {ratio:.2f} ---")

        # Generate instances
        episodes = []
        for _ in range(args.num_instances):
            raw = generate_raw_instance(data_cfg, py_rng)
            assignments = simulate_metaheuristic_assignment(raw.n, raw.m, py_rng)
            non_empty = [i for i, a in enumerate(assignments) if len(a) > 0]
            m_idx = py_rng.choice(non_empty) if non_empty else 0
            job_idxs = assignments[m_idx]

            ep = make_single_machine_episode(
                raw,
                m_idx,
                job_idxs,
                py_rng,
                deadline_slack_ratio_min=ratio,
                deadline_slack_ratio_max=ratio,
            )
            episodes.append(ep)

        batch = batch_from_episodes(
            episodes,
            N_job_pad=int(variant_config.env.N_job_pad),
            K_period_pad=250,
            T_max_pad=500,
        )

        # Run decoding to get action traces (greedy or SGBS)
        if args.decoder == "sgbs":
            decode_res = sgbs(
                model,
                variant_config.env,
                device,
                batch,
                beta=int(args.sgbs_beta),
                gamma=int(args.sgbs_gamma),
                max_depth_steps=(
                    int(args.sgbs_max_depth_steps)
                    if args.sgbs_max_depth_steps is not None
                    else None
                ),
                max_wait_slots=(
                    int(args.max_wait_slots)
                    if args.max_wait_slots is not None
                    else None
                ),
                wait_logit_penalty=float(args.wait_logit_penalty),
                makespan_penalty=float(args.makespan_penalty),
            )
        else:
            decode_res = greedy_decode(
                model,
                variant_config.env,
                device,
                batch,
                max_wait_slots=(
                    int(args.max_wait_slots)
                    if args.max_wait_slots is not None
                    else None
                ),
                wait_logit_penalty=float(args.wait_logit_penalty),
                makespan_penalty=float(args.makespan_penalty),
            )

        # Run DP baselines
        spt_res = spt_lpt_with_dp(variant_config.env, device, batch, which="spt")
        lpt_res = spt_lpt_with_dp(variant_config.env, device, batch, which="lpt")

        # Process each instance
        for i in range(len(episodes)):
            single = _slice_single_instance(batch, i)
            n_jobs = int(single["n_jobs"][0])
            T_limit = int(single["T_limit"][0])
            p_subset = single["p_subset"][0][:n_jobs].astype(np.int32)
            ct = single["ct"][0]
            e_single = int(single["e_single"][0])

            if decode_res[i].actions is None:
                continue

            # Baseline decoding
            bars_baseline, energy_baseline, cmax_baseline = (
                _action_trace_to_bars_baseline(
                    decode_res[i].actions,
                    variant_config.env,
                    p_subset,
                    ct,
                    e_single,
                    T_limit,
                    single["period_starts"][0],
                    single["Tk"][0],
                    single["ck"][0],
                    int(single["K"][0]),
                )
            )

            # Cmax-aware decoding
            bars_cmax, energy_cmax, cmax_cmax = _action_trace_to_bars_cmax_aware(
                decode_res[i].actions,
                variant_config.env,
                p_subset,
                ct,
                e_single,
                T_limit,
                single["period_starts"][0],
                single["Tk"][0],
                single["ck"][0],
                int(single["K"][0]),
            )

            results.append(
                {
                    "instance_idx": i,
                    "slack_ratio": ratio,
                    "n_jobs": n_jobs,
                    "T_limit": T_limit,
                    "decoder": args.decoder,
                    "decoder_total_energy": float(
                        getattr(decode_res[i], "total_energy", float("nan"))
                    ),
                    "sgbs_beta": (
                        int(args.sgbs_beta) if args.decoder == "sgbs" else None
                    ),
                    "sgbs_gamma": (
                        int(args.sgbs_gamma) if args.decoder == "sgbs" else None
                    ),
                    "baseline_energy": energy_baseline,
                    "baseline_cmax": cmax_baseline,
                    "cmax_aware_energy": energy_cmax,
                    "cmax_aware_cmax": cmax_cmax,
                    "spt_dp_energy": spt_res[i].total_energy,
                    "lpt_dp_energy": lpt_res[i].total_energy,
                    "energy_diff": energy_cmax - energy_baseline,
                    "cmax_diff": cmax_cmax - cmax_baseline,
                }
            )

            # Visualize first few
            if i < args.num_viz:
                schedules = {
                    "Cmax-Aware": {
                        "energy": energy_cmax,
                        "cmax": cmax_cmax,
                        "bars": bars_cmax,
                        "complete": len(bars_cmax) == n_jobs,
                    },
                    "Baseline": {
                        "energy": energy_baseline,
                        "cmax": cmax_baseline,
                        "bars": bars_baseline,
                        "complete": len(bars_baseline) == n_jobs,
                    },
                    "SPT+DP": {
                        "energy": spt_res[i].total_energy,
                        "cmax": (
                            max(
                                (st + int(p_subset[j]))
                                for j, st in zip(
                                    spt_res[i].job_sequence, spt_res[i].start_times
                                )
                            )
                            if spt_res[i].start_times
                            else 0
                        ),
                        "bars": _sequence_schedule_to_bars(
                            spt_res[i].job_sequence, spt_res[i].start_times, p_subset
                        ),
                        "complete": True,
                    },
                    "LPT+DP": {
                        "energy": lpt_res[i].total_energy,
                        "cmax": (
                            max(
                                (st + int(p_subset[j]))
                                for j, st in zip(
                                    lpt_res[i].job_sequence, lpt_res[i].start_times
                                )
                            )
                            if lpt_res[i].start_times
                            else 0
                        ),
                        "bars": _sequence_schedule_to_bars(
                            lpt_res[i].job_sequence, lpt_res[i].start_times, p_subset
                        ),
                        "complete": True,
                    },
                }

                fname = f"cmax_compare_{scale_str}_seed{args.eval_seed}_slack{ratio:.2f}_idx{i}.png"
                visualize_schedule(
                    out_dir / fname,
                    {
                        "n_jobs": n_jobs,
                        "T_limit": T_limit,
                        "Tk": single["Tk"][0],
                        "ck": single["ck"][0],
                        "period_starts": single["period_starts"][0],
                        "K": int(single["K"][0]),
                    },
                    schedules,
                    f" | slack={ratio:.2f} idx={i}",
                )

    # Save results
    import pandas as pd

    df = pd.DataFrame(results)
    csv_path = out_dir / f"cmax_aware_results_{scale_str}_seed{args.eval_seed}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results: {csv_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<20} | {'Avg Energy':<12} | {'Avg Cmax':<10}")
    print("-" * 50)
    print(
        f"{'Baseline':<20} | {df['baseline_energy'].mean():<12.2f} | {df['baseline_cmax'].mean():<10.1f}"
    )
    print(
        f"{'Cmax-Aware':<20} | {df['cmax_aware_energy'].mean():<12.2f} | {df['cmax_aware_cmax'].mean():<10.1f}"
    )
    print(f"{'SPT+DP':<20} | {df['spt_dp_energy'].mean():<12.2f} | -")
    print(f"{'LPT+DP':<20} | {df['lpt_dp_energy'].mean():<12.2f} | -")
    print("=" * 70)

    avg_energy_diff = df["energy_diff"].mean()
    avg_cmax_diff = df["cmax_diff"].mean()
    print(f"\nCmax-Aware vs Baseline:")
    print(f"  Energy difference: {avg_energy_diff:+.2f} (negative = Cmax-Aware better)")
    print(f"  Cmax difference: {avg_cmax_diff:+.1f} (negative = Cmax-Aware better)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Cmax-aware decoding for price-family variants"
    )
    parser.add_argument(
        "--checkpoint", "-c", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--variant", type=str, default="ppo_family_q4_beststart", help="Variant ID"
    )
    parser.add_argument(
        "--eval_seed", type=int, default=55, help="Random seed for evaluation"
    )
    parser.add_argument(
        "--num_instances", type=int, default=32, help="Number of instances to evaluate"
    )
    parser.add_argument(
        "--num_viz",
        type=int,
        default=4,
        help="Number of schedules to visualize per slack ratio",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="large",
        choices=["small", "medium", "large"],
        help="Instance scale",
    )
    parser.add_argument(
        "--slack_ratios",
        type=str,
        default="0.22",
        help="Comma-separated slack ratios to test",
    )
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu or cuda)"
    )

    # Decoder selection
    parser.add_argument(
        "--decoder",
        type=str,
        default="greedy",
        choices=["greedy", "sgbs"],
        help="Decoder used to produce the action trace (greedy or SGBS)",
    )
    parser.add_argument(
        "--sgbs_beta", type=int, default=16, help="SGBS beam width (beta)"
    )
    parser.add_argument(
        "--sgbs_gamma",
        type=int,
        default=8,
        help="SGBS expansion top-k per node (gamma)",
    )
    parser.add_argument(
        "--sgbs_max_depth_steps",
        type=int,
        default=None,
        help="Max rollout steps for SGBS (default: env.N_job_pad + 5)",
    )
    parser.add_argument(
        "--max_wait_slots",
        type=int,
        default=None,
        help="Optional inference-time cap on waiting (non-family variants only)",
    )
    parser.add_argument(
        "--wait_logit_penalty",
        type=float,
        default=0.0,
        help="Optional inference-time logit penalty proportional to waiting",
    )
    parser.add_argument(
        "--makespan_penalty",
        type=float,
        default=0.0,
        help="Optional inference-time return regularizer for makespan during decoding",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load variant config
    variant_id_map = {
        "ppo_family_q4_beststart": VariantID.PPO_FAMILY_Q4_BESTSTART,
        "ppo_family_q4": VariantID.PPO_FAMILY_Q4,
        "ppo_family_ctx13": VariantID.PPO_FAMILY_Q4_CTX13,
    }
    vid = variant_id_map.get(args.variant)
    if vid is None:
        print(f"Unknown variant: {args.variant}, using PPO_FAMILY_Q4_CTX13")
        vid = VariantID.PPO_FAMILY_Q4_CTX13

    variant_config = get_variant_config(vid)

    # Force best_family_start for decoding
    variant_config.env.use_best_family_start = True

    # Restrict data config by scale
    data_cfg = _restrict_data_config(variant_config.data, args.scale)
    scale_str = args.scale or "mixed"

    print(f"Variant: {vid.value}")
    print(f"Scale: {scale_str}")
    print(f"Data config T_max_choices: {data_cfg.T_max_choices}")

    # Load model
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = _load_checkpoint(ckpt_path, device)
    model_state = _extract_model_state(ckpt)

    model = build_model(variant_config)
    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()

    # Run evaluation
    run_evaluation(args, variant_config, data_cfg, model, device, scale_str)


if __name__ == "__main__":
    main()

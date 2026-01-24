"""
Diagnostic Evaluation: Q-Sequence Backbone Comparison (Attention vs CWE).

This script performs deep analysis beyond final energy costs:

1. Q-VALUE ACCURACY
   - How close are predicted Q-values to actual DP costs?
   - MAE, RMSE, Pearson/Spearman correlation
   - Scatter plots: predicted vs actual

2. RANKING QUALITY (Listwise)
   - For each state, sample multiple job choices
   - Do models correctly rank them by true DP cost?
   - Pairwise accuracy, Kendall's tau, NDCG

3. PRICE EXPLOITATION
   - Are jobs scheduled in cheap price periods?
   - Distribution of job placements vs price percentiles
   - Comparison with random baseline

4. SCHEDULE VISUALIZATION
   - Gantt charts with price curve overlay
   - Visual comparison of model decisions

Usage:
    python PaST/run_diagnostic_q_backbones.py --num_instances 16 --num_viz 4
    python PaST/run_diagnostic_q_backbones.py --scale medium --num_ranking_samples 10
"""

import argparse
import os
import sys
import random
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import stats

# Add project root
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PaST.config import VariantID, get_variant_config
from PaST.q_sequence_model import build_q_model
from PaST.batch_dp_solver import BatchSequenceDPSolver
from PaST.sm_benchmark_data import (
    generate_raw_instance,
    simulate_metaheuristic_assignment,
    make_single_machine_episode,
)
from PaST.run_eval_eas_ppo_short_base import batch_from_episodes, _load_checkpoint
from PaST.run_eval_q_sequence import greedy_decode_q_sequence
from PaST.baselines_sequence_dp import _dp_schedule_fixed_order
from PaST.sequence_env import GPUBatchSequenceEnv

# =============================================================================
# Model Configurations
# =============================================================================

MODELS = {
    "Attention": {
        "name": "Q-Seq (Attention)",
        "path": "runs_p100/ppo_q_seq/checkpoints/best.pt",
        "variant_id": VariantID.Q_SEQUENCE,
    },
    "CWE": {
        "name": "Q-Seq (CWE)",
        "path": "runs_p100/q-seq-cwe/checkpoints/checkpoint_55.pt",
        "variant_id": VariantID.Q_SEQUENCE_CWE_CTX13,
    },
}

# =============================================================================
# Helpers
# =============================================================================


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    p_past = Path("PaST") / path_str
    if p_past.exists():
        return p_past
    return p


def safe_extract_state_dict(ckpt: Dict) -> Dict:
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            return ckpt["model_state"]
        if (
            "runner" in ckpt
            and isinstance(ckpt["runner"], dict)
            and "model" in ckpt["runner"]
        ):
            return ckpt["runner"]["model"]
        if "model" in ckpt:
            return ckpt["model"]
        keys = list(ckpt.keys())
        if any(k.startswith("encoder.") or k.startswith("q_head.") for k in keys):
            return ckpt
    raise ValueError("Could not extract state dict from checkpoint")


def load_q_model(model_key: str, device: torch.device):
    cfg = MODELS[model_key]
    var_cfg = get_variant_config(cfg["variant_id"])
    model = build_q_model(var_cfg)

    ckpt_path = resolve_path(cfg["path"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = _load_checkpoint(ckpt_path, device)
    state_dict = safe_extract_state_dict(ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, var_cfg


# =============================================================================
# 1. Q-VALUE ACCURACY ANALYSIS
# =============================================================================


@dataclass
class QValueSample:
    """Single sample of predicted Q vs actual DP cost."""

    model_name: str
    instance_idx: int
    step: int
    job_idx: int
    q_predicted: float
    dp_cost_actual: float


def collect_q_value_samples(
    model,
    var_cfg,
    episodes: List,
    device: torch.device,
    model_name: str,
    num_samples_per_instance: int = 20,
) -> List[QValueSample]:
    """
    Collect Q-value predictions and compare with actual DP costs.

    Uses GPUBatchSequenceEnv for proper observation construction.
    For each step, samples multiple job choices and computes their true DP costs.
    """
    samples = []
    model.eval()

    for inst_idx, ep in enumerate(episodes):
        n_jobs = len(ep.p_subset)
        ct = np.array(ep.ct, dtype=np.float32)
        p_subset = np.array(ep.p_subset, dtype=np.int32)
        e_single = ep.e_single
        T_limit = ep.T_limit

        # Create batch for this instance
        batch = batch_from_episodes([ep], N_job_pad=int(var_cfg.env.N_job_pad))

        # Create sequence environment
        env = GPUBatchSequenceEnv(
            batch_size=1,
            env_config=var_cfg.env,
            device=device,
        )
        obs = env.reset(batch)

        sequence = []
        remaining_jobs = list(range(n_jobs))

        step = 0
        while remaining_jobs and step < num_samples_per_instance:
            # Get observations from environment
            jobs_t = obs["jobs"]  # [1, N_pad, F_job]
            periods_t = obs["periods"]  # [1, K_pad, F_period]
            ctx_t = obs["ctx"]  # [1, F_ctx]

            # Get job mask (1=valid in env, need True=invalid for model)
            job_mask_float = obs.get("job_mask", env.job_available)
            if job_mask_float.dtype != torch.bool:
                job_mask = job_mask_float < 0.5
            else:
                job_mask = ~job_mask_float

            # Get Q-values for current state
            with torch.no_grad():
                q_values = model(
                    jobs=jobs_t,
                    periods_local=periods_t,
                    ctx=ctx_t,
                    job_mask=job_mask,
                )  # [1, N_pad]

            q_np = q_values[0].cpu().numpy()

            # For each available job, record Q-value and compute actual DP cost
            for job_idx in remaining_jobs[: min(5, len(remaining_jobs))]:
                q_pred = float(q_np[job_idx])

                # Complete sequence with this job + remaining by SPT
                test_seq = sequence + [job_idx]
                remaining_after = [j for j in remaining_jobs if j != job_idx]
                remaining_sorted = sorted(remaining_after, key=lambda j: p_subset[j])
                full_seq = test_seq + remaining_sorted

                # Compute actual DP cost
                proc_ordered = [int(p_subset[j]) for j in full_seq]
                dp_cost, _ = _dp_schedule_fixed_order(
                    processing_times=proc_ordered,
                    ct=ct.tolist(),
                    e_single=e_single,
                    T_limit=T_limit,
                )

                samples.append(
                    QValueSample(
                        model_name=model_name,
                        instance_idx=inst_idx,
                        step=step,
                        job_idx=job_idx,
                        q_predicted=q_pred,
                        dp_cost_actual=float(dp_cost),
                    )
                )

            # Greedy selection: pick job with lowest Q
            valid_q = np.full(len(q_np), np.inf)
            for j in remaining_jobs:
                valid_q[j] = q_np[j]
            chosen_job = int(np.argmin(valid_q))

            sequence.append(chosen_job)
            remaining_jobs.remove(chosen_job)

            # Step environment
            action = torch.tensor([chosen_job], device=device)
            obs, _, done, _ = env.step(action)

            if done.all():
                break

            step += 1

    return samples


def analyze_q_accuracy(samples: List[QValueSample], out_dir: Path):
    """Analyze Q-value prediction accuracy."""
    df = pd.DataFrame(
        [
            {
                "model": s.model_name,
                "instance": s.instance_idx,
                "step": s.step,
                "job": s.job_idx,
                "q_pred": s.q_predicted,
                "dp_actual": s.dp_cost_actual,
            }
            for s in samples
        ]
    )

    results = {}

    for model_name in df["model"].unique():
        model_df = df[df["model"] == model_name]

        q_pred = model_df["q_pred"].values
        dp_actual = model_df["dp_actual"].values

        # Filter out inf values
        valid = np.isfinite(q_pred) & np.isfinite(dp_actual)
        q_pred = q_pred[valid]
        dp_actual = dp_actual[valid]

        if len(q_pred) < 2:
            continue

        mae = np.mean(np.abs(q_pred - dp_actual))
        rmse = np.sqrt(np.mean((q_pred - dp_actual) ** 2))
        pearson_r, pearson_p = stats.pearsonr(q_pred, dp_actual)
        spearman_r, spearman_p = stats.spearmanr(q_pred, dp_actual)

        results[model_name] = {
            "MAE": mae,
            "RMSE": rmse,
            "Pearson r": pearson_r,
            "Spearman ρ": spearman_r,
            "N_samples": len(q_pred),
        }

    # Print results
    print("\n" + "=" * 60)
    print("Q-VALUE ACCURACY ANALYSIS")
    print("=" * 60)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    # Scatter plot: Q_pred vs DP_actual
    fig, axes = plt.subplots(
        1, len(df["model"].unique()), figsize=(6 * len(df["model"].unique()), 5)
    )
    if len(df["model"].unique()) == 1:
        axes = [axes]

    colors = {"Q-Seq (Attention)": "#2196F3", "Q-Seq (CWE)": "#4CAF50"}

    for ax, model_name in zip(axes, df["model"].unique()):
        model_df = df[df["model"] == model_name]
        q_pred = model_df["q_pred"].values
        dp_actual = model_df["dp_actual"].values

        valid = np.isfinite(q_pred) & np.isfinite(dp_actual)
        q_pred = q_pred[valid]
        dp_actual = dp_actual[valid]

        ax.scatter(dp_actual, q_pred, alpha=0.5, s=20, c=colors.get(model_name, "#666"))

        # Diagonal
        min_val = min(dp_actual.min(), q_pred.min())
        max_val = max(dp_actual.max(), q_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5)

        ax.set_xlabel("Actual DP Cost", fontsize=11)
        ax.set_ylabel("Predicted Q-Value", fontsize=11)
        ax.set_title(
            f"{model_name}\nPearson r={results[model_name]['Pearson r']:.3f}",
            fontsize=12,
        )
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "q_value_accuracy_scatter.png", dpi=150)
    plt.close()

    return results


# =============================================================================
# 2. RANKING QUALITY ANALYSIS
# =============================================================================


@dataclass
class RankingSample:
    """Ranking data for one state with multiple candidate actions."""

    model_name: str
    instance_idx: int
    step: int
    job_indices: List[int]
    q_values: List[float]
    dp_costs: List[float]


def collect_ranking_samples(
    model,
    var_cfg,
    episodes: List,
    device: torch.device,
    model_name: str,
    num_ranking_samples: int = 5,
) -> List[RankingSample]:
    """
    For each state, collect Q-values and DP costs for ALL available jobs.
    This tests if the model correctly ranks alternatives.
    """
    samples = []
    model.eval()

    for inst_idx, ep in enumerate(episodes):
        n_jobs = len(ep.p_subset)
        ct = np.array(ep.ct, dtype=np.float32)
        p_subset = np.array(ep.p_subset, dtype=np.int32)
        e_single = ep.e_single
        T_limit = ep.T_limit

        batch = batch_from_episodes([ep], N_job_pad=int(var_cfg.env.N_job_pad))

        # Create sequence environment
        env = GPUBatchSequenceEnv(
            batch_size=1,
            env_config=var_cfg.env,
            device=device,
        )
        obs = env.reset(batch)

        sequence = []
        remaining_jobs = list(range(n_jobs))

        step = 0
        while (
            remaining_jobs and step < num_ranking_samples and len(remaining_jobs) >= 2
        ):
            # Get observations from environment
            jobs_t = obs["jobs"]  # [1, N_pad, F_job]
            periods_t = obs["periods"]  # [1, K_pad, F_period]
            ctx_t = obs["ctx"]  # [1, F_ctx]

            # Get job mask (1=valid in env, need True=invalid for model)
            job_mask_float = obs.get("job_mask", env.job_available)
            if job_mask_float.dtype != torch.bool:
                job_mask = job_mask_float < 0.5
            else:
                job_mask = ~job_mask_float

            # Get Q-values
            with torch.no_grad():
                q_values = (
                    model(
                        jobs=jobs_t,
                        periods_local=periods_t,
                        ctx=ctx_t,
                        job_mask=job_mask,
                    )[0]
                    .cpu()
                    .numpy()
                )

            # Collect Q-values and DP costs for all remaining jobs
            job_indices = []
            q_vals = []
            dp_costs = []

            for job_idx in remaining_jobs:
                job_indices.append(job_idx)
                q_vals.append(float(q_values[job_idx]))

                # Compute DP cost: current sequence + this job + SPT for rest
                test_seq = sequence + [job_idx]
                remaining_after = [j for j in remaining_jobs if j != job_idx]
                remaining_sorted = sorted(remaining_after, key=lambda j: p_subset[j])
                full_seq = test_seq + remaining_sorted

                proc_ordered = [int(p_subset[j]) for j in full_seq]
                dp_cost, _ = _dp_schedule_fixed_order(
                    processing_times=proc_ordered,
                    ct=ct.tolist(),
                    e_single=e_single,
                    T_limit=T_limit,
                )
                dp_costs.append(float(dp_cost))

            samples.append(
                RankingSample(
                    model_name=model_name,
                    instance_idx=inst_idx,
                    step=step,
                    job_indices=job_indices,
                    q_values=q_vals,
                    dp_costs=dp_costs,
                )
            )

            # Greedy step
            valid_q = np.full(len(q_values), np.inf)
            for j in remaining_jobs:
                valid_q[j] = q_values[j]
            chosen_job = int(np.argmin(valid_q))

            sequence.append(chosen_job)
            remaining_jobs.remove(chosen_job)

            # Step environment
            action = torch.tensor([chosen_job], device=device)
            obs, _, done, _ = env.step(action)

            if done.all():
                break

            step += 1

    return samples


def analyze_ranking_quality(samples: List[RankingSample], out_dir: Path):
    """Analyze ranking quality using pairwise accuracy and Kendall's tau."""
    results = {}

    for model_name in set(s.model_name for s in samples):
        model_samples = [s for s in samples if s.model_name == model_name]

        pairwise_correct = 0
        pairwise_total = 0
        kendall_taus = []
        top1_correct = 0
        top1_total = 0

        for sample in model_samples:
            if len(sample.job_indices) < 2:
                continue

            q_vals = np.array(sample.q_values)
            dp_costs = np.array(sample.dp_costs)

            # Pairwise accuracy
            for i in range(len(q_vals)):
                for j in range(i + 1, len(q_vals)):
                    # Lower is better for both Q and DP cost
                    q_order = q_vals[i] < q_vals[j]  # True if i is predicted better
                    dp_order = dp_costs[i] < dp_costs[j]  # True if i is actually better

                    if q_order == dp_order:
                        pairwise_correct += 1
                    pairwise_total += 1

            # Kendall's tau
            if len(q_vals) >= 2:
                tau, _ = stats.kendalltau(q_vals, dp_costs)
                if not np.isnan(tau):
                    kendall_taus.append(tau)

            # Top-1 accuracy (does model pick the best job?)
            model_best = np.argmin(q_vals)
            actual_best = np.argmin(dp_costs)
            if model_best == actual_best:
                top1_correct += 1
            top1_total += 1

        results[model_name] = {
            "Pairwise Accuracy": pairwise_correct / max(1, pairwise_total),
            "Kendall's τ": np.mean(kendall_taus) if kendall_taus else 0.0,
            "Top-1 Accuracy": top1_correct / max(1, top1_total),
            "N_states": len(model_samples),
        }

    print("\n" + "=" * 60)
    print("RANKING QUALITY ANALYSIS (Listwise)")
    print("=" * 60)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    # Bar chart comparison
    if len(results) >= 2:
        fig, ax = plt.subplots(figsize=(10, 5))

        metrics_to_plot = ["Pairwise Accuracy", "Kendall's τ", "Top-1 Accuracy"]
        x = np.arange(len(metrics_to_plot))
        width = 0.35

        colors = {"Q-Seq (Attention)": "#2196F3", "Q-Seq (CWE)": "#4CAF50"}

        for i, (model_name, metrics) in enumerate(results.items()):
            values = [metrics.get(m, 0) for m in metrics_to_plot]
            offset = (i - 0.5) * width
            ax.bar(
                x + offset,
                values,
                width,
                label=model_name,
                color=colors.get(model_name, "#666"),
            )

        ax.set_ylabel("Score", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_to_plot)
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.set_title("Ranking Quality Comparison", fontsize=12)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / "ranking_quality.png", dpi=150)
        plt.close()

    return results


# =============================================================================
# 3. PRICE EXPLOITATION ANALYSIS
# =============================================================================


@dataclass
class ScheduleInfo:
    """Info about a scheduled job."""

    model_name: str
    instance_idx: int
    job_idx: int
    processing_time: int
    start_time: int
    end_time: int
    avg_price: float
    price_percentile: float  # 0-100, lower = cheaper


def analyze_price_exploitation(
    model,
    var_cfg,
    episodes: List,
    device: torch.device,
    model_name: str,
) -> List[ScheduleInfo]:
    """
    Run greedy decoding and analyze where jobs are placed relative to prices.
    """
    schedules = []

    results = greedy_decode_q_sequence(
        model,
        var_cfg,
        batch_from_episodes(episodes, N_job_pad=int(var_cfg.env.N_job_pad)),
        device,
    )

    for inst_idx, (ep, res) in enumerate(zip(episodes, results)):
        ct = np.array(ep.ct, dtype=np.float32)
        p_subset = np.array(ep.p_subset, dtype=np.int32)

        # Get job sequence and start times from DPResult
        job_seq = res.job_sequence
        start_times = res.start_times

        # Map jobs to their schedule info
        for i, job_idx in enumerate(job_seq):
            if job_idx >= len(p_subset):
                continue

            proc_time = int(p_subset[job_idx])
            start_t = start_times[i]
            end_t = start_t + proc_time

            # Compute average price during job execution
            if end_t > start_t and end_t <= len(ct):
                avg_price = np.mean(ct[start_t:end_t])
            else:
                avg_price = ct[min(start_t, len(ct) - 1)]

            # Compute price percentile (lower = cheaper)
            price_percentile = stats.percentileofscore(ct, avg_price)

            schedules.append(
                ScheduleInfo(
                    model_name=model_name,
                    instance_idx=inst_idx,
                    job_idx=job_idx,
                    processing_time=int(p_subset[job_idx]),
                    start_time=start_t,
                    end_time=end_t,
                    avg_price=avg_price,
                    price_percentile=price_percentile,
                )
            )

    return schedules


def visualize_price_exploitation(all_schedules: List[ScheduleInfo], out_dir: Path):
    """Visualize how models exploit cheap prices."""
    df = pd.DataFrame(
        [
            {
                "model": s.model_name,
                "instance": s.instance_idx,
                "job": s.job_idx,
                "proc_time": s.processing_time,
                "start": s.start_time,
                "end": s.end_time,
                "avg_price": s.avg_price,
                "price_pctl": s.price_percentile,
            }
            for s in all_schedules
        ]
    )

    print("\n" + "=" * 60)
    print("PRICE EXPLOITATION ANALYSIS")
    print("=" * 60)

    for model_name in df["model"].unique():
        model_df = df[df["model"] == model_name]

        # Weight by processing time (more work in cheap periods = better)
        weighted_pctl = np.average(
            model_df["price_pctl"], weights=model_df["proc_time"]
        )
        mean_pctl = model_df["price_pctl"].mean()

        print(f"\n{model_name}:")
        print(f"  Mean price percentile: {mean_pctl:.1f}% (lower = cheaper)")
        print(f"  Work-weighted percentile: {weighted_pctl:.1f}%")
        print(
            f"  Jobs in cheap periods (≤25%): {(model_df['price_pctl'] <= 25).sum()} / {len(model_df)}"
        )
        print(
            f"  Jobs in expensive periods (≥75%): {(model_df['price_pctl'] >= 75).sum()} / {len(model_df)}"
        )

    # Histogram of price percentiles
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {"Q-Seq (Attention)": "#2196F3", "Q-Seq (CWE)": "#4CAF50"}

    for model_name in df["model"].unique():
        model_df = df[df["model"] == model_name]
        ax.hist(
            model_df["price_pctl"],
            bins=20,
            alpha=0.6,
            label=model_name,
            color=colors.get(model_name, "#666"),
            weights=model_df["proc_time"] / model_df["proc_time"].sum(),
        )

    ax.axvline(
        50, color="red", linestyle="--", alpha=0.7, label="Median (random baseline)"
    )
    ax.set_xlabel("Price Percentile (lower = cheaper)", fontsize=11)
    ax.set_ylabel("Fraction of Work", fontsize=11)
    ax.set_title("Distribution of Work Across Price Percentiles", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "price_exploitation_histogram.png", dpi=150)
    plt.close()

    return df


# =============================================================================
# 4. GANTT CHART VISUALIZATION
# =============================================================================


def plot_gantt_comparison(
    episodes: List,
    models: Dict,
    device: torch.device,
    out_dir: Path,
    num_viz: int = 4,
):
    """Create Gantt charts comparing model schedules with price overlay."""

    out_dir.mkdir(parents=True, exist_ok=True)

    for inst_idx in range(min(num_viz, len(episodes))):
        ep = episodes[inst_idx]
        ct = np.array(ep.ct, dtype=np.float32)
        p_subset = np.array(ep.p_subset, dtype=np.int32)
        n_jobs = len(p_subset)
        T_limit = ep.T_limit

        fig, axes = plt.subplots(
            len(models) + 1, 1, figsize=(14, 3 * (len(models) + 1)), sharex=True
        )

        # Top panel: Price curve
        ax_price = axes[0]
        ax_price.fill_between(range(len(ct)), ct, alpha=0.3, color="orange")
        ax_price.plot(ct, color="darkorange", linewidth=1.5)
        ax_price.set_ylabel("Price", fontsize=10)
        ax_price.set_title(
            f"Instance {inst_idx}: Price Curve (T_limit={T_limit}, n_jobs={n_jobs})",
            fontsize=12,
        )
        ax_price.axvline(
            T_limit, color="red", linestyle="--", alpha=0.7, label="Deadline"
        )
        ax_price.legend(loc="upper right")
        ax_price.grid(alpha=0.3)

        # Normalize prices for coloring
        price_norm = Normalize(vmin=ct.min(), vmax=ct.max())
        cmap = plt.cm.RdYlGn_r  # Red = expensive, Green = cheap

        # Create schedules for each model
        batch_single = batch_from_episodes([ep], N_job_pad=50)

        for ax_idx, (model_key, model_data) in enumerate(models.items()):
            ax = axes[ax_idx + 1]
            model = model_data["model"]
            var_cfg = model_data["var_cfg"]

            # Get greedy schedule - returns DPResult with job_sequence and start_times
            result = greedy_decode_q_sequence(model, var_cfg, batch_single, device)[0]

            # Get job sequence and schedule from DPResult
            job_seq = result.job_sequence
            start_times = result.start_times
            energy = result.total_energy

            # Plot Gantt bars
            for bar_idx, job_idx in enumerate(job_seq):
                start_t = start_times[bar_idx]
                proc_time = int(p_subset[job_idx])
                end_t = start_t + proc_time

                # Color by average price during execution
                if end_t > start_t and end_t <= len(ct):
                    avg_price = np.mean(ct[start_t:end_t])
                else:
                    avg_price = ct[min(start_t, len(ct) - 1)]

                bar_color = cmap(price_norm(avg_price))

                ax.barh(
                    bar_idx,
                    proc_time,
                    left=start_t,
                    height=0.6,
                    color=bar_color,
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax.text(
                    start_t + proc_time / 2,
                    bar_idx,
                    f"J{job_idx}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )

            ax.set_ylabel("Job Order", fontsize=10)
            ax.set_title(
                f"{MODELS[model_key]['name']} (Energy: {energy:.1f})", fontsize=11
            )
            ax.axvline(T_limit, color="red", linestyle="--", alpha=0.7)
            ax.set_yticks(range(n_jobs))
            ax.set_yticklabels([f"{i+1}" for i in range(n_jobs)])
            ax.grid(axis="x", alpha=0.3)

        axes[-1].set_xlabel("Time", fontsize=11)

        # Colorbar
        sm = ScalarMappable(cmap=cmap, norm=price_norm)
        sm.set_array([])
        cbar = fig.colorbar(
            sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.02
        )
        cbar.set_label("Price Level", fontsize=10)

        plt.tight_layout()
        plt.savefig(out_dir / f"gantt_instance_{inst_idx}.png", dpi=150)
        plt.close()

    print(f"\n📊 Gantt charts saved to {out_dir}/")


# =============================================================================
# MAIN
# =============================================================================


def run_diagnostics(args):
    print("\n" + "=" * 70)
    print("Q-SEQUENCE BACKBONE DIAGNOSTIC ANALYSIS")
    print(f"Scale: {args.scale}, Instances: {args.num_instances}")
    print("=" * 70)

    device = torch.device(args.device)
    out_dir = Path(args.out_dir) if args.out_dir else Path("diagnostic_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Models
    print("\n📦 Loading models...")
    models = {}
    for key in ["Attention", "CWE"]:
        try:
            model, var_cfg = load_q_model(key, device)
            models[key] = {
                "model": model,
                "var_cfg": var_cfg,
                "name": MODELS[key]["name"],
            }
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  ✓ {MODELS[key]['name']}: {n_params:,} params")
        except Exception as e:
            print(f"  ✗ {MODELS[key]['name']}: {e}")

    if len(models) == 0:
        print("❌ No models loaded. Exiting.")
        return

    # 2. Generate Instances
    print(f"\n📋 Generating {args.num_instances} instances...")

    base_cfg = get_variant_config(VariantID.Q_SEQUENCE)
    if args.scale == "small":
        T_choices = [t for t in base_cfg.data.T_max_choices if int(t) <= 100]
    elif args.scale == "medium":
        T_choices = [t for t in base_cfg.data.T_max_choices if 100 < int(t) <= 350]
    else:
        T_choices = [t for t in base_cfg.data.T_max_choices if int(t) > 350]

    base_cfg.data.T_max_choices = T_choices

    py_rng = random.Random(args.eval_seed)
    episodes = []

    for i in range(args.num_instances):
        raw = generate_raw_instance(base_cfg.data, py_rng, instance_id=i)
        assignments = simulate_metaheuristic_assignment(raw.n, raw.m, py_rng)
        non_empty = [idx for idx, a in enumerate(assignments) if len(a) > 0]
        m_idx = py_rng.choice(non_empty) if non_empty else 0

        ratio = py_rng.uniform(0.2, 0.8)  # Random deadline tightness
        ep = make_single_machine_episode(
            raw,
            m_idx,
            assignments[m_idx],
            py_rng,
            deadline_slack_ratio_min=ratio,
            deadline_slack_ratio_max=ratio,
        )
        episodes.append(ep)

    # 3. Q-Value Accuracy Analysis
    print("\n🔬 Analyzing Q-value accuracy...")
    all_q_samples = []
    for key, model_data in models.items():
        samples = collect_q_value_samples(
            model_data["model"],
            model_data["var_cfg"],
            episodes[: args.num_instances // 2],
            device,
            model_data["name"],
            num_samples_per_instance=10,
        )
        all_q_samples.extend(samples)

    q_results = analyze_q_accuracy(all_q_samples, out_dir)

    # 4. Ranking Quality Analysis
    print("\n🔬 Analyzing ranking quality...")
    all_ranking_samples = []
    for key, model_data in models.items():
        samples = collect_ranking_samples(
            model_data["model"],
            model_data["var_cfg"],
            episodes[: args.num_instances // 2],
            device,
            model_data["name"],
            num_ranking_samples=args.num_ranking_samples,
        )
        all_ranking_samples.extend(samples)

    ranking_results = analyze_ranking_quality(all_ranking_samples, out_dir)

    # 5. Price Exploitation Analysis
    print("\n🔬 Analyzing price exploitation...")
    all_schedules = []
    for key, model_data in models.items():
        schedules = analyze_price_exploitation(
            model_data["model"],
            model_data["var_cfg"],
            episodes,
            device,
            model_data["name"],
        )
        all_schedules.extend(schedules)

    price_df = visualize_price_exploitation(all_schedules, out_dir)

    # 6. Gantt Visualization
    if args.num_viz > 0:
        print(f"\n📊 Generating {args.num_viz} Gantt chart comparisons...")
        plot_gantt_comparison(
            episodes, models, device, out_dir / "gantt", num_viz=args.num_viz
        )

    # 7. Summary Report
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    summary_data = []
    for model_name in set(s.model_name for s in all_q_samples):
        row = {"Model": model_name}
        if model_name in q_results:
            row.update({f"Q_{k}": v for k, v in q_results[model_name].items()})
        if model_name in ranking_results:
            row.update({f"Rank_{k}": v for k, v in ranking_results[model_name].items()})

        # Price exploitation summary
        model_sched = [s for s in all_schedules if s.model_name == model_name]
        if model_sched:
            row["Price_Mean_Pctl"] = np.mean([s.price_percentile for s in model_sched])

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(out_dir / "diagnostic_summary.csv", index=False)
    print(f"\n💾 Summary saved to {out_dir}/diagnostic_summary.csv")
    print(f"📊 Visualizations saved to {out_dir}/")

    print("\n✅ Diagnostic analysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnostic Q-Sequence Backbone Analysis"
    )

    parser.add_argument(
        "--scale", type=str, default="small", choices=["small", "medium", "large"]
    )
    parser.add_argument("--num_instances", type=int, default=16)
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument(
        "--num_ranking_samples",
        type=int,
        default=5,
        help="Number of decision points per instance for ranking analysis",
    )
    parser.add_argument(
        "--num_viz",
        type=int,
        default=4,
        help="Number of Gantt chart visualizations to generate",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out_dir", type=str, default="diagnostic_out")

    args = parser.parse_args()
    run_diagnostics(args)

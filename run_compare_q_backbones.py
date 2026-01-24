"""
Compare Q-Sequence Backbone Variants: Attention (PaSTEncoder) vs CWE (Sparse).

This script evaluates two Q-Sequence model backbones on the same instances:
1. Attention-based: Full transformer encoder (PaSTEncoder)
2. CWE: Candidate-Window sparse attention encoder

Both use the same DuelingQHead and are trained with Q-learning on DP costs.

Usage:
    # Quick comparison (Greedy only)
    python PaST/run_compare_q_backbones.py --scale small --num_instances 32

    # Full comparison with SGBS
    python PaST/run_compare_q_backbones.py --scale small --num_instances 64 --use_sgbs --beta 4 --gamma 4

    # Visualize specific instances
    python PaST/run_compare_q_backbones.py --scale small --num_instances 8 --visualize
"""

import argparse
import os
import sys
import time
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PaST.config import VariantID, get_variant_config
from PaST.q_sequence_model import build_q_model, QModelWrapper
from PaST.sm_benchmark_data import (
    generate_raw_instance,
    simulate_metaheuristic_assignment,
    make_single_machine_episode,
)
from PaST.run_eval_eas_ppo_short_base import batch_from_episodes, _load_checkpoint
from PaST.run_eval_q_sequence import greedy_decode_q_sequence, sgbs_q_sequence
from PaST.baselines_sequence_dp import spt_lpt_with_dp

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
    """Try to resolve path relative to CWD or CWD/PaST."""
    p = Path(path_str)
    if p.exists():
        return p
    p_past = Path("PaST") / path_str
    if p_past.exists():
        return p_past
    return p


def safe_extract_state_dict(ckpt: Dict) -> Dict:
    """Robust extraction of state dict handling nested keys."""
    if isinstance(ckpt, dict):
        # Q-sequence training checkpoints use "model_state"
        if "model_state" in ckpt:
            return ckpt["model_state"]
        # PPO runner checkpoints
        if (
            "runner" in ckpt
            and isinstance(ckpt["runner"], dict)
            and "model" in ckpt["runner"]
        ):
            return ckpt["runner"]["model"]
        if "model" in ckpt:
            return ckpt["model"]
        # Heuristic: if it looks like a flat state dict
        keys = list(ckpt.keys())
        if any(k.startswith("encoder.") or k.startswith("q_head.") for k in keys):
            return ckpt
    raise ValueError("Could not extract state dict from checkpoint")


def load_q_model(model_key: str, device: torch.device):
    """Load a Q-sequence model by key."""
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
# Evaluation
# =============================================================================


@dataclass
class EvalResult:
    model_name: str
    method: str
    instance_idx: int
    ratio: float
    energy: float
    time_sec: float
    n_jobs: int
    T_limit: int


def evaluate_model(
    model,
    var_cfg,
    episodes: List,
    device: torch.device,
    model_name: str,
    ratio: float,
    use_sgbs: bool = False,
    beta: int = 4,
    gamma: int = 4,
) -> List[EvalResult]:
    """Evaluate a model on a list of episodes."""
    results = []

    batch = batch_from_episodes(episodes, N_job_pad=int(var_cfg.env.N_job_pad))

    # --- Greedy ---
    t0 = time.time()
    greedy_res = greedy_decode_q_sequence(model, var_cfg, batch, device)
    t_greedy = time.time() - t0

    for i, res in enumerate(greedy_res):
        results.append(
            EvalResult(
                model_name=model_name,
                method="Greedy+DP",
                instance_idx=i,
                ratio=ratio,
                energy=res.total_energy,
                time_sec=t_greedy / len(greedy_res),
                n_jobs=len(episodes[i].p_subset),
                T_limit=episodes[i].T_limit,
            )
        )

    # --- SGBS ---
    if use_sgbs:
        t0 = time.time()
        sgbs_res = sgbs_q_sequence(
            model, var_cfg, batch, device, beta=beta, gamma=gamma
        )
        t_sgbs = time.time() - t0

        for i, res in enumerate(sgbs_res):
            results.append(
                EvalResult(
                    model_name=model_name,
                    method=f"SGBS+DP(β{beta}γ{gamma})",
                    instance_idx=i,
                    ratio=ratio,
                    energy=res.total_energy,
                    time_sec=t_sgbs / len(sgbs_res),
                    n_jobs=len(episodes[i].p_subset),
                    T_limit=episodes[i].T_limit,
                )
            )

    return results


def evaluate_baselines(
    episodes: List,
    device: torch.device,
    ratio: float,
) -> List[EvalResult]:
    """Evaluate SPT+DP and LPT+DP baselines."""
    results = []

    base_cfg = get_variant_config(VariantID.Q_SEQUENCE)
    batch = batch_from_episodes(episodes, N_job_pad=int(base_cfg.env.N_job_pad))

    for which in ["spt", "lpt"]:
        t0 = time.time()
        res_list = spt_lpt_with_dp(base_cfg.env, device, batch, which=which)
        t_baseline = time.time() - t0

        for i, res in enumerate(res_list):
            results.append(
                EvalResult(
                    model_name="Baseline",
                    method=f"{which.upper()}+DP",
                    instance_idx=i,
                    ratio=ratio,
                    energy=res.total_energy,
                    time_sec=t_baseline / len(res_list),
                    n_jobs=len(episodes[i].p_subset),
                    T_limit=episodes[i].T_limit,
                )
            )

    return results


# =============================================================================
# Visualization
# =============================================================================


def visualize_comparison(df: pd.DataFrame, out_dir: Path, args):
    """Create comparison visualizations."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Bar chart: Mean energy by model and method
    fig, ax = plt.subplots(figsize=(10, 6))

    summary = df.groupby(["model_name", "method"])["energy"].mean().reset_index()
    summary = summary.sort_values("energy")

    colors = {
        "Q-Seq (Attention)": "#2196F3",
        "Q-Seq (CWE)": "#4CAF50",
        "Baseline": "#9E9E9E",
    }

    x_labels = []
    x_positions = []
    bar_colors = []
    energies = []

    for idx, row in summary.iterrows():
        label = f"{row['model_name']}\n{row['method']}"
        x_labels.append(label)
        x_positions.append(len(x_positions))
        bar_colors.append(colors.get(row["model_name"], "#666666"))
        energies.append(row["energy"])

    bars = ax.bar(
        x_positions, energies, color=bar_colors, edgecolor="black", linewidth=0.5
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean Energy Cost", fontsize=11)
    ax.set_title(
        f"Q-Sequence Backbone Comparison ({args.scale} scale, {args.num_instances} instances)",
        fontsize=12,
    )
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, energies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out_dir / "backbone_comparison_bar.png", dpi=150)
    plt.close()

    # 2. Box plot by model
    fig, ax = plt.subplots(figsize=(10, 6))

    models = df["model_name"].unique()
    model_data = [df[df["model_name"] == m]["energy"].values for m in models]

    bp = ax.boxplot(model_data, labels=models, patch_artist=True)
    for patch, model in zip(bp["boxes"], models):
        patch.set_facecolor(colors.get(model, "#666666"))
        patch.set_alpha(0.7)

    ax.set_ylabel("Energy Cost", fontsize=11)
    ax.set_title("Energy Distribution by Model", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "backbone_comparison_box.png", dpi=150)
    plt.close()

    # 3. Scatter: Attention vs CWE (instance-wise)
    if (
        "Q-Seq (Attention)" in df["model_name"].values
        and "Q-Seq (CWE)" in df["model_name"].values
    ):
        fig, ax = plt.subplots(figsize=(8, 8))

        # Get greedy results for both
        attn_df = df[
            (df["model_name"] == "Q-Seq (Attention)") & (df["method"] == "Greedy+DP")
        ]
        cwe_df = df[(df["model_name"] == "Q-Seq (CWE)") & (df["method"] == "Greedy+DP")]

        if len(attn_df) > 0 and len(cwe_df) > 0:
            merged = attn_df.merge(
                cwe_df, on=["instance_idx", "ratio"], suffixes=("_attn", "_cwe")
            )

            ax.scatter(
                merged["energy_attn"],
                merged["energy_cwe"],
                alpha=0.6,
                s=50,
                c="#2196F3",
                edgecolor="black",
                linewidth=0.5,
            )

            # Diagonal line
            max_val = max(merged["energy_attn"].max(), merged["energy_cwe"].max()) * 1.1
            ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="Equal")

            ax.set_xlabel("Attention Energy", fontsize=11)
            ax.set_ylabel("CWE Energy", fontsize=11)
            ax.set_title("Instance-wise Comparison (Greedy+DP)", fontsize=12)
            ax.legend()
            ax.set_aspect("equal")
            ax.grid(alpha=0.3)

            # Count wins
            attn_wins = (merged["energy_attn"] < merged["energy_cwe"]).sum()
            cwe_wins = (merged["energy_cwe"] < merged["energy_attn"]).sum()
            ties = (merged["energy_attn"] == merged["energy_cwe"]).sum()

            ax.text(
                0.05,
                0.95,
                f"Attention wins: {attn_wins}\nCWE wins: {cwe_wins}\nTies: {ties}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        plt.tight_layout()
        plt.savefig(out_dir / "backbone_scatter.png", dpi=150)
        plt.close()

    print(f"\n📊 Visualizations saved to {out_dir}/")


# =============================================================================
# Main
# =============================================================================


def run_comparison(args):
    print("\n" + "=" * 70)
    print("Q-SEQUENCE BACKBONE COMPARISON: Attention vs CWE")
    print(
        f"Scale: {args.scale}, Instances: {args.num_instances}, Seed: {args.eval_seed}"
    )
    if args.use_sgbs:
        print(f"SGBS: β={args.beta}, γ={args.gamma}")
    print("=" * 70)

    device = torch.device(args.device)

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
        print("\n❌ No models loaded. Exiting.")
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
    instances = []

    for i in range(args.num_instances):
        raw = generate_raw_instance(base_cfg.data, py_rng, instance_id=i)
        assignments = simulate_metaheuristic_assignment(raw.n, raw.m, py_rng)
        non_empty = [idx for idx, a in enumerate(assignments) if len(a) > 0]
        m_idx = py_rng.choice(non_empty) if non_empty else 0
        instances.append({"raw": raw, "m_idx": m_idx, "job_idxs": assignments[m_idx]})

    # 3. Generate Episodes per Ratio
    ratios = np.linspace(0.0, 1.0, 5)
    episodes_by_ratio = []

    for r_idx, ratio in enumerate(ratios):
        current_episodes = []
        for inst_idx, inst in enumerate(instances):
            inst_seed = args.eval_seed + inst_idx + (r_idx * 10000)
            rng_ep = random.Random(inst_seed)
            ep = make_single_machine_episode(
                inst["raw"],
                inst["m_idx"],
                inst["job_idxs"],
                rng_ep,
                deadline_slack_ratio_min=ratio,
                deadline_slack_ratio_max=ratio,
            )
            current_episodes.append(ep)
        episodes_by_ratio.append(current_episodes)

    print(f"  Generated episodes for {len(ratios)} deadline ratios")

    # 4. Evaluate
    all_results = []

    for key, model_data in models.items():
        print(f"\n🔍 Evaluating {model_data['name']}...")

        for r_idx, ratio in enumerate(ratios):
            episodes = episodes_by_ratio[r_idx]
            results = evaluate_model(
                model_data["model"],
                model_data["var_cfg"],
                episodes,
                device,
                model_data["name"],
                ratio,
                use_sgbs=args.use_sgbs,
                beta=args.beta,
                gamma=args.gamma,
            )
            all_results.extend(results)

            # Progress
            greedy_mean = np.mean(
                [r.energy for r in results if r.method == "Greedy+DP"]
            )
            print(f"    Ratio {ratio:.2f}: Greedy mean = {greedy_mean:.2f}")

    # 5. Baselines
    print("\n🔍 Evaluating Baselines (SPT+DP, LPT+DP)...")
    for r_idx, ratio in enumerate(ratios):
        episodes = episodes_by_ratio[r_idx]
        baseline_results = evaluate_baselines(episodes, device, ratio)
        all_results.extend(baseline_results)

    # 6. Compile Results
    df = pd.DataFrame(
        [
            {
                "model_name": r.model_name,
                "method": r.method,
                "instance_idx": r.instance_idx,
                "ratio": r.ratio,
                "energy": r.energy,
                "time_sec": r.time_sec,
                "n_jobs": r.n_jobs,
                "T_limit": r.T_limit,
            }
            for r in all_results
        ]
    )

    # 7. Save Results
    out_dir = Path(args.out_dir) if args.out_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"q_backbone_comparison_{args.scale}_seed{args.eval_seed}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n💾 Results saved to {out_csv}")

    # 8. Summary
    print("\n" + "=" * 70)
    print("SUMMARY (Mean Energy)")
    print("=" * 70)

    summary = (
        df.groupby(["model_name", "method"])["energy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary = summary.sort_values("mean")
    summary.columns = ["Model", "Method", "Mean Energy", "Std", "Count"]
    print(summary.to_string(index=False))

    # Highlight comparison
    if (
        "Q-Seq (Attention)" in df["model_name"].values
        and "Q-Seq (CWE)" in df["model_name"].values
    ):
        attn_greedy = df[
            (df["model_name"] == "Q-Seq (Attention)") & (df["method"] == "Greedy+DP")
        ]["energy"].mean()
        cwe_greedy = df[
            (df["model_name"] == "Q-Seq (CWE)") & (df["method"] == "Greedy+DP")
        ]["energy"].mean()

        print("\n" + "-" * 40)
        print("HEAD-TO-HEAD (Greedy+DP)")
        print("-" * 40)
        print(f"  Attention: {attn_greedy:.2f}")
        print(f"  CWE:       {cwe_greedy:.2f}")

        if cwe_greedy < attn_greedy:
            imp = (attn_greedy - cwe_greedy) / attn_greedy * 100
            print(f"  → CWE wins by {imp:.2f}%")
        elif attn_greedy < cwe_greedy:
            imp = (cwe_greedy - attn_greedy) / cwe_greedy * 100
            print(f"  → Attention wins by {imp:.2f}%")
        else:
            print(f"  → Tie")

    # 9. Visualize
    if args.visualize:
        visualize_comparison(df, out_dir / "viz", args)

    print("\n✅ Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Q-Sequence Backbone Variants")

    parser.add_argument(
        "--scale",
        type=str,
        default="small",
        choices=["small", "medium", "large"],
        help="Instance scale",
    )
    parser.add_argument(
        "--num_instances", type=int, default=32, help="Number of instances to evaluate"
    )
    parser.add_argument(
        "--eval_seed", type=int, default=42, help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--use_sgbs",
        action="store_true",
        help="Also run SGBS decoding (slower but better)",
    )
    parser.add_argument("--beta", type=int, default=4, help="SGBS beam width")
    parser.add_argument("--gamma", type=int, default=4, help="SGBS expansion width")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for evaluation",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None, help="Output directory for results"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Generate comparison visualizations"
    )

    args = parser.parse_args()
    run_comparison(args)

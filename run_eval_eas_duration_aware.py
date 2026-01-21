"""
EAS Evaluation Script for ppo_duration_aware_family variant.

Supports:
- EAS-Lay standalone
- SGBS+EAS hybrid
- Comparison with greedy, SGBS, SPT+DP, LPT+DP
- Epsilon-constraint analysis
- Duration-aware family decoding

Note: This script includes the monkeypatch for correct action mask handling
in duration-aware mode (consistent with run_eval_duration_aware_viz.py).

Example:
    python PaST/run_eval_eas_duration_aware.py \\
        --checkpoint PaST/runs_p100/ppo_duration_aware_family/checkpoints/best.pt \\
        --eval_seed 42 --num_instances 16 \\
        --scale small --method sgbs_eas \\
        --max_iterations 50
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import torch
import yaml

# Set matplotlib backend before import
os.environ.setdefault("MPLBACKEND", "Agg")

from PaST.baselines_sequence_dp import DPResult, spt_lpt_with_dp
from PaST.config import DataConfig, VariantID, get_variant_config
from PaST.past_sm_model import build_model
from PaST.sgbs import greedy_decode, sgbs, DecodeResult
from PaST.eas import EASConfig, eas_batch, EASResult
from PaST.sgbs_eas import SGBSEASConfig, sgbs_eas_batch, SGBSEASResult
from PaST.sm_benchmark_data import (
    generate_raw_instance,
    simulate_metaheuristic_assignment,
    make_single_machine_episode,
    SingleMachineEpisode,
)
import random
import PaST.sgbs as sgbs_module  # For monkeypatch


# =============================================================================
# Monkeypatch for Duration-Aware Action Mask
# =============================================================================


def _monkeypatch_sgbs_for_duration_aware():
    """
    CRITICAL WORKAROUND:
    The default `sgbs._completion_feasible_action_mask` logic supports `use_price_families` (slot-based)
    but assumes a slot-to-family mapping that is incorrect for DURATION-AWARE families (which depend on p).
    It also runs very slowly (O(N*T)) per step.

    Since the duration-aware environment (`SingleMachinePeriodEnv` with `use_duration_aware_families=True`)
    already computes the exact correct action mask in `_get_obs()["action_mask"]`, we can safely
    disable the redundant (and incorrect) extra masking in SGBS for this script.
    """

    def _dummy_mask(env, obs):
        if "action_mask" in obs:
            return obs["action_mask"].float()
        return torch.ones((env.batch_size, env.action_dim), device=env.device)

    print(
        ">> Monkeypatching PaST.sgbs._completion_feasible_action_mask to bypass slow/incorrect logic"
    )
    sgbs_module._completion_feasible_action_mask = _dummy_mask


# Apply patch immediately
_monkeypatch_sgbs_for_duration_aware()


# =============================================================================
# Utility Functions
# =============================================================================


def batch_from_episodes(
    episodes: List[SingleMachineEpisode],
    N_job_pad: int = 50,
    K_period_pad: int = 250,
    T_max_pad: int = 500,
) -> Dict[str, np.ndarray]:
    """Batch a list of episodes into numpy arrays."""
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
        n = min(episode.n_jobs, N_job_pad)
        k = min(episode.K, K_period_pad)
        t_max = min(episode.T_max, T_max_pad)

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
    return data if isinstance(data, dict) else {}


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    return ckpt


def _extract_model_state(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    if "runner" in ckpt and isinstance(ckpt["runner"], dict):
        runner = ckpt["runner"]
        if "model" in runner:
            return runner["model"]
    if "model" in ckpt:
        return ckpt["model"]
    raise KeyError("Could not find model state in checkpoint")


def _restrict_data_config(
    data: DataConfig,
    scale: Optional[str],
) -> DataConfig:
    cfg = replace(data)

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


# =============================================================================
# Main Evaluation Functions
# =============================================================================


def run_evaluation(
    args,
    variant_config,
    data_cfg,
    model,
    device,
    scale_str,
):
    """Run full evaluation with all methods."""
    print("\n" + "=" * 70)
    print(f"EAS EVALUATION (duration_aware) - {args.method.upper()}")
    print(f"Scale: {scale_str}, Instances: {args.num_instances}")
    print("=" * 70)

    # Parse SGBS parameters
    betas = [int(x) for x in str(args.beta).split(",")]
    gammas = [int(x) for x in str(args.gamma).split(",")]
    bg_pairs = list(zip(betas, gammas)) if len(betas) == len(gammas) else [(betas[0], gammas[0])]

    # Setup output directory
    out_dir = Path(args.out_dir) if args.out_dir else (Path(args.checkpoint).parent.parent / "eas_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate instances
    py_rng = random.Random(args.eval_seed)
    ratios = np.linspace(0.0, 1.0, args.epsilon_steps)

    # Generate base instances
    base_instances = []
    for _ in range(args.num_instances):
        raw = generate_raw_instance(data_cfg, py_rng)
        assignments = simulate_metaheuristic_assignment(raw.n, raw.m, py_rng)
        non_empty = [i for i, a in enumerate(assignments) if len(a) > 0]
        m_idx = py_rng.choice(non_empty) if non_empty else 0
        base_instances.append({
            "raw": raw,
            "m_idx": m_idx,
            "job_idxs": assignments[m_idx],
        })

    results = []

    for r_idx, ratio in enumerate(ratios):
        print(f"\nSlack ratio {ratio:.2f} ({r_idx+1}/{len(ratios)})...")

        # Create episodes
        episodes = []
        for b in base_instances:
            ep = make_single_machine_episode(
                b["raw"], b["m_idx"], b["job_idxs"], py_rng,
                deadline_slack_ratio_min=ratio,
                deadline_slack_ratio_max=ratio,
            )
            episodes.append(ep)

        batch = batch_from_episodes(
            episodes,
            N_job_pad=int(variant_config.env.N_job_pad),
        )

        # Run Greedy
        print("  Running Greedy...")
        t0 = time.time()
        greedy_res = greedy_decode(model, variant_config.env, device, batch)
        greedy_time = time.time() - t0

        # Run SGBS
        print(f"  Running SGBS (β={bg_pairs[0][0]}, γ={bg_pairs[0][1]})...")
        t0 = time.time()
        sgbs_res = sgbs(
            model=model,
            env_config=variant_config.env,
            device=device,
            batch_data=batch,
            beta=bg_pairs[0][0],
            gamma=bg_pairs[0][1],
        )
        sgbs_time = time.time() - t0

        # Run EAS or SGBS+EAS
        if args.method == "eas":
            print(f"  Running EAS ({args.max_iterations} iters)...")
            eas_config = EASConfig(
                learning_rate=args.eas_lr,
                il_weight=args.eas_il_weight,
                samples_per_iter=args.samples_per_iter,
                max_iterations=args.max_iterations,
            )
            t0 = time.time()
            eas_res = eas_batch(
                model, variant_config.env, device, batch, eas_config
            )
            eas_time = time.time() - t0

        else:  # sgbs_eas
            print(f"  Running SGBS+EAS ({args.max_iterations} iters)...")
            sgbs_eas_config = SGBSEASConfig(
                sgbs_beta=bg_pairs[0][0],
                sgbs_gamma=bg_pairs[0][1],
                eas_learning_rate=args.eas_lr,
                eas_il_weight=args.eas_il_weight,
                samples_per_iter=args.samples_per_iter,
                max_iterations=args.max_iterations,
            )
            t0 = time.time()
            eas_res = sgbs_eas_batch(
                model, variant_config.env, device, batch, sgbs_eas_config
            )
            eas_time = time.time() - t0

        # Run DP baselines
        print("  Running SPT+DP and LPT+DP...")
        spt_res = spt_lpt_with_dp(variant_config.env, device, batch, which="spt")
        lpt_res = spt_lpt_with_dp(variant_config.env, device, batch, which="lpt")

        # Collect results
        for i in range(len(episodes)):
            row = {
                "instance_idx": i,
                "slack_ratio": float(ratio),
                "T_limit": int(episodes[i].T_limit),
                "n_jobs": int(episodes[i].n_jobs),
                "greedy_energy": greedy_res[i].total_energy,
                "sgbs_energy": sgbs_res[i].total_energy,
                "eas_energy": eas_res[i].best_energy,
                "spt_dp_energy": spt_res[i].total_energy,
                "lpt_dp_energy": lpt_res[i].total_energy,
                "eas_iterations": eas_res[i].iterations,
                "eas_time": eas_res[i].time_seconds,
            }
            results.append(row)

    # Save results
    df = pd.DataFrame(results)
    out_csv = out_dir / f"eas_duration_aware_{scale_str}_seed{args.eval_seed}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}")

    # Print summary
    print_summary(df, args.method)

    # Plot results
    plot_results(df, out_dir, scale_str, args.eval_seed, "duration_aware")


def print_summary(df: pd.DataFrame, method: str):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    methods = ["greedy", "sgbs", "eas", "spt_dp", "lpt_dp"]
    method_names = ["Greedy", "SGBS", method.upper(), "SPT+DP", "LPT+DP"]

    print(f"{'Method':<15} | {'Mean Energy':<12} | {'Improvement vs Greedy':<20}")
    print("-" * 55)

    greedy_mean = df["greedy_energy"].replace([np.inf, -np.inf], np.nan).mean()

    for m, name in zip(methods, method_names):
        col = f"{m}_energy"
        if col in df.columns:
            mean_e = df[col].replace([np.inf, -np.inf], np.nan).mean()
            if greedy_mean > 0 and not np.isnan(mean_e):
                imp = ((greedy_mean - mean_e) / greedy_mean) * 100
                print(f"{name:<15} | {mean_e:<12.2f} | {imp:+.2f}%")
            else:
                print(f"{name:<15} | {mean_e:<12.2f} | -")

    # EAS-specific stats
    if "eas_iterations" in df.columns:
        avg_iters = df["eas_iterations"].mean()
        avg_time = df["eas_time"].mean()
        print(f"\nEAS Stats: {avg_iters:.1f} avg iterations, {avg_time:.2f}s avg time")

    print("=" * 70)


def plot_results(df: pd.DataFrame, out_dir: Path, scale_str: str, seed: int, variant: str):
    """Plot energy vs slack ratio for all methods."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    slacks = sorted(df["slack_ratio"].unique())

    methods = {
        "greedy_energy": ("Greedy", "--", "o"),
        "sgbs_energy": ("SGBS", "-", "s"),
        "eas_energy": ("EAS/SGBS+EAS", "-", "*"),
        "spt_dp_energy": ("SPT+DP", ":", "^"),
        "lpt_dp_energy": ("LPT+DP", ":", "v"),
    }

    for col, (label, ls, marker) in methods.items():
        if col not in df.columns:
            continue

        means = []
        for s in slacks:
            vals = df[df["slack_ratio"] == s][col].replace([np.inf, -np.inf], np.nan)
            means.append(vals.mean())

        ax.plot(slacks, means, label=label, linestyle=ls, marker=marker, markersize=6)

    ax.set_xlabel("Slack Ratio (0=Tightest, 1=Loosest)")
    ax.set_ylabel("Mean Energy")
    ax.set_title(f"EAS Evaluation ({variant}) - {scale_str} instances")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / f"eas_{variant}_{scale_str}_seed{seed}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EAS Evaluation for ppo_duration_aware_family")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")

    # Instance generation
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--num_instances", type=int, default=16)
    parser.add_argument("--scale", type=str, default="small",
                        choices=["small", "medium", "large"])
    parser.add_argument("--epsilon_steps", type=int, default=5,
                        help="Number of slack ratio steps")

    # Method selection
    parser.add_argument("--method", type=str, default="sgbs_eas",
                        choices=["eas", "sgbs_eas"],
                        help="Search method: eas (standalone) or sgbs_eas (hybrid)")

    # SGBS parameters
    parser.add_argument("--beta", type=str, default="4", help="SGBS beam width")
    parser.add_argument("--gamma", type=str, default="4", help="SGBS expansion factor")

    # EAS parameters
    parser.add_argument("--max_iterations", type=int, default=50,
                        help="Max EAS iterations per instance")
    parser.add_argument("--eas_lr", type=float, default=0.003,
                        help="EAS learning rate")
    parser.add_argument("--eas_il_weight", type=float, default=0.01,
                        help="EAS imitation learning weight (lambda)")
    parser.add_argument("--samples_per_iter", type=int, default=32,
                        help="Samples per EAS iteration")

    # Output
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Load config and model
    ckpt_path = Path(args.checkpoint)
    run_dir = ckpt_path.parent.parent
    config_path = run_dir / "config.yaml"

    # Use PPO_DURATION_AWARE_FAMILY variant
    variant_config = get_variant_config(VariantID.PPO_DURATION_AWARE_FAMILY)

    # Restrict to scale
    data_cfg = _restrict_data_config(variant_config.data, args.scale)

    device = torch.device(args.device)

    # Load model
    ckpt = _load_checkpoint(ckpt_path, device)
    model_state = _extract_model_state(ckpt)

    model = build_model(variant_config)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    print(f"Loaded model from {ckpt_path}")
    print(f"Device: {device}")

    # Run evaluation
    run_evaluation(args, variant_config, data_cfg, model, device, args.scale)

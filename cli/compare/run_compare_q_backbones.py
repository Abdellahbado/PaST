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
    episode_to_dict,
)
from PaST.cli.eval.run_eval_eas_ppo_short_base import (
    batch_from_episodes,
    _load_checkpoint,
)
from PaST.cli.eval.run_eval_q_sequence import greedy_decode_q_sequence, sgbs_q_sequence
from PaST.baselines_sequence_dp import spt_lpt_with_dp, dp_schedule_for_job_sequence


def _latest_checkpoint_in_dir(ckpt_dir: Path) -> Optional[Path]:
    """Return latest checkpoint_*.pt in a directory based on the numeric suffix."""
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        return None

    candidates = list(ckpt_dir.glob("checkpoint_*.pt"))
    if not candidates:
        return None

    def _ckpt_num(p: Path) -> int:
        stem = p.stem  # checkpoint_40
        try:
            return int(stem.split("checkpoint_")[-1])
        except Exception:
            return -1

    candidates.sort(key=_ckpt_num)
    return candidates[-1]


def _best_checkpoint_in_dir(ckpt_dir: Path) -> Optional[Path]:
    """Return a "best" checkpoint if present (best_model.pt or best.pt)."""
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        return None

    for fname in ["best_model.pt", "best.pt"]:
        cand = ckpt_dir / fname
        if cand.exists() and cand.is_file():
            return cand
    return None


def resolve_checkpoint_path(path_str: str, checkpoint_selector: str = "latest") -> Path:
    """Resolve a checkpoint path.

    Accepts either:
    - a direct checkpoint file path
    - a directory containing checkpoints (picks latest checkpoint_*.pt)
    - a run directory containing a checkpoints/ subdir

    Falls back to best checkpoints if present.
    """
    p = resolve_path(path_str)

    if p.exists() and p.is_file():
        return p

    ckpt_dir = p
    if p.exists() and p.is_dir():
        # If user points at the run dir, use its checkpoints/ subdir.
        if (p / "checkpoints").is_dir():
            ckpt_dir = p / "checkpoints"

        checkpoint_selector = (checkpoint_selector or "latest").strip().lower()

        if checkpoint_selector in {"best", "best_model"}:
            best = _best_checkpoint_in_dir(ckpt_dir)
            if best is not None:
                return best

        latest = _latest_checkpoint_in_dir(ckpt_dir)
        if latest is not None:
            return latest

        # Fallbacks (common naming across scripts)
        for fname in ["best_model.pt", "best.pt", "latest.pt", "checkpoint.pt"]:
            cand = ckpt_dir / fname
            if cand.exists() and cand.is_file():
                return cand

    # Last resort: return original (will trigger a clear FileNotFoundError)
    return p


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
        # Can be a checkpoint file, a checkpoints/ directory, or a run directory.
        # We'll auto-pick latest checkpoint_*.pt if a directory is provided.
        "path": "runs_p100/q-seq-cwe/checkpoints",
        "variant_id": VariantID.Q_SEQUENCE_CWE_CTX13,
    },
}

# =============================================================================
# Helpers
# =============================================================================


def resolve_path(path_str: str) -> Path:
    """Try to resolve path relative to CWD or CWD/PaST.

    Accepts:
    - absolute paths
    - workspace-relative paths
    - paths prefixed with "PaST/" (and can also resolve after stripping that prefix)
    """
    p = Path(path_str)
    if p.exists():
        return p

    # Try relative to the PaST package root.
    p_past = Path("PaST") / path_str
    if p_past.exists():
        return p_past

    # If user provided PaST/..., also try stripping it.
    parts = list(p.parts)
    if parts and parts[0] == "PaST":
        p_stripped = Path(*parts[1:])
        if p_stripped.exists():
            return p_stripped
        p_stripped2 = Path("PaST") / p_stripped
        if p_stripped2.exists():
            return p_stripped2

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


class RandomQSequenceNet(torch.nn.Module):
    """A deterministic pseudo-random Q model.

    Important: Q-values are deterministic *per state* (based on the job-availability
    mask), not per forward-call. This avoids accidentally giving RandomQ extra
    stochastic exploration during SGBS just because SGBS evaluates many nodes.
    """

    def __init__(self, seed: int):
        super().__init__()
        self.seed = int(seed)

    def reset(self) -> None:
        # Kept for API symmetry with real models; no internal state required.
        return None

    def forward(
        self,
        jobs_t: torch.Tensor,
        periods_t: torch.Tensor,
        ctx_t: torch.Tensor,
        invalid_mask: torch.Tensor,
    ) -> torch.Tensor:
        # jobs_t: [B, N_pad, F], invalid_mask: [B, N_pad]
        B, N = int(jobs_t.shape[0]), int(jobs_t.shape[1])

        # Derive a cheap deterministic fingerprint of the state from the valid-action mask.
        # valid_mask: 1 for valid jobs, 0 for invalid.
        valid_mask = (~invalid_mask).to(torch.int64)
        # A simple polynomial hash over mask bits.
        idx = torch.arange(N, device=jobs_t.device, dtype=torch.int64) + 1
        mask_hash = (valid_mask * (idx * 1315423911)).sum(dim=-1)  # (B,)

        # Also include remaining job count to distinguish early/late steps.
        remaining = valid_mask.sum(dim=-1)  # (B,)

        # Build a reproducible pseudo-random value per action.
        # Use integer mixing, then map to [0,1).
        base = (
            torch.tensor(self.seed, device=jobs_t.device, dtype=torch.int64)
            + mask_hash
            + remaining * 2654435761
        )
        # action-specific mix
        action_ids = idx.view(1, N).expand(B, N)
        x = base.view(B, 1) + action_ids * 97531
        x = (x ^ (x >> 13)) * 1274126177
        x = x ^ (x >> 16)

        # Convert to float in [0,1). Use uint32 mask for stable behaviour.
        x_u32 = (x & 0xFFFFFFFF).to(torch.float32)
        q = x_u32 / 4294967296.0
        return q


def load_q_model_from_spec(
    *,
    model_name: str,
    variant_id: VariantID,
    checkpoint_path: str,
    device: torch.device,
    checkpoint_selector: str = "latest",
):
    """Load a Q-sequence model from an explicit checkpoint path."""
    var_cfg = get_variant_config(variant_id)
    model = build_q_model(var_cfg)

    ckpt_path = resolve_checkpoint_path(
        checkpoint_path, checkpoint_selector=checkpoint_selector
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = _load_checkpoint(ckpt_path, device)
    state_dict = safe_extract_state_dict(ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, var_cfg


@dataclass
class ModelSpec:
    key: str
    name: str
    path: str
    variant_id: VariantID


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
    sgbs_rollout: str = "model",
    sgbs_rollout_mix_prob: float = 0.5,
    sgbs_rollout_seed: int = 0,
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
            model,
            var_cfg,
            batch,
            device,
            beta=beta,
            gamma=gamma,
            rollout_policy=str(sgbs_rollout),
            rollout_mix_prob=float(sgbs_rollout_mix_prob),
            rollout_seed=int(sgbs_rollout_seed),
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


def evaluate_random_permutation_baseline(
    episodes: List,
    ratio: float,
    seed: int,
) -> List[EvalResult]:
    """Evaluate a random job sequence baseline (RandomPerm+DP)."""
    rng = random.Random(int(seed))
    results: List[EvalResult] = []

    t0 = time.time()
    dp_results = []
    for ep in episodes:
        n = int(getattr(ep, "n_jobs"))
        seq = list(range(n))
        rng.shuffle(seq)
        # dp_schedule_for_job_sequence expects a dict shaped (1, ...)
        single = {
            "n_jobs": np.asarray([int(ep.n_jobs)], dtype=np.int32),
            "p_subset": np.asarray(ep.p_subset[None, :], dtype=np.int32),
            "ct": np.asarray(ep.ct[None, :], dtype=np.int32),
            "T_limit": np.asarray([int(ep.T_limit)], dtype=np.int32),
            "e_single": np.asarray([int(ep.e_single)], dtype=np.int32),
        }
        dp_results.append(dp_schedule_for_job_sequence(single, seq))
    t_total = time.time() - t0

    for i, res in enumerate(dp_results):
        results.append(
            EvalResult(
                model_name="Baseline",
                method="RandomPerm+DP",
                instance_idx=i,
                ratio=ratio,
                energy=res.total_energy,
                time_sec=t_total / max(1, len(dp_results)),
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

    specs: List[ModelSpec] = []

    if not args.no_attention:
        specs.append(
            ModelSpec(
                key="Attention",
                name=args.attention_name or MODELS["Attention"]["name"],
                path=args.attention_ckpt or MODELS["Attention"]["path"],
                variant_id=VariantID.Q_SEQUENCE,
            )
        )

    # Support either one or two CWE checkpoints.
    if args.cwe_ckpt_1:
        specs.append(
            ModelSpec(
                key="CWE_1",
                name=args.cwe_name_1 or "Q-Seq (CWE #1)",
                path=args.cwe_ckpt_1,
                variant_id=VariantID.Q_SEQUENCE_CWE_CTX13,
            )
        )
    else:
        # Backward-compatible default CWE path.
        specs.append(
            ModelSpec(
                key="CWE",
                name=args.cwe_name_1 or MODELS["CWE"]["name"],
                path=MODELS["CWE"]["path"],
                variant_id=VariantID.Q_SEQUENCE_CWE_CTX13,
            )
        )

    if args.cwe_ckpt_2:
        specs.append(
            ModelSpec(
                key="CWE_2",
                name=args.cwe_name_2 or "Q-Seq (CWE #2)",
                path=args.cwe_ckpt_2,
                variant_id=VariantID.Q_SEQUENCE_CWE_CTX13,
            )
        )

    models = {}
    for spec in specs:
        try:
            model, var_cfg = load_q_model_from_spec(
                model_name=spec.name,
                variant_id=spec.variant_id,
                checkpoint_path=spec.path,
                device=device,
                checkpoint_selector=args.checkpoint_selector,
            )
            models[spec.key] = {
                "model": model,
                "var_cfg": var_cfg,
                "name": spec.name,
                "path": spec.path,
            }
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  ✓ {spec.name}: {n_params:,} params")
        except Exception as e:
            print(f"  ✗ {spec.name}: {e}")

    # Optional: add a random-Q baseline model (supports greedy and SGBS)
    if args.include_random_q:
        try:
            var_cfg_rand = get_variant_config(VariantID.Q_SEQUENCE)
            rand_model = RandomQSequenceNet(seed=int(args.random_seed))
            rand_model.to(device)
            rand_model.eval()
            models["RandomQ"] = {
                "model": rand_model,
                "var_cfg": var_cfg_rand,
                "name": args.random_name,
                "path": "<random>",
            }
            print(f"  ✓ {args.random_name}: (random baseline)")
        except Exception as e:
            print(f"  ✗ {args.random_name}: {e}")

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

    def _finite_mean_and_infeasible_count(
        values: List[float],
    ) -> tuple[float, int, int]:
        arr = np.asarray(values, dtype=np.float64)
        finite_mask = np.isfinite(arr)
        infeasible = int((~finite_mask).sum())
        finite_n = int(finite_mask.sum())
        mean_finite = float(arr[finite_mask].mean()) if finite_n > 0 else float("inf")
        return mean_finite, infeasible, int(arr.size)

    for key, model_data in models.items():
        print(f"\n🔍 Evaluating {model_data['name']}...")

        # Reset random model call counter per run for reproducibility
        if hasattr(model_data["model"], "reset"):
            try:
                model_data["model"].reset()
            except Exception:
                pass

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
                sgbs_rollout=args.sgbs_rollout,
                sgbs_rollout_mix_prob=args.sgbs_rollout_mix_prob,
                sgbs_rollout_seed=args.sgbs_rollout_seed + int(r_idx) * 1000,
            )
            all_results.extend(results)

            # Progress
            greedy_energies = [r.energy for r in results if r.method == "Greedy+DP"]
            greedy_mean, infeasible, total_n = _finite_mean_and_infeasible_count(
                greedy_energies
            )
            if infeasible > 0:
                print(
                    f"    Ratio {ratio:.2f}: Greedy mean (finite) = {greedy_mean:.2f} | infeasible={infeasible}/{total_n}"
                )
            else:
                print(f"    Ratio {ratio:.2f}: Greedy mean = {greedy_mean:.2f}")

    # 5. Baselines
    print("\n🔍 Evaluating Baselines (SPT+DP, LPT+DP)...")
    for r_idx, ratio in enumerate(ratios):
        episodes = episodes_by_ratio[r_idx]
        baseline_results = evaluate_baselines(episodes, device, ratio)
        all_results.extend(baseline_results)

        if args.include_random_perm:
            rand_perm_results = evaluate_random_permutation_baseline(
                episodes,
                ratio,
                seed=int(args.random_seed) + int(r_idx) * 1000,
            )
            all_results.extend(rand_perm_results)

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
    print(
        "Note: Mean/Std are computed over finite energies only; Infeasible counts are shown separately."
    )

    # Replace infinities for summary stats (but keep raw df as-is for CSV/debugging).
    df_summary = df.copy()
    df_summary["energy"] = df_summary["energy"].replace([np.inf, -np.inf], np.nan)

    summary = (
        df_summary.groupby(["model_name", "method"])["energy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    infeasible_counts = (
        df.groupby(["model_name", "method"])["energy"]
        .apply(lambda s: int((~np.isfinite(np.asarray(s, dtype=np.float64))).sum()))
        .reset_index(name="infeasible_count")
    )

    summary = summary.merge(infeasible_counts, on=["model_name", "method"], how="left")
    summary = summary.sort_values("mean")
    summary.columns = [
        "Model",
        "Method",
        "Mean Energy",
        "Std",
        "Count",
        "Infeasible",
    ]
    print(summary.to_string(index=False))

    # 8b. Fair summary: restrict to the subset of (instance_idx, ratio) where *all* methods are feasible.
    # This avoids a method looking better simply because it has infeasible (inf) rows that were excluded.
    try:
        wide = df.pivot_table(
            index=["instance_idx", "ratio"],
            columns=["model_name", "method"],
            values="energy",
            aggfunc="first",
        )
        # Keep only rows where every column is finite.
        finite_mask = np.isfinite(wide.to_numpy(dtype=np.float64)).all(axis=1)
        wide_common = wide.loc[finite_mask]

        if len(wide_common) > 0 and len(wide_common) < len(wide):
            print("\n" + "=" * 70)
            print("COMMON-FEASIBLE SUMMARY (Mean Energy)")
            print("=" * 70)
            print(
                f"Common-feasible rows: {len(wide_common)}/{len(wide)} (dropped {len(wide) - len(wide_common)} rows with any infeasibility)"
            )

            # Compute mean/std/count on the common-feasible subset.
            common_stats = []
            for model_name, method in wide_common.columns:
                vals = wide_common[(model_name, method)].to_numpy(dtype=np.float64)
                common_stats.append(
                    {
                        "Model": model_name,
                        "Method": method,
                        "Mean Energy": float(np.mean(vals)),
                        "Std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                        "Count": int(len(vals)),
                    }
                )

            common_df = pd.DataFrame(common_stats).sort_values("Mean Energy")
            print(common_df.to_string(index=False))
    except Exception as e:
        print(f"\n[warn] Could not compute common-feasible summary: {e}")

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
        "--sgbs_rollout",
        type=str,
        default="model",
        choices=[
            "model",
            "random",
            "spt",
            "lpt",
            "mix_model_spt",
            "mix_model_random",
        ],
        help=(
            "Rollout/completion policy inside SGBS simulation. "
            "Expansion still uses the model's Q; this only changes how candidates are completed before DP scoring."
        ),
    )
    parser.add_argument(
        "--sgbs_rollout_mix_prob",
        type=float,
        default=0.5,
        help="For mix_model_* rollouts: probability of using the model vs the heuristic during rollout.",
    )
    parser.add_argument(
        "--sgbs_rollout_seed",
        type=int,
        default=None,
        help="Seed for stochastic rollouts (defaults to eval_seed).",
    )

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

    parser.add_argument(
        "--checkpoint_selector",
        type=str,
        default="latest",
        choices=["latest", "best_model", "best"],
        help=(
            "How to select a checkpoint when a directory is provided: "
            "'latest' picks the highest checkpoint_*.pt; 'best'/'best_model' uses best_model.pt or best.pt if present"
        ),
    )

    # --- Override checkpoint paths (useful for ad-hoc comparisons) ---
    parser.add_argument(
        "--attention_ckpt",
        type=str,
        default=None,
        help=(
            "Checkpoint path for the attention Q-seq model. "
            "Can be a file, checkpoints/ directory, or run directory (containing checkpoints/)."
        ),
    )
    parser.add_argument(
        "--attention_name",
        type=str,
        default=None,
        help="Display name for the attention checkpoint in the output table.",
    )
    parser.add_argument(
        "--no_attention",
        action="store_true",
        help="Do not load/evaluate the attention model (compare CWE checkpoints only).",
    )

    parser.add_argument(
        "--cwe_ckpt_1",
        type=str,
        default=None,
        help=(
            "First CWE Q-seq checkpoint path (file/dir/run). If omitted, uses the script default CWE path."
        ),
    )
    parser.add_argument(
        "--cwe_name_1",
        type=str,
        default=None,
        help="Display name for CWE checkpoint #1.",
    )
    parser.add_argument(
        "--cwe_ckpt_2",
        type=str,
        default=None,
        help="Second CWE Q-seq checkpoint path (file/dir/run).",
    )
    parser.add_argument(
        "--cwe_name_2",
        type=str,
        default=None,
        help="Display name for CWE checkpoint #2.",
    )

    # --- Random baselines ---
    parser.add_argument(
        "--include_random_perm",
        action="store_true",
        help="Include a RandomPerm+DP baseline (random job order, then DP timing).",
    )
    parser.add_argument(
        "--include_random_q",
        action="store_true",
        help="Include a RandomQ baseline model (supports greedy and SGBS decoding).",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Seed for random baselines (defaults to eval_seed if omitted).",
    )
    parser.add_argument(
        "--random_name",
        type=str,
        default="Baseline (RandomQ)",
        help="Display name for the RandomQ baseline model.",
    )

    args = parser.parse_args()

    if args.random_seed is None:
        args.random_seed = int(args.eval_seed)

    if args.sgbs_rollout_seed is None:
        args.sgbs_rollout_seed = int(args.eval_seed)

    run_comparison(args)

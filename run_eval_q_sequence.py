"""Evaluation script for Q-Sequence model with SGBS and baselines.

Evaluates the Q-sequence model trained via supervised regression on DP costs.
Uses the QModelWrapper to provide SGBS-compatible logits from Q-values.

Methods:
- Greedy: argmin Q-value for job selection, DP for optimal scheduling
- SGBS(β, γ): Simulation-guided beam search with DP scheduling
- SPT+DP: Shortest Processing Time order + optimal DP scheduling
- LPT+DP: Longest Processing Time order + optimal DP scheduling

Example:
    python -m PaST.run_eval_q_sequence \\
        --checkpoint runs_p100/ppo_q_seq/checkpoints/best.pt \\
        --variant_id q_sequence \\
        --eval_seed 1337 \\
        --num_instances 64 \\
        --scale small \\
        --beta 4 --gamma 4
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
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
import yaml

# Set matplotlib backend before import
os.environ.setdefault("MPLBACKEND", "Agg")

from PaST.baselines_sequence_dp import (
    DPResult,
    dp_schedule_for_job_sequence,
    spt_lpt_with_dp,
    spt_sequence,
    lpt_sequence,
)
from PaST.config import DataConfig, VariantID, get_variant_config
from PaST.q_sequence_model import build_q_model, QSequenceNet, QModelWrapper
from PaST.sm_benchmark_data import generate_episode_batch
from PaST.sequence_env import GPUBatchSequenceEnv


# =============================================================================
# Utilities
# =============================================================================


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
        # Older PyTorch versions
        ckpt = torch.load(path, map_location=device)
    return ckpt


def _extract_q_model_state(ckpt: Any) -> Dict[str, Any]:
    """Extract model state dict from checkpoint.
    
    Handles multiple formats:
    - Direct state_dict (from best_model.pt / final_model.pt)
    - Nested dict with 'model_state' key (from checkpoint_X.pt)
    """
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dict")
    
    # Format 1: checkpoint_X.pt with 'model_state' key
    if "model_state" in ckpt:
        return ckpt["model_state"]
    
    # Format 2: Direct state_dict (keys are layer names)
    # Check if it looks like a state_dict by examining keys
    sample_keys = list(ckpt.keys())[:5]
    if any("." in k for k in sample_keys):
        # Likely a state_dict (has nested module names)
        return ckpt
    
    raise KeyError("Could not find model state in checkpoint")


def _resolve_run_dir(run_dir_arg: str) -> Path:
    raw = Path(run_dir_arg)
    if raw.exists():
        return raw
    
    pkg_root = Path(__file__).resolve().parent
    candidate = pkg_root / raw
    if candidate.exists():
        return candidate
    
    cwd = Path.cwd()
    candidate = cwd / "PaST" / raw
    if candidate.exists():
        return candidate
    
    parts = list(raw.parts)
    if parts and parts[0] == "PaST":
        stripped = Path(*parts[1:])
        candidate = pkg_root / stripped
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


def _mean(x: List[float]) -> float:
    arr = np.array(x, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return float("nan")
    return float(arr[finite].mean())


# =============================================================================
# Greedy Decode for Q-Sequence
# =============================================================================


def greedy_decode_q_sequence(
    model: QSequenceNet,
    variant_config,
    batch_data: Dict[str, Any],
    device: torch.device,
) -> List[DPResult]:
    """Greedy decode using Q-values: argmin Q(s, j) for each step.
    
    Uses GPUBatchSequenceEnv to get properly constructed observations,
    then selects argmin Q for job ordering.
    After determining the job sequence, uses DP to compute optimal scheduling.
    """
    B = int(batch_data["n_jobs"].shape[0])
    env_config = variant_config.env
    
    results = []
    
    model.eval()
    with torch.no_grad():
        for i in range(B):
            single = _slice_single_instance(batch_data, i)
            n = int(single["n_jobs"][0])
            
            # Create sequence environment for this instance
            env = GPUBatchSequenceEnv(
                batch_size=1,
                env_config=env_config,
                device=device,
            )
            obs = env.reset(single)
            
            sequence = []
            
            for step in range(n):
                # Get observation tensors from env
                jobs_t = obs["jobs"]  # [1, N_pad, F_job]
                periods_t = obs["periods"]  # [1, K_pad, F_period]
                ctx_t = obs["ctx"]  # [1, F_ctx]
                
                # Get job mask from environment (1=valid, need to convert to True=invalid)
                job_mask_float = obs.get("job_mask", env.job_available)
                if job_mask_float.dtype != torch.bool:
                    job_mask = job_mask_float < 0.5  # True = INVALID
                else:
                    job_mask = ~job_mask_float  # Invert if bool
                
                # Get Q-values
                q_values = model(jobs_t, periods_t, ctx_t, job_mask)
                
                # Mask invalid jobs
                q_values = q_values.masked_fill(job_mask, float("inf"))
                
                # Select argmin Q
                action = q_values.argmin(dim=-1)
                
                sequence.append(int(action.item()))
                
                # Step environment
                obs, _, done, _ = env.step(action)
                
                if done.all():
                    break
            
            # Use DP to schedule the sequence
            result = dp_schedule_for_job_sequence(single, sequence)
            results.append(result)
    
    return results


def sgbs_q_sequence(
    model: QSequenceNet,
    variant_config,
    batch_data: Dict[str, Any],
    device: torch.device,
    beta: int,
    gamma: int,
    temperature: float = 1.0,
) -> List[DPResult]:
    """SGBS decode for Q-sequence model.
    
    Since Q-sequence uses job-only actions (not job×slack), we implement a custom
    beam search over job sequences using the sequence environment.
    
    Algorithm (simplified SGBS):
    1. Expand: For each beam node, consider top-γ jobs by Q-value (lower Q = better)
    2. Simulate: Complete each candidate sequence with greedy rollout
    3. Prune: Keep top-β candidates by total energy (from DP scheduling)
    """
    B = int(batch_data["n_jobs"].shape[0])
    env_config = variant_config.env
    
    results = []
    
    model.eval()
    with torch.no_grad():
        for i in range(B):
            single = _slice_single_instance(batch_data, i)
            n = int(single["n_jobs"][0])
            
            # Special case: small instance, just do greedy
            if n <= 2 or (beta == 1 and gamma == 1):
                result = _greedy_single_instance(model, variant_config, single, device)
                results.append(result)
                continue
            
            # SGBS beam search for this instance
            result = _sgbs_single_instance(
                model, variant_config, single, device, 
                beta=beta, gamma=gamma, temperature=temperature
            )
            results.append(result)
    
    return results


def _greedy_single_instance(
    model: QSequenceNet,
    variant_config,
    single: Dict[str, Any],
    device: torch.device,
) -> DPResult:
    """Helper: greedy decode for a single instance."""
    env_config = variant_config.env
    n = int(single["n_jobs"][0])
    
    env = GPUBatchSequenceEnv(batch_size=1, env_config=env_config, device=device)
    obs = env.reset(single)
    
    sequence = []
    
    for _ in range(n):
        jobs_t = obs["jobs"]
        periods_t = obs["periods"]
        ctx_t = obs["ctx"]
        
        job_mask_float = obs.get("job_mask", env.job_available)
        job_mask = job_mask_float < 0.5 if job_mask_float.dtype != torch.bool else ~job_mask_float
        
        q_values = model(jobs_t, periods_t, ctx_t, job_mask)
        q_values = q_values.masked_fill(job_mask, float("inf"))
        
        action = q_values.argmin(dim=-1)
        sequence.append(int(action.item()))
        
        obs, _, done, _ = env.step(action)
        if done.all():
            break
    
    return dp_schedule_for_job_sequence(single, sequence)


def _sgbs_single_instance(
    model: QSequenceNet,
    variant_config,
    single: Dict[str, Any],
    device: torch.device,
    beta: int,
    gamma: int,
    temperature: float = 1.0,
) -> DPResult:
    """SGBS for a single instance with Q-sequence model."""
    env_config = variant_config.env
    n = int(single["n_jobs"][0])
    
    # Beam node: (partial_sequence, available_jobs_set)
    # Start with empty sequence, all jobs available
    initial_beam = [([], set(range(n)))]
    
    best_energy = float("inf")
    best_sequence = list(range(n))  # Fallback
    
    for step in range(n):
        candidates = []
        
        for partial_seq, available in initial_beam:
            if not available:
                # Sequence complete - evaluate
                result = dp_schedule_for_job_sequence(single, partial_seq)
                if result.total_energy < best_energy:
                    best_energy = result.total_energy
                    best_sequence = partial_seq
                continue
            
            # Create env at this state
            env = GPUBatchSequenceEnv(batch_size=1, env_config=env_config, device=device)
            obs = env.reset(single)
            
            # Replay partial sequence
            for a in partial_seq:
                obs, _, _, _ = env.step(torch.tensor([a], device=device))
            
            # Get Q-values for remaining jobs
            jobs_t = obs["jobs"]
            periods_t = obs["periods"]
            ctx_t = obs["ctx"]
            
            job_mask_float = obs.get("job_mask", env.job_available)
            job_mask = job_mask_float < 0.5 if job_mask_float.dtype != torch.bool else ~job_mask_float
            
            q_values = model(jobs_t, periods_t, ctx_t, job_mask)
            q_values = q_values.masked_fill(job_mask, float("inf"))
            
            # Get top-gamma jobs by Q-value (lower is better)
            # Note: we want smallest Q values, so use negative for topk
            neg_q = -q_values[0]
            neg_q = neg_q.masked_fill(job_mask[0], float("-inf"))
            
            k_eff = min(gamma, len(available))
            if k_eff > 0:
                topk = torch.topk(neg_q, k=k_eff)
                top_jobs = topk.indices.cpu().tolist()
                
                for job in top_jobs:
                    if job in available:
                        new_seq = partial_seq + [job]
                        new_available = available - {job}
                        candidates.append((new_seq, new_available))
        
        if not candidates:
            break
        
        # Simulate: complete each candidate with greedy and evaluate
        scored_candidates = []
        for partial_seq, available in candidates:
            # Complete with greedy
            full_seq = list(partial_seq)
            remaining = sorted(available)  # Fallback ordering
            
            if remaining:
                # Create env and complete greedily
                env = GPUBatchSequenceEnv(batch_size=1, env_config=env_config, device=device)
                obs = env.reset(single)
                
                # Replay to current state
                for a in partial_seq:
                    obs, _, _, _ = env.step(torch.tensor([a], device=device))
                
                # Greedy completion
                for _ in remaining:
                    jobs_t = obs["jobs"]
                    periods_t = obs["periods"]
                    ctx_t = obs["ctx"]
                    
                    job_mask_float = obs.get("job_mask", env.job_available)
                    job_mask = job_mask_float < 0.5 if job_mask_float.dtype != torch.bool else ~job_mask_float
                    
                    q_values = model(jobs_t, periods_t, ctx_t, job_mask)
                    q_values = q_values.masked_fill(job_mask, float("inf"))
                    
                    action = q_values.argmin(dim=-1)
                    full_seq.append(int(action.item()))
                    
                    obs, _, done, _ = env.step(action)
                    if done.all():
                        break
            
            # Evaluate with DP
            result = dp_schedule_for_job_sequence(single, full_seq)
            scored_candidates.append((result.total_energy, partial_seq, available, full_seq))
            
            # Track best
            if result.total_energy < best_energy:
                best_energy = result.total_energy
                best_sequence = full_seq
        
        # Prune: keep top-beta by energy
        scored_candidates.sort(key=lambda x: x[0])
        initial_beam = [(seq, avail) for _, seq, avail, _ in scored_candidates[:beta]]
    
    return dp_schedule_for_job_sequence(single, best_sequence)


# =============================================================================
# Visualization
# =============================================================================


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
        print("matplotlib not available, skipping visualization")
        return
    
    n_jobs = int(instance_data["n_jobs"])
    T_limit = int(instance_data["T_limit"])
    Tk = instance_data["Tk"]
    ck = instance_data["ck"]
    period_starts = instance_data["period_starts"]
    K = int(instance_data["K"])
    
    method_order = list(schedules.keys())
    
    fig_h = 2.0 + 0.8 * len(method_order)
    fig, (ax_top, ax_jobs) = plt.subplots(
        2, 1, figsize=(14, fig_h), sharex=True,
        gridspec_kw={"height_ratios": [1.0, max(2.0, 0.7 * len(method_order))]}
    )
    
    # Price period colors
    price_colors = {1: "#d4f1d4", 2: "#fff4b3", 3: "#ffd9b3", 4: "#ffb3b3"}
    
    # Draw price periods
    for k in range(K):
        start = int(period_starts[k])
        dur = int(Tk[k])
        price = int(ck[k])
        if start >= T_limit:
            continue
        dur = min(dur, T_limit - start)
        rect = mpatches.Rectangle(
            (start, 0), dur, 1.0,
            facecolor=price_colors.get(price, "#cccccc"),
            edgecolor="black", linewidth=0.7
        )
        ax_top.add_patch(rect)
        ax_top.text(
            start + dur / 2, 0.5, f"c={price}",
            ha="center", va="center", fontsize=8
        )
    
    ax_top.set_xlim(0, T_limit)
    ax_top.set_ylim(0, 1)
    ax_top.set_ylabel("Price")
    ax_top.set_title(f"Q-Sequence Evaluation{title_suffix}")
    
    # Draw schedules
    job_colors = plt.cm.tab20(np.linspace(0, 1, max(n_jobs, 1)))
    
    y_pos = 0
    for method_name, sched in schedules.items():
        bars = sched.get("bars", [])
        energy = sched.get("energy", 0)
        
        for bar in bars:
            job_id = bar["job_id"]
            start = bar["start"]
            end = bar["end"]
            dur = end - start
            
            color = job_colors[job_id % len(job_colors)]
            rect = mpatches.Rectangle(
                (start, y_pos + 0.1), dur, 0.8,
                facecolor=color, edgecolor="black", linewidth=0.5
            )
            ax_jobs.add_patch(rect)
            
            # Label
            if dur > 5:
                ax_jobs.text(
                    start + dur / 2, y_pos + 0.5, f"J{job_id}",
                    ha="center", va="center", fontsize=7
                )
        
        # Method label
        ax_jobs.text(
            -T_limit * 0.02, y_pos + 0.5,
            f"{method_name}: E={energy:.1f}",
            ha="right", va="center", fontsize=9
        )
        
        y_pos += 1
    
    ax_jobs.set_xlim(-T_limit * 0.15, T_limit)
    ax_jobs.set_ylim(0, len(method_order))
    ax_jobs.set_xlabel("Time")
    ax_jobs.set_ylabel("Method")
    ax_jobs.axvline(x=T_limit, color="red", linestyle="--", linewidth=1.5, label="Deadline")
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote visualization: {out_path}")


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate Q-Sequence model with SGBS + baselines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Run directory containing checkpoints/. If provided, uses --which to select checkpoint.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Direct path to checkpoint file. Alternative to --run_dir.",
    )
    p.add_argument(
        "--variant_id",
        type=str,
        default="q_sequence",
        help="Variant ID (q_sequence or q_sequence_ctx13).",
    )
    p.add_argument(
        "--which",
        type=str,
        default="best",
        choices=["best", "latest"],
        help="Checkpoint name under <run_dir>/checkpoints (only used with --run_dir).",
    )
    p.add_argument("--eval_seed", type=int, required=True, help="Seed for instance generation.")
    p.add_argument("--num_instances", type=int, default=64, help="Number of instances to evaluate.")
    p.add_argument("--num_viz", type=int, default=0, help="Number of visualizations to generate.")
    p.add_argument(
        "--scale",
        type=str,
        default=None,
        choices=["small", "medium", "large"],
        help="Instance scale filter.",
    )
    p.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    
    # SGBS parameters
    p.add_argument("--beta", type=str, default="4", help="SGBS beta (or comma-separated list).")
    p.add_argument("--gamma", type=str, default="4", help="SGBS gamma (or comma-separated list).")
    p.add_argument("--temperature", type=float, default=1.0, help="Temperature for Q->logits conversion.")
    
    # Output
    p.add_argument("--out_dir", type=str, default=None, help="Output directory for results.")
    p.add_argument("--out_csv", type=str, default=None, help="CSV output path.")
    p.add_argument("--out_json", type=str, default=None, help="JSON summary output path.")
    
    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    # Validate args
    if args.run_dir is None and args.checkpoint is None:
        raise ValueError("Provide either --run_dir or --checkpoint")
    if args.run_dir is not None and args.checkpoint is not None:
        raise ValueError("Provide only one of --run_dir or --checkpoint")
    
    # Resolve checkpoint path
    if args.run_dir is not None:
        run_dir = _resolve_run_dir(args.run_dir)
        ckpt_path = run_dir / "checkpoints" / f"{args.which}.pt"
    else:
        ckpt_path = Path(args.checkpoint)
        run_dir = ckpt_path.parent.parent if ckpt_path.parent.name == "checkpoints" else ckpt_path.parent
    
    # Device
    requested_device = args.device or "cuda"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but unavailable; falling back to CPU")
        requested_device = "cpu"
    device = torch.device(requested_device)
    
    # Load variant config
    try:
        variant_id = VariantID(args.variant_id)
    except ValueError:
        print(f"Warning: Unknown variant {args.variant_id}, using q_sequence")
        variant_id = VariantID.Q_SEQUENCE
    
    variant_config = get_variant_config(variant_id)
    
    # Build and load model
    print(f"Loading checkpoint from: {ckpt_path}")
    model = build_q_model(variant_config).to(device)
    ckpt = _load_checkpoint(ckpt_path, device)
    model_state = _extract_q_model_state(ckpt)
    model.load_state_dict(model_state)
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    
    # Parse SGBS parameters
    betas = [int(x) for x in args.beta.split(",")]
    gammas = [int(x) for x in args.gamma.split(",")]
    
    # Create beta-gamma pairs
    if len(betas) == len(gammas) and len(betas) > 1:
        bg_pairs = list(zip(betas, gammas))
    elif len(betas) == 1:
        bg_pairs = [(betas[0], g) for g in gammas]
    elif len(gammas) == 1:
        bg_pairs = [(b, gammas[0]) for b in betas]
    else:
        bg_pairs = [(b, g) for b in betas for g in gammas]
    
    print(f"SGBS configurations: {bg_pairs}")
    
    # Prepare data config
    data_cfg = _restrict_data_config(variant_config.data, args.scale)
    scale_str = args.scale or "all"
    
    # Generate evaluation batch
    print(f"Generating {args.num_instances} instances (scale={scale_str}, seed={args.eval_seed})...")
    batch = generate_episode_batch(
        batch_size=int(args.num_instances),
        config=data_cfg,
        seed=int(args.eval_seed),
        N_job_pad=int(variant_config.env.N_job_pad),
        K_period_pad=250,
        T_max_pad=500,
    )
    
    rows: List[Dict[str, Any]] = []
    
    # === Greedy ===
    print("Running Greedy decode...", end=" ", flush=True)
    t0 = time.perf_counter()
    greedy_res = greedy_decode_q_sequence(model, variant_config, batch, device)
    greedy_time = time.perf_counter() - t0
    print(f"done ({greedy_time:.2f}s)")
    
    # === SGBS ===
    sgbs_res_map: Dict[str, List[DPResult]] = {}
    for b_val, g_val in bg_pairs:
        label = f"{b_val}-{g_val}"
        print(f"Running SGBS(β={b_val}, γ={g_val})...", end=" ", flush=True)
        t0 = time.perf_counter()
        sgbs_res = sgbs_q_sequence(
            model, variant_config, batch, device, beta=b_val, gamma=g_val, temperature=args.temperature
        )
        sgbs_time = time.perf_counter() - t0
        print(f"done ({sgbs_time:.2f}s)")
        sgbs_res_map[label] = sgbs_res
    
    # === SPT+DP ===
    print("Running SPT+DP...", end=" ", flush=True)
    t0 = time.perf_counter()
    spt_res = spt_lpt_with_dp(variant_config.env, device, batch, which="spt")
    spt_time = time.perf_counter() - t0
    print(f"done ({spt_time:.2f}s)")
    
    # === LPT+DP ===
    print("Running LPT+DP...", end=" ", flush=True)
    t0 = time.perf_counter()
    lpt_res = spt_lpt_with_dp(variant_config.env, device, batch, which="lpt")
    lpt_time = time.perf_counter() - t0
    print(f"done ({lpt_time:.2f}s)")
    
    # Collect results
    for i in range(int(args.num_instances)):
        row = {
            "instance": i,
            "greedy_energy": greedy_res[i].total_energy,
            "spt_dp_energy": spt_res[i].total_energy,
            "lpt_dp_energy": lpt_res[i].total_energy,
        }
        for label, res_list in sgbs_res_map.items():
            row[f"sgbs_{label}_energy"] = res_list[i].total_energy
        rows.append(row)
    
    # Compute means
    greedy_mean = _mean([r.total_energy for r in greedy_res])
    spt_mean = _mean([r.total_energy for r in spt_res])
    lpt_mean = _mean([r.total_energy for r in lpt_res])
    sgbs_means = {label: _mean([r.total_energy for r in res]) for label, res in sgbs_res_map.items()}
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"Q-SEQUENCE EVALUATION RESULTS")
    print(f"Checkpoint: {ckpt_path.name} | Scale: {scale_str} | Seed: {args.eval_seed}")
    print("=" * 70)
    print(f"{'Method':<25} | {'Mean Energy':<12} | {'vs Greedy':<15}")
    print("-" * 60)
    print(f"{'Greedy':<25} | {greedy_mean:<12.2f} | {'baseline':<15}")
    
    for label, mean_e in sgbs_means.items():
        imp = ((greedy_mean - mean_e) / greedy_mean * 100) if greedy_mean > 0 else 0
        print(f"{f'SGBS({label})':<25} | {mean_e:<12.2f} | {f'{imp:+.2f}%':<15}")
    
    imp_spt = ((greedy_mean - spt_mean) / greedy_mean * 100) if greedy_mean > 0 else 0
    imp_lpt = ((greedy_mean - lpt_mean) / greedy_mean * 100) if greedy_mean > 0 else 0
    print(f"{'SPT+DP':<25} | {spt_mean:<12.2f} | {f'{imp_spt:+.2f}%':<15}")
    print(f"{'LPT+DP':<25} | {lpt_mean:<12.2f} | {f'{imp_lpt:+.2f}%':<15}")
    print("=" * 70)
    
    # === Output ===
    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    first_bg = bg_pairs[0]
    out_csv = (
        Path(args.out_csv) if args.out_csv
        else out_dir / f"eval_q_seq_{scale_str}_seed{args.eval_seed}_b{first_bg[0]}_g{first_bg[1]}.csv"
    )
    out_json = Path(args.out_json) if args.out_json else out_csv.with_suffix(".json")
    
    # Write CSV
    fieldnames = ["instance", "greedy_energy", "spt_dp_energy", "lpt_dp_energy"]
    for label in sgbs_res_map.keys():
        fieldnames.append(f"sgbs_{label}_energy")
    
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    
    # Write JSON summary
    summary = {
        "variant_id": args.variant_id,
        "checkpoint": str(ckpt_path),
        "scale": scale_str,
        "eval_seed": int(args.eval_seed),
        "num_instances": int(args.num_instances),
        "sgbs_configs": [{"beta": b, "gamma": g} for b, g in bg_pairs],
        "temperature": float(args.temperature),
        "means": {
            "greedy_energy": greedy_mean,
            "spt_dp_energy": spt_mean,
            "lpt_dp_energy": lpt_mean,
            **{f"sgbs_{label}_energy": mean_e for label, mean_e in sgbs_means.items()},
        },
        "outputs": {
            "csv": str(out_csv),
            "json": str(out_json),
        },
    }
    
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nWrote: {out_csv}")
    print(f"Wrote: {out_json}")
    
    # === Visualizations ===
    if args.num_viz > 0:
        print(f"\nGenerating {args.num_viz} visualizations...")
        for viz_idx in range(min(int(args.num_viz), int(args.num_instances))):
            single = _slice_single_instance(batch, viz_idx)
            n_jobs = int(single["n_jobs"][0])
            T_limit = int(single["T_limit"][0])
            p_subset = single["p_subset"][0][:n_jobs].astype(np.int32)
            
            schedules = {}
            
            # Greedy
            schedules["Greedy"] = {
                "energy": greedy_res[viz_idx].total_energy,
                "bars": _sequence_schedule_to_bars(
                    greedy_res[viz_idx].job_sequence,
                    greedy_res[viz_idx].start_times,
                    p_subset
                ),
            }
            
            # SGBS variants
            for label, res_list in sgbs_res_map.items():
                schedules[f"SGBS({label})"] = {
                    "energy": res_list[viz_idx].total_energy,
                    "bars": _sequence_schedule_to_bars(
                        res_list[viz_idx].job_sequence,
                        res_list[viz_idx].start_times,
                        p_subset
                    ),
                }
            
            # SPT+DP
            schedules["SPT+DP"] = {
                "energy": spt_res[viz_idx].total_energy,
                "bars": _sequence_schedule_to_bars(
                    spt_res[viz_idx].job_sequence,
                    spt_res[viz_idx].start_times,
                    p_subset
                ),
            }
            
            # LPT+DP
            schedules["LPT+DP"] = {
                "energy": lpt_res[viz_idx].total_energy,
                "bars": _sequence_schedule_to_bars(
                    lpt_res[viz_idx].job_sequence,
                    lpt_res[viz_idx].start_times,
                    p_subset
                ),
            }
            
            viz_path = out_dir / f"schedule_q_seq_{scale_str}_seed{args.eval_seed}_idx{viz_idx}.png"
            visualize_schedule(
                viz_path,
                {
                    "n_jobs": n_jobs,
                    "T_limit": T_limit,
                    "Tk": single["Tk"][0],
                    "ck": single["ck"][0],
                    "period_starts": single["period_starts"][0],
                    "K": int(single["K"][0]),
                },
                schedules,
                f" | seed={args.eval_seed} idx={viz_idx}",
            )


if __name__ == "__main__":
    main()

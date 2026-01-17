"""Enhanced evaluation script with schedule visualization for ppo_family_q4_ctx13.

Extends run_eval_family_q4.py with:
- Gantt-style schedule visualization (price periods + job bars)
- Multi-scale support (small/medium/large instances)
- Side-by-side comparison of all methods

Example:
    # Quick visualization on a small instance
    python run_eval_family_q4_viz.py \
        --run_dir runs_p100/ppo_family_q4_ctx13/seed_0 \
        --which best --eval_seed 1337 \
        --num_instances 16 --num_viz 2 \
        --scale small --beta 4 --gamma 4

    # Full evaluation on large instances
    python run_eval_family_q4_viz.py \
        --run_dir runs_p100/ppo_family_q4_ctx13/seed_0 \
        --which best --eval_seed 1337 \
        --num_instances 64 --num_viz 3 \
        --scale large --beta 8 --gamma 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

# Set matplotlib backend before import
os.environ.setdefault("MPLBACKEND", "Agg")

from PaST.baselines_sequence_dp import DPResult, spt_lpt_with_dp, spt_sequence, lpt_sequence, dp_schedule_for_job_sequence
from PaST.config import DataConfig, VariantID, get_variant_config
from PaST.past_sm_model import build_model
from PaST.sgbs import greedy_decode, sgbs, DecodeResult
from PaST.sm_benchmark_data import generate_episode_batch


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


def _action_trace_to_bars_family(
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
) -> Tuple[List[Dict[str, Any]], float]:
    """Convert action trace to bars for price-family variant.
    
    In price-family mode, action = job_id * num_families + family_id.
    We need to compute the actual start time based on price family.
    """
    num_families = env_config.num_price_families
    
    # Compute price quantiles to determine slot families
    ct_valid = ct[:T_limit]
    if len(ct_valid) > 0:
        q25, q50, q75 = np.quantile(ct_valid, [0.25, 0.5, 0.75])
    else:
        q25 = q50 = q75 = 0
    
    # Assign each slot to a family
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
    
    for a in actions:
        job_id = int(a) // num_families
        family_id = int(a) % num_families
        p = int(p_subset[job_id])
        
        # Find earliest slot in the target family that allows job to finish by T_limit
        start = None
        for u in range(t, T_limit):
            if slot_families[u] == family_id:
                end_candidate = u + p
                if end_candidate <= T_limit:
                    start = u
                    break
        
        if start is None:
            # Fallback: start now
            start = t
        
        end = start + p
        if end > T_limit:
            # Clip to deadline
            end = T_limit
            start = max(0, end - p)
        
        energy = float(e_single) * float(np.sum(ct[start:end]))
        total_energy += energy
        
        bars.append({
            "job_id": job_id,
            "start": start,
            "end": end,
            "family_id": family_id,
            "energy": energy,
        })
        t = end
    
    return bars, total_energy


def _mean(x: List[float]) -> float:
    arr = np.array(x, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return float("nan")
    return float(arr[finite].mean())


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
        if dur >= 3:
            ax_top.text(start + dur / 2, 0.5, f"p={price}",
                       ha="center", va="center", fontsize=8, fontweight="bold")
    
    ax_top.set_ylim(0, 1)
    ax_top.set_yticks([])
    ax_top.set_title(
        f"Schedule Comparison | n_jobs={n_jobs} T_limit={T_limit}{title_suffix}",
        fontsize=11, fontweight="bold"
    )
    
    legend_elements = [
        mpatches.Patch(facecolor=price_colors[c], edgecolor="black", label=f"price={c}")
        for c in sorted(price_colors.keys())
    ]
    ax_top.legend(handles=legend_elements, loc="upper right", fontsize=8)
    
    # Draw job bars
    job_colors = plt.cm.Set3(np.linspace(0, 1, max(1, n_jobs)))
    
    method_labels = []
    for m in method_order:
        s = schedules[m]
        label = f"{m} (E={s['energy']:.1f})"
        if not s.get("complete", True):
            label += " [INCOMPLETE]"
        method_labels.append(label)
    
    y_positions = list(range(len(method_order)))
    ax_jobs.set_yticks(y_positions)
    ax_jobs.set_yticklabels(method_labels, fontsize=9)
    
    for row, m in enumerate(method_order):
        bars = schedules[m]["bars"]
        for b in bars:
            job_id = int(b["job_id"])
            start = int(b["start"])
            end = int(b["end"])
            dur = max(0, end - start)
            if dur <= 0:
                continue
            ax_jobs.barh(
                row, dur, left=start, height=0.6,
                color=job_colors[job_id % len(job_colors)],
                edgecolor="black", linewidth=0.8
            )
            if dur >= 4:
                ax_jobs.text(start + dur / 2, row, f"J{job_id}",
                           ha="center", va="center", fontsize=7, fontweight="bold")
    
    ax_jobs.set_xlim(0, T_limit)
    ax_jobs.set_xlabel("Time")
    ax_jobs.grid(axis="x", alpha=0.25, linestyle="--")
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate ppo_family_q4_ctx13 with SGBS + baselines and visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--variant_id", type=str, default="ppo_family_q4_ctx13")
    p.add_argument("--which", type=str, default="best", choices=["best", "latest"])
    p.add_argument("--eval_seed", type=int, required=True)
    p.add_argument("--num_instances", type=int, default=64)
    p.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    
    # SGBS parameters
    p.add_argument("--beta", type=int, default=4)
    p.add_argument("--gamma", type=int, default=4)
    
    # Scale and visualization
    p.add_argument("--scale", type=str, default=None, 
                   choices=["small", "medium", "large"],
                   help="Instance scale: small (T<=100), medium (100<T<=350), large (T>350)")
    p.add_argument("--T_max", type=int, default=None, help="Force specific T_max value")
    p.add_argument("--num_viz", type=int, default=2, 
                   help="Number of instances to visualize (0 to skip)")
    
    # Output paths
    p.add_argument("--out_csv", type=str, default=None)
    p.add_argument("--out_json", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None, help="Directory for visualization outputs")
    
    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    if args.run_dir is None and args.checkpoint is None:
        raise ValueError("Provide either --run_dir or --checkpoint")
    if args.run_dir is not None and args.checkpoint is not None:
        raise ValueError("Provide only one of --run_dir or --checkpoint")
    
    run_dir: Path | None = None
    run_cfg: Dict[str, Any] | None = None
    if args.run_dir is not None:
        run_dir = _resolve_run_dir(args.run_dir)
        run_cfg_path = run_dir / "run_config.yaml"
        if run_cfg_path.exists():
            run_cfg = _load_yaml(run_cfg_path)
    
    if args.checkpoint is not None:
        variant_id_str = str(args.variant_id)
        ckpt_path = Path(args.checkpoint)
    else:
        assert run_cfg is not None
        variant_id_str = str(run_cfg.get("variant_id", args.variant_id))
        ckpt_path = run_dir / "checkpoints" / f"{args.which}.pt"
    
    requested_device = str(
        args.device or (run_cfg.get("device") if run_cfg else None) or "cuda"
    )
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but unavailable; falling back to CPU")
        requested_device = "cpu"
    device = torch.device(requested_device)
    
    variant_config = get_variant_config(VariantID(variant_id_str))
    
    # Restrict data config by scale if specified
    data_cfg = _restrict_data_config(variant_config.data, args.scale, args.T_max)
    
    scale_str = args.scale or "mixed"
    print("=" * 70)
    print(f"SGBS Evaluation with Visualization")
    print("=" * 70)
    print(f"Variant: {variant_id_str}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Scale: {scale_str} (T_max choices: {data_cfg.T_max_choices})")
    print(f"Device: {device}")
    print(f"Instances: {args.num_instances}, Viz: {args.num_viz}")
    print(f"SGBS: beta={args.beta}, gamma={args.gamma}")
    print("=" * 70)
    
    # Load model
    model = build_model(variant_config).to(device)
    ckpt = _load_checkpoint(ckpt_path, device)
    model_state = _extract_model_state(ckpt)
    model.load_state_dict(model_state)
    model.eval()
    
    # Generate batch with restricted data config
    batch = generate_episode_batch(
        batch_size=int(args.num_instances),
        config=data_cfg,
        seed=int(args.eval_seed),
        N_job_pad=int(variant_config.env.N_job_pad),
        K_period_pad=250,
        T_max_pad=500,
    )
    
    # Run methods
    print("\nRunning greedy decode...")
    t0 = time.perf_counter()
    greedy_res = greedy_decode(model, variant_config.env, device, batch)
    greedy_time = time.perf_counter() - t0
    print(f"  Greedy done in {greedy_time:.2f}s")
    
    print(f"Running SGBS(beta={args.beta}, gamma={args.gamma})...")
    t0 = time.perf_counter()
    sgbs_res = sgbs(
        model=model,
        env_config=variant_config.env,
        device=device,
        batch_data=batch,
        beta=int(args.beta),
        gamma=int(args.gamma),
    )
    sgbs_time = time.perf_counter() - t0
    print(f"  SGBS done in {sgbs_time:.2f}s")
    
    print("Running SPT+DP...")
    t0 = time.perf_counter()
    spt_res = spt_lpt_with_dp(variant_config.env, device, batch, which="spt")
    spt_time = time.perf_counter() - t0
    print(f"  SPT+DP done in {spt_time:.2f}s")
    
    print("Running LPT+DP...")
    t0 = time.perf_counter()
    lpt_res = spt_lpt_with_dp(variant_config.env, device, batch, which="lpt")
    lpt_time = time.perf_counter() - t0
    print(f"  LPT+DP done in {lpt_time:.2f}s")
    
    # Collect per-instance results
    rows = []
    for i in range(int(args.num_instances)):
        rows.append({
            "instance": i,
            "n_jobs": int(batch["n_jobs"][i]),
            "T_limit": int(batch["T_limit"][i]),
            "greedy_energy": greedy_res[i].total_energy,
            "sgbs_energy": sgbs_res[i].total_energy,
            "spt_dp_energy": spt_res[i].total_energy,
            "lpt_dp_energy": lpt_res[i].total_energy,
        })
    
    # Summary statistics
    greedy_mean = _mean([r.total_energy for r in greedy_res])
    sgbs_mean = _mean([r.total_energy for r in sgbs_res])
    spt_mean = _mean([r.total_energy for r in spt_res])
    lpt_mean = _mean([r.total_energy for r in lpt_res])
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{variant_id_str} | {args.which}.pt | seed={args.eval_seed} | N={args.num_instances} | scale={scale_str}")
    print(f"  greedy: {greedy_mean:.4f}  (time {greedy_time:.2f}s)")
    print(f"  sgbs(b={args.beta},g={args.gamma}): {sgbs_mean:.4f}  (time {sgbs_time:.2f}s)")
    print(f"  spt+dp: {spt_mean:.4f}  (time {spt_time:.2f}s)")
    print(f"  lpt+dp: {lpt_mean:.4f}  (time {lpt_time:.2f}s)")
    
    if greedy_mean > 0:
        sgbs_improvement = (greedy_mean - sgbs_mean) / greedy_mean * 100
        print(f"\n  SGBS improvement over greedy: {sgbs_improvement:.2f}%")
    
    # Visualization
    if args.num_viz > 0:
        print(f"\nGenerating {args.num_viz} schedule visualizations...")
        out_dir = Path(args.out_dir) if args.out_dir else (run_dir or Path.cwd())
        
        for viz_idx in range(min(int(args.num_viz), int(args.num_instances))):
            single = _slice_single_instance(batch, viz_idx)
            n_jobs = int(single["n_jobs"][0])
            T_limit = int(single["T_limit"][0])
            p_subset = single["p_subset"][0][:n_jobs].astype(np.int32)
            
            schedules = {}
            
            # SPT+DP schedule
            schedules["SPT+DP"] = {
                "energy": spt_res[viz_idx].total_energy,
                "bars": _sequence_schedule_to_bars(
                    spt_res[viz_idx].job_sequence,
                    spt_res[viz_idx].start_times,
                    p_subset
                ),
                "complete": True,
            }
            
            # LPT+DP schedule 
            schedules["LPT+DP"] = {
                "energy": lpt_res[viz_idx].total_energy,
                "bars": _sequence_schedule_to_bars(
                    lpt_res[viz_idx].job_sequence,
                    lpt_res[viz_idx].start_times,
                    p_subset
                ),
                "complete": True,
            }
            
            # Greedy - from action trace
            if greedy_res[viz_idx].actions is not None:
                greedy_bars, greedy_energy = _action_trace_to_bars_family(
                    greedy_res[viz_idx].actions,
                    variant_config.env,
                    p_subset,
                    single["ct"][0],
                    int(single["e_single"][0]),
                    T_limit,
                    single["period_starts"][0],
                    single["Tk"][0],
                    single["ck"][0],
                    int(single["K"][0]),
                )
                schedules["Greedy"] = {
                    "energy": greedy_res[viz_idx].total_energy,
                    "bars": greedy_bars,
                    "complete": len(greedy_bars) == n_jobs,
                }
            
            # SGBS - from action trace
            if sgbs_res[viz_idx].actions is not None:
                sgbs_bars, sgbs_energy = _action_trace_to_bars_family(
                    sgbs_res[viz_idx].actions,
                    variant_config.env,
                    p_subset,
                    single["ct"][0],
                    int(single["e_single"][0]),
                    T_limit,
                    single["period_starts"][0],
                    single["Tk"][0],
                    single["ck"][0],
                    int(single["K"][0]),
                )
                schedules["SGBS"] = {
                    "energy": sgbs_res[viz_idx].total_energy,
                    "bars": sgbs_bars,
                    "complete": len(sgbs_bars) == n_jobs,
                }
            
            out_png = out_dir / f"schedule_viz_{scale_str}_seed{args.eval_seed}_idx{viz_idx}.png"
            visualize_schedule(
                out_png,
                {
                    "n_jobs": n_jobs,
                    "T_limit": T_limit,
                    "Tk": single["Tk"][0],
                    "ck": single["ck"][0],
                    "period_starts": single["period_starts"][0],
                    "K": int(single["K"][0]),
                },
                schedules,
                f" | seed={args.eval_seed} idx={viz_idx}"
            )
    
    # Save CSV and JSON
    default_base = run_dir if run_dir is not None else Path.cwd()
    ckpt_tag = args.which if args.checkpoint is None else Path(args.checkpoint).stem
    out_csv = (
        Path(args.out_csv) if args.out_csv
        else default_base / f"eval_family_{scale_str}_{ckpt_tag}_seed{args.eval_seed}_b{args.beta}_g{args.gamma}.csv"
    )
    out_json = Path(args.out_json) if args.out_json else out_csv.with_suffix(".json")
    
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)
    
    summary = {
        "variant_id": variant_id_str,
        "checkpoint": args.which,
        "eval_seed": int(args.eval_seed),
        "num_instances": int(args.num_instances),
        "scale": scale_str,
        "beta": int(args.beta),
        "gamma": int(args.gamma),
        "means": {
            "greedy_energy": greedy_mean,
            "sgbs_energy": sgbs_mean,
            "spt_dp_energy": spt_mean,
            "lpt_dp_energy": lpt_mean,
        },
        "times_sec": {
            "greedy": float(greedy_time),
            "sgbs": float(sgbs_time),
            "spt_dp": float(spt_time),
            "lpt_dp": float(lpt_time),
        },
    }
    
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nWrote: {out_csv}")
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()

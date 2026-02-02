#!/usr/bin/env python3
"""
Evaluation script for Q-Sequence model with SGBS+EAS test-time fine-tuning.

Runs the SGBS+EAS hybrid algorithm from the SGBS paper (Algorithm 2) for
test-time adaptation of Q-sequence models.

Methods compared:
- Greedy: argmin Q-value, no search
- SGBS(β, γ): Simulation-guided beam search
- SGBS+EAS: Hybrid with test-time fine-tuning
- SPT+DP: Shortest Processing Time baseline

Example:
    python -m PaST.cli.eval.run_eval_eas_q_sequence \\
        --checkpoint "models/ensia hpc/q seq cwe/Checkpoint 85.pt" \\
        --scale small --num_instances 10 --max_iterations 50 \\
        --device cuda --out_dir eas_q_results
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
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Add project root
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PaST.config import VariantID, get_variant_config
from PaST.sm_benchmark_data import generate_episode_batch
from PaST.baselines_sequence_dp import spt_lpt_with_dp

from PaST.artifacts.algorithms.eas_q_sequence import (
    EASQConfig,
    EASQSequenceRunner,
    _greedy_decode,
    _slice_single_instance,
    sgbs_q_sequence_with_eas,
    EASQModelWrapper,
)


# =============================================================================
# Utilities
# =============================================================================


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    """Load checkpoint from file."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    return ckpt


def _extract_model_state(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model state dict from checkpoint."""
    if "model_state" in ckpt:
        return ckpt["model_state"]
    sample_keys = list(ckpt.keys())[:5]
    if any("." in k for k in sample_keys):
        return ckpt
    raise KeyError("Could not find model state in checkpoint")


def _restrict_data_config(data, scale: Optional[str]):
    """Restrict data config to specific scale."""
    cfg = replace(data)
    if scale is None:
        return cfg
    s = scale.lower()
    if s == "small":
        cfg.T_max_choices = [t for t in cfg.T_max_choices if t <= 100]
    elif s == "medium":
        cfg.T_max_choices = [t for t in cfg.T_max_choices if 100 < t <= 350]
    elif s == "large":
        cfg.T_max_choices = [t for t in cfg.T_max_choices if t > 350]
    return cfg


def _mean(vals: List[float]) -> float:
    """Compute mean of finite values."""
    arr = np.array(vals, dtype=np.float64)
    finite = np.isfinite(arr)
    return float(arr[finite].mean()) if finite.any() else float("nan")


# =============================================================================
# Main
# =============================================================================


def parse_args():
    p = argparse.ArgumentParser(description="EAS+SGBS for Q-sequence models")
    
    # Model
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to Q-sequence model checkpoint")
    p.add_argument("--variant_id", type=str, default="q_sequence_cwe_ctx13",
                   help="Variant ID for model config")
    
    # Data
    p.add_argument("--scale", type=str, default="small",
                   choices=["small", "medium", "large"])
    p.add_argument("--num_instances", type=int, default=10)
    p.add_argument("--eval_seed", type=int, default=42)
    
    # Device
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--torch_threads", type=int, default=4)
    
    # SGBS parameters
    p.add_argument("--beta", type=int, default=4, help="SGBS beam width")
    p.add_argument("--gamma", type=int, default=4, help="SGBS expansion factor")
    
    # EAS parameters
    p.add_argument("--max_iterations", type=int, default=50,
                   help="Max EAS iterations per instance")
    p.add_argument("--samples_per_iter", type=int, default=16,
                   help="Samples per EAS iteration")
    p.add_argument("--learning_rate", type=float, default=0.005,
                   help="EAS learning rate")
    p.add_argument("--il_weight", type=float, default=0.02,
                   help="Imitation learning loss weight")
    p.add_argument("--patience", type=int, default=20,
                   help="Early stopping patience")
    p.add_argument("--max_time_per_instance", type=float, default=None,
                   help="Max seconds per instance")
    
    # Output
    p.add_argument("--out_dir", type=str, default="eas_q_results")
    
    # Random baseline
    p.add_argument("--include_random", action="store_true", default=True,
                   help="Include random baseline in comparison")
    p.add_argument("--random_seed", type=int, default=42,
                   help="Random seed for random baseline")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    # Setup
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model config
    try:
        variant_id = VariantID(args.variant_id)
    except ValueError:
        variant_id = VariantID.Q_SEQUENCE_CWE
    
    q_cfg = get_variant_config(variant_id)
    
    # Restrict scale
    q_cfg = replace(q_cfg, data=_restrict_data_config(q_cfg.data, args.scale))
    
    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt_path = Path(args.checkpoint)
    ckpt = _load_checkpoint(ckpt_path, device)
    model_state = _extract_model_state(ckpt)
    
    # Build model
    from PaST.models.q_sequence_model import build_q_model
    q_model = build_q_model(q_cfg)
    q_model.load_state_dict(model_state, strict=False)
    q_model.to(device)
    q_model.eval()
    
    print(f"Model loaded: {type(q_model).__name__}")
    
    # Create EAS config
    eas_config = EASQConfig(
        sgbs_beta=args.beta,
        sgbs_gamma=args.gamma,
        max_iterations=args.max_iterations,
        samples_per_iter=args.samples_per_iter,
        learning_rate=args.learning_rate,
        il_weight=args.il_weight,
        patience=args.patience,
    )
    
    # Generate instances
    print(f"Generating {args.num_instances} instances (scale={args.scale}, seed={args.eval_seed})")
    
    batch = generate_episode_batch(
        batch_size=args.num_instances,
        config=q_cfg.data,
        seed=args.eval_seed,
    )
    
    # Run evaluation
    results = []
    total_start = time.time()
    
    for i in range(args.num_instances):
        single_data = _slice_single_instance(batch, i)
        n_jobs = int(single_data["n_jobs"][0])
        T_limit = int(single_data["T_limit"][0])
        
        print(f"\n=== Instance {i+1}/{args.num_instances}: n={n_jobs}, T={T_limit} ===")
        
        # --- Random baseline (run first for comparison) ---
        random_energy = float("nan")
        random_gap = float("nan")
        if args.include_random:
            import random as py_random
            from PaST.baselines_sequence_dp import dp_schedule_for_job_sequence
            py_rng = py_random.Random(args.random_seed + i)
            random_seq = list(range(n_jobs))
            py_rng.shuffle(random_seq)
            random_result = dp_schedule_for_job_sequence(single_data, random_seq)
            random_energy = float(random_result.total_energy)
        
        # Create EAS runner for this instance
        runner = EASQSequenceRunner(q_model, q_cfg, device, eas_config)
        
        # Run SGBS+EAS
        t0 = time.time()
        eas_result = runner.search(single_data, max_time=args.max_time_per_instance)
        eas_time = time.time() - t0
        
        # Get SGBS-only result (no EAS)
        eas_model_wrapper = EASQModelWrapper(q_model, eas_config)
        eas_model_wrapper.to(device)
        eas_model_wrapper.reset_eas_parameters()
        
        t0 = time.time()
        sgbs_result = sgbs_q_sequence_with_eas(
            model=eas_model_wrapper,
            variant_config=q_cfg,
            single_data=single_data,
            device=device,
            beta=args.beta,
            gamma=args.gamma,
        )
        sgbs_time = time.time() - t0
        
        # Get SPT baseline
        spt_results = spt_lpt_with_dp(q_cfg.env, device, single_data, "spt")
        spt_result = spt_results[0]
        
        # Compute gaps - include random in best if available
        candidates = [eas_result.best_energy, sgbs_result.total_energy, spt_result.total_energy]
        if args.include_random and not np.isnan(random_energy):
            candidates.append(random_energy)
        best_energy = min(candidates)
        
        greedy_gap = (eas_result.greedy_energy - best_energy) / best_energy if best_energy > 0 else 0
        sgbs_gap = (sgbs_result.total_energy - best_energy) / best_energy if best_energy > 0 else 0
        eas_gap = (eas_result.best_energy - best_energy) / best_energy if best_energy > 0 else 0
        spt_gap = (spt_result.total_energy - best_energy) / best_energy if best_energy > 0 else 0
        if args.include_random and not np.isnan(random_energy) and best_energy > 0:
            random_gap = (random_energy - best_energy) / best_energy
        
        row = {
            "instance_idx": i,
            "n_jobs": n_jobs,
            "T_limit": T_limit,
            "greedy_energy": eas_result.greedy_energy,
            "sgbs_energy": sgbs_result.total_energy,
            "eas_energy": eas_result.best_energy,
            "spt_energy": spt_result.total_energy,
            "random_energy": random_energy if args.include_random else float("nan"),
            "best_energy": best_energy,
            "greedy_gap": greedy_gap,
            "sgbs_gap": sgbs_gap,
            "eas_gap": eas_gap,
            "spt_gap": spt_gap,
            "random_gap": random_gap if args.include_random else float("nan"),
            "eas_improvement": (eas_result.greedy_energy - eas_result.best_energy) / eas_result.greedy_energy if eas_result.greedy_energy > 0 else 0,
            "eas_iterations": eas_result.iterations,
            "eas_time_s": eas_time,
            "sgbs_time_s": sgbs_time,
        }
        results.append(row)
        
        print(f"  Greedy: {eas_result.greedy_energy:.2f}")
        print(f"  SGBS:   {sgbs_result.total_energy:.2f} (gap: {sgbs_gap*100:.2f}%)")
        print(f"  EAS:    {eas_result.best_energy:.2f} (gap: {eas_gap*100:.2f}%, iter: {eas_result.iterations}, time: {eas_time:.1f}s)")
        print(f"  SPT:    {spt_result.total_energy:.2f} (gap: {spt_gap*100:.2f}%)")
        if args.include_random:
            print(f"  Random: {random_energy:.2f} (gap: {random_gap*100:.2f}%)")
    
    total_time = time.time() - total_start
    
    # Summary statistics - compute mean of random gap if included
    random_gaps = [r['random_gap'] for r in results if 'random_gap' in r and not np.isnan(r['random_gap'])]
    
    summary = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "scale": args.scale,
            "num_instances": args.num_instances,
            "beta": args.beta,
            "gamma": args.gamma,
            "max_iterations": args.max_iterations,
            "samples_per_iter": args.samples_per_iter,
        },
        "results": {
            "mean_greedy_gap": _mean([r["greedy_gap"] for r in results]),
            "mean_sgbs_gap": _mean([r["sgbs_gap"] for r in results]),
            "mean_eas_gap": _mean([r["eas_gap"] for r in results]),
            "mean_spt_gap": _mean([r["spt_gap"] for r in results]),
            "mean_random_gap": _mean(random_gaps) if random_gaps else float("nan"),
            "mean_eas_improvement": _mean([r["eas_improvement"] for r in results]),
            "mean_eas_iterations": _mean([r["eas_iterations"] for r in results]),
            "mean_eas_time_s": _mean([r["eas_time_s"] for r in results]),
            "total_time_s": total_time,
        },
    }
    
    # Save results
    csv_path = out_dir / f"eas_q_results_{args.scale}_seed{args.eval_seed}.csv"
    json_path = out_dir / f"eas_q_summary_{args.scale}_seed{args.eval_seed}.json"
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Instances: {args.num_instances}")
    print(f"  Mean Greedy Gap: {summary['results']['mean_greedy_gap']*100:.2f}%")
    print(f"  Mean SGBS Gap:   {summary['results']['mean_sgbs_gap']*100:.2f}%")
    print(f"  Mean EAS Gap:    {summary['results']['mean_eas_gap']*100:.2f}%")
    print(f"  Mean SPT Gap:    {summary['results']['mean_spt_gap']*100:.2f}%")
    if args.include_random and random_gaps:
        print(f"  Mean Random Gap: {summary['results']['mean_random_gap']*100:.2f}%")
    print(f"  Mean EAS Improvement: {summary['results']['mean_eas_improvement']*100:.2f}%")
    print(f"  Total Time: {total_time:.1f}s")
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()

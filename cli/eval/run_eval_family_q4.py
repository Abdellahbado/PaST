"""Run evaluation for PaST-SM with SGBS and baselines for ppo_family_q4_ctx13 variant.

This script evaluates the price-family model variant (ppo_family_q4_ctx13) which uses:
- Action space: job × price_family (50 jobs × 4 families = 200 actions)
- Context dimension: 13 (extended with price quantiles + next-slot deltas)

Methods compared:
- Greedy model decode
- SGBS(beta, gamma) - Simulation-guided Beam Search (NeurIPS 2022)
- SPT + DP scheduling (optimal start times for the fixed SPT order)
- LPT + DP scheduling (optimal start times for the fixed LPT order)

Example usage:
    python run_eval_family_q4.py \
        --run_dir runs_p100/ppo_family_q4_ctx13/seed_0 \
        --which best \
        --eval_seed 1337 \
        --num_instances 64 \
        --beta 4 --gamma 4

    # With direct checkpoint path:
    python run_eval_family_q4.py \
        --checkpoint runs_p100/ppo_family_q4_ctx13/seed_0/checkpoints/best.pt \
        --variant_id ppo_family_q4_ctx13 \
        --eval_seed 1337 \
        --num_instances 64 \
        --beta 4 --gamma 4
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

from PaST.baselines_sequence_dp import spt_lpt_with_dp
from PaST.config import VariantID, get_variant_config
from PaST.past_sm_model import build_model
from PaST.sgbs import greedy_decode, sgbs
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
        candidate = cwd / stripped
        if candidate.exists():
            return candidate

    attempted = [
        str(raw),
        str(pkg_root / raw),
        str(cwd / "PaST" / raw),
    ]
    raise FileNotFoundError(
        "Run directory not found. Tried:\n  " + "\n  ".join(attempted)
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate ppo_family_q4_ctx13 checkpoint with SGBS + baselines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Run directory (option A). If it contains run_config.yaml, variant_id/device are inferred.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path (option B). Requires --variant_id.",
    )
    p.add_argument(
        "--variant_id",
        type=str,
        default="ppo_family_q4_ctx13",
        help="Variant id (default: ppo_family_q4_ctx13).",
    )
    p.add_argument(
        "--which",
        type=str,
        default="best",
        choices=["best", "latest"],
        help="Checkpoint name under <run_dir>/checkpoints (only used with --run_dir).",
    )
    p.add_argument("--eval_seed", type=int, required=True)
    p.add_argument("--num_instances", type=int, default=64)
    p.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])

    p.add_argument("--beta", type=int, default=4)
    p.add_argument("--gamma", type=int, default=4)

    p.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="CSV output path (defaults to <run_dir>/eval_family_<which>_seed<seed>_b<beta>_g<gamma>.csv)",
    )
    p.add_argument(
        "--out_json",
        type=str,
        default=None,
        help="JSON summary output path (defaults alongside CSV)",
    )
    return p.parse_args()


def _mean(x: List[float]) -> float:
    arr = np.array(x, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return float("nan")
    return float(arr[finite].mean())


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
        if not args.variant_id:
            raise ValueError("--variant_id is required when using --checkpoint")
        variant_id_str = str(args.variant_id)
        ckpt_path = Path(args.checkpoint)
    else:
        assert (
            run_cfg is not None
        ), "run_cfg should be loaded when using --run_dir with run_config.yaml"
        variant_id_str = str(run_cfg.get("variant_id", args.variant_id))
        ckpt_path = run_dir / "checkpoints" / f"{args.which}.pt"  # type: ignore[operator]

    requested_device = str(
        args.device or (run_cfg.get("device") if run_cfg else None) or "cuda"
    )
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but unavailable; falling back to CPU")
        requested_device = "cpu"
    device = torch.device(requested_device)

    print(f"=" * 70)
    print(f"SGBS Evaluation for PPO Family Q4 CTX13")
    print(f"=" * 70)
    print(f"Variant: {variant_id_str}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Device: {device}")
    print(f"Eval seed: {args.eval_seed}")
    print(f"Num instances: {args.num_instances}")
    print(f"SGBS params: beta={args.beta}, gamma={args.gamma}")
    print(f"=" * 70)

    variant_config = get_variant_config(VariantID(variant_id_str))

    model = build_model(variant_config).to(device)
    ckpt = _load_checkpoint(ckpt_path, device)
    model_state = _extract_model_state(ckpt)
    model.load_state_dict(model_state)
    model.eval()

    print(f"Model loaded. Action dim: {variant_config.env.action_dim}")
    print(f"  Price families: {variant_config.env.num_price_families}")
    print(f"  Context dim: {variant_config.env.F_ctx}")
    print()

    batch = generate_episode_batch(
        batch_size=int(args.num_instances),
        config=variant_config.data,
        seed=int(args.eval_seed),
        N_job_pad=int(variant_config.env.N_job_pad),
        K_period_pad=250,
        T_max_pad=500,
    )

    rows: List[Dict[str, Any]] = []

    # Greedy model
    print("Running greedy decode...")
    t0 = time.perf_counter()
    greedy_res = greedy_decode(model, variant_config.env, device, batch)
    greedy_time = time.perf_counter() - t0
    print(f"  Greedy done in {greedy_time:.2f}s")

    # SGBS
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

    # SPT/LPT + DP
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

    for i in range(int(args.num_instances)):
        rows.append(
            {
                "instance": i,
                "greedy_energy": greedy_res[i].total_energy,
                "sgbs_energy": sgbs_res[i].total_energy,
                "spt_dp_energy": spt_res[i].total_energy,
                "lpt_dp_energy": lpt_res[i].total_energy,
            }
        )

    greedy_mean = _mean([r.total_energy for r in greedy_res])
    sgbs_mean = _mean([r.total_energy for r in sgbs_res])
    spt_mean = _mean([r.total_energy for r in spt_res])
    lpt_mean = _mean([r.total_energy for r in lpt_res])

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(
        f"{variant_id_str} | {args.which}.pt | seed={args.eval_seed} | N={args.num_instances}\n"
        f"  greedy: {greedy_mean:.4f}  (time {greedy_time:.2f}s)\n"
        f"  sgbs(b={args.beta},g={args.gamma}): {sgbs_mean:.4f}  (time {sgbs_time:.2f}s)\n"
        f"  spt+dp: {spt_mean:.4f}  (time {spt_time:.2f}s)\n"
        f"  lpt+dp: {lpt_mean:.4f}  (time {lpt_time:.2f}s)"
    )

    # Compute improvement percentages
    if greedy_mean > 0:
        sgbs_improvement = (greedy_mean - sgbs_mean) / greedy_mean * 100
        print(f"\n  SGBS improvement over greedy: {sgbs_improvement:.2f}%")

    default_base = run_dir if run_dir is not None else Path.cwd()
    ckpt_tag = args.which if args.checkpoint is None else Path(args.checkpoint).stem
    out_csv = (
        Path(args.out_csv)
        if args.out_csv
        else default_base
        / f"eval_family_{ckpt_tag}_seed{args.eval_seed}_b{args.beta}_g{args.gamma}.csv"
    )
    out_json = Path(args.out_json) if args.out_json else out_csv.with_suffix(".json")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "instance",
                "greedy_energy",
                "sgbs_energy",
                "spt_dp_energy",
                "lpt_dp_energy",
            ],
        )
        w.writeheader()
        for row in rows:
            w.writerow(row)

    summary = {
        "variant_id": variant_id_str,
        "checkpoint": args.which,
        "eval_seed": int(args.eval_seed),
        "num_instances": int(args.num_instances),
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
        "outputs": {
            "csv": str(out_csv),
            "json": str(out_json),
        },
    }

    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()

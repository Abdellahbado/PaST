"""Compare full-problem baseline vs ALNS (parallel-machine).

Baseline (as requested):
- CWP assignment
- random sequencing per machine (equivalent to random Q-values ordering)
- DP timing per machine

ALNS:
- CWP assignment + SPT init sequencing (stronger, lower-variance start)
- LNS/ALNS destroy+repair moves over assignment+sequencing
- DP timing evaluation (exact for fixed sequence)

This script produces:
- CSV summary in analysis_out/
- Optional JSON with per-iteration logs and oracle dataset

Example:
  PYTHONPATH=. python -m PaST.cli.compare.compare_alns_vs_baseline_pm \
    --scale small --num_instances 20 --seed 42 --slack_ratio 0.25 \
    --alns_iters 300 --oracle_dataset
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from PaST.config import DataConfig
from PaST.data.sm_benchmark_data import generate_raw_instance, RawInstance
from PaST.solvers.alns_parallel import (
    ALNSConfig,
    _compute_epsilon,
    alns_solve,
    build_baseline_solution_cwp_random,
    eval_to_jsonable,
    save_oracle_dataset_npz,
    solution_to_jsonable,
)


def _ensure_scale_choices(cfg: DataConfig, scale: str) -> None:
    scale = str(scale).lower()
    if scale == "small":
        cfg.T_max_choices = [t for t in cfg.T_max_choices if int(t) <= 80]
    elif scale in {"mls", "medium"}:
        cfg.T_max_choices = [t for t in cfg.T_max_choices if 80 < int(t) <= 300]
    elif scale in {"vls", "large"}:
        cfg.T_max_choices = [t for t in cfg.T_max_choices if int(t) > 300]
    else:
        raise ValueError("scale must be one of: small, mls, vls")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--scale", type=str, default="small")
    p.add_argument("--num_instances", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--slack_ratio",
        type=float,
        default=0.25,
        help="epsilon interpolation in [T_min,T_max]",
    )

    p.add_argument("--alns_iters", type=int, default=250)
    p.add_argument("--alns_no_improve", type=int, default=90)
    p.add_argument("--destroy_frac", type=float, default=0.15)
    p.add_argument("--destroy_max", type=int, default=40)

    p.add_argument("--oracle_dataset", action="store_true")
    p.add_argument("--oracle_max_states", type=int, default=256)

    p.add_argument("--verify_cost", action="store_true")
    p.add_argument(
        "--log_per_machine_every",
        type=int,
        default=0,
        help="If >0, include per-machine (energy,makespan) snapshots every N ALNS iters in JSON logs",
    )

    p.add_argument("--out_prefix", type=str, default="alns_vs_baseline")
    p.add_argument("--save_json", action="store_true")
    return p


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for r in rows:
        vals = []
        for k in keys:
            v = r.get(k)
            if v is None:
                vals.append("")
            else:
                s = str(v)
                if "," in s:
                    s = s.replace(",", ";")
                vals.append(s)
        lines.append(",".join(vals))
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = build_parser().parse_args()

    out_dir = Path("analysis_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DataConfig()
    _ensure_scale_choices(cfg, args.scale)

    rng = random.Random(int(args.seed))

    summary_rows: List[Dict[str, Any]] = []
    json_rows: List[Dict[str, Any]] = []

    for i in range(int(args.num_instances)):
        raw = generate_raw_instance(cfg, rng, instance_id=i)
        epsilon = _compute_epsilon(raw, float(args.slack_ratio))

        # Baseline
        base_rng = random.Random(int(args.seed) * 10_000 + i)
        base_sol, base_ev, base_ms = build_baseline_solution_cwp_random(
            raw, epsilon, base_rng
        )

        # ALNS
        alns_cfg = ALNSConfig(
            max_iters=int(args.alns_iters),
            no_improve_limit=int(args.alns_no_improve),
            destroy_frac=float(args.destroy_frac),
            destroy_max=int(args.destroy_max),
            keep_iter_log=bool(args.save_json),
            oracle_dataset=bool(args.oracle_dataset),
            oracle_max_states=int(args.oracle_max_states),
            verify_cost=bool(args.verify_cost),
            log_per_machine_every=int(args.log_per_machine_every),
        )

        alns_rng = random.Random(int(args.seed) * 100_000 + i)
        alns_out = alns_solve(raw, epsilon, alns_cfg, alns_rng)
        alns_ev = alns_out["best_eval"]
        alns_ms = float(alns_out["solve_time_ms"])

        imp = None
        if base_ev.feasible and alns_ev.feasible:
            if float(base_ev.total_energy) > 0:
                imp = (
                    100.0
                    * (float(base_ev.total_energy) - float(alns_ev.total_energy))
                    / float(base_ev.total_energy)
                )
            else:
                imp = 0.0

        row = {
            "instance_id": int(raw.instance_id),
            "scale": str(raw.scale),
            "m": int(raw.m),
            "n": int(raw.n),
            "T_max": int(raw.T_max),
            "epsilon": int(epsilon),
            "baseline_energy": (
                float(base_ev.total_energy) if base_ev.feasible else "inf"
            ),
            "baseline_makespan": int(base_ev.makespan),
            "baseline_feasible": bool(base_ev.feasible),
            "baseline_time_ms": float(base_ms),
            "alns_energy": float(alns_ev.total_energy) if alns_ev.feasible else "inf",
            "alns_makespan": int(alns_ev.makespan),
            "alns_feasible": bool(alns_ev.feasible),
            "alns_time_ms": float(alns_ms),
            "improvement_pct": float(imp) if imp is not None else "",
        }
        summary_rows.append(row)

        if args.save_json:
            jrow = {
                "raw": asdict(raw),
                "epsilon": int(epsilon),
                "baseline": {
                    "eval": eval_to_jsonable(base_ev),
                    "solution": solution_to_jsonable(base_sol),
                },
                "alns": {
                    "best_eval": eval_to_jsonable(alns_ev),
                    "best_solution": solution_to_jsonable(alns_out["best_solution"]),
                    "iter_log": alns_out.get("iter_log", []),
                },
            }
            json_rows.append(jrow)

        # Save oracle dataset per instance if requested
        if args.oracle_dataset and "oracle_dataset" in alns_out:
            ds = alns_out["oracle_dataset"]
            npz_path = out_dir / f"{args.out_prefix}_oracle_inst{i}_seed{args.seed}.npz"
            save_oracle_dataset_npz(ds, str(npz_path))

        base_str = f"{base_ev.total_energy:.2f}" if base_ev.feasible else "inf"
        alns_str = f"{alns_ev.total_energy:.2f}" if alns_ev.feasible else "inf"
        imp_str = f"{imp:.2f}%" if imp is not None else "n/a"
        print(
            f"[{i+1}/{args.num_instances}] eps={epsilon} baseline={base_str} alns={alns_str} imp={imp_str}"
        )

    ts = int(time.time())
    csv_path = out_dir / f"{args.out_prefix}_{args.scale}_seed{args.seed}_{ts}.csv"
    _write_csv(csv_path, summary_rows)
    print(f"Wrote {csv_path}")

    if args.save_json:
        json_path = (
            out_dir / f"{args.out_prefix}_{args.scale}_seed{args.seed}_{ts}.json"
        )
        json_path.write_text(json.dumps(json_rows, indent=2))
        print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()

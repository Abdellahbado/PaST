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
import concurrent.futures
import json
import os
import random
import sys
import time
from functools import partial
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow running this file directly (python PaST/cli/compare/compare_*.py)
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
    p.add_argument(
        "--json_per_instance",
        action="store_true",
        help="If set with --save_json, write one JSON per instance (faster + less RAM).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel worker processes (0=use all CPU cores).",
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=1,
        help="Task chunk size for parallel mapping (higher reduces overhead).",
    )
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


def _run_one_instance(
    instance_id: int,
    *,
    scale: str,
    seed: int,
    slack_ratio: float,
    alns_cfg_dict: Dict[str, Any],
    out_dir_str: str,
    out_prefix: str,
    save_json: bool,
    json_per_instance: bool,
    oracle_dataset: bool,
    oracle_max_states: int,
    verify_cost: bool,
    log_per_machine_every: int,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[str]]:
    """Worker body (must be top-level for multiprocessing pickling)."""

    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DataConfig()
    _ensure_scale_choices(cfg, scale)

    # Deterministic per-instance RNG (independent of execution order)
    gen_rng = random.Random(int(seed) * 1_000_003 + int(instance_id) * 97)
    raw = generate_raw_instance(cfg, gen_rng, instance_id=int(instance_id))
    epsilon = _compute_epsilon(raw, float(slack_ratio))

    # Baseline
    base_rng = random.Random(int(seed) * 10_000 + int(instance_id))
    base_sol, base_ev, base_ms = build_baseline_solution_cwp_random(
        raw, epsilon, base_rng
    )

    # ALNS
    alns_cfg = ALNSConfig(**alns_cfg_dict)
    # Ensure flags are consistent with worker args
    alns_cfg = ALNSConfig(
        **{
            **alns_cfg_dict,
            "oracle_dataset": bool(oracle_dataset),
            "oracle_max_states": int(oracle_max_states),
            "verify_cost": bool(verify_cost),
            "log_per_machine_every": int(log_per_machine_every),
        }
    )
    alns_rng = random.Random(int(seed) * 100_000 + int(instance_id))
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

    summary = {
        "instance_id": int(raw.instance_id),
        "scale": str(raw.scale),
        "m": int(raw.m),
        "n": int(raw.n),
        "T_max": int(raw.T_max),
        "epsilon": int(epsilon),
        "baseline_energy": float(base_ev.total_energy) if base_ev.feasible else "inf",
        "baseline_makespan": int(base_ev.makespan),
        "baseline_feasible": bool(base_ev.feasible),
        "baseline_time_ms": float(base_ms),
        "alns_energy": float(alns_ev.total_energy) if alns_ev.feasible else "inf",
        "alns_makespan": int(alns_ev.makespan),
        "alns_feasible": bool(alns_ev.feasible),
        "alns_time_ms": float(alns_ms),
        "improvement_pct": float(imp) if imp is not None else "",
    }

    json_row: Optional[Dict[str, Any]] = None
    json_path: Optional[str] = None
    if save_json:
        json_row = {
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

        if json_per_instance:
            json_path = str(
                out_dir / f"{out_prefix}_{scale}_seed{seed}_inst{int(instance_id)}.json"
            )
            Path(json_path).write_text(json.dumps(json_row, indent=2))
            # Avoid sending large payloads back to parent
            json_row = None

    # Save oracle dataset per instance if requested
    if oracle_dataset and "oracle_dataset" in alns_out:
        ds = alns_out["oracle_dataset"]
        npz_path = (
            out_dir / f"{out_prefix}_oracle_inst{int(instance_id)}_seed{seed}.npz"
        )
        save_oracle_dataset_npz(ds, str(npz_path))

    return summary, json_row, json_path


def main() -> None:
    args = build_parser().parse_args()

    out_dir = Path("analysis_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build ALNS config dict once (picklable)
    alns_cfg_dict: Dict[str, Any] = {
        "max_iters": int(args.alns_iters),
        "no_improve_limit": int(args.alns_no_improve),
        "destroy_frac": float(args.destroy_frac),
        "destroy_max": int(args.destroy_max),
        "keep_iter_log": bool(args.save_json),
        "oracle_dataset": bool(args.oracle_dataset),
        "oracle_max_states": int(args.oracle_max_states),
        "verify_cost": bool(args.verify_cost),
        "log_per_machine_every": int(args.log_per_machine_every),
    }

    workers = int(args.workers)
    if workers <= 0:
        workers = max(1, os.cpu_count() or 1)

    summary_rows: List[Dict[str, Any]] = []
    json_rows: List[Dict[str, Any]] = []

    run_one = partial(
        _run_one_instance,
        scale=str(args.scale),
        seed=int(args.seed),
        slack_ratio=float(args.slack_ratio),
        alns_cfg_dict=alns_cfg_dict,
        out_dir_str=str(out_dir),
        out_prefix=str(args.out_prefix),
        save_json=bool(args.save_json),
        json_per_instance=bool(args.json_per_instance),
        oracle_dataset=bool(args.oracle_dataset),
        oracle_max_states=int(args.oracle_max_states),
        verify_cost=bool(args.verify_cost),
        log_per_machine_every=int(args.log_per_machine_every),
    )

    ids = list(range(int(args.num_instances)))

    if workers == 1:
        for idx, i in enumerate(ids, start=1):
            summary, jrow, jpath = run_one(int(i))
            summary_rows.append(summary)
            if jrow is not None:
                json_rows.append(jrow)
            if jpath is not None:
                summary["json_path"] = jpath
            print(f"[{idx}/{len(ids)}] done inst={i}")
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            for idx, (summary, jrow, jpath) in enumerate(
                ex.map(run_one, ids, chunksize=int(args.chunksize)), start=1
            ):
                summary_rows.append(summary)
                if jrow is not None:
                    json_rows.append(jrow)
                if jpath is not None:
                    summary["json_path"] = jpath
                print(f"[{idx}/{len(ids)}] done inst={summary.get('instance_id')}")

    # Keep deterministic ordering in outputs
    summary_rows.sort(key=lambda r: int(r.get("instance_id", 0)))

    ts = int(time.time())
    csv_path = out_dir / f"{args.out_prefix}_{args.scale}_seed{args.seed}_{ts}.csv"
    _write_csv(csv_path, summary_rows)
    print(f"Wrote {csv_path}")

    if args.save_json and (not args.json_per_instance):
        json_path = (
            out_dir / f"{args.out_prefix}_{args.scale}_seed{args.seed}_{ts}.json"
        )
        json_path.write_text(json.dumps(json_rows, indent=2))
        print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()

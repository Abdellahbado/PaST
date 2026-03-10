#!/usr/bin/env python3
"""
Run our C++ DP solver on ALL paper benchmark instances.

Datasets:
  Table 1 (§5.1):  72 instances across 6 datasets (NOSBY + TWOSBY, uniform prices)
  Table 2 (§5.2): 560 instances (benedikt2025b_groups, TWOSBY, OTE prices)
  Figure 9 (§5.3): 560 instances (benedikt2025_groups, TWOSBY, OTE prices)
  Drops ablation:  240 instances (benedikt2025b_drops, TWOSBY, uniform prices)
  GCD analysis:      1 instance  (benedikt2025b_gcd, NOSBY, OTE prices)

Usage:
    python3 cli/experiments/run_all_paper_benchmarks.py [--section SECTION] [--time-limit SEC] [--output CSV]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SOLVER = ROOT / "solvers" / "cpp" / "build" / "stateful_compare"

# Two data directories
TABLE1_DATA = ROOT / "data" / "paper_instances" / "datasets"
TABLE2_DATA = (
    ROOT
    / "data"
    / "green-scheduling-bab"
    / "Iirc.EnergyStatesAndCostsScheduling"
    / "data"
    / "datasets"
)


def load_instance(json_path: Path) -> dict:
    with open(json_path) as f:
        d = json.load(f)
    jobs = [j["ProcessingTime"] for j in d["Jobs"]]
    prices = d["EnergyCosts"]
    noff = len(d.get("OffOnTime", []))
    mtype = "twosby" if noff == 3 else "nosby"
    return {
        "jobs": jobs,
        "prices": prices,
        "n_jobs": len(jobs),
        "horizon": len(prices),
        "machine_type": mtype,
    }


def run_solver_batch(
    instances: list[tuple[str, dict]], solver_path: Path, time_limit: float
) -> list[dict]:
    if not instances:
        return []
    jsonl_lines = []
    for inst_id, inst in instances:
        payload = {
            "instance_id": inst_id,
            "prices": inst["prices"],
            "jobs": inst["jobs"],
            "machine": inst["machine_type"],
        }
        jsonl_lines.append(json.dumps(payload))

    input_data = "\n".join(jsonl_lines) + "\n"
    args = [str(solver_path), "solve-stdin"]
    if time_limit > 0:
        args.append(str(time_limit))

    proc = subprocess.run(
        args,
        input=input_data,
        capture_output=True,
        text=True,
        timeout=int(time_limit * len(instances) + 120),
    )
    if proc.returncode != 0:
        print(f"Solver stderr: {proc.stderr[:500]}", file=sys.stderr)

    results = []
    lines = proc.stdout.strip().split("\n")
    for line in lines[1:]:  # skip header
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) >= 10:
            results.append(
                {
                    "instance_id": parts[0],
                    "n_jobs": int(parts[1]),
                    "horizon": int(parts[2]),
                    "ub": float(parts[3]),
                    "lb": float(parts[4]),
                    "gap_pct": float(parts[5]),
                    "feasible": int(parts[6]),
                    "is_optimal": int(parts[7]),
                    "timed_out": int(parts[8]),
                    "runtime_sec": float(parts[9]),
                }
            )
    return results


def run_dataset(
    dataset_name: str,
    data_dir: Path,
    solver_path: Path,
    time_limit: float,
    batch_size: int = 50,
) -> list[dict]:
    ds_dir = data_dir / dataset_name
    if not ds_dir.exists():
        print(f"  Dataset not found: {ds_dir}", file=sys.stderr)
        return []

    json_files = sorted(ds_dir.glob("*.json"), key=lambda p: int(p.stem))
    instances = []
    for jf in json_files:
        idx = int(jf.stem)
        inst = load_instance(jf)
        inst_id = f"{dataset_name}/{idx}"
        instances.append((inst_id, inst))

    n_total = len(instances)
    mtype = instances[0][1]["machine_type"] if instances else "?"
    print(f"\n{'='*110}")
    print(f"Dataset: {dataset_name} ({n_total} instances, {mtype.upper()})")
    print(f"{'='*110}")

    # Process in batches (to avoid huge stdin for 284 instances)
    all_results = []
    for start in range(0, n_total, batch_size):
        batch = instances[start : start + batch_size]
        batch_results = run_solver_batch(batch, solver_path, time_limit)
        all_results.extend(batch_results)
        done = start + len(batch)
        if n_total > batch_size:
            print(f"  Progress: {done}/{n_total} instances solved...")

    # Print results table
    print(
        f"\n{'Instance':>40} {'n':>5} {'T':>6} {'UB':>14} {'LB':>14} {'gap%':>8} {'opt':>4} {'t[s]':>10}"
    )
    print("-" * 110)

    total_time = 0.0
    total_optimal = 0
    total_open = 0
    for r in all_results:
        gap_str = f"{r['gap_pct']:.4f}" if r["gap_pct"] > 0.001 else "0.0000"
        opt_str = "YES" if r["is_optimal"] else "---"
        print(
            f"{r['instance_id']:>40} {r['n_jobs']:>5} {r['horizon']:>6} "
            f"{r['ub']:>14.1f} {r['lb']:>14.1f} {gap_str:>8} {opt_str:>4} {r['runtime_sec']:>10.4f}"
        )
        total_time += r["runtime_sec"]
        if r["is_optimal"]:
            total_optimal += 1
        else:
            total_open += 1

    print("-" * 110)
    n_res = len(all_results)
    avg_t = total_time / max(n_res, 1)
    print(
        f"{'SUMMARY':>40} optimal={total_optimal}/{n_res}  open={total_open}  "
        f"total_time={total_time:.2f}s  avg={avg_t:.4f}s"
    )
    if total_open > 0:
        # Show details of open gaps
        print(f"\n  OPEN GAPS:")
        for r in all_results:
            if not r["is_optimal"]:
                print(
                    f"    {r['instance_id']:>40} ub={r['ub']:.1f} lb={r['lb']:.1f} gap={r['gap_pct']:.4f}% t={r['runtime_sec']:.4f}s"
                )

    return all_results


def main():
    ap = argparse.ArgumentParser(description="Run solver on ALL paper benchmarks")
    ap.add_argument(
        "--section",
        default="all",
        choices=["1", "2", "fig9", "drops", "gcd", "all"],
        help="Which section: 1 (Table 1), 2 (Table 2), fig9 (Figure 9), drops, gcd, all",
    )
    ap.add_argument("--solver", default=str(SOLVER))
    ap.add_argument(
        "--time-limit",
        type=float,
        default=600,
        help="Time limit per instance (seconds, default: 600)",
    )
    ap.add_argument("--output", default=None, help="Output CSV file")
    ap.add_argument("--dataset", default=None, help="Run specific dataset only")
    ap.add_argument("--batch-size", type=int, default=50)
    args = ap.parse_args()

    solver_path = Path(args.solver)
    if not solver_path.exists():
        print(f"Solver not found: {solver_path}", file=sys.stderr)
        sys.exit(1)

    all_results = []

    # Table 1 datasets (from existing data dir)
    table1_datasets = [
        "benedikt2020a_large_nosby",
        "benedikt2020a_large_twosby",
        "benedikt2020a_medium_nosby",
        "benedikt2020a_medium_twosby",
        "benedikt2020a_prelim",
        "aghelinejad2017a_1",
    ]

    # Table 2 datasets (benedikt2025b_groups — the REAL Table 2)
    table2_datasets = [
        "benedikt2025b_groups",
    ]

    # Figure 9 datasets (benedikt2025_groups)
    fig9_datasets = [
        "benedikt2025_groups",
    ]

    # Drops ablation + GCD
    extra_datasets = [
        "benedikt2025b_drops",
        "benedikt2025b_gcd",
    ]

    # Also check the newly generated Table 1 equivalents
    table1_gen_datasets = [
        "benedikt2025a_large_nosby",
        "benedikt2025a_large_twosby",
        "benedikt2025a_medium_nosby",
        "benedikt2025a_medium_twosby",
        "benedikt2025a_prelim",
    ]

    if args.dataset:
        # Run specific dataset
        for data_dir in [TABLE1_DATA, TABLE2_DATA]:
            if (data_dir / args.dataset).exists():
                all_results.extend(
                    run_dataset(
                        args.dataset,
                        data_dir,
                        solver_path,
                        args.time_limit,
                        args.batch_size,
                    )
                )
                break
        else:
            print(
                f"Dataset '{args.dataset}' not found in either data directory",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        if args.section in ("1", "all"):
            print(f"\n{'#'*110}")
            print(f"# TABLE 1 (§5.1) — Original instances (72 total)")
            print(f"{'#'*110}")
            for ds in table1_datasets:
                if TABLE1_DATA.exists():
                    all_results.extend(
                        run_dataset(
                            ds,
                            TABLE1_DATA,
                            solver_path,
                            args.time_limit,
                            args.batch_size,
                        )
                    )

        if args.section in ("2", "all"):
            print(f"\n{'#'*110}")
            print(
                f"# TABLE 2 (§5.2) — benedikt2025b_groups (560 instances, TWOSBY, OTE prices)"
            )
            print(f"{'#'*110}")
            for ds in table2_datasets:
                all_results.extend(
                    run_dataset(
                        ds, TABLE2_DATA, solver_path, args.time_limit, args.batch_size
                    )
                )

        if args.section in ("fig9", "all"):
            print(f"\n{'#'*110}")
            print(
                f"# FIGURE 9 (§5.3) — benedikt2025_groups (560 instances, TWOSBY, OTE prices)"
            )
            print(f"{'#'*110}")
            for ds in fig9_datasets:
                all_results.extend(
                    run_dataset(
                        ds, TABLE2_DATA, solver_path, args.time_limit, args.batch_size
                    )
                )

        if args.section in ("drops", "all"):
            print(f"\n{'#'*110}")
            print(f"# DROPS ABLATION — benedikt2025b_drops (240 instances, TWOSBY)")
            print(f"{'#'*110}")
            all_results.extend(
                run_dataset(
                    "benedikt2025b_drops",
                    TABLE2_DATA,
                    solver_path,
                    args.time_limit,
                    args.batch_size,
                )
            )

        if args.section in ("gcd", "all"):
            print(f"\n{'#'*110}")
            print(f"# GCD ANALYSIS — benedikt2025b_gcd (1 instance, NOSBY, OTE prices)")
            print(f"{'#'*110}")
            all_results.extend(
                run_dataset(
                    "benedikt2025b_gcd",
                    TABLE2_DATA,
                    solver_path,
                    args.time_limit,
                    args.batch_size,
                )
            )

    # Grand summary
    print(f"\n{'='*110}")
    print(f"GRAND SUMMARY")
    print(f"{'='*110}")
    n = len(all_results)
    n_opt = sum(1 for r in all_results if r["is_optimal"])
    n_open = n - n_opt
    total_t = sum(r["runtime_sec"] for r in all_results)
    print(f"Total instances: {n}")
    print(f"Proven optimal: {n_opt}/{n} ({100*n_opt/max(n,1):.1f}%)")
    print(f"Open gaps: {n_open}")
    print(f"Total solve time: {total_t:.2f}s")
    print(f"Average solve time: {total_t/max(n,1):.4f}s")

    # Write CSV
    if args.output and all_results:
        with open(args.output, "w") as f:
            f.write(
                "instance_id,n_jobs,horizon,ub,lb,gap_pct,feasible,is_optimal,timed_out,runtime_sec\n"
            )
            for r in all_results:
                f.write(
                    f"{r['instance_id']},{r['n_jobs']},{r['horizon']},"
                    f"{r['ub']:.6f},{r['lb']:.6f},{r['gap_pct']:.4f},"
                    f"{r['feasible']},{r['is_optimal']},{r['timed_out']},"
                    f"{r['runtime_sec']:.4f}\n"
                )
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

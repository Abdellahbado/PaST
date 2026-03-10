#!/usr/bin/env python3
"""
Run our C++ DP solver on the paper's actual benchmark instances from:
  https://github.com/CTU-IIG/EnergyStatesAndCostsSchedulingData

Reads JSON instance files, converts them to the solve-stdin JSONL format,
pipes them through the solver, and compares with paper Table 1 results.

Usage:
    python3 cli/experiments/run_paper_instances.py [--dataset DATASET] [--solver PATH] [--time-limit SEC]

Datasets available:
    benedikt2020a_large_nosby     (12 instances, Table 1 top half)
    benedikt2020a_large_twosby    (12 instances, Table 1 bottom half)
    benedikt2020a_medium_nosby    (12 instances)
    benedikt2020a_medium_twosby   (12 instances)
    aghelinejad2017a_1            (12 instances)
    all                           (runs all above)
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
DATA_DIR = ROOT / "data" / "paper_instances" / "datasets"
SOLVER = ROOT / "solvers" / "cpp" / "build" / "stateful_compare"

# Paper Table 1 known optimal costs — NOSBY (keyed by instance file index 0..11)
# Instance order: n=150 λ=1.3,1.6,1.9,2.2  n=170 same  n=190 same
PAPER_TABLE1_NOSBY = {
    0: 8_582,
    1: 8_409,
    2: 8_132,
    3: 8_078,
    4: 10_068,
    5: 9_820,
    6: 9_637,
    7: 9_620,
    8: 12_008,
    9: 11_758,
    10: 11_611,
    11: 11_465,
}

# Paper Table 1 known optimal costs — TWOSBY
PAPER_TABLE1_TWOSBY = {
    0: 21_910,
    1: 21_821,
    2: 21_353,
    3: 21_266,
    4: 25_807,
    5: 25_518,
    6: 25_279,
    7: 25_279,
    8: 30_563,
    9: 30_224,
    10: 30_224,
    11: 30_071,
}

# Paper Table 1 B&B-SPACES algo time (seconds), per instance — NOSBY
PAPER_BB_TIME_NOSBY = {
    0: 0.6,
    1: 0.9,
    2: 0.8,
    3: 1.1,
    4: 0.7,
    5: 0.8,
    6: 1.7,
    7: 2.4,
    8: 0.7,
    9: 1.1,
    10: 2.4,
    11: 2.0,
}

# Paper P-P preprocessing time (seconds), per instance — NOSBY
PAPER_PP_TIME_NOSBY = {
    0: 1.0,
    1: 2.9,
    2: 5.5,
    3: 9.1,
    4: 2.3,
    5: 4.6,
    6: 9.3,
    7: 13.4,
    8: 3.9,
    9: 6.9,
    10: 13.3,
    11: 22.7,
}

# Paper Table 1 B&B-SPACES algo time (seconds) — TWOSBY
PAPER_BB_TIME_TWOSBY = {
    0: 0.9,
    1: 1.2,
    2: 1.3,
    3: 1.4,
    4: 0.8,
    5: 1.1,
    6: 2.2,
    7: 2.0,
    8: 0.9,
    9: 2.3,
    10: 1.9,
    11: 4.5,
}

# Paper P-P preprocessing time (seconds) — TWOSBY
PAPER_PP_TIME_TWOSBY = {
    0: 1.1,
    1: 3.1,
    2: 5.2,
    3: 8.5,
    4: 2.6,
    5: 5.0,
    6: 8.5,
    7: 14.1,
    8: 4.2,
    9: 7.5,
    10: 13.6,
    11: 23.7,
}

# ILP-SPACES optimal costs — benedikt2020a_prelim (NOSBY, n=30/60)
PAPER_PRELIM_NOSBY = {
    0: 1_453,
    1: 3_980,
    2: 2_141,
    3: 3_614,
    4: 2_236,
    5: 3_869,
    6: 1_807,
    7: 3_123,
    8: 1_960,
    9: 3_764,
    10: 2_363,
    11: 4_035,
}

# ILP-SPACES optimal costs — benedikt2020a_medium_nosby
PAPER_MEDIUM_NOSBY = {
    0: 1_426,
    1: 1_394,
    2: 1_394,
    3: 1_394,
    4: 4_290,
    5: 3_994,
    6: 3_836,
    7: 3_833,
    8: 5_920,
    9: 5_686,
    10: 5_431,
    11: 5_373,
}

# ILP-SPACES optimal costs — benedikt2020a_medium_twosby
PAPER_MEDIUM_TWOSBY = {
    0: 3_815,
    1: 3_804,
    2: 3_804,
    3: 3_804,
    4: 10_863,
    5: 10_248,
    6: 9_917,
    7: 9_874,
    8: 15_379,
    9: 14_923,
    10: 14_548,
    11: 14_392,
}

# Instance metadata: (n, lambda) per index
TABLE1_META = {
    0: (150, 1.3),
    1: (150, 1.6),
    2: (150, 1.9),
    3: (150, 2.2),
    4: (170, 1.3),
    5: (170, 1.6),
    6: (170, 1.9),
    7: (170, 2.2),
    8: (190, 1.3),
    9: (190, 1.6),
    10: (190, 1.9),
    11: (190, 2.2),
}


def load_instance(json_path: Path) -> dict:
    """Load a paper instance JSON file and extract jobs + prices."""
    with open(json_path) as f:
        d = json.load(f)

    jobs = [j["ProcessingTime"] for j in d["Jobs"]]
    prices = d["EnergyCosts"]
    meta = d.get("Metadata", {})

    return {
        "jobs": jobs,
        "prices": prices,
        "n_jobs": len(jobs),
        "horizon": len(prices),
        "metadata": meta,
        "machine": {
            "OffOnTime": d.get("OffOnTime"),
            "OnOffTime": d.get("OnOffTime"),
            "OffOnPower": d.get("OffOnPowerConsumption"),
            "OnOffPower": d.get("OnOffPowerConsumption"),
            "OnPower": d.get("OnPowerConsumption"),
            "IdlePower": d.get("IdlePowerConsumption"),
            "OffPower": d.get("OffPowerConsumption"),
        },
    }


def is_nosby(inst: dict) -> bool:
    """Check if instance uses NOSBY (3-state) machine config."""
    m = inst["machine"]
    # NOSBY: OffOnTime is a single-element array [2]
    return (
        isinstance(m["OffOnTime"], list)
        and len(m["OffOnTime"]) == 1
        and m["OffOnTime"][0] == 2
        and m["OnOffTime"][0] == 1
    )


def is_twosby(inst: dict) -> bool:
    """Check if instance uses TWOSBY (5-state) machine config."""
    m = inst["machine"]
    return isinstance(m["OffOnTime"], list) and len(m["OffOnTime"]) == 3


def get_machine_type(inst: dict) -> str:
    """Return machine type string for the solver JSONL protocol."""
    if is_nosby(inst):
        return "nosby"
    if is_twosby(inst):
        return "twosby"
    return "unknown"


def run_solver_batch(
    instances: list[tuple[str, dict]], solver_path: Path, time_limit: float
) -> list[dict]:
    """Run the solver on a batch of instances via solve-stdin."""
    # Build JSONL input
    jsonl_lines = []
    for inst_id, inst in instances:
        payload = {
            "instance_id": inst_id,
            "prices": inst["prices"],
            "jobs": inst["jobs"],
            "machine": get_machine_type(inst),
        }
        line = json.dumps(payload)
        jsonl_lines.append(line)

    input_data = "\n".join(jsonl_lines) + "\n"

    args = [str(solver_path), "solve-stdin"]
    if time_limit > 0:
        args.append(str(time_limit))

    proc = subprocess.run(
        args,
        input=input_data,
        capture_output=True,
        text=True,
        timeout=int(time_limit * len(instances) + 60),
    )

    if proc.returncode != 0:
        print(f"Solver stderr: {proc.stderr}", file=sys.stderr)

    results = []
    lines = proc.stdout.strip().split("\n")
    header = lines[0] if lines else ""
    for line in lines[1:]:
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
    solver_path: Path,
    time_limit: float,
    paper_costs: dict | None = None,
    paper_bb_time: dict | None = None,
    paper_pp_time: dict | None = None,
) -> list[dict]:
    """Run solver on all instances in a dataset directory."""
    ds_dir = DATA_DIR / dataset_name
    if not ds_dir.exists():
        print(f"Dataset not found: {ds_dir}", file=sys.stderr)
        return []

    # Load instances in order
    json_files = sorted(ds_dir.glob("*.json"), key=lambda p: int(p.stem))
    instances = []
    for jf in json_files:
        idx = int(jf.stem)
        inst = load_instance(jf)
        inst_id = f"{dataset_name}_{idx}"
        instances.append((inst_id, inst, idx))

    print(f"\n{'=' * 100}")
    print(f"Dataset: {dataset_name} ({len(instances)} instances)")
    print(f"{'=' * 100}")

    # Check machine types
    supported = [
        (iid, inst, idx)
        for iid, inst, idx in instances
        if get_machine_type(inst) in ("nosby", "twosby")
    ]
    unsupported = len(instances) - len(supported)
    if unsupported > 0:
        print(
            f"  WARNING: {unsupported} instances use unsupported machine config, skipping."
        )
    if not supported:
        print("  No supported instances found. Skipping entire dataset.")
        return []
    instances = supported

    # Run solver
    solver_input = [(iid, inst) for iid, inst, idx in instances]
    results = run_solver_batch(solver_input, solver_path, time_limit)

    # Print results table
    print(
        f"\n{'Instance':>40} {'n':>4} {'h':>5} {'UB':>12} {'LB':>12} {'gap%':>8} {'opt':>4} {'t[s]':>8}",
        end="",
    )
    if paper_costs:
        print(f" {'paper_opt':>12} {'match':>6} {'BB+PP[s]':>9} {'speedup':>8}", end="")
    print()
    print("-" * 130)

    total_time = 0
    total_match = 0
    total_optimal = 0
    for i, r in enumerate(results):
        idx = instances[i][2]
        n, lam = TABLE1_META.get(idx, (r["n_jobs"], "-"))
        print(
            f"{r['instance_id']:>40} {r['n_jobs']:>4} {r['horizon']:>5} "
            f"{r['ub']:>12.0f} {r['lb']:>12.0f} {r['gap_pct']:>8.4f} "
            f"{r['is_optimal']:>4} {r['runtime_sec']:>8.4f}",
            end="",
        )

        if paper_costs and idx in paper_costs:
            p_opt = paper_costs[idx]
            match = "✓" if abs(r["ub"] - p_opt) < 0.5 else "✗"
            if abs(r["ub"] - p_opt) < 0.5:
                total_match += 1
            bb_pp = (paper_bb_time or {}).get(idx, 0) + (paper_pp_time or {}).get(
                idx, 0
            )
            speedup = bb_pp / r["runtime_sec"] if r["runtime_sec"] > 0 else float("inf")
            print(f" {p_opt:>12} {match:>6} {bb_pp:>9.1f} {speedup:>7.1f}x", end="")

        total_time += r["runtime_sec"]
        total_optimal += r["is_optimal"]
        print()

    print("-" * 130)
    n_inst = len(results)
    print(
        f"{'TOTALS':>40} {'':>4} {'':>5} {'':>12} {'':>12} {'':>8} "
        f"{total_optimal:>4} {total_time:>8.4f}",
        end="",
    )
    if paper_costs:
        avg_bb_pp = sum(
            (paper_bb_time or {}).get(instances[i][2], 0)
            + (paper_pp_time or {}).get(instances[i][2], 0)
            for i in range(n_inst)
        ) / max(n_inst, 1)
        avg_our = total_time / max(n_inst, 1)
        overall_speedup = avg_bb_pp / avg_our if avg_our > 0 else 0
        print(
            f" {total_match:>12} {'':>6} {avg_bb_pp:>9.1f} {overall_speedup:>7.1f}x",
            end="",
        )
    print()

    print(
        f"\nSummary: {total_optimal}/{n_inst} proven optimal, avg {total_time/max(n_inst,1):.4f}s"
    )
    if paper_costs:
        print(f"Cost match vs paper: {total_match}/{n_inst}")

    return results


def main():
    ap = argparse.ArgumentParser(
        description="Run solver on paper's actual benchmark instances"
    )
    ap.add_argument(
        "--dataset", default="all", help="Dataset name or 'all' (default: all)"
    )
    ap.add_argument(
        "--solver", default=str(SOLVER), help="Path to stateful_compare binary"
    )
    ap.add_argument(
        "--time-limit",
        type=float,
        default=600,
        help="Time limit per instance in seconds (default: 600)",
    )
    ap.add_argument("--output", default=None, help="Output CSV file path")
    args = ap.parse_args()

    solver_path = Path(args.solver)
    if not solver_path.exists():
        print(f"Solver not found: {solver_path}", file=sys.stderr)
        sys.exit(1)

    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}", file=sys.stderr)
        print(
            "Run: git clone https://github.com/CTU-IIG/EnergyStatesAndCostsSchedulingData.git data/paper_instances"
        )
        sys.exit(1)

    all_results = []

    if args.dataset in ("all", "benedikt2020a_large_nosby"):
        all_results.extend(
            run_dataset(
                "benedikt2020a_large_nosby",
                solver_path,
                args.time_limit,
                paper_costs=PAPER_TABLE1_NOSBY,
                paper_bb_time=PAPER_BB_TIME_NOSBY,
                paper_pp_time=PAPER_PP_TIME_NOSBY,
            )
        )

    if args.dataset in ("all", "benedikt2020a_large_twosby"):
        all_results.extend(
            run_dataset(
                "benedikt2020a_large_twosby",
                solver_path,
                args.time_limit,
                paper_costs=PAPER_TABLE1_TWOSBY,
                paper_bb_time=PAPER_BB_TIME_TWOSBY,
                paper_pp_time=PAPER_PP_TIME_TWOSBY,
            )
        )

    if args.dataset in ("all", "benedikt2020a_medium_nosby"):
        all_results.extend(
            run_dataset(
                "benedikt2020a_medium_nosby",
                solver_path,
                args.time_limit,
                paper_costs=PAPER_MEDIUM_NOSBY,
            )
        )

    if args.dataset in ("all", "benedikt2020a_medium_twosby"):
        all_results.extend(
            run_dataset(
                "benedikt2020a_medium_twosby",
                solver_path,
                args.time_limit,
                paper_costs=PAPER_MEDIUM_TWOSBY,
            )
        )

    if args.dataset in ("all", "benedikt2020a_prelim"):
        all_results.extend(
            run_dataset(
                "benedikt2020a_prelim",
                solver_path,
                args.time_limit,
                paper_costs=PAPER_PRELIM_NOSBY,
            )
        )

    if args.dataset in ("all", "aghelinejad2017a_1"):
        all_results.extend(
            run_dataset(
                "aghelinejad2017a_1",
                solver_path,
                args.time_limit,
            )
        )

    # Write CSV if requested
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

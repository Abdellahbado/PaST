#!/usr/bin/env python3
"""
Generate hard processing-time instances for stress-testing the solver.

The paper uses p_j ~ U[1,5] (5 distinct job lengths). Our solver's relaxed DP
is polynomial in the number of DISTINCT lengths, so harder instances have:
  - Wider ranges: U[1,10] gives 10 types instead of 5
  - Two close values: {8,10} or {9,10} where bin-packing is hard
  - Mixed hard: {3,7} where gcd=1 and packing is combinatorial

Each instance is a JSONL line for solve-stdin: {instance_id, prices, jobs, machine}.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOLVER = ROOT / "solvers" / "cpp" / "build" / "stateful_compare"


def gen_instance(
    n: int,
    lam: float,
    proc_dist: str,  # "U[a,b]" or "{a,b}" or "{a,b,c}"
    seed: int,
    machine: str = "nosby",
) -> dict:
    rng = random.Random(seed)

    # Parse proc_dist
    if proc_dist.startswith("U["):
        lo, hi = map(int, proc_dist[2:-1].split(","))
        jobs = [rng.randint(lo, hi) for _ in range(n)]
    elif proc_dist.startswith("{"):
        vals = list(map(int, proc_dist[1:-1].split(",")))
        jobs = [rng.choice(vals) for _ in range(n)]
    else:
        raise ValueError(f"Unknown dist: {proc_dist}")

    # Compute horizon
    if machine == "nosby":
        startup, shutdown = 2, 1
    elif machine == "twosby":
        startup, shutdown = 4, 1
    else:
        raise ValueError(f"Unknown machine: {machine}")

    lower = startup + sum(jobs) + shutdown
    horizon = int(math.ceil(lam * lower))

    # Generate prices U[1,10]
    prices = [float(rng.randint(1, 10)) for _ in range(horizon)]

    dist_tag = (
        proc_dist.replace("[", "")
        .replace("]", "")
        .replace("{", "")
        .replace("}", "")
        .replace(",", "-")
    )
    return {
        "instance_id": f"hard_{dist_tag}_n{n}_lam{lam:.1f}_s{seed}_{machine}",
        "prices": prices,
        "jobs": jobs,
        "machine": machine,
        "n": n,
        "horizon": horizon,
    }


def main():
    ap = argparse.ArgumentParser(description="Generate and solve hard instances")
    ap.add_argument("--solver", default=str(SOLVER))
    ap.add_argument("--time-limit", type=float, default=60)
    ap.add_argument("--seeds", type=str, default="42,123,456")
    ap.add_argument("--machine", default="nosby", choices=["nosby", "twosby", "both"])
    args = ap.parse_args()

    solver_path = Path(args.solver)
    if not solver_path.exists():
        print(f"Solver not found: {solver_path}", file=sys.stderr)
        sys.exit(1)

    seeds = [int(s) for s in args.seeds.split(",")]
    machines = ["nosby", "twosby"] if args.machine == "both" else [args.machine]

    # Instance configs: (n, lambda, proc_dist)
    configs = [
        # Standard paper range but larger
        (150, 1.3, "U[1,5]"),
        (190, 2.2, "U[1,5]"),
        # Wider proc ranges (more distinct lengths)
        (150, 1.3, "U[1,10]"),
        (190, 1.3, "U[1,10]"),
        (190, 2.2, "U[1,10]"),
        # Two close values (hard bin-packing)
        (100, 1.3, "{8,10}"),
        (150, 1.3, "{8,10}"),
        (100, 1.3, "{9,10}"),
        (150, 1.3, "{9,10}"),
        # Mixed hard
        (100, 1.3, "{3,7}"),
        (150, 1.3, "{3,7}"),
        (150, 1.6, "{3,7}"),
        # Very wide range
        (100, 1.3, "U[1,20]"),
        (150, 1.3, "U[1,20]"),
    ]

    all_instances = []
    for machine in machines:
        for seed in seeds:
            for n, lam, dist in configs:
                inst = gen_instance(n, lam, dist, seed, machine)
                all_instances.append(inst)

    print(f"Generated {len(all_instances)} instances")
    print(
        f"{'Instance':>55} {'n':>4} {'h':>5} {'K':>3} {'UB':>12} {'LB':>12} {'gap%':>8} {'opt':>4} {'t[s]':>8}"
    )
    print("-" * 120)

    # Solve in batches via solve-stdin
    jsonl_lines = [json.dumps(inst) for inst in all_instances]
    input_data = "\n".join(jsonl_lines) + "\n"

    proc = subprocess.run(
        [str(solver_path), "solve-stdin", str(args.time_limit)],
        input=input_data,
        capture_output=True,
        text=True,
        timeout=int(args.time_limit * len(all_instances) + 120),
    )

    if proc.returncode != 0:
        print(f"Solver stderr: {proc.stderr[:500]}", file=sys.stderr)

    lines = proc.stdout.strip().split("\n")
    total_opt = 0
    total_time = 0.0
    max_gap = 0.0
    max_time = 0.0
    problematic = []

    for i, line in enumerate(lines[1:]):  # skip header
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 10:
            continue
        iid = parts[0]
        n_j = int(parts[1])
        horizon = int(parts[2])
        ub = float(parts[3])
        lb = float(parts[4])
        gap = float(parts[5])
        opt = int(parts[7])
        t = float(parts[9])

        # Count distinct job lengths for this instance
        jobs = all_instances[i]["jobs"]
        K = len(set(jobs))

        total_opt += opt
        total_time += t
        max_gap = max(max_gap, gap)
        max_time = max(max_time, t)

        marker = "  " if opt else " !"
        print(
            f"{iid:>55} {n_j:>4} {horizon:>5} {K:>3} {ub:>12.0f} {lb:>12.0f} {gap:>8.4f} {opt:>4} {t:>8.4f}{marker}"
        )

        if not opt:
            problematic.append((iid, gap, t))

    n_total = len(lines) - 1
    print("-" * 120)
    print(
        f"Total: {total_opt}/{n_total} optimal, avg {total_time/max(n_total,1):.4f}s, max {max_time:.4f}s, max_gap {max_gap:.4f}%"
    )

    if problematic:
        print(f"\n{len(problematic)} instances NOT proven optimal:")
        for iid, gap, t in problematic:
            print(f"  {iid}: gap={gap:.4f}%, time={t:.4f}s")


if __name__ == "__main__":
    main()

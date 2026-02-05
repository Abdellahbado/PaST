"""CLI: generate a PaST benchmark instance and solve one single-machine subset with the optimal multiset DP.

This script matches your simplified constraints:
- No per-job releases/deadlines.
- Only processing time matters; jobs with same p are identical.
- Horizon is the benchmark horizon T_max.

We generate a raw benchmark instance using PaST's generator (paper_grid_90),
simulate the meta-heuristic split across machines, select one machine's assigned
jobs, then solve that single-machine subproblem optimally.

Example runs (short):
  conda run -n new-ml-env python PaST/cli/run_optimal_benchmark_dp.py --scale mls --seed 0
  conda run -n new-ml-env python PaST/cli/run_optimal_benchmark_dp.py --scale vls --seed 1

Loop over seeds:
  for s in 0 1 2 3 4; do conda run -n new-ml-env python PaST/cli/run_optimal_benchmark_dp.py --scale mls --seed $s; done
"""

from __future__ import annotations

import argparse
import random
from typing import List, Tuple

import numpy as np

from PaST.config import DataConfig
from PaST.data.sm_benchmark_data import (
    generate_raw_instance,
    simulate_metaheuristic_assignment,
)
from PaST.solvers.optimal_benchmark_dp import solve_optimal_benchmark_dp


def _pick_machine(
    assignments: List[List[int]], policy: str, rng: random.Random, index: int
) -> int:
    if not assignments:
        return 0

    if policy == "max_jobs":
        sizes = [len(a) for a in assignments]
        return int(max(range(len(sizes)), key=lambda i: sizes[i]))

    if policy == "random":
        non_empty = [i for i, a in enumerate(assignments) if len(a) > 0]
        if non_empty:
            return int(rng.choice(non_empty))
        return int(rng.randrange(len(assignments)))

    if policy == "index":
        if index < 0 or index >= len(assignments):
            raise ValueError(
                f"machine-index out of range: {index} (m={len(assignments)})"
            )
        return int(index)

    raise ValueError(f"Unknown machine policy: {policy}")


def _default_T_max(scale: str) -> int:
    if scale == "small":
        return 80
    if scale == "mls":
        return 300
    if scale == "vls":
        return 500
    raise ValueError(f"Unknown scale: {scale}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", choices=["small", "mls", "vls"], default="mls")
    ap.add_argument("--T-max", type=int, default=None, dest="T_max")
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument(
        "--machine-policy",
        choices=["max_jobs", "random", "index"],
        default="max_jobs",
        help="Which machine's job subset to solve",
    )
    ap.add_argument("--machine-index", type=int, default=0)
    ap.add_argument(
        "--assignment-concentration",
        type=float,
        default=1.0,
        help="Dirichlet concentration for the simulated split (lower = more unbalanced)",
    )
    ap.add_argument(
        "--tie-break",
        choices=["cost", "early"],
        default="early",
        help="Tie-break among equal-cost optima (early = prefer earlier schedule)",
    )

    args = ap.parse_args()

    rng = random.Random(int(args.seed))

    config = DataConfig()
    config.sampling_mode = "paper_grid_90"

    T_max = int(args.T_max) if args.T_max is not None else _default_T_max(args.scale)

    raw = generate_raw_instance(
        config=config,
        rng=rng,
        instance_id=int(args.seed),
        T_max=T_max,
        m=args.m,
        n=args.n,
    )

    assignments = simulate_metaheuristic_assignment(
        n=raw.n,
        m=raw.m,
        rng=rng,
        concentration=float(args.assignment_concentration),
    )

    machine_idx = _pick_machine(
        assignments,
        policy=str(args.machine_policy),
        rng=rng,
        index=int(args.machine_index),
    )

    job_indices = assignments[machine_idx]
    p_subset = np.array([raw.p[j] for j in job_indices], dtype=np.int32)

    prices = np.array(raw.ct, dtype=np.float64)

    res = solve_optimal_benchmark_dp(
        p_subset.tolist(), prices, tie_break=str(args.tie_break)
    )

    sum_p = int(np.sum(p_subset)) if len(p_subset) else 0

    # Compact one-line output for easy looping / parsing
    print(
        " ".join(
            [
                f"seed={args.seed}",
                f"scale={raw.scale}",
                f"T={raw.T_max}",
                f"m={raw.m}",
                f"n={raw.n}",
                f"machine={machine_idx}",
                f"n_jobs_sm={len(p_subset)}",
                f"sum_p_sm={sum_p}",
                f"feasible={res.feasible}",
                f"cost={res.cost:.6f}",
                f"finish={res.finish_time}",
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

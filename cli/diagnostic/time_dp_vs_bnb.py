"""Timing comparison: exact DP (correct_dp) vs custom Branch-and-Bound.

Runs on the *unconstrained* single-machine setting supported by bnb_solver_custom:
- all jobs available at t=0
- common deadline horizon T (d=T)
- objective = sum of TOU prices over running slots

Run:
  conda run -n new-ml-env python /Users/mac/Documents/Study/PFE/PaST/cli/diagnostic/time_dp_vs_bnb.py
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import statistics
import time
from pathlib import Path
from typing import List, Tuple

import sys

import numpy as np


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _p90(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    xs_sorted = sorted(xs)
    idx = max(0, min(len(xs_sorted) - 1, int(0.9 * (len(xs_sorted) - 1))))
    return xs_sorted[idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--kind", choices=["valleys", "benchmark", "mixed"], default="valleys"
    )
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--Tmin", type=int, default=50)
    ap.add_argument("--Tmax", type=int, default=80)
    ap.add_argument("--pmax", type=int, default=4)
    ap.add_argument("--bnb-limit", type=float, default=2.0)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[3]
    correct_dp = _load_module("correct_dp", repo / "PaST" / "correct_dp.py")
    bnb_mod = _load_module(
        "bnb_solver_custom", repo / "PaST" / "solvers" / "bnb_solver_custom.py"
    )

    dp_times: List[float] = []
    bnb_times: List[float] = []
    timeouts = 0
    mismatches = 0

    for seed in range(args.seeds):
        rng = random.Random(seed)
        T = rng.randint(args.Tmin, args.Tmax)

        if args.kind == "valleys":
            prices = correct_dp.generate_price_curve_valleys(T, rng)
        elif args.kind == "benchmark":
            prices = correct_dp.generate_benchmark_prices(T, rng)
        else:
            prices = correct_dp.generate_mixed_prices(T, rng)

        # Generate processing times; ensure feasibility (sum(p) <= T).
        for _ in range(1000):
            pts = [rng.randint(1, args.pmax) for _ in range(args.n)]
            if sum(pts) <= T:
                break
        else:
            # Give up; skip seed
            continue

        # ---- Exact DP ----
        correct_dp.set_custom_prices(prices.tolist())
        jobs = [correct_dp.Job(id=i, p=int(pts[i]), r=0, d=T) for i in range(args.n)]
        t0 = time.perf_counter()
        dp_cost, _ = correct_dp.solve_dp_exact(jobs)
        dp_times.append(time.perf_counter() - t0)
        correct_dp.clear_custom_prices()

        # ---- Branch-and-Bound ----
        inst = bnb_mod.Instance(
            n_jobs=args.n,
            processing_times=np.asarray(pts, dtype=np.int32),
            T=T,
            energy_costs=np.asarray(prices, dtype=np.float64),
        )
        solver = bnb_mod.BranchAndBoundSolver(
            inst, time_limit=float(args.bnb_limit), verbose=False
        )
        t1 = time.perf_counter()
        _, bnb_cost = solver.solve()
        bnb_times.append(time.perf_counter() - t1)

        if solver.timed_out:
            timeouts += 1
        else:
            if not (
                (dp_cost == float("inf") and bnb_cost == float("inf"))
                or abs(float(dp_cost) - float(bnb_cost)) < 1e-6
            ):
                mismatches += 1

    def fmt(label: str, xs: List[float]) -> str:
        if not xs:
            return f"{label}: (none)"
        return (
            f"{label}: mean={statistics.mean(xs)*1000:.2f}ms "
            f"median={statistics.median(xs)*1000:.2f}ms "
            f"p90={_p90(xs)*1000:.2f}ms (n={len(xs)})"
        )

    print(
        f"kind={args.kind} N={args.n} seeds={args.seeds} "
        f"T∈[{args.Tmin},{args.Tmax}] bnb_limit={args.bnb_limit}s"
    )
    print(fmt("DP ", dp_times))
    print(fmt("BnB", bnb_times))
    print(f"BnB timeouts: {timeouts} / {len(bnb_times)}")
    if timeouts == 0:
        print(f"Cost mismatches (DP vs BnB): {mismatches}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

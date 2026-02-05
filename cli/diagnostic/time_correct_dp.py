"""Timing harness for PaST.correct_dp (DP vs brute force on small instances).

Run (from repo root):
  conda run -n new-ml-env python PaST/cli/diagnostic/time_correct_dp.py
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from pathlib import Path
from typing import List

import sys


# Ensure repo root is on sys.path even when launched from elsewhere.
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import PaST.correct_dp as cd  # noqa: E402


def _time_call(fn, *args, repeats: int = 1):
    times: List[float] = []
    last = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = fn(*args)
        times.append(time.perf_counter() - t0)
    return times, last


def _gen_prices(kind: str, T: int, rng: random.Random):
    if kind == "valleys":
        return cd.generate_price_curve_valleys(T, rng)
    if kind == "benchmark":
        return cd.generate_benchmark_prices(T, rng)
    if kind == "mixed":
        return cd.generate_mixed_prices(T, rng)
    raise ValueError(f"Unknown kind: {kind}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--kind", choices=["valleys", "benchmark", "mixed"], default="valleys"
    )
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--Tmin", type=int, default=50)
    ap.add_argument("--Tmax", type=int, default=70)
    ap.add_argument("--max-window", type=int, default=15)
    ap.add_argument(
        "--bf",
        action="store_true",
        help="Also time brute force (only feasible for small instances).",
    )
    args = ap.parse_args()

    dp_times: List[float] = []
    bf_times: List[float] = []
    dp_costs: List[float] = []
    bf_costs: List[float] = []

    for seed in range(args.seeds):
        rng = random.Random(seed)
        T = rng.randint(args.Tmin, args.Tmax)
        prices = _gen_prices(args.kind, T, rng)
        cd.set_custom_prices(prices.tolist())
        jobs = cd.generate_random_instance(args.n, T, rng, max_window=args.max_window)

        t0 = time.perf_counter()
        dp_cost, _ = cd.solve_dp_exact(jobs)
        dp_times.append(time.perf_counter() - t0)
        dp_costs.append(dp_cost)

        if args.bf:
            t1 = time.perf_counter()
            bf_cost, _ = cd.solve_brute_force(jobs)
            bf_times.append(time.perf_counter() - t1)
            bf_costs.append(bf_cost)

        cd.clear_custom_prices()

    def summarize(label: str, xs: List[float]):
        if not xs:
            return f"{label}: (none)"
        xs_sorted = sorted(xs)
        p90_idx = max(0, min(len(xs_sorted) - 1, int(0.9 * (len(xs_sorted) - 1))))
        p90 = xs_sorted[p90_idx]
        return (
            f"{label}: mean={statistics.mean(xs)*1000:.2f}ms "
            f"median={statistics.median(xs)*1000:.2f}ms "
            f"p90={p90*1000:.2f}ms "
            f"(n={len(xs)})"
        )

    print(
        f"kind={args.kind} N={args.n} T∈[{args.Tmin},{args.Tmax}] max_window={args.max_window}"
    )
    print(summarize("DP", dp_times))
    if args.bf:
        print(summarize("BF", bf_times))
        # quick correctness spot-check
        mismatches = sum(
            1
            for a, b in zip(dp_costs, bf_costs)
            if not ((a == float("inf") and b == float("inf")) or abs(a - b) < 1e-6)
        )
        print(f"DP vs BF mismatches: {mismatches}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run medium-size instances: Exact DP vs custom BnB, plot + save schedules.

This uses the *unconstrained* single-machine setting supported by bnb_solver_custom:
- all jobs available at t=0
- common deadline horizon T (d=T)
- objective = sum of TOU prices over running slots (idle allowed)

Outputs one PNG per instance to PaST/analysis_out/medium_dp_bnb_viz.

Run (short):
  conda run -n new-ml-env python PaST/cli/diagnostic/run_medium_dp_bnb_viz.py

Tuning examples:
  conda run -n new-ml-env python PaST/cli/diagnostic/run_medium_dp_bnb_viz.py --kind valleys --n 14 --seeds 3 --bnb-limit 20
  conda run -n new-ml-env python PaST/cli/diagnostic/run_medium_dp_bnb_viz.py --kind benchmark --n 16 --T 120 --seeds 2 --bnb-limit 30
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_feasible_pts(rng: random.Random, n: int, pmax: int, T: int) -> List[int]:
    for _ in range(5000):
        pts = [rng.randint(1, pmax) for _ in range(n)]
        if sum(pts) <= T:
            return pts
    # deterministic fallback: all ones
    pts = [1] * n
    if sum(pts) > T:
        raise ValueError("Infeasible: sum(p) > T")
    return pts


def _schedule_cost_from_prices(
    prices: np.ndarray, bars: List[Tuple[int, int, int]]
) -> float:
    # bars: (job_id, start, end)
    if len(prices) == 0:
        return 0.0
    prefix = np.zeros(len(prices) + 1, dtype=np.float64)
    prefix[1:] = np.cumsum(prices.astype(np.float64, copy=False))
    total = 0.0
    for _, s, e in bars:
        s = max(0, int(s))
        e = min(len(prices), int(e))
        if e > s:
            total += float(prefix[e] - prefix[s])
    return float(total)


def _plot_instance(
    out_png: Path,
    prices: np.ndarray,
    dp_bars: List[Tuple[int, int, int]],
    bnb_bars: List[Tuple[int, int, int]],
    title: str,
    subtitle: str,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = len(prices)

    fig_h = 6.0
    fig, (ax_price, ax_sched) = plt.subplots(
        2,
        1,
        figsize=(14, fig_h),
        sharex=True,
        gridspec_kw={"height_ratios": [1.1, 2.2]},
    )

    # Price curve
    x = np.arange(T, dtype=np.int32)
    ax_price.plot(x, prices, linewidth=1.5, color="#1f77b4")
    ax_price.set_ylabel("Price")
    ax_price.grid(alpha=0.25, linestyle="--")
    ax_price.set_title(title + "\n" + subtitle, fontsize=11, fontweight="bold")

    # Schedules (two lanes)
    lanes = [("DP", dp_bars, "#2ca02c"), ("BnB", bnb_bars, "#ff7f0e")]
    ax_sched.set_yticks([0, 1])
    ax_sched.set_yticklabels([f"{name}" for name, _, _ in lanes])

    cmap = plt.cm.Set3

    for lane_y, (name, bars, base_color) in enumerate(lanes):
        for job_id, s, e in bars:
            dur = max(0, int(e) - int(s))
            if dur <= 0:
                continue
            color = cmap((job_id % 12) / 12.0)
            ax_sched.barh(
                lane_y,
                dur,
                left=int(s),
                height=0.55,
                color=color,
                edgecolor="black",
                linewidth=0.9,
            )
            ax_sched.text(
                int(s) + dur / 2,
                lane_y,
                f"J{int(job_id)}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

    ax_sched.set_xlim(0, T)
    ax_sched.set_xlabel("Time")
    ax_sched.grid(axis="x", alpha=0.25, linestyle="--")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--kind", choices=["valleys", "benchmark", "mixed"], default="valleys"
    )
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument(
        "--n",
        type=int,
        default=14,
        help="Number of jobs. For valleys, keep this modest (e.g., 12-14) to avoid BnB blowups.",
    )
    ap.add_argument("--T", type=int, default=120)
    ap.add_argument("--pmax", type=int, default=4)
    ap.add_argument("--bnb-limit", type=float, default=20.0)
    ap.add_argument(
        "--out-dir",
        type=str,
        default="PaST/analysis_out/medium_dp_bnb_viz",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[3]
    correct_dp = _load_module("correct_dp", repo / "PaST" / "correct_dp.py")
    bnb_mod = _load_module(
        "bnb_solver_custom", repo / "PaST" / "solvers" / "bnb_solver_custom.py"
    )

    out_dir = (
        (repo / args.out_dir).resolve()
        if not Path(args.out_dir).is_absolute()
        else Path(args.out_dir)
    )

    for seed in range(args.seeds):
        rng = random.Random(seed)
        T = int(args.T)

        if args.kind == "valleys":
            prices = correct_dp.generate_price_curve_valleys(T, rng)
        elif args.kind == "benchmark":
            prices = correct_dp.generate_benchmark_prices(T, rng).astype(np.float64)
        else:
            prices = correct_dp.generate_mixed_prices(T, rng)

        pts = _ensure_feasible_pts(rng, int(args.n), int(args.pmax), T)

        # ---- Exact DP ----
        correct_dp.set_custom_prices(prices.tolist())
        jobs = [
            correct_dp.Job(id=i, p=int(pts[i]), r=0, d=T) for i in range(int(args.n))
        ]
        t0 = time.perf_counter()
        dp_cost, dp_hist = correct_dp.solve_dp_exact(jobs)
        dp_sec = time.perf_counter() - t0
        correct_dp.clear_custom_prices()
        dp_bars = [(int(j), int(s), int(e)) for (j, s, e) in dp_hist]

        # ---- BnB ----
        inst = bnb_mod.Instance(
            n_jobs=int(args.n),
            processing_times=np.asarray(pts, dtype=np.int32),
            T=T,
            energy_costs=np.asarray(prices, dtype=np.float64),
        )
        solver = bnb_mod.BranchAndBoundSolver(
            inst, time_limit=float(args.bnb_limit), verbose=False
        )
        t1 = time.perf_counter()
        bnb_seq, bnb_cost = solver.solve()
        bnb_sec = time.perf_counter() - t1

        # Convert BnB sequence -> schedule bars using its internal DP for that fixed order.
        # (This matches how the solver evaluates sequences.)
        bnb_pts_in_order = inst.processing_times[
            np.asarray(bnb_seq, dtype=np.int32)
        ].tolist()
        cost_check, starts = solver._dp_evaluate_with_schedule(bnb_pts_in_order)
        bnb_bars = []
        for idx_in_seq, job_id in enumerate(bnb_seq):
            s = int(starts[idx_in_seq])
            e = int(s + int(inst.processing_times[int(job_id)]))
            bnb_bars.append((int(job_id), s, e))

        # Costs computed from bars (sanity)
        dp_cost2 = _schedule_cost_from_prices(
            np.asarray(prices, dtype=np.float64), dp_bars
        )
        bnb_cost2 = _schedule_cost_from_prices(
            np.asarray(prices, dtype=np.float64), bnb_bars
        )

        # Print schedules (compact)
        print("=" * 70)
        print(f"seed={seed} kind={args.kind} N={args.n} T={T} sum(p)={sum(pts)}")
        print(
            f"DP : cost={float(dp_cost):.4f} ({dp_cost2:.4f} bars) time={dp_sec*1000:.2f}ms"
        )
        print(
            f"BnB: cost={float(bnb_cost):.4f} ({bnb_cost2:.4f} bars) time={bnb_sec*1000:.2f}ms timed_out={bool(solver.timed_out)} nodes={int(solver.nodes_explored)}"
        )
        print(f"DP order : {[j for (j,_,_) in dp_bars]}")
        print(f"BnB order: {[j for (j,_,_) in bnb_bars]}")
        print(f"DP bars  : {dp_bars}")
        print(f"BnB bars : {bnb_bars}")

        out_png = out_dir / f"medium_{args.kind}_seed{seed}_N{args.n}_T{T}.png"
        title = f"Medium instance | {args.kind} | N={args.n} T={T} seed={seed}"
        subtitle = (
            f"DP: {dp_sec*1000:.1f}ms cost={float(dp_cost):.3f} | "
            f"BnB: {bnb_sec*1000:.1f}ms cost={float(bnb_cost):.3f} "
            f"(timeout={bool(solver.timed_out)})"
        )
        _plot_instance(
            out_png,
            np.asarray(prices, dtype=np.float64),
            dp_bars,
            bnb_bars,
            title,
            subtitle,
        )
        print(f"Wrote: {out_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

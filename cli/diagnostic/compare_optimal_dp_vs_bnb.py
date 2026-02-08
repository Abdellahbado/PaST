"""Compare optimal benchmark DP vs Branch-and-Bound on generated instances.

What it does
------------
For each seed:
1) Generate a raw benchmark instance via PaST.data.sm_benchmark_data.generate_raw_instance
2) Simulate an assignment of jobs to machines
3) Select ONE machine's subset (same as the environment setup)
4) Solve that single-machine subset using:
   - Optimal multiset DP (PaST.solvers.optimal_benchmark_dp)
   - Branch-and-Bound (PaST.solvers.bnb_solver_custom)
5) Save:
   - a PNG plot (price curve + both schedules)
   - a JSON result file (costs, runtimes, feasibility, etc.)

Parallelism
-----------
Parallelizes across seeds using multiprocessing.

Run as module from repo root so `PaST` imports work:
  conda run -n new-ml-env python -m PaST.cli.diagnostic.compare_optimal_dp_vs_bnb --scale mls --seeds 0:9

Notes
-----
- This compares on a *single-machine subset* (one machine) per seed.
- BnB can be very slow on large subsets; set --bnb-time-limit 0 for "no limit"
  (interpreted as a very large number), but be aware it may take a long time.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np

from PaST.config import DataConfig
from PaST.data.sm_benchmark_data import (
    generate_raw_instance,
    simulate_metaheuristic_assignment,
)
from PaST.solvers.bnb_solver_custom import BranchAndBoundSolver, Instance
from PaST.solvers.optimal_benchmark_dp import solve_optimal_benchmark_dp


def _default_T_max(scale: str) -> int:
    if scale == "small":
        return 80
    if scale == "mls":
        return 300
    if scale == "vls":
        return 500
    raise ValueError(f"Unknown scale: {scale}")


def _parse_seeds(spec: str) -> List[int]:
    """Parse seeds spec.

    Accepts:
    - "0:9" meaning 0..9 inclusive
    - "0:9:2" meaning 0..9 step 2
    - "1,2,5,9"
    """

    spec = spec.strip()
    if "," in spec:
        return [int(x.strip()) for x in spec.split(",") if x.strip()]
    if ":" in spec:
        parts = [p.strip() for p in spec.split(":")]
        if len(parts) == 2:
            a, b = int(parts[0]), int(parts[1])
            step = 1
        elif len(parts) == 3:
            a, b, step = int(parts[0]), int(parts[1]), int(parts[2])
        else:
            raise ValueError(f"Bad --seeds spec: {spec}")
        if step <= 0:
            raise ValueError("step must be positive")
        if a <= b:
            return list(range(a, b + 1, step))
        return list(range(a, b - 1, -step))
    return [int(spec)]


def _pick_machine(
    assignments: List[List[int]], policy: str, rng: np.random.Generator, index: int
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
        return int(rng.integers(0, len(assignments)))
    if policy == "index":
        if index < 0 or index >= len(assignments):
            raise ValueError(
                f"machine-index out of range: {index} (m={len(assignments)})"
            )
        return int(index)
    raise ValueError(f"Unknown machine policy: {policy}")


def _bnb_sequence_to_schedule(
    sequence: List[int], processing_times: np.ndarray, starts: List[int]
) -> Tuple[Tuple[int, int, int], ...]:
    out: List[Tuple[int, int, int]] = []
    for job_id, s in zip(sequence, starts):
        p = int(processing_times[int(job_id)])
        out.append((int(job_id), int(s), int(s + p)))
    return tuple(out)


def _plot_price_and_schedules(
    out_png: str,
    prices: np.ndarray,
    dp_sched: Tuple[Tuple[int, int, int], ...],
    bnb_sched: Tuple[Tuple[int, int, int], ...],
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = int(len(prices))
    x = np.arange(T)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, prices, linewidth=1.5)
    ax.set_xlim(0, max(1, T - 1))
    ax.set_xlabel("time")
    ax.set_ylabel("price")
    ax.set_title(title)

    # Draw schedules as bars below the curve.
    y0 = float(np.min(prices))
    y_span = float(np.max(prices) - np.min(prices) + 1e-9)

    def draw_row(
        sched: Tuple[Tuple[int, int, int], ...], y_frac: float, label: str
    ) -> None:
        y = y0 - y_span * y_frac
        for _, s, e in sched:
            ax.plot([s, e], [y, y], linewidth=6, solid_capstyle="butt")
        ax.text(0, y, label, va="center", ha="left", fontsize=10)

    draw_row(dp_sched, 0.12, "DP")
    draw_row(bnb_sched, 0.22, "BnB")

    ax.set_ylim(y0 - y_span * 0.30, y0 + y_span * 1.05)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def _run_one(seed: int, args_dict: Dict[str, Any]) -> Dict[str, Any]:
    scale = str(args_dict["scale"])
    T_max = int(args_dict["T_max"])
    m = args_dict.get("m")
    n = args_dict.get("n")
    assignment_conc = float(args_dict["assignment_concentration"])
    machine_policy = str(args_dict["machine_policy"])
    machine_index = int(args_dict["machine_index"])
    bnb_time_limit = float(args_dict["bnb_time_limit"])
    dp_time_limit = float(args_dict.get("dp_time_limit", -1.0))
    dp_tie_break = str(args_dict.get("dp_tie_break", "early"))
    out_dir = str(args_dict["out_dir"])

    # Reproducible RNGs
    import random

    py_rng = random.Random(int(seed))
    np_rng = np.random.default_rng(int(seed))

    config = DataConfig()
    config.sampling_mode = "paper_grid_90"

    raw = generate_raw_instance(
        config=config,
        rng=py_rng,
        instance_id=int(seed),
        T_max=T_max,
        m=m,
        n=n,
    )

    assignments = simulate_metaheuristic_assignment(
        n=raw.n,
        m=raw.m,
        rng=py_rng,
        concentration=assignment_conc,
    )

    machine_idx = _pick_machine(assignments, machine_policy, np_rng, machine_index)
    job_indices = assignments[machine_idx]

    p_subset = np.array([raw.p[j] for j in job_indices], dtype=np.int32)
    prices = np.array(raw.ct, dtype=np.float64)

    # --- DP ---
    t0 = time.perf_counter()
    dp_res = solve_optimal_benchmark_dp(
        p_subset.tolist(),
        prices,
        job_ids=range(len(p_subset)),
        tie_break=dp_tie_break,
        time_limit=dp_time_limit,
    )
    dp_time = float(time.perf_counter() - t0)

    # --- BnB ---
    inst = Instance(
        n_jobs=int(len(p_subset)),
        processing_times=p_subset.copy(),
        T=int(raw.T_max),
        energy_costs=prices.copy(),
    )

    if bnb_time_limit <= 0:
        bnb_time_limit = 1e18

    solver = BranchAndBoundSolver(inst, time_limit=float(bnb_time_limit), verbose=False)
    t1 = time.perf_counter()
    seq, bnb_cost = solver.solve()
    bnb_time = float(time.perf_counter() - t1)

    # Recover BnB schedule (start times) for plotting
    if seq:
        pts = inst.processing_times[np.asarray(seq, dtype=np.int32)].tolist()
        _cost2, starts = solver._dp_evaluate_with_schedule(pts)
        bnb_sched = _bnb_sequence_to_schedule(seq, inst.processing_times, starts)
    else:
        bnb_sched = ()

    dp_sched = dp_res.schedule

    # Save artifacts
    seed_tag = f"seed{seed}_T{raw.T_max}_m{raw.m}_n{raw.n}_machine{machine_idx}_nj{len(p_subset)}"
    out_json = os.path.join(out_dir, f"{seed_tag}.json")
    out_png = os.path.join(out_dir, f"{seed_tag}.png")

    payload: Dict[str, Any] = {
        "seed": int(seed),
        "scale": str(raw.scale),
        "T": int(raw.T_max),
        "m": int(raw.m),
        "n": int(raw.n),
        "machine": int(machine_idx),
        "n_jobs_sm": int(len(p_subset)),
        "sum_p_sm": int(np.sum(p_subset)) if len(p_subset) else 0,
        "unique_p_sm": int(len(set(map(int, p_subset.tolist())))),
        "dp": {
            "tie_break": dp_tie_break,
            "feasible": bool(dp_res.feasible),
            "cost": float(dp_res.cost),
            "time_sec": float(dp_time),
            "finish_time": int(dp_res.finish_time),
            "schedule": [list(x) for x in dp_sched],
        },
        "bnb": {
            "timed_out": bool(getattr(solver, "timed_out", False)),
            "cost": float(bnb_cost),
            "time_sec": float(bnb_time),
            "nodes": int(getattr(solver, "nodes_explored", -1)),
            "sequence": [int(x) for x in (seq or [])],
            "schedule": [list(x) for x in bnb_sched],
        },
        "cost_match": bool(
            np.isfinite(dp_res.cost)
            and np.isfinite(bnb_cost)
            and abs(dp_res.cost - float(bnb_cost)) <= 1e-6
        ),
        "out_png": out_png,
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    title = (
        f"{seed_tag} | DP {dp_res.cost:.3f} ({dp_time:.2f}s) "
        f"vs BnB {float(bnb_cost):.3f} ({bnb_time:.2f}s)"
    )
    _plot_price_and_schedules(out_png, prices, dp_sched, bnb_sched, title)

    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=["small", "mls", "vls"], default="mls")
    ap.add_argument("--seeds", type=str, default="0:9")
    ap.add_argument("--T-max", type=int, default=None, dest="T_max")
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default=None)

    ap.add_argument(
        "--machine-policy",
        choices=["max_jobs", "random", "index"],
        default="max_jobs",
    )
    ap.add_argument("--machine-index", type=int, default=0)
    ap.add_argument("--assignment-concentration", type=float, default=1.0)

    ap.add_argument(
        "--bnb-time-limit",
        type=float,
        default=0.0,
        help="Seconds. 0 means no limit (very large).",
    )
    ap.add_argument(
        "--dp-tie-break",
        choices=["cost", "early"],
        default="early",
        help="Tie-break among equal-cost DP optima (early = prefer earlier schedule)",
    )
    ap.add_argument(
        "--dp-time-limit",
        type=float,
        default=-1.0,
        help="Seconds for DP. -1 means no limit.",
    )

    args = ap.parse_args()

    seeds = _parse_seeds(args.seeds)
    T_max = (
        int(args.T_max) if args.T_max is not None else _default_T_max(str(args.scale))
    )

    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(
            "PaST", "analysis_out", f"compare_dp_bnb_{args.scale}_T{T_max}_{ts}"
        )
    else:
        out_dir = str(args.out_dir)

    workers = int(args.workers)
    if workers <= 0:
        workers = max(1, (os.cpu_count() or 2) - 1)

    args_dict: Dict[str, Any] = {
        "scale": str(args.scale),
        "T_max": int(T_max),
        "m": args.m,
        "n": args.n,
        "assignment_concentration": float(args.assignment_concentration),
        "machine_policy": str(args.machine_policy),
        "machine_index": int(args.machine_index),
        "bnb_time_limit": float(args.bnb_time_limit),
        "dp_time_limit": float(args.dp_time_limit),
        "dp_tie_break": str(args.dp_tie_break),
        "out_dir": str(out_dir),
    }

    from multiprocessing import get_context

    ctx = get_context("spawn")
    results: List[Dict[str, Any]] = []

    if workers == 1 or len(seeds) == 1:
        for s in seeds:
            results.append(_run_one(int(s), args_dict))
    else:
        with ctx.Pool(processes=workers) as pool:
            for payload in pool.starmap(_run_one, [(int(s), args_dict) for s in seeds]):
                results.append(payload)

    summary_csv = os.path.join(out_dir, "summary.csv")
    lines = [
        "seed,scale,T,m,n,machine,n_jobs_sm,sum_p_sm,unique_p_sm,dp_cost,dp_time_sec,bnb_cost,bnb_time_sec,bnb_nodes,cost_match,bnb_timed_out"
    ]
    for r in results:
        lines.append(
            ",".join(
                [
                    str(r["seed"]),
                    str(r["scale"]),
                    str(r["T"]),
                    str(r["m"]),
                    str(r["n"]),
                    str(r["machine"]),
                    str(r["n_jobs_sm"]),
                    str(r["sum_p_sm"]),
                    str(r["unique_p_sm"]),
                    str(r["dp"]["cost"]),
                    str(r["dp"]["time_sec"]),
                    str(r["bnb"]["cost"]),
                    str(r["bnb"]["time_sec"]),
                    str(r["bnb"]["nodes"]),
                    str(r["cost_match"]),
                    str(r["bnb"]["timed_out"]),
                ]
            )
        )

    os.makedirs(out_dir, exist_ok=True)
    with open(summary_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {len(results)} results to {out_dir}", flush=True)
    print(f"Summary: {summary_csv}", flush=True)


if __name__ == "__main__":
    main()

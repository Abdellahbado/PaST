#!/usr/bin/env python3
"""
06_ablation_study.py — Ablation study: Banded SPACES vs. Bound-and-Refine DP.

Runs our solver 4 times with different ablation configurations to isolate
the contribution of each component:

  A) "full"         — Banded SPACES + full bound-and-refine   (our method)
  B) "full_spaces"  — Full O(h²) SPACES + full bound-and-refine
  C) "exact_only"   — Banded SPACES + exact DP only (no heuristics/relaxation LB)
  D) "baseline"     — Full O(h²) SPACES + exact DP only

Comparison matrix:
  A vs B  →  Speedup from banded SPACES
  A vs C  →  Speedup from bound-and-refine heuristics + relaxation LB
  A vs D  →  Total speedup from both components
  B vs D  →  Bound-and-refine contribution when SPACES is unoptimized

Instances: Table 2 (§5.2) + Figure 9 (§5.3) — 1120 instances total.
These are the large, hard instances where component contributions are visible.

Usage:
  python3 hpc/06_ablation_study.py [--section all|2|fig9|table1] \\
                                   [--config all|full|full_spaces|exact_only|baseline] \\
                                   [--time-limit 600] \\
                                   [--output-dir hpc/results_ablation]
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOLVER = ROOT / "solvers" / "cpp" / "build" / "stateful_compare"
PAPER_DATASETS = (
    ROOT
    / "data"
    / "green-scheduling-bab"
    / "Iirc.EnergyStatesAndCostsScheduling"
    / "data"
    / "datasets"
)

ABLATION_CONFIGS = ["full", "full_spaces", "exact_only", "baseline"]

ABLATION_HEADER = (
    "config,section,dataset,instance_id,n_jobs,horizon,ub,lb,gap_pct,"
    "feasible,is_optimal,timed_out,runtime_sec,"
    "t_spaces,t_fwd_relax,t_heuristic,t_local_search,"
    "t_bwd_relax,t_two_class,t_exact,step_reached,max_gap"
)

# ── Dataset definitions ──────────────────────────────────────────────────
TABLE1_DATASETS = [
    "benedikt2025a_large_nosby",
    "benedikt2025a_large_twosby",
    "benedikt2025a_medium_nosby",
    "benedikt2025a_medium_twosby",
    "benedikt2025a_prelim",
    "aghelinejad2017a_1",
]
TABLE2_DATASETS = ["benedikt2025b_groups"]
FIG9_DATASETS = ["benedikt2025_groups"]


def log(msg: str, logfile=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if logfile:
        logfile.write(line + "\n")
        logfile.flush()


def load_paper_instance(json_path: Path) -> dict:
    with open(json_path) as f:
        d = json.load(f)
    jobs = [j["ProcessingTime"] for j in d["Jobs"]]
    prices = d["EnergyCosts"]
    noff = len(d.get("OffOnTime", []))
    mtype = "twosby" if noff == 3 else "nosby"
    return {"jobs": jobs, "prices": prices, "machine_type": mtype}


def load_all_instances(dataset_names: list[str], section: str, logfile=None):
    """Load all instances from the given datasets."""
    instances = []
    for ds_name in dataset_names:
        ds_dir = PAPER_DATASETS / ds_name
        if not ds_dir.exists():
            log(f"  WARNING: dataset not found: {ds_dir}", logfile)
            continue
        json_files = sorted(ds_dir.glob("*.json"), key=lambda p: int(p.stem))
        for jf in json_files:
            idx = int(jf.stem)
            inst = load_paper_instance(jf)
            inst_id = f"{ds_name}/{idx}"
            instances.append((section, ds_name, inst_id, inst))
        log(f"  Loaded {len(json_files)} instances from {ds_name}", logfile)
    return instances


def run_ablation_config(
    config_name: str,
    instances: list[tuple[str, str, str, dict]],
    solver_path: Path,
    time_limit: float,
    batch_size: int = 50,
    logfile=None,
) -> list[dict]:
    """Run solver with a specific ablation config on all instances."""
    results = []

    # Group instances into batches
    for start in range(0, len(instances), batch_size):
        batch = instances[start : start + batch_size]

        jsonl_lines = []
        for section, ds_name, inst_id, inst in batch:
            payload = {
                "instance_id": inst_id,
                "prices": inst["prices"],
                "jobs": inst["jobs"],
                "machine": inst.get("machine_type", "nosby"),
            }
            jsonl_lines.append(json.dumps(payload))

        input_data = "\n".join(jsonl_lines) + "\n"
        args = [str(solver_path), "ablation-stdin", config_name]

        batch_timeout = max(time_limit * len(batch) + 300, 7200)

        try:
            proc = subprocess.run(
                args,
                input=input_data,
                capture_output=True,
                text=True,
                timeout=int(batch_timeout),
            )
        except subprocess.TimeoutExpired:
            log(f"  WARNING: Batch timeout ({config_name})", logfile)
            continue

        if proc.returncode != 0 and proc.stderr:
            log(f"  Solver stderr ({config_name}): {proc.stderr[:300]}", logfile)

        lines = proc.stdout.strip().split("\n")
        for line in lines[1:]:  # skip header
            if not line.strip():
                continue
            parts = line.split(",")
            if len(parts) >= 19:
                inst_id = parts[0]
                # Find the matching section/dataset
                sec, ds = "", ""
                for s, d, iid, _ in batch:
                    if iid == inst_id:
                        sec, ds = s, d
                        break
                results.append(
                    {
                        "config": config_name,
                        "section": sec,
                        "dataset": ds,
                        "instance_id": inst_id,
                        "n_jobs": int(parts[1]),
                        "horizon": int(parts[2]),
                        "ub": float(parts[3]),
                        "lb": float(parts[4]),
                        "gap_pct": float(parts[5]),
                        "feasible": int(parts[6]),
                        "is_optimal": int(parts[7]),
                        "timed_out": int(parts[8]),
                        "runtime_sec": float(parts[9]),
                        "t_spaces": float(parts[10]),
                        "t_fwd_relax": float(parts[11]),
                        "t_heuristic": float(parts[12]),
                        "t_local_search": float(parts[13]),
                        "t_bwd_relax": float(parts[14]),
                        "t_two_class": float(parts[15]),
                        "t_exact": float(parts[16]),
                        "step_reached": parts[17],
                        "max_gap": parts[18],
                    }
                )

        done = start + len(batch)
        if len(instances) > batch_size:
            log(f"    [{config_name}] Progress: {done}/{len(instances)}", logfile)

    return results


def write_csv(results: list[dict], path: Path):
    with open(path, "w") as f:
        f.write(ABLATION_HEADER + "\n")
        for r in results:
            f.write(
                f"{r['config']},{r['section']},{r['dataset']},{r['instance_id']},"
                f"{r['n_jobs']},{r['horizon']},"
                f"{r['ub']:.6f},{r['lb']:.6f},{r['gap_pct']:.4f},"
                f"{r['feasible']},{r['is_optimal']},{r['timed_out']},"
                f"{r['runtime_sec']:.4f},"
                f"{r['t_spaces']:.4f},{r['t_fwd_relax']:.4f},"
                f"{r['t_heuristic']:.4f},{r['t_local_search']:.4f},"
                f"{r['t_bwd_relax']:.4f},{r['t_two_class']:.4f},"
                f"{r['t_exact']:.4f},{r['step_reached']},{r['max_gap']}\n"
            )


def print_config_summary(config_name, results, logfile=None):
    if not results:
        return
    n = len(results)
    n_opt = sum(1 for r in results if r["is_optimal"])
    total_t = sum(r["runtime_sec"] for r in results)
    max_t = max(r["runtime_sec"] for r in results)
    avg_t = total_t / max(n, 1)

    # Per-step averages
    avg_steps = {}
    for key in [
        "t_spaces",
        "t_fwd_relax",
        "t_heuristic",
        "t_local_search",
        "t_bwd_relax",
        "t_two_class",
        "t_exact",
    ]:
        avg_steps[key] = sum(r[key] for r in results) / max(n, 1)

    log(
        f"  [{config_name:>12}]  {n_opt:>4}/{n:<4} opt  "
        f"avg={avg_t:>8.4f}s  max={max_t:>8.3f}s  total={total_t:>8.1f}s",
        logfile,
    )
    log(
        f"    Per-step avg: SPACES={avg_steps['t_spaces']:.4f}  "
        f"fwd_relax={avg_steps['t_fwd_relax']:.4f}  "
        f"heur={avg_steps['t_heuristic']:.4f}  "
        f"local={avg_steps['t_local_search']:.4f}  "
        f"bwd={avg_steps['t_bwd_relax']:.4f}  "
        f"2class={avg_steps['t_two_class']:.4f}  "
        f"exact={avg_steps['t_exact']:.4f}",
        logfile,
    )


def print_comparison(all_results: dict[str, list[dict]], logfile=None):
    """Print pairwise speedup comparisons across configs."""
    log("", logfile)
    log("=" * 90, logfile)
    log("  ABLATION COMPARISON", logfile)
    log("=" * 90, logfile)

    # Build per-instance lookup: config -> {instance_id: runtime}
    runtimes = {}
    for cfg, results in all_results.items():
        runtimes[cfg] = {r["instance_id"]: r["runtime_sec"] for r in results}

    pairs = [
        ("full", "full_spaces", "Banded SPACES contribution (A vs B)"),
        ("full", "exact_only", "Bound-and-refine contribution (A vs C)"),
        ("full", "baseline", "Total speedup: both components (A vs D)"),
        ("full_spaces", "baseline", "Bound-and-refine when SPACES is full (B vs D)"),
    ]

    for fast, slow, label in pairs:
        if fast not in runtimes or slow not in runtimes:
            continue
        common = set(runtimes[fast].keys()) & set(runtimes[slow].keys())
        if not common:
            continue

        speedups = []
        for iid in sorted(common):
            t_fast = runtimes[fast][iid]
            t_slow = runtimes[slow][iid]
            if t_fast > 0:
                speedups.append(t_slow / t_fast)

        if not speedups:
            continue

        avg_su = sum(speedups) / len(speedups)
        med_su = sorted(speedups)[len(speedups) // 2]
        max_su = max(speedups)
        min_su = min(speedups)

        t_fast_total = sum(runtimes[fast][iid] for iid in common)
        t_slow_total = sum(runtimes[slow][iid] for iid in common)
        agg_su = t_slow_total / max(t_fast_total, 1e-9)

        log(f"\n  {label}", logfile)
        log(f"    Instances: {len(common)}", logfile)
        log(
            f"    Aggregate speedup: {agg_su:.2f}x  "
            f"(total: {t_fast_total:.1f}s vs {t_slow_total:.1f}s)",
            logfile,
        )
        log(
            f"    Per-instance:  avg={avg_su:.2f}x  med={med_su:.2f}x  "
            f"min={min_su:.2f}x  max={max_su:.2f}x",
            logfile,
        )

    # Step-time comparison table
    log("\n" + "-" * 90, logfile)
    log("  PER-STEP AVERAGE TIMES (seconds)", logfile)
    log("-" * 90, logfile)
    steps = [
        "t_spaces",
        "t_fwd_relax",
        "t_heuristic",
        "t_local_search",
        "t_bwd_relax",
        "t_two_class",
        "t_exact",
    ]
    header = (
        f"  {'Config':>12}"
        + "".join(f"  {s.replace('t_',''):>10}" for s in steps)
        + f"  {'TOTAL':>10}"
    )
    log(header, logfile)
    log("  " + "-" * (len(header) - 2), logfile)

    for cfg in ABLATION_CONFIGS:
        if cfg not in all_results or not all_results[cfg]:
            continue
        results = all_results[cfg]
        n = len(results)
        avgs = [sum(r[s] for r in results) / n for s in steps]
        total = sum(avgs)
        row = (
            f"  {cfg:>12}"
            + "".join(f"  {a:>10.4f}" for a in avgs)
            + f"  {total:>10.4f}"
        )
        log(row, logfile)


def main():
    ap = argparse.ArgumentParser(
        description="Ablation study: Banded SPACES vs Bound-and-Refine"
    )
    ap.add_argument(
        "--section", default="all", choices=["all", "1", "2", "fig9", "table1"]
    )
    ap.add_argument(
        "--config",
        default="all",
        choices=["all"] + ABLATION_CONFIGS,
        help="Which ablation config to run (default: all)",
    )
    ap.add_argument("--time-limit", type=float, default=600)
    ap.add_argument("--output-dir", default=str(ROOT / "hpc" / "results_ablation"))
    ap.add_argument("--batch-size", type=int, default=50)
    args = ap.parse_args()

    solver_path = Path(args.solver) if hasattr(args, "solver") else SOLVER
    if not solver_path.exists():
        print(f"ERROR: Solver not found: {solver_path}")
        print(
            "  Run: cd solvers/cpp/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j"
        )
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logpath = out_dir / "ablation_log.txt"
    logfile = open(logpath, "w")

    log("=" * 90, logfile)
    log("  ABLATION STUDY — Banded SPACES vs. Bound-and-Refine DP", logfile)
    log(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}", logfile)
    log(f"  Solver:     {solver_path}", logfile)
    log(f"  Time limit: {args.time_limit}s per instance", logfile)
    log(f"  Section:    {args.section}", logfile)
    log(f"  Config:     {args.config}", logfile)
    log("=" * 90, logfile)

    # Determine which configs to run
    configs = ABLATION_CONFIGS if args.config == "all" else [args.config]

    # ── Load instances ──────────────────────────────────────────────────
    all_instances = []
    if args.section in ("all", "table1", "1"):
        log("\n--- Loading Table 1 instances ---", logfile)
        all_instances.extend(load_all_instances(TABLE1_DATASETS, "table1", logfile))

    if args.section in ("all", "2"):
        log("\n--- Loading Table 2 instances ---", logfile)
        all_instances.extend(load_all_instances(TABLE2_DATASETS, "table2", logfile))

    if args.section in ("all", "fig9"):
        log("\n--- Loading Figure 9 instances ---", logfile)
        all_instances.extend(load_all_instances(FIG9_DATASETS, "fig9", logfile))

    log(f"\nTotal instances to solve: {len(all_instances)}", logfile)
    log(f"Configs to run: {configs}", logfile)
    log(f"Total solver invocations: {len(all_instances) * len(configs)}", logfile)
    log("", logfile)

    # ── Run each config ────────────────────────────────────────────────
    all_config_results = {}
    grand_start = time.monotonic()

    for cfg in configs:
        log(f"\n{'#' * 90}", logfile)
        log(f"# Running config: {cfg}", logfile)
        log(f"{'#' * 90}", logfile)

        t0 = time.monotonic()
        results = run_ablation_config(
            cfg, all_instances, solver_path, args.time_limit, args.batch_size, logfile
        )
        elapsed = time.monotonic() - t0

        all_config_results[cfg] = results
        log(
            f"\n  Config '{cfg}' done in {elapsed:.1f}s ({len(results)} results)",
            logfile,
        )
        print_config_summary(cfg, results, logfile)

        # Write per-config CSV
        write_csv(results, out_dir / f"ablation_{cfg}.csv")

    # ── Combined CSV ───────────────────────────────────────────────────
    combined = []
    for cfg, results in all_config_results.items():
        combined.extend(results)
    write_csv(combined, out_dir / "ablation_combined.csv")

    # ── Comparison ─────────────────────────────────────────────────────
    print_comparison(all_config_results, logfile)

    grand_elapsed = time.monotonic() - grand_start
    log(f"\n{'=' * 90}", logfile)
    log(f"  ABLATION STUDY COMPLETE — Total wall time: {grand_elapsed:.1f}s", logfile)
    log(f"  Results in: {out_dir}", logfile)
    log(f"{'=' * 90}", logfile)

    logfile.close()
    print(f"\nDone. Results in {out_dir}/")
    print(f"  Combined:  ablation_combined.csv")
    print(f"  Per-config: ablation_{{full,full_spaces,exact_only,baseline}}.csv")
    print(f"  Log:       ablation_log.txt")
    print(f"\nNext: python3 hpc/07_analyze_ablation.py --input-dir {out_dir}")


if __name__ == "__main__":
    main()

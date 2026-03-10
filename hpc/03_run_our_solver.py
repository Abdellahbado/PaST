#!/usr/bin/env python3
"""
03_run_our_solver.py — Run our C++ solver on ALL 1517+ instances.

Sections:
  Table 1 (§5.1):    72 instances (6 datasets × 12 instances, NOSBY+TWOSBY)
  Table 2 (§5.2):   560 instances (benedikt2025b_groups, TWOSBY, OTE prices)
  Figure 9 (§5.3):  560 instances (benedikt2025_groups, TWOSBY, OTE prices)
  Drops ablation:   240 instances (benedikt2025b_drops, TWOSBY)
  GCD analysis:       1 instance  (benedikt2025b_gcd, NOSBY)
  Hard synthetic:    84 instances (gen_hard_instances style)

Output: One CSV per section + combined grand CSV + detailed log.

Usage:
  python3 hpc/03_run_our_solver.py [--section all|1|2|fig9|drops|gcd|synthetic] \\
                                   [--time-limit 600] [--output-dir hpc/results_ours]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import resource
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SOLVER = ROOT / "solvers" / "cpp" / "build" / "stateful_compare"
# Both Table 1 and Table 2+ datasets live in the paper's data directory.
# Table 1 datasets are named benedikt2025a_* in the paper's repo
# (same instances as our benedikt2020a_* — verified byte-identical).
PAPER_DATASETS = (
    ROOT
    / "data"
    / "green-scheduling-bab"
    / "Iirc.EnergyStatesAndCostsScheduling"
    / "data"
    / "datasets"
)

# ── CSV header for all outputs ───────────────────────────────────────────────
CSV_HEADER = (
    "section,dataset,instance_id,n_jobs,horizon,ub,lb,gap_pct,"
    "feasible,is_optimal,timed_out,runtime_sec,peak_rss_kb"
)

# ── Known paper Table 1 optimal costs (for validation) ───────────────────────
PAPER_TABLE1_NOSBY = {
    0: 8582,
    1: 8409,
    2: 8132,
    3: 8078,
    4: 10068,
    5: 9820,
    6: 9637,
    7: 9620,
    8: 12008,
    9: 11758,
    10: 11611,
    11: 11465,
}
PAPER_TABLE1_TWOSBY = {
    0: 21910,
    1: 21821,
    2: 21353,
    3: 21266,
    4: 25807,
    5: 25518,
    6: 25279,
    7: 25279,
    8: 30563,
    9: 30224,
    10: 30224,
    11: 30071,
}


def log(msg: str, logfile=None):
    """Print to stdout and optionally to logfile."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if logfile:
        logfile.write(line + "\n")
        logfile.flush()


def get_system_info() -> str:
    """Collect system info for reproducibility."""
    lines = [
        f"Platform:  {platform.platform()}",
        f"Processor: {platform.processor()}",
        f"Python:    {platform.python_version()}",
    ]
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    lines.append(f"CPU model: {line.split(':')[1].strip()}")
                    break
    except FileNotFoundError:
        try:
            import subprocess as sp

            r = sp.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            )
            if r.returncode == 0:
                lines.append(f"CPU model: {r.stdout.strip()}")
        except Exception:
            pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    lines.append(f"RAM:       {line.split(':')[1].strip()}")
                    break
    except FileNotFoundError:
        pass
    try:
        nproc = os.cpu_count()
        lines.append(f"CPU cores: {nproc}")
    except Exception:
        pass
    return "\n".join(lines)


def load_paper_instance(json_path: Path) -> dict:
    """Load a paper instance JSON and extract jobs/prices/machine."""
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


def solve_batch(
    instances: list[tuple[str, dict]],
    solver_path: Path,
    time_limit: float,
    logfile=None,
) -> list[dict]:
    """Solve a batch of instances via solve-stdin, return parsed results."""
    if not instances:
        return []

    jsonl_lines = []
    for inst_id, inst in instances:
        payload = {
            "instance_id": inst_id,
            "prices": inst["prices"],
            "jobs": inst["jobs"],
            "machine": inst.get("machine_type", inst.get("machine", "nosby")),
        }
        jsonl_lines.append(json.dumps(payload))

    input_data = "\n".join(jsonl_lines) + "\n"
    args = [str(solver_path), "solve-stdin"]
    if time_limit > 0:
        args.append(str(time_limit))

    wall_start = time.monotonic()

    try:
        # Set generous per-batch timeout
        batch_timeout = max(time_limit * len(instances) + 300, 7200)
        proc = subprocess.run(
            args,
            input=input_data,
            capture_output=True,
            text=True,
            timeout=int(batch_timeout),
        )
    except subprocess.TimeoutExpired:
        log(f"  WARNING: Batch timeout after {batch_timeout}s", logfile)
        return []

    wall_elapsed = time.monotonic() - wall_start

    if proc.returncode != 0 and proc.stderr:
        log(f"  Solver stderr: {proc.stderr[:300]}", logfile)

    # Parse peak RSS from /proc if available
    peak_rss_kb = 0
    try:
        ru = resource.getrusage(resource.RUSAGE_CHILDREN)
        peak_rss_kb = ru.ru_maxrss  # KB on Linux, bytes on macOS
        if sys.platform == "darwin":
            peak_rss_kb //= 1024
    except Exception:
        pass

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
                    "peak_rss_kb": peak_rss_kb,
                }
            )

    return results


def run_paper_datasets(
    section_name: str,
    dataset_names: list[str],
    data_dir: Path,
    solver_path: Path,
    time_limit: float,
    batch_size: int,
    logfile=None,
) -> list[dict]:
    """Run solver on paper benchmark datasets (JSON instances)."""
    all_results = []

    for ds_name in dataset_names:
        ds_dir = data_dir / ds_name
        if not ds_dir.exists():
            log(f"  WARNING: Dataset not found: {ds_dir}", logfile)
            continue

        json_files = sorted(ds_dir.glob("*.json"), key=lambda p: int(p.stem))
        instances = []
        for jf in json_files:
            idx = int(jf.stem)
            inst = load_paper_instance(jf)
            inst_id = f"{ds_name}/{idx}"
            instances.append((inst_id, inst))

        n = len(instances)
        mtype = instances[0][1]["machine_type"] if instances else "?"
        log(f"  Dataset: {ds_name} ({n} instances, {mtype.upper()})", logfile)

        # Solve in batches
        ds_results = []
        for start in range(0, n, batch_size):
            batch = instances[start : start + batch_size]
            batch_results = solve_batch(batch, solver_path, time_limit, logfile)
            ds_results.extend(batch_results)
            done = start + len(batch)
            if n > batch_size:
                log(f"    Progress: {done}/{n}", logfile)

        # Tag results with section and dataset
        for r in ds_results:
            r["section"] = section_name
            r["dataset"] = ds_name

        all_results.extend(ds_results)

        # Quick summary
        n_opt = sum(1 for r in ds_results if r["is_optimal"])
        n_open = len(ds_results) - n_opt
        total_t = sum(r["runtime_sec"] for r in ds_results)
        max_t = max((r["runtime_sec"] for r in ds_results), default=0)
        avg_t = total_t / max(len(ds_results), 1)
        log(
            f"    => {n_opt}/{len(ds_results)} optimal, avg={avg_t:.3f}s, max={max_t:.3f}s, total={total_t:.1f}s",
            logfile,
        )

        if n_open > 0:
            for r in ds_results:
                if not r["is_optimal"]:
                    log(
                        f"    OPEN: {r['instance_id']} ub={r['ub']:.1f} lb={r['lb']:.1f} gap={r['gap_pct']:.4f}%",
                        logfile,
                    )

    return all_results


def gen_synthetic_instance(
    n: int, lam: float, proc_dist: str, seed: int, machine: str = "nosby"
) -> dict:
    """Generate a hard synthetic instance (same as scripts/gen_hard_instances.py)."""
    rng = random.Random(seed)

    if proc_dist.startswith("U["):
        lo, hi = map(int, proc_dist[2:-1].split(","))
        jobs = [rng.randint(lo, hi) for _ in range(n)]
    elif proc_dist.startswith("{"):
        vals = list(map(int, proc_dist[1:-1].split(",")))
        jobs = [rng.choice(vals) for _ in range(n)]
    else:
        raise ValueError(f"Unknown dist: {proc_dist}")

    if machine == "nosby":
        startup, shutdown = 2, 1
    elif machine == "twosby":
        startup, shutdown = 4, 1
    else:
        raise ValueError(f"Unknown machine: {machine}")

    lower = startup + sum(jobs) + shutdown
    horizon = int(math.ceil(lam * lower))
    prices = [float(rng.randint(1, 10)) for _ in range(horizon)]

    dist_tag = (
        proc_dist.replace("[", "")
        .replace("]", "")
        .replace("{", "")
        .replace("}", "")
        .replace(",", "-")
    )
    return {
        "instance_id": f"synth_{dist_tag}_n{n}_lam{lam:.1f}_s{seed}_{machine}",
        "prices": prices,
        "jobs": jobs,
        "machine_type": machine,
        "n_jobs": n,
        "horizon": horizon,
        "proc_dist": proc_dist,
    }


def run_synthetic(solver_path: Path, time_limit: float, logfile=None) -> list[dict]:
    """Generate and solve hard synthetic instances."""
    seeds = [42, 123, 456]
    machines = ["nosby", "twosby"]
    configs = [
        (150, 1.3, "U[1,5]"),
        (190, 2.2, "U[1,5]"),
        (150, 1.3, "U[1,10]"),
        (190, 1.3, "U[1,10]"),
        (190, 2.2, "U[1,10]"),
        (100, 1.3, "{8,10}"),
        (150, 1.3, "{8,10}"),
        (100, 1.3, "{9,10}"),
        (150, 1.3, "{9,10}"),
        (100, 1.3, "{3,7}"),
        (150, 1.3, "{3,7}"),
        (150, 1.6, "{3,7}"),
        (100, 1.3, "U[1,20]"),
        (150, 1.3, "U[1,20]"),
    ]

    all_instances = []
    for machine in machines:
        for seed in seeds:
            for n, lam, dist in configs:
                inst = gen_synthetic_instance(n, lam, dist, seed, machine)
                all_instances.append(inst)

    log(f"  Generated {len(all_instances)} synthetic instances", logfile)

    # Solve all
    pairs = [(inst["instance_id"], inst) for inst in all_instances]
    results = solve_batch(pairs, solver_path, time_limit, logfile)

    # Tag
    for r in results:
        r["section"] = "synthetic"
        r["dataset"] = "hard_synthetic"

    n_opt = sum(1 for r in results if r["is_optimal"])
    total_t = sum(r["runtime_sec"] for r in results)
    max_t = max((r["runtime_sec"] for r in results), default=0)
    log(
        f"    => {n_opt}/{len(results)} optimal, max={max_t:.3f}s, total={total_t:.1f}s",
        logfile,
    )

    return results


def write_csv(results: list[dict], path: Path):
    """Write results to CSV."""
    with open(path, "w") as f:
        f.write(CSV_HEADER + "\n")
        for r in results:
            f.write(
                f"{r.get('section','')},{r.get('dataset','')},{r['instance_id']},"
                f"{r['n_jobs']},{r['horizon']},"
                f"{r['ub']:.6f},{r['lb']:.6f},{r['gap_pct']:.4f},"
                f"{r['feasible']},{r['is_optimal']},{r['timed_out']},"
                f"{r['runtime_sec']:.4f},{r.get('peak_rss_kb',0)}\n"
            )


def print_section_summary(section: str, results: list[dict], logfile=None):
    """Print a detailed summary for a section."""
    if not results:
        return

    n = len(results)
    n_opt = sum(1 for r in results if r["is_optimal"])
    n_feas = sum(1 for r in results if r["feasible"])
    n_infeas = n - n_feas
    n_open = n_feas - n_opt
    total_t = sum(r["runtime_sec"] for r in results)
    max_t = max(r["runtime_sec"] for r in results)
    avg_t = total_t / max(n, 1)

    log("", logfile)
    log(f"{'='*80}", logfile)
    log(
        f"  {section}: {n_opt}/{n_feas} optimal"
        f" + {n_infeas} infeasible = {n} total",
        logfile,
    )
    log(f"  Total time: {total_t:.2f}s, Avg: {avg_t:.4f}s, Max: {max_t:.4f}s", logfile)
    if n_open > 0:
        log(f"  *** {n_open} OPEN GAPS ***", logfile)
    log(f"{'='*80}", logfile)

    # Group by dataset
    by_ds = defaultdict(list)
    for r in results:
        by_ds[r.get("dataset", "unknown")].append(r)

    for ds, ds_results in sorted(by_ds.items()):
        ds_n = len(ds_results)
        ds_opt = sum(1 for r in ds_results if r["is_optimal"])
        ds_t = sum(r["runtime_sec"] for r in ds_results)
        ds_max = max(r["runtime_sec"] for r in ds_results)
        ds_avg = ds_t / max(ds_n, 1)
        log(
            f"    {ds:>40}: {ds_opt:>4}/{ds_n:>4} opt  "
            f"avg={ds_avg:>8.3f}s  max={ds_max:>8.3f}s  total={ds_t:>8.1f}s",
            logfile,
        )


def main():
    ap = argparse.ArgumentParser(description="Run our solver on all instances")
    ap.add_argument(
        "--section",
        default="all",
        choices=["all", "1", "2", "fig9", "drops", "gcd", "synthetic"],
    )
    ap.add_argument("--solver", default=str(SOLVER))
    ap.add_argument(
        "--time-limit",
        type=float,
        default=600,
        help="Time limit per instance in seconds (default: 600)",
    )
    ap.add_argument("--output-dir", default=str(ROOT / "hpc" / "results_ours"))
    ap.add_argument("--batch-size", type=int, default=50)
    args = ap.parse_args()

    solver_path = Path(args.solver)
    if not solver_path.exists():
        print(f"ERROR: Solver not found: {solver_path}")
        print(f"  Run: bash hpc/01_build_our_solver.sh")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logpath = out_dir / "run_log.txt"
    logfile = open(logpath, "w")

    log("=" * 80, logfile)
    log(" PaST HPC Benchmark — Our Solver (Relaxed DP + Multiset Exact DP)", logfile)
    log(f" {time.strftime('%Y-%m-%d %H:%M:%S')}", logfile)
    log(f" Solver: {solver_path}", logfile)
    log(f" Time limit: {args.time_limit}s per instance", logfile)
    log("=" * 80, logfile)
    log("", logfile)
    log("--- System Info ---", logfile)
    log(get_system_info(), logfile)
    log("", logfile)

    # Warm-up: run one trivial instance to prime CPU caches and OS scheduler
    log("--- Warm-up: running one trivial instance to prime caches ---", logfile)
    warmup_inst = {
        "instance_id": "warmup",
        "prices": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "jobs": [2, 3],
        "machine_type": "nosby",
    }
    warmup_pairs = [("warmup", warmup_inst)]
    solve_batch(warmup_pairs, solver_path, 10)
    log("  Warm-up done (caches hot)", logfile)
    log("", logfile)

    grand_start = time.monotonic()
    all_results = []

    # ── Table 1 ──────────────────────────────────────────────────────────
    table1_datasets = [
        "benedikt2025a_large_nosby",
        "benedikt2025a_large_twosby",
        "benedikt2025a_medium_nosby",
        "benedikt2025a_medium_twosby",
        "benedikt2025a_prelim",
        "aghelinejad2017a_1",
    ]

    if args.section in ("1", "all"):
        log("", logfile)
        log("#" * 80, logfile)
        log("# TABLE 1 (§5.1) — 72 instances (NOSBY + TWOSBY)", logfile)
        log("#" * 80, logfile)
        results = run_paper_datasets(
            "table1",
            table1_datasets,
            PAPER_DATASETS,
            solver_path,
            args.time_limit,
            args.batch_size,
            logfile,
        )
        all_results.extend(results)
        write_csv(results, out_dir / "table1_results.csv")
        print_section_summary("TABLE 1", results, logfile)

        # Validate costs against paper
        log("\n--- Table 1 Cost Validation ---", logfile)
        for r in results:
            ds = r.get("dataset", "")
            parts = r["instance_id"].split("/")
            if len(parts) == 2:
                idx = int(parts[1])
                if "large_nosby" in ds and idx in PAPER_TABLE1_NOSBY:
                    expected = PAPER_TABLE1_NOSBY[idx]
                    match = "MATCH" if abs(r["ub"] - expected) < 0.5 else "MISMATCH"
                    if match == "MISMATCH":
                        log(
                            f"  {match}: {r['instance_id']} ub={r['ub']:.0f} expected={expected}",
                            logfile,
                        )
                elif "large_twosby" in ds and idx in PAPER_TABLE1_TWOSBY:
                    expected = PAPER_TABLE1_TWOSBY[idx]
                    match = "MATCH" if abs(r["ub"] - expected) < 0.5 else "MISMATCH"
                    if match == "MISMATCH":
                        log(
                            f"  {match}: {r['instance_id']} ub={r['ub']:.0f} expected={expected}",
                            logfile,
                        )
        log("  (Only mismatches shown above. No output = all match.)", logfile)

    # ── Table 2 ──────────────────────────────────────────────────────────
    if args.section in ("2", "all"):
        log("", logfile)
        log("#" * 80, logfile)
        log("# TABLE 2 (§5.2) — 560 instances (benedikt2025b_groups, TWOSBY)", logfile)
        log("#" * 80, logfile)
        results = run_paper_datasets(
            "table2",
            ["benedikt2025b_groups"],
            PAPER_DATASETS,
            solver_path,
            args.time_limit,
            args.batch_size,
            logfile,
        )
        all_results.extend(results)
        write_csv(results, out_dir / "table2_results.csv")
        print_section_summary("TABLE 2", results, logfile)

    # ── Figure 9 ─────────────────────────────────────────────────────────
    if args.section in ("fig9", "all"):
        log("", logfile)
        log("#" * 80, logfile)
        log("# FIGURE 9 (§5.3) — 560 instances (benedikt2025_groups, TWOSBY)", logfile)
        log("#" * 80, logfile)
        results = run_paper_datasets(
            "fig9",
            ["benedikt2025_groups"],
            PAPER_DATASETS,
            solver_path,
            args.time_limit,
            args.batch_size,
            logfile,
        )
        all_results.extend(results)
        write_csv(results, out_dir / "fig9_results.csv")
        print_section_summary("FIGURE 9", results, logfile)

    # ── Drops ablation ───────────────────────────────────────────────────
    if args.section in ("drops", "all"):
        log("", logfile)
        log("#" * 80, logfile)
        log("# DROPS ABLATION — 240 instances (benedikt2025b_drops, TWOSBY)", logfile)
        log("#" * 80, logfile)
        results = run_paper_datasets(
            "drops",
            ["benedikt2025b_drops"],
            PAPER_DATASETS,
            solver_path,
            args.time_limit,
            args.batch_size,
            logfile,
        )
        all_results.extend(results)
        write_csv(results, out_dir / "drops_results.csv")
        print_section_summary("DROPS", results, logfile)

    # ── GCD ──────────────────────────────────────────────────────────────
    if args.section in ("gcd", "all"):
        log("", logfile)
        log("#" * 80, logfile)
        log("# GCD ANALYSIS — 1 instance (benedikt2025b_gcd, NOSBY)", logfile)
        log("#" * 80, logfile)
        results = run_paper_datasets(
            "gcd",
            ["benedikt2025b_gcd"],
            PAPER_DATASETS,
            solver_path,
            args.time_limit,
            args.batch_size,
            logfile,
        )
        all_results.extend(results)
        write_csv(results, out_dir / "gcd_results.csv")
        print_section_summary("GCD", results, logfile)

    # ── Synthetic ────────────────────────────────────────────────────────
    if args.section in ("synthetic", "all"):
        log("", logfile)
        log("#" * 80, logfile)
        log("# HARD SYNTHETIC — 84 instances (stress test)", logfile)
        log("#" * 80, logfile)
        results = run_synthetic(solver_path, args.time_limit, logfile)
        all_results.extend(results)
        write_csv(results, out_dir / "synthetic_results.csv")
        print_section_summary("SYNTHETIC", results, logfile)

    # ── Grand summary ────────────────────────────────────────────────────
    grand_elapsed = time.monotonic() - grand_start
    write_csv(all_results, out_dir / "all_results.csv")

    n = len(all_results)
    n_opt = sum(1 for r in all_results if r["is_optimal"])
    n_feas = sum(1 for r in all_results if r["feasible"])
    n_infeas = n - n_feas
    total_t = sum(r["runtime_sec"] for r in all_results)
    max_t = max((r["runtime_sec"] for r in all_results), default=0)
    avg_t = total_t / max(n, 1)

    log("", logfile)
    log("=" * 80, logfile)
    log("  GRAND SUMMARY", logfile)
    log("=" * 80, logfile)
    log(f"  Total instances:     {n}", logfile)
    log(
        f"  Proven optimal:      {n_opt}/{n_feas} feasible ({100*n_opt/max(n_feas,1):.1f}%)",
        logfile,
    )
    log(f"  Infeasible:          {n_infeas}", logfile)
    log(f"  Open gaps:           {n_feas - n_opt}", logfile)
    log(f"  Total solver time:   {total_t:.2f}s", logfile)
    log(f"  Average solver time: {avg_t:.4f}s", logfile)
    log(f"  Max solver time:     {max_t:.4f}s", logfile)
    log(
        f"  Wall clock time:     {grand_elapsed:.1f}s ({grand_elapsed/60:.1f} min)",
        logfile,
    )
    log("", logfile)

    # Per-section breakdown
    by_section = defaultdict(list)
    for r in all_results:
        by_section[r.get("section", "unknown")].append(r)

    log("  Per-section breakdown:", logfile)
    log(
        f"  {'Section':>15} {'Opt':>8} {'Total':>8} {'Avg(s)':>10} {'Max(s)':>10} {'Sum(s)':>10}",
        logfile,
    )
    for sec in ["table1", "table2", "fig9", "drops", "gcd", "synthetic"]:
        sr = by_section.get(sec, [])
        if not sr:
            continue
        sn = len(sr)
        so = sum(1 for r in sr if r["is_optimal"])
        st = sum(r["runtime_sec"] for r in sr)
        sm = max(r["runtime_sec"] for r in sr)
        sa = st / max(sn, 1)
        log(
            f"  {sec:>15} {so:>5}/{sn:<3} {sn:>8} {sa:>10.4f} {sm:>10.4f} {st:>10.1f}",
            logfile,
        )

    # Open gaps detail
    open_gaps = [r for r in all_results if r["feasible"] and not r["is_optimal"]]
    if open_gaps:
        log("", logfile)
        log(f"  *** {len(open_gaps)} OPEN GAPS ***", logfile)
        for r in open_gaps:
            log(
                f"    {r['instance_id']:>50} ub={r['ub']:.1f} lb={r['lb']:.1f} "
                f"gap={r['gap_pct']:.4f}% t={r['runtime_sec']:.3f}s",
                logfile,
            )
    else:
        log("", logfile)
        log("  *** ZERO OPEN GAPS — ALL FEASIBLE INSTANCES PROVEN OPTIMAL ***", logfile)

    log("", logfile)
    log(f"  Results saved to: {out_dir}/", logfile)
    log(f"  Log file: {logpath}", logfile)
    log("", logfile)
    log("Next step: bash hpc/04_run_paper_solver.sh", logfile)

    logfile.close()


if __name__ == "__main__":
    main()

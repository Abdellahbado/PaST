#!/usr/bin/env python3
"""
04_run_paper_solver.py — Run the paper's C# B&B-SPACES solver on all instances.

The paper's solver uses the Experiments runner which:
  1. Reads an experiments-prescription JSON (specifying solver+config+datasets)
  2. Solves each instance, writing result.json files to a results directory
  3. Each result.json contains: Status, RunningTime, Objective, LowerBound, etc.

We run it on the same instances as our solver, then parse the result JSONs
into a unified CSV for comparison.

Sections:
  Table 1 (§5.1):   72 instances (6 datasets, NOSBY+TWOSBY)
  Table 2 (§5.2):  560 instances (benedikt2025b_groups, TWOSBY)
  Figure 9 (§5.3): 560 instances (benedikt2025_groups, TWOSBY)
  Drops ablation:  240 instances (benedikt2025b_drops, TWOSBY)
  GCD analysis:      1 instance  (benedikt2025b_gcd, NOSBY)

Note: B&B-SPACES uses a pre-built C++ binary (BranchAndBoundJob) that
      needs libgurobi.so + libcplex2212.so at runtime.
      Synthetic instances are only in our solver (paper didn't test on those).

Usage:
  python3 hpc/04_run_paper_solver.py [--section all|table1|table2|fig9|drops|gcd] \\
                                     [--time-limit 600] \\
                                     [--output-dir hpc/results_paper]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER_ROOT = (
    ROOT / "data" / "green-scheduling-bab" / "Iirc.EnergyStatesAndCostsScheduling"
)
PAPER_DATA = PAPER_ROOT / "data"
EXPERIMENTS_PROJECT = PAPER_ROOT / "Iirc.EnergyStatesAndCostsScheduling.Experiments"


def _solver_env() -> dict:
    """Build environment dict for the paper's solver."""
    env = os.environ.copy()
    env["DOTNET_EnableUnsafeBinaryFormatterSerialization"] = "true"
    return env


CSV_HEADER = (
    "section,dataset,instance_id,n_jobs,horizon,ub,lb,gap_pct,"
    "feasible,is_optimal,timed_out,runtime_sec,status,running_time_raw"
)

# ── Table 1 datasets ────────────────────────────────────────────────────
# The paper's repo uses benedikt2025a_* (same instances as our benedikt2020a_*
# in data/paper_instances/datasets — verified byte-identical).
# We use the paper's naming so both solvers produce matching instance_ids.
# The paper uses 3600s for large, 600s for medium/prelim.
TABLE1_DATASETS = [
    # (dataset_name, solver_type) — time limit comes from --time-limit flag
    ("benedikt2025a_large_nosby", "BAB"),
    ("benedikt2025a_large_twosby", "BAB"),
    ("benedikt2025a_medium_nosby", "BAB"),
    ("benedikt2025a_medium_twosby", "BAB"),
    ("benedikt2025a_prelim", "BAB"),
    ("aghelinejad2017a_1", "BAB"),
]


def log(msg: str, logfile=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if logfile:
        logfile.write(line + "\n")
        logfile.flush()


def get_system_info() -> str:
    lines = [
        f"Platform:  {platform.platform()}",
        f"Processor: {platform.processor()}",
    ]
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    lines.append(f"CPU model: {line.split(':')[1].strip()}")
                    break
    except FileNotFoundError:
        try:
            r = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            )
            if r.returncode == 0:
                lines.append(f"CPU model: {r.stdout.strip()}")
        except Exception:
            pass
    try:
        lines.append(f"CPU cores: {os.cpu_count()}")
    except Exception:
        pass
    return "\n".join(lines)


def load_instance_info(json_path: Path) -> dict:
    """Load minimal instance info (n_jobs, horizon) from instance JSON."""
    with open(json_path) as f:
        d = json.load(f)
    return {
        "n_jobs": len(d["Jobs"]),
        "horizon": len(d["EnergyCosts"]),
    }


def parse_timespan(ts_str: str) -> float:
    """Parse C# TimeSpan string like '00:01:23.4567890' to seconds."""
    if not ts_str:
        return 0.0
    # Format: [d.]hh:mm:ss[.fffffff]
    m = re.match(r"(?:(\d+)\.)?(\d+):(\d+):(\d+)(?:\.(\d+))?", ts_str)
    if not m:
        return 0.0
    days = int(m.group(1) or 0)
    hours = int(m.group(2))
    mins = int(m.group(3))
    secs = int(m.group(4))
    frac = float(f"0.{m.group(5)}") if m.group(5) else 0.0
    return days * 86400 + hours * 3600 + mins * 60 + secs + frac


def create_experiment_prescription(
    dataset_name: str,
    time_limit_sec: int,
    solver_type: str = "BAB",
    output_path: Path = None,
) -> Path:
    """Create an experiment prescription JSON for the paper's Experiments runner."""

    if solver_type == "BAB":
        # BranchAndBoundJob — the paper's main solver
        solver_config = {
            "id": "BAB",
            "solverName": "BranchAndBoundJob",
            "substractExtendedInstanceGenerationFromTimeLimit": True,
            "specializedSolverConfig": {
                "usePrimalHeuristicBlockDetection": True,
                "usePrimalHeuristicPackToBlocksByCp": False,
                "primalHeuristicPackToBlocksByCpAllJobs": False,
                "primalHeuristicBlockFinding": "Off",
                "jobsJoiningOnGcd": "WholeTree",
            },
        }
    elif solver_type == "BAB_fig9":
        # BAB with default config for Figure 9 instances
        solver_config = {
            "id": "BAB_default",
            "solverName": "BranchAndBoundJob",
            "substractExtendedInstanceGenerationFromTimeLimit": True,
            "specializedSolverConfig": {
                "usePrimalHeuristicBlockDetection": True,
                "usePrimalHeuristicPackToBlocksByCp": False,
                "primalHeuristicPackToBlocksByCpAllJobs": False,
                "primalHeuristicBlockFinding": "Off",
                "UseIterativeDeepening": False,
                "BranchPriority": "ForcedSpace",
                "jobsJoiningOnGcd": "WholeTree",
            },
        }
    else:
        raise ValueError(f"Unknown solver_type: {solver_type}")

    # Convert seconds to TimeSpan string HH:MM:SS
    hours = time_limit_sec // 3600
    mins = (time_limit_sec % 3600) // 60
    secs = time_limit_sec % 60
    time_str = f"{hours:02d}:{mins:02d}:{secs:02d}"

    prescription = {
        "datasetNames": [dataset_name],
        "globalConfig": {
            "randomSeed": 41,
            "numWorkers": 1,
            "timeLimit": time_str,
        },
        "solvers": [solver_config],
    }

    if output_path is None:
        output_path = Path(f"/tmp/hpc_prescription_{dataset_name}_{solver_type}.json")

    with open(output_path, "w") as f:
        json.dump(prescription, f, indent=2)

    return output_path


def run_paper_experiments(
    dataset_name: str,
    section_name: str,
    solver_type: str,
    time_limit_sec: int,
    results_base_dir: Path,
    logfile=None,
) -> list[dict]:
    """
    Run the paper's Experiments runner on a dataset, ONE INSTANCE AT A TIME.

    Per-instance execution isolates C++ crashes: if the solver segfaults or
    throws on one instance, only that instance is lost — the rest continue.
    Each instance runs in its own dotnet process via a temporary single-instance
    dataset directory.
    """
    solver_id = "BAB" if solver_type == "BAB" else "BAB_default"
    instance_dir = PAPER_DATA / "datasets" / dataset_name

    # Discover all instance files
    instance_files = sorted(
        [f for f in instance_dir.glob("*.json") if not f.name.startswith("._")],
        key=lambda f: int(f.stem),
    )

    log(
        f"  Running paper solver on {dataset_name} "
        f"({solver_type}, {time_limit_sec}s limit) "
        f"-- {len(instance_files)} instances, per-instance mode",
        logfile,
    )

    # Temp single-instance dataset dir (inside paper's data/datasets/ so the
    # Experiments runner can discover it)
    TEMP_DS = "_hpc_single"
    temp_ds_dir = PAPER_DATA / "datasets" / TEMP_DS

    # The Experiments runner writes results to:
    #   {DATA_ROOT}/results/{prescription_stem}/{datasetName}/{solverId}/
    # Compute the prescription once (same file for all instances in this batch).
    presc_path = create_experiment_prescription(TEMP_DS, time_limit_sec, solver_type)
    presc_dest = PAPER_DATA / "experiments-prescriptions" / presc_path.name
    presc_stem = presc_path.stem   # e.g. "hpc_prescription__hpc_single_BAB"
    temp_results_base = PAPER_DATA / "results" / presc_stem / TEMP_DS

    # Ensure experiments-prescriptions dir exists and install the prescription
    presc_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(presc_path, presc_dest)

    all_results: list[dict] = []

    for inst_file in instance_files:
        idx = int(inst_file.stem)
        inst_info = load_instance_info(inst_file)

        # ── Set up temp single-instance dataset ──────────────────────────
        if temp_ds_dir.exists():
            shutil.rmtree(temp_ds_dir)
        temp_ds_dir.mkdir(parents=True)
        shutil.copy2(inst_file, temp_ds_dir / inst_file.name)

        # Clean stale results for this temp dataset
        if temp_results_base.exists():
            shutil.rmtree(temp_results_base)

        cmd = [
            "dotnet", "run",
            "--project", str(EXPERIMENTS_PROJECT),
            "-c", "Release", "--no-build", "--",
            str(PAPER_DATA), presc_path.name,
        ]

        # ── Run solver on single instance ────────────────────────────────
        t0 = time.monotonic()
        proc_ok = True
        stderr_msg = ""

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=time_limit_sec + 120,   # generous wall-clock padding
                cwd=str(PAPER_ROOT),
                env=_solver_env(),
            )
            if proc.returncode != 0:
                proc_ok = False
                stderr_msg = (proc.stderr or "")[:300].strip()
        except subprocess.TimeoutExpired:
            proc_ok = False
            stderr_msg = "wall-clock timeout"

        elapsed = time.monotonic() - t0

        # ── Parse result (may exist even if process crashed) ─────────────
        result_json = temp_results_base / solver_id / f"{idx}.json"
        parsed = False

        if result_json.exists():
            try:
                with open(result_json) as f:
                    rd = json.load(f)

                status = rd.get("Status", "Unknown")
                rt_str = rd.get("RunningTime", "00:00:00")
                runtime = parse_timespan(str(rt_str))
                obj = rd.get("Objective")
                lb = rd.get("LowerBound")
                tlr = rd.get("TimeLimitReached", False)

                ub_val = float(obj) if obj is not None else -1.0
                lb_val = float(lb) if lb is not None else -1.0
                feasible = 1 if status in ("Optimal", "Heuristic") else 0
                is_optimal = 1 if status == "Optimal" else 0
                timed_out = 1 if tlr else 0
                gap_pct = 0.0
                if feasible and lb_val > 0 and ub_val > 0:
                    gap_pct = 100.0 * (ub_val - lb_val) / lb_val

                all_results.append({
                    "section": section_name,
                    "dataset": dataset_name,
                    "instance_id": f"{dataset_name}/{idx}",
                    "n_jobs": inst_info["n_jobs"],
                    "horizon": inst_info["horizon"],
                    "ub": ub_val,
                    "lb": lb_val,
                    "gap_pct": gap_pct,
                    "feasible": feasible,
                    "is_optimal": is_optimal,
                    "timed_out": timed_out,
                    "runtime_sec": runtime,
                    "status": status,
                    "running_time_raw": str(rt_str),
                })
                parsed = True

                sym = "OK" if is_optimal else ("TL" if timed_out else "??")
                log(
                    f"    {idx}: {sym} {status} {runtime:.1f}s "
                    f"(ub={ub_val:.0f} lb={lb_val:.0f})",
                    logfile,
                )
            except Exception as e:
                log(f"    {idx}: result parse error -- {e}", logfile)

        if not parsed:
            # Record as crashed / no result
            all_results.append({
                "section": section_name,
                "dataset": dataset_name,
                "instance_id": f"{dataset_name}/{idx}",
                "n_jobs": inst_info["n_jobs"],
                "horizon": inst_info["horizon"],
                "ub": -1.0,
                "lb": -1.0,
                "gap_pct": 0.0,
                "feasible": 0,
                "is_optimal": 0,
                "timed_out": 0,
                "runtime_sec": elapsed,
                "status": "Crashed",
                "running_time_raw": "",
            })
            # Log both stdout and stderr for diagnostics
            out_snippet = ""
            if "proc" in dir() and hasattr(proc, "stdout") and proc.stdout:
                out_snippet = proc.stdout.strip().split("\n")[-1][:120]
            short_err = (
                stderr_msg.split("\n")[0][:100] if stderr_msg
                else (out_snippet or f"result not found at {result_json}")
            )
            log(f"    {idx}: CRASHED {elapsed:.1f}s -- {short_err}", logfile)

    # Final cleanup of temp dirs
    if temp_ds_dir.exists():
        shutil.rmtree(temp_ds_dir, ignore_errors=True)
    # Remove the entire prescription-scoped results subtree
    presc_results_dir = PAPER_DATA / "results" / presc_stem
    if presc_results_dir.exists():
        shutil.rmtree(presc_results_dir, ignore_errors=True)
    # Clean up installed prescription
    presc_dest.unlink(missing_ok=True)

    # ── Summary ──────────────────────────────────────────────────────────
    n = len(all_results)
    n_opt = sum(1 for r in all_results if r["is_optimal"])
    n_crash = sum(1 for r in all_results if r["status"] == "Crashed")
    n_tlr = sum(1 for r in all_results if r["timed_out"])
    total_t = sum(r["runtime_sec"] for r in all_results)
    max_t = max((r["runtime_sec"] for r in all_results), default=0)
    log(
        f"  => {n_opt}/{n} optimal, {n_crash} crashed, "
        f"{n_tlr} timeouts, total={total_t:.1f}s",
        logfile,
    )
    return all_results


def write_csv(results: list[dict], path: Path):
    with open(path, "w") as f:
        f.write(CSV_HEADER + "\n")
        for r in results:
            f.write(
                f"{r.get('section','')},{r.get('dataset','')},{r['instance_id']},"
                f"{r['n_jobs']},{r['horizon']},"
                f"{r['ub']:.6f},{r['lb']:.6f},{r['gap_pct']:.4f},"
                f"{r['feasible']},{r['is_optimal']},{r['timed_out']},"
                f"{r['runtime_sec']:.4f},{r.get('status','')},{r.get('running_time_raw','')}\n"
            )


def warmup_dotnet(logfile=None):
    """Run a single trivial instance to warm up .NET JIT before timing."""
    log("--- Warm-up: running one trivial instance to prime .NET JIT ---", logfile)

    warmup_ds = "aghelinejad2017a_1"
    warmup_inst = PAPER_DATA / "datasets" / warmup_ds / "0.json"
    if not warmup_inst.exists():
        for ds in sorted(PAPER_DATA.glob("datasets/*/0.json")):
            warmup_ds = ds.parent.name
            warmup_inst = ds
            break

    if not warmup_inst.exists():
        log("  Warm-up skipped: no instance found", logfile)
        return

    presc_path = create_experiment_prescription(warmup_ds, 10, "BAB")
    target_presc = PAPER_DATA / "experiments-prescriptions" / presc_path.name
    shutil.copy2(presc_path, target_presc)

    cmd = [
        "dotnet",
        "run",
        "--project",
        str(EXPERIMENTS_PROJECT),
        "-c",
        "Release",
        "--no-build",
        "--",
        str(PAPER_DATA),
        presc_path.name,
    ]

    t0 = time.monotonic()
    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PAPER_ROOT),
            env=_solver_env(),
        )
    except subprocess.TimeoutExpired:
        pass
    t1 = time.monotonic()

    target_presc.unlink(missing_ok=True)

    # Clean up warm-up results
    warmup_results = PAPER_DATA / "results" / warmup_ds / "BAB"
    if warmup_results.exists():
        shutil.rmtree(warmup_results, ignore_errors=True)

    log(f"  Warm-up done in {t1 - t0:.1f}s (JIT compiled, caches hot)", logfile)


def main():
    ap = argparse.ArgumentParser(
        description="Run paper's B&B-SPACES solver on all instances"
    )
    ap.add_argument(
        "--section",
        default="all",
        choices=["all", "table1", "1", "table2", "2", "fig9", "drops", "gcd"],
    )
    ap.add_argument(
        "--time-limit",
        type=int,
        default=600,
        help="Time limit per instance in seconds (default: 600). "
        "Applied uniformly to all sections for fair comparison.",
    )
    ap.add_argument("--output-dir", default=str(ROOT / "hpc" / "results_paper"))
    ap.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip the .NET JIT warm-up run",
    )
    args = ap.parse_args()

    # Normalize section aliases
    section = args.section
    if section == "1":
        section = "table1"
    elif section == "2":
        section = "table2"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logpath = out_dir / "run_log.txt"
    logfile = open(logpath, "w")

    log("=" * 80, logfile)
    log(" PaST HPC Benchmark — Paper's Solver (B&B-SPACES)", logfile)
    log(f" {time.strftime('%Y-%m-%d %H:%M:%S')}", logfile)
    log(f" Time limit: {args.time_limit}s per instance (default)", logfile)
    log(f" Method: Experiments runner only (pure B&B, no Gurobi/SolverCli)", logfile)
    log("=" * 80, logfile)
    log("", logfile)
    log("--- System Info ---", logfile)
    log(get_system_info(), logfile)
    log("", logfile)

    # Check dotnet
    try:
        r = subprocess.run(["dotnet", "--version"], capture_output=True, text=True)
        log(f".NET SDK: {r.stdout.strip()}", logfile)
    except FileNotFoundError:
        log("ERROR: dotnet not found. Run: bash hpc/00_install_deps.sh", logfile)
        sys.exit(1)

    # Warm up .NET JIT to eliminate cold-start effects
    if not args.skip_warmup:
        warmup_dotnet(logfile)
    log("", logfile)

    grand_start = time.monotonic()
    all_results = []

    # ── Table 1 (§5.1) ──────────────────────────────────────────────────
    if section in ("table1", "all"):
        log(f"\n{'#'*80}", logfile)
        log(f"# TABLE 1 (§5.1) — 72 instances (NOSBY + TWOSBY)", logfile)
        log(f"# Time limit: {args.time_limit}s per instance", logfile)
        log(f"{'#'*80}", logfile)

        for ds_name, solver_type in TABLE1_DATASETS:
            results = run_paper_experiments(
                ds_name, "table1", solver_type, args.time_limit, out_dir, logfile
            )
            all_results.extend(results)

        table1_results = [r for r in all_results if r["section"] == "table1"]
        write_csv(table1_results, out_dir / "table1_results.csv")
        n = len(table1_results)
        n_opt = sum(1 for r in table1_results if r["is_optimal"])
        log(f"  Table 1 total: {n_opt}/{n} optimal", logfile)

    # ── Table 2, Figure 9, Drops, GCD ────────────────────────────────────
    sections_map = {
        "table2": [("benedikt2025b_groups", "BAB")],
        "fig9": [("benedikt2025_groups", "BAB_fig9")],
        "drops": [("benedikt2025b_drops", "BAB")],
        "gcd": [("benedikt2025b_gcd", "BAB")],
    }

    for sec_name, datasets in sections_map.items():
        if section not in (sec_name, "all"):
            continue

        log(f"\n{'#'*80}", logfile)
        log(f"# {sec_name.upper()}", logfile)
        log(f"{'#'*80}", logfile)

        for ds_name, solver_type in datasets:
            results = run_paper_experiments(
                ds_name, sec_name, solver_type, args.time_limit, out_dir, logfile
            )
            all_results.extend(results)
            write_csv(results, out_dir / f"{sec_name}_results.csv")

    # Grand summary
    grand_elapsed = time.monotonic() - grand_start
    write_csv(all_results, out_dir / "all_results.csv")

    n = len(all_results)
    n_opt = sum(1 for r in all_results if r["is_optimal"])
    n_tlr = sum(1 for r in all_results if r["timed_out"])
    total_t = sum(r["runtime_sec"] for r in all_results)
    max_t = max((r["runtime_sec"] for r in all_results), default=0)

    log("", logfile)
    log("=" * 80, logfile)
    log("  GRAND SUMMARY — Paper's Solver", logfile)
    log("=" * 80, logfile)
    log(f"  Total instances:     {n}", logfile)
    log(f"  Proven optimal:      {n_opt}/{n} ({100*n_opt/max(n,1):.1f}%)", logfile)
    log(f"  Time limit reached:  {n_tlr}", logfile)
    log(f"  Total solver time:   {total_t:.2f}s", logfile)
    log(f"  Max solver time:     {max_t:.2f}s", logfile)
    log(
        f"  Wall clock time:     {grand_elapsed:.1f}s ({grand_elapsed/60:.1f} min)",
        logfile,
    )
    log("", logfile)
    log(f"  Results saved to: {out_dir}/", logfile)
    log(f"  Log file: {logpath}", logfile)
    log("", logfile)
    log("Next step: python3 hpc/05_analyze_results.py", logfile)

    logfile.close()


if __name__ == "__main__":
    main()

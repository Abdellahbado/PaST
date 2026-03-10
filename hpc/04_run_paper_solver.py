#!/usr/bin/env python3
"""
04_run_paper_solver.py — Run the paper's C# B&B-SPACES solver on all instances.

The paper's solver uses the Experiments runner which:
  1. Reads an experiments-prescription JSON (specifying solver+config+datasets)
  2. Solves each instance, writing result.json files to a results directory
  3. Each result.json contains: Status, RunningTime, Objective, LowerBound, etc.

We run it on the same instances as our solver, then parse the result JSONs
into a unified CSV for comparison.

Usage:
  python3 hpc/04_run_paper_solver.py [--section all|table2|fig9] \\
                                     [--time-limit 600] \\
                                     [--output-dir hpc/results_paper]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
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
SOLVER_CLI_PROJECT = PAPER_ROOT / "Iirc.EnergyStatesAndCostsScheduling.SolverCli"
EXPERIMENTS_PROJECT = PAPER_ROOT / "Iirc.EnergyStatesAndCostsScheduling.Experiments"

CSV_HEADER = (
    "section,dataset,instance_id,n_jobs,horizon,ub,lb,gap_pct,"
    "feasible,is_optimal,timed_out,runtime_sec,status,running_time_raw"
)


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
                "usePrimalHeuristicPackToBlocksByCp": True,
                "primalHeuristicPackToBlocksByCpAllJobs": True,
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
                "UseIterativeDeepening": False,
                "PrimalHeuristicBlockFinding": True,
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
            "numWorkers": 0,
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
    Run the paper's Experiments runner on a dataset.

    The runner writes result.json files per instance.
    We then parse those into our unified CSV format.
    """
    # Create prescription
    presc_path = create_experiment_prescription(
        dataset_name, time_limit_sec, solver_type
    )
    solver_id = "BAB" if solver_type == "BAB" else "BAB_default"

    log(
        f"  Running paper solver on {dataset_name} ({solver_type}, {time_limit_sec}s limit)",
        logfile,
    )

    # Run the Experiments project
    wall_start = time.monotonic()

    cmd = [
        "dotnet",
        "run",
        "--project",
        str(EXPERIMENTS_PROJECT),
        "-c",
        "Release",
        "--",
        str(PAPER_DATA),
        presc_path.name,
    ]

    # Copy prescription to the experiments-prescriptions dir so it can be found
    import shutil

    target_presc = PAPER_DATA / "experiments-prescriptions" / presc_path.name
    shutil.copy2(presc_path, target_presc)

    log(
        f"  Command: dotnet run --project ...Experiments -c Release -- {PAPER_DATA} {presc_path.name}",
        logfile,
    )

    try:
        overall_timeout = max(time_limit_sec * 700, 86400)  # generous timeout
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=overall_timeout,
            cwd=str(PAPER_ROOT),
        )
        wall_elapsed = time.monotonic() - wall_start
        log(f"  Wall time: {wall_elapsed:.1f}s", logfile)

        if proc.stdout:
            # Print last 20 lines of stdout
            stdout_lines = proc.stdout.strip().split("\n")
            for line in stdout_lines[-20:]:
                log(f"    [C#] {line}", logfile)

        if proc.returncode != 0 and proc.stderr:
            log(f"  Solver stderr: {proc.stderr[:500]}", logfile)
    except subprocess.TimeoutExpired:
        wall_elapsed = time.monotonic() - wall_start
        log(f"  TIMEOUT after {wall_elapsed:.1f}s", logfile)

    # Clean up temp prescription
    target_presc.unlink(missing_ok=True)

    # Parse results from the results directory
    results_dir = PAPER_DATA / "results" / dataset_name / solver_id
    if not results_dir.exists():
        # Some C# runners put results in a different location
        results_dir = PAPER_DATA / "datasets" / dataset_name
        # Actually the Experiments runner creates: data/results/{dataset}/{solver_id}/
        # Let's check
        alt_results = PAPER_DATA / "results"
        if alt_results.exists():
            log(f"  Results dir structure: {list(alt_results.iterdir())}", logfile)

    instance_dir = PAPER_DATA / "datasets" / dataset_name
    all_results = []

    if results_dir.exists():
        for result_file in sorted(results_dir.glob("*.json")):
            idx = int(result_file.stem)
            try:
                with open(result_file) as f:
                    rd = json.load(f)

                # Load instance info
                inst_path = instance_dir / f"{idx}.json"
                inst_info = (
                    load_instance_info(inst_path)
                    if inst_path.exists()
                    else {"n_jobs": 0, "horizon": 0}
                )

                status = rd.get("Status", "Unknown")
                rt_str = rd.get("RunningTime", "00:00:00")
                runtime = parse_timespan(str(rt_str))
                obj = rd.get("Objective")
                lb = rd.get("LowerBound")
                tlr = rd.get("TimeLimitReached", False)

                ub = float(obj) if obj is not None else -1.0
                lb_val = float(lb) if lb is not None else -1.0
                feasible = 1 if status in ("Optimal", "Heuristic") else 0
                is_optimal = 1 if status == "Optimal" else 0
                timed_out = 1 if tlr else 0
                gap_pct = 0.0
                if feasible and lb_val > 0 and ub > 0:
                    gap_pct = 100.0 * (ub - lb_val) / lb_val

                all_results.append(
                    {
                        "section": section_name,
                        "dataset": dataset_name,
                        "instance_id": f"{dataset_name}/{idx}",
                        "n_jobs": inst_info["n_jobs"],
                        "horizon": inst_info["horizon"],
                        "ub": ub,
                        "lb": lb_val,
                        "gap_pct": gap_pct,
                        "feasible": feasible,
                        "is_optimal": is_optimal,
                        "timed_out": timed_out,
                        "runtime_sec": runtime,
                        "status": status,
                        "running_time_raw": str(rt_str),
                    }
                )
            except Exception as e:
                log(f"  Error parsing {result_file}: {e}", logfile)

        n = len(all_results)
        n_opt = sum(1 for r in all_results if r["is_optimal"])
        n_tlr = sum(1 for r in all_results if r["timed_out"])
        total_t = sum(r["runtime_sec"] for r in all_results)
        max_t = max((r["runtime_sec"] for r in all_results), default=0)
        log(
            f"  => {n_opt}/{n} optimal, {n_tlr} timeouts, max={max_t:.1f}s, total={total_t:.1f}s",
            logfile,
        )
    else:
        log(f"  WARNING: Results dir not found: {results_dir}", logfile)
        log(f"  The paper's Experiments runner may write results elsewhere.", logfile)
        log(f"  Check: {PAPER_DATA / 'results'}", logfile)

    return all_results


def run_paper_single_instance(
    instance_path: Path,
    config_path: Path,
    section_name: str,
    dataset_name: str,
    logfile=None,
) -> dict | None:
    """Run paper's SolverCli on a single instance (fallback approach)."""
    idx = int(instance_path.stem)
    inst_info = load_instance_info(instance_path)

    wall_start = time.monotonic()
    try:
        proc = subprocess.run(
            [
                "dotnet",
                "run",
                "--project",
                str(SOLVER_CLI_PROJECT),
                "-c",
                "Release",
                "--",
                str(config_path),
                str(instance_path),
            ],
            capture_output=True,
            text=True,
            timeout=900,
            cwd=str(PAPER_ROOT),
        )
        wall_elapsed = time.monotonic() - wall_start
    except subprocess.TimeoutExpired:
        wall_elapsed = time.monotonic() - wall_start
        return {
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
            "timed_out": 1,
            "runtime_sec": wall_elapsed,
            "status": "Timeout",
            "running_time_raw": f"{wall_elapsed:.3f}s",
        }

    # Parse stdout
    runtime = wall_elapsed
    status = "Unknown"
    ub = -1.0
    lb = -1.0
    for line in proc.stdout.split("\n"):
        if line.startswith("Running time:"):
            rt_str = line.split(":", 1)[1].strip()
            runtime = parse_timespan(rt_str)
        elif line.startswith("Status:"):
            status = line.split(":", 1)[1].strip()
        elif line.startswith("TEC from solver:"):
            ub = float(line.split(":", 1)[1].strip())
        elif line.startswith("Lower bound:"):
            lb = float(line.split(":", 1)[1].strip())

    feasible = 1 if status in ("Optimal", "Heuristic") else 0
    is_optimal = 1 if status == "Optimal" else 0
    gap_pct = 0.0
    if feasible and lb > 0 and ub > 0:
        gap_pct = 100.0 * (ub - lb) / lb

    return {
        "section": section_name,
        "dataset": dataset_name,
        "instance_id": f"{dataset_name}/{idx}",
        "n_jobs": inst_info["n_jobs"],
        "horizon": inst_info["horizon"],
        "ub": ub,
        "lb": lb,
        "gap_pct": gap_pct,
        "feasible": feasible,
        "is_optimal": is_optimal,
        "timed_out": 0,
        "runtime_sec": runtime,
        "status": status,
        "running_time_raw": f"{runtime:.6f}s",
    }


def run_single_instance_batch(
    dataset_name: str,
    section_name: str,
    time_limit_sec: int,
    solver_type: str,
    logfile=None,
) -> list[dict]:
    """
    Fallback: solve each instance individually via SolverCli.
    Use this if the Experiments runner doesn't work.
    """
    instance_dir = PAPER_DATA / "datasets" / dataset_name
    if not instance_dir.exists():
        log(f"  Dataset dir not found: {instance_dir}", logfile)
        return []

    # Create solver config
    hours = time_limit_sec // 3600
    mins = (time_limit_sec % 3600) // 60
    secs = time_limit_sec % 60
    time_str = f"{hours:02d}:{mins:02d}:{secs:02d}"

    if solver_type in ("BAB", "BAB_fig9"):
        config = {
            "randomSeed": 41,
            "timeLimit": time_str,
            "numWorkers": 0,
            "useSerializedExtendedInstance": False,
            "solverName": "BranchAndBoundJob",
            "specializedSolverConfig": {
                "usePrimalHeuristicBlockDetection": True,
                "usePrimalHeuristicPackToBlocksByCp": solver_type == "BAB",
                "primalHeuristicPackToBlocksByCpAllJobs": solver_type == "BAB",
                "jobsJoiningOnGcd": "WholeTree",
            },
        }
    else:
        raise ValueError(f"Unknown solver_type: {solver_type}")

    config_path = Path(f"/tmp/hpc_config_{dataset_name}.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    json_files = sorted(instance_dir.glob("*.json"), key=lambda p: int(p.stem))
    n = len(json_files)
    log(f"  Running SolverCli on {n} instances from {dataset_name}", logfile)

    results = []
    for i, jf in enumerate(json_files):
        r = run_paper_single_instance(
            jf, config_path, section_name, dataset_name, logfile
        )
        if r:
            results.append(r)
        if (i + 1) % 20 == 0 or (i + 1) == n:
            log(f"    Progress: {i+1}/{n}", logfile)

    config_path.unlink(missing_ok=True)
    return results


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


def main():
    ap = argparse.ArgumentParser(description="Run paper's solver on all instances")
    ap.add_argument(
        "--section", default="all", choices=["all", "table2", "fig9", "drops", "gcd"]
    )
    ap.add_argument(
        "--time-limit",
        type=int,
        default=600,
        help="Time limit per instance in seconds (default: 600)",
    )
    ap.add_argument("--output-dir", default=str(ROOT / "hpc" / "results_paper"))
    ap.add_argument(
        "--method",
        default="experiments",
        choices=["experiments", "single"],
        help="'experiments' uses the batch runner, 'single' solves one-by-one",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logpath = out_dir / "run_log.txt"
    logfile = open(logpath, "w")

    log("=" * 80, logfile)
    log(" PaST HPC Benchmark — Paper's Solver (B&B-SPACES)", logfile)
    log(f" {time.strftime('%Y-%m-%d %H:%M:%S')}", logfile)
    log(f" Time limit: {args.time_limit}s per instance", logfile)
    log(f" Method: {args.method}", logfile)
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
        log("ERROR: dotnet not found. Run: bash hpc/02_build_paper_solver.sh", logfile)
        sys.exit(1)

    grand_start = time.monotonic()
    all_results = []

    # Datasets to run
    sections = {
        "table2": [("benedikt2025b_groups", "BAB")],
        "fig9": [("benedikt2025_groups", "BAB_fig9")],
        "drops": [("benedikt2025b_drops", "BAB")],
        "gcd": [("benedikt2025b_gcd", "BAB")],
    }

    for sec_name, datasets in sections.items():
        if args.section not in (sec_name, "all"):
            continue

        log(f"\n{'#'*80}", logfile)
        log(f"# {sec_name.upper()}", logfile)
        log(f"{'#'*80}", logfile)

        for ds_name, solver_type in datasets:
            if args.method == "experiments":
                results = run_paper_experiments(
                    ds_name, sec_name, solver_type, args.time_limit, out_dir, logfile
                )
            else:
                results = run_single_instance_batch(
                    ds_name, sec_name, args.time_limit, solver_type, logfile
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

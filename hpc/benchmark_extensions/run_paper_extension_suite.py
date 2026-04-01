#!/usr/bin/env python3
"""
Run the paper's solver on one formal benchmark-extension suite.

This is a generic custom-dataset runner so the same formal suites can be used
for both our solver and the paper solver without changing the benchmark
definition.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = ROOT / "data" / "green-scheduling-bab" / "Iirc.EnergyStatesAndCostsScheduling"
PAPER_DATA = PAPER_ROOT / "data"
EXPERIMENTS_PROJECT = PAPER_ROOT / "Iirc.EnergyStatesAndCostsScheduling.Experiments"

SUITE_DATASETS = {
    "scalability_large_n": "paperext_scalability_large_n_202604",
    "backup_realistic": "paperext_backup_realistic_202604",
    "k_boundary": "paperext_k_boundary_202604",
}


def parse_timespan(ts_str: str) -> float:
    if not ts_str:
        return 0.0
    m = re.match(r"(?:(\d+)\.)?(\d+):(\d+):(\d+)(?:\.(\d+))?", ts_str)
    if not m:
        return 0.0
    days = int(m.group(1) or 0)
    hours = int(m.group(2))
    mins = int(m.group(3))
    secs = int(m.group(4))
    frac = float(f"0.{m.group(5)}") if m.group(5) else 0.0
    return days * 86400 + hours * 3600 + mins * 60 + secs + frac


def _solver_env() -> dict:
    env = os.environ.copy()
    env["DOTNET_EnableUnsafeBinaryFormatterSerialization"] = "true"
    return env


def create_prescription(dataset_name: str, time_limit_sec: int, out_path: Path) -> None:
    hours = time_limit_sec // 3600
    mins = (time_limit_sec % 3600) // 60
    secs = time_limit_sec % 60
    time_str = f"{hours:02d}:{mins:02d}:{secs:02d}"
    prescription = {
        "datasetNames": [dataset_name],
        "globalConfig": {"randomSeed": 41, "numWorkers": 1, "timeLimit": time_str},
        "solvers": [
            {
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
        ],
    }
    with open(out_path, "w") as f:
        json.dump(prescription, f, indent=2)


def load_instance_info(json_path: Path) -> tuple[int, int]:
    with open(json_path) as f:
        d = json.load(f)
    return len(d["Jobs"]), len(d["EnergyCosts"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the paper solver on a formal extension suite")
    ap.add_argument("--suite", choices=sorted(SUITE_DATASETS.keys()), required=True)
    ap.add_argument("--time-limit", type=int, default=600)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--skip-warmup", action="store_true")
    args = ap.parse_args()

    dataset_name = SUITE_DATASETS[args.suite]
    dataset_dir = PAPER_DATA / "datasets" / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    files = sorted(jf for jf in dataset_dir.glob("*.json") if jf.name != "manifest.json")
    temp_ds = "_paperext_single"
    temp_ds_dir = PAPER_DATA / "datasets" / temp_ds
    presc_local = Path(f"/tmp/paperext_{dataset_name}.json")
    create_prescription(temp_ds, args.time_limit, presc_local)
    presc_dest = PAPER_DATA / "experiments-prescriptions" / presc_local.name
    presc_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(presc_local, presc_dest)
    presc_stem = presc_local.stem
    temp_results_base = PAPER_DATA / "results" / presc_stem / temp_ds / "BAB"

    out_rows = []
    for jf in files:
        n_jobs, horizon = load_instance_info(jf)
        if temp_ds_dir.exists():
            shutil.rmtree(temp_ds_dir)
        temp_ds_dir.mkdir(parents=True)
        shutil.copy2(jf, temp_ds_dir / jf.name)
        if temp_results_base.parent.exists():
            shutil.rmtree(temp_results_base.parent)

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
            presc_local.name,
        ]

        t0 = time.monotonic()
        status = "Crashed"
        runtime_sec = -1.0
        ub = -1.0
        lb = -1.0
        feasible = 0
        is_optimal = 0
        timed_out = 0
        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=args.time_limit + 120,
                cwd=str(PAPER_ROOT),
                env=_solver_env(),
                check=False,
            )
        except subprocess.TimeoutExpired:
            timed_out = 1
            runtime_sec = float(args.time_limit + 120)
            status = "WallTimeout"
        elapsed = time.monotonic() - t0

        result_json = temp_results_base / jf.name
        if result_json.exists():
            with open(result_json) as f:
                rd = json.load(f)
            status = rd.get("Status", "Unknown")
            runtime_sec = parse_timespan(str(rd.get("RunningTime", "")))
            ub = float(rd["Objective"]) if rd.get("Objective") is not None else -1.0
            lb = float(rd["LowerBound"]) if rd.get("LowerBound") is not None else -1.0
            feasible = 1 if status in ("Optimal", "Heuristic") else 0
            is_optimal = 1 if status == "Optimal" else 0
            timed_out = 1 if rd.get("TimeLimitReached", False) else timed_out
        elif runtime_sec < 0:
            runtime_sec = elapsed

        gap_pct = 0.0
        if feasible and ub > 0 and lb > 0:
            gap_pct = 100.0 * (ub - lb) / lb

        out_rows.append(
            {
                "suite": args.suite,
                "dataset": dataset_name,
                "instance_id": f"{dataset_name}/{jf.stem}",
                "n_jobs": n_jobs,
                "horizon": horizon,
                "ub": ub,
                "lb": lb,
                "gap_pct": gap_pct,
                "feasible": feasible,
                "is_optimal": is_optimal,
                "timed_out": timed_out,
                "runtime_sec": runtime_sec,
                "status": status,
            }
        )
        print(f"{jf.stem}: {status} {runtime_sec:.2f}s", flush=True)

    if temp_ds_dir.exists():
        shutil.rmtree(temp_ds_dir, ignore_errors=True)
    if (PAPER_DATA / "results" / presc_stem).exists():
        shutil.rmtree(PAPER_DATA / "results" / presc_stem, ignore_errors=True)
    presc_dest.unlink(missing_ok=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        headers = list(out_rows[0].keys()) if out_rows else [
            "suite","dataset","instance_id","n_jobs","horizon","ub","lb","gap_pct",
            "feasible","is_optimal","timed_out","runtime_sec","status"
        ]
        f.write(",".join(headers) + "\n")
        for row in out_rows:
            f.write(",".join(str(row[h]) for h in headers) + "\n")
    print(f"CSV: {args.output}")


if __name__ == "__main__":
    main()

"""Parallel C++ benchmark runner for arXiv:2506.10405 Section 5.1 instances.

Pipeline
--------
1. Load (or generate) the instance dataset from a JSON file produced by
   run_stateful_paper_benchmark.py --generate-only.
2. Spawn up to --workers C++ solver processes in parallel.
3. Each worker handles a disjoint chunk of instances sequentially (no shared
   memory, no coordination overhead).
4. Merge results into a single CSV.

Memory model
------------
Each C++ process holds exactly ONE instance in memory at a time:
  - SPACES arrays:      O(h²) doubles  ← the dominant cost for large h
  - DP state map:       O(reachable_states * 64 bytes)
For n=190, lambda=2.2, h ≈ 190*2.2*3 ≈ 1254 intervals:
  c_star alone = 1254² * 8 bytes ≈ 12 MB per process.
  DP states are sparse; in practice < 100 MB per process.
  With 8 workers: < 1 GB total — safe on any modern machine.

Usage examples
--------------
# First generate the dataset:
PYTHONPATH=/path/to/PFE \\
  python -m PaST.cli.experiments.run_stateful_paper_benchmark \\
    --generate-only \\
    --n-values 150,170,190 \\
    --lambda-values 1.3,1.6,1.9,2.2 \\
    --seeds 42 \\
    --dataset-out artifacts/paper_stateful_benchmark.json

# Then run this script:
python cli/experiments/run_cpp_benchmark.py \\
    --dataset   artifacts/paper_stateful_benchmark.json \\
    --solver    solvers/cpp/build/stateful_compare \\
    --out       artifacts/results_cpp.csv \\
    --workers   8 \\
    --time-limit 3600

# Quick smoke test (tiny dataset, 2 workers, 60s limit):
python cli/experiments/run_cpp_benchmark.py \\
    --dataset   artifacts/paper_small_test.json \\
    --solver    solvers/cpp/build/stateful_compare \\
    --out       artifacts/results_cpp_smoke.csv \\
    --workers   2 \\
    --time-limit 60
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# Worker: runs one C++ solve-stdin process for a chunk of instances.
# Called inside a child process by ProcessPoolExecutor.
# ---------------------------------------------------------------------------


def _worker(chunk: List[dict], solver_path: str, time_limit: float) -> List[dict]:
    """Solve a list of instances using a single C++ process via stdin pipe.

    Each instance dict must have keys: instance_id, prices, jobs.
    Returns a list of result dicts.
    """
    cmd = [solver_path, "solve-stdin", str(time_limit)]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Build stdin: one JSON line per instance.
    # Format: {"instance_id":"...","prices":[...],"jobs":[...]}
    lines = []
    for inst in chunk:
        obj = {
            "instance_id": inst["instance_id"],
            "prices": inst["prices"],
            "jobs": inst["processing_times"],
        }
        lines.append(json.dumps(obj, separators=(",", ":")))

    stdin_str = "\n".join(lines) + "\n"
    stdout, stderr = proc.communicate(stdin_str)

    if stderr.strip():
        print(f"[worker stderr] {stderr.strip()[:200]}", file=sys.stderr)

    # Parse CSV output from C++ (skip header line)
    results = []
    reader = csv.DictReader(stdout.splitlines())
    for row in reader:
        results.append(dict(row))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run C++ stateful DP solver on a pre-generated JSON dataset in parallel."
    )
    ap.add_argument(
        "--dataset",
        required=True,
        help="Path to the JSON dataset file (from run_stateful_paper_benchmark --generate-only).",
    )
    ap.add_argument(
        "--solver",
        default="solvers/cpp/build/stateful_compare",
        help="Path to the stateful_compare binary.",
    )
    ap.add_argument(
        "--out",
        default="artifacts/results_cpp.csv",
        help="Output CSV file path.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel C++ processes. Default: os.cpu_count().",
    )
    ap.add_argument(
        "--time-limit",
        type=float,
        default=3600.0,
        help="Time limit in seconds per instance (passed to C++ solver). Default: 3600.",
    )
    ap.add_argument(
        "--max-instances",
        type=int,
        default=0,
        help="Cap the number of instances (0 = no cap). Useful for smoke tests.",
    )
    args = ap.parse_args()

    solver = Path(args.solver).resolve()
    if not solver.exists():
        print(f"ERROR: solver not found at {solver}", file=sys.stderr)
        print(
            "Build it first:\n  cd solvers/cpp/build && cmake --build . -j4",
            file=sys.stderr,
        )
        sys.exit(1)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: dataset not found at {dataset_path}", file=sys.stderr)
        print(
            "Generate it first:\n  python -m PaST.cli.experiments.run_stateful_paper_benchmark --generate-only ...",
            file=sys.stderr,
        )
        sys.exit(1)

    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    instances = payload["instances"]
    if args.max_instances > 0:
        instances = instances[: args.max_instances]

    n_workers = args.workers or os.cpu_count() or 1
    n_total = len(instances)

    print(f"dataset:      {dataset_path}", flush=True)
    print(f"instances:    {n_total}", flush=True)
    print(f"workers:      {n_workers}", flush=True)
    print(f"time_limit:   {args.time_limit}s per instance", flush=True)
    print(f"solver:       {solver}", flush=True)
    print(f"output:       {args.out}", flush=True)
    print(flush=True)

    # Split instances into n_workers chunks.
    # Distribute so that each worker gets a roughly equal mix of sizes
    # (interleave rather than block-split to avoid one worker getting all
    # the hard instances).
    chunks: List[List[dict]] = [[] for _ in range(n_workers)]
    for i, inst in enumerate(instances):
        chunks[i % n_workers].append(inst)

    # Remove empty chunks (when n_instances < n_workers)
    chunks = [c for c in chunks if c]
    actual_workers = len(chunks)

    t_start = time.perf_counter()
    all_results: List[dict] = []

    with ProcessPoolExecutor(max_workers=actual_workers) as pool:
        futures = {
            pool.submit(_worker, chunk, str(solver), args.time_limit): wi
            for wi, chunk in enumerate(chunks)
        }
        done = 0
        for fut in as_completed(futures):
            wi = futures[fut]
            try:
                rows = fut.result()
                all_results.extend(rows)
                done += len(rows)
                elapsed = time.perf_counter() - t_start
                print(
                    f"  worker {wi} done: {len(rows)} instances  "
                    f"[{done}/{n_total} total, {elapsed:.1f}s elapsed]",
                    flush=True,
                )
            except Exception as exc:
                print(f"  worker {wi} FAILED: {exc}", file=sys.stderr)

    # Write output CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "instance_id",
        "n_jobs",
        "horizon",
        "cost",
        "feasible",
        "is_optimal",
        "timed_out",
        "runtime_sec",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    total_time = time.perf_counter() - t_start
    print(f"\nDone. {len(all_results)} results written to {out_path}")
    print(
        f"Total wall time: {total_time:.1f}s  (avg {total_time/max(len(all_results),1):.2f}s/instance)"
    )


if __name__ == "__main__":
    main()

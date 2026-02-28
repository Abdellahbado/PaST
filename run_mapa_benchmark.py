#!/usr/bin/env python3
"""Run MAPA (OLLA) solver on the 90 Wang2018 benchmark instances and save results.

This script sweeps the makespan constraint (epsilon) for each instance to generate
a Pareto front (Makespan vs Total Energy Cost) using the new CWP-seeded MAPA heuristic
and the exact C++ DP solver.

The output is saved as a JSON archive compatible with compare_ehs.py.
"""

import argparse
import json
import logging
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Ensure we can import solvers
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from solvers.olla_solver import solve_olla_pareto


def _read_values(path: Path) -> list:
    with open(path) as f:
        # Some files might have multiple values per line, or one per line
        return [float(x) for x in f.read().split() if x.strip()]


def solve_instance(inst_id: int, data_dir: Path) -> list:
    """Solve a single instance, returning a list of dicts for the JSON archive."""
    p_path = data_dir / f"Data_p{inst_id}.txt"
    e_path = data_dir / f"Data_e{inst_id}.txt"
    c_path = data_dir / f"Data_c{inst_id}.txt"

    if not all(x.exists() for x in [p_path, e_path, c_path]):
        return []

    p = [int(x) for x in _read_values(p_path)]
    e = _read_values(e_path)
    ct = np.array(_read_values(c_path), dtype=np.float64)

    t0 = time.perf_counter()
    # Sweep epsilon to generate the Pareto front
    res = solve_olla_pareto(p, e, ct, local_search=True)
    dt = time.perf_counter() - t0

    archive_entries = []
    for makespan, energy in res.front:
        archive_entries.append({
            "instance_id": inst_id,
            "makespan": int(makespan),
            "energy": float(energy),
            "solve_time": float(dt),
        })

    return archive_entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="Benchmark/Data", help="Path to Benchmark Data")
    parser.add_argument("--out", type=str, default="Benchmark/results/archive_mapa.json", help="Output JSON path")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_path = Path(args.out).resolve()

    if not data_dir.is_dir():
        print(f"Error: Data directory {data_dir} not found.")
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running MAPA benchmark on {data_dir} with {args.workers} workers...")
    
    all_entries = []
    t_start = time.time()
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(solve_instance, i, data_dir): i for i in range(1, 91)}
        
        for i, future in enumerate(as_completed(futures)):
            inst_id = futures[future]
            try:
                entries = future.result()
                all_entries.extend(entries)
                print(f"[{i+1:>2}/90] Solved instance {inst_id:>2} -> found {len(entries)} Pareto points")
            except Exception as exc:
                print(f"[{i+1:>2}/90] Instance {inst_id:>2} generated an exception: {exc}")

    t_end = time.time()
    print(f"\nDone! Total time: {t_end - t_start:.2f}s")
    
    with open(out_path, "w") as f:
        json.dump(all_entries, f, indent=2)
    
    print(f"Saved {len(all_entries)} Pareto points to {out_path}")
    print("\nYou can now compare against EHS using:")
    print(f"python compare_ehs.py --glns-archive {out_path} --ehs-dir Benchmark/results/EHS")


if __name__ == "__main__":
    main()

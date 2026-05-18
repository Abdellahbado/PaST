#!/usr/bin/env python3
"""
PLAN18 continuation: run only missing rows, one at a time, with true incremental CSV append.
Memory-safe rules:
- one heavy row at a time
- append single row to CSV instead of rewriting entire file
- do not accumulate all rows in RAM during heavy run
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for p in (str(ROOT), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from hpc.benchmark_extensions.build_extension_suites import build_instance
from run_plan05_paper_groups_extension import stable_seed
from run_plan13_two_track_recovery import run_row
from run_plan18_k_boundary_refine_n1000 import (
    HARD_B_BASE,
    build_plan18_payload,
    normalize_output_row,
    IRREGULAR_REROUTE_ENV,
    N_JOBS,
    LAMBDA,
    TIME_LIMIT,
    MAX_RSS_GB,
    OUT_DIR,
    RAW_CSV,
    BEST_CSV,
    SUM_K_CSV,
    FAIL_CSV,
    build_best_of_route,
    build_summary_by_k,
    build_failure_signatures,
    write_csv,
    load_raw,
    refresh_derived_fields,
    row_key,
)


def append_single_row(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    """Append one row to an existing CSV without loading/rewriting the whole file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    mode = "a" if exists else "w"
    with open(path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)


def get_header(path: Path) -> list[str]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames) if r.fieldnames else []


def run_missing() -> None:
    solver = ROOT / "solvers" / "cpp" / "build" / "stateful_compare"
    if not solver.exists():
        raise FileNotFoundError(f"Missing solver binary: {solver}")

    # Load only keys to know what's missing; keep existing rows for final rebuild
    existing = load_raw(RAW_CSV)
    seen = {row_key(r) for r in existing}

    sizes = HARD_B_BASE[:12]
    fid = "hardB_k12"
    label = "{" + ",".join(str(x) for x in sizes) + "}"
    k = 12

    header = get_header(RAW_CSV)
    if not header:
        raise RuntimeError("Raw CSV header missing; cannot append incrementally.")

    missing = []
    for seed in (0, 1, 2, 3):
        key = (fid, "irregular_reroute", seed)
        if key not in seen:
            missing.append(seed)

    print(f"Missing seeds for {fid} irregular_reroute: {missing}")
    print(f"Existing rows: {len(existing)}")

    for seed in missing:
        print(f"\nRunning {fid} seed={seed} irregular_reroute ...")
        payload = build_plan18_payload(fid, sizes, N_JOBS, LAMBDA, seed)
        raw = run_row(
            fid,
            N_JOBS,
            seed,
            TIME_LIMIT,
            "irregular_reroute",
            dict(IRREGULAR_REROUTE_ENV),
            max_rss_gb=MAX_RSS_GB,
            payload=payload,
        )
        row = normalize_output_row(
            raw,
            family_id=fid,
            family_label=label,
            family_class="hard_irregular_B",
            family_sizes=sizes,
            k=k,
            variant_label="irregular_reroute",
            route_policy="plan18:additive_profile_repair_beam_auto_v1",
        )
        # Append incrementally instead of rewriting whole file
        append_single_row(RAW_CSV, row, header)
        existing.append(row)
        seen.add((fid, "irregular_reroute", seed))
        print(
            f"  -> step={row.get('deciding_step')} opt={row.get('is_optimal')} "
            f"bc={row.get('boundary_class')} rt={row.get('runtime_sec')} "
            f"rc={row.get('solver_returncode')}"
        )

    # Refresh derived fields and rebuild all derived artifacts
    refresh_derived_fields(existing)
    write_csv(RAW_CSV, existing)
    print(f"\nFinal raw rows: {len(existing)}")

    best = build_best_of_route(existing)
    write_csv(BEST_CSV, best)
    print(f"Wrote best-of-route: {BEST_CSV} n={len(best)}")

    sum_k = build_summary_by_k(best)
    write_csv(SUM_K_CSV, sum_k)
    print(f"Wrote summary-by-k: {SUM_K_CSV} n={len(sum_k)}")

    fail = build_failure_signatures(best)
    write_csv(FAIL_CSV, fail)
    print(f"Wrote failure-signatures: {FAIL_CSV} n={len(fail)}")


if __name__ == "__main__":
    run_missing()

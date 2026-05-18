#!/usr/bin/env python3
"""PLAN30: Easy-vs-hard K-scaling story at fixed n=1000.

Run easy contiguous-unit families K=24, K=30 (and optionally K=40).
Compare against existing hard irregular data from PLAN17/18.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_plan05_paper_groups_extension import stable_seed
from run_plan13_two_track_recovery import read_rss_kb, build_preexec_memory_limit, _extract, _read_file_tail
from run_plan17_k_axis_n1000 import build_plan17_payload, sizes_label, arithmetic_descriptors, classify_boundary

SOLVER = ROOT / "solvers" / "cpp" / "build" / "stateful_compare"
OUT_DIR = ROOT / "research" / "k_vs_arithmetic_axes_20260412" / "csv" / "plan30"
RAW_CSV = OUT_DIR / "PLAN30_easy_k_scaling_raw.csv"

BASELINE_ENV = {
    "PAST_RELAXED_BINPACK_SOLVER": "energy_core",
    "PAST_BLOCK_REPAIR_COMPLETION_MODE": "direct",
    "PAST_BLOCK_REPAIR_COMPLETION_DIRECT_MAX_CELLS": "500000000",
    "PAST_BLOCK_REPAIR_EC_STRONGER_CENTER": "0",
    "PAST_BLOCK_REPAIR_EC_DIVERSIFY": "0",
    "PAST_BLOCK_REPAIR_EC_ADAPTIVE_DELTA": "0",
    "PAST_BLOCK_REPAIR_EC_TWO_PHASE": "0",
    "PAST_BLOCK_REPAIR_EG_STATE_KEEP": "60000",
}

DENSE_STEP2_ENV = {
    "PAST_DENSE_UNIT_STEP2_FASTPATH": "1",
    "PAST_DENSE_UNIT_FASTPATH_K_MIN": "8",
}


def _extract_csv_from_tail(path: Path, max_read_bytes: int = 1_000_000) -> dict[str, str]:
    text = _read_file_tail(path, tail_bytes=max_read_bytes)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return {}
    known_fields = {
        "runtime_sec", "is_optimal", "ub", "lb", "gap_pct", "timed_out",
        "fwd_pack_method", "winner_detail", "deciding_step",
    }
    header_idx = -1
    for i, line in enumerate(lines):
        parts = set(line.split(","))
        if parts & known_fields:
            header_idx = i
            break
    if header_idx == -1:
        return {}
    csv_lines = lines[header_idx:]
    rows = list(csv.DictReader(csv_lines))
    return rows[0] if rows else {}


def run_row(family_id, sizes, n_jobs, seed, time_limit, variant_label, env_overrides, max_rss_gb=16.0):
    payload = build_plan17_payload(family_id, sizes, n_jobs, 1.3, seed)
    env = os.environ.copy()
    env.update(BASELINE_ENV)
    env.update(env_overrides)

    cmd = [str(SOLVER), "ablation-stdin", "step1_exact_guided", str(time_limit)]
    external_timeout = int(max(240, time_limit + 120))
    max_rss_kb = int(max_rss_gb * 1024 * 1024)

    t0 = time.monotonic()
    out_file = tempfile.NamedTemporaryFile(prefix="plan28_out_", suffix=".txt", delete=False)
    err_file = tempfile.NamedTemporaryFile(prefix="plan28_err_", suffix=".txt", delete=False)
    out_path = out_file.name
    err_path = err_file.name
    out_file.close()
    err_file.close()

    out_fh = open(out_path, "w", encoding="utf-8")
    err_fh = open(err_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=out_fh, stderr=err_fh,
        text=True, env=env, preexec_fn=build_preexec_memory_limit(max_rss_gb),
    )

    peak_rss_kb = 0
    memory_killed = False
    timed_out = False
    deadline = t0 + external_timeout

    try:
        if proc.stdin is not None:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.close()
    except Exception:
        pass

    while proc.poll() is None:
        rss_kb = read_rss_kb(proc.pid)
        peak_rss_kb = max(peak_rss_kb, rss_kb)
        if rss_kb > max_rss_kb:
            memory_killed = True
            proc.kill()
            break
        if time.monotonic() >= deadline:
            timed_out = True
            proc.kill()
            break
        time.sleep(0.2)

    wall = time.monotonic() - t0
    try:
        proc.wait(timeout=10)
    except Exception:
        pass
    out_fh.close()
    err_fh.close()

    raw = _extract_csv_from_tail(Path(out_path))
    try:
        os.remove(out_path)
    except Exception:
        pass
    try:
        os.remove(err_path)
    except Exception:
        pass

    rc = proc.returncode if proc.returncode is not None else -9

    row = {
        "family_id": family_id,
        "family_sizes": sizes_label(sizes),
        "K": str(len(sizes)),
        "n": str(n_jobs),
        "lambda": "1.3",
        "seed": str(seed),
        "variant_label": variant_label,
        "time_limit_sec": f"{time_limit}",
        "runtime_wall_sec": f"{wall:.4f}",
        "solver_returncode": str(rc),
        "peak_rss_kb": str(peak_rss_kb),
        "peak_rss_gb": f"{peak_rss_kb / (1024.0 * 1024.0):.3f}",
        "memory_killed": "1" if memory_killed else "0",
        "external_timed_out": "1" if timed_out else "0",
    }

    if timed_out or memory_killed:
        status = "memory_limit_kill" if memory_killed else "external_timeout"
        row.update({
            "runtime_sec": f"{time_limit:.4f}", "timed_out": "1", "is_optimal": "0",
            "feasible": "0", "ub": "-1", "lb": "-1", "gap_pct": "nan",
            "deciding_step": status, "fwd_pack_method": "none",
        })
        return row

    if not raw:
        row.update({
            "runtime_sec": f"{wall:.4f}", "timed_out": "1", "is_optimal": "0",
            "feasible": "0", "ub": "-1", "lb": "-1", "gap_pct": "nan",
            "deciding_step": "no_csv_row", "fwd_pack_method": "none",
        })
        return row

    row.update({
        "runtime_sec": _extract(raw, "runtime_sec") or f"{wall:.4f}",
        "timed_out": _extract(raw, "timed_out"),
        "is_optimal": _extract(raw, "is_optimal"),
        "feasible": _extract(raw, "feasible"),
        "ub": _extract(raw, "ub"),
        "lb": _extract(raw, "lb"),
        "gap_pct": _extract(raw, "gap_pct"),
        "deciding_step": _extract(raw, "deciding_step"),
        "fwd_pack_method": _extract(raw, "fwd_pack_method"),
        "fwd_pack_outcome": _extract(raw, "fwd_pack_outcome"),
        "step2_reached": _extract(raw, "fwd_step2_reached"),
        "step2_produced_ub": _extract(raw, "fwd_step2_produced_ub"),
        "dense_unit_fastpath_active": _extract(raw, "fwd_dense_unit_fastpath_active"),
        "count_based_ffd_active": _extract(raw, "fwd_count_based_ffd_active"),
    })
    return row


def append_csv_row(row, path):
    fieldnames = list(row.keys())
    has_header = path.exists() and path.stat().st_size > 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not has_header:
            w.writeheader()
        w.writerow(row)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    families = [
        ("easy_k24_unit", list(range(1, 25)), [0, 1]),
        ("easy_k30_unit", list(range(1, 31)), [0, 1]),
    ]

    time_limit = 1200

    for family_id, sizes, seeds in families:
        for seed in seeds:
            for variant_label, env_overrides in [("baseline", {}), ("dense_step2_fastpath", DENSE_STEP2_ENV)]:
                print(f"[{time.strftime('%H:%M:%S')}] Running {family_id} seed={seed} variant={variant_label}")
                row = run_row(family_id, sizes, 1000, seed, time_limit, variant_label, env_overrides)
                append_csv_row(row, RAW_CSV)
                print(f"  -> runtime={row.get('runtime_sec', 'N/A')} gap={row.get('gap_pct', 'N/A')} deciding={row.get('deciding_step', 'N/A')} rss={row.get('peak_rss_gb', 'N/A')}GB optimal={row.get('is_optimal', 'N/A')}")

    print(f"Done. Raw CSV: {RAW_CSV}")


if __name__ == "__main__":
    main()

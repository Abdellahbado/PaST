#!/usr/bin/env python3
"""PLAN28: Block Realizability Diagnostics and Bounded Repair — Phase A."""

from __future__ import annotations

import csv, json, os, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Also add the research directory for inter-script imports
RESEARCH_DIR = Path(__file__).resolve().parent
if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

from run_plan24_beam_corridor_exact import (
    build_custom_payload, make_variant_env, read_rss_kb,
    build_preexec_memory_limit, _read_file_tail, _extract_csv_from_tail, write_csv
)

SOLVER = ROOT / "solvers" / "cpp" / "build" / "stateful_compare"
OUT_DIR = ROOT / "research" / "k_vs_arithmetic_axes_20260412" / "csv" / "plan28"
RAW_CSV = OUT_DIR / "PLAN28_block_realizability_diagnostics_raw.csv"

BASELINE_ENV = {
    "PAST_RELAXED_BINPACK_SOLVER": "profile_repair_beam",
    "PAST_PROFILE_REALIZATION_SELECTOR_POLICY": "auto_v1",
    "PAST_BLOCK_REPAIR_COMPLETION_MODE": "direct",
    "PAST_BLOCK_REPAIR_COMPLETION_DIRECT_MAX_CELLS": "500000000",
    "PAST_BLOCK_REPAIR_EC_STRONGER_CENTER": "0",
    "PAST_BLOCK_REPAIR_EC_DIVERSIFY": "0",
    "PAST_BLOCK_REPAIR_EC_ADAPTIVE_DELTA": "0",
    "PAST_BLOCK_REPAIR_EC_TWO_PHASE": "0",
    "PAST_BLOCK_REPAIR_EG_STATE_KEEP": "60000",
    # PLAN28: activate block-realizability diagnostics
    "PAST_BLOCK_REALIZ_DIAG": "1",
    "PAST_BLOCK_REALIZ_DIAG_MAX_PATTERNS": "200000",
}

# Extended FAMILY_LENGTHS for Phase A
FAMILY_LENGTHS = {
    # Easy unit-contiguous
    "easy_k8":  [1, 2, 3, 4, 5, 6, 7, 8],
    "easy_k10": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "easy_k12": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    # HardA irregular
    "hardA_k8":  [2, 3, 4, 5, 7, 11, 13, 17],
    "hardA_k10": [2, 3, 4, 5, 7, 11, 13, 17, 19, 23],
    "hardA_k12": [2, 3, 4, 5, 7, 11, 13, 17, 19, 23, 29, 31],
    # HardB irregular
    "hardB_k8":  [3, 5, 7, 11, 13, 17, 19, 23],
    "hardB_k10": [3, 5, 7, 11, 13, 17, 19, 23, 29, 31],
    "hardB_k12": [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41],
}


def build_custom_payload_ext(family_id: str, n_jobs: int, seed: int, lam: float = 1.3):
    """Build payload for both easy and hard families."""
    return build_custom_payload(family_id, n_jobs, seed, lam)


def _extract(raw, *keys):
    for k in keys:
        v = raw.get(k)
        if v is not None and v != "":
            return v
    return ""


def run_row(family_id, n_jobs, seed, time_limit, variant_label, env_overrides, max_rss_gb=16.0):
    payload = build_custom_payload_ext(family_id, n_jobs, seed)
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
    stderr_tail = _read_file_tail(Path(err_path), tail_bytes=8192)[-500:].replace("\n", "\\n").replace("\r", "\\r")

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
        "family_id": family_id, "n": n_jobs, "lambda": "1.3", "seed": seed,
        "variant_label": variant_label, "time_limit_sec": f"{time_limit}",
        "runtime_wall_sec": f"{wall:.4f}", "solver_returncode": str(rc),
        "peak_rss_kb": str(peak_rss_kb), "peak_rss_gb": f"{peak_rss_kb / (1024.0 * 1024.0):.3f}",
        "memory_killed": "1" if memory_killed else "0",
        "external_timed_out": "1" if timed_out else "0",
        "stderr_tail": stderr_tail,
    }

    if timed_out or memory_killed:
        status = "memory_limit_kill" if memory_killed else "external_timeout"
        row.update({
            "runtime_sec": f"{time_limit:.4f}", "timed_out": "1", "is_optimal": "0",
            "feasible": "0", "ub": "-1", "lb": "-1", "gap_pct": "nan",
            "deciding_step": status, "failure_stage": status,
            "winner_detail": "error", "fwd_pack_method": "none",
            "fwd_pack_outcome": status,
        })
        return row

    if not raw:
        row.update({
            "runtime_sec": f"{wall:.4f}", "timed_out": "1", "is_optimal": "0",
            "feasible": "0", "ub": "-1", "lb": "-1", "gap_pct": "nan",
            "deciding_step": "no_csv_row", "failure_stage": "no_csv_row",
            "winner_detail": "error", "fwd_pack_method": "none",
            "fwd_pack_outcome": "no_csv_row",
        })
        return row

    fwd_method = _extract(raw, "fwd_pack_method")
    beam_status = _extract(raw, "fwd_profile_beam_status")
    ub_val = _extract(raw, "ub")
    fwd_outcome = _extract(raw, "fwd_pack_outcome")
    has_beam = (fwd_method == "profile_repair_beam" or beam_status == "feasible")
    no_incumbent = (fwd_method == "none" or ub_val == "-1" or fwd_outcome == "failed" or not has_beam)
    if no_incumbent:
        row.update({
            "runtime_sec": _extract(raw, "runtime_sec") or f"{wall:.4f}",
            "timed_out": _extract(raw, "timed_out") or "0", "is_optimal": "0",
            "feasible": "0", "ub": ub_val or "-1", "lb": _extract(raw, "lb") or "-1",
            "gap_pct": "nan", "deciding_step": "misrouted_or_no_beam_incumbent",
            "failure_stage": "misrouted_or_no_beam_incumbent",
            "winner_detail": "error", "fwd_pack_method": fwd_method or "none",
            "fwd_pack_outcome": fwd_outcome or "failed",
        })
        row["selector_reason"] = _extract(raw, "fwd_profile_selector_reason")
        return row

    deciding_step = "unknown"
    if raw.get("diag_step1_decided") == "1":
        deciding_step = "step1"
    elif raw.get("diag_step2_decided") == "1":
        deciding_step = "step2"
    elif raw.get("diag_step3_decided") == "1":
        deciding_step = "step3"
    elif raw.get("diag_step4_decided") == "1":
        deciding_step = "step4"
    elif raw.get("timed_out") == "1":
        deciding_step = "timeout"

    row.update({
        "runtime_sec": _extract(raw, "runtime_sec") or f"{wall:.4f}",
        "timed_out": _extract(raw, "timed_out"),
        "is_optimal": _extract(raw, "is_optimal"),
        "feasible": _extract(raw, "feasible"),
        "ub": _extract(raw, "ub"),
        "lb": _extract(raw, "lb"),
        "gap_pct": _extract(raw, "gap_pct"),
        "deciding_step": deciding_step,
        "winner_detail": _extract(raw, "winner_detail"),
        "fwd_pack_method": _extract(raw, "fwd_pack_method"),
        "fwd_pack_outcome": _extract(raw, "fwd_pack_outcome"),
        "fwd_profile_beam_candidate_ub": _extract(raw, "fwd_profile_beam_candidate_ub"),
        "fwd_profile_beam_status": _extract(raw, "fwd_profile_beam_status"),
        "fwd_profile_beam_timed_out": _extract(raw, "fwd_profile_beam_timed_out"),
        "fwd_merged_block_count": _extract(raw, "fwd_merged_block_count"),
        # PLAN28 block-realizability diagnostics
        "block_realiz_diag_active": _extract(raw, "block_realiz_diag_active"),
        "block_realiz_blocks_total": _extract(raw, "block_realiz_blocks_total"),
        "block_realiz_bad_blocks": _extract(raw, "block_realiz_bad_blocks"),
        "block_realiz_bad_rate": _extract(raw, "block_realiz_bad_rate"),
        "block_realiz_first_bad_block": _extract(raw, "block_realiz_first_bad_block"),
        "block_realiz_min_finite_patterns": _extract(raw, "block_realiz_min_finite_patterns"),
        "block_realiz_mean_finite_patterns": _extract(raw, "block_realiz_mean_finite_patterns"),
        "block_realiz_base_path_survives": _extract(raw, "block_realiz_base_path_survives"),
        "block_realiz_base_reject_reason": _extract(raw, "block_realiz_base_reject_reason"),
        "block_realiz_diag_time_sec": _extract(raw, "block_realiz_diag_time_sec"),
        "block_realiz_diag_skipped": _extract(raw, "block_realiz_diag_skipped"),
        "block_realiz_diag_skip_reason": _extract(raw, "block_realiz_diag_skip_reason"),
        "block_realiz_per_block_payload": _extract(raw, "block_realiz_per_block_payload"),
    })
    return row


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plan = []

    # Phase A rows:
    # easy K8, K10, K12 — seeds 0,1
    for k in [8, 10, 12]:
        for seed in [0, 1]:
            fam = f"easy_k{k}"
            plan.append((fam, 1000, seed, 1200.0, "baseline"))

    # hardA K8, K10, K12 — seeds 0,1
    for k in [8, 10, 12]:
        for seed in [0, 1]:
            fam = f"hardA_k{k}"
            plan.append((fam, 1000, seed, 1200.0, "baseline"))

    # hardB K8, K10, K12 — seeds 0,1
    for k in [8, 10, 12]:
        for seed in [0, 1]:
            fam = f"hardB_k{k}"
            plan.append((fam, 1000, seed, 1200.0, "baseline"))

    # Patch PLAN24 module's FAMILY_LENGTHS to include all Phase A families
    import run_plan24_beam_corridor_exact as _p24
    _p24.FAMILY_LENGTHS.clear()
    _p24.FAMILY_LENGTHS.update(FAMILY_LENGTHS)

    # Load existing rows
    existing = []
    seen = set()
    if RAW_CSV.exists():
        with open(RAW_CSV, newline="") as f:
            existing = list(csv.DictReader(f))
        seen = {(r["variant_label"], r["family_id"], int(r["seed"])) for r in existing}

    for fam, n, seed, tlim, label in plan:
        key = (label, fam, seed)
        if key in seen:
            continue
        env = {}  # baseline env includes PAST_BLOCK_REALIZ_DIAG=1
        r = run_row(fam, n, seed, tlim, label, env, max_rss_gb=16.0)
        existing.append(r)
        seen.add(key)
        write_csv(RAW_CSV, existing)
        print(
            f"fam={fam} seed={seed} step={r.get('deciding_step')} "
            f"ub={r.get('ub')} gap={r.get('gap_pct')} rt={r.get('runtime_sec')} "
            f"rss={r.get('peak_rss_gb')} "
            f"br_active={r.get('block_realiz_diag_active')} "
            f"br_blocks={r.get('block_realiz_blocks_total')} "
            f"br_bad={r.get('block_realiz_bad_blocks')} "
            f"br_bad_rate={r.get('block_realiz_bad_rate')} "
            f"br_survives={r.get('block_realiz_base_path_survives')} "
            f"br_reject={r.get('block_realiz_base_reject_reason')} "
            f"br_min_fin={r.get('block_realiz_min_finite_patterns')} "
            f"br_mean_fin={r.get('block_realiz_mean_finite_patterns')} "
            f"br_time={r.get('block_realiz_diag_time_sec')}"
        )

    print(f"\nDone. Raw CSV: {RAW_CSV}")
    print(f"Total rows: {len(existing)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""PLAN24B: forced-entry corridor exact DP diagnostic."""

from __future__ import annotations

import csv, json, os, resource, subprocess, sys, tempfile, time, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_plan05_paper_groups_extension import build_payload
from run_plan24_beam_corridor_exact import (
    build_custom_payload, make_variant_env, read_rss_kb,
    build_preexec_memory_limit, _read_file_tail, _extract_csv_from_tail, write_csv
)

SOLVER = ROOT / "solvers" / "cpp" / "build" / "stateful_compare"
OUT_DIR = ROOT / "research" / "k_vs_arithmetic_axes_20260412" / "csv" / "plan24b"
RAW_CSV = OUT_DIR / "PLAN24B_forced_corridor_raw.csv"

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
}


def _extract(raw, *keys):
    for k in keys:
        v = raw.get(k)
        if v is not None and v != "":
            return v
    return ""


def run_row(family_id, n_jobs, seed, time_limit, variant_label, env_overrides, max_rss_gb=16.0):
    payload = build_custom_payload(family_id, n_jobs, seed)
    env = os.environ.copy()
    env.update(BASELINE_ENV)
    env.update(env_overrides)

    cmd = [str(SOLVER), "ablation-stdin", "step1_exact_guided", str(time_limit)]
    external_timeout = int(max(240, time_limit + 120))
    max_rss_kb = int(max_rss_gb * 1024 * 1024)

    t0 = time.monotonic()
    out_file = tempfile.NamedTemporaryFile(prefix="plan24b_out_", suffix=".txt", delete=False)
    err_file = tempfile.NamedTemporaryFile(prefix="plan24b_err_", suffix=".txt", delete=False)
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
        "t_fwd_pack_profile_beam": _extract(raw, "fwd_t_pack_profile_beam"),
        "exact_diag_mode": _extract(raw, "exact_diag_mode"),
        "exact_diag_variant": _extract(raw, "exact_diag_variant"),
        "exact_diag_elapsed": _extract(raw, "exact_diag_elapsed"),
        "exact_diag_states_reached": _extract(raw, "exact_diag_states_reached"),
        "exact_diag_states_expanded": _extract(raw, "exact_diag_states_expanded"),
        "exact_diag_pruned_bound": _extract(raw, "exact_diag_pruned_bound"),
        "exact_diag_pruned_relaxed": _extract(raw, "exact_diag_pruned_relaxed"),
        "exact_diag_pruned_completion": _extract(raw, "exact_diag_pruned_completion"),
        "exact_diag_pruned_type_aware": _extract(raw, "exact_diag_pruned_type_aware"),
        "exact_diag_pruned_dominance": _extract(raw, "exact_diag_pruned_dominance"),
        "exact_diag_corridor_enabled": _extract(raw, "exact_diag_corridor_enabled"),
        "exact_diag_corridor_delta": _extract(raw, "exact_diag_corridor_delta"),
        "exact_diag_corridor_pruned": _extract(raw, "exact_diag_corridor_pruned"),
        "exact_diag_corridor_infeasible": _extract(raw, "exact_diag_corridor_infeasible"),
        "corridor_force_entry": _extract(raw, "corridor_force_entry"),
        "corridor_max_states": _extract(raw, "corridor_max_states"),
        "corridor_time_limit": _extract(raw, "corridor_time_limit"),
        "stop_reason": _extract(raw, "stop_reason"),
    })
    return row


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plan = []

    # hardA_k10 seed=0
    for variant in ["standard_step4", "forced_corridor_delta1_300s", "forced_corridor_delta2_300s"]:
        if variant == "standard_step4":
            env = make_variant_env("standard_step4")
            tlim = 1200.0
        elif variant == "forced_corridor_delta1_300s":
            env = make_variant_env("corridor_delta1")
            env["PAST_EXACT_CORRIDOR_FORCE_ENTRY"] = "1"
            env["PAST_EXACT_CORRIDOR_TIME_LIMIT"] = "300"
            tlim = 1200.0  # Give enough overall time for beam + forced exact
        elif variant == "forced_corridor_delta2_300s":
            env = make_variant_env("corridor_delta2")
            env["PAST_EXACT_CORRIDOR_FORCE_ENTRY"] = "1"
            env["PAST_EXACT_CORRIDOR_TIME_LIMIT"] = "300"
            tlim = 1200.0  # Give enough overall time for beam + forced exact
        plan.append((variant, env, "hardA_k10", 1000, 0, tlim, 16.0))

    # hardB_k10 seed=2
    for variant in ["standard_step4", "forced_corridor_delta1_300s", "forced_corridor_delta2_300s"]:
        if variant == "standard_step4":
            env = make_variant_env("standard_step4")
            tlim = 1200.0
        elif variant == "forced_corridor_delta1_300s":
            env = make_variant_env("corridor_delta1")
            env["PAST_EXACT_CORRIDOR_FORCE_ENTRY"] = "1"
            env["PAST_EXACT_CORRIDOR_TIME_LIMIT"] = "300"
            tlim = 1200.0  # Give enough overall time for beam + forced exact
        elif variant == "forced_corridor_delta2_300s":
            env = make_variant_env("corridor_delta2")
            env["PAST_EXACT_CORRIDOR_FORCE_ENTRY"] = "1"
            env["PAST_EXACT_CORRIDOR_TIME_LIMIT"] = "300"
            tlim = 1200.0  # Give enough overall time for beam + forced exact
        plan.append((variant, env, "hardB_k10", 1000, 2, tlim, 16.0))

    # Load existing rows
    existing = []
    seen = set()
    if RAW_CSV.exists():
        with open(RAW_CSV, newline="") as f:
            existing = list(csv.DictReader(f))
        seen = {(r["variant_label"], r["family_id"], int(r["seed"])) for r in existing}

    for label, env, fam, n, seed, tlim, mem in plan:
        key = (label, fam, seed)
        if key in seen:
            continue
        r = run_row(fam, n, seed, tlim, label, env, max_rss_gb=mem)
        existing.append(r)
        seen.add(key)
        write_csv(RAW_CSV, existing)
        print(f"{label} family={fam} seed={seed} step={r.get('deciding_step')} ub={r.get('ub')} gap={r.get('gap_pct')} rt={r.get('runtime_sec')} rss={r.get('peak_rss_gb')} mode={r.get('exact_diag_mode')} stop={r.get('stop_reason')}")

    print(f"\nDone. Raw CSV: {RAW_CSV}")
    print(f"Total rows: {len(existing)}")


if __name__ == "__main__":
    main()

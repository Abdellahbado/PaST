#!/usr/bin/env python3
"""PLAN33: Certified Anytime Hard-K Prepass."""

from __future__ import annotations
import csv, json, os, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
RESEARCH_DIR = Path(__file__).resolve().parent
if str(RESEARCH_DIR) not in sys.path: sys.path.insert(0, str(RESEARCH_DIR))

from run_plan24_beam_corridor_exact import (
    build_custom_payload, read_rss_kb,
    build_preexec_memory_limit, _read_file_tail, _extract_csv_from_tail, write_csv
)

SOLVER = ROOT / "solvers" / "cpp" / "build" / "stateful_compare"
OUT_DIR = RESEARCH_DIR / "csv" / "plan33"
RAW_CSV = OUT_DIR / "PLAN33_cert_anytime_raw.csv"

BASE_ENV = {
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

FAMILY_LENGTHS = {
    "hardA_k12": [2,3,4,5,7,11,13,17,19,23,29,31],
    "hardB_k12": [3,5,7,11,13,17,19,23,29,31,37,41],
    "hardA_k10": [2,3,4,5,7,11,13,17,19,23],
    "hardB_k10": [3,5,7,11,13,17,19,23,29,31],
}

# PLAN32C baseline: serial anytime initial UB (75 trials + local search, skip forward DP)
PLAN32C_BASELINE_ENV = {
    "PAST_ANYTIME_INITIAL_UB": "1",
    "PAST_ANYTIME_INITIAL_UB_TRIALS": "75",
    "PAST_ANYTIME_INITIAL_UB_LOCAL_SEARCH": "1",
    "PAST_ANYTIME_INITIAL_UB_ONLY": "1",
    "PAST_ANYTIME_RETURN_ON_TIMEOUT": "1",
}

# PLAN33: certified anytime prepass (5 trials + polish + LB + gap stop)
# Self-contained — the cert prepass does its own serial portfolio.
# Do NOT add PAST_ANYTIME_INITIAL_UB=1 (redundant 75-trial block
# would exhaust the time budget before the semigroup LB runs).
PLAN33_ENV = {
    "PAST_CERT_ANYTIME_PREPASS": "1",
    "PAST_CERT_ANYTIME_K_MIN": "10",
    "PAST_CERT_ANYTIME_GAP_STOP_PCT": "0.1",
    "PAST_CERT_ANYTIME_TRIALS": "5",
    "PAST_CERT_ANYTIME_POLISH": "1",
}

VARIANTS = {
    "plan32c_baseline": PLAN32C_BASELINE_ENV,
    "plan33_cert_prepass": PLAN33_ENV,
}


def _extract(raw, *keys):
    for k in keys:
        v = raw.get(k)
        if v is not None and v != "": return v
    return ""


def run_row(family_id, n_jobs, seed, time_limit, variant_label, env_overrides, max_rss_gb=16.0):
    payload = build_custom_payload(family_id, n_jobs, seed)
    env = os.environ.copy()
    env.update(BASE_ENV)
    env.update(env_overrides)

    cmd = [str(SOLVER), "ablation-stdin", "step1_exact_guided", str(time_limit)]
    external_timeout = int(max(240, time_limit + 120))
    max_rss_kb = int(max_rss_gb * 1024 * 1024)

    t0 = time.monotonic()
    out_file = tempfile.NamedTemporaryFile(prefix="plan33_", suffix=".txt", delete=False)
    err_file = tempfile.NamedTemporaryFile(prefix="plan33_err_", suffix=".txt", delete=False)
    out_path = out_file.name; err_path = err_file.name
    out_file.close(); err_file.close()

    out_fh = open(out_path, "w"); err_fh = open(err_path, "w")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=out_fh, stderr=err_fh,
                            text=True, env=env, preexec_fn=build_preexec_memory_limit(max_rss_gb))
    if proc.stdin: proc.stdin.write(json.dumps(payload) + "\n"); proc.stdin.close()

    peak_rss_kb = 0; memory_killed = False; timed_out = False; deadline = t0 + external_timeout
    while proc.poll() is None:
        rss_kb = read_rss_kb(proc.pid); peak_rss_kb = max(peak_rss_kb, rss_kb)
        if rss_kb > max_rss_kb: memory_killed = True; proc.kill(); break
        if time.monotonic() >= deadline: timed_out = True; proc.kill(); break
        time.sleep(0.2)

    wall = time.monotonic() - t0
    try: proc.wait(timeout=10)
    except: pass
    out_fh.close(); err_fh.close()

    raw = _extract_csv_from_tail(Path(out_path))
    stderr_tail = _read_file_tail(Path(err_path), tail_bytes=8192)[-500:].replace("\n", "\\n")
    for p in [out_path, err_path]:
        try: os.remove(p)
        except: pass

    rc = proc.returncode if proc.returncode is not None else -9

    row = {
        "family_id": family_id, "n": n_jobs, "lambda": "1.3", "seed": seed,
        "variant_label": variant_label, "time_limit_sec": f"{time_limit}",
        "runtime_wall_sec": f"{wall:.4f}", "solver_returncode": str(rc),
        "peak_rss_kb": str(peak_rss_kb), "peak_rss_gb": f"{peak_rss_kb/(1024*1024):.3f}",
        "memory_killed": "1" if memory_killed else "0",
        "external_timed_out": "1" if timed_out else "0", "stderr_tail": stderr_tail,
    }

    # Always try to parse the solver CSV row, even on timeout
    if not raw:
        # No CSV row found — use fallback
        if memory_killed:
            row.update({"runtime_sec": f"{time_limit:.4f}", "timed_out": "1", "is_optimal": "0",
                         "feasible": "0", "ub": "-1", "lb": "-1", "gap_pct": "nan",
                         "deciding_step": "memory_limit_kill", "failure_stage": "memory_limit_kill",
                         "winner_detail": "error", "fwd_pack_method": "none"})
        elif timed_out:
            row.update({"runtime_sec": f"{time_limit:.4f}", "timed_out": "1", "is_optimal": "0",
                         "feasible": "0", "ub": "-1", "lb": "-1", "gap_pct": "nan",
                         "deciding_step": "external_timeout", "failure_stage": "external_timeout",
                         "winner_detail": "error", "fwd_pack_method": "none"})
        else:
            row.update({"runtime_sec": f"{wall:.4f}", "timed_out": "1", "is_optimal": "0",
                         "feasible": "0", "ub": "-1", "lb": "-1", "gap_pct": "nan",
                         "deciding_step": "no_csv_row", "winner_detail": "error",
                         "fwd_pack_method": "none"})
        return row

    # Parse solver CSV row
    ub_val = _extract(raw, "ub")
    no_incumbent = (ub_val in ("-1", "-1.000000", ""))
    if no_incumbent:
        row.update({"runtime_sec": _extract(raw, "runtime_sec") or f"{wall:.4f}",
                     "timed_out": _extract(raw, "timed_out") or "0", "is_optimal": "0",
                     "feasible": "0", "ub": "-1", "lb": _extract(raw, "lb") or "-1",
                     "gap_pct": "nan", "deciding_step": "no_incumbent",
                     "winner_detail": "error", "fwd_pack_method": _extract(raw, "fwd_pack_method") or "none"})
        return row

    deciding_step = "unknown"
    step_reached = _extract(raw, "step_reached", "deciding_step")
    if "cert_anytime_prepass" in step_reached: deciding_step = "cert_anytime_prepass"
    elif raw.get("diag_step1_decided") == "1": deciding_step = "step1"
    elif raw.get("diag_step2_decided") == "1": deciding_step = "step2"
    elif raw.get("diag_step3_decided") == "1": deciding_step = "step3"
    elif raw.get("diag_step4_decided") == "1": deciding_step = "step4"
    elif raw.get("timed_out") == "1": deciding_step = "timeout"

    row.update({
        "runtime_sec": _extract(raw, "runtime_sec") or f"{wall:.4f}",
        "timed_out": _extract(raw, "timed_out", "is_timed_out"),
        "is_optimal": _extract(raw, "is_optimal"),
        "feasible": _extract(raw, "feasible"),
        "ub": _extract(raw, "ub"), "lb": _extract(raw, "lb"),
        "gap_pct": _extract(raw, "gap_pct"),
        "deciding_step": deciding_step,
        "winner_detail": _extract(raw, "winner_detail"),
        "fwd_pack_method": _extract(raw, "fwd_pack_method"),
        "fwd_pack_outcome": _extract(raw, "fwd_pack_outcome"),
        # PLAN32 anytime diagnostics
        "anytime_initial_ub": _extract(raw, "anytime_initial_ub"),
        "anytime_initial_ub_source": _extract(raw, "anytime_initial_ub_source"),
        "anytime_time_to_first_ub": _extract(raw, "anytime_time_to_first_ub"),
        "anytime_initial_ub_valid": _extract(raw, "anytime_initial_ub_valid"),
        # PLAN33 cert anytime diagnostics
        "cert_anytime_enabled": _extract(raw, "cert_anytime_enabled"),
        "cert_anytime_triggered": _extract(raw, "cert_anytime_triggered"),
        "cert_anytime_stopped": _extract(raw, "cert_anytime_stopped"),
        "cert_anytime_initial_ub": _extract(raw, "cert_anytime_initial_ub"),
        "cert_anytime_lb": _extract(raw, "cert_anytime_lb"),
        "cert_anytime_gap_pct": _extract(raw, "cert_anytime_gap_pct"),
        "cert_anytime_best_policy": _extract(raw, "cert_anytime_best_policy"),
        "cert_anytime_finite_candidates": _extract(raw, "cert_anytime_finite_candidates"),
        "cert_anytime_time_to_first_ub": _extract(raw, "cert_anytime_time_to_first_ub"),
        "cert_anytime_time_total": _extract(raw, "cert_anytime_time_total"),
        "cert_anytime_polish_used": _extract(raw, "cert_anytime_polish_used"),
        "cert_anytime_ub_before_polish": _extract(raw, "cert_anytime_ub_before_polish"),
        "cert_anytime_ub_after_polish": _extract(raw, "cert_anytime_ub_after_polish"),
    })
    return row


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["A", "B", "C", "all"], default="A")
    ap.add_argument("--time-limit", type=float, default=2400.0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import run_plan24_beam_corridor_exact as _p24
    _p24.FAMILY_LENGTHS.clear(); _p24.FAMILY_LENGTHS.update(FAMILY_LENGTHS)

    if args.phase == "A":
        # Phase A: correctness smoke — K12 s3 + K10 s0,s1
        plan = []
        for fam in ["hardA_k12", "hardB_k12"]:
            for seed in [3]:
                plan.append((fam, 1000, seed, args.time_limit, "plan32c_baseline", PLAN32C_BASELINE_ENV))
                plan.append((fam, 1000, seed, args.time_limit, "plan33_cert_prepass", PLAN33_ENV))
        for fam in ["hardA_k10", "hardB_k10"]:
            for seed in [0, 1]:
                plan.append((fam, 1000, seed, args.time_limit, "plan32c_baseline", PLAN32C_BASELINE_ENV))
                plan.append((fam, 1000, seed, args.time_limit, "plan33_cert_prepass", PLAN33_ENV))
    elif args.phase == "B":
        # Phase B: no-regression check all 8 hard K12 rows
        plan = []
        for fam in ["hardA_k12", "hardB_k12"]:
            for seed in [0, 1, 2, 3]:
                plan.append((fam, 1000, seed, args.time_limit, "plan32c_baseline", PLAN32C_BASELINE_ENV))
                plan.append((fam, 1000, seed, args.time_limit, "plan33_cert_prepass", PLAN33_ENV))
    elif args.phase == "C":
        # Phase C: optional K10 check
        plan = []
        for fam in ["hardA_k10", "hardB_k10"]:
            for seed in [0, 1]:
                plan.append((fam, 1000, seed, args.time_limit, "plan32c_baseline", PLAN32C_BASELINE_ENV))
                plan.append((fam, 1000, seed, args.time_limit, "plan33_cert_prepass", PLAN33_ENV))
    else:
        plan = []
        for fam in ["hardA_k12", "hardB_k12"]:
            for seed in [0, 1, 2, 3]:
                plan.append((fam, 1000, seed, args.time_limit, "plan32c_baseline", PLAN32C_BASELINE_ENV))
                plan.append((fam, 1000, seed, args.time_limit, "plan33_cert_prepass", PLAN33_ENV))

    existing = []; seen = set()
    if RAW_CSV.exists():
        with open(RAW_CSV, newline="") as f:
            existing = list(csv.DictReader(f))
        seen = {(r["variant_label"], r["family_id"], int(r["seed"])) for r in existing}

    for fam, n, seed, tlim, label, ev in plan:
        key = (label, fam, seed)
        if key in seen: continue
        print(f"[PLAN33 {args.phase}] variant={label:<20} family={fam:<12} seed={seed} ...", flush=True)
        r = run_row(fam, n, seed, tlim, label, ev, max_rss_gb=16.0)
        existing.append(r); seen.add(key)
        write_csv(RAW_CSV, existing)
        print(f"  step={r.get('deciding_step'):<20} ub={r.get('ub'):<18} "
              f"lb={r.get('lb'):<18} gap={r.get('gap_pct'):>10} "
              f"rt={r.get('runtime_sec'):>8} "
              f"cert_stop={r.get('cert_anytime_stopped') or '0':>1} "
              f"cert_pol={r.get('cert_anytime_best_policy') or 'none':<16} "
              f"cert_ub={r.get('cert_anytime_initial_ub') or '-1':<18}", flush=True)

    print(f"\nDone. Raw CSV: {RAW_CSV} Total: {len(existing)}")

if __name__ == "__main__":
    main()

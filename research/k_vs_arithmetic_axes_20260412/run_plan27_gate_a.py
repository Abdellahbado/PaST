#!/usr/bin/env python3
"""PLAN27: Step-3 adaptive survivor policy experiments.

Gate A: hardA_k10 seeds 0-3, hardB_k10 seeds 0-3
Variants: standard_beam, uniform_mult2, ambig_scoreband_mult2,
          late_ambig, residual_aware, late_residual_ambig
"""

from __future__ import annotations

import csv, json, os, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_plan24_beam_corridor_exact import (
    build_custom_payload, read_rss_kb, build_preexec_memory_limit,
    _extract_csv_from_tail, _read_file_tail,
)

SOLVER = ROOT / "solvers" / "cpp" / "build" / "stateful_compare"
OUT_DIR = ROOT / "research" / "k_vs_arithmetic_axes_20260412" / "csv" / "plan27"
RAW_CSV = OUT_DIR / "PLAN27_step3_adaptive_survivor_raw.csv"

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

VARIANTS = {
    "standard_beam": {
        "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_POLICY": "off",
        "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_MAX": "1",
        "PAST_PROFILE_REPAIR_BEAM_SCORE_POLICY": "default",
        "PAST_PROFILE_REPAIR_BEAM_RESIDUAL_WEIGHT": "0.0",
        "PAST_PROFILE_REPAIR_BEAM_LATE_FRAC": "0.0",
    },
    "uniform_mult2": {
        "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_POLICY": "uniform",
        "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_MAX": "2",
        "PAST_PROFILE_REPAIR_BEAM_SCORE_POLICY": "default",
        "PAST_PROFILE_REPAIR_BEAM_RESIDUAL_WEIGHT": "0.0",
        "PAST_PROFILE_REPAIR_BEAM_LATE_FRAC": "0.0",
    },
    "ambig_scoreband_mult2": {
        "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_POLICY": "ambig_scoreband",
        "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_MAX": "2",
        "PAST_PROFILE_REPAIR_BEAM_SCORE_POLICY": "default",
        "PAST_PROFILE_REPAIR_BEAM_RESIDUAL_WEIGHT": "0.0",
        "PAST_PROFILE_REPAIR_BEAM_LATE_FRAC": "0.0",
    },
    "late_ambig": {
        "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_POLICY": "late_ambig",
        "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_MAX": "2",
        "PAST_PROFILE_REPAIR_BEAM_SCORE_POLICY": "default",
        "PAST_PROFILE_REPAIR_BEAM_RESIDUAL_WEIGHT": "0.0",
        "PAST_PROFILE_REPAIR_BEAM_LATE_FRAC": "0.35",
    },
    "residual_aware": {
        "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_POLICY": "off",
        "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_MAX": "1",
        "PAST_PROFILE_REPAIR_BEAM_SCORE_POLICY": "residual_aware",
        "PAST_PROFILE_REPAIR_BEAM_RESIDUAL_WEIGHT": "0.2",
        "PAST_PROFILE_REPAIR_BEAM_LATE_FRAC": "0.0",
    },
    "late_residual_ambig": {
        "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_POLICY": "late_ambig",
        "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_MAX": "2",
        "PAST_PROFILE_REPAIR_BEAM_SCORE_POLICY": "residual_aware",
        "PAST_PROFILE_REPAIR_BEAM_RESIDUAL_WEIGHT": "0.2",
        "PAST_PROFILE_REPAIR_BEAM_LATE_FRAC": "0.35",
    },
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
    out_file = tempfile.NamedTemporaryFile(prefix="plan27_out_", suffix=".txt", delete=False)
    err_file = tempfile.NamedTemporaryFile(prefix="plan27_err_", suffix=".txt", delete=False)
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
            "fwd_profile_beam_score_policy": env_overrides.get("PAST_PROFILE_REPAIR_BEAM_SCORE_POLICY", "default"),
            "fwd_profile_beam_residual_weight": env_overrides.get("PAST_PROFILE_REPAIR_BEAM_RESIDUAL_WEIGHT", "0.0"),
            "fwd_profile_beam_late_frac": env_overrides.get("PAST_PROFILE_REPAIR_BEAM_LATE_FRAC", "0.0"),
            "fwd_profile_beam_key_multi_policy": env_overrides.get("PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_POLICY", "off"),
            "fwd_profile_beam_key_multi_max": env_overrides.get("PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_MAX", "1"),
        })
        return row

    if not raw:
        row.update({
            "runtime_sec": f"{wall:.4f}", "timed_out": "1", "is_optimal": "0",
            "feasible": "0", "ub": "-1", "lb": "-1", "gap_pct": "nan",
            "deciding_step": "no_csv_row", "failure_stage": "no_csv_row",
            "winner_detail": "error", "fwd_pack_method": "none",
            "fwd_pack_outcome": "no_csv_row",
            "fwd_profile_beam_score_policy": env_overrides.get("PAST_PROFILE_REPAIR_BEAM_SCORE_POLICY", "default"),
            "fwd_profile_beam_residual_weight": env_overrides.get("PAST_PROFILE_REPAIR_BEAM_RESIDUAL_WEIGHT", "0.0"),
            "fwd_profile_beam_late_frac": env_overrides.get("PAST_PROFILE_REPAIR_BEAM_LATE_FRAC", "0.0"),
            "fwd_profile_beam_key_multi_policy": env_overrides.get("PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_POLICY", "off"),
            "fwd_profile_beam_key_multi_max": env_overrides.get("PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_MAX", "1"),
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
            "fwd_profile_beam_score_policy": _extract(raw, "fwd_profile_beam_score_policy") or env_overrides.get("PAST_PROFILE_REPAIR_BEAM_SCORE_POLICY", "default"),
            "fwd_profile_beam_residual_weight": _extract(raw, "fwd_profile_beam_residual_weight") or env_overrides.get("PAST_PROFILE_REPAIR_BEAM_RESIDUAL_WEIGHT", "0.0"),
            "fwd_profile_beam_late_frac": _extract(raw, "fwd_profile_beam_late_frac") or env_overrides.get("PAST_PROFILE_REPAIR_BEAM_LATE_FRAC", "0.0"),
            "fwd_profile_beam_key_multi_policy": _extract(raw, "fwd_profile_beam_key_multi_policy") or env_overrides.get("PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_POLICY", "off"),
            "fwd_profile_beam_key_multi_max": _extract(raw, "fwd_profile_beam_key_multi_max") or env_overrides.get("PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_MAX", "1"),
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
        "fwd_profile_beam_base_width": _extract(raw, "fwd_profile_beam_base_width"),
        "fwd_profile_beam_avg_width": _extract(raw, "fwd_profile_beam_avg_width"),
        "fwd_profile_beam_max_width": _extract(raw, "fwd_profile_beam_max_width"),
        "fwd_profile_beam_states_considered": _extract(raw, "fwd_profile_beam_states_considered"),
        "fwd_profile_beam_states_kept": _extract(raw, "fwd_profile_beam_states_kept"),
        "fwd_profile_beam_score_policy": _extract(raw, "fwd_profile_beam_score_policy"),
        "fwd_profile_beam_residual_weight": _extract(raw, "fwd_profile_beam_residual_weight"),
        "fwd_profile_beam_residual_mean_penalty": _extract(raw, "fwd_profile_beam_residual_mean_penalty"),
        "fwd_profile_beam_residual_max_penalty": _extract(raw, "fwd_profile_beam_residual_max_penalty"),
        "fwd_profile_beam_late_frac": _extract(raw, "fwd_profile_beam_late_frac"),
        "fwd_profile_beam_key_multi_policy": _extract(raw, "fwd_profile_beam_key_multi_policy"),
        "fwd_profile_beam_key_multi_max": _extract(raw, "fwd_profile_beam_key_multi_max"),
        "exact_diag_mode": _extract(raw, "exact_diag_mode"),
        "exact_diag_variant": _extract(raw, "exact_diag_variant"),
        "exact_diag_elapsed": _extract(raw, "exact_diag_elapsed"),
        "exact_diag_states_reached": _extract(raw, "exact_diag_states_reached"),
        "exact_diag_states_expanded": _extract(raw, "exact_diag_states_expanded"),
        "exact_diag_pruned_bound": _extract(raw, "exact_diag_pruned_bound"),
        "exact_diag_pruned_relaxed": _extract(raw, "exact_diag_pruned_relaxed"),
        "exact_diag_pruned_completion": _extract(raw, "exact_diag_pruned_completion"),
        "exact_diag_pruned_type_aware": _extract(raw, "exact_diag_pruned_type_aware"),
        "exact_diag_timed_out": _extract(raw, "exact_diag_timed_out"),
        "exact_diag_corridor_enabled": _extract(raw, "exact_diag_corridor_enabled"),
        "exact_diag_corridor_delta": _extract(raw, "exact_diag_corridor_delta"),
        "exact_diag_corridor_pruned": _extract(raw, "exact_diag_corridor_pruned"),
    })
    return row


def write_csv(rows, path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def append_csv_row(row, path):
    fieldnames = list(row.keys())
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Ensure raw CSV exists with header
    if not RAW_CSV.exists():
        with open(RAW_CSV, "w", newline="") as f:
            pass

    families = [
        ("hardA_k10", 1000, [0, 1, 2, 3]),
        ("hardB_k10", 1000, [0, 1, 2, 3]),
    ]

    time_limit = 1200
    all_rows = []

    for family_id, n_jobs, seeds in families:
        for seed in seeds:
            for variant_label, env_overrides in VARIANTS.items():
                print(f"[{time.strftime('%H:%M:%S')}] Running {family_id} seed={seed} variant={variant_label}")
                row = run_row(family_id, n_jobs, seed, time_limit, variant_label, env_overrides)
                all_rows.append(row)
                append_csv_row(row, RAW_CSV)
                print(f"  -> runtime={row.get('runtime_sec', 'N/A')} gap={row.get('gap_pct', 'N/A')} deciding={row.get('deciding_step', 'N/A')} rss={row.get('peak_rss_gb', 'N/A')}GB")

    print(f"Done. Raw CSV: {RAW_CSV}")
    return all_rows


if __name__ == "__main__":
    main()

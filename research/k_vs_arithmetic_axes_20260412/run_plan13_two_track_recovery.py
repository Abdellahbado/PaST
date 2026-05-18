#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_plan05_paper_groups_extension import build_payload

SOLVER = ROOT / "solvers" / "cpp" / "build" / "stateful_compare"
OUT_DIR = ROOT / "research" / "k_vs_arithmetic_axes_20260412" / "csv" / "plan15"

RELAX_DIAG_CSV = OUT_DIR / "PLAN15_dense_unit_relax_diagnosis.csv"
RELAX_FASTPATH_COMPARE_CSV = OUT_DIR / "PLAN15_dense_unit_relax_fastpath_compare.csv"
ENERGY_PROFILE_COMPARE_CSV = OUT_DIR / "PLAN15_dense_unit_energy_profile_compare.csv"
SMOKE_120_CSV = OUT_DIR / "PLAN15_dense_unit_1_20_smoke.csv"

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


def parse_csv_row(stdout: str) -> dict[str, str]:
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    if len(lines) < 2:
        return {}
    rows = list(csv.DictReader(lines))
    return rows[0] if rows else {}


def read_rss_kb(pid: int) -> int:
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True)
        raw = out.strip()
        return int(raw) if raw else 0
    except Exception:
        return 0


def build_preexec_memory_limit(max_rss_gb: float):
    limit_bytes = int(max_rss_gb * 1024 * 1024 * 1024)

    def _apply_limit() -> None:
        try:
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_DATA, (limit_bytes, limit_bytes))
        except Exception:
            pass

    return _apply_limit


def _extract(raw: dict[str, str], *keys: str) -> str:
    for k in keys:
        v = raw.get(k)
        if v is not None and v != "":
            return v
    return ""


def _read_file_tail(path: Path, tail_bytes: int = 8192) -> str:
    """Read only the last tail_bytes of a file without loading it all."""
    try:
        size = path.stat().st_size
    except Exception:
        return ""
    read_size = min(size, tail_bytes)
    try:
        with open(path, "rb") as f:
            if size > read_size:
                f.seek(size - read_size)
                # discard potential partial first line
                f.readline()
            return f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_csv_from_tail(path: Path, max_read_bytes: int = 1_000_000) -> dict[str, str]:
    """Read only the trailing max_read_bytes of stdout to parse the CSV row."""
    text = _read_file_tail(path, tail_bytes=max_read_bytes)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return {}
    # Detect CSV header by looking for known solver fields
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
        # No recognizable header in tail; solver likely did not emit CSV.
        return {}
    csv_lines = lines[header_idx:]
    rows = list(csv.DictReader(csv_lines))
    return rows[0] if rows else {}


def run_row(
    family_id: str,
    n_jobs: int,
    seed: int,
    time_limit: float,
    variant_label: str,
    env_overrides: dict[str, str],
    max_rss_gb: float = 12.0,
    rss_poll_sec: float = 0.2,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if payload is None:
        payload, _ = build_payload(family_id, n_jobs, 1.3, seed)
    env = os.environ.copy()
    env.update(BASELINE_ENV)
    env.update(env_overrides)

    cmd = [str(SOLVER), "ablation-stdin", "step1_exact_guided", str(time_limit)]
    external_timeout = int(max(240, time_limit + 120))
    max_rss_kb = int(max_rss_gb * 1024 * 1024)

    t0 = time.monotonic()
    out_file = tempfile.NamedTemporaryFile(
        prefix="plan15_out_", suffix=".txt", delete=False
    )
    err_file = tempfile.NamedTemporaryFile(
        prefix="plan15_err_", suffix=".txt", delete=False
    )
    out_path = out_file.name
    err_path = err_file.name
    out_file.close()
    err_file.close()

    out_fh = open(out_path, "w", encoding="utf-8")
    err_fh = open(err_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=out_fh,
        stderr=err_fh,
        text=True,
        env=env,
        preexec_fn=build_preexec_memory_limit(max_rss_gb),
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
        time.sleep(max(0.1, rss_poll_sec))

    wall = time.monotonic() - t0
    try:
        proc.wait(timeout=10)
    except Exception:
        pass
    out_fh.close()
    err_fh.close()

    # Memory-safe: never read the full temp files into Python heap.
    raw = _extract_csv_from_tail(Path(out_path))
    stderr_tail = (
        _read_file_tail(Path(err_path), tail_bytes=8192)[-500:]
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )

    try:
        os.remove(out_path)
    except Exception:
        pass
    try:
        os.remove(err_path)
    except Exception:
        pass

    rc = proc.returncode if proc.returncode is not None else -9

    row: dict[str, Any] = {
        "family_id": family_id,
        "n": n_jobs,
        "lambda": "1.3",
        "seed": seed,
        "variant_label": variant_label,
        "time_limit_sec": f"{time_limit}",
        "runtime_wall_sec": f"{wall:.4f}",
        "solver_returncode": str(rc),
        "peak_rss_kb": str(peak_rss_kb),
        "peak_rss_gb": f"{peak_rss_kb / (1024.0 * 1024.0):.3f}",
        "memory_killed": "1" if memory_killed else "0",
        "external_timed_out": "1" if timed_out else "0",
        "stderr_tail": stderr_tail,
    }

    if timed_out or memory_killed:
        status = "memory_limit_kill" if memory_killed else "external_timeout"
        row.update(
            {
                "runtime_sec": f"{time_limit:.4f}",
                "timed_out": "1",
                "is_optimal": "0",
                "feasible": "0",
                "ub": "-1",
                "lb": "-1",
                "gap_pct": "nan",
                "deciding_step": status,
                "failure_stage": status,
                "winner_detail": "error",
                "fwd_pack_method": "none",
                "fwd_pack_outcome": status,
                "step2_reached": "0",
                "step2_produced_ub": "0",
            }
        )
        return row

    if not raw:
        row.update(
            {
                "runtime_sec": f"{wall:.4f}",
                "timed_out": "1",
                "is_optimal": "0",
                "feasible": "0",
                "ub": "-1",
                "lb": "-1",
                "gap_pct": "nan",
                "deciding_step": "no_csv_row",
                "failure_stage": "no_csv_row",
                "winner_detail": "error",
                "fwd_pack_method": "none",
                "fwd_pack_outcome": "no_csv_row",
                "step2_reached": "0",
                "step2_produced_ub": "0",
            }
        )
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

    failure_stage = "completed"
    if raw.get("timed_out") == "1" and deciding_step == "timeout":
        failure_stage = "solver_timeout"
    elif _extract(raw, "fwd_block_dp_status") == "timeout":
        failure_stage = "step3_timeout"
    elif _extract(raw, "fwd_pack_outcome") in {"failed", "infeasible"}:
        failure_stage = "pack_failed"

    row.update(
        {
            "runtime_sec": _extract(raw, "runtime_sec") or f"{wall:.4f}",
            "timed_out": _extract(raw, "timed_out"),
            "is_optimal": _extract(raw, "is_optimal"),
            "feasible": _extract(raw, "feasible"),
            "ub": _extract(raw, "ub"),
            "lb": _extract(raw, "lb"),
            "gap_pct": _extract(raw, "gap_pct"),
            "deciding_step": deciding_step,
            "failure_stage": failure_stage,
            "winner_detail": _extract(raw, "winner_detail"),
            "fwd_pack_method": _extract(raw, "fwd_pack_method"),
            "fwd_pack_outcome": _extract(raw, "fwd_pack_outcome"),
            "selector_policy": _extract(raw, "fwd_profile_selector_policy"),
            "selector_decision": _extract(raw, "fwd_profile_selector_decision"),
            "selector_reason": _extract(raw, "fwd_profile_selector_reason"),
            "step3_mode": _extract(raw, "fwd_profile_step3_incumbent_mode"),
            "block_dp_status": _extract(raw, "fwd_block_dp_status"),
            "diag_exact_dp_used": _extract(raw, "diag_exact_dp_used"),
            "step2_reached": _extract(raw, "fwd_step2_reached"),
            "step2_produced_ub": _extract(raw, "fwd_step2_produced_ub"),
            "step2_ub": _extract(raw, "fwd_profile_step2_ub"),
            # PLAN14 / PLAN15 dense toggles
            "dense_unit_fastpath_active": _extract(
                raw, "fwd_dense_unit_fastpath_active"
            ),
            "count_based_ffd_active": _extract(raw, "fwd_count_based_ffd_active"),
            "dense_unit_relax_fastpath_active": _extract(
                raw, "fwd_dense_unit_relax_fastpath_active"
            ),
            "dense_unit_energy_profile_active": _extract(
                raw, "fwd_dense_unit_energy_profile_active"
            ),
            "dense_unit_relax_fastpath_fallback": _extract(
                raw, "fwd_dense_unit_relax_fastpath_fallback"
            ),
            "dense_unit_energy_profile_fallback": _extract(
                raw, "fwd_dense_unit_energy_profile_fallback"
            ),
            "dense_unit_relax_mode": _extract(raw, "fwd_dense_unit_relax_mode"),
            # coarse timers
            "t_fwd_relax": _extract(raw, "t_fwd_relax"),
            "t_exact": _extract(raw, "t_exact"),
            "t_fwd_pack_profile_recovery": _extract(raw, "t_fwd_pack_profile_recovery"),
            "t_fwd_pack_merge_blocks": _extract(raw, "t_fwd_pack_merge_blocks"),
            "t_fwd_pack_to_first_candidate": _extract(
                raw, "t_fwd_pack_to_first_candidate"
            ),
            "t_fwd_pack_ffd_only": _extract(raw, "t_fwd_pack_ffd_only"),
            "t_fwd_pack_heuristic": _extract(raw, "t_fwd_pack_heuristic"),
            "t_fwd_pack_profile_beam": _extract(raw, "fwd_t_pack_profile_beam"),
            "t_fwd_pack_block_dp_exact": _extract(raw, "fwd_t_pack_block_dp_exact"),
            # PLAN15 dense split timers
            "t_dense_spaces_or_lb": _extract(raw, "t_dense_spaces_or_lb"),
            "t_dense_profile_dp": _extract(raw, "t_dense_profile_dp"),
            "t_dense_profile_recovery": _extract(raw, "t_dense_profile_recovery"),
            "t_dense_block_build": _extract(raw, "t_dense_block_build"),
            "t_dense_job_materialization": _extract(raw, "t_dense_job_materialization"),
            "t_dense_step2_pack": _extract(raw, "t_dense_step2_pack"),
            "t_dense_pre_step2_total": _extract(raw, "t_dense_pre_step2_total"),
            # extra timing
            "t_fwd_ec_completion": _extract(raw, "fwd_ec_time_completion"),
            "t_fwd_ec_pattern_generation": _extract(
                raw, "fwd_ec_time_pattern_generation"
            ),
            "t_fwd_ec_phase1": _extract(raw, "fwd_ec_time_phase1"),
            # PLAN20 beam diagnostics
            "fwd_profile_beam_base_width": _extract(raw, "fwd_profile_beam_base_width"),
            "fwd_profile_beam_avg_width": _extract(raw, "fwd_profile_beam_avg_width"),
            "fwd_profile_beam_max_width": _extract(raw, "fwd_profile_beam_max_width"),
            "fwd_profile_beam_states_considered": _extract(raw, "fwd_profile_beam_states_considered"),
            "fwd_profile_beam_states_kept": _extract(raw, "fwd_profile_beam_states_kept"),
            "fwd_profile_beam_pruned_over": _extract(raw, "fwd_profile_beam_pruned_over"),
            "fwd_profile_beam_pruned_suffix": _extract(raw, "fwd_profile_beam_pruned_suffix"),
            "fwd_profile_beam_pruned_discrepancy": _extract(raw, "fwd_profile_beam_pruned_discrepancy"),
            "fwd_profile_beam_discrepancy_budget": _extract(raw, "fwd_profile_beam_discrepancy_budget"),
            "fwd_profile_beam_discrepancy_depth": _extract(raw, "fwd_profile_beam_discrepancy_depth"),
            "fwd_profile_beam_status": _extract(raw, "fwd_profile_beam_status"),
            "fwd_profile_beam_timed_out": _extract(raw, "fwd_profile_beam_timed_out"),
            "fwd_profile_beam_candidate_ub": _extract(raw, "fwd_profile_beam_candidate_ub"),
            "fwd_profile_beam_plus_candidate_ub": _extract(raw, "fwd_profile_beam_plus_candidate_ub"),
            "fwd_profile_beam_improved_over_step2": _extract(raw, "fwd_profile_beam_improved_over_step2"),
        }
    )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def row_key(label: str, fam: str, n: int, seed: int) -> tuple[str, str, int, int]:
    return (label, fam, n, seed)


def build_cache(*paths: Path) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    cache: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for path in paths:
        for r in load_csv_rows(path):
            try:
                key = row_key(
                    str(r.get("variant_label", "")),
                    str(r.get("family_id", "")),
                    int(r.get("n", "-1")),
                    int(r.get("seed", "-1")),
                )
            except Exception:
                continue
            cache[key] = r
    return cache


def run_plan(
    rows_plan: list[tuple[str, dict[str, str], str, int, int, float, float]],
    out_csv: Path,
    cache: dict[tuple[str, str, int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = load_csv_rows(out_csv)
    seen = {
        row_key(
            str(r.get("variant_label", "")),
            str(r.get("family_id", "")),
            int(r.get("n", "-1")),
            int(r.get("seed", "-1")),
        )
        for r in rows
    }
    for label, env, fam, n, seed, tlim, mem in rows_plan:
        key = row_key(label, fam, n, seed)
        if key in seen:
            continue

        if key in cache:
            row = dict(cache[key])
            rows.append(row)
            seen.add(key)
            write_csv(out_csv, rows)
            print(
                f"reused {label} family={fam} n={n} seed={seed} "
                f"step={row.get('deciding_step')} fail={row.get('failure_stage')} "
                f"runtime={row.get('runtime_sec')} ub={row.get('ub')} lb={row.get('lb')}"
            )
            continue

        row = run_row(fam, n, seed, tlim, label, env, max_rss_gb=mem)
        rows.append(row)
        seen.add(key)
        cache[key] = row
        write_csv(out_csv, rows)
        print(
            f"{label} family={fam} n={n} seed={seed} "
            f"step={row.get('deciding_step')} fail={row.get('failure_stage')} "
            f"runtime={row.get('runtime_sec')} ub={row.get('ub')} lb={row.get('lb')}"
        )
    return rows


def main() -> None:
    if not SOLVER.exists():
        raise FileNotFoundError(f"Missing solver binary: {SOLVER}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = build_cache(
        RELAX_DIAG_CSV,
        RELAX_FASTPATH_COMPARE_CSV,
        ENERGY_PROFILE_COMPARE_CSV,
        SMOKE_120_CSV,
    )

    dense_step2_fastpath_env = {
        "PAST_DENSE_UNIT_STEP2_FASTPATH": "1",
        "PAST_DENSE_UNIT_FASTPATH_K_MIN": "8",
    }
    dense_step2_count_env = {
        **dense_step2_fastpath_env,
        "PAST_COUNT_BASED_FFD": "1",
    }
    dense_relax_fastpath_env = {
        **dense_step2_fastpath_env,
        "PAST_DENSE_UNIT_RELAX_FASTPATH": "1",
    }
    dense_relax_fastpath_count_env = {
        **dense_relax_fastpath_env,
        "PAST_COUNT_BASED_FFD": "1",
    }
    dense_energy_profile_env = {
        **dense_step2_fastpath_env,
        "PAST_DENSE_UNIT_ENERGY_PROFILE": "1",
    }

    # Phase A — deeper timing split inside forward relaxation/profile
    # required rows + optional n=6000 seed 0
    phase_a_plan = [
        ("baseline", {}, "g12345678910", 3500, 0, 1200.0, 16.0),
        (
            "step2_fastpath",
            dense_step2_fastpath_env,
            "g12345678910",
            3500,
            0,
            1200.0,
            16.0,
        ),
        (
            "step2_fastpath",
            dense_step2_fastpath_env,
            "g12345678910",
            5000,
            0,
            1200.0,
            16.0,
        ),
        (
            "step2_fastpath",
            dense_step2_fastpath_env,
            "g12345678910",
            5000,
            1,
            1200.0,
            16.0,
        ),
        (
            "step2_fastpath_optional",
            dense_step2_fastpath_env,
            "g12345678910",
            6000,
            0,
            1500.0,
            16.0,
        ),
    ]
    phase_a_rows = run_plan(phase_a_plan, RELAX_DIAG_CSV, cache)

    # Phase B — dense-unit relaxation fastpath compare
    # compare PLAN14 dense-step2-fastpath vs new dense-relax-fastpath
    phase_b_plan = [
        (
            "step2_fastpath",
            dense_step2_fastpath_env,
            "g12345678910",
            3500,
            0,
            1200.0,
            16.0,
        ),
        (
            "relax_fastpath",
            dense_relax_fastpath_env,
            "g12345678910",
            3500,
            0,
            1200.0,
            16.0,
        ),
        (
            "step2_fastpath",
            dense_step2_fastpath_env,
            "g12345678910",
            5000,
            0,
            1200.0,
            16.0,
        ),
        (
            "step2_fastpath",
            dense_step2_fastpath_env,
            "g12345678910",
            5000,
            1,
            1200.0,
            16.0,
        ),
        (
            "relax_fastpath",
            dense_relax_fastpath_env,
            "g12345678910",
            5000,
            0,
            1200.0,
            16.0,
        ),
        (
            "relax_fastpath",
            dense_relax_fastpath_env,
            "g12345678910",
            5000,
            1,
            1200.0,
            16.0,
        ),
        (
            "relax_fastpath_count",
            dense_relax_fastpath_count_env,
            "g12345678910",
            5000,
            0,
            1200.0,
            16.0,
        ),
        (
            "relax_fastpath_count",
            dense_relax_fastpath_count_env,
            "g12345678910",
            5000,
            1,
            1200.0,
            16.0,
        ),
    ]
    phase_b_rows = run_plan(phase_b_plan, RELAX_FASTPATH_COMPARE_CSV, cache)

    # attempt n=6000 seeds 0/1 only if n=5000 relax_fastpath succeeds on both seeds
    relax_5000_ok = {
        int(r.get("seed", -1))
        for r in phase_b_rows
        if r.get("variant_label") == "relax_fastpath"
        and int(r.get("n", -1)) == 5000
        and r.get("is_optimal") == "1"
    }
    if relax_5000_ok == {0, 1}:
        phase_b_n6000 = [
            (
                "relax_fastpath",
                dense_relax_fastpath_env,
                "g12345678910",
                6000,
                0,
                1800.0,
                16.0,
            ),
            (
                "relax_fastpath",
                dense_relax_fastpath_env,
                "g12345678910",
                6000,
                1,
                1800.0,
                16.0,
            ),
        ]
        extra_rows = run_plan(phase_b_n6000, RELAX_FASTPATH_COMPARE_CSV, cache)
        phase_b_rows.extend(extra_rows)

    # Phase C — dense-unit aggregate / energy-profile experiment
    phase_c_plan = [
        (
            "energy_profile",
            dense_energy_profile_env,
            "g12345678910",
            5000,
            0,
            1200.0,
            16.0,
        ),
        (
            "energy_profile",
            dense_energy_profile_env,
            "g12345678910",
            5000,
            1,
            1200.0,
            16.0,
        ),
    ]
    phase_c_rows = run_plan(phase_c_plan, ENERGY_PROFILE_COMPARE_CSV, cache)

    # optional n=6000 seed0 for energy-profile if safe
    phase_c_5000_ok = all(
        r.get("is_optimal") == "1" for r in phase_c_rows if int(r.get("n", -1)) == 5000
    )
    if phase_c_5000_ok:
        extra_c_rows = run_plan(
            [
                (
                    "energy_profile_optional",
                    dense_energy_profile_env,
                    "g12345678910",
                    6000,
                    0,
                    1800.0,
                    16.0,
                )
            ],
            ENERGY_PROFILE_COMPARE_CSV,
            cache,
        )
        phase_c_rows.extend(extra_c_rows)

    # Phase E — {1..20} smoke (after {1..10} diagnosis)
    # Always attempt smoke rows as requested.
    smoke_plan = [
        (
            "dense_unit_relax_fastpath",
            dense_relax_fastpath_env,
            "g1234567891011121314151617181920",
            1000,
            0,
            900.0,
            16.0,
        ),
        (
            "dense_unit_relax_fastpath",
            dense_relax_fastpath_env,
            "g1234567891011121314151617181920",
            2000,
            0,
            900.0,
            16.0,
        ),
    ]

    smoke_rows = run_plan(smoke_plan, SMOKE_120_CSV, cache)

    # Optional n=3500 smoke if first two are exact and memory-safe
    smoke_ok = all(
        r.get("is_optimal") == "1" and r.get("memory_killed") == "0"
        for r in smoke_rows
        if int(r.get("n", -1)) in {1000, 2000}
    )
    if smoke_ok:
        smoke_rows.extend(
            run_plan(
                [
                    (
                        "dense_unit_relax_fastpath_optional",
                        dense_relax_fastpath_env,
                        "g1234567891011121314151617181920",
                        3500,
                        0,
                        1200.0,
                        16.0,
                    )
                ],
                SMOKE_120_CSV,
                cache,
            )
        )

    print(f"Wrote {RELAX_DIAG_CSV} rows={len(phase_a_rows)}")
    print(f"Wrote {RELAX_FASTPATH_COMPARE_CSV} rows={len(phase_b_rows)}")
    print(f"Wrote {ENERGY_PROFILE_COMPARE_CSV} rows={len(phase_c_rows)}")
    print(f"Wrote {SMOKE_120_CSV} rows={len(smoke_rows)}")


if __name__ == "__main__":
    main()

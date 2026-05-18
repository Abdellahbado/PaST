#!/usr/bin/env python3
"""PLAN24: beam-guided Step-4 exact corridor experiments.

Variants:
- standard_step4: current behavior, no corridor
- corridor_delta0: exact beam prefix trajectory only
- corridor_delta1: +/-1 per type
- corridor_delta2: +/-2 per type
- corridor_widen_0_1_2: time-sliced widening schedule
"""

from __future__ import annotations

import csv
import json
import math
import os
import resource
import statistics
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
OUT_DIR = ROOT / "research" / "k_vs_arithmetic_axes_20260412" / "csv" / "plan24"
RAW_CSV = OUT_DIR / "PLAN24_beam_corridor_exact_raw.csv"

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

FAMILY_LENGTHS = {
    "hardA_k10": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
    "hardB_k10": [3, 5, 7, 11, 13, 17, 19, 23, 29, 31],
    "hardA_k12": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37],
    "hardB_k12": [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41],
}


def build_custom_payload(family_id: str, n_jobs: int, seed: int, lam: float = 1.3):
    lengths = FAMILY_LENGTHS[family_id]
    ec = [{"from_date": "2019-01-21T00:00:00", "repeat_count": 1}][seed % 1]
    import random
    rng = random.Random(
        700000 + 131 * seed + 1009 * n_jobs + 17 * int(round(lam * 100)) + sum(ord(c) for c in family_id)
    )
    jobs = [rng.choice(lengths) for _ in range(n_jobs)]
    from hpc.benchmark_extensions.build_extension_suites import build_instance
    inst = build_instance(
        name=f"plan24/{family_id}_n{n_jobs}_lam{lam:.1f}_s{seed}",
        family=family_id,
        jobs_list=jobs,
        horizon_multiplier=lam,
        ec_config=ec,
        metadata={
            "processing_group": lengths,
            "K": len(lengths),
            "seed": seed,
            "lambda": lam,
            "paper_machine": "twosby",
        },
    )
    payload = {
        "instance_id": inst["name"],
        "prices": inst["prices"],
        "jobs": inst["jobs"],
        "machine": "twosby",
    }
    return payload


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
    try:
        size = path.stat().st_size
    except Exception:
        return ""
    read_size = min(size, tail_bytes)
    try:
        with open(path, "rb") as f:
            if size > read_size:
                f.seek(size - read_size)
                f.readline()
            return f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


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


def run_row(
    family_id: str,
    n_jobs: int,
    seed: int,
    time_limit: float,
    variant_label: str,
    env_overrides: dict[str, str],
    max_rss_gb: float = 16.0,
    rss_poll_sec: float = 0.2,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if payload is None:
        payload = build_custom_payload(family_id, n_jobs, seed)
    env = os.environ.copy()
    env.update(BASELINE_ENV)
    env.update(env_overrides)

    cmd = [str(SOLVER), "ablation-stdin", "step1_exact_guided", str(time_limit)]
    external_timeout = int(max(240, time_limit + 120))
    max_rss_kb = int(max_rss_gb * 1024 * 1024)

    t0 = time.monotonic()
    out_file = tempfile.NamedTemporaryFile(prefix="plan24_out_", suffix=".txt", delete=False)
    err_file = tempfile.NamedTemporaryFile(prefix="plan24_err_", suffix=".txt", delete=False)
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
            }
        )
        return row

    # Route sanity check for PLAN24
    fwd_method = _extract(raw, "fwd_pack_method")
    beam_status = _extract(raw, "fwd_profile_beam_status")
    ub_val = _extract(raw, "ub")
    fwd_outcome = _extract(raw, "fwd_pack_outcome")
    has_beam_incumbent = (
        fwd_method == "profile_repair_beam" or beam_status == "feasible"
    )
    no_incumbent = (
        fwd_method == "none"
        or ub_val == "-1"
        or fwd_outcome == "failed"
        or not has_beam_incumbent
    )
    if no_incumbent:
        row.update(
            {
                "runtime_sec": _extract(raw, "runtime_sec") or f"{wall:.4f}",
                "timed_out": _extract(raw, "timed_out") or "0",
                "is_optimal": "0",
                "feasible": "0",
                "ub": ub_val or "-1",
                "lb": _extract(raw, "lb") or "-1",
                "gap_pct": "nan",
                "deciding_step": "misrouted_or_no_beam_incumbent",
                "failure_stage": "misrouted_or_no_beam_incumbent",
                "winner_detail": "error",
                "fwd_pack_method": fwd_method or "none",
                "fwd_pack_outcome": fwd_outcome or "failed",
            }
        )
        # Still copy available diagnostics for debugging
        row["selector_reason"] = _extract(raw, "fwd_profile_selector_reason")
        row["selector_policy"] = _extract(raw, "fwd_profile_selector_policy")
        row["fwd_profile_beam_status"] = beam_status
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
            "t_fwd_relax": _extract(raw, "t_fwd_relax"),
            "t_exact": _extract(raw, "t_exact"),
            "t_fwd_pack_profile_recovery": _extract(raw, "t_fwd_pack_profile_recovery"),
            "t_fwd_pack_merge_blocks": _extract(raw, "t_fwd_pack_merge_blocks"),
            "t_fwd_pack_to_first_candidate": _extract(raw, "t_fwd_pack_to_first_candidate"),
            "t_fwd_pack_ffd_only": _extract(raw, "t_fwd_pack_ffd_only"),
            "t_fwd_pack_heuristic": _extract(raw, "t_fwd_pack_heuristic"),
            "t_fwd_pack_profile_beam": _extract(raw, "fwd_t_pack_profile_beam"),
            "t_fwd_pack_block_dp_exact": _extract(raw, "fwd_t_pack_block_dp_exact"),
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
            "fwd_profile_exact_improved_over_step2": _extract(raw, "fwd_profile_exact_improved_over_step2"),
            "fwd_profile_step2_ub": _extract(raw, "fwd_profile_step2_ub"),
            "fwd_profile_exact_candidate_ub": _extract(raw, "fwd_profile_exact_candidate_ub"),
            "fwd_profile_selector_policy": _extract(raw, "fwd_profile_selector_policy"),
            "fwd_profile_selector_decision": _extract(raw, "fwd_profile_selector_decision"),
            "fwd_profile_selector_reason": _extract(raw, "fwd_profile_selector_reason"),
            "fwd_profile_selector_hard_alarm": _extract(raw, "fwd_profile_selector_hard_alarm"),
            "exact_diag_variant": _extract(raw, "exact_diag_variant"),
            "exact_diag_mode": _extract(raw, "exact_diag_mode"),
            "exact_diag_initial_ub": _extract(raw, "exact_diag_initial_ub"),
            "exact_diag_final_ub": _extract(raw, "exact_diag_final_ub"),
            "exact_diag_elapsed": _extract(raw, "exact_diag_elapsed"),
            "exact_diag_states_reached": _extract(raw, "exact_diag_states_reached"),
            "exact_diag_states_expanded": _extract(raw, "exact_diag_states_expanded"),
            "exact_diag_pruned_bound": _extract(raw, "exact_diag_pruned_bound"),
            "exact_diag_pruned_relaxed": _extract(raw, "exact_diag_pruned_relaxed"),
            "exact_diag_pruned_completion": _extract(raw, "exact_diag_pruned_completion"),
            "exact_diag_pruned_type_aware": _extract(raw, "exact_diag_pruned_type_aware"),
            "exact_diag_pruned_dominance": _extract(raw, "exact_diag_pruned_dominance"),
            "exact_diag_timed_out": _extract(raw, "exact_diag_timed_out"),
            "exact_diag_exhaustive": _extract(raw, "exact_diag_exhaustive"),
            "exact_diag_corridor_enabled": _extract(raw, "exact_diag_corridor_enabled"),
            "exact_diag_corridor_delta": _extract(raw, "exact_diag_corridor_delta"),
            "exact_diag_corridor_pruned": _extract(raw, "exact_diag_corridor_pruned"),
            "exact_diag_corridor_infeasible": _extract(raw, "exact_diag_corridor_infeasible"),
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


def run_plan(
    rows_plan: list[tuple[str, dict[str, str], str, int, int, float, float]],
    out_csv: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = load_csv_rows(out_csv)
    seen = {
        (
            str(r.get("variant_label", "")),
            str(r.get("family_id", "")),
            int(r.get("n", "-1")),
            int(r.get("seed", "-1")),
        )
        for r in rows
    }
    for label, env, fam, n, seed, tlim, mem in rows_plan:
        key = (label, fam, n, seed)
        if key in seen:
            continue

        row = run_row(fam, n, seed, tlim, label, env, max_rss_gb=mem)
        rows.append(row)
        seen.add(key)
        write_csv(out_csv, rows)
        print(
            f"{label} family={fam} n={n} seed={seed} "
            f"step={row.get('deciding_step')} ub={row.get('ub')} lb={row.get('lb')} "
            f"gap={row.get('gap_pct')} rt={row.get('runtime_sec')} rss={row.get('peak_rss_gb')}"
        )
    return rows


def make_variant_env(variant: str) -> dict[str, str]:
    env: dict[str, str] = {}
    if variant == "standard_step4":
        pass
    elif variant == "corridor_delta0":
        env["PAST_EXACT_CORRIDOR_ENABLE"] = "1"
        env["PAST_EXACT_CORRIDOR_DELTA"] = "0"
        env["PAST_EXACT_CORRIDOR_SOURCE"] = "profile_beam"
    elif variant == "corridor_delta1":
        env["PAST_EXACT_CORRIDOR_ENABLE"] = "1"
        env["PAST_EXACT_CORRIDOR_DELTA"] = "1"
        env["PAST_EXACT_CORRIDOR_SOURCE"] = "profile_beam"
    elif variant == "corridor_delta2":
        env["PAST_EXACT_CORRIDOR_ENABLE"] = "1"
        env["PAST_EXACT_CORRIDOR_DELTA"] = "2"
        env["PAST_EXACT_CORRIDOR_SOURCE"] = "profile_beam"
    elif variant == "corridor_widen_0_1_2":
        env["PAST_EXACT_CORRIDOR_ENABLE"] = "1"
        env["PAST_EXACT_CORRIDOR_DELTA"] = "1"
        env["PAST_EXACT_CORRIDOR_SOURCE"] = "profile_beam"
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return env


def main() -> None:
    if not SOLVER.exists():
        raise FileNotFoundError(f"Missing solver binary: {SOLVER}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase A — smoke test
    print("=== Phase A: smoke test ===")
    phase_a_plan = [
        ("standard_step4", make_variant_env("standard_step4"), "hardA_k10", 1000, 0, 1200.0, 16.0),
        ("corridor_delta0", make_variant_env("corridor_delta0"), "hardA_k10", 1000, 0, 1200.0, 16.0),
        ("corridor_delta1", make_variant_env("corridor_delta1"), "hardA_k10", 1000, 0, 1200.0, 16.0),
        ("corridor_delta2", make_variant_env("corridor_delta2"), "hardA_k10", 1000, 0, 1200.0, 16.0),
    ]
    phase_a_rows = run_plan(phase_a_plan, RAW_CSV)

    # Check if smoke succeeded (at least standard produced a row)
    standard_smoke = [r for r in phase_a_rows if r.get("variant_label") == "standard_step4" and int(r.get("seed", -1)) == 0]
    if not standard_smoke or standard_smoke[0].get("deciding_step") not in {"step3", "step4"}:
        print("Phase A smoke FAILED: standard_step4 did not reach step3/4. Stopping.")
        return

    print("=== Phase B: K=10 corridor comparison ===")
    phase_b_plan = []
    for seed in [0, 1, 2, 3]:
        for fam in ["hardA_k10", "hardB_k10"]:
            for variant in ["standard_step4", "corridor_delta1", "corridor_delta2", "corridor_widen_0_1_2"]:
                phase_b_plan.append(
                    (variant, make_variant_env(variant), fam, 1000, seed, 1200.0, 16.0)
                )
    phase_b_rows = run_plan(phase_b_plan, RAW_CSV)

    # Decide if K=12 probe is warranted
    print("=== Evaluating K=12 signal ===")
    k10_rows = [r for r in phase_b_rows if r.get("variant_label") != "standard_step4"]
    any_exact = any(r.get("is_optimal") == "1" for r in k10_rows)
    any_better_gap = False
    any_faster = False
    any_fewer_states = False
    for r in k10_rows:
        std_match = [s for s in phase_b_rows if s.get("family_id") == r.get("family_id") and int(s.get("seed", -1)) == int(r.get("seed", -1)) and s.get("variant_label") == "standard_step4"]
        if not std_match:
            continue
        std = std_match[0]
        try:
            if float(r.get("gap_pct", "inf")) < float(std.get("gap_pct", "inf")) - 1e-6:
                any_better_gap = True
        except Exception:
            pass
        try:
            if float(r.get("runtime_sec", "inf")) < float(std.get("runtime_sec", "inf")) * 0.9:
                any_faster = True
        except Exception:
            pass
        try:
            if float(r.get("exact_diag_states_reached", "inf")) < float(std.get("exact_diag_states_reached", "inf")) * 0.9:
                any_fewer_states = True
        except Exception:
            pass

    print(f"K=10 signal: exact={any_exact} better_gap={any_better_gap} faster={any_faster} fewer_states={any_fewer_states}")

    if any_exact or any_better_gap or any_faster or any_fewer_states:
        print("=== Phase C: K=12 probe ===")
        # Determine best corridor variant from K=10
        variant_gaps: dict[str, list[float]] = {}
        for r in k10_rows:
            v = str(r.get("variant_label", ""))
            try:
                g = float(r.get("gap_pct", "nan"))
                if not math.isnan(g):
                    variant_gaps.setdefault(v, []).append(g)
            except Exception:
                pass
        best_variant = min(variant_gaps, key=lambda v: statistics.mean(variant_gaps[v])) if variant_gaps else "corridor_delta1"
        phase_c_plan = [
            ("standard_step4", make_variant_env("standard_step4"), "hardA_k12", 1000, 0, 1200.0, 16.0),
            (best_variant, make_variant_env(best_variant), "hardA_k12", 1000, 0, 1200.0, 16.0),
            ("standard_step4", make_variant_env("standard_step4"), "hardB_k12", 1000, 0, 1200.0, 16.0),
            (best_variant, make_variant_env(best_variant), "hardB_k12", 1000, 0, 1200.0, 16.0),
        ]
        run_plan(phase_c_plan, RAW_CSV)
    else:
        print("Phase C skipped: no signal from K=10")

    print(f"Done. Raw CSV: {RAW_CSV}")


if __name__ == "__main__":
    main()

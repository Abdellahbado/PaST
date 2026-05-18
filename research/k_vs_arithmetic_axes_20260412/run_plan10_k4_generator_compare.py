#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import os
import random
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SOLVER = ROOT / "solvers" / "cpp" / "build" / "stateful_compare"
PLAN10_DIR = ROOT / "research" / "k_vs_arithmetic_axes_20260412" / "csv" / "plan10"
BASELINE_CSV = PLAN10_DIR / "PLAN10_k4_speedup_baseline.csv"
DP4_CSV = PLAN10_DIR / "PLAN10_k4_generator_dp4.csv"
DFS_OPT_CSV = PLAN10_DIR / "PLAN10_k4_generator_dfs_opt.csv"
COMPARE_CSV = PLAN10_DIR / "PLAN10_k4_generator_compare.csv"

DATASET_DIR = (
    ROOT
    / "data"
    / "green-scheduling-bab"
    / "Iirc.EnergyStatesAndCostsScheduling"
    / "data"
    / "datasets"
    / "paperext_profile_repair_smallk_nscale_plus_20260409"
)

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hpc.benchmark_extensions.build_extension_suites import build_instance


BASE_ENV = {
    "PAST_RELAXED_BINPACK_SOLVER": "energy_core",
    "PAST_BLOCK_REPAIR_COMPLETION_MODE": "direct",
    "PAST_BLOCK_REPAIR_COMPLETION_DIRECT_MAX_CELLS": "500000000",
    "PAST_BLOCK_REPAIR_EC_STRONGER_CENTER": "0",
    "PAST_BLOCK_REPAIR_EC_DIVERSIFY": "0",
    "PAST_BLOCK_REPAIR_EC_ADAPTIVE_DELTA": "0",
    "PAST_BLOCK_REPAIR_EC_TWO_PHASE": "0",
    "PAST_BLOCK_REPAIR_EG_STATE_KEEP": "60000",
}

EC_CONFIGS = [
    {"from_date": "2019-01-21T00:00:00", "repeat_count": 1},
    {"from_date": "2019-04-08T00:00:00", "repeat_count": 1},
]


def stable_seed(family_id: str, n_jobs: int, lam: float, seed: int) -> int:
    lam_tag = int(round(lam * 100))
    return 700000 + 131 * seed + 1009 * n_jobs + 17 * lam_tag + sum(ord(c) for c in family_id)


def build_g3567_payload(n_jobs: int, seed: int, lam: float = 1.3) -> dict[str, Any]:
    lengths = [3, 5, 6, 7]
    rng = random.Random(stable_seed("g3567", n_jobs, lam, seed))
    jobs = [rng.choice(lengths) for _ in range(n_jobs)]
    ec = EC_CONFIGS[seed % len(EC_CONFIGS)]
    inst = build_instance(
        name=f"plan10/g3567_n{n_jobs}_lam{lam:.1f}_s{seed}",
        family="g3567",
        jobs_list=jobs,
        horizon_multiplier=lam,
        ec_config=ec,
        metadata={
            "processing_group": lengths,
            "K": len(lengths),
            "seed": seed,
            "lambda": lam,
            "paper_group": "{3,5,6,7}",
            "paper_machine": "twosby",
        },
    )
    return {
        "instance_id": inst["name"],
        "prices": inst["prices"],
        "jobs": inst["jobs"],
        "machine": "twosby",
    }


def load_json_payload(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if "instance_id" in obj and "prices" in obj and "jobs" in obj:
        return {
            "instance_id": obj["instance_id"],
            "prices": obj["prices"],
            "jobs": obj["jobs"],
            "machine": obj.get("machine", "nosby"),
        }

    if "Intervals" in obj and "Jobs" in obj:
        prices = [float(x.get("EnergyCost", 0.0)) for x in obj.get("Intervals", [])]
        jobs = [int(x.get("ProcessingTime", 0)) for x in obj.get("Jobs", [])]
        return {
            "instance_id": path.stem,
            "prices": prices,
            "jobs": jobs,
            "machine": "nosby",
        }

    return {
        "instance_id": path.stem,
        "prices": [],
        "jobs": [],
        "machine": "nosby",
    }


def parse_solver_stdout(stdout: str) -> dict[str, str]:
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    if len(lines) < 2:
        return {}
    rows = list(csv.DictReader(lines))
    return rows[0] if rows else {}


def read_rss_kb(pid: int) -> int:
    proc = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return 0
    txt = proc.stdout.strip()
    if not txt:
        return 0
    try:
        return int(txt)
    except ValueError:
        return 0


def run_one_case(
    payload: dict[str, Any],
    variant: str,
    extra_env: dict[str, str],
    rss_limit_kb: int,
    time_limit_sec: float,
    poll_sec: float = 1.0,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(BASE_ENV)
    env.update(extra_env)

    cmd = [str(SOLVER), "ablation-stdin", "step1_exact_guided", str(time_limit_sec)]
    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    assert proc.stdin is not None
    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()
    proc.stdin.close()
    proc.stdin = None

    peak_rss_kb = 0
    killed_for_rss = False
    while proc.poll() is None:
        rss = read_rss_kb(proc.pid)
        peak_rss_kb = max(peak_rss_kb, rss)
        if rss_limit_kb > 0 and rss > rss_limit_kb:
            killed_for_rss = True
            try:
                proc.send_signal(signal.SIGKILL)
            except ProcessLookupError:
                pass
            break
        time.sleep(poll_sec)

    stdout, stderr = proc.communicate()
    wall_sec = time.monotonic() - t0
    parsed = parse_solver_stdout(stdout)

    row: dict[str, Any] = {
        "variant": variant,
        "instance_id": payload["instance_id"],
        "machine": payload.get("machine", ""),
        "solver_returncode": proc.returncode,
        "status": "ok" if (proc.returncode == 0 and parsed) else "error",
        "memory_killed": int(killed_for_rss),
        "peak_rss_kb": peak_rss_kb,
        "wall_sec": f"{wall_sec:.4f}",
        "stderr_tail": (stderr or "")[-500:].replace("\n", "\\n").replace("\r", "\\r"),
    }
    row.update(parsed)
    if killed_for_rss:
        row["status"] = "memory_killed"
    return row


def g3567_case_id(n: int, seed: int) -> str:
    return f"g3567_n{n}_lam1.3_s{seed}"


def continuity_case_id(n: int, seed: int) -> str:
    return f"continuity_3567plus_n{n}_s{seed}"


def build_gate_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    continuity_files = [
        (continuity_case_id(3500, 0), DATASET_DIR / "0008_profile_smallk_3567_plus_n3500_s0.json", "continuity_3567_plus", "3567_plus", 3500, 0),
        (continuity_case_id(3500, 1), DATASET_DIR / "0009_profile_smallk_3567_plus_n3500_s1.json", "continuity_3567_plus", "3567_plus", 3500, 1),
        (continuity_case_id(5000, 0), DATASET_DIR / "0010_profile_smallk_3567_plus_n5000_s0.json", "continuity_3567_plus", "3567_plus", 5000, 0),
        (continuity_case_id(5000, 1), DATASET_DIR / "0011_profile_smallk_3567_plus_n5000_s1.json", "continuity_3567_plus", "3567_plus", 5000, 1),
    ]
    for case_id, path, kind, family_id, n, seed in continuity_files:
        payload = load_json_payload(path)
        payload["machine"] = "nosby"
        cases.append(
            {
                "case_id": case_id,
                "kind": kind,
                "family_id": family_id,
                "n": n,
                "seed": seed,
                "payload": payload,
            }
        )

    for n in (2500, 3500, 5000):
        for seed in (0, 1):
            cases.append(
                {
                    "case_id": g3567_case_id(n, seed),
                    "kind": "paper_group",
                    "family_id": "g3567",
                    "n": n,
                    "seed": seed,
                    "payload": build_g3567_payload(n, seed, 1.3),
                }
            )
    return cases


def write_csv(path: Path, rows: list[dict[str, Any]], field_order: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if field_order is None:
        fieldnames: list[str] = []
        for r in rows:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
    else:
        extras: list[str] = []
        for r in rows:
            for k in r.keys():
                if k not in field_order and k not in extras:
                    extras.append(k)
        fieldnames = field_order + extras
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_int(v: Any, default: int = 0) -> int:
    try:
        return int(str(v))
    except Exception:
        return default


def main() -> None:
    if not SOLVER.exists():
        raise FileNotFoundError(f"Missing solver binary: {SOLVER}")

    cases = build_gate_cases()

    rss_limit_kb = int(16.5 * 1024 * 1024)
    time_limit_sec = 3600.0

    variants = [
        (
            "dp4_generator",
            {"PAST_BLOCK_REPAIR_PATTERN_DP_K": "4"},
            lambda c: True,
        ),
        (
            "dp4_generator_dedup_off",
            {
                "PAST_BLOCK_REPAIR_PATTERN_DP_K": "4",
                "PAST_BLOCK_REPAIR_EC_SIGNATURE_DEDUP": "0",
            },
            lambda c: True,
        ),
    ]

    all_rows: list[dict[str, Any]] = []
    for variant, extra_env, case_filter in variants:
        variant_cases = [c for c in cases if case_filter(c)]
        print(f"=== Running variant: {variant} ===", flush=True)
        for i, case in enumerate(variant_cases, start=1):
            print(
                f"[{i}/{len(variant_cases)}] {variant} {case['family_id']} n={case['n']} seed={case['seed']}",
                flush=True,
            )
            raw = run_one_case(
                payload=case["payload"],
                variant=variant,
                extra_env=extra_env,
                rss_limit_kb=rss_limit_kb,
                time_limit_sec=time_limit_sec,
                poll_sec=1.0,
            )
            row = {
                "variant": variant,
                "case_id": case["case_id"],
                "kind": case["kind"],
                "family_id": case["family_id"],
                "n": case["n"],
                "seed": case["seed"],
                "machine": raw.get("machine", ""),
                "status": raw.get("status", ""),
                "solver_returncode": raw.get("solver_returncode", ""),
                "peak_rss_kb": raw.get("peak_rss_kb", ""),
                "wall_sec": raw.get("wall_sec", ""),
                "instance_id": raw.get("instance_id", ""),
                "runtime_sec": raw.get("runtime_sec", ""),
                "ub": raw.get("ub", ""),
                "lb": raw.get("lb", ""),
                "gap_pct": raw.get("gap_pct", ""),
                "is_optimal": raw.get("is_optimal", ""),
                "exact": "exact" if str(raw.get("is_optimal", "")) == "1" else "not_exact",
                "diag_step3_decided": raw.get("diag_step3_decided", ""),
                "diag_step4_decided": raw.get("diag_step4_decided", ""),
                "fwd_ec_time_pattern_generation": raw.get("fwd_ec_time_pattern_generation", ""),
                "fwd_ec_time_exact_core": raw.get("fwd_ec_time_exact_core", ""),
                "fwd_ec_generated_patterns_total": raw.get("fwd_ec_generated_patterns_total", ""),
                "fwd_ec_retained_patterns_total": raw.get("fwd_ec_retained_patterns_total", ""),
                "fwd_pack_method": raw.get("fwd_pack_method", ""),
                "winner_detail": raw.get("winner_detail", ""),
                "ec_from": raw.get("ec_from", ""),
                "ec_repeat": raw.get("ec_repeat", ""),
                "memory_killed": raw.get("memory_killed", 0),
                "stderr_tail": raw.get("stderr_tail", ""),
            }
            all_rows.append(row)

            ok_exact = to_int(row.get("is_optimal", 0), 0) == 1
            print(
                f"    status={row['status']} exact={ok_exact} runtime={row.get('runtime_sec','')}s "
                f"patgen={row.get('fwd_ec_time_pattern_generation','')} rss_kb={row.get('peak_rss_kb','')}",
                flush=True,
            )

    dp4_rows = [r for r in all_rows if r["variant"] == "dp4_generator"]
    write_csv(DP4_CSV, dp4_rows)

    dfs_rows = [r for r in all_rows if r["variant"] == "dfs_optimized"]
    if dfs_rows:
        write_csv(DFS_OPT_CSV, dfs_rows)

    baseline_rows = read_csv(BASELINE_CSV)
    compare_rows: list[dict[str, Any]] = []
    for r in baseline_rows:
        compare_rows.append(
            {
                "variant": "baseline_generator",
                "case_id": r.get("case_id", ""),
                "kind": r.get("kind", ""),
                "family_id": r.get("family_id", ""),
                "n": r.get("n", ""),
                "seed": r.get("seed", ""),
                "is_optimal": r.get("is_optimal", ""),
                "exact": "exact" if str(r.get("is_optimal", "")) == "1" else "not_exact",
                "runtime_sec": r.get("runtime_sec", ""),
                "gap_pct": r.get("gap_pct", ""),
                "diag_step3_decided": r.get("diag_step3_decided", ""),
                "diag_step4_decided": r.get("diag_step4_decided", ""),
                "fwd_ec_time_pattern_generation": r.get("fwd_ec_time_pattern_generation", ""),
                "fwd_ec_time_exact_core": r.get("fwd_ec_time_exact_core", ""),
                "fwd_ec_generated_patterns_total": r.get("fwd_ec_generated_patterns_total", ""),
                "fwd_ec_retained_patterns_total": r.get("fwd_ec_retained_patterns_total", ""),
            }
        )

    for r in all_rows:
        compare_rows.append(
            {
                "variant": r.get("variant", ""),
                "case_id": r.get("case_id", ""),
                "kind": r.get("kind", ""),
                "family_id": r.get("family_id", ""),
                "n": r.get("n", ""),
                "seed": r.get("seed", ""),
                "is_optimal": r.get("is_optimal", ""),
                "exact": "exact" if str(r.get("is_optimal", "")) == "1" else "not_exact",
                "runtime_sec": r.get("runtime_sec", ""),
                "gap_pct": r.get("gap_pct", ""),
                "diag_step3_decided": r.get("diag_step3_decided", ""),
                "diag_step4_decided": r.get("diag_step4_decided", ""),
                "fwd_ec_time_pattern_generation": r.get("fwd_ec_time_pattern_generation", ""),
                "fwd_ec_time_exact_core": r.get("fwd_ec_time_exact_core", ""),
                "fwd_ec_generated_patterns_total": r.get("fwd_ec_generated_patterns_total", ""),
                "fwd_ec_retained_patterns_total": r.get("fwd_ec_retained_patterns_total", ""),
            }
        )

    write_csv(
        COMPARE_CSV,
        compare_rows,
        field_order=[
            "variant",
            "case_id",
            "kind",
            "family_id",
            "n",
            "seed",
            "is_optimal",
            "exact",
            "runtime_sec",
            "gap_pct",
            "diag_step3_decided",
            "diag_step4_decided",
            "fwd_ec_time_pattern_generation",
            "fwd_ec_time_exact_core",
            "fwd_ec_generated_patterns_total",
            "fwd_ec_retained_patterns_total",
        ],
    )

    print(f"Wrote: {DP4_CSV}")
    if dfs_rows:
        print(f"Wrote: {DFS_OPT_CSV}")
    print(f"Wrote: {COMPARE_CSV}")


if __name__ == "__main__":
    main()

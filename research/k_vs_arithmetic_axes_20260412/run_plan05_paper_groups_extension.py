#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import subprocess
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hpc.benchmark_extensions.build_extension_suites import build_instance

SOLVER = ROOT / "solvers" / "cpp" / "build" / "stateful_compare"

OUT_DIR = ROOT / "research" / "k_vs_arithmetic_axes_20260412"
CSV_DIR = OUT_DIR / "csv" / "plan05"
BASELINE_CSV = CSV_DIR / "PAPER_GROUPS_PLAN05_baseline.csv"
N_EXT_CSV = CSV_DIR / "PAPER_GROUPS_PLAN05_n_extension.csv"
LAMBDA_EXT_CSV = CSV_DIR / "PAPER_GROUPS_PLAN05_lambda_extension.csv"

BASELINE_SUMMARY_CSV = CSV_DIR / "PAPER_GROUPS_PLAN05_baseline_summary.csv"
N_EXT_SUMMARY_CSV = CSV_DIR / "PAPER_GROUPS_PLAN05_n_extension_summary.csv"
LAMBDA_EXT_SUMMARY_CSV = CSV_DIR / "PAPER_GROUPS_PLAN05_lambda_extension_summary.csv"
FINAL_SUMMARY_CSV = CSV_DIR / "PAPER_GROUPS_PLAN05_final_summary.csv"
SUMMARY_MD = OUT_DIR / "PAPER_GROUPS_EXTENSION_SUMMARY.md"


GROUPS: "OrderedDict[str, list[int]]" = OrderedDict(
    [
        ("g12345678910", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        (
            "g1234567891011121314151617181920",
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        ),
        ("g12357", [1, 2, 3, 5, 7]),
        ("g246810", [2, 4, 6, 8, 10]),
        ("g24", [2, 4]),
        ("g3567", [3, 5, 6, 7]),
        ("g37", [3, 7]),
        ("g810", [8, 10]),
    ]
)

PHASE2_ORDER = [
    "g810",
    "g12345678910",
    "g3567",
    "g37",
    "g246810",
    "g12357",
    "g24",
]

LABELS = {
    "g12345678910": "{1,2,3,4,5,6,7,8,9,10}",
    "g1234567891011121314151617181920": "{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}",
    "g12357": "{1,2,3,5,7}",
    "g246810": "{2,4,6,8,10}",
    "g24": "{2,4}",
    "g3567": "{3,5,6,7}",
    "g37": "{3,7}",
    "g810": "{8,10}",
}

EC_CONFIGS = [
    {"from_date": "2019-01-21T00:00:00", "repeat_count": 1},
    {"from_date": "2019-04-08T00:00:00", "repeat_count": 1},
]


def parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_strs(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def row_key(row: dict[str, Any]) -> tuple[str, int, str, int, str] | None:
    try:
        return (
            str(row.get("family_id", "")),
            int(row.get("n", "0")),
            str(row.get("lambda", "")),
            int(row.get("seed", "0")),
            str(row.get("phase", "")),
        )
    except Exception:
        return None


def stable_seed(family_id: str, n_jobs: int, lam: float, seed: int) -> int:
    lam_tag = int(round(lam * 100))
    return (
        700000
        + 131 * seed
        + 1009 * n_jobs
        + 17 * lam_tag
        + sum(ord(c) for c in family_id)
    )


def build_payload(
    family_id: str,
    n_jobs: int,
    lam: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    lengths = GROUPS[family_id]
    ec = EC_CONFIGS[seed % len(EC_CONFIGS)]
    rng = random.Random(stable_seed(family_id, n_jobs, lam, seed))
    jobs = [rng.choice(lengths) for _ in range(n_jobs)]

    inst = build_instance(
        name=f"plan05/{family_id}_n{n_jobs}_lam{lam:.1f}_s{seed}",
        family=family_id,
        jobs_list=jobs,
        horizon_multiplier=lam,
        ec_config=ec,
        metadata={
            "processing_group": lengths,
            "K": len(lengths),
            "seed": seed,
            "lambda": lam,
            "paper_group": LABELS[family_id],
            "paper_machine": "twosby",
        },
    )

    payload = {
        "instance_id": inst["name"],
        "prices": inst["prices"],
        "jobs": inst["jobs"],
        "machine": "twosby",
    }
    return payload, inst


def parse_ablation_row(stdout: str) -> dict[str, str]:
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    if len(lines) < 2:
        return {}
    rows = list(csv.DictReader(lines))
    return rows[0] if rows else {}


def deciding_step(row: dict[str, str]) -> str:
    if row.get("diag_step1_decided") == "1":
        return "step1"
    if row.get("diag_step2_decided") == "1":
        return "step2"
    if row.get("diag_step3_decided") == "1":
        return "step3"
    if row.get("diag_step4_decided") == "1":
        return "step4"
    if row.get("timed_out") == "1":
        return "timeout"
    return "unknown"


def step3_mode(row: dict[str, str]) -> str:
    mode = (row.get("fwd_profile_step3_incumbent_mode") or "").strip().lower()
    if mode in ("exact", "beam"):
        return mode
    method = (row.get("fwd_pack_method") or "").lower()
    if "profile_realization_dp_exact" in method:
        return "exact"
    if "profile_repair_beam" in method:
        return "beam"
    return "none"


def exact_entered(row: dict[str, str]) -> int:
    if row.get("diag_exact_dp_used") == "1":
        return 1
    if safe_float(row.get("t_exact", "0")) > 0.000001:
        return 1
    mode = (row.get("exact_diag_mode") or "").strip().lower()
    if mode and mode != "none":
        return 1
    return 0


def run_one(
    payload: dict[str, Any],
    time_limit: float,
) -> tuple[dict[str, str], str, str, int]:
    cmd = [str(SOLVER), "ablation-stdin", "step1_exact_guided", str(time_limit)]
    ext_timeout = int(max(time_limit + 120, 240))
    try:
        proc = subprocess.run(
            cmd,
            input=(json.dumps(payload) + "\n"),
            capture_output=True,
            text=True,
            timeout=ext_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {}, "", "external_timeout", 124

    parsed = parse_ablation_row(proc.stdout)
    stderr_tail = (proc.stderr or "")[-400:]
    stderr_tail = stderr_tail.replace("\r", "\\r").replace("\n", "\\n")
    return parsed, stderr_tail, "ok", proc.returncode


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict[str, str]] = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))

    all_rows: list[dict[str, Any]] = []
    all_rows.extend(existing)
    all_rows.extend(rows)

    # Keep only the latest row per logical key so reruns replace stale rows
    # instead of duplicating them.
    last_idx_by_key: dict[tuple[str, int, str, int, str], int] = {}
    for idx, r in enumerate(all_rows):
        k = row_key(r)
        if k is not None:
            last_idx_by_key[k] = idx

    deduped_rows: list[dict[str, Any]] = []
    for idx, r in enumerate(all_rows):
        k = row_key(r)
        if k is not None and last_idx_by_key.get(k) != idx:
            continue
        deduped_rows.append(r)

    fieldnames: list[str] = []
    for r in deduped_rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(deduped_rows)


def normalize_result_row(row: dict[str, Any]) -> dict[str, Any]:
    ub = safe_float(row.get("ub", ""))
    lb = safe_float(row.get("lb", ""))
    feasible = str(row.get("feasible", "")) == "1"
    timed_out = str(row.get("timed_out", "")) == "1"

    finite_bounds = math.isfinite(ub) and math.isfinite(lb) and ub >= 0.0 and lb >= 0.0

    if not finite_bounds or not feasible:
        row["feasible"] = "0"
        row["is_optimal"] = "0"
        row["gap_pct"] = "nan"
        if timed_out:
            row["deciding_step"] = "timeout"
            row["winner_detail"] = "timeout"
        else:
            row["deciding_step"] = "unresolved"
            row["winner_detail"] = "unresolved"
        return row

    if not math.isfinite(safe_float(row.get("gap_pct", ""))):
        if lb > 0.0:
            row["gap_pct"] = f"{100.0 * max(0.0, ub - lb) / lb:.4f}"
        else:
            row["gap_pct"] = "0.0000"

    if not row.get("winner_detail", "") or row.get("winner_detail", "") == "none":
        row["winner_detail"] = row.get("deciding_step", "") or "unknown"

    return row


def load_keys(path: Path) -> set[tuple[str, int, str, int, str]]:
    if not path.exists():
        return set()
    out: set[tuple[str, int, str, int, str]] = set()
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                out.add(
                    (
                        r.get("family_id", ""),
                        int(r.get("n", "0")),
                        r.get("lambda", ""),
                        int(r.get("seed", "0")),
                        r.get("phase", ""),
                    )
                )
            except Exception:
                continue
    return out


def run_specs(
    specs: list[dict[str, Any]],
    out_csv: Path,
    resume: bool,
) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    seen = load_keys(out_csv) if resume else set()

    total = len(specs)
    for i, sp in enumerate(specs, start=1):
        key = (
            sp["family_id"],
            sp["n"],
            f"{sp['lambda']:.1f}",
            sp["seed"],
            sp["phase"],
        )
        if key in seen:
            continue

        payload, inst = build_payload(
            sp["family_id"], sp["n"], sp["lambda"], sp["seed"]
        )

        t0 = time.monotonic()
        raw, stderr_tail, status, rc = run_one(payload, sp["time_limit_sec"])
        wall = time.monotonic() - t0

        if status != "ok" or rc != 0 or not raw:
            row = {
                "phase": sp["phase"],
                "family": LABELS[sp["family_id"]],
                "family_id": sp["family_id"],
                "K": len(GROUPS[sp["family_id"]]),
                "n": sp["n"],
                "lambda": f"{sp['lambda']:.1f}",
                "seed": sp["seed"],
                "runtime_sec": f"{wall:.4f}",
                "ub": "-1",
                "lb": "-1",
                "gap_pct": "nan",
                "timed_out": "1",
                "is_optimal": "0",
                "feasible": "0",
                "deciding_step_raw": "timeout",
                "deciding_step": "timeout",
                "step3_mode": "none",
                "exact_dp_entered": "0",
                "main_pack_method": "none",
                "selector_policy": "",
                "selector_decision": "",
                "selector_reason": "",
                "winner_detail_raw": "error",
                "winner_detail": "error",
                "fwd_pack_outcome": "",
                "exact_diag_mode": "",
                "instance_id": payload["instance_id"],
                "horizon": len(payload["prices"]),
                "machine": "twosby",
                "ec_from": sp["ec_from"],
                "ec_repeat": sp["ec_repeat"],
                "solver_mode": "ablation-stdin step1_exact_guided",
                "time_limit_sec": sp["time_limit_sec"],
                "solver_returncode": str(rc),
                "stderr_tail": stderr_tail,
            }
        else:
            row = {
                "phase": sp["phase"],
                "family": LABELS[sp["family_id"]],
                "family_id": sp["family_id"],
                "K": len(GROUPS[sp["family_id"]]),
                "n": sp["n"],
                "lambda": f"{sp['lambda']:.1f}",
                "seed": sp["seed"],
                "runtime_sec": raw.get("runtime_sec", f"{wall:.4f}"),
                "ub": raw.get("ub", ""),
                "lb": raw.get("lb", ""),
                "gap_pct": raw.get("gap_pct", ""),
                "timed_out": raw.get("timed_out", ""),
                "is_optimal": raw.get("is_optimal", ""),
                "feasible": raw.get("feasible", ""),
                "deciding_step_raw": deciding_step(raw),
                "deciding_step": deciding_step(raw),
                "step3_mode": step3_mode(raw),
                "exact_dp_entered": str(exact_entered(raw)),
                "main_pack_method": raw.get("fwd_pack_method", ""),
                "selector_policy": raw.get("fwd_profile_selector_policy", ""),
                "selector_decision": raw.get("fwd_profile_selector_decision", ""),
                "selector_reason": raw.get("fwd_profile_selector_reason", ""),
                "winner_detail_raw": raw.get("winner_detail", ""),
                "winner_detail": raw.get("winner_detail", ""),
                "fwd_pack_outcome": raw.get("fwd_pack_outcome", ""),
                "exact_diag_mode": raw.get("exact_diag_mode", ""),
                "instance_id": raw.get("instance_id", payload["instance_id"]),
                "horizon": raw.get("horizon", str(len(payload["prices"]))),
                "machine": "twosby",
                "ec_from": sp["ec_from"],
                "ec_repeat": sp["ec_repeat"],
                "solver_mode": "ablation-stdin step1_exact_guided",
                "time_limit_sec": sp["time_limit_sec"],
                "solver_returncode": str(rc),
                "stderr_tail": stderr_tail,
                "diag_step1_decided": raw.get("diag_step1_decided", ""),
                "diag_step2_decided": raw.get("diag_step2_decided", ""),
                "diag_step3_decided": raw.get("diag_step3_decided", ""),
                "diag_step4_decided": raw.get("diag_step4_decided", ""),
                "diag_exact_dp_used": raw.get("diag_exact_dp_used", ""),
                "fwd_profile_step3_incumbent_mode": raw.get(
                    "fwd_profile_step3_incumbent_mode", ""
                ),
            }

        row = normalize_result_row(row)

        rows_out.append(row)
        write_rows(out_csv, [row])
        print(
            f"[{i}/{total}] {sp['phase']} {LABELS[sp['family_id']]} n={sp['n']} lam={sp['lambda']:.1f} s={sp['seed']} "
            f"gap={row.get('gap_pct', '')} step={row.get('deciding_step', '')} t={row.get('runtime_sec', '')}"
        )
    return rows_out


def aggregate_rows(
    rows: list[dict[str, Any]], group_fields: list[str]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[tuple(r.get(k, "") for k in group_fields)].append(r)

    out: list[dict[str, Any]] = []
    for key, chunk in grouped.items():
        runtimes = [
            safe_float(r.get("runtime_sec"))
            for r in chunk
            if math.isfinite(safe_float(r.get("runtime_sec")))
        ]
        gaps = [
            safe_float(r.get("gap_pct"))
            for r in chunk
            if math.isfinite(safe_float(r.get("gap_pct")))
        ]
        exact_count = sum(1 for r in chunk if r.get("is_optimal") == "1")
        timeout_count = sum(1 for r in chunk if r.get("timed_out") == "1")
        finite_count = sum(
            1
            for r in chunk
            if math.isfinite(safe_float(r.get("ub")))
            and math.isfinite(safe_float(r.get("lb")))
            and safe_float(r.get("ub")) >= 0
            and safe_float(r.get("lb")) >= 0
        )
        step_mode = Counter(
            r.get("deciding_step", "") for r in chunk if r.get("deciding_step", "")
        )
        dominant_step = step_mode.most_common(1)[0][0] if step_mode else ""

        row = {k: v for k, v in zip(group_fields, key)}
        row.update(
            {
                "runs": len(chunk),
                "exact_count": exact_count,
                "timeout_count": timeout_count,
                "finite_count": finite_count,
                "avg_runtime_sec": f"{statistics.mean(runtimes):.4f}"
                if runtimes
                else "",
                "max_runtime_sec": f"{max(runtimes):.4f}" if runtimes else "",
                "avg_gap_pct": f"{statistics.mean(gaps):.6f}" if gaps else "",
                "max_gap_pct": f"{max(gaps):.6f}" if gaps else "",
                "dominant_deciding_step": dominant_step,
            }
        )
        out.append(row)
    return out


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def classify_families(
    n_rows: list[dict[str, str]],
    lambda_rows: list[dict[str, str]],
    tiny_gap_pct: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    by_family_n: dict[str, dict[int, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in n_rows:
        by_family_n[r["family_id"]][int(r["n"])].append(r)

    by_family_lambda: dict[str, dict[float, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in lambda_rows:
        by_family_lambda[r["family_id"]][float(r["lambda"])].append(r)

    for family_id in GROUPS.keys():
        fam_n = by_family_n.get(family_id, {})
        tested_ns = sorted(fam_n.keys())
        if not tested_ns:
            continue

        largest_tested_n = max(tested_ns)
        largest_exact_n = -1
        largest_tiny_gap_n = -1

        step_counter = Counter()
        step4_count = 0
        total_count = 0
        for n in tested_ns:
            chunk = fam_n[n]
            all_exact = True
            all_finite_tiny = True
            for r in chunk:
                total_count += 1
                step = r.get("deciding_step", "")
                if step:
                    step_counter[step] += 1
                if step == "step4":
                    step4_count += 1

                if r.get("is_optimal") != "1":
                    all_exact = False
                g = safe_float(r.get("gap_pct"))
                ub = safe_float(r.get("ub"))
                lb = safe_float(r.get("lb"))
                if not (
                    math.isfinite(g)
                    and math.isfinite(ub)
                    and math.isfinite(lb)
                    and ub >= 0
                    and lb >= 0
                    and g <= tiny_gap_pct
                ):
                    all_finite_tiny = False

            if all_exact:
                largest_exact_n = n
            if all_finite_tiny:
                largest_tiny_gap_n = n

        dominant_step = step_counter.most_common(1)[0][0] if step_counter else ""

        lam13 = by_family_lambda.get(family_id, {}).get(1.3, [])
        lam30 = by_family_lambda.get(family_id, {}).get(3.0, [])
        lambda_effect = "mixed"
        if lam13 and lam30:
            rt13 = statistics.mean(
                [
                    safe_float(r.get("runtime_sec"))
                    for r in lam13
                    if math.isfinite(safe_float(r.get("runtime_sec")))
                ]
            )
            rt30 = statistics.mean(
                [
                    safe_float(r.get("runtime_sec"))
                    for r in lam30
                    if math.isfinite(safe_float(r.get("runtime_sec")))
                ]
            )
            g13_vals = [
                safe_float(r.get("gap_pct"))
                for r in lam13
                if math.isfinite(safe_float(r.get("gap_pct")))
            ]
            g30_vals = [
                safe_float(r.get("gap_pct"))
                for r in lam30
                if math.isfinite(safe_float(r.get("gap_pct")))
            ]
            g13 = statistics.mean(g13_vals) if g13_vals else float("nan")
            g30 = statistics.mean(g30_vals) if g30_vals else float("nan")

            rt_ratio = (
                (rt30 / rt13)
                if (math.isfinite(rt13) and rt13 > 0 and math.isfinite(rt30))
                else float("nan")
            )
            if math.isfinite(g13) and math.isfinite(g30) and math.isfinite(rt_ratio):
                if g30 <= g13 and rt_ratio <= 1.35:
                    lambda_effect = "helps_or_neutral"
                elif g30 > g13 or rt_ratio > 1.35:
                    lambda_effect = "hurts"

        step4_share = (step4_count / total_count) if total_count else 0.0
        if (
            largest_exact_n == largest_tested_n
            and dominant_step in ("step1", "step2")
            and step4_share < 0.2
        ):
            cls = "easy-scalable"
        elif (
            step4_share >= 0.4
            and largest_exact_n < largest_tested_n
            and largest_tiny_gap_n == largest_tested_n
        ):
            cls = "step4-limited"
        else:
            cls = "step3-dominated but practical"

        out.append(
            {
                "family": LABELS[family_id],
                "family_id": family_id,
                "K": len(GROUPS[family_id]),
                "largest_tested_n_lambda_1p3": largest_tested_n,
                "largest_exact_n_all_seeds": largest_exact_n
                if largest_exact_n >= 0
                else "",
                "largest_finite_tiny_gap_n_all_seeds": largest_tiny_gap_n
                if largest_tiny_gap_n >= 0
                else "",
                "dominant_deciding_step": dominant_step,
                "step4_share": f"{100.0 * step4_share:.1f}%",
                "lambda_effect": lambda_effect,
                "classification": cls,
            }
        )

    return out


def build_phase_specs(
    seeds_baseline: list[int],
    seeds_n: list[int],
    seeds_lambda: list[int],
    baseline_lambdas: list[float],
    n_values_phase2: list[int],
    n_values_large: list[int],
    time_limit_baseline: float,
    time_limit_n: float,
    time_limit_n_large: float,
    time_limit_lambda: float,
    run_lambda_families: list[str],
    lambda_representative_n: dict[str, list[int]],
    lambda_values_phase3: list[float],
    easy_extend_families: list[str],
    phase2_families: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_specs: list[dict[str, Any]] = []
    n_specs: list[dict[str, Any]] = []
    lambda_specs: list[dict[str, Any]] = []

    for family_id in GROUPS.keys():
        for n in [50, 100, 150, 200]:
            for lam in baseline_lambdas:
                for seed in seeds_baseline:
                    ec = EC_CONFIGS[seed % len(EC_CONFIGS)]
                    baseline_specs.append(
                        {
                            "phase": "phase1_baseline",
                            "family_id": family_id,
                            "n": n,
                            "lambda": lam,
                            "seed": seed,
                            "ec_from": ec["from_date"],
                            "ec_repeat": ec["repeat_count"],
                            "time_limit_sec": time_limit_baseline,
                        }
                    )

    for family_id in phase2_families:
        for n in n_values_phase2:
            for seed in seeds_n:
                ec = EC_CONFIGS[seed % len(EC_CONFIGS)]
                n_specs.append(
                    {
                        "phase": "phase2_n_core",
                        "family_id": family_id,
                        "n": n,
                        "lambda": 1.3,
                        "seed": seed,
                        "ec_from": ec["from_date"],
                        "ec_repeat": ec["repeat_count"],
                        "time_limit_sec": time_limit_n,
                    }
                )

    for family_id in easy_extend_families:
        if family_id not in phase2_families:
            continue
        for n in n_values_large:
            for seed in seeds_n:
                ec = EC_CONFIGS[seed % len(EC_CONFIGS)]
                n_specs.append(
                    {
                        "phase": "phase2_n_extended",
                        "family_id": family_id,
                        "n": n,
                        "lambda": 1.3,
                        "seed": seed,
                        "ec_from": ec["from_date"],
                        "ec_repeat": ec["repeat_count"],
                        "time_limit_sec": time_limit_n_large,
                    }
                )

    for family_id in run_lambda_families:
        reps = lambda_representative_n.get(family_id, [])
        for n in reps:
            for lam in lambda_values_phase3:
                for seed in seeds_lambda:
                    ec = EC_CONFIGS[seed % len(EC_CONFIGS)]
                    lambda_specs.append(
                        {
                            "phase": "phase3_lambda",
                            "family_id": family_id,
                            "n": n,
                            "lambda": lam,
                            "seed": seed,
                            "ec_from": ec["from_date"],
                            "ec_repeat": ec["repeat_count"],
                            "time_limit_sec": time_limit_lambda,
                        }
                    )
    return baseline_specs, n_specs, lambda_specs


def easy_family_detection(
    n_rows: list[dict[str, str]], seeds_n: list[int]
) -> list[str]:
    by_family_n: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in n_rows:
        if int(r.get("n", "0")) == 1000:
            by_family_n[r["family_id"]].append(r)

    easy: list[str] = []
    for family_id in GROUPS.keys():
        rows = by_family_n.get(family_id, [])
        if len(rows) < len(seeds_n):
            continue
        all_exact = all(r.get("is_optimal") == "1" for r in rows)
        avg_rt = statistics.mean([safe_float(r.get("runtime_sec")) for r in rows])
        if all_exact and avg_rt <= 120.0:
            easy.append(family_id)
    return easy


def representative_ns_for_lambda(
    n_rows: list[dict[str, str]], fallback_groups: list[str]
) -> dict[str, list[int]]:
    by_family = defaultdict(set)
    for r in n_rows:
        by_family[r["family_id"]].add(int(r["n"]))

    reps: dict[str, list[int]] = {}
    for family_id in fallback_groups:
        ns = sorted(by_family.get(family_id, set()))
        if not ns:
            continue
        small = 100 if 100 in ns else ns[0]
        medium = 500 if 500 in ns else ns[min(len(ns) - 1, max(0, len(ns) // 2))]
        if 1500 in ns:
            large = 1500
        elif 1000 in ns:
            large = 1000
        else:
            large = ns[-1]
        reps[family_id] = sorted({small, medium, large})
    return reps


def write_summary_markdown(
    baseline_summary: list[dict[str, Any]],
    n_summary: list[dict[str, Any]],
    lambda_summary: list[dict[str, Any]],
    final_summary: list[dict[str, Any]],
    seeds_baseline: list[int],
    seeds_n: list[int],
    seeds_lambda: list[int],
    lambda_scope_note: str,
) -> None:
    lines: list[str] = []
    lines.append("# PAPER GROUPS EXTENSION SUMMARY (PLAN 05)")
    lines.append("")
    lines.append(
        "This note reports the clean extension campaign on the seven Section-5.2 processing-time groups under the current cleaned 4-step policy."
    )
    lines.append("")
    lines.append("## Seed Policy")
    lines.append("")
    lines.append(f"- Phase 1 baseline seeds: `{seeds_baseline}`")
    lines.append(f"- Phase 2 n-extension seeds: `{seeds_n}`")
    lines.append(f"- Phase 3 lambda-extension seeds: `{seeds_lambda}`")
    lines.append(
        "- Energy profile source dates are deterministic per seed and fixed to OTE 2019 (`from_date` in CSV)."
    )
    lines.append("")
    lines.append("## Lambda Phase Scope")
    lines.append("")
    lines.append(f"- {lambda_scope_note}")
    lines.append("")
    lines.append("## Final Classification Table")
    lines.append("")
    lines.append(
        "| Family | K | Largest n@lambda=1.3 | Largest exact n | Largest finite tiny-gap n | Dominant step | Lambda effect | Classification |"
    )
    lines.append("|---|---:|---:|---:|---:|---|---|---|")
    for r in final_summary:
        lines.append(
            "| "
            + f"{r['family']} | {r['K']} | {r['largest_tested_n_lambda_1p3']} | {r.get('largest_exact_n_all_seeds', '')} | "
            + f"{r.get('largest_finite_tiny_gap_n_all_seeds', '')} | {r['dominant_deciding_step']} | {r['lambda_effect']} | {r['classification']} |"
        )

    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Baseline CSV: `{BASELINE_CSV}`")
    lines.append(f"- n-extension CSV: `{N_EXT_CSV}`")
    lines.append(f"- lambda-extension CSV: `{LAMBDA_EXT_CSV}`")
    lines.append(f"- Final summary CSV: `{FINAL_SUMMARY_CSV}`")

    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run Plan 05 paper-groups extension campaign"
    )
    ap.add_argument(
        "--phase",
        choices=["all", "baseline", "n", "lambda", "summaries"],
        default="all",
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--baseline-seeds", default="0,1")
    ap.add_argument("--n-seeds", default="0,1")
    ap.add_argument("--lambda-seeds", default="0,1")
    ap.add_argument("--baseline-lambdas", default="1.3,1.6,1.9,2.2")
    ap.add_argument("--phase2-n-values", default="300,400,500,600,750,1000")
    ap.add_argument("--phase2-n-large", default="1500,2500,3500,5000")
    ap.add_argument(
        "--phase2-families",
        default="",
        help="Optional comma-separated subset of family IDs for Phase-2 runs; default is all plan order families.",
    )
    ap.add_argument(
        "--force-extend-families",
        default="",
        help="Comma-separated family IDs to always include in large-n extension (e.g., g12345678910,g3567).",
    )
    ap.add_argument("--phase3-lambdas", default="1.3,1.6,1.9,2.2,2.5,3.0")
    ap.add_argument("--time-limit-baseline", type=float, default=240.0)
    ap.add_argument("--time-limit-n", type=float, default=300.0)
    ap.add_argument("--time-limit-n-large", type=float, default=600.0)
    ap.add_argument("--time-limit-lambda", type=float, default=300.0)
    ap.add_argument(
        "--lambda-scope",
        choices=["all7", "fallback3"],
        default="fallback3",
        help="Use full 7-family lambda grid or fallback 3 representative families.",
    )
    ap.add_argument("--tiny-gap-pct", type=float, default=0.05)
    args = ap.parse_args()

    if not SOLVER.exists():
        raise FileNotFoundError(f"Solver not found: {SOLVER}")

    seeds_baseline = parse_ints(args.baseline_seeds)
    seeds_n = parse_ints(args.n_seeds)
    seeds_lambda = parse_ints(args.lambda_seeds)

    baseline_lambdas = parse_floats(args.baseline_lambdas)
    n_values_phase2 = parse_ints(args.phase2_n_values)
    n_values_large = parse_ints(args.phase2_n_large)
    lambda_values_phase3 = parse_floats(args.phase3_lambdas)
    force_extend_families = parse_strs(args.force_extend_families)
    phase2_families = (
        parse_strs(args.phase2_families) if args.phase2_families else list(PHASE2_ORDER)
    )
    phase2_families = [f for f in phase2_families if f in GROUPS]

    if args.phase in ("all", "baseline"):
        baseline_specs, _, _ = build_phase_specs(
            seeds_baseline=seeds_baseline,
            seeds_n=seeds_n,
            seeds_lambda=seeds_lambda,
            baseline_lambdas=baseline_lambdas,
            n_values_phase2=n_values_phase2,
            n_values_large=n_values_large,
            time_limit_baseline=args.time_limit_baseline,
            time_limit_n=args.time_limit_n,
            time_limit_n_large=args.time_limit_n_large,
            time_limit_lambda=args.time_limit_lambda,
            run_lambda_families=[],
            lambda_representative_n={},
            lambda_values_phase3=lambda_values_phase3,
            easy_extend_families=[],
            phase2_families=phase2_families,
        )
        print(f"Running baseline rows: {len(baseline_specs)}")
        run_specs(baseline_specs, BASELINE_CSV, resume=args.resume)

    if args.phase in ("all", "n"):
        n_rows_existing = read_csv(N_EXT_CSV)
        easy_families = easy_family_detection(n_rows_existing, seeds_n)
        for fam in force_extend_families:
            if fam in GROUPS and fam not in easy_families:
                easy_families.append(fam)
        if not easy_families:
            # Conservative initial guess before first n-core run finishes.
            easy_families = ["g12345678910", "g12357", "g246810", "g24", "g3567", "g37"]
        easy_families = [f for f in easy_families if f in phase2_families]

        _, n_specs, _ = build_phase_specs(
            seeds_baseline=seeds_baseline,
            seeds_n=seeds_n,
            seeds_lambda=seeds_lambda,
            baseline_lambdas=baseline_lambdas,
            n_values_phase2=n_values_phase2,
            n_values_large=n_values_large,
            time_limit_baseline=args.time_limit_baseline,
            time_limit_n=args.time_limit_n,
            time_limit_n_large=args.time_limit_n_large,
            time_limit_lambda=args.time_limit_lambda,
            run_lambda_families=[],
            lambda_representative_n={},
            lambda_values_phase3=lambda_values_phase3,
            easy_extend_families=easy_families,
            phase2_families=phase2_families,
        )
        print(
            f"Running n-extension rows: {len(n_specs)} (easy extensions={easy_families})"
        )
        run_specs(n_specs, N_EXT_CSV, resume=args.resume)

    if args.phase in ("all", "lambda"):
        n_rows = read_csv(N_EXT_CSV)

        if args.lambda_scope == "all7":
            run_lambda_families = list(GROUPS.keys())
            lambda_scope_note = "Phase 3 run on all 7 paper groups."
        else:
            run_lambda_families = ["g12345678910", "g3567", "g810"]
            lambda_scope_note = (
                "Fallback rule triggered: Phase 2 all-family n<=1000 completed; "
                "Phase 3 run on 3 representative groups {1..10}, {3,5,6,7}, {8,10}."
            )

        reps = representative_ns_for_lambda(n_rows, run_lambda_families)
        _, _, lambda_specs = build_phase_specs(
            seeds_baseline=seeds_baseline,
            seeds_n=seeds_n,
            seeds_lambda=seeds_lambda,
            baseline_lambdas=baseline_lambdas,
            n_values_phase2=n_values_phase2,
            n_values_large=n_values_large,
            time_limit_baseline=args.time_limit_baseline,
            time_limit_n=args.time_limit_n,
            time_limit_n_large=args.time_limit_n_large,
            time_limit_lambda=args.time_limit_lambda,
            run_lambda_families=run_lambda_families,
            lambda_representative_n=reps,
            lambda_values_phase3=lambda_values_phase3,
            easy_extend_families=[],
            phase2_families=phase2_families,
        )

        print(
            f"Running lambda-extension rows: {len(lambda_specs)} families={run_lambda_families} reps={reps}"
        )
        run_specs(lambda_specs, LAMBDA_EXT_CSV, resume=args.resume)

        # Persist note for summary-generation pass.
        (OUT_DIR / "PAPER_GROUPS_PLAN05_lambda_scope.txt").write_text(
            lambda_scope_note + "\n", encoding="utf-8"
        )

    if args.phase in ("all", "summaries"):
        baseline_rows = read_csv(BASELINE_CSV)
        n_rows = read_csv(N_EXT_CSV)
        lambda_rows = read_csv(LAMBDA_EXT_CSV)

        baseline_summary = aggregate_rows(
            baseline_rows, ["family", "family_id", "K", "n", "lambda"]
        )
        n_summary = aggregate_rows(
            n_rows, ["family", "family_id", "K", "n", "lambda", "phase"]
        )
        lambda_summary = aggregate_rows(
            lambda_rows, ["family", "family_id", "K", "n", "lambda"]
        )
        final_summary = classify_families(n_rows, lambda_rows, args.tiny_gap_pct)

        write_csv(BASELINE_SUMMARY_CSV, baseline_summary)
        write_csv(N_EXT_SUMMARY_CSV, n_summary)
        write_csv(LAMBDA_EXT_SUMMARY_CSV, lambda_summary)
        write_csv(FINAL_SUMMARY_CSV, final_summary)

        scope_note_path = OUT_DIR / "PAPER_GROUPS_PLAN05_lambda_scope.txt"
        lambda_scope_note = (
            scope_note_path.read_text(encoding="utf-8").strip()
            if scope_note_path.exists()
            else ""
        )
        write_summary_markdown(
            baseline_summary=baseline_summary,
            n_summary=n_summary,
            lambda_summary=lambda_summary,
            final_summary=final_summary,
            seeds_baseline=seeds_baseline,
            seeds_n=seeds_n,
            seeds_lambda=seeds_lambda,
            lambda_scope_note=lambda_scope_note,
        )

        print(
            f"Wrote summaries:\n- {BASELINE_SUMMARY_CSV}\n- {N_EXT_SUMMARY_CSV}\n- {LAMBDA_EXT_SUMMARY_CSV}\n- {FINAL_SUMMARY_CSV}\n- {SUMMARY_MD}"
        )


if __name__ == "__main__":
    main()

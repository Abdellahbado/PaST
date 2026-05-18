#!/usr/bin/env python3
"""
PLAN18: targeted irregular K-boundary refinement at fixed n=1000.

Focus: hard irregular ladder A and hard irregular ladder B.
K values: 8, 10, 12 (skip 14 unless needed to resolve 12-vs-16 break).
Seeds: 0, 1, 2, 3.
Budget: memory cap 16 GB, one heavy row at a time, external timeout 1320s.
Routes: baseline energy_core + additive profile_repair_beam/auto_v1 reroute.

Produces:
- PLAN18_k_boundary_refine_n1000_raw.csv
- PLAN18_k_boundary_refine_best_of_route.csv
- PLAN18_k_boundary_refine_summary_by_k.csv
- PLAN18_k_boundary_refine_failure_signatures.csv
"""

from __future__ import annotations

import csv
import math
import random
import statistics
import sys
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

OUT_DIR = ROOT / "research" / "k_vs_arithmetic_axes_20260412" / "csv" / "plan18"
RAW_CSV = OUT_DIR / "PLAN18_k_boundary_refine_n1000_raw.csv"
BEST_CSV = OUT_DIR / "PLAN18_k_boundary_refine_best_of_route.csv"
SUM_K_CSV = OUT_DIR / "PLAN18_k_boundary_refine_summary_by_k.csv"
FAIL_CSV = OUT_DIR / "PLAN18_k_boundary_refine_failure_signatures.csv"

LAMBDA = 1.3
N_JOBS = 1000
SEEDS = (0, 1, 2, 3)
TIME_LIMIT = 1200.0
MAX_RSS_GB = 12.0
EXTERNAL_TIMEOUT = int(max(240, TIME_LIMIT + 120))

# Rows to force rerun (suspicious or missing from prior partial run)
FORCE_RERUN: set[tuple[str, str, int]] = set()

EC_CONFIGS = [
    {"from_date": "2019-01-21T00:00:00", "repeat_count": 1},
    {"from_date": "2019-04-08T00:00:00", "repeat_count": 1},
]

IRREGULAR_REROUTE_ENV: dict[str, str] = {
    "PAST_RELAXED_BINPACK_SOLVER": "profile_repair_beam",
    "PAST_PROFILE_REALIZATION_SELECTOR_POLICY": "auto_v1",
}

HARD_A_BASE = [
    2, 3, 4, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67,
]

HARD_B_BASE = [
    3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
]


def sizes_label(sizes: list[int]) -> str:
    return "{" + ",".join(str(x) for x in sizes) + "}"


def build_plan18_payload(
    family_id: str,
    sizes: list[int],
    n_jobs: int,
    lam: float,
    seed: int,
) -> dict[str, Any]:
    ec = EC_CONFIGS[seed % len(EC_CONFIGS)]
    rng = random.Random(stable_seed(family_id, n_jobs, lam, seed))
    jobs = [rng.choice(sizes) for _ in range(n_jobs)]
    label = sizes_label(sizes)
    inst = build_instance(
        name=f"plan18/{family_id}_n{n_jobs}_lam{lam:.1f}_s{seed}",
        family=family_id,
        jobs_list=jobs,
        horizon_multiplier=lam,
        ec_config=ec,
        metadata={
            "processing_group": sizes,
            "K": len(sizes),
            "seed": seed,
            "lambda": lam,
            "paper_group": label,
            "paper_machine": "twosby",
            "plan18": "1",
        },
    )
    return {
        "instance_id": inst["name"],
        "prices": inst["prices"],
        "jobs": inst["jobs"],
        "machine": "twosby",
    }


def gcd_list(values: list[int]) -> int:
    g = abs(values[0])
    for x in values[1:]:
        g = math.gcd(g, abs(x))
    return g


def semigroup_density_prefix_100(sizes: list[int]) -> float:
    if not sizes:
        return float("nan")
    m = min(sizes)
    if m <= 0:
        return float("nan")
    ok = [False] * 101
    ok[0] = True
    for t in range(1, 101):
        for s in sizes:
            if s <= t and ok[t - s]:
                ok[t] = True
                break
    return sum(ok[1:101]) / 100.0


def arithmetic_descriptors(sizes: list[int]) -> dict[str, Any]:
    s_sorted = sorted(set(sizes))
    has_one = 1 if 1 in sizes else 0
    lo, hi = min(sizes), max(sizes)
    contiguous = 1 if sizes == list(range(lo, hi + 1)) else 0
    gaps = [b - a for a, b in zip(s_sorted, s_sorted[1:])] if len(s_sorted) > 1 else []
    max_gap = max(gaps) if gaps else 0
    mean_gap = statistics.mean(gaps) if gaps else 0.0
    return {
        "has_one": has_one,
        "contiguous": contiguous,
        "gcd": gcd_list(sizes),
        "min_size": lo,
        "max_size": hi,
        "range_width": hi - lo,
        "max_gap": max_gap,
        "mean_gap": f"{mean_gap:.6f}",
        "semigroup_density_prefix_100": f"{semigroup_density_prefix_100(sizes):.6f}",
    }


def _parse_rc(row: dict[str, Any]) -> int:
    try:
        return int(float(str(row.get("solver_returncode", "0"))))
    except Exception:
        return 0


def classify_boundary(row: dict[str, Any], k: int) -> tuple[str, str]:
    mem = str(row.get("memory_killed", "0")) == "1"
    ext = str(row.get("external_timed_out", "0")) == "1"
    rc = _parse_rc(row)
    sel = (row.get("selector_reason") or "").strip()
    ds = (row.get("deciding_step") or "").strip()
    opt = str(row.get("is_optimal", "0")) == "1"
    ub = str(row.get("ub", "-1")).strip()
    step3_mode = (row.get("step3_mode") or "").strip().lower()
    method = (row.get("fwd_pack_method") or "").strip().lower()

    if mem or ds == "memory_limit_kill":
        return "memory_failure", "rss_cap"
    if rc == -6 or rc == -11:
        return "crash", f"returncode={rc}"
    if k == 2 and sel == "non_mainline_solver":
        return "misrouted", "k2_energy_core_bypass"
    if ext or ds in ("external_timeout", "timeout"):
        if ub == "-1" or ub == "":
            return "timeout_no_incumbent", ds or "timeout"
        return "timeout_no_incumbent", "timeout_with_ub"
    if ds == "no_csv_row":
        return "unresolved_other", "no_csv_row"

    if opt:
        if ds == "step2":
            return "easy_step2_exact", ""
        if ds == "step3":
            if step3_mode == "exact":
                return "step3_exact", ""
            if step3_mode == "beam":
                return "step3_beam_exact", ""
            if method == "profile_repair_beam":
                return "step3_beam_exact", "fwd_pack_method=profile_repair_beam"
            return "step3_exact", f"fwd_pack_method={method or 'unknown'}"
        if ds == "step4":
            return "unresolved_other", "closed_step4_global_exact"
        if ds == "step1":
            return "easy_step2_exact", "closed_step1"
        return "unresolved_other", f"optimal_unexpected_step={ds}"

    try:
        gap = float(str(row.get("gap_pct", "nan")))
    except Exception:
        gap = float("nan")
    if ub not in ("-1", "") and str(row.get("lb", "-1")).strip() not in ("-1", ""):
        if not math.isnan(gap) and gap > 1e-6:
            return "finite_gap", f"gap_pct={gap}"
        if not math.isnan(gap):
            return "unresolved_other", f"nonopt_small_gap={gap}"
    return "unresolved_other", "nonopt_no_class"


def worse_class(a: str, b: str) -> str:
    order = [
        "misrouted",
        "crash",
        "memory_failure",
        "timeout_no_incumbent",
        "finite_gap",
        "unresolved_other",
        "step3_beam_exact",
        "step3_exact",
        "easy_step2_exact",
    ]
    ia = order.index(a) if a in order else len(order)
    ib = order.index(b) if b in order else len(order)
    return a if ia <= ib else b


def normalize_output_row(
    base: dict[str, Any],
    *,
    family_id: str,
    family_label: str,
    family_class: str,
    family_sizes: list[int],
    k: int,
    variant_label: str,
    route_policy: str,
) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    desc = arithmetic_descriptors(family_sizes)
    out["family_id"] = family_id
    out["family_label"] = family_label
    out["family_class"] = family_class
    out["family_sizes"] = ",".join(str(x) for x in family_sizes)
    out["K"] = str(k)
    out["lambda"] = str(LAMBDA)
    out["variant_label"] = variant_label
    out["route_policy"] = route_policy
    out["wall_runtime_sec"] = out.get("runtime_wall_sec", "")
    out["external_timeout"] = str(out.get("external_timed_out", out.get("external_timeout", "0")))
    out["peak_rss_gb"] = out.get("peak_rss_gb", "")
    for kdesc, v in desc.items():
        out[kdesc] = str(v)
    out["arithmetic_class_label"] = (
        "irregular_primes_from2"
        if family_class == "hard_irregular_A"
        else "irregular_primes_from3"
    )
    bc, bnote = classify_boundary(out, k)
    out["boundary_class"] = bc
    out["boundary_detail"] = bnote
    return out


def refresh_derived_fields(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["wall_runtime_sec"] = row.get("runtime_wall_sec", row.get("wall_runtime_sec", ""))
        row["external_timeout"] = str(row.get("external_timed_out", row.get("external_timeout", "0")))
        try:
            k = int(str(row.get("K", "0")))
        except Exception:
            k = 0
        bc, bnote = classify_boundary(row, k)
        row["boundary_class"] = bc
        row["boundary_detail"] = bnote


def load_raw(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def row_key(r: dict[str, Any]) -> tuple[str, str, int]:
    try:
        s = int(r.get("seed", -1))
    except Exception:
        s = -1
    return (str(r.get("family_id", "")), str(r.get("variant_label", "")), s)


def run_hard_ladder(
    prefix: str,
    family_class: str,
    base: list[int],
    ks: list[int],
    rows: list[dict[str, Any]],
    seen: set[tuple[str, str, int]],
) -> None:
    for target_k in ks:
        sizes = base[:target_k]
        fid = f"{prefix}_k{target_k}"
        label = sizes_label(sizes)
        k = len(sizes)

        # Baseline for all seeds
        for seed in SEEDS:
            key = (fid, "baseline", seed)
            if key in seen:
                continue
            payload = build_plan18_payload(fid, sizes, N_JOBS, LAMBDA, seed)
            raw = run_row(
                fid,
                N_JOBS,
                seed,
                TIME_LIMIT,
                "baseline",
                {},
                max_rss_gb=MAX_RSS_GB,
                payload=payload,
            )
            row = normalize_output_row(
                raw,
                family_id=fid,
                family_label=label,
                family_class=family_class,
                family_sizes=sizes,
                k=k,
                variant_label="baseline",
                route_policy="plan18:accepted_energy_core_baseline",
            )
            rows.append(row)
            seen.add(key)
            write_csv(RAW_CSV, rows)
            print(
                f"[{prefix}] {fid} baseline seed={seed} K={k} "
                f"step={row.get('deciding_step')} opt={row.get('is_optimal')} "
                f"bc={row.get('boundary_class')} rt={row.get('runtime_sec')}"
            )

        # Irregular reroute for all seeds
        for seed in SEEDS:
            key = (fid, "irregular_reroute", seed)
            if key in seen:
                continue
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
                family_class=family_class,
                family_sizes=sizes,
                k=k,
                variant_label="irregular_reroute",
                route_policy="plan18:additive_profile_repair_beam_auto_v1",
            )
            rows.append(row)
            seen.add(key)
            write_csv(RAW_CSV, rows)
            print(
                f"[{prefix}] {fid} irregular_reroute seed={seed} K={k} "
                f"step={row.get('deciding_step')} opt={row.get('is_optimal')} "
                f"bc={row.get('boundary_class')} rt={row.get('runtime_sec')}"
            )


def build_best_of_route(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    For each (family_id, K, seed), pick best route:
    - exact beats non-exact
    - smaller gap beats larger gap
    - feasible incumbent beats no incumbent
    - lower runtime breaks ties
    """
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for r in all_rows:
        try:
            k = int(r.get("K", "0"))
            seed = int(r.get("seed", "-1"))
        except Exception:
            continue
        fid = str(r.get("family_id", ""))
        if not fid:
            continue
        groups.setdefault((fid, k, seed), []).append(r)

    out: list[dict[str, Any]] = []
    for (fid, k, seed), rs in sorted(groups.items()):
        def score(r: dict[str, Any]) -> tuple[int, float, int, float]:
            opt = 1 if str(r.get("is_optimal", "0")) == "1" else 0
            try:
                ub = float(str(r.get("ub", "-1")))
                lb = float(str(r.get("lb", "-1")))
                has_incumbent = 1 if (ub >= 0 and lb >= 0) else 0
            except Exception:
                has_incumbent = 0
            try:
                gap = float(str(r.get("gap_pct", "nan")))
                if math.isnan(gap) or not has_incumbent:
                    gap = float("inf")
            except Exception:
                gap = float("inf")
            try:
                rt = float(str(r.get("runtime_sec", "inf")))
            except Exception:
                rt = float("inf")
            return (-opt, gap, -has_incumbent, rt)

        best = min(rs, key=score)
        out.append(dict(best))
    return out


def classify_failure_signature(row: dict[str, Any]) -> str:
    """
    Classify the dominant failure mode for a non-exact best-of-route row.
    """
    mem = str(row.get("memory_killed", "0")) == "1"
    rc = _parse_rc(row)
    ext = str(row.get("external_timed_out", "0")) == "1"
    ds = (row.get("deciding_step") or "").strip().lower()
    ub = str(row.get("ub", "-1")).strip()
    step3_mode = (row.get("step3_mode") or "").strip().lower()

    if mem:
        return "memory_failure"
    if rc == -6 or rc == -11:
        return "crash"
    if ext or ds in ("external_timeout", "timeout"):
        if ub == "-1" or ub == "":
            return "no_incumbent_timeout"
        return "no_incumbent_timeout"  # timeout with UB still means timeout
    if ds == "step3":
        try:
            gap = float(str(row.get("gap_pct", "nan")))
            if not math.isnan(gap) and gap > 1e-6:
                return "finite_gap_after_step3"
        except Exception:
            pass
    if ds == "step4":
        try:
            gap = float(str(row.get("gap_pct", "nan")))
            if not math.isnan(gap) and gap > 1e-6:
                return "finite_gap_after_step4"
        except Exception:
            pass
    sel_reason = (row.get("selector_reason") or "").strip()
    if sel_reason in ("non_mainline_solver", "bypass"):
        return "selector_bypass"
    fp_outcome = (row.get("fwd_pack_outcome") or "").strip().lower()
    if fp_outcome == "failed" or ds == "pack_failed":
        return "pack_failed_after_relax"
    return "unresolved_other"


def build_summary_by_k(best_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_kfc: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for r in best_rows:
        try:
            k = int(r.get("K", "-1"))
        except Exception:
            continue
        fc = str(r.get("family_class", ""))
        by_kfc.setdefault((k, fc), []).append(r)

    out: list[dict[str, Any]] = []
    for (k, fc), rs in sorted(by_kfc.items()):
        exact = sum(1 for r in rs if str(r.get("is_optimal", "0")) == "1")
        n = len(rs)
        rts = []
        for r in rs:
            try:
                rts.append(float(str(r.get("runtime_sec", "nan"))))
            except Exception:
                pass
        worst = "easy_step2_exact"
        for r in rs:
            worst = worse_class(worst, str(r.get("boundary_class", "unresolved_other")))
        out.append({
            "K": str(k),
            "family_class": fc,
            "n_rows": str(n),
            "exact_rows": f"{exact}/{n}",
            "boundary_worst_of_rows": worst,
            "mean_runtime_sec": f"{statistics.mean(rts):.3f}" if rts else "",
            "family_ids_sample": ",".join(sorted({str(r.get('family_id')) for r in rs})[:5]),
        })
    return out


def build_failure_signatures(best_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in best_rows:
        if str(r.get("is_optimal", "0")) == "1":
            continue
        sig = classify_failure_signature(r)
        out.append({
            "family_id": str(r.get("family_id", "")),
            "K": str(r.get("K", "")),
            "seed": str(r.get("seed", "")),
            "route_policy": str(r.get("route_policy", "")),
            "deciding_step": str(r.get("deciding_step", "")),
            "gap_pct": str(r.get("gap_pct", "")),
            "ub": str(r.get("ub", "")),
            "lb": str(r.get("lb", "")),
            "runtime_sec": str(r.get("runtime_sec", "")),
            "memory_killed": str(r.get("memory_killed", "0")),
            "external_timed_out": str(r.get("external_timed_out", "0")),
            "solver_returncode": str(r.get("solver_returncode", "0")),
            "failure_signature": sig,
            "boundary_class": str(r.get("boundary_class", "")),
            "boundary_detail": str(r.get("boundary_detail", "")),
        })
    return out


def main() -> None:
    solver = ROOT / "solvers" / "cpp" / "build" / "stateful_compare"
    if not solver.exists():
        raise FileNotFoundError(f"Missing solver binary: {solver}")

    rows = load_raw(RAW_CSV)
    refresh_derived_fields(rows)
    # Drop suspicious / incomplete rows so they are rerun cleanly
    rows = [r for r in rows if row_key(r) not in FORCE_RERUN]
    seen = {row_key(r) for r in rows}

    ks = [8, 10, 12]
    run_hard_ladder("hardA", "hard_irregular_A", HARD_A_BASE, ks, rows, seen)
    run_hard_ladder("hardB", "hard_irregular_B", HARD_B_BASE, ks, rows, seen)

    refresh_derived_fields(rows)
    write_csv(RAW_CSV, rows)
    print(f"Wrote {RAW_CSV} n={len(rows)}")

    best = build_best_of_route(rows)
    write_csv(BEST_CSV, best)
    print(f"Wrote {BEST_CSV} n={len(best)}")

    sum_k = build_summary_by_k(best)
    write_csv(SUM_K_CSV, sum_k)
    print(f"Wrote {SUM_K_CSV} n={len(sum_k)}")

    fail = build_failure_signatures(best)
    write_csv(FAIL_CSV, fail)
    print(f"Wrote {FAIL_CSV} n={len(fail)}")


if __name__ == "__main__":
    main()

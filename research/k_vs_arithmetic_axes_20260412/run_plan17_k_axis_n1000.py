#!/usr/bin/env python3
"""
PLAN17: fixed n=1000 K-axis ladders (easy unit vs hard irregular A/B).

See RESULTS.md / LOG.md for interpretation. Uses explicit routing per policy;
does not mix variants in a single row.
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

OUT_DIR = ROOT / "research" / "k_vs_arithmetic_axes_20260412" / "csv" / "plan17"
RAW_CSV = OUT_DIR / "PLAN17_k_axis_n1000_raw.csv"
SUM_FAMILY = OUT_DIR / "PLAN17_k_axis_n1000_summary_by_family.csv"
SUM_K = OUT_DIR / "PLAN17_k_axis_n1000_summary_by_k.csv"
BOUNDARY_CSV = OUT_DIR / "PLAN17_k_axis_boundary_classification.csv"

LAMBDA = 1.3
N_JOBS = 1000
SEEDS = (0, 1)
TIME_LIMIT = 900.0
MAX_RSS_GB = 16.0
EXTERNAL_TIMEOUT = int(max(240, TIME_LIMIT + 120))

EC_CONFIGS = [
    {"from_date": "2019-01-21T00:00:00", "repeat_count": 1},
    {"from_date": "2019-04-08T00:00:00", "repeat_count": 1},
]

K2_ENV: dict[str, str] = {
    "PAST_RELAXED_BINPACK_SOLVER": "profile_repair_beam",
    "PAST_PROFILE_REALIZATION_SELECTOR_POLICY": "auto_v1",
}

DENSE_STEP2_ENV: dict[str, str] = {
    "PAST_DENSE_UNIT_STEP2_FASTPATH": "1",
    "PAST_DENSE_UNIT_FASTPATH_K_MIN": "8",
}

IRREGULAR_REROUTE_ENV: dict[str, str] = {
    "PAST_RELAXED_BINPACK_SOLVER": "profile_repair_beam",
    "PAST_PROFILE_REALIZATION_SELECTOR_POLICY": "auto_v1",
}

HARD_A_BASE = [
    2,
    3,
    4,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
]

HARD_B_BASE = [
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
]


def sizes_label(sizes: list[int]) -> str:
    return "{" + ",".join(str(x) for x in sizes) + "}"


def build_plan17_payload(
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
        name=f"plan17/{family_id}_n{n_jobs}_lam{lam:.1f}_s{seed}",
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
            "plan17": "1",
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
    """Fraction of {1..100} representable as nonnegative combo of sizes (min size positive)."""
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
    """
    Returns (boundary_class, detail_note).
    """
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
            # Enum in task list has no Step-4 bucket; record detail on raw row.
            return "unresolved_other", "closed_step4_global_exact"
        if ds == "step1":
            return "easy_step2_exact", "closed_step1"
        return "unresolved_other", f"optimal_unexpected_step={ds}"

    # not optimal
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
    # align timer name requested by PLAN17
    if "t_fwd_pack_profile_beam" not in out or out.get("t_fwd_pack_profile_beam") == "":
        out["t_fwd_pack_profile_beam"] = out.get(
            "t_fwd_pack_profile_beam", base.get("t_fwd_pack_profile_beam", "")
        )
    for kdesc, v in desc.items():
        out[kdesc] = str(v)
    out["arithmetic_class_label"] = (
        "contiguous_unit_with_1"
        if family_class == "easy_unit"
        else (
            "irregular_primes_from2"
            if family_class == "hard_irregular_A"
            else "irregular_primes_from3"
        )
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


def should_rerun_irregular_baseline(row: dict[str, Any]) -> bool:
    """PLAN17 rule: additive profile reroute if baseline bypasses Step-3 or no useful incumbent."""
    if str(row.get("memory_killed", "0")) == "1":
        return False
    if str(row.get("is_optimal", "0")) == "1":
        return False
    sel = (row.get("selector_reason") or "").strip()
    if sel == "non_mainline_solver":
        return True
    if str(row.get("external_timed_out", "0")) == "1":
        return True
    ub = str(row.get("ub", "-1")).strip()
    if ub == "-1" or ub == "":
        return True
    return False


def run_easy_ladder(rows: list[dict[str, Any]], seen: set[tuple[str, str, int]]) -> None:
    specs = [
        ("easy_k2_unit", [1, 2]),
        ("easy_k4_unit", list(range(1, 5))),
        ("easy_k6_unit", list(range(1, 7))),
        ("easy_k8_unit", list(range(1, 9))),
        ("easy_k10_unit", list(range(1, 11))),
        ("easy_k12_unit", list(range(1, 13))),
        ("easy_k16_unit", list(range(1, 17))),
        ("easy_k20_unit", list(range(1, 21))),
    ]
    for fid, sizes in specs:
        k = len(sizes)
        label = sizes_label(sizes)
        variants: list[tuple[str, dict[str, str], str]] = []
        if k == 2:
            variants.append(
                (
                    "baseline",
                    dict(K2_ENV),
                    "plan17_k2:profile_repair_beam+auto_v1",
                )
            )
        elif k >= 8:
            variants.append(("baseline", {}, "plan17_easy:accepted_energy_core_baseline"))
            variants.append(
                (
                    "dense_step2_fastpath",
                    dict(DENSE_STEP2_ENV),
                    "plan17_easy:dense_step2_fastpath_kmin8+energy_core_baseline",
                )
            )
        else:
            variants.append(("baseline", {}, "plan17_easy:accepted_energy_core_baseline"))

        for variant_label, env_extra, route_policy in variants:
            for seed in SEEDS:
                key = (fid, variant_label, seed)
                if key in seen:
                    continue
                payload = build_plan17_payload(fid, sizes, N_JOBS, LAMBDA, seed)
                raw = run_row(
                    fid,
                    N_JOBS,
                    seed,
                    TIME_LIMIT,
                    variant_label,
                    env_extra,
                    max_rss_gb=MAX_RSS_GB,
                    payload=payload,
                )
                row = normalize_output_row(
                    raw,
                    family_id=fid,
                    family_label=label,
                    family_class="easy_unit",
                    family_sizes=sizes,
                    k=k,
                    variant_label=variant_label,
                    route_policy=route_policy,
                )
                rows.append(row)
                seen.add(key)
                write_csv(RAW_CSV, rows)
                print(
                    f"[easy] {fid} {variant_label} seed={seed} K={k} "
                    f"step={row.get('deciding_step')} opt={row.get('is_optimal')} "
                    f"rt={row.get('runtime_sec')} bc={row.get('boundary_class')}"
                )


def run_hard_ladder(
    prefix: str,
    family_class: str,
    base: list[int],
    rows: list[dict[str, Any]],
    seen: set[tuple[str, str, int]],
) -> None:
    ks = [4, 6, 8, 10, 12, 16, 20]
    for target_k in ks:
        sizes = base[:target_k]
        fid = f"{prefix}_k{target_k}"
        label = sizes_label(sizes)
        k = len(sizes)
        if k == 4:
            seeds_to_run = list(SEEDS)
            for seed in seeds_to_run:
                key = (fid, "baseline", seed)
                if key in seen:
                    continue
                payload = build_plan17_payload(fid, sizes, N_JOBS, LAMBDA, seed)
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
                    route_policy="plan17_k4:energy_core_direct_package",
                )
                rows.append(row)
                seen.add(key)
                write_csv(RAW_CSV, rows)
                print(
                    f"[{prefix}] {fid} baseline seed={seed} K={k} "
                    f"step={row.get('deciding_step')} opt={row.get('is_optimal')} "
                    f"bc={row.get('boundary_class')}"
                )
            continue

        # K >= 6
        for seed in SEEDS:
            key = (fid, "irregular_baseline", seed)
            if key in seen:
                continue
            payload = build_plan17_payload(fid, sizes, N_JOBS, LAMBDA, seed)
            raw = run_row(
                fid,
                N_JOBS,
                seed,
                TIME_LIMIT,
                "irregular_baseline",
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
                variant_label="irregular_baseline",
                route_policy="plan17_irregular:accepted_energy_core_baseline",
            )
            rows.append(row)
            seen.add(key)
            write_csv(RAW_CSV, rows)
            print(
                f"[{prefix}] {fid} irregular_baseline seed={seed} K={k} "
                f"step={row.get('deciding_step')} opt={row.get('is_optimal')} "
                f"bc={row.get('boundary_class')}"
            )

        for seed in SEEDS:
            bkey = (fid, "irregular_baseline", seed)
            if bkey not in seen:
                continue
            base_row = next(
                (r for r in rows if row_key(r) == (fid, "irregular_baseline", seed)), None
            )
            if base_row is None:
                continue
            if not should_rerun_irregular_baseline(base_row):
                continue
            rkey = (fid, "irregular_profile_reroute", seed)
            if rkey in seen:
                continue
            payload = build_plan17_payload(fid, sizes, N_JOBS, LAMBDA, seed)
            raw = run_row(
                fid,
                N_JOBS,
                seed,
                TIME_LIMIT,
                "irregular_profile_reroute",
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
                variant_label="irregular_profile_reroute",
                route_policy="plan17_irregular:additive_profile_repair_beam_auto_v1",
            )
            rows.append(row)
            seen.add(rkey)
            write_csv(RAW_CSV, rows)
            print(
                f"[{prefix}] {fid} irregular_profile_reroute seed={seed} K={k} "
                f"step={row.get('deciding_step')} opt={row.get('is_optimal')} "
                f"bc={row.get('boundary_class')}"
            )


def build_summaries(all_rows: list[dict[str, Any]]) -> None:
    # by family_id + variant_label: keep baseline/reroute separated
    by_fam: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in all_rows:
        fid = str(r.get("family_id", ""))
        vl = str(r.get("variant_label", ""))
        by_fam.setdefault((fid, vl), []).append(r)

    fam_rows: list[dict[str, Any]] = []
    for (fid, vl), rs in sorted(by_fam.items()):
        rts = []
        exact = 0
        seeds_present: list[str] = []
        worst = "easy_step2_exact"
        for r in rs:
            try:
                rts.append(float(str(r.get("runtime_sec", "nan"))))
            except Exception:
                pass
            if str(r.get("is_optimal", "0")) == "1":
                exact += 1
            seed = str(r.get("seed", "")).strip()
            if seed != "":
                seeds_present.append(seed)
            worst = worse_class(worst, str(r.get("boundary_class", "unresolved_other")))
        fc = str(rs[0].get("family_class", ""))
        k = str(rs[0].get("K", ""))
        rss_vals: list[float] = []
        for r in rs:
            try:
                v = float(str(r.get("peak_rss_gb", "nan")))
            except Exception:
                continue
            if v == v:
                rss_vals.append(v)
        fam_rows.append(
            {
                "family_id": fid,
                "family_class": fc,
                "K": k,
                "variant_label": vl,
                "route_policy": str(rs[0].get("route_policy", "")),
                "seeds_present": ",".join(sorted(set(seeds_present))),
                "n_rows": str(len(rs)),
                "exact_rows": f"{exact}/{len(rs)}",
                "boundary_worst_of_rows": worst,
                "mean_runtime_sec": f"{statistics.mean(rts):.3f}" if rts else "",
                "mean_peak_rss_gb": f"{statistics.mean(rss_vals):.3f}" if rss_vals else "",
            }
        )
    write_csv(SUM_FAMILY, fam_rows)

    # by K, family_class, and variant_label
    by_kfc: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for r in all_rows:
        try:
            kk = int(r.get("K", "-1"))
        except Exception:
            continue
        fc = str(r.get("family_class", ""))
        vl = str(r.get("variant_label", ""))
        by_kfc.setdefault((kk, fc, vl), []).append(r)

    k_rows: list[dict[str, Any]] = []
    for (kk, fc, vl) in sorted(by_kfc.keys()):
        rs = by_kfc[(kk, fc, vl)]
        rts = []
        exact = 0
        worst = "easy_step2_exact"
        for r in rs:
            try:
                rts.append(float(str(r.get("runtime_sec", "nan"))))
            except Exception:
                pass
            if str(r.get("is_optimal", "0")) == "1":
                exact += 1
            worst = worse_class(worst, str(r.get("boundary_class", "unresolved_other")))
        k_rows.append(
            {
                "K": str(kk),
                "family_class": fc,
                "variant_label": vl,
                "n_rows": str(len(rs)),
                "exact_rows": f"{exact}/{len(rs)}",
                "boundary_worst_of_rows": worst,
                "mean_runtime_sec": f"{statistics.mean(rts):.3f}" if rts else "",
                "family_ids_sample": ",".join(sorted({str(r.get('family_id')) for r in rs})[:5]),
            }
        )
    write_csv(SUM_K, k_rows)


def build_boundary_table(all_rows: list[dict[str, Any]]) -> None:
    """
    One row per (family_id, variant_label): worst boundary across seeds,
    plus per-seed classes for audit.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in all_rows:
        key = (str(r.get("family_id", "")), str(r.get("variant_label", "")))
        groups.setdefault(key, []).append(r)

    out: list[dict[str, Any]] = []
    for (fid, vl), rs in sorted(groups.items()):
        by_seed: dict[int, str] = {}
        for r in rs:
            try:
                by_seed[int(r.get("seed", -1))] = str(r.get("boundary_class", ""))
            except Exception:
                pass
        worst = "easy_step2_exact"
        for s in SEEDS:
            worst = worse_class(worst, by_seed.get(s, "unresolved_other"))
        try:
            k = int(rs[0].get("K", "0"))
        except Exception:
            k = -1
        out.append(
            {
                "family_id": fid,
                "family_label": str(rs[0].get("family_label", "")),
                "family_class": str(rs[0].get("family_class", "")),
                "K": str(k),
                "variant_label": vl,
                "boundary_worst_of_seeds": worst,
                "boundary_seed_0": by_seed.get(0, ""),
                "boundary_seed_1": by_seed.get(1, ""),
                "route_policy": str(rs[0].get("route_policy", "")),
            }
        )
    write_csv(BOUNDARY_CSV, out)


def main() -> None:
    solver = ROOT / "solvers" / "cpp" / "build" / "stateful_compare"
    if not solver.exists():
        raise FileNotFoundError(f"Missing solver binary: {solver}")

    rows = load_raw(RAW_CSV)
    refresh_derived_fields(rows)
    seen = {row_key(r) for r in rows}

    run_easy_ladder(rows, seen)
    run_hard_ladder("hardA", "hard_irregular_A", HARD_A_BASE, rows, seen)
    run_hard_ladder("hardB", "hard_irregular_B", HARD_B_BASE, rows, seen)

    refresh_derived_fields(rows)
    write_csv(RAW_CSV, rows)
    build_summaries(rows)
    build_boundary_table(rows)

    print(f"Wrote {RAW_CSV} n={len(rows)}")
    print(f"Wrote {SUM_FAMILY}")
    print(f"Wrote {SUM_K}")
    print(f"Wrote {BOUNDARY_CSV}")


if __name__ == "__main__":
    main()

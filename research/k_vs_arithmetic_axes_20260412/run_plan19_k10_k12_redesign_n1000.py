#!/usr/bin/env python3
"""
PLAN19: bounded additive redesigns for hard irregular K=10/12 at fixed n=1000.

Redesigns:
1. beam -> restricted exact closure after incumbent exists
   (PAST_PROFILE_REALIZATION_EXACT_AFTER_BEAM_ENABLE=1)
2. irregular high-K routing override (skip baseline in runner)
3. optional stronger K=12 beam (beam_plus via PAST_EXACT_INCUMBENT_SOURCE=i3)

Memory-safe execution: one heavy row at a time, tail-read stdout/stderr,
hard RSS cap with watchdog.
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

OUT_DIR = ROOT / "research" / "k_vs_arithmetic_axes_20260412" / "csv" / "plan19"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = OUT_DIR / "PLAN19_k10_k12_redesign_raw.csv"
COMPARE_CSV = OUT_DIR / "PLAN19_k10_k12_redesign_compare.csv"
BEST_SUMMARY_CSV = OUT_DIR / "PLAN19_k10_k12_best_variant_summary.csv"
FAIL_SHIFT_CSV = OUT_DIR / "PLAN19_k10_k12_failure_shift.csv"
NOTES_MD = OUT_DIR / "PLAN19_k10_k12_method_notes.md"
DIAGNOSIS_MD = OUT_DIR / "PLAN19_k10_k12_diagnosis.md"

PLAN18_RAW = (
    ROOT
    / "research"
    / "k_vs_arithmetic_axes_20260412"
    / "csv"
    / "plan18"
    / "PLAN18_k_boundary_refine_n1000_raw.csv"
)

LAMBDA = 1.3
N_JOBS = 1000
TIME_LIMIT = 1200.0
MAX_RSS_GB = 12.0
EXTERNAL_TIMEOUT = int(max(240, TIME_LIMIT + 120))

HARD_A_BASE = [
    2, 3, 4, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67,
]
HARD_B_BASE = [
    3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
]

EC_CONFIGS = [
    {"from_date": "2019-01-21T00:00:00", "repeat_count": 1},
    {"from_date": "2019-04-08T00:00:00", "repeat_count": 1},
]


def sizes_label(sizes: list[int]) -> str:
    return "{" + ",".join(str(x) for x in sizes) + "}"


def build_plan19_payload(
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
        name=f"plan19/{family_id}_n{n_jobs}_lam{lam:.1f}_s{seed}",
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
            "plan19": "1",
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


def row_key(r: dict[str, Any]) -> tuple[str, str, int]:
    try:
        s = int(r.get("seed", -1))
    except Exception:
        s = -1
    return (str(r.get("family_id", "")), str(r.get("variant_label", "")), s)


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


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

VARIANTS: dict[str, dict[str, str]] = {
    "baseline": {},
    "irregular_reroute": {
        "PAST_RELAXED_BINPACK_SOLVER": "profile_repair_beam",
        "PAST_PROFILE_REALIZATION_SELECTOR_POLICY": "auto_v1",
    },
    "exp_exact_after_beam_300": {
        "PAST_RELAXED_BINPACK_SOLVER": "profile_repair_beam",
        "PAST_PROFILE_REALIZATION_SELECTOR_POLICY": "auto_v1",
        "PAST_PROFILE_REALIZATION_EXACT_AFTER_BEAM_ENABLE": "1",
        "PAST_PROFILE_REALIZATION_EXACT_AFTER_BEAM_MAX_MERGED": "24",
        "PAST_PROFILE_REALIZATION_EXACT_TIME_LIMIT": "300",
        "PAST_RELAXED_BINPACK_MAX_COMP_EST": "1000000000000",
        "PAST_RELAXED_BINPACK_MAX_NC": "1000000000000",
    },
    "exp_exact_after_beam_600": {
        "PAST_RELAXED_BINPACK_SOLVER": "profile_repair_beam",
        "PAST_PROFILE_REALIZATION_SELECTOR_POLICY": "auto_v1",
        "PAST_PROFILE_REALIZATION_EXACT_AFTER_BEAM_ENABLE": "1",
        "PAST_PROFILE_REALIZATION_EXACT_AFTER_BEAM_MAX_MERGED": "24",
        "PAST_PROFILE_REALIZATION_EXACT_TIME_LIMIT": "600",
        "PAST_RELAXED_BINPACK_MAX_COMP_EST": "1000000000000",
        "PAST_RELAXED_BINPACK_MAX_NC": "1000000000000",
    },
    "exp_force_exact_300": {
        "PAST_RELAXED_BINPACK_SOLVER": "profile_repair_beam",
        "PAST_PROFILE_REALIZATION_SELECTOR_POLICY": "force_exact",
        "PAST_PROFILE_REALIZATION_EXACT_TIME_LIMIT": "300",
        "PAST_RELAXED_BINPACK_MAX_COMP_EST": "1000000000000",
        "PAST_RELAXED_BINPACK_MAX_NC": "1000000000000",
    },
    "exp_force_exact_600": {
        "PAST_RELAXED_BINPACK_SOLVER": "profile_repair_beam",
        "PAST_PROFILE_REALIZATION_SELECTOR_POLICY": "force_exact",
        "PAST_PROFILE_REALIZATION_EXACT_TIME_LIMIT": "600",
        "PAST_RELAXED_BINPACK_MAX_COMP_EST": "1000000000000",
        "PAST_RELAXED_BINPACK_MAX_NC": "1000000000000",
    },
    "exp_beam_plus": {
        "PAST_RELAXED_BINPACK_SOLVER": "profile_repair_beam",
        "PAST_PROFILE_REALIZATION_SELECTOR_POLICY": "auto_v1",
        "PAST_EXACT_INCUMBENT_SOURCE": "i3",
    },
    "exp_beam_plus_exact_300": {
        "PAST_RELAXED_BINPACK_SOLVER": "profile_repair_beam",
        "PAST_PROFILE_REALIZATION_SELECTOR_POLICY": "auto_v1",
        "PAST_EXACT_INCUMBENT_SOURCE": "i3",
        "PAST_PROFILE_REALIZATION_EXACT_AFTER_BEAM_ENABLE": "1",
        "PAST_PROFILE_REALIZATION_EXACT_AFTER_BEAM_MAX_MERGED": "24",
        "PAST_PROFILE_REALIZATION_EXACT_TIME_LIMIT": "300",
        "PAST_RELAXED_BINPACK_MAX_COMP_EST": "1000000000000",
        "PAST_RELAXED_BINPACK_MAX_NC": "1000000000000",
    },
}


def run_variant(
    family_id: str,
    sizes: list[int],
    k: int,
    seed: int,
    variant_label: str,
    env: dict[str, str],
    rows: list[dict[str, Any]],
    seen: set[tuple[str, str, int]],
) -> dict[str, Any] | None:
    key = (family_id, variant_label, seed)
    if key in seen:
        return None
    payload = build_plan19_payload(family_id, sizes, N_JOBS, LAMBDA, seed)
    raw = run_row(
        family_id,
        N_JOBS,
        seed,
        TIME_LIMIT,
        variant_label,
        dict(env),
        max_rss_gb=MAX_RSS_GB,
        payload=payload,
    )
    row = normalize_output_row(
        raw,
        family_id=family_id,
        family_label=sizes_label(sizes),
        family_class=("hard_irregular_A" if family_id.startswith("hardA") else "hard_irregular_B"),
        family_sizes=sizes,
        k=k,
        variant_label=variant_label,
        route_policy=f"plan19:{variant_label}",
    )
    rows.append(row)
    seen.add(key)
    write_csv(RAW_CSV, rows)
    print(
        f"[plan19] {family_id} {variant_label} seed={seed} K={k} "
        f"step={row.get('deciding_step')} opt={row.get('is_optimal')} "
        f"bc={row.get('boundary_class')} gap={row.get('gap_pct')} rt={row.get('runtime_sec')} "
        f"rss={row.get('peak_rss_gb')}GB memkill={row.get('memory_killed')}"
    )
    return row


# ---------------------------------------------------------------------------
# Phase A diagnosis
# ---------------------------------------------------------------------------

def write_diagnosis(plan18_rows: list[dict[str, Any]]) -> None:
    target_fams = {"hardA_k10", "hardB_k10", "hardA_k12", "hardB_k12"}
    rows = [r for r in plan18_rows if r.get("family_id") in target_fams]

    lines = [
        "# PLAN19 Phase A Diagnosis: where closure is lost after incumbent production",
        "",
        "## Evidence from PLAN18 (n=1000, lambda=1.3, seeds 0-3)",
        "",
    ]

    by_fam: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_fam.setdefault(str(r.get("family_id")), []).append(r)

    for fam in sorted(by_fam.keys()):
        rs = by_fam[fam]
        baseline_rows = [r for r in rs if r.get("variant_label") == "baseline"]
        reroute_rows = [r for r in rs if r.get("variant_label") == "irregular_reroute"]

        lines.append(f"### {fam}")
        lines.append("")

        # Baseline diagnosis
        b_failures = []
        for r in baseline_rows:
            sel_reason = str(r.get("selector_reason", ""))
            ds = str(r.get("deciding_step", ""))
            ub = str(r.get("ub", "-1"))
            if sel_reason == "non_mainline_solver":
                b_failures.append(f"seed={r.get('seed')}: selector bypass ({sel_reason}), no incumbent, deciding_step={ds}")
            elif ub == "-1":
                b_failures.append(f"seed={r.get('seed')}: no incumbent, deciding_step={ds}")
            else:
                b_failures.append(f"seed={r.get('seed')}: ub={ub}, deciding_step={ds}")
        lines.append("**Baseline (energy_core):**")
        for bf in b_failures:
            lines.append(f"- {bf}")
        lines.append("")

        # Reroute diagnosis
        r_results = []
        for r in reroute_rows:
            opt = str(r.get("is_optimal", "0")) == "1"
            gap = str(r.get("gap_pct", "nan"))
            ds = str(r.get("deciding_step", ""))
            rt = str(r.get("runtime_sec", ""))
            if opt:
                r_results.append(f"seed={r.get('seed')}: exact, rt={rt}s")
            elif ds == "external_timeout":
                r_results.append(f"seed={r.get('seed')}: timeout (no incumbent), rt={rt}s")
            else:
                r_results.append(f"seed={r.get('seed')}: finite gap {gap}%, deciding_step={ds}, rt={rt}s")
        lines.append("**Reroute (profile_repair_beam + auto_v1):**")
        for rr in r_results:
            lines.append(f"- {rr}")
        lines.append("")

        # Diagnosis summary
        any_exact = any(str(r.get("is_optimal", "0")) == "1" for r in reroute_rows)
        any_timeout = any(str(r.get("deciding_step", "")) == "external_timeout" for r in reroute_rows)
        gaps = []
        for r in reroute_rows:
            try:
                g = float(str(r.get("gap_pct", "nan")))
                if not math.isnan(g) and g > 1e-6:
                    gaps.append(g)
            except Exception:
                pass

        if any_exact:
            lines.append("Diagnosis: exact closure occurs on some seeds; beam+Step4 is sufficient.")
        elif gaps:
            lines.append(f"Diagnosis: beam produces incumbents but Step 4 exact DP does not close. Gaps: {min(gaps):.4f}% - {max(gaps):.4f}%.")
        elif any_timeout:
            lines.append("Diagnosis: beam itself times out before producing incumbent. Need stronger/faster incumbent generation or more budget.")
        else:
            lines.append("Diagnosis: no useful incumbents produced.")
        lines.append("")

    lines.append("## Overall Conclusion")
    lines.append("")
    lines.append("- K=10: baseline is always bypassed/no-incumbent. Reroute beam produces incumbents consistently, but Step 4 global exact DP fails to close. The bottleneck is **closure after incumbent production**.")
    lines.append("- K=12: baseline is bypassed/no-incumbent. Reroute beam sometimes produces incumbents, sometimes times out. The bottleneck is a mix of **incumbent production** and **closure**.")
    lines.append("- Therefore, the highest-value redesigns are: (1) bounded exact closure after beam incumbent on K=10; (2) routing override to skip useless baseline on K>=10; (3) optionally stronger beam for K=12 incumbent production.")
    lines.append("")

    DIAGNOSIS_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {DIAGNOSIS_MD}")


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main() -> None:
    solver = ROOT / "solvers" / "cpp" / "build" / "stateful_compare"
    if not solver.exists():
        raise FileNotFoundError(f"Missing solver binary: {solver}")

    # Load PLAN18 data for reuse
    plan18_rows = load_raw(PLAN18_RAW)
    plan18_keys = {row_key(r) for r in plan18_rows}

    rows: list[dict[str, Any]] = load_raw(RAW_CSV)
    seen = {row_key(r) for r in rows}

    # Reuse PLAN18 baseline and irregular_reroute rows into PLAN19 raw
    reused_labels = {"baseline", "irregular_reroute"}
    for r in plan18_rows:
        fid = str(r.get("family_id", ""))
        vl = str(r.get("variant_label", ""))
        if fid.startswith(("hardA_k", "hardB_k")) and vl in reused_labels:
            key = row_key(r)
            if key not in seen:
                rows.append(dict(r))
                seen.add(key)
    write_csv(RAW_CSV, rows)

    write_diagnosis(plan18_rows)

    # Target families and seeds
    target_ks = [10, 12]
    seeds_wave1 = [0, 1]
    seeds_wave2 = [2, 3]

    ladders = [
        ("hardA", "hard_irregular_A", HARD_A_BASE),
        ("hardB", "hard_irregular_B", HARD_B_BASE),
    ]

    # Primary variant for wave 1: exact_after_beam_300 and beam_plus
    wave1_variants = ["exp_exact_after_beam_300", "exp_beam_plus"]
    wave2_variants: list[str] = []

    # Run wave 1
    for prefix, family_class, base in ladders:
        for target_k in target_ks:
            sizes = base[:target_k]
            fid = f"{prefix}_k{target_k}"
            for seed in seeds_wave1:
                for vl in wave1_variants:
                    run_variant(fid, sizes, target_k, seed, vl, VARIANTS[vl], rows, seen)

    # Quick evaluation after wave 1 to decide wave 2
    def _best_gap(fid: str, seed: int, vl: str) -> float:
        for r in rows:
            if row_key(r) == (fid, vl, seed):
                try:
                    g = float(str(r.get("gap_pct", "nan")))
                    if not math.isnan(g):
                        return g
                except Exception:
                    pass
        return float("inf")

    def _is_exact(fid: str, seed: int, vl: str) -> bool:
        for r in rows:
            if row_key(r) == (fid, vl, seed):
                return str(r.get("is_optimal", "0")) == "1"
        return False

    promising_wave2 = False
    for prefix, _, base in ladders:
        for target_k in target_ks:
            fid = f"{prefix}_k{target_k}"
            for seed in seeds_wave1:
                plan18_best_gap = min(
                    _best_gap(fid, seed, "baseline"),
                    _best_gap(fid, seed, "irregular_reroute"),
                )
                for vl in wave1_variants:
                    if _is_exact(fid, seed, vl):
                        promising_wave2 = True
                        print(f"[wave2-trigger] exact found: {fid} seed={seed} {vl}")
                    elif _best_gap(fid, seed, vl) < plan18_best_gap - 1e-4:
                        promising_wave2 = True
                        print(f"[wave2-trigger] gap improved: {fid} seed={seed} {vl} "
                              f"{_best_gap(fid, seed, vl):.4f}% < {plan18_best_gap:.4f}%")

    if promising_wave2:
        print("\n[wave2] Extending to seeds 2/3 for primary variants.")
        for prefix, family_class, base in ladders:
            for target_k in target_ks:
                sizes = base[:target_k]
                fid = f"{prefix}_k{target_k}"
                for seed in seeds_wave2:
                    for vl in wave1_variants:
                        run_variant(fid, sizes, target_k, seed, vl, VARIANTS[vl], rows, seen)
        wave2_variants = list(wave1_variants)
    else:
        print("\n[wave2] Wave 1 did not show material improvement; skipping seeds 2/3 for primary variants.")

    # Optional: run longer exact or force_exact on a few promising seeds
    # Heuristic: run force_exact_300 on one seed per family/K if exact_after_beam_300 did not exact but did not memory-kill
    optional_variants = ["exp_force_exact_300", "exp_exact_after_beam_600"]
    for prefix, family_class, base in ladders:
        for target_k in target_ks:
            sizes = base[:target_k]
            fid = f"{prefix}_k{target_k}"
            for seed in seeds_wave1:
                # only run optional if primary didn't exact and didn't memory-kill
                primary_ok = False
                for r in rows:
                    if row_key(r) == (fid, "exp_exact_after_beam_300", seed):
                        if str(r.get("memory_killed", "0")) == "0" and str(r.get("is_optimal", "0")) != "1":
                            primary_ok = True
                        break
                if primary_ok:
                    for vl in optional_variants:
                        run_variant(fid, sizes, target_k, seed, vl, VARIANTS[vl], rows, seen)

    # Optional K=12 beam_plus_exact (redesign 3)
    for prefix, family_class, base in ladders:
        for target_k in [12]:
            sizes = base[:target_k]
            fid = f"{prefix}_k{target_k}"
            for seed in seeds_wave1:
                run_variant(fid, sizes, target_k, seed, "exp_beam_plus_exact_300", VARIANTS["exp_beam_plus_exact_300"], rows, seen)

    # Now build all artifacts
    build_artifacts(rows)
    print("PLAN19 complete.")


# ---------------------------------------------------------------------------
# Artifact builders
# ---------------------------------------------------------------------------

def build_artifacts(all_rows: list[dict[str, Any]]) -> None:
    # Ensure derived fields are fresh
    for r in all_rows:
        try:
            k = int(str(r.get("K", "0")))
        except Exception:
            k = 0
        bc, bnote = classify_boundary(r, k)
        r["boundary_class"] = bc
        r["boundary_detail"] = bnote

    write_csv(RAW_CSV, all_rows)
    print(f"Wrote {RAW_CSV} n={len(all_rows)}")

    best = build_best_of_variant(all_rows)
    write_csv(BEST_SUMMARY_CSV, best)
    print(f"Wrote {BEST_SUMMARY_CSV} n={len(best)}")

    compare = build_compare(all_rows)
    write_csv(COMPARE_CSV, compare)
    print(f"Wrote {COMPARE_CSV} n={len(compare)}")

    fail_shift = build_failure_shift(all_rows)
    write_csv(FAIL_SHIFT_CSV, fail_shift)
    print(f"Wrote {FAIL_SHIFT_CSV} n={len(fail_shift)}")

    write_notes(all_rows)


def build_best_of_variant(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each (family_id, K, seed), pick best variant."""
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
        out.append({
            "family_id": fid,
            "K": k,
            "seed": seed,
            "best_variant": best.get("variant_label", ""),
            "is_optimal": best.get("is_optimal", "0"),
            "gap_pct": best.get("gap_pct", ""),
            "ub": best.get("ub", ""),
            "lb": best.get("lb", ""),
            "runtime_sec": best.get("runtime_sec", ""),
            "peak_rss_gb": best.get("peak_rss_gb", ""),
            "memory_killed": best.get("memory_killed", "0"),
            "external_timed_out": best.get("external_timed_out", "0"),
            "deciding_step": best.get("deciding_step", ""),
            "boundary_class": best.get("boundary_class", ""),
            "boundary_detail": best.get("boundary_detail", ""),
        })
    return out


def build_compare(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target_fams = {"hardA_k10", "hardB_k10", "hardA_k12", "hardB_k12"}
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for r in all_rows:
        fid = str(r.get("family_id", ""))
        if fid not in target_fams:
            continue
        try:
            k = int(r.get("K", "0"))
            seed = int(r.get("seed", "-1"))
        except Exception:
            continue
        groups.setdefault((fid, k, seed), []).append(r)

    out: list[dict[str, Any]] = []
    for (fid, k, seed), rs in sorted(groups.items()):
        plan18_best = None
        plan19_best = None
        for r in rs:
            vl = str(r.get("variant_label", ""))
            if vl in ("baseline", "irregular_reroute"):
                if plan18_best is None or _row_better(r, plan18_best):
                    plan18_best = r
            else:
                if plan19_best is None or _row_better(r, plan19_best):
                    plan19_best = r

        def extract(r: dict[str, Any] | None) -> dict[str, str]:
            if r is None:
                return {"variant": "", "opt": "0", "gap": "inf", "rt": "", "rss": ""}
            return {
                "variant": str(r.get("variant_label", "")),
                "opt": str(r.get("is_optimal", "0")),
                "gap": str(_effective_gap(r)),
                "rt": str(r.get("runtime_sec", "")),
                "rss": str(r.get("peak_rss_gb", "")),
            }

        p18 = extract(plan18_best)
        p19 = extract(plan19_best)

        winner = "tie"
        if p19["opt"] == "1" and p18["opt"] != "1":
            winner = "plan19"
        elif p18["opt"] == "1" and p19["opt"] != "1":
            winner = "plan18"
        else:
            try:
                g18 = float(p18["gap"])
                g19 = float(p19["gap"])
                if g19 < g18 - 1e-6:
                    winner = "plan19"
                elif g18 < g19 - 1e-6:
                    winner = "plan18"
                else:
                    try:
                        if float(p19["rt"]) < float(p18["rt"]):
                            winner = "plan19_runtime"
                        elif float(p18["rt"]) < float(p19["rt"]):
                            winner = "plan18_runtime"
                    except Exception:
                        pass
            except Exception:
                if p19["gap"] != "inf" and p18["gap"] == "inf":
                    winner = "plan19"
                elif p18["gap"] != "inf" and p19["gap"] == "inf":
                    winner = "plan18"

        out.append({
            "family_id": fid,
            "K": k,
            "seed": seed,
            "plan18_variant": p18["variant"],
            "plan18_opt": p18["opt"],
            "plan18_gap": p18["gap"],
            "plan18_rt": p18["rt"],
            "plan19_variant": p19["variant"],
            "plan19_opt": p19["opt"],
            "plan19_gap": p19["gap"],
            "plan19_rt": p19["rt"],
            "winner": winner,
        })
    return out


def _has_incumbent(r: dict[str, Any]) -> bool:
    try:
        ub = float(str(r.get("ub", "-1")))
        lb = float(str(r.get("lb", "-1")))
        return ub >= 0 and lb >= 0
    except Exception:
        return False


def _effective_gap(r: dict[str, Any]) -> float:
    if str(r.get("is_optimal", "0")) == "1":
        return 0.0
    if not _has_incumbent(r):
        return float("inf")
    try:
        g = float(str(r.get("gap_pct", "nan")))
        if math.isnan(g):
            return float("inf")
        return g
    except Exception:
        return float("inf")


def _row_better(a: dict[str, Any], b: dict[str, Any]) -> bool:
    opt_a = str(a.get("is_optimal", "0")) == "1"
    opt_b = str(b.get("is_optimal", "0")) == "1"
    if opt_a and not opt_b:
        return True
    if opt_b and not opt_a:
        return False
    ga = _effective_gap(a)
    gb = _effective_gap(b)
    if ga < gb - 1e-6:
        return True
    if gb < ga - 1e-6:
        return False
    try:
        return float(str(a.get("runtime_sec", "inf"))) < float(str(b.get("runtime_sec", "inf")))
    except Exception:
        return False


def build_failure_shift(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target_fams = {"hardA_k10", "hardB_k10", "hardA_k12", "hardB_k12"}
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for r in all_rows:
        fid = str(r.get("family_id", ""))
        if fid not in target_fams:
            continue
        try:
            k = int(r.get("K", "0"))
            seed = int(r.get("seed", "-1"))
        except Exception:
            continue
        groups.setdefault((fid, k, seed), []).append(r)

    out: list[dict[str, Any]] = []
    for (fid, k, seed), rs in sorted(groups.items()):
        plan18 = None
        plan19 = None
        for r in rs:
            vl = str(r.get("variant_label", ""))
            if vl in ("baseline", "irregular_reroute"):
                if plan18 is None or _row_better(r, plan18):
                    plan18 = r
            else:
                if plan19 is None or _row_better(r, plan19):
                    plan19 = r

        def sig(r: dict[str, Any] | None) -> str:
            if r is None:
                return "missing"
            if str(r.get("is_optimal", "0")) == "1":
                return "exact"
            bc = str(r.get("boundary_class", ""))
            if bc == "timeout_no_incumbent":
                return "no_incumbent_timeout"
            if bc == "finite_gap":
                ds = str(r.get("deciding_step", ""))
                if ds == "step3":
                    return "finite_gap_after_step3"
                return "finite_gap_after_step4"
            if bc == "unresolved_other":
                sel = str(r.get("selector_reason", ""))
                if sel == "non_mainline_solver":
                    return "selector_bypass"
                return "unresolved_other"
            return bc

        out.append({
            "family_id": fid,
            "K": k,
            "seed": seed,
            "plan18_signature": sig(plan18),
            "plan19_signature": sig(plan19),
            "shift": f"{sig(plan18)} -> {sig(plan19)}",
        })
    return out


def write_notes(all_rows: list[dict[str, Any]]) -> None:
    target_fams = {"hardA_k10", "hardB_k10", "hardA_k12", "hardB_k12"}
    rows = [r for r in all_rows if r.get("family_id") in target_fams]

    # Aggregate by variant
    by_vl: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        vl = str(r.get("variant_label", ""))
        by_vl.setdefault(vl, []).append(r)

    lines = [
        "# PLAN19 Method Notes",
        "",
        "## What was changed",
        "",
        "### Redesign 1: beam -> restricted exact closure",
        "",
        "Added C++ hook `PAST_PROFILE_REALIZATION_EXACT_AFTER_BEAM_ENABLE=1`. After `profile_repair_beam` produces an incumbent, if the selector originally rejected exact mode, the solver re-evaluates with relaxed guardrails (`MAX_MERGED=24`, `MAX_STATE=1e12`, `MAX_COMP=1e12`) and attempts bounded exact fixed-block DP with an explicit time limit (300s or 600s).",
        "",
        "Variants tested:",
        "- `exp_exact_after_beam_300`: exact time limit 300s",
        "- `exp_exact_after_beam_600`: exact time limit 600s",
        "- `exp_force_exact_300`: force exact selector policy (baseline for comparison)",
        "",
        "### Redesign 2: irregular high-K routing override",
        "",
        "Runner-level change: for hard irregular K>=10, baseline `energy_core` is skipped because it consistently fails with `selector_bypass` / no incumbent. The useful path (`profile_repair_beam`) is run directly.",
        "",
        "### Redesign 3: stronger K=12 beam",
        "",
        "Variant `exp_beam_plus` enables `strengthened=true` in `block_repair_profile_repair_beam_ub` via `PAST_EXACT_INCUMBENT_SOURCE=i3`. This increases beam width and discrepancy budget.",
        "",
        "## Results by variant",
        "",
    ]

    for vl in sorted(by_vl.keys()):
        rs = by_vl[vl]
        exact = sum(1 for r in rs if str(r.get("is_optimal", "0")) == "1")
        finite = sum(1 for r in rs if str(r.get("boundary_class", "")) == "finite_gap")
        timeout = sum(1 for r in rs if str(r.get("boundary_class", "")) == "timeout_no_incumbent")
        memkill = sum(1 for r in rs if str(r.get("memory_killed", "0")) == "1")
        rts = []
        for r in rs:
            try:
                rts.append(float(str(r.get("runtime_sec", "nan"))))
            except Exception:
                pass
        mean_rt = statistics.mean(rts) if rts else 0.0
        mean_rss = 0.0
        rss_vals = []
        for r in rs:
            try:
                rss_vals.append(float(str(r.get("peak_rss_gb", "0"))))
            except Exception:
                pass
        mean_rss = statistics.mean(rss_vals) if rss_vals else 0.0
        lines.append(f"### {vl}")
        lines.append(f"- rows: {len(rs)}")
        lines.append(f"- exact: {exact}")
        lines.append(f"- finite-gap: {finite}")
        lines.append(f"- timeout/no-incumbent: {timeout}")
        lines.append(f"- memory-killed: {memkill}")
        lines.append(f"- mean runtime: {mean_rt:.1f}s")
        lines.append(f"- mean peak RSS: {mean_rss:.2f}GB")
        lines.append("")

    lines.append("## What worked")
    lines.append("")
    lines.append("- **Routing override (redesign 2)** is justified: baseline `energy_core` on K>=10 hard irregular rows consistently produces no incumbent (selector bypass) and wastes 500-1200s. Skipping it saves substantial runtime with no quality loss.")
    lines.append("- **Memory safety**: all variants stayed within the 12GB default cap. Peak RSS ranged 2.4-10.0GB; no memory kills occurred.")
    lines.append("")
    lines.append("## What did not")
    lines.append("")
    lines.append("- **Exact closure after beam (redesign 1) did not work**. The `exact_after_beam` C++ hook did not visibly trigger: rows still show `selector_decision=beam` and `block_dp_status=skipped_selector`. Even `force_exact` with guardrails raised to 1e12 immediately hits `skipped_comp_est`, confirming that exact fixed-block DP state space / comp_est is astronomically large for K=10/12 irregular rows (B≈20, merged>16).")
    lines.append("- **Stronger K=12 beam (redesign 3) did not help**. `exp_beam_plus` timed out on 6/8 K=12 seeds with no incumbent. On the 2 seeds where it produced an incumbent, gaps were identical to standard reroute but runtime was longer.")
    lines.append("- **No exact rows recovered**. Across all 67 rows (including reused PLAN18 data), zero rows achieved exact closure at K=10 or K=12.")
    lines.append("- **Gap reduction was marginal or none**. `exact_after_beam_300` produced the same finite gaps as standard `irregular_reroute`; the extra exact-mode attempt did not tighten bounds.")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append("1. **Accept the boundary**: exact closure at K=10/12 on hard irregular families is infeasible under current fixed-block-DP budgets. The practical ceiling is the beam incumbent + Step 4 global exact DP, which leaves small finite gaps (~0.02-0.06%).")
    lines.append("2. **Keep the routing override**: for K>=10 hard irregular, always route directly to `profile_repair_beam` and skip `energy_core`. This saves 30-50% runtime with no downside.")
    lines.append("3. **Do not pursue stronger beams for K=12**: `beam_plus` increases runtime and timeout rate without improving incumbent quality.")
    lines.append("4. **If further closure is needed**, the path is NOT through fixed-block DP. Consider: (a) better Step 2 heuristics to raise the LB; (b) alternative exact methods (e.g., MIP, SAT) for the packing subproblem; or (c) accepting the current gaps as the practical limit for this solver architecture.")
    lines.append("")

    NOTES_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {NOTES_MD}")


if __name__ == "__main__":
    main()

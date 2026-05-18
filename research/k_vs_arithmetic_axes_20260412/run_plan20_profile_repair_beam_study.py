#!/usr/bin/env python3
"""
PLAN20: improve and understand profile_repair_beam on hard irregular K=8/10/12 at fixed n=1000.

Phases:
A. Beam diagnostics — run representative rows with patched runner to collect beam metrics
B. Implement 2–4 bounded beam refinement variants based on Phase A analysis
C. Measure variants against current best route
D. Build artifacts and decide what is worth keeping

Scope: hardA_k8/k10/k12 and hardB_k8/k10/k12, n=1000, seeds 0–3.
Route policy: profile_repair_beam main; energy_core baseline as reference only.
"""

from __future__ import annotations

import csv
import math
import os
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

OUT_DIR = ROOT / "research" / "k_vs_arithmetic_axes_20260412" / "csv" / "plan20"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = OUT_DIR / "PLAN20_profile_repair_beam_raw.csv"
COMPARE_CSV = OUT_DIR / "PLAN20_profile_repair_beam_compare.csv"
BEST_SUMMARY_CSV = OUT_DIR / "PLAN20_profile_repair_beam_best_summary.csv"
DIAG_CSV = OUT_DIR / "PLAN20_profile_repair_beam_diagnostics.csv"
FAIL_SHIFT_CSV = OUT_DIR / "PLAN20_profile_repair_beam_failure_shift.csv"
NOTES_MD = OUT_DIR / "PLAN20_profile_repair_beam_notes.md"
PHASEA_MD = OUT_DIR / "PLAN20_phaseA_beam_diagnostics.md"

PLAN18_RAW = (
    ROOT
    / "research"
    / "k_vs_arithmetic_axes_20260412"
    / "csv"
    / "plan18"
    / "PLAN18_k_boundary_refine_n1000_raw.csv"
)
PLAN19_RAW = (
    ROOT
    / "research"
    / "k_vs_arithmetic_axes_20260412"
    / "csv"
    / "plan19"
    / "PLAN19_k10_k12_redesign_raw.csv"
)

LAMBDA = 1.3
N_JOBS = 1000
TIME_LIMIT = 1200.0
MAX_RSS_GB = 12.0

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


def build_plan20_payload(
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
        name=f"plan20/{family_id}_n{n_jobs}_lam{lam:.1f}_s{seed}",
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
            "plan20": "1",
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

BASELINE_ENV: dict[str, str] = {}

STANDARD_BEAM_ENV: dict[str, str] = {
    "PAST_RELAXED_BINPACK_SOLVER": "profile_repair_beam",
    "PAST_PROFILE_REALIZATION_SELECTOR_POLICY": "auto_v1",
}

VARIANTS: dict[str, dict[str, str]] = {
    "baseline": BASELINE_ENV,
    "standard_beam": STANDARD_BEAM_ENV,
    # Variant 1: modest width increase
    "beam_wider": {
        **STANDARD_BEAM_ENV,
        "PAST_PROFILE_REPAIR_BEAM_WIDTH_MAX": "900000",  # 4.5x default 200K base
    },
    # Variant 2: deeper discrepancy exploration
    "beam_deeper": {
        **STANDARD_BEAM_ENV,
        "PAST_PROFILE_REPAIR_BEAM_DISC_BUDGET": "2",
        "PAST_PROFILE_REPAIR_BEAM_DISC_DEPTH": "6",
    },
    # Variant 3: local search refinement
    "beam_local": {
        **STANDARD_BEAM_ENV,
        "PAST_PROFILE_REPAIR_BEAM_LOCAL_PASSES": "1",
        "PAST_PROFILE_REPAIR_BEAM_LOCAL_MAX_MERGED": "24",
    },
    # Variant 4: scoring weight shift toward feasibility
    "beam_weighted": {
        **STANDARD_BEAM_ENV,
        "PAST_PROFILE_REPAIR_BEAM_W_FEAS": "1.0",
        "PAST_PROFILE_REPAIR_BEAM_W_LOCAL": "0.8",
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
    payload = build_plan20_payload(family_id, sizes, N_JOBS, LAMBDA, seed)
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
        route_policy=f"plan20:{variant_label}",
    )
    rows.append(row)
    seen.add(key)
    write_csv(RAW_CSV, rows)
    print(
        f"[plan20] {family_id} {variant_label} seed={seed} K={k} "
        f"step={row.get('deciding_step')} opt={row.get('is_optimal')} "
        f"bc={row.get('boundary_class')} gap={row.get('gap_pct')} rt={row.get('runtime_sec')} "
        f"rss={row.get('peak_rss_gb')}GB memkill={row.get('memory_killed')}"
    )
    return row


# ---------------------------------------------------------------------------
# Phase A: Beam diagnostics
# ---------------------------------------------------------------------------

def run_phase_a(rows: list[dict[str, Any]], seen: set[tuple[str, str, int]]) -> None:
    """Run standard_beam on all target families/seeds to collect beam diagnostics."""
    ladders = [
        ("hardA", "hard_irregular_A", HARD_A_BASE),
        ("hardB", "hard_irregular_B", HARD_B_BASE),
    ]
    target_ks = [8, 10, 12]
    seeds = [0, 1, 2, 3]

    for prefix, family_class, base in ladders:
        for target_k in target_ks:
            sizes = base[:target_k]
            fid = f"{prefix}_k{target_k}"
            for seed in seeds:
                # Skip baseline for K>=10 (known useless from PLAN19)
                if target_k <= 8:
                    run_variant(fid, sizes, target_k, seed, "baseline", VARIANTS["baseline"], rows, seen)
                run_variant(fid, sizes, target_k, seed, "standard_beam", VARIANTS["standard_beam"], rows, seen)


def build_phase_a_diagnostics(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract beam structural metrics from standard_beam rows."""
    target_fams = {f"hardA_k{k}" for k in [8, 10, 12]} | {f"hardB_k{k}" for k in [8, 10, 12]}
    rows = [r for r in all_rows if r.get("family_id") in target_fams and r.get("variant_label") == "standard_beam"]

    out: list[dict[str, Any]] = []
    for r in rows:
        def _f(key: str) -> float:
            try:
                return float(str(r.get(key, "nan")))
            except Exception:
                return float("nan")

        def _i(key: str) -> int:
            try:
                return int(float(str(r.get(key, "0"))))
            except Exception:
                return 0

        out.append({
            "family_id": str(r.get("family_id", "")),
            "K": str(r.get("K", "")),
            "seed": str(r.get("seed", "")),
            "is_optimal": str(r.get("is_optimal", "0")),
            "gap_pct": str(r.get("gap_pct", "nan")),
            "ub": str(r.get("ub", "")),
            "lb": str(r.get("lb", "")),
            "runtime_sec": str(r.get("runtime_sec", "")),
            "t_fwd_pack_profile_beam": str(r.get("t_fwd_pack_profile_beam", "")),
            "beam_base_width": str(r.get("fwd_profile_beam_base_width", "")),
            "beam_avg_width": str(r.get("fwd_profile_beam_avg_width", "")),
            "beam_max_width": str(r.get("fwd_profile_beam_max_width", "")),
            "beam_states_considered": str(r.get("fwd_profile_beam_states_considered", "")),
            "beam_states_kept": str(r.get("fwd_profile_beam_states_kept", "")),
            "beam_pruned_over": str(r.get("fwd_profile_beam_pruned_over", "")),
            "beam_pruned_suffix": str(r.get("fwd_profile_beam_pruned_suffix", "")),
            "beam_pruned_discrepancy": str(r.get("fwd_profile_beam_pruned_discrepancy", "")),
            "beam_discrepancy_budget": str(r.get("fwd_profile_beam_discrepancy_budget", "")),
            "beam_discrepancy_depth": str(r.get("fwd_profile_beam_discrepancy_depth", "")),
            "beam_status": str(r.get("fwd_profile_beam_status", "")),
            "beam_timed_out": str(r.get("fwd_profile_beam_timed_out", "0")),
            "beam_candidate_ub": str(r.get("fwd_profile_beam_candidate_ub", "")),
            "beam_improved_over_step2": str(r.get("fwd_profile_beam_improved_over_step2", "0")),
            "deciding_step": str(r.get("deciding_step", "")),
            "boundary_class": str(r.get("boundary_class", "")),
            "boundary_detail": str(r.get("boundary_detail", "")),
            "peak_rss_gb": str(r.get("peak_rss_gb", "")),
        })
    return out


def write_phase_a_analysis(diag_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# PLAN20 Phase A: Beam Diagnostics",
        "",
        "## Method",
        "",
        "Ran `profile_repair_beam` with `auto_v1` selector on hard irregular K=8/10/12, n=1000, seeds 0-3.",
        "Beam structural metrics are now extracted from the solver via the patched runner.",
        "",
        "## Results by K",
        "",
    ]

    by_k: dict[int, list[dict[str, Any]]] = {}
    for r in diag_rows:
        try:
            k = int(r.get("K", "0"))
        except Exception:
            continue
        by_k.setdefault(k, []).append(r)

    for k in sorted(by_k.keys()):
        rs = by_k[k]
        exact = sum(1 for r in rs if str(r.get("is_optimal", "0")) == "1")
        n = len(rs)

        def _fvals(key: str) -> list[float]:
            out = []
            for r in rs:
                try:
                    v = float(str(r.get(key, "nan")))
                    if not math.isnan(v):
                        out.append(v)
                except Exception:
                    pass
            return out

        def _ivals(key: str) -> list[int]:
            out = []
            for r in rs:
                try:
                    v = int(float(str(r.get(key, "0"))))
                    out.append(v)
                except Exception:
                    pass
            return out

        beam_times = _fvals("t_fwd_pack_profile_beam")
        considered = _ivals("beam_states_considered")
        kept = _ivals("beam_states_kept")
        widths = _fvals("beam_avg_width")
        max_widths = _fvals("beam_max_width")
        timeouts = sum(1 for r in rs if str(r.get("beam_timed_out", "0")) == "1")

        lines.append(f"### K={k} (n={n}, exact={exact})")
        lines.append("")
        if beam_times:
            lines.append(f"- Beam stage runtime: {statistics.mean(beam_times):.1f}s mean, {min(beam_times):.1f}s min, {max(beam_times):.1f}s max")
        if considered:
            lines.append(f"- States considered: {statistics.mean(considered):.0f} mean, {min(considered)} min, {max(considered)} max")
        if kept:
            lines.append(f"- States kept: {statistics.mean(kept):.0f} mean, {min(kept)} min, {max(kept)} max")
        if widths:
            lines.append(f"- Avg width: {statistics.mean(widths):.1f} mean, {min(widths):.1f} min, {max(widths):.1f} max")
        if max_widths:
            lines.append(f"- Max width: {statistics.mean(max_widths):.1f} mean, {min(max_widths):.1f} min, {max(max_widths):.1f} max")
        lines.append(f"- Beam timeouts: {timeouts}/{n}")
        lines.append("")

        # Per-row detail
        lines.append("**Per-row details:**")
        for r in sorted(rs, key=lambda x: (str(x.get("family_id")), int(x.get("seed", 0)))):
            fid = r.get("family_id", "")
            seed = r.get("seed", "")
            opt = r.get("is_optimal", "0")
            gap = r.get("gap_pct", "nan")
            bt = r.get("t_fwd_pack_profile_beam", "")
            st = r.get("beam_status", "")
            to = r.get("beam_timed_out", "0")
            cons = r.get("beam_states_considered", "")
            kept = r.get("beam_states_kept", "")
            lines.append(f"- {fid} s={seed} opt={opt} gap={gap}% beam_time={bt}s status={st} timed_out={to} considered={cons} kept={kept}")
        lines.append("")

    lines.append("## Observations & Diagnosis")
    lines.append("")

    # Automatic observations
    k10_rows = by_k.get(10, [])
    k12_rows = by_k.get(12, [])

    if k10_rows:
        timeouts_k10 = sum(1 for r in k10_rows if str(r.get("beam_timed_out", "0")) == "1")
        finite_gaps_k10 = sum(1 for r in k10_rows if str(r.get("is_optimal", "0")) != "1" and str(r.get("gap_pct", "")) not in ("", "nan", "inf"))
        if timeouts_k10 == 0 and finite_gaps_k10 > 0:
            lines.append("- **K=10**: Beam completes consistently but Step 4 exact DP fails to close gaps. The bottleneck is closure after incumbent production, not beam search itself.")
        elif timeouts_k10 > 0:
            lines.append(f"- **K=10**: Beam times out on {timeouts_k10}/{len(k10_rows)} rows. Incumbent production is partially failing.")

    if k12_rows:
        timeouts_k12 = sum(1 for r in k12_rows if str(r.get("beam_timed_out", "0")) == "1")
        finite_gaps_k12 = sum(1 for r in k12_rows if str(r.get("is_optimal", "0")) != "1" and str(r.get("gap_pct", "")) not in ("", "nan", "inf"))
        if timeouts_k12 > 0:
            lines.append(f"- **K=12**: Beam times out on {timeouts_k12}/{len(k12_rows)} rows. Incumbent production is the primary bottleneck.")
        elif finite_gaps_k12 > 0:
            lines.append(f"- **K=12**: Beam completes but leaves finite gaps. Like K=10, closure is the bottleneck.")

    lines.append("")
    lines.append("## Implications for Variant Design")
    lines.append("")
    lines.append("Based on these diagnostics, the bounded variants should target:")
    lines.append("1. **K=12 timeout rows**: modest width or discrepancy increase to improve incumbent quality without catastrophic slowdown.")
    lines.append("2. **K=10/12 finite-gap rows**: local search refinement or scoring adjustments to improve incumbent quality.")
    lines.append("3. **Avoid**: simultaneous broad increases (width + discrepancy + local passes) as tested in PLAN19 `beam_plus`, which caused timeouts.")
    lines.append("")

    PHASEA_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {PHASEA_MD}")


# ---------------------------------------------------------------------------
# Phase B: Bounded variants
# ---------------------------------------------------------------------------

def run_phase_b(rows: list[dict[str, Any]], seen: set[tuple[str, str, int]]) -> None:
    """Run 4 bounded beam variants on all target families/seeds."""
    ladders = [
        ("hardA", "hard_irregular_A", HARD_A_BASE),
        ("hardB", "hard_irregular_B", HARD_B_BASE),
    ]
    target_ks = [8, 10, 12]
    seeds = [0, 1, 2, 3]
    variant_labels = ["beam_wider", "beam_deeper", "beam_local", "beam_weighted"]

    for prefix, family_class, base in ladders:
        for target_k in target_ks:
            sizes = base[:target_k]
            fid = f"{prefix}_k{target_k}"
            for seed in seeds:
                for vl in variant_labels:
                    run_variant(fid, sizes, target_k, seed, vl, VARIANTS[vl], rows, seen)


# ---------------------------------------------------------------------------
# Phase C: Artifact builders
# ---------------------------------------------------------------------------

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
    target_fams = {f"hardA_k{k}" for k in [8, 10, 12]} | {f"hardB_k{k}" for k in [8, 10, 12]}
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

    def _effective_gap(r: dict[str, Any]) -> float:
        if str(r.get("is_optimal", "0")) == "1":
            return 0.0
        try:
            ub = float(str(r.get("ub", "-1")))
            lb = float(str(r.get("lb", "-1")))
            if ub < 0 or lb < 0:
                return float("inf")
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

    out: list[dict[str, Any]] = []
    for (fid, k, seed), rs in sorted(groups.items()):
        standard = None
        best_variant = None
        for r in rs:
            vl = str(r.get("variant_label", ""))
            if vl == "standard_beam":
                standard = r
            else:
                if best_variant is None or _row_better(r, best_variant):
                    best_variant = r

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

        std = extract(standard)
        bst = extract(best_variant)

        winner = "tie"
        if bst["opt"] == "1" and std["opt"] != "1":
            winner = "variant"
        elif std["opt"] == "1" and bst["opt"] != "1":
            winner = "standard"
        else:
            try:
                gs = float(std["gap"])
                gb = float(bst["gap"])
                if gb < gs - 1e-6:
                    winner = "variant"
                elif gs < gb - 1e-6:
                    winner = "standard"
                else:
                    try:
                        if float(bst["rt"]) < float(std["rt"]):
                            winner = "variant_runtime"
                        elif float(std["rt"]) < float(bst["rt"]):
                            winner = "standard_runtime"
                    except Exception:
                        pass
            except Exception:
                if bst["gap"] != "inf" and std["gap"] == "inf":
                    winner = "variant"
                elif std["gap"] != "inf" and bst["gap"] == "inf":
                    winner = "standard"

        out.append({
            "family_id": fid,
            "K": k,
            "seed": seed,
            "standard_opt": std["opt"],
            "standard_gap": std["gap"],
            "standard_rt": std["rt"],
            "best_variant": bst["variant"],
            "best_variant_opt": bst["opt"],
            "best_variant_gap": bst["gap"],
            "best_variant_rt": bst["rt"],
            "winner": winner,
        })
    return out


def build_failure_shift(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target_fams = {f"hardA_k{k}" for k in [8, 10, 12]} | {f"hardB_k{k}" for k in [8, 10, 12]}
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
        if bc == "memory_failure":
            return "memory_failure"
        if bc == "crash":
            return "crash"
        return "unresolved_other"

    out: list[dict[str, Any]] = []
    for (fid, k, seed), rs in sorted(groups.items()):
        standard = None
        best = None
        for r in rs:
            vl = str(r.get("variant_label", ""))
            if vl == "standard_beam":
                standard = r
            elif best is None:
                best = r
            else:
                # pick best non-standard
                opt_cur = str(best.get("is_optimal", "0")) == "1"
                opt_new = str(r.get("is_optimal", "0")) == "1"
                if opt_new and not opt_cur:
                    best = r
                elif opt_new == opt_cur:
                    try:
                        gap_cur = float(str(best.get("gap_pct", "inf")))
                        gap_new = float(str(r.get("gap_pct", "inf")))
                        if gap_new < gap_cur - 1e-6:
                            best = r
                    except Exception:
                        pass

        out.append({
            "family_id": fid,
            "K": k,
            "seed": seed,
            "standard_signature": sig(standard),
            "best_variant_signature": sig(best),
            "shift": f"{sig(standard)} -> {sig(best)}",
        })
    return out


def write_notes(all_rows: list[dict[str, Any]]) -> None:
    target_fams = {f"hardA_k{k}" for k in [8, 10, 12]} | {f"hardB_k{k}" for k in [8, 10, 12]}
    rows = [r for r in all_rows if r.get("family_id") in target_fams]

    by_vl: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        vl = str(r.get("variant_label", ""))
        by_vl.setdefault(vl, []).append(r)

    lines = [
        "# PLAN20 Method Notes",
        "",
        "## Goal",
        "",
        "Improve and understand `profile_repair_beam` on hard irregular K=8/10/12 at fixed n=1000 through bounded beam refinements and diagnostics.",
        "",
        "## Variants Tested",
        "",
    ]

    variant_descriptions = {
        "baseline": "Standard energy_core baseline (reference only, K<=8)",
        "standard_beam": "Default profile_repair_beam with auto_v1 selector (current best route)",
        "beam_wider": "Modest width increase: WIDTH_MAX=900000 (4.5x base vs default 3x)",
        "beam_deeper": "Deeper discrepancy: DISC_BUDGET=2, DISC_DEPTH=6 (vs default 1, 4)",
        "beam_local": "Local search: LOCAL_PASSES=1, LOCAL_MAX_MERGED=24 (vs default 0, 32)",
        "beam_weighted": "Scoring shift: W_FEAS=1.0, W_LOCAL=0.8 (vs default 0.75, 0.6)",
    }

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
        rss_vals = []
        for r in rs:
            try:
                rss_vals.append(float(str(r.get("peak_rss_gb", "0"))))
            except Exception:
                pass
        mean_rss = statistics.mean(rss_vals) if rss_vals else 0.0
        lines.append(f"### {vl}")
        lines.append(variant_descriptions.get(vl, ""))
        lines.append(f"- rows: {len(rs)}")
        lines.append(f"- exact: {exact}")
        lines.append(f"- finite-gap: {finite}")
        lines.append(f"- timeout/no-incumbent: {timeout}")
        lines.append(f"- memory-killed: {memkill}")
        lines.append(f"- mean runtime: {mean_rt:.1f}s")
        lines.append(f"- mean peak RSS: {mean_rss:.2f}GB")
        lines.append("")

    lines.append("## Results Summary")
    lines.append("")

    # Compare standard vs best variant
    compare = build_compare(all_rows)
    wins = {"variant": 0, "standard": 0, "tie": 0, "variant_runtime": 0, "standard_runtime": 0}
    for c in compare:
        w = c.get("winner", "tie")
        wins[w] = wins.get(w, 0) + 1

    lines.append(f"- Standard beam wins: {wins.get('standard', 0)} (exact or better gap)")
    lines.append(f"- Variant wins: {wins.get('variant', 0)} (exact or better gap)")
    lines.append(f"- Runtime ties: {wins.get('variant_runtime', 0)} variant, {wins.get('standard_runtime', 0)} standard")
    lines.append(f"- True ties: {wins.get('tie', 0)}")
    lines.append("")

    # Failure shift
    fail_shift = build_failure_shift(all_rows)
    improved = sum(1 for f in fail_shift if "exact" in f.get("shift", ""))
    worsened = sum(1 for f in fail_shift if "timeout" in f.get("shift", "") and "exact" not in f.get("shift", ""))
    lines.append(f"- Failure mode improvements: {improved} rows shifted toward exact")
    lines.append(f"- Failure mode worsening: {worsened} rows shifted toward timeout")
    lines.append("")

    lines.append("## What Worked")
    lines.append("")
    lines.append("(To be filled after experiments complete.)")
    lines.append("")
    lines.append("## What Did Not")
    lines.append("")
    lines.append("(To be filled after experiments complete.)")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append("(To be filled after experiments complete.)")
    lines.append("")

    NOTES_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {NOTES_MD}")


def build_artifacts(all_rows: list[dict[str, Any]]) -> None:
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

    diag = build_phase_a_diagnostics(all_rows)
    write_csv(DIAG_CSV, diag)
    print(f"Wrote {DIAG_CSV} n={len(diag)}")

    write_phase_a_analysis(diag)
    write_notes(all_rows)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-after-phase-a", action="store_true", help="Stop after Phase A diagnostics")
    parser.add_argument("--phase-b-variants", type=str, default="", help="Comma-separated variant labels for Phase B")
    parser.add_argument("--phase-b-ks", type=str, default="", help="Comma-separated K values for Phase B (e.g., 10,12)")
    args = parser.parse_args()

    solver = ROOT / "solvers" / "cpp" / "build" / "stateful_compare"
    if not solver.exists():
        raise FileNotFoundError(f"Missing solver binary: {solver}")

    rows: list[dict[str, Any]] = load_raw(RAW_CSV)
    seen = {row_key(r) for r in rows}

    # Reuse PLAN18/19 standard_beam and baseline rows if present, but we need
    # beam diagnostics so only reuse if the row already has beam fields.
    for src_path in [PLAN18_RAW, PLAN19_RAW]:
        if src_path.exists():
            for r in load_raw(src_path):
                fid = str(r.get("family_id", ""))
                vl = str(r.get("variant_label", ""))
                if fid.startswith(("hardA_k", "hardB_k")) and vl in ("baseline", "irregular_reroute"):
                    # Map irregular_reroute -> standard_beam for consistency
                    if vl == "irregular_reroute":
                        r = dict(r)
                        r["variant_label"] = "standard_beam"
                        r["route_policy"] = "plan20:standard_beam"
                        vl = "standard_beam"
                    key = (fid, vl, int(r.get("seed", -1)))
                    if key not in seen:
                        # Check if it has beam diagnostics
                        has_beam = any(
                            r.get(f"fwd_profile_beam_{f}")
                            for f in ["base_width", "avg_width", "max_width", "status"]
                        )
                        if has_beam:
                            rows.append(dict(r))
                            seen.add(key)
                            print(f"reused {vl} {fid} seed={r.get('seed')} (has beam diagnostics)")
                        else:
                            print(f"skip reused {vl} {fid} seed={r.get('seed')} (no beam diagnostics, will rerun)")

    # Phase A: Run standard_beam (and baseline for K<=8) to collect beam diagnostics
    print("\n=== PHASE A: Beam diagnostics ===")
    run_phase_a(rows, seen)

    # Build Phase A artifacts immediately so we can diagnose before Phase B
    diag = build_phase_a_diagnostics(rows)
    write_csv(DIAG_CSV, diag)
    write_phase_a_analysis(diag)

    if args.stop_after_phase_a:
        print("\n=== STOPPED AFTER PHASE A (as requested) ===")
        return

    # Phase B: Run bounded variants
    print("\n=== PHASE B: Bounded beam variants ===")
    if args.phase_b_variants:
        variant_labels = [v.strip() for v in args.phase_b_variants.split(",") if v.strip()]
    else:
        variant_labels = ["beam_wider", "beam_deeper", "beam_local", "beam_weighted"]

    if args.phase_b_ks:
        target_ks = [int(v.strip()) for v in args.phase_b_ks.split(",") if v.strip()]
    else:
        target_ks = [8, 10, 12]

    ladders = [
        ("hardA", "hard_irregular_A", HARD_A_BASE),
        ("hardB", "hard_irregular_B", HARD_B_BASE),
    ]
    seeds = [0, 1, 2, 3]

    for prefix, family_class, base in ladders:
        for target_k in target_ks:
            sizes = base[:target_k]
            fid = f"{prefix}_k{target_k}"
            for seed in seeds:
                for vl in variant_labels:
                    if vl not in VARIANTS:
                        print(f"WARNING: unknown variant {vl}, skipping")
                        continue
                    run_variant(fid, sizes, target_k, seed, vl, VARIANTS[vl], rows, seen)

    # Phase C: Build all artifacts
    print("\n=== PHASE C: Build artifacts ===")
    build_artifacts(rows)
    print("PLAN20 complete.")


if __name__ == "__main__":
    main()

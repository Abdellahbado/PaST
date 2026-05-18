#!/usr/bin/env python3
"""
PLAN20B: focused beam-survivor study on 4 anchor rows.

Anchor rows:
- hardA_k8, seed=1
- hardA_k8, seed=3
- hardA_k10, seed=0
- hardA_k10, seed=1

Experiments:
1. Per-key multiplicity
2. Diversity-aware survivor buckets
3. Score ablation
4. Discrepancy relevance
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

OUT_DIR = ROOT / "research" / "k_vs_arithmetic_axes_20260412" / "csv" / "plan20b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = OUT_DIR / "PLAN20B_beam_survivor_raw.csv"
COMPARE_CSV = OUT_DIR / "PLAN20B_beam_survivor_compare.csv"
SUMMARY_CSV = OUT_DIR / "PLAN20B_beam_survivor_summary.csv"
DIAG_MD = OUT_DIR / "PLAN20B_beam_survivor_diagnostics.md"

LAMBDA = 1.3
N_JOBS = 1000
TIME_LIMIT = 1200.0
MAX_RSS_GB = 12.0

HARD_A_BASE = [
    2, 3, 4, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67,
]

EC_CONFIGS = [
    {"from_date": "2019-01-21T00:00:00", "repeat_count": 1},
    {"from_date": "2019-04-08T00:00:00", "repeat_count": 1},
]


def sizes_label(sizes: list[int]) -> str:
    return "{" + ",".join(str(x) for x in sizes) + "}"


def build_plan20b_payload(
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
        name=f"plan20b/{family_id}_n{n_jobs}_lam{lam:.1f}_s{seed}",
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
            "plan20b": "1",
        },
    )
    return {
        "instance_id": inst["name"],
        "prices": inst["prices"],
        "jobs": inst["jobs"],
        "machine": "twosby",
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
    payload = build_plan20b_payload(family_id, sizes, N_JOBS, LAMBDA, seed)
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
    # Normalize and add experiment_family
    raw["experiment_family"] = variant_label.split("_")[0] if "_" in variant_label else "baseline"
    raw["family_id"] = family_id
    raw["K"] = str(k)
    raw["family_label"] = sizes_label(sizes)
    raw["family_class"] = "hard_irregular_A"
    raw["family_sizes"] = ",".join(str(x) for x in sizes)
    rows.append(raw)
    seen.add(key)
    write_csv(RAW_CSV, rows)
    print(
        f"[plan20b] {family_id} {variant_label} seed={seed} K={k} "
        f"step={raw.get('deciding_step')} opt={raw.get('is_optimal')} "
        f"gap={raw.get('gap_pct')} rt={raw.get('runtime_sec')} "
        f"rss={raw.get('peak_rss_gb')}GB memkill={raw.get('memory_killed')}"
    )
    return raw


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


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

STANDARD_BEAM_ENV: dict[str, str] = {
    "PAST_RELAXED_BINPACK_SOLVER": "profile_repair_beam",
    "PAST_PROFILE_REALIZATION_SELECTOR_POLICY": "auto_v1",
}

VARIANTS: dict[str, dict[str, str]] = {
    "standard_beam": STANDARD_BEAM_ENV,
    # Experiment 1: per-key multiplicity
    "multiplicity_2": {
        **STANDARD_BEAM_ENV,
        "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI": "2",
    },
    "multiplicity_3": {
        **STANDARD_BEAM_ENV,
        "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI": "3",
    },
    # Experiment 2: diversity buckets
    "bucket_70_30_feas": {
        **STANDARD_BEAM_ENV,
        "PAST_PROFILE_REPAIR_BEAM_BUCKET_SPLIT": "70,30",
        "PAST_PROFILE_REPAIR_BEAM_BUCKET_METRIC": "feas",
    },
    "bucket_70_30_local": {
        **STANDARD_BEAM_ENV,
        "PAST_PROFILE_REPAIR_BEAM_BUCKET_SPLIT": "70,30",
        "PAST_PROFILE_REPAIR_BEAM_BUCKET_METRIC": "local",
    },
    # Experiment 3: score ablation
    "weight_center_low": {
        **STANDARD_BEAM_ENV,
        "PAST_PROFILE_REPAIR_BEAM_W_CENTER": "0.5",
    },
    "weight_feas_high": {
        **STANDARD_BEAM_ENV,
        "PAST_PROFILE_REPAIR_BEAM_W_FEAS": "1.25",
    },
    "weight_arith_high": {
        **STANDARD_BEAM_ENV,
        "PAST_PROFILE_REPAIR_BEAM_W_ARITH": "1.5",
    },
    # Experiment 4: discrepancy
    "disc_off": {
        **STANDARD_BEAM_ENV,
        "PAST_PROFILE_REPAIR_BEAM_DISC_BUDGET": "0",
        "PAST_PROFILE_REPAIR_BEAM_DISC_DEPTH": "0",
    },
    "disc_deep": {
        **STANDARD_BEAM_ENV,
        "PAST_PROFILE_REPAIR_BEAM_DISC_BUDGET": "3",
        "PAST_PROFILE_REPAIR_BEAM_DISC_DEPTH": "8",
    },
}

ANCHORS = [
    ("hardA_k8", 8, [1, 3]),
    ("hardA_k10", 10, [0, 1]),
]


def main() -> None:
    solver = ROOT / "solvers" / "cpp" / "build" / "stateful_compare"
    if not solver.exists():
        raise FileNotFoundError(f"Missing solver binary: {solver}")

    rows: list[dict[str, Any]] = load_raw(RAW_CSV)
    seen = {row_key(r) for r in rows}

    for family_id, k, seeds in ANCHORS:
        sizes = HARD_A_BASE[:k]
        for seed in seeds:
            for vl, env in VARIANTS.items():
                run_variant(family_id, sizes, k, seed, vl, env, rows, seen)

    # Build artifacts
    build_artifacts(rows)
    print("PLAN20B complete.")


def build_artifacts(all_rows: list[dict[str, Any]]) -> None:
    write_csv(RAW_CSV, all_rows)
    print(f"Wrote {RAW_CSV} n={len(all_rows)}")

    compare = build_compare(all_rows)
    write_csv(COMPARE_CSV, compare)
    print(f"Wrote {COMPARE_CSV} n={len(compare)}")

    summary = build_summary(all_rows)
    write_csv(SUMMARY_CSV, summary)
    print(f"Wrote {SUMMARY_CSV} n={len(summary)}")

    write_diagnostics(all_rows)


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


def build_compare(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each anchor, compare standard_beam vs each variant."""
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
        standard = None
        variants: list[dict[str, Any]] = []
        for r in rs:
            vl = str(r.get("variant_label", ""))
            if vl == "standard_beam":
                standard = r
            else:
                variants.append(r)

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

        for vr in variants:
            v = extract(vr)
            winner = "tie"
            if v["opt"] == "1" and std["opt"] != "1":
                winner = "variant"
            elif std["opt"] == "1" and v["opt"] != "1":
                winner = "standard"
            else:
                try:
                    gs = float(std["gap"])
                    gv = float(v["gap"])
                    if gv < gs - 1e-6:
                        winner = "variant"
                    elif gs < gv - 1e-6:
                        winner = "standard"
                    else:
                        try:
                            if float(v["rt"]) < float(std["rt"]):
                                winner = "variant_runtime"
                            elif float(std["rt"]) < float(v["rt"]):
                                winner = "standard_runtime"
                        except Exception:
                            pass
                except Exception:
                    if v["gap"] != "inf" and std["gap"] == "inf":
                        winner = "variant"
                    elif std["gap"] != "inf" and v["gap"] == "inf":
                        winner = "standard"

            out.append({
                "family_id": fid,
                "K": k,
                "seed": seed,
                "standard_opt": std["opt"],
                "standard_gap": std["gap"],
                "standard_rt": std["rt"],
                "variant_label": v["variant"],
                "variant_opt": v["opt"],
                "variant_gap": v["gap"],
                "variant_rt": v["rt"],
                "winner": winner,
            })
    return out


def build_summary(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate by variant."""
    by_vl: dict[str, list[dict[str, Any]]] = {}
    for r in all_rows:
        vl = str(r.get("variant_label", ""))
        by_vl.setdefault(vl, []).append(r)

    out: list[dict[str, Any]] = []
    for vl in sorted(by_vl.keys()):
        rs = by_vl[vl]
        exact = sum(1 for r in rs if str(r.get("is_optimal", "0")) == "1")
        finite = sum(1 for r in rs if _effective_gap(r) < float("inf") and str(r.get("is_optimal", "0")) != "1")
        timeout = sum(1 for r in rs if str(r.get("external_timed_out", "0")) == "1" or str(r.get("timed_out", "0")) == "1")
        memkill = sum(1 for r in rs if str(r.get("memory_killed", "0")) == "1")
        gaps = [_effective_gap(r) for r in rs if _effective_gap(r) < float("inf")]
        mean_gap = statistics.mean(gaps) if gaps else 0.0
        min_gap = min(gaps) if gaps else 0.0
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

        out.append({
            "variant_label": vl,
            "n_rows": len(rs),
            "exact": exact,
            "finite_gap": finite,
            "timeout": timeout,
            "memory_killed": memkill,
            "mean_gap_pct": f"{mean_gap:.4f}" if mean_gap > 0 else "0",
            "min_gap_pct": f"{min_gap:.4f}" if min_gap > 0 else "0",
            "mean_runtime_sec": f"{mean_rt:.1f}",
            "mean_peak_rss_gb": f"{mean_rss:.2f}",
        })
    return out


def write_diagnostics(all_rows: list[dict[str, Any]]) -> None:
    # Build compare for analysis
    compare = build_compare(all_rows)
    summary = build_summary(all_rows)

    # Count wins by experiment family
    wins_by_family: dict[str, dict[str, int]] = {}
    for c in compare:
        fam = str(c.get("variant_label", "")).split("_")[0]
        w = c.get("winner", "tie")
        wins_by_family.setdefault(fam, {}).setdefault(w, 0)
        wins_by_family[fam][w] += 1

    lines = [
        "# PLAN20B Beam Survivor Diagnostics",
        "",
        "## Anchor rows",
        "",
        "- hardA_k8, seeds 1, 3 (near-frontier K=8 failures)",
        "- hardA_k10, seeds 0, 1 (representative K=10 finite-gap failures)",
        "",
        "## Results by experiment family",
        "",
    ]

    for fam in ["multiplicity", "bucket", "weight", "disc"]:
        wins = wins_by_family.get(fam, {})
        variant_wins = wins.get("variant", 0) + wins.get("variant_runtime", 0)
        standard_wins = wins.get("standard", 0) + wins.get("standard_runtime", 0)
        ties = wins.get("tie", 0)
        lines.append(f"### {fam}")
        lines.append(f"- variant wins (better gap or runtime): {variant_wins}")
        lines.append(f"- standard wins: {standard_wins}")
        lines.append(f"- ties: {ties}")
        lines.append("")

    # Summary table
    lines.append("## Summary by variant")
    lines.append("")
    lines.append("| variant | rows | exact | finite | timeout | mean_gap% | mean_rt(s) | mean_rss(GB) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in summary:
        lines.append(
            f"| {s['variant_label']} | {s['n_rows']} | {s['exact']} | {s['finite_gap']} | {s['timeout']} | "
            f"{s['mean_gap_pct']} | {s['mean_runtime_sec']} | {s['mean_peak_rss_gb']} |"
        )
    lines.append("")

    # Beam diagnostics from raw rows
    lines.append("## Beam structural diagnostics")
    lines.append("")
    for r in sorted(all_rows, key=lambda x: (str(x.get("family_id")), int(x.get("seed", 0)), str(x.get("variant_label")))):
        if str(r.get("variant_label", "")) == "standard_beam":
            fid = r.get("family_id", "")
            seed = r.get("seed", "")
            bw = r.get("fwd_profile_beam_base_width", "")
            aw = r.get("fwd_profile_beam_avg_width", "")
            cons = r.get("fwd_profile_beam_states_considered", "")
            kept = r.get("fwd_profile_beam_states_kept", "")
            lines.append(f"- {fid} s={seed}: base_width={bw} avg_width={aw} considered={cons} kept={kept}")
    lines.append("")

    lines.append("## Hypothesis assessment")
    lines.append("")

    # Auto-generate assessment based on compare results
    best_family = ""
    best_variant_score = -1
    for fam, wins in wins_by_family.items():
        score = wins.get("variant", 0) * 2 + wins.get("variant_runtime", 0)
        if score > best_variant_score:
            best_variant_score = score
            best_family = fam

    if best_family == "multiplicity":
        lines.append("1. **Per-key multiplicity**: Most supported hypothesis. Keeping multiple representatives per key improved results on some anchors.")
    elif best_family == "bucket":
        lines.append("1. **Diversity buckets**: Most supported hypothesis. Splitting survivors across metrics helped some anchors.")
    elif best_family == "weight":
        lines.append("1. **Score weights**: Most supported hypothesis. Shifting score emphasis produced better incumbents.")
    elif best_family == "disc":
        lines.append("1. **Discrepancy**: Most supported hypothesis. Changing discrepancy policy improved results.")
    else:
        lines.append("1. **No clear winner**: None of the beam-survivor experiments showed consistent improvement over standard_beam.")

    lines.append("")
    lines.append("2. **Weakened hypotheses**: (To be filled based on full results)")
    lines.append("")
    lines.append("3. **Next direction recommendation**: (To be filled based on full results)")
    lines.append("")

    # Fill in based on actual data
    any_exact = any(str(r.get("is_optimal", "0")) == "1" for r in all_rows if str(r.get("variant_label", "")) != "standard_beam")
    if any_exact:
        lines.append("- At least one variant recovered exact closure on an anchor row where standard_beam did not.")
    else:
        lines.append("- No variant recovered exact closure on any anchor row.")

    # K=10 gap improvement check
    k10_standard = [_effective_gap(r) for r in all_rows if r.get("family_id") == "hardA_k10" and r.get("variant_label") == "standard_beam" and _effective_gap(r) < float("inf")]
    k10_variants = [_effective_gap(r) for r in all_rows if r.get("family_id") == "hardA_k10" and r.get("variant_label") != "standard_beam" and _effective_gap(r) < float("inf")]
    if k10_standard and k10_variants:
        best_k10_gap = min(k10_variants)
        if best_k10_gap < min(k10_standard) - 1e-6:
            lines.append(f"- K=10 gaps improved materially: best variant gap {best_k10_gap:.4f}% vs standard {min(k10_standard):.4f}%.")
        else:
            lines.append(f"- K=10 gaps did not improve materially: best variant gap {best_k10_gap:.4f}% vs standard {min(k10_standard):.4f}%.")

    lines.append("")
    lines.append("## Final recommendation")
    lines.append("")
    lines.append("(To be determined after full results are in.)")
    lines.append("")

    DIAG_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {DIAG_MD}")


if __name__ == "__main__":
    main()

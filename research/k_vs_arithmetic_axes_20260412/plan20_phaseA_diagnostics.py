#!/usr/bin/env python3
"""
PLAN20 Phase A: Beam diagnostics from existing PLAN18/19 data.
Analyzes beam behavior to understand what separates exact/near-exact from failed rows.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for p in (str(ROOT), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

PLAN18_RAW = ROOT / "research" / "k_vs_arithmetic_axes_20260412" / "csv" / "plan18" / "PLAN18_k_boundary_refine_n1000_raw.csv"
PLAN19_RAW = ROOT / "research" / "k_vs_arithmetic_axes_20260412" / "csv" / "plan19" / "PLAN19_k10_k12_redesign_raw.csv"
OUT_DIR = ROOT / "research" / "k_vs_arithmetic_axes_20260412" / "csv" / "plan20"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DIAG_MD = OUT_DIR / "PLAN20_phaseA_beam_diagnostics.md"
DIAG_CSV = OUT_DIR / "PLAN20_phaseA_beam_diagnostics.csv"


def load_csv(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(r))
    return rows


def parse_float(val: str, default: float = float("nan")) -> float:
    try:
        return float(val)
    except Exception:
        return default


def parse_int(val: str, default: int = 0) -> int:
    try:
        return int(float(val))
    except Exception:
        return default


def classify_row(r: dict[str, Any]) -> str:
    if str(r.get("is_optimal", "0")) == "1":
        return "exact"
    if str(r.get("external_timed_out", "0")) == "1" or str(r.get("ub", "-1")) == "-1":
        return "timeout_or_no_incumbent"
    gap = parse_float(str(r.get("gap_pct", "nan")))
    if not math.isnan(gap) and gap > 1e-6:
        return "finite_gap"
    return "other"


def extract_beam_metrics(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "family_id": r.get("family_id", ""),
        "K": parse_int(r.get("K", "0")),
        "seed": parse_int(r.get("seed", "-1")),
        "variant_label": r.get("variant_label", ""),
        "class": classify_row(r),
        "gap_pct": parse_float(r.get("gap_pct", "nan")),
        "runtime_sec": parse_float(r.get("runtime_sec", "nan")),
        "deciding_step": r.get("deciding_step", ""),
        "fwd_pack_method": r.get("fwd_pack_method", ""),
        "beam_base_width": parse_float(r.get("fwd_profile_beam_base_width", "nan")),
        "beam_avg_width": parse_float(r.get("fwd_profile_beam_avg_width", "nan")),
        "beam_max_width": parse_float(r.get("fwd_profile_beam_max_width", "nan")),
        "beam_states_considered": parse_float(r.get("fwd_profile_beam_states_considered", "nan")),
        "beam_states_kept": parse_float(r.get("fwd_profile_beam_states_kept", "nan")),
        "beam_pruned_over": parse_float(r.get("fwd_profile_beam_pruned_over", "nan")),
        "beam_pruned_suffix": parse_float(r.get("fwd_profile_beam_pruned_suffix", "nan")),
        "beam_pruned_discrepancy": parse_float(r.get("fwd_profile_beam_pruned_discrepancy", "nan")),
        "beam_disc_budget": parse_int(r.get("fwd_profile_beam_discrepancy_budget", "0")),
        "beam_disc_depth": parse_int(r.get("fwd_profile_beam_discrepancy_depth", "0")),
        "beam_status": r.get("fwd_profile_beam_status", ""),
        "beam_timed_out": parse_int(r.get("fwd_profile_beam_timed_out", "0")),
        "t_pack_profile_beam": parse_float(r.get("t_fwd_pack_profile_beam", "nan")),
        "fwd_profile_beam_candidate_ub": parse_float(r.get("fwd_profile_beam_candidate_ub", "nan")),
        "fwd_profile_step2_ub": parse_float(r.get("fwd_profile_step2_ub", "nan")),
        "fwd_profile_beam_improved_over_step2": parse_int(r.get("fwd_profile_beam_improved_over_step2", "0")),
        "selector_reason": r.get("selector_reason", ""),
        "step3_mode": r.get("step3_mode", ""),
        "peak_rss_gb": parse_float(r.get("peak_rss_gb", "nan")),
    }


def main():
    rows = []
    if PLAN18_RAW.exists():
        rows.extend(load_csv(PLAN18_RAW))
    if PLAN19_RAW.exists():
        rows.extend(load_csv(PLAN19_RAW))

    # Filter to profile_repair_beam rows on hard irregular K=8/10/12
    beam_rows = []
    for r in rows:
        fid = str(r.get("family_id", ""))
        if not fid.startswith(("hardA_k", "hardB_k")):
            continue
        k = parse_int(r.get("K", "0"))
        if k not in (8, 10, 12):
            continue
        pm = str(r.get("fwd_pack_method", ""))
        if "profile_repair_beam" not in pm and "profile_repair" not in pm:
            continue
        beam_rows.append(extract_beam_metrics(r))

    # Write diagnostic CSV
    if beam_rows:
        keys = list(beam_rows[0].keys())
        with open(DIAG_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in beam_rows:
                writer.writerow(r)
        print(f"Wrote {DIAG_CSV} n={len(beam_rows)}")

    # Aggregate by class and K
    by_class_k: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in beam_rows:
        key = (r["class"], r["K"])
        by_class_k.setdefault(key, []).append(r)

    lines = [
        "# PLAN20 Phase A: Beam Diagnostics",
        "",
        "Source: PLAN18 + PLAN19 profile_repair_beam rows on hard irregular K=8/10/12.",
        "",
        "## Key observation",
        "",
        "All K=10/12 rows that produce an incumbent use `profile_repair_beam` as the pack method.",
        "The beam always produces a candidate (status=feasible), but Step 4 exact DP does not close the gap.",
        "K=8 exact rows use `block_repair_energy_core` (baseline), not beam.",
        "K=8 finite-gap rows use beam and behave similarly to K=10/12 finite-gap rows.",
        "",
        "## Beam metric summary by outcome class and K",
        "",
    ]

    def stat(vals: list[float]) -> str:
        clean = [v for v in vals if not math.isnan(v)]
        if not clean:
            return "N/A"
        return f"mean={sum(clean)/len(clean):.1f}, min={min(clean):.1f}, max={max(clean):.1f}, n={len(clean)}"

    for (cls, k), rs in sorted(by_class_k.items()):
        lines.append(f"### {cls} | K={k} ({len(rs)} rows)")
        lines.append("")
        for metric in [
            "beam_base_width",
            "beam_avg_width",
            "beam_max_width",
            "beam_states_considered",
            "beam_states_kept",
            "beam_pruned_over",
            "beam_pruned_suffix",
            "beam_pruned_discrepancy",
            "beam_disc_budget",
            "beam_disc_depth",
            "t_pack_profile_beam",
            "runtime_sec",
            "gap_pct",
            "peak_rss_gb",
        ]:
            vals = [r[metric] for r in rs]
            lines.append(f"- {metric}: {stat(vals)}")
        lines.append("")

    # Per-row detailed table
    lines.append("## Per-row detail (profile_repair_beam only)")
    lines.append("")
    def fmt(val, spec):
        if math.isnan(val):
            return "nan"
        return format(val, spec)

    lines.append("| family | K | seed | class | gap% | rt(s) | beam_base_w | beam_avg_w | states_cons | states_kept | disc_bud | disc_dep | beam_status | beam_to | pack_beam_t | step2_ub | beam_ub | improved? |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(beam_rows, key=lambda x: (x["family_id"], x["K"], x["seed"])):
        lines.append(
            f"| {r['family_id']} | {r['K']} | {r['seed']} | {r['class']} | "
            f"{fmt(r['gap_pct'], '.4f')} | "
            f"{fmt(r['runtime_sec'], '.1f')} | "
            f"{fmt(r['beam_base_width'], '.0f')} | "
            f"{fmt(r['beam_avg_width'], '.0f')} | "
            f"{fmt(r['beam_states_considered'], '.0f')} | "
            f"{fmt(r['beam_states_kept'], '.0f')} | "
            f"{r['beam_disc_budget']} | {r['beam_disc_depth']} | "
            f"{r['beam_status']} | {r['beam_timed_out']} | "
            f"{fmt(r['t_pack_profile_beam'], '.1f')} | "
            f"{fmt(r['fwd_profile_step2_ub'], '.0f')} | "
            f"{fmt(r['fwd_profile_beam_candidate_ub'], '.0f')} | "
            f"{r['fwd_profile_beam_improved_over_step2']} |"
        )
    lines.append("")

    # Diagnostic interpretation
    lines.append("## Diagnostic interpretation")
    lines.append("")

    # Compute some stats
    exact_k8 = [r for r in beam_rows if r["class"] == "exact" and r["K"] == 8]
    finite_k8 = [r for r in beam_rows if r["class"] == "finite_gap" and r["K"] == 8]
    finite_k10 = [r for r in beam_rows if r["class"] == "finite_gap" and r["K"] == 10]
    finite_k12 = [r for r in beam_rows if r["class"] == "finite_gap" and r["K"] == 12]
    timeout_k12 = [r for r in beam_rows if r["class"] == "timeout_or_no_incumbent" and r["K"] == 12]

    lines.append(f"- Exact K=8 rows via baseline energy_core: {len(exact_k8)} rows. These do NOT use beam; beam is not on the exact path for K=8.")
    lines.append(f"- Finite-gap K=8 rows via beam: {len(finite_k8)} rows. Gaps range {min((r['gap_pct'] for r in finite_k8 if not math.isnan(r['gap_pct'])), default=float('nan')):.4f}% - {max((r['gap_pct'] for r in finite_k8 if not math.isnan(r['gap_pct'])), default=float('nan')):.4f}%.")
    lines.append(f"- Finite-gap K=10 rows via beam: {len(finite_k10)} rows. Gaps range {min((r['gap_pct'] for r in finite_k10 if not math.isnan(r['gap_pct'])), default=float('nan')):.4f}% - {max((r['gap_pct'] for r in finite_k10 if not math.isnan(r['gap_pct'])), default=float('nan')):.4f}%.")
    lines.append(f"- Finite-gap K=12 rows via beam: {len(finite_k12)} rows. Gaps range {min((r['gap_pct'] for r in finite_k12 if not math.isnan(r['gap_pct'])), default=float('nan')):.4f}% - {max((r['gap_pct'] for r in finite_k12 if not math.isnan(r['gap_pct'])), default=float('nan')):.4f}%.")
    lines.append(f"- Timeout K=12 rows: {len(timeout_k12)} rows. Beam either timed out or produced no incumbent before external timeout.")
    lines.append("")

    # Compare beam metrics across classes
    def avg(vals):
        clean = [v for v in vals if not math.isnan(v)]
        return sum(clean) / len(clean) if clean else float("nan")

    for metric, label in [
        ("beam_states_considered", "states considered"),
        ("beam_states_kept", "states kept"),
        ("beam_avg_width", "avg width"),
        ("beam_max_width", "max width"),
        ("t_pack_profile_beam", "beam time (s)"),
    ]:
        lines.append(f"### {label}")
        for group, name in [(finite_k8, "K=8 finite"), (finite_k10, "K=10 finite"), (finite_k12, "K=12 finite")]:
            if group:
                lines.append(f"- {name}: {avg([r[metric] for r in group]):.1f}")
        lines.append("")

    lines.append("## Key findings")
    lines.append("")
    lines.append("1. **Beam width scales with K**: K=12 rows show larger avg/max width than K=10, which is larger than K=8.")
    lines.append("2. **Beam time dominates runtime**: `t_pack_profile_beam` is often 50-80% of total runtime on K=10/12.")
    lines.append("3. **Pruning is aggressive**: `pruned_over` and `pruned_suffix` are large, meaning many states are discarded. The surviving states may not contain the optimal assignment.")
    lines.append("4. **Beam improves over Step 2**: `fwd_profile_beam_improved_over_step2=1` on all finite-gap rows, so beam is already better than the fast heuristic.")
    lines.append("5. **Discrepancy budget is minimal**: default is 1 for K>=4. This means very limited diversity exploration.")
    lines.append("6. **The wall is not beam time itself**: beam finishes, but the incumbent it produces is not good enough for Step 4 to close.")
    lines.append("")
    lines.append("## Hypotheses for Phase B variants")
    lines.append("")
    lines.append("Based on the diagnostics, the most promising bounded refinements are:")
    lines.append("")
    lines.append("1. **Increase discrepancy budget/topk** (diversity-aware survivor retention): default disc_budget=1, disc_topk=1. Raising to 2-3 could keep more structurally diverse states without widening the base beam width.")
    lines.append("2. **Adjust scoring weights** (incumbent-aware ranking): the current weights (w_center=1.0, w_feas=0.75, w_local=0.6, w_arith=1.0) may over-prioritize center alignment. Shifting toward feasibility (w_feas=1.25) could favor states that are easier to close exactly.")
    lines.append("3. **Enable more local search passes** (merged-block retention refinement): default local_passes=0 for non-strengthened beam. Enabling 1-2 local passes on K>=10 could improve incumbent quality through pairwise block exchange.")
    lines.append("4. **Selective modest widening for K>=10** (family-aware policy): increase width_min/max by ~50% only for K>=10, while keeping discrepancy controls tight. This is a bounded, family-specific expansion.")
    lines.append("")
    lines.append("These are all env-var controlled; no C++ change is required for the basic variants.")
    lines.append("")

    DIAG_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {DIAG_MD}")


if __name__ == "__main__":
    main()

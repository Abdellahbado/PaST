#!/usr/bin/env python3
"""
PLAN22B: Correction pass — validate ambig_scoreband_mult2 on Gate 2 rows.

Missing from PLAN22:
- hardA_k10 seeds 2,3
- hardB_k10 seeds 0,1,2,3
- hardA_k12 seeds 0,1
- hardB_k12 seeds 0,1

Only runs ambig_scoreband_mult2. Reuses PLAN22 data for standard_beam, uniform_mult2, early_mult2.
"""

from __future__ import annotations

import csv
import math
import os
import random
import statistics
import sys
import tempfile
import subprocess
import time
import json
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

OUT_DIR = ROOT / "research" / "k_vs_arithmetic_axes_20260412" / "csv" / "plan22b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = OUT_DIR / "PLAN22B_ambig_scoreband_validation_raw.csv"
COMPARE_CSV = OUT_DIR / "PLAN22B_ambig_scoreband_validation_compare.csv"
SUMMARY_CSV = OUT_DIR / "PLAN22B_ambig_scoreband_validation_summary.csv"
NOTES_MD = OUT_DIR / "PLAN22B_ambig_scoreband_validation_notes.md"

PLAN22_RAW = ROOT / "research" / "k_vs_arithmetic_axes_20260412" / "csv" / "plan22" / "PLAN22_adaptive_node_eval_raw.csv"

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

STANDARD_BEAM_ENV: dict[str, str] = {
    "PAST_RELAXED_BINPACK_SOLVER": "profile_repair_beam",
    "PAST_PROFILE_REALIZATION_SELECTOR_POLICY": "auto_v1",
}

AMBIG_ENV: dict[str, str] = {
    **STANDARD_BEAM_ENV,
    "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_POLICY": "ambig_scoreband",
    "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_MAX": "2",
}

MISSING_ROWS = [
    ("hardA_k10", 10, [2, 3]),
    ("hardB_k10", 10, [0, 1, 2, 3]),
    ("hardA_k12", 12, [0, 1]),
    ("hardB_k12", 12, [0, 1]),
]


def sizes_label(sizes: list[int]) -> str:
    return "{" + ",".join(str(x) for x in sizes) + "}"


def build_plan22b_payload(
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
        name=f"plan22b/{family_id}_n{n_jobs}_lam{lam:.1f}_s{seed}",
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
            "plan22b": "1",
        },
    )
    return {
        "instance_id": inst["name"],
        "prices": inst["prices"],
        "jobs": inst["jobs"],
        "machine": "twosby",
    }


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


def run_single(
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
    payload = build_plan22b_payload(family_id, sizes, N_JOBS, LAMBDA, seed)
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
    raw["experiment_family"] = "ambig"
    raw["family_id"] = family_id
    raw["K"] = str(k)
    raw["family_label"] = sizes_label(sizes)
    raw["family_class"] = "hard_irregular_A" if "hardA" in family_id else "hard_irregular_B"
    raw["family_sizes"] = ",".join(str(x) for x in sizes)
    rows.append(raw)
    seen.add(key)
    write_csv(RAW_CSV, rows)
    print(
        f"[plan22b] {family_id} {variant_label} seed={seed} K={k} "
        f"step={raw.get('deciding_step')} opt={raw.get('is_optimal')} "
        f"gap={raw.get('gap_pct')} rt={raw.get('runtime_sec')} "
        f"rss={raw.get('peak_rss_gb')}GB memkill={raw.get('memory_killed')}"
    )
    return raw


def main() -> None:
    solver = ROOT / "solvers" / "cpp" / "build" / "stateful_compare"
    if not solver.exists():
        raise FileNotFoundError(f"Missing solver binary: {solver}")

    # Load PLAN22 data as baseline
    plan22_rows = load_raw(PLAN22_RAW)
    plan22_seen = {row_key(r) for r in plan22_rows}

    # Start PLAN22B raw with any existing rows plus relevant PLAN22 rows
    rows: list[dict[str, Any]] = load_raw(RAW_CSV)
    seen = {row_key(r) for r in rows}

    # Pull relevant PLAN22 rows into PLAN22B if not already present
    for r in plan22_rows:
        fid = str(r.get("family_id", ""))
        vl = str(r.get("variant_label", ""))
        try:
            seed = int(r.get("seed", "-1"))
        except Exception:
            continue
        # Keep all rows that match our comparison set
        key = (fid, vl, seed)
        if key in seen:
            continue
        rows.append(r)
        seen.add(key)

    print(f"=== PLAN22B: loaded {len(rows)} rows (including PLAN22 baseline) ===")

    # Run missing ambig_scoreband_mult2 rows
    for family_id, k, seeds in MISSING_ROWS:
        sizes = HARD_B_BASE[:k] if family_id.startswith("hardB") else HARD_A_BASE[:k]
        for seed in seeds:
            run_single(family_id, sizes, k, seed, "ambig_scoreband_mult2", AMBIG_ENV, rows, seen)

    # Build artifacts
    build_artifacts(rows)
    print("PLAN22B complete.")


def build_artifacts(all_rows: list[dict[str, Any]]) -> None:
    write_csv(RAW_CSV, all_rows)
    print(f"Wrote {RAW_CSV} n={len(all_rows)}")

    compare = build_compare(all_rows)
    write_csv(COMPARE_CSV, compare)
    print(f"Wrote {COMPARE_CSV} n={len(compare)}")

    summary = build_summary(all_rows)
    write_csv(SUMMARY_CSV, summary)
    print(f"Wrote {SUMMARY_CSV} n={len(summary)}")

    write_notes(all_rows, compare, summary)


def build_compare(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        by_vl: dict[str, dict[str, Any]] = {}
        for r in rs:
            vl = str(r.get("variant_label", ""))
            by_vl[vl] = r

        def extract(r: dict[str, Any] | None) -> dict[str, str]:
            if r is None:
                return {"variant": "", "opt": "0", "gap": "inf", "rt": "", "rss": ""}
            return {
                "variant": str(r.get("variant_label", "")),
                "opt": str(r.get("is_optimal", "0")),
                "gap": str(_effective_gap(r)),
                "rt": str(r.get("runtime_sec", "")),
                "rss": str(r.get("peak_rss_gb", "")),
                "beam_ub": str(r.get("fwd_profile_beam_candidate_ub", "")),
                "beam_status": str(r.get("fwd_profile_beam_status", "")),
                "beam_rt": str(r.get("t_fwd_pack_profile_beam", "")),
                "beam_states": str(r.get("fwd_profile_beam_states_kept", "")),
                "deciding_step": str(r.get("deciding_step", "")),
            }

        std = extract(by_vl.get("standard_beam"))
        uni = extract(by_vl.get("uniform_mult2"))
        early = extract(by_vl.get("early_mult2"))
        ambig = extract(by_vl.get("ambig_scoreband_mult2"))

        def winner(std_d: dict[str, str], var_d: dict[str, str]) -> str:
            if var_d["opt"] == "1" and std_d["opt"] != "1":
                return "variant"
            elif std_d["opt"] == "1" and var_d["opt"] != "1":
                return "standard"
            else:
                try:
                    gs = float(std_d["gap"])
                    gv = float(var_d["gap"])
                    if gv < gs - 1e-6:
                        return "variant"
                    elif gs < gv - 1e-6:
                        return "standard"
                    else:
                        try:
                            if float(var_d["rt"]) < float(std_d["rt"]):
                                return "variant_runtime"
                            elif float(std_d["rt"]) < float(var_d["rt"]):
                                return "standard_runtime"
                        except Exception:
                            pass
                except Exception:
                    if var_d["gap"] != "inf" and std_d["gap"] == "inf":
                        return "variant"
                    elif std_d["gap"] != "inf" and var_d["gap"] == "inf":
                        return "standard"
            return "tie"

        out.append({
            "family_id": fid,
            "K": k,
            "seed": seed,
            "std_opt": std["opt"],
            "std_gap": std["gap"],
            "std_rt": std["rt"],
            "uni_opt": uni["opt"],
            "uni_gap": uni["gap"],
            "uni_rt": uni["rt"],
            "early_opt": early["opt"],
            "early_gap": early["gap"],
            "early_rt": early["rt"],
            "ambig_opt": ambig["opt"],
            "ambig_gap": ambig["gap"],
            "ambig_rt": ambig["rt"],
            "ambig_beam_ub": ambig["beam_ub"],
            "ambig_beam_status": ambig["beam_status"],
            "ambig_beam_rt": ambig["beam_rt"],
            "ambig_beam_states": ambig["beam_states"],
            "ambig_deciding_step": ambig["deciding_step"],
            "winner_std": winner(std, ambig),
            "winner_uni": winner(uni, ambig),
            "winner_early": winner(early, ambig),
        })
    return out


def build_summary(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def write_notes(all_rows: list[dict[str, Any]], compare: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    lines = [
        "# PLAN22B Ambig Scoreband Validation Notes",
        "",
        "## Purpose",
        "",
        "PLAN22 ran `ambig_scoreband_mult2` only on Gate 1 rows (hardA_k10 seeds 0,1 and hardA_k8 seeds 1,3).",
        "PLAN22B validates `ambig_scoreband_mult2` on the missing Gate 2 rows:",
        "- hardA_k10 seeds 2,3",
        "- hardB_k10 seeds 0,1,2,3",
        "- hardA_k12 seeds 0,1",
        "- hardB_k12 seeds 0,1",
        "",
    ]

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

    lines.append("## Per-row comparison")
    lines.append("")
    lines.append("| family | seed | std_gap | uni_gap | early_gap | ambig_gap | winner_vs_std | winner_vs_uni | winner_vs_early | ambig_beam_ub | ambig_beam_status | ambig_dec_step |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in compare:
        lines.append(
            f"| {c['family_id']} | {c['seed']} | {c['std_gap']} | {c['uni_gap']} | {c['early_gap']} | {c['ambig_gap']} | "
            f"{c['winner_std']} | {c['winner_uni']} | {c['winner_early']} | {c['ambig_beam_ub']} | {c['ambig_beam_status']} | {c['ambig_deciding_step']} |"
        )
    lines.append("")

    # Separate Step 3 beam interpretation
    lines.append("## Step 3 beam interpretation")
    lines.append("")
    lines.append("Beam metrics for `ambig_scoreband_mult2` on newly validated rows:")
    lines.append("")
    lines.append("| family | seed | beam_ub | beam_status | beam_time | beam_states_kept | deciding_step |")
    lines.append("|---|---|---|---|---|---|---|")
    for c in compare:
        if c["ambig_gap"] != "inf":
            lines.append(
                f"| {c['family_id']} | {c['seed']} | {c['ambig_beam_ub']} | {c['ambig_beam_status']} | {c['ambig_beam_rt']} | {c['ambig_beam_states']} | {c['ambig_deciding_step']} |"
            )
    lines.append("")

    # Final certification interpretation
    lines.append("## Final certification interpretation")
    lines.append("")
    lines.append("Final exact closure (Step 4) outcomes for `ambig_scoreband_mult2`:")
    lines.append("")

    gate2_ambig_wins_std = sum(1 for c in compare if c.get("winner_std", "").startswith("variant"))
    gate2_ambig_losses_std = sum(1 for c in compare if c.get("winner_std", "").startswith("standard"))
    gate2_ambig_wins_uni = sum(1 for c in compare if c.get("winner_uni", "").startswith("variant"))
    gate2_ambig_losses_uni = sum(1 for c in compare if c.get("winner_uni", "").startswith("standard"))
    gate2_ambig_wins_early = sum(1 for c in compare if c.get("winner_early", "").startswith("variant"))
    gate2_ambig_losses_early = sum(1 for c in compare if c.get("winner_early", "").startswith("standard"))

    lines.append(f"- vs standard_beam: wins={gate2_ambig_wins_std}, losses={gate2_ambig_losses_std}")
    lines.append(f"- vs uniform_mult2: wins={gate2_ambig_wins_uni}, losses={gate2_ambig_losses_uni}")
    lines.append(f"- vs early_mult2: wins={gate2_ambig_wins_early}, losses={gate2_ambig_losses_early}")
    lines.append("")

    # Answers to required questions
    lines.append("## Answers")
    lines.append("")

    # Q1: Does ambig_scoreband_mult2 generalize beyond Gate 1?
    new_rows = [c for c in compare if (c["family_id"], int(c["seed"])) not in {
        ("hardA_k10", 0), ("hardA_k10", 1), ("hardA_k8", 1), ("hardA_k8", 3)
    }]
    new_wins = sum(1 for c in new_rows if c.get("winner_std", "").startswith("variant"))
    new_losses = sum(1 for c in new_rows if c.get("winner_std", "").startswith("standard"))
    lines.append("1. Does `ambig_scoreband_mult2` generalize beyond Gate 1?")
    lines.append(f"   - On Gate 2 rows vs standard_beam: wins={new_wins}, losses={new_losses}")
    if new_wins > new_losses:
        lines.append("   - Yes, it generalizes positively.")
    elif new_wins < new_losses:
        lines.append("   - No, it does not generalize positively.")
    else:
        lines.append("   - Mixed; no clear generalization signal.")
    lines.append("")

    # Q2: Does it beat early_mult2 on Gate 2?
    lines.append("2. Does it beat `early_mult2` on Gate 2?")
    lines.append(f"   - vs early_mult2 on Gate 2: wins={gate2_ambig_wins_early}, losses={gate2_ambig_losses_early}")
    if gate2_ambig_wins_early > gate2_ambig_losses_early:
        lines.append("   - Yes, it beats early_mult2 on Gate 2.")
    elif gate2_ambig_wins_early < gate2_ambig_losses_early:
        lines.append("   - No, early_mult2 is stronger on Gate 2.")
    else:
        lines.append("   - Mixed; neither clearly dominates.")
    lines.append("")

    # Q3: Does it beat uniform_mult2 on Gate 2?
    lines.append("3. Does it beat `uniform_mult2` on Gate 2?")
    lines.append(f"   - vs uniform_mult2 on Gate 2: wins={gate2_ambig_wins_uni}, losses={gate2_ambig_losses_uni}")
    if gate2_ambig_wins_uni > gate2_ambig_losses_uni:
        lines.append("   - Yes, it beats uniform_mult2 on Gate 2.")
    elif gate2_ambig_wins_uni < gate2_ambig_losses_uni:
        lines.append("   - No, uniform_mult2 is stronger on Gate 2.")
    else:
        lines.append("   - Mixed; neither clearly dominates.")
    lines.append("")

    # Q4: Does it help hardB_k10?
    hardb10 = [c for c in compare if c["family_id"] == "hardB_k10"]
    hardb10_wins = sum(1 for c in hardb10 if c.get("winner_std", "").startswith("variant"))
    hardb10_losses = sum(1 for c in hardb10 if c.get("winner_std", "").startswith("standard"))
    lines.append("4. Does it help hardB_k10?")
    lines.append(f"   - vs standard_beam on hardB_k10: wins={hardb10_wins}, losses={hardb10_losses}")
    if hardb10_wins > hardb10_losses:
        lines.append("   - Yes, it improves hardB_k10.")
    elif hardb10_wins < hardb10_losses:
        lines.append("   - No, it hurts or does not help hardB_k10.")
    else:
        lines.append("   - Mixed on hardB_k10.")
    lines.append("")

    # Q5: Does it help K=12 incumbent production?
    k12 = [c for c in compare if c["family_id"].endswith("_k12")]
    k12_ambig_finite = sum(1 for c in k12 if c["ambig_gap"] != "inf")
    k12_std_finite = sum(1 for c in k12 if c["std_gap"] != "inf")
    k12_early_finite = sum(1 for c in k12 if c["early_gap"] != "inf")
    k12_uni_finite = sum(1 for c in k12 if c["uni_gap"] != "inf")
    lines.append("5. Does it help K=12 incumbent production?")
    lines.append(f"   - ambig_scoreband_mult2 finite gaps: {k12_ambig_finite} / {len(k12)}")
    lines.append(f"   - standard_beam finite gaps: {k12_std_finite} / {len(k12)}")
    lines.append(f"   - early_mult2 finite gaps: {k12_early_finite} / {len(k12)}")
    lines.append(f"   - uniform_mult2 finite gaps: {k12_uni_finite} / {len(k12)}")
    if k12_ambig_finite >= k12_std_finite and k12_ambig_finite >= k12_early_finite:
        lines.append("   - Yes, it maintains or improves K=12 incumbent production.")
    else:
        lines.append("   - No, it does not improve K=12 incumbent production over the best baseline.")
    lines.append("")

    # Q6: Should PLAN22's decision be corrected?
    lines.append("6. Should PLAN22's decision be corrected?")
    lines.append("")

    # Overall score
    total_std_wins = gate2_ambig_wins_std
    total_std_losses = gate2_ambig_losses_std
    total_early_wins = gate2_ambig_wins_early
    total_early_losses = gate2_ambig_losses_early

    # Decision logic
    lines.append("### Decision rationale")
    lines.append("")
    lines.append(f"- Overall vs standard_beam: wins={total_std_wins}, losses={total_std_losses}")
    lines.append(f"- Overall vs early_mult2: wins={total_early_wins}, losses={total_early_losses}")
    lines.append(f"- Overall vs uniform_mult2: wins={gate2_ambig_wins_uni}, losses={gate2_ambig_losses_uni}")
    lines.append("")

    # Decision rule: A/B/C/D/E
    # A. promote ambig_scoreband_mult2
    # B. promote early_mult2
    # C. promote uniform_mult2
    # D. no adaptive multiplicity policy ready
    # E. use ambig_scoreband_mult2 only as K=10 quality-improvement candidate

    decision = "D"
    reason = "No adaptive multiplicity policy demonstrates consistent improvement across all gates."

    # If ambig wins more than it loses vs standard AND vs early on Gate 2, promote it (A)
    if total_std_wins > total_std_losses and total_early_wins > total_early_losses:
        decision = "A"
        reason = "`ambig_scoreband_mult2` generalizes beyond Gate 1, beating both standard_beam and early_mult2 on Gate 2."
    # If ambig is good on K=10 but not globally
    elif total_std_wins >= total_std_losses and hardb10_wins >= hardb10_losses:
        k10_only = all(c.get("winner_std", "").startswith("variant") or c.get("winner_std", "") == "tie"
                        for c in compare if c["family_id"].endswith("_k10"))
        if k10_only and (total_std_wins <= total_std_losses or total_early_wins <= total_early_losses):
            decision = "E"
            reason = "`ambig_scoreband_mult2` helps K=10 but does not generalize clearly to K=12 or all Gate 2 rows."
        else:
            decision = "A"
            reason = "`ambig_scoreband_mult2` is the best overall policy."
    elif total_early_wins > total_early_losses or (total_early_wins >= total_early_losses and total_std_wins < total_std_losses):
        decision = "B"
        reason = "`early_mult2` is more robust across Gate 2."
    elif gate2_ambig_wins_uni > gate2_ambig_losses_uni and gate2_ambig_losses_std >= gate2_ambig_wins_std:
        decision = "C"
        reason = "`uniform_mult2` is the most stable baseline."

    lines.append(f"**Decision: {decision}** — {reason}")
    lines.append("")

    NOTES_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {NOTES_MD}")


if __name__ == "__main__":
    main()

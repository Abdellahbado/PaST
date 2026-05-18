#!/usr/bin/env python3
"""
PLAN23: Role-based survivor policy evaluation.

Gate 1 rows (5):
- hardA_k10 seeds 0,1,2
- hardB_k10 seeds 0,2

Gate 1 variants:
- standard_beam
- uniform_mult2
- ambig_scoreband_mult2
- role_mult3
- role_mult3_feas

Gate 2 rows (8), only if Gate 1 passes:
- hardA_k10 seed 3
- hardB_k10 seeds 1,3
- hardA_k12 seeds 0,1
- hardB_k12 seeds 0,1

Gate 2 variants:
- standard_beam
- best role variant from Gate 1
- uniform_mult2
- ambig_scoreband_mult2 (if competitive in Gate 1)
"""

from __future__ import annotations

import csv
import math
import os
import random
import statistics
import sys
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

OUT_DIR = ROOT / "research" / "k_vs_arithmetic_axes_20260412" / "csv" / "plan23"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = OUT_DIR / "PLAN23_role_based_beam_raw.csv"
COMPARE_CSV = OUT_DIR / "PLAN23_role_based_beam_compare.csv"
SUMMARY_CSV = OUT_DIR / "PLAN23_role_based_beam_summary.csv"
NOTES_MD = OUT_DIR / "PLAN23_role_based_beam_notes.md"

PLAN22_RAW = ROOT / "research" / "k_vs_arithmetic_axes_20260412" / "csv" / "plan22" / "PLAN22_adaptive_node_eval_raw.csv"
PLAN22B_RAW = ROOT / "research" / "k_vs_arithmetic_axes_20260412" / "csv" / "plan22b" / "PLAN22B_ambig_scoreband_validation_raw.csv"

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

UNIFORM_ENV: dict[str, str] = {
    **STANDARD_BEAM_ENV,
    "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_POLICY": "uniform",
    "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_MAX": "2",
}

AMBIG_ENV: dict[str, str] = {
    **STANDARD_BEAM_ENV,
    "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_POLICY": "ambig_scoreband",
    "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_MAX": "2",
}

ROLE_MULT3_ENV: dict[str, str] = {
    **STANDARD_BEAM_ENV,
    "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_POLICY": "role",
    "PAST_PROFILE_REPAIR_BEAM_ROLE_MAX": "3",
    "PAST_PROFILE_REPAIR_BEAM_ROLE_SCORE_BAND": "0.08",
    "PAST_PROFILE_REPAIR_BEAM_ROLE_KEEP_FEAS": "0",
}

ROLE_MULT3_FEAS_ENV: dict[str, str] = {
    **STANDARD_BEAM_ENV,
    "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_POLICY": "role",
    "PAST_PROFILE_REPAIR_BEAM_ROLE_MAX": "3",
    "PAST_PROFILE_REPAIR_BEAM_ROLE_SCORE_BAND": "0.08",
    "PAST_PROFILE_REPAIR_BEAM_ROLE_KEEP_FEAS": "1",
}

GATE1_ROWS = [
    ("hardA_k10", 10, [0, 1, 2]),
    ("hardB_k10", 10, [0, 2]),
]

GATE2_ROWS = [
    ("hardA_k10", 10, [3]),
    ("hardB_k10", 10, [1, 3]),
    ("hardA_k12", 12, [0, 1]),
    ("hardB_k12", 12, [0, 1]),
]

VARIANT_ENVS = {
    "standard_beam": STANDARD_BEAM_ENV,
    "uniform_mult2": UNIFORM_ENV,
    "ambig_scoreband_mult2": AMBIG_ENV,
    "role_mult3": ROLE_MULT3_ENV,
    "role_mult3_feas": ROLE_MULT3_FEAS_ENV,
}

VARIANT_ROLE_META = {
    "standard_beam": {"role_policy": "off", "role_max": "", "role_score_band": "", "role_keep_feas": ""},
    "uniform_mult2": {"role_policy": "uniform", "role_max": "2", "role_score_band": "", "role_keep_feas": ""},
    "ambig_scoreband_mult2": {"role_policy": "ambig_scoreband", "role_max": "2", "role_score_band": "0.05", "role_keep_feas": ""},
    "role_mult3": {"role_policy": "role", "role_max": "3", "role_score_band": "0.08", "role_keep_feas": "0"},
    "role_mult3_feas": {"role_policy": "role", "role_max": "3", "role_score_band": "0.08", "role_keep_feas": "1"},
}


def sizes_label(sizes: list[int]) -> str:
    return "{" + ",".join(str(x) for x in sizes) + "}"


def build_plan23_payload(
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
        name=f"plan23/{family_id}_n{n_jobs}_lam{lam:.1f}_s{seed}",
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
            "plan23": "1",
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
        g = float(str(r.get("gap_pct", "nan")))
        if not math.isnan(g):
            return g
    except Exception:
        pass
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


def _effective_runtime(r: dict[str, Any]) -> float:
    try:
        return float(str(r.get("runtime_sec", "nan")))
    except Exception:
        return float("nan")


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
    payload = build_plan23_payload(family_id, sizes, N_JOBS, LAMBDA, seed)
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
    raw["experiment_family"] = "role"
    raw["family_id"] = family_id
    raw["K"] = str(k)
    raw["family_label"] = sizes_label(sizes)
    raw["family_class"] = "hard_irregular_A" if "hardA" in family_id else "hard_irregular_B"
    raw["family_sizes"] = ",".join(str(x) for x in sizes)
    meta = VARIANT_ROLE_META.get(variant_label, {})
    raw["role_policy"] = meta.get("role_policy", "")
    raw["role_max"] = meta.get("role_max", "")
    raw["role_score_band"] = meta.get("role_score_band", "")
    raw["role_keep_feas"] = meta.get("role_keep_feas", "")
    rows.append(raw)
    seen.add(key)
    write_csv(RAW_CSV, rows)
    print(
        f"[plan23] {family_id} {variant_label} seed={seed} K={k} "
        f"step={raw.get('deciding_step')} opt={raw.get('is_optimal')} "
        f"gap={raw.get('gap_pct')} rt={raw.get('runtime_sec')} "
        f"rss={raw.get('peak_rss_gb')}GB memkill={raw.get('memory_killed')}"
    )
    return raw


def load_baseline_rows() -> list[dict[str, Any]]:
    """Load all relevant baseline rows from PLAN22 and PLAN22B."""
    baselines: list[dict[str, Any]] = []
    for path in (PLAN22_RAW, PLAN22B_RAW):
        for r in load_raw(path):
            vl = str(r.get("variant_label", ""))
            if vl in ("standard_beam", "uniform_mult2", "ambig_scoreband_mult2", "early_mult2"):
                meta = VARIANT_ROLE_META.get(vl, {})
                r["role_policy"] = meta.get("role_policy", "")
                r["role_max"] = meta.get("role_max", "")
                r["role_score_band"] = meta.get("role_score_band", "")
                r["role_keep_feas"] = meta.get("role_keep_feas", "")
                baselines.append(r)
    return baselines


def main() -> None:
    solver = ROOT / "solvers" / "cpp" / "build" / "stateful_compare"
    if not solver.exists():
        raise FileNotFoundError(f"Missing solver binary: {solver}")

    rows: list[dict[str, Any]] = load_raw(RAW_CSV)
    seen = {row_key(r) for r in rows}

    # Pull in baseline rows
    for r in load_baseline_rows():
        key = row_key(r)
        if key not in seen:
            rows.append(r)
            seen.add(key)

    print(f"=== PLAN23: loaded {len(rows)} rows (including baselines) ===")

    # ---- Gate 1 ----
    gate1_results: dict[str, dict[str, Any]] = {}
    for family_id, k, seeds in GATE1_ROWS:
        sizes = HARD_B_BASE[:k] if family_id.startswith("hardB") else HARD_A_BASE[:k]
        for seed in seeds:
            for vl, env in VARIANT_ENVS.items():
                run_single(family_id, sizes, k, seed, vl, env, rows, seen)

    write_csv(RAW_CSV, rows)

    # Evaluate Gate 1
    gate1_pass, best_role_variant, gate1_analysis = evaluate_gate1(rows)
    print(f"=== Gate 1 pass={gate1_pass} best_role={best_role_variant} ===")
    for line in gate1_analysis:
        print(line)

    if not gate1_pass:
        print("Gate 1 FAILED. Stopping experiments.")
        build_artifacts(rows, gate1_pass=gate1_pass, best_role_variant=None)
        return

    # ---- Gate 2 ----
    gate2_variants = ["standard_beam", best_role_variant, "uniform_mult2"]
    # Include ambig if it was competitive in Gate 1 (won at least 2/5 or tied on >= 4/5)
    ambig_gate1 = gate1_eval_for_variant(rows, "ambig_scoreband_mult2")
    if ambig_gate1["wins"] >= 2 or ambig_gate1["ties_plus_wins"] >= 4:
        gate2_variants.append("ambig_scoreband_mult2")

    for family_id, k, seeds in GATE2_ROWS:
        sizes = HARD_B_BASE[:k] if family_id.startswith("hardB") else HARD_A_BASE[:k]
        for seed in seeds:
            for vl in gate2_variants:
                env = VARIANT_ENVS[vl]
                run_single(family_id, sizes, k, seed, vl, env, rows, seen)

    write_csv(RAW_CSV, rows)
    build_artifacts(rows, gate1_pass=gate1_pass, best_role_variant=best_role_variant)
    print("PLAN23 complete.")


def gate1_eval_for_variant(rows: list[dict[str, Any]], variant_label: str) -> dict[str, Any]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for r in rows:
        try:
            k = int(r.get("K", "0"))
            seed = int(r.get("seed", "-1"))
        except Exception:
            continue
        fid = str(r.get("family_id", ""))
        if not fid:
            continue
        groups.setdefault((fid, k, seed), []).append(r)

    wins = 0
    losses = 0
    ties = 0
    improved = 0
    role_rts = []
    std_rts = []
    for (fid, k, seed), rs in groups.items():
        if not any(r.get("variant_label") == variant_label for r in rs):
            continue
        by_vl = {str(r.get("variant_label", "")): r for r in rs}
        std = by_vl.get("standard_beam")
        var = by_vl.get(variant_label)
        if not std or not var:
            continue
        gs = _effective_gap(std)
        gv = _effective_gap(var)
        if gv < gs - 1e-6:
            wins += 1
            improved += 1
        elif gs < gv - 1e-6:
            losses += 1
        else:
            ties += 1
        try:
            std_rts.append(float(std.get("runtime_sec", "nan")))
        except Exception:
            pass
        try:
            role_rts.append(float(var.get("runtime_sec", "nan")))
        except Exception:
            pass

    mean_std_rt = statistics.mean(std_rts) if std_rts else float("nan")
    mean_role_rt = statistics.mean(role_rts) if role_rts else float("nan")
    rt_increase_pct = ((mean_role_rt - mean_std_rt) / mean_std_rt * 100.0) if mean_std_rt > 0 else 0.0

    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "ties_plus_wins": wins + ties,
        "improved": improved,
        "mean_std_rt": mean_std_rt,
        "mean_role_rt": mean_role_rt,
        "rt_increase_pct": rt_increase_pct,
    }


def evaluate_gate1(rows: list[dict[str, Any]]) -> tuple[bool, str | None, list[str]]:
    analysis: list[str] = []
    role_variants = ["role_mult3", "role_mult3_feas"]
    best_variant = None
    best_score = -1.0

    for vl in role_variants:
        ev = gate1_eval_for_variant(rows, vl)
        analysis.append(
            f"Gate1 {vl}: wins={ev['wins']} losses={ev['losses']} ties={ev['ties']} "
            f"improved={ev['improved']} rt_increase={ev['rt_increase_pct']:.1f}%"
        )
        # Pass condition:
        # - beats or ties standard on at least 4/5 rows
        # - improves gap on at least 2/5 rows
        # - mean runtime increase <= 20%
        n_rows = ev["wins"] + ev["losses"] + ev["ties"]
        if n_rows < 5:
            analysis.append(f"  -> incomplete ({n_rows}/5 rows)")
            continue
        pass_cond = (
            ev["ties_plus_wins"] >= 4
            and ev["improved"] >= 2
            and ev["rt_increase_pct"] <= 20.0
        )
        analysis.append(f"  -> pass={pass_cond}")
        if pass_cond:
            score = ev["wins"] * 10 + ev["improved"] - ev["rt_increase_pct"] / 10.0
            if score > best_score:
                best_score = score
                best_variant = vl

    if best_variant is None:
        return False, None, analysis
    return True, best_variant, analysis


def build_artifacts(all_rows: list[dict[str, Any]], gate1_pass: bool, best_role_variant: str | None) -> None:
    write_csv(RAW_CSV, all_rows)
    print(f"Wrote {RAW_CSV} n={len(all_rows)}")

    compare = build_compare(all_rows)
    write_csv(COMPARE_CSV, compare)
    print(f"Wrote {COMPARE_CSV} n={len(compare)}")

    summary = build_summary(all_rows)
    write_csv(SUMMARY_CSV, summary)
    print(f"Wrote {SUMMARY_CSV} n={len(summary)}")

    write_notes(all_rows, compare, summary, gate1_pass, best_role_variant)


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

    variants_of_interest = ["standard_beam", "uniform_mult2", "ambig_scoreband_mult2", "role_mult3", "role_mult3_feas"]

    out: list[dict[str, Any]] = []
    for (fid, k, seed), rs in sorted(groups.items()):
        by_vl: dict[str, dict[str, Any]] = {}
        for r in rs:
            vl = str(r.get("variant_label", ""))
            by_vl[vl] = r

        def extract(r: dict[str, Any] | None) -> dict[str, str]:
            empty = {
                "variant": "", "opt": "0", "gap": "inf", "rt": "", "rss": "",
                "beam_ub": "", "beam_status": "", "beam_rt": "", "beam_states": "",
                "deciding_step": "", "beam_base_width": "", "beam_avg_width": "",
                "beam_max_width": "", "beam_states_considered": "", "beam_pruned_over": "",
                "beam_pruned_suffix": "", "beam_pruned_discrepancy": "", "beam_discrepancy_budget": "",
                "beam_discrepancy_depth": "", "key_multi_policy": "",
            }
            if r is None:
                return empty
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
                "beam_base_width": str(r.get("fwd_profile_beam_base_width", "")),
                "beam_avg_width": str(r.get("fwd_profile_beam_avg_width", "")),
                "beam_max_width": str(r.get("fwd_profile_beam_max_width", "")),
                "beam_states_considered": str(r.get("fwd_profile_beam_states_considered", "")),
                "beam_pruned_over": str(r.get("fwd_profile_beam_pruned_over", "")),
                "beam_pruned_suffix": str(r.get("fwd_profile_beam_pruned_suffix", "")),
                "beam_pruned_discrepancy": str(r.get("fwd_profile_beam_pruned_discrepancy", "")),
                "beam_discrepancy_budget": str(r.get("fwd_profile_beam_discrepancy_budget", "")),
                "beam_discrepancy_depth": str(r.get("fwd_profile_beam_discrepancy_depth", "")),
                "key_multi_policy": str(r.get("fwd_profile_beam_key_multi_policy", "")),
            }

        std = extract(by_vl.get("standard_beam"))
        row_out: dict[str, Any] = {
            "family_id": fid,
            "K": k,
            "seed": seed,
            "std_opt": std["opt"],
            "std_gap": std["gap"],
            "std_rt": std["rt"],
        }
        for vl in variants_of_interest:
            if vl == "standard_beam":
                continue
            v = extract(by_vl.get(vl))
            prefix = vl.replace("_mult2", "").replace("_mult3", "").replace("_mult3_feas", "")
            row_out[f"{prefix}_opt"] = v["opt"]
            row_out[f"{prefix}_gap"] = v["gap"]
            row_out[f"{prefix}_rt"] = v["rt"]
            row_out[f"{prefix}_beam_ub"] = v["beam_ub"]
            row_out[f"{prefix}_beam_status"] = v["beam_status"]
            row_out[f"{prefix}_beam_rt"] = v["beam_rt"]
            row_out[f"{prefix}_beam_states"] = v["beam_states"]
            row_out[f"{prefix}_deciding_step"] = v["deciding_step"]

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

        for vl in variants_of_interest:
            if vl == "standard_beam":
                continue
            v = extract(by_vl.get(vl))
            prefix = vl.replace("_mult2", "").replace("_mult3", "").replace("_mult3_feas", "")
            row_out[f"winner_{prefix}"] = winner(std, v)

        out.append(row_out)
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

        # Wins vs standard, uniform, ambig
        def count_wins(opponent: str) -> int:
            w = 0
            for r in rs:
                fid = str(r.get("family_id", ""))
                seed = int(r.get("seed", -1))
                # Find opponent row with same family/seed
                opp_rows = [x for x in all_rows if str(x.get("variant_label", "")) == opponent
                            and str(x.get("family_id", "")) == fid and int(x.get("seed", -1)) == seed]
                if not opp_rows:
                    continue
                opp = opp_rows[0]
                g_r = _effective_gap(r)
                g_o = _effective_gap(opp)
                if g_r < g_o - 1e-6:
                    w += 1
                elif abs(g_r - g_o) < 1e-6:
                    try:
                        if float(r.get("runtime_sec", "inf")) < float(opp.get("runtime_sec", "inf")):
                            w += 1
                    except Exception:
                        pass
            return w

        wins_std = count_wins("standard_beam")
        wins_uni = count_wins("uniform_mult2")
        wins_ambig = count_wins("ambig_scoreband_mult2")

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
            "wins_vs_standard": wins_std,
            "wins_vs_uniform": wins_uni,
            "wins_vs_ambig": wins_ambig,
        })
    return out


def write_notes(all_rows: list[dict[str, Any]], compare: list[dict[str, Any]], summary: list[dict[str, Any]], gate1_pass: bool, best_role_variant: str | None) -> None:
    lines = [
        "# PLAN23 Role-Based Beam Notes",
        "",
        "## Purpose",
        "",
        "Test whether role-based survivor representatives are more stable than:",
        "- standard beam",
        "- uniform multiplicity",
        "- ambig_scoreband_mult2",
        "",
        "## Gate 1 result",
        "",
        f"- Gate 1 pass: {gate1_pass}",
    ]
    if best_role_variant:
        lines.append(f"- Best Gate 1 role variant: {best_role_variant}")
    lines.append("")

    lines.append("## Summary by variant")
    lines.append("")
    lines.append("| variant | rows | exact | finite | timeout | mean_gap% | mean_rt(s) | mean_rss(GB) | wins_vs_std | wins_vs_uni | wins_vs_ambig |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in summary:
        lines.append(
            f"| {s['variant_label']} | {s['n_rows']} | {s['exact']} | {s['finite_gap']} | {s['timeout']} | "
            f"{s['mean_gap_pct']} | {s['mean_runtime_sec']} | {s['mean_peak_rss_gb']} | "
            f"{s['wins_vs_standard']} | {s['wins_vs_uniform']} | {s['wins_vs_ambig']} |"
        )
    lines.append("")

    lines.append("## Per-row comparison")
    lines.append("")
    headers = ["family", "seed", "std_gap", "std_rt"]
    for vl in ["uniform_mult2", "ambig_scoreband_mult2", "role_mult3", "role_mult3_feas"]:
        prefix = vl.replace("_mult2", "").replace("_mult3", "").replace("_mult3_feas", "")
        headers.extend([f"{prefix}_gap", f"{prefix}_rt", f"winner_{prefix}"])
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for c in compare:
        cells = [c["family_id"], str(c["seed"]), c["std_gap"], c["std_rt"]]
        for vl in ["uniform_mult2", "ambig_scoreband_mult2", "role_mult3", "role_mult3_feas"]:
            prefix = vl.replace("_mult2", "").replace("_mult3", "").replace("_mult3_feas", "")
            cells.extend([c.get(f"{prefix}_gap", ""), c.get(f"{prefix}_rt", ""), c.get(f"winner_{prefix}", "")])
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Step 3 beam interpretation
    lines.append("## Step 3 beam interpretation")
    lines.append("")
    lines.append("Beam metrics for role variants:")
    lines.append("")
    lines.append("| family | seed | variant | beam_ub | beam_status | beam_time | beam_states_kept | deciding_step |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for c in compare:
        for vl in ["role_mult3", "role_mult3_feas"]:
            prefix = vl.replace("_mult2", "").replace("_mult3", "").replace("_mult3_feas", "")
            gap = c.get(f"{prefix}_gap", "")
            if gap != "inf" and gap != "":
                lines.append(
                    f"| {c['family_id']} | {c['seed']} | {vl} | {c.get(prefix + '_beam_ub', '')} | "
                    f"{c.get(prefix + '_beam_status', '')} | {c.get(prefix + '_beam_rt', '')} | "
                    f"{c.get(prefix + '_beam_states', '')} | {c.get(prefix + '_deciding_step', '')} |"
                )
    lines.append("")

    # Final certification interpretation
    lines.append("## Final certification interpretation")
    lines.append("")
    for vl in ["role_mult3", "role_mult3_feas"]:
        prefix = vl.replace("_mult2", "").replace("_mult3", "").replace("_mult3_feas", "")
        w = 0
        l = 0
        for c in compare:
            g_role = _effective_gap(by_vl_from_compare(c, vl))
            if g_role >= float("inf"):
                continue
            winner = c.get(f"winner_{prefix}", "")
            if winner.startswith("variant"):
                w += 1
            elif winner.startswith("standard"):
                l += 1
        lines.append(f"- {vl} vs standard_beam: wins={w}, losses={l} (over rows where {vl} ran)")
    lines.append("")

    # Answers to required questions
    lines.append("## Answers")
    lines.append("")

    def variant_gap_list(vl: str) -> list[float]:
        return []

    # Q1: Did role-based node evaluation improve over standard?
    q1_wins = {}
    q1_losses = {}
    q1_ties = {}
    for vl in ["role_mult3", "role_mult3_feas"]:
        prefix = vl.replace("_mult2", "").replace("_mult3", "").replace("_mult3_feas", "")
        w = 0
        l = 0
        t = 0
        for c in compare:
            g_role = _effective_gap(by_vl_from_compare(c, vl))
            g_std = _effective_gap(by_vl_from_compare(c, "standard_beam"))
            if g_role >= float("inf") or g_std >= float("inf"):
                continue
            winner = c.get(f"winner_{prefix}", "")
            if winner.startswith("variant"):
                w += 1
            elif winner.startswith("standard"):
                l += 1
            else:
                t += 1
        q1_wins[vl] = w
        q1_losses[vl] = l
        q1_ties[vl] = t
    best_role = max(["role_mult3", "role_mult3_feas"], key=lambda vl: q1_wins[vl] - q1_losses[vl])
    lines.append("1. Did role-based node evaluation improve over standard?")
    for vl in ["role_mult3", "role_mult3_feas"]:
        lines.append(f"   - {vl}: wins={q1_wins[vl]}, losses={q1_losses[vl]}, ties={q1_ties[vl]} (over rows where both ran)")
    if q1_wins[best_role] > q1_losses[best_role]:
        lines.append(f"   - Yes, {best_role} wins more often than it loses.")
    elif q1_wins[best_role] < q1_losses[best_role]:
        lines.append(f"   - No, {best_role} loses more often than it wins.")
    else:
        lines.append("   - Mixed; no clear directional signal.")
    lines.append("")

    # Q2: Did it improve over uniform multiplicity?
    lines.append("2. Did it improve over uniform multiplicity?")
    for vl in ["role_mult3", "role_mult3_feas"]:
        prefix = vl.replace("_mult2", "").replace("_mult3", "").replace("_mult3_feas", "")
        w = 0
        l = 0
        for c in compare:
            g1 = _effective_gap(by_vl_from_compare(c, vl))
            g2 = _effective_gap(by_vl_from_compare(c, "uniform_mult2"))
            if g1 >= float("inf") or g2 >= float("inf"):
                continue
            if g1 < g2 - 1e-6:
                w += 1
            elif g2 < g1 - 1e-6:
                l += 1
        lines.append(f"   - {vl} vs uniform_mult2: wins={w}, losses={l}")
    lines.append("")

    # Q3: Did it improve over ambig_scoreband?
    lines.append("3. Did it improve over ambig_scoreband?")
    for vl in ["role_mult3", "role_mult3_feas"]:
        prefix = vl.replace("_mult2", "").replace("_mult3", "").replace("_mult3_feas", "")
        w = 0
        l = 0
        for c in compare:
            g1 = _effective_gap(by_vl_from_compare(c, vl))
            g2 = _effective_gap(by_vl_from_compare(c, "ambig_scoreband_mult2"))
            if g1 >= float("inf") or g2 >= float("inf"):
                continue
            if g1 < g2 - 1e-6:
                w += 1
            elif g2 < g1 - 1e-6:
                l += 1
        lines.append(f"   - {vl} vs ambig_scoreband_mult2: wins={w}, losses={l}")
    lines.append("")

    # Q4: Did it reduce seed-dependence?
    lines.append("4. Did it reduce seed-dependence?")
    std_gaps_by_family: dict[str, list[float]] = {}
    role_gaps_by_family: dict[str, list[float]] = {}
    for c in compare:
        fam = c["family_id"]
        g_std = _effective_gap(by_vl_from_compare(c, "standard_beam"))
        if g_std < float("inf"):
            std_gaps_by_family.setdefault(fam, []).append(g_std)
        if best_role_variant:
            g_role = _effective_gap(by_vl_from_compare(c, best_role_variant))
            if g_role < float("inf"):
                role_gaps_by_family.setdefault(fam, []).append(g_role)
    for fam in sorted(std_gaps_by_family.keys()):
        std_vals = std_gaps_by_family[fam]
        role_vals = role_gaps_by_family.get(fam, [])
        if len(std_vals) >= 2 and len(role_vals) >= 2:
            std_range = max(std_vals) - min(std_vals)
            role_range = max(role_vals) - min(role_vals)
            lines.append(f"   - {fam}: std_range={std_range:.4f} role_range={role_range:.4f} {'reduced' if role_range < std_range else 'increased'}")
    lines.append("")

    # Q5: Did it help hardB_k10?
    lines.append("5. Did it help hardB_k10?")
    hardb = [c for c in compare if c["family_id"] == "hardB_k10"]
    if best_role_variant:
        prefix = best_role_variant.replace("_mult2", "").replace("_mult3", "").replace("_mult3_feas", "")
        hb_wins = sum(1 for c in hardb if c.get(f"winner_{prefix}", "").startswith("variant"))
        hb_losses = sum(1 for c in hardb if c.get(f"winner_{prefix}", "").startswith("standard"))
        lines.append(f"   - {best_role_variant} vs standard on hardB_k10: wins={hb_wins}, losses={hb_losses}")
        if hb_wins > hb_losses:
            lines.append("   - Yes, it improves hardB_k10.")
        elif hb_wins < hb_losses:
            lines.append("   - No, it does not improve hardB_k10.")
        else:
            lines.append("   - Mixed on hardB_k10.")
    else:
        lines.append("   - N/A (Gate 1 failed)")
    lines.append("")

    # Q6: Did it help K=12 incumbent production?
    lines.append("6. Did it help K=12 incumbent production?")
    k12 = [c for c in compare if c["family_id"].endswith("_k12")]
    for vl in ["standard_beam", "uniform_mult2", "ambig_scoreband_mult2", "role_mult3", "role_mult3_feas"]:
        prefix = vl.replace("_mult2", "").replace("_mult3", "").replace("_mult3_feas", "")
        if prefix == "standard_beam":
            prefix = "std"
        finite = sum(1 for c in k12 if c.get(f"{prefix}_gap", "inf") not in ("inf", ""))
        lines.append(f"   - {vl} finite gaps: {finite} / {len(k12)}")
    lines.append("")

    # Q7: Is the remaining wall Step 3 beam quality or Step 4 certification?
    lines.append("7. Is the remaining wall Step 3 beam quality or Step 4 certification?")
    dec_steps: dict[str, int] = {}
    for c in compare:
        for vl in ["standard_beam", "uniform_mult2", "ambig_scoreband_mult2", "role_mult3", "role_mult3_feas"]:
            prefix = vl.replace("_mult2", "").replace("_mult3", "").replace("_mult3_feas", "")
            ds = c.get(f"{prefix}_deciding_step", "")
            if ds:
                dec_steps[ds] = dec_steps.get(ds, 0) + 1
    for ds, cnt in sorted(dec_steps.items(), key=lambda x: -x[1]):
        lines.append(f"   - {ds}: {cnt}")
    step4_dominant = dec_steps.get("step4", 0) > sum(dec_steps.values()) * 0.5
    if step4_dominant:
        lines.append("   - Step 4 exact DP is the deciding step on most rows. The wall is Step 4 certification, not Step 3 beam quality.")
    else:
        lines.append("   - Step 3 beam is the deciding step on a significant fraction. The wall includes Step 3 beam quality.")
    lines.append("")

    # Final decision
    lines.append("## Final decision")
    lines.append("")

    if not gate1_pass:
        lines.append("**Decision: E** — Gate 1 failed. No survivor-policy change is validated; move next to beam-guided Step 4 certification.")
    else:
        # Compute overall stats for best role variant
        prefix = best_role_variant.replace("_mult2", "").replace("_mult3", "").replace("_mult3_feas", "") if best_role_variant else ""
        total_wins = sum(1 for c in compare if c.get(f"winner_{prefix}", "").startswith("variant"))
        total_losses = sum(1 for c in compare if c.get(f"winner_{prefix}", "").startswith("standard"))
        total_rows = total_wins + total_losses + sum(1 for c in compare if c.get(f"winner_{prefix}", "") == "tie")

        # Decision logic
        if total_wins > total_losses and gate1_pass:
            if best_role_variant == "role_mult3":
                decision = "A"
                reason = "`role_mult3` beats standard on most rows, passes Gate 1, and shows stable improvement."
            else:
                decision = "B"
                reason = "`role_mult3_feas` beats standard on most rows, passes Gate 1, and the feasibility role adds value."
        else:
            decision = "E"
            reason = "Role policy passes Gate 1 but does not demonstrate consistent global improvement across all tested rows."

        lines.append(f"**Decision: {decision}** — {reason}")

    NOTES_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {NOTES_MD}")


def by_vl_from_compare(c: dict[str, Any], vl: str) -> dict[str, Any]:
    """Reconstruct a pseudo-row dict from compare row for gap extraction."""
    prefix = vl.replace("_mult2", "").replace("_mult3", "").replace("_mult3_feas", "")
    if prefix == "standard_beam":
        prefix = "std"
    return {
        "gap_pct": c.get(f"{prefix}_gap", "inf"),
        "is_optimal": c.get(f"{prefix}_opt", "0"),
    }


if __name__ == "__main__":
    main()

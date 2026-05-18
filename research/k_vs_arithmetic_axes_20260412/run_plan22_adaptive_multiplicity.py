#!/usr/bin/env python3
"""
PLAN22: adaptive node evaluation / survivor policy for profile_repair_beam.

Gate 1 rows:
- hardA_k10 seed=0,1
- hardA_k8 seed=1,3

Gate 2 rows (only if Gate 1 passes):
- hardA_k10 seeds 2,3
- hardB_k10 seeds 0,1,2,3
- hardA_k12 seeds 0,1
- hardB_k12 seeds 0,1
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

OUT_DIR = ROOT / "research" / "k_vs_arithmetic_axes_20260412" / "csv" / "plan22"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = OUT_DIR / "PLAN22_adaptive_node_eval_raw.csv"
COMPARE_CSV = OUT_DIR / "PLAN22_adaptive_node_eval_compare.csv"
SUMMARY_CSV = OUT_DIR / "PLAN22_adaptive_node_eval_summary.csv"
NOTES_MD = OUT_DIR / "PLAN22_adaptive_node_eval_notes.md"

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


def build_plan22_payload(
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
        name=f"plan22/{family_id}_n{n_jobs}_lam{lam:.1f}_s{seed}",
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
            "plan22": "1",
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
    payload = build_plan22_payload(family_id, sizes, N_JOBS, LAMBDA, seed)
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
    raw["experiment_family"] = variant_label.split("_")[0] if "_" in variant_label else "baseline"
    raw["family_id"] = family_id
    raw["K"] = str(k)
    raw["family_label"] = sizes_label(sizes)
    raw["family_class"] = "hard_irregular_A" if "hardA" in family_id else "hard_irregular_B"
    raw["family_sizes"] = ",".join(str(x) for x in sizes)
    rows.append(raw)
    seen.add(key)
    write_csv(RAW_CSV, rows)
    print(
        f"[plan22] {family_id} {variant_label} seed={seed} K={k} "
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
    "uniform_mult2": {
        **STANDARD_BEAM_ENV,
        "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_POLICY": "uniform",
        "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_MAX": "2",
    },
    "uniform_mult3_control": {
        **STANDARD_BEAM_ENV,
        "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_POLICY": "uniform",
        "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_MAX": "3",
    },
    "early_mult2": {
        **STANDARD_BEAM_ENV,
        "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_POLICY": "early",
        "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_MAX": "2",
    },
    "ambig_scoreband_mult2": {
        **STANDARD_BEAM_ENV,
        "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_POLICY": "ambig_scoreband",
        "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_MAX": "2",
    },
    "hybrid_mult2": {
        **STANDARD_BEAM_ENV,
        "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_POLICY": "hybrid",
        "PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_MAX": "2",
    },
}

GATE1_ANCHORS = [
    ("hardA_k10", 10, [0, 1]),
    ("hardA_k8", 8, [1, 3]),
]

GATE2_ANCHORS = [
    ("hardA_k10", 10, [2, 3]),
    ("hardB_k10", 10, [0, 1, 2, 3]),
    ("hardA_k12", 12, [0, 1]),
    ("hardB_k12", 12, [0, 1]),
]


def main() -> None:
    solver = ROOT / "solvers" / "cpp" / "build" / "stateful_compare"
    if not solver.exists():
        raise FileNotFoundError(f"Missing solver binary: {solver}")

    rows: list[dict[str, Any]] = load_raw(RAW_CSV)
    seen = {row_key(r) for r in rows}

    # -----------------------------------------------------------------------
    # Gate 1
    # -----------------------------------------------------------------------
    print("=== PLAN22 Gate 1 ===")
    gate1_variants = ["standard_beam", "uniform_mult2", "uniform_mult3_control", "early_mult2", "ambig_scoreband_mult2", "hybrid_mult2"]
    for family_id, k, seeds in GATE1_ANCHORS:
        sizes = HARD_A_BASE[:k]
        for seed in seeds:
            for vl in gate1_variants:
                run_variant(family_id, sizes, k, seed, vl, VARIANTS[vl], rows, seen)

    # Evaluate Gate 1
    gate1_pass, best_policy = evaluate_gate1(rows)
    print(f"Gate 1 result: pass={gate1_pass} best_policy={best_policy}")

    if gate1_pass:
        print("=== PLAN22 Gate 2 ===")
        gate2_variants = ["standard_beam", best_policy]
        # optional uniform_mult2 control if best is not uniform_mult2
        if best_policy != "uniform_mult2":
            gate2_variants.append("uniform_mult2")
        for family_id, k, seeds in GATE2_ANCHORS:
            sizes = HARD_B_BASE[:k] if family_id.startswith("hardB") else HARD_A_BASE[:k]
            for seed in seeds:
                for vl in gate2_variants:
                    run_variant(family_id, sizes, k, seed, vl, VARIANTS[vl], rows, seen)
    else:
        print("Gate 1 failed. Stopping experiments.")

    # Build artifacts
    build_artifacts(rows, gate1_pass, best_policy)
    print("PLAN22 complete.")


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


def evaluate_gate1(rows: list[dict[str, Any]]) -> tuple[bool, str]:
    """
    Gate 1 decision:
    Continue to Gate 2 only if one adaptive policy is not worse than standard
    on at least 3/4 rows and improves either gap or runtime on at least 2/4 rows.
    """
    gate1_keys = []
    for family_id, k, seeds in GATE1_ANCHORS:
        for seed in seeds:
            gate1_keys.append((family_id, seed))

    # Gather results
    by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for r in rows:
        fid = str(r.get("family_id", ""))
        try:
            seed = int(r.get("seed", "-1"))
        except Exception:
            continue
        key = (fid, seed)
        if key not in gate1_keys:
            continue
        vl = str(r.get("variant_label", ""))
        by_key.setdefault(key, {})[vl] = r

    adaptive_policies = ["uniform_mult2", "uniform_mult3_control", "early_mult2", "ambig_scoreband_mult2", "hybrid_mult2"]

    best_policy = ""
    best_score = -1

    for policy in adaptive_policies:
        not_worse = 0
        improved = 0
        total = 0
        for key in gate1_keys:
            group = by_key.get(key, {})
            std = group.get("standard_beam")
            pol = group.get(policy)
            if std is None or pol is None:
                continue
            total += 1
            std_gap = _effective_gap(std)
            pol_gap = _effective_gap(pol)
            std_opt = str(std.get("is_optimal", "0")) == "1"
            pol_opt = str(pol.get("is_optimal", "0")) == "1"

            # not worse: same or better exactness, same or better gap, same or better runtime
            not_worse_flag = False
            if pol_opt and not std_opt:
                not_worse_flag = True
                improved += 1
            elif std_opt and not pol_opt:
                not_worse_flag = False
            else:
                # compare gap
                if pol_gap < std_gap - 1e-6:
                    not_worse_flag = True
                    improved += 1
                elif abs(pol_gap - std_gap) < 1e-6:
                    # compare runtime
                    try:
                        pol_rt = float(str(pol.get("runtime_sec", "nan")))
                        std_rt = float(str(std.get("runtime_sec", "nan")))
                        if pol_rt < std_rt - 1e-3:
                            not_worse_flag = True
                            improved += 1
                        elif pol_rt <= std_rt + 1e-3:
                            not_worse_flag = True
                    except Exception:
                        not_worse_flag = True
                else:
                    not_worse_flag = False

            if not_worse_flag:
                not_worse += 1

        if total == 0:
            continue
        score = not_worse * 10 + improved
        if not_worse >= 3 and improved >= 2 and score > best_score:
            best_score = score
            best_policy = policy

    return (best_policy != ""), best_policy


def build_artifacts(all_rows: list[dict[str, Any]], gate1_pass: bool, best_policy: str) -> None:
    write_csv(RAW_CSV, all_rows)
    print(f"Wrote {RAW_CSV} n={len(all_rows)}")

    compare = build_compare(all_rows)
    write_csv(COMPARE_CSV, compare)
    print(f"Wrote {COMPARE_CSV} n={len(compare)}")

    summary = build_summary(all_rows)
    write_csv(SUMMARY_CSV, summary)
    print(f"Wrote {SUMMARY_CSV} n={len(summary)}")

    write_notes(all_rows, gate1_pass, best_policy)


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


def write_notes(all_rows: list[dict[str, Any]], gate1_pass: bool, best_policy: str) -> None:
    compare = build_compare(all_rows)
    summary = build_summary(all_rows)

    lines = [
        "# PLAN22 Adaptive Node Evaluation Notes",
        "",
        "## Gate 1 result",
        "",
        f"- Gate 1 pass: {gate1_pass}",
    ]
    if best_policy:
        lines.append(f"- Best Gate 1 adaptive policy: {best_policy}")
    lines.append("")

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
    lines.append("| family | seed | variant | gap% | rt(s) | vs_standard |")
    lines.append("|---|---|---|---|---|---|")
    for c in compare:
        lines.append(
            f"| {c['family_id']} | {c['seed']} | {c['variant_label']} | {c['variant_gap']} | {c['variant_rt']} | {c['winner']} |"
        )
    lines.append("")

    lines.append("## Answers")
    lines.append("")

    # 1. Did adaptive node evaluation improve over standard beam?
    std_wins = sum(1 for c in compare if c.get("winner", "").startswith("standard"))
    var_wins = sum(1 for c in compare if c.get("winner", "").startswith("variant"))
    ties = sum(1 for c in compare if c.get("winner", "") == "tie")
    lines.append(f"1. Did adaptive node evaluation improve over standard beam?")
    lines.append(f"   - Variant wins: {var_wins}, Standard wins: {std_wins}, Ties: {ties}")
    if var_wins > std_wins:
        lines.append(f"   - Yes, adaptive variants won more often than standard.")
    elif var_wins < std_wins:
        lines.append(f"   - No, standard beam won more often.")
    else:
        lines.append(f"   - Mixed; no clear winner.")
    lines.append("")

    # 2. Did it improve over naive uniform multiplicity?
    uniform_rows = [r for r in all_rows if r.get("variant_label") == "uniform_mult2"]
    if uniform_rows:
        lines.append(f"2. Did it improve over naive uniform multiplicity?")
        lines.append(f"   - uniform_mult2 is the baseline naive policy. Best adaptive policy is compared above.")
    else:
        lines.append(f"2. Did it improve over naive uniform multiplicity?")
        lines.append(f"   - No uniform_mult2 data.")
    lines.append("")

    # 3. Which policy is best?
    lines.append(f"3. Which policy is best: early, ambig_scoreband, or hybrid?")
    lines.append(f"   - Best Gate 1 policy: {best_policy if best_policy else 'none passed gate'}")
    lines.append("")

    # 4. Did it help K=10 generally, or only one seed?
    k10_rows = [r for r in all_rows if str(r.get("family_id", "")).endswith("_k10")]
    k10_compare = [c for c in compare if str(c.get("family_id", "")).endswith("_k10")]
    k10_var_wins = sum(1 for c in k10_compare if c.get("winner", "").startswith("variant"))
    k10_total = len(k10_compare)
    lines.append(f"4. Did it help K=10 generally, or only one seed?")
    lines.append(f"   - K=10 variant wins: {k10_var_wins} / {k10_total}")
    lines.append("")

    # 5. Did it help K=12 incumbent production?
    k12_rows = [r for r in all_rows if str(r.get("family_id", "")).endswith("_k12")]
    if k12_rows:
        k12_finite = sum(1 for r in k12_rows if _effective_gap(r) < float("inf"))
        lines.append(f"5. Did it help K=12 incumbent production?")
        lines.append(f"   - K=12 rows with finite gap: {k12_finite} / {len(k12_rows)}")
    else:
        lines.append(f"5. Did it help K=12 incumbent production?")
        lines.append(f"   - No K=12 data (Gate 1 may have failed or K=12 not run).")
    lines.append("")

    # 6. Should this become the next promotion candidate?
    lines.append(f"6. Should this become the next promotion candidate?")
    if gate1_pass and best_policy:
        lines.append(f"   - Gate 1 passed with best policy {best_policy}. This is a candidate for promotion.")
    else:
        lines.append(f"   - Gate 1 failed. Adaptive multiplicity is not validated for promotion.")
    lines.append("")

    lines.append("## Final decision")
    lines.append("")
    if not gate1_pass:
        lines.append("**Decision: E** — abandon multiplicity for now and return to post-beam closure.")
    elif best_policy == "hybrid_mult2":
        lines.append("**Decision: A** — promote hybrid_mult2 as the next main candidate.")
    elif best_policy == "ambig_scoreband_mult2":
        lines.append("**Decision: B** — promote ambig_scoreband_mult2 as the next main candidate.")
    elif best_policy == "early_mult2":
        lines.append("**Decision: C** — promote early_mult2 as the next main candidate.")
    else:
        lines.append("**Decision: D** — keep only uniform multiplicity as a diagnostic, not a method.")
    lines.append("")

    NOTES_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {NOTES_MD}")


if __name__ == "__main__":
    main()

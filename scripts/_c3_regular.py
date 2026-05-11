#!/usr/bin/env python3
"""Phase C3-Regular: re-run smoke with capped instances and fixed EHS time limits.

Usage: /opt/miniconda3/envs/new-ml-env/bin/python3 scripts/_c3_regular.py
"""

import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from glns.paper_heuristics import run_ehs

ITER_DIR = PROJECT_ROOT / "research/glns_llm_heuristic_20260422/iterations/20260510_phaseC_adversarial_ehs_benchmark_design"
FAMILIES_DIR = ITER_DIR / "families"
INST_DIR = ITER_DIR / "generated_instances" / "c3_regular"
EVAL_DIR = ITER_DIR / "eval"
NOTES_DIR = ITER_DIR / "notes"

os.makedirs(INST_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(NOTES_DIR, exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────────────
SHORT_BUDGET = 30
LONG_BUDGET = 90  # reduced: n≤150, T≤200 instances complete in reasonable time
N_PER_FAMILY = 3
MAX_N = 150
MAX_T = 200  # tighter cap: T=300 bimodal instances are too slow

# ── Helpers ─────────────────────────────────────────────────────────────────

def load_json(p):
    with open(p) as f:
        return json.load(f)

def save_json(data, p):
    with open(p, "w") as f:
        json.dump(data, f, indent=2)

def _sample_int(rng, lo, hi):
    return lo if lo >= hi else rng.randint(lo, hi)

def _gen_processing_times(rng, dist, n):
    ptype = dist.get("type", "uniform")
    params = dist.get("params", {})
    if ptype == "uniform":
        lo, hi = params.get("low", 1), params.get("high", 10)
        return [rng.randint(lo, hi) for _ in range(n)]
    elif ptype == "bimodal":
        s_lo, s_hi = params.get("small_low", 1), params.get("small_high", 5)
        l_lo, l_hi = params.get("large_low", 10), params.get("large_high", 20)
        sf = params.get("small_fraction", 0.5)
        return [rng.randint(s_lo, s_hi) if rng.random() < sf else rng.randint(l_lo, l_hi) for _ in range(n)]
    elif ptype == "exponential_truncated":
        scale, pmax = params.get("scale", 5.0), params.get("max", 30)
        return [min(int(rng.expovariate(1.0 / scale)) + 1, pmax) for _ in range(n)]
    elif ptype == "normal_truncated":
        mean, std = params.get("mean", 8.0), params.get("std", 3.0)
        pmin, pmax = params.get("min", 1), params.get("max", 30)
        return [max(pmin, min(int(rng.gauss(mean, std) + 0.5), pmax)) for _ in range(n)]
    return [rng.randint(1, 10) for _ in range(n)]

def _gen_machine_rates(rng, dist, m):
    mtype = dist.get("type", "uniform")
    params = dist.get("params", {})
    if mtype == "uniform":
        return [round(rng.uniform(params.get("low", 0.5), params.get("high", 3.0)), 4) for _ in range(m)]
    elif mtype == "step":
        lo, hi = params.get("low_rate", 0.5), params.get("high_rate", 3.0)
        sf = params.get("step_fraction", 0.5)
        n_hi = max(1, int(m * (1 - sf)))
        return [round(hi, 4) for _ in range(n_hi)] + [round(lo, 4) for _ in range(m - n_hi)]
    elif mtype == "exponential":
        scale, pmax = params.get("scale", 1.0), params.get("max", 5.0)
        return [round(min(rng.expovariate(1.0 / scale) * scale, pmax) + 0.1, 4) for _ in range(m)]
    return [round(rng.uniform(0.5, 3.0), 4) for _ in range(m)]

def _gen_tou(rng, tou_type, params, T):
    ct = [1.0] * T
    if tou_type == "single_peak":
        base = params.get("base_price", 1.0)
        mult = params.get("peak_multiplier", 2.0)
        ps, pe = int(params.get("peak_start", 0.3) * T), int(params.get("peak_end", 0.6) * T)
        for t in range(T):
            ct[t] = round(base * (mult if ps <= t < pe else 1.0) + rng.uniform(-0.1, 0.1), 4)
    elif tou_type == "dual_peak":
        base = params.get("base_price", 1.0)
        mult = params.get("peak_multiplier", 2.0)
        p1s, p1e = int(params.get("peak1_start", 0.2) * T), int(params.get("peak1_end", 0.35) * T)
        p2s, p2e = int(params.get("peak2_start", 0.65) * T), int(params.get("peak2_end", 0.8) * T)
        for t in range(T):
            if p1s <= t < p1e or p2s <= t < p2e:
                ct[t] = round(base * mult + rng.uniform(-0.1, 0.1), 4)
            else:
                ct[t] = round(base + rng.uniform(-0.1, 0.1), 4)
    elif tou_type == "high_variance":
        base, noise = params.get("base_price", 1.0), params.get("noise_range", 2.0)
        ct = [round(max(0.1, base + rng.uniform(-noise, noise)), 4) for _ in range(T)]
    elif tou_type == "low_variance":
        base, noise = params.get("base_price", 1.0), params.get("noise_range", 0.2)
        ct = [round(max(0.1, base + rng.uniform(-noise, noise)), 4) for _ in range(T)]
    elif tou_type == "monotonic_increasing":
        s, e = params.get("start_price", 0.5), params.get("end_price", 3.0)
        ct = [round(s + (e - s) * t / max(T - 1, 1), 4) for t in range(T)]
    elif tou_type == "monotonic_decreasing":
        s, e = params.get("start_price", 3.0), params.get("end_price", 0.5)
        ct = [round(s + (e - s) * t / max(T - 1, 1), 4) for t in range(T)]
    elif tou_type == "step_function":
        base, mult = params.get("base_price", 1.0), params.get("step_multiplier", 3.0)
        ss = int(params.get("step_start", 0.5) * T)
        ct = [round(base * mult if t >= ss else base, 4) for t in range(T)]
    return ct


def gen_instance(rng, family, idx, base_seed):
    nr = {"min": min(family["n_jobs_range"]["min"], MAX_N), "max": min(family["n_jobs_range"]["max"], MAX_N)}
    mr = family["m_machines_range"]
    tr = {"min": min(family["horizon_T_range"]["min"], MAX_T), "max": min(family["horizon_T_range"]["max"], MAX_T)}
    n = _sample_int(rng, nr["min"], nr["max"])
    m = _sample_int(rng, mr["min"], mr["max"])
    T = _sample_int(rng, tr["min"], tr["max"])
    p = _gen_processing_times(rng, family["processing_time_distribution"], n)
    e = _gen_machine_rates(rng, family["machine_rate_distribution"], m)
    ct = _gen_tou(rng, family.get("TOU_price_profile_type", "single_peak"), family.get("TOU_price_profile_params", {}), T)

    inst = {"n": n, "m": m, "T": T, "p": p, "e": e, "ct": ct,
            "instance_id": f"c3r_{family['family_name']}_{base_seed}",
            "metadata": {"family_name": family["family_name"], "arm": family.get("_arm", "?"),
                         "seed": base_seed, "expected_mechanism": family.get("expected_EHS_failure_mechanism", "?"),
                         "feasible": sum(p) <= m * T}}
    return inst


def eval_instance(inst, budget):
    try:
        t0 = time.time()
        front = run_ehs(inst, rng=random.Random(42), time_limit_seconds=float(budget), eps_ordering="expensive_source_first")
        elapsed = time.time() - t0
        if not front:
            return {"feasible": False, "front_size": 0, "runtime_s": round(elapsed, 1),
                    "timeout": elapsed >= budget * 0.9, "deepest_cmax": -1, "best_energy": float("inf"),
                    "shallowest_cmax": -1, "error": "empty_front"}
        s = front[0]
        pts = [(sched.cmax, sched.energy) for sched in front if sched.feasible]
        return {"feasible": s.feasible, "front_size": len(front),
                "runtime_s": round(elapsed, 1), "timeout": elapsed >= budget * 0.9,
                "deepest_cmax": min(p[0] for p in pts) if pts else -1,
                "best_energy": min(p[1] for p in pts) if pts else float("inf"),
                "shallowest_cmax": max(p[0] for p in pts) if pts else -1, "error": ""}
    except Exception as e:
        return {"feasible": False, "front_size": 0, "runtime_s": 0, "error": str(e)[:200],
                "timeout": False, "deepest_cmax": -1, "best_energy": float("inf"), "shallowest_cmax": -1}


# ── C3B-Regular: Select families ────────────────────────────────────────────

print("=" * 60)
print("Phase C3-Regular: Re-selecting families with n≤150, T≤300")
print("=" * 60)

llm_data = load_json(FAMILIES_DIR / "llm_families.json")
random_data = load_json(FAMILIES_DIR / "random_families.json")
human_data = load_json(FAMILIES_DIR / "human_sweep_families.json")

llm_fams = llm_data.get("families", [])
random_fams = random_data.get("families", [])
human_fams = human_data.get("families", [])

# Filter: skip giant families (first_khat_dominance_giant, n_max > 150)
def is_eligible(fam):
    nr = fam["n_jobs_range"]
    return nr["min"] <= MAX_N and nr["max"] <= MAX_N

# Select 2 eligible LLM families, or take what's available
eligible_llm = [f for f in llm_fams if is_eligible(f)]
if len(eligible_llm) < 2:
    # Relax: use families with n_min ≤ 150 (cap at generation time)
    eligible_llm = [f for f in llm_fams if f["n_jobs_range"]["min"] <= MAX_N]

selected_llm = eligible_llm[:2]
selected_random = random_fams[:2]
selected_human = human_fams[:2]

selection = {
    "description": "C3-Regular family selection. Capped at n≤150, T≤300.",
    "constraints": {"max_n": MAX_N, "max_T": MAX_T},
    "selected": {
        "llm": [f["family_name"] for f in selected_llm],
        "random": [f["family_name"] for f in selected_random],
        "human": [f["family_name"] for f in selected_human],
    }
}

save_json(selection, NOTES_DIR / "c3_regular_selected_families.json")
for arm, fams in [("LLM", selected_llm), ("Random", selected_random), ("Human", selected_human)]:
    for f in fams:
        nr = f["n_jobs_range"]
        print(f"  {arm}: {f['family_name']} → {f.get('expected_EHS_failure_mechanism', '?')} (n=[{nr['min']},{nr['max']}])")

# ── C3C-Regular: Generate instances ─────────────────────────────────────────

print("\n" + "=" * 60)
print("C3C-Regular: Generating instances (capped)")
print("=" * 60)

rng = random.Random(42)
all_instances = []

for arm, fams in [("llm", selected_llm), ("random", selected_random), ("human", selected_human)]:
    for f in fams:
        f["_arm"] = arm
        base = f.get("seed_behavior", {}).get("base_seed", 50000)
        for i in range(N_PER_FAMILY):
            seed = base + i * 17
            inst = gen_instance(rng, f, i, seed)
            inst["metadata"]["_arm"] = arm  # tag arm on the inst
            for retry in range(10):
                if sum(inst["p"]) <= inst["m"] * inst["T"]:
                    break
                inst["n"] = max(10, inst["n"] - 5)
                inst["p"] = _gen_processing_times(random.Random(seed + retry * 100), f["processing_time_distribution"], inst["n"])
                inst["T"] = min(MAX_T, inst["T"] + 20)
            inst["metadata"]["feasible"] = sum(inst["p"]) <= inst["m"] * inst["T"]

            save_json(inst, INST_DIR / f"{inst['instance_id']}.json")
            all_instances.append(inst)
            print(f"  ✅ {inst['instance_id']}: n={inst['n']}, m={inst['m']}, T={inst['T']}, sum_p={sum(inst['p'])}, ok={inst['metadata']['feasible']}")

manifest = {"n_instances": len(all_instances),
            "n_feasible": sum(1 for i in all_instances if i["metadata"]["feasible"]),
            "instances": [{"id": i["instance_id"], "n": i["n"], "m": i["m"], "T": i["T"],
                           "sum_p": sum(i["p"]), "feasible": i["metadata"]["feasible"]}
                          for i in all_instances]}
save_json(manifest, INST_DIR / "manifest.json")

# ── C3D-Regular: EHS Evaluation ─────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"C3D-Regular: EHS eval @ {SHORT_BUDGET}s / {LONG_BUDGET}s")
print("=" * 60)

csv_path = EVAL_DIR / "c3_regular_raw.csv"
fields = ["instance_id","arm","family_name","expected_mechanism","n","m","T","sum_p",
          "feasible_instance","budget","feasible_ehs","front_size","runtime_s","timeout",
          "deepest_cmax","best_energy","shallowest_cmax","error"]

BUDGETS = [("short", SHORT_BUDGET), ("long", LONG_BUDGET)]

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for inst in all_instances:
        arm = inst["metadata"]["_arm"]
        fam = inst["metadata"]["family_name"]
        mech = inst["metadata"]["expected_mechanism"]
        s_p = sum(inst["p"])

        for blabel, budget in BUDGETS:
            print(f"  {inst['instance_id']} @{blabel}({budget}s)...", end=" ", flush=True)
            m = eval_instance(inst, budget)
            row = {"instance_id": inst["instance_id"], "arm": arm, "family_name": fam,
                   "expected_mechanism": mech, "n": inst["n"], "m": inst["m"],
                   "T": inst["T"], "sum_p": s_p,
                   "feasible_instance": inst["metadata"]["feasible"],
                   "budget": blabel, "feasible_ehs": m["feasible"],
                   "front_size": m["front_size"], "runtime_s": m["runtime_s"],
                   "timeout": m["timeout"], "deepest_cmax": m["deepest_cmax"],
                   "best_energy": m["best_energy"], "shallowest_cmax": m["shallowest_cmax"],
                   "error": m["error"]}
            writer.writerow(row)
            f.flush()
            status = "✅" if m["feasible"] else "❌"
            err = f" [{m['error'][:50]}]" if m["error"] else ""
            print(f"{status} f={m['front_size']}, cmax={m['deepest_cmax']}, "
                  f"E={m['best_energy']:.0f}, {m['runtime_s']:.1f}s{err}", flush=True)

# ── C3E-Regular: Compare ────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("C3E-Regular: Arm Comparison")
print("=" * 60)

rows = []
with open(csv_path, newline="") as f:
    for row in csv.DictReader(f):
        rows.append(row)

inst_data = defaultdict(lambda: {"short": None, "long": None})
for r in rows:
    inst_data[r["instance_id"]][r["budget"]] = r

summary_rows = []
for inst_id, budgets in sorted(inst_data.items()):
    short = budgets.get("short")
    long = budgets.get("long")
    if not short:
        continue

    fs_s = int(short["front_size"])
    fs_l = int(long["front_size"])
    dc_s = int(short["deepest_cmax"])
    dc_l = int(long["deepest_cmax"])
    cmax_delta = dc_s - dc_l if dc_s > 0 and dc_l > 0 else -1

    high_yield = False
    yield_reason = "none"

    if short["error"] or long["error"]:
        yield_reason = f"ehs_error: {short.get('error','')[:40]}"
    elif not short["feasible_ehs"] or not long["feasible_ehs"]:
        yield_reason = "ehs_infeasible"
    elif fs_s == 0 and fs_l > 0:
        high_yield = True
        yield_reason = f"zero_at_short→{fs_l}_at_long"
    elif fs_l - fs_s >= 2:
        high_yield = True
        yield_reason = f"fs_growth_+{fs_l - fs_s}"
    elif fs_s <= 1 and fs_l >= 3:
        high_yield = True
        yield_reason = f"near_zero_at_short (1→{fs_l})"
    elif cmax_delta > 5:
        high_yield = True
        yield_reason = f"cmax_improved_by_{cmax_delta}"
    elif short["timeout"] == "True" and not long["timeout"] == "True":
        high_yield = True
        yield_reason = "short_timeout_long_ok"
    elif fs_s > 0 and fs_l == fs_s:
        yield_reason = "no_growth_saturated"
    else:
        yield_reason = f"normal (Δfs={fs_l-fs_s})"

    summary_rows.append({
        "instance_id": inst_id,
        "arm": short["arm"],
        "family_name": short["family_name"],
        "expected_mechanism": short["expected_mechanism"],
        "n": short["n"], "m": short["m"], "T": short["T"],
        "short_front_size": fs_s, "short_runtime_s": short["runtime_s"],
        "short_deepest_cmax": dc_s, "short_best_energy": short["best_energy"],
        "long_front_size": fs_l, "long_runtime_s": long["runtime_s"],
        "long_deepest_cmax": dc_l, "long_best_energy": long["best_energy"],
        "cmax_delta": dc_s - dc_l if dc_s > 0 and dc_l > 0 else 0,
        "high_yield": high_yield, "yield_reason": yield_reason,
    })

sum_path = EVAL_DIR / "c3_regular_summary.csv"
with open(sum_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)

# Arm stats
arm_stats = defaultdict(lambda: {"total": 0, "high_yield": 0, "fronts": [], "mechanisms": set()})
for r in summary_rows:
    a = r["arm"]
    arm_stats[a]["total"] += 1
    if r["high_yield"]:
        arm_stats[a]["high_yield"] += 1
    arm_stats[a]["fronts"].append(f"{r['short_front_size']}→{r['long_front_size']}")
    arm_stats[a]["mechanisms"].add(r["expected_mechanism"])

print(f"\n  {'Arm':<10} {'Inst':>5} {'Yield':>6} {'Rate':>8}  Short→Long Fronts        Mechanisms")
print(f"  {'-'*75}")
for arm in ["llm", "random", "human"]:
    s = arm_stats[arm]
    rate = f"{s['high_yield']/s['total']*100:.0f}%" if s["total"] else "N/A"
    fronts = ", ".join(s["fronts"])
    mechs = ", ".join(sorted(s["mechanisms"]))
    print(f"  {arm:<10} {s['total']:>5} {s['high_yield']:>6} {rate:>8}  {fronts:<25} {mechs}")

print("\n  Per-instance:")
for r in summary_rows:
    tag = "🔴" if r["high_yield"] else "🟢"
    print(f"  {tag} {r['instance_id']:<45} {r['arm']:<8} "
          f"fs={r['short_front_size']:>2}→{r['long_front_size']:>2} "
          f"cmax={r['short_deepest_cmax']:>3}→{r['long_deepest_cmax']:>3} "
          f"[{r['yield_reason'][:55]}]")

# Gate decision
llm_ry = arm_stats["llm"]["high_yield"] / max(arm_stats["llm"]["total"], 1)
random_ry = arm_stats["random"]["high_yield"] / max(arm_stats["random"]["total"], 1)
human_ry = arm_stats["human"]["high_yield"] / max(arm_stats["human"]["total"], 1)

print(f"\n  Gate Assessment:")
print(f"    LLM:    {arm_stats['llm']['high_yield']}/{arm_stats['llm']['total']} = {llm_ry:.0%}")
print(f"    Random: {arm_stats['random']['high_yield']}/{arm_stats['random']['total']} = {random_ry:.0%}")
print(f"    Human:  {arm_stats['human']['high_yield']}/{arm_stats['human']['total']} = {human_ry:.0%}")

if llm_ry >= 2 * max(random_ry, human_ry) and llm_ry >= 0.33:
    gate = "STRONG: LLM ≥ 2× both baselines."
elif llm_ry >= max(random_ry, human_ry) and llm_ry >= 0.33:
    gate = "MODERATE: LLM ties/beats baselines."
elif llm_ry < max(random_ry, human_ry):
    gate = "FAIL: LLM worse than at least one baseline."
else:
    gate = "WEAK: LLM similar, insufficient margin."

print(f"    → {gate}")

# Write decision
dp = NOTES_DIR / "c3_regular_decision.md"
with open(dp, "w") as f:
    f.write("# C3-Regular Smoke Decision\n\n")
    f.write(f"**Date**: 2026-05-10\n")
    f.write(f"**Instances**: {len(all_instances)} (n≤{MAX_N}, T≤{MAX_T})\n")
    f.write(f"**Budgets**: {SHORT_BUDGET}s / {LONG_BUDGET}s\n")
    f.write(f"**EHS time-limit fix**: cooperative deadline checks in SGH/A-SGH loops\n\n")
    f.write("## Yield Rates\n\n")
    f.write("| Arm | Instances | High-Yield | Rate |\n")
    f.write("|-----|----------|------------|------|\n")
    for arm in ["llm", "random", "human"]:
        s = arm_stats[arm]
        rate = f"{s['high_yield']/s['total']*100:.0f}%" if s["total"] else "N/A"
        f.write(f"| {arm} | {s['total']} | {s['high_yield']} | {rate} |\n")
    f.write(f"\n## Gate: {gate}\n\n")
    f.write("## Per-Instance\n\n")
    for r in summary_rows:
        f.write(f"- `{r['instance_id']}` ({r['arm']}, {r['expected_mechanism']}): "
                f"n={r['n']} m={r['m']} T={r['T']} — "
                f"fs={r['short_front_size']}→{r['long_front_size']} "
                f"cmax={r['short_deepest_cmax']}→{r['long_deepest_cmax']} "
                f"{'HIGH' if r['high_yield'] else 'low'} [{r['yield_reason']}]\n")
    f.write("\n## Mechanism Match Analysis\n\n")
    for r in summary_rows:
        f.write(f"- **{r['expected_mechanism']}**: {'MATCH' if r['high_yield'] else 'NO_FAILURE'} "
                f"— {r['yield_reason']}\n")

print(f"  Decision: {dp}")
print(f"\n✅ C3-Regular smoke complete.")

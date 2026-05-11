#!/usr/bin/env python3
"""Fill in missing C3-Regular eval rows as INCOMPLETE and run analysis."""
import csv, json, os, sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

ITER_DIR = PROJECT_ROOT / "research/glns_llm_heuristic_20260422/iterations/20260510_phaseC_adversarial_ehs_benchmark_design"
INST_DIR = ITER_DIR / "generated_instances" / "c3_regular"
EVAL_DIR = ITER_DIR / "eval"
NOTES_DIR = ITER_DIR / "notes"
os.makedirs(EVAL_DIR, exist_ok=True)

def load_json(p):
    with open(p) as f: return json.load(f)

BUDGETS = [("short", 30), ("long", 90)]
csv_path = EVAL_DIR / "c3_regular_raw.csv"
manifest = load_json(INST_DIR / "manifest.json")

# Load existing rows
existing = []
seen = set()
if csv_path.exists():
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            existing.append(row)
            seen.add(row["instance_id"] + "_" + row["budget"])

# Fill missing with appropriate error markers
for inst_info in manifest["instances"]:
    with open(INST_DIR / f"{inst_info['id']}.json") as fi:
        inst = json.load(fi)
    arm = inst["metadata"].get("_arm", inst["metadata"].get("arm", "?"))
    fam = inst["metadata"]["family_name"]
    mech = inst["metadata"].get("expected_mechanism", "?")
    s_p = sum(inst["p"])
    infeasible = not inst_info.get("feasible", True)

    for blabel, budget in BUDGETS:
        key = inst_info["id"] + "_" + blabel
        if key not in seen:
            if infeasible:
                err = "instance_infeasible"
            else:
                err = "eval_incomplete_timeout"
            existing.append({
                "instance_id": inst_info["id"], "arm": arm, "family_name": fam,
                "expected_mechanism": mech, "n": inst["n"], "m": inst["m"],
                "T": inst["T"], "sum_p": s_p,
                "feasible_instance": not infeasible, "budget": blabel,
                "feasible_ehs": False, "front_size": 0, "runtime_s": 0,
                "timeout": True, "deepest_cmax": -1, "best_energy": float("inf"),
                "shallowest_cmax": -1, "error": err,
            })

# Rewrite complete CSV
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=existing[0].keys())
    writer.writeheader()
    writer.writerows(existing)

# Group by instance
inst_data = defaultdict(lambda: {"short": None, "long": None})
for r in existing:
    r["front_size"] = int(r["front_size"])
    r["feasible_ehs"] = r.get("feasible_ehs") in (True, "True")
    r["timeout"] = r.get("timeout") in (True, "True")
    r["runtime_s"] = float(r["runtime_s"])
    r["feasible_instance"] = r.get("feasible_instance") in (True, "True")
    inst_data[r["instance_id"]][r["budget"]] = r

# Analysis
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
    rt_s = float(short["runtime_s"])
    rt_l = float(long["runtime_s"])
    err = short.get("error", "") or long.get("error", "")
    infeasible = not short.get("feasible_instance", True)

    high_yield = False
    yield_reason = "none"

    if infeasible:
        yield_reason = "infeasible_instance"
    elif "incomplete" in err or "timeout" in err:
        yield_reason = "eval_incomplete"
    elif err:
        yield_reason = f"error: {err[:40]}"
    elif not short["feasible_ehs"] or not long["feasible_ehs"]:
        yield_reason = "ehs_returned_empty"
    elif fs_s == 0 and fs_l > 0:
        high_yield = True
        yield_reason = f"zero_at_short→{fs_l}_at_long"
    elif fs_l - fs_s >= 2:
        high_yield = True
        yield_reason = f"fs_growth_+{fs_l - fs_s}"
    elif fs_s <= 1 and fs_l >= 3:
        high_yield = True
        yield_reason = f"near_zero→{fs_l}"
    elif dc_s > 0 and dc_l > 0 and (dc_s - dc_l) > 3:
        high_yield = True
        yield_reason = f"cmax_Δ={dc_s - dc_l}"
    elif rt_s > 90 and fs_s <= 1:
        high_yield = True
        yield_reason = "very_slow_single_point"
    elif fs_s > 0 and fs_l == fs_s:
        yield_reason = f"saturated_at_{fs_s}"
    else:
        yield_reason = f"normal Δfs={fs_l-fs_s}"

    summary_rows.append({
        "instance_id": inst_id, "arm": short["arm"], "family_name": short["family_name"],
        "expected_mechanism": short["expected_mechanism"],
        "n": short["n"], "m": short["m"], "T": short["T"],
        "infeasible": infeasible, "short_front_size": fs_s, "short_runtime_s": rt_s,
        "short_deepest_cmax": dc_s, "short_best_energy": short.get("best_energy", "inf"),
        "long_front_size": fs_l, "long_runtime_s": rt_l,
        "long_deepest_cmax": dc_l, "long_best_energy": long.get("best_energy", "inf") if long else "inf",
        "cmax_delta": dc_s - dc_l if dc_s > 0 and dc_l > 0 else 0,
        "high_yield": high_yield, "yield_reason": yield_reason,
    })

sum_path = EVAL_DIR / "c3_regular_summary.csv"
with open(sum_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)

# Arm stats
arm_stats = defaultdict(lambda: {"total": 0, "high_yield": 0, "evaluable": 0, "infeasible": 0, "incomplete": 0, "fronts": [], "mechanisms": set()})
for r in summary_rows:
    a = r["arm"]
    arm_stats[a]["total"] += 1
    if r["high_yield"]:
        arm_stats[a]["high_yield"] += 1
    if r["infeasible"]:
        arm_stats[a]["infeasible"] += 1
    elif "incomplete" in r["yield_reason"]:
        arm_stats[a]["incomplete"] += 1
    else:
        arm_stats[a]["evaluable"] += 1
    arm_stats[a]["fronts"].append(f"{r['short_front_size']}→{r['long_front_size']}")
    arm_stats[a]["mechanisms"].add(r["expected_mechanism"])

print(f"\n{'='*70}")
print(f"C3-Regular Results")
print(f"{'='*70}")
print(f"\n  {'Arm':<10} {'Inst':>5} {'Eval':>5} {'Infeas':>7} {'Incom':>5} {'Yield':>6} {'Rate':>8}  Front Growth")
print(f"  {'-'*80}")
for arm in ["llm", "random", "human"]:
    s = arm_stats[arm]
    evaluable = s["total"] - s["infeasible"] - s["incomplete"]
    rate = f"{s['high_yield']/evaluable*100:.0f}%" if evaluable else "N/A"
    rate_total = f"{s['high_yield']/s['total']*100:.0f}%" if s["total"] else "N/A"
    fronts = ", ".join(s["fronts"])
    print(f"  {arm:<10} {s['total']:>5} {evaluable:>5} {s['infeasible']:>7} {s['incomplete']:>5} {s['high_yield']:>6} {rate:>8}  {fronts}")

print(f"\n  Per-instance:")
for r in summary_rows:
    tag = "🔴" if r["high_yield"] else "  "
    print(f"  {tag} {r['instance_id']:<50} {r['arm']:<8} "
          f"fs={r['short_front_size']:>2}→{r['long_front_size']:>2} "
          f"cmax={r['short_deepest_cmax']:>3}→{str(r['long_deepest_cmax']):>3} "
          f"[{r['yield_reason'][:50]}]")

# Gate
llm_y = arm_stats["llm"]["high_yield"]
llm_e = max(arm_stats["llm"]["total"] - arm_stats["llm"]["infeasible"] - arm_stats["llm"]["incomplete"], 1)
random_y = arm_stats["random"]["high_yield"]
random_e = max(arm_stats["random"]["total"] - arm_stats["random"]["infeasible"] - arm_stats["random"]["incomplete"], 1)
human_y = arm_stats["human"]["high_yield"]
human_e = max(arm_stats["human"]["total"] - arm_stats["human"]["infeasible"] - arm_stats["human"]["incomplete"], 1)

llm_ry = llm_y / llm_e
random_ry = random_y / random_e
human_ry = human_y / human_e

print(f"\n  Yield Rates (evaluable instances only):")
print(f"    LLM:    {llm_y}/{llm_e} = {llm_ry:.0%}")
print(f"    Random: {random_y}/{random_e} = {random_ry:.0%}")
print(f"    Human:  {human_y}/{human_e} = {human_ry:.0%}")

if llm_ry >= 2 * max(random_ry, human_ry) and llm_ry >= 0.33:
    gate = "STRONG: LLM ≥ 2× both baselines."
elif llm_ry >= max(random_ry, human_ry) and llm_ry >= 0.33:
    gate = "MODERATE: LLM ties/beats best baseline."
elif llm_ry < max(random_ry, human_ry):
    gate = "FAIL: LLM worse than at least one baseline."
else:
    gate = "WEAK: LLM similar, insufficient margin."

print(f"\n  Gate: {gate}")

# Write decision
dp = NOTES_DIR / "c3_regular_decision.md"
with open(dp, "w") as f:
    f.write("# C3-Regular Smoke Decision\n\n")
    f.write(f"**Date**: 2026-05-10\n")
    f.write(f"**Instances**: 18 (n≤{150}, T≤200)\n")
    f.write(f"**Budgets**: 30s / 90s\n")
    f.write(f"**EHS time-limit fix**: cooperative deadline checks in SGH/A-SGH loops\n\n")
    f.write(f"**Evaluable**: LLM={llm_e}, Random={random_e}, Human={human_e}\n\n")
    f.write("## Yield Rates\n\n")
    f.write("| Arm | Instances | Evaluable | Infeasible | Incomplete | High-Yield | Rate (eval) |\n")
    f.write("|-----|----------|-----------|------------|------------|------------|-------------|\n")
    for arm in ["llm", "random", "human"]:
        s = arm_stats[arm]
        eval_n = s["total"] - s["infeasible"] - s["incomplete"]
        max_hy = max(eval_n, 1)
        rate = f"{s['high_yield']/max_hy*100:.0f}%" if eval_n > 0 else "N/A"
        f.write(f"| {arm} | {s['total']} | {eval_n} | {s['infeasible']} | {s['incomplete']} | {s['high_yield']} | {rate} |\n")

    f.write(f"\n## Gate: {gate}\n\n")
    f.write("## Per-Instance\n\n")
    for r in summary_rows:
        f.write(f"- `{r['instance_id']}` ({r['arm']}, {r['expected_mechanism']}): "
                f"n={r['n']} m={r['m']} T={r['T']} — "
                f"fs={r['short_front_size']}→{r['long_front_size']} "
                f"cmax={r['short_deepest_cmax']}→{r['long_deepest_cmax']} "
                f"{'HIGH' if r['high_yield'] else 'low'} [{r['yield_reason']}]\n")
    f.write("\n## Mechanism Match\n\n")
    for r in summary_rows:
        if r["infeasible"] or "incomplete" in r["yield_reason"]:
            continue
        match = "MATCH" if r["high_yield"] else "NO_FAILURE"
        f.write(f"- **{r['expected_mechanism']}**: {match} — {r['yield_reason']} "
                f"(n={r['n']}, m={r['m']}, T={r['T']})\n")

print(f"\n  Decision: {dp}")
print("✅ C3-Regular analysis complete.")

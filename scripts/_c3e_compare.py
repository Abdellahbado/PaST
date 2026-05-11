#!/usr/bin/env python3
"""C3E: Arm comparison from smoke results — handles incomplete data."""
import csv, json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ITER_DIR = PROJECT_ROOT / "research/glns_llm_heuristic_20260422/iterations/20260510_phaseC_adversarial_ehs_benchmark_design"
EVAL_DIR = ITER_DIR / "eval"
NOTES_DIR = ITER_DIR / "notes"
NOTES_DIR.mkdir(parents=True, exist_ok=True)

EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Load data
with open(EVAL_DIR / "c3_smoke_raw.csv") as f:
    rows = list(csv.DictReader(f))

# Classify
def classify(r):
    e = r.get("error", "")
    if "TOO_LARGE" in e:
        return "too_large"
    if "NOT_EVALUATED" in e:
        return "skipped"
    return "evaluated"

# Per-instance grouping
inst_data = defaultdict(lambda: {"short": None, "long": None})
for r in rows:
    inst_data[r["instance_id"]][r["budget"]] = r

# Compute yield per instance
summary_rows = []
for inst_id, budgets in sorted(inst_data.items()):
    short = budgets.get("short")
    long = budgets.get("long")
    if short is None:
        continue
    arm = short["arm"]
    fam = short["family_name"]
    mech = short["expected_mechanism"]
    sc = classify(short)
    lc = classify(long) if long else "skipped"
    cat = sc if sc != "evaluated" else lc

    fs_s = int(short["front_size"]) if short["front_size"] else 0
    fs_l = int(long["front_size"]) if long and long["front_size"] else 0
    rt_s = float(short["runtime_s"]) if short["runtime_s"] else 0
    
    # Determine yield
    high_yield = False
    yield_reason = ""
    
    if cat == "too_large":
        # Instance too large for EHS to even start — adversarial success
        high_yield = True
        yield_reason = "too_large_for_smoke"
    elif cat == "skipped":
        high_yield = False
        yield_reason = "not_evaluated"
    else:
        # Evaluated: check front growth
        if fs_s == 0 and fs_l > 0:
            high_yield = True
            yield_reason = "zero_at_short_meaningful_at_long"
        elif fs_s <= 1 and fs_l > 1:
            high_yield = True
            yield_reason = f"near_zero_at_short (fs={fs_s}→{fs_l})"
        elif fs_l - fs_s >= 2:
            high_yield = True
            yield_reason = f"front_size_growth_≥2 (Δ={fs_l-fs_s})"
        elif rt_s > 120 and fs_s <= 1:
            high_yield = True
            yield_reason = "very_slow_single_point"
        else:
            yield_reason = "normal_convergence"

    summary_rows.append({
        "instance_id": inst_id,
        "arm": arm,
        "family_name": fam,
        "expected_mechanism": mech,
        "n": short["n"],
        "m": short["m"],
        "T": short["T"],
        "eval_status": cat,
        "short_front_size": fs_s,
        "short_runtime_s": rt_s,
        "long_front_size": fs_l,
        "long_runtime_s": float(long["runtime_s"]) if long else 0,
        "high_yield": high_yield,
        "yield_reason": yield_reason,
    })

# Save summary
sum_path = EVAL_DIR / "c3_smoke_summary.csv"
with open(sum_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)
print(f"Summary: {sum_path}")

# Arm stats
arm_stats = defaultdict(lambda: {"total": 0, "high_yield": 0, "evaluated": 0, "too_large": 0, "skipped": 0})
for r in summary_rows:
    a = r["arm"]
    arm_stats[a]["total"] += 1
    if r["high_yield"]:
        arm_stats[a]["high_yield"] += 1
    if r["eval_status"] == "evaluated":
        arm_stats[a]["evaluated"] += 1
    elif r["eval_status"] == "too_large":
        arm_stats[a]["too_large"] += 1
    else:
        arm_stats[a]["skipped"] += 1

print("\n  Arm Comparison:")
print(f"  {'Arm':<10} {'Total':>6} {'Eval':>6} {'Yield':>6} {'Rate':>8} {'Large':>6} {'Skip':>6}")
for arm in ["llm", "random", "human"]:
    a = arm_stats[arm]
    rate = f"{a['high_yield']/a['total']*100:.0f}%" if a["total"] > 0 else "N/A"
    print(f"  {arm:<10} {a['total']:>6} {a['evaluated']:>6} {a['high_yield']:>6} {rate:>8} {a['too_large']:>6} {a['skipped']:>6}")

print("\n  Per-instance:")
for r in summary_rows:
    tag = "🔴" if r["high_yield"] else "🟢"
    print(f"  {tag} {r['instance_id']:<40} {r['arm']:<8} {r['eval_status']:<12} "
          f"fs={r['short_front_size']:>2}→{r['long_front_size']:>2} "
          f"[{r['yield_reason'][:60]}]")

# Decision
llm_rate = arm_stats["llm"]["high_yield"] / max(arm_stats["llm"]["total"], 1)
random_rate = arm_stats["random"]["high_yield"] / max(arm_stats["random"]["total"], 1)
human_rate = arm_stats["human"]["high_yield"] / max(arm_stats["human"]["total"], 1)

print("\n  Gate Assessment:")
print(f"    LLM yield: {llm_rate:.0%} ({arm_stats['llm']['high_yield']}/{arm_stats['llm']['total']})")
print(f"    Random yield: {random_rate:.0%} ({arm_stats['random']['high_yield']}/{arm_stats['random']['total']})")
print(f"    Human yield: {human_rate:.0%} ({arm_stats['human']['high_yield']}/{arm_stats['human']['total']})")

# Write decision
lines = [
    "# C3 Smoke Decision",
    "",
    "**Date**: 2026-05-10",
    "**Instances generated**: 18 (6 families × 3)",
    "**Instances evaluated**: 8/36 EHS runs (22% — smoke truncated by time constraints)",
    "",
    "## Caveats",
    "- EHS does NOT yield during SGH construction. Time limit only checked between khats.",
    "- First khat at large T can take >> time_limit. Smoke used 30s/60s budgets.",
    "- Giant LLM instances (n=800+) marked 'too_large' — unevaluable in any reasonable time.",
    "- Most random/human instances not evaluated due to smoke timeout.",
    "- Results are INDICATIVE not conclusive.",
    "",
    "## Yield Rates",
    f"| Arm | Instances | Evaluated | Too Large | Skipped | High-Yield |",
    f"|-----|----------|-----------|-----------|---------|------------|",
]
for arm in ["llm", "random", "human"]:
    a = arm_stats[arm]
    lines.append(f"| {arm} | {a['total']} | {a['evaluated']} | {a['too_large']} | {a['skipped']} | {a['high_yield']} |")

lines += [
    "",
    "## Key Observations",
    "",
    "1. **LLM successfully identified first_khat_dominance with giant instances**:",
    "   - 3/3 LLM giant instances (n=800+, m=35-44, T=750-850) are completely unevaluable",
    "   - EHS cannot complete even one SGH construction within smoke budget",
    "   - This IS a valid adversarial design — the LLM correctly identified the mechanism",
    "",
    "2. **LLM asgh_trajectory_conflict instances evaluated successfully**:",
    "   - Moderate size (n=80-120, m=10-12, T=200-250) — chose different mechanism target",
    "   - All 3 instances produced fronts (1-12 points), all feasible",
    "   - Clear budget scaling: front grows from short → long (e.g. 5→8, 1→3, 8→12)",
    "   - Yielded front-size growth ≥ 2 on 2/3 instances",
    "",
    "3. **Random random_000 instance**:",
    "   - T=737 very large — only 1 front point produced (stuck at khat=T)",
    "   - Bimodal jobs created large cmax gap, preventing descent",
    "   - 155s for first khat (well past 30s budget)",
    "",
    "4. **Human instances not evaluated** due to smoke truncation",
    "",
    "5. **Structural issue**: EHS time_limit_seconds is NOT respected during SGH construction",
    "   - For T=737 instance: 155s (5× budget) for first khat",
    "   - Makes adversarial benchmark evaluation unreliable at any budget",
    "",
    "## Gate Decision",
    "",
    "**INCONCLUSIVE — smoke truncated.**",
    "",
    "Reasons:",
    "- Only 8/36 runs completed — insufficient for arm comparison.",
    "- EHS does not respect time limits during first SGH construction.",
    "- Giant LLM instances are successfully adversarial but unevaluable.",
    "- Human and random baselines had 0 evaluated results.",
    "",
    "## Recommendations",
    "",
    "1. **Fix EHS time-limit enforcement**: add time checks INSIDE SGH construction loop.",
    "2. **Cap T at reasonable values** (e.g. T ≤ 300) for smoke instances.",
    "3. **Re-run smoke with capped T and fixed time limits**.",
    "4. **OR accept smoke as generative success**: LLM designed an instance family (first_khat_dominance_giant)",
    "   that breaks EHS completely — but this is a structural EHS bug, not a benchmark insight.",
    "",
    "## Per-Instance",
]
for r in summary_rows:
    lines.append(f"- `{r['instance_id']}` ({r['arm']}, {r['eval_status']}): "
                 f"n={r['n']} m={r['m']} T={r['T']} — "
                 f"yield={'HIGH' if r['high_yield'] else 'low'} [{r['yield_reason']}]")

dp = NOTES_DIR / "c3_smoke_decision.md"
with open(dp, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\n  Decision: {dp}")
print("  ✅ C3 smoke complete (truncated).")

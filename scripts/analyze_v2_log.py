#!/usr/bin/env python3
"""Analyze V2 experiment log: complete_new.log"""
import re
import sys
from collections import defaultdict
import statistics

LOG = "/Users/mac/Documents/Study/PFE/PaST/ADP/logs/logs/rigourous/complete_new.log"

# ── Parse experiment boundaries ──────────────────────────────────
lines = open(LOG, encoding="utf-8", errors="replace").readlines()

# Parse run headers like "[1/15] TRAIN hard-small → EVAL hard-small : ramp_mlp"
run_re = re.compile(r"\[(\d+)/(\d+)\]\s+(.*?):\s+(\S+)")
# Parse gap lines like "seed=200 beam=5 exact=313.5 L=313.5 Z=336.0 P=324.0 gapL=0.00% gapZ=7.18% gapP=3.35%"
gap_re = re.compile(
    r"seed=(\d+)\s+beam=(\d+)\s+exact=([\d.]+)\s+L=([\d.]+)\s+Z=([\d.]+)\s+P=([\d.]+)\s+"
    r"gapL=([\d.]+)%\s+gapZ=([\d.]+)%\s+gapP=([\d.]+)%"
)
# Parse R² lines
r2_re = re.compile(r"R²\s*=?\s*([\d.]+)")
# Parse training summary lines
train_summary_re = re.compile(r"\[pool\]\s+Final training:\s+(\d+)\s+samples")
# Parse CSV result lines
csv_re = re.compile(r"CSV\s+results?\s+saved?\s+to\s+(\S+)", re.IGNORECASE)
# Parse experiment header
exp_header_re = re.compile(r"\[(\d+)/(\d+)\]\s+(TRAIN.*?|EVAL.*?):\s+(\S+)")
# Parse skipped seeds
skip_re = re.compile(r"seed=(\d+)\s+INFEASIBLE")

# ── Identify experiment blocks ───────────────────────────────────
current_run = None
current_phase = None  # "Bhard", "H", "E", "F", "G"
runs = []  # list of dicts

for i, line in enumerate(lines):
    # Detect experiment phase
    if "Experiment B-hard" in line:
        current_phase = "B-hard"
    elif "Experiment H" in line or "Non-Repeating" in line:
        current_phase = "H"
    elif "Experiment E" in line or "Epsilon" in line:
        current_phase = "E"
    elif "Experiment F" in line or "Drift" in line or "Weak Repeat" in line:
        current_phase = "F"
    elif "Experiment G" in line or "Forecast Bias" in line:
        current_phase = "G"

    # Detect run header
    m = run_re.search(line)
    if m:
        run_num, total, desc, config = m.groups()
        current_run = {
            "phase": current_phase,
            "run_num": int(run_num),
            "total": int(total),
            "desc": desc.strip(),
            "config": config.strip(),
            "gaps": [],  # list of (seed, beam, gapL, gapZ, gapP, exact, L, Z, P)
            "r2": [],
            "infeasible": 0,
            "line_start": i,
        }
        runs.append(current_run)

    # Parse gap lines
    mg = gap_re.search(line)
    if mg and current_run is not None:
        seed, beam, exact, lval, zval, pval, gapL, gapZ, gapP = mg.groups()
        current_run["gaps"].append(
            {
                "seed": int(seed),
                "beam": int(beam),
                "exact": float(exact),
                "L": float(lval),
                "Z": float(zval),
                "P": float(pval),
                "gapL": float(gapL),
                "gapZ": float(gapZ),
                "gapP": float(gapP),
            }
        )

    # Parse R²
    mr = r2_re.search(line)
    if mr and current_run is not None and "R²" in line:
        current_run["r2"].append(float(mr.group(1)))

    # Parse infeasible
    if "INFEASIBLE" in line and current_run is not None:
        current_run["infeasible"] += 1

# ── Also find the error at the end ──────────────────────────────
errors = []
for i, line in enumerate(lines):
    if "error:" in line.lower() or "Traceback" in line or "Exception" in line:
        errors.append((i + 1, line.strip()))

print("=" * 80)
print("V2 EXPERIMENT LOG ANALYSIS")
print("=" * 80)

# ── Error report ─────────────────────────────────────────────────
print("\n### ERRORS FOUND ###")
if errors:
    for lnum, msg in errors:
        print(f"  Line {lnum}: {msg}")
else:
    print("  None")

# ── Overall summary per phase ────────────────────────────────────
print("\n### EXPERIMENT RUNS SUMMARY ###")
phase_runs = defaultdict(list)
for r in runs:
    phase_runs[r["phase"]].append(r)

for phase in ["B-hard", "H", "E", "F", "G"]:
    pr = phase_runs.get(phase, [])
    if not pr:
        print(f"\n  Phase {phase}: NOT RUN or NO DATA")
        continue
    total_gaps = sum(len(r["gaps"]) for r in pr)
    total_infeasible = sum(r["infeasible"] for r in pr)
    print(
        f"\n  Phase {phase}: {len(pr)} runs, {total_gaps} gap measurements, {total_infeasible} infeasible"
    )
    for r in pr:
        ngaps = len(r["gaps"])
        r2_str = f"R²={r['r2'][-1]:.4f}" if r["r2"] else "no R²"
        if ngaps > 0:
            avg_gapL = statistics.mean(g["gapL"] for g in r["gaps"])
            avg_gapP = statistics.mean(g["gapP"] for g in r["gaps"])
            avg_gapZ = statistics.mean(g["gapZ"] for g in r["gaps"])
            print(
                f"    [{r['run_num']}/{r['total']}] {r['config']:30s} "
                f"n={ngaps:3d}  avgGapL={avg_gapL:6.2f}%  avgGapP={avg_gapP:6.2f}%  "
                f"avgGapZ={avg_gapZ:6.2f}%  {r2_str}"
            )
        else:
            print(
                f"    [{r['run_num']}/{r['total']}] {r['config']:30s} n=0 (no eval data)  {r2_str}"
            )

# ── Detailed per-profile analysis ────────────────────────────────
print("\n" + "=" * 80)
print("### DETAILED: WHEN L BEATS P vs WHEN P BEATS L ###")
print("=" * 80)

all_gaps = []
for r in runs:
    for g in r["gaps"]:
        g["phase"] = r["phase"]
        g["config"] = r["config"]
        g["desc"] = r["desc"]
        all_gaps.append(g)

if not all_gaps:
    print("  No gap data found!")
    sys.exit(0)

# Extract profile and model from config like "ramp_mlp"
for g in all_gaps:
    parts = g["config"].rsplit("_", 1)
    if len(parts) == 2:
        g["profile"] = parts[0]
        g["model"] = parts[1]
    else:
        g["profile"] = g["config"]
        g["model"] = "?"

# ── L vs P head-to-head ──────────────────────────────────────────
l_better = [g for g in all_gaps if g["gapL"] < g["gapP"]]
p_better = [g for g in all_gaps if g["gapP"] < g["gapL"]]
tied = [g for g in all_gaps if g["gapL"] == g["gapP"]]
l_optimal = [g for g in all_gaps if g["gapL"] == 0.0]
p_optimal = [g for g in all_gaps if g["gapP"] == 0.0]

print(f"\n  Total gap measurements: {len(all_gaps)}")
print(
    f"  L strictly better than P: {len(l_better)} ({100*len(l_better)/len(all_gaps):.1f}%)"
)
print(
    f"  P strictly better than L: {len(p_better)} ({100*len(p_better)/len(all_gaps):.1f}%)"
)
print(f"  Tied: {len(tied)} ({100*len(tied)/len(all_gaps):.1f}%)")
print(
    f"  L finds optimal (gapL=0): {len(l_optimal)} ({100*len(l_optimal)/len(all_gaps):.1f}%)"
)
print(
    f"  P finds optimal (gapP=0): {len(p_optimal)} ({100*len(p_optimal)/len(all_gaps):.1f}%)"
)

# ── Per-profile breakdown ────────────────────────────────────────
print("\n### PER-PROFILE BREAKDOWN (across all beams) ###")
profiles = sorted(set(g["profile"] for g in all_gaps))
for prof in profiles:
    pg = [g for g in all_gaps if g["profile"] == prof]
    if not pg:
        continue
    avg_gapL = statistics.mean(g["gapL"] for g in pg)
    avg_gapP = statistics.mean(g["gapP"] for g in pg)
    avg_gapZ = statistics.mean(g["gapZ"] for g in pg)
    l_wins = sum(1 for g in pg if g["gapL"] < g["gapP"])
    p_wins = sum(1 for g in pg if g["gapP"] < g["gapL"])
    ties = sum(1 for g in pg if g["gapL"] == g["gapP"])
    l_opt = sum(1 for g in pg if g["gapL"] == 0.0)
    p_opt = sum(1 for g in pg if g["gapP"] == 0.0)
    adv = avg_gapP - avg_gapL
    print(f"\n  {prof}:")
    print(
        f"    n={len(pg)}  avgGapL={avg_gapL:.2f}%  avgGapP={avg_gapP:.2f}%  "
        f"avgGapZ={avg_gapZ:.2f}%  L_advantage={adv:+.2f}pp"
    )
    print(f"    L wins: {l_wins}  P wins: {p_wins}  Tied: {ties}")
    print(f"    L optimal: {l_opt}/{len(pg)}  P optimal: {p_opt}/{len(pg)}")

# ── Per-model breakdown ──────────────────────────────────────────
print("\n### PER-MODEL BREAKDOWN ###")
models = sorted(set(g["model"] for g in all_gaps))
for mod in models:
    mg = [g for g in all_gaps if g["model"] == mod]
    if not mg:
        continue
    avg_gapL = statistics.mean(g["gapL"] for g in mg)
    avg_gapP = statistics.mean(g["gapP"] for g in mg)
    l_wins = sum(1 for g in mg if g["gapL"] < g["gapP"])
    p_wins = sum(1 for g in mg if g["gapP"] < g["gapL"])
    adv = avg_gapP - avg_gapL
    print(
        f"  {mod:15s}: n={len(mg):4d}  avgGapL={avg_gapL:.2f}%  avgGapP={avg_gapP:.2f}%  "
        f"L_adv={adv:+.2f}pp  L_wins={l_wins}  P_wins={p_wins}"
    )

# ── Per-beam breakdown ───────────────────────────────────────────
print("\n### PER-BEAM BREAKDOWN ###")
beams = sorted(set(g["beam"] for g in all_gaps))
for b in beams:
    bg = [g for g in all_gaps if g["beam"] == b]
    avg_gapL = statistics.mean(g["gapL"] for g in bg)
    avg_gapP = statistics.mean(g["gapP"] for g in bg)
    l_wins = sum(1 for g in bg if g["gapL"] < g["gapP"])
    p_wins = sum(1 for g in bg if g["gapP"] < g["gapL"])
    adv = avg_gapP - avg_gapL
    print(
        f"  beam={b:3d}: n={len(bg):4d}  avgGapL={avg_gapL:.2f}%  avgGapP={avg_gapP:.2f}%  "
        f"L_adv={adv:+.2f}pp  L_wins={l_wins}  P_wins={p_wins}"
    )

# ── WORST CASES: where P beats L the most ───────────────────────
print("\n### TOP 20 WORST CASES: P beats L by most margin ###")
p_better_sorted = sorted(p_better, key=lambda g: g["gapL"] - g["gapP"], reverse=True)
for i, g in enumerate(p_better_sorted[:20]):
    diff = g["gapL"] - g["gapP"]
    print(
        f"  {i+1}. {g['config']:30s} seed={g['seed']} beam={g['beam']} "
        f"gapL={g['gapL']:.2f}% gapP={g['gapP']:.2f}% (P better by {diff:.2f}pp) "
        f"[{g['desc']}]"
    )

# ── BEST CASES: where L beats P the most ────────────────────────
print("\n### TOP 20 BEST CASES: L beats P by most margin ###")
l_better_sorted = sorted(l_better, key=lambda g: g["gapP"] - g["gapL"], reverse=True)
for i, g in enumerate(l_better_sorted[:20]):
    diff = g["gapP"] - g["gapL"]
    print(
        f"  {i+1}. {g['config']:30s} seed={g['seed']} beam={g['beam']} "
        f"gapL={g['gapL']:.2f}% gapP={g['gapP']:.2f}% (L better by {diff:.2f}pp) "
        f"[{g['desc']}]"
    )

# ── Cross-size: small vs medium eval ─────────────────────────────
print("\n### CROSS-SIZE GENERALIZATION ###")
for phase in ["B-hard"]:
    prs = phase_runs.get(phase, [])
    small_runs = [r for r in prs if "small" in r["desc"].lower()]
    medium_runs = [r for r in prs if "medium" in r["desc"].lower()]

    if small_runs:
        sg = [g for r in small_runs for g in r["gaps"]]
        avg_gapL_s = statistics.mean(g["gapL"] for g in sg) if sg else 0
        avg_gapP_s = statistics.mean(g["gapP"] for g in sg) if sg else 0
        print(
            f"  Small eval:  n={len(sg):4d}  avgGapL={avg_gapL_s:.2f}%  avgGapP={avg_gapP_s:.2f}%"
        )
    if medium_runs:
        mg = [r for r in medium_runs for g in r["gaps"]]
        mg_gaps = [g for r in medium_runs for g in r["gaps"]]
        avg_gapL_m = statistics.mean(g["gapL"] for g in mg_gaps) if mg_gaps else 0
        avg_gapP_m = statistics.mean(g["gapP"] for g in mg_gaps) if mg_gaps else 0
        print(
            f"  Medium eval: n={len(mg_gaps):4d}  avgGapL={avg_gapL_m:.2f}%  avgGapP={avg_gapP_m:.2f}%"
        )

# ── Per-profile + per-model detailed (L vs P) ───────────────────
print("\n### PROFILE × MODEL MATRIX (L advantage in pp) ###")
header = f"  {'Profile':<20s}"
for mod in models:
    header += f" {mod:>12s}"
print(header)
for prof in profiles:
    row = f"  {prof:<20s}"
    for mod in models:
        pg = [g for g in all_gaps if g["profile"] == prof and g["model"] == mod]
        if pg:
            avg_L = statistics.mean(g["gapL"] for g in pg)
            avg_P = statistics.mean(g["gapP"] for g in pg)
            adv = avg_P - avg_L
            row += f" {adv:+11.2f}p"
        else:
            row += f" {'—':>12s}"
    print(row)

# ── Eval size breakdown per profile ──────────────────────────────
print("\n### EVAL SIZE × PROFILE (B-hard only, L advantage in pp) ###")
for desc_filter, label in [("small", "Small"), ("medium", "Medium")]:
    print(f"\n  {label} eval:")
    for prof in profiles:
        gaps_f = [
            g
            for r in phase_runs.get("B-hard", [])
            if desc_filter in r["desc"].lower()
            for g in r["gaps"]
            if g["profile"] == prof
        ]
        if gaps_f:
            avg_L = statistics.mean(g["gapL"] for g in gaps_f)
            avg_P = statistics.mean(g["gapP"] for g in gaps_f)
            avg_Z = statistics.mean(g["gapZ"] for g in gaps_f)
            adv = avg_P - avg_L
            p_wins_here = sum(1 for g in gaps_f if g["gapP"] < g["gapL"])
            l_wins_here = sum(1 for g in gaps_f if g["gapL"] < g["gapP"])
            print(
                f"    {prof:<20s} n={len(gaps_f):3d}  gapL={avg_L:.2f}%  gapP={avg_P:.2f}%  "
                f"gapZ={avg_Z:.2f}%  L_adv={adv:+.2f}pp  L_wins={l_wins_here} P_wins={p_wins_here}"
            )

# ── R² summary ───────────────────────────────────────────────────
print("\n### R² TRAINING QUALITY ###")
for r in runs:
    if r["r2"]:
        print(f"  {r['config']:30s} R²={r['r2'][-1]:.4f}")

# ── Interesting stats ────────────────────────────────────────────
print("\n### DIFFICULTY DISTRIBUTION ###")
exact_costs = [g["exact"] for g in all_gaps]
if exact_costs:
    print(f"  Exact cost range: [{min(exact_costs):.1f}, {max(exact_costs):.1f}]")
    print(f"  Mean exact cost: {statistics.mean(exact_costs):.1f}")
    zero_gap_l = sum(1 for g in all_gaps if g["gapL"] == 0.0)
    zero_gap_p = sum(1 for g in all_gaps if g["gapP"] == 0.0)
    print(
        f"  Zero-gap L: {zero_gap_l}/{len(all_gaps)} ({100*zero_gap_l/len(all_gaps):.1f}%)"
    )
    print(
        f"  Zero-gap P: {zero_gap_p}/{len(all_gaps)} ({100*zero_gap_p/len(all_gaps):.1f}%)"
    )

    # High-gap instances
    high_gap_l = [g for g in all_gaps if g["gapL"] > 5.0]
    high_gap_p = [g for g in all_gaps if g["gapP"] > 5.0]
    print(f"  gapL > 5%: {len(high_gap_l)}")
    print(f"  gapP > 5%: {len(high_gap_p)}")

print("\n" + "=" * 80)
print("END OF ANALYSIS")
print("=" * 80)

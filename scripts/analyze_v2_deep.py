#!/usr/bin/env python3
"""Analyze Experiment H and two_block failure modes from V2 log."""
import re

LOG = "/Users/mac/Documents/Study/PFE/PaST/ADP/logs/logs/rigourous/complete_new.log"
lines = open(LOG, encoding="utf-8", errors="replace").readlines()

gap_re = re.compile(
    r"seed=(\d+)\s+beam=(\d+)\s+exact=([\d.]+)\s+L=([\d.]+)\s+Z=([\d.]+)\s+P=([\d.]+)\s+"
    r"gapL=([\d.]+)%\s+gapZ=([\d.]+)%\s+gapP=([\d.]+)%"
)

sections = {
    "H1-mlp": (7487, 7991),
    "H1-elasticnet": (7992, 8337),
    "H1-poly": (8338, 8683),
    "H2-ramp/mlp": (8686, 8810),
    "H2-ramp/elasticnet": (8811, 8935),
    "H2-ramp/poly": (8936, 9060),
    "H2-dpeak/mlp": (9061, 9185),
    "H2-dpeak/elasticnet": (9186, 9310),
    "H2-dpeak/poly": (9311, 9435),
    "H2-dtou/mlp": (9436, 9560),
    "H2-dtou/elasticnet": (9561, 9685),
    "H2-dtou/poly": (9686, 9812),
}

print("=== EXPERIMENT H RESULTS ===\n")
h1_all, h2_all = [], []
for name, (start, end) in sections.items():
    gaps = []
    for i in range(start - 1, min(end, len(lines))):
        m = gap_re.search(lines[i])
        if m:
            seed, beam, exact, lval, zval, pval, gapL, gapZ, gapP = m.groups()
            gaps.append((float(gapL), float(gapP), float(gapZ)))
    if not gaps:
        print(f"  {name:30s}: NO GAP DATA")
        continue
    n = len(gaps)
    avgL = sum(g[0] for g in gaps) / n
    avgP = sum(g[1] for g in gaps) / n
    avgZ = sum(g[2] for g in gaps) / n
    l_wins = sum(1 for g in gaps if g[0] < g[1])
    p_wins = sum(1 for g in gaps if g[1] < g[0])
    ties = sum(1 for g in gaps if g[0] == g[1])
    adv = avgP - avgL
    print(
        f"  {name:30s}: n={n:3d}  gapL={avgL:.2f}%  gapP={avgP:.2f}%  gapZ={avgZ:.2f}%  "
        f"L_adv={adv:+.2f}pp  Lwins={l_wins} Pwins={p_wins} T={ties}"
    )
    if name.startswith("H1"):
        h1_all.extend(gaps)
    else:
        h2_all.extend(gaps)

if h1_all:
    n = len(h1_all)
    avgL = sum(g[0] for g in h1_all) / n
    avgP = sum(g[1] for g in h1_all) / n
    print(
        f"\n  H1 OVERALL (trained on NR):  n={n:3d}  gapL={avgL:.2f}%  gapP={avgP:.2f}%  L_adv={avgP-avgL:+.2f}pp"
    )
if h2_all:
    n = len(h2_all)
    avgL = sum(g[0] for g in h2_all) / n
    avgP = sum(g[1] for g in h2_all) / n
    print(
        f"  H2 OVERALL (zero-shot):     n={n:3d}  gapL={avgL:.2f}%  gapP={avgP:.2f}%  L_adv={avgP-avgL:+.2f}pp"
    )

# ── two_block_elasticnet medium analysis ──
print("\n\n=== TWO_BLOCK_ELASTICNET: MEDIUM EVAL ANALYSIS ===")
# Collect ALL gap lines from two_block_elasticnet runs (multiple medium evals at line ~7386+)
# More robust: find all gap lines that belong to two_block_elasticnet medium runs
in_target = False
target_gaps = []
for i, line in enumerate(lines):
    if (
        "[15/15]" in line
        and "medium" in line.lower()
        and "two_block_elasticnet" in line
    ):
        in_target = True
        continue
    if in_target and ("[" in line and "/15]" in line and "two_block" not in line):
        in_target = False
    if in_target:
        m = gap_re.search(line)
        if m:
            seed, beam, exact, lval, zval, pval, gapL, gapZ, gapP = m.groups()
            target_gaps.append(
                {
                    "seed": int(seed),
                    "beam": int(beam),
                    "exact": float(exact),
                    "L": float(lval),
                    "P": float(pval),
                    "gapL": float(gapL),
                    "gapP": float(gapP),
                    "gapZ": float(gapZ),
                    "line": i + 1,
                }
            )

p_worse = [g for g in target_gaps if g["gapL"] > g["gapP"]]
l_worse = [g for g in target_gaps if g["gapP"] > g["gapL"]]
print(f"  Total gap measures: {len(target_gaps)}")
print(f"  L strictly better: {len(l_worse)}")
print(f"  P strictly better: {len(p_worse)}")
if p_worse:
    print(f"\n  TOP 15 WORST (L underperforms P):")
    for g in sorted(p_worse, key=lambda x: x["gapL"] - x["gapP"], reverse=True)[:15]:
        print(
            f"    seed={g['seed']} beam={g['beam']} exact={g['exact']:.0f} "
            f"L={g['L']:.0f}(gap={g['gapL']:.2f}%) P={g['P']:.0f}(gap={g['gapP']:.2f}%)  "
            f"diff={g['gapL']-g['gapP']:.2f}pp  line={g['line']}"
        )

# ── Check: seed 204 instance properties ──
print("\n=== SEED 204 ACROSS ALL PROFILES (looking for pattern) ===")
seed204 = []
for i, line in enumerate(lines):
    if "seed=204" in line:
        m = gap_re.search(line)
        if m:
            seed, beam, exact, lval, zval, pval, gapL, gapZ, gapP = m.groups()
            seed204.append(
                {
                    "beam": int(beam),
                    "exact": float(exact),
                    "L": float(lval),
                    "P": float(pval),
                    "Z": float(zval),
                    "gapL": float(gapL),
                    "gapP": float(gapP),
                    "gapZ": float(gapZ),
                    "line": i + 1,
                }
            )
for g in seed204:
    marker = " <<<P BETTER" if g["gapL"] > g["gapP"] else ""
    print(
        f"  line={g['line']} beam={g['beam']} exact={g['exact']:.0f} "
        f"L={g['L']:.0f}(gap={g['gapL']:.2f}%) P={g['P']:.0f}(gap={g['gapP']:.2f}%) "
        f"Z={g['Z']:.0f}(gap={g['gapZ']:.2f}%){marker}"
    )

# ── Distribution of gapL across two_block vs other profiles in medium eval ──
print("\n=== MEDIUM EVAL: PROFILE COMPARISON (B-hard) ===")
# Lines ~1266 to ~7473 = B-hard
bhard_medium_gaps = {}
current_config = None
for i, line in enumerate(lines[:7473]):
    run_m = re.search(r"\[\d+/15\]\s+EVAL medium.*?:\s+(\S+)", line)
    if run_m:
        current_config = run_m.group(1)
    if current_config and "medium" in lines[max(0, i - 10) : i + 1].__repr__().lower():
        m = gap_re.search(line)
        if m:
            seed, beam, exact, lval, zval, pval, gapL, gapZ, gapP = m.groups()
            bhard_medium_gaps.setdefault(current_config, []).append(
                (float(gapL), float(gapP), float(gapZ))
            )

for cfg in sorted(bhard_medium_gaps.keys()):
    gs = bhard_medium_gaps[cfg]
    n = len(gs)
    avgL = sum(g[0] for g in gs) / n
    avgP = sum(g[1] for g in gs) / n
    avgZ = sum(g[2] for g in gs) / n
    p_better = sum(1 for g in gs if g[1] < g[0])
    print(
        f"  {cfg:30s}: n={n:3d}  gapL={avgL:.2f}%  gapP={avgP:.2f}%  gapZ={avgZ:.2f}%  Pwins={p_better}"
    )

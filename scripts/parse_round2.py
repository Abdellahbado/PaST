#!/usr/bin/env python3
"""Parse round_2.log and produce clean tabular results.

Usage:  python3 scripts/parse_round2.py [logfile]
Default: ADP/logs/logs/rigourous/round_2.log
"""
import re, sys, os
from collections import defaultdict, OrderedDict

log_path = sys.argv[1] if len(sys.argv) > 1 else "ADP/logs/logs/rigourous/round_2.log"

if not os.path.isfile(log_path):
    print(f"ERROR: {log_path} not found"); sys.exit(1)

with open(log_path, 'rb') as f:
    raw = f.read()
text = raw.decode('utf-8', errors='replace')
lines = text.split('\n')

# ── Tag patterns ─────────────────────────────────────────────────────
TAG_RE = re.compile(r'(?:RUN|CROSS|EVAL)\s+(\S+)')

current_exp = None
skipped = []
data_info = OrderedDict()       # tag -> (samples, r2_train, r2_test)
eval_rows = []                  # (tag, beam, n, gapL_m, gapL_md, gapZ_m, gapZ_md, gapP_m, gapP_md, speed)
last_samples = 0

for line in lines:
    line = line.strip()
    if not line:
        continue

    # New experiment tag
    m = TAG_RE.search(line)
    if m:
        tag = m.group(1)
        # Clean up mangled tags (interleaved output)
        tag = re.sub(r'[=:].*', '', tag).strip()
        if tag:
            current_exp = tag
            last_samples = 0

    # SKIP
    if 'SKIP (exists)' in line:
        m_skip = re.search(r'SKIP \(exists\)\s+(\S+)', line)
        if m_skip:
            csv = m_skip.group(1)
            tag = os.path.splitext(os.path.basename(csv))[0]
            skipped.append(tag)

    # Pooled samples
    m_pool = re.search(r'Pooled samples:\s*(\d[\d,]*)', line)
    if m_pool and current_exp:
        last_samples = int(m_pool.group(1).replace(',', ''))

    # R2 scores
    m_r2 = re.search(r'R2_train=([\d.]+)\s+R2_test=([\d.]+)', line)
    if m_r2 and current_exp:
        data_info[current_exp] = (last_samples, float(m_r2.group(1)), float(m_r2.group(2)))

    # Evaluation beam lines
    m_beam = re.search(
        r'beam=\s*(\d+)\s+n=\s*(\d+)\s+'
        r'gapL=\s*([\d.]+)%/\s*([\d.]+)%\s+'
        r'gapZ=\s*([\d.]+)%/\s*([\d.]+)%\s+'
        r'gapP=\s*([\d.]+)%/\s*([\d.]+)%\s+'
        r'speedL=([\d.]+)x',
        line
    )
    if m_beam and current_exp:
        eval_rows.append((
            current_exp,
            int(m_beam.group(1)), int(m_beam.group(2)),
            float(m_beam.group(3)), float(m_beam.group(4)),
            float(m_beam.group(5)), float(m_beam.group(6)),
            float(m_beam.group(7)), float(m_beam.group(8)),
            float(m_beam.group(9)),
        ))

# ── Group by experiment tag ──────────────────────────────────────────
exp_beams = defaultdict(list)
for row in eval_rows:
    exp_beams[row[0]].append(row[1:])

# ── Output ───────────────────────────────────────────────────────────
print()
print("=" * 80)
print(f"  SKIPPED experiments ({len(skipped)}) — STALE DATA from previous bad run")
print("=" * 80)
for s in sorted(skipped):
    print(f"  {s}")

print()
print("=" * 90)
print("  TRAINING DATA SUMMARY (newly trained only)")
print("=" * 90)
print(f"  {'Experiment':<48} {'Samples':>8} {'R2_train':>9} {'R2_test':>9}")
print("  " + "-" * 86)
for tag, (samp, r2tr, r2te) in data_info.items():
    print(f"  {tag:<48} {samp:>8,} {r2tr:>9.4f} {r2te:>9.4f}")

for beam_w in [20, 10, 5]:
    rows_for_beam = []
    for tag in dict.fromkeys(r[0] for r in eval_rows):
        for bd in exp_beams[tag]:
            if bd[0] == beam_w:
                rows_for_beam.append((tag, *bd))
    if not rows_for_beam:
        continue
    print()
    print("=" * 130)
    print(f"  EVALUATION @ beam={beam_w}   gapL=Learned  gapP=PriceHeuristic  gapZ=ZeroVhat  (mean/median)")
    print("=" * 130)
    print(f"  {'Experiment':<48} {'n':>3} {'gapL_m':>7} {'gapL_md':>8} {'gapP_m':>7} {'gapP_md':>8} {'gapZ_m':>7} {'gapZ_md':>8} {'speed':>6}")
    print("  " + "-" * 126)
    for tag, beam, n, gl_m, gl_md, gz_m, gz_md, gp_m, gp_md, sp in rows_for_beam:
        print(f"  {tag:<48} {n:>3} {gl_m:>6.2f}% {gl_md:>6.2f}% {gp_m:>6.2f}% {gp_md:>6.2f}% {gz_m:>6.2f}% {gz_md:>6.2f}% {sp:>5.1f}x")

print()
print("=" * 80)
print("  SAMPLES PER INSTANCE (trained experiments)")
print("=" * 80)
for tag, (samp, r2tr, r2te) in data_info.items():
    per_inst = samp / 300 if samp > 0 else 0
    flag = " *** DATA-STARVED ***" if per_inst < 500 else ""
    print(f"  {tag:<48} {samp:>8,}  ({per_inst:>6.0f}/inst){flag}")

#!/usr/bin/env python3
"""Re-run PLAN27 variants that were broken (late_ambig, residual_aware, late_residual_ambig)."""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from run_plan27_gate_a import VARIANTS, run_row, append_csv_row, RAW_CSV

families = [
    ("hardA_k10", 1000, [0, 1, 2, 3]),
    ("hardB_k10", 1000, [0, 1, 2, 3]),
]

time_limit = 1200
broken_variants = ['late_ambig', 'residual_aware', 'late_residual_ambig']

# Read already completed
completed = set()
if RAW_CSV.exists():
    with open(RAW_CSV) as f:
        reader = csv.reader(f)
        for parts in reader:
            if len(parts) >= 5:
                completed.add((parts[0], int(parts[3]), parts[4]))

print(f"Already completed: {len(completed)} rows")

# Delete broken variant rows from CSV
rows_to_keep = []
if RAW_CSV.exists():
    with open(RAW_CSV) as f:
        reader = csv.reader(f)
        for parts in reader:
            if len(parts) >= 5 and parts[4] in broken_variants:
                continue
            rows_to_keep.append(parts)

with open(RAW_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rows_to_keep)

print(f"Kept {len(rows_to_keep)} rows, removed broken variants")

remaining = []
for family_id, n_jobs, seeds in families:
    for seed in seeds:
        for variant_label in broken_variants:
            remaining.append((family_id, n_jobs, seed, variant_label))

print(f"Re-running: {len(remaining)} rows")

for i, (family_id, n_jobs, seed, variant_label) in enumerate(remaining):
    print(f"[{i+1}/{len(remaining)}] Running {family_id} seed={seed} variant={variant_label}")
    env_overrides = VARIANTS[variant_label]
    row = run_row(family_id, n_jobs, seed, time_limit, variant_label, env_overrides)
    append_csv_row(row, RAW_CSV)
    print(f"  -> runtime={row.get('runtime_sec', 'N/A')} gap={row.get('gap_pct', 'N/A')} deciding={row.get('deciding_step', 'N/A')} rss={row.get('peak_rss_gb', 'N/A')}GB")

print(f"Done. Raw CSV: {RAW_CSV}")

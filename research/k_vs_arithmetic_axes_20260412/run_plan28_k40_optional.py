#!/usr/bin/env python3
"""PLAN30 optional K=40 run."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from run_plan28_easy_k_scaling import run_row, append_csv_row, RAW_CSV, DENSE_STEP2_ENV

families = [
    ("easy_k40_unit", list(range(1, 41)), [0, 1]),
]

time_limit = 1800

for family_id, sizes, seeds in families:
    for seed in seeds:
        for variant_label, env_overrides in [("baseline", {}), ("dense_step2_fastpath", DENSE_STEP2_ENV)]:
            print(f"[{__import__('time').strftime('%H:%M:%S')}] Running {family_id} seed={seed} variant={variant_label}")
            row = run_row(family_id, sizes, 1000, seed, time_limit, variant_label, env_overrides)
            append_csv_row(row, RAW_CSV)
            print(f"  -> runtime={row.get('runtime_sec', 'N/A')} gap={row.get('gap_pct', 'N/A')} deciding={row.get('deciding_step', 'N/A')} rss={row.get('peak_rss_gb', 'N/A')}GB optimal={row.get('is_optimal', 'N/A')}")

print("Done.")

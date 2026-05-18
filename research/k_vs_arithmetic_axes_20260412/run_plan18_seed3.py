#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

from run_plan18_k_boundary_refine_n1000 import (
    HARD_B_BASE, load_raw, write_csv, RAW_CSV, N_JOBS, LAMBDA, TIME_LIMIT, MAX_RSS_GB,
    IRREGULAR_REROUTE_ENV, build_plan18_payload, normalize_output_row
)
from run_plan13_two_track_recovery import run_row

rows = load_raw(RAW_CSV)
print(f'Current rows: {len(rows)}')
sizes = HARD_B_BASE[:12]
fid = 'hardB_k12'
label = '{' + ','.join(str(x) for x in sizes) + '}'
seed = 3
print(f'Running {fid} seed={seed}...')
payload = build_plan18_payload(fid, sizes, N_JOBS, LAMBDA, seed)
raw = run_row(fid, N_JOBS, seed, TIME_LIMIT, 'irregular_reroute',
              dict(IRREGULAR_REROUTE_ENV), max_rss_gb=MAX_RSS_GB, payload=payload)
row = normalize_output_row(
    raw, family_id=fid, family_label=label, family_class='hard_irregular_B',
    family_sizes=sizes, k=12, variant_label='irregular_reroute',
    route_policy='plan18:additive_profile_repair_beam_auto_v1'
)
rows.append(row)
write_csv(RAW_CSV, rows)
print(f'Done: step={row.get("deciding_step")}, opt={row.get("is_optimal")}, rt={row.get("runtime_sec")}, rc={row.get("solver_returncode")}')
print(f'Total rows: {len(rows)}')

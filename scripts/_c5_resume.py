#!/usr/bin/env python3
"""Resume C5 EHS evaluation from partial checkpoint."""

import csv, json, os, sys, time, random
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path.cwd()))
from glns.paper_heuristics import run_ehs

INST_DIR = Path('research/glns_llm_heuristic_20260422/iterations/20260510_phaseC_adversarial_ehs_benchmark_design/generated_instances/c5_validation')
EVAL_DIR = Path('research/glns_llm_heuristic_20260422/iterations/20260510_phaseC_adversarial_ehs_benchmark_design/eval')
RAW_CSV = EVAL_DIR / 'c5_raw_runs.csv'

BUDGETS = {'short': 30, 'medium': 120, 'long': 300}

# Read already completed instance IDs
done_ids = set()
if RAW_CSV.exists():
    with open(RAW_CSV) as f:
        for row in csv.DictReader(f):
            done_ids.add(row['instance_id'])

print(f'Already done: {len(done_ids)} instances')

# Find remaining instances
inst_files = sorted(INST_DIR.glob('c5_*.json'))
remaining = [f for f in inst_files if f.stem not in done_ids]
print(f'Remaining: {len(remaining)} instances')

if not remaining:
    print('All done!')
    sys.exit(0)

# Load existing rows
existing_rows = []
if RAW_CSV.exists():
    with open(RAW_CSV) as f:
        existing_rows = list(csv.DictReader(f))
    print(f'Loaded {len(existing_rows)} existing rows')

t_start = time.time()

def compute_hv(points, ref_point):
    if not points: return 0.0
    sorted_pts = sorted(points, key=lambda x: x[0])
    hv = 0.0
    prev_cmax = 0
    prev_energy = ref_point[1]
    for cmax, energy in sorted_pts:
        if cmax >= ref_point[0] or energy >= ref_point[1]: continue
        width = cmax - prev_cmax
        if width > 0 and energy < prev_energy:
            hv += width * (ref_point[1] - energy)
        if energy < prev_energy: prev_energy = energy
        prev_cmax = cmax
    width = ref_point[0] - prev_cmax
    if width > 0: hv += width * (ref_point[1] - prev_energy)
    return hv

all_rows = list(existing_rows)

for i, fp in enumerate(remaining):
    with open(fp) as f: inst = json.load(f)
    meta = inst['metadata']
    base = {
        'instance_id': inst['instance_id'],
        'arm': meta['arm'], 'family_name': meta['family_name'],
        'expected_mechanism': meta['expected_mechanism'],
        'n': inst['n'], 'm': inst['m'], 'T': inst['T'],
        'sum_p': meta['sum_p'], 'feasible_instance': meta['feasible'],
    }

    if not meta['feasible']:
        for bn, bs in BUDGETS.items():
            all_rows.append({**base, 'budget': bn, 'budget_s': bs, 'feasible_ehs': False,
                'front_size': 0, 'runtime_s': 0.0, 'timeout': True,
                'deepest_cmax': inst['T'], 'best_energy': 'inf', 'shallowest_cmax': inst['T'],
                'hypervolume': 0.0, 'ref_cmax': inst['T']+1, 'ref_energy': 100000.0, 'error': 'infeasible'})
        continue

    # Evaluate at all 3 budgets
    budget_results = {}
    all_pts = []
    for bn in ['short', 'medium', 'long']:
        bs = BUDGETS[bn]
        t0 = time.time()
        rng = random.Random(meta['seed'] + hash(bn) % 10000)
        try:
            results = run_ehs(inst, rng=rng, time_limit_seconds=bs,
                             use_history=True, use_exchange=True, use_esr=True)
            rt = time.time() - t0
            pts = [(s.cmax, s.energy) for s in results]
            budget_results[bn] = {
                'ok': True, 'fs': len(results), 'rt': round(rt, 2),
                'to': rt >= bs * 0.95,
                'dc': max((s.cmax for s in results), default=inst['T']),
                'be': min((s.energy for s in results), default=float('inf')),
                'sc': min((s.cmax for s in results), default=inst['T']),
                'pts': pts, 'err': '',
            }
            all_pts.extend(pts)
        except Exception as e:
            budget_results[bn] = {
                'ok': False, 'fs': 0, 'rt': 0.0, 'to': True,
                'dc': inst['T'], 'be': 'inf', 'sc': inst['T'],
                'pts': [], 'err': str(e)[:200],
            }

    # Compute shared reference point for HV
    if all_pts:
        ref_cmax = max(p[0] for p in all_pts) + 1
        ref_energy = max(p[1] for p in all_pts) * 1.1 + 1.0
    else:
        ref_cmax = inst['T'] + 1
        ref_energy = 100000.0

    for bn in ['short', 'medium', 'long']:
        r = budget_results[bn]
        bs = BUDGETS[bn]
        hv = compute_hv(r['pts'], (ref_cmax, ref_energy))
        all_rows.append({**base, 'budget': bn, 'budget_s': bs,
            'feasible_ehs': r['ok'], 'front_size': r['fs'],
            'runtime_s': r['rt'], 'timeout': r['to'],
            'deepest_cmax': r['dc'], 'best_energy': round(r['be'], 6) if r['be'] != float('inf') else 'inf',
            'shallowest_cmax': r['sc'],
            'hypervolume': round(hv, 6),
            'ref_cmax': ref_cmax, 'ref_energy': round(ref_energy, 6),
            'error': r['err']})

    # Progress
    done = len(done_ids) + i + 1
    total = len(inst_files)
    elapsed = time.time() - t_start
    eta = (elapsed / (i + 1)) * (len(remaining) - i - 1) if i < len(remaining) - 1 else 0
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {done}/{total} "
          f"({done/total*100:.0f}%) elapsed={elapsed:.0f}s ETA={eta:.0f}s "
          f"last={fp.name[:55]}")

    # Save checkpoint every 3 instances
    if (i + 1) % 3 == 0 and all_rows:
        with open(RAW_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)

# Final save
with open(RAW_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
    w.writeheader()
    w.writerows(all_rows)

total_time = time.time() - t_start
print(f'\n[{datetime.now().strftime("%H:%M:%S")}] Done: {len(all_rows)} total rows ({len(remaining)} new instances) in {total_time:.0f}s ({total_time/3600:.1f}h)')
print(f'Saved: {RAW_CSV}')

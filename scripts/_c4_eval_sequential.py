#!/usr/bin/env python3
"""C4 EHS evaluation — sequential with progress."""

import csv, json, os, sys, time, random
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ITER_DIR = PROJECT_ROOT / 'research/glns_llm_heuristic_20260422/iterations/20260510_phaseC_adversarial_ehs_benchmark_design'
EVAL_DIR = ITER_DIR / 'eval'
INST_DIR = ITER_DIR / 'generated_instances' / 'c4_validation'
os.makedirs(EVAL_DIR, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))
from glns.paper_heuristics import run_ehs

SHORT_BUDGET = 30
LONG_BUDGET = 90

def eval_one_instance(inst_path):
    with open(inst_path) as f:
        inst = json.load(f)
    meta = inst['metadata']
    base = {
        'instance_id': inst['instance_id'],
        'arm': meta['arm'],
        'family_name': meta['family_name'],
        'expected_mechanism': meta['expected_mechanism'],
        'n': inst['n'],
        'm': inst['m'],
        'T': inst['T'],
        'sum_p': meta['sum_p'],
        'feasible_instance': meta['feasible'],
    }
    rows = []
    for budget_name, budget_s in [('short', SHORT_BUDGET), ('long', LONG_BUDGET)]:
        t0 = time.time()
        if not meta['feasible']:
            rows.append({**base, 'budget': budget_name, 'feasible_ehs': False, 'front_size': 0,
                        'runtime_s': 0.0, 'timeout': True, 'deepest_cmax': inst['T'],
                        'best_energy': 'inf', 'shallowest_cmax': inst['T'],
                        'error': 'infeasible_instance'})
            continue
        try:
            rng = random.Random(meta['seed'] + hash(str(budget_s)) % 10000)
            results = run_ehs(inst, rng=rng, time_limit_seconds=budget_s,
                             use_history=True, use_exchange=True, use_esr=True)
            runtime = time.time() - t0
            front_size = len(results)
            deepest_cmax = max((s.cmax for s in results), default=inst['T'])
            best_energy = min((s.energy for s in results), default=float('inf'))
            shallowest_cmax = min((s.cmax for s in results), default=inst['T'])
            timed_out = runtime >= budget_s * 0.95
            rows.append({**base, 'budget': budget_name, 'feasible_ehs': True,
                        'front_size': front_size, 'runtime_s': round(runtime, 2),
                        'timeout': timed_out, 'deepest_cmax': deepest_cmax,
                        'best_energy': round(best_energy, 6) if best_energy != float('inf') else 'inf',
                        'shallowest_cmax': shallowest_cmax, 'error': ''})
        except Exception as e:
            rows.append({**base, 'budget': budget_name, 'feasible_ehs': False, 'front_size': 0,
                        'runtime_s': 0.0, 'timeout': True, 'deepest_cmax': inst['T'],
                        'best_energy': 'inf', 'shallowest_cmax': inst['T'],
                        'error': str(e)[:200]})
    return rows

def main():
    inst_files = sorted(INST_DIR.glob('c4v_*.json'))
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Found {len(inst_files)} instances")
    print(f"Budgets: {SHORT_BUDGET}s / {LONG_BUDGET}s")
    est = len(inst_files) * (SHORT_BUDGET + LONG_BUDGET)
    print(f"Estimated sequential: {est}s ({est/60:.0f}min)")
    print()

    all_rows = []
    t_start = time.time()

    for i, fp in enumerate(inst_files):
        rows = eval_one_instance(fp)
        all_rows.extend(rows)
        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t_start
            done = i + 1
            eta = (elapsed / done) * (len(inst_files) - done) if done > 0 else 0
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] {done}/{len(inst_files)} "
                  f"({done/len(inst_files)*100:.0f}%) elapsed={elapsed:.0f}s ETA={eta:.0f}s "
                  f"last={fp.name[:50]}")

        # Periodic save every 10 instances
        if (i + 1) % 10 == 0 and all_rows:
            with open(EVAL_DIR / 'c4_validation_raw.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                writer.writeheader()
                writer.writerows(all_rows)
            print(f"  [checkpoint] saved {len(all_rows)} rows")

    # Final save
    with open(EVAL_DIR / 'c4_validation_raw.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    total_time = time.time() - t_start
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done: {len(all_rows)} runs in {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"Saved: eval/c4_validation_raw.csv")

if __name__ == '__main__':
    main()

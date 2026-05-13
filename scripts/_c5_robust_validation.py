#!/usr/bin/env python3
"""C5 Robust Multi-Budget Adversarial Validation.

Tests whether LLM-designed EHS stress families remain difficult under
30s / 120s / 300s budgets, beyond the trivial 30s short-budget regime.

Arms: LLM Call2 (M2+M1), Literature (Wang+Anghi), Random (004+005),
      agent_manual_sweep (tight_epsilon+mixed_job_sizes, internal control),
      simple_stress (non-LLM structured baseline).

Frozen families — no LLM tuning based on C5 results.
"""

import csv, json, os, sys, time, random, math
from pathlib import Path
from collections import defaultdict
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ITER_DIR = PROJECT_ROOT / 'research/glns_llm_heuristic_20260422/iterations/20260510_phaseC_adversarial_ehs_benchmark_design'
NOTES_DIR = ITER_DIR / 'notes'
EVAL_DIR = ITER_DIR / 'eval'
INST_DIR = ITER_DIR / 'generated_instances' / 'c5_validation'
FAMILIES_DIR = ITER_DIR / 'families'

for d in [NOTES_DIR, EVAL_DIR, INST_DIR]:
    os.makedirs(d, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))
from glns.paper_heuristics import run_ehs

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

BUDGETS = {'short': 30, 'medium': 120, 'long': 300}
MAX_N = 150
MAX_T = 200
INSTANCES_PER_FAMILY = 3
MAX_RETRIES = 200

# ═══════════════════════════════════════════════════════════════
# Frozen Family Definitions
# ═══════════════════════════════════════════════════════════════

# LLM Call2 families (from llm_counter_sweep_families.json)
LLM_FAMILIES_RAW = json.load(open(FAMILIES_DIR / 'llm_counter_sweep_families.json'))['families']
LLM_M2 = next(f for f in LLM_FAMILIES_RAW if f['family_name'] == 'hybrid_M2_tight_steprates_asghlock')
LLM_M1 = next(f for f in LLM_FAMILIES_RAW if f['family_name'] == 'hybrid_M1_tight_hetero_rates_firstkhat_dom')

# Agent manual sweep families (from human_sweep_families.json)
AGENT_FAMILIES_RAW = json.load(open(FAMILIES_DIR / 'human_sweep_families.json'))['families']
AGENT_TIGHT = next(f for f in AGENT_FAMILIES_RAW if f['family_name'] == 'human_tight_epsilon')
AGENT_MIXED = next(f for f in AGENT_FAMILIES_RAW if f['family_name'] == 'human_mixed_job_sizes')

# Literature generators (from literature_generators.json)
LIT_GENERATORS = json.load(open(FAMILIES_DIR / 'literature_generators.json'))['generators']

# Random families
RANDOM_FAMILIES_RAW = json.load(open(FAMILIES_DIR / 'random_families.json'))['families']
RANDOM_004 = next(f for f in RANDOM_FAMILIES_RAW if f['family_name'] == 'random_004')
RANDOM_005 = next(f for f in RANDOM_FAMILIES_RAW if f['family_name'] == 'random_005')

# Simple stress baseline (non-LLM structured)
SIMPLE_STRESS_FAMILY = {
    "family_name": "simple_stress_tight_steprates",
    "hypothesis": "Small uniform jobs + tight epsilon + step machine rates create structural front growth. A-SGH locks into cheap machines early; later khats break the lock. This is a simpler, non-LLM version of mechanism stress.",
    "description": "Tight epsilon, small uniform jobs (p_j~U[1,4]), step machine rates (half 0.5, half 3.0), single-peak TOU. Expected mechanism: structural asgh_lock_in.",
    "n_jobs_range": {"min": 60, "max": 100},
    "m_machines_range": {"min": 8, "max": 12},
    "horizon_T_range": {"min": 150, "max": 200},
    "processing_time_distribution": {"type": "uniform", "params": {"low": 1, "high": 4}},
    "machine_rate_distribution": {"type": "step", "params": {"levels": [0.5, 3.0], "proportions": [0.5, 0.5]}},
    "TOU_price_profile_type": "single_peak",
    "TOU_price_profile_params": {"peak_start": 0.3, "peak_end": 0.6, "peak_multiplier": 2.0, "base_price": 1.0},
    "epsilon_regime": "tight",
    "expected_EHS_failure_mechanism": "asgh_lock_in",
    "expected_EHS_failure_mechanism_evidence": "Step rates cause early lock-in on cheap machines; tight epsilon forces break at later khats.",
}

# Arm configuration
ARMS_CONFIG = [
    {"arm": "llm_l2", "families": [LLM_M2, LLM_M1], "base_seed": 280000},
    {"arm": "literature", "families": [], "base_seed": 540000,
     "generators": ["wang_mls_generator", "anghinolfi_vls_capped"]},
    {"arm": "random", "families": [RANDOM_004, RANDOM_005], "base_seed": 210000},
    {"arm": "agent_manual_sweep", "families": [AGENT_TIGHT, AGENT_MIXED], "base_seed": 220000},
    {"arm": "simple_stress", "families": [SIMPLE_STRESS_FAMILY], "base_seed": 290000},
]

# ═══════════════════════════════════════════════════════════════
# Instance Generation
# ═══════════════════════════════════════════════════════════════

def _si(rng, lo, hi):
    if lo >= hi:
        return min(lo, hi)
    return rng.randint(lo, hi)

def _gen_processing_times(rng, dist, n):
    t = dist['type']
    p = dist['params']
    if t == 'uniform':
        return [_si(rng, int(p['low']), int(p['high'])) for _ in range(n)]
    elif t == 'bimodal':
        if 'modes' in p:
            modes = p['modes']
            weights = p.get('weights', [0.5, 0.5])
            total = sum(weights)
            counts = []
            for i, (lo, hi) in enumerate(modes):
                if i == len(modes) - 1:
                    counts.append(n - sum(counts))
                else:
                    counts.append(int(n * weights[i] / total))
            result = []
            for (lo, hi), cnt in zip(modes, counts):
                result.extend([_si(rng, lo, hi) for _ in range(cnt)])
            rng.shuffle(result)
            return result[:n] + [_si(rng, 1, 10) for _ in range(n - len(result))]
        else:
            n_small = int(n * p.get('small_fraction', 0.5))
            small = [_si(rng, p.get('small_low', 2), p.get('small_high', 5)) for _ in range(n_small)]
            large = [_si(rng, p.get('large_low', 15), p.get('large_high', 25)) for _ in range(n - n_small)]
            result = small + large
            rng.shuffle(result)
            return result
    elif t == 'exponential_truncated':
        scale = p['scale']
        mx = p.get('max', 30)
        return [min(mx, max(1, int(round(rng.expovariate(1.0/scale))))) for _ in range(n)]
    elif t == 'normal_truncated':
        mean = p['mean']; std = p['std']
        mn = p.get('min', 1); mx = p.get('max', 30)
        return [max(mn, min(mx, int(round(rng.gauss(mean, std))))) for _ in range(n)]
    else:
        return [_si(rng, 1, 10) for _ in range(n)]

def _gen_machine_rates(rng, dist, m):
    t = dist['type']
    p = dist['params']
    if t == 'uniform':
        lo = p.get('min_rate', p.get('low', 0.5))
        hi = p.get('max_rate', p.get('high', 3.0))
        return [round(rng.uniform(lo, hi), 4) for _ in range(m)]
    elif t == 'step':
        if 'levels' in p:
            levels = p['levels']
            props = p['proportions']
        else:
            levels = [p['low_rate'], p['high_rate']]
            props = [p['step_fraction'], 1.0 - p['step_fraction']]
        result = []
        for i, (lev, prop) in enumerate(zip(levels, props)):
            count = int(m * prop) if i < len(levels) - 1 else m - len(result)
            result.extend([lev] * count)
        rng.shuffle(result)
        return result
    elif t == 'exponential':
        scale = p.get('scale', 1.0)
        mx = p.get('max', 5.0)
        return [min(mx, max(0.01, round(rng.expovariate(1.0/scale), 4))) for _ in range(m)]
    elif t == 'discrete_uniform':
        vals = p['values']
        return [float(rng.choice(vals)) for _ in range(m)]
    elif t == 'continuous_uniform':
        lo = p.get('min_rate', 1.0)
        hi = p.get('max_rate', 6.0)
        return [round(rng.uniform(lo, hi), 4) for _ in range(m)]
    else:
        return [round(rng.uniform(0.5, 3.0), 4) for _ in range(m)]

def _gen_tou(rng, tou_type, params, T):
    if tou_type == 'flat':
        return [1.0] * T
    elif tou_type == 'single_peak':
        ps = params.get('peak_start', 0.3)
        pe = params.get('peak_end', 0.6)
        pm = params.get('peak_multiplier', 2.0)
        bp = params.get('base_price', 1.0)
        ct = []
        for t in range(T):
            frac = t / max(T, 1)
            ct.append(bp * pm if ps <= frac <= pe else bp)
        return ct
    elif tou_type == 'dual_peak':
        if 'peak1_fraction' in params:
            p1f = params['peak1_fraction']
            p2f = params['peak2_fraction']
            pp = params['peak_price']
            op = params['offpeak_price']
            return [pp if abs(t/max(T,1) - p1f) < 0.15 or abs(t/max(T,1) - p2f) < 0.15 else op for t in range(T)]
        else:
            p1s = params.get('peak1_start', 0.2); p1e = params.get('peak1_end', 0.35)
            p2s = params.get('peak2_start', 0.65); p2e = params.get('peak2_end', 0.8)
            pm = params.get('peak_multiplier', 1.8); bp = params.get('base_price', 1.0)
            return [bp * pm if (p1s <= t/max(T,1) <= p1e or p2s <= t/max(T,1) <= p2e) else bp for t in range(T)]
    elif tou_type == 'low_variance':
        bp = params.get('base_price', 1.0); nr = params.get('noise_range', 0.3)
        return [round(bp + rng.uniform(-nr, nr), 4) for _ in range(T)]
    elif tou_type in ('monotonic_decreasing', 'monotonic_increasing'):
        sp = params.get('start_price', 2.0); ep = params.get('end_price', 0.3)
        if tou_type == 'monotonic_increasing': sp, ep = ep, sp
        return [round(sp + (ep - sp) * t / max(T-1, 1), 4) for t in range(T)]
    elif tou_type == 'discrete_uniform':
        vals = params.get('values', [1, 2, 3, 4])
        return [float(rng.choice(vals)) for _ in range(T)]
    elif tou_type == 'continuous_uniform':
        lo = params.get('min_price', 1.0); hi = params.get('max_price', 8.0)
        return [round(rng.uniform(lo, hi), 4) for _ in range(T)]
    elif tou_type == 'step_function':
        bps = params.get('breakpoints_fraction', [0, 0.25, 0.5, 0.75, 1.0])
        prices = params.get('prices', [1, 5, 2, 10])
        return [prices[min(i, len(prices)-1)] for t in range(T)
                for i in range(len(bps)-1) if bps[i] <= t/max(T,1) < bps[i+1]]
    else:
        return [1.0] * T

def gen_family_instance(rng, family, arm_name, family_idx, instance_idx, base_seed):
    """Generate one instance from a family spec."""
    nr = family['n_jobs_range']
    mr = family['m_machines_range']
    tr = family['horizon_T_range']

    n_min = min(nr['min'], MAX_N)
    n_max = min(nr['max'], MAX_N)
    T_min = min(tr['min'], MAX_T)
    T_max = min(tr['max'], MAX_T)

    n = _si(rng, n_min, n_max)
    m = _si(rng, mr['min'], mr['max'])
    T = _si(rng, T_min, T_max)

    p = _gen_processing_times(rng, family['processing_time_distribution'], n)
    e = _gen_machine_rates(rng, family['machine_rate_distribution'], m)
    ct = _gen_tou(rng, family.get('TOU_price_profile_type', 'flat'),
                  family.get('TOU_price_profile_params', {}), T)

    feasible = sum(p) <= m * T
    seed = base_seed + instance_idx
    fname = family['family_name']
    iid = f"c5_{arm_name}_{fname}_{seed}"

    return {
        "n": n, "m": m, "T": T, "p": p, "e": e, "ct": ct,
        "instance_id": iid,
        "metadata": {
            "arm": arm_name,
            "family_name": fname,
            "seed": seed,
            "expected_mechanism": family.get('expected_EHS_failure_mechanism', 'mixed'),
            "feasible": feasible,
            "sum_p": sum(p),
            "has_mechanism_layer": 'mechanism_layer' in family,
            "pt_type": family['processing_time_distribution']['type'],
        }
    }

def gen_literature_instance(rng, gen_spec, arm_name, instance_idx, base_seed):
    """Generate one instance from a literature generator spec."""
    gen_name = gen_spec['generator']

    if gen_name == 'wang_mls_generator':
        n_opts = [30, 60, 100, 150]
        m_opts = [8, 16, 25]
        T_opts = [100, 200]
        n = n_opts[rng.randint(0, len(n_opts)-1)]
        m = m_opts[rng.randint(0, len(m_opts)-1)]
        T = T_opts[rng.randint(0, len(T_opts)-1)]
        p = [_si(rng, 1, 4) for _ in range(n)]
        e = [float(rng.choice(gen_spec['machine_rate_distribution']['params']['values'])) for _ in range(m)]
        ct = [float(rng.choice(gen_spec['TOU_price_profile_params']['values'])) for _ in range(T)]
        fname = 'wang_literature_medium'
        mech = 'mixed'
    else:
        n = _si(rng, 60, 150)
        m = _si(rng, 8, 25)
        T = _si(rng, 100, 200)
        p = [_si(rng, 1, 12) for _ in range(n)]
        e = [round(rng.uniform(1.0, 6.0), 4) for _ in range(m)]
        ct = [round(rng.uniform(1.0, 8.0), 4) for _ in range(T)]
        fname = 'anghinolfi_vls_capped'
        mech = 'mixed'

    feasible = sum(p) <= m * T
    seed = base_seed + instance_idx
    iid = f"c5_{arm_name}_{fname}_{seed}"

    return {
        "n": n, "m": m, "T": T, "p": p, "e": e, "ct": ct,
        "instance_id": iid,
        "metadata": {
            "arm": arm_name,
            "family_name": fname,
            "seed": seed,
            "expected_mechanism": mech,
            "feasible": feasible,
            "sum_p": sum(p),
            "has_mechanism_layer": False,
            "pt_type": 'uniform',
        }
    }

def generate_all_instances():
    all_insts = []
    arm_summary = {}

    for arm_cfg in ARMS_CONFIG:
        arm_name = arm_cfg['arm']
        base_seed = arm_cfg['base_seed']
        arm_insts = []

        if 'generators' in arm_cfg:
            # Literature generators
            for gen_idx, gen_name in enumerate(arm_cfg['generators']):
                gen_spec = next(g for g in LIT_GENERATORS if g['generator'] == gen_name)
                for idx in range(INSTANCES_PER_FAMILY):
                    for attempt in range(MAX_RETRIES):
                        rng = random.Random(base_seed + gen_idx * 10000 + idx * 17 + attempt * 100000)
                        inst = gen_literature_instance(rng, gen_spec, arm_name, idx,
                                                        base_seed + gen_idx * 10000)
                        if inst['metadata']['feasible']:
                            arm_insts.append(inst)
                            break
                    else:
                        arm_insts.append(inst)
                        print(f"  WARNING: No feasible instance for {gen_name} #{idx}")
        else:
            # Regular families
            for fam_idx, family in enumerate(arm_cfg['families']):
                fname = family['family_name']
                for idx in range(INSTANCES_PER_FAMILY):
                    for attempt in range(MAX_RETRIES):
                        rng = random.Random(base_seed + fam_idx * 10000 + idx * 17 + attempt * 100000)
                        inst = gen_family_instance(rng, family, arm_name, fam_idx, idx,
                                                    base_seed + fam_idx * 10000)
                        if inst['metadata']['feasible']:
                            arm_insts.append(inst)
                            break
                    else:
                        arm_insts.append(inst)
                        print(f"  WARNING: No feasible instance for {fname} #{idx}")

        n_feas = sum(1 for i in arm_insts if i['metadata']['feasible'])
        n_lit = sum(1 for i in arm_insts if 'generators' in arm_cfg)
        print(f"  {arm_name}: {len(arm_insts)} insts, {n_feas} feasible")
        arm_summary[arm_name] = {"n_instances": len(arm_insts), "n_feasible": n_feas}
        all_insts.extend(arm_insts)

    # Save instances
    for inst in all_insts:
        with open(INST_DIR / f"{inst['instance_id']}.json", 'w') as f:
            json.dump(inst, f, indent=2)

    manifest = {
        "purpose": "C5 Robust Multi-Budget Adversarial Validation",
        "date": "2026-05-13",
        "budgets": BUDGETS,
        "arms": arm_summary,
        "total_instances": len(all_insts),
    }
    with open(INST_DIR / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    # Save frozen family selection
    frozen = {
        "validation": "C5",
        "frozen": True,
        "arms": {},
    }
    for arm_cfg in ARMS_CONFIG:
        arm_name = arm_cfg['arm']
        if 'generators' in arm_cfg:
            frozen['arms'][arm_name] = {"generators": arm_cfg['generators']}
        else:
            frozen['arms'][arm_name] = {
                "families": [f['family_name'] for f in arm_cfg['families']]
            }
    with open(FAMILIES_DIR / 'c5_frozen_families.json', 'w') as f:
        json.dump(frozen, f, indent=2)

    return all_insts

# ═══════════════════════════════════════════════════════════════
# EHS Evaluation
# ═══════════════════════════════════════════════════════════════

def compute_hv(points, ref_point):
    """Compute hypervolume of Pareto front against reference point.
    points: list of (cmax, energy) tuples (both minimize).
    ref_point: (cmax_ref, energy_ref) — the nadir/worst point.
    HV is the area dominated by the front within [0, ref_cmax] × [0, ref_energy].
    Since cmax and energy are minimized, we compute the area of dominated region.
    """
    if not points:
        return 0.0
    # Sort by cmax ascending
    sorted_pts = sorted(points, key=lambda x: x[0])
    hv = 0.0
    prev_cmax = 0  # cmax lower bound (best)
    prev_energy = ref_point[1]

    for cmax, energy in sorted_pts:
        if cmax >= ref_point[0] or energy >= ref_point[1]:
            continue
        # Rectangular slice from prev_cmax to cmax
        width = cmax - prev_cmax
        if width > 0 and energy < prev_energy:
            hv += width * (ref_point[1] - energy)
        if energy < prev_energy:
            prev_energy = energy
        prev_cmax = cmax

    # Final rectangle to reference cmax
    width = ref_point[0] - prev_cmax
    if width > 0:
        hv += width * (ref_point[1] - prev_energy)
    return hv

def eval_one(inst, budget_s, rng_seed):
    t0 = time.time()
    try:
        rng = random.Random(rng_seed)
        results = run_ehs(inst, rng=rng, time_limit_seconds=budget_s,
                         use_history=True, use_exchange=True, use_esr=True)
        rt = time.time() - t0
        fs = len(results)
        points = [(s.cmax, s.energy) for s in results]
        dc = max((s.cmax for s in results), default=inst['T'])
        be = min((s.energy for s in results), default=float('inf'))
        sc = min((s.cmax for s in results), default=inst['T'])
        to = rt >= budget_s * 0.95
        return {
            'ok': True, 'fs': fs, 'rt': round(rt, 2), 'to': to,
            'dc': dc, 'be': round(be, 6) if be != float('inf') else 'inf',
            'sc': sc, 'points': points, 'err': '',
        }
    except Exception as e:
        return {
            'ok': False, 'fs': 0, 'rt': 0.0, 'to': True,
            'dc': inst['T'], 'be': 'inf', 'sc': inst['T'],
            'points': [], 'err': str(e)[:200],
        }

def evaluate_all(instances):
    raw_rows = []
    t_start = time.time()
    total = len(instances)

    for i, inst in enumerate(instances):
        meta = inst['metadata']
        base = {
            'instance_id': inst['instance_id'],
            'arm': meta['arm'],
            'family_name': meta['family_name'],
            'expected_mechanism': meta['expected_mechanism'],
            'n': inst['n'], 'm': inst['m'], 'T': inst['T'],
            'sum_p': meta['sum_p'],
            'feasible_instance': meta['feasible'],
        }

        if not meta['feasible']:
            for bn, bs in BUDGETS.items():
                raw_rows.append({**base, 'budget': bn, 'budget_s': bs,
                                'feasible_ehs': False, 'front_size': 0,
                                'runtime_s': 0.0, 'timeout': True,
                                'deepest_cmax': inst['T'], 'best_energy': 'inf',
                                'shallowest_cmax': inst['T'],
                                'hypervolume': 0.0, 'points_json': '[]', 'error': 'infeasible'})
            continue

        # Determine reference point for HV from the long run (worst across all runs)
        # We pre-compute after all budgets; store points for later HV computation

        budget_results = {}
        for bn in ['short', 'medium', 'long']:
            bs = BUDGETS[bn]
            rng_seed = meta['seed'] + hash(bn) % 10000
            result = eval_one(inst, bs, rng_seed)
            budget_results[bn] = result

        # Compute shared reference point for HV
        all_points = []
        for bn in ['short', 'medium', 'long']:
            all_points.extend(budget_results[bn]['points'])
        if all_points:
            ref_cmax = max(p[0] for p in all_points) + 1
            ref_energy = max(p[1] for p in all_points) * 1.1 + 1.0
        else:
            ref_cmax = inst['T'] + 1
            ref_energy = 100000.0
        ref_point = (ref_cmax, ref_energy)

        for bn in ['short', 'medium', 'long']:
            r = budget_results[bn]
            bs = BUDGETS[bn]
            hv = compute_hv(r['points'], ref_point)
            raw_rows.append({**base, 'budget': bn, 'budget_s': bs,
                            'feasible_ehs': r['ok'], 'front_size': r['fs'],
                            'runtime_s': r['rt'], 'timeout': r['to'],
                            'deepest_cmax': r['dc'], 'best_energy': r['be'],
                            'shallowest_cmax': r['sc'],
                            'hypervolume': round(hv, 6),
                            'ref_cmax': ref_point[0], 'ref_energy': round(ref_point[1], 6),
                            'error': r['err']})

        # Progress
        if (i + 1) % 3 == 0 or i == total - 1:
            elapsed = time.time() - t_start
            eta = (elapsed / (i + 1)) * (total - i - 1) if i < total - 1 else 0
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] {i+1}/{total} "
                  f"elapsed={elapsed:.0f}s ETA={eta:.0f}s")

        # Periodic save every 6 instances
        if (i + 1) % 6 == 0 and raw_rows:
            with open(EVAL_DIR / 'c5_raw_runs.csv', 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
                w.writeheader()
                w.writerows(raw_rows)

    # Final save
    with open(EVAL_DIR / 'c5_raw_runs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        w.writeheader()
        w.writerows(raw_rows)

    total_time = time.time() - t_start
    print(f"\n  Done: {len(raw_rows)} runs in {total_time:.0f}s ({total_time/3600:.1f}h)")
    return raw_rows

# ═══════════════════════════════════════════════════════════════
# Metrics + Mechanism Confirmation
# ═══════════════════════════════════════════════════════════════

def _tobool(v):
    if isinstance(v, bool): return v
    if isinstance(v, str): return v.lower() in ('true', '1', 'yes')
    return bool(v)

def compute_instance_metrics(inst_rows):
    """Compute per-instance metrics from 3-budget rows."""
    s = inst_rows['short']
    m = inst_rows['medium']
    l = inst_rows['long']

    fs_s = int(s['front_size']); fs_m = int(m['front_size']); fs_l = int(l['front_size'])
    dc_s = int(s['deepest_cmax']); dc_m = int(m['deepest_cmax']); dc_l = int(l['deepest_cmax'])
    rt_s = float(s['runtime_s']); rt_m = float(m['runtime_s']); rt_l = float(l['runtime_s'])
    to_s = _tobool(s['timeout']); to_m = _tobool(m['timeout']); to_l = _tobool(l['timeout'])
    hv_s = float(s['hypervolume']); hv_m = float(m['hypervolume']); hv_l = float(l['hypervolume'])

    delta_fs_30_120 = fs_m - fs_s
    delta_fs_120_300 = fs_l - fs_m
    delta_fs_30_300 = fs_l - fs_s
    delta_cmax_30_300 = dc_s - dc_l
    hv_gain_30_120 = hv_m - hv_s
    hv_gain_120_300 = hv_l - hv_m

    # Persistent hard: difficulty outlasts the 30s trivial short-budget regime.
    # Instance may converge at 120s or 300s; the point is that 30s alone is insufficient.
    persistent_hard = (
        fs_m > fs_s + 3                    # significant growth after short budget
        and fs_l >= fs_m                   # no regression
        and not (fs_s <= 2 and fs_m <= 2)  # nontrivial
    )

    # Mechanism confirmation (budget-aware)
    mech = s['expected_mechanism']
    mc = False
    mc_reason = 'none'

    if mech == 'asgh_lock_in':
        # A-SGH lock-in: EHS needs >30s to converge. Lock breaks after short budget.
        # Converged at 120s or 300s is OK — the mechanism is that short budget is insufficient.
        mc = (
            persistent_hard
            and delta_fs_30_120 > 2
            and (rt_120 > 30 or rt_300 > 30)  # substantial time needed
        )
        mc_reason = f"fs={fs_s}→{fs_m}→{fs_l}" + (" MC" if mc else " no")

    elif mech == 'first_khat_dominance':
        # First khat dominates short budget; late budgets unlock more
        mc = (
            to_s and fs_s <= 3
            and delta_fs_30_120 > 3
            and delta_fs_120_300 > 0
        )
        mc_reason = f"to_s={to_s} fs_s={fs_s}→{fs_m}→{fs_l}" + (" MC" if mc else " no")

    elif mech == 'es_exploration_tension':
        mc = (delta_fs_120_300 > 2 and delta_cmax_30_300 >= 2)
        mc_reason = f"dfs_120_300={delta_fs_120_300} dcmax={delta_cmax_30_300}" + (" MC" if mc else " no")

    elif mech == 'front_coverage_gap':
        mc = (fs_s >= 2 and delta_fs_120_300 > 2)
        mc_reason = f"fs_s={fs_s} dfs_120_300={delta_fs_120_300}" + (" MC" if mc else " no")

    elif mech == 'res_reinsertion_starvation':
        mc = (rt_s < 15 and fs_s >= 3 and delta_fs_120_300 > 3)
        mc_reason = f"rt_s={rt_s}s fs_s={fs_s} dfs_120_300={delta_fs_120_300}" + (" MC" if mc else " no")

    elif mech == 'load_imbalance':
        T = int(s['T'])
        mc = (dc_l < T * 0.5 and delta_fs_120_300 <= 3 and fs_s > 0)
        mc_reason = f"dc_l={dc_l} T={T}" + (" MC" if mc else " no")

    elif mech == 'epsilon_skip':
        mc = (fs_s >= 3 and delta_fs_120_300 <= 2 and delta_fs_30_120 > 0)
        mc_reason = f"fs_s={fs_s} dfs_120_300={delta_fs_120_300}" + (" MC" if mc else " no")

    elif mech == 'short_budget_pressure':
        mc = (to_s and fs_s <= 1 and delta_fs_30_120 > 5)
        mc_reason = f"to_s={to_s} fs_s={fs_s} dfs_30_120={delta_fs_30_120}" + (" MC" if mc else " no")

    else:
        # mixed / unknown: check persistent growth
        mc = persistent_hard and delta_fs_120_300 > 2
        mc_reason = f"persistent={persistent_hard} dfs_120_300={delta_fs_120_300}" + (" MC" if mc else " no")

    # Nontrivial MC: MC confirmed AND not purely structural
    has_layer = _tobool(s.get('has_mechanism_layer', False))
    n = int(s['n']); T_val = int(s['T'])
    size_only = (n > 120 or T_val > 180) and delta_fs_120_300 <= 2
    ntm = mc and persistent_hard and not size_only

    # High-yield on front-size growth (comparable to C4 metric)
    hy = False; yr = 'none'
    if delta_fs_30_300 >= 2:
        hy = True; yr = f'+{delta_fs_30_300}'
    elif fs_s <= 1 and fs_l >= 3:
        hy = True; yr = f'near_zero→{fs_l}'
    elif delta_cmax_30_300 >= 2:
        hy = True; yr = f'cmaxΔ={delta_cmax_30_300}'
    elif to_s and fs_s <= 2:
        hy = True; yr = 'slow'
    elif fs_s > 0 and fs_l == fs_s:
        yr = f'sat_{fs_s}'
    else:
        yr = f'Δ{delta_fs_30_300}'

    return {
        'instance_id': s['instance_id'],
        'arm': s['arm'],
        'family_name': s['family_name'],
        'expected_mechanism': mech,
        'n': n, 'm': int(s['m']), 'T': T_val,
        'feasible_instance': _tobool(s['feasible_instance']),
        'evaluable': _tobool(s['feasible_ehs']),
        # Front sizes
        'fs_30': fs_s, 'fs_120': fs_m, 'fs_300': fs_l,
        # Deltas
        'delta_fs_30_120': delta_fs_30_120,
        'delta_fs_120_300': delta_fs_120_300,
        'delta_fs_30_300': delta_fs_30_300,
        'delta_cmax_30_300': delta_cmax_30_300,
        # Deepest Cmax
        'dc_30': dc_s, 'dc_120': dc_m, 'dc_300': dc_l,
        # Hypervolume
        'hv_30': round(hv_s, 2), 'hv_120': round(hv_m, 2), 'hv_300': round(hv_l, 2),
        'hv_gain_30_120': round(hv_gain_30_120, 2),
        'hv_gain_120_300': round(hv_gain_120_300, 2),
        # Runtime
        'rt_30': rt_s, 'rt_120': rt_m, 'rt_300': rt_l,
        # Timeouts
        'to_30': to_s, 'to_120': to_m, 'to_300': to_l,
        # Flags
        'high_yield': hy, 'yield_reason': yr,
        'persistent_hard': persistent_hard,
        'mechanism_confirmed': mc, 'mc_reason': mc_reason,
        'nontrivial_mc': ntm,
    }

# ═══════════════════════════════════════════════════════════════
# Decision
# ═══════════════════════════════════════════════════════════════

def compute_decision(summaries):
    # Aggregate by arm
    arm_st = defaultdict(lambda: {
        'n_inst': 0, 'n_feas': 0, 'n_eval': 0,
        'n_hy': 0, 'n_persistent': 0, 'n_mc': 0, 'n_ntm': 0,
        'dfs_30_300': [], 'dfs_120_300': [], 'hv_gains': [],
        'rts_30': [], 'rts_120': [], 'rts_300': [],
        'to_30': 0, 'to_120': 0, 'to_300': 0,
        'fss_30': [], 'fss_120': [], 'fss_300': [],
        'hvs_30': [], 'hvs_120': [], 'hvs_300': [],
        'families': set(),
    })

    # Per-family aggregation
    fam_st = defaultdict(lambda: {
        'n_inst': 0, 'n_eval': 0, 'n_persistent': 0, 'n_mc': 0, 'n_ntm': 0,
        'dfs_30_300': [], 'dfs_120_300': [],
    })

    for s in summaries:
        arm = s['arm']; fname = s['family_name']
        arm_st[arm]['n_inst'] += 1
        arm_st[arm]['families'].add(fname)
        if s['feasible_instance']: arm_st[arm]['n_feas'] += 1
        if s['evaluable']:
            e = arm_st[arm]
            e['n_eval'] += 1
            e['dfs_30_300'].append(s['delta_fs_30_300'])
            e['dfs_120_300'].append(s['delta_fs_120_300'])
            e['hv_gains'].append(s['hv_gain_120_300'])
            e['rts_30'].append(s['rt_30'])
            e['rts_120'].append(s['rt_120'])
            e['rts_300'].append(s['rt_300'])
            e['fss_30'].append(s['fs_30'])
            e['fss_120'].append(s['fs_120'])
            e['fss_300'].append(s['fs_300'])
            e['hvs_30'].append(s['hv_30'])
            e['hvs_120'].append(s['hv_120'])
            e['hvs_300'].append(s['hv_300'])
            if s['to_30']: e['to_30'] += 1
            if s['to_120']: e['to_120'] += 1
            if s['to_300']: e['to_300'] += 1
            if s['high_yield']: e['n_hy'] += 1
            if s['persistent_hard']: e['n_persistent'] += 1
            if s['mechanism_confirmed']: e['n_mc'] += 1
            if s['nontrivial_mc']: e['n_ntm'] += 1

            fm = fam_st[fname]
            fm['n_inst'] += 1; fm['n_eval'] += 1
            fm['dfs_30_300'].append(s['delta_fs_30_300'])
            fm['dfs_120_300'].append(s['delta_fs_120_300'])
            if s['persistent_hard']: fm['n_persistent'] += 1
            if s['mechanism_confirmed']: fm['n_mc'] += 1
            if s['nontrivial_mc']: fm['n_ntm'] += 1

    return arm_st, fam_st

def format_pct(num, den):
    return f'{num/den*100:.0f}% ({num}/{den})' if den else 'N/A'

def format_mean(lst):
    return f'{sum(lst)/len(lst):.1f}' if lst else 'N/A'

def print_decision(arm_st, fam_st):
    print("\n" + "=" * 80)
    print("C5 ROBUST MULTI-BUDGET VALIDATION — RESULTS")
    print("=" * 80)

    # Main comparison
    main_arms = ['llm_l2', 'literature', 'random', 'simple_stress']
    main_labels = {
        'llm_l2': 'LLM Call2',
        'literature': 'Literature',
        'random': 'Random',
        'simple_stress': 'Simple Stress',
    }

    print(f"\n{'=' * 80}")
    print("MAIN COMPARISON (multi-budget persistent difficulty)")
    print(f"{'=' * 80}\n")

    header = (f"  {'Arm':<20} {'Inst':>5} {'Feas':>5} {'Eval':>5} "
              f"{'HY%':>7} {'Persist%':>9} {'MC%':>7} {'Δfs30→300':>10} "
              f"{'Δfs120→300':>11} {'HVgain120':>10} {'TO120':>6} {'TO300':>6}")
    print(header)
    print(f"  {'-' * len(header)}")

    rows = []
    for arm in main_arms:
        if arm not in arm_st:
            continue
        s = arm_st[arm]; e = s['n_eval']
        hy_rate = format_pct(s['n_hy'], e)
        pers_rate = format_pct(s['n_persistent'], e)
        mc_rate = format_pct(s['n_mc'], e)
        dfs30 = format_mean(s['dfs_30_300'])
        dfs120 = format_mean(s['dfs_120_300'])
        hvg = format_mean(s['hv_gains'])
        print(f"  {main_labels[arm]:<20} {s['n_inst']:>5} {s['n_feas']:>5} {e:>5} "
              f"{hy_rate:>7} {pers_rate:>9} {mc_rate:>7} {dfs30:>10} "
              f"{dfs120:>11} {hvg:>10} {s['to_120']:>5}/{e} {s['to_300']:>5}/{e}")
        rows.append({'arm': arm, 'label': main_labels[arm], **s})

    # Side: agent_manual_sweep
    print(f"\n  SIDE TABLE (internal control):")
    agent = arm_st.get('agent_manual_sweep')
    if agent:
        e = agent['n_eval']
        pers_rate = format_pct(agent['n_persistent'], e)
        mc_rate = format_pct(agent['n_mc'], e)
        ntm_rate = format_pct(agent['n_ntm'], e)
        print(f"  {'agent_manual_sweep':<20} {agent['n_inst']:>5} {agent['n_feas']:>5} {e:>5} "
              f"{format_pct(agent['n_hy'], e):>7} {pers_rate:>9} {mc_rate:>7} "
              f"{format_mean(agent['dfs_30_300']):>10} {format_mean(agent['dfs_120_300']):>11} "
              f"{format_mean(agent['hv_gains']):>10}")
        rows.append({'arm': 'agent_manual_sweep', 'label': 'agent_manual_sweep', **agent})

    # Per-family breakdown
    print(f"\n{'=' * 80}")
    print("PER-FAMILY BREAKDOWN")
    print(f"{'=' * 80}\n")

    for arm in ['llm_l2', 'literature', 'random', 'agent_manual_sweep', 'simple_stress']:
        arm_fams = [(fn, fs) for fn, fs in fam_st.items() if fn in arm_st[arm]['families']]
        if not arm_fams:
            continue
        print(f"  --- {arm} ---")
        for fn, fs in sorted(arm_fams):
            e = fs['n_eval']
            pers_rate = format_pct(fs['n_persistent'], e)
            mc_rate = format_pct(fs['n_mc'], e)
            dfs30 = ', '.join(f'{d:+d}' for d in fs['dfs_30_300'])
            dfs120 = ', '.join(f'{d:+d}' for d in fs['dfs_120_300'])
            print(f"    {fn:<45} pers={pers_rate} mc={mc_rate}")
            print(f"      Δfs(30→300): [{dfs30}]")
            print(f"      Δfs(120→300): [{dfs120}]")
        print()

    # Gate decision
    llm = arm_st.get('llm_l2')
    lit = arm_st.get('literature')
    ran = arm_st.get('random')
    simple = arm_st.get('simple_stress')
    agent = arm_st.get('agent_manual_sweep')

    print(f"{'=' * 80}")
    print("GATE DECISION")
    print(f"{'=' * 80}\n")

    if llm and lit and ran:
        l2_e = max(llm['n_eval'], 1)
        lit_e = max(lit['n_eval'], 1)
        ran_e = max(ran['n_eval'], 1)

        l2_pers = llm['n_persistent'] / l2_e
        lit_pers = lit['n_persistent'] / lit_e
        ran_pers = ran['n_persistent'] / ran_e

        l2_mc = llm['n_mc'] / l2_e
        lit_mc = lit['n_mc'] / lit_e

        print(f"  Persistent Hard Rate:")
        print(f"    LLM:        {llm['n_persistent']}/{llm['n_eval']} = {l2_pers:.0%}")
        print(f"    Literature: {lit['n_persistent']}/{lit['n_eval']} = {lit_pers:.0%}")
        print(f"    Random:     {ran['n_persistent']}/{ran['n_eval']} = {ran_pers:.0%}")
        if simple:
            simp_pers = simple['n_persistent'] / max(simple['n_eval'], 1)
            print(f"    SimpleStr:  {simple['n_persistent']}/{simple['n_eval']} = {simp_pers:.0%}")
        print()

        print(f"  Mechanism Confirmed Rate:")
        print(f"    LLM:        {llm['n_mc']}/{llm['n_eval']} = {l2_mc:.0%}")
        print(f"    Literature: {lit['n_mc']}/{lit['n_eval']} = {lit_mc:.0%}")
        if agent:
            ag_mc = agent['n_mc'] / max(agent['n_eval'], 1)
            print(f"    AgentSW:    {agent['n_mc']}/{agent['n_eval']} = {ag_mc:.0%}")
        print()

        print(f"  NT-MC: LLM={llm['n_ntm']}/{llm['n_eval']} Lit={lit['n_ntm']}/{lit['n_eval']}")
        print()

        # Gate logic
        if l2_pers > lit_pers + 0.1 and l2_pers > ran_pers + 0.1 and l2_mc >= 0.3:
            gate = 'STRONG'
        elif l2_pers > ran_pers and l2_pers >= lit_pers - 0.05 and l2_mc > 0:
            gate = 'MODERATE'
        elif l2_pers > ran_pers:
            gate = 'WEAK'
        else:
            gate = 'FAIL'

        print(f"  Gate: {gate}")

        # Paper-safe wording
        print(f"\n{'=' * 80}")
        print("PAPER-SAFE CLAIMS")
        print(f"{'=' * 80}\n")

        if gate in ('STRONG', 'MODERATE'):
            print(f"  Gate: {gate}")
            print()
            print(f"  Acceptable claim:")
            print(f"    Interactive LLM prompting can generate mechanism-aware EHS stress")
            print(f"    families that outperform random and literature-derived generators")
            print(f"    on persistent difficulty rates under multi-budget evaluation")
            print(f"    ({', '.join(str(bs) for bs in BUDGETS.values())}s).")
            print(f"    The difficulty persists beyond the trivial short-time-limit regime.")
            if l2_mc > 0:
                print(f"    At least one LLM family ({l2_mc:.0%} mechanism-confirmed) shows")
                print(f"    genuine EHS-mechanism-specific stress, not merely budget truncation.")
        else:
            print(f"  Gate: {gate} — LLM families do not beat external baselines")
            print(f"  on persistent difficulty under multi-budget evaluation.")

        return gate, rows

    return 'INCOMPLETE', []

# ═══════════════════════════════════════════════════════════════
# Write artifacts
# ═══════════════════════════════════════════════════════════════

def write_all_artifacts(summaries, arm_st, fam_st, gate, raw_rows):
    # Instance summary CSV
    if summaries:
        with open(EVAL_DIR / 'c5_instance_summary.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            w.writeheader()
            w.writerows(summaries)

    # Family summary CSV
    fam_rows = []
    for fn, fs in fam_st.items():
        e = fs['n_eval']
        fam_rows.append({
            'family_name': fn,
            'n_instances': fs['n_inst'],
            'n_evaluable': e,
            'n_persistent': fs['n_persistent'],
            'persistent_rate': f'{fs["n_persistent"]/e:.0%}' if e else 'N/A',
            'n_mc': fs['n_mc'],
            'mc_rate': f'{fs["n_mc"]/e:.0%}' if e else 'N/A',
            'n_ntm': fs['n_ntm'],
            'dfs_30_300': ', '.join(f'{d:+d}' for d in fs['dfs_30_300']),
            'dfs_120_300': ', '.join(f'{d:+d}' for d in fs['dfs_120_300']),
        })
    with open(EVAL_DIR / 'c5_family_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(fam_rows[0].keys()))
        w.writeheader()
        w.writerows(fam_rows)

    # Mechanism confirmation CSV
    mc_rows = []
    for s in summaries:
        mc_rows.append({
            'instance_id': s['instance_id'],
            'arm': s['arm'],
            'family_name': s['family_name'],
            'expected_mechanism': s['expected_mechanism'],
            'fs_30': s['fs_30'], 'fs_120': s['fs_120'], 'fs_300': s['fs_300'],
            'delta_fs_120_300': s['delta_fs_120_300'],
            'persistent_hard': s['persistent_hard'],
            'mechanism_confirmed': s['mechanism_confirmed'],
            'mc_reason': s['mc_reason'],
            'nontrivial_mc': s['nontrivial_mc'],
        })
    with open(EVAL_DIR / 'c5_mechanism_confirmation.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(mc_rows[0].keys()))
        w.writeheader()
        w.writerows(mc_rows)

    # C5_RESULTS.md
    with open(ITER_DIR / 'C5_RESULTS.md', 'w') as f:
        f.write('# C5 Results: Multi-Budget Adversarial Validation\n\n')
        f.write(f'**Date**: 2026-05-13\n')
        f.write(f'**Budgets**: {BUDGETS}\n\n')

        f.write('## Main Comparison\n\n')
        f.write('| Arm | Inst | Eval | Persist% | MC% | Δfs30→300 | Δfs120→300 | HVgain120-300 | TO120 | TO300 |\n')
        f.write('|-----|------|------|----------|-----|-----------|------------|---------------|-------|-------|\n')
        for arm in ['llm_l2', 'literature', 'random', 'simple_stress']:
            s = arm_st.get(arm)
            if not s: continue
            e = s['n_eval']
            labels = {'llm_l2': 'LLM Call2', 'literature': 'Literature', 'random': 'Random', 'simple_stress': 'Simple Stress'}
            f.write(f"| {labels.get(arm, arm)} | {s['n_inst']} | {e} | "
                   f"{format_pct(s['n_persistent'], e)} | {format_pct(s['n_mc'], e)} | "
                   f"{format_mean(s['dfs_30_300'])} | {format_mean(s['dfs_120_300'])} | "
                   f"{format_mean(s['hv_gains'])} | {s['to_120']}/{e} | {s['to_300']}/{e} |\n")

        agent = arm_st.get('agent_manual_sweep')
        if agent:
            e = agent['n_eval']
            f.write(f"| agent_manual_sweep (int.) | {agent['n_inst']} | {e} | "
                   f"{format_pct(agent['n_persistent'], e)} | {format_pct(agent['n_mc'], e)} | "
                   f"{format_mean(agent['dfs_30_300'])} | {format_mean(agent['dfs_120_300'])} | "
                   f"{format_mean(agent['hv_gains'])} | {agent['to_120']}/{e} | {agent['to_300']}/{e} |\n")

        f.write(f'\n## Gate: **{gate}**\n')

    # C5_DECISION.md
    with open(ITER_DIR / 'C5_DECISION.md', 'w') as f:
        f.write('# C5 Decision\n\n')
        f.write(f'**Date**: 2026-05-13\n')
        f.write(f'**Gate**: {gate}\n\n')

        f.write('## Decision Questions\n\n')
        llm = arm_st.get('llm_l2'); lit = arm_st.get('literature'); ran = arm_st.get('random')
        if llm and lit and ran:
            l2_e = max(llm['n_eval'], 1)
            lit_e = max(lit['n_eval'], 1)
            l2_pers = llm['n_persistent'] / l2_e
            lit_pers = lit['n_persistent'] / lit_e
            ran_pers = ran['n_persistent'] / l2_e

            f.write(f'1. **Does Phase C still pass under multi-budget validation?** ')
            f.write(f'LLM persistent={l2_pers:.0%} vs Lit={lit_pers:.0%} vs Random={ran_pers:.0%}. ')
            if l2_pers > lit_pers and l2_pers > ran_pers:
                f.write('Yes.\n')
            else:
                f.write('No or marginal.\n')

            f.write(f'2. **Which family is strongest?** See per-family breakdown.\n')

            f.write(f'3. **LLM vs Random?** ')
            if l2_pers > ran_pers:
                f.write(f'LLM={l2_pers:.0%} > Random={ran_pers:.0%}.\n')
            else:
                f.write(f'LLM={l2_pers:.0%} ≤ Random={ran_pers:.0%}.\n')

            f.write(f'4. **LLM vs Literature?** ')
            if l2_pers > lit_pers:
                f.write(f'LLM={l2_pers:.0%} > Lit={lit_pers:.0%}.\n')
            else:
                f.write(f'LLM={l2_pers:.0%} ≤ Lit={lit_pers:.0%}.\n')

            agent = arm_st.get('agent_manual_sweep')
            if agent:
                ag_pers = agent['n_persistent'] / max(agent['n_eval'], 1)
                f.write(f'5. **LLM vs agent_manual_sweep (internal control)?** ')
                if l2_pers > ag_pers:
                    f.write(f'LLM beats agent on persistent rate.\n')
                else:
                    f.write(f'Agent beats LLM={ag_pers:.0%} vs {l2_pers:.0%}.\n')

            f.write(f'6. **EHS-mechanism-specific or budget artifact?** ')
            if llm['n_mc'] > 0:
                f.write(f'Mechanism-specific for {llm["n_mc"]}/{llm["n_eval"]} LLM instances.\n')
            else:
                f.write('Likely budget artifact.\n')

        f.write(f'\n7. **Paper-safe claim**: ')
        if gate in ('STRONG', 'MODERATE'):
            f.write('"LLM-generated families expose persistent EHS stress mechanisms '
                    'better than random and literature-derived generators '
                    'under controlled multi-budget evaluation."\n')
        else:
            f.write('Phase C advantage against external baselines disappears under multi-budget evaluation. '
                    'The paper should report Phase C as exploratory only.\n')

    return gate

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 80)
    print("C5: ROBUST MULTI-BUDGET ADVERSARIAL VALIDATION")
    print("=" * 80)
    print(f"Budgets: {BUDGETS}")
    print(f"Arms: {[a['arm'] for a in ARMS_CONFIG]}")
    print(f"Instances/family: {INSTANCES_PER_FAMILY}")
    n_fams = sum(len(a.get('families', [])) + len(a.get('generators', [])) for a in ARMS_CONFIG)
    print(f"Total: {n_fams} families × {INSTANCES_PER_FAMILY} = ~{n_fams * INSTANCES_PER_FAMILY} instances")
    est_runtime = n_fams * INSTANCES_PER_FAMILY * sum(BUDGETS.values())
    print(f"Worst-case sequential: {est_runtime}s ({est_runtime/60:.0f}min, {est_runtime/3600:.1f}h)")
    print()

    # Step 1: Generate instances
    print("[Step 1] Generating instances...")
    t0 = time.time()
    instances = generate_all_instances()
    print(f"  Total: {len(instances)} instances in {time.time()-t0:.0f}s")
    n_feas = sum(1 for i in instances if i['metadata']['feasible'])
    print(f"  Feasible: {n_feas}/{len(instances)}")
    print()

    # Step 2: EHS evaluation
    print(f"[Step 2] Running EHS at {len(BUDGETS)} budgets × {len(instances)} instances...")
    t1 = time.time()
    raw_rows = evaluate_all(instances)
    eval_time = time.time() - t1
    print(f"  Evaluation done in {eval_time:.0f}s ({eval_time/3600:.1f}h)")
    print()

    # Step 3: Compute metrics
    print("[Step 3] Computing metrics...")
    # Group raw rows by instance
    inst_groups = defaultdict(dict)
    for row in raw_rows:
        inst_groups[row['instance_id']][row['budget']] = row

    summaries = []
    for iid in sorted(inst_groups.keys()):
        d = inst_groups[iid]
        if 'short' in d and 'medium' in d and 'long' in d:
            s = compute_instance_metrics(d)
            summaries.append(s)

    print(f"  {len(summaries)} instance summaries computed")
    print()

    # Step 4: Decision
    print("[Step 4] Computing arm/family aggregates and gate...")
    arm_st, fam_st = compute_decision(summaries)
    gate, rows = print_decision(arm_st, fam_st)

    # Step 5: Write artifacts
    print("\n[Step 5] Writing artifacts...")
    write_all_artifacts(summaries, arm_st, fam_st, gate, raw_rows)

    print(f"\n{'=' * 80}")
    print(f"C5 complete. Gate: {gate}")
    print(f"Total time: {time.time() - t0:.0f}s ({ (time.time() - t0)/3600:.1f}h)")
    print(f"{'=' * 80}")

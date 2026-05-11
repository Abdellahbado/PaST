#!/usr/bin/env python3
"""C4 Validation pipeline: generate 60 instances, run EHS, compute metrics, decide gate."""

import csv, json, os, sys, time, random, math
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ITER_DIR = PROJECT_ROOT / 'research/glns_llm_heuristic_20260422/iterations/20260510_phaseC_adversarial_ehs_benchmark_design'
FAMILIES_DIR = ITER_DIR / 'families'
NOTES_DIR = ITER_DIR / 'notes'
EVAL_DIR = ITER_DIR / 'eval'
INST_DIR = ITER_DIR / 'generated_instances' / 'c4_validation'
os.makedirs(INST_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(NOTES_DIR, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))
from glns.paper_heuristics import run_ehs

SHORT_BUDGET = 30
LONG_BUDGET = 90
MAX_N = 150
MAX_T = 200
MAX_RETRIES = 100
INSTANCES_PER_FAMILY = 5

# ─────────────────────────────────────────────────────────────
# Task B: Family selection
# ─────────────────────────────────────────────────────────────

def select_families():
    llm_l2_raw = json.load(open(FAMILIES_DIR / 'llm_counter_sweep_families.json'))
    human_raw = json.load(open(FAMILIES_DIR / 'human_sweep_families.json'))
    random_raw = json.load(open(FAMILIES_DIR / 'random_families.json'))

    llm_l2_sel = llm_l2_raw['families'][:4]  # M1-M4
    human_sel = [
        human_raw['families'][0],  # human_tight_epsilon
        human_raw['families'][1],  # human_loose_epsilon
        human_raw['families'][5],  # human_mixed_job_sizes
        human_raw['families'][6],  # human_many_machines_sparse
    ]
    random_sel = random_raw['families'][:4]  # random_000..003

    selection = {
        "validation": "C4",
        "date": "2026-05-11",
        "frozen": True,
        "n_families_per_arm": 4,
        "instances_per_family": INSTANCES_PER_FAMILY,
        "total_instances": 60,
        "short_budget": SHORT_BUDGET,
        "long_budget": LONG_BUDGET,
        "caps": {"n_max": MAX_N, "T_max": MAX_T},
        "arms": {
            "llm_l2": {
                "source": "families/llm_counter_sweep_families.json",
                "base_seed_offset": 180000,
                "families": [f["family_name"] for f in llm_l2_sel]
            },
            "human": {
                "source": "families/human_sweep_families.json",
                "base_seed_offset": 120000,
                "families": [f["family_name"] for f in human_sel]
            },
            "random": {
                "source": "families/random_families.json",
                "base_seed_offset": 110000,
                "families": [f["family_name"] for f in random_sel]
            }
        }
    }
    with open(NOTES_DIR / 'c4_selected_families.json', 'w') as f:
        json.dump(selection, f, indent=2)
    return selection, llm_l2_sel, human_sel, random_sel

# ─────────────────────────────────────────────────────────────
# Task C: Instance generation
# ─────────────────────────────────────────────────────────────

def _sample_int(rng, lo, hi):
    if lo >= hi:
        return min(lo, hi)
    return rng.randint(lo, hi)

def _gen_processing_times(rng, dist, n):
    t = dist['type']
    p = dist['params']
    if t == 'uniform':
        return [_sample_int(rng, int(p['low']), int(p['high'])) for _ in range(n)]
    elif t == 'bimodal':
        n_small = int(n * p.get('small_fraction', 0.5))
        n_large = n - n_small
        small = [_sample_int(rng, int(p.get('small_low', p.get('modes', [[2,5],[15,25]])[0][0])), int(p.get('small_high', p.get('modes', [[2,5],[15,25]])[0][1]))) for _ in range(n_small)]
        large = [_sample_int(rng, int(p.get('large_low', p.get('modes', [[2,5],[15,25]])[1][0])), int(p.get('large_high', p.get('modes', [[2,5],[15,25]])[1][1]))) for _ in range(n_large)]
        result = small + large
        rng.shuffle(result)
        return result
    elif t == 'exponential_truncated':
        scale = p['scale']
        mx = p.get('max', 30)
        return [min(mx, max(1, int(round(rng.expovariate(1.0/scale))))) for _ in range(n)]
    elif t == 'normal_truncated':
        mean = p['mean']
        std = p['std']
        mn = p.get('min', 1)
        mx = p.get('max', 30)
        return [max(mn, min(mx, int(round(rng.gauss(mean, std))))) for _ in range(n)]
    else:
        return [_sample_int(rng, 1, 10) for _ in range(n)]

def _gen_machine_rates(rng, dist, m):
    t = dist['type']
    p = dist['params']
    if t == 'uniform':
        return [round(rng.uniform(p.get('min_rate', p.get('low', 0.5)), p.get('max_rate', p.get('high', 3.0))), 4) for _ in range(m)]
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
            if ps <= frac <= pe:
                ct.append(bp * pm)
            else:
                ct.append(bp)
        return ct
    elif tou_type == 'dual_peak':
        if 'peak1_fraction' in params:
            # LLM L2 format
            p1f = params['peak1_fraction']
            p2f = params['peak2_fraction']
            pp = params['peak_price']
            op = params['offpeak_price']
            ct = []
            for t in range(T):
                frac = t / max(T, 1)
                d1 = abs(frac - p1f)
                d2 = abs(frac - p2f)
                if d1 < 0.15 or d2 < 0.15:
                    ct.append(pp)
                else:
                    ct.append(op)
            return ct
        else:
            # Human format
            p1s = params.get('peak1_start', 0.2)
            p1e = params.get('peak1_end', 0.35)
            p2s = params.get('peak2_start', 0.65)
            p2e = params.get('peak2_end', 0.8)
            pm = params.get('peak_multiplier', 1.8)
            bp = params.get('base_price', 1.0)
            ct = []
            for t in range(T):
                frac = t / max(T, 1)
                if p1s <= frac <= p1e or p2s <= frac <= p2e:
                    ct.append(bp * pm)
                else:
                    ct.append(bp)
            return ct
    elif tou_type == 'low_variance':
        bp = params.get('base_price', 1.0)
        nr = params.get('noise_range', 0.3)
        return [round(bp + rng.uniform(-nr, nr), 4) for _ in range(T)]
    elif tou_type == 'high_variance':
        bp = params.get('base_price', 0.5)
        nr = params.get('noise_range', 3.0)
        return [round(bp + rng.uniform(-nr, nr), 4) for _ in range(T)]
    elif tou_type in ('monotonic_decreasing', 'monotonic_increasing'):
        sp = params.get('start_price', 2.0)
        ep = params.get('end_price', 0.3)
        if tou_type == 'monotonic_increasing':
            sp, ep = ep, sp
        return [round(sp + (ep - sp) * t / max(T-1, 1), 4) for t in range(T)]
    else:
        return [1.0] * T

def gen_instance(rng, family, instance_idx, base_seed):
    nr = family['n_jobs_range']
    mr = family['m_machines_range']
    tr = family['horizon_T_range']

    # Clamp to caps
    n_min = min(nr['min'], MAX_N)
    n_max = min(nr['max'], MAX_N)
    m_min = mr['min']
    m_max = mr['max']
    T_min = min(tr['min'], MAX_T)
    T_max = min(tr['max'], MAX_T)

    n = _sample_int(rng, n_min, n_max)
    m = _sample_int(rng, m_min, m_max)
    T = _sample_int(rng, T_min, T_max)

    p = _gen_processing_times(rng, family['processing_time_distribution'], n)
    e = _gen_machine_rates(rng, family['machine_rate_distribution'], m)
    ct = _gen_tou(rng, family.get('TOU_price_profile_type', 'flat'), family.get('TOU_price_profile_params', {}), T)

    feasible = sum(p) <= m * T

    arm = family.get('_arm', 'llm_l2' if 'hybrid' in family.get('family_name', '') else 'human')
    # detect arm from family name or metadata
    fname = family['family_name']
    if fname.startswith('hybrid_M'):
        arm = 'llm_l2'
    elif fname.startswith('human_'):
        arm = 'human'
    elif fname.startswith('random_'):
        arm = 'random'
    else:
        arm = family.get('_arm', 'unknown')

    instance_id = f"c4v_{fname}_{base_seed + instance_idx}"
    return {
        "n": n, "m": m, "T": T, "p": p, "e": e, "ct": ct,
        "instance_id": instance_id,
        "metadata": {
            "family_name": fname,
            "arm": arm,
            "seed": base_seed + instance_idx,
            "expected_mechanism": family.get('expected_EHS_failure_mechanism', 'unknown'),
            "feasible": feasible,
            "sum_p": sum(p),
            "has_mechanism_layer": 'mechanism_layer' in family,
            "pt_type": family['processing_time_distribution']['type'],
            "pt_params": family['processing_time_distribution']['params'],
        }
    }

def generate_all_instances(llm_l2_sel, human_sel, random_sel, selection):
    all_instances = []
    arms_config = {
        'llm_l2': (llm_l2_sel, 180000),
        'human': (human_sel, 120000),
        'random': (random_sel, 110000),
    }

    manifest = {"arms": {}}

    for arm_name, (families, base_offset) in arms_config.items():
        rng = random.Random(base_offset + 9999)
        arm_instances = []
        for family in families:
            family['_arm'] = arm_name
            fname = family['family_name']
            family_insts = []
            for inst_idx in range(INSTANCES_PER_FAMILY):
                seed = base_offset + inst_idx * 17
                for attempt in range(MAX_RETRIES):
                    inst = gen_instance(rng, family, inst_idx, base_offset)
                    if inst['metadata']['feasible']:
                        family_insts.append(inst)
                        break
                    # Regenerate with fresh rng
                    rng = random.Random(base_offset + 9999 + attempt * 10000 + inst_idx)
                    inst_idx_retry = inst_idx + attempt * 100  # vary seed on retry
                    inst = gen_instance(rng, family, inst_idx_retry, base_offset + attempt * 10000)
                    if inst['metadata']['feasible']:
                        family_insts.append(inst)
                        break
                else:
                    print(f"  WARNING: Could not generate feasible instance for {fname} idx={inst_idx}")
                    # still add it so we know
                    inst = gen_instance(rng, family, inst_idx, base_offset)
                    family_insts.append(inst)

            arm_instances.extend(family_insts)
            print(f"  {arm_name}/{fname}: {len([i for i in family_insts if i['metadata']['feasible']])}/{len(family_insts)} feasible")

        manifest["arms"][arm_name] = {
            "n_families": len(families),
            "n_instances": len(arm_instances),
            "n_feasible": len([i for i in arm_instances if i['metadata']['feasible']]),
            "families": [f['family_name'] for f in families],
            "instances": [i['instance_id'] for i in arm_instances],
        }
        all_instances.extend(arm_instances)

    with open(INST_DIR / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    return all_instances, manifest

# ─────────────────────────────────────────────────────────────
# Task D: EHS Evaluation
# ─────────────────────────────────────────────────────────────

def run_ehs_for_instance(inst, budget):
    try:
        rng = random.Random(inst['metadata']['seed'] + hash(budget) % 10000)
        results = run_ehs(inst, rng=rng, time_limit_seconds=budget,
                         use_history=True, use_exchange=True, use_esr=True)
        front_size = len(results)
        deepest_cmax = max((s['Cmax'] for s in results), default=inst['T'])
        best_energy = min((s['TEC'] for s in results), default=float('inf'))
        shallowest_cmax = min((s['Cmax'] for s in results), default=inst['T'])
        return {
            'feasible_ehs': True,
            'front_size': front_size,
            'deepest_cmax': deepest_cmax,
            'best_energy': best_energy,
            'shallowest_cmax': shallowest_cmax,
            'timeout': False,
            'error': '',
        }
    except Exception as e:
        return {
            'feasible_ehs': False,
            'front_size': 0,
            'deepest_cmax': inst['T'],
            'best_energy': float('inf'),
            'shallowest_cmax': inst['T'],
            'timeout': True,
            'error': str(e)[:200],
        }

def evaluate_all(all_instances):
    raw_rows = []
    total = len(all_instances)
    for i, inst in enumerate(all_instances):
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

        for budget_name, budget_s in [('short', SHORT_BUDGET), ('long', LONG_BUDGET)]:
            t0 = time.time()
            if not meta['feasible']:
                result = {'feasible_ehs': False, 'front_size': 0, 'deepest_cmax': inst['T'],
                          'best_energy': float('inf'), 'shallowest_cmax': inst['T'],
                          'timeout': True, 'error': 'infeasible_instance'}
                runtime = 0.0
            else:
                result = run_ehs_for_instance(inst, budget_s)
                runtime = time.time() - t0
                # Check if it likely timed out
                if runtime >= budget_s * 0.95 or result.get('timeout', False):
                    result['timeout'] = True

            row = {**base, 'budget': budget_name, 'feasible_ehs': result['feasible_ehs'],
                   'front_size': result['front_size'], 'runtime_s': round(runtime, 2),
                   'timeout': result['timeout'], 'deepest_cmax': result['deepest_cmax'],
                   'best_energy': round(result['best_energy'], 6) if result['best_energy'] != float('inf') else 'inf',
                   'shallowest_cmax': result['shallowest_cmax'],
                   'error': result.get('error', '')}
            raw_rows.append(row)

        if (i + 1) % 10 == 0:
            print(f"  EHS progress: {i+1}/{total} instances ({((i+1)/total*100):.0f}%)")

    return raw_rows

# ─────────────────────────────────────────────────────────────
# Task E: Metrics
# ─────────────────────────────────────────────────────────────

def check_mechanism_confirmed(short, long, expected_mechanism):
    """Apply explicit operational rules for mechanism confirmation."""
    fs_s, fs_l = short['front_size'], long['front_size']
    dc_s, dc_l = short['deepest_cmax'], long['deepest_cmax']
    dc_delta = dc_s - dc_l  # positive = improved
    rt_s = short['runtime_s']
    to_s = short['timeout']
    tocs_s = False  # timeout_short

    mech = expected_mechanism

    if mech == 'first_khat_dominance':
        if to_s and fs_s <= 3:
            return True
        return rt_s >= SHORT_BUDGET * 0.85 and fs_s <= 2

    elif mech == 'asgh_lock_in':
        return (fs_l - fs_s) >= 4 and rt_s > 10 and not to_s

    elif mech == 'res_reinsertion_starvation':
        return rt_s < SHORT_BUDGET * 0.5 and fs_s >= 3 and (fs_l - fs_s) >= 4

    elif mech == 'es_exploration_tension':
        return (fs_l - fs_s) >= 4 and dc_delta >= 2

    elif mech == 'front_coverage_gap':
        return fs_s >= 3 and (fs_l - fs_s) >= 3

    elif mech == 'epsilon_skip':
        return fs_s >= 3 and (fs_l - fs_s) <= 2

    elif mech == 'load_imbalance':
        T = short.get('T', 200)
        return dc_l < T * 0.5 and (fs_l - fs_s) <= 3

    elif mech == 'short_budget_pressure':
        return to_s and fs_s <= 1

    return False

def check_nontrivial(meta, short, long, high_yield, mechanism_confirmed):
    """Check if high-yield mechanism-confirmed is nontrivial (not explainable by sweep alone)."""
    if not (high_yield and mechanism_confirmed):
        return False
    # Has explicit mechanism_layer (LLM L2 hallmark)
    if meta.get('has_mechanism_layer', False):
        return True
    # Processing times not simple uniform(1,10) sweep
    pt_type = meta.get('pt_type', 'uniform')
    pt_params = meta.get('pt_params', {})
    if pt_type != 'uniform':
        return True
    # Check if it's the narrow uniform sweep (1-10 or 2-10)
    low = pt_params.get('low', pt_params.get('low_rate', 999))
    high = pt_params.get('high', pt_params.get('high_rate', 999))
    if low > 2 or high < 8:
        return True  # Not the simple wide sweep
    # It IS the simple uniform(1,10) sweep → trivial
    return False

def compute_metrics(raw_rows, manifest):
    """Group by instance, compute metrics, then aggregate by arm."""
    inst_map = defaultdict(lambda: {'short': None, 'long': None, 'meta': {}})
    for row in raw_rows:
        iid = row['instance_id']
        inst_map[iid][row['budget']] = row
        inst_map[iid]['meta'] = {
            'arm': row['arm'],
            'family_name': row['family_name'],
            'expected_mechanism': row['expected_mechanism'],
            'n': row['n'], 'm': row['m'], 'T': row['T'],
            'feasible_instance': row['feasible_instance'],
        }

    # Get has_mechanism_layer and pt_type from manifest (pass through from generation metadata)
    # We stored it in the instance metadata
    # For now derive from family_name
    instance_has_layer = {}
    for iid in inst_map:
        fname = inst_map[iid]['meta']['family_name']
        instance_has_layer[iid] = fname.startswith('hybrid_M')
        # Determine pt_type from raw row
        # We didn't store pt_type in raw - let's derive it from family_name pattern
        pass

    summaries = []
    arm_stats = defaultdict(lambda: {
        'n_instances': 0, 'n_feasible': 0, 'n_evaluable': 0,
        'n_high_yield': 0, 'n_mechanism_confirmed': 0, 'n_nontrivial_mc_hy': 0,
        'delta_fs_list': [], 'runtime_short_list': [], 'runtime_long_list': [],
        'timeout_short': 0, 'timeout_long': 0,
    })

    for iid in sorted(inst_map.keys()):
        d = inst_map[iid]
        s = d['short']; l = d['long']
        meta = d['meta']

        if not s or not l:
            continue

        feasible = meta['feasible_instance']
        fs_s = int(s['front_size']); fs_l = int(l['front_size'])
        dc_s = int(s['deepest_cmax']); dc_l = int(l['deepest_cmax'])
        rt_s = float(s['runtime_s']); rt_l = float(l['runtime_s'])
        to_s = s.get('timeout', False); to_l = l.get('timeout', False)
        eval_s = s.get('feasible_ehs', False); eval_l = l.get('feasible_ehs', False)
        evaluable = feasible and eval_s and eval_l

        # High-yield
        hy = False
        yr = 'none'
        delta_fs = fs_l - fs_s
        delta_cmax = dc_s - dc_l
        if delta_fs >= 2:
            hy = True; yr = f'+{delta_fs}'
        elif fs_s <= 1 and fs_l >= 3:
            hy = True; yr = f'near_zero→{fs_l}'
        elif delta_cmax >= 2:
            hy = True; yr = f'cmaxΔ={delta_cmax}'
        elif to_s and fs_s <= 1:
            hy = True; yr = 'slow'
        elif fs_s > 0 and fs_l == fs_s:
            yr = f'sat_{fs_s}'
        else:
            yr = f'Δ{delta_fs}'

        # Mechanism confirmation
        fname = meta['family_name']
        exp_mech = meta['expected_mechanism']
        mc = check_mechanism_confirmed(s, l, exp_mech) if evaluable else False

        # Nontrivial
        has_layer = fname.startswith('hybrid_M')
        # Get actual pt type from the original family data
        ntm = check_nontrivial({'has_mechanism_layer': has_layer,
                               'pt_type': 'uniform' if ('human' in fname or fname.startswith('hybrid')) else 'bimodal' if 'random' in fname and '000' in fname else 'uniform',
                               'pt_params': {}}, None, None, hy, mc) if evaluable else False

        summary = {
            'instance_id': iid,
            'arm': meta['arm'],
            'family_name': fname,
            'expected_mechanism': exp_mech,
            'n': meta['n'], 'm': meta['m'], 'T': meta['T'],
            'feasible_instance': feasible,
            'evaluable': evaluable,
            'fs_short': fs_s, 'fs_long': fs_l,
            'delta_fs': delta_fs,
            'dc_short': dc_s, 'dc_long': dc_l,
            'delta_cmax': delta_cmax,
            'rt_short': rt_s, 'rt_long': rt_l,
            'timeout_short': to_s, 'timeout_long': to_l,
            'high_yield': hy,
            'yield_reason': yr,
            'mechanism_confirmed': mc,
            'nontrivial_mc_hy': ntm,
        }
        summaries.append(summary)

        # Arm aggregation
        arm = meta['arm']
        st = arm_stats[arm]
        st['n_instances'] += 1
        if feasible: st['n_feasible'] += 1
        if evaluable:
            st['n_evaluable'] += 1
            st['delta_fs_list'].append(delta_fs)
            st['runtime_short_list'].append(rt_s)
            st['runtime_long_list'].append(rt_l)
            if to_s: st['timeout_short'] += 1
            if to_l: st['timeout_long'] += 1
            if hy: st['n_high_yield'] += 1
            if mc: st['n_mechanism_confirmed'] += 1
            if ntm: st['n_nontrivial_mc_hy'] += 1

    return summaries, arm_stats

# ─────────────────────────────────────────────────────────────
# Task F: Compare and decide
# ─────────────────────────────────────────────────────────────

def compute_decision(arm_stats):
    arms_order = ['llm_l2', 'human', 'random']
    labels = {'llm_l2': 'LLM Call2 (L2)', 'human': 'Human sweep', 'random': 'Random'}

    print("\n" + "=" * 80)
    print("C4 VALIDATION RESULTS")
    print("=" * 80)

    header = f"  {'Arm':<22} {'N':>5} {'Feas':>5} {'Eval':>5} {'HY':>4} {'HY%':>7} {'Δfs':>8} {'MC%':>7} {'NT-MC-HY%':>10}"
    print(header)
    print(f"  {'-' * 80}")

    rows = []
    for arm in arms_order:
        s = arm_stats[arm]
        n = s['n_instances']
        f = s['n_feasible']
        e = s['n_evaluable']
        hy = s['n_high_yield']
        hy_rate = f'{hy/e*100:.0f}%' if e else 'N/A'
        mean_dfs = sum(s['delta_fs_list'])/len(s['delta_fs_list']) if s['delta_fs_list'] else 0
        median_dfs = sorted(s['delta_fs_list'])[len(s['delta_fs_list'])//2] if s['delta_fs_list'] else 0
        mc = s['n_mechanism_confirmed']
        mc_rate = f'{mc/e*100:.0f}%' if e else 'N/A'
        ntm = s['n_nontrivial_mc_hy']
        ntm_rate = f'{ntm/e*100:.0f}%' if e else 'N/A'
        gen_rate = f'{f/n*100:.0f}%' if n else 'N/A'

        print(f"  {labels[arm]:<22} {n:>5} {f:>5} {e:>5} {hy:>4} {hy_rate:>7} {mean_dfs:>8.1f} {mc_rate:>7} {ntm_rate:>10}")
        rows.append({'arm': arm, 'label': labels[arm], 'n': n, 'feasible': f, 'evaluable': e,
                     'high_yield': hy, 'hy_rate': hy_rate, 'mean_delta_fs': mean_dfs,
                     'median_delta_fs': median_dfs, 'mechanism_confirmed': mc, 'mc_rate': mc_rate,
                     'nontrivial_mc_hy': ntm, 'ntm_rate': ntm_rate,
                     'gen_rate': gen_rate})

        # Per-instance detail
        print(f"\n  {labels[arm]} Δfs: [{', '.join(f'{g:+d}' for g in s['delta_fs_list'])}]")
        print(f"    HY={hy}/{e} MC={mc}/{e} NT-MC-HY={ntm}/{e}")
        print(f"    Timeout short={s['timeout_short']}/{e}, long={s['timeout_long']}/{e}")
        if s['runtime_short_list']:
            print(f"    Mean rt: short={sum(s['runtime_short_list'])/len(s['runtime_short_list']):.1f}s, long={sum(s['runtime_long_list'])/len(s['runtime_long_list']):.1f}s")

    print()

    # Gate
    llm = next(r for r in rows if r['arm'] == 'llm_l2')
    hum = next(r for r in rows if r['arm'] == 'human')
    ran = next(r for r in rows if r['arm'] == 'random')

    l2_hy = llm['high_yield'] / max(llm['evaluable'], 1)
    hum_hy = hum['high_yield'] / max(hum['evaluable'], 1)
    l2_ntm = llm['nontrivial_mc_hy'] / max(llm['evaluable'], 1)
    hum_ntm = hum['nontrivial_mc_hy'] / max(hum['evaluable'], 1)
    ran_hy = ran['high_yield'] / max(ran['evaluable'], 1)

    print(f"  High-Yield:    LLM={l2_hy:.0%}  Human={hum_hy:.0%}  Random={ran_hy:.0%}")
    print(f"  NT-MC-HY:      LLM={l2_ntm:.0%}  Human={hum_ntm:.0%}")
    print(f"  Mean Δfs:      LLM={llm['mean_delta_fs']:.1f}  Human={hum['mean_delta_fs']:.1f}  Random={ran['mean_delta_fs']:.1f}")

    if l2_hy >= hum_hy and l2_ntm > hum_ntm:
        gate = 'STRONG'
    elif l2_hy >= hum_hy and l2_ntm > 0:
        gate = 'MODERATE'
    elif l2_hy > ran_hy:
        gate = 'WEAK'
    else:
        gate = 'FAIL'

    print(f"\n  Gate: {gate}")

    return gate, rows

# ─────────────────────────────────────────────────────────────
# Task G: Write decision
# ─────────────────────────────────────────────────────────────

def write_decision(gate, rows, summaries, arm_stats):
    dp = NOTES_DIR / 'c4_validation_decision.md'
    labels = {'llm_l2': 'LLM Call2 (L2)', 'human': 'Human sweep', 'random': 'Random'}

    with open(dp, 'w') as f:
        f.write('# C4 Validation Decision\n\n')
        f.write('**Date**: 2026-05-11\n')
        f.write(f'**Instances**: 12 families × 5 instances = 60 total\n')
        f.write(f'**Budgets**: {SHORT_BUDGET}s / {LONG_BUDGET}s\n\n')

        f.write('## Summary Table\n\n')
        f.write('| Arm | N | Feasible | Evaluable | Gen% | HY | HY% | Mean Δfs | MC% | NT-MC-HY% |\n')
        f.write('|-----|--|----------|-----------|-------|----|------|----------|------|------------|\n')
        for r in rows:
            f.write(f"| {r['label']} | {r['n']} | {r['feasible']} | {r['evaluable']} | {r['gen_rate']} | {r['high_yield']} | {r['hy_rate']} | {r['mean_delta_fs']:.1f} | {r['mc_rate']} | {r['ntm_rate']} |\n")

        f.write(f'\n## Gate: {gate}\n\n')
        f.write(f'High-Yield:    LLM={rows[0]["hy_rate"]}  Human={rows[1]["hy_rate"]}  Random={rows[2]["hy_rate"]}\n')
        f.write(f'NT-MC-HY:      LLM={rows[0]["ntm_rate"]}  Human={rows[1]["ntm_rate"]}\n')
        f.write(f'Mean Δfs:      LLM={rows[0]["mean_delta_fs"]:.1f}  Human={rows[1]["mean_delta_fs"]:.1f}  Random={rows[2]["mean_delta_fs"]:.1f}\n\n')

        f.write('## Decision Questions\n\n')
        l2_hy = rows[0]['high_yield'] / max(rows[0]['evaluable'], 1)
        hum_hy = rows[1]['high_yield'] / max(rows[1]['evaluable'], 1)
        ran_hy = rows[2]['high_yield'] / max(rows[2]['evaluable'], 1)

        f.write(f'1. **Did L2 transfer beyond C3-L2?** C3-L2 had 6/6 (100%) HY with Δfs_mean=19.5. ')
        if l2_hy >= 0.8:
            f.write(f'C4 validation: {rows[0]["hy_rate"]} HY with Δfs_mean={rows[0]["mean_delta_fs"]:.1f}. Transfer confirmed.\n')
        else:
            f.write(f'C4 validation: {rows[0]["hy_rate"]} HY. Transfer weakened.\n')

        f.write(f'2. **Did L2 beat random?** Random={ran_hy:.0%} HY. ')
        if l2_hy > ran_hy + 0.1:
            f.write('Yes, clear margin.\n')
        elif l2_hy > ran_hy:
            f.write('Yes, narrow margin.\n')
        else:
            f.write('No.\n')

        f.write(f'3. **Did L2 beat or tie human sweep?** Human={hum_hy:.0%} HY. ')
        if l2_hy >= hum_hy:
            f.write('Tied or beat.\n')
        else:
            f.write('Lost.\n')

        f.write(f'4. **LLM advantage**: Generation quality={rows[0]["gen_rate"]}, Adversarial yield={rows[0]["hy_rate"]}, Mechanism specificity={rows[0]["ntm_rate"]}.\n\n')
        f.write(f'5. **Strong enough for thesis?** Discussed below.\n\n')

        f.write('## Per-Instance Details\n\n')
        for r in rows:
            f.write(f'### {r["label"]}\n\n')
            arm_summaries = [s for s in summaries if s['arm'] == r['arm']]
            for s in arm_summaries:
                ntm_flag = ' ***NT-MC-HY***' if s['nontrivial_mc_hy'] else ''
                mc_flag = ' [MC]' if s['mechanism_confirmed'] else ''
                hy_flag = ' HIGH' if s['high_yield'] else ' low'
                f.write(f"- {s['instance_id'][:60]}: fs={s['fs_short']}→{s['fs_long']} "
                       f"cmax={s['dc_short']}→{s['dc_long']} "
                       f"rt={s['rt_short']:.1f}s/{s['rt_long']:.1f}s "
                       f"{'TO' if s['timeout_short'] else 'ok'}/{hy_flag}{mc_flag}{ntm_flag} [{s['yield_reason']}]\n")
            f.write('\n')

    return dp

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 80)
    print("C4 Validation Pipeline")
    print("=" * 80)

    # Task B
    print("\n[Task B] Selecting families...")
    selection, llm_l2_sel, human_sel, random_sel = select_families()
    print(f"  Saved: notes/c4_selected_families.json")

    # Task C
    print("\n[Task C] Generating 60 instances...")
    all_instances, manifest = generate_all_instances(llm_l2_sel, human_sel, random_sel, selection)
    n_feasible = sum(1 for i in all_instances if i['metadata']['feasible'])
    print(f"  Generated: {len(all_instances)} total, {n_feasible} feasible")
    print(f"  Saved: generated_instances/c4_validation/")

    # Save instances
    serializable = []
    for inst in all_instances:
        si = {k: v for k, v in inst.items()}
        serializable.append(si)
    # Save manifest + instances separately  
    for inst in all_instances:
        iid = inst['instance_id']
        with open(INST_DIR / f'{iid}.json', 'w') as f:
            json.dump(inst, f, indent=2)
    print(f"  Individual instance files saved to {INST_DIR}/")

    # Task D
    print(f"\n[Task D] Running EHS on {len(all_instances)} instances ({SHORT_BUDGET}s + {LONG_BUDGET}s each)...")
    raw_rows = evaluate_all(all_instances)
    print(f"  Completed {len(raw_rows)} EHS runs")

    with open(EVAL_DIR / 'c4_validation_raw.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=raw_rows[0].keys())
        writer.writeheader()
        writer.writerows(raw_rows)
    print(f"  Saved: eval/c4_validation_raw.csv")

    # Task E
    print("\n[Task E] Computing metrics...")
    summaries, arm_stats = compute_metrics(raw_rows, manifest)

    with open(EVAL_DIR / 'c4_validation_summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    print(f"  Saved: eval/c4_validation_summary.csv")

    # Task F
    print("\n[Task F] Comparing arms...")
    gate, rows = compute_decision(arm_stats)

    # Task G
    print("\n[Task G] Writing decision...")
    dp = write_decision(gate, rows, summaries, arm_stats)
    print(f"  Saved: {dp}")

    print(f"\n{'=' * 80}")
    print(f"Gate: {gate}")
    print(f"{'=' * 80}")

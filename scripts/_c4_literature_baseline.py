#!/usr/bin/env python3
"""C4-Lit: Literature baseline + framing revision.

Creates literature_generators.json, generates instances, evaluates EHS,
reframes the comparison with agent_manual_sweep as internal control.
"""

import csv, json, os, sys, time, random, math
from pathlib import Path
from collections import defaultdict
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ITER_DIR = PROJECT_ROOT / 'research/glns_llm_heuristic_20260422/iterations/20260510_phaseC_adversarial_ehs_benchmark_design'
FAMILIES_DIR = ITER_DIR / 'families'
NOTES_DIR = ITER_DIR / 'notes'
EVAL_DIR = ITER_DIR / 'eval'
INST_DIR = ITER_DIR / 'generated_instances' / 'c4_literature_baseline'
os.makedirs(INST_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(NOTES_DIR, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))
from glns.paper_heuristics import run_ehs

SHORT_BUDGET = 30
LONG_BUDGET = 90
MAX_N_CAP = 150
MAX_T_CAP = 200
INSTANCES_PER_GENERATOR = 20

# ═══════════════════════════════════════════════════════════════
# TASK A+B: Literature generators JSON
# ═══════════════════════════════════════════════════════════════

LITERATURE_GENERATORS = {
    "description": "Literature-derived benchmark generators for Phase C baseline comparison.",
    "date": "2026-05-11",
    "generators": [
        {
            "generator": "wang_mls_generator",
            "source": "Wang et al. (2018) Journal of Cleaner Production 193; Gaggero et al. (2023) EJOR 311. Instances 31-60 (medium-scale).",
            "hypothesis": "Medium-scale instances from the published benchmark family. p_j small (1-4), discrete machine rates and TOU prices, standard literature parameter ranges.",
            "description": "Medium-scale Wang benchmark family, n in {30, 60, 100, 150}, capped T to 200 for Phase C compatibility.",
            "family_name": "wang_literature_medium",
            "n_jobs_range": {"min": 30, "max": 150},
            "m_machines_range": {"min": 8, "max": 25},
            "horizon_T_range": {"min": 100, "max": 200},
            "processing_time_distribution": {
                "type": "uniform",
                "params": {"low": 1, "high": 4}
            },
            "machine_rate_distribution": {
                "type": "discrete_uniform",
                "params": {"values": [1, 2, 3]}
            },
            "TOU_price_profile_type": "discrete_uniform",
            "TOU_price_profile_params": {"values": [1, 2, 3, 4]},
            "epsilon_regime": "medium",
            "expected_EHS_failure_mechanism": "mixed",
            "expected_EHS_failure_mechanism_evidence": "Published benchmark family. Medium-scale instances designed to test solver scalability.",
            "generated_instances_count": INSTANCES_PER_GENERATOR,
            "validity_constraints": {
                "feasibility_guarantee": True,
                "min_total_work": 100,
                "n_per_machine_min": 3
            },
            "rejection_conditions": [
                "sum(p_j) > m*T",
                "all e[h] equal",
                "all ct[t] equal",
                "n < 2*m"
            ],
            "seed_behavior": {
                "base_seed": 500000,
                "expected_seed_variance": "medium"
            }
        },
        {
            "generator": "anghinolfi_vls_capped",
            "source": "Anghinolfi et al. (2021); Gaggero et al. (2023) EJOR 311. Instances 61-90 (large-scale). CAPPED for Phase C compatibility.",
            "hypothesis": "Capped VLS-style instances. Larger p_j spread (1-12), more machines, richer TOU, but n/T capped for Phase C evaluation budget.",
            "description": "CAPPED VLS variant: n=60-150, m=8-25, T=100-200, p_j~U[1,12], e_h~U[1,6], c_t~U[1,8]. NOT the original VLS benchmark.",
            "family_name": "anghinolfi_vls_capped",
            "n_jobs_range": {"min": 60, "max": 150},
            "m_machines_range": {"min": 8, "max": 25},
            "horizon_T_range": {"min": 100, "max": 200},
            "processing_time_distribution": {
                "type": "uniform",
                "params": {"low": 1, "high": 12}
            },
            "machine_rate_distribution": {
                "type": "continuous_uniform",
                "params": {"min_rate": 1.0, "max_rate": 6.0}
            },
            "TOU_price_profile_type": "continuous_uniform",
            "TOU_price_profile_params": {"min_price": 1.0, "max_price": 8.0},
            "epsilon_regime": "medium",
            "expected_EHS_failure_mechanism": "mixed",
            "expected_EHS_failure_mechanism_evidence": "Larger p_j spread and wider e/c ranges than Wang. Should produce more diverse and harder scheduling problems.",
            "generated_instances_count": INSTANCES_PER_GENERATOR,
            "validity_constraints": {
                "feasibility_guarantee": True,
                "min_total_work": 150,
                "n_per_machine_min": 3
            },
            "rejection_conditions": [
                "sum(p_j) > m*T",
                "all e[h] equal",
                "all ct[t] equal",
                "n < 2*m"
            ],
            "seed_behavior": {
                "base_seed": 510000,
                "expected_seed_variance": "medium"
            }
        }
    ]
}

# ═══════════════════════════════════════════════════════════════
# Instance generation (literature generators)
# ═══════════════════════════════════════════════════════════════

def _si(rng, lo, hi):
    return rng.randint(lo, hi) if lo <= hi else lo

def gen_wang_instance(rng, gen, seed):
    """Generate one instance from the Wang literature generator."""
    # Select parameter combination randomly
    n_options = [30, 60, 100, 150]
    m_options = [8, 16, 25]
    T_options = [100, 200]  # capped from [100, 300]

    n = n_options[rng.randint(0, len(n_options)-1)]
    m = m_options[rng.randint(0, len(m_options)-1)]
    T = T_options[rng.randint(0, len(T_options)-1)]

    p = [_si(rng, 1, 4) for _ in range(n)]
    # Discrete rates from {1, 2, 3}
    e = [float(rng.choice([1, 2, 3])) for _ in range(m)]
    # Discrete TOU prices from {1, 2, 3, 4}
    ct = [float(rng.choice([1, 2, 3, 4])) for _ in range(T)]

    feasible = sum(p) <= m * T
    iid = f"c4lit_wang_mls_{seed}"
    return {
        "n": n, "m": m, "T": T, "p": p, "e": e, "ct": ct,
        "instance_id": iid,
        "metadata": {
            "arm": "literature",
            "family_name": "wang_literature_medium",
            "generator": "wang_mls_generator",
            "expected_mechanism": "mixed",
            "seed": seed,
            "feasible": feasible,
            "sum_p": sum(p),
            "is_literature_baseline": True,
            "source": "Wang et al. (2018) medium-scale, capped",
        }
    }

def gen_anghinolfi_instance(rng, gen, seed):
    """Generate one capped Anghinolfi-style instance."""
    n = _si(rng, 60, 150)
    m = _si(rng, 8, 25)
    T = _si(rng, 100, 200)

    p = [_si(rng, 1, 12) for _ in range(n)]
    e = [round(rng.uniform(1.0, 6.0), 4) for _ in range(m)]
    ct = [round(rng.uniform(1.0, 8.0), 4) for _ in range(T)]

    feasible = sum(p) <= m * T
    iid = f"c4lit_anghinolfi_capped_{seed}"
    return {
        "n": n, "m": m, "T": T, "p": p, "e": e, "ct": ct,
        "instance_id": iid,
        "metadata": {
            "arm": "literature",
            "family_name": "anghinolfi_vls_capped",
            "generator": "anghinolfi_vls_capped",
            "expected_mechanism": "mixed",
            "seed": seed,
            "feasible": feasible,
            "sum_p": sum(p),
            "is_literature_baseline": True,
            "is_capped": True,
            "source": "Anghinolfi et al. (2021) VLS, capped for Phase C",
        }
    }

def generate_literature_instances():
    """Generate instances from both literature generators."""
    rng = random.Random(500000 + 9999)
    all_insts = []

    # Wang: 20 instances
    for i in range(INSTANCES_PER_GENERATOR):
        seed = 500000 + i * 17
        for attempt in range(100):
            inst = gen_wang_instance(rng, LITERATURE_GENERATORS['generators'][0], seed)
            if inst['metadata']['feasible']:
                all_insts.append(inst)
                break
            seed = 500000 + attempt * 10000 + i
        else:
            inst['instance_id'] += f"_retry{attempt}"
            all_insts.append(inst)

    # Anghinolfi capped: 20 instances
    for i in range(INSTANCES_PER_GENERATOR):
        seed = 510000 + i * 17
        for attempt in range(100):
            inst = gen_anghinolfi_instance(rng, LITERATURE_GENERATORS['generators'][1], seed)
            if inst['metadata']['feasible']:
                all_insts.append(inst)
                break
            seed = 510000 + attempt * 10000 + i
        else:
            inst['instance_id'] += f"_retry{attempt}"
            all_insts.append(inst)

    return all_insts

# ═══════════════════════════════════════════════════════════════
# EHS Evaluation
# ═══════════════════════════════════════════════════════════════

def eval_one(inst, budget_s):
    try:
        t0 = time.time()
        rng = random.Random(inst['metadata']['seed'] + hash(str(budget_s)) % 10000)
        results = run_ehs(inst, rng=rng, time_limit_seconds=budget_s,
                         use_history=True, use_exchange=True, use_esr=True)
        rt = time.time() - t0
        fs = len(results)
        dc = max((s.cmax for s in results), default=inst['T'])
        be = min((s.energy for s in results), default=float('inf'))
        sc = min((s.cmax for s in results), default=inst['T'])
        to = rt >= budget_s * 0.95
        return (True, fs, round(rt, 2), to, dc, round(be, 6) if be != float('inf') else 'inf', sc, '')
    except Exception as e:
        return (False, 0, 0.0, True, inst['T'], 'inf', inst['T'], str(e)[:200])

def evaluate_all(instances):
    rows = []
    for i, inst in enumerate(instances):
        meta = inst['metadata']
        base = {
            'instance_id': inst['instance_id'],
            'arm': 'literature',
            'family_name': meta['family_name'],
            'expected_mechanism': meta['expected_mechanism'],
            'n': inst['n'], 'm': inst['m'], 'T': inst['T'],
            'sum_p': meta['sum_p'],
            'feasible_instance': meta['feasible'],
        }
        for bn, bs in [('short', SHORT_BUDGET), ('long', LONG_BUDGET)]:
            fe, fs, rt, to, dc, be, sc, err = eval_one(inst, bs)
            rows.append({**base, 'budget': bn, 'feasible_ehs': fe, 'front_size': fs,
                        'runtime_s': rt, 'timeout': to, 'deepest_cmax': dc,
                        'best_energy': be, 'shallowest_cmax': sc, 'error': err})
        if (i+1) % 5 == 0:
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] {i+1}/{len(instances)} instances")

    # Save
    with open(EVAL_DIR / 'c4_literature_baseline_raw.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows

# ═══════════════════════════════════════════════════════════════
# Load previous C4 data for comparison
# ═══════════════════════════════════════════════════════════════

def load_c4_summary():
    path = EVAL_DIR / 'c4_validation_summary.csv'
    if path.exists():
        with open(path) as f:
            return list(csv.DictReader(f))
    return []

# ═══════════════════════════════════════════════════════════════
# Metrics + Decision
# ═══════════════════════════════════════════════════════════════

def _tobool(v):
    if isinstance(v, bool): return v
    if isinstance(v, str): return v.lower() in ('true', '1', 'yes')
    return bool(v)

def compute_summaries(raw_rows):
    inst_map = defaultdict(lambda: {'short': None, 'long': None})
    for row in raw_rows:
        inst_map[row['instance_id']][row['budget']] = row

    summaries = []
    for iid in sorted(inst_map.keys()):
        d = inst_map[iid]; s = d['short']; l = d['long']
        if not s or not l: continue

        feasible = _tobool(s['feasible_instance'])
        eval_s = _tobool(s['feasible_ehs']); eval_l = _tobool(l['feasible_ehs'])
        ev = feasible and eval_s and eval_l

        fs_s = int(s['front_size']); fs_l = int(l['front_size'])
        dc_s = int(s['deepest_cmax']); dc_l = int(l['deepest_cmax'])
        rt_s = float(s['runtime_s']); rt_l = float(l['runtime_s'])
        to_s_str = s.get('timeout', 'False')
        to_s = _tobool(to_s_str)
        df = fs_l - fs_s; dd = dc_s - dc_l

        hy = False; yr = 'none'
        if df >= 2: hy = True; yr = f'+{df}'
        elif fs_s <= 1 and fs_l >= 3: hy = True; yr = f'near_zero→{fs_l}'
        elif dd >= 2: hy = True; yr = f'cmaxΔ={dd}'
        elif to_s and fs_s <= 1: hy = True; yr = 'slow'
        elif fs_s > 0 and fs_l == fs_s: yr = f'sat_{fs_s}'
        else: yr = f'Δ{df}'

        summaries.append({
            'instance_id': iid, 'arm': s['arm'], 'family_name': s['family_name'],
            'n': int(s['n']), 'm': int(s['m']), 'T': int(s['T']),
            'feasible_instance': feasible, 'evaluable': ev,
            'fs_short': fs_s, 'fs_long': fs_l, 'delta_fs': df,
            'dc_short': dc_s, 'dc_long': dc_l, 'delta_cmax': dd,
            'rt_short': rt_s, 'rt_long': rt_l,
            'timeout_short': to_s,
            'high_yield': hy, 'yield_reason': yr,
        })
    return summaries

def arm_stats_from_summaries(summaries):
    st = defaultdict(lambda: {'n_inst': 0, 'n_feas': 0, 'n_eval': 0, 'n_hy': 0,
                              'dfs': [], 'rts': [], 'rtl': [], 'tos': 0, 'tol': 0})
    for s in summaries:
        arm = s['arm']; st[arm]['n_inst'] += 1
        if _tobool(s.get('feasible_instance', False)): st[arm]['n_feas'] += 1
        if _tobool(s.get('evaluable', False)):
            st[arm]['n_eval'] += 1
            st[arm]['dfs'].append(int(s['delta_fs']))
            st[arm]['rts'].append(float(s['rt_short']))
            st[arm]['rtl'].append(float(s['rt_long']))
            if _tobool(s.get('timeout_short', False)): st[arm]['tos'] += 1
            if _tobool(s.get('high_yield', False)): st[arm]['n_hy'] += 1
    return st

# ═══════════════════════════════════════════════════════════════
# TASK D: Reframed comparison
# ═══════════════════════════════════════════════════════════════

def build_reframed_table(lit_summaries, c4_summaries):
    """Build the main comparison table with literature baseline, and side table with agent_manual_sweep."""

    # Classify C4 summaries: rename "human" → "agent_manual_sweep"
    reframed = []
    for s in c4_summaries:
        s2 = dict(s)
        if s2['arm'] == 'human':
            s2['arm'] = 'agent_manual_sweep'
            s2['display_arm'] = 'agent_manual_sweep'
        elif s2['arm'] == 'llm_l2':
            s2['display_arm'] = 'LLM Call2'
        elif s2['arm'] == 'random':
            s2['display_arm'] = 'Random (schema)'
        reframed.append(s2)

    # Literature summaries
    for s in lit_summaries:
        s2 = dict(s)
        s2['display_arm'] = 'Literature (Wang+Anghi)'
        reframed.append(s2)

    # Build arm stats
    st = defaultdict(lambda: {'n_inst': 0, 'n_feas': 0, 'n_eval': 0, 'n_hy': 0,
                              'dfs': [], 'rts': [], 'rtl': [], 'tos': 0, 'tol': 0})
    for s in reframed:
        arm = s['arm']; st[arm]['n_inst'] += 1
        if _tobool(s.get('feasible_instance', False)): st[arm]['n_feas'] += 1
        if _tobool(s.get('evaluable', False)):
            st[arm]['n_eval'] += 1
            st[arm]['dfs'].append(int(s['delta_fs']))
            st[arm]['rts'].append(float(s['rt_short']))
            st[arm]['rtl'].append(float(s['rt_long']))
            if _tobool(s.get('timeout_short', False)): st[arm]['tos'] += 1
            if _tobool(s.get('high_yield', False)): st[arm]['n_hy'] += 1

    return st, reframed

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 80)
    print("C4-Lit: Literature Baseline + Framing Revision")
    print("=" * 80)

    # Task A+B: Save literature generators JSON
    print("\n[Task A+B] Creating literature_generators.json...")
    with open(FAMILIES_DIR / 'literature_generators.json', 'w') as f:
        json.dump(LITERATURE_GENERATORS, f, indent=2)
    print(f"  Saved: families/literature_generators.json")

    print("\n[Naming revision] 'human' → 'agent_manual_sweep' in documentation.")
    print("  'human_sweep_families.json' file preserved (rename in text only).")

    # Task C: Generate instances
    print(f"\n[Task C] Generating {INSTANCES_PER_GENERATOR*2} literature instances...")
    lit_insts = generate_literature_instances()
    n_wang = sum(1 for i in lit_insts if 'wang' in i['metadata']['family_name'])
    n_anghi = sum(1 for i in lit_insts if 'anghinolfi' in i['metadata']['family_name'])
    n_feas = sum(1 for i in lit_insts if i['metadata']['feasible'])
    print(f"  Wang: {n_wang}, Anghinolfi: {n_anghi}, Feasible: {n_feas}/{len(lit_insts)}")

    for inst in lit_insts:
        with open(INST_DIR / f"{inst['instance_id']}.json", 'w') as f:
            json.dump(inst, f, indent=2)

    manifest = {
        "purpose": "C4 Literature baseline validation",
        "date": "2026-05-11",
        "generators": ["wang_mls_generator", "anghinolfi_vls_capped"],
        "source": "families/literature_generators.json",
        "n_instances_per_generator": INSTANCES_PER_GENERATOR,
        "total_instances": len(lit_insts),
        "n_feasible": n_feas,
        "budgets": {"short": SHORT_BUDGET, "long": LONG_BUDGET},
    }
    with open(INST_DIR / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved: {INST_DIR}/")

    # Task C.2: EHS evaluation
    print(f"\n[Task C.2] Running EHS on {len(lit_insts)} literature instances...")
    lit_raw = evaluate_all(lit_insts)
    print(f"  Completed {len(lit_raw)} EHS runs")

    # Compute summaries
    lit_summaries = compute_summaries(lit_raw)
    with open(EVAL_DIR / 'c4_literature_baseline_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(lit_summaries[0].keys()))
        w.writeheader()
        w.writerows(lit_summaries)

    # Load C4 data
    c4_summaries = load_c4_summary()
    print(f"  Loaded {len(c4_summaries)} C4 summaries, {len(lit_summaries)} lit summaries")

    # Task D: Build reframed comparison
    print("\n[Task D] Building reframed comparison tables...")
    st, reframed = build_reframed_table(lit_summaries, c4_summaries)

    # Print comparison
    print("\n" + "=" * 80)
    print("MAIN COMPARISON (literature + LLM + random)")
    print("=" * 80)

    main_arms = ['llm_l2', 'literature', 'random']
    main_labels = {'llm_l2': 'LLM Call2', 'literature': 'Literature (Wang+Anghi)', 'random': 'Random (schema)'}

    print(f"\n  {'Arm':<30} {'N':>5} {'Feas':>5} {'Eval':>5} {'HY':>4} {'HY%':>7} {'MeanΔfs':>8} {'TOs':>6}")
    print(f"  {'-' * 75}")
    for arm in main_arms:
        s = st[arm]
        n = s['n_inst']; f = s['n_feas']; e = s['n_eval']; hy = s['n_hy']
        hy_rate = f'{hy/e*100:.0f}%' if e else 'N/A'
        mdfs = sum(s['dfs'])/len(s['dfs']) if s['dfs'] else 0
        print(f"  {main_labels[arm]:<30} {n:>5} {f:>5} {e:>5} {hy:>4} {hy_rate:>7} {mdfs:>8.1f} {s['tos']:>4}/{e}")

        if s['dfs']:
            print(f"    Δfs: [{', '.join(f'{g:+d}' for g in s['dfs'])}]")

    print(f"\n  SIDE TABLE (internal control):")
    print(f"  {'Arm':<30} {'N':>5} {'Feas':>5} {'Eval':>5} {'HY':>4} {'HY%':>7} {'MeanΔfs':>8} {'TOs':>6}")
    agent = st['agent_manual_sweep'] if 'agent_manual_sweep' in st else st.get('human', None)
    if agent:
        hy_a = agent['n_hy']; e_a = agent['n_eval']
        mdfs_a = sum(agent['dfs'])/len(agent['dfs']) if agent['dfs'] else 0
        print(f"  {'agent_manual_sweep (internal)':<30} {agent['n_inst']:>5} {agent['n_feas']:>5} {e_a:>5} {hy_a:>4} {f'{hy_a/e_a*100:.0f}%' if e_a else 'N/A':>7} {mdfs_a:>8.1f} {agent['tos']:>4}/{e_a}")

    # Gate
    llm = st['llm_l2']; lit = st['literature']; ran = st['random']
    l2_hy = llm['n_hy']/max(llm['n_eval'],1)
    lit_hy = lit['n_hy']/max(lit['n_eval'],1)
    ran_hy = ran['n_hy']/max(ran['n_eval'],1)
    l2_mdfs = sum(llm['dfs'])/len(llm['dfs']) if llm['dfs'] else 0
    lit_mdfs = sum(lit['dfs'])/len(lit['dfs']) if lit['dfs'] else 0

    print(f"\n  High-Yield:  LLM={l2_hy:.0%}  Literature={lit_hy:.0%}  Random={ran_hy:.0%}")
    print(f"  Mean Δfs:    LLM={l2_mdfs:.1f}  Literature={lit_mdfs:.1f}")

    if l2_hy > lit_hy and l2_hy > ran_hy:
        gate = 'STRONG'
    elif l2_hy >= lit_hy and l2_hy > ran_hy:
        gate = 'MODERATE'
    elif l2_hy > ran_hy:
        gate = 'WEAK'
    else:
        gate = 'FAIL'
    print(f"\n  Gate (LLM vs Literature vs Random): {gate}")

    # Task D.3: Write decision
    dp = NOTES_DIR / 'c4_literature_baseline_decision.md'
    with open(dp, 'w') as f:
        f.write('# C4 Literature Baseline Decision\n\n')
        f.write('**Date**: 2026-05-11\n')
        f.write(f'**Literature generators**: `wang_mls_generator` (20 inst) + `anghinolfi_vls_capped` (20 inst)\n')
        f.write(f'**Budgets**: {SHORT_BUDGET}s / {LONG_BUDGET}s\n\n')

        f.write('## Naming Revision\n\n')
        f.write('The previous `human` / `human_sweep` arm has been renamed to **`agent_manual_sweep`** (internal control). ')
        f.write('It was generated during our agentic research workflow, not by an independent human expert or published method. ')
        f.write('It is NOT presented as the main independent baseline. Instead, we use:\n\n')
        f.write('- **Main baselines**: random schema generator + literature-derived generators (Wang 2018, Anghinolfi 2021)\n')
        f.write('- **Internal control**: agent_manual_sweep (reported separately)\n\n')

        f.write('## Main Comparison Table\n\n')
        f.write('| Arm | N | Feasible | Evaluable | HY | HY% | Mean Δfs | TO (short) |\n')
        f.write('|-----|--|----------|-----------|----|-----|----------|------------|\n')
        for arm in main_arms:
            s = st[arm]; e = s['n_eval']; hy = s['n_hy']
            hy_rate = f'{hy/e*100:.0f}%' if e else 'N/A'
            mdfs = sum(s['dfs'])/len(s['dfs']) if s['dfs'] else 0
            f.write(f"| {main_labels[arm]} | {s['n_inst']} | {s['n_feas']} | {e} | {hy} | {hy_rate} | {mdfs:.1f} | {s['tos']}/{e} |\n")

        if agent:
            e_a = agent['n_eval']; hy_a = agent['n_hy']
            mdfs_a = sum(agent['dfs'])/len(agent['dfs']) if agent['dfs'] else 0
            f.write(f"| agent_manual_sweep (internal) | {agent['n_inst']} | {agent['n_feas']} | {e_a} | {hy_a} | {f'{hy_a/e_a*100:.0f}%' if e_a else 'N/A'} | {mdfs_a:.1f} | {agent['tos']}/{e_a} |\n")

        f.write(f'\n## Gate: {gate}\n\n')
        f.write(f'High-Yield:  LLM={l2_hy:.0%}  Literature={lit_hy:.0%}  Random={ran_hy:.0%}\n\n')
        f.write('## Decision Questions\n\n')
        f.write(f'1. **Does LLM beat literature baseline?** Literature={lit_hy:.0%} HY. ')
        if l2_hy > lit_hy: f.write('Yes.\n')
        elif l2_hy >= lit_hy: f.write('Tied.\n')
        else: f.write('No.\n')
        f.write(f'2. **Does LLM beat random?** Random={ran_hy:.0%} HY. ')
        if l2_hy > ran_hy: f.write('Yes.\n')
        else: f.write('No.\n')
        f.write(f'3. **Is the revised framing cleaner?** Yes. Literature benchmark + random are proper external baselines. The agent_manual_sweep is an internal stress-control, not a claims competitor.\n\n')

        f.write('## Per-Instance (Literature)\n\n')
        for s in lit_summaries:
            f.write(f"- {s['instance_id'][:60]}: fs={s['fs_short']}→{s['fs_long']} cmax={s['dc_short']}→{s['dc_long']} rt={s['rt_short']}s/{s['rt_long']}s {'HIGH' if _tobool(s['high_yield']) else 'low'} [{s['yield_reason']}]\n")

    print(f"\n  Decision: {dp}")
    print(f"\n{'='*80}")
    print(f"C4-Lit complete. Gate: {gate}")

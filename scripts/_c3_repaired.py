#!/usr/bin/env python3
"""Phase C3-Regular Repair: generate evaluable random/human baseline instances
+ run fair EHS comparison against frozen LLM instances.

Usage: /opt/miniconda3/envs/new-ml-env/bin/python3 scripts/_c3_repaired.py
"""

import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from glns.paper_heuristics import run_ehs
from phaseC_adversarial_family_generation import load_schema
from phaseC_adversarial_family_generation import TOU_PARAMS_PRESETS, VALID_TOU_TYPES, VALID_FAILURE_MECHANISMS

ITER_DIR = PROJECT_ROOT / "research/glns_llm_heuristic_20260422/iterations/20260510_phaseC_adversarial_ehs_benchmark_design"
FAMILIES_DIR = ITER_DIR / "families"
INST_DIR = ITER_DIR / "generated_instances" / "c3_regular"
REPAIR_DIR = ITER_DIR / "generated_instances" / "c3_repair"
EVAL_DIR = ITER_DIR / "eval"
NOTES_DIR = ITER_DIR / "notes"
os.makedirs(REPAIR_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(NOTES_DIR, exist_ok=True)

MAX_N = 150
MAX_T = 200
SHORT_BUDGET = 30
LONG_BUDGET = 90
N_PER_ARM = 6

# ── Helpers ─────────────────────────────────────────────────────────────────

def load_json(p):
    with open(p) as f: return json.load(f)

def save_json(data, p):
    with open(p, "w") as f: json.dump(data, f, indent=2)

def _si(rng, lo, hi): return lo if lo >= hi else rng.randint(lo, hi)

def _gen_pt(rng, dist, n):
    pt, pr = dist.get("type","uniform"), dist.get("params",{})
    if pt == "uniform":
        return [rng.randint(pr.get("low",1), pr.get("high",10)) for _ in range(n)]
    elif pt == "bimodal":
        sl,sh = pr.get("small_low",1), pr.get("small_high",5)
        ll,lh = pr.get("large_low",10), pr.get("large_high",20)
        sf = pr.get("small_fraction",0.5)
        return [rng.randint(sl,sh) if rng.random()<sf else rng.randint(ll,lh) for _ in range(n)]
    elif pt == "exponential_truncated":
        sc,pm = pr.get("scale",5.0), pr.get("max",30)
        return [min(int(rng.expovariate(1.0/sc))+1, pm) for _ in range(n)]
    elif pt == "normal_truncated":
        mn,std = pr.get("mean",8.0), pr.get("std",3.0)
        return [max(pr.get("min",1), min(int(rng.gauss(mn,std)+0.5), pr.get("max",30))) for _ in range(n)]
    return [rng.randint(1,10) for _ in range(n)]

def _gen_mr(rng, dist, m):
    mt,pr = dist.get("type","uniform"), dist.get("params",{})
    if mt == "uniform":
        return [round(rng.uniform(pr.get("low",0.5),pr.get("high",3.0)),4) for _ in range(m)]
    elif mt == "step":
        lo,hi,sf = pr.get("low_rate",0.5), pr.get("high_rate",3.0), pr.get("step_fraction",0.5)
        nh = max(1,int(m*(1-sf)))
        return [round(hi,4)]*nh + [round(lo,4)]*(m-nh)
    elif mt == "exponential":
        sc = pr.get("scale",1.0)
        return [round(min(rng.expovariate(1.0/sc)*sc, pr.get("max",5.0))+0.1,4) for _ in range(m)]
    return [round(rng.uniform(0.5,3.0),4) for _ in range(m)]

def _gen_tou(rng, ttype, pr, T):
    ct = [1.0]*T
    if ttype == "single_peak":
        b,m = pr.get("base_price",1.0), pr.get("peak_multiplier",2.0)
        ps,pe = int(pr.get("peak_start",0.3)*T), int(pr.get("peak_end",0.6)*T)
        for t in range(T): ct[t]=round(b*(m if ps<=t<pe else 1.0)+rng.uniform(-0.1,0.1),4)
    elif ttype == "dual_peak":
        b,m=pr.get("base_price",1.0),pr.get("peak_multiplier",2.0)
        p1s,p1e=int(pr.get("peak1_start",0.2)*T),int(pr.get("peak1_end",0.35)*T)
        p2s,p2e=int(pr.get("peak2_start",0.65)*T),int(pr.get("peak2_end",0.8)*T)
        for t in range(T):
            ct[t]=round(b*m+rng.uniform(-0.1,0.1),4) if (p1s<=t<p1e or p2s<=t<p2e) else round(b+rng.uniform(-0.1,0.1),4)
    elif ttype == "high_variance":
        b,n=pr.get("base_price",1.0),pr.get("noise_range",2.0)
        ct=[round(max(0.1,b+rng.uniform(-n,n)),4) for _ in range(T)]
    elif ttype == "low_variance":
        b,n=pr.get("base_price",1.0),pr.get("noise_range",0.2)
        ct=[round(max(0.1,b+rng.uniform(-n,n)),4) for _ in range(T)]
    elif ttype == "monotonic_increasing":
        s,e=pr.get("start_price",0.5),pr.get("end_price",3.0)
        ct=[round(s+(e-s)*t/max(T-1,1),4) for t in range(T)]
    elif ttype == "monotonic_decreasing":
        s,e=pr.get("start_price",3.0),pr.get("end_price",0.5)
        ct=[round(s+(e-s)*t/max(T-1,1),4) for t in range(T)]
    elif ttype == "step_function":
        b,m=pr.get("base_price",1.0),pr.get("step_multiplier",3.0)
        ss=int(pr.get("step_start",0.5)*T)
        ct=[round(b*m if t>=ss else b,4) for t in range(T)]
    return ct


def make_random_family_capped(rng, idx):
    """Generate a random family with T_max ≤ MAX_T, n_max ≤ MAX_N built in."""
    m_min, m_max = rng.randint(3,10), rng.randint(6,15)
    n_min, n_max = rng.randint(max(15, 2*m_min), MAX_N), rng.randint(max(15, 2*m_min), MAX_N)
    n_min, n_max = min(n_min, n_max), max(n_min, n_max)

    proc_type = rng.choice(["uniform","bimodal","exponential_truncated","normal_truncated"])
    if proc_type == "uniform":
        pl,ph = rng.randint(1,3), rng.randint(pl+2, max(pl+3,10))
        proc_params = {"low":pl, "high":ph}
    elif proc_type == "bimodal":
        proc_params = {"small_low":1,"small_high":4,"large_low":6,"large_high":12,
                       "small_fraction":round(rng.uniform(0.3,0.7),2)}
    elif proc_type == "exponential_truncated":
        proc_params = {"scale":round(rng.uniform(2.0,6.0),1), "max":20}
    else:
        proc_params = {"mean":round(rng.uniform(4.0,8.0),1),"std":round(rng.uniform(1.0,3.0),1),"min":1,"max":20}

    mr_type = rng.choice(["uniform","step","exponential"])
    if mr_type == "uniform":
        mr_params = {"low":round(rng.uniform(0.3,1.5),2),"high":round(rng.uniform(2.0,5.0),2)}
    elif mr_type == "step":
        mr_params = {"low_rate":round(rng.uniform(0.3,1.0),2),"high_rate":round(rng.uniform(2.0,4.0),2),
                     "step_fraction":round(rng.uniform(0.3,0.7),2)}
    else:
        mr_params = {"scale":round(rng.uniform(0.5,2.0),2),"max":5.0}

    tou_type = rng.choice([t for t in VALID_TOU_TYPES if t not in ("flat","custom")])
    tou_params = TOU_PARAMS_PRESETS.get(tou_type,{}).copy()
    if "base_price" in tou_params: tou_params["base_price"] = round(rng.uniform(0.5,2.5),2)
    if "peak_multiplier" in tou_params: tou_params["peak_multiplier"] = round(rng.uniform(1.3,2.5),2)

    t_min = rng.randint(80, 120)
    t_max = rng.randint(t_min, MAX_T)

    mech = rng.choice(VALID_FAILURE_MECHANISMS[:-1])

    return {
        "family_name": f"repair_random_{idx:03d}",
        "hypothesis": f"Random family for C3-Regular repair #{idx}. Targeting {mech} with n≈{(n_min+n_max)//2}, m≈{(m_min+m_max)//2}, T≤{MAX_T}.",
        "description": f"Repair random family: n=[{n_min},{n_max}], m=[{m_min},{m_max}], T=[{t_min},{t_max}], {proc_type} proc, {mr_type} rates, {tou_type} TOU, medium epsilon.",
        "n_jobs_range":{"min":n_min,"max":n_max},
        "m_machines_range":{"min":m_min,"max":m_max},
        "horizon_T_range":{"min":t_min,"max":t_max},
        "processing_time_distribution":{"type":proc_type,"params":proc_params},
        "machine_rate_distribution":{"type":mr_type,"params":mr_params},
        "TOU_price_profile_type":tou_type,
        "TOU_price_profile_params":tou_params,
        "epsilon_regime":"medium",
        "expected_EHS_failure_mechanism":mech,
        "expected_EHS_failure_mechanism_evidence":"C3 repair random baseline.",
        "generated_instances_count":8,
        "validity_constraints":{"feasibility_guarantee":True,"min_total_work":50,"n_per_machine_min":2},
        "rejection_conditions":["sum(p_j)>m*T","all e[h] equal","all ct[t] equal","n<2*m"],
        "seed_behavior":{"base_seed":70000+idx*100,"expected_seed_variance":"medium"},
    }


def gen_and_check(rng, family, base_seed, max_retries=50):
    """Generate an instance, retry until feasible and T≤MAX_T, n≤MAX_N."""
    for attempt in range(max_retries):
        nr = family["n_jobs_range"]
        mr = family["m_machines_range"]
        tr = family["horizon_T_range"]
        n = min(_si(rng, nr["min"], nr["max"]), MAX_N)
        m = _si(rng, mr["min"], mr["max"])
        T = min(_si(rng, tr["min"], tr["max"]), MAX_T)
        p = _gen_pt(rng, family["processing_time_distribution"], n)
        e = _gen_mr(rng, family["machine_rate_distribution"], m)
        ct = _gen_tou(rng, family.get("TOU_price_profile_type","single_peak"),
                      family.get("TOU_price_profile_params",{}), T)

        if sum(p) > m * T:
            continue  # retry with fresh random

        inst = {"n":n,"m":m,"T":T,"p":p,"e":e,"ct":ct,
                "instance_id": f"c3repair_{family['family_name']}_{base_seed+attempt}",
                "metadata":{"family_name":family["family_name"],
                            "expected_mechanism":family.get("expected_EHS_failure_mechanism","?"),
                            "feasible":True,"seed":base_seed+attempt}}
        return inst
    return None


def eval_one(inst, budget):
    try:
        t0 = time.time()
        front = run_ehs(inst, rng=random.Random(42), time_limit_seconds=float(budget),
                        eps_ordering="expensive_source_first")
        elapsed = time.time()-t0
        if not front:
            return {"feasible":False,"front_size":0,"runtime_s":round(elapsed,1),
                    "timeout":elapsed>=budget*0.9,"deepest_cmax":-1,"best_energy":float("inf"),
                    "shallowest_cmax":-1,"error":"empty_front"}
        s = front[0]
        pts = [(sc.cmax,sc.energy) for sc in front if sc.feasible]
        return {"feasible":s.feasible,"front_size":len(front),
                "runtime_s":round(elapsed,1),"timeout":elapsed>=budget*0.9,
                "deepest_cmax":min(p[0] for p in pts) if pts else -1,
                "best_energy":min(p[1] for p in pts) if pts else float("inf"),
                "shallowest_cmax":max(p[0] for p in pts) if pts else -1,"error":""}
    except Exception as e:
        return {"feasible":False,"front_size":0,"runtime_s":0,"error":str(e)[:200],
                "timeout":False,"deepest_cmax":-1,"best_energy":float("inf"),"shallowest_cmax":-1}


# ════════════════════════════════════════════════════════════════════════════
# Task A: Load frozen LLM results (unchanged)
# ════════════════════════════════════════════════════════════════════════════

print("="*70)
print("Task A: Loading frozen LLM results")
print("="*70)

existing_raw = EVAL_DIR / "c3_regular_raw.csv"
llm_rows = []
if existing_raw.exists():
    with open(existing_raw) as f:
        for row in csv.DictReader(f):
            if row["arm"] == "llm":
                llm_rows.append(row)
print(f"  Loaded {len(llm_rows)} LLM eval rows (frozen, unchanged)")

llm_llm_fams = FAMILIES_DIR / "llm_families.json"
llm_data = load_json(llm_llm_fams) if llm_llm_fams.exists() else {"families":[]}

# ════════════════════════════════════════════════════════════════════════════
# Task B: Generate repaired random + human instances
# ════════════════════════════════════════════════════════════════════════════

print("\n"+"="*70)
print("Task B: Generating repaired baseline instances")
print("="*70)

rng = random.Random(123)

# ── Random repair: 2 families × 3 instances each ──
random_fams = [make_random_family_capped(rng, i) for i in range(2)]
random_insts = []
for fam in random_fams:
    base = fam["seed_behavior"]["base_seed"]
    fam["_arm"] = "random"
    for i in range(3):
        inst = gen_and_check(rng, fam, base + i * 17, max_retries=100)
        if inst is None:
            print(f"  ❌ Could not generate feasible instance for {fam['family_name']}")
            continue
        inst["metadata"]["_arm"] = "random"
        save_json(inst, REPAIR_DIR / f"{inst['instance_id']}.json")
        random_insts.append(inst)
        print(f"  ✅ random: {inst['instance_id']}: n={inst['n']} m={inst['m']} T={inst['T']} sum_p={sum(inst['p'])}")

# ── Human repair: use existing human instances ──
human_data = load_json(FAMILIES_DIR / "human_sweep_families.json")
human_fams = {f["family_name"]: f for f in human_data.get("families",[])}

# Re-generate 6 human instances (3 from human_tight_epsilon, 3 from human_loose_epsilon)
human_insts = []
for fam_name in ["human_tight_epsilon", "human_loose_epsilon"]:
    fam = human_fams[fam_name]
    fam["_arm"] = "human"
    base = fam.get("seed_behavior",{}).get("base_seed", 20000)
    for i in range(3):
        inst = gen_and_check(rng, fam, base + i * 17, max_retries=100)
        if inst is None:
            print(f"  ❌ Could not generate feasible instance for {fam_name}")
            continue
        inst["metadata"]["_arm"] = "human"
        save_json(inst, REPAIR_DIR / f"{inst['instance_id']}.json")
        human_insts.append(inst)
        print(f"  ✅ human:  {inst['instance_id']}: n={inst['n']} m={inst['m']} T={inst['T']} sum_p={sum(inst['p'])}")

print(f"\n  Generated: random={len(random_insts)}, human={len(human_insts)}")

# ════════════════════════════════════════════════════════════════════════════
# Task C: EHS evaluation of all instances
# ════════════════════════════════════════════════════════════════════════════

print("\n"+"="*70)
print(f"Task C: EHS evaluation @ {SHORT_BUDGET}s / {LONG_BUDGET}s")
print("="*70)

all_eval_rows = list(llm_rows)  # freeze LLM rows

# Load LLM instances for re-evaluation (they're frozen, already have results stored)
# Only evaluate random + human
# But also re-evaluate LLM to ensure same time-limit fix was used

# Re-evaluate ALL instances (LLM + random + human) for consistency
llm_insts = []
for inst_info in load_json(INST_DIR / "manifest.json")["instances"]:
    iid = inst_info["id"]
    if "asgh" not in iid and "es_local" not in iid:
        continue
    ipath = INST_DIR / f"{iid}.json"
    if ipath.exists():
        inst = load_json(ipath)
        if inst["metadata"].get("feasible", True):
            inst["metadata"]["_arm"] = "llm"
            llm_insts.append(inst)

all_insts = llm_insts + random_insts + human_insts

csv_path = EVAL_DIR / "c3_regular_repaired_raw.csv"
fields = ["instance_id","arm","family_name","expected_mechanism","n","m","T","sum_p",
          "feasible_instance","budget","feasible_ehs","front_size","runtime_s","timeout",
          "deepest_cmax","best_energy","shallowest_cmax","error"]
BUDGETS = [("short",SHORT_BUDGET),("long",LONG_BUDGET)]

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for inst in all_insts:
        arm = inst["metadata"].get("_arm","?")
        fam = inst["metadata"]["family_name"]
        mech = inst["metadata"].get("expected_mechanism","?")
        sp = sum(inst["p"])
        for blabel, budget in BUDGETS:
            print(f"  {inst['instance_id']} @{blabel}({budget}s)...", end=" ", flush=True)
            m = eval_one(inst, budget)
            row = {"instance_id":inst["instance_id"],"arm":arm,"family_name":fam,
                   "expected_mechanism":mech,"n":inst["n"],"m":inst["m"],"T":inst["T"],
                   "sum_p":sp,"feasible_instance":True,"budget":blabel,
                   "feasible_ehs":m["feasible"],"front_size":m["front_size"],
                   "runtime_s":m["runtime_s"],"timeout":m["timeout"],
                   "deepest_cmax":m["deepest_cmax"],"best_energy":m["best_energy"],
                   "shallowest_cmax":m["shallowest_cmax"],"error":m["error"]}
            writer.writerow(row); f.flush()
            status = "✅" if m["feasible"] else "❌"
            err = f" [{m['error'][:40]}]" if m["error"] else ""
            print(f"{status} f={m['front_size']} cmax={m['deepest_cmax']} E={m['best_energy']:.0f} {m['runtime_s']:.1f}s{err}", flush=True)

print(f"\n  Saved: {csv_path} ({len(all_insts)*2} rows)")

# ════════════════════════════════════════════════════════════════════════════
# Task D: Dual metrics — generation quality + adversarial yield
# ════════════════════════════════════════════════════════════════════════════

print("\n"+"="*70)
print("Task D: Metrics")
print("="*70)

# Reload all rows
rows = []
with open(csv_path, newline="") as f:
    for row in csv.DictReader(f):
        rows.append(row)

# Per-instance grouping
inst_data = defaultdict(lambda: {"short":None,"long":None,"arm":"?","fam":"?","mech":"?"})
for r in rows:
    iid = r["instance_id"]
    inst_data[iid]["arm"] = r["arm"]
    inst_data[iid]["fam"] = r["family_name"]
    inst_data[iid]["mech"] = r["expected_mechanism"]
    inst_data[iid]["n"] = int(r["n"])
    inst_data[iid]["m"] = int(r["m"])
    inst_data[iid]["T"] = int(r["T"])
    inst_data[iid][r["budget"]] = r

# ── Generation-quality metrics ──
gen_stats = defaultdict(lambda: {"total":0,"feasible_inst":0,"evaluable":0,"completed_pairs":0,
                                  "infeasible":0,"timeout":0,"error":0})
for iid, d in inst_data.items():
    a = d["arm"]
    gen_stats[a]["total"] += 1
    gen_stats[a]["feasible_inst"] += 1  # all instances in this set are feasible

    short = d.get("short")
    long = d.get("long")
    evaluable = False
    if short:
        se = short.get("error","")
        if not se or "empty_front" in se:
            gen_stats[a]["completed_pairs"] += 1
            evaluable = True
        elif "timeout" in se.lower():
            gen_stats[a]["timeout"] += 1
        else:
            gen_stats[a]["error"] += 1
    if long:
        le = long.get("error","")
        if not le or "empty_front" in le:
            gen_stats[a]["completed_pairs"] += 1
            evaluable = True
        elif "timeout" in le.lower():
            gen_stats[a]["timeout"] += 1
        else:
            gen_stats[a]["error"] += 1

    if evaluable:
        gen_stats[a]["evaluable"] += 1

print("\n  ── Generation Quality ──")
print(f"  {'Arm':<10} {'Inst':>5} {'Feas':>5} {'Eval':>5} {'Pairs':>6} {'T/O':>5} {'Err':>5}  {'Eval Rate':>10}")
for arm in ["llm","random","human"]:
    s = gen_stats[arm]
    evrate = f"{s['evaluable']/s['total']*100:.0f}%" if s["total"] else "N/A"
    print(f"  {arm:<10} {s['total']:>5} {s['feasible_inst']:>5} {s['evaluable']:>5} "
          f"{s['completed_pairs']:>6} {s['timeout']:>5} {s['error']:>5}  {evrate:>10}")

# ── Adversarial yield (evaluable only) ──
summary_rows = []
for iid, d in sorted(inst_data.items()):
    a, fam, mech = d["arm"], d["fam"], d["mech"]
    short = d.get("short")
    long = d.get("long")
    if not short or not long:
        continue

    fs_s = int(short["front_size"])
    fs_l = int(long["front_size"])
    dc_s = int(short["deepest_cmax"])
    dc_l = int(long["deepest_cmax"])
    rt_s = float(short["runtime_s"])
    rt_l = float(long["runtime_s"])
    e_s, e_l = short.get("error",""), long.get("error","")

    evaluable = not (e_s and "empty_front" not in e_s) and not (e_l and "empty_front" not in e_l)
    if not evaluable:
        continue

    high_yield = False
    yield_reason = "none"

    if not short["feasible_ehs"] == "True" or not long["feasible_ehs"] == "True":
        yield_reason = "ehs_infeasible"
    elif fs_s == 0 and fs_l > 0:
        high_yield = True
        yield_reason = f"zero_short→{fs_l}_long"
    elif fs_l - fs_s >= 2:
        high_yield = True
        yield_reason = f"fs_growth_+{fs_l-fs_s}"
    elif fs_s <= 1 and fs_l >= 3:
        high_yield = True
        yield_reason = f"near_zero→{fs_l}"
    elif dc_s > 0 and dc_l > 0 and (dc_s - dc_l) >= 2:
        high_yield = True
        yield_reason = f"cmax_improved_{dc_s-dc_l}"
    elif rt_s > SHORT_BUDGET * 1.5 and fs_s <= 1:
        high_yield = True
        yield_reason = "very_slow_produces_little"
    elif fs_s > 0 and fs_l == fs_s:
        yield_reason = f"saturated_at_{fs_s}"
    else:
        yield_reason = f"normal_Δfs={fs_l-fs_s}"

    summary_rows.append({
        "instance_id":iid,"arm":a,"family_name":fam,"expected_mechanism":mech,
        "n":d["n"],"m":d["m"],"T":d["T"],
        "short_feasible":short["feasible_ehs"],"short_front_size":fs_s,
        "short_runtime_s":rt_s,"short_deepest_cmax":dc_s,"short_best_energy":short.get("best_energy","inf"),
        "long_feasible":long["feasible_ehs"],"long_front_size":fs_l,
        "long_runtime_s":rt_l,"long_deepest_cmax":dc_l,"long_best_energy":long.get("best_energy","inf"),
        "high_yield":high_yield,"yield_reason":yield_reason,
    })

sum_path = EVAL_DIR / "c3_regular_repaired_summary.csv"
with open(sum_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
    writer.writeheader(); writer.writerows(summary_rows)

# Arm yield stats
yield_stats = defaultdict(lambda: {"total":0,"high_yield":0,"fronts":[],"mechs":set()})
for r in summary_rows:
    a = r["arm"]
    yield_stats[a]["total"] += 1
    if r["high_yield"]: yield_stats[a]["high_yield"] += 1
    yield_stats[a]["fronts"].append(f"{r['short_front_size']}→{r['long_front_size']}")
    yield_stats[a]["mechs"].add(r["expected_mechanism"])

print("\n  ── Adversarial Yield (evaluable only) ──")
print(f"  {'Arm':<10} {'Eval':>5} {'Yield':>6} {'Rate':>8}  Front Growth")
for arm in ["llm","random","human"]:
    s = yield_stats[arm]
    rate = f"{s['high_yield']/s['total']*100:.0f}%" if s["total"] else "N/A"
    fronts = ", ".join(s["fronts"])
    print(f"  {arm:<10} {s['total']:>5} {s['high_yield']:>6} {rate:>8}  {fronts}")

print("\n  Per-instance yield:")
for r in summary_rows:
    tag = "🔴" if r["high_yield"] else "  "
    print(f"  {tag} {r['instance_id']:<55} {r['arm']:<8} "
          f"fs={r['short_front_size']:>2}→{r['long_front_size']:>2} "
          f"cmax={r['short_deepest_cmax']:>3}→{str(r['long_deepest_cmax']):>3} "
          f"[{r['yield_reason'][:50]}]")

# ════════════════════════════════════════════════════════════════════════════
# Task E: Write decision
# ════════════════════════════════════════════════════════════════════════════

print("\n"+"="*70)
print("Task E: Gate Decision")
print("="*70)

# Generation quality
llm_gq = gen_stats["llm"]["evaluable"]/max(gen_stats["llm"]["total"],1)
random_gq = gen_stats["random"]["evaluable"]/max(gen_stats["random"]["total"],1)
human_gq = gen_stats["human"]["evaluable"]/max(gen_stats["human"]["total"],1)

# Adversarial yield
llm_ay = yield_stats["llm"]["high_yield"]/max(yield_stats["llm"]["total"],1)
random_ay = yield_stats["random"]["high_yield"]/max(yield_stats["random"]["total"],1)
human_ay = yield_stats["human"]["high_yield"]/max(yield_stats["human"]["total"],1)

print(f"  Generation quality: LLM={llm_gq:.0%}, Random={random_gq:.0%}, Human={human_gq:.0%}")
print(f"  Adversarial yield:  LLM={llm_ay:.0%}, Random={random_ay:.0%}, Human={human_ay:.0%}")

# Gate
if llm_ay >= 2*max(random_ay, human_ay) and llm_ay >= 0.33 and llm_gq >= max(random_gq, human_gq):
    gate = "STRONG: LLM has better generation quality AND ≥2× high-yield."
elif llm_ay >= max(random_ay, human_ay) and llm_ay >= 0.33 and llm_gq >= max(random_gq, human_gq):
    gate = "MODERATE: LLM has better generation quality, ties/beats on yield."
elif llm_ay >= max(random_ay, human_ay) and llm_ay >= 0.33:
    gate = "MODERATE: LLM ties/beats on yield, but generation quality not clearly better."
elif llm_ay < max(random_ay, human_ay):
    gate = "FAIL: Repaired baselines beat LLM on yield."
else:
    gate = "WEAK: LLM similar to repaired baselines."

print(f"\n  Gate: {gate}")

dp = NOTES_DIR / "c3_regular_repaired_decision.md"
with open(dp, "w") as f:
    f.write("# C3-Regular Repaired Decision\n\n")
    f.write("**Date**: 2026-05-10\n")
    f.write(f"**Instances per arm**: 6 (n≤{MAX_N}, T≤{MAX_T})\n")
    f.write(f"**Budgets**: {SHORT_BUDGET}s / {LONG_BUDGET}s\n\n")

    f.write("## Generation Quality\n\n")
    f.write("| Arm | Instances | Feasible | Evaluable | Pairs | Timeout | Error | Rate |\n")
    f.write("|-----|----------|----------|-----------|-------|---------|-------|------|\n")
    for arm in ["llm","random","human"]:
        s = gen_stats[arm]
        rate = f"{s['evaluable']/s['total']*100:.0f}%" if s["total"] else "N/A"
        f.write(f"| {arm} | {s['total']} | {s['feasible_inst']} | {s['evaluable']} | "
                f"{s['completed_pairs']} | {s['timeout']} | {s['error']} | {rate} |\n")

    f.write("\n## Adversarial Yield (evaluable instances only)\n\n")
    f.write("| Arm | Evaluable | High-Yield | Rate |\n")
    f.write("|-----|----------|------------|------|\n")
    for arm in ["llm","random","human"]:
        s = yield_stats[arm]
        rate = f"{s['high_yield']/s['total']*100:.0f}%" if s["total"] else "N/A"
        f.write(f"| {arm} | {s['total']} | {s['high_yield']} | {rate} |\n")

    f.write(f"\n## Gate: {gate}\n\n")
    f.write(f"- LLM generation quality: {llm_gq:.0%}\n")
    f.write(f"- Random generation quality: {random_gq:.0%}\n")
    f.write(f"- Human generation quality: {human_gq:.0%}\n")
    f.write(f"- LLM adversarial yield: {llm_ay:.0%}\n")
    f.write(f"- Random adversarial yield: {random_ay:.0%}\n")
    f.write(f"- Human adversarial yield: {human_ay:.0%}\n\n")

    f.write("## Per-Instance Yield\n\n")
    for r in summary_rows:
        f.write(f"- `{r['instance_id']}` ({r['arm']}, {r['expected_mechanism']}): "
                f"n={r['n']} m={r['m']} T={r['T']} — "
                f"fs={r['short_front_size']}→{r['long_front_size']} "
                f"cmax={r['short_deepest_cmax']}→{r['long_deepest_cmax']} "
                f"{'HIGH' if r['high_yield'] else 'low'} [{r['yield_reason']}]\n")

print(f"\n  Decision: {dp}")
print("✅ C3-Regular repaired complete.")

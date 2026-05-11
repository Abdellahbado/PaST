#!/usr/bin/env python3
"""Complete remaining C3-Regular runs + analysis. Uses existing partial CSV."""
import csv, json, os, random, sys, time
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from glns.paper_heuristics import run_ehs

ITER_DIR = PROJECT_ROOT / "research/glns_llm_heuristic_20260422/iterations/20260510_phaseC_adversarial_ehs_benchmark_design"
INST_DIR = ITER_DIR / "generated_instances" / "c3_regular"
EVAL_DIR = ITER_DIR / "eval"
NOTES_DIR = ITER_DIR / "notes"
os.makedirs(EVAL_DIR, exist_ok=True)

def load_json(p):
    with open(p) as f: return json.load(f)

def save_json(data, p):
    with open(p, "w") as f: json.dump(data, f, indent=2)

manifest = load_json(INST_DIR / "manifest.json")
BUDGETS = [("short", 30), ("long", 90)]
csv_path = EVAL_DIR / "c3_regular_raw.csv"

# Load existing
existing = {}
if csv_path.exists():
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            existing[row["instance_id"] + "_" + row["budget"]] = row

# Determine incomplete runs
pending = []
for inst_info in manifest["instances"]:
    with open(INST_DIR / f"{inst_info['id']}.json") as fi:
        inst = json.load(fi)
    for blabel, budget in BUDGETS:
        key = inst_info["id"] + "_" + blabel
        if key not in existing:
            pending.append((inst, inst_info, blabel, budget))

print(f"Pending: {len(pending)} runs")
if not pending:
    print("All runs complete.")

# Run pending evaluations
fields = ["instance_id","arm","family_name","expected_mechanism","n","m","T","sum_p",
          "feasible_instance","budget","feasible_ehs","front_size","runtime_s","timeout",
          "deepest_cmax","best_energy","shallowest_cmax","error"]

with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    for inst, inst_info, blabel, budget in pending:
        arm = inst["metadata"].get("_arm", inst["metadata"].get("arm", "?"))
        fam = inst["metadata"]["family_name"]
        mech = inst["metadata"].get("expected_mechanism", "?")
        s_p = sum(inst["p"])

        # Skip infeasible instances
        if not inst_info.get("feasible", True):
            row = {"instance_id": inst_info["id"], "arm": arm, "family_name": fam,
                "expected_mechanism": mech, "n": inst["n"], "m": inst["m"], "T": inst["T"],
                "sum_p": s_p, "feasible_instance": False,
                "budget": blabel, "feasible_ehs": False, "front_size": 0,
                "runtime_s": 0, "timeout": True, "deepest_cmax": -1,
                "best_energy": float("inf"), "shallowest_cmax": -1,
                "error": "instance_infeasible"}
            writer.writerow(row)
            print(f"  SKIP {inst_info['id']} @{blabel}: infeasible instance", flush=True)
            continue

        print(f"  {inst_info['id']} @{blabel}({budget}s)...", end=" ", flush=True)
        try:
            t0 = time.time()
            front = run_ehs(inst, rng=random.Random(42), time_limit_seconds=float(budget),
                            eps_ordering="expensive_source_first")
            elapsed = time.time() - t0
            if not front:
                row = {"instance_id": inst_info["id"], "arm": arm, "family_name": fam,
                    "expected_mechanism": mech, "n": inst["n"], "m": inst["m"], "T": inst["T"],
                    "sum_p": s_p, "feasible_instance": True,
                    "budget": blabel, "feasible_ehs": False, "front_size": 0,
                    "runtime_s": round(elapsed, 1), "timeout": elapsed >= budget * 0.9,
                    "deepest_cmax": -1, "best_energy": float("inf"), "shallowest_cmax": -1,
                    "error": "empty_front"}
            else:
                s = front[0]
                pts = [(sc.cmax, sc.energy) for sc in front if sc.feasible]
                row = {"instance_id": inst_info["id"], "arm": arm, "family_name": fam,
                    "expected_mechanism": mech, "n": inst["n"], "m": inst["m"], "T": inst["T"],
                    "sum_p": s_p, "feasible_instance": True,
                    "budget": blabel, "feasible_ehs": s.feasible, "front_size": len(front),
                    "runtime_s": round(elapsed, 1), "timeout": elapsed >= budget * 0.9,
                    "deepest_cmax": min(p[0] for p in pts) if pts else -1,
                    "best_energy": min(p[1] for p in pts) if pts else float("inf"),
                    "shallowest_cmax": max(p[0] for p in pts) if pts else -1,
                    "error": ""}
            writer.writerow(row)
            f.flush()
            status = "✅" if row["feasible_ehs"] else "❌"
            err = f" [{row['error'][:40]}]" if row["error"] else ""
            print(f"{status} f={row['front_size']}, cmax={row['deepest_cmax']}, "
                  f"E={row['best_energy']:.0f}, {row['runtime_s']:.1f}s{err}", flush=True)
        except Exception as e:
            row = {"instance_id": inst_info["id"], "arm": arm, "family_name": fam,
                "expected_mechanism": mech, "n": inst["n"], "m": inst["m"], "T": inst["T"],
                "sum_p": s_p, "feasible_instance": True,
                "budget": blabel, "feasible_ehs": False, "front_size": 0,
                "runtime_s": 0, "timeout": True, "deepest_cmax": -1,
                "best_energy": float("inf"), "shallowest_cmax": -1,
                "error": str(e)[:200]}
            writer.writerow(row)
            print(f"❌ ERR: {str(e)[:60]}", flush=True)

print(f"\nAll evaluations complete.", flush=True)

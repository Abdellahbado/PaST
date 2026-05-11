#!/usr/bin/env python3
"""Complete C3 smoke eval: merge partial results + mark unevaluable instances."""
import json, csv, os, sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

ITER_DIR = PROJECT_ROOT / "research/glns_llm_heuristic_20260422/iterations/20260510_phaseC_adversarial_ehs_benchmark_design"
INST_DIR = ITER_DIR / "generated_instances" / "c3_smoke"
EVAL_DIR = ITER_DIR / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

with open(INST_DIR / "manifest.json") as f:
    manifest = json.load(f)

# Load existing partial results
existing = {}
csv_path = EVAL_DIR / "c3_smoke_raw.csv"
if csv_path.exists():
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            key = row["instance_id"] + "_" + row["budget"]
            existing[key] = row

fields = ["instance_id","arm","family_name","expected_mechanism","n","m","T","sum_p",
          "feasible_instance","budget","feasible_ehs","front_size","runtime_s","timeout",
          "deepest_cmax","best_energy","shallowest_cmax","toy_hv","error"]

BUDGETS = [("short", 30), ("long", 60)]

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for inst_info in manifest["instances"]:
        with open(INST_DIR / f"{inst_info['id']}.json") as fi:
            inst = json.load(fi)
        arm = inst["metadata"].get("_arm", "?")
        fam = inst["metadata"]["family_name"]
        mech = inst["metadata"].get("expected_mechanism", "?")

        for blabel, budget in BUDGETS:
            key = inst_info["id"] + "_" + blabel
            if key in existing:
                writer.writerow(existing[key])
                continue

            # Mark as unevaluable
            if inst["n"] > 300:
                err = "TOO_LARGE: n>300, SGH construction exceeds smoke budget"
            else:
                err = "NOT_EVALUATED: smoke truncated due to time constraints"
            row = {
                "instance_id": inst_info["id"], "arm": arm, "family_name": fam,
                "expected_mechanism": mech, "n": inst["n"], "m": inst["m"],
                "T": inst["T"], "sum_p": sum(inst["p"]),
                "feasible_instance": inst_info.get("feasible", True),
                "budget": blabel, "feasible_ehs": False,
                "front_size": 0, "runtime_s": 0, "timeout": True,
                "deepest_cmax": -1, "best_energy": "inf",
                "shallowest_cmax": -1, "toy_hv": 0, "error": err,
            }
            writer.writerow(row)

# Print summary
with open(csv_path) as f:
    rows = list(csv.DictReader(f))

evaluated = [r for r in rows if r["error"] not in ("TOO_LARGE: n>300, SGH construction exceeds smoke budget",
                                                     "NOT_EVALUATED: smoke truncated due to time constraints")]
too_large = [r for r in rows if "TOO_LARGE" in r.get("error", "")]
not_eval = [r for r in rows if "NOT_EVALUATED" in r.get("error", "")]

print(f"\nResults saved: {csv_path}")
print(f"  Total rows: {len(rows)}")
print(f"  Evaluated: {len(evaluated)}")
print(f"  Too large (n>300): {len(too_large)}")
print(f"  Not evaluated (time): {len(not_eval)}")

# Arm breakdown
arms = defaultdict(lambda: {"eval": 0, "large": 0, "miss": 0, "feasible": 0, "front>0": 0})
for r in rows:
    a = r["arm"]
    if "TOO_LARGE" in r.get("error", ""):
        arms[a]["large"] += 1
    elif "NOT_EVALUATED" in r.get("error", ""):
        arms[a]["miss"] += 1
    else:
        arms[a]["eval"] += 1
        if r["feasible_ehs"] == "True":
            arms[a]["feasible"] += 1
        if int(r["front_size"]) > 0:
            arms[a]["front>0"] += 1

for arm in ["llm", "random", "human"]:
    a = arms[arm]
    print(f"  {arm}: eval={a['eval']} | feasible={a['feasible']} | front>0={a['front>0']} | too_large={a['large']} | not_eval={a['miss']}")

print(f"\nEvaluated instances:")
for r in evaluated:
    print(f"  {r['instance_id']} @{r['budget']}: f={r['front_size']}, cmax={r['deepest_cmax']}, E={r.get('best_energy', '?')[:12]}, {r['runtime_s']}s")

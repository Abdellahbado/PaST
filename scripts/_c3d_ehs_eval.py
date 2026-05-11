#!/usr/bin/env python3
"""C3 Smoke EHS Eval — standalone runner using conda Python."""
import json, random, time, csv, os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Must be run with conda env Python to get Cython DP
from glns.paper_heuristics import run_ehs

ITER_DIR = PROJECT_ROOT / "research/glns_llm_heuristic_20260422/iterations/20260510_phaseC_adversarial_ehs_benchmark_design"
INST_DIR = ITER_DIR / "generated_instances" / "c3_smoke"
EVAL_DIR = ITER_DIR / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

BUDGETS = [("short", 30), ("long", 60)]

def eval_one(inst, budget):
    try:
        t0 = time.time()
        front = run_ehs(inst, rng=random.Random(42), time_limit_seconds=float(budget),
                         eps_ordering="expensive_source_first")
        elapsed = time.time() - t0
        s = front[0] if front else None
        pts = [(sched.cmax, sched.energy) for sched in front if sched.feasible]
        return {
            "feasible": s.feasible if s else False,
            "front_size": len(front),
            "runtime_s": round(elapsed, 1),
            "timeout": elapsed >= budget * 0.9,
            "deepest_cmax": min(p[0] for p in pts) if pts else -1,
            "best_energy": min(p[1] for p in pts) if pts else float("inf"),
            "shallowest_cmax": max(p[0] for p in pts) if pts else -1,
            "shallowest_energy": max(p[1] for p in pts) if pts else float("inf"),
            "toy_hv": 0,
            "error": "",
        }
    except Exception as e:
        return {"feasible": False, "front_size": 0, "runtime_s": 0, "error": str(e)[:200],
                "deepest_cmax": -1, "best_energy": float("inf"), "shallowest_cmax": -1,
                "timeout": False, "toy_hv": 0}

with open(INST_DIR / "manifest.json") as f:
    manifest = json.load(f)

# Skip the giant LLM instances (n=800+) — they can't complete even one khat
# This is itself a valid diagnostic result
insts_to_eval = [i for i in manifest["instances"]
                 if i["n"] <= 300]
print(f"Evaluating {len(insts_to_eval)} instances (n≤300, skipping giant LLM instances)", flush=True)

csv_path = EVAL_DIR / "c3_smoke_raw.csv"
fields = ["instance_id","arm","family_name","expected_mechanism","n","m","T","sum_p",
          "feasible_instance","budget","feasible_ehs","front_size","runtime_s","timeout",
          "deepest_cmax","best_energy","shallowest_cmax","toy_hv","error"]

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for inst_info in insts_to_eval:
        with open(INST_DIR / f"{inst_info['id']}.json") as fi:
            inst = json.load(fi)
        arm = inst["metadata"].get("_arm", "?")
        fam = inst["metadata"]["family_name"]
        mech = inst["metadata"].get("expected_mechanism", "?")

        for blabel, budget in BUDGETS:
            print(f"  {inst_info['id']} @{blabel}({budget}s)...", end=" ", flush=True)
            m = eval_one(inst, budget)
            row = {
                "instance_id": inst_info["id"], "arm": arm, "family_name": fam,
                "expected_mechanism": mech, "n": inst["n"], "m": inst["m"],
                "T": inst["T"], "sum_p": sum(inst["p"]),
                "feasible_instance": inst_info.get("feasible", True),
                "budget": blabel, "feasible_ehs": m.get("feasible", False),
                "front_size": m.get("front_size", 0),
                "runtime_s": m.get("runtime_s", 0),
                "timeout": m.get("timeout", False),
                "deepest_cmax": m.get("deepest_cmax", -1),
                "best_energy": m.get("best_energy", float("inf")),
                "shallowest_cmax": m.get("shallowest_cmax", -1),
                "toy_hv": m.get("toy_hv", 0),
                "error": m.get("error", ""),
            }
            writer.writerow(row)
            f.flush()
            status = "✅" if m.get("feasible") else "❌"
            err = f" [{m.get('error', '')[:60]}]" if m.get("error") else ""
            print(f"{status} f={m.get('front_size', 0)}, "
                  f"cmax={m.get('deepest_cmax', -1)}, "
                  f"E={m.get('best_energy', 0):.0f}, "
                  f"{m.get('runtime_s', 0):.0f}s{err}", flush=True)

# Add giant LLM instances as unevaluable (mark them)
with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    for inst_info in manifest["instances"]:
        if inst_info["n"] > 300:
            with open(INST_DIR / f"{inst_info['id']}.json") as fi:
                inst = json.load(fi)
            arm = inst["metadata"].get("_arm", "?")
            fam = inst["metadata"]["family_name"]
            mech = inst["metadata"].get("expected_mechanism", "?")
            for blabel, budget in BUDGETS:
                row = {"instance_id": inst_info["id"], "arm": arm, "family_name": fam,
                    "expected_mechanism": mech, "n": inst["n"], "m": inst["m"],
                    "T": inst["T"], "sum_p": sum(inst["p"]),
                    "feasible_instance": inst_info.get("feasible", True),
                    "budget": blabel, "feasible_ehs": False,
                    "front_size": 0, "runtime_s": 0, "timeout": True,
                    "deepest_cmax": -1, "best_energy": float("inf"),
                    "shallowest_cmax": -1, "toy_hv": 0,
                    "error": "instance too large for smoke (n>300, SGH construction exceeds budget)"}
                writer.writerow(row)

print(f"✅ Done. Results: {csv_path}", flush=True)

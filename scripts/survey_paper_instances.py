#!/usr/bin/env python3
"""Survey all datasets in the paper's benchmark repository."""
import json, os, glob

BASE = "/Users/mac/Documents/Study/PFE/PaST/data/paper_instances/datasets"

for ds_name in sorted(os.listdir(BASE)):
    ds_path = os.path.join(BASE, ds_name)
    if not os.path.isdir(ds_path):
        continue
    files = sorted(glob.glob(os.path.join(ds_path, "*.json")))
    print(f"\n=== {ds_name} ({len(files)} instances) ===")
    if files:
        d = json.load(open(files[0]))
        meta = d.get("Metadata", {})
        jobs = [j["ProcessingTime"] for j in d["Jobs"]]
        costs = d["EnergyCosts"]
        print(f"  First instance: n={len(jobs)}, h={len(costs)}")
        print(
            f"  Proc times: min={min(jobs)}, max={max(jobs)}, unique={sorted(set(jobs))}"
        )
        print(f"  Metadata: {meta}")
        print(f"  OffOnTime={d['OffOnTime']}, OnOffTime={d['OnOffTime']}")
        print(
            f"  OffOnPower={d['OffOnPowerConsumption']}, OnOffPower={d['OnOffPowerConsumption']}"
        )
        print(
            f"  OnPower={d['OnPowerConsumption']}, IdlePower={d['IdlePowerConsumption']}, OffPower={d['OffPowerConsumption']}"
        )
        print(
            f"  OffIdleTime={d.get('OffIdleTime')}, IdleOffTime={d.get('IdleOffTime')}"
        )
        print(
            f"  OffIdlePower={d.get('OffIdlePowerConsumption')}, IdleOffPower={d.get('IdleOffPowerConsumption')}"
        )

        # Show range of n, h across all instances
        ns = set()
        hs = set()
        ptgroups = set()
        for f in files:
            dd = json.load(open(f))
            ns.add(len(dd["Jobs"]))
            hs.add(len(dd["EnergyCosts"]))
            pts = tuple(sorted(set(j["ProcessingTime"] for j in dd["Jobs"])))
            ptgroups.add(pts)
        print(f"  Range: n in {sorted(ns)}, h in {sorted(hs)}")
        print(
            f"  Proc time groups: {sorted(ptgroups)[:5]}{'...' if len(ptgroups)>5 else ''}"
        )

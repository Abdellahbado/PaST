#!/usr/bin/env python3
import json, os

for ds in ["benedikt2020a_prelim", "aghelinejad2017a_1"]:
    path = f"data/paper_instances/datasets/{ds}"
    files = sorted(
        [f for f in os.listdir(path) if f.endswith(".json")],
        key=lambda x: int(x.split(".")[0]),
    )
    print(f"\n=== {ds} ({len(files)} files) ===")
    for fn in files:
        with open(f"{path}/{fn}") as f:
            d = json.load(f)
        n = len(d["Jobs"])
        h = len(d["EnergyCosts"])
        off_on = d.get("OffOnTime", [])
        n_off = len(off_on)
        is_nosby = n_off == 1 and off_on[0] == 2
        pjs = sorted(set(j["ProcessingTime"] for j in d["Jobs"]))
        sum_pj = sum(j["ProcessingTime"] for j in d["Jobs"])
        feasible = sum_pj + 3 <= h  # off->on(2) + jobs + on->off(1)
        print(
            f"  {fn:8s} n={n:3d} h={h:4d} off_levels={n_off} nosby={is_nosby} pj={pjs} sum_pj={sum_pj} feasible={feasible}"
        )

# Check prelim results
print()
for ds in ["benedikt2020a_prelim"]:
    rpath = f"data/paper_instances/results/{ds}"
    if os.path.exists(rpath):
        for sub in os.listdir(rpath):
            subpath = f"{rpath}/{sub}"
            if os.path.isdir(subpath):
                for method in os.listdir(subpath):
                    mpath = f"{subpath}/{method}"
                    if os.path.isdir(mpath):
                        nfiles = len(
                            [f for f in os.listdir(mpath) if f.endswith(".json")]
                        )
                        print(f"  Results: {sub}/{method} - {nfiles} files")
                        # Check first result
                        for rf in sorted(os.listdir(mpath))[:1]:
                            with open(f"{mpath}/{rf}") as f:
                                r = json.load(f)
                            print(
                                f'    Sample {rf}: Cost={r.get("Objective","?")}, LB={r.get("LowerBound","?")}, TLR={r.get("TimeLimitReached","?")}'
                            )
    else:
        print(f"No results for {ds}")

# Check TWOSBY instance details
print("\n=== TWOSBY instance sample ===")
with open("data/paper_instances/datasets/benedikt2020a_large_twosby/0.json") as f:
    d = json.load(f)
print(f'n={len(d["Jobs"])}, h={len(d["EnergyCosts"])}')
print(f'OffOnTime={d["OffOnTime"]}')
print(f'OnOffTime={d["OnOffTime"]}')
print(f'OffOnPower={d["OffOnPowerConsumption"]}')
print(f'OnOffPower={d["OnOffPowerConsumption"]}')
print(f'OnPower={d["OnPowerConsumption"]}')
print(f'IdlePower={d["IdlePowerConsumption"]}')
print(f'OffPower={d["OffPowerConsumption"]}')
# Check for any additional transition keys
all_keys = set(d.keys()) - {
    "Jobs",
    "EnergyCosts",
    "Metadata",
    "OffOnTime",
    "OnOffTime",
    "OffOnPowerConsumption",
    "OnOffPowerConsumption",
    "OnPowerConsumption",
    "IdlePowerConsumption",
    "OffPowerConsumption",
}
print(f"Other keys: {all_keys}")
# Check if there are idle<->off transitions
for key in sorted(d.keys()):
    if "idle" in key.lower() or "Idle" in key:
        if key != "IdlePowerConsumption":
            print(f"  {key} = {d[key]}")
    if (
        "off" in key.lower()
        and "Off" in key
        and key
        not in [
            "OffOnTime",
            "OnOffTime",
            "OffOnPowerConsumption",
            "OnOffPowerConsumption",
            "OffPowerConsumption",
        ]
    ):
        print(f"  {key} = {d[key]}")

#!/usr/bin/env python3
"""Check for instances with scaled machine params (timeMul > 1)."""
import json, glob, os

base = "/Users/mac/Documents/Study/PFE/PaST/data/green-scheduling-bab/Iirc.EnergyStatesAndCostsScheduling/data/datasets/benedikt2025_groups"
files = sorted(glob.glob(os.path.join(base, "*.json")))
configs = {}
for fp in files:
    with open(fp) as f:
        d = json.load(f)
    key = (
        tuple(d["OffOnTime"]),
        tuple(d["OnOffTime"]),
        d["OnPowerConsumption"],
        d["IdlePowerConsumption"],
    )
    if key not in configs:
        configs[key] = []
    configs[key].append(os.path.basename(fp))

for k, v in configs.items():
    print(
        f"Config: OffOnTime={list(k[0])}, OnOffTime={list(k[1])}, OnPower={k[2]}, IdlePower={k[3]}"
    )
    print(f"  {len(v)} instances")

# Also check max T and n
max_T = 0
max_n = 0
for fp in files:
    with open(fp) as f:
        d = json.load(f)
    T = len(d["EnergyCosts"])
    n = len(d["Jobs"])
    if T > max_T:
        max_T = T
    if n > max_n:
        max_n = n
print(f"\nMax n={max_n}, Max T={max_T}")
print(f"Total instances: {len(files)}")

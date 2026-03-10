#!/usr/bin/env python3
"""Quick check of all generated datasets."""
import json, os, glob

base = "/Users/mac/Documents/Study/PFE/PaST/data/green-scheduling-bab/Iirc.EnergyStatesAndCostsScheduling/data/datasets"
for ds in sorted(os.listdir(base)):
    dpath = os.path.join(base, ds)
    if not os.path.isdir(dpath):
        continue
    files = sorted(glob.glob(os.path.join(dpath, "*.json")))
    n = len(files)
    if n == 0:
        print(f"{ds}: 0 instances")
        continue
    # Read first instance
    with open(files[0]) as f:
        d = json.load(f)
    nj = len(d["Jobs"])
    T = len(d["EnergyCosts"])
    noff = len(d.get("OffOnTime", []))
    mtype = "TWOSBY" if noff == 3 else "NOSBY"
    pts = sorted(set(j["ProcessingTime"] for j in d["Jobs"]))
    print(f"{ds}: {n} inst, first: n={nj}, T={T}, {mtype}, ptimes={pts}")

    # For groups, show range of sizes
    if "groups" in ds and n > 1:
        max_n = 0
        max_T = 0
        for fp in files:
            with open(fp) as f:
                dd = json.load(f)
            nn = len(dd["Jobs"])
            tt = len(dd["EnergyCosts"])
            if nn > max_n:
                max_n = nn
            if tt > max_T:
                max_T = tt
        print(f"  range: n up to {max_n}, T up to {max_T}")
print("DONE")

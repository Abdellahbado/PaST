#!/usr/bin/env python3
"""Analyze exact DP feasibility with ACTUAL instance counts."""
import json, glob, os
from collections import Counter

base = "/Users/mac/Documents/Study/PFE/PaST/data/green-scheduling-bab/Iirc.EnergyStatesAndCostsScheduling/data/datasets/benedikt2025_groups"
files = sorted(glob.glob(os.path.join(base, "*.json")))

exact_ok = 0
exact_skip = 0
skip_detail = {}

for fp in files:
    with open(fp) as f:
        d = json.load(f)
    nj = len(d["Jobs"])
    T = len(d["EnergyCosts"])
    cnts = Counter(j["ProcessingTime"] for j in d["Jobs"])
    pts = sorted(cnts.keys())
    K = len(pts)

    # Simulate the exact NC computation from the solver
    NC = 1
    feasible = True
    for i, p in enumerate(pts):
        if NC * (cnts[p] + 1) > 200_000:
            feasible = False
            break
        NC *= cnts[p] + 1

    total_cells = NC * (T + 2) if feasible else float("inf")
    ok = feasible and total_cells <= 50_000_000

    if ok:
        exact_ok += 1
    else:
        exact_skip += 1
        group = tuple(pts)
        key = (group, nj)
        if key not in skip_detail:
            skip_detail[key] = {
                "count": 0,
                "NC": NC if feasible else "overflow",
                "T": T,
                "cells": total_cells if feasible else "overflow",
            }
        skip_detail[key]["count"] += 1

print(f"Exact DP feasible: {exact_ok}/{len(files)}")
print(f"Exact DP skipped:  {exact_skip}/{len(files)}")
print(f"\nSkipped instance groups:")
for (group, n), info in sorted(skip_detail.items()):
    print(
        f"  {list(group)} n={n}: {info['count']} inst, NC={info['NC']}, T={info['T']}, cells={info['cells']}"
    )

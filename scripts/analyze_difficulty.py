#!/usr/bin/env python3
"""Analyze Table 2 instance difficulty by K (distinct proc times) and n."""
import json, glob, os
from collections import Counter, defaultdict

base = "/Users/mac/Documents/Study/PFE/PaST/data/green-scheduling-bab/Iirc.EnergyStatesAndCostsScheduling/data/datasets/benedikt2025_groups"
files = sorted(glob.glob(os.path.join(base, "*.json")))

k_dist = Counter()
by_group = defaultdict(list)
for fp in files:
    with open(fp) as f:
        d = json.load(f)
    nj = len(d["Jobs"])
    T = len(d["EnergyCosts"])
    pts = sorted(set(j["ProcessingTime"] for j in d["Jobs"]))
    K = len(pts)
    k_dist[K] += 1
    group_key = tuple(pts)
    by_group[group_key].append((nj, T, os.path.basename(fp)))

print("K distribution:")
for k in sorted(k_dist.keys()):
    print(f"  K={k}: {k_dist[k]} instances")

print(f"\nBy processing-time group:")
for group in sorted(by_group.keys()):
    instances = by_group[group]
    K = len(group)
    n_values = sorted(set(nj for nj, T, fn in instances))
    # Check if exact DP is feasible: NC = product(max_ci + 1)
    # For each n, NC = product(n+1) ^ K / ... well, NC = (max_ci+1)^K in worst case
    print(f"  {list(group)} (K={K}): {len(instances)} inst, n={n_values}")
    for n in n_values:
        # Worst case NC for K types with n jobs
        NC = 1
        feasible = True
        for i in range(K):
            NC *= n + 1
            if NC > 200000:
                feasible = False
                break
        max_T = max(T for nj, T, fn in instances if nj == n)
        cells = NC * max_T if feasible else float("inf")
        exact_ok = cells <= 50_000_000 if feasible else False
        n_inst = sum(1 for nj, T, fn in instances if nj == n)
        print(
            f"    n={n:>4}: {n_inst:>3} inst, max_T={max_T:>5}, NC={NC if feasible else 'TOO_BIG':>10}, cells={'OK' if exact_ok else 'SKIP'}"
        )

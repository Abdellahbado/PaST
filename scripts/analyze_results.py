#!/usr/bin/env python3
"""Analyze Table 2 results — group by processing-time set, compare with paper."""
import csv, json, os, sys
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(
    "data/green-scheduling-bab/Iirc.EnergyStatesAndCostsScheduling/data/datasets/benedikt2025_groups"
)
CSV_PATH = "/tmp/table2_results.csv"

# Load results
rows = list(csv.DictReader(open(CSV_PATH)))

# Load instance metadata to get processing time groups
inst_meta = {}
for f in sorted(DATA_DIR.glob("*.json")):
    with open(f) as fp:
        inst = json.load(fp)
    iid = f"benedikt2025_groups/{f.stem}"
    ptimes = sorted(set(j["ProcessingTime"] for j in inst["Jobs"]))
    n = len(inst["Jobs"])
    T = len(inst["EnergyCosts"])
    inst_meta[iid] = {
        "ptimes": tuple(ptimes),
        "n": n,
        "T": T,
        "ptime_counts": {
            p: sum(1 for j in inst["Jobs"] if j["ProcessingTime"] == p) for p in ptimes
        },
    }

# Build groups: (ptimes_tuple, n) -> list of results
groups = defaultdict(list)
for r in rows:
    iid = r["instance_id"]
    meta = inst_meta.get(iid)
    if not meta:
        continue
    key = (meta["ptimes"], meta["n"])
    groups[key].append(
        {
            "iid": iid,
            "runtime": float(r["runtime_sec"]),
            "ub": float(r["ub"]),
            "lb": float(r["lb"]),
            "opt": r["is_optimal"] == "1",
            "T": meta["T"],
        }
    )

# Print analysis
print("=" * 100)
print("TABLE 2 DETAILED ANALYSIS — By Processing-Time Group and Job Count")
print("=" * 100)

total_opt = 0
total_inst = 0

for key in sorted(groups.keys(), key=lambda k: (len(k[0]), k[0], k[1])):
    ptimes, n = key
    results = groups[key]
    K = len(ptimes)
    ntot = len(results)
    nopt = sum(1 for r in results if r["opt"])
    times = [r["runtime"] for r in results]
    avg_t = sum(times) / len(times)
    max_t = max(times)
    min_t = min(times)
    tot_t = sum(times)

    total_opt += nopt
    total_inst += ntot

    print(f"\nK={K}  ptimes={set(ptimes)}  n={n}  |  {nopt}/{ntot} optimal")
    print(
        f"  Time: avg={avg_t:.2f}s  min={min_t:.2f}s  max={max_t:.2f}s  total={tot_t:.1f}s"
    )

print("\n" + "=" * 100)
print(f"TOTAL: {total_opt}/{total_inst} optimal")
print("=" * 100)

# Now aggregate by K for paper comparison
print("\n\nAGGREGATED BY K (distinct processing times):")
print("-" * 80)
by_K = defaultdict(list)
for key, results in groups.items():
    K = len(key[0])
    by_K[K].extend(results)

print(f"{'K':>3} {'#inst':>6} {'#opt':>6} {'avg_t':>10} {'max_t':>10} {'total_t':>10}")
for K in sorted(by_K.keys()):
    results = by_K[K]
    times = [r["runtime"] for r in results]
    nopt = sum(1 for r in results if r["opt"])
    print(
        f"{K:>3} {len(results):>6} {nopt:>6} {sum(times)/len(times):>10.2f} {max(times):>10.2f} {sum(times):>10.1f}"
    )

# Aggregate by n
print("\n\nAGGREGATED BY n:")
print("-" * 80)
by_n = defaultdict(list)
for key, results in groups.items():
    by_n[key[1]].extend(results)

print(f"{'n':>5} {'#inst':>6} {'#opt':>6} {'avg_t':>10} {'max_t':>10} {'total_t':>10}")
for n in sorted(by_n.keys()):
    results = by_n[n]
    times = [r["runtime"] for r in results]
    nopt = sum(1 for r in results if r["opt"])
    print(
        f"{n:>5} {len(results):>6} {nopt:>6} {sum(times)/len(times):>10.2f} {max(times):>10.2f} {sum(times):>10.1f}"
    )


# Paper Table 2 comparison: The paper reports B&B-SPACES results
# From the paper (Table 2): times are in seconds, for n=50 and n=100
# We'll print our times in the same format for easy comparison
print("\n\n" + "=" * 100)
print("PAPER TABLE 2 COMPARISON FORMAT")
print("Our solver vs Paper's B&B-SPACES (10-min time limit)")
print("=" * 100)
print()

# Group by (K, ptimes, n) for table format
for K in sorted(by_K.keys()):
    K_groups = defaultdict(list)
    for key, results in groups.items():
        if len(key[0]) == K:
            K_groups[(key[0], key[1])] = results

    print(f"\n--- K = {K} ---")
    for ptimes, n in sorted(K_groups.keys()):
        results = K_groups[(ptimes, n)]
        times = [r["runtime"] for r in results]
        nopt = sum(1 for r in results if r["opt"])
        pstr = str(set(ptimes))
        print(
            f"  p={pstr:20s} n={n:>4}: {nopt}/{len(results)} opt, "
            f"avg={sum(times)/len(times):.3f}s, max={max(times):.3f}s"
        )

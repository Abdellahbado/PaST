#!/usr/bin/env python3
"""Get ground truth optimal costs from ILP-SPACES results for all NOSBY datasets."""
import json, os

base = "data/paper_instances/results"

for ds_dir in sorted(os.listdir(base)):
    rpath = os.path.join(base, ds_dir, ds_dir, "ILP-SPACES")
    if not os.path.exists(rpath):
        rpath2 = os.path.join(base, ds_dir)
        # Check subdirectories
        for sub in sorted(os.listdir(rpath2)):
            rp = os.path.join(rpath2, sub, "ILP-SPACES")
            if os.path.exists(rp):
                print(f"\n=== {ds_dir}/{sub} / ILP-SPACES ===")
                for i in range(20):
                    fp = os.path.join(rp, f"{i}.json")
                    if not os.path.exists(fp):
                        continue
                    with open(fp) as f:
                        r = json.load(f)
                    print(
                        f"  {i:2d}: Cost={r['Objective']:>8}  LB={r['LowerBound']:>8}  TLR={r['TimeLimitReached']}"
                    )
        continue
    print(f"\n=== {ds_dir} / ILP-SPACES ===")
    for i in range(20):
        fp = os.path.join(rpath, f"{i}.json")
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            r = json.load(f)
        print(
            f"  {i:2d}: Cost={r['Objective']:>8}  LB={r['LowerBound']:>8}  TLR={r['TimeLimitReached']}"
        )

# Also check for aghelinejad in datasets dir
ag_base = "data/paper_instances/datasets/aghelinejad2017a_1"
if os.path.exists(ag_base):
    print(f"\n=== aghelinejad2017a_1 instance info ===")
    for i in range(12):
        fp = os.path.join(ag_base, f"{i}.json")
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            d = json.load(f)
        jobs = d["Jobs"]
        h = len(d["EnergyCosts"])
        pj = [j["ProcessingTime"] for j in jobs]
        print(f"  {i:2d}: n={len(jobs)}, h={h}, sum_pj={sum(pj)}, max_pj={max(pj)}")

# Check aghelinejad results
for meth in ["ILP-REF", "ILP-SPACES"]:
    ag_res = (
        f"data/paper_instances/results/aghelinejad2017a_1/aghelinejad2017a_1/{meth}"
    )
    if os.path.exists(ag_res):
        print(f"\n=== aghelinejad2017a_1 / {meth} ===")
        for i in range(12):
            fp = os.path.join(ag_res, f"{i}.json")
            if not os.path.exists(fp):
                continue
            with open(fp) as f:
                r = json.load(f)
            print(
                f"  {i:2d}: Cost={r['Objective']:>8}  LB={r['LowerBound']:>8}  TLR={r['TimeLimitReached']}"
            )
    else:
        print(f"\n{ag_res} doesn't exist")

# Check what datasets have results
print("\n=== All result subdirs ===")
for ds_dir in sorted(os.listdir(base)):
    rpath = os.path.join(base, ds_dir)
    for sub in sorted(os.listdir(rpath)):
        subp = os.path.join(rpath, sub)
        if os.path.isdir(subp):
            methods = [
                m for m in os.listdir(subp) if os.path.isdir(os.path.join(subp, m))
            ]
            print(f"  {ds_dir}/{sub}: {', '.join(sorted(methods)[:3])}...")

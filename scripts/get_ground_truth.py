#!/usr/bin/env python3
"""Get ground truth optimal costs from ILP-SPACES results for all NOSBY datasets."""
import json, os

base = "data/paper_instances/results"


def show_dataset(ds, methods=("ILP-SPACES", "ILP-REF")):
    for meth in methods:
        rpath = os.path.join(base, ds, ds, meth)
        if not os.path.exists(rpath):
            continue
        print(f"\n=== {ds} / {meth} ===")
        for i in range(12):
            fp = os.path.join(rpath, f"{i}.json")
            if not os.path.exists(fp):
                continue
            with open(fp) as f:
                r = json.load(f)
            obj = r.get("Objective", "?")
            lb = r.get("LowerBound", "?")
            tlr = r.get("TimeLimitReached", "?")
            t = r.get("Time", "?")
            print(f"  {i:2d}: Cost={obj:>8}  LB={lb:>8}  TLR={tlr}  Time={t}")


for ds in [
    "benedikt2020a_prelim",
    "aghelinejad2017a_1",
    "benedikt2020a_large_twosby",
    "benedikt2020a_medium_twosby",
]:
    show_dataset(ds)

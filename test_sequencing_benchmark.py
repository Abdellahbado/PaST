#!/usr/bin/env python3
"""Benchmark the sequencing layer on real Wang2018 instances."""
import time
from glns.benchmark_loader import load_benchmark_instances
from glns.sequencing import evaluate_assignment, make_initial_assignment

insts = load_benchmark_instances("Benchmark/Data")

for scale in ["small", "mls", "large"]:
    subset = [i for i in insts if i.get("scale") == scale]
    if not subset:
        continue
    inst = subset[0]
    assign = make_initial_assignment(inst)

    t0 = time.perf_counter()
    e_opt, c_opt, _, _ = evaluate_assignment(assign, inst, sequencing_mode="optimal")
    dt_opt = time.perf_counter() - t0

    t0 = time.perf_counter()
    e_heu, c_heu, _, _ = evaluate_assignment(assign, inst, sequencing_mode="heuristic")
    dt_heu = time.perf_counter() - t0

    print(
        f"{scale:6s} inst={inst['instance_id']} n={inst['n']:3d} m={inst['m']:2d} "
        f"K={len(set(inst['p']))} | "
        f"opt: E={e_opt:.1f} C={c_opt} ({dt_opt:.3f}s) | "
        f"heu: E={e_heu:.1f} C={c_heu} ({dt_heu:.3f}s)"
    )

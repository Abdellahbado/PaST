"""
CLI for Evolutionary Algorithm Benchmark.

Runs the EA metaheuristic on benchmark instances and compares against:
1. CWP-only (greedy assignment, no optimization)
2. CWP + SPT + DP (existing baseline)
3. EA (Evolutionary Algorithm) with Fast Init -> Fast Eval -> Refinement

Usage:
    conda run -n new-ml-env python -m PaST.cli.run_ea_solver --scale small --num_instances 5
    conda run -n new-ml-env python -m PaST.cli.run_ea_solver --scale mls --num_instances 3 --pop_size 50
"""

import argparse
import random
import time
from typing import List, Tuple

import numpy as np

from PaST.config import DataConfig
from PaST.data.sm_benchmark_data import generate_raw_instance, RawInstance
from PaST.solvers.cwp_solver import construct_schedule_CWP, get_machine_job_sets
from PaST.solvers.baselines_sequence_dp import spt_sequence
from PaST.solvers.evolutionary_solver import (
    GeneticSolver,
    EAConfig,
    Individual,
    evaluate_machine,
)


def cwp_only_solve(instance: RawInstance, epsilon: int) -> Tuple[float, int, float]:
    """CWP-only baseline: greedy assignment, no sequence optimization."""
    t0 = time.perf_counter()

    cwp_result = construct_schedule_CWP(
        processing_times=instance.p,
        energy_rates=[float(e) for e in instance.e],
        ct=np.array(instance.ct, dtype=np.float64),
        epsilon=epsilon,
        top_k=80,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return cwp_result.total_cost, cwp_result.makespan, elapsed_ms


def cwp_spt_dp_solve(instance: RawInstance, epsilon: int) -> Tuple[float, int, float]:
    """CWP + SPT + DP baseline: greedy assignment, SPT sequences, DP scheduling."""
    t0 = time.perf_counter()

    cwp_result = construct_schedule_CWP(
        processing_times=instance.p,
        energy_rates=[float(e) for e in instance.e],
        ct=np.array(instance.ct, dtype=np.float64),
        epsilon=epsilon,
        top_k=80,
    )

    job_sets = get_machine_job_sets(cwp_result, instance.m)
    ct = np.array(instance.ct, dtype=np.int32)

    total_energy = 0.0
    max_makespan = 0

    for m_idx in range(instance.m):
        jobs = job_sets[m_idx]
        if not jobs:
            continue

        # SPT sequence for this machine's jobs
        procs = np.array([instance.p[j] for j in jobs], dtype=np.int32)
        n_m = len(jobs)
        spt_order = spt_sequence(procs, n_m)
        seq = [jobs[i] for i in spt_order]

        energy, start_times, makespan = evaluate_machine(
            job_indices=jobs,
            sequence=seq,
            processing_times=instance.p,
            ct=ct,
            e_rate=instance.e[m_idx],
            T_limit=epsilon,
        )

        total_energy += energy
        max_makespan = max(max_makespan, makespan)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return total_energy, max_makespan, elapsed_ms


def run_benchmark(
    scale: str,
    num_instances: int,
    seed: int,
    pop_size: int,
    generations: int,
    verbose: bool,
):
    """Run benchmark comparisons."""
    config = DataConfig()
    rng_inst = random.Random(seed)

    if scale == "small":
        T_max_choices = [50, 80]
    elif scale == "mls":
        T_max_choices = [100, 300, 350]
    else:  # vls
        T_max_choices = [500]

    ea_config = EAConfig(
        pop_size=pop_size,
        generations=generations,
        crossover_prob=0.8,
        mutation_prob=0.1,
        refine_top_k=1,
    )

    # Results storage
    results = {
        "CWP-only": {"energy": [], "makespan": [], "time_ms": []},
        "CWP+SPT+DP": {"energy": [], "makespan": [], "time_ms": []},
        "EA": {"energy": [], "makespan": [], "time_ms": []},
    }

    print(f"\n{'='*80}")
    print(f"EA Benchmark — scale={scale}, instances={num_instances}")
    print(f"Config: pop={pop_size}, gen={generations}, refine=1")
    print(f"{'='*80}")
    print()

    header = (
        f"{'#':>3} | {'m':>3} {'n':>4} {'T':>4} | "
        f"{'CWP-only':>10} | {'SPT+DP':>10} | {'EA':>10} | "
        f"{'Impr(%)':>8} | {'EA time':>10}"
    )
    print(header)
    print("-" * len(header))

    for i in range(num_instances):
        T_max = rng_inst.choice(T_max_choices)
        instance = generate_raw_instance(config, rng_inst, instance_id=i, T_max=T_max)
        epsilon = T_max

        # 1. CWP-only
        e_cwp, mk_cwp, t_cwp = cwp_only_solve(instance, epsilon)

        # 2. CWP + SPT + DP
        e_spt, mk_spt, t_spt = cwp_spt_dp_solve(instance, epsilon)

        # 3. EA Solver
        solver = GeneticSolver(
            processing_times=instance.p,
            energy_rates=instance.e,
            ct=np.array(instance.ct, dtype=np.int32),
            T_limit=epsilon,
            n_machines=instance.m,
            config=ea_config,
            seed=seed + i,
        )
        
        t0 = time.perf_counter()
        sol, best_log = solver.solve(verbose=verbose)
        e_ea = sol.total_energy
        mk_ea = sol.makespan
        t_ea = (time.perf_counter() - t0) * 1000

        # Store results
        results["CWP-only"]["energy"].append(e_cwp)
        results["CWP-only"]["makespan"].append(mk_cwp)
        results["CWP-only"]["time_ms"].append(t_cwp)

        results["CWP+SPT+DP"]["energy"].append(e_spt)
        results["CWP+SPT+DP"]["makespan"].append(mk_spt)
        results["CWP+SPT+DP"]["time_ms"].append(t_spt)

        results["EA"]["energy"].append(e_ea)
        results["EA"]["makespan"].append(mk_ea)
        results["EA"]["time_ms"].append(t_ea)

        # Improvement over CWP-only
        if e_cwp > 0 and np.isfinite(e_cwp) and np.isfinite(e_ea):
            improvement = (e_cwp - e_ea) / e_cwp * 100
        else:
            improvement = 0.0

        print(
            f"{i:>3} | {instance.m:>3} {instance.n:>4} {T_max:>4} | "
            f"{e_cwp:>10.1f} | {e_spt:>10.1f} | {e_ea:>10.1f} | "
            f"{improvement:>7.2f}% | {t_ea:>8.0f}ms"
        )
        
        if verbose:
            print(f"Log: {best_log}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    for method in results:
        energies = np.array(results[method]["energy"])
        times = np.array(results[method]["time_ms"])
        valid = np.isfinite(energies)

        if np.any(valid):
            print(f"\n{method}:")
            print(f"  Energy: mean={np.mean(energies[valid]):.1f}, "
                  f"std={np.std(energies[valid]):.1f}")
            print(f"  Time:   mean={np.mean(times[valid]):.0f}ms, "
                  f"max={np.max(times[valid]):.0f}ms")

    # Improvement stats
    e_cwp_arr = np.array(results["CWP-only"]["energy"])
    e_ea_arr = np.array(results["EA"]["energy"])
    valid = np.isfinite(e_cwp_arr) & np.isfinite(e_ea_arr) & (e_cwp_arr > 0)
    if np.any(valid):
        improvements = (e_cwp_arr[valid] - e_ea_arr[valid]) / e_cwp_arr[valid] * 100
        print(f"\nEA vs CWP-only improvement:")
        print(f"  Mean: {np.mean(improvements):.2f}%")
        print(f"  Min:  {np.min(improvements):.2f}%")
        print(f"  Max:  {np.max(improvements):.2f}%")
        
    e_spt_arr = np.array(results["CWP+SPT+DP"]["energy"])
    valid2 = np.isfinite(e_spt_arr) & np.isfinite(e_ea_arr) & (e_spt_arr > 0)
    if np.any(valid2):
        improvements2 = (e_spt_arr[valid2] - e_ea_arr[valid2]) / e_spt_arr[valid2] * 100
        print(f"\nEA vs CWP+SPT+DP improvement:")
        print(f"  Mean: {np.mean(improvements2):.2f}%")
        print(f"  Min:  {np.min(improvements2):.2f}%")
        print(f"  Max:  {np.max(improvements2):.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="EA Metaheuristic for Parallel Machine TOU Scheduling"
    )
    parser.add_argument(
        "--scale", default="small", choices=["small", "mls", "vls"],
        help="Instance scale"
    )
    parser.add_argument(
        "--num_instances", type=int, default=5,
        help="Number of instances to test"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pop_size", type=int, default=50)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    run_benchmark(
        scale=args.scale,
        num_instances=args.num_instances,
        seed=args.seed,
        pop_size=args.pop_size,
        generations=args.generations,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

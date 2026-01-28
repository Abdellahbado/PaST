"""
Test script for CWP (Cheapest-Window Placement) Solver.

Tests the CWP heuristic on benchmark instances of different scales:
- Small: m ∈ {3,5,7}, n ∈ {6,10,15,20,25}, T ∈ {50,80}
- MLS (Medium-Large Scale): m ∈ {8,16,25}, n ∈ {30,60,100,150,200}, T ∈ {100,300,350}
- VLS (Very Large Scale): m ∈ {25,30,40}, n ∈ {250,300,350,400,500}, T = 500

Usage:
    conda activate new-ml-env
    python -m PaST.cli.test_cwp_solver
"""

import random
import time
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from PaST.config import DataConfig
from PaST.sm_benchmark_data import generate_raw_instance, RawInstance
from PaST.solvers.cwp_solver import (
    construct_schedule_CWP,
    get_machine_job_sets,
    CWPResult,
)


@dataclass
class TestResult:
    """Result of testing CWP on a single instance."""
    
    scale: str
    m: int
    n: int
    T_max: int
    K: int  # Number of periods
    
    # CWP results
    cwp_cost: float
    cwp_makespan: int
    cwp_feasible: bool
    cwp_time_ms: float
    
    # Job distribution across machines
    jobs_per_machine: List[int]


def evaluate_cwp_on_instance(
    instance: RawInstance,
    epsilon: int = None,
    top_k: int = 80,
) -> TestResult:
    """Evaluate CWP heuristic on a raw benchmark instance.
    
    Args:
        instance: Raw benchmark instance
        epsilon: Optional makespan limit (defaults to T_max)
        top_k: Window search limit for speedup
        
    Returns:
        TestResult with evaluation metrics
    """
    if epsilon is None:
        epsilon = instance.T_max
    
    # Run CWP
    start_time = time.perf_counter()
    result = construct_schedule_CWP(
        processing_times=instance.p,
        energy_rates=[float(e) for e in instance.e],
        ct=np.array(instance.ct, dtype=np.float64),
        epsilon=epsilon,
        top_k=top_k,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    # Get job distribution
    job_sets = get_machine_job_sets(result, instance.m)
    jobs_per_machine = [len(js) for js in job_sets]
    
    return TestResult(
        scale=instance.scale,
        m=instance.m,
        n=instance.n,
        T_max=instance.T_max,
        K=len(instance.Tk),
        cwp_cost=result.total_cost,
        cwp_makespan=result.makespan,
        cwp_feasible=result.feasible,
        cwp_time_ms=elapsed_ms,
        jobs_per_machine=jobs_per_machine,
    )


def run_scale_tests(
    scale: str,
    n_instances: int = 10,
    seed: int = 42,
) -> List[TestResult]:
    """Run CWP tests on instances of a given scale.
    
    Args:
        scale: "small", "mls", or "vls"
        n_instances: Number of instances to test
        seed: Random seed
        
    Returns:
        List of TestResult objects
    """
    config = DataConfig()
    rng = random.Random(seed)
    
    # Set T_max based on scale
    if scale == "small":
        T_max_choices = [50, 80]
    elif scale == "mls":
        T_max_choices = [100, 300, 350]
    else:  # vls
        T_max_choices = [500]
    
    results = []
    for i in range(n_instances):
        T_max = rng.choice(T_max_choices)
        instance = generate_raw_instance(config, rng, instance_id=i, T_max=T_max)
        result = evaluate_cwp_on_instance(instance)
        results.append(result)
    
    return results


def print_results_summary(results: List[TestResult], scale: str) -> None:
    """Print summary statistics for a set of test results."""
    
    if not results:
        print(f"No results for scale '{scale}'")
        return
    
    print(f"\n{'='*70}")
    print(f"CWP Results for Scale: {scale.upper()}")
    print(f"{'='*70}")
    
    n_feasible = sum(1 for r in results if r.cwp_feasible)
    avg_cost = np.mean([r.cwp_cost for r in results])
    avg_time = np.mean([r.cwp_time_ms for r in results])
    avg_makespan = np.mean([r.cwp_makespan for r in results])
    
    print(f"Instances tested: {len(results)}")
    print(f"Feasible solutions: {n_feasible}/{len(results)} ({100*n_feasible/len(results):.1f}%)")
    print(f"Average cost: {avg_cost:.2f}")
    print(f"Average makespan: {avg_makespan:.1f}")
    print(f"Average solve time: {avg_time:.2f} ms")
    
    # Size statistics
    avg_m = np.mean([r.m for r in results])
    avg_n = np.mean([r.n for r in results])
    avg_T = np.mean([r.T_max for r in results])
    avg_K = np.mean([r.K for r in results])
    
    print(f"\nInstance sizes:")
    print(f"  Machines (m): {avg_m:.1f} average")
    print(f"  Jobs (n): {avg_n:.1f} average")
    print(f"  Horizon (T): {avg_T:.1f} average")
    print(f"  Periods (K): {avg_K:.1f} average")
    
    # Job distribution
    all_jobs_per_machine = []
    for r in results:
        all_jobs_per_machine.extend(r.jobs_per_machine)
    
    print(f"\nJob distribution per machine:")
    print(f"  Min: {min(all_jobs_per_machine)}")
    print(f"  Max: {max(all_jobs_per_machine)}")
    print(f"  Avg: {np.mean(all_jobs_per_machine):.1f}")
    print(f"  Std: {np.std(all_jobs_per_machine):.1f}")
    
    # Detailed per-instance results
    print(f"\nPer-instance results:")
    print(f"{'ID':>3} {'m':>3} {'n':>4} {'T':>4} {'K':>4} {'Cost':>10} {'Makespan':>8} {'Time(ms)':>10} {'Feasible':>8}")
    print("-" * 70)
    for i, r in enumerate(results):
        feas_str = "Yes" if r.cwp_feasible else "NO"
        print(f"{i:>3} {r.m:>3} {r.n:>4} {r.T_max:>4} {r.K:>4} {r.cwp_cost:>10.2f} {r.cwp_makespan:>8} {r.cwp_time_ms:>10.2f} {feas_str:>8}")


def main():
    """Main test function."""
    print("=" * 70)
    print("CWP (Cheapest-Window Placement) Solver Test Suite")
    print("=" * 70)
    
    seed = 42
    n_instances_per_scale = 10
    
    all_results = {}
    
    for scale in ["small", "mls", "vls"]:
        print(f"\nRunning tests for scale: {scale}...")
        results = run_scale_tests(scale, n_instances_per_scale, seed)
        all_results[scale] = results
        print_results_summary(results, scale)
    
    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    
    total_instances = sum(len(r) for r in all_results.values())
    total_feasible = sum(sum(1 for x in r if x.cwp_feasible) for r in all_results.values())
    all_times = [x.cwp_time_ms for r in all_results.values() for x in r]
    
    print(f"Total instances: {total_instances}")
    print(f"Total feasible: {total_feasible} ({100*total_feasible/total_instances:.1f}%)")
    print(f"Average solve time: {np.mean(all_times):.2f} ms")
    print(f"Max solve time: {np.max(all_times):.2f} ms")
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

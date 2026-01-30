"""
Diagnostic: How much does ASSIGNMENT matter in parallel machine scheduling?

For each instance:
1. Generate 100 random job-to-machine assignments
2. For each assignment, use SPT sequencing + DP timing on each machine
3. Compute total energy
4. Measure gap: (max_cost - min_cost) / min_cost

If gap >> 5% (the sequencing gap), then assignment is where the learning signal is!
"""

import torch
import numpy as np
import random
from typing import List, Dict, Any, Tuple

from PaST.config import DataConfig
from PaST.sm_benchmark_data import generate_raw_instance, RawInstance
from PaST.solvers.baselines_sequence_dp import spt_sequence, _dp_schedule_fixed_order


def schedule_single_machine(
    job_indices: List[int],
    processing_times: List[int],
    ct: np.ndarray,
    e_single: int,
    T_limit: int,
) -> float:
    """Schedule jobs on a single machine: SPT + DP."""
    if len(job_indices) == 0:
        return 0.0
    
    # Get processing times for this subset
    p_subset = [processing_times[j] for j in job_indices]
    n = len(p_subset)
    
    # SPT sequence (indices 0..n-1 in local ordering)
    local_seq = list(range(n))
    local_seq.sort(key=lambda i: (p_subset[i], i))
    
    # Reorder p_subset according to SPT
    proc_ordered = [p_subset[i] for i in local_seq]
    
    # DP schedule
    total_energy, _ = _dp_schedule_fixed_order(
        processing_times=proc_ordered,
        ct=ct,
        e_single=e_single,
        T_limit=T_limit,
    )
    
    return total_energy


def evaluate_assignment(
    assignment: List[int],  # assignment[j] = machine index for job j
    instance: RawInstance,
) -> float:
    """Compute total energy for a given assignment."""
    m = instance.m
    n = instance.n
    T_limit = instance.T_max
    ct = np.array(instance.ct, dtype=np.int32)
    
    total_energy = 0.0
    
    for machine_idx in range(m):
        # Jobs assigned to this machine
        job_indices = [j for j in range(n) if assignment[j] == machine_idx]
        
        if len(job_indices) == 0:
            continue
        
        # Check feasibility: sum of processing times <= T_limit
        machine_load = sum(instance.p[j] for j in job_indices)
        if machine_load > T_limit:
            return float('inf')  # Infeasible
        
        energy = schedule_single_machine(
            job_indices=job_indices,
            processing_times=instance.p,
            ct=ct,
            e_single=instance.e[machine_idx],
            T_limit=T_limit,
        )
        
        if not np.isfinite(energy):
            return float('inf')
        
        total_energy += energy
    
    return total_energy


def generate_random_assignment(n: int, m: int, rng: random.Random) -> List[int]:
    """Generate a random assignment of n jobs to m machines."""
    return [rng.randint(0, m - 1) for _ in range(n)]


def generate_balanced_random_assignment(n: int, m: int, rng: random.Random) -> List[int]:
    """Generate a more balanced random assignment."""
    # Assign roughly n/m jobs per machine, with some randomness
    assignment = []
    jobs_per_machine = [0] * m
    target = n / m
    
    for j in range(n):
        # Prefer machines that are under-loaded
        weights = [max(0.1, target - jobs_per_machine[i]) for i in range(m)]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        
        machine = rng.choices(range(m), weights=probs, k=1)[0]
        assignment.append(machine)
        jobs_per_machine[machine] += 1
    
    return assignment


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    
    config = DataConfig()
    # Parallel machine instances (2-4 machines, 15-30 jobs)
    config.n_machines_min = 2
    config.n_machines_max = 4
    config.n_jobs_min = 15
    config.n_jobs_max = 30
    
    print("=" * 70)
    print("DIAGNOSTIC: Assignment Gap in Parallel Machine Scheduling")
    print("=" * 70)
    print(f"Config: m ∈ [{config.n_machines_min}, {config.n_machines_max}], "
          f"n ∈ [{config.n_jobs_min}, {config.n_jobs_max}]")
    print()
    
    gaps = []
    num_instances = 100
    num_assignments = 100
    
    for i in range(num_instances):
        T_max = random.choice([80, 100, 120])
        instance = generate_raw_instance(config, random.Random(seed + i), instance_id=i, T_max=T_max)
        
        costs = []
        rng = random.Random(seed + i + 1000)
        
        for _ in range(num_assignments):
            # Use balanced random to avoid too many infeasible
            assignment = generate_balanced_random_assignment(instance.n, instance.m, rng)
            cost = evaluate_assignment(assignment, instance)
            costs.append(cost)
        
        # Filter out infeasible
        valid_costs = [c for c in costs if np.isfinite(c)]
        
        if len(valid_costs) < 10:
            # Too few feasible, skip
            continue
        
        min_c = min(valid_costs)
        max_c = max(valid_costs)
        
        if min_c > 0:
            gap = (max_c - min_c) / min_c * 100.0
            gaps.append(gap)
    
    gaps = np.array(gaps)
    
    print(f"Results over {len(gaps)} instances with sufficient feasible assignments:")
    print(f"  Mean Gap:   {gaps.mean():.2f}%")
    print(f"  Median Gap: {np.median(gaps):.2f}%")
    print(f"  Max Gap:    {gaps.max():.2f}%")
    print(f"  Min Gap:    {gaps.min():.2f}%")
    print(f"  Std Dev:    {gaps.std():.2f}%")
    
    # Percentiles
    print()
    print("Percentiles:")
    for p in [25, 50, 75, 90, 95]:
        print(f"  P{p}: {np.percentile(gaps, p):.2f}%")
    
    print()
    print("=" * 70)
    print("COMPARISON TO SEQUENCING GAP")
    print("=" * 70)
    print("Sequencing Gap (from previous test): Mean 5.47%, Max 19.35%")
    print(f"Assignment Gap (this test):         Mean {gaps.mean():.2f}%, Max {gaps.max():.2f}%")
    
    if gaps.mean() > 10:
        print("\n✅ ASSIGNMENT HAS HIGH IMPACT! Learning should focus here.")
    elif gaps.mean() > 5:
        print("\n⚠️ Assignment has moderate impact, comparable to sequencing.")
    else:
        print("\n❌ Assignment gap is small. May not be worth learning.")


if __name__ == "__main__":
    main()

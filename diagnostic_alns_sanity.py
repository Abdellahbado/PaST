"""
Sanity Check: Can simple ALNS (without learning) improve assignments?

This tests whether destroy-repair operators can find better solutions.
If ALNS improves significantly over random, then PPO-ALNS is worth pursuing.
"""

import numpy as np
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass

from PaST.config import DataConfig
from PaST.sm_benchmark_data import generate_raw_instance, RawInstance
from PaST.solvers.baselines_sequence_dp import spt_sequence, _dp_schedule_fixed_order


@dataclass
class Solution:
    assignment: List[int]  # assignment[j] = machine for job j
    cost: float


def evaluate(assignment: List[int], instance: RawInstance) -> float:
    """Compute total energy for assignment + SPT + DP on each machine."""
    m = instance.m
    n = instance.n
    T_limit = instance.T_max
    ct = np.array(instance.ct, dtype=np.int32)
    
    total = 0.0
    for machine_idx in range(m):
        jobs = [j for j in range(n) if assignment[j] == machine_idx]
        if not jobs:
            continue
        
        load = sum(instance.p[j] for j in jobs)
        if load > T_limit:
            return float('inf')
        
        # SPT + DP
        p_subset = [instance.p[j] for j in jobs]
        local_seq = list(range(len(jobs)))
        local_seq.sort(key=lambda i: p_subset[i])
        proc_ordered = [p_subset[i] for i in local_seq]
        
        energy, _ = _dp_schedule_fixed_order(
            processing_times=proc_ordered,
            ct=ct,
            e_single=instance.e[machine_idx],
            T_limit=T_limit,
        )
        
        if not np.isfinite(energy):
            return float('inf')
        
        total += energy
    
    return total


def random_assignment(n: int, m: int, rng: random.Random) -> List[int]:
    """Generate balanced random assignment."""
    assignment = []
    counts = [0] * m
    target = n / m
    
    for j in range(n):
        weights = [max(0.1, target - counts[i]) for i in range(m)]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        machine = rng.choices(range(m), weights=probs, k=1)[0]
        assignment.append(machine)
        counts[machine] += 1
    
    return assignment


# ============ DESTROY OPERATORS ============

def destroy_random_swap(assignment: List[int], k: int, m: int, rng: random.Random) -> Tuple[List[int], List[int]]:
    """Remove k random jobs from their machines."""
    n = len(assignment)
    removed_jobs = rng.sample(range(n), min(k, n))
    new_assignment = assignment.copy()
    for j in removed_jobs:
        new_assignment[j] = -1  # Mark as unassigned
    return new_assignment, removed_jobs


def destroy_expensive(assignment: List[int], k: int, instance: RawInstance, rng: random.Random) -> Tuple[List[int], List[int]]:
    """Remove jobs that contribute most to cost (long jobs on expensive machines)."""
    n = len(assignment)
    # Score each job: p[j] * e[machine]
    scores = [(instance.p[j] * instance.e[assignment[j]], j) for j in range(n)]
    scores.sort(reverse=True)
    
    # Remove top-k with some randomness
    top_k = min(k * 2, n)
    candidates = [j for _, j in scores[:top_k]]
    removed = rng.sample(candidates, min(k, len(candidates)))
    
    new_assignment = assignment.copy()
    for j in removed:
        new_assignment[j] = -1
    return new_assignment, removed


def destroy_machine_clear(assignment: List[int], m: int, rng: random.Random) -> Tuple[List[int], List[int]]:
    """Clear one random machine."""
    machine = rng.randint(0, m - 1)
    removed = [j for j, a in enumerate(assignment) if a == machine]
    new_assignment = assignment.copy()
    for j in removed:
        new_assignment[j] = -1
    return new_assignment, removed


# ============ REPAIR OPERATORS ============

def repair_greedy(assignment: List[int], removed: List[int], instance: RawInstance) -> List[int]:
    """Assign each removed job to machine with lowest marginal cost increase."""
    new_assignment = assignment.copy()
    m = instance.m
    T_limit = instance.T_max
    ct = np.array(instance.ct, dtype=np.int32)
    
    for j in removed:
        best_machine = 0
        best_cost = float('inf')
        
        for machine in range(m):
            # Check feasibility
            jobs_on_machine = [jj for jj in range(len(new_assignment)) if new_assignment[jj] == machine]
            load = sum(instance.p[jj] for jj in jobs_on_machine) + instance.p[j]
            if load > T_limit:
                continue
            
            # Estimate cost (simple: duration * energy rate)
            cost = instance.p[j] * instance.e[machine]
            if cost < best_cost:
                best_cost = cost
                best_machine = machine
        
        new_assignment[j] = best_machine
    
    return new_assignment


def repair_load_balance(assignment: List[int], removed: List[int], instance: RawInstance) -> List[int]:
    """Assign to least loaded feasible machine."""
    new_assignment = assignment.copy()
    m = instance.m
    T_limit = instance.T_max
    
    for j in removed:
        loads = []
        for machine in range(m):
            jobs_on_machine = [jj for jj in range(len(new_assignment)) if new_assignment[jj] == machine]
            load = sum(instance.p[jj] for jj in jobs_on_machine)
            if load + instance.p[j] <= T_limit:
                loads.append((load, machine))
        
        if loads:
            # Pick least loaded
            loads.sort()
            new_assignment[j] = loads[0][1]
        else:
            # Fallback: random
            new_assignment[j] = random.randint(0, m - 1)
    
    return new_assignment


def run_alns(instance: RawInstance, max_iters: int = 50, seed: int = 0) -> Tuple[float, float, int]:
    """Run simple ALNS and return (initial_cost, best_cost, iters)."""
    rng = random.Random(seed)
    m = instance.m
    n = instance.n
    
    # Initial solution (random balanced)
    assignment = random_assignment(n, m, rng)
    current_cost = evaluate(assignment, instance)
    best_cost = current_cost
    best_assignment = assignment.copy()
    initial_cost = current_cost
    
    if not np.isfinite(current_cost):
        return float('inf'), float('inf'), 0
    
    # SA parameters
    temp = current_cost * 0.1
    cooling = 0.95
    
    destroy_ops = [
        lambda a: destroy_random_swap(a, 3, m, rng),
        lambda a: destroy_expensive(a, 3, instance, rng),
        lambda a: destroy_machine_clear(a, m, rng),
    ]
    
    repair_ops = [repair_greedy, repair_load_balance]
    
    for t in range(max_iters):
        # Select operators
        destroy = rng.choice(destroy_ops)
        repair = rng.choice(repair_ops)
        
        # Destroy
        partial, removed = destroy(assignment)
        if not removed:
            continue
        
        # Repair
        new_assignment = repair(partial, removed, instance)
        new_cost = evaluate(new_assignment, instance)
        
        if not np.isfinite(new_cost):
            continue
        
        # SA acceptance
        delta = new_cost - current_cost
        if delta < 0 or rng.random() < np.exp(-delta / temp):
            assignment = new_assignment
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_assignment = assignment.copy()
        
        temp *= cooling
    
    return initial_cost, best_cost, max_iters


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    
    config = DataConfig()
    config.n_machines_min = 2
    config.n_machines_max = 4
    config.n_jobs_min = 15
    config.n_jobs_max = 30
    
    print("=" * 70)
    print("SANITY CHECK: Can ALNS improve random assignments?")
    print("=" * 70)
    
    improvements = []
    
    for i in range(50):
        T_max = random.choice([80, 100, 120])
        instance = generate_raw_instance(config, random.Random(seed + i), instance_id=i, T_max=T_max)
        
        initial, best, iters = run_alns(instance, max_iters=50, seed=seed + i + 1000)
        
        if np.isfinite(initial) and np.isfinite(best) and initial > 0:
            improvement = (initial - best) / initial * 100.0
            improvements.append(improvement)
    
    improvements = np.array(improvements)
    
    print(f"\nResults over {len(improvements)} instances:")
    print(f"  Mean Improvement: {improvements.mean():.2f}%")
    print(f"  Median Improvement: {np.median(improvements):.2f}%")
    print(f"  Max Improvement: {improvements.max():.2f}%")
    print(f"  Min Improvement: {improvements.min():.2f}%")
    
    print()
    if improvements.mean() > 5:
        print("✅ ALNS significantly improves over random! PPO-ALNS is viable.")
    else:
        print("⚠️ ALNS only marginally improves. PPO guidance may or may not help.")


if __name__ == "__main__":
    main()

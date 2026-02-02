
import sys
import os
import random
import numpy as np

# Add project root to path
sys.path.insert(0, os.getcwd())

from PaST.config import DataConfig
from PaST.data.sm_benchmark_data import generate_raw_instance
from PaST.solvers.alns_parallel import (
    ALNSConfig, 
    build_initial_solution_cwp_spt, 
    _compute_epsilon,
    _destroy_expensive_jobs,
    _destroy_random,
    _best_insertion_for_job,
    evaluate_solution,
    FullEval
)

def check_operators():
    print("Initializing Instance...")
    scale = "medium"
    seed = 42
    
    cfg = ALNSConfig()
    
    # 1. Setup Instance
    data_cfg = DataConfig()
    # We need to manually filter T_max choices as per pm_alns_env._ensure_scale_choices
    if scale == "medium":
         data_cfg.T_max_choices = [t for t in data_cfg.T_max_choices if 80 < int(t) <= 300]
         
    rng = random.Random(seed)
    raw = generate_raw_instance(data_cfg, rng, instance_id=0)
    epsilon = _compute_epsilon(raw, slack_ratio=0.25)
    
    # 2. Build Initial Solution
    cur_sol, cur_ev = build_initial_solution_cwp_spt(raw, epsilon, top_k=80)
    print(f"Initial Energy: {cur_ev.total_energy}")
    
    # 3. Test Expensive Destroy + Greedy Repair
    print("\n--- Testing Expensive Destroy + Greedy Repair ---")
    k = 4 # destroy count
    
    cycling_count = 0
    improved_count = 0
    worse_count = 0
    accepted_count = 0
    
    # We will simulate a simple hill climber (Greedy Accept)
    # using the SAME starting solution to see "Move Diversity"
    
    trials = 50
    # Create copies to avoid mutating baseline
    import copy
    
    unique_outcomes = set()
    
    for i in range(trials):
        # Sol copy
        sol = copy.deepcopy(cur_sol)
        ev = cur_ev # FullEval is immutable-ish (dataclass frozen=True for sub-parts?) No, FullEval is frozen=True.
        
        # Destroy Expensive
        # _destroy_expensive_jobs(raw, epsilon, sol, eval_, k, rng)
        # Note: it returns (new_sol, removed, touched)
        dest_sol, removed, touched = _destroy_expensive_jobs(raw, epsilon, sol, ev, k, rng)
        
        if not removed:
            print(f"Trial {i}: No jobs removed (fallback to random?)")
            continue
            
        # Repair Greedy
        # We need to re-insert 'removed' jobs one by one greedily
        # _best_insertion_for_job(raw, epsilon, sol, eval_, job, cfg, rng)
        
        rep_sol = dest_sol
        rep_ev = evaluate_solution(raw, rep_sol, epsilon) # Re-eval partial?
        # Typically ALNS computes partial eval incrementally or scratch. 
        # _destroy functions return a solution with jobs pop'd.
        # We need a partial_eval for repair.
        rep_ev = evaluate_solution(raw, rep_sol, epsilon)
        
        # Sort removed jobs? Random shuffle usually.
        rng.shuffle(removed)
        
        # Insert loop
        valid_repair = True
        for job in removed:
            sol2, ev2, mi = _best_insertion_for_job(
                raw, epsilon, rep_sol, rep_ev, job, cfg, rng
            )
            if sol2 is None:
                valid_repair = False
                break
            rep_sol = sol2
            rep_ev = ev2
            
        if not valid_repair:
            # Infeasible repair
            continue
            
        # Check outcome
        delta = rep_ev.total_energy - cur_ev.total_energy
        
        # Hash solution to check uniqueness (just sequence lists)
        sol_hash = str(rep_sol.sequences)
        unique_outcomes.add(sol_hash)
        
        if abs(delta) < 1e-9:
            cycling_count += 1
        elif delta < 0:
            improved_count += 1
        else:
            worse_count += 1
            
    print(f"Trials: {trials}")
    print(f"Unique Outcomes (Diversity): {len(unique_outcomes)}")
    print(f"Null Moves (Exact Return): {cycling_count}")
    print(f"Improved: {improved_count}")
    print(f"Worse: {worse_count}")

    # 4. Test Random Destroy + Greedy Repair
    print("\n--- Testing Random Destroy + Greedy Repair ---")
    
    unique_outcomes_rnd = set()
    cycling_count = 0
    improved_count = 0
    worse_count = 0
    
    for i in range(trials):
        sol = copy.deepcopy(cur_sol)
        ev = cur_ev
        
        dest_sol, removed, touched = _destroy_random(sol, k, rng)
        
        if not removed:
            continue
            
        rep_sol = dest_sol
        rep_ev = evaluate_solution(raw, rep_sol, epsilon)
        rng.shuffle(removed)
        
        valid_repair = True
        for job in removed:
            sol2, ev2, mi = _best_insertion_for_job(
                raw, epsilon, rep_sol, rep_ev, job, cfg, rng
            )
            if sol2 is None:
                valid_repair = False
                break
            rep_sol = sol2
            rep_ev = ev2
            
        if not valid_repair:
            continue
            
        delta = rep_ev.total_energy - cur_ev.total_energy
        sol_hash = str(rep_sol.sequences)
        unique_outcomes_rnd.add(sol_hash)
        
        if abs(delta) < 1e-9:
            cycling_count += 1
        elif delta < 0:
            improved_count += 1
        else:
            worse_count += 1
            
    print(f"Trials: {trials}")
    print(f"Unique Outcomes (Diversity): {len(unique_outcomes_rnd)}")
    print(f"Null Moves (Exact Return): {cycling_count}")
    print(f"Improved: {improved_count}")
    print(f"Worse: {worse_count}")

if __name__ == "__main__":
    check_operators()

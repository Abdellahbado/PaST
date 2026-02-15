"""
Diagnostic: Compare CWP Cost vs SGBS Evaluation of CWP Assignment.

Hypothesis: The EA evaluates assignments using SGBS. If SGBS finds a worse 
sequence/schedule than CWP's internal constructive heuristic for the SAME 
assignment, the EA will underestimate the CWP solution's quality and might 
discard it, leading to a final result worse than CWP.
"""

import numpy as np
import random
from PaST.config import DataConfig
from PaST.data.sm_benchmark_data import generate_raw_instance
from PaST.solvers.cwp_solver import construct_schedule_CWP
from PaST.solvers.evolutionary_solver import evaluate_machine, random_q_sgbs_sequence

def test_cwp_vs_sgbs_decoding():
    seed = 42
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    config = DataConfig()
    
    # Instance 2 (from previous run where EA (1532) > CWP (1525))
    # m=5, n=60, T=80
    instance = generate_raw_instance(config, rng, instance_id=2, T_max=80)
    
    # 1. Run CWP
    cwp_res = construct_schedule_CWP(
        instance.p, 
        [float(e) for e in instance.e], 
        np.array(instance.ct, dtype=np.float64), 
        epsilon=80, 
        top_k=80
    )
    
    print(f"CWP Original Cost: {cwp_res.total_cost}")
    
    # 2. Evaluate CWP Assignment using SGBS (as done in EA)
    total_sgbs_cost = 0.0
    
    # Group jobs by machine from CWP assignment
    jobs_per_machine = [[] for _ in range(instance.m)]
    for j, m_idx in cwp_res.assignment.items():
        if m_idx >= 0:
            jobs_per_machine[m_idx].append(j)
            
    print("\nMachine-wise comparison:")
    for m_idx in range(instance.m):
        jobs = jobs_per_machine[m_idx]
        if not jobs:
            continue
            
        # CWP's cost for this machine
        cwp_m_cost = cwp_res.machine_costs[m_idx]
        
        # SGBS Evaluation
        _, sgbs_m_cost, _ = random_q_sgbs_sequence(
            job_indices=jobs,
            processing_times=instance.p,
            ct=np.array(instance.ct, dtype=np.int32),
            e_rate=int(instance.e[m_idx]),
            T_limit=80,
            beta=2,     # Matching EA config
            gamma=2,
            rng=np_rng
        )
        
        diff = sgbs_m_cost - cwp_m_cost
        print(f"M{m_idx}: CWP={cwp_m_cost:.1f}, SGBS={sgbs_m_cost:.1f}, Diff={diff:+.1f}")
        
        total_sgbs_cost += sgbs_m_cost
        
    print(f"\nTotal SGBS Cost for CWP Assignment: {total_sgbs_cost}")
    print(f"Diff: {total_sgbs_cost - cwp_res.total_cost:.1f}")

if __name__ == "__main__":
    test_cwp_vs_sgbs_decoding()

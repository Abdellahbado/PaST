
import torch
import numpy as np
import random
import time
from typing import List, Dict, Any

from PaST.config import DataConfig
from PaST.sm_benchmark_data import generate_raw_instance, RawInstance
from PaST.solvers.batch_dp_solver import BatchSequenceDPSolver

def make_batch_from_instance(instance: RawInstance, n_jobs: int) -> Dict[str, torch.Tensor]:
    """
    Creates a batch dict for BatchSequenceDPSolver from a RawInstance.
    Repetitively copies the instance B=100 times for batch testing.
    """
    BATCH_SIZE = 100
    
    # Extract data from instance (assuming m=1, so we take all jobs and machine 0)
    p = np.array(instance.p, dtype=np.int32)
    e_single = instance.e[0]
    ct = np.array(instance.ct, dtype=np.int32)
    T_max = instance.T_max
    
    # Assume T_limit = T_max for sequencing test
    T_limit = T_max
    
    # Pad if necessary? BatchSequenceDPSolver works with variable lengths if masked,
    # but here we solve 100 permutations of the SAME set of jobs, so shapes are identical.
    
    # Create tensors
    # processing_times: (B, N)
    p_tensor = torch.tensor(p, dtype=torch.long).unsqueeze(0).expand(BATCH_SIZE, -1)
    
    # ct: (B, T)
    ct_tensor = torch.tensor(ct, dtype=torch.long).unsqueeze(0).expand(BATCH_SIZE, -1)
    
    # e_single: (B,)
    e_tensor = torch.tensor(e_single, dtype=torch.long).expand(BATCH_SIZE)
    
    # T_limit: (B,)
    T_tensor = torch.tensor(T_limit, dtype=torch.long).expand(BATCH_SIZE)
    
    # sequence_lengths: (B,)
    n_tensor = torch.tensor(n_jobs, dtype=torch.long).expand(BATCH_SIZE)
    
    return {
        "processing_times": p_tensor,
        "ct": ct_tensor,
        "e_single": e_tensor,
        "T_limit": T_tensor,
        "sequence_lengths": n_tensor,
        "n_jobs": n_jobs
    }

def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Configuration for single machine instances
    # We want to test sequencing hardness, so we generate a load typical for one machine.
    # Scale "small" usually has n=20-50, m=2-5. So n/m ~ 10.
    # Let's target n=15 to 20 jobs on 1 machine.
    config = DataConfig()
    config.n_machines_min = 1
    config.n_machines_max = 1
    config.n_jobs_min = 15
    config.n_jobs_max = 20
    
    print(f"Generating 100 random instances with n_jobs in [{config.n_jobs_min}, {config.n_jobs_max}]...")
    
    gaps = []
    
    for i in range(100):
        # Generate instance
        # T_max choice: usually tight enough to matter but feasible.
        # Let's pick T_max such that utilization is reasonable.
        # If mean p=3, 20 jobs => sum p=60. T_max=80 is reasonably tight.
        T_max = 80
        
        instance = generate_raw_instance(config, random.Random(seed + i), instance_id=i, T_max=T_max)
        n = instance.n
        
        # Prepare batch data
        batch_data = make_batch_from_instance(instance, n)
        
        # Generate 100 random permutations
        # sequences: (B, N)
        sequences = []
        base_seq = list(range(n))
        for _ in range(100):
            seq = base_seq.copy()
            random.shuffle(seq)
            sequences.append(seq)
        
        sequences_tensor = torch.tensor(sequences, dtype=torch.long)
        
        # Run Solver
        # Returns (B,) costs
        costs = BatchSequenceDPSolver.solve(
            job_sequences=sequences_tensor,
            processing_times=batch_data["processing_times"],
            ct=batch_data["ct"],
            e_single=batch_data["e_single"],
            T_limit=batch_data["T_limit"],
            sequence_lengths=batch_data["sequence_lengths"]
        )
        
        # Filter out infeasible sequences (inf cost)
        valid_costs = costs[torch.isfinite(costs)]
        
        if len(valid_costs) == 0:
            # print(f"Instance {i}: All 100 sequences infeasible.")
            continue
            
        min_c = valid_costs.min().item()
        max_c = valid_costs.max().item()
        
        if min_c > 0:
            gap_percent = (max_c - min_c) / min_c * 100.0
            gaps.append(gap_percent)
            # print(f"Instance {i}: Min={min_c:.1f}, Max={max_c:.1f}, Gap={gap_percent:.2f}%")
        else:
            # Zero cost (unlikely unless energy is zero)
            gaps.append(0.0)

    # Statistics
    gaps = np.array(gaps)
    print(f"\nResults over {len(gaps)} valid instances:")
    print(f"Mean Gap:   {gaps.mean():.2f}%")
    print(f"Median Gap: {np.median(gaps):.2f}%")
    print(f"Max Gap:    {gaps.max():.2f}%")
    print(f"Min Gap:    {gaps.min():.2f}%")
    
    threshold_low = 3.0
    threshold_high = 5.0
    
    print("\nConclusion:")
    if gaps.mean() < threshold_low:
        print(f"Gap is SMALL (<{threshold_low}%). Sequencing matters little.")
    elif gaps.mean() > threshold_high:
        print(f"Gap is SIGNIFICANT (>{threshold_high}%). Sequencing matters.")
    else:
        print(f"Gap is MODERATE ({threshold_low}-{threshold_high}%). Hints at structure but signal is weak.")

if __name__ == "__main__":
    main()

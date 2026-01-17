import sys
import os
sys.path.append(os.getcwd())
try:
    from PaST.batch_dp_solver import BatchSequenceDPSolver
except ImportError:
    from batch_dp_solver import BatchSequenceDPSolver

def test_solver():
    B = 2
    N = 3
    T_max = 10
    
    # Prices = 1 everywhere
    ct = torch.ones((B, T_max), dtype=torch.int32)
    
    # Jobs: p=2
    # Sequences: 0, 1, 2
    job_sequences = torch.tensor([[0,1,2], [2,1,0]], dtype=torch.long)
    processing_times = torch.full((B, N), 2, dtype=torch.int32)
    
    # Energy rate = 1
    e_single = torch.ones((B,), dtype=torch.int32)
    
    # T_limit = 10
    T_limit = torch.full((B,), 10, dtype=torch.int32)
    
    print("Testing solver...")
    costs = BatchSequenceDPSolver.solve(
        job_sequences=job_sequences,
        processing_times=processing_times,
        ct=ct,
        e_single=e_single,
        T_limit=T_limit
    )
    
    print("Costs:", costs)
    
    # Expected cost:
    # 3 jobs of length 2 = total duration 6.
    # Price is 1. Energy rate 1.
    # Total energy = 6 * 1 * 1 = 6.
    # Any valid schedule has cost 6.
    
    assert (costs > 0).all(), "Costs should be positive"
    assert (costs == 6).all(), f"Expected cost 6, got {costs}"

if __name__ == "__main__":
    test_solver()

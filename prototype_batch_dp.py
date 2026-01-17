"""Prototype: Batch Sequence DP Solver for PaST-SM

This script implements and benchmarks a fully vectorized PyTorch DP solver
that computes the optimal start times (and total cost) for a fixed sequence of jobs.
This is the core "Timing Layer" for the proposed `ppo_sequence` variant.
"""

import time
import torch
import torch.nn.functional as F


def batch_dp_solve(
    job_sequences: torch.Tensor,   # (B, N) job indices
    processing_times: torch.Tensor, # (B, N) duration of each job
    ct: torch.Tensor,              # (B, T_max_pad) price curve
    e_single: torch.Tensor,        # (B,) energy scalar
    T_limit: torch.Tensor,         # (B,) deadline
) -> torch.Tensor:
    """
    Computes optimal energy cost for a batch of fixed job sequences.
    
    Algorithm:
    Let DP[b, t] = min cost to complete the prefix of jobs 0...i such that 
    the i-th job finishes at or before time t.
    
    Transitions:
    To finish job i at time t_end, it must have started at t_start = t_end - p[i].
    The previous job (i-1) must have finished by t_start.
    So: Cost(end at t_end) = DP_prev[t_start] + JobCost(start at t_start).
    
    DP_curr[t] = min_{u <= t} ( Cost(end at u) )
               = min( DP_curr[t-1], Cost(end at t) )  <-- Prefix Min
    """
    B, N = job_sequences.shape
    device = job_sequences.device
    T_max = ct.shape[1]
    
    # Precompute prefix sum of prices for O(1) interval cost
    # P[b, t] = sum(ct[b, :t])
    price_prefix = torch.zeros(
        (B, T_max + 1), dtype=torch.float32, device=device
    )
    price_prefix[:, 1:] = torch.cumsum(ct.float(), dim=1)
    
    # Initialize DP state: min cost to have finished 0 jobs by time t.
    # Cost is 0 for all t >= 0.
    # min_prev[b, t] = 0.0
    min_prev = torch.zeros((B, T_max + 1), dtype=torch.float32, device=device)
    
    # Create time indices for broadcasting
    t_indices = torch.arange(T_max + 1, device=device).unsqueeze(0) # (1, T+1)
    
    # Process jobs in sequence (sequential loop over N, vectorized over B and T)
    for i in range(N):
        # Gather processing time for the current job in the sequence
        # We need p corresponding to job_sequences[:, i]
        # processing_times is (B, N) where columns are job IDs 0..N-1
        # So we gather from processing_times using job_sequences[:, i]
        job_idx = job_sequences[:, i] # (B,)
        p = torch.gather(processing_times, 1, job_idx.unsqueeze(1)).squeeze(1) # (B,)
        
        # We want to compute cost for job starting at t_start.
        # Length p varies per batch item. 
        # JobCost[b, t_start] = (P[b, t_start + p[b]] - P[b, t_start]) * e_single[b]
        
        # To keep it fully vectorized without a loop over p values, we can use gather.
        # But t_start ranges from 0 to T_max.
        # t_end = t_start + p
        
        # Let's map everything to t_end domain (completion time domain).
        # t_end goes from 0 to T_max.
        # t_start = t_end - p
        
        # Valid t_end must be >= p (since t_start >= 0)
        # Also t_end <= T_limit
        
        # 1. Compute t_start for all possible t_end
        p_expanded = p.unsqueeze(1) # (B, 1)
        t_start = t_indices - p_expanded # (B, T+1)
        
        # 2. Identify valid start times (t_start >= 0)
        # Note: We also need previous job to have finished by t_start.
        # This is handled by fetching min_prev[t_start].
        # If t_start < 0, it's invalid (cost = inf).
        valid_mask = t_start >= 0
        
        # Clamp negative indices to 0 to avoid gather error, then mask result
        safe_t_start = t_start.clamp(min=0).long()
        
        # 3. Retrieve min_prev cost
        # cost_prev[b, t_end] = min_prev[b, safe_t_start]
        cost_prev = torch.gather(min_prev, 1, safe_t_start)
        
        # 4. Compute Job Energy Cost
        # Energy = (P[b, t_end] - P[b, t_start]) * e
        # Note: price_prefix is (B, T+1). t_indices matches columns.
        # P_end: gather at t_indices
        # P_start: gather at safe_t_start
        price_end = torch.gather(price_prefix, 1, t_indices)
        price_start = torch.gather(price_prefix, 1, safe_t_start)
        energy_cost = (price_end - price_start) * e_single.unsqueeze(1)
        
        # 5. Total cost to finish at exactly t_end
        total_cost_at_end = cost_prev + energy_cost
        
        # Apply validity mask (t_start >= 0)
        total_cost_at_end = torch.where(
            valid_mask, total_cost_at_end, torch.full_like(total_cost_at_end, float('inf'))
        )
        
        # Apply Deadline Constraint?
        # Ideally, we just check T_limit at the very end. 
        # But intermediate steps must be feasible too? No, as long as final fits.
        # Actually in this DP, if t_end > T_limit, it's effectively invalid for the final step.
        # We can handle it at the end.
        
        # 6. Compute new min_prev (Prefix Min)
        # min_prev[t] = min(total_cost_at_end[0...t])
        # We can use torch.cummin (available in newer PyTorch) or scan
        if hasattr(torch, 'cummin'):
            min_curr, _ = torch.cummin(total_cost_at_end, dim=1)
        else:
            # Fallback for older torch (slow loop, but torch usually has cummin now)
            vals = []
            curr = total_cost_at_end[:, 0]
            vals.append(curr)
            for k in range(1, T_max + 1):
                curr = torch.min(curr, total_cost_at_end[:, k])
                vals.append(curr)
            min_curr = torch.stack(vals, dim=1)
            
        min_prev = min_curr
        
    # Final result: min cost to finish all N jobs by T_limit
    # Gather at T_limit indices
    # T_limit might be smaller than T_max_pad
    
    # Clamp T_limit to be within bounds (safety)
    safe_T_limit = T_limit.clamp(max=T_max).long().unsqueeze(1)
    final_cost = torch.gather(min_prev, 1, safe_T_limit).squeeze(1)
    
    return final_cost


def benchmark():
    print("Running implementation benchmark...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Parameters matches real use case
    B = 100
    N = 50
    T_max = 500
    
    # Mock data
    job_sequences = torch.argsort(torch.randn(B, N, device=device), dim=1) # Random perms
    processing_times = torch.randint(2, 10, (B, N), device=device).float()
    ct = torch.randint(1, 10, (B, T_max), device=device)
    e_single = torch.ones(B, device=device)
    T_limit = torch.randint(300, 500, (B,), device=device)
    
    # Warmup
    for _ in range(5):
        _ = batch_dp_solve(job_sequences, processing_times, ct, e_single, T_limit)
        
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    # Benchmark
    t0 = time.perf_counter()
    n_iters = 10
    with torch.no_grad():
        for _ in range(n_iters):
             _ = batch_dp_solve(job_sequences, processing_times, ct, e_single, T_limit)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t1 = time.perf_counter()
    
    avg_time = (t1 - t0) / n_iters
    print(f"Average time per batch (B={B}, N={N}, T={T_max}): {avg_time*1000:.2f} ms")
    print(f"Throughput: {B / avg_time:.1f} episodes/sec")
    
    # Validation vs Greedy Check
    # Let's verify correctness on a trivial case (B=1)
    print("\nVerifying logic on trivial case...")
    B_small = 1
    N_small = 3
    T_small = 20
    
    # Job 0: p=2, Job 1: p=3, Job 2: p=2
    # Seq: 0, 1, 2
    seq = torch.tensor([[0, 1, 2]], device=device)
    p = torch.tensor([[2, 3, 2]], device=device).float()
    
    # Price: 10 everywhere, except window [5, 8) is 1
    # Optimal:
    # J0 (2): [5, 7) cost 2
    # J1 (3): [7, 8) cost 1, [8,10) cost 20 -> 21
    # Total wait?
    # This DP finds optimal timing.
    # Prices: [10, 10, 10, 10, 10, 1, 1, 1, 10, 10...]
    # Time 5, 6, 7 are cheap.
    ct_small = torch.full((1, T_small), 10, device=device)
    ct_small[0, 5:8] = 1
    e = torch.tensor([1.0], device=device)
    T_lim = torch.tensor([20], device=device)
    
    cost = batch_dp_solve(seq, p, ct_small, e, T_lim)
    print(f"Computed Cost: {cost.item()}")
    
    # Manual calc:
    # J0 (len 2) best: [5, 7) -> cost 1+1 = 2. Ends at 7.
    # J1 (len 3) starts >= 7. Best: [7, 10).
    #   [7, 8) price 1. [8, 10) price 10.
    #   Total 1 + 10 + 10 = 21. Ends at 10.
    # J2 (len 2) starts >= 10. Best [10, 12). Cost 20.
    # Total = 2 + 21 + 20 = 43.
    print(f"Expected Cost: 43.0")
    
    if abs(cost.item() - 43.0) < 1e-5:
        print("Validation PASSED.")
    else:
        print("Validation FAILED.")

if __name__ == "__main__":
    benchmark()

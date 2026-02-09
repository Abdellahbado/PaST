
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import time
from PaST.solvers.optimal_benchmark_dp import solve_optimal_benchmark_dp

def test_anytime_dp():
    print("Testing Anytime DP...")
    
    # Sparse / Numba test
    # K=7, counts=6 each. Radices=7.
    # State space 7^7 = 823,543.
    # T * states = 500 * 823k = 411M > 12M threshold -> Triggers Sparse DP
    print("-" * 20)
    print("Testing Sparse/Numba DP (K=7)...")
    
    T = 500
    prices = np.sin(np.linspace(0, 20, T)) + 2.0
    
    p = []
    for length in [1, 2, 3, 4, 5, 6, 7]:
        p.extend([length] * 6)
    
    print(f"Instance: T={T}, {len(p)} jobs, K={len(set(p))}")
    
    print(f"Solving with very short time limit (0.1s)...")
    t0 = time.time()
    res = solve_optimal_benchmark_dp(p, prices, time_limit=0.1)
    dur = time.time() - t0
    
    print(f"Result in {dur:.4f}s:")
    print(f"  Feasible: {res.feasible}")
    print(f"  Cost: {res.cost}")
    print(f"  Is Optimal: {res.is_optimal}")
    print(f"  Timed Out: {res.timed_out}")
    print(f"  Scheduled jobs: {len(res.schedule)} / {len(p)}")
    
    if res.feasible and not res.is_optimal and res.timed_out and res.cost < float('inf'):
        print("SUCCESS: Sparse DP returned separate sub-optimal solution on timeout.")
    else:
        print("FAILURE: Sparse DP did not return expected partial solution.")
        if not res.timed_out:
            print("  Reason: Solver finished too fast, increase difficulty.")
        elif res.cost == float('inf'):
             print("  Reason: Returned Infinity cost.")


    print("-" * 20)
    print("Solving with sufficient time limit (5.0s)...")
    t0 = time.time()
    res_opt = solve_optimal_benchmark_dp(p, prices, time_limit=5.0)
    dur = time.time() - t0
    print(f"Result in {dur:.4f}s:")
    print(f"  Feasible: {res_opt.feasible}")
    print(f"  Cost: {res_opt.cost}")
    print(f"  Is Optimal: {res_opt.is_optimal}")
    
    if res_opt.feasible and res_opt.is_optimal:
        print("SUCCESS: Returned optimal solution when time allowed.")
    else:
        print("FAILURE: Did not return optimal solution.")

if __name__ == "__main__":
    test_anytime_dp()

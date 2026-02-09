import pandas as pd
import numpy as np
import os

def analyze(csv_path, scale_label):
    df = pd.read_csv(csv_path)
    print(f"\n=== Analysis for {scale_label} (N={len(df)}) ===")
    
    # Runtimes
    print(f"Avg Time: DP={df['dp_time_sec'].mean():.3f}s | BnB={df['bnb_time_sec'].mean():.3f}s")
    print(f"Max Time: DP={df['dp_time_sec'].max():.3f}s | BnB={df['bnb_time_sec'].max():.3f}s")
    
    # Optimality / Timeouts
    dp_optimal = df['dp_is_optimal'].sum()
    bnb_timeouts = df['bnb_timed_out'].sum()
    print(f"DP Optimal: {dp_optimal}/{len(df)} | BnB Timeouts: {bnb_timeouts}/{len(df)}")
    
    # Cost Comparison
    # Note: If BnB timed out, it might have a higher cost than DP (or lower if it found a good feasible but didn't prove optimality, but usually DP wins on timeout)
    df['cost_diff'] = df['bnb_cost'] - df['dp_cost'] # Positive means DP is better (lower cost)
    dp_wins_cost = (df['cost_diff'] > 1e-6).sum()
    bnb_wins_cost = (df['cost_diff'] < -1e-6).sum()
    ties_cost = (abs(df['cost_diff']) <= 1e-6).sum()
    
    print(f"Cost Wins: DP={dp_wins_cost} | BnB={bnb_wins_cost} | Ties={ties_cost}")
    
    # Makespan Comparison
    df['ms_diff'] = df['bnb_makespan'] - df['dp_makespan']
    dp_wins_ms = (df['ms_diff'] > 0).sum()
    bnb_wins_ms = (df['ms_diff'] < 0).sum()
    ties_ms = (df['ms_diff'] == 0).sum()
    
    print(f"Makespan Wins: DP={dp_wins_ms} | BnB={bnb_wins_ms} | Ties={ties_ms}")
    
    # Aggregates
    print(f"Avg Cost: DP={df['dp_cost'].mean():.2f} | BnB={df['bnb_cost'].mean():.2f}")
    print(f"Avg Makespan: DP={df['dp_makespan'].mean():.1f} | BnB={df['bnb_makespan'].mean():.1f}")

if __name__ == "__main__":
    small_csv = "/Users/mac/Documents/Study/PFE/PaST/analysis_out/rigorous_small_200s/summary.csv"
    medium_csv = "/Users/mac/Documents/Study/PFE/PaST/analysis_out/rigorous_mls_200s/summary.csv"
    
    if os.path.exists(small_csv):
        analyze(small_csv, "Small (T=80)")
    if os.path.exists(medium_csv):
        analyze(medium_csv, "Medium (T=300)")

"""Aggregate DP vs BnB Results.

Scans all subdirectories in PaST/analysis_out matching 'compare_dp_bnb_*'
and aggregates results from the JSON files.

Metrics reported:
- Success rate (feasible/optimal)
- Runtime (mean ± std)
- Cost (mean ± std)
- Makespan (mean ± std)

Filters out:
- Instances where DP returned infinity (timeout/infeasible)
- Instances where BnB returned infinity
"""

import glob
import json
import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Any

def scan_and_aggregate(root_dir: str):
    # Pattern to match: PaST/analysis_out/compare_dp_bnb_*
    search_pattern = os.path.join(root_dir, "compare_dp_bnb_*", "*.json")
    files = glob.glob(search_pattern)
    
    print(f"Found {len(files)} JSON files in {root_dir}/compare_dp_bnb_*")
    
    data = []
    
    for fpath in files:
        try:
            with open(fpath, "r") as f:
                res = json.load(f)
            
            # Extract key metrics
            dp = res.get("dp", {})
            bnb = res.get("bnb", {})
            
            # Check feasibility and timeouts
            dp_cost = dp.get("cost", float("inf"))
            bnb_cost = bnb.get("cost", float("inf"))
            
            dp_feasible = np.isfinite(dp_cost)
            bnb_feasible = np.isfinite(bnb_cost)
            
            bnb_timed_out = bnb.get("timed_out", False)
            
            # Basic info
            scale = res.get("scale", "unknown")
            n_jobs = res.get("n_jobs_sm")
            unique_p = res.get("unique_p_sm")
            
            row = {
                "file": os.path.basename(fpath),
                "scale": scale,
                "n_jobs": n_jobs,
                "unique_p": unique_p,
                
                # Feasibility metrics
                "dp_feasible": dp_feasible,
                "bnb_feasible": bnb_feasible,
                "bnb_timed_out": bnb_timed_out,
                
                # Metrics (use NaN for infeasible to avoid polluting means)
                "dp_cost": dp_cost if dp_feasible else np.nan,
                "dp_time": dp.get("time_sec", np.nan),
                "dp_makespan": dp.get("finish_time", np.nan),
                
                "bnb_cost": bnb_cost if bnb_feasible else np.nan,
                "bnb_time": bnb.get("time_sec", np.nan),
                "bnb_nodes": bnb.get("nodes", 0),
            }
            
            # Comparison metrics (only if both feasible)
            if dp_feasible and bnb_feasible:
                row["cost_diff"] = dp_cost - bnb_cost
                row["time_ratio"] = dp.get("time_sec", 1e-9) / max(1e-9, bnb.get("time_sec", 1e-9))
                
                # Classification
                if abs(dp_cost - bnb_cost) < 1e-9:
                    row["cmp"] = "equal"
                elif dp_cost < bnb_cost:
                    row["cmp"] = "dp_better"
                else:
                    row["cmp"] = "bnb_better"
            else:
                row["cost_diff"] = np.nan
                row["time_ratio"] = np.nan
                row["cmp"] = "infeasible"
            
            # Calculate BnB makespan
            bnb_sched = bnb.get("schedule", [])
            bnb_makespan = np.nan
            if bnb_sched:
                end_times = [item[2] for item in bnb_sched if len(item) >= 3]
                if end_times:
                    bnb_makespan = max(end_times)
            row["bnb_makespan"] = bnb_makespan
            
            data.append(row)
            
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            continue
            
    if not data:
        print("No valid instances found.")
        return

    df = pd.DataFrame(data)
    
    print("\n" + "="*80)
    print(f"AGGREGATED RESULTS ({len(df)} instances processed)")
    print("="*80)
    
    # Group by scale
    for scale, group in df.groupby("scale"):
        print(f"\n--- Scale: {scale} (Total: {len(group)}) ---")
        
        # Feasibility
        n_dp_feas = group["dp_feasible"].sum()
        n_bnb_feas = group["bnb_feasible"].sum()
        print(f"Feasible: DP={n_dp_feas}/{len(group)}, BnB={n_bnb_feas}/{len(group)}")
        
        # Filter for valid comparisons for cost/time stats
        valid = group[group["dp_feasible"] & group["bnb_feasible"]]
        
        if len(valid) == 0:
            print("  No instances where both solvers found a solution.")
            continue
            
        # Comparison Breakdown
        counts = valid["cmp"].value_counts()
        n_equal = counts.get("equal", 0)
        n_dp_better = counts.get("dp_better", 0)
        n_bnb_better = counts.get("bnb_better", 0)
        
        print(f"\n[COMPARISON (n={len(valid)})]")
        print(f"  DP == BnB: {n_equal} ({n_equal/len(valid)*100:.1f}%)")
        print(f"  DP < BnB:  {n_dp_better} ({n_dp_better/len(valid)*100:.1f}%) (DP better)")
        print(f"  DP > BnB:  {n_bnb_better} ({n_bnb_better/len(valid)*100:.1f}%) (BnB better)")
        
        if n_dp_better > 0:
            timeouts = valid[(valid["cmp"] == "dp_better") & (valid["bnb_timed_out"])]
            print(f"  -> In {len(timeouts)}/{n_dp_better} 'DP better' cases, BnB timed out.")

        # Cost statistics
        print(f"\n[COST]")
        print(f"  DP Mean:  {valid['dp_cost'].mean():.2f} ± {valid['dp_cost'].std():.2f}")
        print(f"  BnB Mean: {valid['bnb_cost'].mean():.2f} ± {valid['bnb_cost'].std():.2f}")
        print(f"  Avg Diff: {valid['cost_diff'].mean():.2f} (DP - BnB)")
        
        # Time statistics
        print(f"\n[TIME (seconds)]")
        print(f"  DP Mean:  {valid['dp_time'].mean():.4f} ± {valid['dp_time'].std():.4f}")
        print(f"  BnB Mean: {valid['bnb_time'].mean():.4f} ± {valid['bnb_time'].std():.4f}")
        print(f"  Med Ratio: {valid['time_ratio'].median():.2f}x (DP/BnB)")
        
        # Makespan statistics
        print(f"\n[MAKESPAN]")
        print(f"  DP Mean:  {valid['dp_makespan'].mean():.1f} ± {valid['dp_makespan'].std():.1f}")
        print(f"  BnB Mean: {valid['bnb_makespan'].mean():.1f} ± {valid['bnb_makespan'].std():.1f}")
        
    print("\n" + "="*80)
    print("Top 5 Cases where DP Cost < BnB Cost (BnB suboptimal/timeout):")
    better_dp = df[df["cmp"] == "dp_better"].sort_values("cost_diff")
    if not better_dp.empty:
        print(better_dp[["file", "dp_cost", "bnb_cost", "bnb_timed_out", "cost_diff"]].head(5).to_string(index=False))
    else:
        print("None")
        
    print("\nTop 5 Cases where BnB Cost < DP Cost (DP suboptimal/timeout?):")
    better_bnb = df[df["cmp"] == "bnb_better"].sort_values("cost_diff", ascending=False)
    if not better_bnb.empty:
        print(better_bnb[["file", "dp_cost", "bnb_cost", "bnb_timed_out", "cost_diff"]].head(5).to_string(index=False))
    else:
        print("None")
        
    print("\nTop 5 Slowest DP Instances (Feasible):")
    slowest = df[df["dp_feasible"]].sort_values("dp_time", ascending=False).head(5)
    print(slowest[["file", "n_jobs", "unique_p", "dp_time", "bnb_time"]].to_string(index=False))

if __name__ == "__main__":
    root = "PaST/analysis_out"
    if len(sys.argv) > 1:
        root = sys.argv[1]
    scan_and_aggregate(root)

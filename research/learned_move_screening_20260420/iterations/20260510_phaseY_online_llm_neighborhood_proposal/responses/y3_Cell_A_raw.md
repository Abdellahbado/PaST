# DeepSeek V4 Pro Response — Cell_A

## Proposal

```json
{
  "proposal_name": "y3_attack_highgap_rate2_sources_small_medium",
  "move_family": "insert_inter",
  "source_machines": ["M24", "M19", "M22", "M6", "M17"],
  "target_machines": ["M12", "M0", "M20", "M15", "M21"],
  "job_size_classes": ["small", "medium"],
  "max_candidates": 20,
  "ranking_hint": "cost_gap",
  "diversity_rule": "per_source",
  "fallback_if_empty": "top_s2_same_budget",
  "rationale": "M24(gap=124,underexplored)/M19(gap=112)/M22(gap=96)/M6(gap=108)/M17(gap=92) are the top-5 gap machines—highest hidden repacking room where cheap LB is most inaccurate. All are rate=2 with many small/medium jobs. Current targets M19/M22 are saturated(CT=5). Switch to underexplored near-empty targets M12/M0/M20/M15/M21 with rate=3-4 and slack=346—lower energy cost than the empty rate=5-6 machines. cost_gap ranking prioritizes biggest-gap sources first."
}
```

## Analysis

1. **Most expensive machines**: M7/M8/M14 have highest EC (1594 each) at rate=1. These are fully loaded with medium/large jobs. However, their gap=0 (EC=RLB), indicating that the cheap LB agrees with exact DP cost—there is no hidden re-packing room. These machines are not the best attack surface.

2. **Biggest gaps (hidden re-packing room)**: M24 (gap=124), M19 (gap=112), M6 (gap=108), M22 (gap=96), M17 (gap=92). These rate=2 machines have significant discrepancies between cheap LB and exact DP, meaning the DP can potentially find better schedules if given different job mixes.

3. **Underexplored sources**: M24, M19, M22 have CS=0 (never hit in core shortlist as sources), despite being among the top gap machines. They have been targeted heavily (CT=5 for M19/M22) but never used as sources.

4. **Target saturation**: M19 and M22 have CT=5 (targeted every round), suggesting further targeting yields diminishing returns. The underexplored targets list shows empty machines M18/M16/M11/M9/M5 but these are all rate=5-6 (expensive energy). Better targets are the near-empty rate=3-4 machines: M12, M0, M20, M15, M21 (all have slack=346, 1 job, rate 3-4).

5. **Job size strategy**: Recent accepted moves used medium jobs (size 5-7). The job size distribution in the highest cost quartile shows 38% small, 30% medium, 32% large. Small-medium gives the most flexibility while avoiding the large jobs that require more slack.

6. **Prior results**: Prior best TEC from Phase S is 6852 (Δ=-94), Phase S llm_exception is 6869 (Δ=-77). The random baseline best is 6893 (Δ=-53). This proposal aims to find improvements by attacking previously unexplored source-target combinations.

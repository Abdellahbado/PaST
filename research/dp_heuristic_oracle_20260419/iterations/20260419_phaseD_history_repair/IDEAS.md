# Ideas

## Candidate paths considered

1. `history_repair_dp_ranked`
   - A-SGH-style repair from previous `epsilon` assignment
   - displaced jobs are reinserted with DP-LB-ranked machine choices
   - no post-repair local search

2. `history_repair_dp_ranked_relocate`
   - same repair as above
   - then relocate-only exact-DP cleanup

3. `history_repair_priority_displaced`
   - same repair scaffold
   - displaced-job order prioritized by disruption score (`rate * p`)
   - no post-repair local search

4. `history_repair_priority_displaced_relocate`
   - prioritized displaced repair
   - then relocate-only exact-DP cleanup

## This pass selections

- implemented and tested:
  - `history_repair_dp_ranked`
  - `history_repair_priority_displaced_relocate`

- deferred:
  - `history_repair_dp_ranked_relocate`
  - `history_repair_priority_displaced`

Reason for bounded scope:

- keep implementation thin while testing both dimensions:
  - DP-ranked repair behavior
  - impact of relocate cleanup after repair

# PLAN28: Block Realizability Diagnostics and Bounded Repair

## Objective

Test whether hard irregular `K=8/10/12`, `n=1000` rows fail because Step-1 recovered blocks are not locally realizable enough for the finite job multiset.

This is a bounded experiment. First diagnose. Repair only if the diagnostics show signal.

## Read first

Work in `/Users/mac/Documents/Study/PFE/PaST`.

Read these files before editing code:

1. `research/k_vs_arithmetic_axes_20260412/ACTIVE.md`
2. `research/k_vs_arithmetic_axes_20260412/iterations/20260428_realizability_aware_blocks/SUMMARY.md`
3. `research/k_vs_arithmetic_axes_20260412/iterations/20260428_realizability_aware_blocks/PROBLEM.md`
4. `research/k_vs_arithmetic_axes_20260412/iterations/20260428_realizability_aware_blocks/IDEAS.md`
5. `research/k_vs_arithmetic_axes_20260412/iterations/20260421_paper_group_recovery/SUMMARY.md`
6. `research/k_vs_arithmetic_axes_20260412/csv/plan26/PLAN26_multi_idea_notes.md`
7. `research/k_vs_arithmetic_axes_20260412/METHOD_BOUNDARIES.md`
8. `solvers/cpp/stateful_dp_solver.hpp`
9. `solvers/cpp/stateful_dp_solver.cpp`
10. `solvers/cpp/stateful_compare.cpp`
11. `research/k_vs_arithmetic_axes_20260412/run_plan18_k_boundary_refine_n1000.py`
12. `research/k_vs_arithmetic_axes_20260412/run_plan26_multi_idea.py` if present

## Do not do

- Do not delete old artifacts.
- Do not change accepted baseline defaults silently.
- Do not revive `smart_reconstruct(...)` as the main method.
- Do not continue Step-4 corridor exact DP.
- Do not implement full column generation, branch-and-price, MIP/SAT, or a new global relaxation.
- Do not mark repair-only UB gains as exact optimality.

## Phase A: diagnostics only

Add block-realizability diagnostics around the recovered blocks used by Step 3.

Required aggregate CSV fields:

- `block_realiz_diag_active`
- `block_realiz_blocks_total`
- `block_realiz_bad_blocks`
- `block_realiz_bad_rate`
- `block_realiz_first_bad_block`
- `block_realiz_min_finite_patterns`
- `block_realiz_mean_finite_patterns`
- `block_realiz_base_path_survives`
- `block_realiz_base_reject_reason`
- `block_realiz_diag_time_sec`
- `block_realiz_diag_skipped`
- `block_realiz_diag_skip_reason`

If feasible, also write a per-block diagnostic CSV:

- `csv/plan28/PLAN28_block_realizability_blocks.csv`

Per-block fields should include:

- `case_id`, `family`, `K`, `seed`, `block_index`
- `block_start`, `block_end`, `block_work`
- `chosen_counts`
- `chosen_counts_local_feasible`
- `finite_pattern_count`
- `best_local_pattern_cost`
- `reject_reason`

Diagnostics must be capped. If a block is too expensive to diagnose, mark it skipped and continue.

## Phase A rows

Run only this diagnostic gate first:

- easy K8: `{1,2,3,4,5,6,7,8}`, `n=1000`, seeds `0,1`
- easy K10: `{1,2,3,4,5,6,7,8,9,10}`, `n=1000`, seeds `0,1`
- easy K12: `{1,2,3,4,5,6,7,8,9,10,11,12}`, `n=1000`, seeds `0,1`
- hardA K8: `{2,3,4,5,7,11,13,17}`, `n=1000`, seeds `0,1`
- hardA K10: `{2,3,4,5,7,11,13,17,19,23}`, `n=1000`, seeds `0,1`
- hardA K12: `{2,3,4,5,7,11,13,17,19,23,29,31}`, `n=1000`, seeds `0,1`
- hardB K8: `{3,5,7,11,13,17,19,23}`, `n=1000`, seeds `0,1`
- hardB K10: `{3,5,7,11,13,17,19,23,29,31}`, `n=1000`, seeds `0,1`
- hardB K12: `{3,5,7,11,13,17,19,23,29,31,37,41}`, `n=1000`, seeds `0,1`

Artifacts:

- `csv/plan28/PLAN28_block_realizability_diagnostics_raw.csv`
- `csv/plan28/PLAN28_block_realizability_diagnostics_summary.csv`
- `csv/plan28/PLAN28_block_realizability_notes.md`

Gate A passes only if diagnostics separate easy from hard rows or correlate with final gap/no-incumbent outcome.

If Gate A fails, stop and document Decision C.

## Phase B: bounded block repair

Run only if Gate A passes.

Implement an experimental block repair toggle. The repair modifies only the recovered block list passed into Step 3.

Suggested env vars:

- `PAST_BLOCK_REALIZ_REPAIR=1`
- `PAST_BLOCK_REALIZ_REPAIR_POLICY=merge_bad_prev|merge_bad_next|merge_bad_triplet|small_boundary_shift|best_local_repair`
- `PAST_BLOCK_REALIZ_REPAIR_MAX_BLOCKS=3`
- `PAST_BLOCK_REALIZ_DIAG_MAX_PATTERNS=200000`
- `PAST_BLOCK_REALIZ_REPAIR_TIME_LIMIT=120`

Required repair variants:

- `baseline`
- `diag_only`
- `merge_bad_prev`
- `merge_bad_next`
- `merge_bad_triplet`
- `best_local_repair`

Phase B rows:

- hardA K10, seeds `0,1,2,3`
- hardB K10, seeds `0,1,2,3`

Artifacts:

- `csv/plan28/PLAN28_block_repair_raw.csv`
- `csv/plan28/PLAN28_block_repair_compare.csv`
- `csv/plan28/PLAN28_block_repair_summary.csv`

Gate B passes if:

- at least `4/8` rows improve final UB or final gap versus baseline,
- mean final gap improves,
- no memory kills,
- timeout count does not increase,
- runtime increase is at most `20%` unless gap improvement is material and documented.

## Phase C: controls only if Phase B passes

Run small controls:

- hardA K8 seeds `0,1`
- hardB K8 seeds `0,1`
- hardA K12 seed `0`
- hardB K12 seed `0`

Do not run a broad campaign.

## Memory safety

- One heavy solver row at a time.
- Default RSS cap: `16 GB`.
- Absolute cap: `20 GB`, only if explicitly justified in notes.
- Flush every row immediately.
- Do not buffer full solver output in memory.
- Store stderr/stdout tails only.
- If RSS exceeds cap, kill the row, mark `memory_killed=1`, and continue.

## Final decision labels

- `A`: diagnostics correlate and repair improves enough to expand.
- `B`: diagnostics correlate but simple repairs fail; next method needs smarter recovered-block construction.
- `C`: diagnostics do not correlate; abandon this direction.
- `D`: implementation blocker, diagnostics inconclusive.
- `E`: repair improves only one family or seed subset; keep as targeted option, not default.

## Required docs to update

Update only after results exist:

- `research/k_vs_arithmetic_axes_20260412/LOG.md`
- `research/k_vs_arithmetic_axes_20260412/iterations/20260428_realizability_aware_blocks/RESULTS.md`
- `research/k_vs_arithmetic_axes_20260412/iterations/20260428_realizability_aware_blocks/BLOCKERS.md`
- `research/k_vs_arithmetic_axes_20260412/iterations/20260428_realizability_aware_blocks/SUMMARY.md`
- `research/k_vs_arithmetic_axes_20260412/RESULTS.md` only if the result changes the public method story
- `research/k_vs_arithmetic_axes_20260412/BLOCKERS.md` only if the result changes the public blocker story

Return only when Phase A is complete. If Phase A passes, continue through Phase B without asking. If Phase B passes, run Phase C controls. If a gate fails, stop after documenting the failure clearly.

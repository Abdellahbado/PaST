# PLAN29: Multi-View Adjacent Block Reconstruction

## Objective

Test whether hard irregular fixed-`n=1000` K-axis rows improve when Step 3 receives several alternative adjacent coarsenings of the recovered block list.

This is a bounded method experiment. It is not a broad campaign.

## Read first

Work in `/Users/mac/Documents/Study/PFE/PaST`.

Read in this order:

1. `research/k_vs_arithmetic_axes_20260412/ACTIVE.md`
2. `research/k_vs_arithmetic_axes_20260412/iterations/20260428_multiview_block_reconstruction/SUMMARY.md`
3. `research/k_vs_arithmetic_axes_20260412/iterations/20260428_multiview_block_reconstruction/PROBLEM.md`
4. `research/k_vs_arithmetic_axes_20260412/iterations/20260428_multiview_block_reconstruction/IDEAS.md`
5. `research/k_vs_arithmetic_axes_20260412/iterations/20260428_realizability_aware_blocks/SUMMARY.md`
6. `research/k_vs_arithmetic_axes_20260412/csv/plan28/PLAN28_block_realizability_notes.md`
7. `research/k_vs_arithmetic_axes_20260412/iterations/20260421_paper_group_recovery/SUMMARY.md`
8. `solvers/cpp/stateful_dp_solver.hpp`
9. `solvers/cpp/stateful_dp_solver.cpp`
10. `solvers/cpp/stateful_compare.cpp`
11. `research/k_vs_arithmetic_axes_20260412/run_plan18_k_boundary_refine_n1000.py`

## Hard constraints

- Only merge adjacent recovered blocks.
- Do not use PLAN28 `bad_rate` or local-block infeasibility as the coarsening signal.
- Do not revive `smart_reconstruct(...)`.
- Do not continue Step-4 corridor exact DP.
- Do not change accepted defaults silently.
- Do not mark repair/coarsening UB gains as exact unless normal `UB=LB` certification occurs.
- Keep all new behavior behind explicit experimental env vars.

## Implementation

Add an experimental block-view layer inside `pack_recovered_blocks(...)`, after the current overlap/touching merge produces `merged`.

Suggested env vars:

- `PAST_BLOCK_VIEW_POLICY=baseline|coarsen2|coarsen3|target_b|price_preserve|arith_adaptive|best_of_views`
- `PAST_BLOCK_VIEW_TARGET_B=12`
- `PAST_BLOCK_VIEW_MAX_VIEWS=6`
- `PAST_BLOCK_VIEW_TIME_LIMIT_SEC=900`

Required view policies:

1. `baseline`: current behavior.
2. `coarsen2`: merge adjacent blocks in pairs.
3. `coarsen3`: merge adjacent blocks in triples.
4. `target_B12`: greedily remove adjacent boundaries until about 12 blocks remain.
5. `target_B8`: greedily remove adjacent boundaries until about 8 blocks remain.
6. `price_preserve_B12`: remove weak price-boundaries first, but preserve boundaries with large local price jumps.
7. `arith_adaptive`: if hard irregular and many blocks are short relative to `max(lengths)`, coarsen around short blocks.

Implementation detail:

- For single-view policies, replace `merged` with that view before Step 3.
- For `best_of_views`, run Step 3 beam on each view with a bounded per-view budget and keep the best UB.
- If `best_of_views` is too invasive, skip it and document why. Single-view comparison is enough for Gate A.

Required CSV diagnostics:

- `block_view_policy`
- `block_view_original_blocks`
- `block_view_final_blocks`
- `block_view_removed_boundaries`
- `block_view_target_b`
- `block_view_price_preserve_used`
- `block_view_arith_adaptive_used`
- `block_view_selected`
- `block_view_eval_count`
- `block_view_best_ub`
- `block_view_time_sec`

## Phase A: K10 gate

Run:

- hardA K10 `{2,3,4,5,7,11,13,17,19,23}`, `n=1000`, seeds `0,1,2,3`
- hardB K10 `{3,5,7,11,13,17,19,23,29,31}`, `n=1000`, seeds `0,1,2,3`

Variants:

- `baseline`
- `coarsen2`
- `coarsen3`
- `target_B12`
- `target_B8`
- `price_preserve_B12`
- `arith_adaptive`

Artifacts:

- `csv/plan29/PLAN29_multiview_block_reconstruction_raw.csv`
- `csv/plan29/PLAN29_multiview_block_reconstruction_compare.csv`
- `csv/plan29/PLAN29_multiview_block_reconstruction_summary.csv`
- `csv/plan29/PLAN29_multiview_block_reconstruction_notes.md`

Gate A passes only if one view has:

- at least `4/8` rows not worse than baseline and improved in UB/gap,
- mean gap improves,
- no memory kills,
- timeout count does not increase,
- runtime increase <= 25%, unless the gap improvement is material and documented.

If Gate A fails, stop and document Decision C.

## Phase B: small controls if Gate A passes

Only if Gate A passes, run:

- hardA K8 seeds `0,1`
- hardB K8 seeds `0,1`
- hardA K12 seed `0`
- hardB K12 seed `1`

Use only:

- `baseline`
- the best Phase A view

Do not run a broad campaign.

## Memory safety

- One heavy solver row at a time.
- Default RSS cap: 16 GB.
- Absolute cap: 20 GB only if explicitly justified.
- Flush each row immediately.
- Do not load full stdout/stderr into memory.
- Store only output tails.
- If memory cap is exceeded, kill the row, mark `memory_killed=1`, and continue.

## Final decision labels

- `A`: one coarsening policy improves K10 and controls; candidate for paper method.
- `B`: K10 improves but controls are mixed; targeted option only.
- `C`: no useful improvement; abandon block-view reconstruction.
- `D`: implementation blocker/inconclusive.
- `E`: improves runtime only, not quality.

## Required docs after results

Update:

- `research/k_vs_arithmetic_axes_20260412/LOG.md`
- `research/k_vs_arithmetic_axes_20260412/iterations/20260428_multiview_block_reconstruction/RESULTS.md`
- `research/k_vs_arithmetic_axes_20260412/iterations/20260428_multiview_block_reconstruction/BLOCKERS.md`
- `research/k_vs_arithmetic_axes_20260412/iterations/20260428_multiview_block_reconstruction/SUMMARY.md`

Update thread-level `RESULTS.md` and `BLOCKERS.md` only if the result changes the public method story.

Return only after Phase A is complete. If Phase A passes, continue through Phase B before returning.

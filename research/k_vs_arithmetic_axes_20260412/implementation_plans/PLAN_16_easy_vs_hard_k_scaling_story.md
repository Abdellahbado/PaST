# PLAN 16: Easy-vs-Hard K-Scaling Story

## Objective

Build a paper-facing fixed-`n=1000` K-axis comparison:

- **easy arithmetic**: contiguous unit ladder `{1,2,...,K}`;
- **hard arithmetic**: irregular non-unit ladders from PLAN17/18.

The intended claim is:

> the same pipeline scales much farther in K when arithmetic is favorable, while
> irregular non-unit arithmetic degrades around K=8-10.

This is a benchmark/story experiment, not a solver redesign.

## Existing Evidence

Read first:

- `csv/plan17/PLAN17_k_axis_n1000_summary_by_k.csv`
- `csv/plan17/PLAN17_k_axis_boundary_classification.csv`
- `csv/plan18/PLAN18_k_boundary_refine_best_of_route.csv`
- `csv/plan18/PLAN18_k_boundary_refine_summary_by_k.csv`
- `RESULTS.md`
- `PAPER_RESULTS_READY.md`
- `CURRENT_RESULTS_INDEX.md`

Current easy-unit evidence from PLAN17:

- `{1,2}` through `{1,...,20}` are exact at fixed `n=1000`, `lambda=1.3`,
  seeds `0,1`.
- Closure is `easy_step2_exact`, so the story is Step-2 driven.

Current hard evidence:

- hard irregular ladders are exact through K=6;
- K=8 is mixed exact/finite-gap;
- K=10 has finite-gap incumbents but no exact closure;
- K=12 is mostly timeout/no-incumbent or finite-gap under reroute;
- K=16/20 hard rows are timeout-dominated.

## Scope

Do not:

- tune Step 3;
- change Step 4;
- run local corridor;
- add new exact methods;
- change accepted defaults.

Do:

- extend easy-unit K scaling;
- preserve the default pipeline;
- run the existing dense-unit Step-2 fastpath where already available;
- record whether closure happens at Step 2;
- compare against already measured hard irregular boundaries.

## Easy Families To Run

Use contiguous unit ladders:

- `easy_k24_unit = {1,2,...,24}`
- `easy_k30_unit = {1,2,...,30}`

Optional only if K=30 is exact and memory/time safe:

- `easy_k40_unit = {1,2,...,40}`

Fixed parameters:

- `n=1000`
- `lambda=1.3`
- seeds `0,1`

Optional robustness:

- seeds `2,3` for K=30 only, if K=30 is exact for seeds `0,1`.

## Pipeline Policy

Use current accepted/default pipeline first.

For K >= 8, also run the existing dense-unit Step-2 fastpath variant if already
available:

- `baseline`
- `dense_step2_fastpath`

Expected result:

- Step 2 closes via `ffd` / dense fastpath;
- Step 3 should not be the main mechanism;
- Step 4 should not be needed.

## Hard Comparison

Do not rerun the full hard campaign unless necessary.

Use existing PLAN17/18 hard rows:

- hardA/hardB K=8;
- hardA/hardB K=10;
- hardA/hardB K=12;
- optionally K=16/20 timeout rows as high-K contrast.

Hard family definitions:

- hardA K=8: `{2,3,4,5,7,11,13,17}`
- hardA K=10: `{2,3,4,5,7,11,13,17,19,23}`
- hardA K=12: `{2,3,4,5,7,11,13,17,19,23,29,31}`
- hardB K=8: `{3,5,7,11,13,17,19,23}`
- hardB K=10: `{3,5,7,11,13,17,19,23,29,31}`
- hardB K=12: `{3,5,7,11,13,17,19,23,29,31,37,41}`

## Required Artifacts

Create:

- `csv/plan30/PLAN30_easy_k_scaling_raw.csv`
- `csv/plan30/PLAN30_easy_k_scaling_summary.csv`
- `csv/plan30/PLAN30_easy_vs_hard_k_boundary.csv`
- `csv/plan30/PLAN30_easy_vs_hard_notes.md`

Raw CSV fields must include:

- family id;
- family sizes;
- K;
- n;
- lambda;
- seed;
- variant label;
- runtime;
- peak RSS;
- timeout/memory kill;
- exact flag;
- UB/LB/gap;
- deciding step;
- `fwd_pack_method`;
- Step-2 reached/produced UB flags if available;
- dense fastpath active flag if available.

## Summary Table Requirements

The final table should support:

> favorable contiguous-unit arithmetic scales much farther in K than irregular
> non-unit arithmetic under the same pipeline.

Recommended columns:

- arithmetic class: `easy_contiguous_unit` or `hard_irregular`;
- family sizes;
- K;
- exact rows / total rows;
- worst boundary class;
- mean runtime;
- deciding step;
- method responsible for closure/incumbent.

## Success Criteria

Minimum:

- K=24 easy-unit exact for seeds `0,1`.

Strong:

- K=30 easy-unit exact for seeds `0,1`.

Very strong:

- K=40 easy-unit exact for seeds `0,1`, or K=30 exact for seeds `0-3`.

If K=30 fails:

- diagnose whether failure is Step-2 runtime, memory, or unexpected downstream
  routing;
- do not redesign the solver unless the fix is harness/routing only.

## Memory Safety

- one heavy process at a time;
- default RSS cap 16 GB;
- absolute max 20 GB only if justified and recorded;
- flush CSV after every row;
- no full stdout/stderr buffering;
- skip K=40 if K=30 is not exact and memory-safe.

## Documentation Updates

Update:

- `LOG.md`
- `RESULTS.md`
- `PAPER_RESULTS_READY.md`
- `CURRENT_RESULTS_INDEX.md`
- `PRESENTATION_RESULTS_SUMMARY.md` if present/current;
- `iterations/20260421_paper_group_recovery/SUMMARY.md`

Do not change accepted paper-group `n` frontiers unless new paper-group rows are
actually run.

## Final Decision Labels

Use exactly one:

- **A**: easy arithmetic exact through K=30 or higher; paper story strengthened.
- **B**: easy arithmetic exact through K=24 but not K=30; boundary documented.
- **C**: easy arithmetic fails before K=24; diagnose pipeline/routing issue.
- **D**: run blocked by harness/memory issue; no conclusion.


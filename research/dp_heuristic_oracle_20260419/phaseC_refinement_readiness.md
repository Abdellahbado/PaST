# Phase C Refinement Readiness

Date: 2026-04-19

## Objective

Decide which post-assignment local-search refinements should continue after a multi-path comparison.

## Status

- **Ready / Passed** for a focused next phase.

## Continue

1. `greedy_dp_local_search_relocate_only` as the new default refinement candidate
   - same TEC as baseline local search on all tested rows
   - materially fewer evaluated moves (swap evaluations removed)
   - lower local-search DP-call counts on broader rows

2. `greedy_dp_local_search_priority_machines` as optional hard-row mode
   - significantly better TEC on `61/350`
   - reduced search effort in tested rows
   - but unstable across rows (regresses on `46/120`)

## Stop (for now)

- `greedy_dp_local_search_screened_swap`
  - no practical advantage over relocate-only in Stage 1
  - additional complexity without measurable return in this pass

## Metric integrity

- Reporting debt resolved by splitting exact DP calls into:
  - `exact_dp_calls_initial`
  - `exact_dp_calls_local_search_only`

## Recommendation

- Make relocate-only the default Phase C refinement baseline.
- Keep priority-machines behind a variant flag for targeted hard instances.
- Do not invest further in screened swaps until evidence suggests relocate-only misses quality.

# Phase C DP Local Search Readiness

Date: 2026-04-19

## Objective

Determine whether bounded DP-evaluated local search after baseline assignment is worth continuing beyond `greedy_dp`.

## Status

- **Ready / Passed** for continuation as an optimization branch.

## Evidence

From `research/dp_heuristic_oracle_20260419/phaseC_dp_local_search_results.md`:

- TEC improved on `46/120`, `61/350`, `64/77`; tie on `90/82`
- no row regressed versus `greedy_dp`
- two rows (`46/120`, `64/77`) reached exact CP-SAT values
- all runs stayed far below 16 GB RSS

## Runtime tradeoff

- relative runtime overhead exists versus `greedy_dp`
- absolute runtimes remain small on the tested subset
- overhead is acceptable for this research phase

## Risks and caveats

- swap neighborhood can be expensive with low acceptance on some rows
- assignment-conditioned LB remains diagnostic only and not a global certificate

## Recommendation

- Continue the branch, targeting move-screening/pruning to cut expensive non-improving swap evaluations while preserving TEC gains.

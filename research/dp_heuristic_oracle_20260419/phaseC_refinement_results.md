# Phase C Refinement Results

Date: 2026-04-19

## Scope

Run a multi-path refinement search inside post-assignment DP local search.

Baseline variants:

- `greedy_dp`
- `greedy_dp_local_search`

Refinement variants implemented:

- `greedy_dp_local_search_relocate_only` (Path A)
- `greedy_dp_local_search_screened_swap` (Path B + narrow paper-inspired equal-size/near-size swap restriction)
- `greedy_dp_local_search_priority_machines` (Path C)

## Small reporting debt fixed

In `solvers/cpp/parallel_heuristic_compare.cpp`, the misleading metric was corrected by splitting:

- `exact_dp_calls_initial`
- `exact_dp_calls_local_search_only`

So local-search-only DP effort is now explicitly separated from initialization.

## Stage 1 (quick screen)

Rows: `46/120`, `61/350`.

### Stage 1 results table

| row | variant | TEC | gap vs exact | runtime_sec | accepted (r/s) | evaluated (r/s) | DP calls (init/LS) | dominant |
|---|---|---:|---:|---:|---|---|---|---|
| 46/120 | greedy_dp | 112 | 0.087379 | 0.008084 | 0 (0/0) | 0/0 | 0/0 | none |
| 46/120 | greedy_dp_local_search | 103 | 0.000000 | 0.099531 | 5 (5/0) | 545/0 | 5/48 | relocate |
| 46/120 | relocate_only | 103 | 0.000000 | 0.097193 | 5 (5/0) | 545/0 | 5/48 | relocate |
| 46/120 | screened_swap | 103 | 0.000000 | 0.097034 | 5 (5/0) | 545/0 | 5/48 | relocate |
| 46/120 | priority_machines | 112 | 0.087379 | 0.021090 | 0 (0/0) | 119/199 | 5/39 | none |
| 61/350 | greedy_dp | 7053 | 0.061879 | 1.164387 | 0 (0/0) | 0/0 | 0/0 | none |
| 61/350 | greedy_dp_local_search | 7046 | 0.060825 | 2.021783 | 5 (5/0) | 107/0 | 12/30 | relocate |
| 61/350 | relocate_only | 7046 | 0.060825 | 2.312738 | 5 (5/0) | 107/0 | 12/30 | relocate |
| 61/350 | screened_swap | 7046 | 0.060825 | 2.338255 | 5 (5/0) | 107/0 | 12/30 | relocate |
| 61/350 | priority_machines | 7015 | 0.056007 | 1.386081 | 5 (5/0) | 39/0 | 12/14 | relocate |

Stage 1 selection decision:

- Keep for Stage 2: `relocate_only`, `priority_machines`.
- Rationale: `priority_machines` gives best TEC on hard row `61/350`; `relocate_only` tests whether swaps are unnecessary while preserving quality.
- `screened_swap` is not selected for Stage 2 because Stage 1 showed no benefit over relocate-only.

## Stage 2 (confirmation)

Rows: `64/77`, `90/82`.
Variants run: `greedy_dp`, `greedy_dp_local_search`, `relocate_only`, `priority_machines`.

### Stage 2 results table

| row | variant | TEC | gap vs exact | runtime_sec | accepted (r/s) | evaluated (r/s) | DP calls (init/LS) | dominant |
|---|---|---:|---:|---:|---|---|---|---|
| 64/77 | greedy_dp | 30598 | 0.000589 | 0.210178 | 0 (0/0) | 0/0 | 0/0 | none |
| 64/77 | greedy_dp_local_search | 30580 | 0.000000 | 0.387491 | 1 (1/0) | 411/12572 | 17/396 | relocate |
| 64/77 | relocate_only | 30580 | 0.000000 | 0.133795 | 1 (1/0) | 411/0 | 17/73 | relocate |
| 64/77 | priority_machines | 30580 | 0.000000 | 0.298707 | 1 (1/0) | 148/7806 | 17/291 | relocate |
| 90/82 | greedy_dp | 53294 | 0.000000 | 0.083737 | 0 (0/0) | 0/0 | 0/0 | none |
| 90/82 | greedy_dp_local_search | 53294 | 0.000000 | 0.493232 | 0 (0/0) | 174/15114 | 22/183 | none |
| 90/82 | relocate_only | 53294 | 0.000000 | 0.260764 | 0 (0/0) | 174/0 | 22/39 | none |
| 90/82 | priority_machines | 53294 | 0.000000 | 0.297565 | 0 (0/0) | 51/6665 | 22/74 | none |

## Improvement deltas

### Versus `greedy_dp`

- `relocate_only`:
  - `46/120`: `-9`
  - `61/350`: `-7`
  - `64/77`: `-18`
  - `90/82`: `0`

- `priority_machines`:
  - `46/120`: `0`
  - `61/350`: `-38`
  - `64/77`: `-18`
  - `90/82`: `0`

### Versus current `greedy_dp_local_search`

- `relocate_only`:
  - TEC: equal on all 4 rows
  - runtime: lower on `46/120`, `64/77`, `90/82`; higher on `61/350`
  - evaluations: drastically fewer swaps (zero)

- `priority_machines`:
  - TEC: worse on `46/120`, better on `61/350`, equal on `64/77`, `90/82`
  - runtime/evaluations: generally reduced vs baseline local search
  - quality variance across rows suggests signal instability

## RSS (max resident set size)

Selected rows from run outputs:

- `46/120` all-variants run: `5,079,040` bytes
- `61/350` all-variants run: `42,844,160` bytes
- `64/77` selected runs: up to `11,321,344` bytes
- `90/82` selected runs: up to `5,619,712` bytes

All remain far below 16 GB.

## Decision against criteria

Criteria recap:

1. same or better TEC than `greedy_dp_local_search` with materially fewer evaluations
2. same or better TEC with materially lower runtime
3. better TEC on `61/350`

Assessment:

- `relocate_only`: **continue**
  - meets (1) and often (2); same TEC as baseline local search on all tested rows with massive swap-evaluation reduction.

- `priority_machines`: **conditional continue (as optional branch)**
  - meets (3) strongly on `61/350` (`7046 -> 7015`), but regresses on `46/120`.
  - useful as targeted hard-row mode, not as default replacement.

- `screened_swap`: **stop for now**
  - no observed gain over `relocate_only` in Stage 1; extra complexity without return.

## Best variant after this pass

- Default best: `greedy_dp_local_search_relocate_only` (robust quality parity with much lower search effort).
- Best on hard row `61/350`: `greedy_dp_local_search_priority_machines`.

# Phase B DP-Guided Assignment Results

Date: 2026-04-19

## Scope

Phase B tests whether assignment guidance from fast DP-based lower-bound scores improves quality beyond simple greedy assignment, with exact single-machine DP used as optimizer in both compared variants.

Compared variants:

- `greedy_dp`
- `dp_guided_assignment_dp`

Both are fixed-`epsilon` and heuristic-only.

## Implementation

Updated:

- `solvers/cpp/parallel_heuristic_compare.cpp`

Added Phase B variant:

- `dp_guided_assignment_dp`

Assignment logic:

1. process jobs in non-increasing length order
2. for each feasible target machine, score post-insertion machine assignment using per-machine DP-based lower-bound estimate
3. choose machine with best score (tie-break with immediate insertion cost)
4. after assignment, optimize each machine with exact single-machine DP

## Commands

```bash
/usr/bin/time -l "/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/build/parallel_heuristic_compare" paper-instance 46 120 all "/Users/mac/Documents/Study/PFE/PaST/temp/paper_exact_repo/instances" 30
/usr/bin/time -l "/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/build/parallel_heuristic_compare" paper-instance 61 350 all "/Users/mac/Documents/Study/PFE/PaST/temp/paper_exact_repo/instances" 30
/usr/bin/time -l "/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/build/parallel_heuristic_compare" paper-instance 64 77 all "/Users/mac/Documents/Study/PFE/PaST/temp/paper_exact_repo/instances" 30
/usr/bin/time -l "/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/build/parallel_heuristic_compare" paper-instance 90 82 all "/Users/mac/Documents/Study/PFE/PaST/temp/paper_exact_repo/instances" 30
```

## Runtime / RSS

Process-level max RSS from runs stayed well below 16 GB:

- highest observed in this batch: `37,273,600` bytes (`61/350` run)

## Results

| instance / epsilon | `greedy_dp` TEC | `dp_guided_assignment_dp` TEC | delta |
|---|---:|---:|---:|
| `46 / 120` | 112 | 154 | +42 |
| `61 / 350` | 7053 | 10402 | +3349 |
| `64 / 77` | 30598 | 30580 | -18 |
| `90 / 82` | 53294 | 53294 | 0 |

Interpretation:

- one row improves (`64/77`)
- one row ties (`90/82`)
- two rows degrade substantially (`46/120`, `61/350`)

So there is no clear additional gain from this Phase B DP-guided assignment strategy.

## Exact-gap view

Using Phase A exact references (`TEC_exact`):

- `46/120` exact `103`: gap worsens from `0.087379` to `0.495146`
- `61/350` exact `6642`: gap worsens from `0.061879` to `0.566697`
- `64/77` exact `30580`: gap improves from `0.000589` to `0.000000`
- `90/82` exact `53294`: unchanged at `0.000000`

## Decision

Phase B decision rule is **not met**:

- DP-guided assignment does not provide clear additional quality gain over `greedy_dp` on this subset.

Branch conclusion at Phase B:

- DP is useful as machine optimizer (Phase A positive)
- current DP-guided assignment scoring is not useful enough on this subset

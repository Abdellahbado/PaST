# Phase C DP Local Search Results

Date: 2026-04-19

## Scope

Evaluate whether bounded local search after baseline assignment improves over `greedy_dp`.

New variant:

- `greedy_dp_local_search`

Mechanism:

- start from the same baseline assignment used by `greedy_dp`
- explore relocate and swap moves
- evaluate touched machines with exact single-machine DP
- accept only strict TEC improvements
- bounded by `ls_time_cap_sec=10`, `ls_max_rounds=5`, `ls_max_moves_per_round=20000`

## Build

```bash
cmake --build "/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/build" --target parallel_heuristic_compare
```

## Commands run

```bash
/usr/bin/time -l "/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/build/parallel_heuristic_compare" paper-instance 46 120 all "/Users/mac/Documents/Study/PFE/PaST/temp/paper_exact_repo/instances" 30 10 5 20000
/usr/bin/time -l "/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/build/parallel_heuristic_compare" paper-instance 61 350 all "/Users/mac/Documents/Study/PFE/PaST/temp/paper_exact_repo/instances" 30 10 5 20000
/usr/bin/time -l "/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/build/parallel_heuristic_compare" paper-instance 64 77 all "/Users/mac/Documents/Study/PFE/PaST/temp/paper_exact_repo/instances" 30 10 5 20000
/usr/bin/time -l "/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/build/parallel_heuristic_compare" paper-instance 90 82 all "/Users/mac/Documents/Study/PFE/PaST/temp/paper_exact_repo/instances" 30 10 5 20000
```

## TEC and exact-gap impact

Exact references:

- `46/120 = 103`
- `61/350 = 6642`
- `64/77 = 30580`
- `90/82 = 53294`

| instance / epsilon | `greedy_dp` TEC | `greedy_dp_local_search` TEC | change | exact-gap (`greedy_dp`) | exact-gap (`local_search`) |
|---|---:|---:|---:|---:|---:|
| `46 / 120` | 112 | 103 | -9 | 0.087379 | 0.000000 |
| `61 / 350` | 7053 | 7046 | -7 | 0.061879 | 0.060825 |
| `64 / 77` | 30598 | 30580 | -18 | 0.000589 | 0.000000 |
| `90 / 82` | 53294 | 53294 | 0 | 0.000000 | 0.000000 |

`greedy_dp_local_search` improves or ties `greedy_dp` on all required rows.

## Local-search activity metrics

| instance / epsilon | accepted moves | relocate accepted | swap accepted | relocate evaluated | swap evaluated | dominant move | exact DP calls (LS) |
|---|---:|---:|---:|---:|---:|---|---:|
| `46 / 120` | 5 | 5 | 0 | 545 | 0 | relocate | 53 |
| `61 / 350` | 5 | 5 | 0 | 107 | 0 | relocate | 42 |
| `64 / 77` | 1 | 1 | 0 | 411 | 12572 | relocate | 413 |
| `90 / 82` | 0 | 0 | 0 | 174 | 15114 | none | 205 |

Accepted improvements are relocate-dominated. Swap exploration can be expensive without accepted gains on some rows.

Metric note update:

- the earlier `exact_dp_calls_local_search` field mixed initialization and local-search calls.
- this is now corrected in code/reporting by splitting into:
  - `exact_dp_calls_initial`
  - `exact_dp_calls_local_search_only`
- refinement-phase reports use the split metrics.

## Runtime cost

Driver runtime (`runtime_sec` in CSV):

- `46/120`: `0.008893 -> 0.168997` (~19.0x)
- `61/350`: `1.323677 -> 2.342739` (~1.8x)
- `64/77`: `0.033357 -> 0.395049` (~11.8x)
- `90/82`: `0.048947 -> 0.181747` (~3.7x)

Absolute times remain low on the tested subset, but relative overhead is visible.

## Memory

Process max RSS from `/usr/bin/time -l`:

- `46/120`: `4,931,584` bytes
- `61/350`: `38,780,928` bytes
- `64/77`: `11,304,960` bytes
- `90/82`: `9,125,888` bytes

All well below the 16 GB cap.

## Decision

- Phase C is positive: local search provides non-negative quality impact on all rows, with strict improvements on 3/4 rows.
- Branch is worth continuing, with first optimization priority on pruning/screening swap move evaluations.

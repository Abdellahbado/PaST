# Phase A ESR Replacement Results (Corrected)

Date: 2026-04-19

## Scope

This report re-runs Phase A after fixing the lower-bound bug in:

- `solvers/cpp/parallel_heuristic_compare.cpp`

Compared variants (same assignment baseline):

- `greedy_esr`: LPT-style greedy assignment + paper-style sequence-preserving ESR per machine
- `greedy_dp`: LPT-style greedy assignment + exact single-machine DP per machine

## What was wrong before

The previous `relaxed_machine_lb(...)` compressed a machine assignment to:

- unique lengths
- total work

and ignored multiplicities. That can produce values that are not valid lower bounds for the exact same assigned multiset.

So the prior LB diagnostic is not trusted anymore.

## Code correction

Fixes made in `solvers/cpp/parallel_heuristic_compare.cpp`:

1. `relaxed_machine_lb(...)` now uses the actual assigned multiset counts (`lengths` + `totals`) and calls:
   - `dp::solve_relaxed_dp_lb_feas(...)` first
   - fallback to `dp::solve_relaxed_dp_lb(...)` if needed
2. Added a hard safety check: if computed LB exceeds a known feasible machine cost, do not keep it.
3. Added safe fallback bound:
   - `fallback_slot_lb = rate * sum(cheapest total_work slots)`
   - this is always a valid lower bound for the same assigned multiset.
4. Driver now reports `machine_lb_source` so fallback usage is explicit.

## Files changed

- `solvers/cpp/parallel_heuristic_compare.cpp`
- `solvers/cpp/CMakeLists.txt`

## Benchmark root used

- `/Users/mac/Documents/Study/PFE/PaST/temp/paper_exact_repo/instances`

## Exact commands used

Build:

```bash
cmake -S "/Users/mac/Documents/Study/PFE/PaST/solvers/cpp" -B "/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/build" && cmake --build "/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/build" --target parallel_heuristic_compare
```

Corrected Phase A rerun:

```bash
/usr/bin/time -l "/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/build/parallel_heuristic_compare" paper-instance 46 120 all "/Users/mac/Documents/Study/PFE/PaST/temp/paper_exact_repo/instances" 30
/usr/bin/time -l "/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/build/parallel_heuristic_compare" paper-instance 61 350 all "/Users/mac/Documents/Study/PFE/PaST/temp/paper_exact_repo/instances" 30
/usr/bin/time -l "/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/build/parallel_heuristic_compare" paper-instance 64 77 all "/Users/mac/Documents/Study/PFE/PaST/temp/paper_exact_repo/instances" 30
/usr/bin/time -l "/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/build/parallel_heuristic_compare" paper-instance 90 82 all "/Users/mac/Documents/Study/PFE/PaST/temp/paper_exact_repo/instances" 30
```

Exact references (same CP-SAT branch):

```bash
/usr/bin/time -l python3 "/Users/mac/Documents/Study/PFE/PaST/solvers/parallel_f2_cp_sat.py" --instance 46 --epsilon 120 --data-dir "/Users/mac/Documents/Study/PFE/PaST/temp/paper_exact_repo/instances" --time-limit 180 --workers 8
/usr/bin/time -l python3 "/Users/mac/Documents/Study/PFE/PaST/solvers/parallel_f2_cp_sat.py" --instance 61 --epsilon 350 --data-dir "/Users/mac/Documents/Study/PFE/PaST/temp/paper_exact_repo/instances" --time-limit 240 --workers 8
/usr/bin/time -l python3 "/Users/mac/Documents/Study/PFE/PaST/solvers/parallel_f2_cp_sat.py" --instance 64 --epsilon 77 --data-dir "/Users/mac/Documents/Study/PFE/PaST/temp/paper_exact_repo/instances" --time-limit 180 --workers 8
/usr/bin/time -l python3 "/Users/mac/Documents/Study/PFE/PaST/solvers/parallel_f2_cp_sat.py" --instance 90 --epsilon 82 --data-dir "/Users/mac/Documents/Study/PFE/PaST/temp/paper_exact_repo/instances" --time-limit 180 --workers 8
```

## Runtime and RSS

Heuristic process-level RSS from corrected reruns:

- `46/120`: `3,620,864` bytes
- `61/350`: `37,273,600` bytes
- `64/77`: `10,289,152` bytes
- `90/82`: `8,404,992` bytes

All far below the 16 GB cap.

## TEC comparison: `greedy_esr` vs `greedy_dp`

| instance / epsilon | TEC `greedy_esr` | TEC `greedy_dp` | DP improvement |
|---|---:|---:|---:|
| `46 / 120` | 118 | 112 | -6 |
| `61 / 350` | 7263 | 7053 | -210 |
| `64 / 77` | 30640 | 30598 | -42 |
| `90 / 82` | 53300 | 53294 | -6 |

`greedy_dp` still improves TEC on all tested rows.

## Exact-gap comparison against CP-SAT optimum

Exact references:

- `46/120`: `TEC_exact = 103`
- `61/350`: `TEC_exact = 6642`
- `64/77`: `TEC_exact = 30580`
- `90/82`: `TEC_exact = 53294`

Exact relative gaps `(TEC_variant - TEC_exact) / TEC_exact`:

| instance / epsilon | gap `greedy_esr` | gap `greedy_dp` |
|---|---:|---:|
| `46 / 120` | 0.145631 | 0.087379 |
| `61 / 350` | 0.093496 | 0.061879 |
| `64 / 77` | 0.001962 | 0.000589 |
| `90 / 82` | 0.000113 | 0.000000 |

`greedy_dp` still improves exact-gap on all rows.

## Assignment-conditioned LB status (corrected wording)

### What is safe to claim

- The previous multiplicity-collapse bug was removed.
- For `greedy_dp` schedules, the reported machine-level LB values are guarded by fallback and LB-vs-feasible sanity checks.

### What is **not** claimed

- We do **not** claim assignment-conditioned LB for `greedy_esr` as a validated safe bound for decision making.
- `greedy_esr` LB numbers below are diagnostic only and not used as a readiness gate.
- On harder rows, several machines use fallback LB (`machine_lb_source`), so tightness is mixed.

Schedule-level assignment-conditioned LB diagnostics:

| instance / epsilon | LB (`greedy_esr`) | LB (`greedy_dp`) |
|---|---:|---:|
| `46 / 120` | 107 | 107 |
| `61 / 350` | 6781 | 6781 |
| `64 / 77` | 30604 | 30598 |
| `90 / 82` | 53294 | 53294 |

These are assignment-conditioned diagnostics only, not global bounds.

## Decision outcome for Phase A (after correction)

- Phase A lower-bound overclaim is corrected in reporting: `greedy_esr` LB is treated as diagnostic only.
- Main Phase A conclusion is unchanged: `greedy_dp` clearly improves TEC over `greedy_esr` on the tested subset.
- Phase A still passes and supports continuation to Phase B.

## Phase B initial run (same corrected driver)

Added Phase B variant:

- `dp_guided_assignment_dp`: DP-guided assignment scoring + exact single-machine DP optimizer

Observed TEC vs `greedy_dp`:

| instance / epsilon | `greedy_dp` | `dp_guided_assignment_dp` | change |
|---|---:|---:|---:|
| `46 / 120` | 112 | 154 | +42 (worse) |
| `61 / 350` | 7053 | 10402 | +3349 (worse) |
| `64 / 77` | 30598 | 30580 | -18 (better) |
| `90 / 82` | 53294 | 53294 | 0 |

Phase B does not show clear additional improvement over `greedy_dp`.

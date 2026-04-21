# Results

## Refinement search update (same iteration)

Added multi-path local-search refinements and compared them under a staged protocol.

New variants in `solvers/cpp/parallel_heuristic_compare.cpp`:

- `greedy_dp_local_search_relocate_only`
- `greedy_dp_local_search_screened_swap`
- `greedy_dp_local_search_priority_machines`

Also fixed metric honesty:

- replaced ambiguous `exact_dp_calls_local_search` with:
  - `exact_dp_calls_initial`
  - `exact_dp_calls_local_search_only`

Stage 1 (`46/120`, `61/350`) and Stage 2 (`64/77`, `90/82`) results are reported in:

- `research/dp_heuristic_oracle_20260419/phaseC_refinement_results.md`

Key outcomes:

- `relocate_only` matches baseline local-search TEC on all tested rows while removing swap-search overhead.
- `priority_machines` gives the best `61/350` TEC (`7015`) but regresses on `46/120`.
- `screened_swap` showed no gain over relocate-only in Stage 1.

Decision from refinement pass:

- continue `relocate_only` as default candidate
- keep `priority_machines` as optional hard-row mode
- stop screened-swap path for now

## Variant added

- `greedy_dp_local_search` in `solvers/cpp/parallel_heuristic_compare.cpp`
- neighborhood: one-job relocate + one-for-one swap
- acceptance: strict TEC improvement only
- machine-cost evaluation: exact single-machine DP on touched machines with cache

## Fixed-`epsilon` rows

- `46/120`, `61/350`, `64/77`, `90/82`

## TEC comparison vs `greedy_dp`

- `46/120`: `112 -> 103` (improves by `-9`)
- `61/350`: `7053 -> 7046` (improves by `-7`)
- `64/77`: `30598 -> 30580` (improves by `-18`)
- `90/82`: `53294 -> 53294` (no change)

`greedy_dp_local_search` is never worse on tested rows; improves on 3/4 rows.

## Exact reference context (CP-SAT)

- `46/120` exact = `103`  -> local search reaches exact
- `61/350` exact = `6642` -> still above exact
- `64/77` exact = `30580` -> local search reaches exact
- `90/82` exact = `53294` -> already exact before local search

## Local-search activity

- `46/120`: accepted `5` (relocate `5`, swap `0`), evaluated relocate `545`, swap `0`, exact DP calls `53`
- `61/350`: accepted `5` (relocate `5`, swap `0`), evaluated relocate `107`, swap `0`, exact DP calls `42`
- `64/77`: accepted `1` (relocate `1`, swap `0`), evaluated relocate `411`, swap `12572`, exact DP calls `413`
- `90/82`: accepted `0`, evaluated relocate `174`, swap `15114`, exact DP calls `205`

Dominant accepted move type is relocate on improving rows.

## Runtime overhead (driver runtime_sec)

- `46/120`: `0.008893` (`greedy_dp`) -> `0.168997` (local search)
- `61/350`: `1.323677` -> `2.342739`
- `64/77`: `0.033357` -> `0.395049`
- `90/82`: `0.048947` -> `0.181747`

Overhead is moderate-to-large in relative terms, but absolute runtime remains small on tested rows.

## Memory (process max RSS)

- `46/120`: `4,931,584` bytes
- `61/350`: `38,780,928` bytes
- `64/77`: `11,304,960` bytes
- `90/82`: `9,125,888` bytes

All runs remain far below 16 GB.

## Decision

Phase C signal is positive enough to continue as a bounded heuristic branch:

- quality: improves or ties `greedy_dp` on all tested rows
- practicality: memory safe; runtime overhead acceptable for current scope

Main caution:

- many evaluated swaps did not produce accepted improvements; further screening/pruning is the first optimization target.

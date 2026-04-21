# Phase J Insert Screening Redesign Results

Date: 2026-04-20

## Scope

- single hard point only: `instance 61`, `epsilon 347`
- objective: recover no-screen-discovered improving `insert_inter` behavior with analytical bounded screening/ranking
- no ML, no ALNS, no frontier

## Implemented redesigns

In `solvers/cpp/parallel_heuristic_compare.cpp`:

- `vnd_exact_dp_insert_rank_v1`
  - source-target dual-pressure ranking (Idea A)
  - gap-aware source priority (Idea D)
- `vnd_exact_dp_insert_rank_diverse`
  - per-source preserved candidate diversity (Idea B)
  - two-stage screening/rerank (Idea C)
  - gap-aware source priority (Idea D)

Both variants:

- use bounded deterministic multistart starting family (same as Phase H)
- focus on `insert_inter` only in this pass
- use exact DP acceptance on touched machines only

## Required comparisons at `61/347`

- `greedy_dp`: `7088`
- `greedy_dp_local_search_relocate_only`: `7081`
- previous `vnd_exact_dp` baseline: `6944`
- Phase I no-screen best: `6920`
- `vnd_exact_dp_insert_rank_v1`: `6908`
- `vnd_exact_dp_insert_rank_diverse`: `6884`

## Metrics by redesigned variant

### `vnd_exact_dp_insert_rank_v1`

- final TEC: `6908`
- improvement vs `6944`: `-36`
- gap to no-screen `6920`: `-12` (better)
- gap to paper EHS `6710`: `+198`
- gap to reference `6643`: `+265`
- runtime: `25.36 s`
- max RSS: `246,300,672` bytes
- exact local-search evaluations (misses after initial): `22`
- accepted insert moves: `4`
- evaluated insert candidates: `29160`
- stop reason: `no_improving_move`

### `vnd_exact_dp_insert_rank_diverse`

- final TEC: `6884`
- improvement vs `6944`: `-60`
- gap to no-screen `6920`: `-36` (better)
- gap to paper EHS `6710`: `+174`
- gap to reference `6643`: `+241`
- runtime: `68.42 s`
- max RSS: `1,055,866,880` bytes
- exact local-search evaluations (misses after initial): `17`
- accepted insert moves: `4`
- evaluated insert candidates: `29160`
- stop reason: `no_improving_move`

## Interpretation

- Redesign clearly recovered Phase I improving move pattern (accepted `insert_inter` moves appear and improve TEC).
- Diversity + two-stage rerank improved quality further than rank_v1.
- Main bottleneck shifted from ranking quality to **screening efficiency/memory overhead** in the diverse variant.

## Branch decision

- Continue this branch, but narrowly: keep insert-ranking redesign and optimize candidate-pool efficiency (cap and structure), without expanding scope beyond `61/347` yet.

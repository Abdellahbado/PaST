# Phase H VND-Exact-Oracle Results

Date: 2026-04-20

## Scope executed

- new method family probe only (Phase H)
- fixed epsilon only (`61/347`)
- no frontier, no epsilon oscillation, no broad benchmark

## Prototype implemented

- variant: `vnd_exact_dp`
- implementation file: `solvers/cpp/parallel_heuristic_compare.cpp`
- neighborhoods: `swap_intra`, `swap_inter`, `insert_inter` (VND order)
- acceptance: epsilon-feasible + strict exact-DP TEC improvement on touched machine(s)

## Small bounded enhancement beyond minimum

- deterministic bounded multistart initialization (4 starts, randomized LPT with fixed seeds, RCL size 3)
- fallback to deterministic greedy LPT if multistart yields no feasible candidate

## Main metrics (`61/347`)

- final TEC: `6944`
- gap to paper EHS (`6710`): `+234`
- gap to reference (`6643`): `+301`
- runtime (`/usr/bin/time -l` command wall): `32.04 s`
- algorithm runtime field: `31.328390 s`
- max RSS (`/usr/bin/time -l`): `259,375,104` bytes
- accepted moves: `0`
  - `swap_intra`: `0`
  - `swap_inter`: `0`
  - `insert_inter`: `0`
- evaluated moves:
  - `swap_intra`: `3742`
  - `swap_inter`: `25000`
  - `insert_inter`: `6000`
- exact DP calls:
  - initial: `12`
  - local-search phase: `6`
- exact-DP cache:
  - hits: `127`
  - misses: `18`
  - hit rate: `87.59%`
- stop reason: `no_improving_move`

## Required same-epsilon comparisons

- `greedy_dp`: `7088`
- `greedy_dp_local_search_relocate_only`: `7053`
- `vnd_exact_dp`: `6944`
- paper EHS: `6710`
- reference exact/F2-init: `6643`

Derived deltas:

- `vnd_exact_dp` vs `greedy_dp`: `-144`
- `vnd_exact_dp` vs relocate-only baseline: `-109`

## Interpretation

- The Phase H point is materially better than current heuristic baselines at same epsilon.
- However, move-level evidence is weak in this run: no neighborhood move accepted, so gain appears to come from stronger bounded initialization rather than successful VND transitions.
- This is positive quality signal but incomplete mechanism signal.

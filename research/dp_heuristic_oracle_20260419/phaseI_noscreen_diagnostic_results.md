# Phase I No-Screen Diagnostic Results

Date: 2026-04-20

## Scope

- primary target only: `instance 61`, `epsilon 347`
- diagnostic objective: separate "aggressive screening" vs "true 1-move local optimum"
- no frontier and no multi-epsilon sweep

## Implementation summary

- Added variant `phaseI_noscreen_diagnostic` in `solvers/cpp/parallel_heuristic_compare.cpp`.
- Start state: best bounded multistart incumbent (same deterministic multistart family used in Phase H).
- Diagnostic engine:
  - exact evaluation of feasible `insert_inter` moves in tested batch, with no pre-exact shortlist pruning
  - optional `swap_inter` exact evaluation when budget remains
  - bounded by exact-evaluation caps and time cap
- Extra bounded enhancement used: source-machine prioritization by highest exact TEC and exact-eval cap ladder (`64/256/1024`).

## Main metrics (`61/347`)

From the cap ladder runs:

- cap `64`
  - starting TEC: `6944`
  - best TEC found: `6922`
  - improving 1-move found: **yes**
  - exact-evaluated moves: `64`
    - `insert_inter`: `48`
    - `swap_inter`: `16`
  - accepted improving moves: `1` (`insert_inter`)
  - runtime: `38.25 s` wall (`37.275882 s` algorithm field)
  - max RSS: `239,501,312` bytes
  - stop reason: `diag_found_improving_move`

- cap `256`
  - starting TEC: `6944`
  - best TEC found: `6920`
  - improving 1-move found: **yes**
  - exact-evaluated moves: `149`
    - `insert_inter`: `149`
    - `swap_inter`: `0`
  - accepted improving moves: `1` (`insert_inter`)
  - runtime: `57.02 s` wall (`57.012513 s` algorithm field)
  - max RSS: `251,248,640` bytes
  - stop reason: `diag_found_improving_move`

- cap `1024`
  - starting TEC: `6944`
  - best TEC found: `6920`
  - improving 1-move found: **yes**
  - exact-evaluated moves: `359`
    - `insert_inter`: `359`
    - `swap_inter`: `0`
  - accepted improving moves: `1` (`insert_inter`)
  - runtime: `61.06 s` wall (`61.053379 s` algorithm field)
  - max RSS: `254,705,664` bytes
  - stop reason: `diag_found_improving_move`

## Diagnostic conclusion

Evidence supports:

- **screening too aggressive** (improving `insert_inter` 1-moves appear under no-screen exact evaluation from same start).

Evidence does not support:

- **true 1-move local optimum** under current tested neighborhoods.

## Recommended next branch

- Keep single-point scope (`61/347`) and redesign screening/ranking so that shortlist preserves move patterns that passed no-screen exact diagnostic, while keeping bounded exact-evaluation cost.
- Defer larger-neighborhood or ALNS branch until this screening redesign is validated.

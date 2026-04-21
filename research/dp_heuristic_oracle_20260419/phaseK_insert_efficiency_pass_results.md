# Phase K Insert Efficiency Pass Results

Date: 2026-04-20

## Scope

- single point only: `instance 61`, `epsilon 347`
- one bounded non-ML pass on Phase J insert-screening branch

## Compared variants

- `greedy_dp`
- `greedy_dp_local_search_relocate_only`
- Phase H `vnd_exact_dp`
- Phase J best `vnd_exact_dp_insert_rank_diverse`
- Phase K:
  - `vnd_exact_dp_insert_rank_diverse_trimmed`
  - `vnd_exact_dp_insert_rank_diverse_budgeted`

## Baseline references

- old screened baseline: `6944`
- no-screen best: `6920`
- Phase J best: `6884`
- paper EHS: `6710`
- reference: `6643`

## Results table (new Phase K variants)

### `vnd_exact_dp_insert_rank_diverse_trimmed`

- final TEC: `6884`
- improvement vs `6884`: `0` (tie)
- gap to EHS `6710`: `+174`
- gap to reference `6643`: `+241`
- runtime: `75.01 s` (CSV runtime field)
- max RSS: `992,165,888` bytes
- accepted insert moves: `4`
- evaluated insert candidates: `19,416`
- exact local-search evaluations: `15`
- stop reason: `no_improving_move`

### `vnd_exact_dp_insert_rank_diverse_budgeted`

- final TEC: `6884`
- improvement vs `6884`: `0` (tie)
- gap to EHS `6710`: `+174`
- gap to reference `6643`: `+241`
- runtime: `73.97 s` (CSV runtime field)
- max RSS: `965,541,888` bytes
- accepted insert moves: `4`
- evaluated insert candidates: `22,440`
- exact local-search evaluations: `15`
- stop reason: `no_improving_move`

## Efficiency comparison vs Phase J best (`vnd_exact_dp_insert_rank_diverse`)

Phase J diverse (rerun in this pass):

- TEC `6884`
- runtime `73.87 s`
- max RSS `971,849,728` bytes
- evaluated insert candidates `29,160`
- exact local-search evals `17`

Relative changes:

- trimmed:
  - screened insert candidates: `29160 -> 19416` (about `-33.4%`)
  - exact LS evals: `17 -> 15`
  - runtime: slightly worse (`73.87 -> 75.01 s`)
  - RSS: slightly worse (`971,849,728 -> 992,165,888`)

- budgeted:
  - screened insert candidates: `29160 -> 22440` (about `-23.0%`)
  - exact LS evals: `17 -> 15`
  - runtime: near tie/slightly worse (`73.87 -> 73.97 s`)
  - RSS: slightly better (`971,849,728 -> 965,541,888`), not material

## Interpretation

- Quality: preserved (both tie `6884`), not improved.
- Efficiency: candidate-screening work reduced materially, but runtime/RSS gains are not material.
- Branch signal: insert-focused analytical structure is validated; additional handcrafted passes likely have diminishing return on this point.

## Decision

- Best hand-crafted base remains the diverse insert-focused family at TEC `6884`.
- Prefer `vnd_exact_dp_insert_rank_diverse_trimmed` as a cleaner feature/base design for learning handoff (strongest screening reduction while keeping quality).
- Recommend pivot to learning-based move screening/ranking, keeping exact-DP touched-machine acceptance unchanged.

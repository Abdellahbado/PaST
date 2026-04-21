# Results

Runs executed at `instance 61`, `epsilon 347`.

Baselines:

- `greedy_dp`: `7088`
- `greedy_dp_local_search_relocate_only`: `7081`
- Phase H `vnd_exact_dp`: `6944`
- Phase J best `vnd_exact_dp_insert_rank_diverse`: `6884`

Phase K variants:

- `vnd_exact_dp_insert_rank_diverse_trimmed`
  - TEC: `6884` (tie vs Phase J best)
  - accepted insert moves: `4`
  - screened insert candidates: `19416` (vs `29160`, about `-33%`)
  - exact LS evals: `15` (vs `17`)
  - runtime: `75.01 s`
  - max RSS: `992,165,888` bytes
  - stop reason: `no_improving_move`

- `vnd_exact_dp_insert_rank_diverse_budgeted`
  - TEC: `6884` (tie vs Phase J best)
  - accepted insert moves: `4`
  - screened insert candidates: `22440` (vs `29160`, about `-23%`)
  - exact LS evals: `15` (vs `17`)
  - runtime: `73.97 s`
  - max RSS: `965,541,888` bytes
  - stop reason: `no_improving_move`

Interpretation:

- Quality was preserved but not improved (`6884` retained).
- Candidate screening work was materially reduced.
- Wall/runtime and RSS did not improve materially in this environment.
- Remaining non-ML headroom appears limited; this suggests pivot to learned screening/ranking while keeping this insert-focused exact-DP acceptance core.

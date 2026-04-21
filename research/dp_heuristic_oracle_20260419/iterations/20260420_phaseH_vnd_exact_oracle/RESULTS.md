# Results

Main run executed:

- instance `61`, epsilon `347`, variant `vnd_exact_dp`

Key outcome:

- TEC `6944`
- improvement vs `greedy_dp` (`7088`): `-144`
- improvement vs `greedy_dp_local_search_relocate_only` (`7053`): `-109`
- gap to paper EHS (`6710`): `+234`
- gap to reference (`6643`): `+301`

Search behavior summary:

- accepted moves: `0` (`swap_intra=0`, `swap_inter=0`, `insert_inter=0`)
- evaluated moves:
  - `swap_intra=3742`
  - `swap_inter=25000`
  - `insert_inter=6000`
- exact DP calls: initial `12`, local-search phase `6`
- cache hits/misses: `127 / 18`
- stop reason: `no_improving_move`

Interpretation:

- quality gain mainly comes from stronger initialization (bounded multistart), not accepted neighborhood moves in this run.
- despite no accepted moves, resulting TEC is materially better than current relocate-only baseline at same epsilon.

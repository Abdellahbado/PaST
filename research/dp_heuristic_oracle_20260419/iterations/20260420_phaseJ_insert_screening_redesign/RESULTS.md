# Results

Main runs executed at `61/347`.

Reference points:

- `greedy_dp`: `7088`
- `greedy_dp_local_search_relocate_only`: `7081`
- old `vnd_exact_dp`: `6944`
- Phase I no-screen best: `6920`

Phase J redesigned variants:

- `vnd_exact_dp_insert_rank_v1`: `6908`
  - accepted insert moves: `4`
  - evaluated insert candidates: `29160`
  - exact local-search calls: `22`
  - runtime: `25.36 s`, max RSS: `246,300,672` bytes

- `vnd_exact_dp_insert_rank_diverse`: `6884`
  - accepted insert moves: `4`
  - evaluated insert candidates: `29160`
  - exact local-search calls: `17`
  - runtime: `68.42 s`, max RSS: `1,055,866,880` bytes

Interpretation:

- Both redesigned insert-focused variants beat `6944` and recover (and surpass) the no-screen `6920` quality signal.
- Diversity + two-stage rerank improved quality further, but at high memory/runtime cost due to larger candidate pools.

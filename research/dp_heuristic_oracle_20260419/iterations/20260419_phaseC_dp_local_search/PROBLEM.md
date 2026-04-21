# Problem

Phase C question:

- after building the baseline LPT-style assignment, can a bounded DP-evaluated local search (`relocate` + `swap`) improve TEC over `greedy_dp`?

Scope constraints:

- keep assignment construction simple (same baseline as Phase A)
- use exact single-machine DP only for touched machines during move evaluation
- accept only strict TEC-improving moves
- enforce deterministic, bounded search (`ls_time_cap_sec`, `ls_max_rounds`, `ls_max_moves_per_round`)
- keep heavy runs one at a time and below 16 GB RSS

Evaluation rows:

- `46/120`, `61/350`, `64/77`, `90/82`

Decision criterion:

- continue this branch only if `greedy_dp_local_search` shows consistent net TEC gains vs `greedy_dp` at acceptable runtime overhead on the tested rows.

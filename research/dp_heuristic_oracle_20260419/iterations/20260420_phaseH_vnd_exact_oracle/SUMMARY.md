# Summary

Phase H starts a new method family: VND-inspired fixed-epsilon heuristic with exact DP as the machine oracle.

Scope executed:

- one bounded run on `61/347`
- no EOA oscillation, no frontier expansion, no multi-instance sweep.

Implemented prototype:

- `vnd_exact_dp` in `solvers/cpp/parallel_heuristic_compare.cpp`
- neighborhoods in VND order: `swap_intra`, `swap_inter`, `insert_inter`
- exact-DP-on-touched-machines acceptance with caching and bounded screening/evaluation.

Result signal:

- TEC `6944` at `61/347`
- materially better than current heuristic baselines at same epsilon (`7088`, `7053`)
- still behind paper/reference (`6710`, `6643`)

Behavioral caveat:

- no neighborhood move accepted in this run; gain came from bounded multistart initialization.

Decision posture:

- continuation is conditionally justified because quality improved materially over current branch baseline.
- next step must explicitly target move-level acceptance signal under same single-point scope.

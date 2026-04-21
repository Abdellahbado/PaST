# Summary

Phase F introduced a new method family probe: configuration decomposition with DP pricing.

What was implemented:

- design note with explicit formulation and CG caution:
  - `research/dp_heuristic_oracle_20260419/phaseF_configuration_probe_design.md`
- C++ enumerator/pricer:
  - `solvers/cpp/phaseF_config_probe.cpp`
- Python master IP solver:
  - `solvers/phaseF_config_master_probe.py`

Main required result:

- `46/77` master solved (`OPTIMAL`) with TEC `103`, matching exact reference and paper EHS at same epsilon.

Measured `46/77` costs:

- configurations: `4536` (per class), priced pairs: `13608`
- pricing runtime: `18.124586 s`
- master runtime: `8.657083 s`
- total runtime: `26.782123 s`

Optional follow-up checks (same instance only):

- `46/73` and `46/120` also solved with TEC `103`.

Decision signal:

- regime-1 formulation is viable and exact on the tested row.
- keep CG as a separate next step; do not assume `solve_pricing_dp` is drop-in branch-and-price ready yet.

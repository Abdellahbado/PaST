# Problem

Date: 2026-04-20

Phase H at `61/347` reached `6944` but had no accepted neighborhood move and only `6` exact local-search evaluations, indicating a screening/ranking failure mode.

Phase I no-screen diagnostic proved that improving `insert_inter` 1-moves exist from the same start and improved to `6920`.

Phase J objective:

- redesign analytical `insert_inter` screening/ranking (not ML)
- recover no-screen move pattern with bounded exact-DP effort
- keep single-point scope (`61/347`) and exact DP only on touched machines.

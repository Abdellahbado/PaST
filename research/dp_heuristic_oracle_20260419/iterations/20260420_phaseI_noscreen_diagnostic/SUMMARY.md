# Summary

Phase I performs a bounded diagnostic at `61/347` to separate two hypotheses after Phase H:

- H1: Phase H screening was too aggressive and hid improving 1-moves.
- H2: the best multistart incumbent was already a true 1-move local optimum.

Executed method:

- start from Phase-H-style best multistart incumbent
- run no-screen exact move evaluation on bounded feasible batches
- required neighborhood: `insert_inter`
- optional neighborhood included in implementation: `swap_inter` when budget remains
- cap ladder used: `64`, `256`, `1024`

Observed outcome:

- improving `insert_inter` move found at all tested caps
- best TEC improved from `6944` to `6920`
- evidence supports H1 (screening too aggressive), not H2.

Recommended next branch:

- stay on single-point scope and redesign screening/ranking so that shortlist admits moves similar to the no-screen improving moves; keep exact-eval budget bounded.

# Problem

Date: 2026-04-20

Phase H at `61/347` improved TEC (`6944`) against current heuristic baselines but accepted `0` VND neighborhood moves.

Open diagnostic question:

- did shortlist/screening hide improving 1-moves, or
- is the best multistart incumbent already a true 1-move local optimum for tested neighborhoods?

Phase I target:

- run one bounded no-screen exact-move diagnostic at `61/347`
- focus on `insert_inter` (required) and `swap_inter` (optional if budget remains)
- evaluate tested candidate batch exactly (no LB shortlist pruning before exact DP) and decide which hypothesis is supported.

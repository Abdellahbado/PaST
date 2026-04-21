# Phase I No-Screen Diagnostic Design

Date: 2026-04-20

Competing hypotheses:

1. Screening was too aggressive and hid improving 1-moves.
2. The best multistart incumbent is already a true 1-move local optimum.

Neighborhoods tested:

- required: `insert_inter`
- optional: `swap_inter` (evaluated only if budget remains after insert batch)

What "no screening" means in this pass:

- for the tested batch, feasible candidates are enumerated directly and exact-evaluated on touched machines
- no relaxed-LB shortlist pruning before exact DP for those tested candidates

Bounded exact-evaluated batch:

- start from best multistart Phase-H-style incumbent at `61/347`
- source machines prioritized by highest current exact TEC (top 8)
- exact-eval cap ladder: `64`, `256`, `1024`
- wall-clock cap: `30s` per run
- first-improvement apply-once diagnostic behavior (accept best improving tested move if found)

Outcome interpretation:

- supports "screening too aggressive" if at least one improving 1-move is found in no-screen exact batches
- supports "true 1-move local optimum" if no improving 1-move is found across bounded no-screen batches

Next branch by outcome:

- if improving moves exist: redesign screening/ranking to retain those moves under bounded cost
- if none exist: stop 1-job neighborhood line and move to stronger diversification (more multistart or larger neighborhoods/destroy-repair)

# Summary

This iteration tests a bounded local-search extension after baseline assignment.

Question:

- can exact single-machine DP, used only to evaluate local moves, improve `greedy_dp` without exploding runtime/memory?

Implemented:

- new variant `greedy_dp_local_search` in `solvers/cpp/parallel_heuristic_compare.cpp`
- neighborhoods: relocate + swap
- strict-improvement acceptance on total TEC
- deterministic bounded controls:
  - `ls_time_cap_sec`
  - `ls_max_rounds`
  - `ls_max_moves_per_round`
- added reporting fields: accepted/evaluated moves, dominant move, final loads, local-search exact DP calls

Checkpoint results on required rows:

- `46/120`: `112 -> 103` (matches exact)
- `61/350`: `7053 -> 7046`
- `64/77`: `30598 -> 30580` (matches exact)
- `90/82`: `53294 -> 53294` (already exact)

Interpretation:

- local search improves or ties `greedy_dp` on all tested rows
- accepted improvements are relocate-dominated
- swap exploration cost is visible; pruning is the first optimization target

Resource profile:

- all runs stayed far below 16 GB RSS
- runtime increased vs `greedy_dp`, but remained small in absolute terms on tested rows

Decision:

- branch is worth continuing with bounded local-search engineering, especially move screening to reduce expensive non-improving swap evaluations.

Refinement-search checkpoint (same iteration, 2026-04-19):

- considered multiple paths: relocate-only, screened swap, machine prioritization, and a narrow paper-inspired equal-size/near-size swap restriction.
- implemented and tested three refinement variants:
  - `greedy_dp_local_search_relocate_only`
  - `greedy_dp_local_search_screened_swap`
  - `greedy_dp_local_search_priority_machines`
- fixed metric honesty by splitting DP calls into initialization vs local-search-only.

Refinement outcomes:

- `relocate_only` matches baseline local-search TEC on all tested rows with much lower swap-related effort.
- `priority_machines` gives best hard-row result on `61/350` but is not uniformly robust (`46/120` regression).
- `screened_swap` did not beat relocate-only in Stage 1.

Current recommendation:

- continue with `relocate_only` as default refinement direction
- keep `priority_machines` as optional targeted mode for hard rows
- stop screened-swap path for now

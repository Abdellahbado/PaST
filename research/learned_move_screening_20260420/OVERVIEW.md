# Overview

This research thread studies a learning-based extension of the insert-focused exact-DP heuristic for the BPMSTP benchmark in:

- `temp/paper_exact_repo/instances`

This is a new thread because the objective has changed.

The predecessor thread:

- `research/dp_heuristic_oracle_20260419/`

asked:

- can a handcrafted DP-centered heuristic compete with the paper's EHS?

This thread asks a different question:

- can a learning-based move-ranking component improve the validated insert-focused exact-DP heuristic without replacing the exact DP oracle?

## Stable inherited facts from the predecessor thread

- exact single-machine DP is stronger than the paper's ESR retiming step
- DP-guided greedy assignment from scratch was not useful
- bounded local search only became competitive after:
  - multistart initialization
  - insert-focused screening redesign
- the best handcrafted fixed-`epsilon` heuristic result so far at `61/347` is:
  - `6884`
- the main handcrafted bottleneck is now:
  - which `insert_inter` moves should receive exact DP evaluation

## Working method view for this thread

At fixed makespan cap `epsilon`:

1. build a bounded multistart incumbent
2. generate many candidate `insert_inter` moves
3. rank a small subset of those moves
4. evaluate only that subset with exact DP on touched machines
5. accept only exact-DP-improving moves

The learning insertion point is Step 3.

## Thesis-level purpose

The goal is not necessarily to beat EHS on every row.

The goal is to produce a clear and publishable learning-based method:

- exact DP remains the verifier / oracle
- learning decides which moves are most worth exact evaluation
- analytical features and lower-bound signals remain part of the method

## Re-entry rule

For a fresh conversation or resume:

1. read `START_HERE.md`
2. read `ACTIVE.md`
3. read the referenced iteration `SUMMARY.md`
4. read only the minimum extra files needed from:
   - `reference/`
   - `history/phase_reports/`

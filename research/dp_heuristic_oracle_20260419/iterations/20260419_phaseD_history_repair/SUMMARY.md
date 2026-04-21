# Summary

Phase D explores whether continuity-aware repair (paper A-SGH style) is the missing scaffold.

What was implemented:

- new history-chain execution mode in `solvers/cpp/parallel_heuristic_compare.cpp`
- two DP-enhanced repair prototypes:
  - `history_repair_dp_ranked`
  - `history_repair_priority_displaced_relocate`
- repair metrics added to CSV:
  - `epsilon_prev`
  - `displaced_jobs`
  - `reinsertion_candidates_scored`
  - `exact_dp_evals_repair`
  - `exact_dp_evals_post_repair_local_search`
  - `relocate_cleanup_used`

Paper-aligned transitions tested:

- `46: 77 -> 73`
- `61: 347 -> 345`
- `64: 79 -> 77`
- `90: 84 -> 82`

Key findings:

- `history_repair_dp_ranked` is not competitive and breaks on tighter transitions.
- `history_repair_priority_displaced_relocate` is better and achieves one same-`epsilon` win over paper EHS (`46/73: 103 vs 104`), interpreted as continuity + relocate-cleanup signal rather than reinsertion-alone strength.
- on `61`, history-repair with relocate cleanup beats one-shot relocate at tested points, but remains far above paper EHS.
- robustness is currently insufficient (`64` and `90` chain failures).

Decision:

- branch is promising but not yet ready as a default replacement.
- continue only with the prioritized+relocate path after targeted feasibility hardening; stop DP-ranked-only path.

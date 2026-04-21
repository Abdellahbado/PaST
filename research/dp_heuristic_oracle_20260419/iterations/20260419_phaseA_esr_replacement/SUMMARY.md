# Summary

This iteration launches a new heuristic-DP research line.

Question:

- Can the single-machine DP improve BPMSTP heuristic quality when used as a machine optimizer, starting with ESR replacement?

Current plan:

1. Run a narrow fixed-`epsilon` experiment
2. Hold assignment fixed
3. Compare paper-style ESR against our exact single-machine DP
4. Measure quality against exact CP-SAT values
5. Only if that succeeds, test DP-guided assignment scoring

Success criterion:

- clear TEC improvement from DP machine optimization on most tested rows

Failure criterion:

- ESR replacement produces little or no quality gain, or impractical runtime for the gained quality

Current checkpoint (2026-04-19):

- Implemented `solvers/cpp/parallel_heuristic_compare.cpp` and wired build target.
- Ran Phase A rows on corrected benchmark root:
  - `46/120`, `61/350`, `64/77`, `90/82`.
- On fixed assignment, `greedy_dp` improved TEC vs `greedy_esr` on all tested rows.
- Exact-gap against CP-SAT also improved on all rows.
- Memory remained far below the 16 GB cap.

Lower-bound correction update:

- previous LB diagnostic was invalid due to multiplicity loss in `relaxed_machine_lb(...)`.
- this was corrected; reported assignment-conditioned LBs are now safe by construction
  (relaxed-DP LB when valid, otherwise safe slot-based fallback).

Decision at this checkpoint:

- Phase A passes the stated success criterion; branch can proceed to Phase B experiments.

Phase B checkpoint (same date):

- added `dp_guided_assignment_dp` variant (DP-guided assignment + DP machine optimization).
- result vs `greedy_dp` is mixed and mostly negative on tested rows.
- no clear additional quality gain from DP-guided assignment in this phase.

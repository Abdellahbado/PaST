# Phase A ESR Replacement Readiness

Date: 2026-04-19

## Readiness check against Phase A objective (corrected)

Objective:

- hold machine assignment fixed
- compare paper-style ESR vs exact single-machine DP
- decide whether DP gives meaningful TEC improvement

Status:

- **Ready / Passed** for Phase A decision, after LB correction.

## Evidence summary

From `research/dp_heuristic_oracle_20260419/phaseA_esr_replacement_results.md`:

- `greedy_dp` beats `greedy_esr` on all tested rows (`46/120`, `61/350`, `64/77`, `90/82`)
- exact-gap to CP-SAT optimum improves on all tested rows
- heavy runs stayed far below `16 GB` RSS
- `greedy_dp` beats `greedy_esr` on all tested rows (`46/120`, `61/350`, `64/77`, `90/82`)
- exact-gap to CP-SAT optimum improves on all tested rows
- heavy runs stayed far below `16 GB` RSS

## Constraints compliance

- no full `epsilon` frontier loop implementation was added
- no full A-SGH or full R-ES implementation was added
- no exact global DP-proof branch was revived
- implementation is a thin comparison driver plus CMake wiring

## Lower-bound diagnostic status

- The old LB diagnostic was invalid because multiplicities were dropped in `relaxed_machine_lb(...)`.
- This has been corrected in code.
- Reporting correction: do not treat `greedy_esr` assignment-conditioned LB as a validated safe decision bound.
- In this phase, readiness is gated by TEC and exact-gap behavior, not ESR LB diagnostics.
- Where LB is reported, it is diagnostic and guarded by fallback/sanity checks in the driver.

## Remaining caveats

- assignment baseline is intentionally simple (LPT-style greedy insertion under `epsilon`)
- assignment-conditioned LB remains diagnostic only and not a global optimality certificate
- many large-row machine LBs currently come from safe fallback bounds, so LB tightness is limited

These caveats are expected and consistent with Phase A scope.

## Recommendation

- Proceed to Phase B (`CODER_PROMPT_PHASE_B_DP_GUIDED_ASSIGNMENT.md`) is justified by TEC improvement.
- Treat LB diagnostics as safe-but-mixed-strength (not the gating criterion).

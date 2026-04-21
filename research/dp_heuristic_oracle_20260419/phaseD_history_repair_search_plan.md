# Phase D History-Repair Search Plan

Date: 2026-04-19

## 1) What paper EHS is doing structurally

From `Papers/Exact and heuristic.txt`:

- **A-SGH** (Split-greedy with assignment history): solve `Khat` sequence by reusing previous solution and repairing only assignments that become infeasible when `Khat` decreases.
- **R-ES**: local search with rescheduling to improve TEC after constructive/repair stage.
- **ESR**: exact single-machine sequence-preserving timing optimizer used as a strong machine-level improver.

So EHS strength is a scaffold: continuity over decreasing `Khat` + repair + local improvement + ESR.

## 2) What we already improved

- We already replaced ESR-like machine optimization with exact single-machine DP and validated gains (Phase A).
- We validated post-assignment local search and relocate-only refinement (Phase C).

## 3) What is likely still missing

- continuity-aware repair across neighboring `epsilon` values (paper A-SGH-like history reuse) is not present in one-shot variants.

## 4) Candidate DP-enhanced history-repair paths

### Path A: A-SGH-style repair with DP-ranked reinsertion

- Start from assignment at `epsilon+1`.
- Remove jobs that violate `epsilon`.
- Reinsert displaced jobs by ranking candidate machines using DP-safe LB deltas.
- Use exact DP for touched-machine acceptance.

Plausible: preserves history while using DP where strong.
Risk: naive displacement policy may break feasibility on tighter steps.

### Path B: Path A + relocate-only cleanup

- Same repair as Path A.
- Then run relocate-only exact-DP local search.

Plausible: combines continuity scaffold with best validated improver.
Risk: extra runtime without fixing poor repair decisions.

### Path C: priority repair on displaced set

- Reinsert displaced jobs in disruption-priority order (e.g., high `rate*p`), not naive order.
- Keep DP-ranked candidate machine scoring.

Plausible: may reduce early bad reinsertion choices in hard rows.
Risk: priority signal may be unstable across instances.

### Path D: narrow exchange after repair

- very restricted exchange neighborhood after repair, exact-DP-evaluable.

Plausible: could recover specific local misses.
Risk: complexity creep; lower priority than A/B/C.

## 5) Selected for this implementation pass

- `history_repair_dp_ranked` (Path A)
- `history_repair_priority_displaced_relocate` (Path C + Path B)

## 6) Deferred for now

- `history_repair_dp_ranked_relocate`
- `history_repair_priority_displaced` without cleanup
- Path D exchange extension

Reason: keep this pass bounded while testing both pure-repair and repair+cleanup behavior.

# Phase D History-Repair Readiness

Date: 2026-04-19

## Objective

Assess whether history-aware repair with DP-guided machine-local decisions should continue as the next main branch.

## Status

- **Conditionally ready**: continue a narrowed path only.

## Keep

- `history_repair_priority_displaced_relocate`
  - beats one-shot relocate on hard row `61` at tested points
  - achieves one same-`epsilon` win over paper EHS (`46/73`), interpreted as continuity + relocate-cleanup signal (not standalone reinsertion-strength proof)

## Stop (for now)

- `history_repair_dp_ranked`
  - weaker quality and no robustness advantage
  - does not beat one-shot relocate baseline on tested points

## Current blocker to broad continuation

- chain infeasibility on `64` and `90` tight transitions

## Next required step before scaling

- strengthen displacement and reinsertion fallback in repair step so chain can remain feasible on tighter transitions.

## Recommendation

- continue Phase D with only the prioritized+relocate path, and make feasibility robustness the sole near-term target before any wider claim versus paper EHS.

# Summary

## Status: Direction stopped (Decision C)

This iteration tested whether alternative adjacent coarsenings of the recovered block list could improve beam incumbent quality on hard K10 families. No single coarsening view passed the gate.

## What was done

- Implemented 6 alternative block coarsening views in `pack_recovered_blocks`
- Ran 56-row Phase A gate (hardA_k10 + hardB_k10, seeds 0-3, 7 variants each)
- All rows memory-safe, no timeouts

## Key result

**No single variant improves ≥ 4/8 K10 anchor rows.** Best is 3/8:
- `target_B12`: 3 improved (hardA s2, hardB s1, hardB s2), 1 same
- `target_B8`: 3 improved (hardB s0, s1, s2), 5 worse
- `price_preserve_B12`: 3 improved (hardA s2, hardB s1, s2), 1 same

## Why

Coarsening loses price-profile fidelity. The beam's cost estimates become less accurate with wider blocks. Wider blocks also explode pattern counts (larger capacities → more candidate patterns), counteracting the intended benefit of fewer layers.

## Decision

**Decision C**: Adjacent block coarsening does not improve beam incumbent quality. The beam is already near-optimal with the standard block structure.

## Next

A new direction is needed for the hard-family closure problem. Block-level approaches (PLAN28 diagnostics, PLAN29 coarsening) have been exhausted.

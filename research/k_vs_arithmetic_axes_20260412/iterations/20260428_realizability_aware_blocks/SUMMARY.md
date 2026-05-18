# Summary

## Status: Direction stopped (Decision C)

This iteration tested whether block-realizability diagnostics could separate easy unit-contiguous rows from hard irregular rows at fixed `n=1000`. The diagnostics fail to separate.

## What was done

- Implemented block-realizability diagnostics in the C++ solver, triggered by `PAST_BLOCK_REALIZ_DIAG=1`
- Ran a 18-row Phase A diagnostic gate (easy K8/K10/K12, hardA K8/K10/K12, hardB K8/K10/K12, seeds 0/1 at `n=1000`)
- All rows memory-safe (peak RSS 0.6–5.7 GB)

## Key result

**`block_realiz_base_path_survives = 0` for ALL 17 rows with beam incumbents.** The beam's chosen counts are never locally feasible at block 0 — this is universal across easy and hard families.

**`block_realiz_bad_rate = 50%`** for both easy families and hardA_k10. A diagnostic that produces the same value for a gap-0 row (easy) and a gap>0 row (hard) does not separate.

## Why this happened

The beam validates the full global sequence. Block-local evaluation (`evaluate_profile_block_counts`) is stricter — it requires counts to be independently schedulable within each block. The beam's counts pass global validation but always fail block-local validation at block 0. This is a structural mismatch, not a quality signal.

Easy families bypass this by closing at Step 2 (FFD/FFI) without ever needing the beam's blocks. Hard families must use the beam, and the beam's blocks are universally bad at the beginning.

## Decision

**Decision C**: Diagnostics do not separate easy from hard. Stop this direction. No Phase B repair or Phase C controls.

## Next

The active direction for this iteration is complete. A new direction or iteration should address a different aspect of the hard-family closure problem.

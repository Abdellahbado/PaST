# PLAN24 Beam Corridor Exact DP - Notes

## Experiment
Test whether a beam-guided Step-4 exact corridor (restricting exact DP to count vectors near the Step 3 beam's prefix-count trajectory) improves exact closure, gap, or runtime on hard irregular K=10 rows.

## Variants
- `standard_step4`: profile-repair beam + normal sparse exact DP, no corridor
- `corridor_delta0`: exact beam prefix trajectory only (delta=0)
- `corridor_delta1`: +/-1 per type around beam trajectory
- `corridor_delta2`: +/-2 per type around beam trajectory
- `corridor_widen_0_1_2`: time-sliced widening schedule (delta=1)

## Target rows
- hardA_k10: seeds 0-3, n=1000
- hardB_k10: seeds 0-3, n=1000
- K=12 probe skipped (no signal from K=10)

## Critical fix
- Initial smoke run (11 rows) used PAST_RELAXED_BINPACK_SOLVER=energy_core which produced no beam incumbent (ub=-1). These rows are preserved in PLAN24_invalid_energy_core_misroute_raw.csv.
- Corrected to PAST_RELAXED_BINPACK_SOLVER=profile_repair_beam with PAST_PROFILE_REALIZATION_SELECTOR_POLICY=auto_v1.
- Added route sanity check in runner to detect misrouted rows.

## Results summary
- 33 total valid rows (4 smoke + 29 Phase B)
- All rows reach step4 with fwd_pack_method=profile_repair_beam, beam_status=feasible
- Corridor correctly activated: corridor_enabled=1 with correct deltas for all corridor variants
- Zero corridor pruning: exact_diag_corridor_pruned=0 for all 24 corridor rows. No state was ever pruned.
- Zero corridor infeasibility: exact_diag_corridor_infeasible=0 for all rows
- Sparse exact DP skipped: exact_diag_states_reached=0, exact_diag_elapsed~0 for all rows. Mode is sparse_skip_theoretical.
- Identical UB/LB/gap: All corridor variants produce exactly the same results as standard_step4.
- No exact closure: is_optimal=0 for all rows.
- Comparable runtime: No significant overhead or benefit from the corridor.
- Memory well under cap: Max RSS ~8.1 GB < 16 GB.

## Root cause
The sparse exact DP mode `sparse_skip_theoretical` means the solver determines the theoretical lower bound is already tight enough, so it skips the exact search entirely. This prevents the corridor from having any effect because no states are ever generated or pruned.

On hard irregular K=10 instances with profile-repair beam, the beam produces a strong incumbent within the theoretical bound. The sparse exact DP concludes the search is fruitless and skips.

## Decision
**D**: No evidence that beam-guided exact corridor improves exact closure, gap, or runtime on hard irregular K=10 rows. The corridor machinery is functional but the sparse exact DP skips the search, making corridor pruning irrelevant. With zero pruning in 24 corridor rows across 8 instances, the approach is unlikely to yield benefits even if exact DP ran.

## Artifacts
- PLAN24_beam_corridor_exact_raw.csv: 33 valid rows
- PLAN24_invalid_energy_core_misroute_raw.csv: 11 invalid rows (energy_core misroute)
- PLAN24_beam_corridor_exact_compare.csv: Side-by-side comparison per family/seed
- PLAN24_beam_corridor_exact_summary.csv: Aggregate statistics per variant
- PLAN24_beam_corridor_exact_notes.md: This file

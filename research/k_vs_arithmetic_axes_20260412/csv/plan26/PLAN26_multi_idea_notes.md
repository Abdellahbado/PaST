# PLAN26: Validate PLAN25 Local Corridor + Multi-Idea Improvement Queue — Notes

## Objective
Validate whether PLAN25 local corridor is mechanically correct, and if not, run a bounded multi-idea improvement queue for hard irregular K=10.

## Phase 0 — Correctness Repairs
Implemented before any experiment:
1. **Fixed `lb=ub` bug**: In `stateful_compare.cpp`, local corridor UB improvements no longer set `lb=ub`. Local corridor is exact only inside the corridor, not globally.
2. **Added `LocalCorridorDiag` fields** for alignment validation:
   - `beam_counts_size`, `merged_blocks`, `block_count_mismatch`
   - `target_offset_l1`, `target_in_corridor`
   - `base_candidates_finite`, `empty_candidate_blocks`, `first_empty_layer`
   - `base_path_survives`, `base_path_cost`, `base_path_reject_reason`
3. **Fixed `merged_blocks` propagation**: Added `merged_blocks` to `RecoveredBlockPackingResult`, populated it in `pack_recovered_blocks`, and propagated to `RelaxedDPResult` in all three `solve_relaxed_dp_with_binpack` exit paths. `stateful_compare.cpp` now uses `fwd.merged_blocks` directly instead of reconstructing from `fwd.block_profile`.
4. **Added base-path survival simulation** in `beam_corridor_local_dp`: explicitly checks whether the beam's own count vector survives a layer-by-layer simulation before running the full DP.

## Phase 1 — Validate Local Corridor Alignment (one planned row missing)

### Target Rows
- hardA_k10, n=1000, seed=0
- hardB_k10, n=1000, seed=2

### Variants
- `standard_step4`
- `local_corridor_delta1_300s`
- `local_corridor_delta2_300s`

### Key Diagnostic Results

| Instance | Variant | Base Path Survives | Mismatch | Target L1 | Empty Blocks | Reject Reason |
|---|---|---|---|---|---|---|
| hardA_k10 s0 | delta1 | 0 | 0 | 0 | 2 | `base_candidate_not_found_at_layer_0` |
| hardA_k10 s0 | delta2 | 0 | 0 | 0 | 2 | `base_candidate_not_found_at_layer_0` |
| hardB_k10 s2 | delta1 | 0 | 0 | 0 | 2 | `base_candidate_not_found_at_layer_0` |

Note: the planned `hardB_k10 s2 local_corridor_delta2_300s` row is not present in the raw artifact. The Phase 1 conclusion is still supported by the common layer-0 base-candidate failure on both tested families, but the Phase 1 grid is not fully complete.

### Interpretation
- **Block count mismatch = 0**: `merged_blocks` and `beam_chosen_counts` have the same size. Block partition alignment is correct after the fix.
- **Target offset L1 = 0**: The beam assigns all jobs (no remaining counts). Target is within corridor.
- **Base path does NOT survive**: `base_candidate_not_found_at_layer_0` on ALL local corridor rows.
- **Root cause**: `evaluate_profile_block_counts` returns `kInf` for the base beam candidate on at least 2 blocks per instance.

### Why `evaluate_profile_block_counts` returns `kInf`
`generate_energy_core_patterns` generates patterns by **work capacity** (total job lengths ≤ block capacity), not by **schedulability** (whether the specific multiset can actually be scheduled in the block given machine transition constraints). The beam selects patterns based on surrogate scores (center deviation, feasibility pressure, local deviation, arithmetic pressure) and then validates the **full global sequence** using `solve_fixed_sequence` on the entire horizon.

The local corridor DP, however, evaluates each block's counts **independently** using `evaluate_profile_block_counts` with local block views. If the block boundaries don't align with actual schedule boundaries (which they don't, because they're recovered from a relaxed DP path), the base beam counts may not be independently schedulable within each block. This is a **fundamental mismatch** between the beam's global validation and the corridor's local validation.

### Phase 1 Conclusion
**Local corridor is invalid** under its current design. The base beam path fails because block-local schedulability is not guaranteed by the beam. Phase 2 (candidate-set expansion) and Phase 3 (multi-center corridor) are blocked.

## Phase 2 & 3 — Cancelled
Blocked by Phase 1 finding. No candidate-set or multi-center variant can succeed if the base beam path itself is rejected by the block-local evaluator.

## Phase 4 — Step-3 Beam Scoring Variants (partial)

### Available Variants
Only `ambig_scoreband_mult2` was available in the current code and was tested. The originally requested `residual_aware` and `late_ambig` scoring policies were not implemented in this pass, so Phase 4 should be read as a partial fallback check, not a complete Step-3 scoring study.

### Rows
- hardA_k10 seeds 0, 2
- hardB_k10 seeds 0, 2

### Results

| Instance | Variant | UB | Gap% | Runtime(s) |
|---|---|---|---|---|
| hardA_k10 s0 | standard | 118,817,429 | 0.0273 | 496.2 |
| hardA_k10 s0 | ambig_scoreband_mult2 | 118,820,437 | 0.0299 | 460.0 |
| hardA_k10 s2 | standard | 123,402,702 | 0.0217 | 601.9 |
| hardA_k10 s2 | ambig_scoreband_mult2 | 123,413,654 | 0.0306 | 494.8 |
| hardB_k10 s0 | standard | 149,439,243 | 0.0391 | 738.1 |
| hardB_k10 s0 | ambig_scoreband_mult2 | 149,436,775 | 0.0375 | 652.0 |
| hardB_k10 s2 | standard | 145,046,294 | 0.0450 | 719.0 |
| hardB_k10 s2 | ambig_scoreband_mult2 | 145,037,455 | 0.0389 | 622.3 |

### Analysis
- **hardA_k10**: `ambig_scoreband_mult2` is WORSE on both seeds (gap 0.0299% vs 0.0273%; 0.0306% vs 0.0217%).
- **hardB_k10**: `ambig_scoreband_mult2` is BETTER on both seeds (gap 0.0375% vs 0.0391%; 0.0389% vs 0.0450%).
- **Runtime**: `ambig_scoreband_mult2` is faster on all 4 rows (by 36-120s).

### Promotion Check
- At least 3/4 rows not worse: **2/4** (fails — worse on hardA_k10 seeds 0 and 2).
- At least 2/4 rows improve UB/gap: **2/4** (passes — improves on hardB_k10 seeds 0 and 2).

Since the first criterion fails, **do not promote `ambig_scoreband_mult2` globally**.

This confirms PLAN22B's corrected decision: `ambig_scoreband_mult2` is a K=10 quality-improvement candidate for hardB instances only, not a global policy.

## Memory Safety
All rows completed without memory kills. Peak RSS:
- Standard: 5.3–7.3 GB
- Local corridor: 4.9–6.5 GB
- ambig_scoreband_mult2: 5.3–7.6 GB

All well within the 16 GB cap.

## Final Decision: **C**

**Local corridor invalid due to block/path mismatch; do not use until repaired.**

The local corridor DP assumes that the beam's chosen counts are independently schedulable within each recovered block. The beam does not guarantee this — it only validates the full global sequence. This is a fundamental design mismatch, not a bug that can be fixed by tuning delta or candidate generation.

To make local corridor valid, one of the following would be required:
1. Replace block-local evaluation with global sequence evaluation (major redesign).
2. Generate only actually schedulable patterns per block (may severely limit candidate pool).
3. Abandon the block-layered DP and use a global state-space approach with local offset encoding.

None of these are justified by the current evidence. The beam incumbent is already near-optimal (gaps < 0.05%). The marginal potential improvement does not justify a major corridor redesign.

**No tested variant (corridor or existing Step-3 scoring) improves consistently across the K=10 hard irregular distribution.** The requested new Step-3 policies (`residual_aware`, `late_ambig`) remain untested.

## Artifacts
- `csv/plan26/PLAN26_multi_idea_raw.csv` (11 rows)
- `csv/plan26/PLAN26_multi_idea_compare.csv`
- `csv/plan26/PLAN26_multi_idea_notes.md` (this file)
- `implementation_plans/PLAN26_beam_corridor_multi_idea_queue.md`

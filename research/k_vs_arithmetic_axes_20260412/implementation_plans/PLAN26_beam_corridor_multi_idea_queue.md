# PLAN26: Validate PLAN25 Local Corridor + Multi-Idea Improvement Queue

## Current Blocker Summary

PLAN25 implemented `beam_corridor_local_dp()` with local offset encoding to avoid PLAN24B's int64 overflow. Smoke test on hardA_k10 s0 and hardB_k10 s2 ran without crashes, but all local-corridor rows returned `infeasible_corridor` with `best_ub=inf`.

Critical unanswered questions:
1. Does `merged_seg` used by local corridor match the blocks used by `profile_repair_beam`?
2. Does the base beam count vector have finite local evaluation cost?
3. Does the zero-offset beam path survive every layer of the local DP?
4. Is `lb=ub` incorrectly set when corridor-limited UB improves?

PLAN24B proved global mixed-radix encoding overflows for K=10 at n=1000. PLAN25's local offset encoding avoids that, but the method remains unvalidated.

## Hypotheses

H1: `infeasible_corridor` is caused by block partition mismatch between `fwd.block_profile` (reconstructed in `stateful_compare.cpp`) and the actual `merged` blocks used by `profile_repair_beam`.
H2: `infeasible_corridor` is caused by `evaluate_profile_block_counts` returning `kInf` for the base beam candidate on some block.
H3: The beam prefix assigns all jobs (target_offset=0), so the base path should survive. If it doesn't, there's a bug in the DP transition logic.
H4: Even if the base path survives, the corridor is too narrow to find improvements. Candidate-set expansion might help.
H5: A single beam center is too restrictive. Multiple near-best centers might capture better completions.
H6: If corridor variants fail, Step-3 beam scoring changes (residual-aware, late-diversity) might improve incumbent quality.

## Exact Variant Queue

### Phase 0 — Mandatory Correctness Repair (done before any experiment)
- Fix `stateful_compare.cpp`: if local corridor returns `local_ub < ub`, update `ub` only, do NOT set `lb=ub`.
- Add 12 new `LocalCorridorDiag` fields for alignment validation.
- Use `fwd.merged_blocks` directly instead of reconstructing from `fwd.block_profile`.
- Populate `merged_blocks` in `RecoveredBlockPackingResult` and propagate to `RelaxedDPResult`.

### Phase 1 — Validate Base-Path Survival
Target rows:
- hardA_k10, n=1000, seed=0
- hardB_k10, n=1000, seed=2

Variants:
- `standard_step4` (baseline)
- `local_corridor_delta1_300s`
- `local_corridor_delta2_300s`

Success criteria:
- `local_corridor_base_path_survives=1`
- `local_corridor_base_path_cost` finite
- `local_corridor_block_count_mismatch=0`
- `local_corridor_target_offset_l1=0` (beam assigns all jobs)
- No `lb=ub` from corridor
- Memory safe

Stop rule: If `base_path_survives=0` or `block_count_mismatch=1`, STOP corridor variants. Fix the bug before continuing.

### Phase 2 — Candidate-Set Variants
Run only if Phase 1 validates base-path survival.

Variant A: `energy_topN`
- Include energy/core generated block patterns as candidates.
- Always keep beam base vector.
- Keep top N per block by local cost.
- Env: `PAST_LOCAL_CORRIDOR_CAND_POLICY=energy_topN`, `PAST_LOCAL_CORRIDOR_CAND_TOP_N=50`

Variant B: `beam_neighborhood`
- Always keep beam base vector.
- Add perturbations ranked by local block cost + closeness to beam.
- Env: `PAST_LOCAL_CORRIDOR_CAND_POLICY=beam_neighborhood`, `PAST_LOCAL_CORRIDOR_CAND_TOP_N=100`

Variant C: `residual_guided`
- Prefer candidates that reduce remaining global count imbalance.
- Env: `PAST_LOCAL_CORRIDOR_CAND_POLICY=residual_guided`, `PAST_LOCAL_CORRIDOR_CAND_TOP_N=100`

Run rows: hardA_k10 s0, hardB_k10 s2, delta=1 and delta=2.

Stop rule: If no variant finds a feasible path (all `infeasible_corridor`), proceed to Phase 3.

### Phase 3 — Multi-Center Corridor
Run only if Phase 2 gives feasible local paths but no UB improvement.

Hypothesis: single beam prefix is too restrictive.
Implement: `PAST_LOCAL_CORRIDOR_MULTI_CENTER=1`, `PAST_LOCAL_CORRIDOR_CENTER_COUNT=2/3`.

Blocked if: near-best beam terminal paths are not cheaply available.
Run rows: hardA_k10 s0, hardB_k10 s2, delta=1 and delta=2.

Stop rule: If multi-center is blocked or shows no improvement, proceed to Phase 4.

### Phase 4 — Step-3 Beam Scoring Variants
Run only if corridor remains invalid or useless.

Variant D: `residual_aware` beam scoring
- Env: `PAST_PROFILE_REPAIR_BEAM_SCORE_POLICY=residual_aware`

Variant E: `late_ambig` diversity
- Env: `PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_POLICY=late_ambig`, `PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_MAX=2`

Compare against: standard `profile_repair_beam`, `ambig_scoreband_mult2`.
Rows: hardA_k10 seeds 0,2; hardB_k10 seeds 0,2.

Promotion rule: Promote only if >=3/4 rows not worse AND >=2/4 rows improve UB or gap.

## Gate/Stop Rules

1. Phase 0 must complete and build must succeed before any experiment.
2. Phase 1 is mandatory. If base path does not survive, corridor is invalid. Stop corridor branch, proceed to Phase 4.
3. Phase 2 runs only if Phase 1 passes.
4. Phase 3 runs only if Phase 2 finds feasible paths but no improvement.
5. Phase 4 runs only if corridor is abandoned.
6. Do not run K=12 unless K=10 shows clear positive signal.
7. Do not run delta=3 unless delta=2 improves.
8. Do not claim global optimality from local corridor DP.

## Memory Rules

- RSS cap: 16 GB default.
- Absolute max: 20 GB only if justified and recorded.
- One heavy solver process at a time.
- Flush CSV after every row.
- No full stdout/stderr buffering.

## Artifact List

Created in `research/k_vs_arithmetic_axes_20260412/csv/plan26/`:
- `PLAN26_multi_idea_raw.csv`
- `PLAN26_multi_idea_compare.csv`
- `PLAN26_multi_idea_summary.csv`
- `PLAN26_multi_idea_notes.md`

## Final Decision Labels

- **A**: Local corridor fixed and useful; expand K=10 seeds 0-3.
- **B**: Local corridor valid but not useful; keep diagnostic only.
- **C**: Local corridor invalid due block/path mismatch; do not use until repaired.
- **D**: Candidate-set expansion improves incumbent; expand that variant.
- **E**: Multi-center corridor improves incumbent; expand that variant.
- **F**: Step-3 beam scoring improves incumbent; expand that variant.
- **G**: No tested variant improves; stop this branch and report boundary honestly.

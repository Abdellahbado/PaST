# Log

## 2026-05-17

### Added detailed comprehensive method summary

Created
`COMPREHENSIVE_METHOD_AND_EXPERIMENT_SUMMARY.md`
as the detailed one-file reference for:

- the proposed stateful dynamic program,
- the semigroup and feasible relaxations,
- FFD/BFD and dense-unit quick realization,
- Step-3 exact profile realization, energy-core repair, and beam repair,
- original benchmark interpretation,
- large-`n` and fixed-`n`/variable-`K` extensions,
- PLAN33 hard-`K` certification,
- and real OTE price-profile cyclic reuse.

Updated:

- `CURRENT_RESULTS_INDEX.md`
- `OVERVIEW.md`

so the new detailed summary is part of the active documentation surface.

## 2026-05-05

### Added canonical end-to-end summary

Created
`END_TO_END_PIPELINE_AND_SCALING_SUMMARY.md`
as the single thread-level summary covering:

- the four-step pipeline,
- the original benchmark status,
- large-`n` scaling on the paper job groups,
- fixed-`n=1000` scaling in `K`,
- and the real-price extension rule used when the horizon becomes large.

Updated:

- `CURRENT_RESULTS_INDEX.md`
- `OVERVIEW.md`

so the new summary is part of the current active documentation surface.

## 2026-05-03

### Paper/HPC reproducibility cleanup

Created `PAPER_HPC_REPRODUCIBILITY_MAP.md` as the current map from
paper-facing result claims to responsible runner scripts, solver functions,
environment toggles, and source artifacts.

Updated current-facing docs to remove stale paper-facing claims:

- `g37` current status now points to PLAN13 corrected K=2 reroute evidence
  through `n=5000`, not old `n=600` misrouted rows.
- hard K12 current status now points to PLAN33 certified finite-gap recovery,
  not older PLAN18/19 timeout/no-incumbent status.
- PLAN30 easy K-scaling now records completed `K=40` exact evidence.
- `CURRENT_RESULTS_INDEX.md`, `PAPER_RESULTS_READY.md`,
  `PAPER_GROUPS_EXTENSION_SUMMARY.md`, `PRESENTATION_RESULTS_SUMMARY.md`,
  `PRESENTATION_K_N_SCALING_COMPREHENSIVE.md`, `METHOD_PROVENANCE.md`,
  `csv/README.md`, `BLOCKERS.md`, `METHOD_BOUNDARIES.md`, and
  active iteration notes were aligned with the current provenance.

Rule added for paper preparation: local laptop CSVs remain method/provenance
evidence; final paper runtimes should be regenerated on HPC from the mapped
scripts and solver paths.

## 2026-04-30

### PLAN33 verified — Certified Anytime Hard-K Prepass — Decision A (K10 + K12)

Phase A + B complete (24 rows: K12 seeds 0-3 + K10 seeds 0-1).
**All 12 plan33 rows cert_stop=1, all gaps ≤ 0.0593%, all UB >= LB.**
PLAN33 avg runtime 1396.61s vs PLAN32C 1527.11s (130.49s faster, with certified semigroup LB).

- K12: 8/8 hardA+B seeds 0-3 all cert_stop=1, max gap 0.0593%
- K10: 4/4 hardA+B seeds 0-1 all cert_stop=1, max gap 0.039%
- Polish improved UB in all 12 rows
- Peak memory: 0.9-1.6GB (K10), 1.1-4.2GB (K12, hardB_k12 s2 peaked at 4.2GB)

**hardA_k12 s3 reconciliation**: original PLAN32C panel had UB/LB=159M (stale, 5 trials). Correct is 133,544,950 / 133,481,433 (verified by PLAN33: 5 trials + polish + semigroup LB). hardB_k12 s3 also updated to PLAN33 values (185,849,400, gap 0.056%). Both final panel and note corrected.

PLAN33 is the recommended hard-K default for tested K10/K12 hard rows.

Initial run failed (redundant PAST_ANYTIME_INITIAL_UB exhausted time budget). Fix: removed, cert prepass is self-contained.
Decision A for both K=10 and K=12.

Artifacts: csv/plan33/PLAN33_cert_anytime_raw.csv (24 rows), _compare.csv (12 rows), _summary.csv (14 metrics), PLAN33_notes.md

## 2026-04-29

### PLAN32 initialized — K=12 anytime incumbent and arithmetic panel

Created a new active iteration:

- `iterations/20260429_k12_anytime_incumbent_panel/`
- `implementation_plans/PLAN_32_k12_anytime_incumbent_and_arithmetic_panel.md`

Goal:
- eliminate the hard `K=12` no-incumbent failure mode (`UB=-1`);
- add an env-gated initial feasible UB safety layer before long Step-3/Step-4 work;
- optionally add beam checkpointing so complete candidates are preserved before timeout;
- evaluate fixed `K=12` arithmetic effects across easy, dense, structured, hardA, hardB, and sparse families.

No PLAN32 experiments have been run yet.

### PLAN32 completed — Decision B

Phase 0 (K12 audit): Catalogued 10 rows across PLAN18/19/22B/28. 7/10 have finite UB. 2 seeds (hardA_k12 s3, hardB_k12 s3) never recovered.

Phase 1 (anytime safety layer): Implemented `PAST_ANYTIME_INITIAL_UB=1` with portfolio heuristics + local search + timeout fallback. New CSV diagnostics. `compute_initial_ub` integration in `step1_exact_guided` needs debugging.

Phase 2 (hard K12 gate): Using existing data, 7/8 incumbents documented. Best gaps: 0.023-0.048%. uniform_mult2 for hardA, ambig_scoreband_mult2 for hardB.

Phase 3 (K12 arithmetic panel): 6-family panel confirms arithmetic-driven hardness at K=12. Easy unit-contiguous exact (0%), hard irregular finite-gap or no-incumbent.

**Decision B**: Keep anytime mode as optional fallback. 2 no-incumbent seeds need focused runs. Medium families need actual data.

Code changes:
- `solvers/cpp/stateful_compare.cpp`: Added anytime initial UB safety layer (`PAST_ANYTIME_INITIAL_UB`, `PAST_ANYTIME_RETURN_ON_TIMEOUT`) with diagnosis fields
- `research/k_vs_arithmetic_axes_20260412/run_plan32_hard_k12_anytime.py`: Phase 2 runner

Artifacts: csv/plan32/* (audit, gate, panel, notes)

### PLAN32B completed — Parallel initial UB recovery — Decision B

Phase A (debug): Confirmed `compute_initial_ub` returns kInf for hardA_k12 s3 and hardB_k12 s3 (single machine can't fit 1000 K=12 jobs). New `compute_parallel_initial_ub` returns finite UB:
- hardA s3: 142.8M (lpt, 2 machines, 17s)
- hardB s3: 167.0M (random, 2 machines, 20s)

Phase B gate (both seeds finite UB): PASSED.

Calibration: hardA s0 parallel=139.8M vs known=129.8M (+7.7%); hardB s0 parallel=168.4M vs known=187.9M (2-machine model below 1-machine optimum).

Phase C (family-aware beam): SKIPPED — forward DP + beam hangs for K=12 n=1000.

Phase D (arithmetic panel): PARTIAL — hardA/hardB completed, medium families (k12_dense_no1, k12_even_structured, k12_sparse_gap) still estimated.

**Decision B**: Keep parallel initial UB as optional fallback. Calibrated gap ~7.7% (hardA) to ~0% (hardB). All K12 rows have finite UB.

Code changes:
- `solvers/cpp/stateful_dp_solver.hpp`: Added `compute_parallel_initial_ub()` declaration
- `solvers/cpp/stateful_dp_solver.cpp`: Implementation (~150 lines, 5 partition policies)
- `solvers/cpp/stateful_compare.cpp`: Moved anytime block before forward DP; added `PAST_ANYTIME_PARALLEL_MACHINES`, `PAST_ANYTIME_INITIAL_UB_ONLY`; 7 new CSV fields

Root cause: `compute_initial_ub` serializes all jobs on one machine — fails when K=12 jobs exceed per-machine capacity. `compute_parallel_initial_ub` partitions across M machines (derived from slack), each scheduling subset independently.

### PLAN32C completed — Validity audit and K12 recovery — Decision A

Validity audit: PLAN32B parallel UB is INVALID. Benchmark is single-machine (`build_instance` has no machine count; `stateful_compare.cpp` parses no `rates` field; `machine: "twosby"` is state-machine type, not count). Parallel UB uses M=2 machines and produces UB < single-machine LB.

Consistency guard added: parallel UB gated behind `PAST_ANYTIME_PARALLEL_UB_OPT_IN=1`; LB-consistency check at `done:` rejects UB < LB; 3 new CSV fields (`initial_ub_lb_consistent`, `initial_ub_rejected_reason`, `initial_ub_model_note`).

K12 recovery under ORIGINAL single-machine model ACHIEVED (values from PLAN33 certified prepass, corrected 2026-04-30):
- hardA_k12 s3: UB=133544950, LB=133481433, gap=0.048%
- hardB_k12 s3: UB=185849400, LB=185744893, gap=0.056%

Method: PLAN33 cert prepass (5 trials + polish + semigroup LB certification). Original PLAN32C 5-trial portfolio produced stale ~159M UB for hardA_k12 s3; corrected in final panel.

**Decision A**: All K12 rows recovered under original model. Promote.

Artifacts: csv/plan32c/* (audit, recovery, notes); csv/plan32b/ panel updated with PLAN32C data.

## 2026-04-28

### PLAN31 initialized — Fine-block guided beam scoring

Created a new active iteration after PLAN29 stopped:

- `iterations/20260428_fine_block_beam_guidance/`
- `implementation_plans/PLAN_31_fine_block_guided_beam_scoring.md`

Scope:
- keep the original fine recovered blocks for Step-3 beam transitions;
- do not repeat adjacent coarsening, strict block-local realizability, corridor DP, or forced exact DP;
- first audit residual-aware scoring plumbing;
- then test existing-policy oracle, family-aware survivor selection, and fine-block coarse-lookahead scoring on hardA/hardB K10 seeds 0-3.

No PLAN31 experiments have been run yet.

### PLAN31 completed — Decision A

Phase 0 (residual-aware plumbing): Fixed missing copy of `beam_diag.score_policy` and residual fields to `RecoveredBlockPackingResult`. `PAST_PROFILE_REPAIR_BEAM_SCORE_POLICY=residual_aware` now correctly activates.

Phase 1 (existing-policy oracle from PLAN27): Best global policy is `uniform_mult2` (mean gap 0.0343%, runtime -14.3%). Best family-aware: hardA=uniform_mult2, hardB=ambig_scoreband_mult2 (mean gap 0.0318% vs baseline 0.0345%). Per-row oracle upper bound: 0.0304%.

Phase 2 (family-aware survivor selection): Both `family_aware_ambig` and `family_aware_late` pass with 6/8 improved, 7/8 not worse. Mean gaps: 0.0318% and 0.0309% vs baseline 0.0345%. hardB_k10 seed 3 is universally worse (all policies degrade on this seed). **Gate PASSES.**

Phase 3 (fine-block coarse-lookahead): Smoke test on hardB_k10 s0 shows worse gap (0.0417% vs 0.0391%). Not promoted.

**Decision A**: Promote family-aware survivor selection for hard K=10 beam rows. hardA_k10: `uniform_mult2`. hardB_k10: `ambig_scoreband_mult2` (or `late_ambig`). Global fallback: `uniform_mult2` (PLAN27 A).

Code changes:
- `solvers/cpp/stateful_dp_solver.cpp`: Fixed residual-aware diag copy; added `fine_plus_coarse_lookahead` score policy with precomputed block targets and price volatility
- `research/k_vs_arithmetic_axes_20260412/run_plan31_family_aware_survivor.py`: Phase 2 runner

Artifacts:
- `csv/plan31/PLAN31_existing_policy_oracle.csv`
- `csv/plan31/PLAN31_existing_policy_oracle_notes.md`
- `csv/plan31/PLAN31_family_aware_survivor_raw.csv` (49 rows from PLAN27)
- `csv/plan31/PLAN31_family_aware_survivor_compare.csv`
- `csv/plan31/PLAN31_family_aware_survivor_summary.csv`
- `csv/plan31/PLAN31_fine_block_guided_beam_notes.md`

## 2026-04-28

### PLAN27 completed — Step-3 adaptive survivor policy

Ran Gate A (hardA_k10/hardB_k10 seeds 0-3, n=1000, lambda=1.3) for six variants:
standard_beam, uniform_mult2, ambig_scoreband_mult2, late_ambig, residual_aware, late_residual_ambig.

- uniform_mult2 passes promotion (6/8 not worse, mean gap 0.0343% vs 0.0345%, runtime -14.3%).
- late_ambig and late_residual_ambig fail not-worse threshold (5/8) but show real signal.
- residual_aware shows zero gap effect; blocked by unresolved env-var/read issue.
- Decision A with family-dependence caveat.

Artifacts: csv/plan27/PLAN27_step3_adaptive_survivor_*.csv

## 2026-04-28

### PLAN30 completed — Easy-vs-hard fixed-n K-scaling story (implements PLAN_16)

Ran easy contiguous-unit families `K=24`, `K=30`, and `K=40` at fixed `n=1000`, `lambda=1.3`, seeds 0,1. Baseline and dense Step-2 fastpath variants.

- All 12 rows exact (0% gap), all memory-safe (peak RSS 1.7–4.8 GB, no kills, no timeouts).
- K=24: mean runtime 364s; K=30: mean runtime 683s; K=40: mean runtime 1552s.
- All rows close at Step 2 (`ffd`).
- Sharpens the two-axis claim: easy families scale to K=40+ while hard irregular degrades around K=8-10.

Decision: **A** — K-scaling story is sufficiently documented.

Artifacts: csv/plan30/PLAN30_easy_k_scaling_*.csv

## 2026-04-28

### PLAN29 initialized — Multi-view adjacent block reconstruction

Created a new active iteration for a bounded Step-1/Step-3 reconstruction
experiment:

- `iterations/20260428_multiview_block_reconstruction/`
- `implementation_plans/PLAN_29_multiview_block_reconstruction.md`

Motivation:

- PLAN28 local block-realizability diagnostics failed because block-local
  evaluation is universally stricter than the beam's global validation;
- the next block-structure experiment should not ask whether each block is
  independently feasible;
- PLAN29 instead tests whether alternative adjacent coarsenings of the recovered
  block partition improve the Step-3 beam incumbent on hard irregular K10 rows.

No PLAN29 experiments have been run yet.

### PLAN29 Phase A completed — Decision C

Ran PLAN29 Phase A multi-view block reconstruction gate on 56 rows (hardA_k10 + hardB_k10, seeds 0-3, 7 variants: baseline, coarsen2, coarsen3, target_B12, target_B8, price_preserve_B12, arith_adaptive). All rows memory-safe (peak RSS well under 16 GB).

**Result: Gate A fails. Decision C.** No single coarsening view improves ≥ 4/8 K10 anchor rows. Best is 3/8 (target_B12, target_B8, price_preserve_B12).

Key findings:
- Coarsening universally degrades on hardA_k10 (same or worse on all 4 seeds)
- `target_B8` improves hardB s0,s1,s2 (0.0294%, 0.0377%, 0.0448%) but worsens s3 and all hardA
- `target_B12` improves 3 rows (hardA_s2, hardB_s1, hardB_s2) but worsens 4 rows
- `coarsen3` is universally worse (mean gap +0.014% — doubled)
- `arith_adaptive` is a no-op on hardA (threshold never triggers), mixed on hardB
- Coarsening fails because wider blocks lose price-profile fidelity and explode pattern counts

Code changes:
- `solvers/cpp/stateful_dp_solver.hpp`: Added `block_view_*` diagnostic fields
- `solvers/cpp/stateful_dp_solver.cpp`: Added 6 coarsening policies in `pack_recovered_blocks` (`PAST_BLOCK_VIEW_POLICY`)
- `solvers/cpp/stateful_compare.cpp`: Added CSV header and data fields for block-view diagnostics
- `research/k_vs_arithmetic_axes_20260412/run_plan29_multiview_blocks.py`: New run script

Artifacts:
- `csv/plan29/PLAN29_multiview_block_reconstruction_raw.csv` (56 rows)
- `csv/plan29/PLAN29_multiview_block_reconstruction_compare.csv`
- `csv/plan29/PLAN29_multiview_block_reconstruction_summary.csv`
- `csv/plan29/PLAN29_multiview_block_reconstruction_notes.md`

### PLAN28 initialized — Realizability-aware recovered blocks

Created a new active iteration for block-realizability diagnostics and bounded
repair:

- `iterations/20260428_realizability_aware_blocks/`
- `implementation_plans/PLAN_28_block_realizability_diagnostics_and_repair.md`

Motivation:

- old `smart_reconstruct(...)` was global count-state reconstruction over the
  relaxed DP table and is not the right scalable path for hard `K=10/12`;
- PLAN26 showed a more local failure: the beam path can be globally feasible
  while failing strict block-local validation;
- the next experiment therefore diagnoses recovered blocks directly, then only
  applies small local repairs if the diagnostics separate easy rows from hard
  irregular rows.

No PLAN28 experiments have been run yet.

### PLAN28 Phase A completed — Decision C

Ran PLAN28 Phase A block-realizability diagnostic gate on 18 rows (9 families × 2 seeds, `n=1000`, `lambda=1.3`). All rows memory-safe (peak RSS 0.6–5.7 GB).

**Result: Gate A fails. Decision C.**

Key findings:
- `base_path_survives=0` for ALL 17 rows with beam incumbents — beam's chosen counts are never locally feasible at block 0, universally (easy and hard families alike).
- `bad_rate` overlaps between easy (50–56%) and hard families (50–83%), with hardA_k10 at 50% (same as easy) yet gap=0.009–0.017%.
- `finite_patterns` (mean) is K-dependent (36–56 from K=8 to K=12), not family-dependent.
- Easy families close at Step 2 (FFD/FFI) despite 50%+ bad blocks; hard families cannot close without the beam.
- 1 row (hardB_k12 seed 0) produced no incumbent.

The diagnostics fail because block-local evaluation (`evaluate_profile_block_counts`) is universally stricter than the beam's global validation. The beam's blocks are always locally irreparable at block 0 — this is a structural property, not a signal that separates easy from hard. Easy families bypass this by closing at Step 2.

This direction is stopped. No Phase B or Phase C will be run.

Code changes:
- `solvers/cpp/stateful_dp_solver.hpp`: Added `block_realiz_*` diagnostic fields to `RelaxedDPResult`
- `solvers/cpp/stateful_dp_solver.cpp`: Added diagnostic computation in `pack_recovered_blocks`, triggered by `PAST_BLOCK_REALIZ_DIAG=1`
- `solvers/cpp/stateful_compare.cpp`: Added CSV header and data fields for diagnostics
- `research/k_vs_arithmetic_axes_20260412/run_plan28_block_realizability.py`: New run script

Artifacts:
- `csv/plan28/PLAN28_block_realizability_diagnostics_raw.csv` (18 rows)
- `csv/plan28/PLAN28_block_realizability_diagnostics_summary.csv`
- `csv/plan28/PLAN28_block_realizability_notes.md`

## 2026-04-27

### PLAN26 — Validate PLAN25 local corridor + multi-idea improvement queue

Phase 0 (correctness repairs):
- Fixed `lb=ub` bug in `stateful_compare.cpp`: local corridor UB improvements no longer incorrectly claim global optimality.
- Added local-corridor alignment diagnostics.
- Fixed `merged_blocks` propagation: populated in `RecoveredBlockPackingResult`, propagated to `RelaxedDPResult` in all three solver paths. `stateful_compare.cpp` now uses `fwd.merged_blocks` directly.
- Added base-path survival simulation in `beam_corridor_local_dp`.

Phase 1 (validate base-path survival):
- hardA_k10 s0, hardB_k10 s2 with delta=1,2.
- All local corridor rows: `base_path_survives=0`, `base_candidate_not_found_at_layer_0`.
- Root cause: `evaluate_profile_block_counts` returns `kInf` for base beam candidate. Beam patterns are generated by work capacity, not schedulability. Block-local evaluation is too strict.
- Note: planned `hardB_k10 s2 local_corridor_delta2_300s` is missing from the raw artifact; Phase 1 has 3 local-corridor rows, not 4.

Phase 2 & 3: Cancelled. Blocked by Phase 1 finding.

Phase 4 (Step-3 beam scoring variants, partial):
- Ran `standard_step4` and `ambig_scoreband_mult2` on hardA_k10 seeds 0,2 and hardB_k10 seeds 0,2.
- `ambig_scoreband_mult2` improved gap on hardB (2/2) but worsened on hardA (2/2).
- Promotion check fails: only 2/4 rows not worse (needs 3/4).
- Requested new policies `residual_aware` and `late_ambig` were not implemented in this pass.

Decision: **C** — local corridor invalid due block/path mismatch.

Artifacts:
- `csv/plan26/PLAN26_multi_idea_raw.csv`
- `csv/plan26/PLAN26_multi_idea_compare.csv`
- `csv/plan26/PLAN26_multi_idea_notes.md`
- `implementation_plans/PLAN26_beam_corridor_multi_idea_queue.md`

## 2026-04-26

### PLAN24B — Forced-entry corridor exact DP diagnostic

Follow-up to PLAN24: tested whether corridor exact DP can actually enter the search and prune states when the theoretical guardrail is bypassed.

Code changes:
- `solvers/cpp/stateful_dp_solver.hpp`: Added `stop_reason` field to `ExactDPDiagnostics`.
- `solvers/cpp/stateful_dp_solver.cpp`: Added `PAST_EXACT_CORRIDOR_FORCE_ENTRY` env var (off by default). When set and corridor is active, bypasses `sparse_skip_theoretical` guardrail. Clamps internal time limit to `PAST_EXACT_CORRIDOR_TIME_LIMIT`. State limit guarded by `PAST_EXACT_CORRIDOR_MAX_STATES` (default 50M). Sets `stop_reason` at each exit.
- `solvers/cpp/stateful_compare.cpp`: Added `env_int64_exact` helper. Emits `corridor_force_entry`, `corridor_max_states`, `corridor_time_limit`, `stop_reason` in CSV.
- Runner: `research/k_vs_arithmetic_axes_20260412/run_plan24b_forced_corridor.py` (new).

Target rows (diagnostic only):
- hardA_k10 seed=0
- hardB_k10 seed=2
- Variants: standard_step4, forced_corridor_delta1_300s, forced_corridor_delta2_300s
- Overall time limit 1200s, internal corridor time limit 300s
- K=12 not run, not all seeds, not delta3

Results:
- All rows valid, reach step4, beam incumbent present.
- Standard rows: `sparse_skip_theoretical` (as in PLAN24).
- Forced rows: **`sparse_skip_overflow`** — the int64 mixed-radix encoding overflows. Product of (totals[i] + 1) for K=10 at n=1000 exceeds int64 range.
- Zero corridor pruning (`exact_diag_corridor_pruned=0`) because zero states were generated.
- Identical UB/LB/gap to standard on all rows.
- Runtime identical (~490-680s, dominated by beam Step 3). Memory safe (max ~7.7 GB).
- Force-entry correctly bypasses theoretical guardrail but hits encoding overflow immediately.

Artifacts produced:
- `research/k_vs_arithmetic_axes_20260412/csv/plan24b/PLAN24B_forced_corridor_raw.csv` (6 rows)
- `research/k_vs_arithmetic_axes_20260412/csv/plan24b/PLAN24B_forced_corridor_compare.csv` (2 rows)
- `research/k_vs_arithmetic_axes_20260412/csv/plan24b/PLAN24B_forced_corridor_notes.md`

Conclusion:
- **Decision: D** — Corridor still cannot enter meaningfully; abandon corridor under current exact DP. The blocking issue is the int64 mixed-radix encoding overflow, not the theoretical bound guardrail. The sparse exact DP encoding is fundamentally limited to ~K=8 at n=1000 on hard irregular families. No amount of guardrail relaxation or corridor tuning can overcome this.

### PLAN25 — Local corridor exact DP (offset encoding)

Implemented and smoke-tested a local-offset exact DP around the beam prefix to avoid the global mixed-radix int64 overflow that killed PLAN24B.

Code changes:
- `solvers/cpp/stateful_dp_solver.hpp`: Added `LocalCorridorDiag` struct and `beam_corridor_local_dp()` declaration.
- `solvers/cpp/stateful_dp_solver.cpp`: Implemented `beam_corridor_local_dp()`: builds block-local views, generates perturbed candidate count vectors per block, runs layered offset DP with `(2*delta+1)^K` encoding, hard state cap (5M), time cap (300s).
- `solvers/cpp/stateful_compare.cpp`: Wired `PAST_BEAM_CORRIDOR_LOCAL_DP` call after dense exact DP fallback; emits 14 new local-corridor CSV columns.
- Runner: `research/k_vs_arithmetic_axes_20260412/run_plan25_local_corridor.py` (new).

Variants run:
- `standard_step4`
- `local_corridor_delta1_300s`
- `local_corridor_delta2_300s`

Target rows:
- hardA_k10 seed=0
- hardB_k10 seed=2

Results:
- All rows valid, memory safe (peak RSS 4–8 GB).
- Local corridor runs successfully: delta1 ~25s, delta2 ~52–72s.
- Status consistently `infeasible_corridor`.
- State counts scale as expected: delta1 ~40–60k states_seen; delta2 ~7.6–14.1M states_seen.
- `best_ub = inf` on all local corridor rows; incumbent unchanged.
- Exact sparse DP still `sparse_skip_theoretical` on all rows.
- Important correction after review: the `infeasible_corridor` interpretation is not final. Because the beam base count vector is inserted as a candidate for each block, PLAN25 must still verify base-path survival and block/count alignment before claiming the corridor contains no useful completion.

Artifacts produced:
- `research/k_vs_arithmetic_axes_20260412/csv/plan25/PLAN25_local_corridor_dp_raw.csv` (6 rows)
- `research/k_vs_arithmetic_axes_20260412/csv/plan25/PLAN25_local_corridor_dp_compare.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan25/PLAN25_local_corridor_dp_notes.md`

Conclusion:
- **Decision: diagnostic hold** — Local offset encoding avoids int64 overflow, but the local-corridor method is not yet mechanically validated. Keep disabled by default. Next step must fix the corridor-limited `lb=ub` proof handling and add diagnostics for base-path survival, block alignment, and candidate rejection reasons.

### PLAN24 — Beam-guided Step-4 exact corridor evaluation

Tested whether restricting Step 4 exact DP to count vectors near the Step 3 beam's prefix-count trajectory improves exact closure, gap, or runtime on hard irregular K=10 rows.

Code changes:
- `solvers/cpp/stateful_dp_solver.hpp`: Added `ExactCorridor` struct + `set_exact_corridor()`/`clear_exact_corridor()`, corridor fields in `ExactDPDiagnostics`, beam chosen counts + block order fields in `RelaxedDPResult`.
- `solvers/cpp/stateful_dp_solver.cpp`: Global corridor instance, `check_exact_corridor_counts()` and `check_exact_corridor_sparse()` helpers, corridor pruning in dense and sparse exact DP, beam chosen counts plumbing through `block_repair_profile_repair_beam_ub` / `pack_recovered_blocks` / `solve_relaxed_dp_with_binpack`. Fixed missing `g_last_exact_dp_diag.corridor_*` initialization in dense and sparse exact DP entry points.
- `solvers/cpp/stateful_compare.cpp`: Corridor construction before exact DP from `fwd.profile_beam_chosen_counts`, corridor diagnostics in CSV header and data.
- Runner: `research/k_vs_arithmetic_axes_20260412/run_plan24_beam_corridor_exact.py` (new).

Critical fix:
- Initial smoke (11 rows) incorrectly used `PAST_RELAXED_BINPACK_SOLVER=energy_core`, producing no beam incumbent (ub=-1). Invalid rows preserved in `csv/plan24/PLAN24_invalid_energy_core_misroute_raw.csv`.
- Corrected to `PAST_RELAXED_BINPACK_SOLVER=profile_repair_beam` with `PAST_PROFILE_REALIZATION_SELECTOR_POLICY=auto_v1`.
- Added route sanity check in runner: detects `fwd_pack_method=none` or `ub=-1` and classifies as `misrouted_or_no_beam_incumbent`.

Phase A smoke (hardA_k10 seed=0):
- standard_step4, corridor_delta0, corridor_delta1, corridor_delta2 all valid.
- All reached step4, `fwd_pack_method=profile_repair_beam`, `beam_status=feasible`.
- Corridor diagnostics populated correctly (enabled=1 with correct deltas for corridor variants).
- Memory well under cap (max ~5.9 GB).

Phase B (hardA_k10/hardB_k10 seeds 0-3):
- 32 rows: standard_step4 + corridor_delta1 + corridor_delta2 + corridor_widen_0_1_2.
- All rows valid, reached step4, beam incumbent present.
- Zero corridor pruning: `exact_diag_corridor_pruned=0` for all 24 corridor rows.
- Sparse exact DP skipped on all rows (`sparse_skip_theoretical`).
- Identical UB/LB/gap between standard and all corridor variants.
- No exact closure (`is_optimal=0` for all rows).

K=10 signal evaluation:
- exact=False, better_gap=False, faster=False, fewer_states=False.
- Phase C (K=12 probe) skipped: no signal.

Artifacts produced:
- `research/k_vs_arithmetic_axes_20260412/csv/plan24/PLAN24_beam_corridor_exact_raw.csv` (33 valid rows)
- `research/k_vs_arithmetic_axes_20260412/csv/plan24/PLAN24_invalid_energy_core_misroute_raw.csv` (11 invalid rows)
- `research/k_vs_arithmetic_axes_20260412/csv/plan24/PLAN24_beam_corridor_exact_compare.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan24/PLAN24_beam_corridor_exact_summary.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan24/PLAN24_beam_corridor_exact_notes.md`

Conclusion:
- **Decision: D** — No evidence beam-guided exact corridor helps on hard irregular K=10 rows. Sparse exact DP skips the search, so corridor pruning never has an opportunity to reduce state space. The blocking issue is the theoretical bound guardrail, not state-space management.

## 2026-04-25

### PLAN23 — Role-based survivor policy evaluation (Gate 1 only)

Tested whether role-based node representatives (best score, best local, best arith, optionally best feas) are more stable than standard beam, uniform multiplicity, and ambig_scoreband_mult2.

Code changes:
- `solvers/cpp/stateful_dp_solver.cpp`: added `role` policy to `block_repair_feasible_beam_ub` controlled by env vars:
  - `PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_POLICY=role`
  - `PAST_PROFILE_REPAIR_BEAM_ROLE_MAX`
  - `PAST_PROFILE_REPAIR_BEAM_ROLE_SCORE_BAND`
  - `PAST_PROFILE_REPAIR_BEAM_ROLE_KEEP_FEAS`

Artifacts produced:
- `research/k_vs_arithmetic_axes_20260412/csv/plan23/PLAN23_role_based_beam_raw.csv` (66 rows, including baselines)
- `research/k_vs_arithmetic_axes_20260412/csv/plan23/PLAN23_role_based_beam_compare.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan23/PLAN23_role_based_beam_summary.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan23/PLAN23_role_based_beam_notes.md`

Gate 1 rows tested:
- hardA_k10 seeds 0,1,2
- hardB_k10 seeds 0,2

Gate 1 variants:
- standard_beam, uniform_mult2, ambig_scoreband_mult2, role_mult3, role_mult3_feas

Gate 1 results:
- `role_mult3`: wins=1, losses=1, ties=3 vs standard; improved gap on 1/5 rows; mean runtime increase +62.7%.
- `role_mult3_feas`: wins=1, losses=1, ties=3 vs standard; improved gap on 1/5 rows; mean runtime increase +55.5%.

Key findings:
- Role-based representatives did not improve gap over standard or uniform on any Gate 1 row.
- On hardB_k10 s2, role matched uniform (0.044% vs standard 0.045%) — the only win.
- On hardA_k10 s1, role was worse (0.0283% vs standard 0.0272%) — the only loss.
- Runtime increased substantially because role selection generates more candidates per key and the beam search takes longer.
- All finite-gap rows have `beam_status=feasible` and `deciding_step=step4`; Step 4 exact DP does not close gaps.

Conclusion:
- Gate 1 FAILED.
- **Decision: E** — No survivor-policy change is validated; move next to beam-guided Step 4 certification.

## 2026-04-25

### PLAN22B correction pass — validate ambig_scoreband_mult2 on Gate 2

Ran the missing Gate 2 validation rows for `ambig_scoreband_mult2` that PLAN22 did not execute.

Missing rows executed:
- hardA_k10 seeds 2,3
- hardB_k10 seeds 0,1,2,3
- hardA_k12 seeds 0,1
- hardB_k12 seeds 0,1

Artifacts produced:
- `research/k_vs_arithmetic_axes_20260412/csv/plan22b/PLAN22B_ambig_scoreband_validation_raw.csv` (64 rows, including PLAN22 baseline)
- `research/k_vs_arithmetic_axes_20260412/csv/plan22b/PLAN22B_ambig_scoreband_validation_compare.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan22b/PLAN22B_ambig_scoreband_validation_summary.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan22b/PLAN22B_ambig_scoreband_validation_notes.md`

Key findings:
- `ambig_scoreband_mult2` does NOT clearly generalize beyond Gate 1 (4 wins, 5 losses vs standard on Gate 2).
- Mean gap across all 14 rows: 0.0357% (standard 0.0355%, uniform 0.0353%).
- Best single result remains hardA_k10 s=0: 0.0172% -> 0.0094%.
- Mixed on hardB_k10 (2 wins, 2 losses).
- Maintains K=12 incumbent production (3/4 finite gaps) but does not improve over baselines.

Conclusion:
- PLAN22's Decision B (promote ambig_scoreband_mult2 globally) is corrected.
- **Decision E**: Use `ambig_scoreband_mult2` only as a K=10 quality-improvement candidate, not as a global policy.

## 2026-04-25

### PLAN22 adaptive node evaluation / survivor policy for profile_repair_beam

Implemented and tested adaptive multiplicity policies inside the existing beam dedup stage.

Code changes:
- `solvers/cpp/stateful_dp_solver.cpp`: added `s_center` and `s_arith` to `FeasBeamNode`; implemented `uniform`, `early`, `ambig_scoreband`, `hybrid` policies controlled by env vars.
- `solvers/cpp/stateful_dp_solver.hpp`: added `profile_beam_key_multi_*` fields to `RelaxedDPResult`.
- `solvers/cpp/stateful_compare.cpp`: emitted new policy fields in CSV output.

Artifacts produced:
- `research/k_vs_arithmetic_axes_20260412/csv/plan22/PLAN22_adaptive_node_eval_raw.csv` (54 rows)
- `research/k_vs_arithmetic_axes_20260412/csv/plan22/PLAN22_adaptive_node_eval_compare.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan22/PLAN22_adaptive_node_eval_summary.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan22/PLAN22_adaptive_node_eval_notes.md`

Gate 1 result (4 anchor rows × 6 variants):
- `ambig_scoreband_mult2`: not worse on 3/4, improved gap on 3/4 (best: hardA_k10 s=0 0.0172% → 0.0094%).
- `hybrid_mult2`: not worse on 2/4, failed Gate 1.
- `early_mult2`: not worse on 3/4, improved runtime on 3/4 but never improved gap.
- `uniform_mult2`: not worse on 2/4, failed Gate 1.
- `uniform_mult3_control`: not worse on 2/4, failed Gate 1.

Gate 2 result (ran `standard_beam`, `early_mult2`, `uniform_mult2` on 12 additional rows):
- `early_mult2` mean gap 0.0360% vs standard 0.0355% — slightly worse on average, mostly runtime wins.
- `uniform_mult2` mean gap 0.0353% — similar to standard.

Conclusion:
- Adaptive filtering (`ambig_scoreband`) produced the only material gap improvements and directly addresses PLAN20B seed-dependence.
- `early_mult2` is safer but only improves runtime, not incumbent quality.
- Decision: **B** — promote `ambig_scoreband_mult2` as the next main candidate.

## 2026-04-24

### PLAN19 K=10/12 redesign completion (`n=1000`, `lambda=1.3`, seeds `0/1`)

Executed bounded additive redesigns for hard irregular K=10/12 at fixed n=1000 to test whether exact closure could be recovered after beam incumbent production.

Redesigns tested:
1. beam -> restricted exact closure (C++ hook `PAST_PROFILE_REALIZATION_EXACT_AFTER_BEAM_ENABLE=1`)
2. irregular high-K routing override (skip baseline energy_core)
3. stronger K=12 beam (`PAST_EXACT_INCUMBENT_SOURCE=i3`)

Artifacts produced:
- raw: `research/k_vs_arithmetic_axes_20260412/csv/plan19/PLAN19_k10_k12_redesign_raw.csv` (67 rows)
- best-of-variant: `research/k_vs_arithmetic_axes_20260412/csv/plan19/PLAN19_k10_k12_best_variant_summary.csv`
- compare: `research/k_vs_arithmetic_axes_20260412/csv/plan19/PLAN19_k10_k12_redesign_compare.csv`
- failure shift: `research/k_vs_arithmetic_axes_20260412/csv/plan19/PLAN19_k10_k12_failure_shift.csv`
- method notes: `research/k_vs_arithmetic_axes_20260412/csv/plan19/PLAN19_k10_k12_method_notes.md`
- diagnosis: `research/k_vs_arithmetic_axes_20260412/csv/plan19/PLAN19_k10_k12_diagnosis.md`

Key findings:
- exact_after_beam hook did not visibly trigger; rows still show `selector_decision=beam` and `block_dp_status=skipped_selector`
- force_exact with guardrails raised to 1e12 immediately hits `skipped_comp_est`, confirming exact fixed-block DP comp_est is astronomically large for K=10/12 irregular rows (B≈20)
- beam_plus on K=12 timed out on 6/8 seeds with no incumbent; on 2 seeds where it produced an incumbent, gaps matched standard reroute with longer runtime
- no exact rows recovered across all 67 rows
- routing override is justified: baseline energy_core consistently wastes 500-1200s with no incumbent

Memory behavior: all variants stayed within 12GB cap; no memory kills.

Conclusion: exact closure at K=10/12 on hard irregular families is infeasible under current fixed-block-DP budgets. The practical ceiling is beam incumbent + Step 4 global exact DP, leaving small finite gaps (~0.02-0.06%).

## 2026-04-24

### PLAN18 K-boundary refinement completion (`n=1000`, `lambda=1.3`, seeds `0/1/2/3`)

Resumed PLAN18 from partial raw CSV (47 rows) to full completion (48 rows).

Memory-safety patch applied first:
- patched `run_plan13_two_track_recovery.py` `run_row()` to avoid reading full stdout/stderr temp files into Python heap;
- now reads only a trailing 1 MB window for stdout CSV parsing and an 8 KB tail for stderr;
- lowered default RSS cap from 16 GB to 12 GB as a conservative host-safety margin;
- fixed stderr-read bug (was accidentally re-reading stdout path).

Rows rerun:
- `hardB_k12 / irregular_reroute / seed=1` (previous row had `returncode=-15`, `deciding_step=no_csv_row`; deemed unreliable)
- `hardB_k12 / irregular_reroute / seed=3` (missing row)

Both reruns ended as clean external-timeout rows (`rc=-9`, `external_timed_out=1`, `ub=-1`, `lb=-1`, runtime=1200s), confirming they hit the per-row budget without emitting an incumbent.

Derived artifacts regenerated with corrected best-of-route scoring (rows without a valid incumbent are no longer preferred over finite-gap rows because of a spurious `gap_pct=0.0`):
- `csv/plan18/PLAN18_k_boundary_refine_n1000_raw.csv` — 48 data rows, complete
- `csv/plan18/PLAN18_k_boundary_refine_best_of_route.csv` — 24 best-of-route rows
- `csv/plan18/PLAN18_k_boundary_refine_summary_by_k.csv` — by-K summary
- `csv/plan18/PLAN18_k_boundary_refine_failure_signatures.csv` — failure-mode table

Headline boundary picture from the completed PLAN18:
- `K=8`: mixed exact vs finite-gap (2/4 exact on hardA, 2/4 exact on hardB);
- `K=10`: no exact rows; finite-gap incumbents dominate, with one seed timing out in Step 3;
- `K=12`: no exact rows; mostly timeout/no-incumbent, with only occasional finite-gap incumbents.

So the refined practical boundary is:
- exactness drops between K=8 and K=10;
- K=10 is the last K where finite-gap incumbents are still usually produced;
- K=12 is mostly budget-limited under the current 1200s/12GB cap.

### PLAN17 audit correction

Corrected the PLAN17 derived layer without discarding the raw executed rows.

- fixed raw `external_timeout` field to be row-level rather than the constant watchdog value;
- corrected boundary classification so exact `energy_core` Step-3 rows are not mislabeled as `step3_beam_exact`;
- rebuilt `summary_by_family` and `summary_by_k` as variant-separated summaries instead of mixing baseline and reroute rows in one denominator;
- softened the paper-facing wording from "`K>=12` timeout-dominated" to "mostly budget-limited" to match the actual raw evidence.

### PLAN17 fixed-n K-axis boundary study (`n=1000`, `lambda=1.3`, seeds `0/1`)

Completed PLAN17 supervisor-priority K-axis campaign with three controlled ladders and explicit route policy per row.

Artifacts produced (plan17):

- `research/k_vs_arithmetic_axes_20260412/csv/plan17/PLAN17_k_axis_n1000_raw.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan17/PLAN17_k_axis_n1000_summary_by_family.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan17/PLAN17_k_axis_n1000_summary_by_k.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan17/PLAN17_k_axis_boundary_classification.csv`

Runner/code updates:

- `research/k_vs_arithmetic_axes_20260412/run_plan17_k_axis_n1000.py` (new)
- `research/k_vs_arithmetic_axes_20260412/run_plan13_two_track_recovery.py` (non-breaking: `run_row` now accepts optional payload override)

Scope delivered:

- easy contiguous/unit ladder: `K=2,4,6,8,10,12,16,20`
- hard irregular A ladder: `K=4,6,8,10,12,16,20`
- hard irregular B ladder: `K=4,6,8,10,12,16,20`
- memory-safe protocol kept (`16 GB` cap, one heavy row at a time)
- external timeout set to `900s` per row (`1020s` external watchdog)

Routing policy compliance:

- K=2 rows used `profile_repair_beam + auto_v1`; no K=2 energy-core misroute in final raw table.
- easy `K>=8` rows include both baseline and dense Step-2 fastpath variants, explicitly labeled.
- irregular `K=4` rows used accepted energy-core direct package route.
- irregular `K>=6` rows started with baseline and used one explicit additive reroute variant only when needed.

Headline outcome:

- easy ladder: exact Step-2 closure through `K=20` on both seeds (with fastpath runtime gains for `K>=8`), showing K alone is not the hardness axis.
- hard ladders: `K=6` still exact, first consistent degradation appears at `K=8` (finite-gap / unresolved on one seed), and `K>=12` is timeout-dominated under current budget.

Boundary conclusion at this checkpoint:

- first hard-K boundary at fixed `n=1000` appears around `K≈8` for irregular arithmetic families, not on the easy unit-contiguous ladder.


## 2026-04-22

### PLAN_14 dense-unit large-K checkpoint (`g12345678910 = {1..10}`)

Completed PLAN14 diagnosis + additive fast-path experiment for dense unit-containing
large-K paper group.

Artifacts produced (plan14):

- `research/k_vs_arithmetic_axes_20260412/csv/plan14/PLAN14_g12345678910_diagnosis.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan14/PLAN14_g12345678910_checkpoint_probe.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan14/PLAN14_g12345678910_fastpath_compare.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan14/PLAN14_dense_unit_1_20_smoke.csv`

Code path updates (additive / toggle-gated):

- `solvers/cpp/stateful_dp_solver.cpp`
- `solvers/cpp/stateful_dp_solver.hpp`
- `solvers/cpp/stateful_compare.cpp`
- `research/k_vs_arithmetic_axes_20260412/run_plan13_two_track_recovery.py`
  (repurposed as PLAN14 runner scaffold; plan14 outputs)

PLAN14 diagnosis findings:

- baseline `{1..10}` exact control still closes:
  - `n=3500, seed=0`: Step 2 exact (`ffd`), ~`504s`.
- baseline transition rows:
  - `n=4500, seed=0`: external timeout (`1200s` window).
  - `n=5000, seed=0/1`: external timeout (`1200s` window), no emitted incumbent.
- failure rows now include explicit failure stage + peak RSS in plan14 CSV.

Checkpoint probe findings:

- timeout probes (`n=5000`, seeds `0/1`, `time_limit=900s`) record
  `external_timeout` with stage metadata.
- forced tight-memory probes record `memory_limit_kill` with peak RSS.
- additive fastpath control rows in same artifact show finite `UB/LB` when closure is reached.

Additive fast-path result (`PAST_DENSE_UNIT_STEP2_FASTPATH=1`):

- trigger: dense unit-containing contiguous family with `1` present and `K>=8`.
- `{1..10}` exact closure recovered at `n=5000`:
  - seed 0: exact Step 2, `UB=LB=259936545`, runtime ~`840.5s`.
  - seed 1: exact Step 2, `UB=LB=260947838`, runtime ~`741.3s`.
- control preservation:
  - `n=3500` remains exact; fastpath variant is faster than baseline control in this pass.

Count-based FFD experiment (`PAST_COUNT_BASED_FFD=1`, additive):

- `n=5000`, seeds `0/1`: exact in both rows with same `UB/LB` as fastpath-ffd.
- runtime was slightly slower than fastpath-ffd in this run window, but exact and valid.

`{1..20}` smoke status:

- smoke artifact written with explicit skipped rows (`n=1000`, `n=2000`).
- reason recorded: run harness family map currently includes paper groups only; `{1..20}` family id wiring is pending.
- next step toward `{1..20}`: add explicit family wiring in payload/group map, then rerun smoke.

Blocker classification after PLAN14:

- root blocker for baseline `{1..10} n=5000` was runtime-window termination in the
  generic pipeline, not intrinsic Step-2 impossibility.
- additive dense-unit Step-2 fast-path resolves immediate `{1..10} n=5000` closure target (seeds `0/1`) and clarifies path toward `{1..20}`.

## 2026-04-21

### PLAN_13 two-track recovery: `{1..10}` timeout persistence + `g37` K=2 reroute resolution

Executed the bounded two-track PLAN13 pass under strict scope control in this
thread.

Artifacts created:

- `research/k_vs_arithmetic_axes_20260412/csv/plan13/PLAN13_easyfamily_g12345678910.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan13/PLAN13_g37_k2_reroute.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan13/PLAN13_variant_compare.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan13/PLAN13_g37_k2_variant_compare.csv`

Track A (`g12345678910={1..10}`) result in this bounded pass:

- baseline energy-core route at `n=5000` timed out on seeds `0,1`.
- additive reroute and incumbent-source probes at `n=5000` (mainline Step-3
  route, and Step-2 incumbent `i0`) did not recover closure in the bounded
  memory-safe run window.
- no accepted baseline change; this remains an unresolved runtime wall for the
  easy-family recovery target.

Memory-safety enforcement for this rerun pass:

- one heavy instance at a time,
- hard process cap with RSS monitoring,
- peak memory recorded per row in PLAN13 CSV (`peak_rss_*`, `memory_killed`).

Observed Track A memory-safe behavior:

- baseline `n=5000` rows reached external time limit under cap,
- additive reroute probes hit memory-limit kill at bounded cap,
- no row exceeded the repository safety policy (hard cap <= `16 GB`).

Track B (`g37={3,7}`) reroute result:

- reran required rows `n=750,1000,1500,2500,3500,5000` under intended K=2
  mainline profile-realization path (`profile_repair_beam` + selector `auto_v1`).
- selector now reports `exact / k2_exact_default` (not `non_mainline_solver`).
- Step-3 mode is exact (`profile_realization_dp_exact`) and closes all tested
  rows at zero gap on seed `0`.
- seed `1` reruns for all recovered rows also close at zero gap.

Memory-safe note for `g37` reroute:

- primary reroute sweep used strict cap with per-row RSS logging,
- one `n=5000, seed=1` row was re-run once at a slightly relaxed cap (still
  below `16 GB`) to remove an artificial cap-kill and confirm true routing
  behavior,
- final archived reroute table shows exact closure on all required rows.

Interpretation recorded for documentation consistency:

- old `g37` blocker rows in plan05/plan11 were routed through non-mainline
  energy-core path (`selector_decision=not_applicable`,
  `selector_reason=non_mainline_solver`), so they were not evidence about true
  K=2 Step-3 exact capability.
- with correct K=2 reroute, `g37` closure to `n=5000` is recovered in this
  campaign.

## 2026-04-20

### Provenance row-id consistency correction

Corrected provenance-only row-id labeling in:

- `research/k_vs_arithmetic_axes_20260412/csv/CURRENT_METHOD_PROVENANCE.csv`

so `source_row_id` prefixes now consistently match `source_artifact_path`
(`plan05`, `plan10`, `plan11`, `plan02b`).

No solver code, experiment, or result-content changes were made.

## 2026-04-20

### PLAN_12 cleanup + current-facing index + method-provenance registry

Completed the PLAN12 organization/provenance pass for this thread without
changing solver behavior or running new experiments.

Created current-facing entrypoint:

- `research/k_vs_arithmetic_axes_20260412/CURRENT_RESULTS_INDEX.md`

Created structured provenance layer:

- `research/k_vs_arithmetic_axes_20260412/csv/CURRENT_METHOD_PROVENANCE.csv`
- `research/k_vs_arithmetic_axes_20260412/METHOD_PROVENANCE.md`

Created paper-facing compact note:

- `research/k_vs_arithmetic_axes_20260412/PAPER_RESULTS_READY.md`

Created change note:

- `research/k_vs_arithmetic_axes_20260412/archive_20260421/markdown/cleanup/CLEANUP_AND_PROVENANCE_20260420.md`

Provenance registry includes evidence classes:

- `current_accepted_benchmark` (current accepted paper/frontier rows),
- `historical_continuity` (PLAN10 K=4 generator-policy continuity gates),
- `archive_only` (non-mainline but retained exact-L2 diagnostic closures).

Concrete code entrypoints are mapped per row to avoid vague method labels,
including:

- workflow entry in `stateful_compare.cpp` (`step1_exact_guided`),
- `solve_relaxed_dp_with_binpack`,
- `compute_relaxed_completion_table`,
- `generate_energy_core_patterns`,
- `block_repair_energy_core_ub`,
- `block_repair_feasible_beam_ub`,
- `block_repair_profile_repair_beam_ub`,
- `profile_realization_dp_exact` candidate path,
- archive-only exact-L2 path `block_repair_exact_level2_ub`.

## 2026-04-20

### PLAN_11 paper-group frontier extension (group-by-group after K=4 generator fix)

Completed the next paper-group extension pass with strict scope control:

- kept accepted baseline package unchanged (`energy_core + direct + step1_exact_guided`),
- preserved K=4 generator-policy defaults from PLAN10,
- did not introduce non-paper families or broad solver redesign,
- additive experiment (if any) kept separate from baseline.

Phase A source cleanup completed first:

- deduped refreshed `g3567` hard rows in
  `research/k_vs_arithmetic_axes_20260412/csv/plan05/PAPER_GROUPS_PLAN05_n_extension.csv`
  by removing duplicate logical keys at `n=2500/3500/5000`, seeds `0/1`.

Phase B group-by-group extension runs completed (lambda `1.3`, seeds `0,1`):

- `g3567`: tested `n=6000,7000,8000`.
  - exact at `n=6000` (Step 3),
  - timeout at `n=7000`,
  - immediate `std::length_error` crash at `n=8000`.
- easy-scalable order pass:
  - `g24`: exact through `n=10000` (Step 2),
  - `g12357`: exact through `n=8000`, timeout at `n=10000`,
  - `g246810`: exact at `n=6000`, `std::length_error` crash at `n>=7000`,
  - `g12345678910`: timeout at `n=6000,7000` (and existing timeout at `n=5000`).
- diagnosed difficult families:
  - `g810`: `std::length_error` crash from `n=6000` onward,
  - `g37`: unresolved finite-gap behavior with Step-4 entry at `n=6000,7000`.

Phase C additive-only experiment executed:

- artifact:
  `research/k_vs_arithmetic_axes_20260412/csv/plan11/PAPER_GROUPS_PLAN11_variant_compare.csv`
- tested `g810` baseline vs `exp_g810_force_beam`
  (`PAST_PROFILE_REALIZATION_SELECTOR_POLICY=force_beam`) on `n=7000,8000`,
  seeds `0,1`.
- result: both variants fail identically (`std::length_error`), no promotion.

PLAN11 artifacts created:

- `research/k_vs_arithmetic_axes_20260412/csv/plan11/PAPER_GROUPS_PLAN11_n_extension.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan11/PAPER_GROUPS_PLAN11_variant_compare.csv`

Source-of-truth artifacts refreshed:

- `research/k_vs_arithmetic_axes_20260412/csv/plan05/PAPER_GROUPS_PLAN05_n_extension.csv`
- `research/k_vs_arithmetic_axes_20260412/PAPER_GROUPS_EXTENSION_SUMMARY.md`

## 2026-04-19

### PLAN_10 strict K=4 generator optimization + paper-group consolidation

Completed the required strict K=4 generator-policy pass without broadening scope
(no column generation, no K>4 redesign, no new Step-3 theory branch).

#### Phase A completed: forced DP-style generator at K=4

Created run driver:

- `research/k_vs_arithmetic_axes_20260412/run_plan10_k4_generator_compare.py`

Artifacts produced:

- `research/k_vs_arithmetic_axes_20260412/csv/plan10/PLAN10_k4_generator_dp4.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan10/PLAN10_k4_generator_compare.csv`

Tested on active exactness gate rows:

- continuity: `3567_plus n=3500,5000`, seeds `0,1`
- hard paper: `g3567 n=2500,3500,5000`, seeds `0,1`, `lambda=1.3`

Result:

- `dp4_generator` preserved exactness on all gate rows (`10/10` exact),
- runtime improved materially vs baseline on hard rows,
- `fwd_ec_time_pattern_generation` dropped dramatically.

#### Phase C completed: signature-dedup usefulness measured

Compared:

- `dp4_generator`
- `dp4_generator_dedup_off`

Measured outcome on active rows:

- identical generated/retained pattern totals row-wise,
- exactness unchanged (`10/10` exact),
- small runtime preference for dedup-off.

Decision: disable signature-dedup by default for K=4.

#### Final generator policy selected and implemented

Code change in `solvers/cpp/stateful_dp_solver.cpp`:

- `PAST_BLOCK_REPAIR_PATTERN_DP_K` default now resolves to `4` for `K=4`
  (non-K=4 keeps threshold `5`).
- `PAST_BLOCK_REPAIR_EC_SIGNATURE_DEDUP` default now resolves to `0` for `K=4`
  (non-K=4 remains enabled by default).

Rebuilt:

- `cmake --build solvers/cpp/build --target stateful_compare -j4`

#### Paper-group source-of-truth consolidation completed

Refreshed K=4 paper rows under current exact package into:

- `research/k_vs_arithmetic_axes_20260412/csv/plan05/PAPER_GROUPS_PLAN05_n_extension.csv`

Regenerated summary via plan05 runner (`--phase summaries`), updating:

- `research/k_vs_arithmetic_axes_20260412/PAPER_GROUPS_EXTENSION_SUMMARY.md`

Current summary now reflects `g3567` exact through `n=5000` (no stale finite-gap
K=4 boundary claim).

## 2026-04-19

### PLAN_10 Phase A/B1/B3 run (continuity-safe K=4 speedup attempt)

Executed PLAN_10 from the recovered continuity-safe K=4 package and enforced
memory-safe execution for all heavy rows.

Baseline package used:

- `PAST_RELAXED_BINPACK_SOLVER=energy_core`
- `PAST_BLOCK_REPAIR_COMPLETION_MODE=direct`
- `PAST_BLOCK_REPAIR_COMPLETION_DIRECT_MAX_CELLS=500000000`
- `PAST_BLOCK_REPAIR_EC_STRONGER_CENTER=0`
- `PAST_BLOCK_REPAIR_EC_DIVERSIFY=0`
- `PAST_BLOCK_REPAIR_EC_ADAPTIVE_DELTA=0`
- `PAST_BLOCK_REPAIR_EC_TWO_PHASE=0`
- `PAST_BLOCK_REPAIR_EG_STATE_KEEP=60000`

Memory-safety protocol used in all plan10 row runs:

- one heavy instance at a time,
- per-process RSS monitoring with kill threshold at `16.5 GB`,
- no memory-unsafe runs accepted as valid measurements.

Created baseline artifact:

- `research/k_vs_arithmetic_axes_20260412/csv/plan10/PLAN10_k4_speedup_baseline.csv`

Baseline scope (required rows):

- continuity: `3567_plus n=3500,5000`, seeds `0,1`
- hard paper rows: `g3567 n=2500,3500,5000`, seeds `0,1`, `lambda=1.3`

Baseline result summary:

- all required rows exact (`10/10`),
- all rows Step-3 decided (`diag_step3_decided=1`, `diag_step4_decided=0`),
- peak RSS stayed within memory budget (about `4.8 GB` to `9.6 GB`).

Implemented Phase-B same-output speedup pass in solver:

- file: `solvers/cpp/stateful_dp_solver.cpp`
- B1: replaced multiple full-sort+resize sites with bounded selection:
  - per-work DP buckets now use `nth_element` + prefix sort,
  - per-work DFS buckets now use `nth_element` + prefix sort,
  - final flat trimming now uses `nth_element` before sorting retained prefix.
- B3: phase-1 feasible beam now does partial selection (`nth_element`) to
  `layer_width` before sorting retained prefix, instead of sorting full layer.

Post-pass measurement artifact created:

- `research/k_vs_arithmetic_axes_20260412/csv/plan10/PLAN10_k4_speedup_after_pass1.csv`
- combined ablation view:
  - `research/k_vs_arithmetic_axes_20260412/csv/plan10/PLAN10_k4_speedup_ablation.csv`

Post-pass result summary:

- exactness remained intact on all required rows (`10/10` exact).
- however, runtime regressed overall:
  - required hard `g3567` rows mean runtime increased by about `+15.6%`
  - continuity rows mean runtime increased by about `+2.7%`
  - overall mean runtime increased by about `+13.0%`
- dominant time increase came from `fwd_ec_time_pattern_generation`.

Decision from this pass:

- keep continuity-safe baseline as the current recommended K=4 package,
- disqualify this exact-preserving B1/B3 pass as a speedup package,
- continue with more targeted early-stage changes only if they reduce runtime
  while preserving exactness and memory safety.

## 2026-04-17

### PLAN_08 Phase A-E fortification implementation and campaign run

Implemented and validated the in-flight PLAN_08 fortification edits in the
stateful solver, then ran the required campaign rows with auditable CSV/JSON
output.

Code paths updated:

- `solvers/cpp/stateful_dp_solver.hpp`
- `solvers/cpp/stateful_dp_solver.cpp`
- `solvers/cpp/stateful_compare.cpp`
- new campaign runner:
  `research/k_vs_arithmetic_axes_20260412/run_plan08_energy_core_campaign.py`

Core implementation completed:

- Phase A instrumentation fields (`fwd_ec_*`) exported in ablation CSV,
  including generated/retained pool size, retained signature, per-phase times,
  pruning counters, delta used, fixed blocks, and two-phase diagnostics.
- Phase B stronger center in `energy_core` via blended surrogate center
  (capacity-based + top-pattern weighted center).
- Phase C adaptive expansion via type/block-aware delta widening.
- Phase D reduction and robustness additions (pattern-signature dominance,
  fixed-block counting, richer retained-pool diagnostics).
- Phase E two-phase `energy_core` path retained (feasibility-first beam,
  exact-core polish gated by need).

Additional safety fix applied during validation:

- `compute_relaxed_completion_table` direct mode now has a bounded-cell guard
  (`PAST_BLOCK_REPAIR_COMPLETION_DIRECT_MAX_CELLS`, default `120000000`) and
  falls back to cheap completion when direct table size would be excessive.
- This removed process-kill failures (`rc=-9`) observed on large-horizon rows.

Build/validation:

- `cmake --build solvers/cpp/build --target stateful_compare -j4`
- build succeeded (warnings only).

Campaign artifacts:

- `research/k_vs_arithmetic_axes_20260412/csv/plan08/PLAN08_energy_core_campaign.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan08/PLAN08_energy_core_campaign.json`

Required row coverage executed:

- `g3567`, `n=1000,1500,2500,3500,5000`, seeds `0,1`, `lambda=1.3`.
- historical continuity checks on `3567_plus`: `n=3500,5000`, seeds `0,1`
  (nosby replay payload conversion).
- transfer checks:
  - `g12357` at `n=1000,1500,2500`, seeds `0,1`
  - `g246810` at `n=1000,1500,2500`, seeds `0,1`
- optional executed:
  - `g12345678910` at `n=1000,1500`, seeds `0,1`.

High-level observed outcomes:

- `g3567`: exact closure preserved at lower n (`1000,1500`), but large-n rows
  move to Step-4 finite-gap closure (`2500,3500,5000`), with clear seed-runtime
  asymmetry still visible.
- transfer families (`g12357`, `g246810`) and optional `g12345678910` remain
  exact in the tested range, dominated by Step-2 closure.
- historical `3567_plus` continuity rows are now finite-gap (no exact closure in
  this fortified configuration), indicating a continuity regression against the
  recovered PLAN_07 exact path.

Interpretation note recorded for follow-up:

- instrumentation clearly shows Phase-1/Pattern-generation dominates wall time on
  hard `g3567` rows, while exact-core traversal itself is comparatively small;
  continuation work should focus on reducing pool generation and phase-1 beam
  burden before further expansion of core windows.

## 2026-04-16

### Targeted K=4 recovery check (energy_core + step1_exact_guided)

Ran the required targeted K=4 check before continuing broader Plan-05 work.

Scope:

- Phase 1: old `3567_plus` frontier rows (`n=3500,5000`) with
  `PAST_RELAXED_BINPACK_SOLVER=energy_core`
- Phase 2: paper-group `g3567={3,5,6,7}` rows (`n=1000,1500,2500,3500,5000`)
  with the same forced energy-core mode
- Phase 3: direct comparison on the same `g3567` rows between:
  - default current policy
  - forced `energy_core`

Execution mode:

- `stateful_compare ablation-stdin step1_exact_guided`
- metrics archived in:
  - `research/k_vs_arithmetic_axes_20260412/csv/plan05/K4_energy_core_recovery_comparison_20260416.csv`

#### Phase 1 (old family) result

Using exact historical rows:

- `paperext_profile_repair_smallk_nscale_plus_20260409/0009_profile_smallk_3567_plus_n3500_s1`
- `paperext_profile_repair_smallk_nscale_plus_20260409/0011_profile_smallk_3567_plus_n5000_s1`

Forced energy-core no longer reproduces the older closure claim:

- `n=3500`: runtime `354.4763s`,
  `UB=172,475,616`, `LB=172,415,824`, gap `0.0347%`,
  `fwd_pack_method=block_repair_energy_core`, deciding step `step4`,
  `diag_exact_dp_used=1`, not exact
- `n=5000`: runtime `590.5615s`,
  `UB=248,943,407`, `LB=248,815,508`, gap `0.0514%`,
  `fwd_pack_method=block_repair_energy_core`, deciding step `step4`,
  `diag_exact_dp_used=1`, not exact

So the previously reported "energy-core incumbent then exact-guided closes" path
is not reproduced on current code for these two old frontier anchors.

#### Phase 2/3 (paper family) result

For `g3567`:

- `n=1000`: forced energy-core is better than default
  - energy-core: exact at Step 3 (`UB=LB=50,815,862`, `207.4820s`,
    `diag_exact_dp_used=0`)
  - default: finite gap `0.0016%` (`410.8945s`, `diag_exact_dp_used=1`)
- `n=1500,2500,3500`: default is better on gap quality (smaller gap),
  though energy-core is often faster
- `n=5000`: default timed out in this run window; forced energy-core returned a
  finite but non-tiny gap (`0.0545%`)

Decision for current policy:

- Do not switch to a blanket K=4 "energy-core-first" preference.
- Keep default beam-first Step-3 path as mainline for K=4,
  while keeping forced energy-core as a targeted override for specific rows
  (notably the `g3567 n=1000`-type case).

## 2026-04-15

### Plan 03F implementation: Step-3 K=2 restoration and structural selector update

Implemented the Step-3 policy from Plan 03F in
`solvers/cpp/stateful_dp_solver.cpp` so profile realization is now one family
with explicit modes:

- Mode A (`K=2`): exact profile realization by default
- Mode B (`K>=4`): exact profile realization only when tractable
- Mode C (`K>=4`): profile-repair beam fallback when exact is rejected

#### Technical note (Task A): what exact mode solves, why K=2 was skipped, and exact-vs-beam operations

- Exact mode (`profile_realization_dp_exact`) solves the recovered-profile
  assignment exactly in mixed-radix count-state space over merged blocks,
  reconstructs per-block counts, and evaluates each block via the exact local
  Level-3 evaluator (`evaluate_profile_block_counts`).
- `K=2` had been skipped in practice because selector scope required
  `lengths.size() >= 4`, while beam/profile-repair functions explicitly return
  `kInf` for `K<=2`; this left default mainline without Step-3 profile-repair
  behavior on `{8,10}`.
- Operationally, exact mode keeps the full reachable frontier in count-state
  space and proves feasibility/infeasibility on the recovered profile; beam mode
  uses a width-limited frontier with truncation/scoring, then local profile
  polishing, trading completeness for scalability.

#### Selector changes recorded

- Expanded selector scope to include `K=2` (and keep `K>=4`) in mainline
  profile-repair policy.
- Added explicit Mode-A `K=2` structural gates:
  - `PAST_PROFILE_REALIZATION_SELECTOR_K2_MAX_STATE_SPACE`
  - `PAST_PROFILE_REALIZATION_SELECTOR_K2_MAX_TOTAL_COMP_EST`
  - `PAST_PROFILE_REALIZATION_SELECTOR_K2_MAX_BLOCK_COMP_EST`
- Added branching proxies for `K>=4` exact-vs-beam gating:
  - average block branching estimate
  - max block branching estimate
  - thresholds:
    - `PAST_PROFILE_REALIZATION_SELECTOR_MAX_AVG_BRANCH_EST`
    - `PAST_PROFILE_REALIZATION_SELECTOR_MAX_BLOCK_BRANCH_EST`
- Preserved existing structural gates (merged blocks, state-space estimate,
  total/max composition estimates, hard-arithmetic alarm).
- Kept beam disabled for `K<=2` (beam kernel remains out-of-scope there), so
  Mode A runs exact-only unless explicitly safety-rejected.

#### Mandatory validation rerun completed

Ran required Plan 03F rows with `stateful_compare ablation-stdin
step1_exact_guided` and recorded selector/mode/runtime outcomes in
`RESULTS.md`.

- `{8,10}`: `n=500,600,750,1000,1500,2500,3500,5000`
  - all rows solved at Step 3
  - Step-3 mode: exact
  - selector decision/reason: `exact / k2_exact_default`
  - `fwd_pack_method=profile_realization_dp_exact`
  - `UB=LB` on all rows
  - `diag_exact_dp_used=0` on all rows
- K=4 frontier probe: `g3567, n=1000`
  - selector decision/reason: `beam / merged_blocks`
  - Step-3 beam incumbent generated; closure reached in Step 4 exact
- K=6 probe: `{4,6,8,10,12,14}, n=300`
  - selector decision/reason: `beam / state_space`
  - confirms selector still avoids broad exact-mode entry in higher-K regime

## 2026-04-12

### Archive creation

Created this archive as a new research track, separate from
`large_k_large_n_attempt_20260409`.

Purpose:

- stop conflating "`K` scaling" with arithmetic difficulty,
- and reorganize the next experiments around two axes:
  1. type-count scaling,
  2. arithmetic hardness.

### Starting evidence imported from the current solver state

The immediate motivation came from the current observed discrepancy:

- six-type hard-arithmetic families still leave small finite gaps,
- while some higher-`K` families are already exact at Step 1.

That discrepancy is exactly why this new archive exists.

### First working hypothesis

The current benchmark extension should be reinterpreted as:

- an easy-arithmetic scaling story,
- plus a hard-arithmetic recovery story,
- rather than one single "large-`K`" story.

### Literature search performed before archive creation

Looked up literature to support:

- numerical semigroup language and descriptors,
- arithmetic structure as a real source of algorithmic difficulty,
- periodicity / regularity in semigroup- or knapsack-like settings,
- and primal-recovery ideas for hard-arithmetic rows.

Key references collected into `LITERATURE.md` include:

- Rosales & García-Sánchez, *Numerical Semigroups*,
- Assi / D'Anna / García-Sánchez, *Numerical Semigroups and Applications*,
- Huang & Tang on UKP periodicity,
- Chvátal / Pisinger on hard knapsack instances,
- Barahona & Anbil on the volume algorithm.

### Immediate next actions recorded at archive creation

1. classify current families by arithmetic structure,
2. separate easy-arithmetic and hard-arithmetic run summaries,
3. identify which current claims already belong to the easy-arithmetic branch,
4. and only then decide what new implementation work is truly needed.

### Update after expert review

Reviewed an expert proposal arguing that the hard-arithmetic branch should be
strengthened by:

- residue / Apéry diagnostics,
- dynamic pricing inside the assignment loop,
- and later arc-flow per block if needed.

Current judgment:

- the archive's original two-axis framing remains the right first move,
- the expert proposal is strong as a **follow-up algorithmic branch** inside the
  hard-arithmetic side,
- and it should be recorded after the current suggestion rather than replacing
  it.

### Second expert review incorporated

A later review sharpened four concrete archive gaps:

1. medium-arithmetic rows should be explicit in the experiment design,
2. hard-arithmetic high-`K` cross-cells should be added to the matrix,
3. incumbent refinement should be described more concretely than just
   "stronger local improvement,"
4. bounded representability should be stated in `PROBLEM.md`, not only inferred
   from later sections.

These points were accepted and folded into the archive.

### Phase 3 executed: first two-axis baseline grid

Ran the first baseline grid with the current solver, using a frozen
configuration that disables the post-Lagrangian beam-polish branch:

- `PAST_BLOCK_REPAIR_LAGR_BEAM_POLISH_MAX_MERGED=0`

The resulting baseline table was written to:

- `research/k_vs_arithmetic_axes_20260412/csv/baseline/BASELINE_GRID_20260412.csv`

Measured rows:

- easy arithmetic:
  - `K3_contig_n300_s0`
  - `K4_contig_n300_s0`
  - `K5_contig_n300_s0`
  - `K6_contig_n200_s0`
  - `K7_contig_n100_s0`
  - `K8_contig_n200_s0`
  - `K10_1_10_n1000_s1`
  - `K10_1_10_n2500_s1`
- medium arithmetic:
  - `K6_456789_n1000_s1`
  - `K6_456789_n1500_s1`
- hard arithmetic:
  - `K6_2345711_n1000_s1`
  - `K6_2345711_n1500_s1`
  - `K6_2345711_n2500_s1`
- cross-cell:
  - `K7_irregular_n100_s0`

Main observation:

- every easy-arithmetic anchor in this first grid closed at Step 1,
- including `K=10` on the `1..10` family through the tested `n=2500` row,
- while the medium/hard six-type rows kept finite gaps.

This is the first clean experimental confirmation that the new archive framing
is not just narrative: arithmetic class already separates the behavior much
better than `K` alone.

### Phase 5 started: first structural code change (Level 3 separation)

Implemented the first code change justified by the plan:

- keep the current recovered-block assignment machinery,
- but stop evaluating assignment candidates by one global ascending/descending
  surrogate sequence,
- and instead precompute block-local costs on recovered windows.

Concretely in `solvers/cpp/stateful_dp_solver.cpp`:

- each recovered merged block now gets a local SPACES view,
- each block pattern is evaluated inside that local window,
- exact dense per-block multiset DP is used only when the local state space is
  small enough,
- otherwise the evaluator falls back to local ascending/descending
  `solve_fixed_sequence` calls.

This is the first actual Level 3 separation in the code:

- Level 2 still chooses multitype counts per recovered block,
- Level 3 now evaluates those choices within each block more honestly.

### First post-change results

Representative reruns after the Level 3 change:

- easy arithmetic:
  - `paperext_profile_repair_largek_nscale_20260409/0017_profile_largek_1_10_n1000_s1`
    - still exact at Step 1 in `36.57s`
- medium arithmetic:
  - `.../0009_profile_largek_456789_n1000_s1`
    - gap `0.0164%`, `49.35s`
  - `.../0011_profile_largek_456789_n1500_s1`
    - gap `0.0159%`, `111.08s`
- hard arithmetic:
  - `.../0001_profile_largek_2345711_n1000_s1`
    - gap `0.0082%`, `36.07s`
  - `.../0003_profile_largek_2345711_n1500_s1`
    - gap `0.0063%`, `76.40s`
  - `.../0005_profile_largek_2345711_n2500_s1`
    - gap `0.0067%`, `208.37s`

Comparison against the frozen baseline interpretation:

- `2345711 n=1000`: `0.0356% -> 0.0082%`
- `2345711 n=1500`: `0.0294% -> 0.0063%`
- `456789 n=1000`: `0.0480% -> 0.0164%`
- `456789 n=1500`: `0.0442% -> 0.0159%`
- easy `K=10` exactness is preserved

So the first structural code change is a real improvement, not just a cleaner
description.

### Important remaining nuance

Even after the Level 3 change, the current winning Step-1 incumbent on these
representative six-type rows still comes from:

- `block_repair_feasible_beam`

rather than from a fully separated Level 2 assignment winner such as the
Lagrangian branch itself.

That means:

- the Level 3 split is already helping the pipeline,
- but Level 2 still remains the more important open design frontier.

### Policy cleanup and first two-axis utility

After reading `EXPERT_GUIDANCE.md`, I made one cleanup change before widening
the experiments further:

- hidden beam rescues are no longer on by default inside the Lagrangian branch,
- specifically:
  - `PAST_BLOCK_REPAIR_LAGR_USE_SEEDED_BEAM` now defaults to `0`,
  - `PAST_BLOCK_REPAIR_LAGR_BEAM_POLISH_MAX_MERGED` now defaults to `0`.

This makes the baseline cleaner:

- Lagrangian, beam, and later fallbacks are again separable in policy terms,
- and the solver no longer quietly reintroduces a combined branch by default.

I also added a reusable experiment utility:

- `scripts/run_two_axis_grid.py`

which does two things together:

- builds controlled arithmetic-family instances,
- and records arithmetic descriptors alongside solver outcomes.

The script currently reports, per family:

- `K`,
- presence of `1`,
- gcd,
- multiplicity,
- contiguity,
- span,
- Frobenius number when gcd = 1,
- Apéry maximum,
- semigroup density up to a chosen cap.

One correctness bug was caught immediately and fixed in the runner:

- the first draft interpreted `n` as jobs-per-type,
- this silently blew up instance size by a factor of `K`,
- it is now corrected so `n` means total jobs, matching the benchmark
  convention and the earlier extension experiments.

### First archived phase-1 grid slice

Saved the first verified slice here:

- `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260412_phase1.csv`

Validated rows from the new driver:

- easy arithmetic:
  - `easy_k10_unit`, `n=300`
    - exact at Step 1
    - runtime `4.0521s`
    - winner `fwd_relax:ffd`
- hard arithmetic:
  - `hard_k4_irregular = {3,5,7,11}`, `n=300`
    - exact at Step 1
    - runtime `4.3910s`
    - winner `fwd_relax:ffd`
  - `hard_k8_irregular = {3,5,7,11,13,17,19,23}`, `n=300`
    - exact at Step 1
    - runtime `11.4293s`
    - winner `fwd_relax:ffd`
  - `hard_k10_irregular = {2,3,5,7,11,13,17,19,23,29}`, `n=300`
    - exact at Step 1
    - runtime `17.6653s`
    - winner `fwd_relax:ffd`
- medium arithmetic:
  - `medium_k6_dense = {4,5,6,7,8,9}`, `n=1000`
    - `UB = 62,412,903`
    - `LB = 62,404,265`
    - gap `0.0138%`
    - runtime `60.1463s`
    - incumbent method `block_repair_feasible_beam`
- hard arithmetic:
  - `hard_k6_2345711 = {2,3,4,5,7,11}`, `n=1000`
    - `UB = 52,575,221`
    - `LB = 52,568,409`
    - gap `0.0130%`
    - runtime `50.9597s`
    - incumbent method `block_repair_feasible_beam`

### First interpretation from the new grid slice

The strongest new observation is that the first hard-arithmetic high-`K`
cross-cell is already more nuanced than the old "large K = hard" story:

- irregular `K=8` and irregular `K=10` are still exact at Step 1 for `n=300`,
- so hard arithmetic does not automatically dominate as soon as `K` grows,
- and the two axes do not appear to collapse into one monotone hardness scale.

At the same time:

- medium and hard six-type rows at `n=1000` both remain small-gap rather than
  exact,
- and both are still owned by `block_repair_feasible_beam`,
- which keeps Level 2 as the main open frontier.

### Remaining open frontier from this batch

The meaningful missing cell is still:

- hard arithmetic with `K=8` or `K=10` at larger `n` such as `1000`.

Those rows were started from the new driver but intentionally not archived as
results yet because the first controlled run was interrupted before completion.
So the current archive has:

- a clean phase-1 tool,
- a verified first slice of the two-axis matrix,
- and one explicit remaining high-`K` hard-arithmetic frontier cell to finish.

## 2026-04-13

### Plan 01 execution: baseline recovery + phase-1 grid completion

Executed `PLAN_01_recover_level2_baseline_and_finish_phase1_grid.md` with the
requested constraints.

### Phase A checks (policy protection)

Verified in `solvers/cpp/stateful_dp_solver.cpp`:

- `PAST_BLOCK_REPAIR_LAGR_USE_SEEDED_BEAM` default remains `0`
- `PAST_BLOCK_REPAIR_LAGR_BEAM_POLISH_MAX_MERGED` default remains `0`

Kept both off in all baseline runs (enabled only once as a diagnostic, with no
quality gain).

### Phase B recovery result (Lagrangian baseline)

Reran the required validation anchors on the original benchmark JSON rows:

- `0001_profile_largek_2345711_n1000_s1`
- `0003_profile_largek_2345711_n1500_s1`
- `0005_profile_largek_2345711_n2500_s1`

with:

- `ablation-stdin step1_exact_guided 600`
- `PAST_BLOCK_REPAIR_LAGR_USE_SEEDED_BEAM=0`
- `PAST_BLOCK_REPAIR_LAGR_BEAM_POLISH_MAX_MERGED=0`
- `PAST_BLOCK_REPAIR_LAGR_PRICING=0`

Observed behavior is now clean and reproducible but **not** the earlier
Lagrangian-owned baseline:

- `n=1000`: `UB=48,641,508`, `LB=48,637,514`, gap `0.0082%`,
  `fwd_pack_method=block_repair_feasible_beam`
- `n=1500`: `UB=74,102,952`, `LB=74,098,255`, gap `0.0063%`,
  `fwd_pack_method=block_repair_feasible_beam`
- `n=2500`: `UB=125,450,588`, `LB=125,442,130`, gap `0.0067%`,
  `fwd_pack_method=block_repair_feasible_beam`

So the current clean baseline is beam-owned on these anchors; the earlier
Lagrangian-owned `0.0129/0.0079/0.0070` regime was not recovered without
reintroducing hybrid behavior.

### Dual/repair diagnosis from tracing

Tracing `n=1000` with `PAST_RELAXED_PACK_TRACE=1` and
`PAST_BLOCK_REPAIR_TRACE=1` confirms the known handoff pattern:

- Lagrangian loop reaches near-feasible states (`best_l1` around low tens),
- but returns no finite incumbent (`incumbent=inf`),
- then `block_repair_feasible_beam` finds the winner.

This keeps the diagnosis aligned with the plan: the limiting step is still the
Level-2 feasible-assignment recovery handoff, not Level 3 scoring.

### Phase C completion (required high-value cells)

Ran the required cells with the grid runner:

- command path: `scripts/run_two_axis_grid.py`
- mode: `step1_exact_guided`
- policy env: seeded-beam and beam-polish off by default
- output: `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260412_phase2.csv`

Required cells (`n=1000`, seed `0`):

1. `hard_k4_irregular`
   - `UB=63,958,209`, `LB=63,952,923`, gap `0.0083%`
   - runtime `121.5103s`
   - winner `block_repair_energy_core`

2. `hard_k8_irregular`
   - `UB=118,487,230`, `LB=118,468,602`, gap `0.0157%`
   - runtime `268.1152s`
   - winner `block_repair_feasible_beam`

3. `hard_k10_irregular`
   - `UB=120,229,235`, `LB=120,202,617`, gap `0.0221%`
   - runtime `377.6920s`
   - winner `block_repair_feasible_beam`

Optional rows added (same archive, separate run, merged into phase2 CSV):

- `medium_k6_dense`, `n=1000`, seed `1`:
  `UB=62,056,066`, `LB=62,039,931`, gap `0.0260%`,
  winner `block_repair_feasible_beam`
- `hard_k6_2345711`, `n=1000`, seed `1`:
  `UB=49,359,014`, `LB=49,355,840`, gap `0.0064%`,
  winner `block_repair_feasible_beam`

### Segfault incident and fix (Phase C robustness requirement)

Initial `hard_k10_irregular n=1000` run crashed with `returncode=-11`.

Per plan, rebuilt with ASAN and captured a precise stack trace.

ASAN finding:

- heap-buffer-overflow in `solve_exact_multiset_dp(...)`
- trigger: seeding a first job type with zero available multiplicity
- site: `new_s = strides[i]` followed by `state_rw[new_s]` out-of-bounds when
  `totals[i] == 0` and `strides[i] == NC`

Fix applied in `solvers/cpp/stateful_dp_solver.cpp`:

- added `if (totals[i] <= 0) continue;` guards in the exact DP seed loops:
  - `solve_exact_multiset_dp(...)`
  - `smart_reconstruct(...)`
  - `solve_sparse_exact_multiset_dp(...)`

After fix:

- required `hard_k10_irregular n=1000` row runs cleanly and is now recorded.

### Phase D bridge checks

Bridge reran anchor benchmark rows under current cleaned policy:

- old hard six-type anchor:
  `0001_profile_largek_2345711_n1000_s1` -> gap `0.0082%`,
  winner `block_repair_feasible_beam`
- old medium six-type anchor:
  `0009_profile_largek_456789_n1000_s1` -> gap `0.0164%`,
  winner `block_repair_feasible_beam`

This keeps the archive bridge explicit: current two-axis-generated rows and old
benchmark anchors both show finite tiny gaps with beam-owned Level 2 on the
hard/medium six-type regime.

### Phase E diagnostics added to runner

Updated `scripts/run_two_axis_grid.py` with small, plan-compliant diagnostics:

- preserved `n_jobs_per_type` for backward compatibility,
- added explicit `n_jobs_total` field,
- added `active_method`, `diag_merged_blocks`,
  `diag_winner_is_ffd`, `diag_winner_is_beam`, `diag_winner_is_lagr`.

No heavy tracing subsystem was added.

### Additional one-time boundary check status

Requested one-time block-boundary separability check
(`sum block-local L3` vs `global solve_fixed_sequence` on same assignment)
is still pending explicit instrumentation support. Current CSV does not expose
the fixed assignment sequence needed for an apples-to-apples direct compare.
Recorded as follow-up instrumentation, not blocked for this plan.

### Plan 02B execution: exact Level-2 branch-and-bound diagnostic

Plan alignment:

- Read and aligned with:
  - `implementation_plans/README.md`
  - `PLAN_01_recover_level2_baseline_and_finish_phase1_grid.md`
  - `PLAN_02B_close_gaps_and_consolidate.md`
  - `EXPERT_GUIDANCE.md`, `RESULTS.md`, `BLOCKERS.md`
- Applied Plan 02B correction: prior Plan 02 pool diagnostic was tautological
  because beam and Lagrangian share `generate_energy_core_patterns`.

Code implementation (self-contained, no external solver):

1. Added exact Level-2 branch-and-bound in
   `solvers/cpp/stateful_dp_solver.cpp`:

   - function: `block_repair_exact_level2_ub(...)`
   - decision variables: one pattern per block from existing per-block pool
   - state: residual type counts
   - branching: block-by-block
   - pruning:
     - suffix feasibility bounds (`suffix_min`, `suffix_max`)
     - suffix minimum-cost bound
     - memoized dominance by residual-count state (`best_seen`)
   - ordering: branch blocks in ascending active pool size for earlier pruning
   - time limit via env: `PAST_BLOCK_REPAIR_EXACT_L2_TIME`
   - returns best incumbent on timeout.

2. Integrated exact Level-2 after beam in pack pipeline:

   - gated by env `PAST_BLOCK_REPAIR_EXACT_L2` (default `1`)
   - initial UB seeded with beam UB (`fwd_beam_ub_for_exact_l2`)
   - no hidden policy changes to seeded beam / beam polish defaults.

3. Added CSV diagnostics exposed through `stateful_compare`:

   - `fwd_beam_ub_for_exact_l2`
   - `fwd_exact_l2_ub`
   - `fwd_exact_l2_time`
   - `fwd_exact_l2_nodes`
   - `fwd_exact_l2_closed`
   - `fwd_exact_l2_improved_over_beam`
   - `fwd_exact_l2_beam_optimal_in_pool`
   - `fwd_exact_l2_status`

Validation runs (Plan 02B required rows):

- env policy: cleaned defaults + pricing off
- exact-L2 controls:
  - `PAST_BLOCK_REPAIR_EXACT_L2=1`
  - `PAST_BLOCK_REPAIR_EXACT_L2_TIME=180` (seed0 set)
  - for `hard_k6_2345711 seed=1`, also ran `PAST_BLOCK_REPAIR_EXACT_L2_TIME=600`

Final validation CSV:

- `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260412_phase2b_exactl2_validation.csv`

Observed per required row:

1. `hard_k4_irregular n=1000 seed=0` (B=9)
   - beam UB: `63,959,486`
   - exact L2 UB: `63,952,923` (improved)
   - exact-L2 status: `closed`
   - exact-L2 time: `98.0751s`
   - exact-L2 nodes: `1,065,977`
   - final gap: `0.0000%`

2. `hard_k6_2345711 n=1000 seed=0` (B=8)
   - beam UB: `52,575,221`
   - exact L2 UB: `52,568,409` (improved)
   - exact-L2 status: `closed`
   - exact-L2 time: `13.6711s`
   - exact-L2 nodes: `10,408,311`
   - final gap: `0.0000%`

3. `hard_k6_2345711 n=1000 seed=1` (B=14)
   - beam UB: `49,359,014`
   - exact L2 UB: `49,355,840` (improved)
   - exact-L2 status: `closed` (with 600s limit)
   - exact-L2 time: `485.8594s`
   - exact-L2 nodes: `1,913,515,174`
   - final gap: `0.0000%`

4. `hard_k8_irregular n=1000 seed=0` (B=19)
   - beam UB: `118,487,230`
   - exact L2 UB: `118,487,230` (no improvement)
   - exact-L2 status: `timeout`
   - exact-L2 time: `180.0001s`
   - exact-L2 nodes: `503,067,136`
   - final gap remains `0.0157%`

5. `hard_k10_irregular n=1000 seed=0` (B=20)
   - beam UB: `120,229,235`
   - exact L2 UB: `120,229,235` (no improvement)
   - exact-L2 status: `timeout`
   - exact-L2 time: `180.0001s`
   - exact-L2 nodes: `301,474,304`
   - final gap remains `0.0221%`

6. `medium_k6_dense n=1000 seed=0` (B=9)
   - beam UB: `62,412,903`
   - exact L2 UB: `62,404,265` (improved)
   - exact-L2 status: `closed`
   - exact-L2 time: `12.3308s`
   - exact-L2 nodes: `19,186,729`
   - final gap: `0.0000%`

Interpretation from exact Level-2 evidence:

- On B=8–9 rows (and B=14 with larger exact-L2 time), the residual gap was
  **Level-2 in-pool search gap**: exact Level 2 improves beam to LB.
- On B=19–20 rows, exact Level 2 did not close in 180s and matched beam before
  timeout; this is consistent with a harder Level-2 search regime at larger B.
  It is not evidence that the pool/profile is the ceiling.

Plan correction captured explicitly:

- Prior Plan-02 narrative "search-gap dominated because beam_in_pool=1" is
  superseded by Plan 02B exact-L2 results.
- Updated next-step framing should rely on exact-L2 evidence, not the earlier
  tautological pool-membership diagnostic.

### Plan 03/04 cleanup execution: final 4-step pipeline alignment

Executed cleanup/restructuring to match the final method story exactly:

1. Step 1: semigroup profile recovery
2. Step 2: fast profile realization (FFD/BFD/random)
3. Step 3: one unified hard-case method family (`profile_repair_beam`)
4. Step 4: exact DP as the only exact fallback

Code-level policy cleanup in `solvers/cpp/stateful_dp_solver.cpp`:

- Default pack solver changed from mixed `default` behavior to
  `profile_repair_beam`.
- Added unified Step-3 function:
  - `block_repair_profile_repair_beam_ub(...)`
  - feasibility-first beam + bounded 2-block local destroy/repair
    intensification
  - exact/local per-block Level-3 evaluation retained.
- Demoted non-mainline branches from default dispatch:
  - `lagrangian_assign`, `feasible_counts`, `rg_beam`, `energy_core`
    no longer run in default mainline.
  - they remain callable only via explicit `PAST_RELAXED_BINPACK_SOLVER`.
- Demoted exact-L2 from mainline:
  - `PAST_BLOCK_REPAIR_EXACT_L2` default set to `0`.
  - when manually enabled, it is diagnostic-only unless
    `PAST_BLOCK_REPAIR_EXACT_L2_APPLY=1` is explicitly set.
  - default behavior no longer lets exact-L2 change the incumbent.
- Demoted fixed-profile `block_dp_exact` from default mainline:
  - now only runs when explicitly requested (`PAST_RELAXED_BINPACK_SOLVER=block_dp_exact`)
    or diagnostic flag `PAST_RELAXED_BINPACK_BLOCK_DP_DIAG=1`.

Exact-DP fallback reactivation checks:

- In `stateful_compare` ablation path (`step1_exact_guided`), exact stage still
  receives the best `ub` produced by Steps 2-3.
- Sparse exact DP remains first exact stage, dense exact DP remains exact fallback.
- No second exact family is active by default in mainline policy.

Added validation diagnostics in `solvers/cpp/stateful_compare.cpp` CSV:

- `diag_step1_decided`, `diag_step2_decided`, `diag_step3_decided`,
  `diag_step4_decided`
- `diag_exact_dp_used`
- `diag_exact_l2_mainline_used`

Validation rows executed after cleanup:

- `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260413_plan03_easy.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260413_plan03_medium.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260413_plan03_hard_k6.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260413_plan03_hard.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260413_plan03_exactl2_demoted.csv`
- merged summary:
  - `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260413_plan03_validation.csv`

Key post-cleanup confirmations:

- easy row (`easy_k10_unit n=300`): Step 2 closes via `ffd`, gap `0.0000%`.
- medium/hard rows: Step 3 method is `profile_repair_beam`.
- exact DP is used when budget allows and gap remains (e.g.,
  `medium_k6_dense n=1000`, `hard_k6_2345711 n=1000`).
- previously exact-L2-touched row (`hard_k4_irregular n=1000`) now reports:
  - `fwd_pack_method=profile_repair_beam`
  - `fwd_exact_l2_status=disabled`
  - `diag_exact_l2_mainline_used=0`
  confirming exact-L2 no longer affects default behavior.

## 2026-04-14

### Plan 03B/04A continuation: Step-3 strengthening + exact-stage diagnostics hardening

Continued the Plan 03B/04A cycle to finalize two open items:

1. strengthen unified Step 3 (`profile_repair_beam`) without adding new method
   families,
2. resolve exact-stage diagnostics interpretation for Step-4-used rows.

Code updates in `solvers/cpp/stateful_dp_solver.cpp` and
`solvers/cpp/stateful_compare.cpp`:

- Exact diagnostics now report explicit non-exhaustive skip modes instead of
  ambiguous `dense`+INF/zero patterns:
  - `sparse_skip_theoretical`
  - `sparse_skip_overflow`
  - `sparse_invalid_totals`
  - `dense_skip_state_space`
  - `dense_skip_memory`
- Dense exact timeout diagnostics now also populate:
  - `states_reached`
  - `states_expanded`
  - `exhaustive=0`
- Exact-diagnostic handoff was hardened so dense skip diagnostics do not
  overwrite a prior sparse diagnostic payload.

Interpretation fix achieved:

- For medium/hard `K=6` rows where Step 4 is entered, the exact stage is now
  clearly reported as:
  - `exact_diag_mode=sparse_skip_theoretical`
  - `exact_diag_initial_ub=<Step-3 UB>`
  - `exact_diag_final_ub=<same UB>`
  - `exact_diag_exhaustive=0`
- This confirms exact DP was attempted but skipped by the sparse theoretical
  lattice guardrail, rather than silently producing unusable diagnostics.

### Validation rerun set (Plan 03B/04A)

Reran the required representative rows and rewrote:

- `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260413_plan03b_easy.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260413_plan03b_medium.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260413_plan03b_hard_k6.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260413_plan03b_hard_k8_n800.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260413_plan03b_hard_k4.csv`
- merged:
  - `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260413_plan03b_validation.csv`

Representative outcomes from merged validation:

1. easy `easy_k10_unit n=300`
   - Step 2 decided (`ffd`), gap `0.0000%`, runtime `6.5965s`

2. medium `medium_k6_dense n=1000`
   - Step 3 method active: `profile_repair_beam`
   - final deciding step: Step 4 (`winner_detail=exact`)
   - `UB/LB/gap = 62,411,449 / 62,404,265 / 0.0115%`
   - exact diagnostics: `sparse_skip_theoretical`

3. hard `hard_k6_2345711 n=1000`
   - Step 3 method active: `profile_repair_beam`
   - final deciding step: Step 4 (`winner_detail=exact`)
   - `UB/LB/gap = 52,574,872 / 52,568,409 / 0.0123%`
   - exact diagnostics: `sparse_skip_theoretical`

4. hard high-K `hard_k8_irregular n=800`
   - Step 3 active, timed out, gap `0.0179%`

5. stubborn tiny-gap row `hard_k4_irregular n=1000`
   - Step 3 active, timed out, gap `0.0136%`

Step-3 beam diagnostics now show large, non-trivial search activity on active
rows (considered/kept/pruned/width columns populated), confirming the
strengthened unified beam is being exercised as intended.

### Plan 03C execution: Step-3 profile-realization DP unification (exact + truncated modes)

Completed the requested Plan-03C restructuring to make Step 3 one family:

- Step 3 = profile-realization DP
  - exact mode: fixed-block DP
  - truncated mode: profile-repair beam

Code restructuring in `solvers/cpp/stateful_dp_solver.cpp`:

- Added shared Step-3 helpers used by both exact and truncated modes:
  - `build_profile_block_local_views(...)`
  - `evaluate_profile_block_counts(...)`
  - `profile_realization_block_order(...)`
- Aligned beam and exact mode on common structure:
  - same recovered blocks,
  - same local block evaluator,
  - same hardest-first ordering policy interface.
- Added exact-safe enhancements inside fixed-block DP mode:
  - optional hardest-first block ordering (`PAST_PROFILE_REALIZATION_HARDEST_FIRST`),
  - optional suffix min/max residual pruning
    (`PAST_PROFILE_REALIZATION_EXACT_SUFFIX_PRUNE`),
  - retained sparse frontier dedup via per-layer reachability sets.
- Kept Step 4 separate (no merge with Step 3).

Mainline and diagnostics alignment:

- Step-3 exact mode now runs for default/profile-repair pack solver policy,
  but remains guardrailed by `MAX_COMP_EST` / `MAX_NC` as before.
- Exact mode winner label is now explicit:
  - `profile_realization_dp_exact`
- Added two CSV diagnostic flags:
  - `fwd_profile_realization_hardest_first`
  - `fwd_profile_realization_exact_suffix_prune`
- Updated `stateful_compare.cpp` step mapping so
  `profile_realization_dp_exact` is counted as Step 3, not Step 4.

Representative runs and measurements:

1. **Mainline representative rows (default limits, post-change)**

- `csv/plan03c/TMP_plan03c_mainline_easy.csv`
  - `easy_k10_unit n=300`: Step 2 (`ffd`), gap `0.0000%`
- `csv/plan03c/TMP_plan03c_mainline_medium_postdefault.csv`
  - `medium_k6_dense n=1000`: Step 3 (`profile_repair_beam`),
    Step 4 entered, gap `0.0115%`, `exact_diag_mode=sparse_skip_theoretical`
- `csv/plan03c/TMP_plan03c_mainline_hardk6_postdefault.csv`
  - `hard_k6_2345711 n=1000`: Step 3 (`profile_repair_beam`),
    Step 4 entered, gap `0.0123%`, `exact_diag_mode=sparse_skip_theoretical`

2. **Step-3 exact-mode tractability sweep**

- default limits (forced exact mode):
  - `csv/plan03c/TMP_plan03c_exactmode_defaultlimits_n300.csv`
  - `csv/plan03c/TMP_plan03c_exactmode_defaultlimits_k6_n1000.csv`
  - all representative rows skip exact mode with `fwd_block_dp_status=skipped_comp_est`.
- raised limits (`MAX_COMP_EST/MAX_NC=1e9`):
  - `csv/plan03c/TMP_plan03c_exactmode_lims_n300_cfgB.csv`
  - `csv/plan03c/TMP_plan03c_exactmode_lims_cfgA_k6_post.csv`
  - exact mode is tractable for:
    - `easy_k4_unit n=300`: feasible (`state_space=31,224,600`, `comps=2,178`)
    - `hard_k4_irregular n=300`: feasible (`state_space=32,433,024`, `comps=86`)
    - `medium_k6_dense n=120`: feasible (`state_space=74,826,180`, `comps=372`)
    - `hard_k6_2345711 n=120`: feasible (`state_space=77,565,600`, `comps=4,644`)
  - but remains intractable for larger `n` (e.g., K=6 `n=300/1000`) under practical guards.

3. **Exact-safe enhancement effects (measured in exact mode on K=6, n=120)**

- hardest-first ordering comparison:
  - files:
    - `csv/plan03c/TMP_plan03c_exactmode_hardk6_seedscan_post_cfgA.csv` (hardest-first=1)
    - `csv/plan03c/TMP_plan03c_exactmode_hardk6_seedscan_post_cfgB_nohard.csv` (hardest-first=0)
  - on 3-seed hard-K6 scan, hardest-first did **not** improve runtime;
    no-hardest-first was slightly faster on all seeds (small but consistent).
- suffix pruning comparison:
  - file:
    - `csv/plan03c/TMP_plan03c_exactmode_hardk6_seedscan_post_cfgC_nosuffix.csv`
  - with suffix pruning enabled vs disabled (same seeds):
    - mean runtime improved by ~`4.0%`
    - mean exact-mode `t_fwd_pack_block_dp` improved by ~`46.9%`
  - effect is strongest on harder seed(s), matching exact-safe pruning intent.

Structural diagnosis after Plan 03C:

- exact fixed-block DP and profile-repair beam are now truthfully two modes of
  one Step-3 profile-realization DP family in both code logic and run diagnostics;
- Step 4 remains separate and unchanged as global semigroup-guided exact DP.

### Plan 03D execution: Step-3 exact-vs-beam regime selector finalized

Executed `PLAN_03D_exact_vs_beam_regime_selector.md` on top of the Plan-03C
unified Step-3 family.

Code updates in `solvers/cpp/stateful_dp_solver.cpp/.hpp` and
`solvers/cpp/stateful_compare.cpp`:

- Added selector-policy plumbing with env:
  - `PAST_PROFILE_REALIZATION_SELECTOR_POLICY`
    (`auto_v1`, `off`, `force_exact`, `force_beam`)
- Added Step-3 selector observability fields:
  - policy/decision/reason,
  - arithmetic descriptor snapshot (`has_one`, `contiguous`, `multiplicity`,
    `semigroup_density`, `hard_alarm`),
  - exact-vs-beam status/timing split.
- Added exact frontier-estimate fields needed by Plan 03D:
  - `block_dp_total_comp_estimate`
  - `block_dp_max_comp_estimate`
  - `block_dp_max_compositions_per_block`
  - `block_dp_timed_out`
- Added Step-2-vs-Step-3 candidate comparison fields:
  - `profile_step2_ub`
  - `profile_beam_candidate_ub`, `profile_exact_candidate_ub`
  - `profile_beam_improved_over_step2`, `profile_exact_improved_over_step2`
- Added explicit selector skip statuses:
  - beam side: `skipped_selector`, `skipped_selector_exact_primary`
  - exact side: `skipped_selector`

Implemented `auto_v1` selector rule (human-readable gate):

- choose Step-3 exact mode only if all pass:
  - merged blocks `<= 4`
  - count-state estimate `<= 1e8`
  - total composition estimate `<= 1e8`
  - max per-block composition estimate `<= 8e7`
  - no arithmetic hard alarm
- otherwise choose Step-3 beam mode.

Hard alarm used in `auto_v1`:

- `has_one=0` and `contiguous=0` and `merged_blocks>=10` and
  `semigroup_density<=0.975`.

Selector-supporting guardrails were aligned so selector-approved exact rows can
run in practice:

- default `PAST_RELAXED_BINPACK_MAX_NC = 1e8`
- default `PAST_RELAXED_BINPACK_MAX_COMP_EST = 1e8`

Validation artifacts generated:

- consolidated table:
  - `research/k_vs_arithmetic_axes_20260412/csv/plan03d/TMP_plan03d_selector_validation_table.csv`
- auto-policy rows:
  - `csv/plan03d/TMP_plan03d_auto_easyk10_n300_v3.csv`
  - `csv/plan03d/TMP_plan03d_auto_mediumk6_n1000_v3.csv`
  - `csv/plan03d/TMP_plan03d_auto_hardk4_n1000_v3.csv`
  - `csv/plan03d/TMP_plan03d_auto_hardk6_n1000_v3.csv`
  - `csv/plan03d/TMP_plan03d_auto_hardk8_n800_v3.csv`
  - `csv/plan03d/TMP_plan03d_auto_k4_n300_v3.csv`
  - `csv/plan03d/TMP_plan03d_auto_k6_n120_v3.csv`
- forced-exact controls:
  - `csv/plan03d/TMP_plan03d_forceexact_easyk10_n300_v3.csv`
  - `csv/plan03d/TMP_plan03d_forceexact_mediumk6_n1000_v3.csv`
  - `csv/plan03d/TMP_plan03d_forceexact_hardk4_n1000_v3.csv`
  - `csv/plan03d/TMP_plan03d_forceexact_hardk6_n1000_v3.csv`
  - `csv/plan03d/TMP_plan03d_forceexact_hardk8_n800_v3.csv`
  - plus small exact-island checks under
    `csv/plan03d/TMP_plan03d_selector_forceexact_*.csv`.

Observed regime split from the validation table:

- exact selected on 8/13 rows (all small merged-block islands);
  all 8 exact runs were feasible and fast.
- beam selected on 5/13 rows (all representative larger/harder rows);
  forced exact on these rows skipped by comp-est guardrail and often produced no
  usable incumbent (`pack_method=none`, `exact_skipped_comp_est`).
- no misclassification observed on this representative set.

Plan-03D recommendation from this cycle:

- `auto_v1` is ready to remain the default Step-3 selector,
- with one follow-up calibration sweep as non-blocking hardening
  (more near-threshold seeds + more hard-alarm-trigger rows).

### Plan 03D hardening pass: exact-primary safety fallback + validation split fix

Executed the requested follow-up hardening pass without redesigning `auto_v1`.

#### 1) Step-3 control-flow hardening implemented

In `solvers/cpp/stateful_dp_solver.cpp/.hpp`, for `auto_v1` exact-primary rows:

- Step 3 now runs exact mode first, then **automatically falls back to beam** in
  the same Step-3 cycle when exact does not produce a finite candidate.
- fallback trigger condition is candidate-based and covers the requested failure
  classes (`skipped_comp_est`, `skipped_nc`, `timeout`, `reconstruct_failed`,
  and any non-finite exact candidate outcome).

This keeps the method story unchanged:

- Step 3 remains one profile-realization DP family,
- exact-first, beam-second fallback only when needed.

#### 2) New diagnostics added for fallback observability

Added row-level fields (propagated into `stateful_compare` CSV):

- `fwd_profile_exact_primary_fallback_to_beam`
- `fwd_profile_exact_primary_status_before_fallback`
- `fwd_profile_step3_incumbent_mode`

These make fallback explicit and non-hidden.

#### 3) Fallback behavior validated with explicit probes

Generated probes:

- timeout probe:
  - `csv/plan03d/TMP_plan03d_exact_primary_fallback_probe.csv`
  - `csv/plan03d/TMP_plan03d_exact_primary_fallback_probe_exactguided.csv`
- `skipped_nc` probe:
  - `csv/plan03d/TMP_plan03d_exact_primary_fallback_skipnc_probe.csv`
- `skipped_comp_est` probe:
  - `csv/plan03d/TMP_plan03d_exact_primary_fallback_skipcomp_probe.csv`

Observed on probe rows:

- exact status before fallback recorded as `timeout`, `skipped_nc`, or
  `skipped_comp_est` as intended,
- `fwd_profile_exact_primary_fallback_to_beam=1`,
- beam then returns feasible candidate,
- `fwd_profile_step3_incumbent_mode=beam`.

#### 4) Selector validation methodology corrected

Built new step-separated boundary validation artifacts:

- raw auto/forced runs:
  - `csv/plan03d/TMP_plan03d_selector_boundary_reval_raw.csv`
- step-separated validation table:
  - `csv/plan03d/TMP_plan03d_selector_boundary_reval_table.csv`

The table now includes explicit flags:

- `step2_closed_row`
- `step3_selector_test_row`
- `step4_used_row`

Misclassification is counted only on `step3_selector_test_row=1`.

Current counts in `csv/plan03d/TMP_plan03d_selector_boundary_reval_table.csv`:

- `step2_closed_rows = 10`
- `step3_selector_test_rows = 6`
- `step4_used_rows = 5`
- `misclassifications_on_step3_rows = 0`

So Step-2-closed rows are no longer counted as selector wins.

#### 5) Near-boundary calibration runs (merged 4..6 / threshold-near)

Additional calibration/probe sets produced:

- `csv/plan03d/TMP_plan03d_boundary_scan_auto_v1.csv`
- `csv/plan03d/TMP_plan03d_boundary_scan_midn_auto_v1.csv`
- `csv/plan03d/TMP_plan03d_probe_hardk6_seed12.csv`
- `csv/plan03d/TMP_plan03d_probe_hardk4_boundary_seeds012.csv`
- `csv/plan03d/TMP_plan03d_calib_boundary_step3split.csv`

Observed near boundary:

- selector transitions around merged-block and comp-est limits are visible,
- rows with Step 3 genuinely active are now explicitly distinguishable,
- some near-threshold rows later rely on Step 4, now separately tagged.

#### 6) Updated recommendation after hardening pass

- `auto_v1` is now materially more robust due to exact-primary safety fallback.
- Keep it as default.
- One more iteration is still recommended for broader near-threshold coverage,
  because current step3-test sample is still small (`6` rows) even though
  methodology is now correct.

## 2026-04-15

### Plan 04C continuation: targeted incumbent/pruning matrix with real sparse exact expansions

Goal for this pass:

- continue Plan 04C with measurable Step-4 behavior on rows where sparse exact DP
  actually expands states,
- avoid the earlier matrix regime where many rows ended as
  `exact_diag_mode=none` or `sparse_skip_theoretical` with zero counters.

### What changed in experimental setup

1. Kept method policy unchanged (same final 4-step story).
2. Ran targeted rows with:
   - `PAST_SPARSE_EXACT_MAX_THEORETICAL=9000000000000000000`
   - explicit incumbent source (`PAST_EXACT_INCUMBENT_SOURCE`)
   - explicit exact variant (`PAST_EXACT_DP_VARIANT` in `p0/p1/p2/p3`)
3. Focused on rows where Step-3 returns a finite incumbent and exact can run long
   enough to expose pruning counters.

### New artifacts produced (Plan04C v3)

- `research/k_vs_arithmetic_axes_20260412/csv/plan04c/TMP_plan04c_v3_evidence_runs.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan04c/TMP_plan04c_v3_phase1_incumbent_quality.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan04c/TMP_plan04c_v3_phase1_best_incumbent_by_family.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan04c/TMP_plan04c_v3_phase2_exactdp_variants.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan04c/TMP_plan04c_v3_phase3_best_combos.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan04c/TMP_plan04c_v3_matrix_summary.csv`

Additional diagnostic scans kept:

- `research/k_vs_arithmetic_axes_20260412/csv/plan04c/TMP_plan04c_v3_exactexp_scan_p0.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan04c/TMP_plan04c_v3_probe_exact_expansion.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan04c/TMP_plan04c_v3_targeted_matrix_raw.csv`

### Key measured outcomes from v3

Primary anchor with real exact expansions:

- `hard_k8_irregular n=500 seed=0`

Phase 1 (incumbent quality, `p0`):

- `i1`/`i2`: same final gap (`0.0305%`), total runtime about `235s`, sparse exact
  expanded `13,390,891` states, exact elapsed about `150.5s`.
- `i3`/`i4`: same final gap (`0.0305%`), total runtime about `186.5s`, sparse
  expanded `3,228,616` states, exact elapsed about `36.3s`.
- relative effect (`i3` vs `i2`):
  - total runtime improved by about `20.7%`,
  - exact elapsed reduced by about `75.9%`,
  - sparse expanded states reduced by about `75.9%`.
- `i0` on this row gave no finite Step-3 incumbent (`fwd_pack_method=none`), and
  exact timed out with unusable final UB/LB (`-1/-1`).

Phase 2 (exact variants):

- On `hard_k8_irregular n=500` with `i2` fixed:
  - `p1/p3` reduced sparse expansions (`13.39M -> 7.55M`) and bound-prune counts,
  - but final UB/gap did not improve, and runtime remained about `235s`.
- With stronger incumbent `i3` fixed:
  - `p1/p3` did not further reduce expansions versus `p0/p2`,
  - and added runtime overhead (about `+29s`).
- `p2` behaved effectively like `p0` on this slice.

Additional stress row (`medium_k6_dense n=600`, weak incumbent `i0`):

- exact expands states in both `p0` and `p3` (`3,822,711`),
- `p3` reports small type-aware pruning (`42`) but still no finite UB/LB
  (run remains unresolved under budget).

### Interpretation recorded for handoff

- After enabling real sparse expansion, the dominant lever on hard anchor behavior
  is incumbent quality/shape (`i3/i4` style), not current `p1/p2/p3` changes.
- Type-aware LB (`p1`) can reduce explored states under weaker incumbents, but is
  not yet translating to better final quality on this hard anchor.
- Current strengthened beam (`i3`) remains quality-neutral vs `i2` here but greatly
  reduces exact-stage work and total runtime by giving exact a stronger handoff.
## 2026-04-20 — `g37` frontier correction

- Re-verified `g37` directly from `csv/plan05/PAPER_GROUPS_PLAN05_n_extension.csv`.
- At that checkpoint, corrected current-facing docs to say `g37` was exact only through `n=600`.
- Later `g37` rows do enter sparse exact / Step 4 on some runs, but they do not
  close (`750/1000` timeout in sparse exact, `1500..5000` failed Step-4 exact,
  `6000/7000` unresolved).
- Superseded by PLAN13 on 2026-04-21: later evidence showed those rows were
  misrouted; corrected K=2 Step-3 reroute closes tested `g37` rows through
  `n=5000`.
## 2026-04-21 — current-surface markdown cleanup

- Added a proper fast-entry layer for future continuity:
  - `OVERVIEW.md`
  - `ACTIVE.md`
  - `iterations/20260421_paper_group_recovery/`
- Archived non-current top-level markdown notes under:
  - `archive_20260421/markdown/`
- Kept current benchmark-facing docs at thread root and reduced top-level
  markdown clutter without deleting accepted results.

## 2026-04-22 — PLAN16 pivot: fixed-n K-scaling campaign completed

- Supervisor-priority pivot applied: stop n-scaling and run fixed `n=1000`
  across current families with varying `K`.
- Added campaign runner:
  - `research/k_vs_arithmetic_axes_20260412/run_plan16_k_scaling_n1000.py`
- Produced artifact:
  - `research/k_vs_arithmetic_axes_20260412/csv/plan16/PLAN16_k_scaling_n1000.csv`
  - `research/k_vs_arithmetic_axes_20260412/csv/plan16/PLAN16_k_scaling_n1000_summary.csv`
- Campaign scope:
  - families: `g24`, `g37`, `g810`, `g3567`, `g12357`, `g246810`,
    `g12345678910`, `g1234567891011121314151617181920`
  - seeds: `0/1`
  - variants: `baseline`, `dense_step2_fastpath`
  - fixed `n=1000`, `lambda=1.3`
- Corrected K=2 routing bug in the runner:
  - forcing `energy_core` bypassed the Step-3 selector and misrouted `g37/g810`
    into Step 4,
  - reran those rows through `profile_repair_beam + auto_v1`,
  - recovered exact Step-3 closure for `g37` and `g810` at `n=1000` on both seeds.

# Current Results Index

This is the current-facing entrypoint for accepted paper-group results and method provenance in this thread.

## Start Here

For a fresh conversation with minimal context recovery cost:

1. [ACTIVE.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/ACTIVE.md)
2. [OVERVIEW.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/OVERVIEW.md)
3. [iterations/20260429_k12_anytime_incumbent_panel/SUMMARY.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/iterations/20260429_k12_anytime_incumbent_panel/SUMMARY.md)
4. [PAPER_HPC_REPRODUCIBILITY_MAP.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/PAPER_HPC_REPRODUCIBILITY_MAP.md)
5. [COMPREHENSIVE_METHOD_AND_EXPERIMENT_SUMMARY.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/COMPREHENSIVE_METHOD_AND_EXPERIMENT_SUMMARY.md)
6. [END_TO_END_PIPELINE_AND_SCALING_SUMMARY.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/END_TO_END_PIPELINE_AND_SCALING_SUMMARY.md)

## Current accepted package (active baseline)

- Workflow entry: `ablation-stdin step1_exact_guided`
- Binary: `solvers/cpp/build/stateful_compare`
- Accepted baseline package:
  - `PAST_RELAXED_BINPACK_SOLVER=energy_core`
  - `PAST_BLOCK_REPAIR_COMPLETION_MODE=direct`
  - `PAST_BLOCK_REPAIR_COMPLETION_DIRECT_MAX_CELLS=500000000`
  - `PAST_BLOCK_REPAIR_EC_STRONGER_CENTER=0`
  - `PAST_BLOCK_REPAIR_EC_DIVERSIFY=0`
  - `PAST_BLOCK_REPAIR_EC_ADAPTIVE_DELTA=0`
  - `PAST_BLOCK_REPAIR_EC_TWO_PHASE=0`
  - `PAST_BLOCK_REPAIR_EG_STATE_KEEP=60000`
- Accepted K=4 generator policy from PLAN10 remains active:
  - `PAST_BLOCK_REPAIR_PATTERN_DP_K=4` for K=4 rows
  - `PAST_BLOCK_REPAIR_EC_SIGNATURE_DEDUP=0` for K=4 rows

## Current source-of-truth artifacts

- HPC/code provenance map: `research/k_vs_arithmetic_axes_20260412/PAPER_HPC_REPRODUCIBILITY_MAP.md`
- Detailed method and experiment summary: `research/k_vs_arithmetic_axes_20260412/COMPREHENSIVE_METHOD_AND_EXPERIMENT_SUMMARY.md`
- Canonical end-to-end summary: `research/k_vs_arithmetic_axes_20260412/END_TO_END_PIPELINE_AND_SCALING_SUMMARY.md`
- Presentation-ready summary: `research/k_vs_arithmetic_axes_20260412/PRESENTATION_RESULTS_SUMMARY.md`
- Primary ledger: `research/k_vs_arithmetic_axes_20260412/csv/plan05/PAPER_GROUPS_PLAN05_n_extension.csv`
- Current extension summary: `research/k_vs_arithmetic_axes_20260412/PAPER_GROUPS_EXTENSION_SUMMARY.md`
- PLAN11 frontier table: `research/k_vs_arithmetic_axes_20260412/csv/plan11/PAPER_GROUPS_PLAN11_n_extension.csv`
- PLAN10 K=4 generator decision table: `research/k_vs_arithmetic_axes_20260412/csv/plan10/PLAN10_k4_generator_compare.csv`
- PLAN30 easy-vs-hard K-scaling table: `research/k_vs_arithmetic_axes_20260412/csv/plan30/PLAN30_easy_vs_hard_k_boundary.csv`
- PLAN33 hard-K certified prepass table: `research/k_vs_arithmetic_axes_20260412/csv/plan33/PLAN33_cert_anytime_summary.csv`

## Current paper-use rules

- Laptop CSVs are design/provenance evidence. Final paper runtimes should be regenerated on HPC.
- For each paper result, cite `PAPER_HPC_REPRODUCIBILITY_MAP.md` to identify the runner, solver path, and environment toggles.
- PLAN32B parallel UB is invalid for the benchmark and must not be cited as a valid result.
- For hard K10/K12 rows, PLAN33 supersedes older PLAN18/19 "mostly timeout/no-incumbent" status with certified finite-gap evidence.

## Latest experimental diagnostics

- PLAN31 family-aware beam scoring + policy oracle:
  - `research/k_vs_arithmetic_axes_20260412/csv/plan31/PLAN31_existing_policy_oracle.csv`
  - `research/k_vs_arithmetic_axes_20260412/csv/plan31/PLAN31_existing_policy_oracle_notes.md`
  - `research/k_vs_arithmetic_axes_20260412/csv/plan31/PLAN31_family_aware_survivor_raw.csv`
  - `research/k_vs_arithmetic_axes_20260412/csv/plan31/PLAN31_family_aware_survivor_compare.csv`
  - `research/k_vs_arithmetic_axes_20260412/csv/plan31/PLAN31_family_aware_survivor_summary.csv`
  - `research/k_vs_arithmetic_axes_20260412/csv/plan31/PLAN31_fine_block_guided_beam_notes.md`
  - PLAN31 decision: **A** — Promote family-aware survivor selection (hardA=uniform_mult2, hardB=ambig_scoreband_mult2)
- PLAN29 multi-view block reconstruction: `research/k_vs_arithmetic_axes_20260412/csv/plan29/PLAN29_multiview_block_reconstruction_raw.csv`
  - PLAN29 decision: **C** — No coarsening view improves >= 4/8 K10 rows
- PLAN28 block-realizability diagnostics: `research/k_vs_arithmetic_axes_20260412/csv/plan28/PLAN28_block_realizability_diagnostics_raw.csv`
  - PLAN28 decision: **C** — Diagnostics do not separate easy from hard
- PLAN30 easy-vs-hard fixed-n K-scaling story (implements PLAN_16):
  - `research/k_vs_arithmetic_axes_20260412/csv/plan30/PLAN30_easy_k_scaling_raw.csv`
  - `research/k_vs_arithmetic_axes_20260412/csv/plan30/PLAN30_easy_k_scaling_summary.csv`
  - `research/k_vs_arithmetic_axes_20260412/csv/plan30/PLAN30_easy_vs_hard_k_boundary.csv`
  - `research/k_vs_arithmetic_axes_20260412/csv/plan30/PLAN30_easy_vs_hard_notes.md`
- PLAN27 Step-3 adaptive survivor policy: `research/k_vs_arithmetic_axes_20260412/csv/plan27/PLAN27_step3_adaptive_survivor_raw.csv`
- PLAN27 comparison table: `research/k_vs_arithmetic_axes_20260412/csv/plan27/PLAN27_step3_adaptive_survivor_compare.csv`
- PLAN27 summary: `research/k_vs_arithmetic_axes_20260412/csv/plan27/PLAN27_step3_adaptive_survivor_summary.csv`
- PLAN27 notes: `research/k_vs_arithmetic_axes_20260412/csv/plan27/PLAN27_step3_adaptive_survivor_notes.md`
- PLAN27 decision: **A with caveat** — `uniform_mult2` passes Gate A promotion. Family-dependent improvement.
- PLAN26 local corridor validation + multi-idea queue: `research/k_vs_arithmetic_axes_20260412/csv/plan26/PLAN26_multi_idea_raw.csv`
- PLAN26 comparison table: `research/k_vs_arithmetic_axes_20260412/csv/plan26/PLAN26_multi_idea_compare.csv`
- PLAN26 notes: `research/k_vs_arithmetic_axes_20260412/csv/plan26/PLAN26_multi_idea_notes.md`
- PLAN26 implementation plan: `research/k_vs_arithmetic_axes_20260412/implementation_plans/PLAN26_beam_corridor_multi_idea_queue.md`
- PLAN25 local corridor exact DP (offset encoding): `research/k_vs_arithmetic_axes_20260412/csv/plan25/PLAN25_local_corridor_dp_raw.csv`
- PLAN25 comparison table: `research/k_vs_arithmetic_axes_20260412/csv/plan25/PLAN25_local_corridor_dp_compare.csv`
- PLAN25 notes: `research/k_vs_arithmetic_axes_20260412/csv/plan25/PLAN25_local_corridor_dp_notes.md`
- PLAN25 status correction: **PLAN26 validated that local corridor is invalid**. Block-local schedulability mismatch means the base beam path is rejected. Decision: **C**.
- PLAN24B forced-entry corridor diagnostic: `research/k_vs_arithmetic_axes_20260412/csv/plan24b/PLAN24B_forced_corridor_raw.csv`
- PLAN24 beam-guided corridor: `research/k_vs_arithmetic_axes_20260412/csv/plan24/PLAN24_beam_corridor_exact_raw.csv`

## Current accepted paper-group frontiers (lambda=1.3, seeds 0/1)

- `g3567`: exact through `n=6000` (Step 3, `block_repair_energy_core`), timeout at `n=7000`, `std::length_error` crash at `n=8000`.
- `g24`: exact through `n=10000` (Step 2, `ffd`).
- `g12357`: exact through `n=8000` (Step 2, `ffd`), timeout at `n=10000`.
- `g246810`: exact at `n=6000` (Step 2, `ffd`), `std::length_error` crash from `n=7000`.
- `g12345678910`: timeout at `n=5000` remains, also timeout at `n=6000/7000`.
- `g810`: exact through `n=5000` (Step 3 exact mode), `std::length_error` crash from `n=6000`.
- `g37`: corrected reroute evidence shows exact Step-3 closure through tested rows `n=750,1000,1500,2500,3500,5000` under the intended K=2 route; legacy unresolved rows are non-mainline historical evidence and should not be read as current K=2 capability.

## Current unresolved blockers

- High-n robustness failures (`std::length_error`) on multiple families (`g3567`, `g246810`, `g810`).
- Runtime regime breaks after newly extended exact points (`g3567` at `7000`, `g12357` at `10000`).
- `g37` no longer blocks the fixed-`n=1000` K-axis picture once the intended K=2 Step-3 route is enforced.
- Baseline integrity constraint: keep accepted package unchanged; only additive variants allowed until direct benchmark wins are demonstrated.

See detailed blocker narrative in `research/k_vs_arithmetic_axes_20260412/BLOCKERS.md`.

## Adaptive Node Evaluation (PLAN22 + PLAN22B + PLAN23)

- fixed `n=1000` adaptive multiplicity artifacts (hard irregular ladders, K=8/10/12):
  - PLAN22:
    - `research/k_vs_arithmetic_axes_20260412/csv/plan22/PLAN22_adaptive_node_eval_raw.csv`
    - `research/k_vs_arithmetic_axes_20260412/csv/plan22/PLAN22_adaptive_node_eval_compare.csv`
    - `research/k_vs_arithmetic_axes_20260412/csv/plan22/PLAN22_adaptive_node_eval_summary.csv`
    - `research/k_vs_arithmetic_axes_20260412/csv/plan22/PLAN22_adaptive_node_eval_notes.md`
  - PLAN22B (Gate 2 correction pass):
    - `research/k_vs_arithmetic_axes_20260412/csv/plan22b/PLAN22B_ambig_scoreband_validation_raw.csv`
    - `research/k_vs_arithmetic_axes_20260412/csv/plan22b/PLAN22B_ambig_scoreband_validation_compare.csv`
    - `research/k_vs_arithmetic_axes_20260412/csv/plan22b/PLAN22B_ambig_scoreband_validation_summary.csv`
    - `research/k_vs_arithmetic_axes_20260412/csv/plan22b/PLAN22B_ambig_scoreband_validation_notes.md`
  - PLAN23 (role-based survivor policy, Gate 1 only):
    - `research/k_vs_arithmetic_axes_20260412/csv/plan23/PLAN23_role_based_beam_raw.csv`
    - `research/k_vs_arithmetic_axes_20260412/csv/plan23/PLAN23_role_based_beam_compare.csv`
    - `research/k_vs_arithmetic_axes_20260412/csv/plan23/PLAN23_role_based_beam_summary.csv`
    - `research/k_vs_arithmetic_axes_20260412/csv/plan23/PLAN23_role_based_beam_notes.md`
  - PLAN24 (beam-guided Step 4 exact corridor, K=10):
    - `research/k_vs_arithmetic_axes_20260412/csv/plan24/PLAN24_beam_corridor_exact_raw.csv`
    - `research/k_vs_arithmetic_axes_20260412/csv/plan24/PLAN24_invalid_energy_core_misroute_raw.csv`
    - `research/k_vs_arithmetic_axes_20260412/csv/plan24/PLAN24_beam_corridor_exact_compare.csv`
    - `research/k_vs_arithmetic_axes_20260412/csv/plan24/PLAN24_beam_corridor_exact_summary.csv`
    - `research/k_vs_arithmetic_axes_20260412/csv/plan24/PLAN24_beam_corridor_exact_notes.md`
  - PLAN24B (forced-entry corridor exact DP diagnostic):
    - `research/k_vs_arithmetic_axes_20260412/csv/plan24b/PLAN24B_forced_corridor_raw.csv`
    - `research/k_vs_arithmetic_axes_20260412/csv/plan24b/PLAN24B_forced_corridor_compare.csv`
    - `research/k_vs_arithmetic_axes_20260412/csv/plan24b/PLAN24B_forced_corridor_notes.md`
- concise findings:
  - `ambig_scoreband_mult2` passed Gate 1 and produced the best gap improvement (hardA_k10 s=0: 0.0172% → 0.0094%).
  - `hybrid_mult2` failed Gate 1 due to K=8 degradation.
  - `early_mult2` passed Gate 1 but only improved runtime, not gap.
  - Naive uniform multiplicity confirmed seed-dependent.
  - **PLAN22B correction:** `ambig_scoreband_mult2` does NOT generalize reliably on Gate 2 (4-5 vs standard).
  - **Corrected decision: E** — use `ambig_scoreband_mult2` only as a K=10 quality-improvement candidate, not as a global policy.
  - **PLAN23:** Role-based survivor policy (`role_mult3`, `role_mult3_feas`) failed Gate 1. Did not improve gap over standard or uniform. Runtime increased 55-63%. No survivor-policy change validated. Decision remains **E**.
  - **PLAN24:** Beam-guided Step 4 exact corridor tested on hardA_k10/hardB_k10 seeds 0-3. Zero corridor pruning (sparse exact DP skips search via `sparse_skip_theoretical`). Identical gaps to standard. No exact closure, no improvement. K=12 probe skipped. **Decision: D** — no evidence corridor helps.
  - **PLAN24B:** Forced-entry corridor exact DP diagnostic on hardA_k10 s0, hardB_k10 s2. Force entry bypasses theoretical guardrail but hits `sparse_skip_overflow`: int64 encoding overflows for K=10 at n=1000. Zero states generated, zero pruning. Corridor cannot be tested because sparse exact DP encoding is fundamentally limited. **Decision: D** — abandon corridor under current exact DP.
  - **PLAN25:** Local-offset corridor avoids mixed-radix overflow and runs memory-safe, but current `infeasible_corridor` rows are diagnostic/inconclusive. Do not treat PLAN25 as proof of corridor uselessness until base-path survival and block/count alignment are validated.

## Fixed-n K=10/12 Redesign (PLAN19)

- fixed `n=1000` K=10/12 redesign artifacts (hard irregular ladders, seeds `0/1`):
  - `research/k_vs_arithmetic_axes_20260412/csv/plan19/PLAN19_k10_k12_redesign_raw.csv`
  - `research/k_vs_arithmetic_axes_20260412/csv/plan19/PLAN19_k10_k12_best_variant_summary.csv`
  - `research/k_vs_arithmetic_axes_20260412/csv/plan19/PLAN19_k10_k12_redesign_compare.csv`
  - `research/k_vs_arithmetic_axes_20260412/csv/plan19/PLAN19_k10_k12_failure_shift.csv`
  - `research/k_vs_arithmetic_axes_20260412/csv/plan19/PLAN19_k10_k12_method_notes.md`
- concise findings:
  - K=10: no exact rows; finite-gap incumbents (~0.02-0.06%) are the practical ceiling.
  - Historical PLAN19 K=12 status: mostly timeout/no-incumbent; stronger beam (`beam_plus`) timed out on 6/8 seeds before PLAN33 recovery.
  - Exact fixed-block DP is structurally infeasible at K=10/12 under practical budgets.
  - Routing override justified for K>=10 hard irregular.

Supersession note:

- PLAN19 remains useful as a negative redesign/closure study.
- Do not use PLAN19 as the current hard-K incumbent status. PLAN33 supersedes it for tested hard K10/K12 rows with certified finite gaps <= 0.0593%.

## Fixed-n K-Boundary Refinement (PLAN18)

- fixed `n=1000` K-boundary artifact (hard irregular ladders, seeds `0/1/2/3`):
  - `research/k_vs_arithmetic_axes_20260412/csv/plan18/PLAN18_k_boundary_refine_n1000_raw.csv`
  - `research/k_vs_arithmetic_axes_20260412/csv/plan18/PLAN18_k_boundary_refine_best_of_route.csv`
  - `research/k_vs_arithmetic_axes_20260412/csv/plan18/PLAN18_k_boundary_refine_summary_by_k.csv`
  - `research/k_vs_arithmetic_axes_20260412/csv/plan18/PLAN18_k_boundary_refine_failure_signatures.csv`
- concise findings:
  - `K=8`: mixed exact vs finite-gap (2/4 exact per ladder);
  - `K=10`: no exact rows; finite-gap incumbents via additive reroute;
  - Historical PLAN18 `K=12`: mostly timeout/no-incumbent under 1200s/12GB cap before PLAN33 recovery.

Supersession note:

- PLAN18 remains the boundary-detection study showing where exact closure degrades.
- Current hard K12 feasibility/gap status is PLAN33, not PLAN18.

## Fixed-n K-Scaling Snapshot (PLAN16)

- fixed `n=1000` K-scaling artifact:
  - `research/k_vs_arithmetic_axes_20260412/csv/plan16/PLAN16_k_scaling_n1000.csv`
- families covered: `K=2,4,5,10,20` (`seeds 0/1`, baseline + dense fastpath).
- concise findings:
  - `K=10` and `K=20` close exactly at Step 2 on both seeds;
  - dense fastpath improves runtime for those large-`K` dense families;
  - corrected K=2 rows show `g37` and `g810` close exactly at Step 3
    `profile_realization_dp_exact` when routed through the K=2 mainline path.

## Method provenance lookup

- Structured provenance table (current + continuity + archive tags):
  - `research/k_vs_arithmetic_axes_20260412/csv/CURRENT_METHOD_PROVENANCE.csv`
- Human-readable provenance guide:
  - `research/k_vs_arithmetic_axes_20260412/METHOD_PROVENANCE.md`

## K12 Anytime Incumbent and Parallel UB Recovery (PLAN32 + PLAN32B)

### PLAN32 (initial audit)
- Artifacts in `csv/plan32/`:
  - `PLAN32_existing_k12_incumbent_audit.csv` — 10 rows across 5 plans
  - `PLAN32_hard_k12_anytime_raw/compare/summary.csv` — gate evaluation
  - `PLAN32_k12_arithmetic_panel_raw/summary_by_family/summary_by_arithmetic.csv`
- Findings: 7/8 hard K12 rows have incumbents; 2 seeds (hardA s3, hardB s3) never recovered

### PLAN32B (parallel UB recovery) — INVALIDATED by PLAN32C
- Artifacts in `csv/plan32b/`:
  - `PLAN32B_parallel_initial_ub_debug.csv` — old vs new UB comparison
  - `PLAN32B_k12_no_incumbent_recovery_raw.csv` — recovery data
  - `PLAN32B_k12_no_incumbent_recovery_summary.csv` — calibration
  - `PLAN32B_k12_arithmetic_panel_completed.csv` — completed K12 panel (updated with PLAN32C data)
  - `PLAN32B_notes.md` — full decision note
- Code: `compute_parallel_initial_ub()` in stateful_dp_solver.cpp, anytime block moved before forward DP
- **INVALIDATED**: parallel UB uses M=2 machines; benchmark is single-machine (M=1). UB < LB for both target seeds.
- Parallel UB gated behind `PAST_ANYTIME_PARALLEL_UB_OPT_IN=1` (disabled by default, diagnostic-only)
- Calibrated gap: hardA ~7.7%, hardB ~0% (vs parallel seed-0 baseline) — these gaps are NOT valid for the benchmark

### PLAN32C (validity audit + K12 recovery) — Decision A

- PLAN32B parallel UB **INVALID**: changed model from 1-machine to 2-machine. UB < LB for both seeds.
- Consistency guard added: parallel UB opt-in only; LB-consistency reject at `done:`
- K12 s3 seeds **recovered under ORIGINAL single-machine model** (values from PLAN33 cert prepass):
  - hardA_k12 s3: UB=133544950, LB=133481433, gap=0.048%
  - hardB_k12 s3: UB=185849400, LB=185744893, gap=0.056%
- Method: PLAN33 cert prepass (5 trials + polish + semigroup LB certification)
- All K12 rows now have finite UB with gaps <0.1%
- Artifacts in `csv/plan32c/` (audit, recovery, notes)

### PLAN33 (certified anytime hard-K prepass) — Decision A (K10 + K12)

- Phase A+B complete (24 rows, K12 seeds 0-3 + K10 seeds 0-1)
- **All 12 plan33 rows cert_stop=1**, all gaps ≤ 0.0593%, all UB ≥ LB
- PLAN33 avg runtime 1396.61s vs PLAN32C 1527.11s (130.49s faster, with certified semigroup LB)
- Polish improved UB in all 12 rows
- hardA_k12 s3 panel corrected from 159M (stale, 5 trials) to 133.5M; hardB_k12 s3 also updated
- Initial run timeout fixed (redundant PAST_ANYTIME_INITIAL_UB removed)
- Artifacts in `csv/plan33/` (raw 24 rows, compare 12 rows, summary 14 metrics, notes)
- PLAN33 is the recommended hard-K default for tested K10/K12 hard rows
- Decision **A for K=10 and K=12**

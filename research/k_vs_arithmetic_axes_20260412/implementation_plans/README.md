# Implementation Plans

This subfolder is a handoff-oriented extension of the main
`k_vs_arithmetic_axes_20260412` archive.

Purpose:

- hold concrete implementation plans for the next coding cycles,
- keep those plans separate from the higher-level research framing in
  `PLAN.md`, `IDEAS.md`, and `EXPERT_GUIDANCE.md`,
- and make it easy to hand a single plan to another coder without forcing them
  to reconstruct the context from the whole archive.

Conventions:

- each plan should be self-contained,
- each plan should state:
  - objective,
  - exact files to inspect or edit,
  - experiments to run,
  - success criteria,
  - and fallback paths if the main path fails.
- plans should be ordered and named so later coders can execute them with
  minimal interpretation.

Current contents:

- `PLAN_01_recover_level2_baseline_and_finish_phase1_grid.md` (COMPLETED)
- `PLAN_02_dynamic_pricing_and_paper_consolidation.md` (Phase A done — pricing SKIPPED)
- `PLAN_02B_close_gaps_and_consolidate.md` (diagnostic archive)
- `PLAN_03_cleanup_and_pipeline_simplification.md` (COMPLETED — policy cleaned)
- `PLAN_03_step3_unified_profile_repair.md` (COMPLETED — beam is Step 3)
- `PLAN_03C_profile_realization_dp_unification.md` (conceptual — adopted)
- `PLAN_03D_exact_vs_beam_regime_selector.md` (conceptual)
- `PLAN_03E_beam_survivor_selection.md` (NEW — focused frontier-retention study)
- `PLAN_03F_restore_k2_and_mmkp_selector.md` (NEW — restore missing `K=2` exact profile repair and add structural exact-vs-beam selection)
- `PLAN_04_exact_dp_reactivation.md` (direction set)
- `PLAN_04A_incumbent_guided_exact_dp.md` (direction set)
- `PLAN_04B_admissible_pruning_proposals.md` (CURRENT — concrete pruning enhancements)
- `PLAN_04C_incumbent_and_pruning_matrix.md` (NEW — coupled Step-3/Step-4 implementation plan)
- `PLAN_05_paper_groups_extended_n_lambda.md` (NEW — extend the paper's 7 Table-2 processing-time groups to larger `n` and varied `lambda`)
- `PLAN_06_step3_restricted_master_prototype.md` (NEW — practical prototype of the Step-3 unified restricted-master theory before any full CG / branch-price implementation)
- `PLAN_07_k4_recovery_and_closure.md` (NEW — recover or replace the historical `K=4` energy-core + exact-guided closure path, then close the active `K=4` benchmark regime before broad extension resumes)
- `PLAN_08_energy_core_fortification_and_k_growth.md` (NEW — literature-grounded strengthening of the `energy_core` path before any full CG / branch-price implementation, with K=4 stabilization first and K=6 pilots second)
- `PLAN_09_k4_continuity_first_ablation.md` (NEW — continuity-first ablation plan: restore a continuity-safe direct-completion policy, isolate the regressing fortification features, and close the full active K=4 regime before any broader tuning resumes)
- `PLAN_10_k4_early_pruning_speedup.md` (NEW — continuity-safe speedup pass: keep the recovered K=4 package exact, then reduce runtime by optimizing pattern generation and phase-1 frontier retention before any broader method changes)
- `PLAN_11_paper_groups_n_scaling_and_group_local_optimization.md` (NEW — continue the paper-group `n` frontier after the K=4 generator fix, keep the accepted baseline stable, and allow only additive clearly labeled experimental optimizations when a specific group needs help)
- `PLAN_12_deep_cleanup_and_method_provenance_registry.md` (NEW — deep cleanup of the current thread surface plus a structured method-provenance registry so accepted rows point to concrete solver paths and code entrypoints)
- `PLAN_13_g12345678910_easy_family_recovery_and_g37_closure_push.md` (NEW — treat `{1..10}` and `g37` as different weak-family problems: recover easy-family behavior for `{1..10}` toward `n=5000–6000`, and run a separate exact-fallback diagnosis for `g37`, whose true exact frontier is only `n=600`)
- `PLAN_14_dense_unit_largek_step2_scaling.md` (NEW — diagnose why dense unit-containing `{1..10}` times out at `n=5000` despite Step-2 `ffd` closures through `n=3500`, then add an explicit experimental dense-unit Step-2 fast path as preparation for `{1..20}`)
- `PLAN_15_step3_adaptive_survivor_policy.md` (NEW — evidence-updated Step-3 survivor plan: stop corridor work, avoid stale quota/role priorities from PLAN_03E, and test uniform multiplicity, late ambiguity, residual-aware scoring, and a possible family-aware selector on hard irregular K=10)
- `PLAN_16_easy_vs_hard_k_scaling_story.md` (NEW — extend fixed-n K-scaling for easy contiguous-unit arithmetic beyond K=20, then present the contrast against already measured hard irregular K=8/10/12 boundaries)
- `PLAN_28_block_realizability_diagnostics_and_repair.md` (NEW — diagnose whether hard irregular K-axis failures come from recovered blocks that are relaxed-feasible but locally hard to realize; run bounded block repair only if diagnostics show signal)
- `PLAN_29_multiview_block_reconstruction.md` (NEW — test adjacent block coarsening views before Step 3 beam; remove selected boundaries rather than using failed PLAN28 local-feasibility diagnostics)
- `PLAN_31_fine_block_guided_beam_scoring.md` (NEW — after PLAN29 failed, keep fine blocks and improve Step-3 beam survivor scoring via residual-audit, family-aware selection, and coarse-lookahead scoring)
- `PLAN_32_k12_anytime_incumbent_and_arithmetic_panel.md` (NEW — make hard K=12 anytime-safe by returning a feasible incumbent, then test fixed-K arithmetic effects across several K=12 size families)

Plan 01 completed with Recommendation 2: Level 2 needs escalation.
Plan 02 Phase A ran a pool diagnostic, but it was tautological (beam
and Lagrangian share the same pattern pool by construction — see
errata below). Dynamic pricing skipped. Plan 02B replaces Phases B–D
with: self-contained exact Level 2 branch-and-bound (no external
solvers) to establish ground truth, then paper consolidation.

Errata:

- Plan 02 contains a factual error: it claims the beam "does not
  operate on a pre-enumerated pattern pool." This is FALSE — the
  beam calls `generate_energy_core_patterns` and searches the same
  fixed pool as the Lagrangian. Plan 02B corrects this. If reading
  Plan 02, treat it as superseded by Plan 02B on all points about
  beam architecture and pool-vs-search diagnostics.

Current strategic direction:

- `PLAN_02B` is useful as a diagnostic archive item, but it should NOT define
  the final method story because it introduces a second exact layer.
- The preferred long-term cleanup is now:
  - keep semigroup/profile recovery,
  - keep fast realization for easy arithmetic,
  - unify fixed-block DP and beam as one profile-realization DP family,
  - use exact fixed-block DP when the recovered-profile frontier is tractable,
  - use beam mode when the exact frontier is predicted to be too large,
  - and restore semigroup-guided exact DP as the only exact fallback.

Benchmark-facing direction now also includes:

- first, consolidate and extend the paper's own Section-5.2 processing-time
  groups (`7` groups) under the cleaned four-step pipeline,
- scale `n` first, then vary `lambda`/horizon slack,
- and only after that move to beyond-paper harder custom families.

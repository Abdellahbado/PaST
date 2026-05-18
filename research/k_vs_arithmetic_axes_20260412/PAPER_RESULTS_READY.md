# Paper Results Ready

This note is a compact paper-facing snapshot for the current accepted benchmark story.

## Reproducibility / code map

For HPC reruns and code provenance, use:

- `research/k_vs_arithmetic_axes_20260412/PAPER_HPC_REPRODUCIBILITY_MAP.md`

Local laptop CSVs are method-selection and provenance evidence. Final paper
runtimes should be regenerated on HPC from the runners and solver paths listed
in that map.

## PLAN33 certified anytime hard-K prepass (2026-04-30) — Decision A (K10 + K12)

Target: turn the serial initial UB into a certified prepass that early-stops when gap is already small, avoiding expensive beam/exact DP.

Artifacts:
- `research/k_vs_arithmetic_axes_20260412/csv/plan33/PLAN33_cert_anytime_raw.csv` (24 rows)
- `research/k_vs_arithmetic_axes_20260412/csv/plan33/PLAN33_cert_anytime_compare.csv` (12 head-to-head)
- `research/k_vs_arithmetic_axes_20260412/csv/plan33/PLAN33_cert_anytime_summary.csv` (14 metrics)
- `research/k_vs_arithmetic_axes_20260412/csv/plan33/PLAN33_notes.md`

Headline:
- All 12 plan33 rows early-stop with cert_stop=1, all gaps ≤ 0.0593%, all UB >= LB.
- PLAN33 avg runtime 1396.61s vs PLAN32C 1527.11s (130.49s faster, + certified semigroup LB).
- Polish improved UB in all 12 rows. PLAN33 uses 5 trials + polish; PLAN32C baseline used 75 trials.
- hardA_k12 s3 panel corrected from 159M (stale, 5 trials) to 133.5M; hardB_k12 s3 also updated.
- PLAN33 is the recommended hard-K default for tested K10/K12 hard rows.
- Decision A for both K=10 and K=12.

## PLAN32C K12 recovery note (2026-04-29) — corrected 2026-04-30

Target: recover the two hard K12 seeds that never had finite incumbents (hardA_k12 s3, hardB_k12 s3) under the original single-machine model.

Artifacts:
- `research/k_vs_arithmetic_axes_20260412/csv/plan32c/PLAN32C_hard_k12_final_panel.csv` — final 8-row K12 panel
- `research/k_vs_arithmetic_axes_20260412/csv/plan32c/PLAN32C_parallel_ub_validity_audit.csv` — 9-check audit
- `research/k_vs_arithmetic_axes_20260412/csv/plan32c/PLAN32C_k12_recovery_after_validity_check.csv` — recovery data
- `research/k_vs_arithmetic_axes_20260412/csv/plan32c/PLAN32C_notes.md` — technical notes
- `research/k_vs_arithmetic_axes_20260412/PLAN32C_K12_FINAL_NOTE.md` — paper-facing summary

Headline:
- PLAN32B parallel initial UB was invalidated: benchmark is single-machine (M=1); parallel UB used M=2, producing UB < single-machine LB.
- Serial `compute_initial_ub` (single-machine portfolio) together with PLAN33 certified prepass (5 trials + polish + semigroup LB) recovers both missing seeds with verified gaps.
- hardA_k12 s3: UB=133,544,950, LB=133,481,433, gap=0.048% (PLAN33)
- hardB_k12 s3: UB=185,849,400, LB=185,744,893, gap=0.056% (PLAN33)
- 8/8 hard K12 rows now have valid finite incumbents under original single-machine model. All gaps ≤0.056%.
- No exact closure (all rows finite-gap). Wider K12 arithmetic panel (medium families) still estimated, not run.

## PLAN19 K=10/12 redesign note (2026-04-24)

Target: test whether exact closure can be recovered at K=10/12 hard irregular via bounded additive redesigns.

Artifacts:
- `research/k_vs_arithmetic_axes_20260412/csv/plan19/PLAN19_k10_k12_redesign_raw.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan19/PLAN19_k10_k12_redesign_compare.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan19/PLAN19_k10_k12_failure_shift.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan19/PLAN19_k10_k12_method_notes.md`

Headline:
- K=10: no exact rows; finite-gap incumbents (~0.02-0.06%) are the practical ceiling.
- Historical PLAN19 K=12 status: mostly timeout/no-incumbent; occasional finite-gap incumbents before PLAN33 recovery.
- Exact fixed-block DP is structurally infeasible at K=10/12 under practical budgets (confirmed by `force_exact` with 1e12 guardrails hitting `skipped_comp_est`).
- Routing override justified: skip baseline `energy_core` for K>=10 hard irregular (saves 30-50% runtime, no quality loss).
- Stronger beam (`beam_plus`) disqualified: increases timeouts without improving gaps.

Current-use note:
- PLAN19 is retained as a negative redesign/closure study.
- For current hard K10/K12 paper-facing incumbent quality, use PLAN33 instead.

## PLAN18 fixed-n K-boundary refinement note (2026-04-24)

Controlled K-axis artifacts (fixed `n=1000`, `lambda=1.3`, seeds `0/1/2/3`):

- `research/k_vs_arithmetic_axes_20260412/csv/plan18/PLAN18_k_boundary_refine_n1000_raw.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan18/PLAN18_k_boundary_refine_best_of_route.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan18/PLAN18_k_boundary_refine_summary_by_k.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan18/PLAN18_k_boundary_refine_failure_signatures.csv`

PLAN18 headline:

- K=8: mixed exact vs finite-gap (2/4 exact on each hard irregular ladder);
- K=10: no exact rows; finite-gap incumbents dominate via additive reroute;
- Historical PLAN18 K=12 status: mostly budget-limited (timeout/no-incumbent) under 1200s/12GB cap, before PLAN33 recovery.

Refined boundary: exactness drops between K=8 and K=10. K=10 is the last K where finite-gap incumbents are usually produced. K=12 is beyond the current practical budget for hard irregular arithmetic at n=1000.

Dominant failure mode:
- K=10: `finite_gap_after_step4` via additive `profile_repair_beam/auto_v1`;
- K=12: `no_incumbent_timeout`.

Current-use note:
- PLAN18 is the boundary-detection study showing where exact closure degrades.
- It is superseded by PLAN33 for the current hard K12 finite-incumbent status.

## PLAN17 fixed-n K-axis boundary note (2026-04-23)

Controlled K-axis artifacts (fixed `n=1000`, `lambda=1.3`, seeds `0/1`):

- `research/k_vs_arithmetic_axes_20260412/csv/plan17/PLAN17_k_axis_n1000_raw.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan17/PLAN17_k_axis_n1000_summary_by_family.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan17/PLAN17_k_axis_n1000_summary_by_k.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan17/PLAN17_k_axis_boundary_classification.csv`

PLAN17 headline:

- easy unit-contiguous ladder is exact through `K=20` at fixed `n=1000`;
- hard irregular ladders are exact through `K=6`, with first degradation around `K=8`;
- irregular `K>=12` rows are mostly budget-limited in current budget (`900s`, `16 GB`), with one remaining finite-gap reroute row at `hardA_k12`.

Paper-facing interpretation from this pass:

- K alone is not the hardness axis; arithmetic structure controls whether larger K remains easy or enters the hard regime.
- use the corrected variant-separated PLAN17 summaries and boundary table; do not interpret mixed baseline+rereoute denominators from the original draft.


## Accepted method statement (current)

All accepted paper-group extension results are under:

- workflow: `ablation-stdin step1_exact_guided`
- baseline package: `energy_core + direct`
- accepted K=4 generator specialization from PLAN10:
  - DP-style generator active for K=4
  - K=4 signature-dedup default off

No baseline replacement was made in PLAN11.
PLAN14 introduces additive experimental behavior only (no silent baseline rewrite).

## Current paper-group frontiers (lambda=1.3, seeds 0/1)

- `g3567`: exact to `n=6000`; timeout at `n=7000`; `length_error` crash at `n=8000`.
- `g24`: exact to `n=10000`.
- `g12357`: exact to `n=8000`; timeout at `n=10000`.
- `g246810`: exact at `n=6000`; crash from `n=7000`.
- `g12345678910`:
  - baseline: exact to `n=3500`, timeout from `n=5000` in tested windows;
  - PLAN14 additive dense-unit fast-path: exact at `n=5000` on seeds `0/1` (Step 2, `UB=LB`).
- `g810`: exact to `n=5000`; crash from `n=6000`.
- `g37`:
  - old accepted ledger: exact through `n=600` only because later rows were
    misrouted;
  - current corrected evidence: PLAN13 reroute closes tested rows
    `n=750,1000,1500,2500,3500,5000` on seeds `0/1` through Step-3
    `profile_realization_dp_exact`.

## PLAN13 correction note (2026-04-21)

- A dedicated reroute pass showed that prior unresolved `g37` rows were run under
  non-mainline routing (`selector_reason=non_mainline_solver`), so they did not
  test the intended K=2 Step-3 exact profile-realization path.
- Under proper K=2 reroute (`selector_decision=exact`,
  `selector_reason=k2_exact_default`), `g37` closes at Step 3 with `UB=LB`
  through tested rows `n=750,1000,1500,2500,3500,5000` (seeds `0/1`).

## PLAN14 dense-unit recovery note (2026-04-22)

- Target family: `g12345678910 = {1,2,3,4,5,6,7,8,9,10}`.
- Baseline diagnosis confirmed:
  - exact Step-2 closure at `n=3500`;
  - baseline runtime-window timeout at `n=5000` (seeds `0/1`) with no usable
    incumbent row emitted on those failures.
- Additive toggle-gated recovery:
  - `PAST_DENSE_UNIT_STEP2_FASTPATH=1` routes dense unit-containing large-`K`
    rows to early Step-2 `ffd` closure path;
  - `{1..10}` closes exactly at `n=5000` on seeds `0/1` (`UB=LB`).
- Additional additive variant:
  - `PAST_COUNT_BASED_FFD=1` also closes `{1..10}` exactly at `n=5000`
    (seeds `0/1`) in PLAN14 comparisons.

## PLAN16 fixed-n K-scaling note (2026-04-22)

- Priority pivot applied: fixed `n=1000`, vary group `K` across current families.
- Artifact:
  - `research/k_vs_arithmetic_axes_20260412/csv/plan16/PLAN16_k_scaling_n1000.csv`
- Rows run:
  - families `g24`, `g37`, `g810`, `g3567`, `g12357`, `g246810`,
    `g12345678910`, `g1234567891011121314151617181920`
  - seeds `0/1`
  - variants `baseline` and `dense_step2_fastpath`
- Key outcomes:
  - `K=10` and `K=20` close exactly at Step 2 on both seeds;
  - dense fastpath improves mean runtime at `K=10` and `K=20`;
  - `g37` and `g810` also close exactly at this `n`, but through the K=2
    Step-3 exact profile-realization path rather than Step 2.

## Primary evidence artifacts

- baseline and accepted extension ledgers:
  - `research/k_vs_arithmetic_axes_20260412/csv/plan05/PAPER_GROUPS_PLAN05_n_extension.csv`
  - `research/k_vs_arithmetic_axes_20260412/csv/plan11/PAPER_GROUPS_PLAN11_n_extension.csv`
- PLAN14 dense-unit artifacts:
  - `research/k_vs_arithmetic_axes_20260412/csv/plan14/PLAN14_g12345678910_diagnosis.csv`
  - `research/k_vs_arithmetic_axes_20260412/csv/plan14/PLAN14_g12345678910_checkpoint_probe.csv`
  - `research/k_vs_arithmetic_axes_20260412/csv/plan14/PLAN14_g12345678910_fastpath_compare.csv`
  - `research/k_vs_arithmetic_axes_20260412/csv/plan14/PLAN14_dense_unit_1_20_smoke.csv`
- summary index:
  - `research/k_vs_arithmetic_axes_20260412/PAPER_GROUPS_EXTENSION_SUMMARY.md`

## Method provenance lookup

- `research/k_vs_arithmetic_axes_20260412/csv/CURRENT_METHOD_PROVENANCE.csv`
- `research/k_vs_arithmetic_axes_20260412/METHOD_PROVENANCE.md`

## Current blocker sentence

The frontier is now best characterized as (i) high-`n` robustness/runtime issues
in generic mode (`length_error` and timeouts), and (ii) exact-closure limits on
hard irregular large-`K` rows. Feasible incumbents are no longer the main hard-K
blocker after PLAN33; the remaining issue is proving exact optimality or
reproducing the certified finite-gap panel on HPC.

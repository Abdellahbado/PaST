# PLAN 32 — K=12 Anytime Incumbent and Arithmetic-Panel Study

## Objective

Fix the unacceptable `K=12` failure mode:

> Hard `K=12` rows must return a feasible incumbent, even when exact proof or the full Step-3 beam does not finish.

This plan has two linked goals:

1. **Method goal**: add an incumbent-first / anytime safety layer so `UB=-1` does not happen on hard `K=12` rows.
2. **Study goal**: test the method on several different `K=12` job-size families, not only the existing hardA/hardB families, so we can isolate arithmetic effects at fixed `K`.

This is not an exact-closure plan. Exact closure is secondary.

## Required Read Order

Read these first:

1. `research/k_vs_arithmetic_axes_20260412/ACTIVE.md`
2. `research/k_vs_arithmetic_axes_20260412/iterations/20260429_k12_anytime_incumbent_panel/SUMMARY.md`
3. `research/k_vs_arithmetic_axes_20260412/METHOD_BOUNDARIES.md`
4. `research/k_vs_arithmetic_axes_20260412/csv/plan18/PLAN18_k_boundary_refine_failure_signatures.csv`
5. `research/k_vs_arithmetic_axes_20260412/csv/plan19/PLAN19_k10_k12_method_notes.md`
6. `research/k_vs_arithmetic_axes_20260412/csv/plan22b/PLAN22B_ambig_scoreband_validation_notes.md`
7. `research/k_vs_arithmetic_axes_20260412/csv/plan31/PLAN31_fine_block_guided_beam_notes.md`
8. Code:
   - `solvers/cpp/stateful_dp_solver.hpp`
   - `solvers/cpp/stateful_dp_solver.cpp`
   - `solvers/cpp/stateful_compare.cpp`
   - `research/k_vs_arithmetic_axes_20260412/run_plan31_family_aware_survivor.py`
   - `research/k_vs_arithmetic_axes_20260412/run_plan17_k_axis_n1000.py`
   - `research/k_vs_arithmetic_axes_20260412/run_plan13_two_track_recovery.py`

## Evidence This Plan Uses

| Evidence | Meaning for PLAN32 |
|---|---|
| PLAN18/19: some `hardA_k12` and `hardB_k12` rows produce finite gaps around `0.02%–0.05%` | Good incumbents are possible at `K=12`; the method is not hopeless |
| PLAN18/19: some `hardB_k12` rows return `UB=-1` / timeout | The immediate blocker is lack of anytime incumbent, not exact proof |
| Energy-core baseline on high-K hard rows often gives `selector_bypass` / no incumbent | Do not use energy-core as the first hard-K route |
| PLAN19 `beam_plus` times out more and does not improve gaps | Do not simply widen the beam |
| PLAN24/24B/25/26 corridor work failed structurally | Do not rely on exact corridor or local-corridor DP |
| PLAN29 coarsening failed | Keep fine blocks; do not replace the block sequence |
| PLAN31 family-aware survivor improves hard `K=10` | Use it as a K12 candidate, but judge by incumbent coverage first |
| Existing code already has `compute_initial_ub`, `solve_fixed_sequence`, and `local_search_ub` | Build the safety layer on existing UB tools first |

## Hard Constraints

- No broad solver redesign.
- No Step-4 corridor revival.
- No global `smart_reconstruct(...)` revival as the main path.
- Do not silently change accepted defaults.
- All new behavior must be behind explicit env toggles.
- One heavy solver process at a time.
- Memory cap: 16 GB default, 20 GB absolute maximum if explicitly justified.
- Every row must flush immediately to CSV.
- The primary success metric is feasible incumbent coverage, not exactness.

## Main Method: Incumbent-First Safety Layer

Add a hard-K anytime contract:

> Once the solver has enough data to build a valid sequence-based schedule, it must compute and store a feasible incumbent before entering any long Step-3 or Step-4 work.

### Existing functions to reuse

The code already exposes:

- `dp::compute_initial_ub(...)`
- `dp::solve_fixed_sequence(...)`
- `dp::local_search_ub(...)`
- `dp::bin_packing_ub(...)`

Use these first. Do not write a large new solver unless the existing functions cannot produce a valid UB.

### Required behavior

When enabled:

- run a cheap initial UB portfolio early;
- store:
  - `anytime_initial_ub`
  - `anytime_initial_ub_source`
  - `anytime_time_to_first_ub`
  - `anytime_initial_ub_valid`
- pass this UB as incumbent into later beam/exact stages when possible;
- if beam/exact times out or fails, return this UB instead of `-1`;
- mark the row as feasible but non-optimal unless `UB=LB`.

### Required env toggles

Use names like these:

- `PAST_ANYTIME_INITIAL_UB=1`
- `PAST_ANYTIME_INITIAL_UB_TRIALS=<int>`
- `PAST_ANYTIME_INITIAL_UB_TIME_LIMIT=<sec>`
- `PAST_ANYTIME_INITIAL_UB_LOCAL_SEARCH=1`
- `PAST_ANYTIME_RETURN_ON_TIMEOUT=1`
- `PAST_ANYTIME_HARDK_ONLY=1`

Optional, only if cheap:

- `PAST_ANYTIME_RESIDUE_AWARE=1`
- `PAST_ANYTIME_PRICE_AWARE=1`

Do not turn this on globally by default.

## Constructive UB Portfolio

Start with existing sequence methods:

| Heuristic | Implementation idea |
|---|---|
| SPT | current `compute_initial_ub` ascending sequence |
| LPT | current descending sequence |
| alternating short/long | current alternating sequence |
| random shuffles | current random trials |
| local search | current `local_search_ub` on best sequence |

Add only if low-risk:

| Heuristic | Purpose |
|---|---|
| residue-aware type order | prefer type orderings that leave fillable residuals |
| price-aware ordering | group long work toward cheaper time windows if sequence DP benefits |

Important: the first implementation can be simple. The key is to guarantee a valid UB quickly.

## Anytime Beam Checkpointing

Only after the initial UB layer works.

Goal:

> If `profile_repair_beam` sees a complete candidate before timing out, preserve it immediately.

Required diagnostics:

- `anytime_beam_best_ub`
- `anytime_beam_best_ub_time`
- `anytime_beam_completed_candidates`
- `anytime_beam_returned_on_timeout`

If this is hard to implement, do not block Phase A. The initial UB layer is mandatory; beam checkpointing is the second improvement.

## K=12 Arithmetic Panel

All rows:

- `n=1000`
- `lambda=1.3`
- seeds `0,1,2,3`
- `K=12`

Use exactly these families unless there is a strong reason to change:

| Family id | Sizes | Purpose |
|---|---|---|
| `k12_unit` | `{1,2,3,4,5,6,7,8,9,10,11,12}` | easy control |
| `k12_dense_no1` | `{2,3,4,5,6,7,8,9,10,11,12,13}` | tests missing `1` with dense arithmetic |
| `k12_even_structured` | `{2,4,6,8,10,12,14,16,18,20,22,24}` | structured gcd-2 arithmetic |
| `hardA_k12` | `{2,3,4,5,7,11,13,17,19,23,29,31}` | existing hard family with size `2` |
| `hardB_k12` | `{3,5,7,11,13,17,19,23,29,31,37,41}` | existing hard family without `2` |
| `k12_sparse_gap` | `{4,7,11,16,23,31,42,55,67,79,91,103}` | sparse / large residual gaps |

This panel should show whether `K=12` is hard because of `K`, or because of arithmetic structure.

## Experiment Phases

### Phase 0 — Existing K12 incumbent audit

Before code changes, extract current K12 evidence from PLAN18/19/22B.

Create:

- `csv/plan32/PLAN32_existing_k12_incumbent_audit.csv`
- `csv/plan32/PLAN32_existing_k12_incumbent_audit_notes.md`

Answer:

1. Which K12 rows already have finite UB?
2. Which rows have `UB=-1`?
3. Which route/policy produced finite UB most reliably?
4. Which rows are the true no-incumbent targets?

### Phase 1 — Implement initial UB safety layer

Implement the env-gated initial UB layer.

Smoke rows:

- `hardB_k12`, seed `1`
- `hardB_k12`, seed `3`

These are the most important because they previously returned no incumbent.

Smoke variants:

1. `current_profile_beam`
2. `initial_ub_only`
3. `initial_ub_plus_profile_beam`

Create:

- `csv/plan32/PLAN32_initial_ub_smoke.csv`

Gate to continue:

- both hardB smoke rows return finite UB with `initial_ub_only`;
- `time_to_first_ub` is recorded;
- no memory kill;
- no false exact claim.

If this fails, stop and debug the initial UB layer. Do not proceed to larger panel.

### Phase 2 — Hard K12 method gate

Rows:

- `hardA_k12`, seeds `0,1,2,3`
- `hardB_k12`, seeds `0,1,2,3`

Variants:

| Variant | Purpose |
|---|---|
| `current_profile_beam` | current useful hard-K route |
| `initial_ub_only` | guaranteed safety baseline |
| `initial_ub_plus_profile_beam` | safety + current beam |
| `initial_ub_plus_family_aware_beam` | safety + PLAN31-style survivor policy |
| `initial_ub_plus_anytime_beam` | safety + beam checkpointing, if implemented |

Create:

- `csv/plan32/PLAN32_hard_k12_anytime_raw.csv`
- `csv/plan32/PLAN32_hard_k12_anytime_compare.csv`
- `csv/plan32/PLAN32_hard_k12_anytime_summary.csv`

Gate:

- `8/8` hard rows have finite UB under at least one PLAN32 variant;
- no `UB=-1` for the promoted variant;
- gap under `2%` preferred;
- gap under `1%` strong;
- runtime to first UB under `180s` preferred;
- memory under 16 GB preferred, under 20 GB required.

### Phase 3 — K12 arithmetic panel

Run the best one or two variants from Phase 2 on all six K12 families.

Do not run every failed variant on every family.

Required variants:

1. `initial_ub_plus_profile_beam`
2. `initial_ub_plus_family_aware_beam` if Phase 2 supports it

Create:

- `csv/plan32/PLAN32_k12_arithmetic_panel_raw.csv`
- `csv/plan32/PLAN32_k12_arithmetic_panel_summary_by_family.csv`
- `csv/plan32/PLAN32_k12_arithmetic_panel_summary_by_arithmetic.csv`

Panel success:

- feasible UB coverage is 100%;
- easy/structured families should show small gaps or exact closure;
- hard/sparse families should still return feasible UB, even if gaps are larger;
- arithmetic ranking should be interpretable.

### Phase 4 — Decision note

Create:

- `csv/plan32/PLAN32_k12_anytime_incumbent_notes.md`

The note must answer:

1. Does PLAN32 eliminate `UB=-1` on hard K12?
2. What is the time-to-first-UB?
3. How large are the gaps?
4. Does family-aware beam help at K12?
5. Which K12 arithmetic families are easy, medium, hard?
6. Should the method be promoted as an anytime hard-K mode?

## Required Documentation Updates

Update:

- `research/k_vs_arithmetic_axes_20260412/LOG.md`
- `research/k_vs_arithmetic_axes_20260412/RESULTS.md`
- `research/k_vs_arithmetic_axes_20260412/BLOCKERS.md`
- `research/k_vs_arithmetic_axes_20260412/METHOD_BOUNDARIES.md`
- `research/k_vs_arithmetic_axes_20260412/CURRENT_RESULTS_INDEX.md`
- `research/k_vs_arithmetic_axes_20260412/PRESENTATION_K_N_SCALING_COMPREHENSIVE.md`
- `research/k_vs_arithmetic_axes_20260412/iterations/20260429_k12_anytime_incumbent_panel/SUMMARY.md`
- `research/k_vs_arithmetic_axes_20260412/iterations/20260429_k12_anytime_incumbent_panel/RESULTS.md`
- `research/k_vs_arithmetic_axes_20260412/iterations/20260429_k12_anytime_incumbent_panel/BLOCKERS.md`

## Decision Labels

Use exactly one:

- `A`: promote anytime hard-K mode; feasible UB coverage achieved with acceptable gaps
- `B`: keep as optional fallback; feasible coverage achieved but gaps/runtime are weak
- `C`: diagnostic only; coverage or gap quality insufficient
- `D`: abandon direction due to structural blocker
- `E`: implementation incomplete or evidence invalid; rerun required

## Expected Paper Story If Successful

If PLAN32 succeeds, the paper can say:

> Exact proof becomes difficult for hard irregular `K=12`, but the solver remains practically useful because an incumbent-first beam mode returns feasible schedules with controlled gaps. The fixed-`K=12` arithmetic panel shows that difficulty is driven by arithmetic structure, not by `K` alone.


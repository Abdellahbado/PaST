# PLAN 07 — K=4 Recovery And Closure

Date: 2026-04-16

Status: NEW

## 1. Purpose

This plan is a targeted `K=4` recovery plan.

We previously reported that the method closed meaningful `K=4` frontier rows
through at least `n=5000`. That historical result must either be faithfully
recovered and generalized, or the gap must be closed again with a controlled
replacement path.

This plan is **not** open-ended method research. It is a bounded recovery task
with a primary path and limited fallback paths.

## 2. What We Know Already

### Historical closure path

The old established `K=4` closure path is:

- Step 1 incumbent from `block_repair_energy_core`
- then `step1_exact_guided` closes

This is documented in:

- [research/large_k_large_n_attempt_20260409/RESULTS.md](/Users/mac/Documents/Study/PFE/PaST/research/large_k_large_n_attempt_20260409/RESULTS.md)
- [research/large_k_large_n_attempt_20260409/LOG.md](/Users/mac/Documents/Study/PFE/PaST/research/large_k_large_n_attempt_20260409/LOG.md)

Representative rows:

- `paperext_profile_repair_smallk_nscale_plus_20260409/0009_profile_smallk_3567_plus_n3500_s1`
- `paperext_profile_repair_smallk_nscale_plus_20260409/0011_profile_smallk_3567_plus_n5000_s1`

### Important caution

Recent negative revalidation is **not yet trustworthy** as proof that the old
closure is gone, because:

- the rerun did not clearly reproduce the exact historical payloads,
- the objective scale did not match the historical rows,
- and the effective exact-guided policy was not fully verified as apples-to-apples.

So the first task is not to speculate. It is to reconstruct the historical
winning path faithfully.

## 3. Main Objective

Recover a practical `K=4` policy that closes the active `K=4` benchmark rows,
not just one hand-picked row.

Success means:

1. the old `3567_plus` closure path is either faithfully recovered or replaced
   by a stronger current path,
2. the current paper-group `g3567 = {3,5,6,7}` rows are closed through the
   tested extension range,
3. the final `K=4` policy is explicit, reproducible, and documented,
4. broad benchmark work does not resume until `K=4` is actually closed.

## 4. Scope Of The K=4 Regime

This plan treats the following as the active `K=4` regime:

### Historical extension family

- `3567_plus`
- at minimum:
  - `n = 1500, 2000, 2500, 3000, 3500, 5000`
  - both seeds if available from the original suite

### Paper-facing family

- `g3567 = {3,5,6,7}`
- at minimum:
  - `n = 600, 750, 1000, 1500, 2500, 3500, 5000`
  - `lambda = 1.3`
  - current Plan-05 seed policy

If `lambda=1.3` closes cleanly, then extend to the representative Plan-05
lambda slice for `g3567`.

## 5. Files To Inspect First

- [research/large_k_large_n_attempt_20260409/RESULTS.md](/Users/mac/Documents/Study/PFE/PaST/research/large_k_large_n_attempt_20260409/RESULTS.md)
- [research/large_k_large_n_attempt_20260409/LOG.md](/Users/mac/Documents/Study/PFE/PaST/research/large_k_large_n_attempt_20260409/LOG.md)
- [research/large_k_large_n_attempt_20260409/BLOCKERS.md](/Users/mac/Documents/Study/PFE/PaST/research/large_k_large_n_attempt_20260409/BLOCKERS.md)
- [research/k_vs_arithmetic_axes_20260412/RESULTS.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/RESULTS.md)
- [research/k_vs_arithmetic_axes_20260412/LOG.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/LOG.md)
- [research/k_vs_arithmetic_axes_20260412/METHOD_BOUNDARIES.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/METHOD_BOUNDARIES.md)
- [research/k_vs_arithmetic_axes_20260412/BLOCKERS.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/BLOCKERS.md)
- [research/k_vs_arithmetic_axes_20260412/csv/plan05/K4_energy_core_recovery_comparison_20260416.csv](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/csv/plan05/K4_energy_core_recovery_comparison_20260416.csv)
- [hpc/benchmark_extensions/build_profile_repair_suites.py](/Users/mac/Documents/Study/PFE/PaST/hpc/benchmark_extensions/build_profile_repair_suites.py)
- [research/k_vs_arithmetic_axes_20260412/run_plan05_paper_groups_extension.py](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/run_plan05_paper_groups_extension.py)
- [solvers/cpp/stateful_compare.cpp](/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/stateful_compare.cpp)
- [solvers/cpp/stateful_dp_solver.cpp](/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/stateful_dp_solver.cpp)

## 6. Primary Path

### Phase A — Reconstruct the historical `K=4` winning path faithfully

Do **not** start by changing the solver.

First reconstruct the exact historical conditions as closely as possible:

1. identify the original `3567_plus` generator and seed policy,
2. confirm the EC configuration used on the historical rows,
3. confirm the exact `step1_exact_guided` behavior at that time,
4. verify the small-`K` exact-guidance/completion policy that restored the
   `K=4` checkpoint,
5. rerun the historical `3567_plus` anchors in an apples-to-apples way.

Required result:

- the rerun must match the **objective scale** of the archive rows before any
  regression claim is accepted.

If the objective scale does not match:

- stop calling it a historical revalidation,
- diagnose payload/policy mismatch first.

### Phase B — Recover the historical path on current code

Once the historical payloads/settings are matched, restore the old effective
path on current code:

- Step 1:
  - `block_repair_energy_core`
- then:
  - `step1_exact_guided`
- with the cheap small-`K` guidance regime that historically restored the `K=4`
  checkpoint

Required outcome:

- `3567_plus n=3500`
- `3567_plus n=5000`

must close again if the old method still exists in substance.

### Phase C — Generalize from `3567_plus` to paper-group `g3567`

After the historical path is trusted again, test it on:

- `g3567 n=1000`
- `g3567 n=1500`
- `g3567 n=2500`
- `g3567 n=3500`
- `g3567 n=5000`

The goal is not just to show one nice row. The goal is to make this the real
`K=4` method if it works broadly.

## 7. Allowed Fallback Paths

If the pure historical path does not close all active `K=4` rows, use only the
following controlled fallback directions.

### Fallback 1 — Strengthen the Step-1 energy-core incumbent

Allowed actions:

- widen or retune the energy-core pattern pool in a controlled way,
- improve the core-generation thresholds,
- add a small exact polish on the restricted core,
- keep memory bounded and `K=4`-specific if needed.

Not allowed:

- arbitrary large pattern explosion,
- broad solver redesign,
- pushing memory until the laptop dies.

### Fallback 2 — `K=4` hybrid Step-3 policy

Allowed path:

- `energy_core` incumbent first,
- then exact profile realization or exact-guided continuation,
- then beam only as a bounded fallback if exact/core fails to close,
- then Step 4 only as the last resort.

This is still one controlled `K=4` policy, not a method zoo.

### Fallback 3 — Restricted-master augmentation for `K=4` only

If fixed energy-core pools are the bottleneck, the only allowed “new” direction
in this plan is a **small K=4-only augmentation loop**:

- start from the energy-core pattern pool,
- add a bounded number of extra missing patterns,
- rerun the exact/core solve,
- measure whether closure improves.

This is acceptable because it is a limited version of the Step-3 restricted
master idea, not a full new research branch.

Do **not** implement full column generation or branch-and-price in this plan.

### Fallback 4 — K=4-specific exact guidance corrections

If Step 1 is already giving a strong incumbent but exact still fails to close,
allowed exact-stage fixes are:

- restore the historical cheap-suffix guidance for `K<=4`,
- remove or gate exact-stage behavior that harms small `K`,
- improve incumbent handoff and exact-start conditions for `K=4`.

Do not turn this into a general Step-4 redesign task.

## 8. What Must Not Happen

Do not:

- continue broad Plan-05 extension while `K=4` is open,
- accept non-apples-to-apples reruns as proof of regression,
- rely on external timeout comparisons as final evidence,
- add local search,
- add exact-L2 back into the mainline,
- jump straight to full column generation,
- use “finite tiny gap” as a stopping condition for this plan.

This plan ends only when the targeted `K=4` rows are closed or there is a
precisely documented, evidence-backed blocker.

## 9. Required Experiments

### Experiment Group 1 — Historical revalidation

Run the full historical `3567_plus` suite if feasible; otherwise at minimum:

- `n=3500, s=1`
- `n=5000, s=1`

Record:

- payload identity / generator provenance
- runtime
- UB
- LB
- gap
- `fwd_pack_method`
- exact-guided diagnostics
- deciding step

### Experiment Group 2 — Active paper-family closure

Run:

- `g3567`: `n=600, 750, 1000, 1500, 2500, 3500, 5000`

Record the same fields, plus:

- whether the method was:
  - historical path,
  - tuned energy-core,
  - hybrid exact/core,
  - or bounded augmentation

### Experiment Group 3 — Head-to-head policy table

At minimum compare on the active hard rows:

- current default
- historical energy-core path
- tuned energy-core path if different
- hybrid path if introduced

The winner must be chosen by:

- exact closure first,
- then gap,
- then runtime,
- then stability / reproducibility.

## 10. Success Criteria

This plan succeeds only if all of the following are true:

1. the old `3567_plus` path is either faithfully recovered or conclusively
   replaced by a stronger current `K=4` path,
2. the active `g3567` `n`-extension rows are closed through the tested range,
3. the final `K=4` policy is explicit and reproducible,
4. the archive clearly distinguishes:
   - historical evidence,
   - recovered current evidence,
   - and any remaining true blocker.

## 11. Required Archive Updates

Update:

- [research/k_vs_arithmetic_axes_20260412/LOG.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/LOG.md)
- [research/k_vs_arithmetic_axes_20260412/RESULTS.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/RESULTS.md)
- [research/k_vs_arithmetic_axes_20260412/BLOCKERS.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/BLOCKERS.md)
- [research/k_vs_arithmetic_axes_20260412/METHOD_BOUNDARIES.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/METHOD_BOUNDARIES.md)

Also create a dedicated note if useful:

- `K4_RECOVERY_NOTE.md`

## 12. Final Deliverable

Do not return with:

- only partial recovery,
- only finite-gap rows,
- or only speculative diagnosis.

Return only when you have:

1. reconstructed the historical `K=4` path faithfully,
2. either recovered it or replaced it with a stronger current path,
3. closed the active `K=4` rows in the benchmark regime,
4. documented the final `K=4` policy clearly,
5. and stated whether broad benchmark work can resume.

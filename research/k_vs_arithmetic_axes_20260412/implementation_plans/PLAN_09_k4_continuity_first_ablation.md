# PLAN 09 — K=4 Continuity-First Ablation And Full Closure

Date: 2026-04-18

Status: NEW

## 1. Purpose

The recent probe work shows that the current fortified `energy_core` package is
not uniformly bad. It is failing for a more specific reason:

- the hard `K=4` continuity rows become exact again when the direct-completion
  cap is raised,
- while center/diversification changes appear to be secondary contributors.

So the next task is **not** broad tuning.
It is:

1. restore a continuity-safe direct-completion policy,
2. isolate which fortification features are safe,
3. recover exact closure on the full active `K=4` regime,
4. and only then continue performance tuning.

## 2. What The Probe Evidence Already Says

On the hard continuity row `3567_plus n=3500 s=1`:

- baseline fortified package:
  - finite gap
  - Step 4 used
- `direct500m`:
  - exact
  - Step 4 not used
- `direct950m`:
  - exact
  - Step 4 not used

Secondary signals:

- `center0` improves the gap,
- `center0_nodiv` improves it further,
- `nodiversify` alone also improves it,
- `two_phase_off` is faster but still not exact,
- `keep128` and larger expansion did not rescue closure.

This means:

- the **main blocker** is the current direct-completion guard/cap policy,
- the **secondary suspects** are the new center/diversification choices,
- and the exact-core stage itself is still not the dominant problem.

## 3. Main Objective

Close **all active `K=4` rows** again under a clean, reproducible package.

That means:

### Historical continuity family

- `3567_plus`
- required:
  - `n = 3500, 5000`
  - seeds `0,1`

### Paper-facing family

- `g3567 = {3,5,6,7}`
- required:
  - `n = 600, 750, 1000, 1500, 2500, 3500, 5000`
  - seeds `0,1`
  - `lambda = 1.3`

This plan does **not** end at “finite tiny gaps.”
It ends only when these rows are closed or a precise measured blocker is
established.

## 4. Required Order Of Work

### Phase A — Restore a continuity-safe direct-completion policy

Before testing secondary knobs, recover the direct-completion behavior that
actually returns exact closure on the hard continuity rows.

Required tasks:

1. identify the smallest safe direct-completion cap/policy that recovers exact
   continuity on:
   - `3567_plus n=3500 s=1`
   - `3567_plus n=5000 s=1`
2. extend the same check to:
   - `s=0`
3. convert that from a one-off probe into a real policy:
   - either a K=4-specific cap,
   - or an adaptive cap rule,
   - or a continuity-safe direct/cheap selection rule

Do **not** leave the fix as a manual probe-only environment override.

### Phase B — Controlled secondary ablation

Once direct-completion continuity is restored, test the secondary suspects on
top of the restored cap:

1. blended center vs old center
2. diversification on/off
3. adaptive delta widening on/off
4. two-phase on/off

The objective is not to keep all new features.
The objective is to find the strongest package that still preserves exact K=4
closure.

### Phase C — Rebuild the final K=4 package

After the ablation, choose one final K=4 package:

- continuity-safe direct completion
- best safe center choice
- best safe diversification setting
- best safe delta policy
- best safe phase-1/two-phase behavior

This package must be:

- explicit,
- reproducible,
- and benchmark-facing.

### Phase D — Full K=4 closure rerun

Run the full active K=4 regime with the final chosen package:

#### Historical continuity
- `3567_plus n=3500,5000`, seeds `0,1`

#### Paper group
- `g3567 n=600,750,1000,1500,2500,3500,5000`, seeds `0,1`, `lambda=1.3`

Do not stop early after the easy rows.

## 5. Allowed Methods And Fallbacks

### Allowed

- K=4-specific direct-completion cap/policy
- K=4-specific gating for center/diversification if justified
- preserving instrumentation and safety observability
- keeping the useful direct-table safety guard, if continuity-safe

### Allowed only if continuity is restored first

- runtime tuning of pattern generation
- phase-1 tuning
- light K=4-only cleanup of pattern retention

### Not allowed yet

- broader K>4 transfer work
- pricing-lite / augmentation
- full column generation / branch-price
- local search
- exact-L2 return to mainline

## 6. Required Measurements

For every ablation package and every required row record:

- exact / not exact
- runtime
- UB
- LB
- gap
- Step 4 used or not
- `fwd_ec_time_pattern_generation`
- `fwd_ec_time_phase1`
- `fwd_ec_time_exact_core`
- `fwd_ec_delta_used`
- generated / retained patterns
- direct-completion policy actually chosen

## 7. Decision Rule

The best package is chosen by:

1. exact closure on **all active K=4 rows**
2. then runtime
3. then seed stability
4. then simplicity / robustness of the policy

Any package that loses exact closure on required `K=4` rows is disqualified,
even if it is faster.

## 8. Required Deliverables

Create or update a dedicated ablation artifact such as:

- `research/k_vs_arithmetic_axes_20260412/csv/plan08/PLAN09_k4_continuity_ablation.csv`

and update:

- [research/k_vs_arithmetic_axes_20260412/LOG.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/LOG.md)
- [research/k_vs_arithmetic_axes_20260412/RESULTS.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/RESULTS.md)
- [research/k_vs_arithmetic_axes_20260412/BLOCKERS.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/BLOCKERS.md)
- [research/k_vs_arithmetic_axes_20260412/METHOD_BOUNDARIES.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/METHOD_BOUNDARIES.md)
- [research/k_vs_arithmetic_axes_20260412/archive_20260421/markdown/k4_history/ENERGY_CORE_FORTIFICATION_NOTE.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/archive_20260421/markdown/k4_history/ENERGY_CORE_FORTIFICATION_NOTE.md)

If helpful, also create:

- `K4_CONTINUITY_RECOVERY_NOTE.md`

## 9. Success Criteria

This plan succeeds only if:

1. direct-completion continuity is restored as a real policy,
2. the regression-causing feature set is identified,
3. a final continuity-safe K=4 package is selected,
4. all active K=4 rows are closed again,
5. and only then is the path reopened for later performance tuning or larger-K
   transfer work.

## 10. Final Principle

Do not optimize on top of a regressed method.

First:
- recover exact `K=4`

Then:
- tune runtime

Not the other way around.

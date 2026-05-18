# PLAN 13 — `{1..10}` Easy-Family Recovery and `g37` Closure Push

## Objective

Focus the next benchmark effort on the two underperforming paper groups:

1. `g12345678910 = {1..10}`
2. `g37 = {3,7}`

But do **not** treat them as one generic “weak group” problem.

They are weak for different reasons:

- `{1..10}` is an easy-arithmetic family that is currently underperforming.
- `g37` is a genuine closure-limited family.

So this plan has two separate tracks.

## Why this split is necessary

### A. `{1..10}` is suspiciously weak, not obviously intrinsically weak

Current evidence in this thread says:

- `{1..10}` contains `1`,
- it is contiguous,
- earlier notes explicitly classify it as easy arithmetic,
- and it is Step-2/FFD exact through `n=3500`.

Current paper-group ledger:

- exact through `n=3500`,
- then timeout at `n=5000`,
- and timeout again at `n=6000,7000`.

That pattern suggests:

- not “family is fundamentally hard at `n=5000`,”
- but “workflow/policy is failing to capitalize on an easy arithmetic family.”

This is the best candidate for a recovery-style benchmark improvement.

### B. `g37` is a real closure problem

Current ledger says:

- exact only through `n=600`,
- `n=750,1000` already fail in sparse exact,
- `n=1500,2500,3500,5000` enter Step 4 but do not close,
- `n=6000,7000` remain unresolved.

That is not the same as `{1..10}`.

So for `g37` the task is not “recover easy-family behavior.”
It is:

- push a real closure frontier.

## Scope

This task is limited to:

- `{1..10}` paper-family recovery to at least `n=5000`, ideally `n=6000`
- `g37` exact-fallback diagnosis and, if possible, closure recovery beyond `n=600`

Do not broaden this into:

- all-family extension,
- robustness/crash debugging on unrelated groups,
- column generation,
- or global solver redesign.

## Hard constraints

- Keep the accepted current baseline reproducible.
- No silent policy rewrite.
- Any new idea must be additive and explicitly experimental.
- If an experiment is tested, baseline and experiment must be compared directly.

## Track A — `{1..10}` easy-family recovery

### Hypothesis

The current timeout at `n>=5000` is more likely a policy/workflow miss than a
fundamental hardness boundary.

Reason:

- easy arithmetic,
- contiguous lengths,
- `1` present,
- Step 2 / FFD already dominant and exact through `n=3500`.

### Goal

Get `{1..10}` to:

- at least `n=5000`,
- ideally `n=6000`,

under the accepted pipeline story.

### Required diagnostic order

1. Re-run `{1..10}` `n=5000` under the current accepted baseline with detailed
   diagnostics.
2. Determine where the runtime is actually going:
   - Step 1 only,
   - Step 2 candidate generation,
   - unnecessary Step 3 entry,
   - or downstream exact work after an already-good easy-family incumbent.
3. Confirm whether Step 2 already produces a strong/closing incumbent and the
   solver simply fails to terminate early enough.

### Allowed bounded experiments for `{1..10}`

Only additive experiments are allowed.

Most promising directions:

1. easy-family early-accept policy experiment
   - only for contiguous/has-one groups,
   - only as an experimental toggle,
   - intended to stop unnecessary downstream work when Step 2 is already
     obviously sufficient.

2. selector-gating experiment
   - keep easy families out of unnecessary Step-3/Step-4 work,
   - again only as explicit experiment, not default rewrite.

3. cheap incumbent-certification experiment
   - if Step 2 is already exact or practically exact on this family, test a
     bounded confirmation path rather than letting the full downstream pipeline
     consume time.

Do **not** start with broad tuning of energy-core or beam for this track.

## Track B — `g37` closure push

### Hypothesis

`g37` is not failing for the same reason as `{1..10}`.

This is a real low-`K`, sparse-arithmetic exact-fallback failure regime.

### Goal

Clarify why exact fallback fails beyond `n=600`, then recover the frontier if a
bounded fix exists.

### Required diagnostic order

1. Re-check the `g37` rows to confirm the true exact boundary (`n=600`) and the
   later failed Step-4 entries.
2. Inspect the failed rows `n=750,1000,1500,2500,3500,5000,6000,7000`:
   - deciding step,
   - whether Step 4 enters,
   - whether the issue is budget, pruning, or missing incumbent quality.

### Allowed bounded experiments for `g37`

Only additive experiments are allowed.

Most promising directions:

1. exact-vs-beam selector experiment specific to this family
2. bounded exact-guided rescue settings
3. family-local incumbent handoff experiment

Do **not** assume FFD/easy-family reasoning applies here.

## Required artifacts

Create:

- `research/k_vs_arithmetic_axes_20260412/csv/plan13/PLAN13_easyfamily_g12345678910.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan13/PLAN13_g37_closure.csv`

If additive experiments are used, create:

- `research/k_vs_arithmetic_axes_20260412/csv/plan13/PLAN13_variant_compare.csv`

## Required docs

Update:

- `research/k_vs_arithmetic_axes_20260412/LOG.md`
- `research/k_vs_arithmetic_axes_20260412/RESULTS.md`
- `research/k_vs_arithmetic_axes_20260412/BLOCKERS.md`
- `research/k_vs_arithmetic_axes_20260412/METHOD_BOUNDARIES.md`
- `research/k_vs_arithmetic_axes_20260412/PAPER_RESULTS_READY.md`
- `research/k_vs_arithmetic_axes_20260412/PAPER_GROUPS_EXTENSION_SUMMARY.md`

## Success criteria

This task succeeds if:

1. `{1..10}` is pushed to at least `n=5000`, ideally `n=6000`,
2. `g37` is clarified correctly and, if possible, recovered beyond `n=600`,
3. the two families are treated as distinct problems,
4. any new idea remains additive and explicitly experimental,
5. and the paper-facing frontier story becomes stronger and clearer.

# PLAN 11 — Paper-Group n-Scaling With Group-Local Optimization

## Objective

Resume the paper-group campaign after the K=4 generator fix.

This plan has two simultaneous goals:

1. extend the paper's processing-time groups further in `n` under the cleaned
   four-step pipeline,
2. allow bounded, clearly marked experimental optimizations when a specific
   group becomes too slow or stops closing.

The important constraint is:

- do **not** destabilize the now-working K=4 package,
- do **not** rewrite the existing mainline policy in place,
- and do **not** broaden this into a new method family.

Any new optimization must be introduced as an explicit experimental branch,
toggle, or helper path that is easy to compare against the accepted baseline.

## Starting point

Current accepted K=4 result:

- `energy_core + direct + step1_exact_guided`
- `K=4` now uses the DP-style pattern generator by default
- K=4 signature dedup is disabled by default
- active hard K=4 rows are exact and much faster than before

Current paper-group summary:

- `g3567 = {3,5,6,7}` is exact through `n=5000`
- several other groups are already exact through `n=5000`
- some groups remain structurally harder or less cleanly extended

So the next benchmark-facing question is:

> for each paper group, how far can we scale `n`, and which stage becomes the
> practical limit?

## Scope

Stay inside the paper's seven processing-time groups only.

Do not start beyond-paper custom families in this task.

Target groups:

- `g24 = {2,4}`
- `g37 = {3,7}`
- `g810 = {8,10}`
- `g3567 = {3,5,6,7}`
- `g12357 = {1,2,3,5,7}`
- `g246810 = {2,4,6,8,10}`
- `g12345678910 = {1..10}`

## Main rule for code changes

The existing accepted solver path is the baseline and must remain readable and
reproducible.

If you notice a plausible improvement while extending a group:

- do **not** silently rewrite the accepted policy,
- do **not** replace existing defaults without proof,
- add the change as a clearly named experimental toggle, helper, or comparison
  path,
- keep the accepted baseline runnable side-by-side,
- and document exactly what the experiment is meant to test.

In short:

- baseline stays stable,
- new ideas are additive and explicitly experimental.

## Recommended execution order

### Phase A — Clean paper-group source of truth first

Before extending further:

1. dedupe the current `g3567` hard rows in
   `csv/plan05/PAPER_GROUPS_PLAN05_n_extension.csv`,
2. regenerate any summary artifacts affected by that dedupe,
3. verify the plan05 source CSV is a clean benchmark ledger again.

This avoids compounding stale or duplicated rows during the next extension pass.

### Phase B — Extend `n` group by group

Do **not** extend every family blindly to the same next frontier in one shot.

Use a group-by-group campaign:

1. pick one family,
2. scale `n` upward from its current verified exact frontier,
3. record:
   - exact / finite-gap / timeout,
   - deciding step,
   - runtime,
   - if Step 3 dominates, capture Step-3 diagnostics,
4. stop when the group clearly changes regime,
5. then move to the next family.

This keeps the diagnosis meaningful.

### Phase C — Optimize only when a specific group needs it

If a family becomes too slow or stops closing, you may test bounded experimental
improvements for that family.

Allowed examples:

- a family-specific generator threshold experiment,
- a family-specific candidate-retention or pruning experiment,
- a family-specific exact-vs-beam selector experiment,
- a bounded K-based or group-based toggle that leaves the baseline untouched.

Not allowed:

- replacing the accepted default policy without comparison evidence,
- broad speculative method redesign,
- column generation / branch-price work,
- hidden rewrites of the current K=4 package.

### Phase D — Promote only proven changes

If an experimental toggle clearly improves a family and does not regress already
accepted rows:

- keep it as an explicit documented option first,
- then decide whether it deserves promotion into the main policy.

Do not auto-promote during the first experiment pass.

## Measurement rules

For each tested row record at minimum:

- family / family_id
- `K`
- `n`
- `lambda`
- seed
- runtime
- `ub`
- `lb`
- gap
- exact / timed out
- deciding step
- main pack method
- any relevant Step-3 diagnostic fields
- experimental variant label

If a new experimental toggle is used, it must appear in the artifact.

## Preferred next frontier

Recommended order after dedupe:

1. continue the hard-but-now-improved `g3567` frontier beyond `n=5000`,
2. then extend the remaining easy-scalable families beyond `n=5000`,
3. keep `g37` and any other difficult groups explicitly diagnosed rather than
   mixed into easy-family averages.

This order gives both:

- one strong stress test (`g3567`),
- and a clean paper-facing extension picture.

## Deliverables

This task should produce:

1. refreshed plan05 CSV artifacts without duplicate paper rows,
2. an updated paper-group summary,
3. a new extension artifact for the next `n` frontier,
4. explicit notes on which groups stayed easy and which became Step-3 or Step-4
   limited,
5. and, if any optimization was tried, a separate experimental comparison table.

## Success criteria

This task succeeds if:

1. the paper-group source-of-truth artifacts are clean,
2. the next `n` frontier is mapped group by group,
3. any new optimization is clearly marked as experimental and additive,
4. the accepted K=4 package remains intact,
5. and the archive clearly states the current paper-group limits.

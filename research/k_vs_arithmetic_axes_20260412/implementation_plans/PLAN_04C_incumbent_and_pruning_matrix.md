# Plan 04C: Coupled Incumbent + Pruning Matrix for Exact DP

## Objective

Strengthen the exact fallback by improving **both**:

1. the incumbent handed into exact DP,
2. the exact-safe pruning/ordering inside exact DP.

This plan exists because improving only one side is not enough:

- a strong exact DP with a weak incumbent still explores too much,
- a strong incumbent with weak exact guidance may still fail to close.

So this cycle must measure the **interaction** between incumbent quality and
exact-DP improvements, not just each piece separately.

---

## Final-method constraints

This plan must preserve the cleaned pipeline:

1. Step 1: semigroup profile recovery
2. Step 2: quick realization
3. Step 3: profile-realization DP family
   - exact mode: fixed-block DP
   - truncated mode: beam
4. Step 4: semigroup-guided exact DP as the only exact fallback

Do not introduce:

- a second exact fallback,
- Lagrangian as co-equal default,
- local search / LNS as a new visible branch,
- exact-L2 back into the mainline.

---

## Core question

For the hard `K=6+` rows:

> Which incumbent source, combined with which exact-safe DP enhancement,
> gives the best practical exact-DP behavior?

“Best” means:

- closes more often,
- or reduces gap more,
- or reaches much stronger exact pruning under the same budget.

---

## Part A. Incumbent sources to test

The coder must not assume there is only one useful incumbent.

### Incumbent source I0 — Quick realization only

Source:

- best of FFD/BFD/random quick realization from Step 2

Purpose:

- baseline
- tells us how much Step 3 is actually helping the exact fallback

### Incumbent source I1 — Fixed-block DP exact mode

Source:

- exact profile-realization DP when the recovered-profile frontier is tractable

Purpose:

- strongest exact Step-3 incumbent source on rows where exact profile
  realization is cheap enough

Important:

- this source only applies on rows where exact fixed-block DP is actually
  runnable under the Step-3 regime rule

### Incumbent source I2 — Current profile_repair_beam

Source:

- current default truncated profile-realization beam

Purpose:

- current mainline baseline incumbent

### Incumbent source I3 — Strengthened beam

Source:

- current beam plus the currently accepted beam-only improvements:
  - arithmetic-aware ranking,
  - bounded discrepancy,
  - adaptive width

Purpose:

- the main candidate for improving exact-DP handoff quality on hard rows

### Incumbent source I4 — Best Step-3 family incumbent

Source:

- whichever is better between:
  - exact fixed-block DP (when tractable)
  - strengthened beam

Purpose:

- realistic best-practical Step-3 handoff

This is likely the long-term default handoff policy if the experiments support
it.

---

## Part B. Exact-DP enhancements to test

Only exact-safe changes are allowed.

### Pruning/ordering variant P0 — Current exact DP

Use as baseline:

- current sparse exact DP
- current dense fallback
- current relaxed/completion/dominance setup

### Variant P1 — Type-aware admissible job-cost lower bound

Add:

- `min_job_cost[j][t]`-style admissible per-type lower bound
- combine with the current bounds via `max(...)`

Purpose:

- inject job-type structure into pruning

This is the first required implementation.

### Variant P2 — Incumbent-guided expansion ordering

Add:

- ordering that prioritizes states most promising relative to current UB
- e.g. lower `g+h` first, then smaller slack-to-UB, then profile/incumbent
  tie-breaks

Purpose:

- tighten `best` earlier and activate existing pruning sooner

This is the second required implementation.

### Variant P3 — P1 + P2 combined

Purpose:

- this is the most important exact-DP experimental variant
- likely the best practical near-term exact configuration

### Variant P4 — Restricted extra dominance (optional)

Only if time allows after P1–P3:

- add exact-safe restricted componentwise dominance in the narrowest form that
  can be proved correct

This is optional in this cycle, not required.

---

## Part C. Required experiment matrix

The coder should not run everything against everything blindly.

Use this matrix.

### Phase 1 — Incumbent quality only

Hold exact DP fixed at P0.

Compare:

- I0
- I1 (where available)
- I2
- I3
- I4

Metrics:

- initial UB passed to exact
- Step-3 runtime
- total runtime
- final gap
- whether exact closes

Goal:

- learn which incumbent source is strongest before touching exact pruning.

### Phase 2 — Exact-DP improvements only

Hold incumbent source fixed at the current best source from Phase 1 on each
row family.

Compare:

- P0
- P1
- P2
- P3

Metrics:

- exact elapsed time
- states reached / expanded
- pruned_bound
- pruned_completion
- pruned_relaxed
- pruned_dominance
- exact final UB / gap
- closed vs timed out

Goal:

- isolate whether type-aware LB or ordering is doing the useful work.

### Phase 3 — Best combinations

Run only the strongest combinations from Phases 1 and 2.

Minimum combinations to test:

- I2 + P0
- I3 + P0
- I2 + P3
- I3 + P3
- I4 + P3

Goal:

- identify the strongest practical end-to-end exact-guided configuration.

---

## Required benchmark rows

At minimum:

### Easy control

- one easy row where Step 2 already closes

Purpose:

- confirm no regression on the easy regime

### Exact-tractable Step-3 rows

- one row where exact fixed-block DP is tractable and useful
- likely a hard `K=4` row or another moderate merged-block case

Purpose:

- test incumbent source I1

### Hard six-type anchors

Both of these are required:

- `medium_k6_dense n=1000`
- `hard_k6_2345711 n=1000`

Purpose:

- these are the main exact-DP stress anchors

### One larger/harder row

At least one of:

- hard `K=6`, larger `n`
- or hard `K=8` irregular

Purpose:

- see whether the exact-DP improvements generalize beyond the easiest hard
  anchors

---

## Implementation order

### Stage 1. Incumbent instrumentation

Before changing behavior, ensure the solver logs clearly:

- which incumbent source exact DP received,
- the exact initial UB,
- Step-3 runtime and method details.

### Stage 2. Implement P1

Add the simple type-aware admissible lower bound first.

Do **not** implement the more complicated “competitive assignment” variant yet.

### Stage 3. Implement P2

Add incumbent-guided ordering only.

Keep it simple and measurable.

### Stage 4. Run Phase-1 and Phase-2 experiments

Do not guess.

Collect the matrix and let it decide whether:

- incumbent quality is still the main bottleneck,
- or exact pruning is finally becoming the bottleneck.

### Stage 5. Run best-combination Phase 3

Only after the first two phases are measured.

---

## Decision rules after the matrix

### If I3/I4 helps a lot but P1/P2 barely help

Interpretation:

- Step 3 incumbent quality is still the dominant bottleneck.

Next move:

- continue Step-3 strengthening
- do not overinvest in exact pruning yet

### If P1/P2 helps a lot even with the current incumbent

Interpretation:

- exact-DP structure was the main bottleneck.

Next move:

- continue exact-safe pruning/dominance work

### If both help materially and P3 beats either alone

Interpretation:

- the exact-guided pipeline really is governed by both parameters together.

Next move:

- adopt the best incumbent + pruning pair as the new exact-guided baseline

### If neither helps much

Interpretation:

- the current global exact DP may need a deeper structural change or the
  Step-3 family may still be too weak.

Next move:

- revisit the Step-3 family first,
- then consider heavier exact-safe redesigns.

---

## Success criteria

This plan succeeds if it produces:

1. one clearly best incumbent source for hard rows,
2. one clearly best exact-DP pruning/ordering variant,
3. a measured answer to whether better incumbents or better exact pruning are
   currently more important,
4. and a clean new default exact-guided configuration if the data supports it.

---

## Deliverables

Required outputs:

- updated solver code
- updated archive docs:
  - `LOG.md`
  - `RESULTS.md`
  - `BLOCKERS.md`
  - `EXPERT_GUIDANCE.md`
- at least one consolidated CSV or table summarizing the incumbent/pruning
  matrix
- a short conclusion:
  - best incumbent source,
  - best pruning variant,
  - and the recommended next move

---

## Important constraint

At every stage, keep the story clean:

- fixed-block DP and beam are Step-3 modes,
- exact DP is Step 4,
- no second exact fallback,
- no drift back into a multi-branch heuristic zoo.

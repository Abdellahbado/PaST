# Plan 03D: Decide When to Use Exact Fixed-Block DP vs Beam Mode

## Objective

Build a principled selector for Step 3:

- when should the solver try **exact fixed-block DP**,
- and when should it go directly to **beam mode**?

This is important for both runtime and paper clarity.

We want a rule that is:

- cheap,
- explainable,
- based on recovered-profile structure,
- and empirically validated.

---

## Core idea

The decision should be based on a **tractability estimate of the
profile-realization frontier**, not just on `K` or `n`.

We already know:

- difficulty is not monotone in `K`,
- arithmetic structure matters,
- recovered block structure matters,
- and exact fixed-block DP is valuable enough to keep.

So the selector should use recovered-profile features, not only raw instance
size.

---

## Candidate predictor features

These features should be computed before Step 3 exact mode starts.

### A. Recovered-profile size

1. number of recovered blocks
2. number of merged blocks
3. total recovered capacity
4. max block capacity

### B. Exact-mode frontier proxies

1. current mixed-radix count-state size estimate
2. per-block composition count estimate
3. total composition estimate across blocks
4. estimated branching factor under current block ordering

### C. Arithmetic-hardness descriptors

1. presence of `1`
2. minimum generator / multiplicity
3. contiguity of the length set
4. Frobenius / Apéry-type features already available in the archive tooling
5. semigroup density / bounded-density proxy if cheap enough

### D. Residual flexibility indicators

1. filler capacity from short jobs
2. number of blocks with very few feasible compositions
3. count of “hard” residue blocks

---

## Implementation tasks

### Task 1. Instrument exact fixed-block DP

Before changing policy, log for rows where exact fixed-block DP is run:

- merged block count,
- state-space estimate,
- composition estimate,
- max compositions per block,
- actual runtime,
- whether it solved, skipped, or timed out,
- whether it improved the incumbent.

This data must be written into results or auxiliary CSVs.

### Task 2. Build a first explicit regime rule

Start with a simple human-readable rule, not a learned model.

Example structure:

- use exact fixed-block DP if:
  - merged blocks <= threshold,
  - estimated composition/state-space <= threshold,
  - and no strong arithmetic-hardness alarm triggers
- otherwise go to beam mode

The first rule should be conservative and explainable.

### Task 3. Validate the boundary

Use archived rows and new representative rows to classify outcomes into:

1. exact mode was correct to try
2. exact mode should have been skipped
3. beam mode was correct
4. beam mode skipped an exact-solvable case

The aim is not perfection in one cycle. It is to discover a useful and
defensible regime boundary.

### Task 4. Refine with block-order-aware estimates

If exact-mode quality depends strongly on block ordering, then the predictor
should use the **post-ordering** composition profile, not only raw block data.

This keeps the selector aligned with the real exact-mode implementation.

---

## Deliverable rule

At the end of this plan, produce:

1. one explicit selector policy for Step 3 exact vs beam mode,
2. supporting table/CSV showing why that rule is reasonable,
3. notes on misclassified rows and how to improve the selector later.

---

## Success criteria

This plan succeeds if:

1. the solver no longer decides “exact vs beam” in an ad hoc way,
2. the rule is based on recovered-profile structure and frontier estimates,
3. the rule is documented clearly enough for the paper,
4. and the validation shows that it captures the main easy-vs-hard separation
   for Step 3.

---

## Fallback if clean classification is still unclear

If no simple selector is good enough yet, deliver this fallback:

1. a ranked list of the most predictive features,
2. a conservative exact-mode gate that avoids the worst blowups,
3. a recommendation for the next cycle:
   - more instrumentation,
   - or a lightweight learned classifier if the paper really needs it.

That is still valuable and still much better than the current implicit policy.

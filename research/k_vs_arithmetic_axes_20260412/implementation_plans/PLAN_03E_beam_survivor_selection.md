# Plan 03E: Beam Survivor Selection and Frontier Retention

## Objective

Strengthen Step 3 by improving the single most important decision inside the
current `profile_repair_beam`:

> **Which states survive to the next block?**

This plan is deliberately narrow. It does **not** introduce:

- local search as a new method family,
- Lagrangian as a default branch,
- exact-L2 back into the mainline,
- dynamic pricing / arc-flow,
- or any second exact method.

It only changes the internal frontier-retention policy of the beam.

---

## Why this plan

The current beam already has:

- suffix-feasibility pruning,
- arithmetic pressure,
- local block evaluation,
- discrepancy control,
- adaptive width.

But after pruning, survivors are still selected primarily by **one scalar
score**, and that makes the beam brittle:

- states good for feasibility can be lost to states good for cost,
- states good for arithmetic flexibility can be lost to states close to the
  proportional center,
- one ranking mistake early in the block sequence may never be repaired.

So the next clean Step-3 improvement is:

> improve survivor selection, not method family.

---

## Current implementation snapshot

Relevant code:

- `/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/stateful_dp_solver.cpp`
- current Step-3 core:
  `block_repair_feasible_beam_ub(...)`

Current scoring components include:

- center score,
- suffix-feasibility pressure,
- arithmetic pressure,
- local block rank,
- discrepancy penalty.

Current survivor policy:

1. generate next-layer candidates,
2. deduplicate by `(count key, discrepancy)`,
3. keep the cheapest score representative,
4. sort globally by score,
5. truncate to width.

This final step is what this plan targets.

---

## Main question

Among all next-layer candidates that survive hard infeasibility pruning:

> Which subset should be kept so the beam remains both feasible and diverse
> enough to find stronger incumbents later?

This is a frontier-design problem, not just a weight-tuning problem.

---

## Required output of this plan

At the end of the cycle, the coder must produce:

1. at least **3 survivor-selection variants** implemented cleanly,
2. one shared instrumentation layer for comparing them,
3. a validation table showing which variant is best on the anchor rows,
4. a recommended new default survivor policy for Step 3.

---

## Variants to implement

The coder should implement these variants in order.

### Variant S0 — Baseline scalar ranking

This is the current behavior and must remain available as the control.

Definition:

- deduplicate by `(count key, discrepancy)`,
- rank by current scalar score,
- truncate to width.

Purpose:

- baseline for all comparisons.

---

### Variant S1 — Stratified quota beam

**Recommended first implementation.**

Idea:

- do not let one scalar score decide the whole frontier,
- reserve slices of the beam for different notions of promise.

Implementation:

After candidate generation and deduplication:

1. build multiple rankings of the same next-layer candidate pool:
   - by overall scalar score,
   - by suffix-feasibility pressure,
   - by arithmetic pressure,
   - by exact local block rank,
   - optionally by lowest discrepancy.

2. allocate survivor quotas, for example:
   - 40% from overall score,
   - 20% from feasibility-best,
   - 20% from arithmetic-best,
   - 20% from local-cost-best.

3. fill the layer by taking the best unseen states from each ranking until the
   width is reached.

Why it is promising:

- preserves useful diversity,
- still easy to explain,
- still fully beam-based,
- no new search family.

Important:

- survivor sets must be deduplicated across quotas,
- if a quota is exhausted, reassign unused capacity to the global-score pool.

What to measure:

- whether kept-state diversity rises,
- whether incumbents improve,
- whether Step-4 initial UB improves.

---

### Variant S2 — Pareto-front survivor filter

Idea:

- instead of one scalar score, keep a small approximate Pareto front before
  width truncation.

Candidate objective vector:

- feasibility pressure,
- arithmetic pressure,
- local rank,
- discrepancy,
- optionally center deviation.

Implementation:

1. define a candidate vector for each next state,
2. remove states that are clearly dominated by another state on all selected
   criteria,
3. if the remaining set is still too large, then rank by the current scalar
   score and truncate.

Important:

- keep this **approximate and bounded**
- do not build a huge quadratic dominance filter without guards

Suggested first version:

- apply Pareto filtering only within buckets of similar overall score,
- or only against a rolling elite pool,
- or cap pairwise comparisons per layer.

Why it is interesting:

- naturally preserves diverse “good for different reasons” states,
- aligns with the concern that one heuristic should not decide everything.

Risk:

- too expensive if implemented naively.

So this is second priority, after S1.

---

### Variant S3 — Novelty-preserving beam

Idea:

- deliberately retain some states that are **structurally different** from the
  current survivors, even if not top-ranked by score.

Novelty measures can be simple:

- Hamming distance in the count vector relative to already kept states,
- residual-type signature novelty,
- residue-class novelty for remaining work,
- difference in block-pattern choice history depth-limited to early blocks.

Implementation:

1. sort candidates by current scalar score,
2. greedily accept the best candidate,
3. for later candidates, accept some only if they add novelty relative to the
   kept set,
4. fall back to scalar-score fill if novelty conditions become too restrictive.

Recommended usage:

- reserve only a small fraction of width (e.g. 10–20%) for novelty survivors,
- keep the rest conventional.

Why it may help:

- avoids beam collapse where all survivors are almost identical,
- especially useful on arithmetic-hard rows where one early misallocation can
  kill all completions later.

Risk:

- novelty can preserve bad states if overused.

So this should be tested as a bounded diversity supplement, not a full policy.

---

### Variant S4 — Two-tier beam (elite + explorer)

Idea:

- explicitly split the frontier into:
  - **elite states**: best by score
  - **explorer states**: second-tier states selected for diversity or
    discrepancy

Implementation:

1. choose elite width `W_e`
2. choose explorer width `W_x`
3. fill elite by global score
4. fill explorer by one alternate rule:
   - best discrepancy states,
   - best arithmetic states,
   - novelty states

This is conceptually cleaner than mixing all policies into one ranking.

Why it is attractive:

- easy to explain
- easy to instrument
- still clearly a beam search

This can be viewed as a more structured version of S1 and S3.

---

## Strong recommendation on priority

The coder should implement and compare in this order:

1. `S0` baseline
2. `S1` stratified quota beam
3. `S4` two-tier beam
4. `S3` novelty-preserving beam
5. `S2` Pareto-front filter only if the simpler variants fail

Rationale:

- S1 and S4 are the cleanest and most likely to help
- S3 is useful but more heuristic
- S2 is elegant but easiest to make too expensive

---

## Shared instrumentation requirements

All variants must report the same diagnostics.

At minimum, per row:

- final UB
- final gap
- Step-3 runtime
- deciding step
- exact-DP initial UB
- exact-DP used or not

And beam-specific:

- total candidates considered
- total survivors kept
- suffix-pruned count
- over-count-pruned count
- discrepancy-pruned count
- average width
- max width
- duplicate-candidate count after dedup

New diagnostics for survivor policy:

- number of distinct survivor buckets used
- survivor diversity score (simple proxy is enough)
- fraction of survivors coming from each quota/tier/policy

If a novelty or stratified variant is used:

- record the actual retained composition by category.

---

## Benchmark rows to compare

Required:

### Easy control

- one row where Step 2 already closes

Purpose:

- confirm no regression in easy regime

### Medium six-type anchor

- `medium_k6_dense n=1000`

Purpose:

- key row where Step 3 matters and exact often enters with a tiny gap

### Hard six-type anchor

- `hard_k6_2345711 n=1000`

Purpose:

- main arithmetic-hard Step-3 test row

### Larger/harder row

At least one:

- hard `K=6` larger `n`,
- or hard `K=8` irregular

Purpose:

- test whether improved survivor logic still helps when the beam is under more
  stress.

### Exact-mode row

One row where exact fixed-block DP is still tractable.

Purpose:

- see whether better beam incumbents help exact mode too,
- not only the global exact DP.

---

## Experimental protocol

### Phase 1 — Survivor policy only

Hold everything else fixed.

Compare:

- S0
- S1
- S4
- optionally S3

Measure:

- Step-3 UB improvement
- Step-3 runtime
- exact-DP initial UB
- final overall gap

Goal:

- identify whether survivor selection alone improves incumbent quality.

### Phase 2 — Interaction with exact DP

Take the best 1–2 survivor policies from Phase 1 and run them through the
current exact-guided pipeline.

Measure:

- whether exact closes more often,
- whether exact reaches more pruning,
- whether exact total runtime decreases.

Goal:

- determine whether better Step-3 survivors actually help Step 4 in practice.

### Phase 3 — Stress test

Run the winning policy on one larger or higher-`K` hard row.

Goal:

- make sure the new policy is not only a small-row trick.

---

## How to know which policy is better

The coder should **not** choose based on Step-3 score alone.

Rank policies using this order:

1. better final UB on the hard anchor rows
2. better exact-DP initial UB
3. better final certified gap after Step 4
4. lower runtime, if the first three are tied

Important:

- if a policy makes Step 3 slower but significantly improves the UB and helps
  exact DP close, that may still be a win
- if a policy only increases diversity diagnostics but does not improve UB, it
  is not good enough

---

## What not to do

- do not add local search as a separate phase in this plan
- do not add destroy/repair neighborhoods yet
- do not add exact-L2
- do not add new method families
- do not hide a new default policy without documenting it

This plan is only about **frontier survival policy** inside the beam.

---

## Success criteria

This plan succeeds if it produces:

1. one clearly better survivor-selection policy than the current scalar
   baseline on the hard anchor rows,
2. evidence that the new policy improves incumbent handoff to exact DP,
3. a clean default recommendation for how Step-3 beam survivors should be
   chosen.

---

## Deliverables

Required outputs:

- code implementing the compared survivor policies
- archive updates:
  - `/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/LOG.md`
  - `/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/RESULTS.md`
  - `/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/BLOCKERS.md`
  - `/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/EXPERT_GUIDANCE.md`
- at least one comparison CSV or table across S0/S1/S4/(S3)
- a short final recommendation on which survivor policy should become default.

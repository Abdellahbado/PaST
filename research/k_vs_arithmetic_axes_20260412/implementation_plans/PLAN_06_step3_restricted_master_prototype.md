# PLAN 06 — Step 3 Restricted-Master Prototype

Date: 2026-04-16

Status: NEW

## 1. Purpose

Turn the new Step-3 unified theory into a practical experiment.

We do **not** want to jump directly to full branch-and-price. That would be a
large research/engineering branch.

Instead, the next justified move is to build a **restricted-master prototype**
that is small enough to implement now, but real enough to test the theory on
actual rows.

The goal is to answer:

> Does an explicit outer loop that alternates between solving a restricted
> recovered-profile master and adding better block patterns outperform the
> current fixed-pool Step-3 methods?

If yes, that validates the framework on the ground.
If no, we learn that the theory is elegant but not yet computationally useful
in our regime.

## 2. Scope

This plan is for a **prototype**, not full branch-and-price.

That means:

- yes: implement one explicit restricted-master loop
- yes: add new patterns on demand
- yes: compare against current exact/core/beam paths
- no: full exact branch-and-price tree
- no: full dual stabilization package
- no: major solver redesign in one jump

## 3. The Right Next Concrete Method

The best immediate next method is:

> **CG-lite / restricted-master profile realization**

with these components:

1. an initial block-pattern pool
2. a master solver over that current pool
3. a pricing/augmentation step that adds missing promising patterns
4. repeat for a limited number of rounds

This is the smallest implementation that is still a real incarnation of the
new unified theory.

## 4. Why Not Full Branch-and-Price Yet

Full branch-and-price would require:

- a true LP restricted master
- stable dual extraction
- exact pricing under reduced cost
- branching rules that preserve pricing structure
- likely stabilization

That is a **big code change**.

This plan instead asks for a **medium-sized code change**:

- add one explicit outer augmentation loop around the existing Step-3 machinery
- use current local evaluators and pattern-generation/search code as subroutines

## 5. Prototype Architecture

### Phase A — Explicit master data extraction

Refactor Step 3 so we can explicitly materialize:

- recovered blocks
- current pattern pool per block
- pattern counts per block
- pattern cost per block

This must be extractable independent of beam/exact traversal.

Deliverable:

- a clean in-memory Step-3 master representation

### Phase B — Restricted master solve on current pool

Use the existing Step-3 exact/beam solvers as the **master traversal** on the
current pool.

In prototype form:

- small/tractable rows:
  - exact profile-realization DP on the current pool
- larger rows:
  - beam master solve on the current pool

So initially, the prototype does **not** yet require a new LP master.
It already gives us a real “restricted master” baseline.

### Phase C — Pattern augmentation loop

Add one outer loop:

1. solve restricted master on current pool
2. inspect residual weakness / dual surrogate
3. generate additional patterns for selected blocks
4. add them to the pool
5. re-solve

This is the first practical experiment with the new theory.

## 6. What To Use For Pricing In The Prototype

For the first prototype, do **not** require full LP dual prices.

Instead implement **two pricing levels**:

### P0 — Dual surrogate pricing (required first)

Use a simple surrogate signal derived from the current master solution:

- under-covered / tight job types
- hard residual types
- block-local cost pressure
- incumbent mismatch

Then search for new patterns that improve those signals.

This is not exact CG, but it is a legitimate first restricted-master
augmentation loop.

### P1 — True reduced-cost pricing (optional second)

Only if Phase P0 is promising:

- add a small LP master
- extract dual prices for type constraints and block convexity constraints
- implement true reduced-cost pricing per block

This is the bridge to real column generation.

## 7. What The Pricing Search Should Actually Do

For a chosen block `b`, pricing must search over feasible block patterns.

The search object is:

- choose a type-count vector for the block
- satisfy block feasibility
- evaluate local block cost exactly
- score the pattern under:
  - original cost
  - or surrogate / reduced-cost objective

This can reuse:

- current pattern generation logic
- local block evaluator
- exact block scheduling DP

The first prototype does **not** need to generate every possible pattern.
It only needs to generate **better missing patterns than the current pool**.

## 8. Concrete Design Axes To Test

The prototype should explicitly test the expert’s 3-axis theory:

### Axis 1 — Column/pattern policy

- initial fixed pool only
- fixed pool + augmented patterns
- core-restricted augmentation

### Axis 2 — Frontier/search policy

- exact master solve
- beam master solve

### Axis 3 — Objective policy

- original cost
- surrogate dual-guided augmentation

## 9. Minimal Experiment Matrix

### Group 1 — `K=2`

Purpose:

- confirm prototype does not break the recovered exact small-case path

Rows:

- `{8,10}` `n=1000`
- `{8,10}` `n=2500`

Expected:

- exact Step-3 already strong
- augmentation should not be necessary

### Group 2 — `K=4`

Purpose:

- the most important validation target

Rows:

- `g3567 n=1000`
- `g3567 n=1500`
- if feasible: `3567_plus n=3500`

Expected:

- current beam/default is not sufficient
- augmentation or core-style restricted master may recover the old stronger path

### Group 3 — `K=6`

Purpose:

- see whether explicit augmentation helps where fixed pools/beam stall

Rows:

- `2345711 n=1000`
- `456789 n=1000`

Expected:

- probably no exact closure
- but possible better incumbent or smaller gap

## 10. Success Criteria

This prototype succeeds if at least one of these happens:

1. on `K=4`, augmentation recovers a stronger incumbent than the current fixed
   pool
2. on a `K=6` anchor, augmentation measurably improves over fixed-pool beam
3. the restricted-master framing becomes operational and inspectable in code

This prototype fails if:

- augmentation never adds useful patterns
- or added patterns do not change master outcomes at all

That would still be a useful scientific result.

## 11. Files To Inspect Or Edit

- [stateful_dp_solver.cpp](/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/stateful_dp_solver.cpp)
- [stateful_compare.cpp](/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/stateful_compare.cpp)
- [STEP3_UNIFIED_THEORY_RESEARCH.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/archive_20260421/markdown/step3_theory/STEP3_UNIFIED_THEORY_RESEARCH.md)
- [step3_unified_theory_critique.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/archive_20260421/markdown/step3_theory/step3_unified_theory_critique.md)
- [step3_one_method_proposal.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/archive_20260421/markdown/step3_theory/step3_one_method_proposal.md)

## 12. Archive Deliverables

Update:

- [LOG.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/LOG.md)
- [RESULTS.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/RESULTS.md)
- [BLOCKERS.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/BLOCKERS.md)

Create if needed:

- `STEP3_RESTRICTED_MASTER_PROTOTYPE_RESULTS.md`

## 13. Practical Recommendation

This is the right next step because:

- it makes the new theory concrete
- it is a medium code change, not a full thesis branch
- it gives a clear yes/no answer on whether restricted-master augmentation is
  worth continuing

## 14. If It Works

If the prototype works, then the next plan is:

- add a tiny LP restricted master
- add true dual prices
- convert surrogate pricing into real reduced-cost pricing

That would be the real bridge to column generation / branch-and-price.

## 15. If It Does Not Work

If the prototype does not help, then:

- keep the theory as a strong interpretation
- but do not invest immediately in full CG
- instead continue with the best current exact/core/beam policy for the paper

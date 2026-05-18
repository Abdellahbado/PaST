# Plan 03C: Unify Fixed-Block DP and Beam as One Profile-Realization DP Family

## Objective

Replace the current Step-3 conceptual split with one clean family:

- **Step 3 = Profile-realization DP**
  - **exact mode**: fixed-block DP (full frontier)
  - **truncated mode**: profile-repair beam (width-limited frontier)

This plan does **not** remove fixed-block DP. It elevates it to the exact core
of Step 3 and treats the beam as its scalable approximation.

That is the intended final paper story.

---

## Why this plan

The current evidence supports the following structural claim:

1. fixed-block DP and profile-repair beam solve the same recovered-profile
   assignment problem,
2. they use the same high-level state object (type-count frontier over
   recovered blocks),
3. they differ primarily in **frontier policy**:
   - keep all states,
   - or keep the best `W` states.

So the clean method is:

> “We solve profile realization exactly when the frontier is tractable, and by
> beam-truncated DP otherwise.”

---

## Required conceptual outcome

At the end of this plan, Step 3 should be explainable as:

1. the recovered block profile defines the exact profile-realization problem,
2. exact fixed-block DP is the full-frontier solver,
3. profile-repair beam is the bounded-frontier solver,
4. both use the same block ordering, exact local evaluator, and feasibility
   checks whenever possible.

---

## Files to inspect

- `/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/stateful_dp_solver.cpp`
- `/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/stateful_compare.cpp`
- `/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/RESULTS.md`
- `/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/BLOCKERS.md`
- `/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/EXPERT_GUIDANCE.md`

Likely relevant functions:

- `pack_recovered_blocks(...)`
- `block_repair_profile_repair_beam_ub(...)`
- archival fixed-block / block-DP exact checker
- exact Level-3 local evaluator used inside the beam

---

## Main implementation tasks

### Task 1. Rewrite Step-3 language in code comments and archive

Change terminology so Step 3 is no longer described as:

- “beam method plus legacy exact subsolver”

and is instead described as:

- “profile-realization DP with exact and truncated modes”

This must be reflected in:

- solver comments,
- archive docs,
- results interpretation.

### Task 2. Align the exact and beam modes structurally

Make the exact fixed-block DP and beam look as much like two modes of the same
solver family as possible.

Concretely, reuse or align:

1. **block ordering**
   - hardest blocks first
   - same ordering policy in exact mode and beam mode

2. **feasibility pruning**
   - suffix min/max type checks
   - suffix work/capacity checks
   - any cheap arithmetic feasibility filters that are exact and safe

3. **local evaluation**
   - exact Level-3 block evaluator where already available
   - do not let exact mode use a much weaker block evaluation than beam mode

### Task 3. Transfer the best exact-safe pruning ideas into fixed-block DP

The expert-highlighted candidates are:

1. suffix min/max per type
2. better block ordering (fewest feasible patterns / hardest-first)
3. sparse frontier management / dedup
4. any exact-safe residual arithmetic feasibility checks

These should be added first to **exact fixed-block DP**, because that is the
main Step-3 exact mode.

### Task 4. Keep beam as the truncated mode

Do not create another Step-3 theory.

Beam remains:

- bounded frontier,
- same recovered blocks,
- same exact local block evaluator,
- same Step-3 family.

The only difference from exact mode should be search budget / frontier width,
not mathematical identity.

---

## Important design constraint

Do **not** fold Step 4 exact DP into Step 3.

This plan only unifies:

- exact recovered-profile realization
- heuristic recovered-profile realization

The global semigroup-guided exact DP remains Step 4 and remains separate.

---

## Block ordering requirement

This is one of the most promising exact-safe enhancements and should be done in
this plan.

Required experiment:

Compare at least two block-order rules inside fixed-block DP:

1. original/current order,
2. hardest-first order.

Suggested hardness signals:

- fewest candidate compositions/patterns,
- smallest residual representability flexibility,
- rare residue classes modulo small generators,
- highest local-cost sensitivity.

Start simple:

- first implement “fewest feasible compositions first”
- then test whether adding an arithmetic tie-break helps.

---

## Success criteria

This plan succeeds if all of the following are true:

1. Step 3 can be described as **one DP family** with exact and truncated modes.
2. Fixed-block DP is clearly retained and elevated, not treated as legacy.
3. At least one exact-safe enhancement from beam-side logic improves fixed-block
   DP behavior or understanding.
4. The code/comments/archive no longer imply that fixed-block DP and beam are
   unrelated methods.

---

## Fallback if full unification is too invasive in one cycle

If full code-level unification is too large, do this minimum acceptable subset:

1. unify the **conceptual documentation** immediately,
2. implement the shared block ordering and shared exact-safe suffix pruning,
3. record what still remains structurally separate in code.

That is still a meaningful improvement and preserves direction.

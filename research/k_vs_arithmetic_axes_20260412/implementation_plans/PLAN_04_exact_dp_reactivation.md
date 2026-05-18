# Plan 04: Exact DP Reactivation

## Goal

Make the semigroup-guided exact DP the **only exact fallback** again, and make
it actually useful after Steps 2–3.

This plan assumes:

- Step 3 is a heuristic repair method only
- exact Level-2 B&B is diagnostic/archive-only
- the final certification burden belongs solely to the exact DP

---

## Principle

Do not add another exact method.

Instead:

- keep the exact DP exact,
- but strengthen it with better incumbents, stronger safe bounds, and better
  state management.

This is consistent with exact-DP literature in scheduling:

- lower bounds
- node merging
- dominance
- heuristic seeding

can all live inside an exact DP without changing its identity:

- [Bürgy, Hertz, Baptiste 2020](https://doi.org/10.1016/j.cor.2020.105063)

---

## What Step 4 should look like conceptually

### Input from Steps 1–3

Step 4 should receive:

1. semigroup relaxation information
2. strongest UB from:
   - fast realization
   - unified Step-3 repair
3. any safe completion lower bounds already available

### Output

Either:

- a proof of optimality,
- or the best certified residual gap under the time budget

### No extra exact branches

Do not present:

- exact-L2 B&B
- fixed-block exact subsolvers
- or any other exact side method

as co-equal exact stages in the final mainline.

---

## Main exact-DP enhancement directions

### 1. Better incumbent injection

The first requirement is simple:

- Step 4 must always receive the best UB from Steps 2–3
- with no policy branch skipping that handoff

This sounds trivial, but it is exactly what makes a DP practical.

### 2. Stronger safe lower bounds inside DP

Use only bounds that do not change exactness:

- semigroup completion bounds
- any existing admissible suffix cost bounds
- state-based lower bounds already validated in the solver

If a new bound is proposed, it must be documented as admissible before being
used for pruning.

### 3. Stronger dominance and node merging

Exact DP should merge states aggressively when safe.

Examples of safe direction:

- same abstract state with worse accumulated cost
- same residual work / machine-state signature with worse bound

The literature supports this style directly:

- lower bounds + node merging + heuristics in exact DP are standard and very
  effective:
  [Bürgy, Hertz, Baptiste 2020](https://doi.org/10.1016/j.cor.2020.105063)

### 4. Better expansion order, not unsafe restriction

If profile or incumbent information is used inside Step 4, use it to:

- prioritize which states are expanded first
- not to throw states away unsafely

This is important:

- profile guidance is welcome as ordering
- profile restriction is not acceptable unless proven safe

---

## Trigger policy for Step 4

Step 4 should remain the final fallback in the story, but its runtime policy
can still be adaptive.

Recommended policy:

1. if Steps 2–3 close, stop
2. otherwise always hand UB/LB to exact DP
3. use one of two budgets:
   - **main benchmark budget** for the default pipeline
   - **extended verification budget** for paper verification runs

This lets the paper say both:

- “the exact DP is the final fallback”
- and
- “we also ran longer exact verification on selected rows”

---

## What should be reused from recent experiments

The recent exact-L2 work should not survive as a second exact stage, but some
of its ideas may still be useful in Step 4 if they can be translated safely:

- stronger incumbent awareness
- suffix minimum-cost reasoning
- state memoization / dominance awareness

If a technique can be absorbed into exact DP without changing the method’s
identity, it is worth trying.

If not, keep it archival only.

---

## Recommended development order

### Phase A. Policy cleanup

Ensure exact DP is once again the only exact method in the declared mainline.

### Phase B. Incumbent handoff audit

Verify for every hard row that:

- the best Step-3 UB is actually handed to exact DP
- exact DP starts from that UB

### Phase C. Safe pruning review

Review the current exact DP for:

- admissible lower bounds
- dominance rules
- node merging opportunities

### Phase D. Expansion-order enhancement

Use incumbent/profile information only to improve:

- search order
- early incumbent confirmation

not to remove exactness.

---

## Success criterion

The final paper story for Step 4 should be:

> “If realization heuristics do not close the gap, we fall back to a
> semigroup-guided exact dynamic program strengthened by admissible lower
> bounds, dominance, and state merging.”

That is a clean and defensible exact-method story.

---

## Execution update (2026-04-13)

Current status after cleanup pass:

- exact-L2 is demoted out of default mainline and does not replace default
  incumbents
- exact DP remains the only exact fallback in the declared pipeline
- exact stage still receives the best UB from Step 3 in ablation mainline
  (`step1_exact_guided` path)

Remaining exact-DP work is performance-focused only (ordering/dominance/bounds)
without introducing any second exact family.

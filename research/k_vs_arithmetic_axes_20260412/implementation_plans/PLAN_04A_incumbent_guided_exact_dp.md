# Plan 04A: Incumbent-Guided Exact DP

## Goal

Keep Step 4 as the only exact fallback, but make it use the incumbent from
Steps 2–3 much more intelligently.

This plan does **not** add a new exact method. It strengthens the existing
semigroup-guided exact DP.

---

## Principle

The exact DP should use the incumbent in three ways only:

1. **pruning**, through admissible lower bounds,
2. **dominance**, by discarding worse equivalent states,
3. **ordering**, by exploring promising states first.

It should not use the incumbent to impose unsafe restrictions.

---

## Literature-backed direction

This is standard exact-DP strengthening:

- Bürgy, Hertz, Baptiste (2020), exact dynamic programming strengthened by
  lower bounds, node merging, and heuristics:
  https://doi.org/10.1016/j.cor.2020.105063

The important message:

- better incumbents help exact DP most when they tighten pruning and expansion
  order,
- not when they create a second exact method beside the DP.

---

## Recommended Step-4 redesign

### A. Guaranteed incumbent handoff

First verify mechanically:

- the best UB from Step 2 or Step 3 always reaches the exact DP,
- no policy branch starts exact DP with a weaker UB,
- incumbent metadata can identify whether the UB came from:
  - FFD-like realization,
  - Step-3 beam realization.

This sounds trivial but is the highest-leverage requirement.

### B. Incumbent-guided expansion order

Use the incumbent to sort DP expansions, not to prune unsafely.

Suggested order signals:

1. lower `g + h` first,
2. smaller estimated slack to incumbent first,
3. closer to the recovered profile / incumbent profile first,
4. closer to incumbent residual-count signature first.

This makes the DP more likely to confirm a strong incumbent quickly and tighten
pruning earlier.

### C. Safe dominance refinement

Review state dominance rules using the incumbent:

- if two states share the same exact abstract signature and one has worse `g`,
  kill it,
- if two states share the same residual-work / machine-state signature and one
  cannot beat the incumbent given the current admissible bound, kill it.

No unsafe heuristic dominance is allowed.

### D. Stronger admissible completion bounds

Keep all pruning bounds admissible.

Priority order:

1. current semigroup completion bounds,
2. stronger profile-derived admissible bounds if provably valid,
3. arithmetic-aware admissible residual bounds only if formally checked.

Do not add surrogate or empirical bounds for pruning.

### E. Profile-guided ordering, not profile restriction

Use the recovered profile and Step-3 incumbent to guide:

- which residual states are explored first,
- which transitions are preferred first.

Do **not** restrict the exact DP to stay near the incumbent unless the
restriction is formally exact.

---

## Concrete implementation sequence

### Phase 1. Audit

Log for representative hard rows:

- initial UB passed into exact DP,
- first incumbent confirmation time,
- states reached / expanded before and after first incumbent-tight pruning,
- pruning reasons.

### Phase 2. Ordering

Implement ordering improvements only:

- incumbent-gap-aware queue ordering,
- incumbent-profile-aware tie breaking,
- early expansion of states closest to the current best realization.

### Phase 3. Dominance

Strengthen safe state merging / dominance only where exact equivalence is
already clear from the DP representation.

### Phase 4. Bound strengthening

Only after ordering and dominance are measured:

- test any stronger admissible completion bounds.

---

## What not to do

- do not add exact-L2 back into the mainline,
- do not present block-DP or branch-and-bound as co-equal exact fallback,
- do not use heuristic similarity to the incumbent as a pruning rule,
- do not turn exact DP into a heuristic search with unsafe restrictions.

---

## Success criterion

Step 4 should be explainable as:

> “If the first three steps do not close the gap, we run a semigroup-guided
> exact dynamic program that exploits the incumbent through admissible pruning,
> safe dominance, and incumbent-guided expansion order.”

That keeps the method exact, clean, and aligned with the paper story.

# Problem

This iteration tests **multi-view adjacent block reconstruction** as a Step-1/Step-3 bridge for hard irregular fixed-`n=1000` K-axis rows.

PLAN28 showed that local block schedulability is not a useful diagnostic: the beam's globally feasible chosen counts fail strict block-local validation at block 0 for easy and hard rows alike. That does not mean block structure is irrelevant. It means the next block experiment must be aligned with how Step 3 actually works: the beam solves a global count-flow problem across block layers.

## Hypothesis

For hard irregular arithmetic, the raw recovered block partition may over-fragment the horizon into too many narrow beam layers. This can make early count decisions brittle and reduce incumbent quality. A small set of alternative adjacent coarsenings may give the same Step-3 beam more useful freedom without redesigning the solver.

## Key distinction from PLAN28

PLAN28 asked:

> Is each block independently locally schedulable?

PLAN29 asks:

> Which adjacent block boundaries should survive before Step 3 beam realization?

Only adjacent blocks may be merged. The experiment changes **which adjacent boundaries are removed**, not the meaning of a block or the global problem.

## Non-goals

- Do not revive `smart_reconstruct(...)`.
- Do not use PLAN28 bad-block rate as the repair signal.
- Do not continue Step-4 corridor exact DP.
- Do not implement column generation, branch-and-price, MIP/SAT, or a new global relaxation.
- Do not change accepted defaults silently.

# Plan 03A: Step 3 Unified Profile Repair

## Goal

Define the ONE method that should survive as Step 3 in the cleaned pipeline.

Recommended method family:

**Profile-guided repair beam with local destroy/repair neighborhoods**

Short name for the paper:

- `profile_repair_beam`

---

## Why this method

This is the best Step-3 choice for both theory and evidence.

### Empirical reason

From the current archive:

- feasible beam is the strongest current Level-2 method
- Lagrangian has been unstable/regressed and is no longer the clean front-runner
- exact-L2 showed that small/moderate open gaps were indeed Level-2 search
  gaps, which reinforces the beam-centered direction

### Theoretical reason

This method stays close to the project’s core ideas:

1. start from semigroup-derived recovered blocks
2. search over multitype block realizations
3. use beam-style bounded search over count vectors
4. intensify by repairing a small subset of blocks locally

This is much closer to the original beam/block theory than:

- a free-standing Lagrangian branch
- a second exact method
- or a full pricing/arc-flow redesign as the mainline

---

## Mathematical interpretation

Step 3 solves:

- a bounded multitype assignment / realization problem over the recovered block
  profile
- with exact or near-exact local block evaluation
- and a heuristic search over the global assignment structure

This is structurally close to:

- multiple subset-sum / multiple knapsack assignment:
  [Caprara, Kellerer, Pferschy 2000](https://doi.org/10.1137/S1052623498348481)
- destroy/repair neighborhood search on combinatorial assignments:
  [Ropke & Pisinger 2006](https://doi.org/10.1287/trsc.1050.0135)
  [clustered VRP large neighborhoods](https://doi.org/10.1016/j.ejor.2018.02.056)

---

## What the unified method should contain

### Core phase: feasibility-first beam

Keep:

- count-vector state
- suffix feasibility pruning
- bounded beam width
- exact Level-3 block evaluation

This remains the main engine for getting the first strong incumbent.

### Intensification phase: local destroy/repair

If the beam returns a finite incumbent but leaves a small gap:

- unfix 2 or 3 blocks
- recompute those blocks by enumerating feasible count vectors under:
  - local capacities
  - residual type demands
- evaluate touched blocks with the current Level-3 exact/local evaluator
- accept only improving replacements

This should be presented as part of the SAME method, not as a second
unrelated heuristic.

In other words:

- the beam produces the incumbent
- local neighborhoods refine it

### Optional adaptivity later

If needed later, adapt the neighborhood choice:

- random block pair
- adjacent blocks
- highest-cost blocks
- highest residual-deviation blocks

But keep that as internal operator selection within the same method family.

---

## What should be removed from Step 3

The final cleaned Step 3 should not keep all of these as co-equal defaults:

- `lagrangian_assign`
- `rg_beam`
- `feasible_counts`
- exact Level-2 B&B

If some of them are kept in code:

- mark them as archival or diagnostic,
- not as part of the final mainline.

---

## Implementation order

### Stage 1

Freeze the current feasible beam as the Step-3 base.

### Stage 2

Add one local destroy/repair layer on top of it:

- start with 2-block neighborhoods
- use exact Level-3 evaluation for touched blocks
- keep wall-clock bounded

### Stage 3

If 2-block neighborhoods help but do not scale enough:

- add 3-block neighborhoods selectively
- or adaptive operator choice

### Stage 4

Only if the beam+local-neighborhood method still stalls on large-B rows:

- then consider out-of-pool augmentation
- for example selective LNS-generated patterns or pricing-lite

That is an escalation within the same Step-3 family, not a replacement of the
method story.

---

## Why not keep Lagrangian as the main Step-3 method

Because current evidence does not support it as the clean winner:

- it drifted from its earlier best state
- it needed multiple corrections and hidden interactions
- the beam still wins the important hard rows
- and its theoretical story is now less aligned with the project than a direct
  profile-guided repair search

Lagrangian can remain in the archive as a useful explored branch, but it should
not currently be the main Step-3 recommendation.

---

## Success criterion

Step 3 should be explainable as:

> “For hard arithmetic, we apply one profile-guided repair beam over the
> recovered blocks, then intensify the best assignment by local destroy/repair
> neighborhoods.”

If you still need to say:

> “sometimes Lagrangian, sometimes beam, sometimes rg-beam, sometimes exact-L2”

then the cleanup failed.

# Candidate Ideas

## 1. Reframe the solver boundary by two orthogonal axes

### Status

Primary conceptual direction for this archive.

### Claim

The solver boundary should not be described only as "large `K`."

Instead, difficulty should be analyzed on two axes:

1. type-count scaling,
2. arithmetic hardness of the type lengths.

### Why it is promising

This immediately explains current observations:

- `K=10` can be easy when the length set is `{1,2,3,4,5,6,7,8,9,10}`,
- `K=6` can be hard when the length set is `{2,3,4,5,7,11}`,
- therefore hardness is not monotone in `K`.

### Implementation consequence

No solver change required to begin. The first action is experimental and
archive-facing:

- classify families,
- rerun by arithmetic class,
- and present the current solver as an arithmetic-aware escalation ladder.

---

## 2. Easy-arithmetic scaling branch

### Status

Immediate run branch.

### Goal

Show how far the current pipeline scales when arithmetic structure is
deliberately favorable.

### Candidate families

- `1..10`
- contiguous `K=8` family `7..14`
- other unit-containing or contiguous families if available

### Expected outcome

- Step 1 should often be exact,
- exact DP may be unnecessary,
- large-`K` alone may turn out to be much less problematic than previously
  suggested.

### Why it matters

This becomes the clean evidence that:

- the solver genuinely scales in `K`,
- and the six-type hard cases should be interpreted as arithmetic-hard rows, not
  as a generic "`K=6` wall."

---

## 3. Hard-arithmetic branch

### Status

Immediate run branch.

### Goal

Stress the arithmetic structure while keeping `K` moderate enough to interpret
what fails.

### Candidate families

- `{2,3,4,5,7,11}`
- `{4,5,6,7,8,9}`
- later additional hard numerical-semigroup families

### Expected outcome

This branch should separate:

- relaxation weakness,
- incumbent-generation weakness,
- and exact-closure weakness.

### Why it matters

This branch will tell us where to spend further algorithmic work:

- better incumbent generation,
- better candidate-pattern generation,
- or stronger exact certification.

---

## 4. Arithmetic-aware incumbent refinement

### Status

First algorithmic follow-up if hard-arithmetic gaps remain.

### Motivation

The current evidence suggests:

- easy arithmetic often closes at Step 1,
- the open rows are mostly hard-arithmetic rows,
- and current small local improvement is too weak.

### Candidate directions

- seeded cost-guided search around the corrected Lagrangian incumbent,
- seeded beam search starting from the corrected Lagrangian assignment rather
  than from scratch,
- stronger neighborhood search on repaired block assignments,
- reuse of the Lagrangian assignment as the center for a narrow cost-guided
  repair core,
- arithmetic-aware pattern-pool widening only on easy small profiles.

### Why it is promising

This keeps the solver story coherent:

- do not redesign the whole pipeline,
- only strengthen the incumbent layer where arithmetic hardness actually bites.

### Current judgment

This remains the best **first** branch for this archive because:

- it matches the current solver evidence,
- it requires the least conceptual jump,
- and it tests the two-axis framing before introducing a new structural method.

### Important caution

This branch should not remain vague.

If it is tried, it should mean one of the following concrete mechanisms:

1. warm-start a short feasibility or cost-guided beam from the Lagrangian block
   assignment,
2. build a seeded local repair that perturbs the incumbent assignment in a
   narrow structured neighborhood,
3. or center the next repair pass explicitly around the incumbent counts rather
   than rerunning a generic search.

---

## 5. Expert extension: bounded-semigroup / dynamic-pricing branch

### Status

Second algorithmic branch, to be considered **after** the current archive
framing is validated.

### Claim

The deeper difficulty on hard-arithmetic families may come less from raw
numerical-semigroup gaps in the unbounded sense, and more from:

- bounded representability,
- bounded representation diversity,
- and the fact that a fixed filtered pattern pool can miss exactly the patterns
  needed by the assignment method.

### Why it is promising

This explains an important observed mismatch:

- a family can have a small Frobenius number and still be hard for the solver,
- because the solver never works in the unbounded semigroup alone,
- it works with bounded type counts and filtered feasible block patterns.

### Candidate directions from the expert branch

- add Apéry / residue-graph diagnostics as arithmetic descriptors,
- replace fixed pattern pools by dynamic pricing inside the Lagrangian loop,
- and treat arc-flow per block as the heavier but cleaner follow-up.

### Current judgment

This looks theoretically strong, but I do **not** think it should replace the
current archive direction.

Instead, it should be appended after the current branch because:

- the two-axis framing is still the right first scientific move,
- the expert branch is best understood as the next hard-arithmetic algorithmic
  escalation,
- and it is stronger as a targeted response to the pattern-ceiling problem than
  as a replacement for the whole archive logic.

---

## 6. Arithmetic-aware family generation

### Status

Possible medium-term dataset branch.

### Goal

Create controlled families where arithmetic hardness varies at fixed `K`.

### Possible levers

- include or exclude `1`,
- vary Frobenius-like gap size,
- vary density / contiguity of the generators,
- vary gcd-normalized spread.
- include hard-arithmetic high-`K` cells, not only moderate-`K` hard families.

### Why it matters

This would turn the new interpretation into a real benchmark design, not only a
post-hoc explanation of current families.

---

## 7. Literature-backed semigroup descriptors

### Status

Paper-facing interpretation branch.

### Goal

Attach simple arithmetic descriptors to each family, for example:

- presence of `1`,
- multiplicity,
- rough Frobenius size,
- density / contiguity,
- Apéry-set or shifted-family behavior if we go deeper.

### Why it matters

If successful, the paper can say more than:

- "family A was hard, family B was easy."

It can say:

- "families with near-complete semigroups are easy for Step 1,"
- "families with larger arithmetic gaps stress profile realization."

## Current implementation order

1. build the archive and rerun current datasets by arithmetic class
2. validate the easy-arithmetic `K`-scaling headline
3. validate the medium-arithmetic branch explicitly, not only easy vs hard
4. add hard-arithmetic high-`K` cross-cells to test whether the two axes compound
5. validate the hard-arithmetic branch as the real open frontier
6. first try arithmetic-aware incumbent refinement on the hard-arithmetic side
5. if the bottleneck still looks like a pattern-ceiling issue, test the expert
   dynamic-pricing branch
7. consider new curated family generation after the current datasets are fully classified

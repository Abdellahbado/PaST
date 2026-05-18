# Experiment Design

## Goal

Design experiments that separate:

1. type-count scaling,
2. arithmetic hardness,

instead of mixing both in one "large-`K`" label.

## Axis definitions

### Axis A. `K` scaling under easy arithmetic

Purpose:

- isolate the dependence on the number of types,
- minimize confounding from semigroup irregularity.

Candidate families:

- `1..10`
- contiguous `K=8` family `7..14`
- other contiguous or unit-containing families if present in the datasets

Primary outputs:

- whether Step 1 is already exact,
- runtime under `step1_exact_guided`,
- exact / near-exact frontier in `n`.

### Axis B. Arithmetic hardness at fixed or moderate `K`

Purpose:

- isolate how arithmetic structure degrades relaxation tightness and profile
  realization.

Candidate families:

- six-type `{2,3,4,5,7,11}`
- six-type `{4,5,6,7,8,9}`
- later curated five- and six-type hard semigroup families if needed

Primary outputs:

- LB quality after Step 1 and after `R_feas`,
- incumbent quality,
- whether the active bottleneck is repair, exact closure, or both.

### Axis C. Medium arithmetic transition rows

Purpose:

- avoid reducing the study to a binary easy/hard contrast,
- and show whether performance degrades smoothly as arithmetic regularity is
  removed.

Candidate families:

- six-type `{4,5,6,7,8,9}`
- other dense contiguous families without `1`

Primary outputs:

- whether they behave closer to easy arithmetic or hard arithmetic,
- and whether the incumbent method switches at the same `K` and `n`.

### Axis D. Cross-cells: hard arithmetic with high `K`

Purpose:

- test whether the two axes compound,
- rather than assuming that hard arithmetic is only relevant at moderate `K`.

Candidate families:

- high-`K` prime-like or sparse irregular sets,
- for example a ten-type irregular family if data or curated rows are available.

Primary outputs:

- exactness / gap frontier,
- runtime growth relative to easy-arithmetic high-`K`,
- and whether the failure mode changes from incumbent quality to pattern
  coverage or exact closure.

## Immediate run plan

### Tier 1. Zero-code validation

1. easy-arithmetic `K=8` rows:
   - confirm exactness and runtime trend as `n` grows
2. easy-arithmetic `K=10` rows:
   - extend beyond `n=1500`
3. medium-arithmetic six-type rows:
   - compare `{4,5,6,7,8,9}` against easy and hard branches
4. hard-arithmetic six-type rows:
   - compare families `2345711` and `456789`
5. if available, curated `K=5` rows:
   - choose one easy arithmetic family and one hard arithmetic family
6. if available, hard-arithmetic high-`K` rows:
   - test at least one `K>=8` irregular family

### Tier 2. Cross-family summary

For each family:

- `K`
- length set
- arithmetic class (`easy`, `medium`, `hard`)
- largest tested `n`
- whether Step 1 is exact
- best finite gap if not exact
- active incumbent method

### Tier 3. Paper-facing tables

#### Table A. Scaling in `K` under easy arithmetic

Columns:

- family
- `K`
- `n`
- runtime
- exact / near-optimal status
- dominant stage

#### Table B. Arithmetic hardness at fixed `K`

Columns:

- family
- arithmetic descriptors
- `K`
- `n`
- LB after Step 1
- final UB
- gap
- active incumbent method

#### Table C. Cross-cells: arithmetic hardness at higher `K`

Columns:

- family
- arithmetic class
- `K`
- `n`
- runtime
- exact / near-optimal status
- active bottleneck

## Immediate next implementation-facing question

The next code changes should only happen **after** we know whether the remaining
gap problem is:

- mostly arithmetic-hard incumbent quality,
- mostly pattern-generation coverage,
- or mostly exact-guidance weakness.

At the moment the evidence suggests:

- easy arithmetic already scales well,
- hard arithmetic still needs stronger incumbent refinement,
- so the next coding branch should target the hard-arithmetic side only.

## Expert add-on after the current plan

If the hard-arithmetic rows still look blocked **after** the current
incumbent-refinement branch, the next experiments should test whether the real
ceiling is the fixed pattern pool itself.

### Branch E1. Arithmetic diagnostics

Measure per family:

- presence of `1`,
- multiplicity,
- simple residue / Apéry descriptors,
- and whether block capacities frequently fall near hard-to-fill residues.

### Branch E2. Dynamic pricing inside the assignment loop

Replace "generate once, then filter" by:

- solving a small bounded-pricing problem per block under current dual weights,
- adding only the patterns that are currently useful,
- and checking whether this removes the quality ceiling on hard-arithmetic rows.

### Branch E3. Arc-flow per block

If pricing still leaves structural gaps, test a compressed arc-flow model per
block as a replacement for explicit pattern enumeration.

### Current judgment

This expert branch should come **after** the archive's current baseline plan,
not before it:

- first validate the two-axis picture,
- then strengthen the incumbent layer,
- only then escalate to pricing / arc-flow if the evidence still points to a
  pattern-ceiling bottleneck.

## Concrete trigger for escalating to pricing

If a hard-arithmetic row shows all of the following:

- the Lagrangian branch returns a feasible but stagnant incumbent,
- seeded incumbent refinement does not improve it,
- and widened fixed pattern pools help only marginally or inconsistently,

then the next branch should immediately be dynamic pricing rather than another
generic local-improvement attempt.

## Logging convention for this archive

When new runs are added, record:

- family / length set,
- arithmetic class,
- `K`, `n`, seed,
- runtime,
- exactness or gap,
- active incumbent method,
- and any observed explanation for success or failure.

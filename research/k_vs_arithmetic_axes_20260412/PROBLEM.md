# Problem

## Title
Two-axis benchmark-extension study: separating type-count scaling from arithmetic hardness in the stateful pricing-aware scheduling solver.

## Core motivation

The current benchmark-extension work showed something important:

- difficulty is **not** explained by `K` alone,
- some high-`K` families are easy because their arithmetic structure makes the
  semigroup relaxation effectively complete,
- while some moderate-`K` families remain hard because recovered-profile
  realization and incumbent generation are arithmetically awkward.

This means the current phrase "scaling in `K`" is not sufficient as a scientific
description of the frontier.

## New research question

> Can we separate the difficulty of the scheduling solver into two independent
> axes:
> 1. type-count scaling (`K`),
> 2. arithmetic hardness of the job-length set,
> and then characterize which parts of the pipeline are responsible for each?

## Proposed interpretation

The method should be presented as one common pipeline with a shared
recovered-block backbone:

1. relaxation (`R_semi`, later `R_feas` if useful),
2. recovered-profile packing / repair,
3. exact closure or certification.

The new claim we want to test is:

- **large `K` alone** is often not the hard part,
- **arithmetic structure** of the type lengths can be an equally important, and
  sometimes dominant, hardness axis.

## Current judgment on the next algorithmic direction

At this stage, the best immediate research direction is still:

- separate the axes cleanly,
- validate the current solver across those axes,
- and only then decide which hard-arithmetic branch deserves heavier
  algorithmic work.

The strongest expert refinement so far is not a contradiction of that plan.
Rather, it sharpens the hard-arithmetic side:

- unbounded semigroup language is useful,
- but the operative solver difficulty may be closer to bounded representability
  and filtered pattern availability,
- which points toward dynamic pricing or arc-flow as later hard-arithmetic
  branches if the current repair layer hits a pattern ceiling.

## Why this matters for the paper

Without separating the axes, a statement such as "the solver becomes hard at
`K=6`" is misleading.

The current evidence already says:

- `{1,2,3,4,5,6,7,8,9,10}` is easy enough that Step 1 is exact through the
  tested `n=3500` rows,
- `{2,3,4,5,7,11}` is much harder even at smaller `K`,
- `{4,5,6,7,8,9}` appears to sit between those two extremes,
- so "hardness vs. `K`" is confounded by semigroup structure.

The new archive exists to remove that confounding.

## Working hypothesis

We should split the study into two axes:

### Axis A. Type-count scaling with easy arithmetic

Use length sets that make the semigroup especially favorable:

- contiguous families,
- families containing `1`,
- or otherwise "dense" numerical semigroups.

This isolates the question:

> How far does the pipeline scale when arithmetic hardness is intentionally low?

### Axis B. Arithmetic hardness at fixed or moderate `K`

Use awkward, sparse, or irregular length sets, such as:

- `{2,3,4,5,7,11}`,
- `{4,5,6,7,8,9}` as a second six-type comparison,
- and later other curated hard semigroup families.

This isolates the question:

> When the arithmetic structure is difficult, which part of the pipeline fails:
> relaxation tightness, primal recovery, or exact closure?

## Immediate experimental consequence

We should stop treating "large-`K`" as one monolithic frontier.

Instead, the next benchmark-facing analysis should report:

- performance by `K`,
- performance by arithmetic-family class,
- and interactions between the two.

## Initial family classification for this archive

### Easy arithmetic

- contiguous families,
- unit-containing families (`1..10`),
- families where the semigroup becomes effectively complete very early.

### Medium arithmetic

- dense families without `1`,
- e.g. contiguous families starting above `1`,
- second-family six-type cases such as `{4,5,6,7,8,9}`.

### Hard arithmetic

- irregular sparse families with larger Frobenius-type gaps,
- e.g. `{2,3,4,5,7,11}`,
- and future curated families chosen to keep semigroup completion weaker and
  cross-block realization harder.

### Important nuance

The archive should not overclaim that Frobenius number alone predicts
difficulty.

Current evidence and expert feedback both suggest the harder question is:

- how bounded type counts and restricted pattern pools interact with the
  arithmetic structure,
- not only how dense the unbounded semigroup is.

This bounded nuance belongs in the main problem statement because it changes the
interpretation of "arithmetic hardness":

- unbounded semigroup completeness can make a family look easy on paper,
- while bounded counts and filtered block-pattern availability can still make
  the realized assignment problem difficult,
- so the operative difficulty axis for the solver is closer to bounded
  representability than to Frobenius number alone.

## Current expected story

The likely paper-facing narrative is:

1. the solver scales well in `K` on easy arithmetic,
2. the main open challenge is arithmetic-hard recovered-profile realization,
3. the large-`K` repair machinery should therefore be described as an
   arithmetic-aware escalation layer, not merely a "`K>=6` patch."

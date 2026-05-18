# Detailed Plan

## Goal

Turn the new two-axis perspective into an actual research and implementation
program, rather than continuing to patch the solver reactively by `K`.

This plan is deliberately ordered:

1. first clarify the framework,
2. then validate it experimentally,
3. then implement only the most justified structural changes,
4. and only escalate to heavier algorithmic redesign if the evidence clearly
   demands it.

The core redesign principle is:

- separate the pipeline into:
  1. block profile,
  2. block assignment,
  3. within-block scheduling,
- then study difficulty along two axes:
  1. type-count scaling,
  2. arithmetic hardness.

---

## Main plan

## Phase 1. Freeze the current solver as a baseline and stop reactive tuning

### Purpose

Before redesigning the story, stop treating each new row as a new patch target.

### Concrete action

Use the current solver state as the baseline branch for this archive:

- keep the current large-`K` incumbent machinery as the comparison baseline,
- do not spend the next cycle tuning the Lagrangian again unless a redesign
  phase explicitly requires it,
- and evaluate all new ideas against this baseline.

### Why this is first

Without freezing the baseline, the paper story will keep drifting and the new
two-axis interpretation will never stabilize.

### Success criterion

We have one stable baseline to compare against in all new tables.

### If this fails

If the current solver state is too unstable or internally inconsistent for a
clean baseline:

1. revert to the last clearly validated corrected-Lagrangian baseline,
2. record that exact baseline in `LOG.md`,
3. only then continue with the rest of the plan.

---

## Phase 2. Recast the method into three explicit levels

### Purpose

The current implementation still mixes:

- block assignment,
- and within-block sequencing/evaluation,

more than it should.

We want the method to be described and eventually implemented as:

1. **Level 1: Block profile**
   - semigroup / feasible relaxation,
   - recovered blocks, capacities, and time windows.
2. **Level 2: Block assignment**
   - assign job counts to blocks under:
     - per-block capacities,
     - global type totals.
3. **Level 3: Within-block scheduling**
   - given one block and a fixed assigned multiset of jobs,
   - compute the minimum-cost schedule for that block exactly.

### Concrete action

Document this decomposition explicitly in the archive and use it to interpret
every current component:

- `R_semi` and related relaxations → Level 1
- Lagrangian / beam / assignment repair → Level 2
- exact per-block sequence evaluation → Level 3

### Why this is next

This is the conceptual redesign that replaces the old reactive
`K=2 -> K=4 -> K=6` patch history.

### Success criterion

The method can be explained as one architecture instead of a stack of cases.

### If this fails

If the decomposition turns out not to match the code or the experiments cleanly:

1. identify exactly which current modules still mix Level 2 and Level 3,
2. record those entanglements as blockers,
3. then make Level 3 separation the first code task before any new theory or
   experiment work.

---

## Phase 3. Run the two-axis baseline grid before new heavy coding

### Purpose

Validate that the two-axis framework actually matches the data.

### Concrete action

Run a structured baseline matrix with the current solver:

#### Axis A. Easy arithmetic

- contiguous or unit-containing families
- target `K = 4, 6, 8, 10`
- measure:
  - Step 1 exactness,
  - runtime,
  - need for later stages

#### Axis B. Medium arithmetic

- dense families without `1`, e.g. `{4,5,6,7,8,9}`
- target `K = 4, 6`, and higher if available
- measure:
  - whether they behave closer to easy or hard

#### Axis C. Hard arithmetic

- irregular/sparse families such as `{2,3,4,5,7,11}`
- target:
  - moderate `K`,
  - and at least one high-`K` cross-cell if available
- measure:
  - whether the failure is:
    - incumbent quality,
    - pattern coverage,
    - or exact closure

### Outputs required

For each family:

- arithmetic class
- `K`
- `n`
- runtime
- exact / gap status
- active incumbent method
- dominant failing stage if not exact

### Why this comes before more coding

Because the new framework must be validated empirically before we invest in a
structural algorithm change.

### Success criterion

We can support the main claim:

- easy arithmetic scales much farther in `K`,
- hard arithmetic is the real open frontier,
- medium arithmetic sits in between.

### If this fails

If the baseline matrix does **not** show a clean two-axis pattern:

1. compute richer arithmetic descriptors,
2. check whether the current family labels are too coarse,
3. add curated families before redesigning the solver further.

---

## Phase 4. Quantify arithmetic hardness with descriptors

### Purpose

Turn the arithmetic axis into something measurable instead of narrative-only.

### Immediate descriptors

For each family, record:

- presence of `1`
- multiplicity (smallest length)
- contiguity / spread
- gcd-normalized length spread
- rough Frobenius-type size
- simple residue / Apéry-style diagnostics

### Important nuance

Do not claim Frobenius number alone explains the solver behavior.

The working interpretation should be:

- unbounded semigroup structure matters,
- but the operative difficulty is closer to bounded representability and
  restricted block-pattern availability.

### Success criterion

Each benchmark family gets a small arithmetic profile in the tables.

### If this fails

If simple descriptors do not correlate with difficulty:

1. add bounded-feasibility proxies,
2. inspect actual recovered block capacities against residue classes,
3. then revisit the arithmetic classification.

---

## Phase 5. First implementation task: separate Level 3 properly

### Purpose

This is the first code change that fits the new framework cleanly.

### Problem it addresses

Current assignment methods still evaluate choices using crude within-block
sequence surrogates or coarse global sequence constructions.

### Concrete action

Replace the current coarse within-block evaluation by an exact or near-exact
per-block scheduling evaluator:

- input:
  - one block,
  - its assigned job multiset,
  - its time window
- output:
  - the true minimum-cost schedule for that block

This should be applied as a Level 3 evaluator:

- after or inside assignment candidate evaluation,
- without redefining the Level 2 combinatorial assignment itself.

### Why this is the first code task

It improves the incumbent side everywhere and makes the architecture cleaner
without yet changing the core Level 2 method.

### Success criterion

- the new evaluator improves or stabilizes UB quality,
- and the code now reflects the Level 2 / Level 3 separation explicitly.

### If this fails

If exact per-block evaluation is too slow or too awkward:

1. add it only on small/medium merged profiles,
2. keep the coarse evaluator as a fallback,
3. and record which regimes justify exact Level 3 evaluation.

---

## Phase 6. Decide whether Level 2 needs only refinement or real redesign

### Purpose

After the two-axis grid and Level 3 separation, decide whether the remaining
hardness is truly a Level 2 structural limitation.

### Decision rule

If hard-arithmetic rows still show:

- strong Level 1 behavior,
- small but persistent gaps,
- and little improvement from better Level 3 evaluation,

then the remaining bottleneck belongs to Level 2.

### Two possible directions

#### Direction 6A. Light Level 2 refinement

Try only one more Level 2 refinement if it is still justified:

- seeded beam from the incumbent assignment,
- seeded cost-guided search centered on incumbent counts,
- or another clearly specified narrow refinement.

This should only be tried if it is now framed cleanly as a Level 2 method,
not as another generic patch.

#### Direction 6B. Structural Level 2 redesign

If the evidence points to a pattern-set ceiling:

- move directly to dynamic pricing inside the assignment loop.

This is the first major redesign candidate because it has:

- a clean theoretical mechanism,
- strong literature support,
- and it directly targets the bounded-pattern bottleneck.

### Success criterion

We choose one Level 2 path based on evidence rather than tuning fatigue.

### If this fails

If neither light refinement nor pricing resolves the pattern ceiling:

1. move to arc-flow per block,
2. or explicitly redesign the Level 2 model around a cleaner block-assignment
   formulation.

---

## Phase 7. Dynamic pricing branch, if needed

### When to activate

Only if the earlier phases show:

- hard arithmetic remains the real bottleneck,
- fixed filtered pattern sets are the main ceiling,
- and lighter Level 2 refinement does not materially improve the hard rows.

### Concrete action

Replace "generate once, then filter" by:

- solve a bounded pricing problem per block under the current dual weights,
- add only the currently relevant patterns,
- repeat within the Level 2 assignment loop.

### Why this is the main structural fallback

It is the cleanest next theory-backed response to the current hard-arithmetic
assignment ceiling.

### Success criterion

Hard-arithmetic rows improve without needing blanket pattern-pool widening.

### If this fails

Escalate to:

- arc-flow / compressed graph per block,
- or a different exact / decomposition model for Level 2.

---

## Phase 8. Paper-facing deliverables

### Deliverable A. A cleaner theory section

The paper should explain:

1. the three levels,
2. the two difficulty axes,
3. and why arithmetic hardness belongs mainly to Level 2.

### Deliverable B. A two-axis experiment section

At minimum:

- easy / medium / hard arithmetic
- low / medium / high `K`
- representative `n` scaling

### Deliverable C. An adaptive pipeline story

The final method should be described as:

- easy arithmetic:
  - Level 1 may already close or nearly close
- hard arithmetic:
  - Level 2 escalation is required
- all cases:
  - Level 3 gives cleaner and stronger incumbents

### Deliverable D. Explicit failure-mode mapping

For unsolved or open-gap cases, identify whether the active bottleneck is:

- Level 1 relaxation,
- Level 2 assignment,
- or Level 3 evaluation / exact closure.

---

## Recommended immediate next step

If we have to choose exactly one next action from this whole plan, it should be:

### Main next step

**Run the two-axis baseline grid with the current solver and add arithmetic
descriptors to the result tables.**

Why this is first:

- it validates the new framework,
- it tells us whether the redesign is actually supported by the data,
- and it prevents us from implementing another heavy method before we know which
  level really needs it.

### Next implementation after that

**Separate Level 3 properly by adding exact per-block scheduling evaluation.**

That is the first implementation step that improves the method while also
making the theory cleaner.

---

## Short fallback ladder

If the main plan stalls, use this order:

1. baseline two-axis grid
2. arithmetic descriptors
3. Level 3 exact per-block evaluation
4. one clean Level 2 seeded refinement
5. dynamic pricing
6. arc-flow
7. curated new family generation if the benchmark matrix is still too uneven

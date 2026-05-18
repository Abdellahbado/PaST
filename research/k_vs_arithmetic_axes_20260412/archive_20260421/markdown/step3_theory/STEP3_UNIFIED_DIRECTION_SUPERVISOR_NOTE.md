# Step 3 Unified Direction: Supervisor Note

Date: 2026-04-17

## Purpose

This note summarizes the current proposed direction for the solver pipeline,
with special focus on **Step 3**. It is intended as a neutral and
supervisor-facing explanation of:

- the current four-step pipeline,
- why Step 3 has become the central algorithmic layer,
- the proposed unified theoretical interpretation of Step 3,
- how the previously separate Step-3 methods fit inside that interpretation,
- what is already justified to claim today,
- and what remains future work rather than implemented fact.

The goal is not to present a finalized theorem or a complete redesign. The goal
is to clarify whether the current direction is conceptually sound and worth
pursuing.

---

## 1. Current Pipeline: Steps 1 Through 4

The current solver is best understood as a four-step pipeline.

### Step 1. Semigroup profile recovery

Step 1 solves a relaxation-driven structural problem. Its role is to:

- produce a strong lower bound,
- recover a block profile or time-window structure,
- and identify a promising coarse decomposition of the instance.

This step is already one of the strongest parts of the method. On easy
families, it can be nearly decisive by itself. On harder families, it still
provides the structural backbone for the rest of the pipeline.

What Step 1 does well:

- strong lower bounds,
- robust structural information,
- stable behavior across many benchmark families.

What Step 1 does not do by itself:

- it does not fully assign the multitype jobs to the recovered blocks,
- and it does not, on difficult rows, certify the original scheduling problem.

### Step 2. Quick realization

Step 2 is a fast constructive realization attempt. Given the profile recovered
in Step 1, it tries to instantiate it quickly using simple packing and
assignment heuristics.

Its purpose is not deep search. Its purpose is to close easy rows cheaply and
to provide a useful incumbent when possible.

What Step 2 does well:

- easy arithmetic families,
- highly fillable contiguous families,
- cases where the Step-1 profile is already easy to realize.

What Step 2 does not do well:

- arithmetic-hard rebalancing across blocks,
- difficult bounded representability,
- instances where several blocks compete for the same difficult job types.

### Step 3. Profile realization

Step 3 is the main middle layer of the method. Its role is:

- to realize the recovered profile from Step 1,
- to decide which job-type counts go into which recovered blocks,
- and to produce a strong feasible incumbent when Step 2 is not sufficient.

This is the layer where the problem changes from “recover useful structure” to
“coordinate exact multitype realization across that structure.”

Historically, several methods were developed for this layer:

- exact realization on the recovered blocks when the state space is still
  manageable,
- specialized small-case repair procedures for very low-dimensional cases,
- restricted-candidate procedures that keep only a promising subset of block
  fillings,
- bounded-search procedures that keep only the most promising partial
  realizations,
- and dual-guided procedures that alter the scoring of assignments to guide the
  search.

The present direction is to reinterpret these as members of **one Step-3
family**, rather than as unrelated alternative methods.

### Step 4. Global exact DP

Step 4 is the final exact authority on the original problem. It uses the best
upper bound available from Steps 2 and 3 and attempts to certify optimality.

Its role is essential, but it is also the most fragile computationally on hard
instances. In particular, for larger `K` and arithmetic-hard families, the
remaining exact search space often stays too large for Step 4 to finish under
reasonable resource limits.

This means that Step 4 is best viewed as:

- the final certifier when the search space has already been collapsed enough,
- not as the main engine that should solve difficult rows from scratch.

---

## 2. Why Step 3 Has Become Central

The current empirical picture is that Step 3 is the real frontier layer.

The pipeline behaves roughly as follows:

- Step 1 gives strong structure and strong lower bounds.
- Step 2 closes easy cases quickly.
- Step 3 determines whether the recovered structure can actually be realized
  well on difficult rows.
- Step 4 can certify optimality only if the incumbent from Steps 2–3 is strong
  enough and the remaining exact search space is already narrow enough.

This is especially important at larger `K`.

The difficulty is not controlled by `K` alone, but larger `K` and harder
arithmetic typically create the same practical effect:

- the Step-3 realization problem becomes combinatorially richer,
- and the Step-4 exact search becomes increasingly expensive.

So the practical bottleneck on hard rows is often not “finding a lower bound,”
but rather:

> how to realize the recovered profile well enough that Step 4 is either
> unnecessary or small enough to finish.

This is why Step 3 deserves a cleaner theory and a cleaner implementation
story. It is the layer where the quality of the final result is often decided.

---

## 3. Motivation for a Unified Step-3 Theory

The project reached a point where Step 3 was being described as a collection of
historical methods:

- some methods tried to solve the recovered-profile realization problem exactly,
- some tried to restrict attention to a smaller promising candidate set,
- some searched broadly but kept only the best partial states,
- some used dual or penalty information to guide the search,
- and some combined these ideas.

That description is operationally useful for development, but weak as a method
story. It creates two problems.

First, it obscures the fact that these procedures are all trying to solve the
same underlying middle-layer problem.

Second, it makes it harder to reason about why one variant should be used for
one regime and another for a different regime.

The present direction is therefore not to claim a brand-new method, but to
state more clearly:

1. what Step 3 is mathematically,
2. how the existing methods relate to that mathematical object,
3. and how that relationship should guide the next implementation steps.

---

## 4. Proposed Unified Interpretation of Step 3

### 4.1 Core statement

The strongest current interpretation is:

> Step 3 solves a **sequential configuration master** over the recovered blocks
> produced by Step 1.

This can also be described as:

- a recovered-profile configuration-selection problem,
- an MMKP-style model at the combinatorial level,
- and a Dantzig-Wolfe or restricted-master structure at the algorithmic level.

These descriptions are compatible, but they emphasize different aspects:

- **MMKP** emphasizes the combinatorial structure. Here this simply means:
  blocks act like classes, candidate block fillings act like choices within
  each class, and the global type totals create coupling constraints across the
  blocks,
- **configuration selection** emphasizes the “choose one filling per block”
  viewpoint,
- **Dantzig-Wolfe / restricted master** emphasizes the decomposition and future
  algorithmic direction.

### 4.2 The master problem

After Step 1, suppose the recovered profile contains blocks indexed by `b`.

For each block `b`, there is a family of candidate feasible fillings or
patterns `p`. A pattern specifies, for that block:

- how many jobs of each type are assigned to it,
- and a corresponding block realization cost.

Define a binary decision variable:

- `lambda_{b,p} = 1` if pattern `p` is selected for block `b`.

Then the Step-3 master is, conceptually:

- choose exactly one pattern per recovered block,
- match the global job-type totals across all blocks,
- minimize total realized cost.

This is the central unifying model.

### 4.3 Why the master is not just a generic MMKP

The MMKP analogy is helpful but incomplete.

The recovered-profile master is richer than a generic flat MMKP for two main
reasons.

First, the pattern costs are highly structured. A block-pattern cost is not an
arbitrary coefficient. It comes from a block-level scheduling computation over
the block’s recovered time window.

Second, the block choices are naturally sequential. Even when the dominant
coupling is global type-count conservation, the solver operates block by block,
and that stage-wise structure is exactly why dynamic programming formulations
are natural here.

For that reason, the most accurate description is:

> a sequential recovered-profile configuration master.

---

## 5. Why This Gives a Natural Restricted-Master or Dantzig-Wolfe View

If each feasible block filling is viewed as a configuration or pattern, then
each pattern becomes a natural **column** of a master problem:

- each column belongs to one block,
- each column contributes a type-count vector,
- each column carries a block realization cost,
- and the master chooses one column per block so that global type demand is
  satisfied.

This is exactly the kind of structure where restricted-master methods and
column-generation ideas are standard in the literature.

This does **not** mean that full column generation or full branch-and-price has
already been implemented. It means the current Step-3 object admits that
interpretation cleanly.

At the current stage, the strongest justified claim is:

> the current Step-3 methods can be understood as policies on the same
> sequential restricted master.

That statement is both strong and defensible without overclaiming maturity of
implementation.

---

## 6. How the Existing Step-3 Methods Fit Inside This Unified View

Under this interpretation, the historical and current methods are not solving
different problems. They are solving the same Step-3 master with different
policies.

Before listing the current variants, it is helpful to describe them in plain
language.

All of them answer the same question:

> given the blocks recovered in Step 1, how should the different job types be
> distributed across those blocks so that the global totals are respected and
> the total realized cost is as small as possible?

They differ mainly in how much of the search space they keep, and in how they
choose which candidate block fillings to examine.

### Exact profile-realization DP

This is the exact mode of the Step-3 family.

Interpretation:

- exact traversal of the recovered-profile state space,
- full or nearly full consideration of feasible block assignments,
- used when the state frontier remains tractable.

In simpler terms:

- if the recovered-profile realization problem is still small enough, we solve
  it directly rather than approximating it.

This is especially justified for:

- `K=2`,
- some `K=4`,
- and any row where the exact recovered-profile state space remains manageable.

### Specialized low-dimensional repair

This is a small-case exact specialization of the same master.

Interpretation:

- exact realization for the very low-dimensional cases,
- exploiting the fact that when there are very few job types, the profile
  realization state space is much smaller and more structured.

This is justified because the state space is dramatically smaller at `K=2`,
which makes exact profile realization practical and low-memory.

### Direct configuration-selection solve

This is a more explicit configuration-selection interpretation of the same
master.

Interpretation:

- choose one configuration per block,
- satisfy global count coupling constraints,
- solve the resulting configuration-selection problem exactly or near-exactly
  when tractable.

This is a particularly natural interpretation for medium-size cases such as
tractable `K=4`.

### Restricted-candidate or “core” solve

This is not a different problem. It is a restriction policy on the same
problem.

Interpretation:

- restrict the candidate pattern set to a promising “core” or kernel,
- then solve the resulting smaller restricted master more aggressively.

This is justified when:

- the full pattern pool is too large,
- but there is reason to believe that only a small subset of patterns will
  matter for a good or exact solution.

In simple terms:

- instead of searching over every possible block filling that appears feasible,
  we first keep only a carefully chosen subset and solve the smaller problem.

### Bounded-frontier search

This is the truncated or scalable mode of the same Step-3 family.

Interpretation:

- traverse the same sequential recovered-profile state space,
- but keep only the best bounded set of frontier states at each stage.

This is justified when:

- exact frontier traversal becomes too expensive,
- but a strong incumbent is still needed quickly.

For harder `K>=4` and especially `K>=6`, this is often the practically
necessary mode.

In simple terms:

- we still search over the same realization problem, but we deliberately keep
  only a limited number of the most promising partial realizations at each
  stage so that the search remains affordable.

### Dual-guided assignment

This is best interpreted as an objective transformation or dual-guided policy
on the same master.

Interpretation:

- the underlying configuration master stays the same,
- but the objective is modified through dual or penalty information to guide
  the search toward promising assignments.

This makes it conceptually different from exact or beam only in search
guidance, not in the underlying Step-3 problem.

In simple terms:

- this does not change the recovered-profile realization problem itself; it
  changes the scoring used to prioritize assignments.

---

## 7. A More Useful Taxonomy: Three Orthogonal Design Axes

The most useful current refinement is to avoid describing the Step-3 variants
as unrelated methods. They are better understood along three orthogonal design
axes.

### Axis 1. Pattern or column policy

Which patterns are available to the master?

- **full**: enumerate all feasible patterns that are tractable to keep,
- **core or kernel**: keep only a restricted promising subset,
- **priced on demand**: generate patterns dynamically rather than fixing the
  entire pool in advance.

### Axis 2. Frontier or search policy

How is the sequential master traversed?

- **exact**: retain the full frontier,
- **beam**: retain only a bounded frontier,
- **greedy or constructive**: follow a small number of guided paths.

### Axis 3. Objective policy

What objective is used to score states or columns?

- **original cost**,
- **dual-guided or Lagrangian-modified cost**,
- **feasibility-first or penalized cost**.

This taxonomy is useful because it turns a flat method list into a design
space. It clarifies that:

- some variants mainly change **which candidate block fillings are considered**,
- some variants mainly change **how much of the search frontier is retained**,
- some variants mainly change **how assignments are scored or guided**,
- and exact profile realization corresponds to the full-frontier exact mode of
  the same master.

---

## 8. Why This Direction Is Practically Motivated

The unified Step-3 interpretation is not only an abstract reframing. It is
motivated by the practical behavior of the current pipeline.

### 8.1 Step 4 is not a good first-rescue layer on hard rows

The global exact DP in Step 4 is exact and important, but on hard rows it is
often too expensive unless Steps 2–3 have already narrowed the search sharply.

This is especially visible when:

- the number of types is moderate or large,
- the arithmetic structure creates fragile representability,
- and many near-optimal residual states survive in the exact search.

In those regimes, Step 4 is best used as:

- a certifier after strong primal recovery,
- not as the main engine that should repair weak Step-3 outputs.

### 8.2 Step 3 already determines the quality of the final incumbent

Empirically, Steps 2–3 often determine whether the final gap is:

- exactly closed,
- tiny but finite,
- or still too loose for Step 4 to certify.

That is why a cleaner Step-3 theory matters. It helps answer:

- why certain regimes favor exact profile realization,
- why others favor restricted or truncated search,
- and what the next principled algorithmic extension should be.

### 8.3 This is a natural continuation, not a detached redesign

The unified theory does not replace the current pipeline. It clarifies it.

The practical reading is:

- Step 1 remains the relaxation and structural recovery layer,
- Step 2 remains the fast realization layer,
- Step 3 becomes a cleanly interpreted sequential configuration master,
- Step 4 remains the final exact certifier.

So this direction is best viewed as a consolidation and sharpening of the
current architecture, not as abandonment of the existing work.

---

## 9. What Is Already Safe To Claim Today

The following points appear safe and well supported by the current archive.

### 9.1 Safe claim: Step 3 is one family, not many unrelated methods

It is now reasonable to state that Step 3 is a single family of
profile-realization methods with:

- exact modes,
- restricted-column modes,
- truncated-search modes,
- and dual-guided variants.

### 9.2 Safe claim: the restricted-master view is legitimate

It is reasonable to state that the recovered-profile realization problem admits
a restricted-master or Dantzig-Wolfe interpretation over block patterns.

This is supported by:

- the structure of the recovered blocks,
- the pattern-selection viewpoint,
- the global type-count coupling,
- and the literature on configuration-based decomposition.

### 9.3 Safe claim: Steps 1–3 form a relaxation-guided primal recovery pipeline

This is a strong and useful description of the overall method:

- Step 1 recovers the structural profile from a relaxation,
- Step 2 attempts fast primal realization,
- Step 3 performs deeper profile realization when Step 2 is insufficient.

This is arguably the cleanest one-sentence narrative for the pipeline up to the
global exact stage.

---

## 10. What Should Still Be Presented As Future Work

The following points are promising, but should remain explicitly future work or
research direction rather than current implemented fact.

### 10.1 Full column generation

It is plausible that a true column-generation loop could be built for Step 3:

- solve a restricted master,
- obtain dual information,
- price new block patterns on demand,
- add those patterns,
- and iterate.

However, this is not yet the correct way to describe the current implementation.

### 10.2 Full branch-and-price

An exact branch-and-price formulation would be a natural exact completion of
the theory, but it should be presented as future work, not as the current
method.

### 10.3 “One exact theorem covers all current modes”

That would overstate the current position. Exact restricted-master or exact
column-generation arguments may support exact variants, but beam and
Lagrangian-guided variants remain heuristic or truncated policies within the
same framework.

---

## 11. Practical Next Step

If this direction is accepted, the most reasonable next implementation step is
not a full branch-and-price rewrite.

A more proportionate next step is:

> an explicit **restricted-master prototype** for Step 3.

That would mean:

- make the Step-3 master explicit in code,
- represent the current pattern pool per block explicitly,
- solve that master with current exact or beam traversal policies,
- add an outer augmentation loop that can inject better missing patterns,
- and test whether this improves the current fixed-pool Step-3 behavior.

This would turn the unified theory into a concrete experiment without requiring
a complete decomposition solver immediately.

---

## 12. Recommended Supervisor-Facing Interpretation

The most neutral and academically appropriate interpretation at this stage is
the following.

1. The pipeline already has a clear four-step structure.
2. Step 3 is the algorithmically decisive middle layer on hard rows.
3. The several historical Step-3 methods are best understood as members of a
   single sequential configuration-master family.
4. The strongest theoretical umbrella for that family is a recovered-profile
   restricted-master or Dantzig-Wolfe interpretation.
5. The practical purpose of this unification is not only conceptual clarity:
   it also gives a principled basis for deciding when to use exact, restricted,
   truncated, or dual-guided Step-3 variants.
6. A natural next experiment is a restricted-master prototype, while full
   column generation or branch-and-price should remain future work.

---

## 13. References Inside the Archive

The main supporting files for this note are:

- [STEP3_UNIFIED_THEORY_RESEARCH.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/archive_20260421/markdown/step3_theory/STEP3_UNIFIED_THEORY_RESEARCH.md)
- [step3_unified_theory_critique.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/archive_20260421/markdown/step3_theory/step3_unified_theory_critique.md)
- [METHOD_BOUNDARIES.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/METHOD_BOUNDARIES.md)
- [PLAN_03C_profile_realization_dp_unification.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/implementation_plans/PLAN_03C_profile_realization_dp_unification.md)
- [PLAN_03F_restore_k2_and_mmkp_selector.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/implementation_plans/PLAN_03F_restore_k2_and_mmkp_selector.md)
- [PLAN_06_step3_restricted_master_prototype.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/implementation_plans/PLAN_06_step3_restricted_master_prototype.md)

---

## 14. Closing summary

The proposed Step-3 direction is not based on introducing a disconnected new
method. It is based on recognizing that the strongest existing Step-3
procedures already solve one common recovered-profile realization problem.

The current proposal is therefore:

- to keep the four-step pipeline,
- to describe Steps 1–3 as relaxation-guided primal recovery,
- to treat Step 3 as a unified sequential configuration master,
- and to use that interpretation to guide the next practical algorithmic
  experiments.

This appears to be a natural and theoretically grounded continuation of the
current work.

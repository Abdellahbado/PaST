# Step 3 Unified Theory Research

Date: 2026-04-16

Supervisor-facing summary:

- [STEP3_UNIFIED_DIRECTION_SUPERVISOR_NOTE.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/archive_20260421/markdown/step3_theory/STEP3_UNIFIED_DIRECTION_SUPERVISOR_NOTE.md)

## Bottom Line

Yes: there is a clean general mother method for Step 3.

The strongest unifying view is:

- **Dantzig-Wolfe / restricted-master configuration optimization**
- on a **recovered-profile pattern master**
- with exact, core-restricted, beam-truncated, and dual-guided variants.

This is stronger than saying only "MMKP".

MMKP is the right combinatorial object.
Dantzig-Wolfe / branch-price is the right algorithmic umbrella.

## 1. The Master Problem

After Step 1 recovers blocks, Step 3 is:

- one recovered block `b` per class
- one feasible filling/pattern `p` per class candidate
- binary decision `lambda_{b,p}` = choose pattern `p` for block `b`

Subject to:

- exactly one pattern chosen per recovered block
- global multiplicities of job types must match totals

Objective:

- minimize total realized cost over chosen block patterns

This is simultaneously:

- a **multiple-choice multidimensional knapsack / MMKP-style model**
- a **configuration-selection model**
- a **set-partitioning / set-packing-style restricted master**

The exact meaning depends on which side is emphasized:

- combinatorial structure: MMKP
- decomposition structure: Dantzig-Wolfe restricted master
- implementation structure: profile-realization DP

## 2. Why Dantzig-Wolfe Is The Best Umbrella

If patterns are the columns, then the recovered-profile model is naturally a
restricted master:

- master variables = pattern-selection variables
- convexity constraints = choose one pattern per block
- coupling constraints = global type counts

This is precisely the kind of structure Dantzig-Wolfe decomposition is built
for.

Then:

- pricing asks whether a missing block pattern with negative reduced cost exists
- branch-and-price gives the exact version
- restricted-column methods give the scalable heuristic versions

So the clean mother framework is:

> **Recovered-profile branch-price / restricted-master optimization**

## 3. How Current And Historical Methods Fit Inside It

### 3.1 Exact fixed-block DP

Interpretation:

- exact solver of the recovered-profile master when the full state frontier is
  tractable

Equivalent view:

- exact dynamic programming over the master columns/configurations

### 3.2 Old `block_repair_dp`

Interpretation:

- specialized exact/small-case master solver for the two-type restricted master

### 3.3 Old `block_repair_mmkp`

Interpretation:

- exact or near-exact MMKP/configuration-selection solve of the same master

This is the clearest direct literature match.

### 3.4 `energy_core`

Interpretation:

- **core / kernel restricted-master search**
- only a promising subset of configurations is retained

This is not a different problem.
It is a different policy for restricting the active column set.

### 3.5 `profile_repair_beam`

Interpretation:

- **width-limited dynamic programming / truncated label-setting**
- over the same recovered-profile master

It differs from exact DP mainly by frontier retention policy.

### 3.6 `lagrangian_assign`

Interpretation:

- **dual-guided restricted-master search**
- or a dual approximation to the coupling constraints

In Dantzig-Wolfe language, this is closely related to reduced-cost guidance and
stabilized pricing.

### 3.7 Arc-flow possibility

Interpretation:

- compact exact reformulation of the same pattern space
- equivalent in spirit to configuration enumeration, but graph-compressed

So arc-flow is not a new theory either.
It is another exact representation of the same master problem.

## 4. The Clean Hierarchy

The best single hierarchy is:

### Mother problem

- recovered-profile configuration master

### Mother algorithm family

- Dantzig-Wolfe / restricted-master optimization

### Exact children

- exact fixed-block DP
- MMKP exact solver
- branch-and-price
- arc-flow exact reformulation

### Restricted / heuristic children

- energy-core / kernel search
- beam-truncated DP
- reduce-and-solve / relax-and-fix restricted master
- Lagrangian-guided restricted search

This gives one theory and many regimes, not many unrelated methods.

## 5. Literature Evidence

### A. Column generation for MMKP

Cherfi and Hifi explicitly studied:

- [A column generation method for the multiple-choice multi-dimensional knapsack problem](https://ideas.repec.org/a/spr/coopap/v46y2010i1p51-73.html)

This is the closest direct confirmation that a Dantzig-Wolfe view is legitimate
for MMKP itself.

### B. Core / kernel methods

- [Development of core to solve the multidimensional multiple-choice knapsack problem](https://doi.org/10.1016/j.cie.2010.12.001)
- [A Core-Based Exact Algorithm for the Multidimensional Multiple Choice Knapsack Problem](https://doi.org/10.1287/ijoc.2019.0909)
- [A two-phase kernel search variant for the multidimensional multiple-choice knapsack problem](https://doi.org/10.1016/j.ejor.2021.05.007)

These support:

- energy-core
- kernel restriction
- exact core expansion
- feasibility-first then quality-first restricted-master search

### C. Reduce-and-solve / restricted master search

- [A “reduce and solve” approach for the multiple-choice multidimensional knapsack problem](https://doi.org/10.1016/j.ejor.2014.05.025)

This supports the view that:

- solving progressively restricted critical subproblems is a standard MMKP
  strategy

### D. Lagrangian-guided neighbourhood / dual guidance

- [Lagrangian heuristic-based neighbourhood search for the multiple-choice multi-dimensional knapsack problem](https://doi.org/10.1080/0305215X.2014.982631)

This supports:

- dual-guided search
- reduced-cost-style steering
- neighbourhoods induced by Lagrangian information

### E. Branch-price umbrella from assignment / packing literature

- [A Branch-and-Price Algorithm for the Generalized Assignment Problem](https://doi.org/10.1287/opre.45.6.831)
- [A Branch-and-Price Algorithm for the Multilevel Generalized Assignment Problem](https://doi.org/10.1287/opre.1060.0323)

These are not MMKP papers, but they strongly justify the exact umbrella:

- master problem with assignment/set-partitioning structure
- pricing subproblem with knapsack / multiple-choice structure

That is very close to the recovered-profile master we have.

### F. Arc-flow exact representation

- [Bin packing and related problems: General arc-flow formulation with graph compression](https://doi.org/10.1016/j.cor.2015.11.009)

This supports:

- arc-flow as an exact compact representation of pattern space
- equivalent in strength to column-generation formulations for packing problems

## 6. The Strongest Unifying Statement

The cleanest statement for the paper is:

> Step 3 solves a recovered-profile configuration master problem. We model each
> recovered block as a class and each feasible block filling as a configuration.
> This yields a Dantzig-Wolfe-style restricted master with global multiplicity
> coupling constraints. Our exact, core-restricted, beam-truncated, and
> dual-guided procedures are different search policies on that same master.

That statement is:

- theoretically clean
- broad enough to include old and new methods
- precise enough to support future design

## 7. Best Candidate For One Single Public Method

If we want one single public Step-3 method that still has internal variants,
the best choice is:

> **Restricted-master profile-realization search**

with internal regimes:

- `exact`
- `core`
- `beam`
- `dual_guided`

That is better than presenting:

- beam
- energy_core
- mmkp
- lagrangian

as separate methods.

## 8. Practical Recommendation

For implementation and paper story:

1. Define Step 3 as a **restricted-master configuration solver**.
2. Keep one public abstraction only.
3. Treat current successful methods as internal policies:
   - exact mode
   - core/kernel mode
   - beam mode
   - dual-guided scoring/pricing mode
4. Keep Step 4 separate, because it solves the original scheduling problem, not
   only the recovered-profile master.

## 9. What This Enables Next

This framework makes several future improvements natural:

- true column generation / pricing-lite over missing block patterns
- stabilized dual guidance
- branch-and-price exact mode
- kernel/core restricted master
- beam as truncated dynamic programming over the master
- arc-flow as compact exact representation

So this is not only a clean interpretation of the past.
It is also the best roadmap for future algorithm design.

## 10. Final Verdict

Yes:

- there is a genuine single mother method
- it is stronger than "MMKP" alone
- the best umbrella is:

> **Dantzig-Wolfe / restricted-master configuration optimization on the
> recovered-profile master**

and your existing methods are best viewed as its instantiations.

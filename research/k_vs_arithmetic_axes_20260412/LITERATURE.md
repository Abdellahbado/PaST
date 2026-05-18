# Literature

This file records literature directions for the new two-axis research program:

1. type-count scaling,
2. arithmetic hardness of the type-length set.

The aim is not only to cite large-`K` packing / repair work, but also to ground
the claim that arithmetic structure itself is an important algorithmic axis.

## 1. Rosales & García-Sánchez: numerical semigroups as the right language

### Reference

Rosales, J. C., and García-Sánchez, P. A.  
**Numerical Semigroups**  
[Springer book](https://doi.org/10.1007/978-1-4419-0160-6)

### Why it matters here

This is the cleanest foundational reference for treating a finite set of job
lengths as generators of a numerical semigroup.

It supports the exact language we need for this archive:

- semigroup completeness,
- Frobenius-type gaps,
- embedding dimension,
- and structural properties of the length set.

### How it may influence our work

- justify the arithmetic-hardness axis formally,
- define family descriptors beyond raw `K`,
- explain why unit-containing families behave so differently.

## 2. Assi / D'Anna / García-Sánchez: applications-oriented numerical semigroups

### Reference

Assi, A., D'Anna, M., and García-Sánchez, P. A.  
**Numerical Semigroups and Applications**  
[Springer book](https://doi.org/10.1007/978-3-030-54943-5)

### Why it matters here

This is useful as a bridge source:

- not only pure semigroup theory,
- but how semigroup structure appears in algorithmic and applied settings.

### How it may influence our work

- justify using semigroup-based descriptors in an optimization paper,
- strengthen the claim that the arithmetic axis is not an ad hoc observation.

## 3. Huang & Tang: periodicity in unbounded knapsack

### Reference

Huang, P. H., and Tang, K.  
**A constructive periodicity bound for the unbounded knapsack problem**  
[Operations Research Letters](https://doi.org/10.1016/j.orl.2012.05.001)

### Why it matters here

Our Step 1 relaxation and recovered-profile logic are not identical to UKP, but
the arithmetic intuition is closely related:

- once the generator set is favorable,
- large capacities become structurally regular,
- and the effective difficulty can collapse.

### How it may influence our work

- support the "easy arithmetic" branch,
- motivate describing some families as entering a regular / near-periodic
  regime much earlier than others.

## 4. Chvátal and Pisinger: hardness is instance-structure dependent, not only size dependent

### References

Chvátal, V.  
**Hard Knapsack Problems**  
[Operations Research 1980](https://doi.org/10.1287/opre.28.6.1402)

Pisinger, D.  
**Where are the hard knapsack problems?**  
[Computers & Operations Research 2005](https://doi.org/10.1016/j.cor.2004.03.002)

### Why they matter here

These are not semigroup papers, but they support an important methodological
claim:

- hardness does not follow from problem size alone,
- benchmark families can be easy or hard because of arithmetic / structural
  details.

### How they may influence our work

- justify splitting "`K` scaling" from "arithmetic hardness,"
- motivate designing benchmark families by structural properties rather than
  only by dimension.

## 5. Canonical coin systems / change-making

### Reference

Fujita-Ramírez? (current useful direct source in this archive context):  
**Characterization of canonical systems with six types of coins for the change-making problem**  
[Theoretical Computer Science 2023](https://doi.org/10.1016/j.tcs.2023.113822)

### Why it matters here

This is not our scheduling problem, but it is directly relevant to the
arithmetic story:

- some generator systems are algorithmically benign,
- others are not,
- and the difference comes from arithmetic structure, not merely from the
  number of generators.

### How it may influence our work

- support the distinction between easy and hard arithmetic families,
- especially for contiguous / nearly-canonical systems versus irregular ones.

## 6. Apéry-set and shifted-family literature

### Reference

Kaplan, N., O'Neill, C., and others  
**Apéry sets of shifted numerical monoids**  
[Advances in Applied Mathematics 2018](https://doi.org/10.1016/j.aam.2018.01.005)

### Why it matters here

This suggests that numerical-semigroup invariants can become regular or
quasipolynomial across structured family shifts.

### How it may influence our work

- if we generate our own arithmetic families later,
- this may help define families whose hardness changes smoothly under controlled
  shifts.

## 7. Volume algorithm / primal recovery

### Reference

Barahona, F., and Anbil, R.  
**The volume algorithm: Producing primal solutions with a subgradient method**  
[IBM / Mathematical Programming reference page](https://researchweb.draco.res.ibm.com/publications/the-volume-algorithm-producing-primal-solutions-with-a-subgradient-method)

### Why it matters here

This remains relevant on the algorithmic side:

- hard-arithmetic rows currently stress incumbent generation,
- not only lower bounds,
- so primal recovery from dual information remains one of the most natural
  refinement directions.

### How it may influence our work

- supports future seeded or averaged primal refinement on hard-arithmetic rows.

## 8. Ramírez Alfonsín and the Frobenius caution

### Reference

Ramírez Alfonsín, J. L.  
**The Diophantine Frobenius Problem**  
[Oxford University Press](https://academic.oup.com/book/9257)

### Why it matters here

This is the right background source for:

- Frobenius numbers,
- Apéry sets,
- residue-class reasoning,
- and why unbounded representability alone is only part of the story.

### How it may influence our work

- supports adding residue / Apéry diagnostics to the archive,
- but also reminds us not to overinterpret Frobenius number as the whole
  hardness explanation for bounded scheduling instances.

## 9. Gilmore-Gomory / pricing as the expert follow-up

### References

Gilmore, P. C., and Gomory, R. E.  
**A Linear Programming Approach to the Cutting-Stock Problem**  
[Operations Research 1961](https://doi.org/10.1287/opre.9.6.849)

Gilmore, P. C., and Gomory, R. E.  
**A Linear Programming Approach to the Cutting-Stock Problem, Part II**  
[Operations Research 1963](https://doi.org/10.1287/opre.11.6.863)

Vanderbeck, F.  
**Computational study of a column generation algorithm for bin packing and cutting stock problems**  
[bibliographic page](https://ftp.iaorifors.com/paper/30121)

### Why they matter here

These are the cleanest references for the expert suggestion that:

- fixed pattern pools are not the only way to run the block-assignment layer,
- useful patterns can be generated on demand by pricing,
- and arithmetic-hard rows may benefit precisely because pricing avoids
  filtering out the needed patterns in advance.

### How they may influence our work

- support a dynamic-pricing extension of the current Lagrangian branch,
- strengthen the theoretical story if the hard-arithmetic frontier turns out to
  be pattern-generation limited.

## 10. Arc-flow as the heavier structural replacement

### References

Valério de Carvalho, J. M.  
**Exact solution of bin-packing problems using column generation and branch-and-bound**  
[journal page / bibliographic access](https://doi.org/10.1023/A:1018952112619)

Brandão, F., and Pedroso, J. P.  
**Bin packing and related problems: General arc-flow formulation with graph compression**  
[Computers & Operations Research 2016](https://doi.org/10.1016/j.cor.2015.11.009)

### Why they matter here

Arc-flow is the cleanest expert alternative to explicit pattern enumeration:

- feasible fillings are encoded as paths in a compressed graph,
- arithmetic structure is handled structurally rather than by filtered pattern
  lists,
- and the method is well grounded in cutting-stock / packing literature.

### How it may influence our work

- keep as the stronger follow-up after dynamic pricing,
- especially if hard-arithmetic failures still look like missed-pattern
  failures rather than incumbent-refinement failures.

## 11. Multiple subset sum as a bridge model

### Reference

Caprara, A., Kellerer, H., and Pferschy, U.  
**The Multiple Subset Sum Problem**  
[SIAM Journal on Optimization](https://doi.org/10.1137/S1052623499357440)

### Why it matters here

This is a very relevant bridge reference for the block-assignment layer:

- multiple capacities,
- global item counts,
- assignment across bins or blocks,
- and exact-fill structure.

### How it may influence our work

- supports treating the recovered-block assignment as a structured multi-bin
  allocation problem,
- gives a closer literature anchor for the hard-arithmetic cross-block
  realization layer than generic knapsack citations alone.

## Main takeaway from the literature

The current literature picture supports the new archive design:

- numerical-semigroup structure gives the right language for the arithmetic
  axis,
- knapsack hardness literature supports the idea that difficulty is strongly
  family-dependent,
- primal-recovery literature remains relevant for the rows where arithmetic
  hardness survives even after good relaxations,
- and the expert follow-up is now clear in the literature too:
  pricing first, arc-flow second, if the hard-arithmetic bottleneck is really a
  pattern-ceiling issue.

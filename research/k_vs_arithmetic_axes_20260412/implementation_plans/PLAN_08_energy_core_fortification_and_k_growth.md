# PLAN 08 — Energy-Core Fortification And K-Growth

Date: 2026-04-17

Status: NEW

## 1. Purpose

The `energy_core` path has now re-emerged as the strongest practical `K=4`
method. The next step is **not** full column generation yet. The next step is
to fortify `energy_core` using the strongest ideas from MMKP/core/kernel and
restricted-master literature, then test whether the strengthened method can
move beyond the current `K=4` regime into selected `K=6` pilots.

This plan is deliberately staged:

1. stabilize and speed up `K=4`,
2. add bounded strengthening ideas that remain compatible with the current
   Step-3 theory,
3. test whether the strengthened core idea transfers to `K>=6`,
4. and only then decide whether true pricing / column generation is warranted.

## 2. Why This Plan Before Full Column Generation

The literature supports a clear ordering of effort:

- **core / kernel / reduce-and-solve** methods are often the fastest way to
  strengthen a configuration master without implementing a full LP master plus
  pricing loop,
- **pricing / branch-price** is the cleanest eventual theory,
- but it is a substantially larger engineering step and should follow only if
  the fortified core stalls.

So for the current project:

- **fortify `energy_core` now**
- **treat pricing-lite as a bounded fallback**
- **defer full CG / branch-price**

## 3. Literature-Grounded Takeaways

### A. Approximate core + exact solve

The MMKP core literature shows that a strong approximate core can be identified
from a relaxation base point and then solved exactly with low memory.

- Ghasemi and Razzazi, 2011:
  - [Development of core to solve the multidimensional multiple-choice knapsack problem](https://doi.org/10.1016/j.cie.2010.12.001)
  - core is built from a relaxation base point and then used in an exact solve
- Mansini and Zanotti, 2020:
  - [A Core-Based Exact Algorithm for the Multidimensional Multiple Choice Knapsack Problem](https://doi.org/10.1287/ijoc.2019.0909)
  - recursive variable fixing, almost constant memory, exact-to-heuristic
    conversion

### B. Two-phase kernel search

Kernel search can separate:

1. fast feasibility construction,
2. then quality improvement with dynamic kernel expansion

- [A two-phase kernel search variant for the multidimensional multiple-choice knapsack problem](https://doi.org/10.1016/j.ejor.2021.05.007)

This is directly relevant to our current `energy_core` issue:

- some hard seeds need a fast feasible closure path,
- others need a quality-focused second phase.

### C. Reduce-and-solve / variable fixing

The MMKP literature repeatedly uses linear-relaxation information to fix
variables or groups and solve only the reduced hard core.

- [A “reduce and solve” approach for the multiple-choice multidimensional knapsack problem](https://doi.org/10.1016/j.ejor.2014.05.025)

This strongly supports:

- fixing “obvious” block/pattern choices,
- isolating uncertain blocks,
- and solving only the reduced configuration core exactly.

### D. Minimal expansion / reduction matters

For multiple-choice knapsack, much of the work is in sorting and reduction, and
strong algorithms focus on **minimal expansion** and **minimal enumeration**.

- [A minimal algorithm for the multiple-choice knapsack problem](https://doi.org/10.1016/0377-2217(95)00015-I)
- [A hybrid dynamic programming/branch-and-bound algorithm for the multiple-choice knapsack problem](https://doi.org/10.1016/0377-0427(93)E0264-M)

This supports:

- more aggressive pattern reduction,
- adaptive expansion instead of fixed-wide cores,
- stronger Lagrangian/reduction bounds inside exact traversal.

### E. Pricing for integrality should be a bounded add-on

Recent branch-price literature supports **pricing for integrality** and
restricted-column repair, but also makes clear that this is a larger algorithmic
step.

- Maher and Rönnberg, 2023:
  - [Integer programming column generation: accelerating branch-and-price using a novel pricing scheme for finding high-quality solutions in set covering, packing, and partitioning problems](https://doi.org/10.1007/s12532-023-00240-w)

This motivates a limited **pricing-lite augmentation** stage for us, not full CG.

### F. If pricing is added later, stabilization matters

If we later move toward true restricted masters, the literature says master
degeneracy and dual instability quickly become practical issues.

- [Dynamic Aggregation of Set-Partitioning Constraints in Column Generation](https://doi.org/10.1287/opre.1050.0222)
- [Dual Inequalities for Stabilized Column Generation Revisited](https://doi.org/10.1287/ijoc.2015.0670)

These are not immediate implementation tasks, but they define the right future
completion path.

## 4. Current Code Grounding

The current `energy_core` implementation already has the right skeleton:

- pattern generation:
  - [generate_energy_core_patterns]( /Users/mac/Documents/Study/PFE/PaST/solvers/cpp/stateful_dp_solver.cpp:2313 )
- exact core traversal:
  - [block_repair_energy_core_ub]( /Users/mac/Documents/Study/PFE/PaST/solvers/cpp/stateful_dp_solver.cpp:2692 )
- pattern-pool reuse in other Step-3 variants:
  - beam / Lagrangian already call the same pattern generator

Important current limitations:

1. the pattern center is still mainly proportional-to-work,
2. the core window uses a simple symmetric `delta` around expected prefix
   counts,
3. expansion is coarse rather than uncertainty-driven,
4. fixing/reduction is still limited,
5. no bounded augmentation loop exists once the initial pattern pool is fixed.

These are exactly the places where the literature suggests improvements.

## 5. Main Objective

Strengthen `energy_core` so that:

1. `K=4` closes robustly with lower seed sensitivity and lower large-`n`
   runtime,
2. the same fortified idea can be tested honestly on selected `K=6` rows,
3. and we learn whether a pricing-lite augmentation is necessary before any
   full restricted-master implementation.

## 6. Implementation Order

### Phase A — Instrumentation First

Before changing the method, add explicit diagnostics for:

- total patterns
- max patterns per block
- per-block retained patterns after pruning
- time spent in:
  - pattern generation
  - completion table
  - exact core traversal
- per-block pruning counters:
  - core window prune
  - suffix prune
  - transition prune
  - bound prune
- delta level actually needed before success
- number of blocks that are effectively “fixed” early

Required output:

- one compact CSV or trace summary per row, not only stderr dumps

Goal:

- explain why `seed 0` is slow when `seed 1` is fast on the same family

### Phase B — Stronger Core Definition

Replace the current purely proportional center with a better **surrogate-relaxation
center**.

Required idea:

- use a cheap relaxation or dual-guided estimate to define:
  - a better per-block target count vector
  - or better prefix target intervals

Allowed implementations:

1. use the existing Lagrangian / dual-like signals already in the solver,
2. or add a cheap linear/surrogate relaxation proxy to estimate the core center,
3. or seed from the best available feasible incumbent and derive a local center
   around its prefix counts

Do not add an external LP solver for this phase.

Success condition:

- fewer patterns survive into the hard blocks
- fewer delta expansions needed

### Phase C — Adaptive Core Expansion

Current expansion is controlled mostly by fixed `delta` schedules.

Replace this with a more adaptive rule:

- start narrow,
- expand only where mismatch/uncertainty is concentrated,
- expand more on a few hard blocks/types instead of uniformly everywhere

Candidate rules:

- type-specific delta based on residual mismatch
- block-specific delta based on prefix uncertainty
- incumbent-aware expansion around good prefix counts

This is strongly supported by:

- core-expansion literature,
- minimal-enumeration knapsack ideas,
- and our own observed seed sensitivity.

### Phase D — Reduction And Dominance

Add stronger pre-core reduction.

Required directions:

1. **pattern dominance**
   - drop patterns that are worse on both local cost and count signature quality
2. **block ordering**
   - visit harder / more discriminating blocks earlier
3. **type/order reduction**
   - prioritize long or scarce types in pattern generation and pruning
4. **safe fixing**
   - if a block has only one strongly surviving pattern under all active
     windows, fix it early

This phase should directly reduce:

- `total_patterns`
- `max_patterns`
- `state_keep` pressure

### Phase E — Two-Phase Energy Core

Split `energy_core` into:

1. **feasibility phase**
   - get an exact/feasible realization quickly
2. **quality phase**
   - polish only if needed

Required rule:

- if feasibility already closes the row, stop
- if feasibility returns a finite incumbent but not closure, run a quality
  polish on the reduced uncertain subset only

This matches the two-phase kernel-search literature and is especially important
for larger `n` and for `K>=6`.

### Phase F — Bounded Augmentation (Pricing-Lite)

Only after Phases A–E are measured.

Implement a **small augmentation loop**:

1. run current fortified energy-core on the initial pool,
2. identify the most constrained or mismatched blocks,
3. generate a bounded number of extra patterns for only those blocks,
4. rerun the exact/core solve

This is **not** full column generation.

It is a practical bridge between:

- today’s fixed core,
- and tomorrow’s restricted-master pricing.

Allowed augmentation triggers:

- high residual mismatch on a type
- many delta expansions
- unusually large seed sensitivity
- pattern scarcity on a hard block

Do not add LP master solving or true reduced-cost pricing in this plan.

### Phase G — K=6 Transfer Pilot

Only after `K=4` is stable.

Run the fortified energy-core on a small selected `K=6` pilot set:

- one easier six-type row
- one medium/hard six-type row

Recommended rows:

- a medium arithmetic six-type row
- a hard arithmetic six-type row already used in the archive

Goal:

- determine whether fortified core is:
  - still useful as an incumbent builder
  - or competitive with beam
  - or too expensive without augmentation

The `K=6` pilot is diagnostic, not yet a blanket policy change.

## 7. What Not To Do

Do not:

- implement full column generation or branch-and-price here
- add an external LP/MIP dependency
- add local search
- reintroduce archived exact-L2 into the mainline
- broaden this into a general Step-4 redesign
- accept runtime improvements on easy rows as proof of success

## 8. Benchmark And Measurement Plan

### Group 1 — K=4 stabilization

At minimum:

- `3567_plus n=3500`
- `3567_plus n=5000`
- `g3567 n=1500`
- `g3567 n=2500`
- `g3567 n=3500`
- `g3567 n=5000`
- both seeds where available

Measure:

- exact closure
- runtime
- pattern counts
- expansion depth
- per-block pruning statistics
- seed variance

### Group 2 — K=6 pilots

At minimum:

- one medium six-type row
- one hard six-type row

Measure:

- finite incumbent quality
- whether closure improves over current beam-only policy
- whether runtime remains bounded enough to justify further work

## 9. Success Criteria

This plan succeeds if:

1. `K=4` runtime and seed sensitivity improve materially while preserving exact
   closure,
2. the method remains low-memory and controlled,
3. the instrumentation clearly explains where runtime goes,
4. the bounded augmentation phase is either shown useful or cleanly ruled out,
5. and the `K=6` pilot tells us whether fortified core is viable beyond `K=4`.

## 10. Deliverables

Update:

- [research/k_vs_arithmetic_axes_20260412/LOG.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/LOG.md)
- [research/k_vs_arithmetic_axes_20260412/RESULTS.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/RESULTS.md)
- [research/k_vs_arithmetic_axes_20260412/BLOCKERS.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/BLOCKERS.md)
- [research/k_vs_arithmetic_axes_20260412/METHOD_BOUNDARIES.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/METHOD_BOUNDARIES.md)

Create, if useful:

- `archive_20260421/markdown/k4_history/ENERGY_CORE_FORTIFICATION_NOTE.md`

## 11. Expected Effort

Rough sizing:

- Phases A–D: **medium** code change
- Phase E: **medium** code change
- Phase F: **medium-to-large** code change
- true CG / branch-price: **large** code change and explicitly out of scope here

## 12. Final Recommendation

The preferred next sequence is:

1. fortify `energy_core`,
2. stabilize `K=4`,
3. test bounded augmentation,
4. run `K=6` pilots,
5. and only then decide whether full pricing / restricted-master work is worth
   the extra implementation cost.

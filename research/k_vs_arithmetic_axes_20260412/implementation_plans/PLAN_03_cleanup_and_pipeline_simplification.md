# Plan 03: Cleanup and Pipeline Simplification

## Purpose

Clean the method at the conceptual and implementation-policy levels so the
paper and the solver tell the same story again.

This plan is not about adding another method. It is about:

1. deciding what belongs in the final method,
2. demoting what is only diagnostic,
3. and restoring the exact DP as the only exact fallback.

---

## Final target pipeline

The cleaned method should be described as four steps:

1. **Step 1: Semigroup profile recovery**
   - compute the semigroup relaxation
   - recover one or a few high-quality block profiles

2. **Step 2: Fast profile realization**
   - FFD/BFD/random packing-style realizers
   - intended for easy arithmetic and easy profiles

3. **Step 3: Unified hard-case profile repair**
   - one method only
   - used when Step 2 leaves a gap
   - must handle arithmetic-hard cases

4. **Step 4: Exact certification**
   - semigroup-guided exact DP
   - the ONLY exact method in the final pipeline

---

## What must leave the final method story

These may remain in the archive as experiments, but should not remain part of
the final conceptual method unless later evidence forces them back in:

- Lagrangian as a co-equal default Level-2 branch
- `rg_beam`
- `feasible_counts`
- exact Level-2 branch-and-bound
- post-Lagrangian beam polish
- any hidden hybrid fallback that changes the solver policy without being part
  of the formal method story

Interpretation:

- exact-L2 was useful diagnostically,
- but it should be archived as evidence, not promoted as a second exact stage.

---

## Step-3 method selection: high-level decision

Based on the current evidence, the preferred Step-3 family is:

**A beam-centered, profile-guided repair method with local out-of-pool
neighborhood repair.**

This should replace the current crowded Level-2 zoo.

Why this is the preferred direction:

1. the current feasible beam is the strongest empirical Level-2 method
2. it is already close to the project’s beam-packing intuition
3. it lives naturally on top of the recovered semigroup profile
4. it can be extended by local destroy/repair neighborhoods without changing
   the overall theory
5. it avoids introducing another exact method

What this means:

- keep the beam logic as the core
- add better local search / neighborhood repair inside the SAME method
- do not keep Lagrangian and beam as co-equal long-term branches

See:

- [PLAN_03_step3_unified_profile_repair.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/implementation_plans/PLAN_03_step3_unified_profile_repair.md)

---

## Step-4 direction: exact DP reactivation

The exact DP must return to being the real final fallback.

That means:

1. it receives the best UB from Steps 2–3
2. it uses only safe pruning and dominance
3. it is the only method allowed to claim certification / exactness

The exact DP should be strengthened, but not replaced by a new exact search
family.

See:

- [PLAN_04_exact_dp_reactivation.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/implementation_plans/PLAN_04_exact_dp_reactivation.md)

---

## Literature-backed rationale

The cleanup direction is consistent with the literature:

- Large neighborhood destroy/repair methods are well established for large
  combinatorial assignment layers:
  [Ropke & Pisinger 2006](https://doi.org/10.1287/trsc.1050.0135)
- Clustered and decomposed assignment+local-order problems are often handled
  by one assignment method plus local neighborhoods, not by many unrelated
  co-equal heuristics:
  [Large multiple neighborhood search for clustered VRP](https://doi.org/10.1016/j.ejor.2018.02.056)
- The Level-2 abstraction is close to multiple subset-sum / multiple knapsack
  assignment:
  [Caprara, Kellerer, Pferschy 2000](https://doi.org/10.1137/S1052623498348481)
- Exact DPs in scheduling commonly become practical by adding lower bounds,
  node merging, and heuristics without ceasing to be DPs:
  [Bürgy, Hertz, Baptiste 2020](https://doi.org/10.1016/j.cor.2020.105063)

---

## Required cleanup actions

1. **Demote exact-L2**
   - keep code only if useful diagnostically
   - remove it from the default final-method narrative
   - do not let it define policy by default

2. **Audit Step 2**
   - keep only the fast realization attempts that clearly belong to the easy
     arithmetic regime
   - if a module is really an exact subsolver, do not advertise it as part of
     the final mainline unless you explicitly want two exact stages

3. **Collapse Step 3 to one method family**
   - beam-centered, profile-guided repair only
   - one method, possibly with internal phases

4. **Refocus Step 4**
   - exact DP as sole certification stage
   - use UB from Steps 2–3 to make it effective

---

## Success criterion

At the end of this cleanup, the method can be explained in one paragraph
without sounding patched:

- semigroup profile
- easy realization
- one hard-case repair method
- one exact certification method

If the method still needs a long list of special-case branches to explain the
Level-2 layer, this cleanup has not succeeded.

---

## Execution update (2026-04-13)

Plan-03 cleanup has been implemented in mainline policy:

- default Step 3 is now `profile_repair_beam`
- default no longer runs `lagrangian_assign`, `rg_beam`, `feasible_counts`,
  or post-Lagrangian beam polish
- exact-L2 is no longer active by default and is diagnostic-only unless
  explicitly applied

Policy mismatch resolved:

- previous default dispatch had a crowded Level-2 branch list under
  `PAST_RELAXED_BINPACK_SOLVER=default`; this has been collapsed so default
  behavior follows one Step-3 family only.

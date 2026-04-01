# Initial Findings On The `K`-Scaling Boundary

Date: 2026-04-01

## Purpose

This note records an initial benchmark specifically designed to test the claim
that increasing the number of distinct job sizes `K` is likely to be the main
boundary of the method.

The goal was not to prove the final boundary rigorously. The goal was to check
whether this claim is already visible in realistic low-variability and
moderate-spread families, using the current production solver.

---

## Benchmark Design

We created a dedicated benchmark generator:

- [generate_k_boundary_benchmark.py](/Users/mac/Documents/Study/PFE/PaST/scripts/generate_k_boundary_benchmark.py)

and a dedicated runner:

- [run_k_boundary_benchmark.py](/Users/mac/Documents/Study/PFE/PaST/scripts/run_k_boundary_benchmark.py)

The benchmark keeps the setting intentionally simple:

- machine fixed to `TWOSBY`,
- horizon multiplier fixed to `lambda = 1.3`,
- real OTE energy-price traces,
- moderate instance sizes,
- only the processing-time group and therefore `K` are changed.

Families used:

| Family | Processing-time groups | `K` | `n` values |
|---|---|---:|---:|
| `K_contig` | `{7,8,9}` | 3 | 100, 200, 300 |
| `K_contig` | `{7,8,9,10}` | 4 | 100, 200, 300 |
| `K_contig` | `{7,8,9,10,11}` | 5 | 100, 200, 300 |
| `K_contig` | `{7,8,9,10,11,12}` | 6 | 100, 200 |
| `K_contig` | `{7,8,9,10,11,12,13}` | 7 | 100, 200 |
| `K_contig` | `{7,8,9,10,11,12,13,14}` | 8 | 100, 200 |
| `K_moderate_spread` | `{3,5,6,7}` | 4 | 100, 200, 300 |
| `K_moderate_spread` | `{5,7,9,11,13}` | 5 | 100, 200 |
| `K_moderate_spread` | `{4,5,6,8,9,11,12}` | 7 | 100, 200 |

This benchmark is meant to be realistic rather than adversarial:

- the contiguous families stay close to the paper's low-variability narrative,
- the moderate-spread families add arithmetic diversity without becoming
  contrived,
- and no scarcity was introduced in this first pass.

---

## What Was Run

We ran:

1. the full default solver path via `ablation-stdin full`,
2. the relaxation hierarchy via `relax-hierarchy-stdin`,

using the representative `s0` seed first.

Output files:

- [k_boundary_202604_s0.csv](/Users/mac/Documents/Study/PFE/PaST/results/k_boundary_202604_s0.csv)
- [k_boundary_202604_highK_s0.csv](/Users/mac/Documents/Study/PFE/PaST/results/k_boundary_202604_highK_s0.csv)

Important limitation:

- the hierarchy run used a short exact-time limit for diagnosis, so many rows do
  not have an exact `opt` value recorded;
- therefore this note is mainly about **pipeline behavior** and **backup need**,
  not yet about exact lower-bound gaps on all rows.

---

## Main Result

In this first realistic `K`-scaling test, simply increasing the number of
distinct job sizes did **not** create a visible failure boundary for the method.

Even up to:

- `K = 8`,
- `n = 200`,

the production solver still closed all tested representative rows at:

- `step_reached = fwd_relax`

That means the instances were solved within the default Phase-1 path:

1. semigroup relaxation,
2. relaxed-profile recovery,
3. heuristic or exact fixed-profile certification,

without needing:

- `R_feas`,
- `R_feas + Lagrangian`,
- smart reconstruction,
- or exact multiset DP fallback.

---

## Compact Result Summary

### Representative `K <= 6` run

| `K` | Rows | Avg runtime | Phase-1 closes | `R_feas > R_semi` |
|---:|---:|---:|---:|---:|
| 3 | 3 | `1.324s` | 3/3 | 0/3 |
| 4 | 6 | `1.273s` | 6/6 | 0/6 |
| 5 | 5 | `1.520s` | 5/5 | 0/5 |
| 6 | 2 | `1.224s` | 2/2 | 0/2 |

### Representative high-`K` run

| `K` | Rows | Avg runtime | Phase-1 closes | `R_feas > R_semi` |
|---:|---:|---:|---:|---:|
| 7 | 4 | `1.375s` | 4/4 | 0/4 |
| 8 | 2 | `2.000s` | 2/2 | 0/2 |

Representative high-`K` rows:

| Instance | `K` | `n` | Runtime | Winning stage |
|---|---:|---:|---:|---|
| `K_contig_p7_8_9_10_11_12_13_n100_s0` | 7 | 100 | `0.484s` | `fwd_relax:random_bf` |
| `K_contig_p7_8_9_10_11_12_13_n200_s0` | 7 | 200 | `2.565s` | `fwd_relax:ffd` |
| `K_contig_p7_8_9_10_11_12_13_14_n100_s0` | 8 | 100 | `0.806s` | `fwd_relax:ffd` |
| `K_contig_p7_8_9_10_11_12_13_14_n200_s0` | 8 | 200 | `3.195s` | `fwd_relax:ffd` |
| `K_moderate_spread_p4_5_6_8_9_11_12_n100_s0` | 7 | 100 | `0.330s` | `fwd_relax:random_ff` |
| `K_moderate_spread_p4_5_6_8_9_11_12_n200_s0` | 7 | 200 | `2.123s` | `fwd_relax:ffd` |

---

## Interpretation

### 1. `K` alone is not yet the boundary

The initial data does **not** support the claim that merely moving from `K=3`
to `K=5`, `6`, `7`, or even `8` already exposes the boundary of the current
method, at least not in these realistic low-variability / moderate-spread
families.

### 2. The practical bottleneck is still not the semigroup stage

The semigroup-based Phase 1 remained dominant. That suggests that:

- increasing `K` changes the transition set size,
- but does not automatically make the semigroup relaxation weak,
- and does not automatically force the expensive fallbacks.

### 3. The backup methods were not needed here

On all representative rows in this benchmark:

- `R_feas = R_semi`

So this benchmark is **not** a backup-method showcase.

### 4. A more precise boundary hypothesis is needed

The more plausible boundary is not:

- "large `K` by itself"

but rather:

- "large `K` combined with bounded scarcity or count-collision structure"

In other words, what likely matters is not only the number of distinct sizes,
but whether those sizes interact in a way that creates:

- hidden inventory ambiguity,
- unpackable relaxed profiles,
- or a large certification search space.

---

## What This Means For The Paper

The proposal that "`K`-scaling should be tested because it may be the boundary"
is good.

But the initial evidence suggests a more careful phrasing:

- `K`-scaling is a worthwhile boundary experiment,
- but realistic low-variability families up to `K=8` do not yet expose that
  boundary in the current solver,
- so `K` alone should not be presented as the proven limit of the method.

The paper-safe statement is:

> Increasing the number of distinct job sizes is a natural structural stress
> axis, because the exact fallback and certification layers depend more
> strongly on per-type combinatorics than the relaxed semigroup DP. However, in
> our initial realistic K-scaling benchmark, the default Phase-1 solver path
> remained effective up to K=8, suggesting that K alone is not sufficient to
> expose the practical boundary of the method.

---

## Recommended Next Step

If we want to find the true structural boundary more honestly, the next
benchmark should combine:

1. larger `K`,
2. bounded scarcity in a few critical types,
3. moderate rather than extreme horizon slack,
4. realistic low-variability or moderate-spread groups.

That would test whether the real limit is:

- not `K` by itself,
- but `K` plus hidden count-collision structure.

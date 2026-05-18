# End-to-End Pipeline and Scaling Summary

Status date: 2026-05-05

This file is the single start-to-finish summary for this research thread. It is
meant to answer four questions in one place:

1. What is the solver pipeline?
2. How did we solve the original benchmark and the benchmark job groups?
3. How did we scale first in `n` and then in `K`?
4. What exactly did we do with the real electricity-price profile when the
   horizon became large?

For code-to-result provenance and later HPC reruns, see:

- [PAPER_HPC_REPRODUCIBILITY_MAP.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/PAPER_HPC_REPRODUCIBILITY_MAP.md)

## 1. The Four-Step Pipeline

The current method is best understood as one pipeline with four stages.

### Step 1 — Semigroup lower bound and profile recovery

We first solve a relaxed dynamic program that ignores the exact finite job
multiset and only reasons about achievable total processing time and total raw
work. This gives:

- a valid lower bound;
- a recovered time/block profile;
- a candidate structure that later stages try to realize with the real jobs.

Main code anchors:

- `compute_relaxed_dp_table(...)`
- `solve_relaxed_dp_with_binpack(...)`

in:

- `solvers/cpp/stateful_dp_solver.cpp`
- `solvers/cpp/stateful_compare.cpp`

### Step 2 — Quick realization

We then try to realize the recovered profile quickly using a direct greedy packer
such as `ffd`.

This is the cheapest realization stage. On easy arithmetic families, Step 2 is
often already enough to close the instance exactly.

Typical code path:

- quick pack / `ffd` logic inside `solve_relaxed_dp_with_binpack(...)`

### Step 3 — Profile realization / repair

If Step 2 is not enough, we keep the recovered profile and use a stronger
realization method.

This is not one single subroutine. It is a family of profile-based realization
methods:

- K=2 exact profile realization:
  - `profile_realization_dp_exact`
- K=4 energy-core repair:
  - `generate_energy_core_patterns(...)`
  - `block_repair_energy_core_ub(...)`
- larger-`K` profile-repair beam:
  - `block_repair_profile_repair_beam_ub(...)`

Conceptually, Step 3 tries to keep the good structure from Step 1 while
repairing the mismatch between the relaxed profile and the true finite job set.

### Step 4 — Global exact fallback

If the profile-based realization stages still do not close the instance, the
solver can invoke a more global exact fallback.

In the current large-`K` story, Step 4 is mainly a certification/backstop
stage. On the hardest irregular rows, the quality of the incumbent still mostly
comes from Step 3.

## 2. Original Benchmark: What We Solved First

Before the large scaling work, the corrected benchmark evidence showed that the
stateful DP pipeline solves the benchmark instances to proven optimality.

Current benchmark-facing evidence in the repository includes:

- `hpc/results_studies/component_ablation_ortools20/report.md`
- `hpc/Benchmark Extension Studies/results_studies/study4_spaces_ablation/report.md`
- `docs/journal_synthesis_202604/unified_findings_and_theory.md`

The paper-facing caution is:

- the corrected benchmark result is strong and already documented;
- final benchmark timings and final paper tables should still be regenerated on
  HPC from the mapped scripts in
  [PAPER_HPC_REPRODUCIBILITY_MAP.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/PAPER_HPC_REPRODUCIBILITY_MAP.md).

The key methodological point from the original benchmark is that the solver is
not relying on one monolithic exact method. It works because the relaxed profile
is often already very informative, and later steps only need to realize or
repair that structure.

## 3. First Scaling Axis: Scale `n` on the Benchmark Job Groups

After the original benchmark, the next study kept the paper job groups and
increased the number of jobs `n`.

Main source artifacts:

- `csv/plan05/PAPER_GROUPS_PLAN05_n_extension.csv`
- `csv/plan11/PAPER_GROUPS_PLAN11_n_extension.csv`
- [PAPER_GROUPS_EXTENSION_SUMMARY.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/PAPER_GROUPS_EXTENSION_SUMMARY.md)

### Main accepted large-`n` frontiers

| Group | Current accepted frontier | Main solving method |
|---|---:|---|
| `g24 = {2,4}` | exact through `n=10000` | Step 2 `ffd` |
| `g12357 = {1,2,3,5,7}` | exact through `n=8000` | Step 2 `ffd` |
| `g3567 = {3,5,6,7}` | exact through `n=6000` | Step 3 `block_repair_energy_core` |
| `g246810 = {2,4,6,8,10}` | exact through `n=6000` | Step 2 `ffd` |
| `g810 = {8,10}` | exact through `n=5000` | Step 3 `profile_realization_dp_exact` |
| `g37 = {3,7}` | exact through tested rows up to `n=5000` | Step 3 `profile_realization_dp_exact` |
| `{1,...,10}` | baseline exact through `n=3500`, recovered to `n=5000` | Step 2 dense-unit fastpath |

### What we learned from `n`-scaling

1. Some groups remain easy because the relaxed profile is already directly
   realizable by the real jobs.
2. K=2 groups such as `g37` and `g810` were not truly hard in the old ledger;
   they were partly a routing issue and close well when sent through the
   intended Step-3 exact profile realization path.
3. K=4 hard rows such as `g3567` need the Step-3 energy-core repair path.
4. Dense contiguous unit-containing groups such as `{1,...,10}` can look hard
   under a generic pipeline, but become easy again when we let Step 2 terminate
   earlier through the dense-unit fastpath.

## 4. Second Scaling Axis: Fix `n=1000` and Scale `K`

After the `n`-scaling study, the focus shifted. Instead of making `n` larger
and larger, we fixed:

- `n = 1000`

and then increased:

- the number of distinct job sizes `K`.

Main source artifacts:

- `csv/plan16/PLAN16_k_scaling_n1000.csv`
- `csv/plan18/PLAN18_k_boundary_refine_summary_by_k.csv`
- `csv/plan30/PLAN30_easy_vs_hard_k_boundary.csv`
- `csv/plan33/PLAN33_cert_anytime_summary.csv`
- [PRESENTATION_K_N_SCALING_COMPREHENSIVE.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/PRESENTATION_K_N_SCALING_COMPREHENSIVE.md)

### Why we changed the axis

Scaling only `n` was no longer enough to explain the method boundary. The
stronger story is that difficulty depends on both:

- the number of job sizes `K`;
- the arithmetic structure of those sizes.

So the fixed-`n` study was designed to isolate the effect of `K`.

### Easy-vs-hard result at fixed `n=1000`

The most important conclusion is:

- `K` alone is not the hardness driver.

Easy contiguous unit families:

- `{1,...,10}` exact
- `{1,...,20}` exact
- `{1,...,24}` exact
- `{1,...,30}` exact
- `{1,...,40}` exact

These close through Step 2 because the relaxed profile is easy to realize.

Hard irregular families behave very differently:

- exact through about `K=6`;
- mixed exact / finite-gap around `K=8`;
- no exact closure at `K=10`;
- at `K=12`, exact proof is still difficult, but valid high-quality incumbents
  and certified small gaps are now recovered.

### Current accepted hard-`K` story

For the tested hard irregular `K=10/K=12` rows:

- the important practical improvement is PLAN33;
- PLAN33 runs a serial anytime feasible upper bound, polishes it, computes the
  semigroup lower bound, and early-stops once the certified gap is small enough.

Current accepted PLAN33 result:

- all tested hard `K=10/K=12` rows have valid certified gaps;
- all certified gaps are at most `0.0593%`.

So the current hard-`K` story is:

- exact proof becomes difficult around `K=8-10`;
- but the method still remains strong at `K=10/12` because it can return
  certified very small gaps.

## 5. How the Price Profile Was Handled During Scaling

This part is important because both the original benchmark and the scaling
studies use real electricity prices.

Main source files:

- `data/green-scheduling-bab/Iirc.EnergyStatesAndCostsScheduling/data/dataset-generators-prescriptions/energy-costs/ote2019.json`
- `data/green-scheduling-bab/Iirc.EnergyStatesAndCostsScheduling/data/dataset-generators-prescriptions/benedikt2025b_groups.json`
- `hpc/benchmark_extensions/build_extension_suites.py`
- [figures/price_profiles/PRICE_PROFILE_VISUALIZATION_NOTE.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/figures/price_profiles/PRICE_PROFILE_VISUALIZATION_NOTE.md)

### Base price source

The prices come from a real hourly electricity-price series:

- OTE 2019 hourly prices.

### The two price scenarios

We use two starting points from that real series:

- `2019-01-21`
- `2019-04-08`

These give two different price horizons. In practice, they are two different
real price scenarios taken from the same real dataset.

### What happens when the instance becomes large

When we scale the instance, the scheduling horizon can become longer than the
available price segment after the chosen start date.

In that case, we do not invent synthetic prices. Instead, the generator wraps
back to the chosen start point and continues again from there. In other words:

- the same real hourly profile is reused cyclically;
- the large instance sees a longer horizon built by appending the same real
  price pattern again from the selected start date.

This is how the large-`n` extension was kept tied to the same real price source.

### What this means experimentally

So the large-scale extension preserves:

- the same real price source;
- the same two start-date scenarios;
- the same overall temporal shape of low-price and high-price periods.

What changes is only:

- the total horizon length needed by the larger instance.

## 6. Which Method Solved Which Part of the Story

This is the shortest useful mapping.

| Result family | Main responsible method |
|---|---|
| Original corrected benchmark | same four-step stateful DP pipeline; final paper numbers should be rerun on HPC |
| Easy benchmark groups (`g24`, `g12357`, `g246810`) | Step 2 direct realization (`ffd`) |
| Hard K=4 benchmark group (`g3567`) | Step 3 energy-core repair |
| K=2 groups (`g37`, `g810`) | Step 3 exact profile realization |
| Dense contiguous unit families `{1,...,K}` | Step 2 dense-unit fastpath |
| Hard irregular `K=10/K=12` rows | Step 3 incumbent generation plus PLAN33 certified anytime prepass |

For exact code anchors, runners, and environment toggles, use:

- [PAPER_HPC_REPRODUCIBILITY_MAP.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/PAPER_HPC_REPRODUCIBILITY_MAP.md)

## 7. Current One-Paragraph Summary

The solver starts from a semigroup lower-bound/profile recovery, tries to
realize that profile quickly, then uses stronger profile-based repair when
needed, and only falls back to global exact search afterward. On the benchmark
job groups, this pipeline scales several families far beyond the original
benchmark sizes, with different Step-2 or Step-3 components responsible for
different arithmetic families. After that, the study fixed `n=1000` and scaled
`K`, which revealed that arithmetic structure matters more than `K` alone:
easy contiguous unit families remain exact through `K=40`, whereas hard
irregular families become non-exact around `K=8-10`, but still admit certified
very small gaps at `K=10/12` through the current hard-`K` package. All scaling
uses the same real OTE 2019 hourly price source; when the horizon becomes too
long, the generator extends it by cyclically reusing the same real profile from
the selected start date.

# Comprehensive Method and Experiment Summary

Status date: 2026-05-17

This is the detailed canonical summary for the stateful DP approach for
single-machine energy-aware scheduling. It collects the method, the relaxation
logic, the heuristic realization layer, the exact certification layer, the
benchmark evidence, the benchmark extensions, and the price-profile handling in
one place.

Shorter entrypoints still exist:

- [END_TO_END_PIPELINE_AND_SCALING_SUMMARY.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/END_TO_END_PIPELINE_AND_SCALING_SUMMARY.md)
- [PAPER_HPC_REPRODUCIBILITY_MAP.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/PAPER_HPC_REPRODUCIBILITY_MAP.md)
- [CURRENT_RESULTS_INDEX.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/CURRENT_RESULTS_INDEX.md)

## 0. Paper-Facing Naming Translation

The code and internal notes use several historical names. For the final paper,
we should use one coherent terminology and mention the code names only in a
reproducibility appendix.

### 0.1 Main pipeline names

| Internal / code name | Paper-facing name | Use in final paper |
|---|---|---|
| `solve_relaxed_dp_with_binpack`, `compute_relaxed_dp_table`, `fwd_relax` | Relaxed block-timing DP | Main Step 1 name |
| `R_semi`, semigroup relaxation | Semigroup block-timing relaxation | Formal lower-bound model |
| `R_feas`, feasible relaxation | Count-aware feasible relaxation | Backup/tighter lower-bound model |
| `ffd`, `bfd`, direct packer | Greedy profile realization | Main Step 2 name |
| `PAST_DENSE_UNIT_STEP2_FASTPATH` | Dense-unit early realization | Scalability enhancement for contiguous unit families |
| `profile_realization_dp_exact`, fixed-block DP | Exact profile realization | Exact Step 3 certifier |
| `block_repair_energy_core_ub`, `energy_core` | Core-restricted profile realization | Focused Step 3 realization for hard small-`K` profiles |
| `block_repair_profile_repair_beam_ub`, profile-repair beam | Beam-limited profile realization | Truncated Step 3 realization for hard large-`K` profiles |
| Step 4 exact DP, sparse/dense exact fallback | Global certification fallback | Final exact/certification layer |
| PLAN33, `PAST_CERT_ANYTIME_PREPASS` | Certified anytime prepass | Hard-`K` certified-gap mechanism |

### 0.2 Step 3 unification

The final paper should not present Step 3 as several unrelated methods. It
should present Step 3 as one **profile realization layer**:

> Given a relaxed block profile from Step 1, assign the finite job multiset to
> the recovered blocks.

The variants differ only in computational fidelity:

| Paper-facing Step 3 variant | Meaning | Exactness status |
|---|---|---|
| Exact profile realization | keeps the full relevant count-vector state for the recovered profile | exact for the recovered profile |
| Core-restricted profile realization | restricts pattern generation around the most relevant profile-filling patterns | exactness depends on the retained pattern set; report as restricted realization unless fully certified |
| Beam-limited profile realization | keeps only a bounded frontier of promising partial fillings | heuristic incumbent generator |

Recommended wording:

> The profile-realization layer admits exact, restricted, and beam-limited
> implementations. These are not separate algorithms; they solve the same
> recovered block-filling problem under different computational budgets.

### 0.3 Names to avoid in the main paper

Avoid these internal names in the main text unless they appear in a
reproducibility table:

- `PLAN10`, `PLAN13`, `PLAN14`, `PLAN30`, `PLAN33`;
- `energy_core`;
- `block repair`;
- `profile_repair_beam`;
- `uniform_mult2`, `ambig_scoreband`, and other beam-policy labels;
- environment variable names such as `PAST_BLOCK_REPAIR_PATTERN_DP_K`.

Use internal names only when mapping paper results back to code in the
reproducibility appendix.

## 1. Problem Setting

We study a single-machine scheduling problem with time-varying electricity
prices. Jobs have processing lengths drawn from a small set of distinct sizes:

- distinct job sizes: `p_1, ..., p_K`;
- finite multiplicities: `n_1, ..., n_K`;
- total work: `W = sum_j n_j p_j`;
- price horizon: an explicit vector of electricity prices over time.

The objective is total energy cost. The machine may process jobs in selected
time intervals and may be idle outside those intervals. For a fixed set of
processing intervals, the identities of jobs inside an interval usually matter
only through whether the interval length can be filled exactly by the available
job multiset.

This observation is the basis of the method:

- first solve a block-timing problem;
- then solve a block-filling problem.

## 2. Core Decomposition

The method separates two questions.

### 2.1 Block timing

The block-timing side asks:

- where should the machine process?
- what are the lengths of the processing blocks?
- what lower bound does this imply?

This is handled by the relaxed dynamic program.

### 2.2 Block filling

The block-filling side asks:

- can the real finite job multiset fill the recovered blocks?
- if yes, can we produce a schedule with cost equal to the lower bound?
- if not, can we repair the profile or at least produce a good incumbent?

This is handled by quick realization, profile realization, beam repair, and
exact fallback.

## 3. The Four-Step Pipeline

The current solver is best understood as a four-step pipeline.

### Step 1 - Relaxed DP and profile recovery

Step 1 computes a lower bound and a recovered block profile.

Main role:

- compute the semigroup lower bound;
- recover a sequence of processing blocks;
- provide the structural backbone for later realization stages.

Main code anchors:

- `compute_relaxed_dp_table(...)`
- `solve_relaxed_dp_with_binpack(...)`

Main files:

- `solvers/cpp/stateful_dp_solver.cpp`
- `solvers/cpp/stateful_compare.cpp`

### Step 2 - Quick realization

Step 2 tries to turn the recovered relaxed profile into a feasible schedule
cheaply.

It uses greedy profile realization methods, mainly:

- `ffd` - first-fit decreasing style packing;
- `bfd` - best-fit decreasing style packing;
- related quick packing attempts and additive dense-unit fast paths.

This stage is the main reason easy arithmetic families close very quickly.

### Step 3 - Profile realization and repair

Step 3 keeps the recovered profile and tries stronger realization methods.

It has several modes:

| Regime | Method | Code anchor |
|---|---|---|
| small `K=2` | exact fixed-block/profile realization | `profile_realization_dp_exact` |
| hard `K=4` | energy-core profile repair | `block_repair_energy_core_ub(...)` |
| larger hard `K` | profile-repair beam | `block_repair_profile_repair_beam_ub(...)` |

The conceptual role is the same in all cases:

- use the Step-1 profile as guidance;
- decide which job counts should fill each recovered block;
- either certify the block profile exactly or generate a strong feasible
  incumbent.

### Step 4 - Global exact fallback

Step 4 is the global exact fallback for the original problem.

It uses the best incumbent from Steps 2 and 3 as an upper bound. It may:

- prove optimality;
- or return a certified finite gap under the current time and memory budget.

For hard large-`K` rows, Step 4 is not the main incumbent generator. The best
incumbent usually comes from Step 3 or from the PLAN33 certified anytime
prepass.

## 4. Relaxation 1: Semigroup Relaxation

The semigroup relaxation is the default lower-bound engine.

### 4.1 Exact count model

For a work amount `x`, the exact bounded representation asks whether:

```text
x = sum_j a_j p_j
0 <= a_j <= n_j
a_j integer
```

This remembers all per-type multiplicities. That is expensive because the exact
state space can grow like:

```text
O(T * product_j (n_j + 1))
```

where `T` is the explicit time horizon.

### 4.2 Semigroup model

The semigroup relaxation drops the upper bounds `a_j <= n_j` and keeps only:

```text
x = sum_j a_j p_j
a_j >= 0
a_j integer
```

Equivalently, it replaces the bounded set of feasible work amounts by the
numerical semigroup generated by the job sizes.

The relaxed DP state is compressed to:

```text
(t, rw)
```

where:

- `t` is the current block end time;
- `rw` is the remaining total work.

The DP forgets which job types remain and remembers only how much work remains.

### 4.3 Why this is tractable

The price horizon is explicit in the input. The DP has at most:

```text
O(T * W)
```

relaxed states, and for feasible instances `W` is bounded by the available
horizon. The relaxation can also be viewed as a shortest-path problem on a
directed acyclic graph over states `(t, rw)`.

So the semigroup relaxation is the fast, scalable lower-bound stage.

### 4.4 What it gives

It gives:

- a valid lower bound;
- a candidate block profile;
- a strong structural guide for realization.

It may fail to respect finite job multiplicities, especially when scarce job
types are overused by the relaxed path.

## 5. Relaxation 2: Feasible Relaxation `R_feas`

`R_feas` is the count-aware backup relaxation.

It keeps the same compressed DP state `(t, rw)` as the semigroup relaxation but
filters transitions that would necessarily overuse a scarce job type.

### 5.1 Transition idea

Suppose the DP wants to place a job of type `j` after already placing total work
`placed`.

`R_feas` allows that transition only if `placed` can be represented while still
leaving at least one job of type `j` unused.

In words:

- "Can the work already done be explained without consuming all copies of type
  `j`?"
- If yes, placing one more `j` remains possible.
- If no, the semigroup transition is blocked.

### 5.2 Inclusion relationship

`R_feas` only deletes semigroup transitions. It never adds new transitions.

The safe relationship is:

```text
LB_semi <= LB_feas <= OPT
```

So `R_feas` is tighter than the semigroup relaxation and remains admissible.

### 5.3 When it matters

The verified evidence is mixed, and this is important for honest reporting.

On the main `benedikt2025b_groups` hard benchmark rows:

- `R_semi` was already exact as a lower bound;
- `R_feas` gave no lower-bound gain;
- enabling it by default was not useful.

On adversarial or count-scarcity tests:

- `R_feas` can close large semigroup gaps;
- it helps when the semigroup relaxation unrealistically reuses scarce job
  types.

Current practical interpretation:

- `R_semi` is the default production lower bound;
- `R_feas` is the principled backup for count-scarcity regimes;
- `R_feas` should not be presented as necessary for the main benchmark.

## 6. Heuristic Realization Layer

The heuristic layer tries to convert the relaxed profile into a valid schedule
before invoking heavier exact methods.

### 6.1 FFD / BFD style realization

The main simple realizers are:

- First Fit Decreasing (`ffd`);
- Best Fit Decreasing (`bfd`);
- small variants used inside the packing attempts.

They attempt to assign the finite jobs to recovered processing blocks.

When they succeed with `UB = LB`, the instance is solved exactly without the
global exact fallback.

### 6.2 Dense-unit fastpath

For contiguous unit-containing families such as `{1, ..., K}`, the relaxed
profile is often directly realizable. The generic pipeline was sometimes
spending time preparing later stages even though Step 2 could close.

The dense-unit fastpath makes this explicit:

- detect dense contiguous unit-containing families;
- run Step 2 early;
- stop when `UB = LB`.

This recovered `{1,...,10}` at `n=5000` and supports the fixed-`n` easy-family
scaling story through `K=40`.

Main toggle:

- `PAST_DENSE_UNIT_STEP2_FASTPATH=1`

## 7. Exact Certification in Step 3

Step 3 is not just a heuristic. In the tractable regimes, it performs exact
profile realization on the recovered blocks.

### 7.1 Fixed-block/profile realization DP

Given recovered blocks, the exact profile-realization problem asks whether the
finite job multiset can fill those blocks exactly:

```text
for each block b:
    sum_j x_bj p_j = block_length_b

for each type j:
    sum_b x_bj = n_j
```

The fixed-block DP keeps the remaining count vector across blocks. This is exact
for the recovered profile, but it can grow quickly with `K`.

This is why it is excellent for:

- `K=2` rows such as `g37={3,7}` and `g810={8,10}`;
- selected tractable fixed-block profiles.

It is not scalable enough to force on hard irregular `K=10/12` rows.

### 7.2 K=4 energy-core repair

For hard `K=4` rows, especially `g3567={3,5,6,7}`, the successful Step-3 path is
the energy-core repair.

The idea is to focus pattern generation around the parts of the recovered
profile that matter most for cost and feasibility, rather than enumerating all
possible block fillings naively.

The accepted implementation improvement was:

- use a DP-style pattern generator for `K=4`;
- disable low-value signature deduplication for `K=4`;
- preserve exactness while materially reducing pattern-generation time.

Main toggles:

- `PAST_BLOCK_REPAIR_PATTERN_DP_K=4`
- `PAST_BLOCK_REPAIR_EC_SIGNATURE_DEDUP=0`

### 7.3 Larger-K profile-repair beam

For harder larger-`K` rows, exact fixed-block DP becomes too large. The beam
mode keeps only a limited set of promising partial realizations.

It is a truncated profile-realization DP:

- same recovered profile;
- same block-filling interpretation;
- but only the best frontier states survive.

This is useful for generating strong incumbents, especially for hard irregular
`K=10/12` rows.

## 8. Certified Anytime Prepass for Hard K

PLAN33 added the current recommended hard-`K` default for tested hard irregular
`K=10/K=12` rows.

It performs:

1. serial feasible upper-bound generation;
2. local polish of the best sequence;
3. semigroup lower-bound computation;
4. certified early stop if the gap is below the threshold.

Important validity correction:

- a previous parallel-machine upper-bound attempt was invalid because the
  benchmark is single-machine;
- PLAN33 uses valid single-machine schedules only.

Current accepted PLAN33 evidence:

- all tested hard `K=10/K=12` rows have valid certified gaps;
- maximum certified gap is `0.0593%`;
- average runtime improves over the previous PLAN32C baseline while adding
  semigroup lower-bound certification.

Main toggle:

- `PAST_CERT_ANYTIME_PREPASS=1`

## 9. How the Original Benchmark Was Solved

The corrected benchmark evidence shows that the stateful DP pipeline solves the
main benchmark instances to proven optimality.

Important source files:

- `hpc/results_studies/component_ablation_ortools20/report.md`
- `hpc/results_studies/study4_spaces_ablation/report.md`
- `docs/journal_synthesis_202604/unified_findings_and_theory.md`
- `docs/archive_20260415/INSTANCE_GENERATION_BUG_REPORT.md`

The paper-facing rule is:

- the local evidence is strong;
- final paper timings should be regenerated on HPC;
- use the corrected benchmark generation path;
- do not use old `hopsCount`-polluted datasets.

### 9.1 Main benchmark interpretation

The key result is not merely that the exact fallback can eventually solve the
benchmark.

The deeper result is:

- the semigroup stage often finds the right block timings;
- exact fixed-block certification can certify those timings;
- therefore many benchmark rows close without needing the global exact fallback.

In the exact-pack ablation, the benchmark closes at the forward/profile stage
once recovered semigroup-optimal block profiles are checked by the exact
fixed-block certifier.

So the benchmark story is:

- Step 1 gives a very strong profile;
- Step 2 or Step-3 certification realizes it;
- global exact DP is not the main explanation for the benchmark success.

## 10. Large-`n` Benchmark Extension

After solving the original benchmark, we extended the paper job groups by
increasing the number of jobs `n`.

Main artifacts:

- `csv/plan05/PAPER_GROUPS_PLAN05_n_extension.csv`
- `csv/plan11/PAPER_GROUPS_PLAN11_n_extension.csv`
- [PAPER_GROUPS_EXTENSION_SUMMARY.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/PAPER_GROUPS_EXTENSION_SUMMARY.md)

Main accepted frontiers:

| Group | Frontier | Main method |
|---|---:|---|
| `g24={2,4}` | exact through `n=10000` | Step 2 `ffd` |
| `g12357={1,2,3,5,7}` | exact through `n=8000` | Step 2 `ffd` |
| `g3567={3,5,6,7}` | exact through `n=6000` | Step 3 energy-core repair |
| `g246810={2,4,6,8,10}` | exact through `n=6000` | Step 2 `ffd` |
| `g810={8,10}` | exact through `n=5000` | Step 3 exact profile realization |
| `g37={3,7}` | exact through tested rows up to `n=5000` | Step 3 exact profile realization |
| `{1,...,10}` | recovered to `n=5000` | Step 2 dense-unit fastpath |

The large-`n` study showed that different arithmetic groups are solved by
different parts of the same pipeline.

## 11. Fixed-`n`, Variable-`K` Study

The next scaling direction fixed:

- `n = 1000`

and increased:

- `K`, the number of distinct job sizes.

Main artifacts:

- `csv/plan16/PLAN16_k_scaling_n1000.csv`
- `csv/plan18/PLAN18_k_boundary_refine_summary_by_k.csv`
- `csv/plan30/PLAN30_easy_vs_hard_k_boundary.csv`
- `csv/plan33/PLAN33_cert_anytime_summary.csv`

Main conclusion:

- `K` alone is not the hardness driver;
- arithmetic structure matters.

### 11.1 Easy contiguous unit families

Families of the form `{1, ..., K}` stay easy because the relaxed profile is
directly realizable.

Accepted evidence:

- exact through `K=40` at fixed `n=1000`.

Main method:

- Step 2 `ffd` / dense-unit fastpath.

### 11.2 Hard irregular families

Hard irregular families degrade much earlier:

- exact through about `K=6`;
- mixed exact / finite-gap around `K=8`;
- finite-gap but no exact closure at `K=10`;
- at `K=12`, exact proof is difficult, but PLAN33 recovers certified small
  gaps.

Main interpretation:

- easy arithmetic lets Step 2 realize the relaxation directly;
- hard arithmetic makes block filling difficult;
- the semigroup lower bound can remain strong even when exact realization is
  hard.

## 12. Price Profile and Scaling Methodology

All benchmark and extension studies use real electricity prices from the OTE
2019 hourly dataset.

Main source files:

- `data/green-scheduling-bab/Iirc.EnergyStatesAndCostsScheduling/data/dataset-generators-prescriptions/energy-costs/ote2019.json`
- `data/green-scheduling-bab/Iirc.EnergyStatesAndCostsScheduling/data/dataset-generators-prescriptions/benedikt2025b_groups.json`
- `hpc/benchmark_extensions/build_extension_suites.py`
- [figures/price_profiles/PRICE_PROFILE_VISUALIZATION_NOTE.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/figures/price_profiles/PRICE_PROFILE_VISUALIZATION_NOTE.md)

### 12.1 Original benchmark price scenarios

The original benchmark prescription uses real OTE 2019 prices and two start
dates:

- `2019-01-21T00:00:00`
- `2019-04-08T00:00:00`

The original benchmark also includes repeated-grid variants, but the large
extension story should be explained separately.

### 12.2 Large extension price handling

When scaling `n` or `K`, the scheduling horizon can exceed the remaining OTE
price segment after the selected start date.

For the extension studies, the generator does not synthesize new prices.
Instead:

- it starts from the selected real OTE start date;
- it reads hourly prices forward;
- when the end of the available OTE segment is reached, it wraps back to the
  same selected start date;
- it continues by cyclically reusing the same real hourly price sequence.

So the large instances are built from repeated real hourly price profiles, not
from artificial flat or random tariffs.

### 12.3 Interpretation and limitation

This preserves:

- the real OTE price source;
- the two real start-date scenarios;
- realistic low-price and high-price temporal structure.

It also means:

- very large horizons reuse a historical profile cyclically;
- this is acceptable as a scaling experiment, but final paper wording should
  state the cyclic reuse clearly.

## 13. Responsible Code Paths

Main binary:

- `solvers/cpp/build/stateful_compare`

Main workflow:

```bash
stateful_compare ablation-stdin step1_exact_guided <time_limit_sec>
```

Core solver files:

- `solvers/cpp/stateful_compare.cpp`
- `solvers/cpp/stateful_dp_solver.cpp`
- `solvers/cpp/stateful_dp_solver.hpp`

Important runners:

- `research/k_vs_arithmetic_axes_20260412/run_plan05_paper_groups_extension.py`
- `research/k_vs_arithmetic_axes_20260412/run_plan10_k4_generator_compare.py`
- `research/k_vs_arithmetic_axes_20260412/run_plan13_two_track_recovery.py`
- `research/k_vs_arithmetic_axes_20260412/run_plan16_k_scaling_n1000.py`
- `research/k_vs_arithmetic_axes_20260412/run_plan17_k_axis_n1000.py`
- `research/k_vs_arithmetic_axes_20260412/run_plan18_k_boundary_refine_n1000.py`
- `research/k_vs_arithmetic_axes_20260412/run_plan28_easy_k_scaling.py`
- `research/k_vs_arithmetic_axes_20260412/run_plan33_cert_anytime.py`

For exact code-to-result mapping, use:

- [PAPER_HPC_REPRODUCIBILITY_MAP.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/PAPER_HPC_REPRODUCIBILITY_MAP.md)

## 14. What Should Be Claimed in the Paper

Safe claims:

- The semigroup relaxation is a valid and tractable lower-bound method under
  the explicit-horizon input model.
- `R_feas` is a tighter admissible backup relaxation, useful when scarce job
  types make semigroup overuse unrealistic.
- On the corrected benchmark, the semigroup-centered profile approach is very
  strong and should be rerun on HPC for final paper timings.
- Many benchmark and extension rows close without global exact fallback because
  the recovered profile can be realized or certified directly.
- Scaling `K` shows that arithmetic structure matters more than `K` alone:
  easy contiguous unit families scale to `K=40`, while hard irregular families
  degrade around `K=8-10`.
- For tested hard `K=10/K=12` rows, PLAN33 gives certified very small finite
  gaps, with maximum gap `0.0593%`.
- Large-horizon experiments reuse real OTE 2019 price profiles cyclically, and
  this should be stated explicitly.

Claims to avoid:

- Do not say the global exact DP is what solved the benchmark generally.
- Do not say `R_feas` is necessary on the main benchmark.
- Do not cite PLAN32B parallel upper bounds as valid; that branch changed the
  model and is invalid for the single-machine benchmark.
- Do not describe `K=12` hard rows as no-incumbent; PLAN33 supersedes that.
- Do not present local laptop runtimes as final paper numbers before HPC rerun.

## 15. Current Bottom Line

The approach is coherent:

1. Use a semigroup dynamic program to solve the block-timing relaxation.
2. Use quick greedy packing to realize easy profiles.
3. Use exact fixed-block/profile realization when the recovered profile is
   tractable.
4. Use energy-core repair or profile-repair beam when the profile is hard but
   still useful.
5. Use global exact fallback or PLAN33 certification when exact closure is too
   expensive.

The experiments support the same story:

- the original benchmark is solved by the semigroup-centered profile approach;
- large-`n` extensions show which paper groups scale and which method handles
  each group;
- fixed-`n`, variable-`K` experiments show that arithmetic structure is the real
  boundary;
- hard `K=10/K=12` rows are not fully exact, but they now have certified small
  gaps under the original single-machine model.

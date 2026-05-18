# Paper/HPC Reproducibility Map

Status date: 2026-05-03

This file maps the result claims we intend to keep for the paper to the code
that produced them. Local laptop timings are useful for method selection, but
paper numbers should be regenerated on HPC from the code paths below.

## Build And Common Entry Point

Build target:

```bash
cmake --build solvers/cpp/build --target stateful_compare -j1
```

Main binary:

```bash
solvers/cpp/build/stateful_compare
```

Main workflow used by the current experiments:

```bash
stateful_compare ablation-stdin step1_exact_guided <time_limit_sec>
```

Instance construction for extension studies:

- `hpc/benchmark_extensions/build_extension_suites.py`
- `research/k_vs_arithmetic_axes_20260412/run_plan05_paper_groups_extension.py`
- `scripts/regenerate_instances.py` for corrected `benedikt2025b_groups` regeneration.

Shared solver code:

- `solvers/cpp/stateful_compare.cpp`
- `solvers/cpp/stateful_dp_solver.cpp`
- `solvers/cpp/stateful_dp_solver.hpp`

## Method Components And Code Anchors

| Method component | What it means | Responsible code |
|---|---|---|
| Semigroup lower bound / profile recovery | Computes relaxed DP and recovered block/profile structure | `compute_relaxed_dp_table(...)`, `solve_relaxed_dp_with_binpack(...)` in `solvers/cpp/stateful_dp_solver.cpp`; called from `stateful_compare.cpp` |
| Step 2 quick realization | Fast greedy realization of the relaxed profile, mainly `ffd`/related packers | `solve_relaxed_dp_with_binpack(...)` and pack-candidate attribution in `solvers/cpp/stateful_dp_solver.cpp` |
| K=2 exact profile realization | Exact Step-3 profile realization for small `K`, e.g. `g37`, `g810` | `profile_realization_dp_exact` candidate path in `solvers/cpp/stateful_dp_solver.cpp`; selected through `PAST_PROFILE_REALIZATION_SELECTOR_POLICY=auto_v1` |
| K=4 energy-core repair | Step-3 pattern/core repair used for hard K=4 rows | `generate_energy_core_patterns(...)`, `block_repair_energy_core_ub(...)` in `solvers/cpp/stateful_dp_solver.cpp` |
| K=4 DP-style pattern generator | PLAN10 speedup for K=4 pattern generation | `PAST_BLOCK_REPAIR_PATTERN_DP_K=4`, `PAST_BLOCK_REPAIR_EC_SIGNATURE_DEDUP=0`; implemented around pattern generation in `solvers/cpp/stateful_dp_solver.cpp` |
| Dense-unit Step-2 fast path | Early Step-2 closure for contiguous unit families `{1,...,K}` | `PAST_DENSE_UNIT_STEP2_FASTPATH=1`; dense-unit logic in `solve_relaxed_dp_with_binpack(...)` / pack path in `solvers/cpp/stateful_dp_solver.cpp` |
| Profile-repair beam | Scalable Step-3 incumbent path for hard irregular larger `K` | `block_repair_profile_repair_beam_ub(...)` in `solvers/cpp/stateful_dp_solver.cpp`; enabled with `PAST_RELAXED_BINPACK_SOLVER=profile_repair_beam` |
| Step-3 survivor multiplicity | Beam survivor-policy variants for K=10 diagnostics | `PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_POLICY`; survivor logic in `block_repair_feasible_beam_ub(...)` / profile beam code |
| Certified anytime hard-K prepass | PLAN33: serial feasible UB + polish + semigroup LB + certified early stop | `compute_initial_ub(...)`, `polish_best_sequence_ub(...)`, `PAST_CERT_ANYTIME_PREPASS=1` block in `stateful_compare.cpp` |
| Global exact fallback | Full exact DP proof attempt after profile methods | exact fallback logic and CSV attribution in `stateful_compare.cpp` / exact DP routines in `stateful_dp_solver.cpp` |

## Paper Result Families To Regenerate On HPC

### 1. Corrected Original Benchmark / 560-Instance Claims

Use for the paper baseline comparison only after rerunning on HPC.

Current evidence files:

- `hpc/results_studies/component_ablation_ortools20/full_exact_pack.csv`
- `hpc/results_studies/component_ablation_ortools20/report.md`
- `hpc/results_studies/study4_spaces_ablation/report.md`
- `docs/journal_synthesis_202604/unified_findings_and_theory.md`
- `docs/archive_20260415/INSTANCE_GENERATION_BUG_REPORT.md`

Responsible code:

- `scripts/regenerate_instances.py`
- `hpc/setup_benchmark_data.sh`
- `hpc/run_revised_ablation_studies.sh`
- `hpc/studies/component_ablation.py`
- `hpc/studies/spaces_ablation.py`
- `solvers/cpp/build/stateful_compare`

Paper-use rule:

- Keep the `560/560` optimality claim only after HPC rerun confirms it.
- Use the corrected benchmark generation path; do not use old `hopsCount`-polluted datasets.

### 2. Paper-Group Large-`n` Extension

Current source artifacts:

- `csv/plan05/PAPER_GROUPS_PLAN05_n_extension.csv`
- `csv/plan11/PAPER_GROUPS_PLAN11_n_extension.csv`
- `PAPER_GROUPS_EXTENSION_SUMMARY.md`

Responsible code:

- `research/k_vs_arithmetic_axes_20260412/run_plan05_paper_groups_extension.py`
- `hpc/benchmark_extensions/build_extension_suites.py`
- `solvers/cpp/build/stateful_compare`

Methods to report:

- `g24`, `g12357`, `g246810`: mostly Step 2 `ffd`
- `g3567`: Step 3 `block_repair_energy_core`
- `g37`, `g810`: Step 3 `profile_realization_dp_exact` when routed correctly
- `{1,...,10}`: Step 2 `ffd`; at `n=5000`, cite dense-unit fastpath as additive recovery, not baseline

### 3. K=4 Generator / Energy-Core Speedup

Current source artifacts:

- `csv/plan10/PLAN10_k4_generator_compare.csv`
- `csv/plan10/PLAN10_k4_generator_dp4.csv`
- `csv/plan10/PLAN10_k4_speedup_baseline.csv`

Responsible code:

- `research/k_vs_arithmetic_axes_20260412/run_plan10_k4_generator_compare.py`
- `generate_energy_core_patterns(...)`
- `block_repair_energy_core_ub(...)`

Required environment:

```bash
PAST_RELAXED_BINPACK_SOLVER=energy_core
PAST_BLOCK_REPAIR_COMPLETION_MODE=direct
PAST_BLOCK_REPAIR_COMPLETION_DIRECT_MAX_CELLS=500000000
PAST_BLOCK_REPAIR_PATTERN_DP_K=4
PAST_BLOCK_REPAIR_EC_SIGNATURE_DEDUP=0
```

Paper-use rule:

- This is a validated K=4 implementation improvement, not a separate algorithmic
  claim.

### 4. Corrected K=2 Routing (`g37`, `g810`)

Current source artifacts:

- `csv/plan13/PLAN13_g37_k2_reroute.csv`
- `csv/plan13/PLAN13_g37_k2_variant_compare.csv`
- `csv/plan16/PLAN16_k_scaling_n1000.csv`

Responsible code:

- `research/k_vs_arithmetic_axes_20260412/run_plan13_two_track_recovery.py`
- `research/k_vs_arithmetic_axes_20260412/run_plan16_k_scaling_n1000.py`
- `profile_realization_dp_exact` path in `solvers/cpp/stateful_dp_solver.cpp`

Required routing:

```bash
PAST_RELAXED_BINPACK_SOLVER=profile_repair_beam
PAST_PROFILE_REALIZATION_SELECTOR_POLICY=auto_v1
```

Paper-use rule:

- Do not cite old `g37` non-mainline rows as a method failure. They were
  misrouted. Current corrected evidence closes tested `g37` rows through
  `n=5000` using Step-3 exact profile realization.

### 5. Dense Unit Families `{1,...,K}`

Current source artifacts:

- `csv/plan14/PLAN14_g12345678910_fastpath_compare.csv`
- `csv/plan16/PLAN16_k_scaling_n1000.csv`
- `csv/plan30/PLAN30_easy_k_scaling_raw.csv`
- `csv/plan30/PLAN30_easy_k_scaling_summary.csv`
- `csv/plan30/PLAN30_easy_vs_hard_k_boundary.csv`

Responsible code:

- `research/k_vs_arithmetic_axes_20260412/run_plan13_two_track_recovery.py` for PLAN14/15 dense-unit recovery artifacts
- `research/k_vs_arithmetic_axes_20260412/run_plan16_k_scaling_n1000.py`
- `research/k_vs_arithmetic_axes_20260412/run_plan28_easy_k_scaling.py` for PLAN30 artifacts
- dense-unit logic in `solvers/cpp/stateful_dp_solver.cpp`

Required environment for fastpath variants:

```bash
PAST_DENSE_UNIT_STEP2_FASTPATH=1
PAST_DENSE_UNIT_FASTPATH_K_MIN=8
```

Paper-use rule:

- The paper story is that easy contiguous unit arithmetic remains exact through
  `K=40` at `n=1000`, while hard irregular arithmetic degrades much earlier.

### 6. Hard Irregular K-Axis Boundary

Current source artifacts:

- `csv/plan17/PLAN17_k_axis_n1000_raw.csv`
- `csv/plan18/PLAN18_k_boundary_refine_n1000_raw.csv`
- `csv/plan18/PLAN18_k_boundary_refine_summary_by_k.csv`
- `csv/plan30/PLAN30_easy_vs_hard_k_boundary.csv`

Responsible code:

- `research/k_vs_arithmetic_axes_20260412/run_plan17_k_axis_n1000.py`
- `research/k_vs_arithmetic_axes_20260412/run_plan18_k_boundary_refine_n1000.py`
- `research/k_vs_arithmetic_axes_20260412/run_plan28_easy_k_scaling.py`

Methods to report:

- `K=8`: mixed exact / finite-gap
- `K=10`: finite-gap, no exact closure in PLAN18
- `K=12`: historical PLAN18/19 rows were budget-limited, but PLAN33 now gives
  certified finite incumbents for the hard K12 panel

Paper-use rule:

- Present this as an arithmetic-structure boundary, not as monotonic hardness
  in `K` alone.

### 7. Hard K10/K12 Certified Anytime Default

Current source artifacts:

- `csv/plan33/PLAN33_cert_anytime_raw.csv`
- `csv/plan33/PLAN33_cert_anytime_compare.csv`
- `csv/plan33/PLAN33_cert_anytime_summary.csv`
- `csv/plan33/PLAN33_notes.md`

Responsible code:

- `research/k_vs_arithmetic_axes_20260412/run_plan33_cert_anytime.py`
- `compute_initial_ub(...)`
- `polish_best_sequence_ub(...)`
- `compute_relaxed_dp_table(...)`
- PLAN33 prepass block in `solvers/cpp/stateful_compare.cpp`

Required environment:

```bash
PAST_RELAXED_BINPACK_SOLVER=profile_repair_beam
PAST_PROFILE_REALIZATION_SELECTOR_POLICY=auto_v1
PAST_CERT_ANYTIME_PREPASS=1
PAST_CERT_ANYTIME_K_MIN=10
PAST_CERT_ANYTIME_GAP_STOP_PCT=0.1
PAST_CERT_ANYTIME_TRIALS=5
PAST_CERT_ANYTIME_POLISH=1
```

Paper-use rule:

- This is the current recommended hard-K default for tested K10/K12 rows.
- It gives certified finite gaps, not exact closure.
- PLAN32B parallel UB is invalid for the benchmark and must not be cited as a
  valid result.

### 8. Step-3 Survivor Policy Diagnostics

Current source artifacts:

- `csv/plan27/PLAN27_step3_adaptive_survivor_summary.csv`
- `csv/plan31/PLAN31_family_aware_survivor_summary.csv`
- `csv/plan31/PLAN31_family_aware_survivor_compare.csv`

Responsible code:

- `research/k_vs_arithmetic_axes_20260412/run_plan27_gate_a.py`
- `research/k_vs_arithmetic_axes_20260412/run_plan31_family_aware_survivor.py`
- `PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_POLICY` logic in `solvers/cpp/stateful_dp_solver.cpp`

Paper-use rule:

- Treat this as an ablation/diagnostic section unless HPC reruns confirm enough
  stable benefit to promote a default policy.

## Results Not To Promote As Main Method

Do not promote the following as main paper methods:

- PLAN24/PLAN24B exact corridor: blocked by exact-DP theoretical skip / int64
  encoding overflow.
- PLAN25/PLAN26 local corridor: invalid due to block/path mismatch.
- PLAN28 block-realizability diagnostic: did not separate easy from hard.
- PLAN29 adjacent block coarsening: no single view generalized.
- PLAN32B parallel UB: invalid model, two-machine partition, rejected by `UB < LB`.

These can appear, at most, as short negative ablations or internal diagnostics.

## Minimum HPC Rerun Checklist

Before writing final paper numbers:

1. Rebuild `stateful_compare` on HPC.
2. Regenerate or verify corrected benchmark data.
3. Rerun original benchmark proof/ablation if it will be claimed.
4. Rerun paper-group large-`n` extension rows.
5. Rerun PLAN30 easy-vs-hard `K` scaling.
6. Rerun PLAN33 hard K10/K12 certified panel.
7. Export final CSVs and regenerate paper tables from HPC outputs only.
8. Keep local CSVs as design/provenance evidence, not final performance evidence.

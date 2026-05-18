# Method Provenance

This note describes how to read `research/k_vs_arithmetic_axes_20260412/csv/CURRENT_METHOD_PROVENANCE.csv` and how solver-method labels map to concrete code entrypoints.

For paper/HPC reruns, also use:

- `research/k_vs_arithmetic_axes_20260412/PAPER_HPC_REPRODUCIBILITY_MAP.md`

That file maps each paper-facing result family to the runner script,
responsible solver code, environment toggles, and source CSV artifact.

## Purpose

For each accepted or benchmark-significant row, provenance should answer:

1. which workflow entry was used,
2. which deciding step and pack method produced the row result,
3. which concrete implementation functions are the relevant code path,
4. which artifact row is the auditable source,
5. whether evidence is current accepted, continuity, or archive-only.

## Provenance table location

- `research/k_vs_arithmetic_axes_20260412/csv/CURRENT_METHOD_PROVENANCE.csv`

Core fields include:

- `logical_row_id`, `family_id`, `K`, `n`, `lambda`, `seed`
- `solution_status`, `deciding_step`, `pack_method`
- `selector_policy`, `selector_decision`
- `accepted_solver_package`
- `source_artifact_path`, `source_row_id`
- `workflow_entry`
- `code_file_paths`, `function_names`, `solver_path_description`
- `important_env_toggles`
- `evidence_class` in:
  - `current_accepted_benchmark`
  - `historical_continuity`
  - `archive_only`

### Source row-id convention

`source_row_id` uses an artifact-family prefix that must match
`source_artifact_path`:

- `plan05/...` for rows sourced from
  `csv/plan05/PAPER_GROUPS_PLAN05_n_extension.csv`
- `plan10/...` for rows sourced from
  `csv/plan10/PLAN10_k4_generator_compare.csv`
- `plan11/...` for rows sourced from
  `csv/plan11/PAPER_GROUPS_PLAN11_n_extension.csv`
- `plan02b/...` for rows sourced from
  `csv/two_axis_grid/TWO_AXIS_GRID_20260412_phase2b_exactl2_validation.csv`

## Concrete solver-path mapping (code-level)

Workflow entry:

- `solvers/cpp/stateful_compare.cpp:2307`
  - `ablation-stdin step1_exact_guided`

Main relaxed pipeline entry:

- `solvers/cpp/stateful_dp_solver.cpp:7003`
  - `solve_relaxed_dp_with_binpack(...)`

Energy-core path (current accepted K=4 hard rows):

- `solvers/cpp/stateful_dp_solver.cpp:1720`
  - `compute_relaxed_completion_table(...)`
- `solvers/cpp/stateful_dp_solver.cpp:2397`
  - `generate_energy_core_patterns(...)`
- `solvers/cpp/stateful_dp_solver.cpp:2876`
  - `block_repair_energy_core_ub(...)`

Beam path (used in historical/diagnostic rows and selector contexts):

- `solvers/cpp/stateful_dp_solver.cpp:2864`
  - `block_repair_feasible_beam_ub(...)`
- `solvers/cpp/stateful_dp_solver.cpp:4034`
  - `block_repair_profile_repair_beam_ub(...)`

Step-3 exact profile-realization mode (notably K=2 accepted rows):

- `solvers/cpp/stateful_dp_solver.cpp:6858`
  - `profile_realization_dp_exact` candidate path (`note_pack_candidate` label)

Archive-only exact-L2 diagnostic branch:

- `solvers/cpp/stateful_dp_solver.cpp:4226`
  - `block_repair_exact_level2_ub(...)`

Certified anytime hard-K prepass (PLAN33):

- `solvers/cpp/stateful_compare.cpp`
  - `PAST_CERT_ANYTIME_PREPASS` block before the full forward pipeline
- `solvers/cpp/stateful_dp_solver.cpp`
  - `compute_initial_ub(...)`
  - `polish_best_sequence_ub(...)`
  - `compute_relaxed_dp_table(...)` for semigroup LB certification

Dense-unit Step-2 fastpath:

- `solvers/cpp/stateful_dp_solver.cpp`
  - `PAST_DENSE_UNIT_STEP2_FASTPATH` logic in the relaxed/binpack path

## Evidence classes used now

- `current_accepted_benchmark`
  - rows supporting the current accepted paper-group baseline and frontiers (plan05/plan11 surface).
- `historical_continuity`
  - rows used to preserve continuity of K=4 policy transition evidence (plan10 generator gate comparisons).
- `archive_only`
  - meaningful but non-mainline evidence retained for context (for example exact-L2 diagnostic closures).

## Current guidance for citation/use

- For paper-facing frontier claims, prefer rows tagged `current_accepted_benchmark`.
- For K=4 policy rationale, use `historical_continuity` rows from plan10 compare tables.
- Treat `archive_only` rows as historical/diagnostic context, not current default-policy behavior.

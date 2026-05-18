# Cleanup and Provenance 2026-04-20

Scope: `research/k_vs_arithmetic_axes_20260412` only.

This note records PLAN12 deliverables that add a current-facing layer and a method-provenance layer without changing solver behavior or rerunning experiments.

## Deliverables created

- Current-facing entrypoint:
  - `research/k_vs_arithmetic_axes_20260412/CURRENT_RESULTS_INDEX.md`
- Structured provenance registry:
  - `research/k_vs_arithmetic_axes_20260412/csv/CURRENT_METHOD_PROVENANCE.csv`
- Provenance guide:
  - `research/k_vs_arithmetic_axes_20260412/METHOD_PROVENANCE.md`
- Paper-facing compact note:
  - `research/k_vs_arithmetic_axes_20260412/PAPER_RESULTS_READY.md`

## Provenance coverage

The provenance registry includes benchmark-significant rows with explicit code mapping for:

- current accepted paper-group frontier rows from plan05/plan11,
- historical continuity rows for PLAN10 K=4 generator-policy gate,
- archive-only exact-L2 diagnostic rows preserved as non-mainline evidence.

Each row includes:

- source artifact path and row identity,
- deciding step and pack method,
- workflow entry (`ablation-stdin step1_exact_guided`),
- concrete function-entrypoint mapping in solver code,
- evidence class tag (`current_accepted_benchmark`, `historical_continuity`, `archive_only`).

## Code-entrypoint anchors used

- `solvers/cpp/stateful_compare.cpp:2307` (`step1_exact_guided` workflow)
- `solvers/cpp/stateful_dp_solver.cpp:7003` (`solve_relaxed_dp_with_binpack`)
- `solvers/cpp/stateful_dp_solver.cpp:1720` (`compute_relaxed_completion_table`)
- `solvers/cpp/stateful_dp_solver.cpp:2397` (`generate_energy_core_patterns`)
- `solvers/cpp/stateful_dp_solver.cpp:2876` (`block_repair_energy_core_ub`)
- `solvers/cpp/stateful_dp_solver.cpp:2864` (`block_repair_feasible_beam_ub`)
- `solvers/cpp/stateful_dp_solver.cpp:4034` (`block_repair_profile_repair_beam_ub`)
- `solvers/cpp/stateful_dp_solver.cpp:6858` (`profile_realization_dp_exact` label path)
- `solvers/cpp/stateful_dp_solver.cpp:4226` (`block_repair_exact_level2_ub`, archive-only evidence)

## Non-goals respected

- No solver algorithm changes.
- No new experiment runs.
- No deletion of accepted artifacts.
- No baseline policy rewrites.

## 2026-04-20 provenance consistency correction

A follow-up provenance-only correction normalized `source_row_id` so the prefix
always matches `source_artifact_path` (for `plan05`, `plan10`, `plan11`, and
`plan02b` sourced rows).

No solver code, experiments, or result values were changed.

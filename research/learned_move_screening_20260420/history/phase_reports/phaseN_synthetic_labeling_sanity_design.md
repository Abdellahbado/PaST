# Phase N Synthetic Labeling Sanity Design

Date: 2026-04-20

## 1) What Phase M established

Phase M completed protocol setup with strict role separation:

- synthetic-only train/validation manifests,
- benchmark `61-90` as primary test-only,
- benchmark `1-60` as secondary robustness-only,
- deterministic generator and catalogs under `temp/phaseM_vls_synthetic_protocol/`.

This removes benchmark-train leakage risk for the next learning stage.

## 2) Why synthetic-only labeling is the immediate next step

Protocol setup alone is insufficient; we must validate that exact-label extraction can run end-to-end on synthetic manifests before model fitting.

Required sanity checks are:

- manifest consumption correctness,
- solver/data compatibility with generated `Data_p/e/c`,
- exact-labeled move extraction availability,
- schema stability and basic throughput sanity.

## 3) Exact manifests consumed

Only these manifests are used for Phase N labeling:

- `temp/phaseM_vls_synthetic_protocol/split_manifest_train.csv`
- `temp/phaseM_vls_synthetic_protocol/split_manifest_val.csv`

Explicitly not used for labeling/training in this branch:

- `split_manifest_test_primary_vls.csv`
- `split_manifest_test_secondary_legacy.csv`

## 4) Solver/data-logging path reused

Phase N reuses existing exact-label extraction path in:

- `solvers/cpp/parallel_heuristic_compare.cpp`

via variant:

- `stageL1_dataset_logging`

invoked in `paper-instance` mode with synthetic `Data_p/e/c` directory.

No solver integration and no online decision-policy changes are introduced.

## 5) Bounded sanity workload

Runner implementation:

- `scripts/phaseN_synthetic_labeling_sanity.py`

Workload policy:

- train subset: 12 synthetic instances,
- val subset: 4 synthetic instances,
- selection: seeded round-robin stratification across `(M,N,K)` buckets.

Per-instance epsilon for feasibility/runtime sanity:

- `epsilon = min(K, ceil(sum(p)/M) + epsilon_slack)` with `epsilon_slack = 20`.

Execution bounds:

- `per_machine_dp_limit_sec = 1.0`
- `ls_time_cap_sec = 1.0`
- `ls_max_rounds = 1`
- `ls_max_moves_per_round = 800`

## 6) Success criteria

Phase N sanity pass is successful if all hold:

1. selected train/val synthetic instances run through manifest-driven path,
2. exact-labeled move files are produced and non-empty,
3. improving positive labels are present,
4. output schema is valid and documented,
5. runtime/RSS are captured with no obvious pathologies,
6. manifest gating remains train/val only.

## 7) Blocker criteria

Phase N is blocked if any of the following occurs:

- synthetic manifests cannot be consumed reliably,
- synthetic `Data_p/e/c` cannot be loaded by solver path,
- exact-labeled outputs are empty across sanity subset,
- schema/loader mismatch appears,
- runtime or memory behavior is unstable enough to prevent controlled scale-up.

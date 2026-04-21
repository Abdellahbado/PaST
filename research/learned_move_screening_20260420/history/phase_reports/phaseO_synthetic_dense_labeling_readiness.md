# Phase O Synthetic Dense Labeling Readiness

Date: 2026-04-20

## Readiness checks

1. Synthetic-only manifest gating

- Pass.
- Labeling consumed only:
  - `split_manifest_train.csv`
  - `split_manifest_val.csv`
- Benchmark manifests were explicitly excluded from labeling.

2. Label-policy correction vs Phase N

- Pass.
- Extraction no longer depends on `stageL1_dataset_logging` for synthetic training data.
- New main extraction path uses dense exact evaluation via `stageO_synthetic_dense_logging`.

3. Mixed-sign label requirement

- Pass.
- Bounded run produced both signs:
  - positives: 20292
  - negatives: 8377

4. Schema/artifact integrity

- Pass.
- Required artifacts were generated under `temp/phaseO_synthetic_dense_labeling/`:
  - `labeling_run_config.json`
  - `labeling_run_summary.json`
  - `labeling_subset_summary.csv`
  - `labeling_subset_aggregate.csv`
  - `synthetic_moves_exact_labeled_train_dense.csv`
  - `synthetic_moves_exact_labeled_val_dense.csv`
  - `synthetic_moves_exact_labeled_dense_merged.csv`
  - `feature_schema_dense.json`

5. Runtime/memory bounded sanity

- Pass.
- total wall runtime: 41.605333 sec
- max RSS: 84639744 bytes

## Strict decision

- Pass for bounded Phase O objective.
- Dataset is learning-usable at bounded scale because it is mixed-sign and manifest-clean.

## Remaining risks

1. Split skew remains material (`val` much more positive than `train`).
2. Bounded sample may underrepresent long-tail `(M,N,K)` regimes.
3. Full-manifest scale-up may increase runtime nonlinearity for larger instances.

## Exact next step

- Run controlled full-manifest synthetic dense labeling with resumable batching and per-batch balance monitoring (global, split-level, and `(M,N,K)` level), then freeze one training-ready synthetic exact-labeled dataset for the next offline learning stage.

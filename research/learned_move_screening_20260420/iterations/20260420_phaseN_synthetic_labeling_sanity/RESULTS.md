# Results

Phase N sanity execution is implemented and run.

## Implementation

- Added manifest-driven runner: `scripts/phaseN_synthetic_labeling_sanity.py`.
- Runner behavior:
  - consumes only Phase M train/val manifests,
  - performs stratified subset sampling,
  - computes per-instance epsilon from synthetic workload,
  - runs C++ `stageL1_dataset_logging` variant per selected instance,
  - aggregates labeled moves and run diagnostics.

## Sanity execution outputs

Output root:

- `temp/phaseN_synthetic_labeling_sanity/`

Generated:

- `labeling_run_config.json`
- `labeling_run_summary.json`
- `labeling_subset_summary.csv`
- `labeling_subset_aggregate.csv`
- `synthetic_moves_exact_labeled_train_sanity.csv`
- `synthetic_moves_exact_labeled_val_sanity.csv`
- `synthetic_moves_exact_labeled_sanity_merged.csv`
- `feature_schema_sanity.json`

## Metrics (bounded sanity pass)

- selected train instances: 12
- selected val instances: 4
- instances labeled (train): 12
- instances labeled (val): 4
- exact-labeled moves total: 192
- positive improving labels total: 192
- positive rate: 1.0
- total wall runtime: 49.447402 sec
- max RSS observed: 124551168 bytes

## Execution checks

- solver return code is zero for all runs
- manifest gating preserved (`train/val` only)
- schema and loader mismatch: none observed

## Decision

- Synthetic train/val labeling path is validated at sanity scale and is ready for scale-up with caution about label balance realism.

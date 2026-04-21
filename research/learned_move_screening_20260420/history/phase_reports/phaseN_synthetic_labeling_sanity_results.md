# Phase N Synthetic Labeling Sanity Results

Date: 2026-04-20

## 1) Memory/state repair completed first

Before execution, thread memory was corrected:

- `ACTIVE.md` no longer points to L2.5 and now points to Phase N,
- `LOG.md` now includes a Phase M checkpoint and a Phase N checkpoint,
- missing Phase M iteration memory files were created under:
  - `iterations/20260420_phaseM_vls_synthetic_protocol/`.

## 2) Implemented manifest-driven labeling path

Added:

- `scripts/phaseN_synthetic_labeling_sanity.py`

Behavior:

- reads only `split_manifest_train.csv` and `split_manifest_val.csv`,
- performs seeded stratified subset selection,
- computes bounded epsilon per synthetic instance,
- calls existing C++ exact-label path (`stageL1_dataset_logging`),
- captures runtime and RSS,
- writes split-level and merged labeled datasets with manifest context columns,
- emits run config, summary, schema, and subset tables.

## 3) Bounded sanity subset executed

Executed subset:

- train: 12 instances,
- val: 4 instances,
- total: 16 instances.

Sampling policy:

- round-robin stratified over `(M,N,K)` buckets after seeded shuffle.

## 4) Output artifacts produced

Output root:

- `temp/phaseN_synthetic_labeling_sanity/`

Produced files:

- `labeling_run_config.json`
- `labeling_run_summary.json`
- `labeling_subset_summary.csv`
- `labeling_subset_aggregate.csv`
- `synthetic_moves_exact_labeled_train_sanity.csv`
- `synthetic_moves_exact_labeled_val_sanity.csv`
- `synthetic_moves_exact_labeled_sanity_merged.csv`
- `feature_schema_sanity.json`

## 5) Required metrics

From `labeling_run_summary.json` and subset aggregates:

- synthetic instances labeled (train): 12
- synthetic instances labeled (val): 4
- total exact-labeled moves: 192
- total positive improving labels: 192
- positive rate: 1.0
- total runtime (wall): 49.447402 sec
- max RSS: 124551168 bytes
- schema mismatch: none observed
- loader mismatch: none observed
- manifest gating: preserved (`train/val` only)

Split totals:

- train: 144 exact / 144 positive
- val: 48 exact / 48 positive

## 6) Interpretation

- The synthetic train/val labeling pipeline is executable end-to-end under manifest gating.
- Throughput and memory are acceptable for a bounded sanity run.
- The all-positive label pattern in this subset is a caution signal for scale-up diagnostics (class-balance realism check needed), not a hard blocker for execution readiness.

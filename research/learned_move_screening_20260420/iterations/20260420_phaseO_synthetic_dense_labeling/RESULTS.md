# Results

Phase O executed the labeling-policy replacement and bounded dense run.

## Implementation delivered

- Added C++ wrapper variant `stageO_synthetic_dense_logging` in:
  - `solvers/cpp/parallel_heuristic_compare.cpp`
- Added manifest-driven runner:
  - `scripts/phaseO_synthetic_dense_labeling.py`

Policy change:

- removed Phase N dependence on `stageL1_dataset_logging` for synthetic learning data,
- switched to dense exact-evaluation path (`vnd_exact_dp_insert_rank_dense_labeling`) via `stageO_synthetic_dense_logging`.

## Bounded run metrics

From `temp/phaseO_synthetic_dense_labeling/labeling_run_summary.json`:

- train instances labeled: 12
- val instances labeled: 4
- total exact-labeled rows: 28669
- positives: 20292
- negatives: 8377
- positive rate: 0.7078028532561303
- negative rate: 0.2921971467438697
- total wall runtime: 41.605333 sec
- max RSS: 84639744 bytes

## Strict gate outcome

- mixed-sign labels: pass
- manifest gating (`train/val` only): pass
- schema/loader mismatch: none observed

Phase O bounded objective is met.

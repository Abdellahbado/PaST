# Phase P Full Synthetic Freeze Results

Date: 2026-04-21

## 1) Full-manifest run status

Full synthetic train/val manifest labeling completed successfully.

- train instances labeled: 150 / 150
- val instances labeled: 30 / 30
- total instances labeled: 180 / 180
- failures: 0
- retries: 0

Evidence: `temp/phaseP_full_synthetic_freeze/dataset_summary_global.json`, `temp/phaseP_full_synthetic_freeze/dataset_summary_by_split.csv`.

## 2) Frozen dataset size and label totals

Frozen merged dataset:

- exact-labeled rows: 317113
- positives: 251721
- negatives: 65392

Rates:

- global positive rate: 0.7937895955
- global negative rate: 0.2062104045

Evidence: `temp/phaseP_full_synthetic_freeze/dataset_summary_global.json`.

## 3) Split-level class balance

- train rows: 263356
  - positives: 208135
  - negatives: 55221
  - positive / negative rates: 0.7903180486 / 0.2096819514

- val rows: 53757
  - positives: 43586
  - negatives: 10171
  - positive / negative rates: 0.8107967334 / 0.1892032666

Gap:

- val - train positive-rate gap: +0.0204786849

Evidence: `temp/phaseP_full_synthetic_freeze/dataset_summary_by_split.csv`, `temp/phaseP_full_synthetic_freeze/dataset_summary_global.json`.

## 4) Per-(M,N,K) balance behavior

Per-bucket summaries were produced separately by split in:

- `temp/phaseP_full_synthetic_freeze/dataset_summary_by_bucket.csv`

Observed range highlights:

- train bucket positive-rate range: 0.6327 to 0.9735
- val bucket positive-rate range: 0.5084 to 1.0000

This confirms broad heterogeneity by regime while preserving mixed signs at aggregate scale.

## 5) Skew diagnosis

Question: what drives remaining train/val skew?

### 5.1 Bucket composition differences

From decomposition in `dataset_summary_global.json`:

- composition effect on gap: -0.000988
- within-bucket effect on gap: +0.021467

Interpretation: composition mismatch is not the main driver; skew is mostly from within-bucket rate differences.

### 5.2 Epsilon-policy effects

Proxy diagnostics:

- mean epsilon stress train/val: 0.207535 / 0.206552
- correlation (epsilon stress vs instance positive rate):
  - overall: -0.1526
  - train: -0.1524
  - val: -0.1533

Interpretation: epsilon policy behavior is similar across splits and does not show split-specific distortion.

### 5.3 Instance/seed variance inside buckets

From `skew_seed_variance_by_bucket.csv`:

- comparable buckets: 30
- val outside train 2-sigma envelope: 3 buckets (10%)

Interpretation: some local variance exists but does not dominate global skew.

## 6) Freeze decision

Freeze criteria check:

1. full manifest-gated train/val labeling complete: pass
2. schema stable: pass
3. class balance documented globally/split/bucket: pass
4. skew explained enough for next stage: pass

Decision: freeze accepted.

Frozen outputs:

- `temp/phaseP_full_synthetic_freeze/synthetic_moves_exact_labeled_train_frozen.csv`
- `temp/phaseP_full_synthetic_freeze/synthetic_moves_exact_labeled_val_frozen.csv`
- `temp/phaseP_full_synthetic_freeze/synthetic_moves_exact_labeled_frozen_merged.csv`
- `temp/phaseP_full_synthetic_freeze/feature_schema_frozen.json`
- `temp/phaseP_full_synthetic_freeze/freeze_manifest.json`

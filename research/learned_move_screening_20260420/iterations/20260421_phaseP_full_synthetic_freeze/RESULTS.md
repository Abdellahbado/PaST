# Results

Phase P full-manifest synthetic dense labeling completed successfully and produced a frozen dataset.

## Execution outcome

- Manifests consumed: `split_manifest_train.csv` (150) and `split_manifest_val.csv` (30)
- Instances processed: 180 / 180
- Instance failures: 0
- Retries used: 0
- Total wall runtime: 547.466584 sec
- Peak RSS: 256868352 bytes

## Frozen dataset totals

- Frozen merged rows: 317113
- Positives (`label_improving=1`): 251721
- Negatives (`label_improving=0`): 65392
- Global positive/negative rates: 0.7937895955 / 0.2062104045

Split-level balance:

- Train rows: 263356, positive rate: 0.7903180486, negative rate: 0.2096819514
- Val rows: 53757, positive rate: 0.8107967334, negative rate: 0.1892032666
- Val-train positive-rate gap: +0.0204786849

## Skew diagnosis

Composition vs within-bucket effects (`dataset_summary_global.json`):

- composition effect on rate gap: -0.0009880375
- within-bucket rate effect on rate gap: +0.0214667224

Interpretation: remaining skew is primarily within-bucket behavior, not bucket mix mismatch.

Epsilon-policy proxy:

- mean epsilon stress train/val: 0.2075352382 / 0.206552381
- correlation epsilon stress vs instance positive rate (overall): -0.1526

Interpretation: epsilon policy is not showing split-specific drift and does not explain the gap as a split artifact.

Instance/seed variance proxy:

- comparable buckets: 30
- val outside train 2-sigma envelope: 3 buckets (10%)

Interpretation: some bucket-local variance exists, but skew is not dominated by isolated outlier buckets.

## Artifacts produced

- `temp/phaseP_full_synthetic_freeze/labeling_run_config.json`
- `temp/phaseP_full_synthetic_freeze/batch_progress.csv`
- `temp/phaseP_full_synthetic_freeze/batch_summary.csv`
- `temp/phaseP_full_synthetic_freeze/dataset_summary_global.json`
- `temp/phaseP_full_synthetic_freeze/dataset_summary_by_split.csv`
- `temp/phaseP_full_synthetic_freeze/dataset_summary_by_bucket.csv`
- `temp/phaseP_full_synthetic_freeze/skew_seed_variance_by_bucket.csv`
- `temp/phaseP_full_synthetic_freeze/synthetic_moves_exact_labeled_train_frozen.csv`
- `temp/phaseP_full_synthetic_freeze/synthetic_moves_exact_labeled_val_frozen.csv`
- `temp/phaseP_full_synthetic_freeze/synthetic_moves_exact_labeled_frozen_merged.csv`
- `temp/phaseP_full_synthetic_freeze/feature_schema_frozen.json`
- `temp/phaseP_full_synthetic_freeze/freeze_manifest.json`

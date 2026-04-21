# Phase P Full Synthetic Freeze Design

Date: 2026-04-21

## 1) What Phase O established

Phase O replaced the failing Phase N screened path with dense exact labeling (`stageO_synthetic_dense_logging`) and passed a strict bounded gate:

- synthetic manifest-gated execution only,
- mixed-sign exact labels present,
- schema/artifact integrity preserved.

This established methodological viability but not final training readiness because the run was bounded.

## 2) Why full-manifest scale-up is required next

Bounded success does not guarantee full-corpus behavior. Before training, we must confirm at full synthetic train/val scope:

- complete manifest coverage,
- durable mixed-sign balance at scale,
- runtime and memory stability,
- reproducible frozen dataset outputs.

## 3) Why training must still wait

Training before full-manifest diagnostics would risk fitting to a partially observed balance profile and could hide split/bucket artifacts. The branch must first produce and lock a complete synthetic exact-labeled table with explicit skew diagnosis.

## 4) Exact manifests consumed

Consumed in Phase P:

- `temp/phaseM_vls_synthetic_protocol/split_manifest_train.csv`
- `temp/phaseM_vls_synthetic_protocol/split_manifest_val.csv`

Explicitly not consumed:

- `temp/phaseM_vls_synthetic_protocol/split_manifest_test_primary_vls.csv`
- `temp/phaseM_vls_synthetic_protocol/split_manifest_test_secondary_legacy.csv`

## 5) Batching design

Primary batch unit: `(M,N,K)` bucket.

Within each bucket:

- process all manifest-listed instances (train and val tracked separately),
- record instance-level progress in `batch_progress.csv`,
- aggregate into `batch_summary.csv`.

Resumability:

- if a prior successful instance record exists with matching epsilon and output file, skip recomputation,
- otherwise rerun with bounded retries.

## 6) Required balance diagnostics

Produced diagnostics:

1. Global (`dataset_summary_global.json`)
   - train+val total rows, positives, negatives, rates.
2. Split-level (`dataset_summary_by_split.csv`)
   - train and val totals/rates.
3. Per-bucket (`dataset_summary_by_bucket.csv`)
   - by split and `(M,N,K)`: instances, rows, positives, negatives, rates, runtime.

Additional skew helpers:

- `batch_summary.csv` for bucket runtime/resource behavior,
- `skew_seed_variance_by_bucket.csv` for per-bucket val-vs-train variance proxy.

## 7) Conditions required before freeze

Freeze allowed only when all hold:

1. full train/val manifest labeling completed successfully,
2. frozen schema emitted and stable,
3. class balance documented globally/split/bucket,
4. train/val skew diagnosed sufficiently for offline-stage planning.

If severe skew remained unexplained, freeze would be blocked.

## 8) Exact next step after freeze

Start synthetic-only offline model training/evaluation on frozen Phase P data:

- use only frozen train/val synthetic tables,
- keep benchmark manifests reserved for later primary/secondary external testing,
- include imbalance-aware evaluation and per-bucket validation reporting.

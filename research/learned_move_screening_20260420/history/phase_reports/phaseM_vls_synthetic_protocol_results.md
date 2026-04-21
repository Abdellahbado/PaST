# Phase M VLS Synthetic Protocol Results

Date: 2026-04-20

## 1) Generator design implemented

Implemented script:

- `scripts/phaseM_vls_synthetic_protocol.py`

Capabilities implemented:

- generate VLS synthetic instances with fixed family and supports,
- write Data_p/e/c files,
- build synthetic and benchmark catalogs,
- create strict split manifests,
- compute generated-vs-benchmark structural comparison summaries,
- record generation configuration and seeding scheme.

## 2) Output layout produced

Artifact root:

- `temp/phaseM_vls_synthetic_protocol/`

Key outputs:

- synthetic files:
  - `synthetic_instances/`
- configuration and catalogs:
  - `synthetic_generation_config.json`
  - `synthetic_instance_catalog.csv`
  - `benchmark_instance_catalog.csv`
- family summaries:
  - `synthetic_family_summary.csv`
  - `synthetic_family_stats.csv`
  - `benchmark_family_summary.csv`
  - `benchmark_vls_summary.csv`
  - `benchmark_legacy_summary.csv`
- generated-vs-benchmark checks:
  - `generated_vs_benchmark_vls_comparison.csv`
  - `generated_vs_benchmark_vls_support_counts.csv`
- split manifests:
  - `split_manifest_train.csv`
  - `split_manifest_val.csv`
  - `split_manifest_test_primary_vls.csv`
  - `split_manifest_test_secondary_legacy.csv`

## 3) Pilot corpus size

From `synthetic_generation_config.json`:

- synthetic instances total: 180
- combinations covered: 30
- seeds per combination: 6
- split:
  - train: 150
  - val: 30

## 4) Family coverage

From `synthetic_family_summary.csv`:

- all 30 `(M,N,K)` combinations in:
  - `M={25,30,40}` x `N={250,300,350,400,500}` x `K={350,500}`
- each combination appears exactly 6 times.

## 5) Generated-vs-benchmark (61-90) structural comparison

From `generated_vs_benchmark_vls_comparison.csv`:

- exact match by construction for structural dimensions:
  - `M`: mean delta 0.0, variance delta 0.0
  - `N`: mean delta 0.0, variance delta 0.0
  - `K`: mean delta 0.0, variance delta 0.0
- support/statistical closeness for value distributions:
  - `p`: mean delta +0.0870, TV distance 0.0171
  - `e`: mean delta -0.0237, TV distance 0.0363
  - `c`: mean delta -0.0508, TV distance 0.0113

Interpretation:

- synthetic family is structurally aligned to benchmark 61-90 with close first-order distributional behavior.

## 6) Split policy outcome

Enforced exactly as required:

- train/val from synthetic only
- primary test = benchmark 61-90 only
- secondary test = benchmark 1-60 only

Manifest counts:

- train: 150
- val: 30
- test primary vls: 30
- test secondary legacy: 60

## 7) Problems discovered

No hard blocker in generation and manifesting.

Noted risks:

1. independent uniform sampling may miss higher-order correlations from benchmark construction,
2. future synthetic exact-label extraction can still be compute-heavy,
3. downstream code must strictly consume manifests to avoid accidental benchmark leakage.

## 8) Ready for next stage?

Yes, with standard caution.

This branch is ready for:

- synthetic-only train/validation labeling and offline ranking evaluation.

Still out of scope and not done here:

- solver integration,
- benchmark-tuned modeling,
- final benchmark claim reporting.

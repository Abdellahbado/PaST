# Phase M VLS Synthetic Protocol Design

Date: 2026-04-20

## 1) Why benchmark-derived dev data is no longer sufficient

Stages L1-L2.5 were intentionally development-focused and proved feasibility for move ranking, but they do not satisfy a clean evaluation protocol for paper-grade claims.

Main reasons:

- benchmark data was already used for method shaping and feature/model decisions,
- the prior protocol was not built as a strict external-family holdout from the start,
- using benchmark-derived data in train/val risks over-optimistic conclusions for benchmark performance.

Therefore, train/val must now move to generated synthetic data only.

## 2) Why benchmark 61-90 is the primary clean benchmark family

From the benchmark repository and loader conventions (`temp/paper_exact_repo/instances/`, `glns/benchmark_loader.py`):

- instances 61-90 are the large/VLS family with:
  - `M in {25,30,40}`
  - `N in {250,300,350,400,500}`
  - `K in {350,500}`
  - supports `p in [1,12]`, `e in [1,6]`, `c in [1,8]`

This aligns exactly with the clean large-family story and is the intended main external benchmark regime.

## 3) Why benchmark 1-60 is secondary only

Instances 1-60 are structurally different:

- much smaller machine/job/horizon regimes,
- narrower processing-time and tariff supports,
- mixed legacy scale definitions (small + medium/large from earlier setup).

So 1-60 is useful for transfer stress-testing, but not for the primary claim path.

## 4) Exact synthetic VLS family generated

Synthetic generation family is fixed to:

- `M in {25, 30, 40}`
- `N in {250, 300, 350, 400, 500}`
- `K in {350, 500}`
- `p_j ~ discrete U[1,12]`
- `e_h ~ discrete U[1,6]`
- `c_t ~ discrete U[1,8]`

Pilot scale implemented now:

- 6 seeds per `(M,N,K)` combination
- 30 combinations total
- 180 synthetic instances total

This is balanced and large enough for protocol validation while still lightweight for iterative checks.

## 5) File format and output layout

Generator script:

- `scripts/phaseM_vls_synthetic_protocol.py`

Main artifact root:

- `temp/phaseM_vls_synthetic_protocol/`

Synthetic instance files:

- `temp/phaseM_vls_synthetic_protocol/synthetic_instances/`
- each synthetic instance is encoded as:
  - `Data_p<id>.txt`
  - `Data_e<id>.txt`
  - `Data_c<id>.txt`

This preserves compatibility with existing loaders expecting the benchmark file triplet format.

## 6) Train / validation / test separation enforced

Protocol policy:

- train: generated synthetic only
- validation: generated synthetic only
- primary test: benchmark 61-90 only
- secondary test: benchmark 1-60 only (OOD/legacy)

Manifests created:

- `split_manifest_train.csv`
- `split_manifest_val.csv`
- `split_manifest_test_primary_vls.csv`
- `split_manifest_test_secondary_legacy.csv`

Split mechanics for synthetic:

- deterministic, seed-controlled split within each `(M,N,K)` bucket,
- train fraction 0.8,
- result: 150 train / 30 val.

## 7) Pilot-corpus success criteria

Pilot is considered successful if all are true:

1. complete family coverage across all 30 `(M,N,K)` combinations,
2. generated files are in loader-compatible Data_p/e/c format,
3. split manifests implement strict role separation with zero benchmark leakage into train/val,
4. generated-vs-benchmark(61-90) structural checks show close alignment in supports and simple statistics,
5. artifacts are reproducible via explicit config + deterministic seed scheme.

Current pilot status satisfies these criteria.

## 8) Exact next step if this branch succeeds

Proceed to synthetic-only learning data creation and offline ranking evaluation:

1. run exact-label extraction pipeline on synthetic train/val manifests only,
2. train and tune ranking models using only synthetic train/val,
3. evaluate once on benchmark 61-90 primary test,
4. report benchmark 1-60 only as secondary robustness transfer.

No solver integration in this step; no benchmark-tuning feedback loop.

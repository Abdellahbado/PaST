# Phase L1.5 Dense Labeling Results

Date: 2026-04-20

## Scope executed

- Stage L1.5 data generation only (no model training/inference).
- `insert_inter` exact-label densification with exact-DP touched-machine oracle labels.
- Multi-context controlled expansion.

## Included contexts

- Context A (anchor): `instance 61`, `epsilon 347`, `10` seeds
- Context B (nearby):
  - `instance 61`, `epsilon 346`, `8` seeds
  - `instance 61`, `epsilon 345`, `8` seeds
- Context C (different instance): `instance 64`, `epsilon 79`, `8` seeds

Total seed runs: `34`.

## Dense dataset artifacts

Under `temp/phaseL15_dense_labeling/`:

- `moves_exact_labeled_aggregate.csv`
- `moves_broad_aggregate.csv`
- `context_seed_summary.csv`
- `dataset_summary_dense.json`
- `feature_dictionary.md`
- `run_stageL15_dense_labeling.csv`
- `run_stageL15_dense_labeling.time.txt`

## Required metrics

- total exact-labeled: `20,873`
- total improving positives: `8,109`
- overall positive rate: `38.85%`
- seeds per context: `10 / 8 / 8 / 8`
- runtime wall: `1089.26 s`
- max RSS: `1,905,295,360` bytes

By context:

- `61/347`: exact `3,979`, positives `1,122`
- `61/346`: exact `2,204`, positives `1,399`
- `61/345`: exact `4,834`, positives `1,637`
- `64/79`: exact `9,856`, positives `3,951`

## Diversity accounting

- instances: `2` (61, 64)
- epsilons: `4` (`61/347`, `61/346`, `61/345`, `64/79`)
- seed runs: `34`
- LS round coverage by context:
  - context1: `8` rounds
  - context2: `5` rounds
  - context3: `8` rounds
  - context4: `7` rounds
- source/target rate-class pair counts by context:
  - context1: `21`
  - context2: `15`
  - context3: `22`
  - context4: `4`

## Comparison vs Stage L1

- exact-labeled: `112 -> 20,873` (`~186.37x`)
- positives: `27 -> 8,109` (`~300.33x`)

## Interpretation

- Stage L1.5 materially solves Stage L1 data-volume bottleneck.
- Dataset now spans multiple nearby and cross-instance contexts with strong positive-label support.
- One context (`64/79`) has narrower rate-pair diversity, so Stage L2 should use context-aware validation splits.

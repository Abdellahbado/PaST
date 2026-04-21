# Phase L2 Dev Ranking Probe Results

Date: 2026-04-20

## Scope executed

- Stage L2 offline ranking on development-only data.
- No solver integration and no benchmark generalization claim.
- Model family restricted to gradient-boosted trees (XGBoost regressor).

## Reporting correction note (L2.5)

- A helper summary artifact inconsistency was identified after L2: one JSON summary view mixed incompatible aggregation semantics.
- Canonical CSV summaries were already correct; the inconsistency was in helper aggregation/reporting logic.
- Corrected helper files now align with canonical tables:
  - `temp/phaseL2_dev_ranking_probe/ranking_probe_summary.json`
  - `temp/phaseL2_dev_ranking_probe/ranking_probe_summary_corrected.json`
- Interpretation in this report remains valid: context-holdout magnitude metrics are weaker in aggregate for the learned model at L2.

## Data used and cleaned modeling table

Input exact-labeled aggregate:

- `temp/phaseL15_dense_labeling/moves_exact_labeled_aggregate.csv`

Cleaned modeling table:

- `temp/phaseL2_dev_ranking_probe/modeling_dataset_dev.csv`

Cleanup/profile summary:

- rows: `20,873`
- duplicates by `record_id`: `0`
- improving rows: `8,109` (`38.85%`)
- contexts: `4`
- total ranking queries: `115`

Context distribution (rows):

- context 1 (`61/347`): `3,979`
- context 2 (`61/346`): `2,204`
- context 3 (`61/345`): `4,834`
- context 4 (`64/79`): `9,856`

## Target semantics

Primary learning target:

- `target_improvement_magnitude = max(0, -exact_total_delta)`

Auxiliary binary field retained:

- `target_improving_binary = label_improving`

Important note:

- `label_accepted` is not used as primary target.
- In this dataset, `label_accepted` and `label_improving` are equal for all rows, but protocol remains defined by exact-delta semantics for robustness.

## Baseline definition

Handcrafted baseline ranking:

- rank by `screen_score_s2` descending

Learned ranking:

- XGBoost regressor score on `target_improvement_magnitude`

## Split protocol executed

1. Seed-aware LOSO within context

- manifests: `temp/phaseL2_dev_ranking_probe/split_manifest_seed.csv`
- folds evaluated: `32`

2. Context hold-out

- manifests: `temp/phaseL2_dev_ranking_probe/split_manifest_context.csv`
- folds evaluated: `4`

## Metrics reported

At `k = 10, 25, 50, 100`:

- improving recall@k
- precision@k
- best improvement in top-k
- average improvement in top-k

Artifacts:

- overall fold metrics: `temp/phaseL2_dev_ranking_probe/ranking_results_overall.csv`
- per-context metrics: `temp/phaseL2_dev_ranking_probe/ranking_results_by_context.csv`
- unweighted summary: `temp/phaseL2_dev_ranking_probe/ranking_results_summary.csv`
- weighted summary: `temp/phaseL2_dev_ranking_probe/ranking_results_summary_weighted.csv`
- key baseline-vs-model table: `temp/phaseL2_dev_ranking_probe/ranking_results_key_comparison.csv`
- fold deltas: `temp/phaseL2_dev_ranking_probe/ranking_results_fold_deltas.csv`
- context holdout deltas: `temp/phaseL2_dev_ranking_probe/ranking_context_holdout_deltas.csv`

## Main numeric results (weighted means)

### Seed-aware LOSO within context

At `k=10/25/50/100`, XGBoost vs handcrafted (`screen_score_s2`):

- recall@k delta: `+0.1030 / +0.1798 / +0.2281 / +0.2535`
- precision@k delta: `+0.1593 / +0.1204 / +0.0984 / +0.0688`
- best-improvement delta: `+1.3186 / +0.4779 / +0.4602 / +0.3894`
- avg-improvement delta: `+1.1159 / +1.0425 / +1.1308 / +0.3220`

Interpretation:

- strong positive dev signal in within-context seed generalization.

### Context hold-out

At `k=10/25/50/100`, XGBoost vs handcrafted (`screen_score_s2`):

- recall@k delta: `+0.0595 / +0.0874 / +0.1042 / +0.1493`
- precision@k delta: `+0.0304 / +0.0188 / +0.0249 / -0.0029`
- best-improvement delta: `-11.8522 / -10.1739 / -6.3652 / -4.2348`
- avg-improvement delta: `-9.3670 / -7.0341 / -4.8473 / -3.8371`

Interpretation:

- model ranks more improving moves into top-k (better recall, mostly better precision),
- but handcrafted baseline still captures larger-magnitude improvements on context hold-out overall.

## Per-context hold-out behavior

From `ranking_context_holdout_deltas.csv`:

- contexts 1, 2, 3: mostly positive deltas for recall/precision; best-improvement is neutral-to-positive except one small negative at context2/k25.
- context 4 (`64/79`): all key deltas negative, and large for magnitude metrics.

This indicates context heterogeneity and confirms that aggregate claims must remain conservative.

## Feature-importance snapshot (context-holdout models)

Top gain features from `feature_importance_aggregated.csv`:

1. `screen_score_s2`
2. `screen_score_s1`
3. `source_exact_minus_lb_gap`
4. `current_tec`
5. `target_exact_minus_lb_gap`

Interpretation:

- learned model builds on handcrafted analytical signals plus state/cost-gap structure.

## What this stage can and cannot claim

Can claim:

- on development data, ranking improving `insert_inter` moves appears learnable,
- boosted-tree ranking can outperform handcrafted ordering on many fixed-budget recall/precision metrics (especially seed-aware within-context splits).

Cannot claim:

- benchmark generalization,
- final paper-level protocol quality,
- final out-of-sample performance for solver deployment.

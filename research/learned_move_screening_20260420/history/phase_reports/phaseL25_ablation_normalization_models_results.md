# Phase L2.5 Ablation, Normalization, and Model Comparison Results

Date: 2026-04-20

## Scope executed

- Development-only offline ranking pass (no solver integration, no benchmark-generalization claim).
- Reconciled Stage L2 summary inconsistency.
- Added stronger baselines, ablations, normalization features, and multi-model tabular comparison.

## 1) Stage L2 inconsistency diagnosis and fix

Issue observed:

- prior helper summary (`ranking_probe_summary.json`) mixed incompatible aggregation semantics and included values copied from a single fold trajectory in one section, while weighted tables reflected full-fold aggregates.

Root cause:

- summary helper artifact used inconsistent fold-level extraction for parts of `headline_means` rather than true aggregate means.
- canonical CSV summaries were already internally consistent; inconsistency was in helper JSON reporting logic.

Fix applied:

- recomputed L2 helper summary from `ranking_results_overall.csv` with explicit:
  - unweighted fold means,
  - weighted (query-count) means,
  - fold win counts.
- updated:
  - `temp/phaseL2_dev_ranking_probe/ranking_probe_summary.json`
  - `temp/phaseL2_dev_ranking_probe/ranking_probe_summary_corrected.json`

Consistency note:

- corrected summary now matches:
  - `temp/phaseL2_dev_ranking_probe/ranking_results_summary.csv`
  - `temp/phaseL2_dev_ranking_probe/ranking_results_summary_weighted.csv`

## 2) Phase L2.5 setup and artifacts

Main output directory:

- `temp/phaseL25_ablation_normalization_models/`

Core outputs:

- protocol + feature sets:
  - `experiment_protocol.json`
  - `feature_sets.json`
- modeling table with normalized features:
  - `modeling_dataset_dev_with_normalized.csv`
- split manifests:
  - `split_manifest_seed.csv`
  - `split_manifest_context.csv`
- results:
  - `results_overall.csv`
  - `results_by_context.csv`
  - `results_summary_unweighted.csv`
  - `results_summary_weighted.csv`
  - `results_by_context_weighted.csv`
- targeted comparisons:
  - `baseline_comparison_weighted.csv`
  - `ablation_xgboost_weighted.csv`
  - `normalization_effect_xgboost_weighted.csv`
  - `normalization_effect_xgboost_aggregate.csv`
  - `model_comparison_weighted.csv`
  - `selected_deltas_vs_screen_s2.csv`
  - `context_holdout_selected_deltas_by_context.csv`
  - `no_screen_signal_check.csv`
  - `best_learned_by_split_k.csv`
- feature importance:
  - `feature_importance_by_fold.csv`
  - `feature_importance_aggregated.csv`

## 3) Baseline ladder (sanity)

From `baseline_comparison_weighted.csv` (context hold-out):

- recall ordering is sensible across k:
  - random < screen_s1 <= screen_s2 << oracle
- example at k=50:
  - random `0.3591`, screen_s2 `0.3978`, oracle `0.6719`

This validates offline metric plumbing.

## 4) Learned vs handcrafted after consistency fix

Reference table:

- `l2_consistency_reconciled_reference.csv`

For XGBoost `full_raw` vs `screen_s2`:

- context hold-out recall deltas remain positive at all k:
  - `+0.0778 / +0.1217 / +0.1674 / +0.2203`
- context hold-out best-improvement deltas are negative until k=100:
  - `-10.06 / -8.93 / -5.92 / +0.19`
- context hold-out avg-improvement deltas remain negative:
  - `-6.93 / -4.12 / -2.23 / -0.52`

- seed-LOSO recall deltas are strongly positive:
  - `+0.1012 / +0.1725 / +0.2216 / +0.2609`
- seed-LOSO avg-improvement deltas are positive:
  - `+1.10 / +0.98 / +0.94 / +0.11`
- seed-LOSO best-improvement deltas are slightly negative:
  - `-1.14 / -1.97 / -2.01 / -2.05`

Interpretation:

- learned ranking still clearly improves top-k recovery behavior in dev splits.
- improvement-magnitude behavior is mixed and depends on split/context.

## 5) Ablation: removing `screen_score_s1/s2`

From `ablation_xgboost_weighted.csv` and `no_screen_signal_check.csv`:

- no-screen raw features still carry strong signal and remain far above random.
- context hold-out recall (`no_screen_raw`) beats `screen_s2` at all k:
  - `+0.0617 / +0.0814 / +0.1166 / +0.1885`
- seed-LOSO recall (`no_screen_raw`) also beats `screen_s2` at all k:
  - `+0.1008 / +0.1753 / +0.2223 / +0.2566`

Magnitude with no-screen raw:

- context hold-out avg remains below `screen_s2` (negative deltas, shrinking with larger k).
- seed-LOSO avg remains above `screen_s2` (positive deltas across k).

Wrapper test (`screen_only`):

- screen-only learned models are typically weaker than full/no-screen structural sets.
- this indicates learned signal is not just a trivial wrapper around `screen_score_s2`.

## 6) Normalization impact

From `normalization_effect_xgboost_weighted.csv` and aggregate file:

- normalization does **not** reliably improve transfer in this pass.
- context hold-out aggregate effects:
  - `full_raw -> full_plus_norm`: mean recall `-0.0185`, mean avg `-6.0209`
  - `no_screen_raw -> no_screen_plus_norm`: mean recall `+0.0115`, mean avg `-7.4295`
- seed-LOSO aggregate effects are near-zero/slightly negative for avg-improvement.

Conclusion:

- current normalization set did not solve cross-context magnitude-transfer issue.

## 7) Model-family comparison

From `model_comparison_weighted.csv` and `model_family_aggregate_full_plus_norm.csv`:

- context hold-out (full_plus_norm):
  - best mean recall: CatBoost (`0.5300`)
  - best mean precision/avg: LightGBM (`0.7070` precision, `18.7181` avg)
  - best mean best-improvement: CatBoost (`32.9370`)
- seed-LOSO (full_plus_norm):
  - XGBoost / LightGBM / CatBoost / RandomForest are all competitive;
  - XGBoost has the highest mean recall (`0.5309`) in this view.
- DecisionTree is generally weaker and less stable.

Takeaway:

- boosted trees are materially stronger than simple trees and generally stronger than random-forest for ranking quality.

## 8) Per-context hold-out behavior

From `context_holdout_selected_deltas_by_context.csv`:

- contexts 1/2/3: most learned variants improve recall and often precision vs `screen_s2`.
- context 4 remains hard; many learned variants lose magnitude metrics heavily at small/medium k.

This confirms that mixed transfer is largely context-specific, not purely random noise.

## What this pass can claim (dev-only)

- learned move ranking signal remains real after stronger controls.
- no-screen ablation still shows meaningful ranking signal above random and often above handcrafted recall.
- boosted-tree families are preferable over simpler tree baselines.

## What this pass cannot claim

- benchmark generalization
- final paper-level train/test protocol validity
- readiness for online solver integration

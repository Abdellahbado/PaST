# Results

Phase Q synthetic-only offline move-ranking training/evaluation completed successfully.

## Data and protocol used

- Train file: `temp/phaseP_full_synthetic_freeze/synthetic_moves_exact_labeled_train_frozen.csv.gz`
- Val file: `temp/phaseP_full_synthetic_freeze/synthetic_moves_exact_labeled_val_frozen.csv`
- Rows: train `263356`, val `53757`
- Class balance:
  - train positive rate `0.790318`
  - val positive rate `0.810797`
- Target: `target_improvement_magnitude = max(0, -exact_total_delta)`
- Ranking query grouping: `(manifest_instance_uid, seed_id, ls_round, current_tec, accepted_improving_moves_so_far, exact_eval_tier, exact_eval_cap)`
- Budgets: `k=10/25/50/100`

## Models and baselines

Models run:

- `XGBoost`: available and trained
- `LightGBM`: available and trained
- `CatBoost`: available and trained

Baselines run:

- random ordering (20-repeat average)
- handcrafted `screen_score_s1`
- handcrafted `screen_score_s2` (primary handcrafted comparator)
- oracle ordering by true target magnitude

## Main validation outcomes

At `k=10/25/50`:

- all learned models tie `screen_score_s2` on recall/precision,
- all learned models beat `screen_score_s2` on best true improvement and average true improvement.

At `k=100`:

- learned models exceed `screen_score_s2` on recall and precision,
- best true improvement ties (`134.05`),
- average true improvement is higher for learned models.

Best-model selection:

- selected default candidate: `xgboost`
- basis: highest aggregate delta vs `screen_score_s2` on recall and precision with strong average-improvement gains.

Aggregate deltas vs `screen_score_s2` (mean over `k=10/25/50/100`):

- `xgboost`: recall `+0.000912`, precision `+0.001198`, best-improvement `+0.578125`, avg-improvement `+2.596531`
- `catboost`: recall `+0.000719`, precision `+0.000969`, best-improvement `+0.553125`, avg-improvement `+2.722906`
- `lightgbm`: recall `+0.000697`, precision `+0.000979`, best-improvement `+0.597917`, avg-improvement `+2.573521`

Baseline ladder sanity:

- random is clearly lower than handcrafted and learned metrics,
- oracle remains upper bound and is slightly above learned at larger budget,
- learned models are close to oracle at `k=100` and stronger than handcrafted on magnitude metrics.

## Artifacts produced

- `research/learned_move_screening_20260420/temp/20260421_run01_xgb_lgbm_catboost/README.md`
- `research/learned_move_screening_20260420/temp/20260421_run01_xgb_lgbm_catboost/command.sh`
- `research/learned_move_screening_20260420/temp/20260421_run01_xgb_lgbm_catboost/metrics/ranking_metrics_by_model.csv`
- `research/learned_move_screening_20260420/temp/20260421_run01_xgb_lgbm_catboost/metrics/ranking_metrics_summary.csv`
- `research/learned_move_screening_20260420/temp/20260421_run01_xgb_lgbm_catboost/metrics/baseline_comparison.csv`
- `research/learned_move_screening_20260420/temp/20260421_run01_xgb_lgbm_catboost/metrics/model_comparison.csv`
- `research/learned_move_screening_20260420/temp/20260421_run01_xgb_lgbm_catboost/artifacts/training_config.json`
- `research/learned_move_screening_20260420/temp/20260421_run01_xgb_lgbm_catboost/artifacts/dataset_profile.json`
- `research/learned_move_screening_20260420/temp/20260421_run01_xgb_lgbm_catboost/artifacts/validation_scored_rows.csv`
- `research/learned_move_screening_20260420/temp/20260421_run01_xgb_lgbm_catboost/artifacts/feature_importance_xgboost.csv`
- `research/learned_move_screening_20260420/temp/20260421_run01_xgb_lgbm_catboost/artifacts/feature_importance_lightgbm.csv`
- `research/learned_move_screening_20260420/temp/20260421_run01_xgb_lgbm_catboost/artifacts/feature_importance_catboost.csv`

## Conclusion

Phase Q succeeds under the clean synthetic-only boundary:

- offline training/evaluation ran cleanly,
- required model families were executed,
- learned ranking provides a meaningful gain over the handcrafted baseline (mainly on magnitude metrics and slightly on higher-budget recall/precision),
- the branch is ready to move to benchmark test-only evaluation with `xgboost` as default candidate.

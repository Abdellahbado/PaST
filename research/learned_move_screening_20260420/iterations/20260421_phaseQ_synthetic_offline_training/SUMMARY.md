# Summary

Phase P froze the synthetic train/val exact-labeled dataset, and Phase Q used that freeze to run synthetic-only offline move-ranking training/evaluation.

What was done:

- used only Phase P frozen synthetic inputs (`synthetic_moves_exact_labeled_train_frozen.csv.gz` and `synthetic_moves_exact_labeled_val_frozen.csv`),
- used no benchmark data for train/tune/validation,
- implemented `scripts/phaseQ_synthetic_offline_training.py` for reproducible offline training/evaluation,
- trained required tabular families (`XGBoost`, `LightGBM`, `CatBoost`) on frozen synthetic train,
- evaluated required baselines (random, handcrafted `screen_score_s1/s2`, oracle) on frozen synthetic val,
- reported fixed-budget ranking metrics at `k=10/25/50/100`, model-vs-baseline deltas, scored validation rows, and feature importances.

Main result:

- all required learned models ran and produced consistent ranking outcomes,
- learned models beat the primary handcrafted baseline (`screen_score_s2`) on top-k improvement-magnitude metrics across budgets,
- at `k=100`, learned models also improve recall and precision over handcrafted,
- bounded run used: `research/learned_move_screening_20260420/temp/20260421_run01_xgb_lgbm_catboost/`,
- best model from that run: `xgboost`.

Decision:

- synthetic-only offline stage is positive and protocol-clean,
- result strength is sufficient to justify benchmark test-only evaluation next,
- exact next step: run benchmark offline test-only evaluation using the frozen Phase Q default candidate (`xgboost`) on primary `61-90` then secondary `1-60`, with no benchmark retraining/tuning.

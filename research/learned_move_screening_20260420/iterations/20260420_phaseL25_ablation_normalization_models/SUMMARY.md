# Summary

Stage L2.5 performed robustness checks for development-only move ranking.

Completed:

- fixed Stage L2 helper-summary inconsistency and aligned reporting artifacts
- added stronger baseline ladder (random, handcrafted, oracle)
- ran required ablations (full, no-screen, screen-only)
- added normalized ratio features and tested transfer impact
- compared tabular models (XGBoost, LightGBM, CatBoost, RandomForest, DecisionTree)

Main findings:

- learned ranking signal remains positive on dev data
- no-screen features still carry strong signal (not just handcrafted-score wrapping)
- normalization did not reliably improve context-holdout magnitude transfer
- boosted trees remain the most promising family

Decision:

- move ranking remains the right learning position,
- but next step must be data-protocol strengthening (generated non-benchmark corpus + strict family-level split) before any solver integration.

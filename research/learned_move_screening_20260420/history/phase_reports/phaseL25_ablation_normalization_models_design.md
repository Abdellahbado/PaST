# Phase L2.5 Ablation, Normalization, and Model Comparison Design

Date: 2026-04-20

## What Stage L2 proved

- On development data, learned ranking gave a clear positive signal on seed-aware within-context splits.
- Context hold-out behavior was mixed: recall/precision gains, but improvement-magnitude gains were unstable.

## What remained unclear

1. whether reported L2 summaries were internally consistent
2. whether gains depended heavily on handcrafted score features
3. whether simple normalization could improve context transfer
4. whether XGBoost was uniquely strong or similar to other tabular models

## Why `screen_score_s1/s2` ablation is necessary

- Stage L2 feature-importance showed heavy reliance on `screen_score_s2`.
- We need to test if structural/raw features alone still carry ranking signal.
- We also need a compact handcrafted-only feature set to detect “model just wraps heuristic score” behavior.

## Why normalization is necessary

- Contexts differ in epsilon and cost scale.
- Raw absolute-value features can hurt cross-context transfer.
- Ratio features may stabilize ranking semantics across contexts.

## Model families compared

- XGBoost
- LightGBM
- CatBoost
- RandomForest
- DecisionTree

Tuning policy: light and comparable, no model-zoo optimization.

## Exact offline baselines

- Baseline A: random ordering
- Baseline B: handcrafted analytical ordering (`screen_score_s2`, plus `screen_score_s1` reference)
- Baseline C: oracle ordering by true target (`target_improvement_magnitude`)
- Baseline D: learned models above

## Feature-set conditions

1. `full_raw`: all prior numeric features including `screen_score_s1/s2`
2. `no_screen_raw`: remove `screen_score_s1/s2`
3. `screen_only`: compact handcrafted-only (`screen_score_s2`, `screen_score_s1`)
4. normalization variants:
   - `full_plus_norm`
   - `no_screen_plus_norm`

## Normalized features added

- `source_exact_cost_over_epsilon`
- `target_exact_cost_over_epsilon`
- `source_gap_ratio`
- `target_gap_ratio`
- `job_processing_time_over_epsilon`
- `job_processing_time_over_source_load`
- `job_processing_time_over_target_post_slack`
- `projected_target_load_after_over_epsilon`
- `projected_source_load_after_over_epsilon`
- `current_tec_over_epsilon`
- `current_tec_minus_start_over_epsilon`

## Splits and metrics

- splits:
  - seed-aware LOSO within context
  - context hold-out
- fixed budgets: `k = 10, 25, 50, 100`
- metrics:
  - recall@k
  - precision@k
  - best true improvement in top-k
  - average true improvement in top-k
- reporting:
  - weighted + unweighted summaries
  - per-context tables
  - fold spread where available

## Continuation criterion after this pass

Continue toward generated-instance corpus design only if:

1. learned ranking remains meaningfully stronger than handcrafted baseline on dev metrics,
2. no-screen feature sets still show nontrivial signal,
3. transfer behavior is at least stable (or improved) under stronger controls.

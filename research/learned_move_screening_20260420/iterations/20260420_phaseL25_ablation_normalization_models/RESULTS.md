# Results

Phase L2.5 completed with consistency reconciliation and expanded offline comparisons.

## L2 consistency fix

- Root cause: helper JSON summary mixed aggregation semantics.
- Canonical L2 CSV summaries were already correct.
- Corrected helper outputs now align with weighted/unweighted tables.

## Baselines and ablations

- Baseline ladder (random < handcrafted < oracle) validated.
- Learned models still improve recall over handcrafted in many settings.
- No-screen ablation remains strong, showing genuine structural signal beyond handcrafted scores.
- Screen-only learned variants underperform richer sets, arguing against pure score-wrapping.

## Normalization

- Added ratio features did not reliably improve context-holdout magnitude behavior.
- Cross-context transfer remains mixed; context 4 remains difficult.

## Model comparison

- Boosted trees (CatBoost/LightGBM/XGBoost) outperform simpler trees overall.
- Best model depends on metric and split; no single universal winner.

## Decision

- Move ranking remains a valid learning position for dev progression.
- Next step should be protocol-level data redesign (generated non-benchmark corpus + strict family-level splits) before any online integration attempt.

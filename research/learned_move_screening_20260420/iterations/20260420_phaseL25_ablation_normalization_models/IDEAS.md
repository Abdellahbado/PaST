# Ideas

## Consistency-first

- Recompute L2 helper summaries directly from fold-level canonical CSV tables.
- Keep weighted/unweighted views explicit and separate.

## Stronger baseline ladder

- Add random and oracle ranking references around handcrafted and learned baselines.

## Signal-isolation ablations

- Run three key feature-set conditions:
  - full
  - no-screen
  - screen-only

## Scale transfer probe

- Add compact ratio/normalization features tied to epsilon, load, and cost gaps.
- Compare raw vs normalized feature sets, especially on context hold-out.

## Model dependence check

- Compare a small tabular set:
  - XGBoost, LightGBM, CatBoost, RandomForest, DecisionTree

# Phase L2.5 Ablation, Normalization, and Model Comparison Readiness

Date: 2026-04-20

## Success criteria check

1. Stage L2 inconsistency reconciled and reporting corrected

- Pass.
- Helper summary artifact corrected and aligned with canonical weighted/unweighted result tables.

2. Stronger baselines + ablations executed

- Pass.
- Random / handcrafted (`s1`,`s2`) / oracle / learned baselines all included.
- Required feature-set ablations completed:
  - full
  - no-screen
  - screen-only

3. Normalization pass executed and assessed

- Pass (negative/mixed outcome).
- Added ratio features and compared raw vs normalized sets.
- No reliable cross-context magnitude improvement was observed.

4. Multi-model tabular comparison completed

- Pass.
- Compared XGBoost, LightGBM, CatBoost, RandomForest, DecisionTree.

## Decision

- Move ranking remains the right learning position for dev progression.
- Signal is genuine (not purely handcrafted-score wrapping), but transfer fragility remains.
- Ready to proceed to next protocol stage only with strict data redesign:
  - generated-instance training corpus
  - family-level train/validation/test separation
  - repeat offline comparisons before any online integration attempt

## Not ready for

- benchmark-level generalization claims
- Stage L3 solver integration
- final paper-level evaluation claims

# Phase L2 Dev Ranking Probe Design

Date: 2026-04-20

## Development-data status (explicit)

- Stage L1 and L1.5 datasets are treated as development-only data.
- This stage is a feasibility probe of learnability, not a final training/evaluation protocol.
- No benchmark generalization claim is made from this stage.

## Cleanliness risks (explicit)

1. Benchmark leakage risk

- Data rows come from benchmark instances that may later appear in evaluation stories.
- Therefore, these data cannot support final external-generalization claims.

2. Selection bias risk

- Exact labels are collected from the dense-labeling policy inside the handcrafted search flow.
- The candidate distribution is policy-induced, not an unbiased sample of all feasible moves.

3. Context imbalance risk

- Context `64/79` contributes a large fraction of rows.
- Aggregate metrics can be misleading unless per-context reporting is mandatory.

4. Label-target ambiguity risk

- In principle, `label_accepted` can differ from `label_improving` due to acceptance/selection policy.
- For this dataset, they are currently equal, but we still define target semantics via exact-delta improvement rather than acceptance status.

## Stage L2 primary target

Primary target for learning:

- `target_improvement_magnitude = max(0, -exact_total_delta)`

Interpretation:

- positive value = exact-DP improving move magnitude
- zero = non-improving move

This keeps target semantics tied to oracle quality signal (`exact_total_delta`) and avoids primary dependence on acceptance-policy labels.

## Baseline ranking

Primary handcrafted baseline:

- rank candidates by `screen_score_s2` descending

Reason:

- it is the logged analytical stage-2 shortlist score used by handcrafted screening logic.

## Dataset cleanup / restructuring policy

Input:

- `temp/phaseL15_dense_labeling/moves_exact_labeled_aggregate.csv`

Modeling table output:

- `temp/phaseL2_dev_ranking_probe/modeling_dataset_dev.csv`

One row per exact-labeled move with preserved:

- `context_id`
- `seed_id`
- search-state identifiers (`ls_round`, `current_tec`, `accepted_improving_moves_so_far`, `exact_eval_tier`, `exact_eval_cap`)

Cleanup checks:

- keep only `epsilon_feasible == 1`
- enforce finite exact labels
- remove duplicate `record_id` if any
- require ranking-query groups with at least 2 rows

Query grouping for ranking metrics:

- `query_id = (context_id, seed_id, ls_round, current_tec, accepted_improving_moves_so_far, exact_eval_tier, exact_eval_cap)`

Metadata output:

- `temp/phaseL2_dev_ranking_probe/dataset_cleanup_protocol.json`

## Split protocol (leakage-aware, dev-only)

Mandatory split families:

1. Seed-aware split (within-context LOSO)

- leave one seed out inside each context
- train on remaining seeds from same context
- test on held-out seed

Manifest:

- `temp/phaseL2_dev_ranking_probe/split_manifest_seed.csv`

2. Context hold-out split

- hold out one full context for test
- train on remaining contexts

Manifest:

- `temp/phaseL2_dev_ranking_probe/split_manifest_context.csv`

## Model family

- gradient-boosted trees only
- first model: XGBoost regressor on `target_improvement_magnitude`
- no model zoo, no neural architectures, no RL, no integration into solver

## Required metrics

At budgets `k = 10, 25, 50, 100`, report:

- recall@k of improving moves
- precision@k
- best exact improvement found within top-k
- average exact improvement within top-k

Comparison policy:

- learned ranker vs handcrafted `screen_score_s2` at same k
- report per-context and global tables
- include split-wise variance / fold-level deltas

## Learnability decision rule (dev-only)

Positive signal if, on dev splits:

- learned ranker beats handcrafted ranking on a meaningful share of fixed-budget metrics (especially recall@k and precision@k), and
- improvements are not purely from one trivial fold.

Negative signal if:

- learned ranker fails to provide meaningful gains over handcrafted baseline on these dev splits.

## If Stage L2 is positive: immediate next step

Do not claim benchmark generalization.

Next step is to build a cleaner protocol:

1. generate a dedicated non-benchmark training corpus (generated/synthetic instance family)
2. define strict final train/validation/test split by instance families and seeds
3. repeat offline ranking with leakage-safe protocol
4. only then move to Stage L3 online integration tests

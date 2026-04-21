# Ideas

## Primary implementation idea

- Add a dense labeling mode (`vnd_exact_dp_insert_rank_dense_labeling`) that keeps insert-focused exact-DP acceptance but evaluates many more feasible insert candidates per seed/round.

## Diversity idea

- Collect from three context families:
  - anchor (`61/347`)
  - nearby epsilons (`61/346`, `61/345`)
  - one additional different instance (`64/79`)

## Schema idea

- Preserve Stage L1 feature groups and add only a few cheap generalization features:
  - `context_id`
  - source exact-cost rank (num/den)
  - target slack rank (num/den)
  - epsilon stress proxy

## Logging outputs

- aggregate broad and exact-labeled files
- per-seed context summary table
- dense metadata summary JSON

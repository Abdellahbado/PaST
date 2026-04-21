# Phase L1.5 Dense Labeling Design

Date: 2026-04-20

## Why Stage L1 is insufficient

Stage L1 proved schema correctness, but not data sufficiency for meaningful offline ranking conclusions.

- Stage L1 exact-labeled size: `112`
- Stage L1 positives: `27`
- single anchor context (`61/347`) only

This is too small and too narrow to support robust split/ablation comparisons.

## How Stage L1.5 increases label volume

- Introduce dense exact-labeling mode: `vnd_exact_dp_insert_rank_dense_labeling`.
- Raise exact-label budget per seed by widening:
  - source coverage
  - per-source retained candidates
  - shortlist cap
  - exact-eval cap per round
  - LS rounds/time budget
- Keep exact DP touched-machine labeling and acceptance semantics unchanged.

## How Stage L1.5 increases diversity

Use multiple controlled contexts rather than one trajectory.

Required and implemented:

1. Context A (anchor hard point)
   - `instance 61`, `epsilon 347`
2. Context B (nearby hard points)
   - `instance 61`, `epsilon 346`
   - `instance 61`, `epsilon 345`
3. Context C (different instance family)
   - selected: `instance 64`, `epsilon 79`
   - reason: structurally different machine/job profile than 61 and already known to be feasible in repository experiments.

## Rows / seeds / contexts included

- `61/347`: 10 deterministic seeds
- `61/346`: 8 deterministic seeds
- `61/345`: 8 deterministic seeds
- `64/79`: 8 deterministic seeds

Total planned seed runs: `34`.

## Dense exact-label collection policy

- Base pipeline remains insert-focused exact-DP local search.
- For every generated `insert_inter` candidate:
  - log broad cheap-feature row
  - if feasible and selected in dense shortlist, exact-evaluate and log exact labels
- Dense mode uses best-improving acceptance within round after evaluating larger candidate subsets.
- Keep two-stream data output:
  - broad stream: `moves_broad_aggregate.csv`
  - exact stream: `moves_exact_labeled_aggregate.csv`

## Minimum dataset target before Stage L2

Stage L2 should not start unless all hold:

- exact-labeled records >= `8,000`
- improving positives >= `1,500`
- at least `4` contexts (instance/epsilon points)
- at least `25` seed runs total

## Risks that remain

- selection bias: dense labels still come from heuristic-generated candidate pipeline, not exhaustive all-move exact labels
- class imbalance can still vary by context/seed
- runtime/memory costs are substantial for dense exact-labeling
- cross-context coverage unevenness (rate-class combinations differ between instances)

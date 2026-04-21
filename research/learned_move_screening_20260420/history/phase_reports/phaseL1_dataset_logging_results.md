# Phase L1 Dataset Logging Results

Date: 2026-04-20

## Scope executed

- Stage L1 only (dataset logging/schema build).
- Anchor only: `instance 61`, `epsilon 347`.
- No model training, no ML inference integration.

## Data source and run policy

- Source family: `vnd_exact_dp_insert_rank_diverse_trimmed`.
- Logging wrapper variant executed: `stageL1_dataset_logging`.
- Multistart seeds used: `12` deterministic seeds.

## Produced artifacts

Under `temp/phaseL1_dataset_logging/`:

- `moves_broad_61_347.csv`
- `moves_exact_labeled_61_347.csv`
- `dataset_summary_61_347.json`
- `feature_dictionary.md`
- `run_61_347_stageL1_dataset_logging.csv`
- `run_61_347_stageL1_dataset_logging.time.txt`

## Required Stage L1 metrics

- seeds used: `12`
- broad candidate records: `144,504`
- exact-labeled records: `112`
- positive improving labels: `27`
- exact positive rate: `24.11%`
- runtime (`/usr/bin/time -l` wall): `368.51 s`
- max RSS (`/usr/bin/time -l`): `2,215,297,024` bytes

## Data quality observations

- Broad stream captures generated candidates with explicit feasibility flag (`epsilon_feasible`).
- Exact stream is cleanly separated and contains exact touched-machine labels.
- Seed-level variability exists (one seed produced no exact-labeled rows), but aggregate positive count is adequate.
- No obvious duplicate groups in exact-labeled records under a practical seed/round/move key check.

## Sufficiency call for Stage L2

- Dataset appears sufficient for first offline ranking experiments at anchor `61/347`.
- Positives are not too sparse for initial tree-based ranking baselines, but class imbalance still requires ranking-aware evaluation (top-k recall/hit-rate focus).

# Results

Stage L1 dataset logging completed at anchor `instance 61`, `epsilon 347`.

Data source used:

- handcrafted base family: `vnd_exact_dp_insert_rank_diverse_trimmed`
- logging wrapper variant: `stageL1_dataset_logging`

Collection summary:

- seeds configured: `12`
- broad candidate records: `144,504`
- exact-labeled records: `112`
- positive improving labels: `27`
- positive rate among exact-labeled: `24.11%`

Runtime/resources:

- wall time (`/usr/bin/time -l`): `368.51 s`
- max RSS: `2,215,297,024` bytes

Dataset quality notes:

- exact-label class balance is acceptable for first offline ranking (`27` positives over `112` exact labels).
- exact record key collisions were not observed under a seed/round/move key check.
- broad stream includes both feasible and infeasible generated candidates (`epsilon_feasible` flag preserved).

Main artifacts:

- `temp/phaseL1_dataset_logging/moves_broad_61_347.csv`
- `temp/phaseL1_dataset_logging/moves_exact_labeled_61_347.csv`
- `temp/phaseL1_dataset_logging/dataset_summary_61_347.json`
- `temp/phaseL1_dataset_logging/feature_dictionary.md`

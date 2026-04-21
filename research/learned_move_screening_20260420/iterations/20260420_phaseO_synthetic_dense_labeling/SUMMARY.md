# Summary

Phase O addressed the Phase N data-quality blocker by replacing the synthetic labeling policy.

What changed:

- synthetic extraction path switched from `stageL1_dataset_logging` to dense wrapper `stageO_synthetic_dense_logging`,
- dense exact move evaluation reused `vnd_exact_dp_insert_rank_dense_labeling`,
- manifest-driven runner added: `scripts/phaseO_synthetic_dense_labeling.py`.

Bounded run outcome (`12 train + 4 val`):

- exact-labeled rows: 28669
- positives: 20292
- negatives: 8377
- positive/negative rates: 0.7078 / 0.2922
- manifest gating: preserved (train/val only)

Decision:

- strict mixed-sign gate passes; branch succeeds at bounded scale.
- next step is controlled full-manifest dense labeling with batch-level class-balance monitoring before freezing offline-learning input.

Thread note:

- root markdown surface was cleaned on 2026-04-21; future re-entry should start from `research/learned_move_screening_20260420/START_HERE.md`

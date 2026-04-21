# Problem

Phase N is the first execution branch on top of Phase M protocol setup.

Objective: validate that synthetic-only exact-label extraction is executable end-to-end using only Phase M train/val manifests, with strict manifest gating and reproducible outputs.

Constraints:

- use only `split_manifest_train.csv` and `split_manifest_val.csv` for labeling,
- do not use benchmark test manifests for training/labeling,
- do not run solver integration or final model training,
- keep the workload bounded for sanity validation (not full-corpus production scale).

# Problem

Phase N validated synthetic manifest plumbing but failed to produce learning-usable labels because the extracted dataset was 100% positive.

Objective for Phase O:

- replace the Stage L1-screened synthetic extraction path with a dense exact-labeling policy,
- keep strict synthetic-only manifest gating (`split_manifest_train.csv` and `split_manifest_val.csv`),
- run bounded sanity first and require mixed-sign labels (`label_improving` includes both `1` and `0`).

Hard constraints:

- no benchmark manifests for training/labeling,
- no model training,
- no solver integration,
- no broad method-family expansion beyond synthetic exact-label policy repair.

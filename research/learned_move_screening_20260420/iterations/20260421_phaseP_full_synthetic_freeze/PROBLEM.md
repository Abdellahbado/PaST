# Problem

Phase O resolved the one-sided-label blocker at bounded scale, but the branch is not training-ready until full synthetic train/val manifests are labeled under the same dense policy with explicit skew diagnostics.

Objective for Phase P:

- run full-manifest dense exact labeling on synthetic train/val only,
- preserve manifest-gated protocol and Phase O labeling policy,
- diagnose train/val class-balance skew at global, split, and `(M,N,K)` levels,
- freeze one reproducible synthetic exact-labeled dataset for the next offline learning stage.

Hard constraints:

- use only `split_manifest_train.csv` and `split_manifest_val.csv`,
- do not consume benchmark test manifests,
- do not train models,
- do not change solver integration,
- do not change labeling policy unless a blocker forces it.

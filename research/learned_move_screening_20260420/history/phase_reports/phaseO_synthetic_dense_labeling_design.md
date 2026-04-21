# Phase O Synthetic Dense Labeling Design

Date: 2026-04-20

## 1) Why Phase N is not training-ready

Phase N output is not learning-usable because it is one-sided:

- `label_improving = 1` for all rows (`192 / 192`),
- no non-improving counterexamples,
- cannot support meaningful ranking/classification discrimination.

Therefore, Phase N is treated as plumbing validation only, not as a training-data branch.

## 2) Why Stage L1-screened path is not acceptable for synthetic learning data

Phase N depended on `stageL1_dataset_logging`, which logs exact outcomes only on a screened candidate path (`vnd_exact_dp_insert_rank_diverse_trimmed`) designed to prioritize promising moves. On synthetic train/val this induces strong positive-selection bias and suppresses negatives.

For learning data, this is unacceptable because it under-represents non-improving moves and distorts label priors.

## 3) New labeling policy used in Phase O

Replace Phase N extraction path with a dense exact-label path:

- C++ wrapper variant: `stageO_synthetic_dense_logging`
- core move policy: `vnd_exact_dp_insert_rank_dense_labeling`
- per-instance output files (no overwrite collisions):
  - `moves_broad_instance_<id>_eps_<e>.csv`
  - `moves_exact_labeled_instance_<id>_eps_<e>.csv`

This keeps exact-label semantics while expanding evaluated candidate coverage beyond trimmed-screen logging.

## 4) How negative examples are encouraged

Negatives are encouraged by policy and budget:

- dense insert-inter exact evaluation mode (higher candidate and exact-eval caps than Stage N path),
- multiple randomized starts per instance (`seeds=4`),
- bounded but non-trivial local rounds/time (`ls_max_rounds=2`, `ls_time_cap_sec=2.0`),
- lower epsilon slack than Phase N (`epsilon_slack=8`) to avoid over-relaxed easy-improvement regimes.

No hard positivity guarantee is assumed; success is gated empirically by mixed-sign labels.

## 5) Manifests consumed

Only Phase M synthetic train/val manifests are allowed for labeling:

- `temp/phaseM_vls_synthetic_protocol/split_manifest_train.csv`
- `temp/phaseM_vls_synthetic_protocol/split_manifest_val.csv`

Explicitly not used for labeling:

- `split_manifest_test_primary_vls.csv`
- `split_manifest_test_secondary_legacy.csv`

## 6) First bounded workload

Run bounded manifest-driven subset first:

- train sample: 12 instances
- val sample: 4 instances
- sampling: round-robin over `(M,N,K)` buckets after seeded shuffle
- epsilon policy: `epsilon=min(K, max(ceil(sum(p)/M), max(p)) + epsilon_slack)`

## 7) Exact success criteria before scale-up

Phase O bounded run is pass only if all hold:

1. non-empty exact-labeled output,
2. mixed-sign labels (`label_improving` includes both `1` and `0`),
3. manifest gating preserved (`train/val` only),
4. no schema/loader breakage,
5. bounded runtime/memory remains manageable.

Additional quality gate:

- if class distribution is still extreme (for example >95/5), do not declare scale readiness; require another policy adjustment first.

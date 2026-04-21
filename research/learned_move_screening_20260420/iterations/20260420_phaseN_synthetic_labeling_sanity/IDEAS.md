# Ideas

## Chosen path

- Implement a manifest-driven Python runner that:
  - reads only Phase M train/val manifests,
  - samples a bounded stratified subset across `(M,N,K)` buckets,
  - computes feasible epsilon targets from synthetic `Data_p` profiles,
  - invokes existing C++ exact-label path (`stageL1_dataset_logging`) per selected instance,
  - captures run metadata, memory/runtime, and labeled-move exports.

## Reuse strategy

- Reuse existing solver/data-logging implementation in `solvers/cpp/parallel_heuristic_compare.cpp`.
- Avoid C++ solver edits for this sanity branch; keep change-surface minimal and reproducible.

## Sanity workload policy

- Bounded workload:
  - train subset: 12 instances,
  - val subset: 4 instances,
  - round-robin stratified selection over family buckets.

## Success signal

- non-empty exact-labeled outputs,
- non-zero improving positives,
- no manifest leakage,
- stable schema and reasonable runtime/RSS envelope.

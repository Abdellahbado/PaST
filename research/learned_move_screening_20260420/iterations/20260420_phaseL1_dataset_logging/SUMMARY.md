# Summary

Stage L1 (dataset logging) is implemented and executed at anchor `61/347`.

What was added:

- logging instrumentation in `solvers/cpp/parallel_heuristic_compare.cpp` for `insert_inter`
- dedicated logging variant: `stageL1_dataset_logging`
- two-stream export:
  - broad candidate stream
  - exact-labeled stream
- metadata + feature dictionary artifacts under `temp/phaseL1_dataset_logging/`

Key numbers:

- seeds: `12`
- broad records: `144,504`
- exact-labeled: `112`
- improving positives: `27` (`24.11%`)

Conclusion:

- Dataset is sufficient for Stage L2 offline ranking at the anchor point.
- Exact label sparsity is present but acceptable for first tree-based ranking experiments.

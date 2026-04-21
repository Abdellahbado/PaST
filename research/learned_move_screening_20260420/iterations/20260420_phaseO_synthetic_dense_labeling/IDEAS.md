# Ideas

1. Reuse prior dense exact-labeling internals (`vnd_exact_dp_insert_rank_dense_labeling`) instead of `stageL1_dataset_logging`.
2. Add a synthetic-oriented wrapper variant in C++ that emits per-instance dense broad/exact CSV files to avoid file overwrite collisions.
3. Keep Phase M manifest plumbing and sampling, but lower epsilon slack and increase exact move budget to expose non-improving labels.
4. Enrich merged outputs with manifest context and epsilon metadata (`epsilon_lb`, `epsilon_used`) for reproducible downstream offline learning.
5. Gate readiness on mixed-sign labels and non-extreme skew, not just non-empty execution.

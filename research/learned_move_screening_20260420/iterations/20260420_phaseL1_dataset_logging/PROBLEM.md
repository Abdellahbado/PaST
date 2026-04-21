# Problem

Stage L1 must build a reliable move-level supervised-learning dataset from the validated insert-focused exact-DP heuristic without adding ML decisions yet.

At anchor `61/347`, we need two compatible streams:

- broad generated-candidate logs for `insert_inter`
- exact-labeled move logs for exact-evaluated candidates

The dataset must preserve the solver's real decision context (features at decision time, exact labels from touched-machine DP), and include multistart diversity (10-20 seeds).

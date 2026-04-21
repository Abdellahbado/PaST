# Summary

This iteration starts the learning-based continuation of the insert-focused exact-DP heuristic.

Inherited conclusion from the predecessor thread:

- the handcrafted method is real but saturating
- the exact bottleneck is move selection under a small exact-DP budget

Planned direction:

- log move-level data from the handcrafted base
- train a supervised move-ranking model for `insert_inter`
- integrate learned ranking while keeping exact DP as final verifier

Why this split was made:

- the objective is now method novelty through learning, not further handcrafted heuristic tuning
- predecessor results remain preserved in:
  - `research/dp_heuristic_oracle_20260419/`

Next concrete step:

- implement Stage L1 data logging and dataset schema

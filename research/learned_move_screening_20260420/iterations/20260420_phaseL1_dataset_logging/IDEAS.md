# Ideas

## Chosen implementation idea

- Keep source heuristic core fixed (`vnd_exact_dp_insert_rank_diverse_trimmed`), add a logging-only wrapper variant (`stageL1_dataset_logging`) for deterministic multistart collection and artifact export.

## Data-stream design

- Dataset A (broad): log every generated `insert_inter` candidate, including infeasible ones, with cheap features and state context.
- Dataset B (exact): log only exact-evaluated candidates with exact touched-machine deltas and binary/continuous labels.

## Seed policy

- Use 12 deterministic seeds at the same anchor point (`61/347`) to diversify incumbents while preserving fixed benchmark conditions.

## Guardrails

- No model training.
- No inference path in solver.
- No expansion beyond anchor point in Stage L1.

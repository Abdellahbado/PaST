# Blockers

## Active blockers

- No hard blocker for Phase P objective. Full synthetic train/val labeling succeeded with complete manifest coverage.

## Resolved concerns

1. Full-manifest runtime viability: resolved (all 180 instances completed; no retries required).
2. Mixed-sign preservation at scale: resolved (65392 negatives retained globally).
3. Split skew explainability: resolved enough for freeze (gap is modest and mostly within-bucket, not composition mismatch).

## Remaining risks (non-blocking)

1. Positive class remains dominant (~79%), so offline training should keep imbalance-aware metrics and loss handling.
2. A small subset of buckets show val rates outside train 2-sigma envelope, indicating local variance that should be monitored during model validation.

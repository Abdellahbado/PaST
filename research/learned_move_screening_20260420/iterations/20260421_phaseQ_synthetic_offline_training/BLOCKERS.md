# Blockers

## Active blockers

- No hard blocker for Phase Q objective.

## Notes / residual risks (non-blocking)

1. Validation is still positive-dominant, so recall/precision can saturate at smaller budgets; magnitude metrics remain important to distinguish learned vs handcrafted ranking.
2. Per-bucket validation diagnostics are inherently noisy because validation has one instance per `(M,N,K)` bucket.
3. Learned-model separation vs handcrafted is modest on recall/precision and stronger on magnitude; this should be rechecked on benchmark test-only manifests next.

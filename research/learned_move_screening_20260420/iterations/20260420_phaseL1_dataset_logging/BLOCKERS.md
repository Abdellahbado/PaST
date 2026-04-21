# Blockers

## Active blockers

- No hard blocker for Stage L1 completion.

## Observed constraints to carry forward

- Exact-labeled stream is much smaller than broad stream by design; Stage L2 must use ranking-aware evaluation and careful split policy.
- One configured seed produced no exact-labeled rows; seed-level variability should be expected and handled in split strategy.
- Runtime and memory are non-trivial for collection even at one anchor row; widening to more rows should be staged.

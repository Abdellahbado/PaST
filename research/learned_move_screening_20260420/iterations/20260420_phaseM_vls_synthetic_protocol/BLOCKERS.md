# Blockers

## Active blockers

- No hard blocker for protocol generation and manifest construction.

## Carried risks

1. Synthetic sampling may miss higher-order benchmark correlations.
2. Exact-label extraction cost can become heavy at larger corpus scales.
3. Leakage risk remains if downstream runners bypass split manifests.

## Mitigation direction

- Enforce manifest-gated runners in next stage and track run metadata.
- Start with bounded sanity execution before full-corpus scaling.

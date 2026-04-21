# Ideas

## Chosen design

- Generate synthetic instances that match benchmark `61-90` structural family exactly on `(M,N,K)` support and closely on value distributions.
- Emit benchmark-compatible `Data_p/e/c` files so existing loaders and solver entrypoints can be reused with minimal change.
- Freeze role-separated manifests as protocol gates (`train`, `val`, `test_primary_vls`, `test_secondary_legacy`).

## Implementation notes

- Deterministic seeding (`root_seed`, `split_seed`) to support full reproducibility.
- Balanced per-combination sampling for pilot coverage across all 30 `(M,N,K)` combinations.
- Catalog and comparison exports to explicitly verify generated-vs-benchmark alignment and prevent silent drift.

## Out-of-scope for this branch

- No benchmark-driven model tuning.
- No solver online integration.
- No alternative ML task broadening.

# Phase M VLS Synthetic Protocol Readiness

Date: 2026-04-20

## Readiness criteria check

1. Synthetic-only train/validation protocol enforced

- Pass.
- Train and validation manifests include generated instances only.

2. Primary and secondary benchmark roles separated

- Pass.
- Primary test manifest is benchmark 61-90 only.
- Secondary robustness manifest is benchmark 1-60 only.

3. VLS synthetic family matches intended benchmark structure

- Pass.
- M/N/K support exactly matches 61-90 family.
- p/e/c support and first-order distribution summaries are close.

4. Reproducibility and artifact completeness

- Pass.
- Deterministic seeding scheme, generation config, catalogs, summaries, and split manifests are all produced.

## Strict decision

- Ready to proceed to the next stage:
  - synthetic-only labeling and offline ranking train/validation evaluation.

## Not yet ready for

- solver integration,
- benchmark-level performance claims,
- any benchmark-driven model tuning loop.

## Remaining risks to carry forward

1. higher-order realism gap between synthetic and benchmark instances,
2. labeling compute cost at larger corpus sizes,
3. accidental split-policy violations if downstream runners bypass manifests.

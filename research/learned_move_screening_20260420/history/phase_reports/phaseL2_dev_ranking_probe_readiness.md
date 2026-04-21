# Phase L2 Dev Ranking Probe Readiness

Date: 2026-04-20

## Success criteria check

Reporting consistency note:

- L2.5 identified and fixed a helper-summary aggregation inconsistency in L2 reporting JSON.
- Canonical weighted/unweighted CSV summaries were unchanged; readiness interpretation remains the same.

1. Cleaned dev dataset and split protocol are explicit and honest

- Pass.
- Modeling table, cleanup metadata, and split manifests are exported and reproducible.
- Development-only status and cleanliness risks are explicitly documented.

2. Learned ranker beats handcrafted baseline on at least part of fixed-budget metrics

- Pass (with caveat).
- On seed-aware LOSO within context, XGBoost consistently improves recall/precision and also improves best/average top-k improvement metrics.
- On context hold-out, XGBoost improves recall strongly but underperforms handcrafted baseline on top-k improvement magnitude metrics in aggregate.

3. Result is strong enough to justify next step

- Pass (for protocol progression, not for deployment claims).
- Signal is sufficient to justify building a cleaner generated-instance training corpus and stricter final evaluation protocol.

## Caveats to carry forward

- Data is benchmark-derived and policy-sampled; no final generalization inference allowed.
- Context imbalance remains material (context 4 dominates rows and behaves differently).
- Magnitude ranking across held-out contexts is not yet reliably better than handcrafted ordering.

## Readiness decision

- Ready to proceed to the next protocol stage:
  - build non-benchmark generated training corpus
  - define strict family-level train/validation/test isolation
  - rerun offline ranking with the same fixed-budget metrics
- Not ready for solver-side ML integration (Stage L3) yet.

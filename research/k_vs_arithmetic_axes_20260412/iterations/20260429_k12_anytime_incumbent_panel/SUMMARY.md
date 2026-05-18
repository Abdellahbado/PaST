# Summary

## Status: Decision A

K12 recovery achieved under original single-machine model. PLAN32B parallel UB was invalid (model mismatch). Serial `compute_initial_ub` finds valid UBs for both no-incumbent seeds.

## PLAN32B invalidation

- Benchmark has M=1 (single-machine, no machine count in instances)
- PLAN32B used M=2 (parallel-machine partition)
- UB < single-machine LB for both seeds → model mismatch confirmed
- Parallel UB gated behind `PAST_ANYTIME_PARALLEL_UB_OPT_IN=1` (disabled)

## PLAN32C validity guard

- LB-consistency check at `done:`: UB < LB rejects incumbent
- 3 new CSV fields: `initial_ub_lb_consistent`, `initial_ub_rejected_reason`, `initial_ub_model_note`

## K12 recovery (original model)

Both previously unrecoverable seeds now have finite UB with <0.1% gap:

| Seed | UB | LB (semigroup) | Gap |
|------|-----|----------------|------|
| hardA_k12 s3 | 133,544,950 | 133,481,433 | 0.048% |
| hardB_k12 s3 | 185,849,400 | 185,744,893 | 0.056% |

Method: serial `compute_initial_ub` (SPT/LPT/alternating/random) finds valid single-machine schedule. Semigroup LB from forward DP validates quality.

## Decision
**A**: K12 recovery achieved under original model. Gaps ≤ 2%. Promote.

## PLAN33 — Certified Anytime Hard-K Prepass — Decision A (K10 + K12)

Phase A+B verified (24 rows). All 12 plan33 rows cert_stop=1, all gaps ≤ 0.0593%, all UB ≥ LB. PLAN33 avg 1396.61s (130.49s faster than PLAN32C 1527.11s) with certified semigroup LB. Polish improved UB in all rows. hardA_k12 s3 corrected from 159M (stale) to 133.5M; hardB_k12 s3 also updated. PLAN33 is the recommended hard-K default for tested K10/K12 hard rows.

## Remaining
- Medium K12 families (k12_dense_no1, k12_even_structured, k12_sparse_gap) estimated, not run
- Phase C (family-aware beam) not run — beam timeout but serial portfolio suffices
- Paper/HPC cleanup started on 2026-05-03:
  `PAPER_HPC_REPRODUCIBILITY_MAP.md` now maps paper-facing results to runner
  scripts, solver functions, env toggles, and source artifacts. Final paper
  runtimes should be regenerated on HPC from that map.

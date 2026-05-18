# Results

## PLAN32B: Parallel UB — INVALIDATED

- `compute_parallel_initial_ub` solves 2-machine model; benchmark is single-machine
- UB < LB for both target seeds (hardA: 142.8M < 159.2M, hardB: 167.0M < 185.7M)
- Calibration confirmed: hardB s0 parallel=168.4M < known single-machine optimum=187.9M

## PLAN32C validity audit

- `build_instance()`: no `rates`, `parallel_machines`, or `M` field
- `stateful_compare.cpp`: parses `prices`/`jobs`/`machine` only
- `machine: "twosby"` = 5-state machine configuration, not count
- M = 1 by construction throughout the benchmark

## PLAN32C guard

- Parallel UB disabled by default (opt-in via `PAST_ANYTIME_PARALLEL_UB_OPT_IN=1`)
- LB-consistency check rejects UB < LB − 1.0
- 3 new CSV diagnostics

## PLAN32C K12 recovery — ACHIEVED

Both no-incumbent seeds recovered under original single-machine model with <0.1% gap:

| Seed | UB | LB | Gap | Method |
|------|-----|----|------|--------|
| hardA_k12 s3 | 133,544,950 | 133,481,433 | 0.048% | cert_anytime_prepass (PLAN33) |
| hardB_k12 s3 | 185,849,400 | 185,744,893 | 0.056% | cert_anytime_prepass (PLAN33) |

## PLAN33: Certified Anytime Hard-K Prepass — Decision A (K10 + K12)

Phase A+B complete (24 rows). All 12 plan33 rows cert_stop=1, all gaps ≤ 0.0593%, all UB ≥ LB.

PLAN33 avg 1396.61s vs PLAN32C 1527.11s (130.49s faster with certified semigroup LB). Polish improved UB in all 12 rows. hardA_k12 s3 corrected from 159M (stale) to 133.5M; hardB_k12 s3 also updated. PLAN33 uses 5 trials + polish; PLAN32C baseline used 75 trials. PLAN33 is the recommended hard-K default for tested K10/K12 hard rows.

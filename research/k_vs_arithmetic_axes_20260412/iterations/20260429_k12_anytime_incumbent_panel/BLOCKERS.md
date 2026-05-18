# Blockers

## Resolved: `compute_initial_ub` returns kInf for K=12
**Root cause**: The function schedules ALL jobs as a single machine sequence. For K=12 n=1000, SPT/LPT/alternating sequences were returning kInf for some seeds. **Fix**: Increased trials (PAST_ANYTIME_INITIAL_UB_TRIALS=5) + moved anytime block before forward DP. Random shuffles sometimes find a feasible ordering.

## Resolved: 2 seeds (hardA_k12 s3, hardB_k12 s3) have no incumbent
**Solution**: PLAN33 serial certified prepass finds valid single-machine sequences and validates them with the semigroup LB. Final gaps are 0.048% for hardA_k12 s3 and 0.056% for hardB_k12 s3.

## Resolved: PLAN32B parallel UB invalid
**Root cause**: `compute_parallel_initial_ub` partitions across M=2 machines, changing the model. **Fix**: Parallel UB disabled by default (opt-in via `PAST_ANYTIME_PARALLEL_UB_OPT_IN=1`). LB-consistency guard added at `done:` label.

## Resolved: Certified gap for K>=10 prepass
**Solution**: PLAN33 provides semigroup LB certification with early-stop at gap ≤ 0.1%. Phase A+B verified: all 12 plan33 rows cert_stop=1, gaps ≤ 0.0593%. PLAN33 avg 1396.61s vs PLAN32C 1527.11s (130.49s faster). PLAN33 is the recommended hard-K default for tested K10/K12 hard rows.

## Resolved: hardA_k12 s3 stale data in PLAN32C final panel
Original panel had UB=159M, LB=159M from 5-trial suboptimal run. Corrected to UB=133,544,950, LB=133,481,433 (PLAN33, 5 trials + polish). hardB_k12 s3 also updated to PLAN33 values.

## Ongoing: Medium K12 families not run
k12_dense_no1, k12_even_structured, k12_sparse_gap are still estimated. Need actual solver runs.

## Ongoing: Forward DP + beam hangs for K=12 n=1000
Beam has time checking (`PAST_PROFILE_REPAIR_BEAM_TIME_LIMIT`) but forward DP's semigroup relaxation takes 300-700s. Beam timeouts at 120-180s without improving. Acceptable — serial portfolio provides excellent UB without beam.

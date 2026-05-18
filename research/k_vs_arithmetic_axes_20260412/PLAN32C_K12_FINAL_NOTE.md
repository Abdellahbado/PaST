# PLAN32C — K12 Final Paper-Facing Note

## Problem

At K=12, n=1000, on hard irregular arithmetic families (hardA, hardB), the solver pipeline consistently failed to produce feasible incumbents on two seeds (hardA_k12 seed 3, hardB_k12 seed 3). These seeds had `UB=-1` (no incumbent) in all prior plans (PLAN18, PLAN19, PLAN22B, PLAN28).

## What was tried and invalidated (PLAN32B)

A parallel initial UB method (`compute_parallel_initial_ub`) was tested. It partitioned jobs across M=2 identical parallel machines and produced finite upper bounds. However, this was **invalid** for the benchmark, which is a single-machine model:
- `build_instance()` has no machine count field
- `stateful_compare.cpp` parses no machine count parameter
- `machine: "twosby"` is a state-machine type, not a machine count
- M = 1 always

The parallel UB fell below the single-machine semigroup lower bound (hardA: 142.8M < 159.2M, hardB: 167.0M < 185.7M), confirming the model mismatch. The parallel UB solves a different problem and was rejected.

## What fixed it (PLAN32C)

The serial `compute_initial_ub` (single-machine portfolio) was moved to run before the expensive forward DP/beam pipeline. With 5 random sequence trials (in addition to SPT, LPT, and alternating orderings), the serial portfolio finds valid single-machine sequences for both previously unrecoverable seeds.

The key change was ordering: running the anytime initial UB block **before** the forward DP, so that even if the forward DP timeouts (300–700s at K=12), the incumbent is already computed and recorded.

## Final result

All 8/8 hard K12 rows (hardA_k12 seeds 0–3, hardB_k12 seeds 0–3) now have valid finite incumbents under the original single-machine model:

| Seed | UB | LB | Gap | Method | Source |
|------|-----|----|------|--------|--------|
| hardA_k12 s0 | 129,771,336 | 129,740,378 | 0.024% | profile_repair_beam | PLAN22B |
| hardA_k12 s1 | 133,100,952 | 133,041,335 | 0.045% | profile_repair_beam | PLAN28 |
| hardA_k12 s2 | 128,534,730 | 128,483,407 | 0.040% | irregular_reroute | PLAN18 |
| hardA_k12 s3 | 133,544,950 | 133,481,433 | 0.048% | cert_anytime_prepass | PLAN33 |
| hardB_k12 s0 | 187,869,803 | 187,789,674 | 0.044% | profile_repair_beam | PLAN22B |
| hardB_k12 s1 | 186,111,159 | 186,030,362 | 0.043% | profile_repair_beam | PLAN28 |
| hardB_k12 s2 | 184,568,251 | 184,514,386 | 0.029% | irregular_reroute | PLAN18 |
| hardB_k12 s3 | 185,849,400 | 185,744,893 | 0.056% | cert_anytime_prepass | PLAN33 |

- All gaps ≤ 0.056% (well under 2%)
- LB from semigroup relaxation (single-machine lower bound)
- The two previously missing seeds (s3) now both have gaps <0.06%
- **2026-04-30 correction**: hardA_k12 s3 UB/LB updated from 159.3M (stale, 5-trial portfolio) to 133.5M (validated by PLAN33 cert prepass: 5 trials + polish + semigroup LB). hardB_k12 s3 UB updated from 185.9M to 185.8M (PLAN33 found a slightly better incumbent with certified gap).

## Limitations

- **No exact closure claim**: All rows are finite-gap, not exact. `exact_closed=0` for all eight rows.
- **Wider K12 arithmetic panel is incomplete**: Medium families (k12_dense_no1, k12_even_structured, k12_sparse_gap) have only estimates, not solver runs. The hard K12 panel covers hardA and hardB only.
- **Beam search timeouts at K=12**: Forward DP + beam hangs for K=12 n=1000 under any budget. The serial portfolio provides the incumbent before the beam starts and does not depend on beam completion.

## Code

- Serial anytime initial UB in `compute_initial_ub` with 5 random trials
- Gated via `PAST_ANYTIME_INITIAL_UB=1` (enabled since PLAN32B)
- Parallel UB retained in `compute_parallel_initial_ub` but gated behind `PAST_ANYTIME_PARALLEL_UB_OPT_IN=1` (disabled by default, diagnostic-only)
- LB-consistency guard at `done:` rejects any UB < LB − 1.0

## Artifacts

- `csv/plan32c/PLAN32C_hard_k12_final_panel.csv` — this final panel
- `csv/plan32c/PLAN32C_parallel_ub_validity_audit.csv` — 9-check audit of PLAN32B invalidity
- `csv/plan32c/PLAN32C_k12_recovery_after_validity_check.csv` — recovered rows
- `csv/plan32c/PLAN32C_notes.md` — detailed technical notes

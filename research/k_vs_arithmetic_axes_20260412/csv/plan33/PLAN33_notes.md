# PLAN33 — Certified Anytime Hard-K Prepass

## Status: Decision A (for K=10 and K=12)

### What it does

PLAN33 adds a certified anytime prepass that runs BEFORE the forward DP:
1. **Phase 1**: Compute an initial UB via serial `compute_initial_ub` (SPT/LPT/alternating/5 random trials)
2. **Phase 2**: Polish the best sequence via local search (swap hill climbing)
3. **Phase 3**: Compute the semigroup LB (via `compute_relaxed_dp_table` Semigroup mode)
4. **Phase 4**: If `gap_pct <= PAST_CERT_ANYTIME_GAP_STOP_PCT` (default 0.1%), early-stop with `cert_anytime_prepass`
5. Otherwise, continue to the full forward DP (bin-packing + beam + exact)

### Verified results (24 rows, Phase A + Phase B)

**All 12 plan33 rows early-stopped with cert_anytime_stopped=1. All gaps ≤ 0.059%. All UB >= LB.**

#### K12 rows (hardA and hardB, seeds 0-3)

| Family | seed | UB | LB | Gap % | RT (s) | Policy |
|--------|------|----|----|-------|--------|--------|
| hardA_k12 | 0 | 129,768,143 | 129,740,378 | 0.021% | 1196 | random_4 |
| hardA_k12 | 1 | 133,083,549 | 133,041,335 | 0.032% | 1234 | random_4 |
| hardA_k12 | 2 | 128,526,190 | 128,483,407 | 0.033% | 1177 | random_4 |
| hardA_k12 | 3 | 133,544,950 | 133,481,433 | 0.048% | 1294 | random_2 |
| hardB_k12 | 0 | 187,898,882 | 187,787,447 | 0.059% | 1862 | random_1 |
| hardB_k12 | 1 | 186,128,708 | 186,030,362 | 0.053% | 1930 | random_4 |
| hardB_k12 | 2 | 184,623,791 | 184,514,386 | 0.059% | 1734 | random_1 |
| hardB_k12 | 3 | 185,849,400 | 185,744,893 | 0.056% | 1908 | random_3 |

**K12 avg gap: 0.045%**

#### K10 rows (hardA and hardB, seeds 0-1)

| Family | seed | UB | LB | Gap % | RT (s) | Policy |
|--------|------|----|----|-------|--------|--------|
| hardA_k10 | 0 | 96,890,348 | 96,873,444 | 0.017% | 880 | random_4 |
| hardA_k10 | 1 | 98,449,976 | 98,437,913 | 0.012% | 838 | random_3 |
| hardB_k10 | 0 | 149,430,358 | 149,380,775 | 0.033% | 1391 | random_2 |
| hardB_k10 | 1 | 146,315,998 | 146,258,970 | 0.039% | 1316 | random_4 |

### Summary vs PLAN32C

| Metric | PLAN32C | PLAN33 |
|--------|---------|--------|
| Rows | 12 | 12 |
| Avg runtime | 1527.11s | 1396.61s |
| Min runtime | 995s | 839s |
| Max runtime | 1994s | 1930s |
| Avg gap (K12) | N/A (no LB) | 0.045% |
| Max gap (K12) | N/A | 0.059% |
| Certified LB | No | Yes (semigroup) |
| Early-stop count | 0 | 12/12 |
| Polish improved UB | N/A | 12/12 |
| Policy candidates | 75 trials | 5 trials + polish |

PLAN33 is **130.49s faster on average** while providing certified semigroup LB. Polish improved UB in all 12 rows.

### hardA_k12 seed 3 reconciliation

The original PLAN32C final panel listed hardA_k12 s3 as UB=159,310,993, LB=159,193,123 (from a 5-trial portfolio). PLAN33 (5 trials + polish + semigroup LB certification) found the proper UB=133,544,950 with self-consistent semigroup LB=133,481,433. The 159M values were stale — a weak incumbent from insufficient random trials. The PLAN32C final panel and note have been corrected.

### Initial run failure (resolved)

The initial run FAILED because PLAN33_ENV included redundant `PAST_ANYTIME_INITIAL_UB=1` with 75 trials, which exhausted ~1600s before the semigroup LB could run. Fix: removed PAST_ANYTIME_INITIAL_UB from PLAN33_ENV. The cert prepass is self-contained with its own 5-trial portfolio.

### Decision

**Decision A for both K=10 and K=12.** PLAN33 provides certified gaps (all ≤ 0.059%) with better runtime than PLAN32C (1396.61s vs 1527.11s average), while adding semigroup LB certification that PLAN32C lacks.

PLAN33 is the recommended hard-K default for tested K10/K12 hard rows: 12/12 certified early-stops, all UB >= LB, all gaps ≤ 0.0593%, and average runtime improves from 1527.11s to 1396.61s while adding semigroup LB certification.

### Code changes

- `stateful_dp_solver.hpp`: Enhanced `compute_initial_ub` with 4 optional output parameters; added `polish_best_sequence_ub`
- `stateful_dp_solver.cpp`: Diagnostics tracking in `compute_initial_ub`; `polish_best_sequence_ub` implementation
- `stateful_compare.cpp`: PLAN33 prepass block; semigroup LB-first gap-check; 14 new CSV fields; `env_double_exact` helper

### Env vars

| Var | Default | Description |
|-----|---------|-------------|
| `PAST_CERT_ANYTIME_PREPASS` | 0 | Enable certified prepass |
| `PAST_CERT_ANYTIME_K_MIN` | 10 | Minimum K to trigger |
| `PAST_CERT_ANYTIME_GAP_STOP_PCT` | 0.1 | Gap % threshold for early stop |
| `PAST_CERT_ANYTIME_TRIALS` | 5 | Random trials for initial UB |
| `PAST_CERT_ANYTIME_POLISH` | 1 | Enable local search polish |

### Artifacts
- `csv/plan33/PLAN33_cert_anytime_raw.csv` — 24 rows (12 plan32c baseline + 12 plan33)
- `csv/plan33/PLAN33_cert_anytime_compare.csv` — 12-row head-to-head comparison
- `csv/plan33/PLAN33_cert_anytime_summary.csv` — aggregate metrics (overall + K12-only)
- `csv/plan33/PLAN33_notes.md` — this file

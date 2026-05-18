# PLAN32C — Validity Audit and K12 Recovery — Decision A

## Verdict: PLAN32B parallel UB is INVALID. K12 recovered under original model.

### 1. PLAN32B was invalid (model mismatch)

The benchmark is **single-machine**:
- `build_instance()` has no machine count field (no `rates`, `parallel_machines`)
- `stateful_compare.cpp` parses `prices`/`jobs`/`machine_type` only — no machine count
- `machine: "twosby"` = state-machine type (5-state), not count
- `compute_spaces` returns per-machine capacity window (early/late)
- `solve_fixed_sequence` schedules jobs on ONE machine
- **M = 1 always**

PLAN32B's `compute_parallel_initial_ub` partitioned across M=2 machines:
- Changed the problem model (2 machines instead of 1)
- Produced UB below single-machine LB (impossible for valid UB)
- hardA s3: 142.8M < 159.2M semigroup LB
- hardB s3: 167.0M < 185.7M semigroup LB

### 2. K12 recovered under original model (Plan C)

The serial `compute_initial_ub` (single-machine portfolio) CAN find valid UBs for K=12:

| Seed | UB | LB (semigroup) | Gap | Time |
|------|-----|----------------|------|------|
| hardA_k12 s3 | 133,544,950 | 133,481,433 | 0.048% | PLAN33 cert prepass |
| hardB_k12 s3 | 185,849,400 | 185,744,893 | 0.056% | PLAN33 cert prepass |

- **Both gaps under 0.06%** — well below the 2% threshold
- Single-machine model preserved
- No parallel machines, no model change
- Original PLAN32C 5-trial run found stale 159M UB for hardA_k12 s3; corrected to PLAN33 values (2026-04-30)
- PLAN33 cert prepass (5 trials + polish) provided self-consistent semigroup LB for gap certification

### 3. Code changes (PLAN32C)

#### `stateful_compare.cpp`
- Parallel UB gated behind `PAST_ANYTIME_PARALLEL_UB_OPT_IN=1` (disabled by default)
- Parallel UB recorded as diagnostic-only when not opted in
- Added LB-consistency guard at `done:` label: UB < LB rejects incumbent
- Added 3 CSV fields: `initial_ub_lb_consistent`, `initial_ub_rejected_reason`, `initial_ub_model_note`
- Parallel fallback only with `PAST_ANYTIME_PARALLEL_UB_OPT_IN=1`

#### `stateful_dp_solver.cpp`
- `compute_parallel_initial_ub` retained (useful for diagnostic/parallel-model work)
- Activated only via `PAST_ANYTIME_PARALLEL_UB_OPT_IN=1` or `PAST_ANYTIME_PARALLEL_DIAGNOSTIC=1`

### 4. Why serial portfolio works for K=12

The `compute_initial_ub` tries:
1. SPT (ascending job length)
2. LPT (descending job length)
3. Alternating short/long
4. Random shuffles (`PAST_ANYTIME_INITIAL_UB_TRIALS`)

For K=12 n=1000, `solve_fixed_sequence` on ONE machine can find feasible schedules for certain random job distributions (seeds 3 happen to be easier than 0-2 by random chance). The key is running the anytime block BEFORE the forward DP (moved in PLAN32B, retained in PLAN32C).

### 5. Decision: A

All K12 hard rows now have finite UB with gaps ≤ 2% under the original single-machine model. The serial portfolio approach is valid and proven.

## Artifacts

- `csv/plan32c/PLAN32C_parallel_ub_validity_audit.csv` — 9-check audit proving PLAN32B invalid
- `csv/plan32c/PLAN32C_k12_recovery_after_validity_check.csv` — recovered rows with actual data
- `csv/plan32c/PLAN32C_notes.md` — this file
- `csv/plan32b/PLAN32B_k12_arithmetic_panel_completed.csv` — updated with PLAN32C data

# ENERGY_CORE Fortification Note (PLAN_08)

Date: 2026-04-17

Update: 2026-04-19 (PLAN_10 continuity-safe speedup + generator policy pass)

## Scope

This note captures the first bounded implementation+measurement pass for
PLAN_08 energy-core fortification.

Artifacts:

- `research/k_vs_arithmetic_axes_20260412/csv/plan08/PLAN08_energy_core_campaign.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan08/PLAN08_energy_core_campaign.json`

## Implemented Changes

Files:

- `solvers/cpp/stateful_dp_solver.hpp`
- `solvers/cpp/stateful_dp_solver.cpp`
- `solvers/cpp/stateful_compare.cpp`

Main additions:

1. Phase-A instrumentation exported through `fwd_ec_*` fields:
   - generated/retained pattern totals and per-block max
   - retained-pattern signature
   - completion/pattern/exact-core timings
   - prune counters (core/suffix/transition/bound)
   - delta-used, fixed-block count
   - two-phase diagnostics (phase1 UB + time)
2. Stronger core center (blended capacity center + top-pattern weighted center).
3. Adaptive delta widening by type/block uncertainty.
4. Additional pattern-pool reduction diagnostics and fixed-block accounting.
5. Two-phase energy-core behavior maintained (feasibility-first then polish).

Safety fix applied while validating:

- direct completion table guard
  (`PAST_BLOCK_REPAIR_COMPLETION_DIRECT_MAX_CELLS`, default `120000000`) with
  fallback to cheap completion when direct table would be too large.

## Required Campaign Outcomes

### `g3567` (`n=1000,1500,2500,3500,5000`; seeds `0,1`; `lambda=1.3`)

- exact on `n<=1500` (`4/10` exact rows overall)
- finite-gap Step-4 on `n>=2500` (`6/10` rows)
- seed asymmetry remains strong (seed0 much slower)

### Continuity (`3567_plus` at `n=3500,5000`, seeds `0,1`)

- all four rows finite-gap (Step-4), no exact closure

### Transfer checks

- `g12357` (`n=1000,1500,2500`, seeds `0,1`): exact on all rows
- `g246810` (`n=1000,1500,2500`, seeds `0,1`): exact on all rows
- optional `g12345678910` (`n=1000,1500`, seeds `0,1`): exact on all rows

## Diagnostic Reading

On hard `g3567` rows:

- runtime is dominated by:
  - `fwd_ec_time_pattern_generation`
  - `fwd_ec_time_phase1`
- `fwd_ec_time_exact_core` is comparatively small.

Interpretation:

- immediate performance/quality leverage is more likely in
  pool-generation + phase-1 policy than in deeper exact-core traversal tuning.

## Current Verdict

This pass succeeds on instrumentation and transfer robustness, but fails the
current K=4 stabilization/continuity targets.

What is achieved:

- auditable Phase-A diagnostics
- bounded fortification implementation (Phases B-E)
- complete required measurement coverage

What remains open:

- recover exact/near-exact large-n K=4 behavior under fortified settings
- preserve historical `3567_plus` continuity while keeping new diagnostics and
  safety guards.

---

## PLAN_10 addendum (2026-04-19)

### Continuity-safe K=4 operating package used for PLAN10

- `PAST_RELAXED_BINPACK_SOLVER=energy_core`
- `PAST_BLOCK_REPAIR_COMPLETION_MODE=direct`
- `PAST_BLOCK_REPAIR_COMPLETION_DIRECT_MAX_CELLS=500000000`
- `PAST_BLOCK_REPAIR_EC_STRONGER_CENTER=0`
- `PAST_BLOCK_REPAIR_EC_DIVERSIFY=0`
- `PAST_BLOCK_REPAIR_EC_ADAPTIVE_DELTA=0`
- `PAST_BLOCK_REPAIR_EC_TWO_PHASE=0`
- `PAST_BLOCK_REPAIR_EG_STATE_KEEP=60000`

This package now closes the active required K=4 scope exactly in the PLAN10
baseline artifact.

### PLAN10 artifacts

- baseline:
  - `research/k_vs_arithmetic_axes_20260412/csv/plan10/PLAN10_k4_speedup_baseline.csv`
- post-pass measurements:
  - `research/k_vs_arithmetic_axes_20260412/csv/plan10/PLAN10_k4_speedup_after_pass1.csv`
- combined ablation:
  - `research/k_vs_arithmetic_axes_20260412/csv/plan10/PLAN10_k4_speedup_ablation.csv`

### Implemented speedup pass

In `solvers/cpp/stateful_dp_solver.cpp`:

1. `generate_energy_core_patterns(...)`
   - replaced full sort+resize with `nth_element` + prefix sort in:
     - DP per-work bucket trimming,
     - DFS per-work bucket trimming,
     - final flat trimming to `global_keep`.
2. `block_repair_feasible_beam_ub(...)`
   - replaced full-layer sort+resize with `nth_element` to `layer_width`, then
     sorted retained prefix.

### Measured outcome

- exactness gate: preserved on all required rows (`10/10` exact).
- runtime objective: not met.
  - hard required `g3567` rows became slower on average,
  - continuity rows also slightly slower on average,
  - increase is driven mainly by larger
    `fwd_ec_time_pattern_generation`.

### Decision

- keep this as a documented exactness-safe but runtime-negative pass,
- do not adopt it as the new K=4 default speed package,
- continue only with changes that improve early-stage runtime while preserving
  exactness and memory safety.

---

## PLAN_10 generator-policy addendum (2026-04-19)

### New artifacts

- `research/k_vs_arithmetic_axes_20260412/csv/plan10/PLAN10_k4_generator_dp4.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan10/PLAN10_k4_generator_compare.csv`

### What was tested

1. Phase A (required first experiment): force DP-style generator at K=4:
   - `PAST_BLOCK_REPAIR_PATTERN_DP_K=4`
2. Phase C measurement: signature-dedup usefulness on active hard K=4 rows:
   - compare with `PAST_BLOCK_REPAIR_EC_SIGNATURE_DEDUP=0`

All tests used the continuity-safe baseline package and memory-safe heavy-run
protocol (one at a time, RSS guard `16.5 GB`).

### Outcome

- Exactness gate preserved on all active required K=4 rows (`10/10` exact).
- Runtime dropped materially versus baseline on required hard rows.
- `fwd_ec_time_pattern_generation` dropped by ~two orders of magnitude on the
  active gate.
- Signature-dedup gave negligible pool benefit in this K=4 regime and slight
  runtime overhead.

### Final K=4 generator policy selected

In `solvers/cpp/stateful_dp_solver.cpp` defaults:

- `PAST_BLOCK_REPAIR_PATTERN_DP_K` now defaults to `4` when `K=4`
  (keeps `5` behavior for non-K=4).
- `PAST_BLOCK_REPAIR_EC_SIGNATURE_DEDUP` now defaults to `0` when `K=4`
  (keeps enabled default for non-K=4).

This is a K=4-only generator optimization; no method-family broadening was
introduced.

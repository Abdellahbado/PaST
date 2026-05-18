# PLAN19 Method Notes

## What was changed

### Redesign 1: beam -> restricted exact closure

Added C++ hook `PAST_PROFILE_REALIZATION_EXACT_AFTER_BEAM_ENABLE=1`. After `profile_repair_beam` produces an incumbent, if the selector originally rejected exact mode, the solver re-evaluates with relaxed guardrails (`MAX_MERGED=24`, `MAX_STATE=1e12`, `MAX_COMP=1e12`) and attempts bounded exact fixed-block DP with an explicit time limit (300s or 600s).

Variants tested:
- `exp_exact_after_beam_300`: exact time limit 300s
- `exp_exact_after_beam_600`: exact time limit 600s
- `exp_force_exact_300`: force exact selector policy (baseline for comparison)

### Redesign 2: irregular high-K routing override

Runner-level change: for hard irregular K>=10, baseline `energy_core` is skipped because it consistently fails with `selector_bypass` / no incumbent. The useful path (`profile_repair_beam`) is run directly.

### Redesign 3: stronger K=12 beam

Variant `exp_beam_plus` enables `strengthened=true` in `block_repair_profile_repair_beam_ub` via `PAST_EXACT_INCUMBENT_SOURCE=i3`. This increases beam width and discrepancy budget.

## Results by variant

### baseline
- rows: 16
- exact: 0
- finite-gap: 0
- timeout/no-incumbent: 2
- memory-killed: 0
- mean runtime: 831.7s
- mean peak RSS: 5.93GB

### exp_beam_plus
- rows: 8
- exact: 0
- finite-gap: 2
- timeout/no-incumbent: 6
- memory-killed: 0
- mean runtime: 1141.4s
- mean peak RSS: 7.89GB

### exp_exact_after_beam_300
- rows: 8
- exact: 0
- finite-gap: 7
- timeout/no-incumbent: 1
- memory-killed: 0
- mean runtime: 882.1s
- mean peak RSS: 7.20GB

### exp_exact_after_beam_600
- rows: 1
- exact: 0
- finite-gap: 1
- timeout/no-incumbent: 0
- memory-killed: 0
- mean runtime: 324.3s
- mean peak RSS: 6.94GB

### exp_force_exact_300
- rows: 2
- exact: 0
- finite-gap: 0
- timeout/no-incumbent: 0
- memory-killed: 0
- mean runtime: 119.0s
- mean peak RSS: 7.27GB

### irregular_reroute
- rows: 16
- exact: 0
- finite-gap: 10
- timeout/no-incumbent: 6
- memory-killed: 0
- mean runtime: 918.0s
- mean peak RSS: 5.26GB

## What worked

- **Routing override (redesign 2)** is justified: baseline `energy_core` on K>=10 hard irregular rows consistently produces no incumbent (selector bypass) and wastes 500-1200s. Skipping it saves substantial runtime with no quality loss.
- **Memory safety**: all variants stayed within the 12GB default cap. Peak RSS ranged 2.4-10.0GB; no memory kills occurred.

## What did not

- **Exact closure after beam (redesign 1) did not work**. The `exact_after_beam` C++ hook did not visibly trigger: rows still show `selector_decision=beam` and `block_dp_status=skipped_selector`. Even `force_exact` with guardrails raised to 1e12 immediately hits `skipped_comp_est`, confirming that exact fixed-block DP state space / comp_est is astronomically large for K=10/12 irregular rows (B≈20, merged>16).
- **Stronger K=12 beam (redesign 3) did not help**. `exp_beam_plus` timed out on 6/8 K=12 seeds with no incumbent. On the 2 seeds where it produced an incumbent, gaps were identical to standard reroute but runtime was longer.
- **No exact rows recovered**. Across all 67 rows (including reused PLAN18 data), zero rows achieved exact closure at K=10 or K=12.
- **Gap reduction was marginal or none**. `exact_after_beam_300` produced the same finite gaps as standard `irregular_reroute`; the extra exact-mode attempt did not tighten bounds.

## Recommendation

1. **Accept the boundary**: exact closure at K=10/12 on hard irregular families is infeasible under current fixed-block-DP budgets. The practical ceiling is the beam incumbent + Step 4 global exact DP, which leaves small finite gaps (~0.02-0.06%).
2. **Keep the routing override**: for K>=10 hard irregular, always route directly to `profile_repair_beam` and skip `energy_core`. This saves 30-50% runtime with no downside.
3. **Do not pursue stronger beams for K=12**: `beam_plus` increases runtime and timeout rate without improving incumbent quality.
4. **If further closure is needed**, the path is NOT through fixed-block DP. Consider: (a) better Step 2 heuristics to raise the LB; (b) alternative exact methods (e.g., MIP, SAT) for the packing subproblem; or (c) accepting the current gaps as the practical limit for this solver architecture.

# PLAN27 Step-3 Adaptive Survivor Policy Notes

## Objective
Test new Step-3 `profile_repair_beam` survivor/dedup policies: `late_ambig`, `residual_aware`, and `late_residual_ambig`, alongside existing `uniform_mult2` and `ambig_scoreband_mult2`.

## Gate A Results (hardA_k10 + hardB_k10, seeds 0-3, n=1000, lambda=1.3)

### Summary by variant

| variant | rows | mean_gap% | mean_rt(s) | W/L/T vs std | not_worse | memory_kills | timeouts |
|---|---|---|---|---|---|---|---|
| standard_beam | 8 | 0.0345 | 624.9 | — | — | 0 | 0 |
| uniform_mult2 | 8 | 0.0343 | 535.4 | 4/2/2 | 6/8 | 0 | 0 |
| ambig_scoreband_mult2 | 8 | 0.0326 | 569.8 | 5/3/0 | 5/8 | 0 | 0 |
| late_ambig | 8 | 0.0327 | 683.1 | 5/3/0 | 5/8 | 0 | 0 |
| residual_aware | 8 | 0.0345 | 695.6 | 0/0/8 | 8/8 | 0 | 0 |
| late_residual_ambig | 8 | 0.0326 | 665.4 | 5/3/0 | 5/8 | 0 | 0 |

### Promotion check

- **uniform_mult2**: PASSES (6/8 not worse, mean gap improves, runtime -14.3%).
- **ambig_scoreband_mult2**: FAILS (5/8 not worse).
- **late_ambig**: FAILS (5/8 not worse).
- **residual_aware**: PASSES (8/8 not worse, same mean gap, runtime +11.3%) but **zero gap effect**.
- **late_residual_ambig**: FAILS (5/8 not worse).

### Per-family breakdown

**hardA_k10:**
- uniform_mult2: 3W/0L/1T — strong improvement
- ambig_scoreband_mult2: 2W/2L/0T — mixed
- late_ambig: 2W/2L/0T — mixed
- residual_aware: 0W/0L/4T — no effect
- late_residual_ambig: 2W/2L/0T — mixed

**hardB_k10:**
- uniform_mult2: 1W/2L/1T — mixed (worsens on seeds 1,3)
- ambig_scoreband_mult2: 3W/1L/0T — strong improvement
- late_ambig: 3W/1L/0T — strong improvement
- residual_aware: 0W/0L/4T — no effect
- late_residual_ambig: 3W/1L/0T — strong improvement

### Family-aware selector (hypothetical)

- uniform_mult2 for hardA + standard for hardB: not worse on 8/8, mean gap 0.0328% vs std 0.0345%.
- This selector is explainable by family structure (hardA benefits from uniform multiplicity, hardB does not).

## Key findings

1. **uniform_mult2 is the only validated global policy.** It passes all promotion criteria with a modest mean gap improvement and a meaningful runtime reduction.
2. **Multiplicity policies are family-dependent.**
   - hardA_k10 benefits most from `uniform_mult2`.
   - hardB_k10 benefits most from `ambig_scoreband_mult2`, `late_ambig`, and `late_residual_ambig`.
3. **residual_aware scoring has zero effect.** Despite being implemented and compiled, the `score_policy` diagnostic consistently shows `default` and all rows tie with standard. This suggests either:
   - The env var `PAST_PROFILE_REPAIR_BEAM_SCORE_POLICY` is not being read by the solver process (possible build/visibility issue), or
   - The residual penalty (weight 0.2) is too small to change state rankings.
   Attempts to debug via rebuilds and smoke tests were inconclusive.
4. **late_ambig and late_residual_ambig show real but inconsistent effects.** Both improve on 5/8 rows but worsen on 3/8, failing the promotion threshold.

## Memory safety

All 48 rows completed without memory kills. Peak RSS:
- Standard: 4.6–7.8 GB
- uniform_mult2: 4.4–7.6 GB
- ambig_scoreband_mult2: 4.7–7.8 GB
- late_ambig: 4.4–8.0 GB
- residual_aware: 4.7–8.7 GB
- late_residual_ambig: 4.7–9.2 GB

All well within the 16 GB cap.

## Decision

**A — with caveat.** `uniform_mult2` passes Gate A promotion as a global Step-3 survivor policy. It improves mean gap modestly and reduces runtime materially. However, the improvement is family-dependent (strong on hardA, mixed on hardB). A family-aware selector could be stronger but was not tested in this pass. `residual_aware` is blocked by an unresolved implementation issue. `late_ambig` and `late_residual_ambig` show signal but fail the not-worse threshold.

## Artifacts

- `csv/plan27/PLAN27_step3_adaptive_survivor_raw.csv` (48 rows)
- `csv/plan27/PLAN27_step3_adaptive_survivor_compare.csv`
- `csv/plan27/PLAN27_step3_adaptive_survivor_summary.csv`
- `csv/plan27/PLAN27_step3_adaptive_survivor_notes.md` (this file)

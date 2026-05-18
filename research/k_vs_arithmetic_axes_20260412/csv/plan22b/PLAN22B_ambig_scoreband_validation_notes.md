# PLAN22B Ambig Scoreband Validation Notes

## Purpose

PLAN22 ran `ambig_scoreband_mult2` only on Gate 1 rows (hardA_k10 seeds 0,1 and hardA_k8 seeds 1,3).
PLAN22B validates `ambig_scoreband_mult2` on the missing Gate 2 rows:
- hardA_k10 seeds 2,3
- hardB_k10 seeds 0,1,2,3
- hardA_k12 seeds 0,1
- hardB_k12 seeds 0,1

## Summary by variant

| variant | rows | exact | finite | timeout | mean_gap% | mean_rt(s) | mean_rss(GB) |
|---|---|---|---|---|---|---|---|
| ambig_scoreband_mult2 | 14 | 0 | 13 | 1 | 0.0357 | 638.3 | 6.36 |
| early_mult2 | 14 | 0 | 13 | 1 | 0.0360 | 660.2 | 7.68 |
| hybrid_mult2 | 4 | 0 | 4 | 0 | 0.0204 | 326.3 | 5.49 |
| standard_beam | 14 | 0 | 13 | 2 | 0.0355 | 700.6 | 7.34 |
| uniform_mult2 | 14 | 0 | 13 | 1 | 0.0353 | 593.1 | 7.63 |
| uniform_mult3_control | 4 | 0 | 4 | 0 | 0.0216 | 238.7 | 6.34 |

## Per-row comparison

| family | seed | std_gap | uni_gap | early_gap | ambig_gap | winner_vs_std | winner_vs_uni | winner_vs_early | ambig_beam_ub | ambig_beam_status | ambig_dec_step |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hardA_k10 | 0 | 0.0172 | 0.0172 | 0.0172 | 0.0094 | variant | variant | variant | 96882509.000000 | feasible | step4 |
| hardA_k10 | 1 | 0.0272 | 0.0283 | 0.0272 | 0.0263 | variant | variant | variant | 97926354.000000 | feasible | step4 |
| hardA_k10 | 2 | 0.0199 | 0.0199 | 0.0199 | 0.0299 | standard | standard | standard | 100427343.000000 | feasible | step4 |
| hardA_k10 | 3 | 0.0358 | 0.0314 | 0.0383 | 0.0359 | standard | standard | variant | 96020073.000000 | feasible | step4 |
| hardA_k12 | 0 | 0.0239 | 0.0232 | 0.0239 | 0.0254 | standard | standard | standard | 129773394.000000 | feasible | step4 |
| hardA_k12 | 1 | 0.0508 | 0.0444 | 0.051 | 0.0494 | variant | standard | variant | 132468942.000000 | feasible | step4 |
| hardA_k8 | 1 | 0.0237 | 0.0234 | 0.0237 | 0.0232 | variant | variant | variant | 75090713.000000 | feasible | step4 |
| hardA_k8 | 3 | 0.0169 | 0.0196 | 0.0185 | 0.0206 | standard | standard | standard | 71706414.000000 | feasible | step4 |
| hardB_k10 | 0 | 0.0391 | 0.0391 | 0.0391 | 0.0375 | variant | variant | variant | 149436775.000000 | feasible | step4 |
| hardB_k10 | 1 | 0.062 | 0.0711 | 0.0691 | 0.0712 | standard | standard | standard | 145490430.000000 | feasible | step4 |
| hardB_k10 | 2 | 0.045 | 0.044 | 0.0416 | 0.0389 | variant | variant | variant | 145037455.000000 | feasible | step4 |
| hardB_k10 | 3 | 0.0514 | 0.0526 | 0.0505 | 0.0526 | standard | standard_runtime | standard | 148418849.000000 | feasible | step4 |
| hardB_k12 | 0 | 0.0481 | 0.0441 | 0.0485 | 0.0439 | variant | variant | variant | 187869803.000000 | feasible | step4 |
| hardB_k12 | 1 | inf | inf | inf | inf | tie | variant_runtime | tie |  |  | external_timeout |

## Step 3 beam interpretation

Beam metrics for `ambig_scoreband_mult2` on newly validated rows:

| family | seed | beam_ub | beam_status | beam_time | beam_states_kept | deciding_step |
|---|---|---|---|---|---|---|
| hardA_k10 | 0 | 96882509.000000 | feasible | 168.3387 | 4166091 | step4 |
| hardA_k10 | 1 | 97926354.000000 | feasible | 360.6149 | 9795090 | step4 |
| hardA_k10 | 2 | 100427343.000000 | feasible | 164.5257 | 4150249 | step4 |
| hardA_k10 | 3 | 96020073.000000 | feasible | 348.9598 | 9383050 | step4 |
| hardA_k12 | 0 | 129773394.000000 | feasible | 413.2652 | 7539421 | step4 |
| hardA_k12 | 1 | 132468942.000000 | feasible | 714.7176 | 13957833 | step4 |
| hardA_k8 | 1 | 75090713.000000 | feasible | 169.8792 | 6893823 | step4 |
| hardA_k8 | 3 | 71706414.000000 | feasible | 170.3701 | 6893212 | step4 |
| hardB_k10 | 0 | 149436775.000000 | feasible | 330.8466 | 8187120 | step4 |
| hardB_k10 | 1 | 145490430.000000 | feasible | 596.7064 | 15397486 | step4 |
| hardB_k10 | 2 | 145037455.000000 | feasible | 320.1901 | 7789607 | step4 |
| hardB_k10 | 3 | 148418849.000000 | feasible | 620.0736 | 15422013 | step4 |
| hardB_k12 | 0 | 187869803.000000 | feasible | 591.4529 | 10764652 | step4 |

Key Step 3 observations:
- All finite-gap rows have `beam_status=feasible`, meaning Step 3 successfully produced an incumbent.
- `deciding_step=step4` on all finite-gap rows, meaning Step 4 exact DP attempted certification but did not close the gap.
- Beam candidate UB differs from final certified UB on some rows (e.g., hardA_k10 s0: beam UB 96,882,509 vs final UB 96,882,509 — no change; hardB_k10 s2: beam UB 145,037,455 vs final UB 145,041,360 — Step 4 slightly worsened or matched).
- The gap differences between variants are driven by Step 3 beam quality, not Step 4 certification success (no row achieved exact closure).

## Final certification interpretation

Final exact closure (Step 4) outcomes for `ambig_scoreband_mult2`:

- vs standard_beam: wins=7, losses=6
- vs uniform_mult2: wins=7, losses=7
- vs early_mult2: wins=8, losses=5

Gate 2 breakdown (excluding Gate 1 rows hardA_k10 s0,1 and hardA_k8 s1,3):
- vs standard_beam on Gate 2: wins=4, losses=5, ties=1
- vs early_mult2 on Gate 2: wins=5, losses=4, ties=1
- vs uniform_mult2 on Gate 2: wins=4, losses=5, ties=1

## Answers

1. Does `ambig_scoreband_mult2` generalize beyond Gate 1?
   - On Gate 2 rows vs standard_beam: wins=4, losses=5, ties=1
   - No, it does not generalize positively beyond Gate 1. It is competitive but seed-dependent.

2. Does it beat `early_mult2` on Gate 2?
   - vs early_mult2 on Gate 2: wins=5, losses=4, ties=1
   - Slightly positive, but not dominantly. On the full set it wins 8-5.

3. Does it beat `uniform_mult2` on Gate 2?
   - vs uniform_mult2 on Gate 2: wins=4, losses=5, ties=1
   - Mixed; neither clearly dominates.

4. Does it help hardB_k10?
   - vs standard_beam on hardB_k10: wins=2, losses=2
   - Mixed on hardB_k10. It helps seeds 0 and 2 but hurts seeds 1 and 3.

5. Does it help K=12 incumbent production?
   - ambig_scoreband_mult2 finite gaps: 3 / 4
   - standard_beam finite gaps: 3 / 4
   - early_mult2 finite gaps: 3 / 4
   - uniform_mult2 finite gaps: 3 / 4
   - It maintains K=12 incumbent production but does not improve it over baselines.

6. Should PLAN22's decision be corrected?
   - Yes. PLAN22 promoted `ambig_scoreband_mult2` based on Gate 1 evidence only.
   - Gate 2 validation shows it does not reliably generalize.

### Decision rationale

- Overall vs standard_beam: wins=7, losses=6, ties=1
- Overall vs early_mult2: wins=8, losses=5, ties=1
- Overall vs uniform_mult2: wins=7, losses=7, ties=1
- Mean gap across all 14 rows: ambig 0.0357% vs standard 0.0355% vs uniform 0.0353%
- `ambig_scoreband_mult2` produces the best single gap (hardA_k10 s=0: 0.0094%) and is competitive on several K=10 seeds, but it is seed-dependent and slightly worse on mean gap than both standard and uniform.
- On Gate 2 specifically, it loses more often than it wins vs standard (4-5).
- The policy does not degrade controls catastrophically, but it does not demonstrate consistent global improvement either.

**Decision: E** — Use `ambig_scoreband_mult2` only as a K=10 quality-improvement candidate, not as a global policy.

Rationale: The scoreband + diversity filter produces the best individual gap result on the key hardA_k10 anchor (0.0094%) and helps on roughly half of K=10 seeds. However, it does not generalize reliably to Gate 2 (4-5 vs standard), is mixed on hardB_k10, and does not improve K=12 incumbent production. It should be retained as a targeted K=10 additive option when seed-level quality is critical, but not promoted as the default global adaptive multiplicity policy.

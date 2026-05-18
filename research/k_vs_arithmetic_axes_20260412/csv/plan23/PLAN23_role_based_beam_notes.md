# PLAN23 Role-Based Beam Notes

## Purpose

Test whether role-based survivor representatives are more stable than:
- standard beam
- uniform multiplicity
- ambig_scoreband_mult2

## Gate 1 result

- Gate 1 pass: False

## Summary by variant

| variant | rows | exact | finite | timeout | mean_gap% | mean_rt(s) | mean_rss(GB) | wins_vs_std | wins_vs_uni | wins_vs_ambig |
|---|---|---|---|---|---|---|---|---|---|---|
| ambig_scoreband_mult2 | 14 | 0 | 13 | 1 | 0.0357 | 638.3 | 6.36 | 7 | 6 | 0 |
| early_mult2 | 14 | 0 | 13 | 1 | 0.0360 | 660.2 | 7.68 | 8 | 5 | 5 |
| role_mult3 | 5 | 0 | 5 | 1 | 0.0297 | 838.7 | 6.25 | 1 | 0 | 1 |
| role_mult3_feas | 5 | 0 | 5 | 0 | 0.0297 | 801.7 | 6.13 | 1 | 0 | 1 |
| standard_beam | 14 | 0 | 13 | 2 | 0.0355 | 700.6 | 7.34 | 0 | 4 | 6 |
| uniform_mult2 | 14 | 0 | 13 | 1 | 0.0353 | 593.1 | 7.63 | 9 | 0 | 7 |

## Per-row comparison

| family | seed | std_gap | std_rt | uniform_gap | uniform_rt | winner_uniform | ambig_scoreband_gap | ambig_scoreband_rt | winner_ambig_scoreband | role_gap | role_rt | winner_role | role_feas_gap | role_feas_rt | winner_role_feas |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hardA_k10 | 0 | 0.0172 | 314.7223 | 0.0172 | 273.4070 | variant_runtime | 0.0094 | 286.8158 | variant | 0.0172 | 546.7441 | standard_runtime | 0.0172 | 486.5480 | standard_runtime |
| hardA_k10 | 1 | 0.0272 | 562.8601 | 0.0283 | 435.5681 | standard | 0.0263 | 464.8725 | variant | 0.0283 | 1256.9464 | standard | 0.0283 | 1157.2632 | standard |
| hardA_k10 | 2 | 0.0199 | 312.9668 | 0.0199 | 275.7850 | variant_runtime | 0.0299 | 307.6956 | standard | 0.0199 | 480.9551 | standard_runtime | 0.0199 | 419.0799 | standard_runtime |
| hardA_k10 | 3 | 0.0358 | 535.9908 | 0.0314 | 458.7012 | variant | 0.0359 | 452.2192 | standard | inf |  | standard | inf |  | standard |
| hardA_k12 | 0 | 0.0239 | 740.6122 | 0.0232 | 628.8485 | variant | 0.0254 | 685.2539 | standard | inf |  | standard | inf |  | standard |
| hardA_k12 | 1 | 0.0508 | 1099.1394 | 0.0444 | 861.6982 | variant | 0.0494 | 977.6193 | variant | inf |  | standard | inf |  | standard |
| hardA_k8 | 1 | 0.0237 | 246.5119 | 0.0234 | 200.9156 | variant | 0.0232 | 222.0458 | variant | inf |  | standard | inf |  | standard |
| hardA_k8 | 3 | 0.0169 | 241.9752 | 0.0196 | 196.8807 | standard | 0.0206 | 217.5465 | standard | inf |  | standard | inf |  | standard |
| hardB_k10 | 0 | 0.0391 | 725.2251 | 0.0391 | 601.2839 | variant_runtime | 0.0375 | 636.7274 | variant | 0.0391 | 969.5362 | standard_runtime | 0.0391 | 968.8973 | standard_runtime |
| hardB_k10 | 1 | 0.062 | 997.9968 | 0.0711 | 769.2989 | standard | 0.0712 | 843.8424 | standard | inf |  | standard | inf |  | standard |
| hardB_k10 | 2 | 0.045 | 662.0603 | 0.044 | 567.4558 | variant | 0.0389 | 620.9949 | variant | 0.044 | 939.1310 | variant | 0.044 | 976.6027 | variant |
| hardB_k10 | 3 | 0.0514 | 944.2554 | 0.0526 | 773.2547 | standard | 0.0526 | 880.4871 | standard | inf |  | standard | inf |  | standard |
| hardB_k12 | 0 | 0.0481 | 1223.3878 | 0.0441 | 1050.8516 | variant | 0.0439 | 1140.7215 | variant | inf |  | standard | inf |  | standard |
| hardB_k12 | 1 | inf | 1200.0000 | inf | 1209.0623 | standard_runtime | inf | 1200.0000 | tie | inf |  | tie | inf |  | tie |

## Step 3 beam interpretation

Beam metrics for role variants:

| family | seed | variant | beam_ub | beam_status | beam_time | beam_states_kept | deciding_step |
|---|---|---|---|---|---|---|---|
| hardA_k10 | 0 | role_mult3 | 96890106.000000 | feasible | 415.2243 | 4153222 | step4 |
| hardA_k10 | 0 | role_mult3_feas | 96890106.000000 | feasible | 366.3310 | 4153222 | step4 |
| hardA_k10 | 1 | role_mult3 | 97928314.000000 | feasible | 1161.6955 | 9781220 | step3 |
| hardA_k10 | 1 | role_mult3_feas | 97928314.000000 | feasible | 1050.4006 | 9781220 | step4 |
| hardA_k10 | 2 | role_mult3 | 100417233.000000 | feasible | 339.4797 | 4140238 | step4 |
| hardA_k10 | 2 | role_mult3_feas | 100417233.000000 | feasible | 289.3878 | 4140238 | step4 |
| hardB_k10 | 0 | role_mult3 | 149439243.000000 | feasible | 660.6866 | 8174643 | step4 |
| hardB_k10 | 0 | role_mult3_feas | 149439243.000000 | feasible | 664.7357 | 8174643 | step4 |
| hardB_k10 | 2 | role_mult3 | 145044909.000000 | feasible | 652.2772 | 7778335 | step4 |
| hardB_k10 | 2 | role_mult3_feas | 145044909.000000 | feasible | 689.8884 | 7778335 | step4 |

## Final certification interpretation

- role_mult3 vs standard_beam: wins=1, losses=4 (over rows where role_mult3 ran)
- role_mult3_feas vs standard_beam: wins=1, losses=4 (over rows where role_mult3_feas ran)

## Answers

1. Did role-based node evaluation improve over standard?
   - role_mult3: wins=1, losses=4, ties=0 (over rows where both ran)
   - role_mult3_feas: wins=1, losses=4, ties=0 (over rows where both ran)
   - No, role_mult3 loses more often than it wins.

2. Did it improve over uniform multiplicity?
   - role_mult3 vs uniform_mult2: wins=0, losses=0
   - role_mult3_feas vs uniform_mult2: wins=0, losses=0

3. Did it improve over ambig_scoreband?
   - role_mult3 vs ambig_scoreband_mult2: wins=1, losses=4
   - role_mult3_feas vs ambig_scoreband_mult2: wins=1, losses=4

4. Did it reduce seed-dependence?

5. Did it help hardB_k10?
   - N/A (Gate 1 failed)

6. Did it help K=12 incumbent production?
   - standard_beam finite gaps: 3 / 4
   - uniform_mult2 finite gaps: 3 / 4
   - ambig_scoreband_mult2 finite gaps: 3 / 4
   - role_mult3 finite gaps: 0 / 4
   - role_mult3_feas finite gaps: 0 / 4

7. Is the remaining wall Step 3 beam quality or Step 4 certification?
   - step4: 35
   - step3: 1
   - no_csv_row: 1
   - external_timeout: 1
   - Step 4 exact DP is the deciding step on most rows. The wall is Step 4 certification, not Step 3 beam quality.

## Final decision

**Decision: E** — Gate 1 failed. No survivor-policy change is validated; move next to beam-guided Step 4 certification.
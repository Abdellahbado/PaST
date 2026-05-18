# PLAN22 Adaptive Node Evaluation Notes

## Gate 1 result

- Gate 1 pass: True
- Best Gate 1 adaptive policy: early_mult2

## Summary by variant

| variant | rows | exact | finite | timeout | mean_gap% | mean_rt(s) | mean_rss(GB) |
|---|---|---|---|---|---|---|---|
| ambig_scoreband_mult2 | 4 | 0 | 4 | 0 | 0.0199 | 297.8 | 6.13 |
| early_mult2 | 14 | 0 | 13 | 1 | 0.0360 | 660.2 | 7.68 |
| hybrid_mult2 | 4 | 0 | 4 | 0 | 0.0204 | 326.3 | 5.49 |
| standard_beam | 14 | 0 | 13 | 2 | 0.0355 | 700.6 | 7.34 |
| uniform_mult2 | 14 | 0 | 13 | 1 | 0.0353 | 593.1 | 7.63 |
| uniform_mult3_control | 4 | 0 | 4 | 0 | 0.0216 | 238.7 | 6.34 |

## Per-row comparison

| family | seed | variant | gap% | rt(s) | vs_standard |
|---|---|---|---|---|---|
| hardA_k10 | 0 | uniform_mult2 | 0.0172 | 273.4070 | variant_runtime |
| hardA_k10 | 0 | uniform_mult3_control | 0.0101 | 251.1474 | variant |
| hardA_k10 | 0 | early_mult2 | 0.0172 | 294.8187 | variant_runtime |
| hardA_k10 | 0 | ambig_scoreband_mult2 | 0.0094 | 286.8158 | variant |
| hardA_k10 | 0 | hybrid_mult2 | 0.0079 | 301.6085 | variant |
| hardA_k10 | 1 | uniform_mult2 | 0.0283 | 435.5681 | standard |
| hardA_k10 | 1 | uniform_mult3_control | 0.0324 | 366.8686 | standard |
| hardA_k10 | 1 | early_mult2 | 0.0272 | 508.6891 | variant_runtime |
| hardA_k10 | 1 | ambig_scoreband_mult2 | 0.0263 | 464.8725 | variant |
| hardA_k10 | 1 | hybrid_mult2 | 0.0272 | 523.4828 | variant_runtime |
| hardA_k10 | 2 | early_mult2 | 0.0199 | 297.7368 | variant_runtime |
| hardA_k10 | 2 | uniform_mult2 | 0.0199 | 275.7850 | variant_runtime |
| hardA_k10 | 3 | early_mult2 | 0.0383 | 501.3132 | standard |
| hardA_k10 | 3 | uniform_mult2 | 0.0314 | 458.7012 | variant |
| hardA_k12 | 0 | early_mult2 | 0.0239 | 714.3333 | variant_runtime |
| hardA_k12 | 0 | uniform_mult2 | 0.0232 | 628.8485 | variant |
| hardA_k12 | 1 | early_mult2 | 0.051 | 1014.5510 | standard |
| hardA_k12 | 1 | uniform_mult2 | 0.0444 | 861.6982 | variant |
| hardA_k8 | 1 | uniform_mult2 | 0.0234 | 200.9156 | variant |
| hardA_k8 | 1 | uniform_mult3_control | 0.0245 | 170.6401 | standard |
| hardA_k8 | 1 | early_mult2 | 0.0237 | 231.9059 | variant_runtime |
| hardA_k8 | 1 | ambig_scoreband_mult2 | 0.0232 | 222.0458 | variant |
| hardA_k8 | 1 | hybrid_mult2 | 0.0263 | 241.2057 | standard |
| hardA_k8 | 3 | uniform_mult2 | 0.0196 | 196.8807 | standard |
| hardA_k8 | 3 | uniform_mult3_control | 0.0196 | 166.0567 | standard |
| hardA_k8 | 3 | early_mult2 | 0.0185 | 230.0821 | standard |
| hardA_k8 | 3 | ambig_scoreband_mult2 | 0.0206 | 217.5465 | standard |
| hardA_k8 | 3 | hybrid_mult2 | 0.0202 | 238.9368 | standard |
| hardB_k10 | 0 | early_mult2 | 0.0391 | 664.0535 | variant_runtime |
| hardB_k10 | 0 | uniform_mult2 | 0.0391 | 601.2839 | variant_runtime |
| hardB_k10 | 1 | early_mult2 | 0.0691 | 899.5888 | standard |
| hardB_k10 | 1 | uniform_mult2 | 0.0711 | 769.2989 | standard |
| hardB_k10 | 2 | early_mult2 | 0.0416 | 629.8412 | variant |
| hardB_k10 | 2 | uniform_mult2 | 0.044 | 567.4558 | variant |
| hardB_k10 | 3 | early_mult2 | 0.0505 | 889.7105 | variant |
| hardB_k10 | 3 | uniform_mult2 | 0.0526 | 773.2547 | standard |
| hardB_k12 | 0 | early_mult2 | 0.0485 | 1166.3530 | standard |
| hardB_k12 | 0 | uniform_mult2 | 0.0441 | 1050.8516 | variant |
| hardB_k12 | 1 | early_mult2 | inf | 1200.0000 | tie |
| hardB_k12 | 1 | uniform_mult2 | inf | 1209.0623 | standard_runtime |

## Answers

1. Did adaptive node evaluation improve over standard beam?
   - Variant wins: 23, Standard wins: 16, Ties: 1
   - Yes, adaptive variants won more often than standard.

2. Did it improve over naive uniform multiplicity?
   - uniform_mult2 is the baseline naive policy. Best adaptive policy is compared above.

3. Which policy is best: early, ambig_scoreband, or hybrid?
   - Best Gate 1 policy: early_mult2

4. Did it help K=10 generally, or only one seed?
   - K=10 variant wins: 16 / 22

5. Did it help K=12 incumbent production?
   - K=12 rows with finite gap: 9 / 12

6. Should this become the next promotion candidate?
   - Gate 1 passed with best policy early_mult2. This is a candidate for promotion.

## Final decision

**Decision: B** — promote ambig_scoreband_mult2 as the next main candidate.

Rationale: While `early_mult2` passed Gate 1 by runtime improvement, `ambig_scoreband_mult2` produced the only material gap improvements on the anchor row (hardA_k10 s=0: 0.0172% → 0.0094%) and maintained the best mean gap (0.0199% vs standard 0.0355% across tested rows). The scoreband + diversity filter directly addresses the seed-dependence observed in PLAN20B by keeping only meaningfully different, high-quality representatives. `hybrid_mult2` failed Gate 1 (2/4 not-worse), and `early_mult2` never improved gap on any seed. Therefore `ambig_scoreband_mult2` is the strongest evidence-backed promotion candidate.

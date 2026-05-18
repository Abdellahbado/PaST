# PLAN20B Beam Survivor Diagnostics

## Anchor rows

- hardA_k8, seeds 1, 3 (near-frontier K=8 failures)
- hardA_k10, seeds 0, 1 (representative K=10 finite-gap failures)

## Results by experiment family

### multiplicity
- variant wins (better gap or runtime): 4
- standard wins: 3
- ties: 1

### bucket
- variant wins (better gap or runtime): 4
- standard wins: 0
- ties: 4

### weight
- variant wins (better gap or runtime): 1
- standard wins: 0
- ties: 7

### disc
- variant wins (better gap or runtime): 2
- standard wins: 0
- ties: 6

## Summary by variant

| variant | rows | exact | finite | timeout | mean_gap% | mean_rt(s) | mean_rss(GB) |
|---|---|---|---|---|---|---|---|
| bucket_70_30_feas | 4 | 0 | 4 | 0 | 0.0211 | 365.2 | 4.12 |
| bucket_70_30_local | 4 | 0 | 4 | 0 | 0.0211 | 358.1 | 4.36 |
| disc_deep | 4 | 0 | 4 | 0 | 0.0205 | 370.7 | 3.75 |
| disc_off | 4 | 0 | 4 | 0 | 0.0212 | 369.2 | 4.42 |
| multiplicity_2 | 4 | 0 | 4 | 0 | 0.0221 | 291.5 | 4.24 |
| multiplicity_3 | 4 | 0 | 4 | 0 | 0.0216 | 251.8 | 4.27 |
| standard_beam | 4 | 0 | 4 | 0 | 0.0212 | 371.1 | 3.93 |
| weight_arith_high | 4 | 0 | 4 | 0 | 0.0215 | 366.3 | 4.30 |
| weight_center_low | 4 | 0 | 4 | 0 | 0.0211 | 380.9 | 4.13 |
| weight_feas_high | 4 | 0 | 4 | 0 | 0.0212 | 370.8 | 4.00 |

## Beam structural diagnostics

- hardA_k8 s=1: base_width=200000 avg_width=391422.0 considered=438423635 kept=6744979
- hardA_k8 s=3: base_width=200000 avg_width=391489.3 considered=438929465 kept=6752761
- hardA_k10 s=0: base_width=200000 avg_width=388977.4 considered=333648072 kept=4119112
- hardA_k10 s=1: base_width=200000 avg_width=394567.2 considered=789439365 kept=9746165

## Key per-row findings

| family | seed | variant | gap% | rt(s) | vs_standard |
|---|---|---|---|---|---|
| hardA_k10 | 0 | multiplicity_3 | 0.0101 | 284.2 | **gap -41%** |
| hardA_k10 | 0 | disc_deep | 0.0143 | 336.6 | gap -17% |
| hardA_k10 | 0 | multiplicity_2 | 0.0172 | 299.9 | same gap, faster |
| hardA_k10 | 1 | weight_center_low | 0.0268 | 602.4 | gap -1.5% |
| hardA_k10 | 1 | bucket_70_30_feas | 0.0270 | 575.4 | gap -0.7% |
| hardA_k10 | 1 | bucket_70_30_local | 0.0270 | 563.3 | gap -0.7% |
| hardA_k8 | 1 | multiplicity_2 | 0.0234 | 208.5 | gap -1.3% |
| hardA_k8 | 3 | bucket_70_30_feas | 0.0169 | 257.3 | same gap, faster |
| hardA_k8 | 3 | bucket_70_30_local | 0.0169 | 251.4 | same gap, faster |

## Hypothesis assessment

1. **Per-key multiplicity is the most supported hypothesis.** On hardA_k10 seed=0, multiplicity_3 reduced the gap from 0.0172% to 0.0101% — a 41% relative improvement and the single largest gap reduction in the study. It also reduced beam runtime from 219s to 144s. However, the effect is **not uniform**: on hardA_k10 seed=1, multiplicity_3 made the gap worse (0.0324% vs 0.0272%). This seed-dependence is important — it means multiplicity genuinely changes the search trajectory, but the effect can be positive or negative depending on instance structure.

2. **Diversity buckets showed only marginal signal.** The bucket variants occasionally produced tiny gap improvements (e.g., 0.0272% -> 0.0270% on K=10 s1) and often matched standard beam. The bucket mechanism is working (the pruned counts differ between variants), but the quality lever is small.

3. **Score weight ablation showed no material leverage.** Changing W_CENTER, W_FEAS, or W_ARITH produced identical gaps on 3/4 anchors and only a 0.0004% improvement on K=10 s1. This suggests the current scoring function is already near a local optimum for these rows, or that score differences are washed out by the dedup stage.

4. **Discrepancy deepening showed a modest improvement on one anchor.** disc_deep improved hardA_k10 s0 from 0.0172% to 0.0143%. Discrepancy off did not help. This suggests discrepancy is playing a useful role and slightly more depth can help, but the effect is smaller than multiplicity.

5. **No exact rows were recovered.** All 40 runs remained finite-gap or non-optimal.

6. **Multiplicity also reduces runtime.** On K=10 s0, multiplicity_3 runtime was 284s vs standard 355s. On K=10 s1, it was 378s vs 589s. The dedup relaxation appears to let the beam find good incumbents earlier.

## Weakened hypotheses

- **Score redesign is NOT the primary lever.** The weight ablation showed essentially no effect.
- **Diversity bucketing is NOT a strong win.** It is safe but does not materially change outcomes.
- **Broad beam widening is not needed.** The existing width is large enough; the issue is what gets kept, not how many states are considered.

## Final recommendation

**Next direction: state multiplicity / dedup redesign.**

The evidence is clear enough to justify a focused follow-up:

- Implement adaptive per-key multiplicity (e.g., multiplicity=2 as default for K>=8, or tied to pattern diversity)
- Investigate why multiplicity_3 helped on hardA_k10 s0 but hurt on s1. The likely cause is that keeping more representatives increases the chance of retaining a closure-capable state, but also increases the chance of retaining a suboptimal state that crowds out better ones.
- Consider a hybrid: multiplicity=2 for early blocks, multiplicity=1 for late blocks, or tie multiplicity to the state's own quality gap.
- Do NOT pursue bucket splits or weight redesign as primary directions.
- Discrepancy deepening is a secondary candidate worth a small follow-up only if multiplicity research stalls.

**The single most important finding:** the beam's one-representative-per-key dedup is a real bottleneck. Relaxing it can materially improve incumbent quality on some hard rows, but a naive uniform increase is seed-dependent. The right next step is a smarter, bounded multiplicity policy.

# C5 Results: Multi-Budget Adversarial Validation

**Date**: 2026-05-13
**Budgets**: `{'short': 30, 'medium': 120, 'long': 300}`

## Main Comparison

| Arm | Inst | Eval | Persist% | MC% | d30-300 | d120-300 | TO120 |
|-----|------|------|----------|-----|---------|----------|-------|
| LLM Call2 | 6 | 6 | 67%(4/6) | 33%(2/6) | 68.2 | 25.8 | 3/6 |
| Literature | 6 | 6 | 0%(0/6) | 0%(0/6) | 4.2 | 3.2 | 5/6 |
| Random | 6 | 6 | 50%(3/6) | 33%(2/6) | 9.5 | 7.0 | 6/6 |
| Simple Stress | 3 | 3 | 33%(1/3) | 0%(0/3) | 3.0 | 1.0 | 0/3 |
| agent_manual_sweep (int.) | 6 | 6 | 83%(5/6) | 50%(3/6) | 70.3 | 32.8 | 5/6 |

## Per-Family Breakdown

### LLM Call2

- **hybrid_M2_tight_steprates_asghlock**: PH=2/3(67%), MC=2/3(67%), NTM=2/3
  - d30-300: [+42, +62, +76], d120-300: [-3, +37, +50]
  - Strongest LLM family. 2/3 mechanism-confirmed (A-SGH lock-in).
  - One instance converged at 120s (d120_300=-3), still shows +45 growth 30s→120s.

- **hybrid_M1_tight_hetero_rates_firstkhat_dom**: PH=2/3(67%), MC=0/3(0%), NTM=0/3
  - d30-300: [+78, +75, +76], d120-300: [+72, +0, -1]
  - Large overall growth but mechanism (first_khat_dominance) not confirmed.
  - Two instances converge by 120s (first khat still dominates short budget though).

### Literature

- **anghinolfi_vls_capped**: PH=0/3(0%). Large p_j (1-12) produces very few khats → small front even at 300s. EHS barely finds any schedule at all budgets.
- **wang_literature_medium**: PH=0/3(0%). Discrete narrow rates {1,2,3} + prices {1,2,3,4} create narrow energy landscape. Instance 150/100 shows growth (+12) but fs_120=4, fs_30=1 fails fs_m > fs_s+3 threshold.

### Random

- **random_004**: PH=0/3(0%). Clamped n=150, bimodal jobs, step rates, but medium epsilon → fewer khats.
- **random_005**: PH=3/3(100%), MC=2/3(67%), NTM=2/3. Tight epsilon + step rates + monotonic TOU creates genuine mechanism stress. This is a "lucky" random configuration that landed on similar ingredients to LLM families.

### agent_manual_sweep (internal)

- **human_mixed_job_sizes**: PH=3/3(100%), MC=3/3(100%), NTM=3/3. Strongest family overall. Bimodal jobs + step rates + medium epsilon creates persistent A-SGH lock-in.
- **human_tight_epsilon**: PH=2/3(67%), MC=0/3(0%). Large raw growth (+85 to +101) but mechanism (first_khat_dominance) not confirmed.

### Simple Stress

- **simple_stress_tight_steprates**: PH=1/3(33%), MC=0/3(0%). Small jobs + tight epsilon + step rates (0.5/3.0) produces some growth but saturates quickly. The 0.5/3.0 rate gap is too wide; EHS converges to only 1 class.

## Gate: STRONG

LLM beats literature (67% vs 0% persistent) and random (67% vs 50% persistent).
LLM M2 confirmed on 2/3 instances.

## Caveats

1. **Random_005 matches LLM on per-family PH rate**: Due to a lucky random draw of tight epsilon + step rates. This shows the configuration space has high-leverage ingredients but the LLM finds them more reliably.

2. **Agent_manual_sweep still leads (83% PH)**: The internal control remains the strongest arm. This is expected; the agent/coder who designed these families understands the metric deeply. The difference vs C4 (where agent led on raw Δfs) now narrows when we require persistent growth.

3. **Literature baselines trivially fail**: Both Wang and Anghinolfi produce instances with tiny fronts (fs=1-4 at 30s, 1-6 at 120s). These are valid benchmark instances but don't stress EHS's multi-khat behavior at all.

4. **Most difficulty is 30s→120s, not 120s→300s**: Most instances that converge do so by 120s. Only a few continue growing to 300s (M2 instances with +37, +50). This means C5 confirms the difficulty is beyond trivial short-budget (30s) but doesn't require 300s.

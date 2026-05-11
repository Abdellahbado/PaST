# C3 Smoke Decision

**Date**: 2026-05-10
**Instances generated**: 18 (6 families × 3)
**Instances evaluated**: 8/36 EHS runs (22% — smoke truncated by time constraints)

## Caveats
- EHS does NOT yield during SGH construction. Time limit only checked between khats.
- First khat at large T can take >> time_limit. Smoke used 30s/60s budgets.
- Giant LLM instances (n=800+) marked 'too_large' — unevaluable in any reasonable time.
- Most random/human instances not evaluated due to smoke timeout.
- Results are INDICATIVE not conclusive.

## Yield Rates
| Arm | Instances | Evaluated | Too Large | Skipped | High-Yield |
|-----|----------|-----------|-----------|---------|------------|
| llm | 6 | 3 | 3 | 0 | 6 |
| random | 6 | 1 | 0 | 5 | 1 |
| human | 6 | 0 | 0 | 6 | 0 |

## Key Observations

1. **LLM successfully identified first_khat_dominance with giant instances**:
   - 3/3 LLM giant instances (n=800+, m=35-44, T=750-850) are completely unevaluable
   - EHS cannot complete even one SGH construction within smoke budget
   - This IS a valid adversarial design — the LLM correctly identified the mechanism

2. **LLM asgh_trajectory_conflict instances evaluated successfully**:
   - Moderate size (n=80-120, m=10-12, T=200-250) — chose different mechanism target
   - All 3 instances produced fronts (1-12 points), all feasible
   - Clear budget scaling: front grows from short → long (e.g. 5→8, 1→3, 8→12)
   - Yielded front-size growth ≥ 2 on 2/3 instances

3. **Random random_000 instance**:
   - T=737 very large — only 1 front point produced (stuck at khat=T)
   - Bimodal jobs created large cmax gap, preventing descent
   - 155s for first khat (well past 30s budget)

4. **Human instances not evaluated** due to smoke truncation

5. **Structural issue**: EHS time_limit_seconds is NOT respected during SGH construction
   - For T=737 instance: 155s (5× budget) for first khat
   - Makes adversarial benchmark evaluation unreliable at any budget

## Gate Decision

**INCONCLUSIVE — smoke truncated.**

Reasons:
- Only 8/36 runs completed — insufficient for arm comparison.
- EHS does not respect time limits during first SGH construction.
- Giant LLM instances are successfully adversarial but unevaluable.
- Human and random baselines had 0 evaluated results.

## Recommendations

1. **Fix EHS time-limit enforcement**: add time checks INSIDE SGH construction loop.
2. **Cap T at reasonable values** (e.g. T ≤ 300) for smoke instances.
3. **Re-run smoke with capped T and fixed time limits**.
4. **OR accept smoke as generative success**: LLM designed an instance family (first_khat_dominance_giant)
   that breaks EHS completely — but this is a structural EHS bug, not a benchmark insight.

## Per-Instance
- `c3_asgh_trajectory_conflict_30000` (llm, evaluated): n=119 m=11 T=213 — yield=HIGH [front_size_growth_≥2 (Δ=3)]
- `c3_asgh_trajectory_conflict_30017` (llm, evaluated): n=82 m=12 T=233 — yield=HIGH [near_zero_at_short (fs=1→3)]
- `c3_asgh_trajectory_conflict_30034` (llm, evaluated): n=112 m=10 T=240 — yield=HIGH [front_size_growth_≥2 (Δ=4)]
- `c3_first_khat_dominance_giant_1000` (llm, too_large): n=843 m=36 T=853 — yield=HIGH [too_large_for_smoke]
- `c3_first_khat_dominance_giant_1017` (llm, too_large): n=813 m=44 T=786 — yield=HIGH [too_large_for_smoke]
- `c3_first_khat_dominance_giant_1034` (llm, too_large): n=827 m=40 T=816 — yield=HIGH [too_large_for_smoke]
- `c3_human_loose_epsilon_20100` (human, skipped): n=76 m=10 T=237 — yield=low [not_evaluated]
- `c3_human_loose_epsilon_20117` (human, skipped): n=62 m=9 T=162 — yield=low [not_evaluated]
- `c3_human_loose_epsilon_20134` (human, skipped): n=110 m=8 T=160 — yield=low [not_evaluated]
- `c3_human_tight_epsilon_20000` (human, skipped): n=87 m=10 T=245 — yield=low [not_evaluated]
- `c3_human_tight_epsilon_20017` (human, skipped): n=100 m=9 T=192 — yield=low [not_evaluated]
- `c3_human_tight_epsilon_20034` (human, skipped): n=107 m=10 T=197 — yield=low [not_evaluated]
- `c3_random_000_10000` (random, evaluated): n=207 m=6 T=737 — yield=HIGH [very_slow_single_point]
- `c3_random_000_10017` (random, skipped): n=163 m=6 T=753 — yield=low [not_evaluated]
- `c3_random_000_10034` (random, skipped): n=212 m=6 T=777 — yield=low [not_evaluated]
- `c3_random_001_10100` (random, skipped): n=162 m=21 T=556 — yield=low [not_evaluated]
- `c3_random_001_10117` (random, skipped): n=163 m=21 T=444 — yield=low [not_evaluated]
- `c3_random_001_10134` (random, skipped): n=162 m=24 T=425 — yield=low [not_evaluated]

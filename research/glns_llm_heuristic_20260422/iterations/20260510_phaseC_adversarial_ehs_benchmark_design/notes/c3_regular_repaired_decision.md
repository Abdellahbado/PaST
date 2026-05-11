# C3-Regular Repaired Decision

**Date**: 2026-05-10
**Instances**: 18 (6 per arm, n≤150, T≤200)
**Budgets**: 30s / 90s

## Generation Quality

| Arm | Instances | Evaluable | Pairs | Rate |
|-----|----------|-----------|-------|------|
| llm | 6 | 6 | 12 | 100% |
| random | 6 | 6 | 12 | 100% |
| human | 6 | 6 | 12 | 100% |

## Adversarial Yield

| Arm | Evaluable | High-Yield | Rate |
|-----|----------|------------|------|
| llm | 6 | 5 | 83% |
| random | 6 | 4 | 67% |
| human | 6 | 6 | 100% |

## Gate: **FAIL** — Human sweep families (100% yield) beat LLM families (83% yield).

Human families (tight/loose epsilon sweeps) produced strong front growth
on all 6 instances because uniform processing times (p_j=1-10) allow
many khat iterations within budget, giving large front-size growth
(Δfs from +7 to +33). This is a structural property of the sweep
design, not an adversarial insight about EHS weaknesses.

LLM families correctly identified EHS mechanisms (A-SGH lock-in,
ES exploration tension) and designed targeted instances. However,
the adversarial yield (83%) is below the human sweep baseline (100%).

Random families (67%) are between LLM and human on yield, confirming
that the yield metric is sensitive to instance size and structure
rather than mechanism-specific adversarial design.

## Per-Instance

-  (llm, asgh_lock_in): n=85 m=6 T=200 — fs=1→5 cmax=200→196 HIGH [fs_growth_+4]
-  (llm, asgh_lock_in): n=91 m=12 T=200 — fs=2→8 cmax=199→193 HIGH [fs_growth_+6]
-  (llm, asgh_lock_in): n=99 m=12 T=200 — fs=4→7 cmax=197→194 HIGH [fs_growth_+3]
-  (llm, es_exploration_tension): n=98 m=8 T=200 — fs=1→1 cmax=200→200 low [saturated_1]
-  (llm, es_exploration_tension): n=93 m=8 T=200 — fs=2→5 cmax=199→196 HIGH [fs_growth_+3]
-  (llm, es_exploration_tension): n=92 m=8 T=200 — fs=1→2 cmax=200→199 HIGH [very_slow_little_output]
-  (human, epsilon_skip): n=98 m=10 T=200 — fs=3→11 cmax=198→190 HIGH [fs_growth_+8]
-  (human, epsilon_skip): n=63 m=8 T=159 — fs=20→53 cmax=140→107 HIGH [fs_growth_+33]
-  (human, epsilon_skip): n=88 m=11 T=200 — fs=4→13 cmax=197→188 HIGH [fs_growth_+9]
-  (human, first_khat_dominance): n=99 m=8 T=200 — fs=4→17 cmax=197→184 HIGH [fs_growth_+13]
-  (human, first_khat_dominance): n=87 m=8 T=196 — fs=1→8 cmax=196→189 HIGH [fs_growth_+7]
-  (human, first_khat_dominance): n=81 m=10 T=174 — fs=6→27 cmax=169→148 HIGH [fs_growth_+21]
-  (random, short_budget_pressure): n=110 m=10 T=98 — fs=3→8 cmax=96→91 HIGH [fs_growth_+5]
-  (random, short_budget_pressure): n=38 m=8 T=102 — fs=56→77 cmax=37→16 HIGH [fs_growth_+21]
-  (random, short_budget_pressure): n=105 m=6 T=102 — fs=1→1 cmax=102→102 HIGH [very_slow_little_output]
-  (random, load_imbalance): n=21 m=9 T=109 — fs=27→27 cmax=20→20 low [saturated_27]
-  (random, load_imbalance): n=55 m=8 T=99 — fs=24→49 cmax=76→51 HIGH [fs_growth_+25]
-  (random, load_imbalance): n=44 m=9 T=86 — fs=51→51 cmax=36→36 low [saturated_51]

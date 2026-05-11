# C3-Regular Smoke Decision

**Date**: 2026-05-10
**Instances**: 18 (n≤150, T≤200)
**Budgets**: 30s / 90s
**EHS time-limit fix**: cooperative deadline checks in SGH (every 5 jobs) and A-SGH loops

## Raw Yield Rates

| Arm | Instances | Evaluable | Infeasible | Incomplete | High-Yield | Rate (evaluable) |
|-----|----------|-----------|------------|------------|------------|------------------|
| llm | 6 | 6 | 0 | 0 | 5 | 83% |
| random | 6 | 1 | 3 | 2 | 1 | 100% |
| human | 6 | 0 | 0 | 6 | 0 | N/A |

## Honest Assessment: INCONCLUSIVE

**The comparison is not valid.** Reasons:

1. **Random arm broken by T≤200 cap**: random_000 was designed for T=728-786 with bimodal jobs. Capping to T=200 made all 3 instances infeasible. random_001 with n=150, m=15-24 was evaluable but very slow (219s for 30s budget). Only 1/6 random instances evaluable.

2. **Human arm not evaluated**: All 6 human instances were skipped due to evaluation timeout. The eval script ran random_001 first (slow) before reaching human instances.

3. **LLM arm was the only robust one**: 6/6 LLM instances evaluable, 5/6 high-yield. LLM designed instances appropriately sized for the eval constraints.

4. **Single-evaluable-instance comparison is meaningless**: Random 1/1 (100%) vs LLM 5/6 (83%) says nothing about relative quality when 5/6 random instances couldn't even be evaluated.

## Per-Instance

### LLM (5/6 high-yield)

- `c3r_asgh_trajectory_conflict_30000` (asgh_lock_in): n=85 m=6 T=200 — HIGH fs=1→4 [fs_growth+3]
- `c3r_asgh_trajectory_conflict_30017` (asgh_lock_in): n=91 m=12 T=200 — HIGH fs=2→8 [fs_growth+6]
- `c3r_asgh_trajectory_conflict_30034` (asgh_lock_in): n=99 m=12 T=200 — HIGH fs=4→10 [fs_growth+6]
- `c3r_es_local_optima_trap_extreme_rates_50000` (es_exploration_tension): n=98 m=8 T=200 — low fs=1→1 [saturated]
- `c3r_es_local_optima_trap_extreme_rates_50017` (es_exploration_tension): n=93 m=8 T=200 — HIGH fs=2→5 [fs_growth+3]
- `c3r_es_local_optima_trap_extreme_rates_50034` (es_exploration_tension): n=92 m=8 T=200 — HIGH fs=1→2 [very_slow]

### Random (1 evaluable)
- `c3r_random_000_*` (3×): all infeasible (bimodal jobs too large for T=200)
- `c3r_random_001_10100`: n=150 m=15 T=200 — HIGH fs=1→1 [very_slow, 219s]
- `c3r_random_001_*` (2×): eval incomplete

### Human (0 evaluable)
- All 6 instances: eval incomplete (time constraints)

## Gate: INCONCLUSIVE

- Cannot determine relative quality when 11/18 non-LLM instances couldn't be evaluated
- LLM demonstrated strong mechanism targeting (5/6 high-yield) with interpretable scaling
- But fair comparison requires redesigning baseline arms for the capped evaluation scale

## Recommendations

1. **Regenerate random families with T≤200 constraint** (not capped from original)
2. **Evaluate human instances first** (they're fastest)
3. **OR accept that LLM's mechanism-aware design is qualitatively different from random parameter sampling**

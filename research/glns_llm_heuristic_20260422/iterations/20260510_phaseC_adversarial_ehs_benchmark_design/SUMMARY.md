# Phase C Summary

**Status**: C3-Regular repaired complete. Gate FAIL.

**Branch**: LLM-guided adversarial benchmark design for EHS.

## C3-Regular Repaired Results (18 instances, n≤150, T≤200, 30s/90s)

### Generation Quality

| Arm | Instances | Evaluable | Pairs | Rate |
|-----|----------|-----------|-------|------|
| llm | 6 | 6 | 12 | 100% |
| random | 6 | 6 | 12 | 100% |
| human | 6 | 6 | 12 | 100% |

All arms had 100% generation quality after repaired generation with feasibility enforcement.

### Adversarial Yield

| Arm | Evaluable | High-Yield | Rate |
|-----|----------|------------|------|
| human | 6 | 6 | **100%** |
| llm | 6 | 5 | 83% |
| random | 6 | 4 | 67% |

### Gate: FAIL
Human sweep families beat LLM on adversarial yield (100% > 83%).

Human advantage is structural: uniform p_j=(1,10) allows many khat iterations within budget, producing large front-size growth (Δfs +7 to +33). This is a property of the sweep design, not an adversarial insight about EHS weaknesses.

LLM families correctly targeted mechanisms (A-SGH lock-in, ES exploration tension) but the yield metric favors families that produce many schedules within a time budget, which is trivially achieved by small, uniform processing times.

## Recommendation
- LLM mechanism-aware design shows qualitative sophistication but doesn't beat simple parameter sweeps on front-size growth yield
- Consider whether front-size growth is the right adversarial metric, or whether mechanism-matching qualitative evidence matters
- Do not proceed to C4/C5 full campaign with current yield metric

# Phase C Summary

**Status**: C3 smoke complete (truncated). Gate INCONCLUSIVE.

**Branch**: LLM-guided adversarial benchmark design for EHS.

## Completed
- C0-C1: Protocol + schema
- C2: Random + human generators, LLM prompt
- C3A: DeepSeek call — 8/8 families valid
- C3B: Family selection (2+2+2)
- C3C: Instance generation (18 instances, all feasible)
- C3D: EHS evaluation — 8/36 runs completed (truncated)
- C3E: Arm comparison — LLM 100% yield vs random 17% (indicative only)

## Key Finding
`run_ehs()` does NOT respect `time_limit_seconds` during SGH construction. First khat can exceed budget by 5×. This blocks reliable adversarial evaluation.

## Recommended Next
1. Add per-iteration time checks inside SGH construction loop
2. Cap T ≤ 300 for smoke instances
3. Re-run smoke with 30s/120s budgets
4. OR: accept that incomplete smoke is sufficient evidence to proceed

## Gate
**INCONCLUSIVE** — insufficient EHS runs for arm comparison. But LLM 100% yield vs 17% random is directionally supportive.

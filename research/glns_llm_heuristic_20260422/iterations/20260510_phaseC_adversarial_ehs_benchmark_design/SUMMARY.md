# Phase C Summary

**Status**: C3 smoke complete. Gate INCONCLUSIVE (baseline arms broken by instance-size caps).

## Completed
- C0-C1: Protocol + schema
- C2: Random + human generators, LLM prompt, DeepSeek call (8/8 valid families)
- C3-Scalability (Track B): LLM giant family correctly identified EHS non-interruptibility
- C3-Regular (Track A): EHS time-limit fix, 18 instances, 30s/90s eval

## C3-Regular Results

| Arm | Evaluable | High-Yield | Rate |
|-----|----------|------------|------|
| LLM | 6/6 | 5 | 83% |
| Random | 1/6 | 1 | 100% |
| Human | 0/6 | 0 | N/A |

**Gate: INCONCLUSIVE** — only 1/12 non-LLM instances evaluable. Cannot compare arms.

## LLM Strengths
- 6/6 instances evaluable vs 1/12 for baselines (robustness across caps)
- `asgh_trajectory_conflict`: 3/3 high-yield, clear front growth from short→long budget
- `es_local_optima_trap_extreme_rates`: 2/3 high-yield, mechanism-targeted

## Infrastructure Fix
- EHS `run_ehs()` now respects time_limit during SGH construction (cooperative deadline checks)

## Recommended Next
- Regenerate random families with built-in T≤200 constraint
- Evaluate human instances first (fastest)
- OR accept qualitative LLM > random evidence from mechanism-aware design

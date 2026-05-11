# C3-Scalability: Giant First-Khat Dominance Diagnostic

**Date**: 2026-05-10
**Track**: C3-Scalability (side diagnostic, not counted in main yield metric)

## Finding

The LLM-designed `first_khat_dominance_giant` family (n=813-843, m=36-44, T=753-853) exposed a structural EHS weakness: **`run_ehs()` cannot be interrupted during the first SGH construction call**, regardless of `time_limit_seconds` setting.

## Evidence

From the C3 smoke pilot (May 10):
- T=737 instance (random_000): 155s for first khat with 30s budget (5× overrun)
- T=800+ instances (LLM giant): completely unevaluable — cannot complete even one khat
- The LLM correctly identified that making the first khat extremely expensive would break EHS

## Root Cause

`split_greedy_heuristic()` runs the SGH construction in a single call without yielding to the time-limit check. The time-limit check in `run_ehs()` only happens between khat iterations (line 872 of paper_heuristics.py). SGH construction is O(n*m*T) and dominates runtime.

## Fix Applied

Added cooperative deadline checks (`_EHS_DEADLINE`) in `split_greedy_heuristic()`:
- Check at start of each duration class
- Check every 5 jobs within a class
- Return None if deadline expired (causes clean early exit)

Added similar checks before `exchange_search_with_rescheduling` and `exact_single_machine_rescheduler`.

This fix enables C3-Regular evaluation with reasonable time-limit compliance.

## LLM Contribution

The LLM correctly:
1. Identified `first_khat_dominance` as a viable EHS failure mechanism
2. Designed instances that maximally stress this mechanism (large n, m, T)
3. The mechanism hypothesis was correct — EHS indeed fails in exactly this way

## Significance

- This is NOT a novel EHS weakness — it's a known property of SGH construction cost
- The LLM's value here is in correctly identifying and exploiting it for benchmark design
- The time-limit fix was needed for fairness in C3-Regular, NOT because the LLM was wrong
- This finding should be reported separately from the C3-Regular yield comparison

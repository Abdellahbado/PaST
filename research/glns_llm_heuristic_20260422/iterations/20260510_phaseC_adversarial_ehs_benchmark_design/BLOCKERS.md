# Phase C Blockers

## Active Blockers

### B-C3.1: EHS time-limit enforcement
`run_ehs()` only checks `time_limit_seconds` between khat iterations, NOT during SGH construction. On large T instances (T≥500), the first SGH construction can take 5× the configured time budget. This makes adversarial benchmark evaluation unreliable:
- T=737 instance: 155s for first khat (with 30s budget)
- T=800+ instances: unevaluable
- **Fix needed**: Add time checks inside `split_greedy_heuristic()` and/or `assignment_history_sgh()`.

### B-C3.2: Giant LLM instances unevaluable
LLM correctly designed `first_khat_dominance_giant` family (n=800+, m=35-44). The instances are valid and feasible, but unevaluable due to B-C3.1. Need B-C3.1 fix first.

## Resolved Blockers

### B-C3.0: DeepSeek API unreachable via Python requests — RESOLVED
Python's `requests` library fails DNS resolution for `api.deepseek.com` in this environment. Resolved by using `curl` via `subprocess.run()` instead.

## Potential Concerns
- Smoke was truncated (8/36 runs). Need to re-run with fixed time limits.
- Human baselines had 0 evaluated results — weak comparison.

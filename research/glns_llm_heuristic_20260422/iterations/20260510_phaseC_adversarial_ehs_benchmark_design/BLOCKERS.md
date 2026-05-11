# Phase C Blockers

## Active Blockers

### B-C3.3: Baseline arms broken by instance-size caps
Random and human family generators were designed for larger parameter ranges. Capping T≤200 and n≤150 for fair C3-Regular evaluation made random_000 instances infeasible and random_001 instances too slow. Human instances weren't reached before eval timeout. Only 1/12 non-LLM instances were evaluable vs 6/6 for LLM.

**Fix**: Regenerate random families with T≤200 constraint built-in, or use different random families that naturally fit the eval scale. Evaluate human instances first (they're fastest).

### B-C3.2: Giant LLM instances unevaluable — RESOLVED
LLM's `first_khat_dominance_giant` family (n=800+) correctly identified a real EHS weakness but is ineligible for C3-Regular yield comparison. Documented in C3-Scalability note.

## Resolved Blockers

### B-C3.1: EHS time-limit enforcement — RESOLVED
Added cooperative deadline checks (`_EHS_DEADLINE`) in `split_greedy_heuristic()` (every 5 jobs), `assignment_history_sgh()`, and before post-SGH operations. `run_ehs()` now sets and clears the deadline. Overruns reduced from 5× budget to ≤40% of budget for n≤150, T≤200.

### B-C3.0: DeepSeek API unreachable via Python requests — RESOLVED
Resolved by using `curl` via `subprocess.run()`.

## Potential Concerns
- C3-Regular comparison is inconclusive due to baseline arms being broken by instance-size caps
- Full campaign would need 8+ hours of EHS runtime even with T≤200

# Phase C Blockers

## Active Blockers

None. C4 validation complete.

## C4 Observations

- **Budget too short for mechanism discrimination**: All 60 instances timed out on 30s short budget. Δfs differences reflect epsilon granularity, not adversarial stress. Consider 60s/120s for any follow-up.
- **NT-MC-HY gap (25% vs 50%)**: LLM mechanism families M1, M3, M4 got 0/5 mechanism confirmation. Operational rules may be too strict, or mechanisms too subtle at 30s.

## Resolved Blockers

### B-C3.3: Baseline arms broken by instance-size caps — RESOLVED
Regenerated random/human families with T≤200 constraint.

### B-C3.2: Giant LLM instances unevaluable — RESOLVED
Documented in C3-Scalability note.

### B-C3.1: EHS time-limit enforcement — RESOLVED
Cooperative deadline checks in `glns/paper_heuristics.py`.

### B-C3.0: DeepSeek API unreachable via Python requests — RESOLVED
curl via subprocess.

## Potential Concerns (for C5 or thesis)

- Budget re-calibration needed for mechanism-rich evaluation
- NT-MC-HY operational rules need refinement
- C5 full campaign would require 8+ hours of EHS runtime

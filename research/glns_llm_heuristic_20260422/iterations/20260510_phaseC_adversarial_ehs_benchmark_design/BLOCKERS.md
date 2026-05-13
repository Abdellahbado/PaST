# Phase C Blockers

## Active Blockers

None. C5 validation complete. Phase C research thread is closed.

## C5 Observations (2026-05-13)

- **Multi-budget confirms genuine difficulty**: LLM M2 family shows persistent A-SGH lock-in from 30s to 120s/300s.
- **random_005 is a lucky configuration**: Tight epsilon + step rates from random generation. Shows the configuration space has high-leverage ingredients. Mitigating: LLM finds them more consistently.
- **Literature baselines trivially fail**: Wang/Anghinolfi produce instances with tiny fronts at all budgets. Good for thesis framing.
- **Agent_manual_sweep still leads**: 83% PH vs 67% LLM. Reported honestly as internal control.

## Resolved Blockers

### B-C3.3: Baseline arms broken by instance-size caps — RESOLVED
Regenerated random/human families with T≤200 constraint.

### B-C3.2: Giant LLM instances unevaluable — RESOLVED
Documented in C3-Scalability note.

### B-C3.1: EHS time-limit enforcement — RESOLVED
Cooperative deadline checks in `glns/paper_heuristics.py`.

### B-C3.0: DeepSeek API unreachable via Python requests — RESOLVED
curl via subprocess.

### B-C4: Short budget too short for mechanism discrimination — RESOLVED by C5
C5 added 120s/300s budgets. LLM M2 confirmed on 2/3 instances under persistent-hard metric.

### B-C4.2: NT-MC-HY gap between LLM and agent — RESOLVED
Gap narrows under multi-budget metric (LLM 33% MC vs agent 50% MC at C5).

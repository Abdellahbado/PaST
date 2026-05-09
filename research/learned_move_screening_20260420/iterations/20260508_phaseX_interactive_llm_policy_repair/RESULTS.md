# Results — Phase X

## X2 Smoke (2026-05-08)

Full smoke on 3 dev cells × 6 arms (trimmed, LLM exc, random exc, score escape,
phaseX example_policy, phaseX random_policy_001). All 18 runs produced feasible
results.

| Inst/Eps | Trimmed | LLM Exc | Random Exc | Score Esc | PhaseX Example | PhaseX Random |
|:--------:|--------:|--------:|----------:|---------:|:--------------:|:-------------:|
| 61/347 | 6884 | 6884 | 6870 | 6884 | 6884 | 6884 |
| 62/290 | 9687 | 9687 | 9687 | 9489 | **9484** | 9503 |
| 65/195 | 27031 | 27031 | 27031 | 26508 | **26508** | 26749 |

Key observations:
- PhaseX example_policy (llm_score, no escape, budget 4→12, quotas=3)
  matches or beats LLM exception on all 3 cells. On 62/290 it improves
  by -203 vs trimmed (-203 vs LLM exc). On 65/195 it improves by -523
  vs trimmed (-523 vs LLM exc). The DSL-based implementation is verified.
- PhaseX example_policy matches score_escape on 61/347 and 65/195,
  beats it slightly on 62/290 (9484 vs 9489).
- Random policy (random_policy_001) produces intermediate results: ties
  baselines on 61/347, partial improvement on 62/290 (9503 vs 9687),
  and partial improvement on 65/195 (26749 vs 27031).
- PhaseX generic runner confirmed functional: C++ reads policy JSON,
  applies 17-field DSL, produces valid exception-lane behavior with
  exact DP verification.

Smoke files:
- `eval/x2_smoke_raw.csv` — all 70 CSV fields for 18 runs
- `eval/x2_smoke_summary.csv` — condensed 32-field summary
- `policies/example_policy.json` — llm_score baseline
- `policies/random_policy_001.json` — generated random policy

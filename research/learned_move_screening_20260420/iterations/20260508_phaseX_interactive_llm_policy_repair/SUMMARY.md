# Summary — Phase X: Interactive LLM Policy Repair

## Status

X2 COMPLETED. Generic C++ PhaseXPolicyJson runner works. Python orchestration
script functional. Full 3-cell × 6-arm smoke passed. PhaseX example_policy equals
or beats both trimmed and LLM exception on all cells.

## Pipeline

| Step | Status |
|------|--------|
| X0 — Initialize | COMPLETED |
| X1 — Define DSL | COMPLETED |
| X2 — Generic policy runner | COMPLETED |
| X3 — Fast dev evaluation | Not started |
| X4 — Interactive LLM loop | Not started |
| X5 — Compare against controls | Not started |
| X6 — Validation | Not started |

## X2 Smoke Results

| Inst/Eps | Trimmed | LLM Exc | Random Exc | Score Esc | Example | Random |
|:--------:|--------:|--------:|----------:|---------:|-------:|-------:|
| 61/347 | 6884 | 6884 | 6870 | 6884 | 6884 | 6884 |
| 62/290 | 9687 | 9687 | 9687 | 9489 | **9484** | 9503 |
| 65/195 | 27031 | 27031 | 27031 | 26508 | **26508** | 26749 |

- PhaseX example_policy matches or beats LLM exception on all cells
- Matches score_escape on 61/347 and 65/195, beats it on 62/290
- Random policy shows intermediate results (no catastrophic regressions)

## Hypothesis

Interactive LLM repair (proposal → eval → feedback → revise) will outperform
one-shot LLM generation and random policy search on a constrained policy DSL.

## Key Design Decisions

- Policy DSL (JSON) constrains LLM output — no arbitrary C++
- 3 fast dev cells for rapid feedback (seconds per eval)
- 5 interaction rounds
- Generic C++ policy runner (no per-round implementation)
- Success defined vs one-shot LLM + random DSL median, not vs EHS

## Artifacts

- `scripts/phaseX_interactive_policy_repair.py`
- `policies/schema.json`, `policies/example_policy.json`
- `policies/random_policy_001.json`
- `eval/x2_smoke_raw.csv`, `eval/x2_smoke_summary.csv`
- `notes/phaseX_policy_dsl.md`

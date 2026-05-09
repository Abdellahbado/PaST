# Summary — Phase X: Interactive LLM Policy Repair

## Status

X4 COMPLETED. 5-round interactive DeepSeek policy repair. Verdict: MINIMUM SUCCESS.
Best LLM policy (Round 2) beats example_policy (-6.3) and random median (-76.3),
does NOT beat random best c000 (+31.4).

## Pipeline

| Step | Status |
|------|--------|
| X0 — Initialize | COMPLETED |
| X1 — Define DSL | COMPLETED |
| X2 — Generic policy runner | COMPLETED |
| X3 — Random DSL baseline campaign | COMPLETED (Case B) |
| X4 — Interactive LLM loop | COMPLETED (MINIMUM SUCCESS) |
| X5 — Compare against controls | Not started |
| X6 — Validation | Not started |

## X4 Outcome

| Metric | Value |
|--------|------:|
| Best LLM round | Round 2 (14285.7) |
| Δ vs example_policy (14292.0) | -6.3 |
| Δ vs random median (14362.0) | -76.3 |
| Δ vs random best c000 (14254.3) | +31.4 |
| Rounds beating example | 2/5 |
| Rounds beating random best | 0/5 |
| Verdict | MINIMUM SUCCESS |

### Key Findings

1. **Interactive feedback works**: LLM improved from +8.7 (R0) to -6.3 (R2)
   by diagnosing per-cell bottlenecks and making targeted changes.
2. **LLM efficiency**: Found a beating policy in 3 attempts (Round 2). Random
   needed 20 attempts, only 2/20 hit.
3. **DSL constraints are binding**: The `max_per_target ≤ 4` cap prevented
   further improvement (Round 3 proposal was capped).
4. **Guard cell breakthrough**: Round 4's hybrid mode achieved the first
   improvement on 61/347 (6873 vs 6884 baseline), suggesting cheap_lb_delta
   signal is informative on tight cells.
5. **Cell trade-off**: Hybrid mode improved tight cells but regressed the
   primary cell — a single fixed-weight scoring cannot optimally serve all
   epsilon regimes.

### Best Policy (Round 2)

llm_score normal mode + cheap_lb_pair escape, max_per_source=4, max_per_target=4,
require_positive_cheap_lb=false, budget 3→11, grow_on_hit=3.

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

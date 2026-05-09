# Summary — Phase X: Interactive LLM Policy Repair

## Status

**PHASE X STOPPED as main LLM-critical direction (2026-05-10).**

X5 WEAK signal closes the branch: interactive LLM DSL tuning does NOT
outperform random best-of-5 under the same 5-attempt budget. X6 validation
on held-out cells is NOT justified. The branch is now archived.

## Why Stopped

- X4 MINIMUM SUCCESS proved interactive feedback works (LLM found beating
  policy in 2 rounds vs random needing 2/20 individual attempts).
- X5 WEAK reversed the comparison: when random gets the same 5-attempt
  budget and picks best-of-5, it beats LLM 75% of the time (15/20 batches).
  LLM at 20th percentile (14285.7) vs random best-of-5 median (14265.0).
- The DSL is flat enough that brute-force random dominates interactive repair.
- Further DSL complexity or more interaction rounds do not address the core
  weakness — the LLM is not generating insight beyond what random search finds.

## What Survives

- DSL-based policy optimization is a valid concept (Case B).
- Interactive LLM feedback is more efficient than single-shot generation (X4
  finding beating policies in fewer total attempts than random).
- Guard-cell breakthrough (hybrid mode: 61/347 6884→6873) shows cell-adaptive
  scoring has potential, but this is a DSL-mechanism observation, not an
  LLM-interaction one.

## What Does Not Survive

- Claim that interactive LLM beats brute-force random under equal budget.
- Validation on held-out cells (proposed in notes/x5_validation_cells.md
  but not executed — not justified given X5 WEAK signal).
- Phase X as the main paper narrative.

## Pipeline

| Step | Status |
|------|--------|
| X0 — Initialize | COMPLETED |
| X1 — Define DSL | COMPLETED |
| X2 — Generic policy runner | COMPLETED |
| X3 — Random DSL baseline campaign | COMPLETED (Case B) |
| X4 — Interactive LLM loop | COMPLETED (MINIMUM SUCCESS) |
| X5 — Random best-of-5 distribution | COMPLETED (WEAK) |
| X6 — Validation | SKIPPED (not justified) |

## X5 Random Best-of-5 Distribution

| Metric | Value |
|--------|------:|
| LLM best-of-5 mean TEC | 14285.7 |
| Random median best-of-5 | 14265.0 |
| Random Q1 best-of-5 | 14252.3 |
| Random Q3 best-of-5 | 14293.3 |
| Random IQR | 41.0 |
| Random beating LLM | 15/20 (75%) |
| LLM percentile rank | 20% |
| Signal | **WEAK** |

### Key Finding

Interactive LLM does NOT beat random best-of-5. When random is given the same
5-attempt budget and picks the best, it finds better policies 75% of the time.
The DSL is flat enough that brute force beats interactive feedback.

### Implications

- The earlier X4 efficiency claim (LLM found beating policy in 2 rounds vs
  random needing 2/20 individual attempts) holds against INDIVIDUAL random
  policies, but not against aggregate best-of-5 random search.
- The DSL is learnable but not sufficiently structured for interactive LLM
  to dominate brute-force random within the same attempt budget.
- Validation is proposed but may not be justified given WEAK signal.

## X2 Smoke Results

| Inst/Eps | Trimmed | LLM Exc | Random Exc | Score Esc | Example | Random |
|:--------:|--------:|--------:|----------:|---------:|-------:|-------:|
| 61/347 | 6884 | 6884 | 6870 | 6884 | 6884 | 6884 |
| 62/290 | 9687 | 9687 | 9687 | 9489 | **9484** | 9503 |
| 65/195 | 27031 | 27031 | 27031 | 26508 | **26508** | 26749 |

- PhaseX example_policy matches or beats LLM exception on all cells
- Matches score_escape on 61/347 and 65/195, beats it on 62/290
- Random policy shows intermediate results (no catastrophic regressions)

## Hypothesis (falsified)

Interactive LLM repair (proposal → eval → feedback → revise) will outperform
one-shot LLM generation and random policy search on a constrained policy DSL.

Falsified at X5: LLM at 20th percentile vs random best-of-5 budget parity.

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

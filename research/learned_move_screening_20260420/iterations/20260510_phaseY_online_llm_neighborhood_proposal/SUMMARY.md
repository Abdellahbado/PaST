# Summary — Phase Y: Online LLM Neighborhood Proposal

## Status

**INITIALIZED (2026-05-10).** No implementation yet. Phase Y is the active
branch following Phase X closure.

## Hypothesis

An LLM is more useful when it sees the current concrete schedule state
and proposes bounded neighborhoods to evaluate, rather than tuning a
static policy DSL (Phase X) or designing C++ operators offline from
aggregate traces (Phase V).

## Pipeline

| Step | Status |
|------|--------|
| Y0 — Trace format + state conditioning | Not started |
| Y1 — C++ variant for neighborhood evaluation | Not started |
| Y2 — Random neighborhood baseline | Not started |
| Y3 — First DeepSeek call on dev cells | Not started |
| Y4 — Held-out validation (if Y3 positive) | Not started |
| Y5 — Compare vs best prior results (if Y4 positive) | Not started |

## Key Design Decisions

- LLM called only at stagnation points, not in every solver round
- LLM receives concrete schedule state, not aggregate statistics
- LLM proposes concrete moves (source/target/job/type), not code or parameters
- Exact DP verifies all proposed moves; only verified improvements accepted
- Random neighborhood proposals as baseline (equal K budget)
- If LLM cannot beat random, stop — no further LLM-critical branches

## Distinction from Prior Phases

| Phase | LLM produces | Representation | Why it failed |
|-------|-------------|----------------|---------------|
| S | Scoring functions (C++) | Python/C++ formulas | scoring ≤ handcrafted |
| V | Operator code (C++) | C++ operator blocks | code too fragile |
| X | Policy parameters (JSON) | DSL with 17 fields | DSL too flat |
| **Y** | **Concrete neighborhoods** | **State → move proposals** | **TBD** |

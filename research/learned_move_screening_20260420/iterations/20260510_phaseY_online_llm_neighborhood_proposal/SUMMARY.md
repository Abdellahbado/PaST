# Summary — Phase Y: Online LLM Neighborhood Proposal

## Status

**Y1 COMPLETED (2026-05-10).** C++ trace instrumentation done, trace-generation
smoke passed on 3 dev cells. No DeepSeek calls, no proposal execution, no LLM
vs random comparisons.

## Hypothesis

An LLM is more useful when it sees the current concrete schedule state
and proposes bounded neighborhoods to evaluate, rather than tuning a
static policy DSL (Phase X) or designing C++ operators offline from
aggregate traces (Phase V).

## Y0 Deliverables

- `traces/schema_state_trace.md` — 6-section state trace format:
  metadata, cell regime, current snapshot, per-machine table (17 columns),
  recent search behavior, candidate pool summaries, prior arm results
- `proposals/schema_neighborhood_proposal.json` — bounded JSON proposal:
  source/target lists (max 5 each), job size classes, max_candidates 
  (≤ 30), ranking hint, diversity rule, fallback, rationale
- `notes/phaseY0_trace_and_proposal_design.md` — design rationale:
  why each field is included/excluded, constraint-to-candidate mapping,
  fairness guarantees, Y1 implementation plan
- `prompts/call0_trace_schema_review.md` — DeepSeek review prompt
  (17 questions across trace sufficiency, proposal executability,
  fairness, overfitting risk, implementation feasibility)

## Y1 Deliverables

- C++: `phaseY_trace_probe` variant (`InsertScreenMode::PhaseYTraceProbe`)
  - 1-start DiverseTrimmed core lane with trace generation
  - `write_phaseY_trace_json` helper: JSON + Markdown output
  - Trace written at end of local search (max_rounds or stagnation)
- `scripts/phaseY_neighborhood_proposal.py` with `--y1-trace-probe`
- Smoke on 3 dev cells: all produce valid traces (~3300 tokens each)
- Trace fields: regime, snapshot, machine state (17 columns, all machines),
  candidate pools (top sources/targets, job size quartiles), prior arms
- Null fields documented: core_source_hits, core_target_hits,
  last_accepted_moves, failed_move_families, underexplored sources/targets

## Pipeline

| Step | Status |
|------|--------|
| Y0 — Trace format + state conditioning | COMPLETED |
| Y1 — C++ variant for trace generation | COMPLETED |
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

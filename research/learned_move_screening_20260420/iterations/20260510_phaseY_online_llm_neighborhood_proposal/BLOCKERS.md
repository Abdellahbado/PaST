# Blockers — Phase Y: Online LLM Neighborhood Proposal

## Active

(none — Y1.1 complete)

## Resolved

### B-Y0.0 — Trace format needs C++ instrumentation before Y1 [RESOLVED 2026-05-10, Y1]

Added `phaseY_trace_probe` variant. Writes JSON + Markdown state traces at
end of DiverseTrimmed local search. Smoke passed on 3 dev cells.

### B-Y1.1 — Search-behavior fields were null [RESOLVED 2026-05-10, Y1.1]

All previously null fields now populated:
- core_source_hits, core_target_hits: per-round pool counters
- starved: derived from core_source_hits==0 AND has jobs
- underexplored_sources/targets: machines with 0 hits, sorted by cost/slack
- last_accepted_moves: ring buffer of up to 10 moves with delta_tec
- failed_summary: evaluated_exact_count + no_improving flag

## Anticipated

### B-Y0.1 — C++ variant for neighborhood evaluation not yet designed [PLANNED]

A new solver variant is needed that reads LLM JSON proposals and generates
candidate triples from source/target constraints. Scoped for Y1.

### B-Y0.2 — Random neighborhood baseline must guarantee fairness [PLANNED]

Random baseline uses same format with weighted random machine selection.
Must verify that the same initial state is used for both LLM and random.
Scoped for Y2.

### B-Y0.3 — DeepSeek overfitting to specific machine IDs on dev cells

The LLM sees exact machine costs and gaps. On dev cells, it could propose
attacking M0 simply because M0 has the highest Gap, rather than because it
diagnoses a generalizable pattern. The trace's cell_label anonymization
helps, but per-machine data is inherently instance-specific. This is the
core risk — the LLM must demonstrate diagnostic reasoning, not pattern
matching.

### B-Y0.4 — The constraint-based approach may be too coarse

If the LLM wants to propose "move small jobs from M0 to M24, but only jobs
that are currently scheduled in the first half of M0's processing order",
the current proposal schema cannot express this (no job position filter).
This may limit the LLM's effectiveness but avoids the unbounded complexity
of per-job proposals. Tradeoff accepted for Y0/Y1.

## Resolved

(none — Y0 design phase had no prior blockers)

# Blockers — Phase Y: Online LLM Neighborhood Proposal

## Active

### B-Y0.0 — Trace format needs C++ instrumentation before Y1 [PLANNED, not blocking]

The existing solver CSV output does not include per-round machine state
snapshots. Phase Y requires new C++ logging at stagnation points to produce
the state trace. This is scoped for Y1 implementation — no blocker for Y0
design.

### B-Y0.1 — C++ variant for neighborhood evaluation not yet designed [PLANNED]

A new solver variant is needed that reads LLM JSON proposals and generates
candidate triples from source/target constraints. Scoped for Y1.

### B-Y0.2 — Random neighborhood baseline must guarantee fairness [PLANNED]

Random baseline uses same format with weighted random machine selection.
Must verify that the same initial state is used for both LLM and random.
Scoped for Y2.

## Anticipated

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

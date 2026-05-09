# Blockers — Phase Y: Online LLM Neighborhood Proposal

## Active

(none — Phase Y not yet implemented)

## Anticipated

### B-Y0.0 — Trace format design needs to balance informativeness with DeepSeek context limit

DeepSeek V4 Pro context window is large but practical prompts should be
well under the limit. The trace format must be compact enough to leave
room for instructions and the proposal response.

### B-Y0.1 — C++ variant for neighborhood evaluation not yet designed

The current C++ solver evaluates moves from the DiverseTrimmed shortlist
and/or exception lane. A new variant is needed that accepts a list of
specific move specifications (source, target, job, type) and evaluates
them with exact DP while maintaining the core lane.

### B-Y0.2 — Random neighborhood baseline complexity

Random neighborhood proposals with equal K budget are straightforward
but must be fair: same move-type distribution, same source/target candidate
set, same K. Implementation must avoid accidentally giving random an
advantage (e.g., sampling from a larger pool than the LLM can see).

## Resolved

(none)

# Phase Y0 — Trace and Proposal Schema Design

## Overview

Phase Y0 defines the two schemas that form the contract between the solver
and the LLM:

1. **State trace schema** (`traces/schema_state_trace.md`): what the solver
   sends to the LLM at a stagnation point.
2. **Neighborhood proposal schema** (`proposals/schema_neighborhood_proposal.json`):
   what the LLM must return, specifying which moves to evaluate.

## Design Decisions

### Why Each Trace Field Is Included

The trace design prioritizes information that enables the LLM to answer:
**"Where should we look for improving moves that the core DiverseTrimmed
shortlist cannot reach?"**

The core insight from prior phases (S, V, X) is that the solver's outside
pool covers only 20% of source machines. The bottleneck is NOT ranking
within the pool — it's which sources and targets enter the pool at all.
The trace must therefore help the LLM identify **underexplored sources**
and **unused targets**.

| Field Group | Key Rationale |
|-------------|---------------|
| **Cell regime** (Section 1) | Context: scale, epsilon, machine count. Epsilon shown because it determines job count per machine (load pressure). Anonymized cell_label prevents instance overfitting. |
| **Current snapshot** (Section 2) | Tells the LLM: are we truly stuck (no_hit_streak=5), or just approaching a cap? Shows what budget remains. `core_lane_stagnation_active` is the key trigger. |
| **Machine state table** (Section 3) | **Core diagnostic**. Each machine row answers: (a) Is this machine expensive? (EC) (b) Is cheap LB wrong here? (Gap) (c) Is it cost-inefficient per unit load? (CD) (d) Can it absorb jobs? (Slack) (e) What job sizes are available? (s/m/l) (f) Is it over-attacked or ignored? (CS/CT) (g) Is it a starved source? (SL) |
| **Recent search behavior** (Section 4) | Shows what the solver tried and what failed. Last 10 accepted moves shows what worked. Failed move families shows what's exhausted. Core/outside pool composition shows diversity metrics. |
| **Candidate pool summaries** (Section 5) | Pre-computed ranked lists that the LLM can reference: top sources by cost/gap/density, top targets by slack/low_cost, underexplored sources/targets. Saves the LLM from having to re-derive these from the raw machine table. |
| **Prior arm results** (Section 6) | Oracle ceiling for dev cells. Shows the LLM what's achievable so it can calibrate ambition. NOT included for held-out cells (prevents oracle leakage). |

### Why Each Trace Field Is Excluded

Many solver-internal fields are excluded because they (a) bloat the context,
(b) encourage overfitting, or (c) are not actionable for move selection:

| Excluded | Reason |
|----------|--------|
| Raw instance ID | Overfitting prevention. Cell label only. |
| S1 score | Redundant with Gap/RLB which are richer signals. |
| Cheap-window price curve | Problem-specific; LLM does not need raw pricing. |
| Per-candidate s2 scores | Too many data points (thousands). Summarized by aggregates. |
| DP cache stats | Implementation detail; irrelevant for WHERE-to-search. |
| Full trajectory | Last 10 moves + summaries sufficient. |
| Swap/move stats | Only insert_inter is relevant for Y0/Y1. |
| CPU/runtime | Not actionable for move diagnosis. |

### How the Proposal Maps to Concrete Candidate Moves

The LLM does NOT propose individual (source, job_position, target) triples.
Instead, it proposes **constraints** that the solver expands into concrete
triples:

1. **Source list** (max 5 machines) → which machines to take jobs FROM
2. **Target list** (max 5 machines) → which machines to move jobs TO
3. **Job size classes** → filter which jobs on those sources to consider
4. **Max candidates** → how many triples to evaluate (DP budget)
5. **Ranking hint** → how to order the candidates
6. **Diversity rule** → how to ensure exploration across machines
7. **Fallback** → what to do if the constraints generate no valid candidates

The solver then:
1. Enumerates all jobs on the source machines matching the size class
2. For each (source, job) pair, finds feasible targets (slack ≥ job size)
3. Generates all (source, job, target) triples
4. Ranks by ranking_hint
5. Applies diversity rule (quota per source/target/pair)
6. Evaluates top-K with exact DP
7. Accepts only verified improvements
8. Falls back if empty

This constraint-based approach is chosen over explicit (source, job, target)
lists because:
- It prevents the LLM from generating thousands of specific triples
- It keeps the proposal compact and parseable
- The solver's internal candidate generation already handles the enumeration
- The LLM's value is in selecting WHICH machines and WHICH job types, not
  in enumerating individual jobs

### Fairness for Random Baseline

The random baseline is carefully designed to be a fair comparison:

| Aspect | LLM | Random |
|--------|-----|--------|
| Source selection | LLM-chosen, up to 5 | Random, up to 5, weighted by EC |
| Target selection | LLM-chosen, up to 5 | Random, up to 5, weighted by slack |
| Job size filter | LLM-chosen | Random subset of {s, m, l} |
| Max candidates | LLM-chosen (≤ 30) | Same value as LLM |
| Ranking | LLM-chosen hint | Fixed 'random' |
| Diversity | LLM-chosen rule | Random uniform choice |
| Fallback | LLM-chosen | Fixed 'top_s2_same_budget' |
| DP verifier | Exact DP | Exact DP |
| Initial state | Same solver state | Same solver state |

The key fairness properties:
- Same K budget for exact DP evaluations
- Same universe of (source, job, target) triples
- Same verifier (exact DP)
- Same initial schedule state
- Random source selection weighted by EC (avoids giving random an advantage
  from low-info sources)

### Why Not Per-Cell Adaptive

The proposal is NOT per-cell adaptive in the sense of using different DSL
parameters per cell. It is **instance-specific**: the LLM sees the concrete
state of ONE cell at ONE decision point and proposes for that specific state.
This is fundamentally different from Phase X, where the LLM tuned a static
policy that had to work for all cells and all states.

### What Counts as a "Hit"

A "hit" is a move that, after exact DP evaluation on both affected machines,
produces a schedule with total exact cost strictly less than the current TEC.
This is identical to the existing solver acceptance criterion. The LLM's
proposal adds no new acceptance mechanism.

### Y1 Implementation Plan (not yet implemented)

Phase Y1 requires two C++ variants:

1. **`phaseY_llm_neighborhood`**: Solver variant that:
   - Runs the normal DiverseTrimmed core lane
   - At stagnation (no_hit_streak ≥ N), snapshots the state trace to a file
   - Reads the LLM's neighborhood proposal JSON
   - Generates candidate triples from the proposal constraints
   - Evaluates with exact DP
   - Accepts verified improvements
   - Falls back on empty/invalid proposals

2. **`phaseY_random_neighborhood`**: Solver variant that:
   - Runs the normal DiverseTrimmed core lane
   - At stagnation, generates a random proposal (matching the LLM's K budget)
   - Generates candidate triples from the random constraints
   - Evaluates with exact DP
   - Accepts verified improvements

Both variants run from the same initial solution and use the same initial
seed (deterministic restart) for the core lane, ensuring the comparison
is at the same solver state.

The Python orchestration script (`scripts/phaseY_neighborhood_proposal.py`)
will:
- Generate state traces by running the solver to stagnation
- Call DeepSeek with the trace + proposal schema + task description
- Parse the JSON proposal from the DeepSeek response
- Validate proposal against the schema
- Run both LLM and random variants with the same trace state
- Compare results

### Trace Generation: New C++ Instrumentation Needed

The existing solver CSV output does NOT include per-round machine state
snapshots. Phase Y requires new C++ instrumentation:

1. **Snapshot at stagnation**: When no_hit_streak ≥ N, write machine-level
   state (EC, RLB, load, job_count, processing_time_histogram, core_hits,
   etc.) to a per-round JSON file.

2. **Track core source/target hit counts**: Per-round counter of which
   machines appeared as sources or targets in the core shortlist.

3. **Track last accepted moves**: Buffer of last 10 accepted moves with
   source, target, job_size, delta_tec, was_exception.

4. **Compute outside pool composition**: Already available from existing
   exception-lane CSV fields (`outside_pool_distinct_src`, etc.).

5. **Compute processing time histograms**: Per-machine counts of jobs in
   [1-4], [5-8], [9-12] buckets. Available from `final_machine_loads` +
   per-machine job data (which the solver already has internally).

This is a bounded set of new instrumentation — no changes to the core
search algorithm, only additional logging at stagnation points.

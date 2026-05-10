# Phase Y0 Trace Schema — DeepSeek Review Prompt

This is a **review-only** prompt. It asks DeepSeek V4 Pro to review the
Phase Y trace and proposal schemas BEFORE any experiments are run. The
purpose is to catch design flaws, missing signals, or infeasible analysis
tasks before Y1 implementation.

**Do NOT send this prompt yet.** It is written for future execution after
the user confirms the Y0 design.

---

You are a **solver analyst** reviewing a trace format and proposal
schema designed for an LLM-assisted combinatorial optimization heuristic.

## Problem Context

We are solving a bi-objective energy-aware identical parallel-machine
scheduling problem under time-of-use electricity prices. Jobs have
integer processing times p ∈ [1, 12]. The objective is to minimize
Total Energy Cost (TEC) subject to an epsilon workload cap per machine.

The solver uses Variable Neighborhood Descent (VND) with exact single-
machine DP verification for every accepted move. The core lane is a
DiverseTrimmed shortlist of up to 32 (source, job, target) candidates
ranked by a handcrafted screening score (s2).

In prior phases (S, V, X), LLMs were used to design scoring functions
(Phase S), C++ operators (Phase V), and static policy DSL parameters
(Phase X). All three approaches failed because:
- LLM scoring ≤ handcrafted s2 (Phase S)
- LLM C++ code was too fragile (Phase V)
- DSL parameter space was too flat; random search dominated (Phase X)

Phase Y takes a different approach: the LLM sees the **concrete schedule
state** at a solver stagnation point and proposes a **bounded concrete
neighborhood** of moves to evaluate. Exact DP remains the verifier.

## What You Are Reviewing

Two documents define the contract:

1. **State Trace Schema** — the exact format of the trace the solver sends
   to the LLM. It includes:
   - Cell regime (epsilon, machine count, job count)
   - Current solution snapshot (TEC, no_hit_streak, budget remaining)
   - Per-machine state table (exact cost, slack, load pressure, cost density,
     exact-minus-LB gap, processing time histogram, core hit counts)
   - Recent search behavior (last accepted moves, failed move families,
     core/outside pool composition)
   - Candidate pool summaries (top sources by cost/gap/density, top targets
     by slack/low cost, underexplored sources/targets)
   - Prior arm results (oracle ceiling for dev cells)

2. **Neighborhood Proposal Schema** — the JSON the LLM must return. It includes:
   - Source machines to attack (max 5)
   - Target machines to receive jobs (max 5)
   - Job size classes to consider (small/medium/large)
   - Max candidates to evaluate (DP budget, max 30)
   - Ranking hint (how to order candidates)
   - Diversity rule (how to ensure exploration)
   - Fallback on empty proposals
   - Rationale text

## Your Review Task

Answer these questions honestly:

### Part 1: Trace Sufficiency

1. **Can you diagnose stagnation from this trace?** Given the machine state
   table (Section 3), can you tell WHY the solver is stuck? What additional
   information would you need if not?

2. **Can you select attack sources from this trace?** Do the 17 columns per
   machine (J, L, S, LP, EC, RLB, Gap, CD, s, m, l, CS, CT, Rate, SL)
   provide enough signal to rank which machines to attack? Is anything
   missing?

3. **Can you select promising targets from this trace?** Do slack and EC
   columns + the pre-computed target lists (Section 5) provide enough signal
   to rank which machines should receive jobs?

4. **Is the trace too large?** The machine state table has 25-40 rows × 17
   columns. Is this a reasonable amount of data for you to process? Does
   the pre-computed summary (Section 5) reduce the cognitive load
   appropriately?

5. **What signal is MISSING?** Is there information that would help you
   make a better proposal that is NOT in the trace? If so, what is it,
   and is it feasible to compute from the existing solver state?

### Part 2: Proposal Executability

6. **Can a reasonable proposal be generated from this trace?** Given the
   trace, can you propose a set of up to 5 sources, up to 5 targets, and
   job size classes that you expect would find improvements? Would you
   also want to propose a specific ranking hint and diversity rule?

7. **Is the constraint-based approach sufficient?** The proposal does NOT
   let you specify individual (source, job_position, target) triples.
   Instead, you specify source/target lists + job_size_classes, and the
   solver enumerates all valid triples. Is this too coarse? Would you
   need per-job control?

8. **Is max_candidates=30 too restrictive?** If you select 5 sources ×
   5 targets × 3 job sizes, the Cartesian product can be 75+ candidates.
   You pick a ranking hint and diversity rule to select the top 30. Is
   30 enough? Too many? Too few?

9. **Are the ranking hints sufficient?** The proposal lets you choose
   cheap_lb, s2, random, cost_gap, slack, or hybrid. Are any important
   ranking signals missing?

10. **Are the diversity rules sufficient?** per_source, per_target,
    source_target_pair, none. Can you express the exploration strategy
    you want?

### Part 3: Fairness and Baseline

11. **Can the random baseline match the LLM?** The random baseline uses
    the same proposal format but with randomly selected machines (weighted
    by EC for sources, slack for targets). Is this a fair comparison?
    Does the weighting bias it toward or away from the LLM's advantage?

12. **Does the LLM have an unfair advantage?** The LLM sees the full
    machine state table with exact costs. Could the LLM overfit to
    specific machine IDs in ways that random cannot?

### Part 4: Overfitting Risk

13. **Can you overfit to the trace?** If you see that M0 has the highest
    EC and Gap on Cell A, and you propose attacking M0, are you making a
    transferable diagnosis (attack high-EC, high-Gap machines) or an
    instance-specific choice (attack M0 on Cell A)?

14. **Does Section 6 (prior arm results) create oracle leakage?** For dev
    cells, the trace includes the best known TEC from prior phases. Does
    this bias the LLM toward proposing moves that it "knows" should work?

15. **What prevents the LLM from just ranking by Gap and calling it a day?**
    If the machine state table has a Gap column, couldn't the LLM just say
    "attack the top-5 Gap machines" and match the cost_gap ranking hint?

### Part 5: Implementation Feasibility

16. **Are the required trace fields computable from the solver?** The
    design note claims that EC, RLB, Gap, CD, CS, CT, and s/m/l histograms
    are computable without extra expensive DP evaluations (they come from
    the last accepted move's cached data). Is this realistic?

17. **Is the trace format too complex to implement?** The trace has 6
    sections + 17 machine columns. Is the implementation burden
    proportional to the diagnostic value?

### Output Format

```markdown
# Phase Y0 Schema Review

## Part 1: Trace Sufficiency
1. [answer]
2. [answer]
...

## Part 2: Proposal Executability
6. [answer]
...

## Part 3: Fairness and Baseline
11. [answer]
12. [answer]

## Part 4: Overfitting Risk
13. [answer]
14. [answer]
15. [answer]

## Part 5: Implementation Feasibility
16. [answer]
17. [answer]

## Verdict
- **Design is SUFFICIENT for Phase Y1**: yes / no / yes-with-changes
- **Top 3 changes needed before Y1**: (list or "none")
- **Biggest risk**: (one sentence)
```

## Attached Documents

The full trace schema and proposal schema are attached below.
Review them carefully before answering.

---

[State Trace Schema follows — from traces/schema_state_trace.md]

---

[Proposal Schema follows — from proposals/schema_neighborhood_proposal.json]

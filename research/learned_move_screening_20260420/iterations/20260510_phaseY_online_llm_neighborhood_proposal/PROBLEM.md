# Problem — Phase Y: Online LLM Neighborhood Proposal

## Hypothesis

An LLM is more useful when it sees the **current concrete schedule state**
and proposes **bounded neighborhoods to evaluate**, rather than tuning a
static policy DSL or designing C++ operators offline from aggregate traces.

Single-machine exact DP remains the verifier.

## Why This Hypothesis

Phase S (0-5D, Stage U): LLM-designed selection/scoring rules did not beat
handcrafted baselines. The LLM's value was diagnostic — it identified
mechanisms (e.g., cheap_lb_delta escape) that a human could implement — but
the human-implemented version was necessary for correctness and performance.

Phase V (V0-V5.2): The LLM produced operators from aggregate trace data,
but the best operator (score_escape_sampler) could not beat the Phase S
LLM exception lane on all cells. Runtime-level operators require C++ precision
that the LLM cannot produce.

Phase X (X0-X5): The LLM tuned a static JSON policy DSL with 17 fields via
interactive feedback. The LLM found a beating policy (MINIMUM SUCCESS at X4)
but was dominated by random brute-force search under equal budget (WEAK at X5).
The DSL is too flat — random search finds good policies 75% of the time.

**Common failure**: all three Phases (S, V, X) gave the LLM a representation
language (scoring formulas, C++ operators, policy DSL) that was either too
coarse or too flat. The LLM's diagnostic ability was empirically validated
(repeatedly) but never produced a runtime advantage over simple baselines.

**Phase Y pivot**: the LLM is given the **concrete schedule state** at a
decision point, not a representation language. Its job is to propose **which
moves to evaluate**, not to generate scoring functions or policy parameters.
This tests the LLM's diagnostic ability in the most direct setting possible:
"Here is the current state. What should we try?"

## Scope

Phase Y tests whether the LLM can propose useful concrete neighborhoods that
exact DP then verifies. The LLM is NOT asked to:
- Generate C++ code
- Tune DSL parameters
- Design scoring functions
- Learn from aggregate statistics

The LLM IS asked to:
- Receive a concrete schedule trace/state at a stagnation or decision point
- Diagnose what is happening (bottleneck sources, slack, moves)
- Propose a bounded set of concrete moves or move restrictions to try
- Receive exact-DP feedback on whether proposed moves helped

## Definition

### Input to LLM

Current schedule trace/state at a search decision point:
- Which sources/machines are tight (close to TEC contribution limit)
- Which sources/machines have slack (can absorb changes)
- Current best schedule quality
- Recent move history (what was tried, what worked, what failed)
- Outside-shortlist candidates (if available)
- Stagnation signal (how many rounds without improvement)

### Output from LLM

A concrete bounded neighborhood proposal:
- Specific sources to focus on
- Specific targets/jobs to consider
- Move type constraints (e.g., only insert_inter, only within epsilon class)
- Budget specification (max moves to evaluate)
- Rationale linking diagnosis to proposal

### Solver Action

- Evaluate proposed moves with exact single-machine DP
- Accept only verified improvements
- Report back to LLM: which moves improved, which didn't, and by how much

### Baseline

Random neighborhood proposals with equal move/DP budget:
- Random source selection
- Random target selection
- Random move-type selection
- Same K budget as LLM

### Success Criteria

- LLM proposals produce more TEC improvement per DP evaluation than random
- LLM correctly diagnoses at least 2/3 held-out cells
- LLM does not produce regressions vs baseline on guard cells

## Non-Goals

- This is NOT a runtime system — LLM latency is high; the test is offline
  (or at worst, at stagnation points where solver would otherwise terminate)
- This is NOT operator design — the LLM proposes moves, not code
- This is NOT a complete solver — exact DP is the verifier; the LLM is only
  an advisor at decision points

## Distinction from Prior Phases

| Phase | What LLM produces | Key representation | Why it failed |
|-------|-------------------|-------------------|---------------|
| S | Scoring functions (C++) | Python/C++ formulas | LLM scoring ≤ handcrafted |
| V | Operator code (C++) | C++ operator blocks | LLM code too buggy/incomplete |
| X | Policy parameters (JSON) | DSL with 17 fields | DSL too flat, random wins |
| **Y** | **Concrete neighborhoods** | **Source/target/move lists** | **TBD** |

Phase Y rejects the static-representation assumption. Instead of asking the
LLM to produce a reusable artifact (scoring function, operator, policy), it
asks the LLM to produce an **instance-specific intervention** at a concrete
decision point. This is the hardest test for the LLM's diagnostic ability
and also the hardest comparison (random can propose random neighborhoods,
and random neighborhood search is a well-known effective strategy).

## Validation Plan

1. Phase Y0: Define trace format for state conditioning (what the LLM sees)
2. Phase Y1: Implement C++ variant that accepts LLM neighborhood proposals
   and evaluates them with exact DP
3. Phase Y2: Implement random-neighborhood baseline with equal budget
4. Phase Y3: Run first DeepSeek call on 1-2 dev cells → iterate if promising
5. Phase Y4: If signal positive, evaluate on held-out cells
6. Phase Y5: If signal positive, compare with best prior results

Stop early if Phase Y0 trace shows state is not sufficiently informative for
LLM diagnosis, or if Phase Y3 shows LLM proposals are no better than random.

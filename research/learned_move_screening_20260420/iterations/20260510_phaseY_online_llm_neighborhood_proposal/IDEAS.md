# Ideas — Phase Y: Online LLM Neighborhood Proposal

## Active Ideas

### Y-I1: State-conditioned neighborhood proposal via DeepSeek

The core idea: at stagnation points, serialize the current schedule state
into a structured prompt, ask DeepSeek V4 Pro to propose a bounded set of
concrete moves, evaluate with exact DP, report results back to the LLM,
and iterate. The LLM receives both the state and the feedback, so it can
correct its diagnosis in subsequent rounds.

Expected advantage over Phase X: the LLM is not tuning a reusable policy
that must work across all cells and all states. It is making a **specific**
decision for a **specific** state. This avoids the DSL flatness problem
because the LLM's proposals are not constrained to a small parameter space.

### Y-I2: LLM as stagnation-breaker, not as solver loop

The LLM is called only at stagnation points (N consecutive rounds without
improvement). The core DiverseTrimmed + exception lane logic runs normally.
When it stalls, the LLM proposes a neighborhood that may include moves the
core lane cannot reach. This preserves exact-DP as the core move evaluator
and uses the LLM only where the core lane fails.

### Y-I3: Trace format — state snapshot + core-lane history

The trace format should include:
- Per-machine: current completion time, job set, slack
- Per-cell: TEC, epsilon, machine count
- Core-lane history: last K rounds (sources tried, improvements found)
- Outside pool: what moves are beyond the core shortlist
- Stagnation counter: how long without improvement

Format should be human-readable (Markdown in prompt) + structured (JSON
for parsing). DeepSeek sees the Markdown version.

### Y-I4: Budgeted move evaluation — K = 20

LLM proposes K = 20 concrete move specifications (source index, target index,
job index range, move type). Solver evaluates up to K moves with exact DP.
Budget is matched for the random baseline: also K = 20 random proposals.

### Y-I5: Comparison with Phase S/V/X best results as oracle ceiling

For cells where a prior phase found better results (e.g., LLM exception
on 62/290, score_escape on 65/195), use that as an oracle ceiling — the
LLM should be able to at least match those results with state-informed
proposals.

## Considered but Deferred

### Neighborhood generation via DP instead of LLM

Could use the exact DP solver to systematically enumerate small deviations
from the current schedule. This would be a much stronger baseline than
random neighborhoods. Deferred until Phase Y shows positive LLM signal —
if LLM cannot beat random, systematic DP enumeration is unnecessary.

### LLM suggests meta-strategy rather than concrete moves

LLM could propose a search strategy (e.g., "try sources with highest
cheap_lb_delta first, then random"). This is closer to Phase S/X and
has the same DSL flatness problem. Deferred. Phase Y tests the LLM's
ability to reason about concrete state, not to generate reusable rules.

### Multi-round online LLM with state feedback

Full online loop where LLM gets feedback after each move evaluation.
This is very expensive (1 DeepSeek call per move). Deferred until the
single-shot proposal mode shows positive signal.

## Rejected

### LLM generates C++ move filters

Phase V proved LLM-generated C++ is too fragile. Rejected.

### LLM generates scoring formulas from state features

Phase S/U proved LLM scoring does not beat handcrafted s2. Rejected.

### LLM generates DSL parameter updates from state

Phase X proved DSL is too flat. Rejected.

## Next Implementation Steps

See PROBLEM.md for the validation plan. Phase Y0 (trace format definition)
is the immediate next step.

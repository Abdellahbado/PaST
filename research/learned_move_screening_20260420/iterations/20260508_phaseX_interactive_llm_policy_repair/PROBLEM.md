# Problem — Phase X: Interactive LLM Policy Repair

## Context

Phase V tested DeepSeek as a trace-conditioned search-operator designer via 5
long design calls → implementation. The LLM correctly diagnosed source
starvation and s2 scoring bottlenecks and produced structurally valid operators
(score escape), but no LLM-designed operator beat all baseline gates
simultaneously at runtime.

The failure pattern was:
- Long design calls produce operators that are "close" but miss budgets or
  edge cases
- The LLM gets no runtime feedback between calls
- The budget / fairness fix (V5.2) was human work, not LLM work

## New Question

Can an **interactive** LLM loop — where the LLM sees the result of each
proposal within seconds/minutes and revises immediately — produce a better
policy than one-shot LLM generation?

This flips the interaction model:
- Phase V: few long calls, no feedback loop
- Phase X: many short rounds, fast feedback is central

## Hypothesis

Interactive LLM repair, using a constrained policy DSL and fast dev evaluation,
will outperform:

1. one-shot LLM policy (round 0);
2. random policy from the same DSL;
3. simple handcrafted score-escape policy;
4. DiverseTrimmed baseline.

## Scope

- Exception-lane only (no source expansion, no core integration)
- Exact DP verifies every accepted move
- Policy DSL constrains the LLM (no arbitrary C++)
- 3 fast dev cells for rapid feedback
- 5 interaction rounds
- 3 validation cells (held out) if gate passes

## Non-goals

- Beating EHS
- Beating global SOTA
- Claiming interactive LLM is better than handcrafted (we already know it wasn't in Phase V)
- We are testing whether **interaction** helps, specifically

## Paper Claim (if successful)

"Interactive LLM repair improves a bounded DP-verified exception-lane policy
over one-shot LLM generation and random policy search."

## Comparison to Phase V

| Aspect | Phase V | Phase X |
|--------|---------|---------|
| Interaction model | Few long design calls | Many short feedback rounds |
| LLM output | C++ implementation | JSON policy DSL |
| Feedback latency | Hours (implementation + eval) | Minutes (eval only) |
| Human implementation | Required each round | None (runner is generic) |
| Goal | Beat LLM exception baseline | Beat one-shot LLM + random |
| LLM role | Operator designer | Policy tuner |

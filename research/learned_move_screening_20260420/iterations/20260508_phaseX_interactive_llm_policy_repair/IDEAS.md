# Ideas — Phase X

## Architecture

### Policy DSL Design

JSON-controlled exception lane. The DSL captures all parameters that
distinguish the policies we've already tested (random exception, LLM
exception, score escape, trimmed=no exception) while being small enough
for the LLM to explore in 5 rounds.

Field rationale:
- `normal_mode`: the scoring function used in normal rounds
- `escape_mode`: fallback strategy when normal mode isn't finding hits
- `switch_after_no_hit`: trigger for escape activation
- `switch_back_on_hit`: whether a hit in escape mode returns to normal
- `initial_budget`, `max_budget`: adaptive budget parameters
- `grow_on_hit`, `shrink_on_miss`: budget adaptation rates
- `max_per_source`, `max_per_target`: diversity quotas
- `require_positive_cheap_lb`: filter out negative lower-bound candidates
- `coverage_bonus`: bonus weight for novel source/target machines
- `random_mix`: mix-in fraction of random selection
- `cheap_lb_weight`, `s2_weight`, `slack_weight`: scoring weights
- `guard_max_budget`: reduced budget on guard/epsilon-tight rounds

### Fast Dev Cells

- 62/290 (medium)
- 65/195 (small-medium)
- 85/300 (large)

Optional guard: 61/347

These are fast to evaluate (~seconds per cell) and represent different
regimes where different policies might excel.

### Interactive Loop Design

5 rounds. Each round:
1. LLM proposes one policy
2. Evaluate on 3 dev cells
3. LLM sees results
4. LLM repairs

The LLM must state what changed and why each round.

### Baselines

- trimmed (no exception lane)
- phaseS_random_exception_lane (seed 0)
- phaseS_llm_exception_lane
- phaseV_score_escape_sampler (V5.2)
- 10+ random DSL policies
- round 0 one-shot LLM policy

### Success Gate (X5)

- best interactive LLM policy beats round 0 one-shot LLM
- beats median random DSL policy
- improves vs trimmed on ≥2/3 fast dev cells
- no catastrophic guard regression

### Validation (X6)

- Run on 3 fresh cells not used in the loop
- Beats one-shot LLM on ≥60%
- Beats random DSL median on ≥60%
- Mean improvement vs trimmed positive

## Risks

1. **DSL too constrained**: The DSL may not express the key thing the LLM
   wants to change. Mitigation: the DSL covers all known effective variations.
2. **3 dev cells not representative**: May overfit. Mitigation: X6 validation
   on held-out cells.
3. **5 rounds too few**: The LLM may not converge. Mitigation: if round 4 is
   still improving, extend to 7 rounds. But 5 is the target for paper framing.
4. **JSON parsing in C++ too complex**: Mitigation: implement a simpler
   key=value format if needed, but JSON is preferred for LLM compatibility.

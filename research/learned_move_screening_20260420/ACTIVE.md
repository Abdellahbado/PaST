# Active Iteration

Current active iteration:

- `iterations/20260508_phaseX_interactive_llm_policy_repair/` — **PHASE X ACTIVE** (X4 complete).

Previous iterations (archived):

- `iterations/20260507_phaseV_trace_conditioned_llm_operators/` — Phase V runtime-LLM branch stopped.
- `iterations/20260503_phaseS_llm_chain_screening_controller/` — Stages 0-5D and Stage U complete.

Current state:

- Phase S source rebuilt from oracle binary (committed `a61c79c`).
- Phase V `score_escape_sampler` rebuilt and fixed (commits `24ca7a7`, `47d7fd3`).
- Phase X1-X2 complete. C++ `PhaseXPolicyJson` and Python orchestration working.
- Phase X3 complete. 20 random DSL policies → Case B (median worse than example, best beats example).
- Phase X4 complete. 5-round interactive DeepSeek loop → MINIMUM SUCCESS.
  Best LLM policy beats example (-6.3) and random median (-76.3), trails random best c000 (+31.4).

Next:

- Phase X5: Compare PhaseX against controls on full paper instances.
- Phase X6: Validation on held-out instances.
- **Next action**: Begin X5 (comparison against controls) or analyze X4 results further.

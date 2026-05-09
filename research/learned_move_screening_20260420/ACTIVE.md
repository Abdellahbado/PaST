# Active Iteration

Current active iteration:

- `iterations/20260508_phaseX_interactive_llm_policy_repair/` — **PHASE X ACTIVE**.

Previous iterations (archived):

- `iterations/20260507_phaseV_trace_conditioned_llm_operators/` — Phase V runtime-LLM branch stopped.
- `iterations/20260503_phaseS_llm_chain_screening_controller/` — Stages 0-5D and Stage U complete.

Current state:

- Phase S source rebuilt from oracle binary (committed `a61c79c`).
- Phase V `score_escape_sampler` rebuilt and fixed (commits `24ca7a7`, `47d7fd3`).
- Phase X1-X2 complete. C++ `PhaseXPolicyJson` and `scripts/phaseX_interactive_policy_repair.py` working.
  Full 3-cell × 6-arm smoke passed.

Next:

- Phase X3: Generate and evaluate 10+ random DSL policies to establish baseline distribution.
- Phase X4: Run 5-round interactive LLM loop.
- **Next action**: Begin X3 (random DSL policy baseline) or X4 (interactive LLM loop). Do not call DeepSeek until instructed.

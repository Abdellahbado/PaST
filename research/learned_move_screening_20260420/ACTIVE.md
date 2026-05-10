# Active Iteration

Current active iteration:

- `iterations/20260510_phaseY_online_llm_neighborhood_proposal/` — **PHASE Y ACTIVE** (Y1.1 complete).

Previous iterations (archived):

- `iterations/20260508_phaseX_interactive_llm_policy_repair/` — Phase X stopped (X5 WEAK, X6 skipped).
- `iterations/20260507_phaseV_trace_conditioned_llm_operators/` — Phase V runtime-LLM branch stopped.
- `iterations/20260503_phaseS_llm_chain_screening_controller/` — Stages 0-5D and Stage U complete.

Current state:

- Phase S: exception-lane architecture + exact-DP verification validated.
- Phase V: runtime-LLM operators stopped — LLM C++ code too fragile.
- Phase X: interactive LLM policy DSL repair stopped — LLM at 20th percentile.
- Phase Y: Y1 trace instrumentation complete; Y1.1 search-behavior fields populated
  (core_source_hits, core_target_hits, starved, underexplored_sources,
  underexplored_targets, last_accepted_moves, failed_summary). Smoke passed
  on 3 dev cells (~3800 tokens each, all machines present).

Next: Phase Y2 (random neighborhood baseline) or Y3 (first DeepSeek call).

Do NOT call DeepSeek until instructed.

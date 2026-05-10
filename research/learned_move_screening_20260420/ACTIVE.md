# Active Iteration

Current active iteration:

- `iterations/20260510_phaseY_online_llm_neighborhood_proposal/` — **PHASE Y CONCLUDED (FAIL)**.

Previous iterations (archived):

- `iterations/20260508_phaseX_interactive_llm_policy_repair/` — Phase X stopped (X5 WEAK, X6 skipped).
- `iterations/20260507_phaseV_trace_conditioned_llm_operators/` — Phase V runtime-LLM branch stopped.
- `iterations/20260503_phaseS_llm_chain_screening_controller/` — Stages 0-5D and Stage U complete.

Current state:

- Phase S: exception-lane architecture + exact-DP verification validated.
- Phase V: runtime-LLM operators stopped — LLM C++ code too fragile.
- Phase X: interactive LLM policy DSL repair stopped — LLM at 20th percentile.
- **Phase Y: CONCLUDED — FAIL.** LLM neighborhood proposals from state traces
  do NOT beat random under equal DP budget. The hypothesis that an LLM can
  diagnose stagnation from machine-level trace data and propose better
  neighborhoods is rejected. All 3 prior LLM approaches (S/V/X/Y) have now
  failed to beat random search.

Next: Open-ended. No active LLM-critical path remains.

# Active Iteration

Current active iteration:

- `iterations/20260510_phaseY_online_llm_neighborhood_proposal/` — **PHASE Y ACTIVE** (initialized, not yet implemented).

Previous iterations (archived):

- `iterations/20260508_phaseX_interactive_llm_policy_repair/` — Phase X stopped (X5 WEAK, X6 skipped).
- `iterations/20260507_phaseV_trace_conditioned_llm_operators/` — Phase V runtime-LLM branch stopped.
- `iterations/20260503_phaseS_llm_chain_screening_controller/` — Stages 0-5D and Stage U complete.

Current state:

- Phase S: exception-lane architecture + exact-DP verification validated.
- Phase V: runtime-LLM operators stopped — LLM C++ code too fragile.
- Phase X: interactive LLM policy DSL repair stopped — LLM at 20th percentile
  vs random best-of-5 under equal budget. DSL too flat.
- Phase Y: newly initialized. LLM sees **concrete schedule state** and proposes
  **bounded neighborhoods** (specific source/target/move lists), not code or
  policy parameters. Exact DP verifies. Random neighborhoods as baseline.

Phase Y hypothesis:
LLM's diagnostic strength (validated across S, U, V, X) is best tested when
the LLM makes instance-specific decisions from concrete state, rather than
designing reusable artifacts from aggregate statistics.

Next: Phase Y0 — trace format design.

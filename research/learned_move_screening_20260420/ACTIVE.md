# Active Iteration

Current active iteration:

- `iterations/20260508_phaseX_interactive_llm_policy_repair/` — **PHASE X ACTIVE** (X5 complete).

Previous iterations (archived):

- `iterations/20260507_phaseV_trace_conditioned_llm_operators/` — Phase V runtime-LLM branch stopped.
- `iterations/20260503_phaseS_llm_chain_screening_controller/` — Stages 0-5D and Stage U complete.

Current state:

- Phase S source rebuilt from oracle binary (committed `a61c79c`).
- Phase V `score_escape_sampler` rebuilt and fixed (commits `24ca7a7`, `47d7fd3`).
- Phase X1-X2 complete. C++ runner and Python orchestration working.
- Phase X3 complete. 20 random DSL policies → Case B.
- Phase X4 complete. 5-round interactive DeepSeek → MINIMUM SUCCESS.
- Phase X5 complete. Random best-of-5 distribution estimator:
  LLM at 20th percentile. **WEAK signal** — interactive LLM does NOT outperform
  random best-of-5 under same 5-attempt budget.

Validation cells (14) proposed in `notes/x5_validation_cells.md` but NOT run.

Next:

- Decide whether to run X6 validation despite WEAK X5 signal.
- If yes: evaluate all baselines + LLM policy + random best-of-5 on 14 cells.
- If no: write X5 summary as final result; Phase X stops here.

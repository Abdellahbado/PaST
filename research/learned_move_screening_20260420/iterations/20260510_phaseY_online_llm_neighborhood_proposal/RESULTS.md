# Results — Phase Y: Online LLM Neighborhood Proposal

No experiments yet. Phase Y is initialized but not yet implemented.

Prior evidence motivating Phase Y:

- Phase S: LLM-designed scoring rules did not beat handcrafted s2.
- Phase V: LLM-generated C++ operators were fragile and did not beat
  fixed exception lanes.
- Phase X: LLM-tuned static DSL policy at 20th percentile vs random
  best-of-5 under equal 5-attempt budget (WEAK signal).

All three prior phases suggest the LLM's diagnostic ability is real but
its representation-language accuracy (scoring formulas, C++ code, DSL
parameters) is insufficient. Phase Y tests whether switching from static
representation languages to concrete state-conditioned proposals unlocks
the LLM's diagnostic strength.

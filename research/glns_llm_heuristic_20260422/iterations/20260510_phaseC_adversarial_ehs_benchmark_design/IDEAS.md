# Phase C Ideas

## Known EHS Weakness Mechanisms (from B6 evidence)

1. **First-khat cost dominates on large instances** (B6.13 run28, B8 Stage 2)
   - SGH construction at khat=T is 100-400s on VLS scale
   - Multi-khat budget-splitting fails because first khat is most expensive
   - _Mechanism_: Make the first khat maximally expensive (high n, high m, spread energy rates)

2. **A-SGH over-conservatism retains too many previous assignments** (B6.11)
   - Keeps 96-98% of jobs from previous khat
   - Release policies don't help — released jobs repair back to same trajectory
   - _Mechanism_: Generate instances where the khat→khat-1 assignment is structurally different from khat→khat

3. **R-ES reinsertion is the primary bottleneck** (B6.4b)
   - 1.81s per khat vs 0.02s for ESR
   - Only 1.4% khat improvement rate from reinsertion
   - _Mechanism_: Generate instances where reinsertion would be valuable but costs too much

4. **ES non-empty dominates over reinsertion** (B6.4b)
   - ES non-empty improves 36.6% of khats vs 1.4% for reinsertion
   - _Mechanism_: Generate instances where local EPS swaps would help but interchange won't

5. **Per-machine sequencing gap is negligible** (B6.17)
   - Post-final gap 0.984% — sequencing surface is closed
   - _Non-mechanism_: Sequencing is NOT a weakness. Don't target it.

6. **Short-budget front density gap** (B6.5b, B6.17b)
   - At 120s, EHS reaches 12.9-71.6% of published HV
   - Fast_mode helps at ≤60s but loses at ≥120s
   - _Mechanism_: Generate instances where per-khat convergence is slow

7. **EHS converges near-completely** (B6.17b)
   - 97.1-97.7% of published HV by 300-1200s
   - _Non-mechanism_: Long-budget convergence is NOT a weakness

## Candidate LLM Family Hypotheses

### F1: High price volatility + narrow feasible region
- Hypothesis: When price volatility is high, SGH construction makes fragile
  energy-vs-cmax tradeoffs that don't survive multi-khat descent
- Parameters: high TOU variance, moderate n/m, tight epsilons

### F2: A-SGH trajectory lock-in
- Hypothesis: When successive khat levels have structurally different optimal
  assignments, A-SGH's 96% job retention becomes a liability
- Parameters: job sizes that force different spread at each khat decrement

### F3: R-ES reinsertion starvation
- Hypothesis: When feasible schedule space is dense at each khat, R-ES
  reinsertion never activates, missing potentially better assignments
- Parameters: wide T, narrow energy rate spread, many feasible alternatives

### F4: ES exploration-vs-exploitation tension
- Hypothesis: ES non-empty finds local improvements that prevent R-ES from
  escaping to better regions
- Parameters: energy-rate heterogeneity, TOU profiles with sharp peaks

### F5: Front coverage scarcity
- Hypothesis: When TOU profile creates discontinuous energy-vs-cmax
  Pareto regions, EHS front misses intermediate points
- Parameters: step-function TOU, bimodal job sizes

### F6: Short-budget pressure on large instances
- Hypothesis: At m≥40, n≥400, EHS under 120s cannot complete even 1 full khat
- Parameters: large n, m, moderate T — but must target mechanism, not just size

### F7: Machine-rate heterogeneity inducing extreme load imbalance
- Hypothesis: Very heterogeneous machine rates cause SGH to concentrate
  all jobs on cheap machines, creating cmax inflation
- Parameters: wide e-rate spread, narrow processing time range

### F8: epsilon skip affecting HV
- Hypothesis: When epsilon spacing is coarse relative to energy rates,
  EHS skips epsilons worth exploring
- Parameters: narrow price levels, wide machine rate differences

## Baseline Generators

### Random families
- Use the same family_schema
- Uniformly sample legal parameter ranges
- Same number of families as LLM

### Human/simple sweeps
- Fixed sweeps over n/m/T/tightness/price_volatility
- Same number of families as LLM
- Design decisions documented explicitly

## Open Questions

- What is the right number of instances per family for smoke?
- Should we validate that generated instances are "realistic BPMSTP"?
- Is Anghinolfi generator appropriate or should we build a richer one?
- Should families be validated against the exact solver (if available) or only EHS?

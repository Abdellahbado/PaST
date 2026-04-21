# Phase C Refinement Search Plan

Date: 2026-04-19

## Candidate paths considered

1. **Path A: relocate-first simplification**
   - Plausible because Phase C accepted moves were already relocate-dominated.
   - Upside: strong runtime/evaluation reduction by removing expensive swaps.
   - Risk: may miss rare but important swap-only improvements.

2. **Path B: screened swap search**
   - Plausible because full swap neighborhood was very large on `64/77` and `90/82` with low acceptance.
   - Implemented screen in this pass: only near-length swap candidates (`|pa-pb| <= 1`) plus safe slot-LB prefilter before exact DP.
   - Upside: keep swap capability while reducing costly exact checks.
   - Risk: over-screening might remove the few useful swaps.

3. **Path C: machine prioritization**
   - Plausible because improvement may be concentrated on a few costly machines.
   - Implemented signal: prioritize source machines with largest `(exact_cost - safe_slot_lb)` gap; search around top-ranked machines first.
   - Upside: fewer move evaluations, potentially faster convergence.
   - Risk: poor signal quality can miss beneficial moves and hurt TEC.

4. **Path D: paper-inspired equal-size exchange transplant**
   - Plausible from paper ES/EPS idea: equal-size period exchange is a core neighborhood component.
   - In our framework, this is represented by the near-length/same-length screened swap path (Path B), evaluated with exact DP acceptance.
   - Upside: paper-aligned neighborhood restriction with bounded complexity.
   - Risk: if relocate already captures gain, this adds engineering but little quality.

## Selected for implementation in this pass

- `greedy_dp_local_search_relocate_only` (Path A)
- `greedy_dp_local_search_screened_swap` (Path B + Path D narrow transplant)
- `greedy_dp_local_search_priority_machines` (Path C)

Also fixed reporting debt:

- split exact DP call metric into:
  - `exact_dp_calls_initial`
  - `exact_dp_calls_local_search_only`

## Deliberately not implemented yet

- multi-pass adaptive screening with dynamic thresholds
- full EPS structure replication from paper R-ES (out of scope)
- larger/multi-job neighborhoods (out of scope)

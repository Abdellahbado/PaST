# Phase E Next-Branch Design Memo

Date: 2026-04-19

## Decision goal

Select the next serious branch after Phase D using evidence from:

- Phase C and Phase D reports
- `solvers/cpp/parallel_heuristic_compare.cpp`
- paper text (`Papers/Exact and heuristic.txt`)
- stored paper EHS fronts (`temp/paper_exact_repo/results/EHS/1/`)

## Candidate ranking

1. **Multi-start / randomized assignment + relocate local search** (**rank #1**)
2. **Stronger post-repair neighborhood or rescue mechanism** (**rank #2**)
3. **Full history-based epsilon sweep with robust repair** (**rank #3**)

## Why this ranking

### 1) Multi-start / randomized assignment + relocate local search (top)

Evidence:

- Phase B showed assignment quality is a major lever; deterministic DP-guided assignment was unstable, but this still indicates assignment structure matters.
- Phase C showed relocate-only cleanup is robust and low complexity.
- Phase D showed continuity ideas can help, but robustness failures on `64` and `90` make full-sweep investment risky right now.

Plausibility:

- Paper SGH includes randomized best-location choice; a bounded multi-start recovers this spirit without reimplementing full EHS.
- Keeps architecture simple: same evaluator/local-search core, only assignment diversification added.

Risk:

- Runtime scales with number of starts; must be bounded.

### 2) Stronger post-repair neighborhood/rescue

Evidence:

- Phase D failures (`64: 78->77`, `90: 83->82`) indicate reinsertion dead-ends.
- A bounded rescue (targeted ejection chain, micro-destroy/repair, or fallback reinsertion order) could improve feasibility continuity.

Why not rank #1:

- More invasive to history-repair internals, higher implementation risk, and still tied to a branch that is not yet robust.

### 3) Full robust history epsilon sweep

Evidence:

- Phase D produced positive signal (`46/73` and partial `61`) but chain robustness is currently insufficient.

Why rank #3:

- Requires solving known blockers first; jumping directly to full sweep would likely amplify failure handling complexity.

## Small prototype (implemented)

Prototype variant added in `solvers/cpp/parallel_heuristic_compare.cpp`:

- `greedy_dp_local_search_relocate_multistart`

Behavior:

- run 8 randomized LPT constructions (restricted candidate list size 3 among cheapest feasible machine insertions)
- for each start, apply existing `greedy_dp_local_search_relocate_only`
- return the best TEC among starts

This is intentionally small and bounded (no new global framework).

## Sanity experiment (instance 61)

Command:

`/usr/bin/time -l ./solvers/cpp/build/parallel_heuristic_compare paper-instance 61 345 greedy_dp_local_search_relocate_multistart "./temp/paper_exact_repo/instances" 30 10 5 20000`

Observed:

- prototype TEC: `6960`
- one-shot relocate baseline TEC (same setup): `7085`
- one-shot `greedy_dp` TEC: `7102`
- paper EHS at same epsilon (`res_61.csv` line for `345`): `6723`

Delta view:

- vs one-shot relocate: `-125`
- vs one-shot greedy_dp: `-142`
- remaining gap to paper EHS: `+237`

Resource profile (timed run):

- runtime: `~147.8 s`
- peak RSS: `~1.63 GB`

## Recommendation

Make **multi-start / randomized assignment + relocate local search** the main next iteration branch.

Rationale:

- strongest immediate quality gain from a minimal prototype,
- aligns with paper's randomized constructive spirit,
- lower implementation risk than immediate full history-sweep hardening,
- leaves option to reuse as seed generator for future history-repair steps.

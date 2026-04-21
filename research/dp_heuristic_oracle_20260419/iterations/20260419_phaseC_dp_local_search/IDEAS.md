# Ideas

## Implemented candidate

- `greedy_dp_local_search`:
  - start from `greedy_dp` assignment
  - evaluate one-job relocate and one-for-one swap neighborhoods
  - recompute exact cost only on touched machines via single-machine DP
  - accept first strict-improving move, restart neighborhood scan

## Engineering choices

- cache exact machine costs keyed by multiset signature to reduce repeated DP solves
- maintain deterministic iteration order over machines/jobs
- expose bounded controls through CLI:
  - `ls_time_cap_sec`
  - `ls_max_rounds`
  - `ls_max_moves_per_round`

## Metrics to track

- TEC, assignment-conditioned LB, runtime, max RSS
- accepted relocate/swap move counts
- evaluated relocate/swap move counts
- dominant accepted move type
- final machine loads
- exact DP calls during local search (cache misses)

## Follow-up ideas (only if Phase C positive)

- stronger move screening before exact DP calls
- limited best-improvement pass per round
- adaptive machine-pair candidate pruning using load/rate heuristics

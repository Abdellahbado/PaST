# Phase D History-Repair Results

Date: 2026-04-19

## Truth sources used

- Paper method source: `Papers/Exact and heuristic.txt`
- Paper repo metadata: `temp/paper_exact_repo/README.md`
- Paper stored fronts (run 1):
  - `temp/paper_exact_repo/results/EHS/1/res_46.csv`
  - `temp/paper_exact_repo/results/EHS/1/res_61.csv`
  - `temp/paper_exact_repo/results/EHS/1/res_64.csv`
  - `temp/paper_exact_repo/results/EHS/1/res_90.csv`
- Paper timing file:
  - `temp/paper_exact_repo/results/EHS/1/time.txt`

Important observed fact:

- cloned paper repo contains data/results, not implementation source code.

## Implemented prototypes

- `history_repair_dp_ranked`
- `history_repair_priority_displaced_relocate`

Implementation file:

- `solvers/cpp/parallel_heuristic_compare.cpp`

New mode:

- `paper-history-chain <instance> <epsilon_start> <epsilon_end> <variant>`

## Paper-aligned transitions tested

- instance `46`: `77 -> 76 -> 75 -> 74 -> 73`
- instance `61`: `347 -> 346 -> 345`
- instance `64`: `79 -> 78 -> 77`
- instance `90`: `84 -> 83 -> 82`

These `epsilon` values are directly present in paper EHS stored fronts.

## Baseline and comparison protocol

For each tested `epsilon`, compared against:

- one-shot `greedy_dp`
- one-shot `greedy_dp_local_search_relocate_only`
- paper EHS TEC at same `epsilon` when available

## Selected quantitative results

### Instance 46 transition (`77 -> 73`)

At `epsilon=73` (same-`epsilon` comparisons):

- paper EHS: `104`
- one-shot `greedy_dp`: `121`
- one-shot relocate-only: `112`
- `history_repair_dp_ranked`: `121`
- `history_repair_priority_displaced_relocate`: `103`

Result:

- At `46/73`, this observed `103` should be interpreted as a continuity-chain outcome with relocate cleanup, not as evidence that reinsertion logic alone is strong.
- It still beats same-`epsilon` paper EHS by `-1` and one-shot relocate-only by `-9` at this point.

### Instance 61 transition (`347 -> 345`)

At `epsilon=346`:

- paper EHS: `6717`
- one-shot relocate-only: `7076`
- `history_repair_dp_ranked`: `7050`
- `history_repair_priority_displaced_relocate`: `7043`

At `epsilon=345`:

- paper EHS: `6723`
- one-shot relocate-only: `7085`
- `history_repair_dp_ranked`: `7070`
- `history_repair_priority_displaced_relocate`: `7039`

Result:

- both history prototypes improve over one-shot relocate baseline on tested points.
- neither approaches paper EHS level on this hard row.

### Instance 64 transition (`79 -> 77`)

- at `epsilon=78`, both history prototypes produce finite TEC but worse than one-shot relocate and paper EHS.
- at `epsilon=77`, both history prototypes fail (infeasible chain step).

Same-`epsilon` references at `77`:

- paper EHS: `30580`
- one-shot relocate-only: `30580`
- history prototypes: infeasible

### Instance 90 transition (`84 -> 82`)

- `history_repair_dp_ranked` yields finite TEC at `83` but worse than one-shot relocate and paper EHS.
- `history_repair_priority_displaced_relocate` fails already at `83`.
- both fail to reach `82` in chain mode.

Same-`epsilon` references:

- paper EHS at `83`: `52510`
- one-shot relocate-only at `83`: `52510`
- history variants: worse or infeasible.

## Exact-gap reporting

Exact references available in this thread for:

- `46/77 = 103`
- `64/77 = 30580`
- `90/82 = 53294`

Computed exact gaps where feasible:

- `46/77`:
  - one-shot `greedy_dp`: `(118-103)/103 = 0.145631`
  - one-shot relocate-only: `(109-103)/103 = 0.058252`
  - history seed at `77`: `(109-103)/103 = 0.058252`

- `64/77`:
  - one-shot `greedy_dp`: `0.000589`
  - one-shot relocate-only: `0.000000`
  - history prototypes: infeasible at `77`

- `90/82`:
  - one-shot `greedy_dp`: `0.000000`
  - one-shot relocate-only: `0.000000`
  - history prototypes: infeasible at `82`

## Runtime and RSS

- all successful runs remain below 16 GB RSS
- highest observed RSS in this pass: `385,204,224` bytes (~367 MB)

## Metrics captured per run

CSV output now includes:

- TEC, runtime, max RSS
- displaced jobs
- reinsertion candidates scored
- exact DP evals during repair
- exact DP evals during post-repair local search
- relocate cleanup usage flag
- accepted/evaluated move counts
- final machine loads

## Decision against requested criteria

1. better TEC than one-shot relocate on `61`:
   - yes (`7039` vs `7085` at `345`, `7043` vs `7076` at `346`) by `history_repair_priority_displaced_relocate`.

2. same or better TEC with cleaner continuity across neighboring epsilons:
   - partial only; continuity still breaks on `64` and `90`.

3. exact same-`epsilon` improvement over paper EHS on one nontrivial row:
   - yes on `46/73`: `103` vs paper `104`, but this is continuity + relocate-cleanup signal, not standalone reinsertion strength proof.

4. credible path to closing gap on `61`:
   - partially credible but still far from paper EHS (`~+316` to `+333` at tested points).

## Conclusion

- Branch has signal (continuity + relocate-cleanup win at `46/73`, improves over one-shot on `61`) but current repair feasibility is not robust enough across instances.
- Continue only with prioritized+relocate prototype after targeted feasibility hardening.
- Stop DP-ranked-only prototype for now.

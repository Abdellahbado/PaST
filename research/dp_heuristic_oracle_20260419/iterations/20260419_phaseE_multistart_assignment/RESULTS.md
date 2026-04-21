# Results

## Design pass outcome

- top-ranked direction: multi-start / randomized assignment plus relocate-only local search
- memo: `research/dp_heuristic_oracle_20260419/phaseE_next_branch_design_memo.md`

## Prototype implemented

Implementation file:

- `solvers/cpp/parallel_heuristic_compare.cpp`

Variant:

- `greedy_dp_local_search_relocate_multistart`

## Sanity experiment on instance 61

Row tested:

- instance `61`, `epsilon=345`

Observed TEC:

- `greedy_dp`: `7102`
- `greedy_dp_local_search_relocate_only`: `7085`
- `greedy_dp_local_search_relocate_multistart`: `6960`
- paper EHS (`res_61.csv`): `6723`

Quality deltas:

- multistart vs relocate-only: `-125`
- multistart vs greedy_dp: `-142`
- multistart vs paper EHS: `+237`

Resource signal (timed run):

- runtime about `147.8` seconds
- peak RSS about `1.63` GB

## Recommendation status

- recommend Phase E multistart branch as main next iteration.

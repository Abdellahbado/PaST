# Results

## Implemented prototypes

- `history_repair_dp_ranked`
- `history_repair_priority_displaced_relocate`

Implemented in:

- `solvers/cpp/parallel_heuristic_compare.cpp`

New mode:

- `paper-history-chain <instance> <epsilon_start> <epsilon_end> <variant>`

## Paper-aligned transitions tested

- `46`: `77 -> 76 -> 75 -> 74 -> 73`
- `61`: `347 -> 346 -> 345`
- `64`: `79 -> 78 -> 77`
- `90`: `84 -> 83 -> 82`

Paper EHS same-`epsilon` references used from:

- `temp/paper_exact_repo/results/EHS/1/res_46.csv`
- `temp/paper_exact_repo/results/EHS/1/res_61.csv`
- `temp/paper_exact_repo/results/EHS/1/res_64.csv`
- `temp/paper_exact_repo/results/EHS/1/res_90.csv`

## Outcome summary

- `history_repair_dp_ranked`:
  - stable but weak on tested transitions
  - never beats one-shot `greedy_dp_local_search_relocate_only`
  - fails to complete chain at `64: 78 -> 77` and `90: 83 -> 82`

- `history_repair_priority_displaced_relocate`:
  - best observed at `46/73`: TEC `103` (beats paper EHS `104` by `-1` and beats one-shot relocate `112`)
  - improves over one-shot relocate on `61` transitions (`346`, `345`) but still far above paper EHS at same `epsilon`
  - fails at `64: 78 -> 77` and already fails at `90: 84 -> 83`

## Resource profile

- all successful runs stayed below 16 GB RSS
- highest observed RSS in this pass remained in MB scale (largest around `385,204,224` bytes from one-shot baseline run)

## Decision signal

- history-repair with relocate cleanup shows promise on narrow transitions (`46`, partly `61`)
- but current repair prototype is not robust enough across `64` and `90` chains

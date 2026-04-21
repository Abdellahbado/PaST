# Ideas

Candidate directions considered in the pre-branch design pass:

1. full history-based epsilon sweep with robust repair
2. multi-start / randomized assignment plus local search
3. stronger post-repair neighborhood or rescue mechanism

Ranking outcome:

1. multi-start / randomized assignment plus local search (selected)
2. stronger post-repair neighborhood/rescue
3. full robust history sweep

Selected initial prototype:

- `greedy_dp_local_search_relocate_multistart`
  - 8 randomized LPT starts
  - restricted candidate list size 3 among cheapest feasible machine insertions
  - per start: existing relocate-only DP local search
  - keep best TEC across starts

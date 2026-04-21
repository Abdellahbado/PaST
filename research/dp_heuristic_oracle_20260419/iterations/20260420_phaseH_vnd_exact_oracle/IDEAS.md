# Ideas

Initial bounded design ideas for Phase H:

1. Keep fixed `epsilon` and focus only on assignment/sequence neighborhoods.
2. Borrow VND neighborhood order from paper: `swap_intra -> swap_inter -> insert_inter`.
3. Keep first-improvement style and restart neighborhood index after each accepted move.
4. Use exact DP only on touched machine(s), not full recomputation for every screened candidate.
5. Add exact-machine-cost caching keyed by machine multiset + rate.
6. Use small deterministic multistart (randomized LPT/RCL) for robustness without large metaheuristic scope.
7. Use bounded shortlist screening (fallback-slot lower bound gain) before exact DP in inter-machine neighborhoods.

# Ideas

1. Reuse Phase H bounded multistart initialization logic to start from a strong incumbent.
2. Replace shortlist/LB-pruned move filtering with direct feasible move enumeration for the tested diagnostic batch.
3. Evaluate touched-machine exact DP cost for every tested move in the batch.
4. Keep bounded runtime using exact-eval caps rather than pre-exact pruning.
5. Prioritize source machines with highest current exact TEC for a more informative bounded batch.
6. Use a small cap ladder (`64`, `256`, `1024`) to test whether improving moves appear with wider exact evaluation.

# Summary

Stage L1.5 densified exact-label collection beyond Stage L1 and expanded to multiple contexts.

Implemented:

- dense labeling variant in solver: `vnd_exact_dp_insert_rank_dense_labeling`
- orchestrator variant: `stageL15_dense_labeling`
- aggregate artifacts under `temp/phaseL15_dense_labeling/`

Coverage:

- contexts: 4 (`61/347`, `61/346`, `61/345`, `64/79`)
- seed runs: 34
- exact-labeled: 20,873
- positives: 8,109

Conclusion:

- Dataset now has materially improved volume and context diversity and is ready for Stage L2 offline ranking experiments.

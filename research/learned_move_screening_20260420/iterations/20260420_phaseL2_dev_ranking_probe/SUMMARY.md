# Summary

Stage L2 development-only ranking probe completed on cleaned L1.5 exact-labeled data.

What was done:

- built cleaned modeling dataset and split manifests under `temp/phaseL2_dev_ranking_probe/`
- used primary target `max(0, -exact_total_delta)`
- trained XGBoost regressor and compared to handcrafted `screen_score_s2` ranking
- evaluated fixed budgets (`k=10/25/50/100`) on seed-aware LOSO and context hold-out splits

Main result:

- strong gains over handcrafted baseline on within-context seed splits
- mixed cross-context result: recall improves, but top-k improvement magnitude is weaker on hold-out due to poor transfer on context `64/79`

Decision:

- evidence is positive enough to continue the learning branch,
- but only via a cleaner next protocol (generated non-benchmark corpus + strict final split design),
- not yet ready for benchmark-level claims or solver integration.

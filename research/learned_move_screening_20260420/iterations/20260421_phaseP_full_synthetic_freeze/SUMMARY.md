# Summary

Phase P executed full-manifest synthetic dense exact labeling and froze one training-ready synthetic dataset.

What was done:

- created Phase P runner `scripts/phaseP_full_synthetic_freeze.py` with manifest validation, resumable progress logging, `(M,N,K)` batching, and freeze metadata,
- labeled all synthetic train/val instances from Phase M manifests using unchanged Phase O dense policy,
- produced required global/split/bucket diagnostics and skew decomposition artifacts,
- froze train/val/merged exact-labeled datasets plus schema and manifest lock file.

Key numbers:

- instances: 180/180 successful (150 train, 30 val), failures: 0
- rows: 317113
- positives/negatives: 251721 / 65392
- train positive rate: 0.790318
- val positive rate: 0.810797
- gap: +0.020479 (val - train)

Skew conclusion:

- skew is primarily within-bucket effect rather than bucket composition mismatch,
- epsilon policy appears stable across splits,
- variance exists in a few buckets but does not invalidate freeze.

Decision:

- Phase P freeze criteria satisfied; synthetic dataset is ready for the next offline synthetic-only model-training stage.

Next step:

- run offline synthetic-only move-ranking training on the frozen train/val dataset, using the established tabular model family (`XGBoost`, `LightGBM`, `CatBoost`) and the same offline baselines/`k=10,25,50,100` evaluation protocol already defined in `reference/PLAN_supervised_move_ranking.md`

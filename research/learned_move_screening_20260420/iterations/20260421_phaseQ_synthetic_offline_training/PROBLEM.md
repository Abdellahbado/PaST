# Problem

Phase P froze a full synthetic train/val exact-labeled dataset and ended the data-creation stage.

Phase Q objective:

- run synthetic-only offline move-ranking training/evaluation on the frozen Phase P train/val files,
- compare required tabular families (`XGBoost`, `LightGBM`, `CatBoost`) against required offline baselines,
- report fixed-budget ranking outcomes at `k=10/25/50/100`,
- choose one default learned model candidate for the next benchmark test-only stage.

Hard constraints:

- use only frozen Phase P synthetic train/val data,
- no benchmark training/tuning/thresholding,
- no labeling-policy changes,
- no solver integration,
- no formulation change beyond move-ranking.

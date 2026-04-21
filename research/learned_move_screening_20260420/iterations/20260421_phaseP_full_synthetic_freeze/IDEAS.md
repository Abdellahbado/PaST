# Ideas

1. Keep Phase O dense policy fixed (`stageO_synthetic_dense_logging` + same epsilon rule) and scale only execution scope from bounded subset to full manifests.
2. Batch by `(M,N,K)` bucket to make failures and skew interpretable; emit instance-level progress plus per-batch aggregates.
3. Make runner resumable by persisting `batch_progress.csv` and skipping already successful instances with matching epsilon.
4. Copy per-instance solver outputs into a Phase P-owned folder and build frozen train/val/merged datasets from successful manifest-covered instances only.
5. Diagnose skew decomposition with three concrete checks: bucket-composition effect, within-bucket rate effect, and instance/seed variance against train bucket spread.
6. Emit a freeze manifest tying output files to exact input manifests and fixed labeling configuration for reproducibility.

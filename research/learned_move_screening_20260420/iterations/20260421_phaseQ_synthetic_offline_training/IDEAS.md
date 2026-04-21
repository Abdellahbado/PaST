# Ideas

1. Keep target semantics identical to prior stages: regress `target_improvement_magnitude = max(0, -exact_total_delta)` and evaluate ranking quality under fixed top-k budgets.
2. Keep handcrafted comparison centered on `screen_score_s2` (primary) with `screen_score_s1` as secondary reference.
3. Use one deterministic train/val split (already frozen by manifest role) and avoid extra split-search/tuning cycles.
4. Handle positive-dominant labels with inverse-frequency sample weights while prioritizing ranking metrics over raw classification accuracy.
5. Export a complete machine-readable artifact set (`training_config`, dataset profile, model/baseline tables, per-k summary, scored validation rows, feature importance) under one Phase Q temp folder.
6. Keep bucket-level diagnostics secondary but included, since val has one instance per `(M,N,K)` bucket.

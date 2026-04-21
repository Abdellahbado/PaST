# Ideas

## Target choice

- Use exact-delta magnitude target: `max(0, -exact_total_delta)`.
- Keep binary improving label as auxiliary only.
- Do not use `label_accepted` as primary target.

## Split policy

- Seed-aware LOSO within context to test seed generalization.
- Full context hold-out to test cross-context transfer.
- Save explicit manifests to prevent row-level leakage.

## Baseline and model

- Baseline: handcrafted `screen_score_s2` descending rank.
- Model: single boosted-tree family (XGBoost regressor) for first probe.

## Reporting policy

- Mandatory fixed-budget metrics at `k=10/25/50/100`.
- Report weighted and unweighted summaries.
- Include per-context deltas so imbalance is explicit.

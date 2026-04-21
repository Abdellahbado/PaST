# Results

Stage L2 dev-only offline ranking probe completed.

## Data cleanup outputs

- cleaned modeling table: `temp/phaseL2_dev_ranking_probe/modeling_dataset_dev.csv`
- cleanup metadata: `temp/phaseL2_dev_ranking_probe/dataset_cleanup_protocol.json`
- profile: `temp/phaseL2_dev_ranking_probe/modeling_dataset_profile.json`
- split manifests:
  - `split_manifest_seed.csv`
  - `split_manifest_context.csv`

## Probe setup

- target: `target_improvement_magnitude = max(0, -exact_total_delta)`
- baseline: `screen_score_s2` descending
- model: XGBoost regressor

## Main outcomes

Seed-aware LOSO within context (weighted):

- recall improvements at `k=10/25/50/100`:
  - `+0.1030 / +0.1798 / +0.2281 / +0.2535`
- precision improvements:
  - `+0.1593 / +0.1204 / +0.0984 / +0.0688`
- best-improvement gains:
  - `+1.3186 / +0.4779 / +0.4602 / +0.3894`

Context hold-out (weighted):

- recall improvements:
  - `+0.0595 / +0.0874 / +0.1042 / +0.1493`
- precision mostly improves, slight drop at `k=100`
- magnitude metrics degrade versus baseline:
  - best-improvement deltas `-11.85 / -10.17 / -6.37 / -4.23`

Per-context hold-out deltas show:

- positive behavior on contexts `1/2/3`
- strong negative behavior on context `4` (`64/79`), dominating magnitude-metric losses

## Conclusion

- Task appears learnable on development data (especially within-context seed generalization).
- Cross-context transfer is mixed; baseline still stronger for top-k improvement magnitude under context hold-out.
- Evidence justifies moving to cleaner non-benchmark data protocol before any stronger claim.

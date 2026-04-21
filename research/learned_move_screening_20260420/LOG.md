# Log

## 2026-04-20

### Attempt
Start a new research thread for the learning-based continuation of the insert-focused exact-DP heuristic.

### Result

- created a separate thread so the objective is cleanly separated from the handcrafted heuristic-oracle experiments
- preserved predecessor evidence as design input rather than mixing it into a new iteration of the old thread
- initialized a planning iteration for supervised move ranking / screening

### Evidence

- `research/learned_move_screening_20260420/OVERVIEW.md`
- `research/learned_move_screening_20260420/LITERATURE.md`
- `research/learned_move_screening_20260420/reference/PLAN_supervised_move_ranking.md`

### Conclusion

The learning branch should be treated as a new method family with the old heuristic thread acting as precursor evidence and baseline source.

### Attempt
Implement Stage L1 move-level dataset logging for learned `insert_inter` ranking at anchor `61/347`.

### Result

- added logging instrumentation to the insert-focused exact-DP base flow in `solvers/cpp/parallel_heuristic_compare.cpp`
- added logging wrapper variant `stageL1_dataset_logging` that runs deterministic multistart seeds and exports Stage L1 artifacts
- generated two compatible datasets:
  - broad generated-candidate stream
  - exact-labeled stream
- produced metadata summary and feature dictionary under `temp/phaseL1_dataset_logging/`

### Evidence

- `research/learned_move_screening_20260420/history/phase_reports/phaseL1_dataset_logging_design.md`
- `temp/phaseL1_dataset_logging/moves_broad_61_347.csv`
- `temp/phaseL1_dataset_logging/moves_exact_labeled_61_347.csv`
- `temp/phaseL1_dataset_logging/dataset_summary_61_347.json`
- `temp/phaseL1_dataset_logging/feature_dictionary.md`

### Conclusion

Stage L1 dataset is usable for Stage L2 offline ranking at the anchor point: broad coverage is high and exact-labeled positives are sufficient for a first ranking experiment.

### Attempt
Run Stage L1.5 dense exact-labeling expansion to materially increase exact-label volume and context diversity before Stage L2.

### Result

- added dense labeling variant `vnd_exact_dp_insert_rank_dense_labeling` and collection wrapper `stageL15_dense_labeling` in `solvers/cpp/parallel_heuristic_compare.cpp`
- preserved Stage L1 schema and added small context/rank stress features for generalization support
- collected aggregate dense dataset across four contexts with deterministic multistart

### Evidence

- `research/learned_move_screening_20260420/history/phase_reports/phaseL15_dense_labeling_design.md`
- `temp/phaseL15_dense_labeling/moves_exact_labeled_aggregate.csv`
- `temp/phaseL15_dense_labeling/moves_broad_aggregate.csv`
- `temp/phaseL15_dense_labeling/context_seed_summary.csv`
- `temp/phaseL15_dense_labeling/dataset_summary_dense.json`

### Conclusion

Stage L1.5 achieved a large step-up in exact labels and multi-context diversity, making the dataset suitable for substantive Stage L2 offline ranking experiments.

### Attempt
Run Stage L2 development-only offline ranking probe with explicit dataset/protocol cleanup and leakage-aware split reporting.

### Result

- created a cleaned modeling table from Stage L1.5 exact-labeled aggregate and documented development-only data status
- defined target semantics using exact-delta improvement magnitude (`max(0, -exact_total_delta)`) rather than acceptance label
- created seed-aware and context-aware split manifests to avoid row-level leakage
- ran XGBoost ranking probe against handcrafted baseline (`screen_score_s2`) at fixed budgets (`k=10/25/50/100`)
- exported global/per-context metric tables, fold deltas, weighted summaries, and feature-importance artifacts

### Evidence

- `research/learned_move_screening_20260420/history/phase_reports/phaseL2_dev_data_status_note.md`
- `research/learned_move_screening_20260420/history/phase_reports/phaseL2_dev_ranking_probe_design.md`
- `research/learned_move_screening_20260420/history/phase_reports/phaseL2_dev_ranking_probe_results.md`
- `research/learned_move_screening_20260420/history/phase_reports/phaseL2_dev_ranking_probe_readiness.md`
- `temp/phaseL2_dev_ranking_probe/modeling_dataset_dev.csv`
- `temp/phaseL2_dev_ranking_probe/ranking_results_key_comparison.csv`
- `temp/phaseL2_dev_ranking_probe/ranking_context_holdout_deltas.csv`

### Conclusion

Development-stage evidence supports learnability (especially within-context seed generalization), but context-holdout magnitude ranking remains mixed; next step should be a cleaner generated-data protocol before any benchmark-level claim or solver integration.

### Attempt
Run Stage L2.5 development-only robustness pass: reconcile L2 reporting consistency, add stronger baselines, run score ablations, test normalization, and compare tabular model families.

### Result

- identified L2 inconsistency source as helper-summary aggregation logic (not canonical metric tables)
- corrected L2 helper summaries to align weighted/unweighted fold aggregates
- executed required baseline ladder: random, handcrafted (`s1/s2`), oracle, learned models
- executed required feature-set ablations:
  - full
  - no-screen
  - screen-only
- added explicit normalized/ratio features and measured transfer impact
- compared tabular families (`xgboost`, `lightgbm`, `catboost`, `random_forest`, `decision_tree`) under seed-LOSO and context-holdout splits

### Evidence

- `research/learned_move_screening_20260420/history/phase_reports/phaseL25_ablation_normalization_models_design.md`
- `research/learned_move_screening_20260420/history/phase_reports/phaseL25_ablation_normalization_models_results.md`
- `research/learned_move_screening_20260420/history/phase_reports/phaseL25_ablation_normalization_models_readiness.md`
- `temp/phaseL25_ablation_normalization_models/results_summary_weighted.csv`
- `temp/phaseL25_ablation_normalization_models/ablation_xgboost_weighted.csv`
- `temp/phaseL25_ablation_normalization_models/model_comparison_weighted.csv`
- `temp/phaseL25_ablation_normalization_models/normalization_effect_xgboost_weighted.csv`

### Conclusion

Move-ranking signal remains positive on development data and survives no-screen ablation, indicating genuine learnable structure beyond handcrafted score wrapping; however, normalization did not reliably fix context-holdout magnitude transfer, so the next step should be generated-instance corpus design with strict family-level split isolation before any online integration.

### Attempt
Run Phase M clean protocol setup for synthetic-only VLS train/val with benchmark role separation.

### Result

- implemented deterministic synthetic VLS generator and emitted loader-compatible `Data_p/e/c` files
- produced synthetic catalogs, benchmark family summaries, generated-vs-benchmark comparison tables, and split manifests
- enforced split policy: synthetic-only train/val, benchmark `61-90` primary test-only, benchmark `1-60` secondary robustness-only

### Evidence

- `research/learned_move_screening_20260420/reference/phaseM_benchmark_role_note.md`
- `research/learned_move_screening_20260420/history/phase_reports/phaseM_vls_synthetic_protocol_design.md`
- `research/learned_move_screening_20260420/history/phase_reports/phaseM_vls_synthetic_protocol_results.md`
- `research/learned_move_screening_20260420/history/phase_reports/phaseM_vls_synthetic_protocol_readiness.md`
- `scripts/phaseM_vls_synthetic_protocol.py`
- `temp/phaseM_vls_synthetic_protocol/split_manifest_train.csv`
- `temp/phaseM_vls_synthetic_protocol/split_manifest_val.csv`
- `temp/phaseM_vls_synthetic_protocol/split_manifest_test_primary_vls.csv`
- `temp/phaseM_vls_synthetic_protocol/split_manifest_test_secondary_legacy.csv`

### Conclusion

Phase M protocol setup is complete and ready for synthetic-only execution work (label extraction and offline learning prep) without benchmark leakage.

### Attempt
Start Phase N by executing a bounded synthetic-only exact-label extraction sanity pass over Phase M train/val manifests.

### Result

- repaired thread-state memory inconsistency (`ACTIVE`/`LOG`) and initialized missing Phase M iteration files
- created Phase N iteration and switched active branch
- implemented manifest-driven runner `scripts/phaseN_synthetic_labeling_sanity.py`
- executed bounded stratified subset (`12 train + 4 val`) using only Phase M train/val manifests
- produced labeling config, run summary, subset aggregates, split/merged labeled datasets, and schema dictionary under `temp/phaseN_synthetic_labeling_sanity/`

### Evidence

- `research/learned_move_screening_20260420/history/phase_reports/phaseN_synthetic_labeling_sanity_design.md`
- `research/learned_move_screening_20260420/history/phase_reports/phaseN_synthetic_labeling_sanity_results.md`
- `research/learned_move_screening_20260420/history/phase_reports/phaseN_synthetic_labeling_sanity_readiness.md`
- `research/learned_move_screening_20260420/iterations/20260420_phaseN_synthetic_labeling_sanity/SUMMARY.md`
- `scripts/phaseN_synthetic_labeling_sanity.py`
- `temp/phaseN_synthetic_labeling_sanity/labeling_run_summary.json`
- `temp/phaseN_synthetic_labeling_sanity/labeling_subset_aggregate.csv`

### Conclusion

Synthetic train/val exact-label pipeline is validated at sanity scale and is ready for controlled scale-up on full manifests before offline model fitting.

### Attempt
Start Phase O to replace one-sided Phase N labeling with synthetic dense exact labeling that explicitly recovers non-improving examples.

### Result

- created and activated Phase O iteration branch for synthetic dense labeling repair
- added new C++ wrapper variant `stageO_synthetic_dense_logging` that reuses dense exact move evaluation (`vnd_exact_dp_insert_rank_dense_labeling`) while writing per-instance outputs
- implemented manifest-driven Python runner `scripts/phaseO_synthetic_dense_labeling.py` using only Phase M train/val manifests
- ran bounded subset (`12 train + 4 val`) and produced mixed-sign exact labels with substantial negative coverage

### Evidence

- `research/learned_move_screening_20260420/history/phase_reports/phaseO_synthetic_dense_labeling_design.md`
- `research/learned_move_screening_20260420/history/phase_reports/phaseO_synthetic_dense_labeling_results.md`
- `research/learned_move_screening_20260420/history/phase_reports/phaseO_synthetic_dense_labeling_readiness.md`
- `research/learned_move_screening_20260420/iterations/20260420_phaseO_synthetic_dense_labeling/SUMMARY.md`
- `scripts/phaseO_synthetic_dense_labeling.py`
- `solvers/cpp/parallel_heuristic_compare.cpp`
- `temp/phaseO_synthetic_dense_labeling/labeling_run_summary.json`
- `temp/phaseO_synthetic_dense_labeling/labeling_subset_aggregate.csv`

### Conclusion

Phase O bounded run passes the strict gate: synthetic exact-labeled data is mixed-sign (not all-positive / not all-negative), so this branch resolves the Phase N labeling-policy blocker and is eligible for controlled scale-up.

## 2026-04-21

### Attempt
Execute Phase P full-manifest synthetic dense labeling, diagnose train/val skew, and freeze one synthetic training-ready dataset.

### Result

- created and activated iteration `20260421_phaseP_full_synthetic_freeze`
- implemented full-manifest/resumable runner `scripts/phaseP_full_synthetic_freeze.py`
- executed all Phase M synthetic manifests (`150 train + 30 val`) with unchanged Phase O dense labeling policy
- produced required machine-readable outputs under `temp/phaseP_full_synthetic_freeze/` including batch progress, batch summary, global/split/bucket summaries, skew variance table, frozen datasets, schema, and freeze manifest
- completed full run with 180/180 successful instances and no retries/failures

### Evidence

- `research/learned_move_screening_20260420/phaseP_full_synthetic_freeze_design.md`
- `research/learned_move_screening_20260420/phaseP_full_synthetic_freeze_results.md`
- `research/learned_move_screening_20260420/phaseP_full_synthetic_freeze_readiness.md`
- `research/learned_move_screening_20260420/iterations/20260421_phaseP_full_synthetic_freeze/SUMMARY.md`
- `scripts/phaseP_full_synthetic_freeze.py`
- `temp/phaseP_full_synthetic_freeze/dataset_summary_global.json`
- `temp/phaseP_full_synthetic_freeze/dataset_summary_by_split.csv`
- `temp/phaseP_full_synthetic_freeze/dataset_summary_by_bucket.csv`
- `temp/phaseP_full_synthetic_freeze/synthetic_moves_exact_labeled_frozen_merged.csv`

### Conclusion

Phase P freeze criteria are satisfied: full synthetic train/val labeling is complete, schema and balance are documented, residual skew is diagnosed and modest, and the branch is ready to start synthetic-only offline model training in the next stage.

## 2026-04-21

### Attempt
Clean the learning-thread markdown surface so re-entry is fast and old phase notes stop cluttering the thread root.

### Result

- added `START_HERE.md` as the single re-entry file for future conversations
- moved stable reusable protocol notes into `reference/`
- moved phase-specific design/results/readiness markdown into `history/phase_reports/`
- kept the thread root limited to the files needed for routing and fast recovery

### Evidence

- `research/learned_move_screening_20260420/START_HERE.md`
- `research/learned_move_screening_20260420/reference/PLAN_supervised_move_ranking.md`
- `research/learned_move_screening_20260420/reference/phaseM_benchmark_role_note.md`
- `research/learned_move_screening_20260420/history/phase_reports/`

### Conclusion

The thread can now be resumed from a new conversation with a short read path: `START_HERE.md` -> `ACTIVE.md` -> current iteration `SUMMARY.md`, with older phase notes separated from the main working surface.

## 2026-04-21

### Attempt
Run Phase Q synthetic-only offline move-ranking training/evaluation on the frozen Phase P train/val dataset.

### Result

- created Phase Q iteration memory for synthetic-only offline ranking evaluation
- trained `XGBoost`, `LightGBM`, and `CatBoost` on frozen synthetic train only
- evaluated against random, `screen_score_s1`, `screen_score_s2`, and oracle baselines on frozen synthetic val only
- learned models produced only modest gains over the strongest handcrafted baseline, with the clearest lift on top-k improvement magnitude and only small recall/precision gains at larger budget
- selected `xgboost` as the default learned candidate from this bounded run

### Evidence

- `research/learned_move_screening_20260420/iterations/20260421_phaseQ_synthetic_offline_training/RESULTS.md`
- `research/learned_move_screening_20260420/iterations/20260421_phaseQ_synthetic_offline_training/SUMMARY.md`

### Conclusion

Phase Q did not justify continuing fine-grained learned move ranking as the main paper direction; the next branch should test an algorithmic mechanism instead.

### Attempt
Create Phase R as a bounded diagnostic for epsilon warm-start continuity on benchmark instance `61`.

### Result

- created and activated iteration `20260421_phaseR_epsilon_warmstart_diagnostic`
- documented the warm-start hypothesis, grounded evidence from Phases I/J/K/Q, and a strict paired warm-start vs fresh-start design
- prepared a detailed coder prompt that can run from the pushed GitHub branch alone

### Evidence

- `research/learned_move_screening_20260420/iterations/20260421_phaseR_epsilon_warmstart_diagnostic/SUMMARY.md`
- `research/learned_move_screening_20260420/phaseR_epsilon_warmstart_diagnostic_design.md`
- `research/learned_move_screening_20260420/phaseR_epsilon_warmstart_diagnostic_coder_prompt.md`

### Conclusion

Phase R is now the active branch. The next step is a single-instance epsilon warm-start diagnostic that can validate or falsify the continuity hypothesis cheaply before any full sweep implementation.

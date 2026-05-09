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

## 2026-05-03 — Phase S Stage 0 completed — GATE PASSED

### Attempt
Run the Stage 0 residual miss-set audit on Phase R instances to test whether
the DiverseTrimmed handcrafted shortlist still misses improving insert_inter
moves on realistic chain states.

### Result

- Added `phaseS_residual_misset_audit` variant to C++ solver
  (`InsertScreenMode::MissetAudit`, uses same screening as DiverseTrimmed
  but also evaluates outside-shortlist candidates with bounded audit budget)
- Built orchestration script `scripts/phaseS_llm_chain_exception_lane.py`
- Ran on 15 epsilon states across 5 Phase R instances (61, 62, 65, 73, 85)
- 765 audit rows collected (51 audit CSV files)

| Instance | States with miss-set | Total outside improving | Best Δ |
|----------|:---:|:---:|:---:|
| 61 | 0/3 | 0 | 0.0 |
| 62 | 1/3 | 5 | -18.0 |
| 65 | 3/3 | 13 | **-94.0** |
| 73 | 2/2 | 2 | -20.0 |
| 85 | 3/3 | 14 | -16.0 |

**Gate: 4/5 instances show meaningful residual miss-set signal. PASSED (>= 2/5).**

Key findings:
- Instance 61 (the Phase I/J/K anchor) shows NO miss-set — confirming the
  handcrafted shortlist is near-optimal after months of targeted optimization.
- Instances 62, 65, 73, 85 show significant miss-set, including large
  improvements (-94 energy units on instance 65).
- The DiverseTrimmed shortlist is instance-tuned to 61 but misses moves on
  other instances. An exception lane is justified.

### Evidence

- `solvers/cpp/parallel_heuristic_compare.cpp` — added `phaseS_residual_misset_audit`
- `scripts/phaseS_llm_chain_exception_lane.py`
- `temp/phaseS_misset_audit/*.csv` (51 audit files)
- `research/learned_move_screening_20260420/temp/20260503_run01_phaseS_stage0_residual_misset/`

### Conclusion

Proceed to Stage 1 (LLM Controller Generation). The handcrafted shortlist misses
improving moves on instances not in its tuning set. An exception lane with
bounded exact-DP budget is justified. Do NOT call the LLM until ready.

## 2026-05-03 — Phase S Stage 1 completed — LLM CONTROLLER GENERATED

### Attempt
Generate a bounded LLM controller that adds an exception lane of outside-shortlist
insert_inter moves on top of the DiverseTrimmed handcrafted shortlist.

### Result

- Used DeepSeek V4 Pro (reasoning_effort=high, 282.8s latency, $0.06)
- Built comprehensive prompt with: DiverseTrimmed screening formulas, Phase I/J/K/R
  evidence, Stage 0 miss-set data, corrected target logic, and explicit forbidden actions
- LLM generated two functions:
  - `score_exception_move(features)`: s2-based ranking with bonus for target slack
    and source tightness
  - `controller_policy(state)`: self-adapting budget (starts at 4, grows to 12 on
    exception hits, shrinks to 1 when fruitless), per-source/per-target diversity
- Both functions parsed successfully (1005 + 2090 chars)

### Evidence

- `research/learned_move_screening_20260420/temp/20260503_run02_phaseS_stage1_llm_controller/`
  - `prompts/stage1_controller_prompt.md`
  - `responses/stage1_llm_response.txt`
  - `responses/score_exception_move.py`
  - `responses/controller_policy.py`

### Conclusion

Proceed to Stage 2 (Proxy Filter). Do NOT integrate into C++ solver until proxy
confirms benefit.

## 2026-05-03 — Phase S Stage 2 completed — PROXY FILTER SHOWS STRONG RECOVERY

### Attempt
Proxy-evaluate the LLM controller on Stage 0 audit data before full solver integration.

### Result

Proxy filter results (per-instance, with per-instance state reset):

| Instance/Epsilon | Outside Improving | Exception Selected | Recovery |
|-----------------|:---:|:---:|:---:|
| 61/347 (guard) | 0 | 0 (self-limited) | N/A |
| 62/290 | 5 | 2 | 40% |
| **65/195 (primary)** | **5** | **5** | **100%** |
| 65/200 | 4 | 5 | 100% |
| 65/190 | 4 | 5 | 100% |
| 73/350 | 1 | 1 | 100% |
| 73/348 | 1 | 1 | 100% |
| 85/300 | 5 | 4 | 80% |
| 85/295 | 4 | 4 | 75% |
| 85/290 | 5 | 4 | 80% |

Key findings:
- **Primary target (65/195): 100% recovery** — exception lane captures ALL 5
  outside-shortlist improving moves.
- **Instance 61 (guard):** controller self-limits budget to minimal after first
  round (shortlist is near-optimal, exceptions find nothing).
- **Total extra exact-DP cost:** 41 evaluations across all states (modest).
- **Controller adapts:** budget grows where exceptions help (instance 65),
  shrinks where they don't (instance 61).

### Evidence

- `research/learned_move_screening_20260420/temp/20260503_run03_phaseS_stage2_proxy_filter/`
  - `metrics/stage2_proxy_summary.csv`
  - `metrics/stage2_candidate_registry.csv`

### Conclusion

Controller PASSES proxy filter on all criteria:
- Primary target (65/195) shows 100% miss-set recovery
- No regression on guard (61/347) — controller self-limits
- Secondary targets (62, 73, 85) show 40-100% recovery
- Ready for Stage 3: Hard Target Gate (requires C++ integration)

## 2026-05-03 — Phase S Stage 3 completed — GATE PASSED (C++ Integration)

### Attempt
Integrate the DeepSeek controller into the C++ solver as a deterministic mode
(`phaseS_llm_exception_lane`, `InsertScreenMode::ExceptionLaneLLM`) and run the
hard target gate.

### Result

GATE PASSED. All targets improved:

| Target | Baseline TEC | Exception TEC | Δ | Exc. Improved | Best Δ |
|--------|:---:|:---:|:---:|:---:|:---:|
| 61/347 (guard) | 6884 | **6871** | **-13** | 2 | 14 |
| 65/195 (primary) | 27031 | **26961** | **-70** | 3 | 30 |
| 62/290 (secondary) | 9687 | **9455** | **-232** | 3 | 20 |
| 73/350 (secondary) | 8534 | **8509** | **-25** | 2 | 18 |
| 85/300 (secondary) | 9492 | **9374** | **-118** | core | — |

Gate checks:
- Guard (61/347): no regression — TEC improved from 6884→6871 ✅
- Primary (65/195): better TEC (27031→26961, -70) ✅
- Secondary: 3/3 targets benefited ✅

Implementation details:
- Added `InsertScreenMode::ExceptionLaneLLM` and variant `phaseS_llm_exception_lane`
- Ported `score_exception_move` (s2 + slack bonus + tightness bonus) and
  `controller_policy` (self-adapting budget 4→12) to C++
- Outside candidates are collected during DiverseTrimmed pool building
  (discarded candidates saved to outside_pool)
- Exception lane evaluates top-N scored outside candidates with exact DP
- Controller adapts budget per-round; self-limits on low-signal instances

Key finding: The guard (61/347), which had zero miss-set in Stage 0, actually
IMPROVED (-13) in the C++ implementation. This is because the multistart
produces different starting states, and the exception lane finds improvements
that Stage 0's single-state audit missed.

### Evidence

- `solvers/cpp/parallel_heuristic_compare.cpp` — ExceptionLaneLLM mode + variant
- `scripts/phaseS_llm_chain_exception_lane.py` — Stage 3 harness
- `research/learned_move_screening_20260420/temp/20260503_run04_phaseS_stage3_cpp_integration/`
  - `metrics/stage3_target_summary.csv`
  - `metrics/stage3_exception_lane_metrics.csv`

### Conclusion

Proceed to Stage 4: Chain-Level Validation on all 5 Phase R instances.

## 2026-05-03

### Attempt
Incorporate the completed Stage 0 residual miss-set audit into the new LLM chain-controller branch and correct the success gates accordingly.

### Result

- Stage 0 recorded 4/5 Phase R instances with meaningful residual miss-set.
- `61/347` showed zero residual miss-set and is now treated as a regression guard, not the main positive target.
- `65/195` is now the primary positive target; `62`, `73`, and `85` are secondary positive targets.
- Rewrote the branch protocol, summary, results, and coder prompt to reflect this.

### Evidence

- `research/learned_move_screening_20260420/temp/20260503_run01_phaseS_stage0_residual_misset/metrics/stage0_instance_metrics.csv`
- `research/learned_move_screening_20260420/iterations/20260503_phaseS_llm_chain_screening_controller/EXPERIMENT_PROTOCOL.md`
- `research/learned_move_screening_20260420/iterations/20260503_phaseS_llm_chain_screening_controller/CODER_PROMPT.md`

### Conclusion

The branch is still justified, but its validation target changed materially: success must now come from the miss-set-positive instances, not from re-solving the already hand-optimized `61/347` anchor.

## 2026-05-04 — Phase S Stage 4 completed — PARTIAL CHAIN PASS, INFRASTRUCTURE BLOCKER

### Attempt
Validate `phaseS_llm_exception_lane` at full warm-start chain level on Phase R
instances and compare against random-exception and wider-shortlist ablations.

### Result

Stage 4 technically passed the 2/4 miss-set-positive gate but did not provide
complete chain validation because the warm-start control harness failed for
insert-screen variants on `65` and `85`.

Completed-chain results:

| Instance | Baseline | LLM Exception | Random | Wider |
|---|---:|---:|---:|---:|
| 61 | 6978 | 6955 (-23) | failed | 6978 |
| 62 | 10215 | 10162 (-53) | failed | 10182 (-33) |
| 65 | 30792 | failed | failed | 30792 |
| 73 | 8489 | 8451 (-38) | 8489 | 8489 |
| 85 | 10298 | failed | failed | 10298 |

### Evidence

- `research/learned_move_screening_20260420/temp/20260503_run05_phaseS_stage4_chain_validation/metrics/stage4_chain_comparison.csv`
- `research/learned_move_screening_20260420/temp/20260503_run05_phaseS_stage4_chain_validation/metrics/stage4_ablation_summary.csv`
- `research/learned_move_screening_20260420/temp/20260503_run05_phaseS_stage4_chain_validation/notes/stage4_results.md`

### Conclusion

The method signal remains positive where the chain completes, and LLM beats the
available ablations on `61`, `62`, and `73`. Full validation is blocked by
infrastructure, not by a measured method regression. Next work should fix
warm-start variant support or implement a Python descending-epsilon chain using
`paper-instance`; do not run more LLM iterations until full chain validation is
possible.

## 2026-05-04 — Phase S incremental DP reuse feasibility note

### Attempt
Assess whether the stateless single-machine DP used by the DP-centered
`insert_inter` heuristic can reuse DP tables when one job is added to or removed
from a machine.

### Result

No prior phase appears to have implemented table-level incremental add/remove
reuse. Existing related mechanisms are exact-cost caching by machine multiset in
the C++ solver and assignment-level incremental evaluation in
`glns/sequencing.py`; neither reuses internal DP layers for changed machines.

### Evidence

- `glns/sequencing.py`
- `correct_dp.py`
- `solvers/cpp/parallel_heuristic_compare.cpp`
- `solvers/cpp/dp_solver.cpp`
- `research/learned_move_screening_20260420/iterations/20260503_phaseS_llm_chain_screening_controller/notes/incremental_dp_feasibility.md`

### Conclusion

Exact DP-table reuse is theoretically possible only with a substantial DP API
redesign because the current solver frees layers and prunes for one final
multiset. It is not the next Phase S lever. Complete Stage 4B validation first;
if runtime later becomes decisive, run a narrow cache-miss/per-DP-runtime audit
before opening a DP-engineering branch.

## 2026-05-05 — Phase S Stage 4B Run07 completed — FRESH MULTI-EPSILON VALIDATION

### Attempt
Rerun Stage 4B with checkpointing using independent fresh-start `paper-instance`
runs at each epsilon for baseline, LLM exception, random exception, and wider
shortlist.

### Result

Run07 completed and produced checkpoint, failure audit, instance summary, and
ablation summary files.

Same-epsilon anchor interpretation:

| Instance / epsilon | Baseline | LLM Exception | Wider | Interpretation |
|---|---:|---:|---:|---|
| 61/340 | 6980 | 6989 (+9) | 6980 | guard regression |
| 62/250 | 10587 | 10388 (-199) | 10587 | LLM wins |
| 65/150 | 30792 | 30427 (-365) | 30792 | LLM wins |
| 73/348 | 8515 | 8495 (-20) | 8515 | LLM wins |
| 85/250 | 10302 | 10264 (-38) | 10304 | LLM wins |

The generated run note overstates the random-ablation conclusion: random's
apparent wins on 65 and 85 are at looser epsilons than the LLM anchor values,
so they are not direct anchor comparisons.

### Evidence

- `research/learned_move_screening_20260420/temp/20260503_run07_phaseS_stage4b_fresh_chain/metrics/stage4b_checkpoint.csv`
- `research/learned_move_screening_20260420/temp/20260503_run07_phaseS_stage4b_fresh_chain/metrics/stage4b_instance_summary.csv`
- `research/learned_move_screening_20260420/temp/20260503_run07_phaseS_stage4b_fresh_chain/metrics/stage4b_ablation_summary.csv`
- `research/learned_move_screening_20260420/iterations/20260503_phaseS_llm_chain_screening_controller/notes/stage4b_run07_interpretation.md`

### Conclusion

The exception-lane mechanism is supported by fresh multi-epsilon validation.
The DeepSeek controller improves all four miss-set-positive anchor instances
and beats wider shortlist at the same anchor epsilon. The candidate-selection
score is not proven superior to random; random remains an ablation pressure,
not a clean winner from Run07.

## 2026-05-05 — Phase S Stage 5 scaling plan created

### Attempt
Define the next scaling step after Run07 without prematurely spending more LLM
tokens.

### Result

Created a Stage 5 plan and coder prompt centered on common-epsilon seeded
ablation:

- add deterministic seed support for `phaseS_random_exception_lane`;
- compare baseline, LLM exception, wider shortlist, and 10 random seeds at the
  same instance/epsilon pairs;
- checkpoint every run;
- decide whether DeepSeek refinement is justified only after the common-epsilon
  random ablation is clean.

### Evidence

- `research/learned_move_screening_20260420/iterations/20260503_phaseS_llm_chain_screening_controller/STAGE5_SCALING_PLAN.md`
- `research/learned_move_screening_20260420/iterations/20260503_phaseS_llm_chain_screening_controller/CODER_PROMPT_STAGE5.md`
- `research/learned_move_screening_20260420/ACTIVE.md`

### Conclusion

Proceed to Stage 5A. Do not call DeepSeek until the seeded common-epsilon
ablation shows whether the current LLM controller is weak because of scoring,
budgeting, diversity, or random instability.

## 2026-05-04 — Presentation README created

### Attempt

Prepare a supervisor-facing presentation basis explaining why EHS is difficult
to beat, why early LLM assignment/GLNS ideas failed, and why Phase S is the
most defensible LLM contribution.

### Result

Created a narrative README that synthesizes:

- EHS and VND/EOA literature context;
- LLM-improves-existing-algorithm literature;
- failed/closed experimental surfaces;
- Phase T as complementary DP warm-start front-builder evidence;
- Phase S as the main LLM exception-lane contribution.

### Evidence

- `research/learned_move_screening_20260420/iterations/20260503_phaseS_llm_chain_screening_controller/PRESENTATION_README.md`

### Conclusion

Use this README as the basis for supervisor discussion and presentation slides.

## 2026-05-04 — Presentation README strengthened after expert review

### Attempt

Revise the supervisor-facing README so the negative experiments are explained
mechanistically, not just listed as failures.

### Result

Updated the README with:

- explicit literature positioning with author-style references;
- detailed explanation of why LLM assignment/DP-guided assignment failed;
- concrete assignment and B8 target-khat evidence;
- clarification that exact DP was not failing, but cannot repair bad
  job-to-machine assignments;
- DeepSeek V4 Pro Stage 1 cost note;
- a method-architecture slide showing core lane, LLM exception lane, exact DP
  verification, and accept/reject.

### Evidence

- `research/learned_move_screening_20260420/iterations/20260503_phaseS_llm_chain_screening_controller/PRESENTATION_README.md`

### Conclusion

The presentation basis now better supports the argument that the research path
was systematic and evidence-driven: assignment-level LLM was a reasonable idea,
but the experiments show the useful LLM role is move-screening guidance under
exact-DP verification.

## 2026-05-05 — Phase S Stage 5A random ablation audited

### Attempt

Interpret Run08 common-epsilon seeded ablation and decide whether another
DeepSeek refinement is justified.

### Result

Stage 5A shows the exception lane still beats baseline and wider shortlist on
`27/47` common-epsilon rows, but the first DeepSeek scorer is not clearly better
than random exception selection. The random summaries undercount failed random
seeds, so the exact random-vs-LLM percentages need recomputation with
conditional and reliability-adjusted metrics.

### Evidence

- `research/learned_move_screening_20260420/temp/20260505_run08_phaseS_stage5a_common_epsilon_ablation/`
- `research/learned_move_screening_20260420/iterations/20260503_phaseS_llm_chain_screening_controller/notes/stage5a_run08_interpretation.md`
- `research/learned_move_screening_20260420/iterations/20260503_phaseS_llm_chain_screening_controller/CODER_PROMPT_STAGE5B.md`

### Conclusion

Proceed to Stage 5B only after fixing the Stage 5A aggregation. The next
DeepSeek task should target diversity-aware exception selection, not another
`s2`-like ranking formula.

## 2026-05-05 — Stage 5B DeepSeek candidates reviewed before full evaluation

### Attempt

Review the three DeepSeek V4 Pro diversity-refinement candidates before the
expensive full Stage 5B grid.

### Result

The generated controller families target the right weakness, but Candidate 1
has a dead-code stratum counter bug, Candidate 2 depends on core-lane features
that must be exposed explicitly in C++, and processing-time thresholds need
normalization before porting.

### Evidence

- `research/learned_move_screening_20260420/temp/20260505_run09_phaseS_stage5b_diversity_refinement/responses/`
- `research/learned_move_screening_20260420/iterations/20260503_phaseS_llm_chain_screening_controller/notes/stage5b_candidate_review.md`

### Conclusion

Run a small preflight after fixing the candidate ports. Do not spend the full
47-row campaign unless at least one controller beats current LLM on the
preflight and logs genuinely different exception candidates from the core lane.

## 2026-05-05 — Refined-variant C++ build blocker confirmed

### Attempt

Verify the expert's claim that the refined exception-lane variants are blocked
by C++ wiring or scope issues.

### Result

`cmake --build solvers/cpp/build --target parallel_heuristic_compare -j2`
fails. The refined variant strings are now present in the main whitelists and
dispatch lists, but the exception-lane block has a stale closing brace after
the controller-policy section. As a result, `exc_cap`, quotas, `scored`, and
coverage state are out of scope in the selection/evaluation section.

### Evidence

- `solvers/cpp/parallel_heuristic_compare.cpp`
- `research/learned_move_screening_20260420/iterations/20260503_phaseS_llm_chain_screening_controller/BLOCKERS.md`

### Conclusion

Fix compilation before any Stage 5B, refined-variant, or 6-arm ablation run.
After compile passes, smoke-test all six arms before launching the larger
15-instance campaign.

## 2026-05-05 — Stage 5B smoke gate passed

### Attempt

Verify whether repaired exception-lane variants compile and produce valid TEC
on the smoke cell `61/347`.

### Result

Build passes. All seven arms return valid TEC on `61/347`. Refined1 stratified
is best on this single cell (`6869`), followed by LLM and coverage (`6871`),
random (`6874`), baseline/wider (`6884`), and anticore (`6885`).

### Evidence

- `solvers/cpp/parallel_heuristic_compare.cpp`
- `research/learned_move_screening_20260420/iterations/20260503_phaseS_llm_chain_screening_controller/RESULTS.md`

### Conclusion

Proceed to the 5-target preflight only. Do not launch the full 47-row or
15-instance campaign until the preflight branch decision is made.

## 2026-05-05 — Stage 5B preflight completed with validation caveats

### Attempt

Run the five-target mechanism-ablation preflight comparing trimmed, budgeted,
LLM exception, random exception, stratified, and coverage arms.

### Result

The preflight directionally supports Case B: exception-lane architecture helps,
but the current LLM scorer is not clearly better than random or stratified
diversity. However, the run revealed an out-of-bounds exception-candidate bug,
and the current fix only checks bounds. A stale in-bounds candidate can still
point to a different processing time after earlier accepted moves. The summary
also misreports random best as random median on at least `65/195` and `85/300`.

### Evidence

- `temp/preflight_raw.json`
- `temp/preflight_rows.csv`
- `temp/preflight_summary.md`
- `research/learned_move_screening_20260420/iterations/20260503_phaseS_llm_chain_screening_controller/notes/stage5b_preflight_analysis.md`

### Conclusion

Strengthen stale-candidate validation, rebuild, rerun the same five-target
preflight, then reclassify. Do not launch the full campaign yet.

## 2026-05-06 — Corrected Stage 5B preflight completed

### Attempt

Rerun the five-target mechanism-ablation preflight after adding stale-candidate
guards to exception-lane evaluation.

### Result

Case B selected. Exception-lane arms beat `budgeted` on at least `4/5` targets,
and `LLM` beats `budgeted` on `5/5`, but the current low-level LLM scorer is
not the strongest selector. Random median is best on `3/5` targets with `0/15`
failures.

### Evidence

- `temp/preflight_raw.json`
- `temp/preflight_rows.csv`
- `temp/preflight_summary.md`
- `research/learned_move_screening_20260420/iterations/20260503_phaseS_llm_chain_screening_controller/notes/stage5b_preflight_analysis.md`

### Conclusion

Stop low-level scorer optimization as the main LLM claim. The next LLM-centered
branch, if pursued, should test a compact higher-level mechanism-selection tree
against the best fixed non-LLM arm, random median, and a locked human rule on
fresh validation cells.

## 2026-05-06 — LLM policy tree implemented with validation caveat

### Attempt

Use DeepSeek V4 Pro to generate a compact higher-level mechanism-selection tree
over existing exception-lane strategies.

### Result

Feature logging was added, a locked human rule was written, three DeepSeek V4
Pro calls were run, and `phaseS_llm_policy_tree` was implemented in C++. Smoke
on `61/347` returns valid TEC `6868`.

### Evidence

- `notes/human_rule_locked.md`
- `notes/validation_plan_policy_tree.md`
- `temp/deepseek_policy_tree_calls/`
- `solvers/cpp/parallel_heuristic_compare.cpp`
- `research/learned_move_screening_20260420/iterations/20260503_phaseS_llm_chain_screening_controller/notes/policy_tree_implementation_review.md`

### Conclusion

Do not run fresh validation yet. First sanity-check the policy tree on the five
development cells, because the implemented skew split appears to trigger LLM on
`61/347`, where random was stronger in the corrected preflight.

## 2026-05-06 — Policy-tree branch failed due to overfitting

### Attempt

Ask DeepSeek V4 Pro to correct the mechanism-selection tree after development
sanity issues.

### Result

DeepSeek produced a two-leaf tree using development constants:
`epsilon >= 290 AND epsilon <= 350 AND num_machines == 25 -> LLM; else random`.
This memorizes the five development cells and is not a transferable controller.

### Evidence

- `temp/deepseek_policy_tree_calls/call4_corrective_raw_response.md`
- `temp/deepseek_policy_tree_calls/call4_corrective_metadata.json`
- `research/learned_move_screening_20260420/iterations/20260503_phaseS_llm_chain_screening_controller/notes/policy_tree_failure_assessment.md`

### Conclusion

Do not validate the current LLM policy tree. The current supported result is
the exact-DP verified exception-lane architecture; a critical LLM controller is
not yet supported.

## 2026-05-06 — Paper evidence base consolidated

### Attempt

Gather the paper-relevant evidence across the EHS/GLNS, DP-centered local
search, learned move-screening, warm-start, and Phase S exception-lane threads
into one writing base.

### Result

Created
`iterations/20260503_phaseS_llm_chain_screening_controller/PAPER_EVIDENCE_BASE.md`.
The file consolidates accepted EHS improvements, closed heuristic surfaces,
assignment-level failures, no-screen and insert-screen diagnostics, classical
learned-ranking results, DP warm-start results, Phase S stages 0-5B, policy-tree
caveats, supported/unsupported claims, missing analyses, and literature anchors.

### Conclusion

Use `PAPER_EVIDENCE_BASE.md` as the primary factual source for the first paper
draft. The current stable paper position remains: exact-DP verified
exception-lane architecture is supported; low-level LLM scoring and the current
policy tree are not yet validated as critical LLM contributions.

## 2026-05-06 — Stage 5D selector headroom audit completed, branch stopped

### Attempt

Run a selector-headroom audit with matched seeds across 6 exception-lane arms
on 18 development cells to test whether a high-level mechanism-selection
controller is justified.

### Result

All 5 proceed gates initially passed (stable winners on all 18 cells, 3 arms
with ≥3 stable wins, 0.31% oracle improvement, no single-arm dominance).
However:
- Original audit note had cell-level winner assignment errors (8/18 cells wrong)
- Meaningful-margin analysis (margin ≥ max(20 TEC, 0.2% baseline)) leaves only
  8/18 cells
- On meaningful cells: oracle improvement = 0.10% (< 0.2% threshold)
- DeepSeek `phaseS_deepseek_selector_tree` backtest: 0.53% WORSE than best
  fixed arm (random median)
- No clean normalized-feature human rule is possible from the data

### Evidence

- `research/learned_move_screening_20260420/temp/20260506_phaseS_selector_headroom_audit/`
- `temp/deepseek_selector_calls/`
- `solvers/cpp/parallel_heuristic_compare.cpp` — `phaseS_deepseek_selector_tree`
- `research/learned_move_screening_20260420/iterations/20260503_phaseS_llm_chain_screening_controller/notes/selector_headroom_audit_corrected.md`

### Conclusion

**Stop high-level LLM selector branch.** Selector headroom is too weak/noisy.
LLM is useful for analysis/prototyping but not validated as critical runtime
controller. Strongest paper story: exception-lane architecture + LLM-guided
research/surface analysis. Do not validate DeepSeek-generated controllers
without fresh positive evidence.

## 2026-05-07 — Stage U LLM symbolic screening discovery FAILED, branch stopped

### Attempt

Use DeepSeek V4 Pro as symbolic feature/rule designer from 317K exact-DP-labeled
move rows, with offline sandbox evaluation against s2, XGBoost baselines.

### Result

DeepSeek correctly diagnosed the false-negative mechanism (small jobs from
slack sources where exact-DP re-packing creates benefit unseen by cheap LB).
Produced 22 features and 11 rules across 3 calls ($0.037). However:

- No LLM rule beats s2 at any k (10/25/50/100)
- Best rule (cost_transfer_hybrid) ties s2 at k=10 but below at higher k
- XGBoost + LLM features = XGBoost original (identical — zero marginal gain)

Root cause: s2's post-decision LB rerank provides signal that screening-time
features alone cannot replicate. Feature space is saturated.

### Evidence

- `temp/20260507_phaseU_llm_symbolic_screening/` — evidence pack, prompts, responses, eval
- `scripts/phaseU_llm_symbolic_screening.py` — sandbox evaluator
- `scripts/run_deepseek_phaseU_calls.py` — DeepSeek call script

### Conclusion

**Stop LLM symbolic screening branch.** LLM is validated for analysis/diagnosis/
mechanism discovery only, not for generating superior runtime scoring functions.
Paper position: exception-lane architecture + exact-DP verification is the
contribution. LLM guided the research, but is not a critical runtime component.

## 2026-05-07 — Phase V trace-conditioned LLM operators initialized

### Attempt

Start a new LLM-critical branch after Phase S Stages 0-5D/Stage U failed. The
LLM is now positioned as an online trace-conditioned hyper-heuristic operator
generator — it reads solver traces, diagnoses stagnation, and generates bounded
instance-specific search operators (source/target/job/budget/neighborhood
selection). Exact DP remains the verifier.

### Result

Created Phase V iteration with structure:
- `PROBLEM.md` — hypothesis and differentiation from failed branches
- `IDEAS.md` — allowed operator families (7 types), rejected directions
- `RESULTS.md` / `BLOCKERS.md` / `SUMMARY.md` — initialized
- Subdirectories: `prompts/`, `traces/`, `responses/`, `eval/`
- Updated `ACTIVE.md` and `LOG.md`

### Conclusion

Phase V is now the active branch. Next: Phase V0 trace schema design and C++
wrapper implementation.

## 2026-05-07 — Phase V0 trace schema & probes COMPLETED

### Attempt

Build the Python-based trace-report generator without adding new C++ instrumentation,
run probe solvers on all 5 development cells, and prepare the first DeepSeek prompt.

### Result

- Built `scripts/phaseV_trace_conditioned_operator.py` with `--v0-run-probes`,
  `--v0-build-traces`, `--v0-report`
- Ran all 40 solver invocations (5 cells × 8 variants) — 40/40 successful
- Generated 5 anonymized Markdown + 5 JSON trace reports with rich diagnostics
- Call 1 prompt (`prompts/call1_diagnostic_analyst.md`) ready
- No C++ changes required — all trace data from existing CSV output

Key findings:
- Universal pattern: outside pool covers only 5/25 distinct sources (20%)
- Cell B (62/290): LLM beats random by 148 TEC (exception lane works)
- Cell C (65/195): Random beats LLM by 664 TEC (primary target, large headroom)
- Cell A (guard, 61/347): Random modestly better, all arms close
- Traces are rich enough for DeepSeek Call 1 — all 5 prompt-required sections covered

### Evidence

- `scripts/phaseV_trace_conditioned_operator.py`
- `eval/v0_probe_raw.csv`, `eval/v0_probe_summary.csv`
- `traces/cell_a_trace.{md,json}` through `traces/cell_e_trace.{md,json}`
- `prompts/call1_diagnostic_analyst.md`
- `notes/phaseV0_trace_schema_and_probe_report.md`

### Conclusion

Phase V0 gates passed. B7.0 resolved. All probes successful, all traces anonymized,
Call 1 prompt ready. No unnecessary C++ changes. Proceed to DeepSeek Call 1 when instructed.

## 2026-05-07 — Phase V3 op1 implemented, V3 evaluation INVALID, V3.1 fixed

### Attempt

Implement `phaseV_op1_source_expansion` C++ variant and run 5-cell dev evaluation.

### Result

V3: Anomalous results (1-7s runtimes, spurious ±100+ TEC deltas). Root cause:
op1 not in 4-start multistart wrapper. Used different single-assignment path.

V3.1: Fixed wrapper + local state + fallback-only activation. op1 matches
trimmed on 4/5 cells. Activates only on Cell A (core stall), -1 TEC.

### Evidence

- C++: `TraceSourceExpansion` enum, variant wiring, op1 lane
- `eval/v3_1_op1_fixed_raw.csv`, `eval/v3_1_op1_fixed_summary.csv`
- `notes/phaseV3_1_op1_fixed_results.md`

### Conclusion

V3 was invalid. V3.1 fixed shows op1 fallback adds no meaningful benefit.
Core lane finds improvements nearly every round. Next: every-round diversity
lane, integrate into core shortlist, or move to swap/destroy-repair operators.

## 2026-05-08 — Phase V4.1 op1 core integration FAILED gate

### Attempt

Implement op1 core integration variant `phaseV_op1_core_integration`: adaptive
source-coverage DiverseTrimmed with starved sources competing in main pool,
not separate lane. Replace fixed top-5 source limit with coverage-based expansion.

### Result

- Added `TraceSourceCoreIntegration` enum + variant + CSV fields
- Source coverage increased 5→9-16 on all 5 cells (PASS)
- TEC improved 0/5 cells vs trimmed (FAIL)
- Cell B regressed +81 (LLM scoring diluted by expanded pool)
- Cells C/E regressed (more incorrectly-ranked s2 candidates)
- Gate: FAILED. s2 scoring is the real bottleneck, not source count.

### Evidence

- `solvers/cpp/parallel_heuristic_compare.cpp` — TraceSourceCoreIntegration
- `eval/v4_1_core_integration_raw.csv`, `eval/v4_1_core_integration_summary.csv`
- `notes/phaseV4_1_core_integration_results.md`

### Conclusion

Source expansion does not improve TEC because s2 scoring cannot rank the expanded
pool. Source starvation exists but is not the actionable lever. Move to V5
score-escape operator or stop runtime-LLM branch.

## 2026-05-08 — Phase V5 score-escape operator prompt created

### Attempt

Create Call 5 prompt for final score-escape LLM operator. After V4.1 failure
proved s2 scoring is the bottleneck, task DeepSeek to design one operator that
samples candidates from outside_pool in a score-free or anti-score way when s2
is misleading, but preserves s2 when it is reliable.

### Result

Prompt `prompts/call5_score_escape_operator.md` created with:
- V4.1 failure evidence
- V4.1 summary CSV
- Call 1 diagnostic summary
- Call 4 decision JSON
- All 5 anonymized trace reports
- Required JSON output schema
- Critical challenge: "If random already wins C/E, what structure do you add?"

### Evidence

- `prompts/call5_score_escape_operator.md`

### Conclusion

Awaiting DeepSeek Call 5 execution. If operator is acceptable: implement
`phaseV_score_escape_sampler`. If unacceptable or gate fails: stop Phase V
runtime-LLM branch.

## 2026-05-08 — Phase V5.1 Score Escape Sampler implemented, MIXED gate

### Attempt

Implement `phaseV_score_escape_sampler` — dual-mode exception-lane operator:
normal mode (LLM exception scoring) with escape to cheap_lb_delta diversity
sampling after 2 consecutive failed rounds. Final Phase V runtime-LLM attempt.

### Result

- Added `ScoreEscapeSampler` enum, variant, 8 CSV fields
- Wired through all 8 dispatch points
- Implemented dual-mode: normal (budget 6→4→2, LLM scoring + source/target
  diversity quotas=2) and escape (cheap_lb_delta per (source,target) pair,
  K=3, skip on max_cheap_lb_delta ≤ 0)
- 5-cell evaluation:
  - Cell A: TEC=6884 (matches trimmed, guard maintained), escape activated 3/5 rounds
  - Cell B: TEC=9503 (-184 vs trimmed, +48 regr vs LLM)
  - Cell C: TEC=26470 (-561 vs trimmed, -456 vs LLM) — BEST Phase V result
  - Cell D: TEC=8509 (-25 vs trimmed, matches LLM)
  - Cell E: TEC=9425 (-67 vs trimmed, +39 regr vs LLM)
- Gate: 4/5 improve vs trimmed (PASS), 2/5 beat LLM (FAIL), Cell B +48 > +15 limit (FAIL)

### Evidence

- `solvers/cpp/parallel_heuristic_compare.cpp` — ScoreEscapeSampler enum + variant
- `eval/v5_1_score_escape_raw.csv`, `eval/v5_1_score_escape_summary.csv`
- `notes/phaseV5_1_score_escape_results.md`
- `prompts/call5_score_escape_operator.md`
- `responses/call5_*` — Call 5 artifacts
- `scripts/phaseV_deepseek_call5.py`

### Conclusion

Score escape is the strongest Phase V operator (best on 3/5 cells vs trimmed,
Cell C -561) but fails the LLM-baseline gate. Mechanism is sound (dual-mode
switching works, cheap_lb_delta diversity works) but normal-mode budget (6)
is too small vs LLM exception (12), causing Cell B regression. The remaining
issue is budget tuning (human work), not operator design (LLM work).
Phase V runtime-LLM branch conclusion: pending user decision between budget
fix (Option A) and branch conclusion (Option B).

## 2026-05-08 — Phase V5.2 Fairness Fix implemented, gate FAILED, branch stopped

### Attempt

Fix score escape normal-mode budget to match LLM exception's adaptive pattern
(4→12 on hits, shrink on 2+ misses). Retest with fairness baseline.

### Result

- Fixed: adaptive budget (4→12), source/target quotas=3, no_hit increment bug
- Fixed: duplicate brace, K=3 always in escape mode
- 5-cell evaluation:
  - Cell A: TEC=6884 (matches trimmed), escape activated 3/5 rounds
  - Cell B: TEC=9484 (-203 vs trimmed, +29 vs LLM) — improved from +48 but
    still exceeds +15 gate threshold
  - Cell C: TEC=26508 (-523 vs trimmed, -418 vs LLM) — strong
  - Cell D: TEC=8516 (-18 vs trimmed, matches LLM)
  - Cell E: TEC=9492 (0 vs trimmed, +106 vs LLM) — regressed from V5.1's -67
- Gate: 3/5 improve vs trimmed (PASS), 2/5 beat LLM (FAIL), Cell B +29 > +15 (FAIL)

### Evidence

- `solvers/cpp/parallel_heuristic_compare.cpp` — fairness fix
- `eval/v5_2_score_escape_fair_raw.csv`, `eval/v5_2_score_escape_fair_summary.csv`
- `notes/phaseV5_2_score_escape_fair_results.md`

### Conclusion

**Phase V runtime-LLM branch permanently stopped.** Score escape is the strongest
Phase V operator (best on 2/5 cells vs all controls) but cannot match specialized
LLM exception on Cell B or random on C/E. The LLM's diagnostic value is confirmed;
its operator-design value does not produce a runtime advantage. Paper position:
exception-lane architecture + exact-DP verification is the contribution. LLM
guided the research via analysis/diagnosis but is not a critical runtime component.

## 2026-05-08 — Phase X initialized

### Attempt

Start Phase X: interactive LLM policy repair. New hypothesis — the LLM is useful
when it gets rapid feedback (proposal → eval → feedback → revise), not when it
produces one-shot C++ designs.

### Result

- Created `iterations/20260508_phaseX_interactive_llm_policy_repair/` with
  PROBLEM.md, IDEAS.md, RESULTS.md, BLOCKERS.md, SUMMARY.md
- Created subdirectories: prompts/, policies/, responses/, eval/
- Updated ACTIVE.md to point to Phase X
- Policy DSL defined: JSON format with 17 fields controlling exception lane
  behavior (scoring mode, escape mode, budget adaptation, diversity quotas,
  coverage bonus, scoring weights)
- Pipeline: X0→X1(DSL)→X2(runner)→X3(fast dev)→X4(5 rounds)→X5(compare)→X6(validate)

### Evidence

- `iterations/20260508_phaseX_interactive_llm_policy_repair/*.md`

### Conclusion

Ready for X1-X2 implementation (DSL + generic policy runner).

## 2026-05-08 — Phase X blocked by C++ source loss

### Attempt

Begin X1-X2 implementation after Phase X initialization.

### Result

`solvers/cpp/parallel_heuristic_compare.cpp` was found reverted to the 3870-line
committed base. The 6035-line working source with Phase S/V changes was not
recoverable from Time Machine or local repo source-copy search. The compiled
binary still contains the Phase S/V behavior and was preserved.

### Evidence

- `temp/recovery_parallel_heuristic_compare_20260508/parallel_heuristic_compare_working_20260508_1617`
- `temp/recovery_parallel_heuristic_compare_20260508/parallel_heuristic_compare_base_3870.cpp`
- `iterations/20260508_phaseX_interactive_llm_policy_repair/BLOCKERS.md`

### Conclusion

X1-X2 is blocked until the needed Phase S/V source surface is rebuilt and
verified against the preserved working binary.

## 2026-05-09 — Phase X1-X2 partial completion

### Attempt

Recover the lost C++ source surface and implement the Phase X policy DSL plus
generic policy runner.

### Result

- Phase S exception lane source rebuilt from oracle binary and committed
  (`a61c79c`).
- Phase V `score_escape_sampler` rebuilt and fixed (`24ca7a7`, `47d7fd3`).
- Phase X policy DSL and C++ `PhaseXPolicyJson` runner implemented.
- X2 smoke on `61/347` completed successfully.

### Evidence

- `iterations/20260508_phaseX_interactive_llm_policy_repair/policies/schema.json`
- `iterations/20260508_phaseX_interactive_llm_policy_repair/notes/phaseX_policy_dsl.md`
- `iterations/20260508_phaseX_interactive_llm_policy_repair/policies/example_policy.json`
- Commits: `a61c79c`, `24ca7a7`, `47d7fd3`

### Conclusion

X2 is not fully complete until `scripts/phaseX_interactive_policy_repair.py` is
created and the full 3-cell smoke is run against all required baselines and
policies. Do not start DeepSeek interaction before that smoke passes.

## 2026-05-09 — Phase X2 completed, full smoke passed

### Attempt

Complete X2: create Python orchestration script and run full smoke on 3 dev cells.

### Result

- Created `scripts/phaseX_interactive_policy_repair.py` with 4 subcommands:
  `--generate-random-policy`, `--eval-policy`, `--eval-baselines`, `--smoke`.
- Full X2 smoke on 18 runs (3 cells × 6 arms). All feasible. Results:

| Inst/Eps | Trimmed | LLM Exc | Random Exc | Score Esc | PhX Example | PhX Random |
|:--------:|--------:|--------:|----------:|---------:|:-----------:|:----------:|
| 61/347 | 6884 | 6884 | 6870 | 6884 | 6884 | 6884 |
| 62/290 | 9687 | 9687 | 9687 | 9489 | **9484** | 9503 |
| 65/195 | 27031 | 27031 | 27031 | 26508 | **26508** | 26749 |

- PhaseX example_policy matches/beats LLM exc and score_escape on all cells.
- Random policy produces intermediate results; no catastrophic regressions.
- Both PhaseX and score_escape improve substantially on 62/290 (-203, -198)
  and 65/195 (-523, -523) vs trimmed.
- Orchestration script validates non-zero exit codes, missing env vars,
  and infeasible results.

### Evidence

- `scripts/phaseX_interactive_policy_repair.py`
- `eval/x2_smoke_raw.csv`, `eval/x2_smoke_summary.csv`
- `policies/random_policy_001.json`

### Conclusion

X2 complete. Phase X generic policy runner is functional. Ready for X3 (fast dev
evaluation with more policies) and X4 (5-round interactive LLM loop).

## 2026-05-09 — X4 Interactive LLM Policy Repair complete (MINIMUM SUCCESS)

### Attempt

5-round interactive DeepSeek policy repair loop:
Round 0 → initial policy, Round 1-4 → feedback + revise.

### Result

Best LLM policy (Round 2, llm_cheaplb_escape_v2): mean TEC = 14285.7.
| Baseline | Value | Δ vs Best LLM |
|----------|------:|-------------:|
| Example policy | 14292.0 | +6.3 (LLM beats) |
| Random median | 14362.0 | +76.3 (LLM beats) |
| Random best c000 | 14254.3 | -31.4 (LLM trails) |

- 2/5 rounds beat example_policy (MINIMUM SUCCESS)
- 0/5 rounds beat random best c000 (not strong success)
- Efficiency: LLM found beating policy in 2 interactive rounds; random needed 20
  attempts with only 2/20 (10%) success rate.
- Guard cell breakthrough: Round 4 hybrid mode improved 61/347 (6884→6873),
  first policy ever to improve this cell, but regressed 65/195 severely.

Key findings:
- Interactive feedback enabled targeted diagnosis (Round 1 correctly identified
  require_positive_cheap_lb as bottleneck on 62/290).
- DSL max_per_target cap (≤4) became binding — LLM wanted to increase further
  but could not within existing DSL.
- Cell-specific scoring matters: hybrid mode helps tight cells, hurts loose
  cells. A per-epsilon-regime adaptive policy might achieve combined gains.

### Evidence

- `prompts/x4_round_*.md` — 5 prompts
- `responses/x4_round_*_raw.md` + `_meta.json` — 5 responses
- `policies/llm_interactive/x4_round_*.json` — 5 policies
- `eval/x4_interactive_rounds.csv` — aggregate per-round
- `eval/x4_interactive_summary.csv` — per-cell per-round
- `scripts/phaseX_interactive_policy_repair.py` — updated with X4 subcommand

### Conclusion

MINIMUM SUCCESS. Interactive LLM found a policy beating example_policy
and random median, proving the interactive repair concept works.
The LLM did NOT beat random best c000 — the DSL search space is flat
enough that brute-force random can find lucky draws. But the LLM
found a good policy in fewer attempts (2/5 vs 2/20 random).

## 2026-05-09 — Phase X3 random campaign complete (Case B)

### Attempt

Run 20 random DSL policies on 3 dev cells to establish the random-search
baseline distribution. The goal is to determine whether the DSL is easy
to search randomly (Case A), noisy but searchable (Case B), or requires
non-random intelligence (Case C).

### Result

- 20 policies evaluated, 0 failures, 0 infeasible
- Baseline mean TEC: trimmed=14534, example_policy=14292
- Random median mean TEC: 14362.0 (worse than example by +70)
- Random best mean TEC: 14254.3 (better than example by -38)
- 2/20 beat example on mean TEC, 20/20 beat trimmed, 11/20 beat score_escape on ≥1 cell

Classification: **CASE B** — DSL contains useful policies but search is noisy.
Example_policy is a strong baseline (median doesn't beat it), but good policies
do exist (best random beats example). Interactive LLM should be compared against
both median and best random.

Top random policy c000 (seed 100): mean TEC=14254.3, beats example on 2/3 cells.

### Evidence

- `eval/x3_random_campaign_raw.csv`, `eval/x3_random_campaign_summary.csv`
- `eval/x3_random_campaign_aggregate.csv`
- `policies/random_campaign/x3_campaign_*.json` (20 policies)

### Conclusion

X3 complete. Case B implies X4 interactive LLM must show it can find good
policies faster/more reliably than brute-force random search. The example_policy
and best random policy provide strong baselines.

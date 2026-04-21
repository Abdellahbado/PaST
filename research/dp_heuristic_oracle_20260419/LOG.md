# Log

## 2026-04-19

### Attempt
Start a new research direction where the single-machine DP is used as a heuristic machine oracle rather than as an exact global proof engine.

### Result
Created a separate research thread with a staged plan:

- Phase A: isolate ESR replacement at fixed `epsilon`
- Phase B: only if Phase A succeeds, test DP-guided assignment scoring
- Phase C: validate heuristic quality against exact fixed-`epsilon` values

### Evidence

- `research/dp_heuristic_oracle_20260419/OVERVIEW.md`
- `research/dp_heuristic_oracle_20260419/LITERATURE.md`
- `research/dp_heuristic_oracle_20260419/DP_HEURISTIC_ORACLE_PLAN.md`
- `research/dp_heuristic_oracle_20260419/CODER_PROMPT_PHASE_A_ESR_REPLACEMENT.md`
- `research/dp_heuristic_oracle_20260419/CODER_PROMPT_PHASE_B_DP_GUIDED_ASSIGNMENT.md`

### Conclusion
The new branch is intentionally narrow and empirical. It does not assume that the DP-based heuristic will beat EHS. It first asks whether ESR replacement alone gives a measurable quality gain.

## 2026-04-19

### Attempt
Execute Phase A ESR replacement experiment with fixed assignment:

- implement comparison driver (`greedy_esr` vs `greedy_dp`)
- run required rows on corrected benchmark root
- compare against CP-SAT exact fixed-`epsilon` references

### Result
Phase A implementation completed and tested.

- Added `solvers/cpp/parallel_heuristic_compare.cpp`
- Updated `solvers/cpp/CMakeLists.txt`
- Ran rows: `46/120`, `61/350`, `64/77`, `90/82`

Observed TEC (`greedy_esr` -> `greedy_dp`):

- `46/120`: `118 -> 112`
- `61/350`: `7263 -> 7053`
- `64/77`: `30640 -> 30598`
- `90/82`: `53300 -> 53294`

### Evidence

- `research/dp_heuristic_oracle_20260419/phaseA_esr_replacement_results.md`
- `research/dp_heuristic_oracle_20260419/phaseA_esr_replacement_readiness.md`
- `research/dp_heuristic_oracle_20260419/iterations/20260419_phaseA_esr_replacement/RESULTS.md`

### Conclusion

Phase A success criterion is met on the tested subset:

- DP machine optimization clearly improves TEC over ESR on most tested rows (here: all tested rows).

Proceed to Phase B is justified.

## 2026-04-19

### Attempt
Correct the Phase A assignment-conditioned LB bug, rerun Phase A, then execute Phase B immediately if Phase A still passes.

### Result

Phase A correction:

- fixed `solvers/cpp/parallel_heuristic_compare.cpp` LB computation to respect assigned multiset multiplicities
- added strict LB safety guard and safe slot-based fallback
- reran full Phase A subset

Phase A corrected TEC (`greedy_esr` -> `greedy_dp`):

- `46/120`: `118 -> 112`
- `61/350`: `7263 -> 7053`
- `64/77`: `30640 -> 30598`
- `90/82`: `53300 -> 53294`

Phase B execution (`greedy_dp` vs `dp_guided_assignment_dp`):

- `46/120`: `112` vs `154` (worse)
- `61/350`: `7053` vs `10402` (worse)
- `64/77`: `30598` vs `30580` (better)
- `90/82`: `53294` vs `53294` (tie)

### Evidence

- `research/dp_heuristic_oracle_20260419/phaseA_esr_replacement_results.md`
- `research/dp_heuristic_oracle_20260419/phaseA_esr_replacement_readiness.md`
- `research/dp_heuristic_oracle_20260419/phaseB_dp_guided_assignment_results.md`
- `research/dp_heuristic_oracle_20260419/phaseB_dp_guided_assignment_readiness.md`

### Conclusion

- Phase A lower-bound issue is corrected and invalid prior claim is removed.
- Phase A still passes on TEC improvement.
- Phase B does not pass its decision rule (no clear additional gain over `greedy_dp`).

## 2026-04-19

### Attempt
Start Phase C branch: keep baseline assignment, then run bounded DP-evaluated local search (`relocate` + `swap`) and compare against `greedy_dp` on required fixed-`epsilon` rows.

### Result

- Added `greedy_dp_local_search` to `solvers/cpp/parallel_heuristic_compare.cpp`.
- Added local-search metrics to CSV output (accepted/evaluated moves, dominant move type, final loads, local-search exact DP calls).
- Ran all required rows one-by-one with `/usr/bin/time -l`.

TEC (`greedy_dp` -> `greedy_dp_local_search`):

- `46/120`: `112 -> 103`
- `61/350`: `7053 -> 7046`
- `64/77`: `30598 -> 30580`
- `90/82`: `53294 -> 53294`

### Evidence

- `solvers/cpp/parallel_heuristic_compare.cpp`
- `research/dp_heuristic_oracle_20260419/iterations/20260419_phaseC_dp_local_search/RESULTS.md`
- `research/dp_heuristic_oracle_20260419/phaseC_dp_local_search_results.md`
- `research/dp_heuristic_oracle_20260419/phaseC_dp_local_search_readiness.md`

### Conclusion

- Phase C is positive on this subset: local search improves or ties `greedy_dp` on all tested rows.
- Runtime overhead is noticeable but manageable; RSS remains far below 16 GB.
- Continue branch with focus on pruning expensive non-improving swap evaluations.

## 2026-04-19

### Attempt
Run a multi-path Phase C refinement search (not a single-path tweak): evaluate relocate-only, screened swap, and machine-priority local search variants under a two-stage protocol.

### Result

- Implemented in `solvers/cpp/parallel_heuristic_compare.cpp`:
  - `greedy_dp_local_search_relocate_only`
  - `greedy_dp_local_search_screened_swap`
  - `greedy_dp_local_search_priority_machines`
- Fixed reporting debt by splitting exact DP calls:
  - `exact_dp_calls_initial`
  - `exact_dp_calls_local_search_only`
- Stage 1 (`46/120`, `61/350`) selected `relocate_only` and `priority_machines` for Stage 2.
- Stage 2 (`64/77`, `90/82`) confirmed:
  - `relocate_only` matches baseline local-search TEC with materially fewer evaluations
  - `priority_machines` best on `61/350` but unstable across rows
  - screened-swap path had no clear return

### Evidence

- `research/dp_heuristic_oracle_20260419/phaseC_refinement_search_plan.md`
- `research/dp_heuristic_oracle_20260419/phaseC_refinement_results.md`
- `research/dp_heuristic_oracle_20260419/phaseC_refinement_readiness.md`

### Conclusion

- Continue with relocate-only as the default refinement path.
- Keep machine-priority mode as optional for hard rows.
- Stop screened-swap path for now.

## 2026-04-19

### Attempt
Start new Phase D branch to test paper-style history-aware repair across decreasing `epsilon` (`Khat`) with DP used only in machine-local roles.

### Result

- Created Phase D iteration and switched active branch.
- Verified paper artifact reality:
  - paper repo contains results/instances only, not implementation source code.
- Implemented in `solvers/cpp/parallel_heuristic_compare.cpp`:
  - history-chain mode: `paper-history-chain`
  - `history_repair_dp_ranked`
  - `history_repair_priority_displaced_relocate`
  - added repair metrics to CSV (`epsilon_prev`, displaced/reinsertion/exact-DP repair counters, post-repair exact-DP counters).
- Ran paper-aligned transitions:
  - `46: 77 -> 73`
  - `61: 347 -> 345`
  - `64: 79 -> 77`
  - `90: 84 -> 82`
- Baselines run at same epsilons:
  - one-shot `greedy_dp`
  - one-shot `greedy_dp_local_search_relocate_only`
  - paper EHS stored fronts at same `epsilon` where available.

### Evidence

- `research/dp_heuristic_oracle_20260419/phaseD_history_repair_search_plan.md`
- `research/dp_heuristic_oracle_20260419/phaseD_history_repair_results.md`
- `research/dp_heuristic_oracle_20260419/phaseD_history_repair_readiness.md`
- `temp/paper_exact_repo/results/EHS/1/res_46.csv`
- `temp/paper_exact_repo/results/EHS/1/res_61.csv`
- `temp/paper_exact_repo/results/EHS/1/res_64.csv`
- `temp/paper_exact_repo/results/EHS/1/res_90.csv`

### Conclusion

- The likely missing strength is indeed continuity-aware repair, not ESR replacement alone.
- `history_repair_priority_displaced_relocate` shows useful signal:
  - improves over one-shot relocate on `61`
  - beats same-`epsilon` paper EHS at `46/73`.
- robustness is still insufficient due to chain infeasibility on `64` and `90` tight steps.
- Continue only prioritized+relocate repair; stop dp-ranked-only repair for now.

## 2026-04-19

### Attempt
Run a focused design-and-prototype pass to choose the next serious branch after Phase D by comparing:

- full robust history sweep
- multi-start randomized assignment plus local search
- stronger post-repair rescue neighborhood

### Result

- Added design memo with explicit ranking and rationale:
  - `research/dp_heuristic_oracle_20260419/phaseE_next_branch_design_memo.md`
- Implemented small bounded prototype in `solvers/cpp/parallel_heuristic_compare.cpp`:
  - `greedy_dp_local_search_relocate_multistart`
  - 8 randomized starts, RCL size 3, existing relocate-only cleanup.
- Ran sanity experiment on `61/345`.

Observed TEC:

- `greedy_dp`: `7102`
- one-shot relocate-only: `7085`
- multistart prototype: `6960`
- paper EHS same epsilon: `6723`

### Evidence

- `research/dp_heuristic_oracle_20260419/phaseE_next_branch_design_memo.md`
- `solvers/cpp/parallel_heuristic_compare.cpp`
- `temp/paper_exact_repo/results/EHS/1/res_61.csv`

### Conclusion

- next main branch should be multi-start randomized assignment plus relocate-only DP local search.
- full history-sweep hardening remains valuable but should be secondary until feasibility/rescue robustness improves.

## 2026-04-19

### Attempt
Start Phase F as a new method-family probe: configuration decomposition with exact DP pricing, bounded to regime 1 and starting at `46/77`.

### Result

- Created design note:
  - `research/dp_heuristic_oracle_20260419/phaseF_configuration_probe_design.md`
- Implemented C++ full-enumeration + pricing tool:
  - `solvers/cpp/phaseF_config_probe.cpp`
- Implemented Python integer master solver:
  - `solvers/phaseF_config_master_probe.py`
- Added CMake target for probe executable.

Primary run (`46/77`):

- job types: `4`
- rate classes: `3`
- configurations per class: `4536`
- priced pairs: `13608`
- pricing runtime: `18.124586 s`
- master runtime: `8.657083 s`
- total runtime: `26.782123 s`
- TEC: `103` (`OPTIMAL`)

Comparisons at `46/77`:

- exact reference: `103`
- paper EHS: `103`
- one-shot `greedy_dp`: `118`
- one-shot relocate-only: `109`

Optional follow-up on same instance after success:

- `46/73` solved `OPTIMAL`, TEC `103`
- `46/120` solved `OPTIMAL`, TEC `103`

### Evidence

- `research/dp_heuristic_oracle_20260419/phaseF_configuration_probe_results.md`
- `research/dp_heuristic_oracle_20260419/phaseF_configuration_probe_readiness.md`
- `temp/phaseF_config_probe/meta_46_77.json`
- `temp/phaseF_config_probe/configs_46_77.csv`
- `temp/phaseF_config_probe/master_46_77.json`

### Conclusion

- regime-1 configuration master formulation is viable and exact on the required first probe.
- keep true column generation as a separate next step; `solve_pricing_dp` is not yet assumed branch-and-price-ready without dedicated reduced-cost mapping validation.

## 2026-04-19

### Attempt
Run one bounded regime-2 restricted-column probe on `61/347`, with explicit reduced-cost mapping note before implementation.

### Result

- Added design note:
  - `research/dp_heuristic_oracle_20260419/phaseG_regime2_cg_design.md`
- Implemented:
  - `solvers/phaseG_regime2_cg_probe.py`
  - `solvers/cpp/phaseG_config_pricer.cpp`
- Main probe (`61/347`) completed.

Measured summary:

- job types: `12`
- rate classes: `6`
- initial columns: `259`
- pricing iterations: `2`
- final columns: `260`
- LP bound: `7024.761959`
- best restricted-master IP TEC: `7040`
- gap to paper EHS (`6710`): `+330`
- gap to reference/F2-init (`6643`): `+397`
- runtime total: `10.486956 s`

### Evidence

- `research/dp_heuristic_oracle_20260419/phaseG_regime2_cg_results.md`
- `research/dp_heuristic_oracle_20260419/phaseG_regime2_cg_readiness.md`
- `temp/phaseG_regime2_cg/summary_61_347.json`
- `temp/phaseG_regime2_cg/iteration_log_61_347.json`

### Conclusion

- probe is feasible and improves over one-shot heuristic baselines at same epsilon.
- pricing loop signal is weak (one add then duplicate-column stall), with large remaining gap to EHS/reference.
- regime-2 path is only conditionally viable if stronger column-diversification/stabilization is implemented; otherwise stop this branch in its current form.

## 2026-04-20

### Attempt
Run a bounded Phase G correction pass on `61/347` to fix duplicate-stop logic, rerun once, and decide whether regime-2 restricted-CG remains viable.

### Result

- Implemented duplicate-stop correction in `solvers/phaseG_regime2_cg_probe.py`:
  - if best negative candidate is duplicate, run bounded threshold pricing on duplicate-hit classes,
  - choose best non-duplicate negative column before stopping.
- Added follow-up design note:
  - `research/dp_heuristic_oracle_20260419/phaseG_regime2_cg_followup_design.md`
- Corrected rerun on `61/347`:
  - initial columns `259`, iterations `12`, final columns `271`
  - LP bound `7024.761959` (unchanged)
  - restricted-master IP TEC `7040` (unchanged)
  - stop reason `max_iter`
- Also corrected Phase D reporting language for `46/73` to avoid reinsertion overclaim.

### Evidence

- `solvers/phaseG_regime2_cg_probe.py`
- `research/dp_heuristic_oracle_20260419/phaseG_regime2_cg_followup_design.md`
- `research/dp_heuristic_oracle_20260419/phaseG_regime2_cg_results.md`
- `research/dp_heuristic_oracle_20260419/phaseG_regime2_cg_readiness.md`
- `temp/phaseG_regime2_cg/summary_61_347.json`
- `temp/phaseG_regime2_cg/iteration_log_61_347.json`
- `research/dp_heuristic_oracle_20260419/phaseD_history_repair_results.md`

### Conclusion

- duplicate-stop bug was real and is now fixed.
- corrected loop adds genuinely new columns but does not improve TEC or LP bound on `61/347`.
- current bounded regime-2 restricted-CG branch should be stopped as non-competitive.

## 2026-04-20

### Attempt
Start Phase H as a new method-family branch: bounded VND-inspired fixed-epsilon search with exact DP as machine oracle, focused on `61/347` only.

### Result

- Created new Phase H iteration and switched active branch.
- Added Phase H design note:
  - `research/dp_heuristic_oracle_20260419/phaseH_vnd_exact_oracle_design.md`
- Implemented `vnd_exact_dp` in:
  - `solvers/cpp/parallel_heuristic_compare.cpp`
- Implemented required neighborhoods in VND order:
  - `swap_intra`, `swap_inter`, `insert_inter`
- Added bounded candidate screening + exact-DP-on-touched-machines acceptance, caching counters, deterministic bounded multistart initialization.
- Ran single required probe at `61/347` and stored artifacts under `temp/phaseH_vnd_exact_oracle/`.

Measured summary (`61/347`):

- `vnd_exact_dp` TEC: `6944`
- baseline `greedy_dp`: `7088`
- baseline `greedy_dp_local_search_relocate_only`: `7053`
- paper EHS (`61/347`): `6710`
- reference/F2-init (`61/347`): `6643`
- runtime: `32.04 s` (wall), max RSS: `259,375,104` bytes
- accepted moves by neighborhood: all `0`

### Evidence

- `research/dp_heuristic_oracle_20260419/phaseH_vnd_exact_oracle_results.md`
- `research/dp_heuristic_oracle_20260419/phaseH_vnd_exact_oracle_readiness.md`
- `research/dp_heuristic_oracle_20260419/iterations/20260420_phaseH_vnd_exact_oracle/SUMMARY.md`
- `temp/phaseH_vnd_exact_oracle/run_61_347_vnd_exact_dp.csv`
- `temp/phaseH_vnd_exact_oracle/summary_61_347.json`

### Conclusion

- Phase H shows material same-epsilon quality improvement over current heuristic baselines.
- move-level VND evidence is weak in this run (no accepted improving move), so continuation should be tightly bounded to confirm mechanism signal before any expansion.

## 2026-04-20

### Attempt
Start Phase I as a bounded no-screen diagnostic at `61/347` to separate:

- screening too aggressive vs
- true 1-move local optimum.

### Result

- Created new iteration and switched active branch:
  - `iterations/20260420_phaseI_noscreen_diagnostic/`
- Added design note:
  - `research/dp_heuristic_oracle_20260419/phaseI_noscreen_diagnostic_design.md`
- Implemented `phaseI_noscreen_diagnostic` variant in:
  - `solvers/cpp/parallel_heuristic_compare.cpp`
- Diagnostic mechanics:
  - start from best bounded multistart incumbent
  - no-screen exact move evaluation for tested feasible batch
  - required `insert_inter`, optional `swap_inter`
  - bounded caps ladder `64/256/1024`

Main `61/347` outcomes:

- cap `64`: start `6944`, best `6922`, exact-evaluated `64` (`insert 48`, `swap 16`)
- cap `256`: start `6944`, best `6920`, exact-evaluated `149` (`insert 149`, `swap 0`)
- cap `1024`: start `6944`, best `6920`, exact-evaluated `359` (`insert 359`, `swap 0`)
- improving move found in all runs (`insert_inter`), stop reason `diag_found_improving_move`

### Evidence

- `research/dp_heuristic_oracle_20260419/phaseI_noscreen_diagnostic_results.md`
- `research/dp_heuristic_oracle_20260419/phaseI_noscreen_diagnostic_readiness.md`
- `temp/phaseI_noscreen_diagnostic/run_61_347_noscreen_cap64.csv`
- `temp/phaseI_noscreen_diagnostic/run_61_347_noscreen_cap256.csv`
- `temp/phaseI_noscreen_diagnostic/run_61_347_noscreen_cap1024.csv`

### Conclusion

- Diagnostic supports "screening too aggressive" and does not support "true 1-move local optimum" at this point.
- Recommended next branch: screening/ranking redesign under same bounded single-point scope.

## 2026-04-20

### Attempt
Start Phase J: bounded insert-focused analytical screening/ranking redesign at `61/347` to recover no-screen improving move behavior efficiently.

### Result

- Created iteration and switched active branch:
  - `iterations/20260420_phaseJ_insert_screening_redesign/`
- Added design note:
  - `research/dp_heuristic_oracle_20260419/phaseJ_insert_screening_redesign_design.md`
- Implemented in `solvers/cpp/parallel_heuristic_compare.cpp`:
  - `vnd_exact_dp_insert_rank_v1` (dual-pressure + gap-aware source priority)
  - `vnd_exact_dp_insert_rank_diverse` (per-source diversity + two-stage rerank + gap-aware source priority)
- Ran required comparisons at `61/347` and saved artifacts under:
  - `temp/phaseJ_insert_screening_redesign/`

Observed TEC:

- `greedy_dp`: `7088`
- `greedy_dp_local_search_relocate_only`: `7081`
- old `vnd_exact_dp`: `6944`
- Phase I no-screen best: `6920`
- `vnd_exact_dp_insert_rank_v1`: `6908`
- `vnd_exact_dp_insert_rank_diverse`: `6884`

### Evidence

- `research/dp_heuristic_oracle_20260419/phaseJ_insert_screening_redesign_results.md`
- `research/dp_heuristic_oracle_20260419/phaseJ_insert_screening_redesign_readiness.md`
- `temp/phaseJ_insert_screening_redesign/run_61_347_insert_rank_v1.csv`
- `temp/phaseJ_insert_screening_redesign/run_61_347_insert_rank_diverse.csv`

### Conclusion

- Phase J successfully recovers and exceeds the no-screen improvement pattern.
- main remaining bottleneck is screening efficiency/memory footprint (especially diverse variant), not inability to find improving insert moves.

## 2026-04-20

### Attempt
Run Phase K as one final bounded non-ML efficiency pass on the insert-screening branch at `61/347`.

### Result

- Created new iteration and switched active branch:
  - `iterations/20260420_phaseK_insert_efficiency_pass/`
- Added design note:
  - `research/dp_heuristic_oracle_20260419/phaseK_insert_efficiency_pass_design.md`
- Implemented in `solvers/cpp/parallel_heuristic_compare.cpp`:
  - `vnd_exact_dp_insert_rank_diverse_trimmed`
  - `vnd_exact_dp_insert_rank_diverse_budgeted`
- Ran required comparisons at `61/347` and saved artifacts under:
  - `temp/phaseK_insert_efficiency_pass/`

Observed TEC:

- `greedy_dp`: `7088`
- `greedy_dp_local_search_relocate_only`: `7081`
- old `vnd_exact_dp`: `6944`
- Phase J best `vnd_exact_dp_insert_rank_diverse`: `6884`
- Phase K `vnd_exact_dp_insert_rank_diverse_trimmed`: `6884`
- Phase K `vnd_exact_dp_insert_rank_diverse_budgeted`: `6884`

Efficiency signal vs Phase J diverse:

- trimmed reduced screened insert candidates `29160 -> 19416` (about `-33%`)
- budgeted reduced screened insert candidates `29160 -> 22440` (about `-23%`)
- runtime/RSS did not improve materially.

### Evidence

- `research/dp_heuristic_oracle_20260419/phaseK_insert_efficiency_pass_results.md`
- `research/dp_heuristic_oracle_20260419/phaseK_insert_efficiency_pass_readiness.md`
- `temp/phaseK_insert_efficiency_pass/run_61_347_insert_rank_diverse_trimmed.csv`
- `temp/phaseK_insert_efficiency_pass/run_61_347_insert_rank_diverse_budgeted.csv`

### Conclusion

- Phase K preserves best known branch quality (`6884`) and tightens screening structure.
- additional handcrafted non-ML headroom appears limited on this point.
- recommended next step is pivot to learning-based move screening/ranking on top of this insert-focused exact-DP base.

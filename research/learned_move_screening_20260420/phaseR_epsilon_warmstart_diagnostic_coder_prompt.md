Continue from the current state in repo root:

- `.`

This task belongs to:

- `research/learned_move_screening_20260420/`

You are starting in a fresh session. Rebuild context from the thread files below and then implement only the bounded diagnostic described here.

## Read first

Read these in order:

1. `research/learned_move_screening_20260420/START_HERE.md`
2. `research/learned_move_screening_20260420/ACTIVE.md`
3. `research/learned_move_screening_20260420/iterations/20260421_phaseR_epsilon_warmstart_diagnostic/SUMMARY.md`
4. `research/learned_move_screening_20260420/phaseR_epsilon_warmstart_diagnostic_design.md`
5. `research/learned_move_screening_20260420/iterations/20260421_phaseQ_synthetic_offline_training/RESULTS.md`

Then inspect the concrete earlier evidence:

6. `temp/phaseI_noscreen_diagnostic/run_61_347_noscreen_cap256.csv`
7. `temp/phaseJ_insert_screening_redesign/run_61_347_insert_rank_diverse.csv`
8. `temp/phaseK_insert_efficiency_pass/run_61_347_insert_rank_diverse.csv`
9. `temp/phaseK_insert_efficiency_pass/run_61_347_insert_rank_diverse_budgeted.csv`
10. `temp/phaseK_insert_efficiency_pass/run_61_347_vnd_exact_dp.csv`

Also inspect the current implementation entry points you will likely need:

11. `solvers/cpp/parallel_heuristic_compare.cpp`
12. any small driver/wrapper scripts you actually use for this diagnostic

## Minimal restored context

You must anchor on these facts before coding:

1. Phase I proved improving insert-inter moves exist on `paper_61` at `epsilon=347`.
   - Evidence file above shows `6944 -> 6920`.

2. Phase J/K improved the same instance further.
   - strongest tracked variant for `paper_61, epsilon=347` is `vnd_exact_dp_insert_rank_diverse`
   - TEC reached `6884`

3. Phase Q learning is not the main next direction.
   - learned ranking gains over `screen_score_s2` are real but marginal
   - do not spend time improving the learning branch here

4. The current hypothesis to test is algorithmic:
   - whether warm-starting across nearby epsilon values creates cumulative gains that fresh-start VND misses

## Objective

Implement a bounded diagnostic on benchmark instance `61`:

- compare warm-started VND versus fresh-start VND
- across a short descending epsilon ladder
- using the same VND family and same exact-DP local oracle

This is a diagnostic branch, not a full solver redesign.

## New iteration

Work inside the already-created active iteration:

- `research/learned_move_screening_20260420/iterations/20260421_phaseR_epsilon_warmstart_diagnostic/`

Update these during the work:

- `RESULTS.md`
- `BLOCKERS.md`
- `SUMMARY.md`
- `research/learned_move_screening_20260420/LOG.md`

Do not create another new phase unless you hit a genuine branch split.

## Required experiment design

Use:

- benchmark instance `61`

Default epsilon ladder:

- `380, 370, 360, 350, 347, 340`

If one nearby value is infeasible or structurally awkward in the current harness, document it and choose the nearest reasonable replacement. Do not broaden the ladder.

At each epsilon, run:

1. fresh-start VND
2. warm-started VND seeded from the final solution at the previous epsilon

The warm-start chain must proceed in descending epsilon order.

## Required base solver

Use the strongest available Phase J/K insert-based VND path already in repo.

Preferred default:

- `vnd_exact_dp_insert_rank_diverse`

If you discover that a closely related Phase K budgeted/trimmed variant is operationally better for this diagnostic, you may use it, but:

- justify the switch explicitly,
- keep the comparison fair,
- and do not introduce a new heuristic family.

## Required outputs

Write everything for this run under one self-contained run folder:

- `temp/phaseR_epsilon_warmstart_diagnostic/<run_id>/`

Example:

- `temp/phaseR_epsilon_warmstart_diagnostic/20260421_run01_inst61_chain_vs_fresh/`

That folder must contain at minimum:

- `README.md`
- `command.sh`
- `config.json`
- `stdout.log`
- `metrics/epsilon_comparison.csv`
- `metrics/epsilon_summary.csv`
- `notes/results.md`
- `notes/readiness.md`
- `notes/blockers.md`

## Required metrics table

In `metrics/epsilon_comparison.csv`, include one row per epsilon with at least:

- `instance_id`
- `epsilon`
- `variant_base`
- `fresh_start_tec`
- `fresh_start_runtime_sec`
- `fresh_start_accepted_moves`
- `warm_start_initial_tec`
- `warm_start_final_tec`
- `warm_start_runtime_sec`
- `warm_start_accepted_moves`
- `carryover_gain_vs_fresh`
- `local_improvement_from_seed`
- `fresh_stop_reason`
- `warm_stop_reason`

Definitions:

- `carryover_gain_vs_fresh = fresh_start_tec - warm_start_final_tec`
- `local_improvement_from_seed = warm_start_initial_tec - warm_start_final_tec`

Also include any other counters that help explain behavior, such as exact-DP calls or accepted insert-inter moves, if cheap to expose.

## Required interpretation

The point of this run is not only “which TEC is lower.”

You must distinguish:

1. improvement due to inherited incumbent quality
2. improvement due to additional local search from the inherited seed

That is why both `warm_start_initial_tec` and `warm_start_final_tec` are mandatory.

## Constraints

Do not:

- revisit learning methods
- change the synthetic protocol
- run benchmark-wide evaluation
- introduce ALNS / destroy-repair / new neighborhoods
- claim paper-level success from this one diagnostic

Keep this branch sharply focused.

## Success criterion

This branch is a pass if the paired table shows that warm-started VND is consistently equal or better than fresh-start VND across the ladder, with a clear cumulative advantage near the hard region.

This branch is a fail or inconclusive result if:

- warm-start is indistinguishable from fresh-start,
- the signal is too noisy to interpret,
- or the implementation cannot actually preserve the incumbent chain correctly.

If it fails, say so directly.

## Final questions to answer directly

1. What exact base VND variant did you use?
2. What epsilon ladder did you run?
3. Did warm-start beat fresh-start, tie it, or fail to help?
4. At which epsilons was the gain largest?
5. How much of the gain came from inherited incumbent quality versus new local improvement from the seed?
6. Does this justify a full epsilon-sweep warm-start branch next?
7. What exact next step should follow?

# Phase R Design

## Objective

Run a bounded diagnostic that tests whether epsilon-to-epsilon warm-start continuity is the missing mechanism behind the current VND gap.

This is not a full heuristic redesign. It is a falsifiable diagnostic.

## Why this branch exists

The project now has three grounded facts:

1. Improving insert-based local-search moves exist.
   - Evidence: `temp/phaseI_noscreen_diagnostic/run_61_347_noscreen_cap256.csv`
   - On `paper_61` at `epsilon=347`, TEC improves from `6944` to `6920`.

2. Better screening helps, but only up to a point.
   - Evidence: `temp/phaseJ_insert_screening_redesign/run_61_347_insert_rank_diverse.csv`
   - Evidence: `temp/phaseK_insert_efficiency_pass/run_61_347_insert_rank_diverse.csv`
   - Best Phase J/K result on the same instance is `6884`.

3. Learning is not the main missing lever in the current formulation.
   - Evidence: `research/learned_move_screening_20260420/iterations/20260421_phaseQ_synthetic_offline_training/RESULTS.md`
   - Learned models beat handcrafted `screen_score_s2` only marginally.

That combination supports one focused hypothesis:

- the real missing mechanism may be cumulative search across nearby epsilon values rather than a better per-move scorer.

## Core hypothesis

If VND at epsilon `e_i` starts from the solution found at nearby epsilon `e_(i-1)`, it may inherit a much better incumbent basin than fresh-start VND. If that is true, warm-started VND should match or beat fresh-start VND at most or all epsilon values in a short descending ladder.

## Experiment scope

Single benchmark instance only:

- `paper_61`

Base solver family:

- the strongest available Phase J/K insert-based exact-DP VND variant, currently `vnd_exact_dp_insert_rank_diverse`

Epsilon ladder:

- a short descending ladder around the hard region, default:
  - `380, 370, 360, 350, 347, 340`

Paired conditions at each epsilon:

1. fresh-start VND
2. warm-start initial incumbent from previous epsilon
3. warm-start final VND result

## Required measurements

At each epsilon, record:

- epsilon
- fresh-start TEC
- fresh-start runtime
- fresh-start accepted moves
- warm-start initial TEC
- warm-start final TEC
- warm-start runtime
- warm-start accepted moves
- carryover gain = `fresh_final_tec - warm_final_tec`
- local improvement from seed = `warm_initial_tec - warm_final_tec`
- note / stop reason

## Decision rule

Interpretation:

1. Warm-start consistently better:
   - Direction B is viable.
   - Move to a larger epsilon-sweep branch.

2. Warm-start equal almost everywhere:
   - continuity is probably not the main missing lever from this starting point.
   - next bottleneck is likely assignment quality or neighborhood scope.

3. Warm-start worse:
   - treat as implementation bug or accounting error until disproven.

## Deliverables

One self-contained run folder under:

- `temp/phaseR_epsilon_warmstart_diagnostic/<run_id>/`

Minimum contents:

- `README.md`
- `command.sh`
- `config.json`
- `stdout.log`
- `metrics/epsilon_comparison.csv`
- `metrics/epsilon_summary.csv`
- `notes/results.md`
- `notes/readiness.md`
- `notes/blockers.md`

## What not to do

- no learning changes
- no benchmark-wide sweep yet
- no new neighborhood family
- no A-SGH / ALNS / assignment-rule mining in this branch
- no paper claims before the paired table exists

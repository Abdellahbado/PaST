# Phase L1 Dataset Logging Design

Date: 2026-04-20

## Scope and anchor

- Scope: Stage L1 only (data logging), no model training, no ML inference.
- Anchor point only: `instance 61`, `epsilon 347`.
- Reference predecessor evidence at this point:
  - Phase H: `6944`
  - Phase I no-screen: `6920`
  - Phase J/K handcrafted best: `6884`
  - paper EHS: `6710`
  - reference: `6643`

## Data source solver variant

- Base family for labels: `vnd_exact_dp_insert_rank_diverse_trimmed`.
- Logging wrapper variant: `stageL1_dataset_logging`.
  - It runs deterministic multistart and calls the same insert-focused exact-DP acceptance core.
  - It is logging-only plumbing, not a new heuristic family.

## Record definitions

### Candidate move record (Dataset A: broad)

- One record per generated `insert_inter` candidate during search.
- Includes cheap context/features regardless of eventual exact evaluation.
- Includes both epsilon-feasible and infeasible generated candidates (`epsilon_feasible` flag).

### Exact-labeled move record (Dataset B: exact)

- One record for each candidate that is exact DP evaluated on touched machines.
- Includes full feature snapshot plus exact labels from oracle evaluation.
- Includes both binary and regression labels.

## Two-stream logging contract

1. Broad stream (`moves_broad_61_347.csv`)
   - Purpose: distribution/context/calibration.
   - Contains all generated `insert_inter` candidates with cheap features.

2. Exact-labeled stream (`moves_exact_labeled_61_347.csv`)
   - Purpose: supervised training/evaluation labels.
   - Contains only exact-evaluated candidates with exact deltas and improve/accept labels.

Shared key:

- `record_id` is used in both streams for compatible linkage.

## Features to log

### Job features

- `job_id`
- `job_processing_time`
- `job_type_id` (mapped to processing-time type for this stage)

### Source machine features

- `source_machine_id`
- `source_rate`
- `source_rate_class`
- `source_exact_cost`
- `source_relaxed_lb`
- `source_exact_minus_lb_gap`
- `source_load`
- `source_utilization`
- `source_job_count`

### Target machine features

- `target_machine_id`
- `target_rate`
- `target_rate_class`
- `target_exact_cost`
- `target_relaxed_lb`
- `target_exact_minus_lb_gap`
- `target_load`
- `target_utilization`
- `target_residual_slack_before`
- `target_job_count`

### Move-combination features

- `projected_target_load_after`
- `projected_source_load_after`
- `source_to_target_rate_diff`
- `cheap_lb_delta_proxy`
- `source_cost_density`
- `target_cost_density`
- `source_top_expensive_flag`
- `target_top_expensive_flag`
- `screen_score_s1`
- `screen_score_s2`

### Search-state features

- `seed_id`
- `ls_round`
- `accepted_improving_moves_so_far`
- `current_tec`
- `current_tec_minus_start`
- `exact_eval_cap`
- `exact_eval_budget_remaining`
- `exact_eval_tier`

### Feasibility context

- `epsilon_feasible`

## Labels (exact stream)

- `exact_evaluation_performed`
- `exact_old_touched_cost_sum`
- `exact_new_touched_cost_sum`
- `exact_total_delta`
- `label_improving`
- `label_accepted`

Label use:

- Binary target: `label_improving`.
- Regression target: `exact_total_delta`.

## Multistart seed policy

- Deterministic multistart at same anchor point (`61/347`).
- Stage L1 choice: `12` seeds (inside requested `10-20` band).
- Seed generation follows deterministic formula in solver wrapper variant.
- Each seed produces broad stream; exact-labeled rows depend on exact-eval path and stopping.

## Stage L1 minimum data sufficiency target

For Stage L1 to be considered usable for Stage L2 offline ranking:

- seeds: at least `10`
- broad records: at least `50k`
- exact-labeled records: at least `80`
- improving positives in exact stream: at least `20`
- positive rate in exact stream: at least `10%`

These are minimum viability targets, not final-data targets.

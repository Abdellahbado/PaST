# Phase Y1 — Trace Probe Results

## Summary

| Cell | Inst/Eps | TEC | Stop Reason | Runtime | Machines | Tokens | OK? |
|------|----------|-----|-------------|---------|----------|--------|-----|
| Cell_A | 61/347 | 6946.0 | max_rounds | 8.2s | 25 | 3299 | YES |
| Cell_B | 62/290 | 9435.0 | max_rounds | 35.6s | 25 | 3297 | YES |
| Cell_C | 65/195 | 27031.0 | max_rounds | 5.0s | 25 | 3365 | YES |

## Smoke Checks

All 3 cells: trace JSON generated, parses as valid JSON, all machines
present, anonymized cell label, token estimates 3297-3365 (well within
DeepSeek context limits).

### What's Included

- Cell regime: epsilon, num_machines, total_jobs, epsilon_regime, job_size_range
- Current snapshot: current_tec, best_tec_episode, no_hit_streak, round,
  total_accepted_moves, exact_dp_evals, core_lane_stagnation_active, 
  exception_lane_active, stop_reason_guard
- Machine state table: all 25 machines with 17 fields (id, jobs, load, slack,
  load_pressure, exact_cost, relaxed_lb, gap, cost_density, small/medium/large
  jobs, core_source_hits, core_target_hits, rate, starved)
- Candidate pool summaries: top 5 sources by cost/gap, top 5 targets by slack,
  job size distribution by cost quartile
- Prior arm results: best known TEC from Phases S/V/X for each dev cell

### Fields Not Yet Tracked (null/informational)

- `core_source_hits` and `core_target_hits`: per-machine hit counts not tracked
  in Y1. Requires accumulating counters in the DiverseTrimmed round loop.
- `last_accepted_moves`: move history not tracked. Requires a ring buffer of
  the last 10 accepted (source, target, job_size, delta_tec) tuples.
- `failed_move_families`: not tracked. Requires classification of exhausted
  candidate patterns.
- `underexplored_sources/targets`: null (depends on core_hits tracking).
- `starved`: null (depends on outside_pool tracking at per-machine granularity).

### Implementation Details

- C++ variant: `phaseY_trace_probe` added as `InsertScreenMode::PhaseYTraceProbe`
- Uses DiverseTrimmed core lane with 1 deterministic start (trace probe mode)
- Trace written at end of DiverseTrimmed local search (max_rounds or stagnation)
- `g_audit_instance_id` set before `evaluate_variant` call for cell labeling
- JSON written to `traces/generated/trace_{cell_label}_r{round}.json`
- Markdown companion written to `traces/generated/trace_{cell_label}_r{round}.md`

### Token Budget Estimate

| Cell | JSON tokens (est.) | Markdown tokens (est.) | Combined |
|------|-------------------:|----------------------:|---------:|
| Cell_A | 3299 | ~1200 | ~4500 |
| Cell_B | 3297 | ~1200 | ~4500 |
| Cell_C | 3365 | ~1200 | ~4600 |

DeepSeek V4 Pro has a large context window. Each trace fits well under 10K tokens.

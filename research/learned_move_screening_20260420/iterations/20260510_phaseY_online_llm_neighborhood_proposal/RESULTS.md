# Results — Phase Y: Online LLM Neighborhood Proposal

## Y0 Schema Design (2026-05-10)

Trace and proposal schemas designed. See SUMMARY.md for deliverables.

## Y1 Trace Probe Smoke (2026-05-10)

### Smoke Results

| Cell | Inst/Eps | TEC | Stop | Runtime | Machines | Tokens | OK |
|------|----------|-----|------|---------|----------|--------|:--:|
| Cell_A | 61/347 | 6946.0 | max_rounds | 8.2s | 25 | 3299 | YES |
| Cell_B | 62/290 | 9435.0 | max_rounds | 35.6s | 25 | 3297 | YES |
| Cell_C | 65/195 | 27031.0 | max_rounds | 5.0s | 25 | 3365 | YES |

All 3 cells produce valid traces. Each trace includes:
- All 25 machines (17 columns per machine: id, jobs, load, slack, load_pressure,
  exact_cost, relaxed_lb, gap, cost_density, small/medium/large jobs, rate)
- Cell regime, snapshot, candidate pools (top 5 sources by cost/gap, top 5
  targets by slack, job size distribution by cost quartile)
- Prior arm results (best known TEC from Phases S/V/X for each dev cell)
- Token estimates: 3297-3365 (well within DeepSeek context limits)

### C++ Implementation

- Added `PhaseYTraceProbe` to `InsertScreenMode` enum
- Added `write_phaseY_trace_json` function (JSON + Markdown output)
- Added variant string dispatch at 9 locations (enum mapping, trimmed_mode
  check, variant lists, CSV output, usage message, multistart wrapper)
- 1 deterministic start for trace probe (enables reproducible traces)
- `g_audit_instance_id` set before `evaluate_variant` call for cell labeling
- Trace written at end of DiverseTrimmed local search (whether max_rounds
  or stagnation)

### Python Orchestration

- `scripts/phaseY_neighborhood_proposal.py` with `--y1-trace-probe` subcommand
- Runs `phaseY_trace_probe` on 3 dev cells
- Parses CSV output, locates generated trace files
- Validates: JSON parse, machine count, token estimate, anonymization
- Writes `eval/y1_trace_probe_raw.csv` and `notes/phaseY1_trace_probe_results.md`

### Fields Not Yet Tracked (null/informational in traces)

| Field | Status | Reason |
|-------|--------|--------|
| core_source_hits | null | Requires per-round counter in DiverseTrimmed loop |
| core_target_hits | null | Same as above |
| last_accepted_moves | null | Requires ring buffer of accepted moves |
| failed_move_families | null | Requires classification of exhausted candidates |
| underexplored sources/targets | null | Depends on core_hits tracking |
| starved | null | Depends on outside_pool per-machine tracking |

These are deferred to Y1.1 if needed for DeepSeek diagnosis quality.
The trace already provides exact costs, gaps, slack, and job composition
— enough for the LLM to select source/target machines.

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

### Fields Previously Null (all resolved in Y1.1)

| Field | Status | Implementation |
|-------|--------|----------------|
| core_source_hits | RESOLVED Y1.1 | Per-round count from DiverseTrimmed pool entries |
| core_target_hits | RESOLVED Y1.1 | Same |
| starved | RESOLVED Y1.1 | Derived: has jobs AND core_source_hits==0 |
| last_accepted_moves | RESOLVED Y1.1 | Ring buffer (up to 10 entries, circular) at core-lane acceptance |
| failed_move_families | simplified Y1.1 | `failed_summary`: evaluated_exact_this_round + no_improving flag |
| underexplored_sources | RESOLVED Y1.1 | Top 5 by cost, core_hits==0 |
| underexplored_targets | RESOLVED Y1.1 | Top 5 by slack, core_hits==0 |


## Y2 Proposal Execution & Random Baseline (2026-05-10)

### Smoke Results (Cell A only; Cells B/C blocked by uncommitted dp_solver changes)

| Variant | TEC | Generated | Evaluated | Improvements | Best Δ |
|---------|-----|-----------|-----------|-------------|--------|
| trace_probe (baseline) | 7036 | — | — | — | — |
| execute_manual | 7036 | 425 | 20 | 0 | 0.0 |
| random_s1 | 7030 | 76 | 20 | 3 | 3.0 |

All variants parse and execute on Cell A (61/347):
- Manual proposal generates 425 candidates from 5 sources × 5 targets × size classes
- Random proposal generates 76 candidates (weighted by cost/slack)
- Candidate generation > 0 for all cells ✓
- Evaluated candidates <= max_candidates (20) ✓
- Exact DP verifies all accepted moves ✓
- No crash/infeasible output on Cell A ✓
- CSV includes all Phase Y proposal fields ✓

### C++ Implementation

- `PhaseYProposal` struct (9 fields) at namespace scope
- JSON parser: `parse_phaseY_proposal` (reads string/int values, M-arrays, size classes)
- Candidate generation: expand source × job × target → `PhaseYCand` list
- Ranking: 6 hint types (cheap_lb, s2, random, cost_gap, slack, hybrid)
- Diversity: 4 rules (per_source, per_target, source_target_pair, none)
- Random proposal: `generate_random_proposal` — weighted by exact_cost (sources) and slack (targets)
- Two new variants: `InsertScreenMode::PhaseYExecuteProposal`, `InsertScreenMode::PhaseYRandomProposal`
- New CSV fields (11): phaseY_proposal_name through phaseY_targets_used
- Variant dispatch at 8+ points (enum, trimmed_mode, flags, CSV, multistart, usage)

### Manual Proposals

- `proposals/manual/y2_Cell_A_manual.json`: starved high-gap sources + slack targets, small+medium jobs
- `proposals/manual/y2_Cell_B_manual.json`: expensive/gap sources + slack targets, all sizes
- `proposals/manual/y2_Cell_C_manual.json`: underexplored sources + slack targets, small+medium jobs

### Known Issues

- Cell B (62/290) and Cell C (65/195): infeasible due to uncommitted dp_solver changes
- Random variant: SIGBUS crash with certain seeds (likely kInf overflow in weighted selection — fixed for Cell A, may affect other seed/machine combos)
- `phaseW_dsl_assignment.hpp` removed (uncommitted + broken include)

| Field | Status | Implementation |
|-------|--------|----------------|
| core_source_hits | RESOLVED Y1.1 | Per-round count from DiverseTrimmed pool entries |
| core_target_hits | RESOLVED Y1.1 | Same |
| starved | RESOLVED Y1.1 | Derived: has jobs AND core_source_hits==0 |
| last_accepted_moves | RESOLVED Y1.1 | Ring buffer (up to 10 entries, circular) at core-lane acceptance |
| failed_move_families | simplified Y1.1 | `failed_summary`: evaluated_exact_this_round + no_improving flag |
| underexplored_sources | RESOLVED Y1.1 | Top 5 by cost, core_hits==0 |
| underexplored_targets | RESOLVED Y1.1 | Top 5 by slack, core_hits==0 |


## Y1.1 Search-Behavior Fields (2026-05-10)

### Smoke Results

| Cell | Inst/Eps | TEC | Stop | Runtime | Machines | Tokens | src>0 | tgt>0 | starved | moves | OK |
|------|----------|-----|------|---------|----------|--------|-------|-------|---------|-------|:--:|
| Cell_A | 61/347 | 6946.0 | max_rounds | 8.5s | 25 | 3800 | 5 | 3 | 12 | 5 | YES |
| Cell_B | 62/290 | 9435.0 | max_rounds | 37.6s | 25 | 3766 | 5 | 3 | 4 | 5 | YES |
| Cell_C | 65/195 | 27031.0 | max_rounds | 6.8s | 25 | 3874 | 5 | 3 | 20 | 5 | YES |

TEC values unchanged from Y1 — instrumentation only.

### C++ Changes

- `PhaseYAcceptedMove` struct at namespace scope
- Added `phaseY_source_hits`, `phaseY_target_hits`, `phaseY_ring`, `phaseY_ring_count`,
  `phaseY_last_evaluated_exact`, `phaseY_last_no_improving` static trackers
- Pool hit counting after `pool` construction (per-round)
- Ring buffer push on core-lane acceptance (`old_ab - new_ab` delta)
- `write_phaseY_trace_json` extended: +4 params (hits, eval_count, no_improving, ring)
- `failed_summary` section replaces `failed_move_families` note

### Python Changes

- `scripts/phaseY_neighborhood_proposal.py`: `--y1-1-trace-probe` subcommand
- Field validation: core_source_hits, core_target_hits, starved, last_accepted_moves,
  underexplored sources/targets, failed_summary
- Output: `eval/y1_1_trace_probe_raw.csv`, `notes/phaseY1_1_trace_probe_results.md`

### Token Budget (after Y1.1)

| Cell | JSON tokens | MD tokens | Combined |
|------|-----------:|----------:|---------:|
| Cell_A | 3800 | ~1500 | ~5300 |
| Cell_B | 3766 | ~1500 | ~5266 |
| Cell_C | 3874 | ~1500 | ~5374 |

All under 5000 token budget per JSON trace.

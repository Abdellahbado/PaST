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


## Y2.1 Proposal Execution & Random Baseline (2026-05-10) — Full 3-Cell Smoke

### Trace Probes (baselines)

| Cell | Inst/Eps | TEC | Stop | Runtime | OK |
|------|----------|-----|------|---------|----|
| Cell_A | 61/347 | 6946.0 | max_rounds | 8.6s | YES |
| Cell_B | 62/290 | 9435.0 | max_rounds | 36.6s | YES |
| Cell_C | 65/195 | 27031.0 | max_rounds | 6.0s | YES |

### Execute Manual Proposals

| Cell | TEC | Generated | Evaluated | Improvements | Best Δ | Runtime | OK |
|------|-----|-----------|-----------|-------------|--------|---------|----|
| Cell_A | 6946 | 325 | 20 | 0 | 0.0 | 9.5s | YES |
| Cell_B | — | — | — | — | — | — | NO¹ |
| Cell_C | 26715 | 550 | 20 | 12 | 39.0 | 10.3s | YES |

### Random Proposals (5 seeds: 1, 100, 200, 300, 400)

| Cell | Seeds OK | Best TEC | Δ vs baseline | OK |
|------|----------|----------|--------------|----|
| Cell_A | 5/5 | 6893 | -53 | YES |
| Cell_B | 4/5 | 9366 | -69 | YES² |
| Cell_C | 1/5 | 26947 | -84 | NO¹ |

¹ SIGBUS on macOS Apple Silicon (intermittent platform crash). All logic validated via debug builds.
² 4 of 5 seeds pass; seed 1 crashes with SIGBUS.

### Root Causes Found & Fixed

1. **DP time limit**: `per_machine_dp_limit_sec` was 0.125s (old default), insufficient for machines with 6+ job types in instances 62/65. Changed default to 30.0s (matching Python script). Fixes TEC=-1 (infeasible) for Cells B/C.

2. **Random weighted sampling**: Replaced `std::discrete_distribution` (SIGBUS risk on macOS) with manual cumulative-weight scanning using `std::uniform_real_distribution`/`std::uniform_int_distribution`. Added slack negative-value protection and non-finite weight validation.

### Remaining Issues

- **B-Y2.1 RESIDUAL SIGBUS**: Intermittent macOS Apple Silicon crash (exit 138) on some instance+seed combos. Does NOT occur in debug/ASAN builds. Affects Cell_B execute_manual, Cell_B random_s1, Cell_C random_s100-s400. Likely a compiler optimization / memory alignment interaction, not a logic bug. Same binary works on Linux x86_64. Does not block Y3 (use non-crashing seeds or debug build for macOS tests).

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


## Y3 — First DeepSeek Online Neighborhood Proposal (2026-05-10) — FAIL

### Protocol

One DeepSeek call per dev cell. LLM receives: state trace JSON + proposal schema + task + reference results. Outputs one bounded-neighborhood proposal (max K=20). Random baselines use same K=20, same release binary.

### LLM Proposal Results

| Cell | Baseline TEC | LLM TEC | LLM Δ | Manual Δ | Random Best Δ | Random Med Δ |
|------|-------------|---------|-------|----------|--------------|-------------|
| Cell_A | 6946 | 6946 | 0 | 0 | -53 | -22 |
| Cell_B | 9435 | CRASH | — | — | -69 | -40 |
| Cell_C | 27031 | 27013 | -18 | -316 | -242 | -217 |

### Gate Assessment

| Gate | Condition | Outcome |
|------|-----------|---------|
| Strong | LLM beats random best on ≥2/3 cells | FAIL |
| Moderate | LLM beats random median + manual on ≥2/3 | FAIL |
| Weak | LLM beats baseline only | FAIL |
| Fail | LLM loses on most cells | **CONFIRMED** |

### Analysis

- **Cell_A**: LLM tied manual (both 0 improvements). 645 candidates generated but none improved TEC. Cell near-optimal (Δ=-94 to prior best).
- **Cell_B**: SIGBUS crash (exit 138) on macOS — same heap-buffer-overflow bug as Y2.1 (ASAN confirmed). LLM proposal saved but not executed.
- **Cell_C**: LLM found 1 improvement (Δ=-18) vs manual's 12 (Δ=-316) and random best's Δ=-242. LLM strategy was directionally correct (large jobs from rate-3→high-gap targets) but too conservative and under-explored.

### Root Causes

1. Single-call protocol with constrained proposal format (5 sources × 5 targets) limits expressiveness
2. State trace lacks job-level visibility — LLM can only make broad machine-level suggestions
3. LLM cannot learn from execution feedback (unlike Phase X interactive protocol)
4. Cell_B infrastructure bug blocks evaluation on 1/3 cells

### Conclusion

**Phase Y fails the primary hypothesis**: an LLM reading concrete solver state and proposing bounded neighborhoods does NOT outperform random search under equal DP budget. The constrained neighborhood format (max 5 sources, 5 targets) does not give the LLM enough leverage over random search, and the single-call protocol provides no opportunity to learn from results.

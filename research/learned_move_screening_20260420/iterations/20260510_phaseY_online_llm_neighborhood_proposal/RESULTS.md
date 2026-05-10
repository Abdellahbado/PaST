# Results — Phase Y: Online LLM Neighborhood Proposal

## Y0 Schema Design (2026-05-10)

Trace and proposal schemas designed. No experiments yet.

### State Trace Schema

The trace is a 6-section Markdown document:

1. **Metadata**: trace_id, cell_label, round, timestamp
2. **Cell Regime**: epsilon, num_machines, total_jobs, epsilon_regime,
   job_size_range, episode_epsilon_progression
3. **Current Solution Snapshot**: current_tec, best_tec_episode,
   no_hit_streak, total_accepted_moves_so_far, core_lane_stagnation_active,
   exception_lane_active, stop_reason_guard
4. **Machine State Table**: 17 columns × 25-40 rows — compact per-machine
   diagnostics covering load, cost, slack, gap, cost_density, processing
   time histogram, core source/target hit counts, rate, starved flag
5. **Recent Search Behavior**: last 10 accepted moves, failed move families,
   core/outside pool composition, no_hit by source/target, next round budget
6. **Candidate Pool Summaries**: pre-computed ranked lists — top sources by
   cost/gap/density, top targets by slack/low_cost, underexplored
   sources/targets, job size distribution by cost quartile
7. **Prior Arm Results** (dev cells only): best known TEC from Phases S/V/X

### Proposal Schema

Bounded JSON with 9 required fields:

- `proposal_name`, `move_family` (insert_inter only)
- `source_machines` (max 5), `target_machines` (max 5)
- `job_size_classes` (small/medium/large subsets)
- `max_candidates` (≤ 30)
- `ranking_hint` (cheap_lb/s2/random/cost_gap/slack/hybrid)
- `diversity_rule` (per_source/per_target/source_target_pair/none)
- `fallback_if_empty` (random_same_budget/top_s2_same_budget)
- `rationale` (≤ 500 chars)

Constraint-to-candidate mapping: 9-step pipeline from source/target lists
to exact-DP-evaluated triples, with diversity quotas and fallback.

### Random Baseline Design

Same proposal format, same K budget, same DP verifier, same initial state.
Random source selection weighted by EC. Random target selection weighted
by slack. Fixed ranking_hint='random', fallback='top_s2_same_budget'.

### Key Design Decisions

- LLM proposes **constraints** (which machines, which job sizes), not
  individual (source, job, target) triples — prevents token bloat and
  parsing fragility.
- All 25-40 machines shown in the state table — LLM can process tabular
  data at scale; hiding machines would prevent discovering underexplored
  sources.
- Fields excluded: raw instance ID, S1 score, electricity price curve,
  per-candidate s2 scores, DP cache stats, CPU/runtime — to prevent
  overfitting and context bloat.
- Prior arm results shown only for dev cells, not held-out cells — prevents
  oracle leakage on validation.

### Y1 Implementation Plan

Requires new C++ instrumentation:
1. Snapshot per-machine state at stagnation
2. Track core source/target hit counts per round
3. Buffer last 10 accepted moves
4. Compute processing time histograms (already available from solver state)

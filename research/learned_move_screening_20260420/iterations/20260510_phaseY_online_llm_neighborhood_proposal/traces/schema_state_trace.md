# Phase Y State Trace Schema

This document defines the exact structure of the state trace sent to the LLM
at a solver stagnation point (N consecutive rounds without improvement).

## Version: Y0.0

---

## Trace Structure

A state trace is a Markdown document with embedded JSON. The LLM receives the
Markdown rendering. The JSON variant is for parsing/automation.

### Section 0 — Trace Metadata

```
| Field | Value |
|-------|-------|
| trace_id | string, unique per snapshot |
| cell_label | anonymized: Cell A..Cell E |
| round | integer, current solver round number |
| timestamp | ISO 8601 |
```

### Section 1 — Cell Regime

Provides context about the problem scale. Anonymized but with epsilon shown
because it is critical for understanding load pressure and job count.

| Field | Type | Description | Rationale |
|-------|------|-------------|-----------|
| cell_label | string | Anonymized cell ID | Prevents overfitting to instance numbers |
| epsilon | integer | Workload cap per machine | Determines job count per machine, load pressure |
| num_machines | integer | Machine count | Scale context for neighborhood size |
| total_jobs | integer | Total jobs on this cell | Provides scale: how many jobs move options exist |
| epsilon_regime | enum | tight (< 250), medium (250-350), loose (> 350) | Coarse regime for pattern-matching without raw thresholds |
| job_size_range | [int, int] | Min/max processing time (typically [1, 12]) | Defines the move universe |
| episode_epsilon_progression | [int, ...] | List of epsilons in the episode so far (if descending-epsilon) | Shows trajectory of previous solve stages |

### Section 2 — Current Solution Snapshot

Describes the current schedule state at the stagnation point.

| Field | Type | Description | Rationale |
|-------|------|-------------|-----------|
| current_tec | number | Total exact cost of the current best schedule | Primary optimization target |
| best_tec_episode | number | Best TEC ever found in the full episode | Ceiling — how close are we? |
| tec_improvement_last_n_rounds | number | TEC change over last N rounds (0 = stagnation) | Quantifies stagnation severity |
| no_hit_streak | integer | Consecutive rounds without ANY move improvement | Determines whether to escalate escape |
| total_rounds_completed | integer | Rounds completed so far in this cell | How far into search are we? |
| total_accepted_moves_so_far | integer | All accepted moves across all rounds | Search effort already expended |
| exact_dp_evals_so_far | integer | All exact DP evaluations so far | Total computational budget spent |
| core_lane_stagnation_active | boolean | Whether the core DiverseTrimmed lane is exhausted (no new source/target pairs) | Key signal: is the shortlist working? |
| exception_lane_active | boolean | Whether exception lane is currently active | Are we already in fallback? |
| stop_reason_guard | enum | none, approaching_time_cap, approaching_round_cap, no_unexplored_sources | Solver self-report on why it may stop soon |

### Section 3 — Machine State Table

**All machines** are shown in a compact table. This is the core diagnostic tool.
Each row is one machine (anonymized as M0...M{n-1}).

| Column | Field | Description | Why Included |
|--------|-------|-------------|--------------|
| M | machine_id | M0..M{n-1} | Anonymous identifier |
| J | job_count | Number of jobs on this machine | How complex is re-packing? |
| L | total_load | Sum of processing times | How full is this machine? |
| S | slack | epsilon - total_load | Remaining capacity — 0 = full |
| LP | load_pressure | L / epsilon, range [0, 1+] | Normalized load — >0.95 = near capacity |
| EC | exact_cost | Exact DP cost of current machine schedule | Which machines drive TEC? |
| RLB | relaxed_lb | Cheap lower bound for this machine | Lower-bound estimate before exact DP |
| Gap | exact_minus_lb_gap | max(0, EC - RLB) | Where cheap LB is most wrong → hidden re-packing room |
| CD | cost_density | EC / max(1, L) | Cost per unit load — expensive machines should lose jobs |
| s | small_jobs | Count of jobs with p ∈ [1, 4] | Small jobs: easier to move, good for re-packing |
| m | medium_jobs | Count of jobs with p ∈ [5, 8] | Medium jobs |
| l | large_jobs | Count of jobs with p ∈ [9, 12] | Large jobs: harder to move, more impact |
| CS | core_source_hits | Times this machine was a SOURCE in core shortlist last 5 rounds | Is this machine over-attacked or under-attacked? |
| CT | core_target_hits | Times this machine was a TARGET in core shortlist last 5 rounds | Is this machine over-targeted or under-targeted? |
| Rate | machine_rate | Energy rate class (1-6, anonymized to 1-6) | Higher rate = more expensive energy |
| SL | starved_sources | Whether this source appeared in outside pool but never in core shortlist | Underexplored sources |

**Table format**: compact 17-column table, one row per machine. Sorted by EC
descending (most expensive machines first).

**Rationale for including ALL machines**: the LLM can process tabular data at
scale. Hiding machines would prevent it from discovering underexplored sources.

**Data freshness**: EC, RLB, Gap, and CD are snapshotted at the current round.
They are recomputable from solver state without expensive extra DP evaluations
if the solver caches these values from the last accepted move.

**Estimated table size**:
- 25 machines × 17 columns ≈ 425 cells → ~1500 tokens (well within DeepSeek limits)
- 40 machines × 17 columns ≈ 680 cells → ~2500 tokens (still fine)

### Section 4 — Recent Search Behavior

Describes what happened in recent rounds to help the LLM diagnose why the
solver is stuck.

| Field | Type | Description | Why Included |
|-------|------|-------------|--------------|
| last_accepted_moves | array of objects | Last 10 accepted moves: {round, source, target, job_size, delta_tec, was_exception} | What worked recently? |
| failed_move_families | array of strings | Categories of moves tried but never accepted in last 5 rounds | What is exhausted? |
| no_hit_by_source | array of {source, count} | Sources where core lane found no hits in last 5 rounds | Which sources are stubborn? |
| no_hit_by_target | array of {target, count} | Targets where core lane found no hits in last 5 rounds | Which targets are saturated? |
| core_shortlist_composition | object | {distinct_sources, distinct_targets, max_src_share, max_tgt_share} | How diverse is the current shortlist? |
| outside_pool_composition | object | {total_candidates, distinct_sources, distinct_targets, max_src_share, max_tgt_share} | What is available beyond the shortlist? |
| outside_pool_source_coverage | float | distinct_pool_sources / num_machines | How many machine floors does the pool peek into? |
| next_round_budget | object | {core_budget, exception_budget, exception_budget_max} | How much DP budget is allocated next? |

**Last accepted moves** example entry:
```json
{
  "round": 42,
  "source": "M3",
  "target": "M17",
  "job_size": "small",
  "delta_tec": -18.0,
  "was_exception": false
}
```

**Failed move families** are string labels like:
- "insert_inter(small jobs, high→low cost)" — tried, no improvement
- "insert_inter(large jobs, cheap targets)" — tried, no improvement
- "exception_lane_random" — tried, no improvement

### Section 5 — Candidate Pool Summaries

Aggregate summaries of what moves are available. The LLM does NOT see individual
candidates (too many), but sees grouped statistics.

| Field | Type | Description | Why Included |
|-------|------|-------------|--------------|
| top_sources_by_cost | array of {machine_id, EC, Gap, CD, job_count} | Top 5 sources by exact cost | Which machines to attack |
| top_sources_by_gap | array of {machine_id, EC, Gap, CD, job_count} | Top 5 sources by exact-Gap (hidden re-packing room) | Where cheap LB is wrong |
| top_sources_by_density | array of {machine_id, EC, Gap, CD, job_count} | Top 5 sources by cost_density (expensive per unit) | Best candidates for job removal |
| top_targets_by_slack | array of {machine_id, slack, LP, job_count} | Top 5 targets by available slack | Best candidates for job insertion |
| top_targets_by_low_cost | array of {machine_id, EC, LP, job_count, rate} | Top 5 targets by low exact cost | Cheapest machines to grow |
| underexplored_sources | array of {machine_id, EC, core_hits, outside_pool_count} | Machines with core_hits=0 in last 5 rounds | Sources the solver never touched |
| underexplored_targets | array of {machine_id, slack, LP, core_hits} | Machines with core_hits=0 as target in last 5 rounds | Targets the solver never considered |
| job_size_distribution_by_cost_quartile | object | Job size mix on machines grouped by cost quartile | Where are small/large jobs concentrated? |

**Job size distribution by cost quartile** format:
```
Cost Q1 (lowest cost machines): small=40%, medium=35%, large=25%
Cost Q2: small=30%, medium=30%, large=40%
Cost Q3: small=25%, medium=30%, large=45%
Cost Q4 (highest cost machines): small=20%, medium=30%, large=50%
```

### Section 6 — Prior Arm Results (if available)

If this cell was evaluated in prior phases (Phase S/V/X), provide the best
known TEC from each arm as an oracle ceiling.

| Arm | Best TEC | Δ vs current | Source Phase |
|-----|----------|-------------|--------------|
| trimmed | number | number | S |
| llm_exception | number | number | S |
| random_best | number | number | S |
| score_escape | number | number | V |
| phaseX_best_random | number | number | X |
| phaseX_best_llm | number | number | X |

This is optional — only included for dev cells where prior results exist.
Not included for held-out validation cells.

---

## Fields Intentionally Excluded

These fields are available from the solver but are NOT included in the trace
to avoid noise, overfitting, or context bloat:

| Excluded Field | Reason |
|----------------|--------|
| Raw instance ID | Prevents overfitting to specific instances. Only cell_label shown. |
| Real machine energy rates (rate class values 1-6) | Included as anonymized Rate column only — no raw cost parameters. |
| S1 score (raw cheap LB gain) | Redundant with RLB and Gap. S2 is richer and includes more signal. |
| Cheap-window electricity price curve | Problem-specific detail; the LLM does not need raw pricing data. |
| Per-candidate s2 score list | Too much data (thousands of candidates). Summarized by aggregate statistics. |
| Trajectory of every past round | Last 10 accepted moves + aggregate summaries are sufficient. |
| DP cache hit/miss rates | Implementation detail, not relevant for move diagnosis. |
| Full solver runtime | Not informative about WHERE to search. |
| Swap move statistics | Only insert_inter moves are relevant for the neighborhood. |
| Exact epsilon progression history | Summarized as episode_epsilon_progression header only. |
| CPU/thread count | Not relevant for move selection. |
| Greedy initial solution detail | The current state is what matters, not how we got there. |

---

## Example Trace (Abbreviated)

A full example trace is shown below for Cell A at Round 12 stagnation:

```markdown
# Solver State Trace — Cell A, Round 12

## 0. Metadata

| Field | Value |
|-------|-------|
| trace_id | cell_a_r12_20260510T120000Z |
| cell_label | Cell A |
| round | 12 |
| timestamp | 2026-05-10T12:00:00Z |

## 1. Cell Regime

| Field | Value |
|-------|-------|
| cell_label | Cell A |
| epsilon | 347 |
| num_machines | 25 |
| total_jobs | 250 |
| epsilon_regime | medium |
| job_size_range | [1, 12] |
| episode_epsilon_progression | [347] |

## 2. Current Solution Snapshot

| Field | Value |
|-------|-------|
| current_tec | 6884.0 |
| best_tec_episode | 6884.0 |
| tec_improvement_last_n_rounds | 0.0 |
| no_hit_streak | 5 |
| total_rounds_completed | 12 |
| total_accepted_moves_so_far | 4 |
| exact_dp_evals_so_far | 73 |
| core_lane_stagnation_active | true |
| exception_lane_active | false |
| stop_reason_guard | no_unexplored_sources |

## 3. Machine State Table

| M | J | L | S | LP | EC | RLB | Gap | CD | s | m | l | CS | CT | Rate | SL |
|:--|--:|--:|--:|----:|----:|----:|----:|-----:|--:|--:|--:|--:|--:|----:|--:|
| M0 | 22 | 335 | 12 | 0.97 | 1498 | 1340 | 158 | 4.5 | 3 | 10 | 9 | 3 | 0 | 5 | yes |
| M1 | 10 | 347 | 0 | 1.00 | 1204 | 1120 | 84 | 3.5 | 2 | 5 | 3 | 2 | 0 | 4 | no |
| M2 | 18 | 280 | 67 | 0.81 | 984 | 920 | 64 | 3.5 | 5 | 8 | 5 | 1 | 3 | 3 | no |
| ... (all 25 machines) |
| M24 | 0 | 0 | 347 | 0.00 | 0 | 0 | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 1 | no |

## 4. Recent Search Behavior

### Last Accepted Moves
| # | Round | Source | Target | Job Size | Δ TEC | Exception? |
|--:|:-----:|--------|--------|:--------:|------:|:----------:|
| 1 | 7 | M0 | M22 | large | -18.0 | no |
| 2 | 5 | M2 | M17 | small | -12.0 | no |
| 3 | 3 | M0 | M14 | medium | -8.0 | no |
| 4 | 1 | M1 | M23 | small | -6.0 | no |

### Failed Move Families
- insert_inter(small jobs, M0→low_cost_machines) — exhausted
- insert_inter(large jobs, M1→slack_machines) — exhausted
- No exception lane activated yet

### Core Shortlist Composition
| Metric | Value |
|--------|-------|
| distinct_sources | 5 |
| distinct_targets | 12 |
| max_src_share | 0.40 (M0 dominates) |
| max_tgt_share | 0.20 |

### Outside Pool Composition
| Metric | Value |
|--------|-------|
| total_candidates | 3609 |
| distinct_sources | 5 |
| distinct_targets | 24 |
| max_src_share | 0.40 |
| max_tgt_share | 0.12 |
| source_coverage | 0.20 (5/25) |

### Next Round Budget
| Metric | Value |
|--------|-------|
| core_budget | 14 |
| exception_budget_base | 0 |
| exception_budget_max | 12 |

## 5. Candidate Pool Summaries

### Top Sources by Cost
| M | EC | Gap | CD | Jobs |
|:--|----:|----:|-----:|-----:|
| M0 | 1498 | 158 | 4.5 | 22 |
| M1 | 1204 | 84 | 3.5 | 10 |
| M2 | 984 | 64 | 3.5 | 18 |
| M3 | 812 | 42 | 3.2 | 15 |
| M4 | 720 | 38 | 3.1 | 14 |

### Top Sources by Gap (hidden re-packing room)
| M | Gap | EC | CD | Jobs |
|:--|----:|----:|-----:|-----:|
| M0 | 158 | 1498 | 4.5 | 22 |
| M1 | 84 | 1204 | 3.5 | 10 |
| M2 | 64 | 984 | 3.5 | 18 |
| M3 | 42 | 812 | 3.2 | 15 |
| M4 | 38 | 720 | 3.1 | 14 |

### Top Sources by Cost Density
| M | CD | EC | Gap | Jobs |
|:--|-----:|----:|----:|-----:|
| M8 | 6.2 | 620 | 30 | 8 |
| M0 | 4.5 | 1498 | 158 | 22 |
| M9 | 4.2 | 460 | 22 | 6 |
| M1 | 3.5 | 1204 | 84 | 10 |
| M2 | 3.5 | 984 | 64 | 18 |

### Top Targets by Slack
| M | Slack | LP | Jobs |
|:--|:-----:|----:|-----:|
| M24 | 347 | 0.00 | 0 |
| M23 | 345 | 0.01 | 2 |
| M22 | 340 | 0.02 | 3 |
| M21 | 335 | 0.03 | 4 |
| M20 | 320 | 0.08 | 5 |

### Top Targets by Low Cost
| M | EC | LP | Jobs | Rate |
|:--|----:|----:|-----:|----:|
| M24 | 0 | 0.00 | 0 | 1 |
| M23 | 4 | 0.01 | 2 | 1 |
| M22 | 12 | 0.02 | 3 | 1 |
| M21 | 16 | 0.03 | 4 | 2 |
| M20 | 28 | 0.08 | 5 | 1 |

### Underexplored Sources (0 core hits in last 5 rounds)
| M | EC | Gap | CD | Core Hits |
|:--|----:|----:|-----:|:---------:|
| M5 | 680 | 35 | 3.0 | 0 |
| M6 | 640 | 28 | 2.8 | 0 |
| M7 | 540 | 24 | 2.5 | 0 |
| ... (20 machines with 0 source hits) |

### Underexplored Targets (0 core hits as target in last 5 rounds)
| M | Slack | LP | Jobs | Core Hits |
|:--|:-----:|----:|-----:|:---------:|
| M15 | 280 | 0.19 | 8 | 0 |
| M16 | 270 | 0.22 | 6 | 0 |
| ... (13 machines with 0 target hits) |

### Job Size Distribution by Cost Quartile
| Quartile | Small (1-4) | Medium (5-8) | Large (9-12) |
|----------|:-----------:|:------------:|:------------:|
| Q4 (highest) | 20% | 30% | 50% |
| Q3 | 25% | 30% | 45% |
| Q2 | 30% | 35% | 35% |
| Q1 (lowest) | 40% | 35% | 25% |

## 6. Prior Arm Results (dev cells only)

| Arm | Best TEC | Δ vs Current | Phase |
|-----|----------|:-----------:|-------|
| trimmed | 6884 | 0 | S |
| llm_exception | 6869 | -15 | S |
| random_best | 6852 | -32 | S |
| score_escape | 6884 | 0 | V |
| phaseX_random_best | 6884 | 0 | X |
| phaseX_llm_best | 6884 | 0 | X |
```

---

## JSON Schema Companion

The JSON trace (`{trace_id}.json`) mirrors the Markdown structure exactly.
It is used for automated parsing and preprocessing before DeepSeek calls.
The Markdown version is what DeepSeek sees in the prompt.

Key JSON structure:
```json
{
  "meta": { "trace_id": "...", "cell_label": "...", "round": 0, ... },
  "regime": { "epsilon": 0, "num_machines": 0, ... },
  "snapshot": { "current_tec": 0.0, ... },
  "machines": [ { "id": "M0", "job_count": 0, ... }, ... ],
  "recent": { "last_accepted_moves": [...], ... },
  "candidate_pools": { "top_sources_by_cost": [...], ... },
  "prior_arms": { "trimmed": 0.0, ... }
}
```

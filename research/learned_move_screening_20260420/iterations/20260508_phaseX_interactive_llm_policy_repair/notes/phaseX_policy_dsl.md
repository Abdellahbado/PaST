# Phase X Policy DSL

## Purpose

A constrained JSON policy DSL controlling the exception lane in the
exact-DP heuristic solver. The LLM explores this space interactively:
propose a policy JSON → evaluate → see results → repair.

## Architecture

The policy controls only the exception lane. The core DiverseTrimmed
shortlist, source ranking, and exact-DP verification remain unchanged.
No source expansion or core integration.

## Fields

### Scoring

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| `normal_mode` | string | `llm_score`, `s2`, `random`, `cheap_lb`, `hybrid` | Scoring in normal rounds |
| `escape_mode` | string | `none`, `cheap_lb_pair`, `random_pair`, `coverage`, `anti_s2` | Scoring in escape rounds |
| `cheap_lb_weight` | float 0-1 | — | Weight of cheap_lb_delta in hybrid mode |
| `s2_weight` | float 0-1 | — | Weight of s2 score in hybrid mode |
| `slack_weight` | float 0-1 | — | Weight of slack bonus in hybrid mode |
| `random_mix` | float 0-1 | — | Random fraction in hybrid mode |
| `coverage_bonus` | float 0-3 | — | Bonus per uncovered machine (coverage mode) |

#### normal_mode details

- `llm_score`: `s2 + slack_bonus + tightness_bonus` (same as LLM exception lane)
  - `slack_bonus = (tgt_slack / epsilon) * 0.5` where `tgt_slack = max(0, epsilon - tgt_load)`
  - `tightness_bonus = src_tightness * 0.2` where `src_tightness = max(0, 1 - src_load/epsilon)`
  - If s2 is invalid (< -1e9), uses fallback: `0.60 * s1 + 0.40 * cheap_lb_delta + 0.30 * source_gap`
- `s2`: Raw s2 score (s1 rerank by machine load distribution)
- `random`: Uniform random via seeded RNG (per-instance, per-round, per-candidate)
- `cheap_lb`: `cheap_lb_delta` (fast lower-bound improvement estimate, no DP)
- `hybrid`: Weighted linear combination of all components

#### escape_mode details

- `none`: No escape — stay in normal mode
- `cheap_lb_pair`: Best cheap_lb_delta per (source, target) pair, sort by delta
- `random_pair`: Random (source, target) pairs
- `coverage`: Reward uncovered machines, priority to novel source/target
- `anti_s2`: `score = max(0, cheap_lb_delta) - s2` (for cells where s2 mis-ranks)

### Budget

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `initial_budget` | int | 1-8 | Starting evaluations per round |
| `max_budget` | int | 4-16 | Upper bound on budget |
| `grow_on_hit` | int | 0-4 | Add on improvement |
| `shrink_on_miss` | int | 0-4 | Remove after 2+ consecutive misses |
| `guard_max_budget` | int | 0-4 | Budget cap on tight-epsilon rounds (eps_per_job ≤ 3.0). 0 = skip on guard |

Budget adaptation:
1. Start: `budget = initial_budget`
2. On hit: `budget = min(max_budget, budget + grow_on_hit)`
3. On 2+ consecutive misses (with shortlist improvement): `budget = max(1, budget - shrink_on_miss)`
4. Guard rounds: `budget = min(budget, guard_max_budget)` or skip if guard_max_budget = 0

### Mode Switching

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `switch_after_no_hit` | int | 0-4 | Consecutive non-improving rounds to trigger escape. 0 = never escape |
| `switch_back_on_hit` | bool | — | Return to normal after improvement in escape |

### Diversity

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `max_per_source` | int | 1-4 | Max candidates per source machine |
| `max_per_target` | int | 1-4 | Max candidates per target machine |

### Filtering

| Field | Type | Description |
|-------|------|-------------|
| `require_positive_cheap_lb` | bool | If true, drop candidates with cheap_lb_delta ≤ 0 |

## Logged Fields (per evaluation)

| Field | Description |
|-------|-------------|
| `policy_name` | Policy identifier |
| `phaseX_normal_rounds` | Rounds spent in normal mode |
| `phaseX_escape_rounds` | Rounds spent in escape mode |
| `phaseX_candidates_considered` | Candidates scored in exception lane |
| `phaseX_candidates_evaluated` | Candidates evaluated with exact DP |
| `phaseX_improvement_count` | Accepted exception moves |
| `phaseX_best_delta` | Best cost delta from exception lane |

## Example Policy

See `policies/example_policy.json` — equivalent to LLM exception lane behavior
(adaptive budget 4→12, s2+slack+tightness scoring, no escape, quotas=3).

## Invocation

```bash
PHASEX_POLICY_PATH=/path/to/policy.json \
  ./parallel_heuristic_compare paper-instance 61 347 phaseX_policy_json \
  temp/paper_exact_repo/instances 30.0 10.0 5 20000
```

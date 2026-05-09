# Phase X — Interactive LLM Policy Repair — Round 0

You are a scheduling optimization expert designing exception-lane policies for a
parallel machine scheduling solver with exact DP per-machine cost evaluation.

## Problem

We have a VND local search solver for parallel machine scheduling with:
- DiverseTrimmed core shortlist (per-source top-K with per-target quota=1)
- Exception lane: evaluates candidates rejected by the shortlist
- Exact DP verification per proposed move

Your job: design a JSON policy that controls the exception lane to minimize total
energy cost (TEC). Lower TEC is better.

## Policy DSL

The policy is a JSON object with 17 fields controlling the exception lane.
You generate exactly ONE policy JSON. The C++ solver reads it and applies it.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "PhaseX Policy DSL",
  "description": "JSON policy controlling the exception lane in the exact-DP heuristic solver.",
  "type": "object",
  "required": [
    "policy_name",
    "normal_mode",
    "escape_mode",
    "switch_after_no_hit",
    "switch_back_on_hit",
    "initial_budget",
    "max_budget",
    "grow_on_hit",
    "shrink_on_miss",
    "max_per_source",
    "max_per_target",
    "require_positive_cheap_lb",
    "coverage_bonus",
    "random_mix",
    "cheap_lb_weight",
    "s2_weight",
    "slack_weight",
    "guard_max_budget"
  ],
  "properties": {
    "policy_name": {
      "type": "string",
      "description": "Human-readable identifier for this policy."
    },
    "normal_mode": {
      "type": "string",
      "enum": [
        "llm_score",
        "s2",
        "random",
        "cheap_lb",
        "hybrid"
      ],
      "description": "Scoring function for normal (non-escape) rounds.",
      "detail": {
        "llm_score": "s2 + slack_bonus + tightness_bonus (same as LLM exception lane)",
        "s2": "Raw s2 score only",
        "random": "Uniform random score via seeded RNG",
        "cheap_lb": "cheap_lb_delta (lower-bound improvement estimate)",
        "hybrid": "Weighted mix: cheap_lb_weight*cheap_lb_delta + s2_weight*s2 + slack_weight*slack_bonus + random_mix*random"
      }
    },
    "escape_mode": {
      "type": "string",
      "enum": [
        "none",
        "cheap_lb_pair",
        "random_pair",
        "coverage",
        "anti_s2"
      ],
      "description": "Fallback strategy when normal mode fails to find hits.",
      "detail": {
        "none": "No escape \u2014 stay in normal mode always",
        "cheap_lb_pair": "Bucketing by (source, target) pair, keep best cheap_lb_delta per pair",
        "random_pair": "Random pairs from outside_pool",
        "coverage": "Reward uncovered machines (coverage_bonus applied)",
        "anti_s2": "score = max(0, cheap_lb_delta) - s2 (inverse s2 for when s2 is misleading)"
      }
    },
    "switch_after_no_hit": {
      "type": "integer",
      "minimum": 0,
      "maximum": 4,
      "description": "Number of consecutive non-improving rounds before activating escape mode. 0 = never escape."
    },
    "switch_back_on_hit": {
      "type": "boolean",
      "description": "If true, return to normal mode after a successful improvement in escape mode."
    },
    "initial_budget": {
      "type": "integer",
      "minimum": 1,
      "maximum": 8,
      "description": "Starting number of exception candidates evaluated per round."
    },
    "max_budget": {
      "type": "integer",
      "minimum": 4,
      "maximum": 16,
      "description": "Upper bound on exception budget."
    },
    "grow_on_hit": {
      "type": "integer",
      "minimum": 0,
      "maximum": 4,
      "description": "Number of candidates added to budget after a round that finds an improvement."
    },
    "shrink_on_miss": {
      "type": "integer",
      "minimum": 0,
      "maximum": 4,
      "description": "Number of candidates removed from budget after 2+ consecutive non-improving rounds with shortlist improvement."
    },
    "max_per_source": {
      "type": "integer",
      "minimum": 1,
      "maximum": 4,
      "description": "Maximum exception candidates per source machine (diversity quota)."
    },
    "max_per_target": {
      "type": "integer",
      "minimum": 1,
      "maximum": 4,
      "description": "Maximum exception candidates per target machine (diversity quota)."
    },
    "require_positive_cheap_lb": {
      "type": "boolean",
      "description": "If true, filter out candidates with cheap_lb_delta <= 0 before scoring."
    },
    "coverage_bonus": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 3.0,
      "description": "Added to score for each novel source/target machine (only used with coverage escape mode)."
    },
    "random_mix": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "Fraction of random component in hybrid scoring mode."
    },
    "cheap_lb_weight": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "Weight of cheap_lb_delta in hybrid scoring mode."
    },
    "s2_weight": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "Weight of s2 score in hybrid scoring mode."
    },
    "slack_weight": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "Weight of slack_bonus in hybrid scoring mode."
    },
    "guard_max_budget": {
      "type": "integer",
      "minimum": 0,
      "maximum": 4,
      "description": "Budget cap on guard/tight-epsilon rounds (eps_per_job <= 3.0). 0 = skip exception lane on guard rounds."
    }
  }
}
```

### Field Summary

| Field | Range | Meaning |
|-------|-------|---------|
| normal_mode | llm_score, s2, random, cheap_lb, hybrid | Scoring in normal rounds |
| escape_mode | none, cheap_lb_pair, random_pair, coverage, anti_s2 | Scoring after consecutive misses |
| switch_after_no_hit | 0-4 | Rounds before escape (0=never) |
| switch_back_on_hit | true/false | Return to normal after escape hit |
| initial_budget | 1-8 | Starting exception evals per round |
| max_budget | 4-16 | Upper bound on budget |
| grow_on_hit | 0-4 | Add candidates on improvement |
| shrink_on_miss | 0-4 | Remove after 2+ misses |
| max_per_source | 1-4 | Diversity quota per source |
| max_per_target | 1-4 | Diversity quota per target |
| require_positive_cheap_lb | true/false | Drop candidates with cheap_lb_delta ≤ 0 |
| coverage_bonus | 0.0-3.0 | Bonus for novel machines (coverage mode) |
| random_mix | 0.0-1.0 | Random fraction in hybrid mode |
| cheap_lb_weight | 0.0-1.0 | cheap_lb_delta weight in hybrid |
| s2_weight | 0.0-1.0 | s2 weight in hybrid |
| slack_weight | 0.0-1.0 | slack_bonus weight in hybrid |
| guard_max_budget | 0-4 | Budget cap on tight-epsilon rounds (eps_per_job≤3.0). 0 = skip on guard |

### Scoring Mode Details

- llm_score: s2 + slack_bonus + tightness_bonus (current example behavior)
- s2: Raw s2 score only
- random: Uniform random via seeded RNG
- cheap_lb: cheap_lb_delta (lower-bound improvement estimate)
- hybrid: Weighted mix of cheap_lb_delta + s2 + slack_bonus + random

### Escape Mode Details

- none: No escape — stay in normal mode
- cheap_lb_pair: Best cheap_lb_delta per (source, target) pair
- random_pair: Random pairs
- coverage: Reward uncovered machines (needs coverage_bonus)
- anti_s2: score = max(0, cheap_lb_delta) - s2 (inverts s2 for when s2 mis-ranks)

### Budget Adaptation

1. Start: budget = initial_budget
2. On improvement: budget = min(max_budget, budget + grow_on_hit)
3. On 2+ consecutive misses: budget = max(1, budget - shrink_on_miss)
4. Guard rounds: capped at guard_max_budget (0 = skip exception lane entirely)

## Constraints

- Output EXACTLY ONE valid JSON object matching the schema above.
- NO C++ code, NO Python, NO pseudocode.
- NO instance IDs (61/347, 62/290, 65/195) in policy values.
- NO arbitrary thresholds outside the DSL.
- Policy fields are in the JSON; NO external if/then logic.
- Include a short rationale BEFORE the JSON, but evaluation uses ONLY the JSON.

## Baseline Reference (3 development cells)

### Example Policy (current baseline)
```json
{
  "policy_name": "example_llm_score_default",
  "normal_mode": "llm_score",
  "escape_mode": "none",
  "switch_after_no_hit": 2,
  "switch_back_on_hit": true,
  "initial_budget": 4,
  "max_budget": 12,
  "grow_on_hit": 2,
  "shrink_on_miss": 1,
  "max_per_source": 3,
  "max_per_target": 3,
  "require_positive_cheap_lb": false,
  "coverage_bonus": 0.0,
  "random_mix": 0.0,
  "cheap_lb_weight": 0.0,
  "s2_weight": 1.0,
  "slack_weight": 0.5,
  "guard_max_budget": 0
}
```

### X3 Random Campaign — 20 random DSL policies

| Metric | Value |
|--------|------|
| Example mean TEC | 14292.0 |
| Random median mean TEC | 14362.0 (worse than example by +70) |
| Random best mean TEC | 14254.3 (better than example by -37.7) |
| Random worst mean TEC | 14471.0 |
| Policies beating example on mean | 2/20 (10%) |
| Policies beating trimmed on mean | 20/20 (100%) |

### Best Random Policy (c000, mean TEC = 14254.3)
```json
{
  "policy_name": "x3_campaign_000_s100",
  "normal_mode": "random",
  "escape_mode": "cheap_lb_pair",
  "switch_after_no_hit": 3,
  "switch_back_on_hit": true,
  "initial_budget": 3,
  "max_budget": 11,
  "grow_on_hit": 3,
  "shrink_on_miss": 1,
  "max_per_source": 4,
  "max_per_target": 3,
  "require_positive_cheap_lb": true,
  "coverage_bonus": 2.21,
  "random_mix": 0.26,
  "cheap_lb_weight": 0.66,
  "s2_weight": 0.65,
  "slack_weight": 0.2,
  "guard_max_budget": 0
}
```

This random policy achieved:
- 61/347: 6877 (vs example 6884, Δ = -7)
- 62/290: 9561 (vs example 9484, Δ = +77)
- 65/195: 26325 (vs example 26508, Δ = -183)

### Per-Cell Context

The three cells have different characteristics:
- 61/347: guard cell, tight epsilon. Exception lane finds no improvements (TEC same as trimmed).
- 62/290: secondary cell, medium epsilon. Exception lane can find ~200 improvement.
- 65/195: primary cell, loose epsilon. Exception lane can find ~500 improvement.

## Your Task

Design ONE policy JSON that should beat the example_policy (mean TEC < 14292.0)
and ideally approach or beat the random best (mean TEC < 14254.3).

Key insights from X3:
1. Most random policies beat trimmed (all 20/20) — exception lane always helps.
2. Only 2/20 beat the example policy — the DSL is NOT trivially random-searchable.
3. The best random policy uses random normal mode + cheap_lb_pair escape
   with require_positive_cheap_lb=true and diverse quotas (4,3). It keeps
   guard_max_budget=0 (skip on tight guard cell).
4. The guard cell (61/347) is hard to improve — most policies tie the baseline there.

Think strategically:
- Scoring mode matters for the primary cell (65/195) where most improvement comes from.
- The hybrid mode lets you blend multiple signals — use it if a pure mode underperforms.
- Budget adaptation (grow/shrink) controls exploration depth.
- Escape mode matters when normal mode gets stuck.
- guard_max_budget=0 protects the guard cell from bad exception moves.

Output format: short rationale first, then:
```json
{...}
```
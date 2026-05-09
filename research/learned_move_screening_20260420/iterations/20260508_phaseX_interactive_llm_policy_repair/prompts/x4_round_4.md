# Phase X — Interactive LLM Policy Repair — Round 4

## Current Best Policy (Round 2/3)
```json
{
  "policy_name": "llm_cheaplb_escape_v2",
  "normal_mode": "llm_score",
  "escape_mode": "cheap_lb_pair",
  "switch_after_no_hit": 2,
  "switch_back_on_hit": true,
  "initial_budget": 3,
  "max_budget": 11,
  "grow_on_hit": 3,
  "shrink_on_miss": 1,
  "max_per_source": 4,
  "max_per_target": 4,
  "require_positive_cheap_lb": false,
  "coverage_bonus": 0.0,
  "random_mix": 0.0,
  "cheap_lb_weight": 0.0,
  "s2_weight": 1.0,
  "slack_weight": 0.5,
  "guard_max_budget": 0
}
```

### Per-Cell TEC Results

| Cell | Your TEC | Example TEC | Δ vs Example | ScoreEsc TEC | Best Random c000 TEC |
|------|---------|------------|-------------|-------------|---------------------|
| 61/347 | 6884 | 6884 | +0 | 6884 | 6877 |
| 62/290 | 9495 | 9484 | +11 | 9484 | 9561 |
| 65/195 | 26478 | 26508 | -30 | 26508 | 26325 |
| **Mean** | **14285.7** | **14292.0** | **-6.3** | **14292.0** | **14254.3** |

### All Round History

| Round | 61/347 | 62/290 | 65/195 | Mean TEC | Δ Example |
|-------|--------|--------|--------|----------|-----------|
| 0 | 6884 | 9534 | 26484 | 14300.7 | +8.7 |
| 1 | 6884 | 9534 | 26484 | 14300.7 | +8.7 |
| 2 | 6884 | 9495 | 26478 | 14285.7 | -6.3 |
| 3 | 6884 | 9495 | 26478 | 14285.7 | -6.3 |

### DSL CONSTRAINT: max_per_target ≤ 4

Round 3 attempted `max_per_target=5` but was capped to 4 by the DSL schema.
Further increases to max_per_target are NOT possible.

### Status

| Baseline | Mean TEC | Δ vs Current Best |
|----------|---------:|-----------------:|
| Example policy | 14292.0 | +6.3 (you BEAT it) |
| Random median | 14362.0 | +76.3 (you beat it) |
| Random best c000 | 14254.3 | -31.4 (you trail) |
| Trimmed | 14534.0 | +248.3 (you beat it) |

## Key Observations

1. **61/347 (guard):** Flat — neither you nor any baseline improves on the tight guard cell.
2. **62/290 (secondary):** You're +11 vs example. The problem is clear: `llm_score` normal mode + `cheap_lb_pair` escape produces TEC=9495 vs the ScoreEsc baseline at 9484. Interestingly, the random best c000 (normal: random, escape: cheap_lb_pair) was WORSE at 9561 on this cell but better on 65/195.
3. **65/195 (primary):** You're -30 vs example. Good, but random best c000 is at 26325 (Δ = -153 vs example). You're at 26478. You need another ~150 improvement on this cell.

## Your Task

Since `max_per_target` is capped at 4, you must find a DIFFERENT strategy.
The current approach (llm_score + cheap_lb_pair escape) has plateaued at 14285.7.

Options to consider:
- **Change normal_mode**: Try `hybrid` to blend cheap_lb_delta with s2. Or try `s2` or `cheap_lb` pure modes. The random best c000 used `random` normal mode (strange but worked on 65/195).
- **Change escape_mode**: Try `anti_s2` (inverts s2 when it mis-ranks) or `coverage` (with coverage_bonus). Or try `none` if the normal mode is strong enough.
- **Tune budget**: current budget=3→11. Maybe a different budget trajectory helps.
- **Switch `require_positive_cheap_lb` back to `true`**: Round 0 had it true but also had max_per_target=3. Now with max_per_target=4 and other changes, require_positive_cheap_lb=true might filter better.
- **Change grow/shrink rate**: grow_on_hit=3 is aggressive (triples budget on hit). Maybe a steadier approach.
- **Change switch threshold**: switch_after_no_hit=2 is moderate. Maybe 1 (switch faster) or 3 (persist longer in normal mode).

Output: EXACTLY ONE JSON, then analysis. State what you changed and why.

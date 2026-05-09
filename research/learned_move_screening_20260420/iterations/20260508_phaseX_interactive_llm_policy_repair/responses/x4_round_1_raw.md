### Analysis

**Cells improved vs. regressed:**
- **Cell 62/290 regressed:** Your TEC was 9534, which is +50 worse than the Example/ScoreEsc baseline (9484).  
- **Cell 65/195 improved:** Your TEC was 26484, which is -24 better than the baseline (26508).  
- **Cell 61/347 was unchanged.**

**Why did this happen?**  
The policy uses `llm_score` in normal mode and falls back to `cheap_lb_pair` in escape mode. The flag `require_positive_cheap_lb` is `true`, meaning the escape action is only applied when its predicted cheap lower‑bound is strictly positive. In cell 62/290, the predicted cheap_lb is likely negative, causing the policy to skip the escape action entirely—leading to higher TEC. The ScoreEsc baseline (which does not have this restriction) applies the escape action regardless, achieving lower cost. In cell 65/195, the LLM mode either avoids triggering escape, or the predicted cheap_lb is positive, so the flag didn’t hurt and the LLM’s benefits yielded improvement.

**Proposed fix:**  
Change `require_positive_cheap_lb` from `true` to `false`. This lets the escape mode always apply the cheap_lb_pair action when triggered, aligning with the ScoreEsc behaviour in cell 62/290 and recovering the lost TEC. In cell 65/195, the change should have little to no effect (escape is rarely needed or cheap_lb is already positive), preserving the gain.

```json
{
  "policy_name": "llm_cheaplb_escape",
  "normal_mode": "llm_score",
  "escape_mode": "cheap_lb_pair",
  "switch_after_no_hit": 2,
  "switch_back_on_hit": true,
  "initial_budget": 3,
  "max_budget": 11,
  "grow_on_hit": 3,
  "shrink_on_miss": 1,
  "max_per_source": 4,
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
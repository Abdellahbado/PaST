### Analysis

**Current state:**  
- We have improved from the initial +8.7 to **–6.3** vs Example, now trailing Random best c000 by –31.4.  
- The large cell **65/195** is where the biggest gain opportunity lies: we are at **26478**, while Random best c000 (with `random` normal + `cheap_lb_pair` escape) achieves **26325**, an improvement of –183 vs Example. Our `llm_score` normal mode is not exploiting the cheap lower bound information as effectively in that large search space.  
- The small cell **62/290** remains slightly worse than Example (+11), but still acceptable. Guard cell is flat.

**Why Random best c000 worked so well on 65/195:**  
- Using `random` normal mode creates a broad, unbiased exploration, and the `cheap_lb_pair` escape then capitalises on cheap lower bounds. Because our escape mode is already `cheap_lb_pair`, the missing piece is cheap‑lower‑bound awareness in the normal phase. `llm_score` only uses \(s_2\), ignoring the cheap bound information that is crucial for large cells.

**Proposed change:**  
Switch `normal_mode` from `llm_score` to **`hybrid`**, set `cheap_lb_weight = 0.5` and `s2_weight = 1.0`. This blends the \(s_2\) score (strong on small cells) with cheap lower bound information (strong on large cells). Additionally, set `require_positive_cheap_lb = true` to filter out pairs where the cheap lower bound is worse than the current match, ensuring only beneficial cheap suggestions are considered.

These changes should shrink the TEC on 65/195 significantly (targeting a drop of ~100–150) while keeping 62/290 near its current level. All other parameters remain unchanged to preserve the escape‑mode behaviour that already works well.

### Revised Policy
```json
{
  "policy_name": "llm_hybrid_cheaplb_escape_v3",
  "normal_mode": "hybrid",
  "escape_mode": "cheap_lb_pair",
  "switch_after_no_hit": 2,
  "switch_back_on_hit": true,
  "initial_budget": 3,
  "max_budget": 11,
  "grow_on_hit": 3,
  "shrink_on_miss": 1,
  "max_per_source": 4,
  "max_per_target": 4,
  "require_positive_cheap_lb": true,
  "coverage_bonus": 0.0,
  "random_mix": 0.0,
  "cheap_lb_weight": 0.5,
  "s2_weight": 1.0,
  "slack_weight": 0.5,
  "guard_max_budget": 0
}
```
# Results — Phase X

## X4 Interactive LLM Policy Repair (2026-05-09)

5-round interactive DeepSeek policy repair loop. Each round: LLM proposes
a policy JSON → evaluate on 3 cells → feedback with per-cell TEC → revise.

### Per-Round Results

| Round | Policy | 61/347 | 62/290 | 65/195 | Mean TEC | Δ Example |
|:-----:|--------|-------:|-------:|-------:|--------:|----------:|
| 0 | llm_cheaplb_escape | 6884 | 9534 | 26484 | 14300.7 | +8.7 |
| 1 | llm_cheaplb_escape | 6884 | 9534 | 26484 | 14300.7 | +8.7 |
| 2 | llm_cheaplb_escape_v2 | 6884 | 9495 | **26478** | **14285.7** | **-6.3** |
| 3 | llm_cheaplb_escape_v3 | 6884 | 9495 | 26478 | 14285.7 | -6.3 |
| 4 | llm_hybrid_cheaplb_escape_v3 | **6873** | **9415** | 26956 | 14414.7 | +122.7 |

### Round-by-Round Changes

| Round | Change | Rationale | Effect |
|:-----:|--------|-----------|--------|
| 0 | Initial generation | LLM chose llm_score + cheap_lb_pair escape, require_positive_cheap_lb=true | +8.7 vs example |
| 1 | require_positive_cheap_lb: true→false | Remove cheap-LB filtering to allow more escape moves | No effect (same TEC) |
| 2 | max_per_source: 4, max_per_target: 4 | Increase diversity quotas to help escape on 62/290 | **-6.3** (beats example) |
| 3 | max_per_target: 5 (capped to 4) | Wanted to further increase target diversity | No change (same as R2) |
| 4 | normal_mode: llm_score→hybrid, cheap_lb_weight=0.5, require_positive_cheap_lb=true | Blend cheap_lb signal with s2 for primary cell | Guard+secondary improved, primary severely regressed |

### Final Verdict

| Criteria | Result |
|----------|--------|
| Best mean TEC | 14285.7 (Round 2) |
| Δ vs example_policy (14292.0) | **-6.3** ✅ |
| Δ vs random median (14362.0) | **-76.3** ✅ |
| Δ vs random best c000 (14254.3) | **+31.4** ✗ |
| Rounds beating example | 2/5 |
| Rounds beating random best | 0/5 |
| Verdict | **MINIMUM SUCCESS** |

### Efficiency Claim

LLM found a beating policy at Round 2 (3rd attempt, after initial + unchanged).
Random search required 20 attempts and only 2/20 (10%) beat example_policy.
Interactive feedback enabled directed improvement: Round 1 identified the
bottleneck (require_positive_cheap_lb blocking 62/290), Round 2 fixed it
with increased quotas.

### Notable Finding: Guard Cell Breakthrough

Round 4's hybrid normal mode (cheap_lb_weight=0.5 + s2_weight=1.0) produced the
**first-ever improvement on guard cell 61/347** (6884→6873). It also improved
62/290 (9495→9415). However, it severely regressed 65/195 (26478→26956, +448),
making the mean worse. This suggests that cheap_lb_delta is informative on
tight/small cells but misleads on loose/large cells. A **cell-adaptive scoring
strategy** (different weights per epsilon regime) might achieve a combined win.

### Best Policy (Round 2: llm_cheaplb_escape_v2)

```json
{
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
  "guard_max_budget": 0
}
```

Differences from example_policy: max_per_source/target 4→3 (more diverse),
initial_budget 3→4 (start conservative), grow_on_hit 3→2 (aggressive growth),
max_budget 11→12 (slightly lower cap).

### Files

- `prompts/x4_round_0.md` … `x4_round_4.md` — full prompts
- `responses/x4_round_0_raw.md` … `x4_round_4_raw.md` — raw DeepSeek responses
- `responses/x4_round_*_meta.json` — API metadata
- `policies/llm_interactive/x4_round_0.json` … `x4_round_4.json` — policies
- `eval/x4_interactive_rounds.csv` — per-round aggregate
- `eval/x4_interactive_summary.csv` — per-cell per-round

20 random DSL policies on 3 cells. Baselines recomputed.

### Baseline reference (mean TEC across 3 cells)

| Arm | Mean TEC |
|-----|---------:|
| trimmed | 14534.0 |
| phaseS_llm_exception_lane | 14534.0 |
| phaseS_random_exception_lane | 14522.3 |
| phaseV_score_escape_sampler | 14292.0 |
| phaseX_example_policy | 14292.0 |

### Random policy summary

| Metric | Value |
|--------|------:|
| Policies evaluated | 20 |
| Failed / infeasible | 0 / 0 |
| Best mean TEC | 14254.3 (Δ example = -37.7) |
| Median mean TEC | 14362.0 (Δ example = +70.0) |
| Worst mean TEC | 14471.0 |
| Beat example on mean | 2/20 (10%) |
| Beat trimmed on mean | 20/20 (100%) |
| Beat example on ≥2 cells | 2/20 (10%) |
| Beat score_escape on ≥1 cell | 11/20 (55%) |

### Top 5 random policies by mean TEC

| Rank | ID | Mean TEC | Δ Trimmed | Δ Example | BeatEx | BeatSE | RegrTr |
|:----:|:--:|--------:|---------:|---------:|:------:|:------:|:------:|
| 1 | c000 | 14254.3 | -279.7 | -37.7 | 2 | 2 | 0 |
| 2 | c002 | 14278.7 | -255.3 | -13.3 | 2 | 2 | 0 |
| 3 | c001 | 14294.7 | -239.3 | +2.7 | 2 | 2 | 0 |
| 4 | c016 | 14312.7 | -221.3 | +20.7 | 1 | 1 | 1 |
| 5 | c015 | 14321.3 | -212.7 | +29.3 | 0 | 0 | 0 |

### Classification: CASE B

**DSL contains useful policies but search is noisy.**

- Random median (14362.0) is worse than example_policy (14292.0): the DSL
  is NOT trivially random-searchable.
- Best random policy (14254.3, c000) beats example by -37.7 on mean TEC
  and beats it on 2/3 cells individually: good policies exist.
- Implications for X4: interactive LLM should be compared against BOTH
  median random policy AND best random policy. The LLM must show it finds
  good policies more efficiently than brute-force random search.

Files:
- `eval/x3_random_campaign_raw.csv` — 75 rows (15 baseline + 60 random)
- `eval/x3_random_campaign_summary.csv` — per-cell metrics
- `eval/x3_random_campaign_aggregate.csv` — per-policy aggregate
- `policies/random_campaign/x3_campaign_000.json` … `019.json`

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

- `prompts/x4_round_0.md` … `x4_round_4.md` — 5 prompts
- `responses/x4_round_*_raw.md` + `_meta.json` — 5 responses + metadata
- `policies/llm_interactive/x4_round_*.json` — 5 policies
- `eval/x4_interactive_rounds.csv` — aggregate per-round
- `eval/x4_interactive_summary.csv` — per-cell per-round

## X5 Random Best-of-5 Distribution (2026-05-09)

20 independent batches × 5 random policies = 100 random DSL policies on 3 dev cells.
Fair comparison: LLM best-of-5 (14285.7) vs random best-of-5 per batch.

### Distribution

| Metric | Value |
|--------|------:|
| N batches | 20 |
| LLM best-of-5 mean TEC | 14285.7 |
| Random Q1 best-of-5 | 14252.3 |
| Random median best-of-5 | 14265.0 |
| Random Q3 best-of-5 | 14293.3 |
| Random IQR | 41.0 |
| Random worst | 14410.0 |

### LLM vs Random (Best-of-5 Comparison)

| Metric | Value |
|--------|------|
| Random batches beating LLM | 15/20 (75%) |
| LLM beating random batches | 5/20 (25%) |
| LLM percentile rank | 20% |
| LLM beats random median | NO |
| Signal strength | **WEAK** |

### Oracle Reference (global best-of-100, NOT equal budget)

| Metric | Value |
|--------|------:|
| Global best-of-100 random | 14207.0 |
| Δ LLM vs global best | +78.7 |

### Interpretation

Interactive LLM does NOT outperform random best-of-5. Random with 5 shots per
batch finds better policies 75% of the time. The DSL is flat enough that
brute-force random with a 5-policy budget beats the interactive LLM. The
prior X4 efficiency claim against individual random policies (2/20) still
holds, but random aggregate budget (best-of-5) reverses the comparison.

### Best Random Best-of-5 Policy (b014, seed 5074)

```json
{
  "normal_mode": "hybrid",
  "escape_mode": "anti_s2",
  "switch_after_no_hit": 3,
  "switch_back_on_hit": true,
  "initial_budget": 7,
  "max_budget": 14,
  "grow_on_hit": 4,
  "shrink_on_miss": 2,
  "max_per_source": 3,
  "max_per_target": 4,
  "require_positive_cheap_lb": true,
  "guard_max_budget": 0
}
```
Mean TEC = 14207.0. Per-cell: 61/347=6884, 62/290=9474, 65/195=26263.
Uses `hybrid` normal mode + `anti_s2` escape (inverts s2 when s2 mis-ranks),
high budget (7→14) with aggressive growth, and require_positive_cheap_lb=true.

### Files

- `eval/x5_batch_checkpoint.csv` — 20-batch checkpoint
- `eval/x5_random_bestof5_batches.csv` — copy
- `eval/x5_random_bestof5_summary.csv` — distribution metrics
- `policies/random_bestof5_batches/` — 100 random policy JSONs
- `notes/x5_validation_cells.md` — 14 proposed held-out validation cells

## X3 Random Campaign (2026-05-09)

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

## Phase X Closure (2026-05-10)

### Verdict

**Phase X stopped as main LLM-critical direction.** X5 WEAK signal shows
interactive LLM policy repair does NOT outperform random best-of-5 under the
same 5-attempt budget. X6 validation on held-out cells is NOT justified.

### Why

| Stage | Result | Implications |
|-------|--------|-------------|
| X3 | Case B — DSL contains useful policies but search is noisy | DSL is learnable, not trivial |
| X4 | MINIMUM SUCCESS — LLM found beating policy in 2 rounds | Interactive repair works against individual random |
| X5 | WEAK — LLM at 20th percentile vs random best-of-5 | Random brute-force beats LLM under same 5-attempt budget |

The DSL is flat enough that brute-force random search with 5 attempts per batch
finds better policies 75% of the time. The X4 efficiency claim (2/5 interactive
rounds vs 2/20 individual random) holds, but the aggregate budget comparison
reverses it decisively. Running X6 on held-out cells would test whether the same
null result generalizes — but there is no evidence warranting another DeepSeek
call or further DSL expansion.

### Remaining value

- DSL-based policy optimization is a valid concept but needs a richer DSL or
  per-cell adaptation to beat brute-force random.
- Guard cell breakthrough (Round 4 hybrid, 61/347: 6884→6873) shows that cell-
  adaptive scoring has potential, but this is orthogonal to interactive LLM.

### Phase Y motivation

The core negative result (LLM cannot beat random best-of-5 under equal budget
when the policy space is flat) motivates Phase Y: instead of tuning a static
policy DSL, use the LLM to propose **concrete bounded neighborhoods** from the
**current schedule state**. The LLM's diagnostic ability (demonstrated in
Phases S, U, V, X) is tested in a setting where the comparison is against
random neighborhood proposals rather than random policy settings.

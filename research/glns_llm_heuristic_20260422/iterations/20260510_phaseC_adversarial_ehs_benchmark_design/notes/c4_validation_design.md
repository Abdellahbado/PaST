# C4 Validation Design

**Date**: 2026-05-11
**Status**: Design frozen before evaluation.

## Purpose

C4 is a **validation** exercise, not another design iteration.
The C3-L2 families (`families/llm_counter_sweep_families.json`) are frozen.
No DeepSeek calls. No family modification.

Goal: Test whether the C3-L2 STRONG result (LLM Call2 100% yield, Δfs_mean=19.5)
transfers to fresh instances beyond the 6 development instances.

## Protocol

- 12 families: 4 LLM Call2 + 4 human sweep + 4 random
- 5 fresh instances per family = 60 total instances
- EHS at 30s (short) and 90s (long) — same as C3-L2
- Caps: n ≤ 150, T ≤ 200
- Only new seeds (base offset + 100000); no C3/C3-L2 seed reuse

## Family Selection (frozen before evaluation)

### Selection rule
1. Prefer families with natural ranges n≤150, T≤200
2. If a range exceeds caps, generate capped instances; do not alter the family JSON
3. Do not select based on expected or observed validation performance
4. 4 families per arm

### LLM Call2 families (4/5 selected)
All 5 naturally fit within caps. Selected first 4 by order:
1. `hybrid_M1_tight_hetero_rates_firstkhat_dom` — first_khat_dominance
2. `hybrid_M2_tight_steprates_asghlock` — asgh_lock_in
3. `hybrid_M3_narrowrates_reinsert_starve` — res_reinsertion_starvation
4. `hybrid_M4_dualpeak_exploration_tension` — es_exploration_tension

Excluded: `hybrid_M5_looseeps_stepTOU_coverage_gap` (loose epsilon, weaker)

### Human sweep families (4/8 selected)
Selected for mechanism diversity (4 different mechanisms) and partial cap-fit:
1. `human_tight_epsilon` — first_khat_dominance (n fits, T needs cap to 200)
2. `human_loose_epsilon` — epsilon_skip (n fits, T needs cap to 200)
3. `human_mixed_job_sizes` — asgh_lock_in (n fits at ≤150, T needs cap to 200)
4. `human_many_machines_sparse` — front_coverage_gap (n and T both fit naturally)

Excluded: high/low price volatility (T > 200), many_small_jobs (n > 150),
few_machines_dense (n > 150, T > 400).

### Random families (4/8 selected)
None fit caps naturally. Selected first 4 for mechanism diversity:
1. `random_000` — load_imbalance
2. `random_001` — epsilon_skip
3. `random_002` — asgh_lock_in
4. `random_003` — es_exploration_tension

### Seed assignment
- Base seeds: LLM L2 = 80000+100000=180000, Human = 20000+100000=120000, Random = 10000+100000=110000
- Instance offsets: 0, 17, 34, 51, 68 (5 per family)

## Metrics

### Per-instance metrics
- `front_size_short`, `front_size_long`
- `delta_front_size` = fs_long - fs_short
- `deepest_cmax_short`, `deepest_cmax_long`
- `delta_cmax` = dc_short - dc_long (positive = cmax improved)
- `runtime_s_short`, `runtime_s_long`
- `timeout_short`, `timeout_long`
- `feasible_ehs_short`, `feasible_ehs_long`
- `high_yield` (see below)
- `mechanism_confirmed` (see below)
- `nontrivial_mechanism_high_yield` (see below)

### High-yield definition
An instance is high-yield if ANY of:
1. `delta_front_size >= 2`
2. `front_size_short <= 1` AND `front_size_long >= 3`
3. `delta_cmax >= 2` (deepest cmax improved by ≥2)
4. short budget timed out AND `front_size_short <= 1` (slow — couldn't complete first khat)

### Mechanism confirmation rules (MUST be explicit and operational)

#### first_khat_dominance
Confirmed if: `timeout_short == True` AND `front_size_short <= 3` AND `runtime_s_short >= 25`
Rationale: First khat dominates; EHS times out before completing it.

#### asgh_lock_in
Confirmed if: `delta_front_size >= 4` AND `runtime_s_short > 10` AND NOT `timeout_short`
Rationale: A-SGH retention creates structure that forces re-optimization across khats.

#### res_reinsertion_starvation
Confirmed if: `runtime_s_short < 15` AND `front_size_short >= 3` AND `delta_front_size >= 4`
Rationale: Fast convergence (no reinsertion overhead), many khats, large front growth.

#### es_exploration_tension
Confirmed if: `delta_front_size >= 4` AND `delta_cmax >= 2`
Rationale: ES explores many local optima, producing diverse front points and cmax improvement.

#### front_coverage_gap
Confirmed if: `front_size_short >= 3` AND `delta_front_size >= 3`
Rationale: Already has a front from first khat, but subsequent khats add significantly more.

#### epsilon_skip
Confirmed if: `front_size_short >= 3` AND `delta_front_size <= 2`
Rationale: Coarse epsilon gets many points early, few added later (or vice versa).

#### load_imbalance
Confirmed if: `deepest_cmax_long < horizon_T * 0.5` AND `delta_front_size <= 3`
Rationale: Load is easy to balance (good cmax), but front doesn't grow much.

#### short_budget_pressure
Confirmed if: `timeout_short == True` AND `front_size_short <= 1`
Rationale: Too many jobs/n for EHS to complete even the first khat.

#### Non-match
If none of the above patterns match the family's declared mechanism,
`mechanism_confirmed = False`.

### Nontrivial mechanism-confirmed high-yield definition
`high_yield == True` AND `mechanism_confirmed == True` AND
NOT explainable solely by generic uniform small-job front growth.

Operational rule:
- If `high_yield` AND `mechanism_confirmed` AND the family has BOTH:
  - An explicit `mechanism_layer` field (present in LLM L2 families, absent in human/random)
  - OR processing_time_distribution is NOT `uniform` with `low <= 2` (i.e., not the simple fine-grained sweep)
  - OR machine_rate_distribution is NOT simple uniform with wide spread
- Then count as nontrivial.

Simplified: A family counts as nontrivial if it has a `mechanism_layer` field
(LLM L2 families) OR if its processing is bimodal/exponential/normal.
Pure uniform(1,10) families are always "trivial" (explainable by sweep alone).

## Arm-level aggregation

| Metric | Definition |
|--------|-----------|
| generation_success_rate | fraction of attempted instances that are feasible |
| evaluable_rate | fraction of feasible instances that completed both short+long |
| high_yield_rate | fraction of evaluable instances that are high-yield |
| mean_delta_fs | mean of delta_front_size for evaluable instances |
| median_delta_fs | median of delta_front_size |
| mechanism_confirmed_rate | fraction of evaluable instances with mechanism_confirmed=True |
| nontrivial_mc_hy_rate | fraction of evaluable instances with nontrivial_mechanism_high_yield=True |
| mean_runtime_short | mean of runtime_s for short budget |
| mean_runtime_long | mean of runtime_s for long budget |
| timeout_rate_short | fraction of short runs that timed out |
| timeout_rate_long | fraction of long runs that timed out |

## Decision gate

| Gate | Condition |
|------|-----------|
| STRONG | LLM L2 ≥ human high-yield rate AND LLM L2 > human on nontrivial mechanism-confirmed high-yield rate |
| MODERATE | LLM L2 ties human on high-yield AND mechanism yield, with stronger qualitative mechanism diversity |
| WEAK | LLM L2 beats random but not human |
| FAIL | LLM L2 does not beat random OR clearly loses to human |

## Decision questions (to answer in c4_validation_decision.md)

1. Did L2 transfer beyond C3-L2?
2. Did L2 beat random?
3. Did L2 beat or tie human sweep?
4. Is the LLM advantage generation quality, adversarial yield, mechanism specificity, or all three?
5. Is this strong enough to proceed to a final thesis experiment?

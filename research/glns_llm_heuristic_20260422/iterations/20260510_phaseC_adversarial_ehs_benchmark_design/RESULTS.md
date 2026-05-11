# Phase C Results

## C0: Protocol Definition — COMPLETED 2026-05-10

## C1: Family Schema — COMPLETED 2026-05-10

## C2: Baseline Generators and LLM Prompt — COMPLETED 2026-05-10

### C2A: Random Family Generator ✅
- 8 families, all validated. `families/random_families.json`

### C2B: Human Sweep Families ✅
- 8 families, all validated. `families/human_sweep_families.json`

### C2C: Family Validation ✅
- Script validates required fields, ranges, rejection conditions.

### C2D: LLM Prompt ✅
- `prompts/call1_llm_family_designer.md`

## C3: Smoke Pilot — COMPLETED (TRUNCATED) 2026-05-10

### C3A: DeepSeek Call ✅
- 8/8 families pass schema validation. All target distinct mechanisms.
- LLM families: first_khat_dominance_giant, asgh_trajectory_conflict, reinsertion_starvation_tight_epsilon, es_local_optima_trap_extreme_rates, front_coverage_gap_step_TOU, short_budget_critical_size, load_imbalance_narrow_jobs_wide_rates, epsilon_skip_narrow_cost_steps
- Cost: ~$0.01
- Artifacts: `responses/call1_family_designer_raw.md`, `responses/call1_family_designer_metadata.json`, `families/llm_families_raw.json`, `families/llm_families.json`

### C3B: Family Selection ✅
- 2 LLM: first_khat_dominance_giant, asgh_trajectory_conflict
- 2 Random: random_000, random_001
- 2 Human: human_tight_epsilon, human_loose_epsilon

### C3C: Instance Generation ✅
- 18 instances generated (3 per family), all feasible
- LLM giant instances: n=813-843, m=36-44, T=753-853
- LLM moderate instances: n=82-119, m=10-12, T=213-240
- Random instances: n=162-212, m=6-24, T=425-777
- Human instances: n=62-110, m=8-10, T=160-245

### C3D: EHS Evaluation ⚠️ TRUNCATED
- **Blocking issue**: `run_ehs()` only checks `time_limit_seconds` between khat iterations. SGH construction at khat=T can take 5× the time budget.
- 8/36 EHS runs completed (22%). Remaining truncated.
- Evaluated: 3 LLM moderate instances (6 runs) + 1 random instance (2 runs)
- LLM giant instances (n=800+) marked "too_large" — cannot complete even first khat
- Human instances: not evaluated

### C3E: Comparison ✅ (indicative only)

| Arm | Instances | Evaluated | Too Large | Skipped | High-Yield | Rate |
|-----|----------|-----------|-----------|---------|------------|------|
| llm | 6 | 3 | 3 | 0 | 6 | 100% |
| random | 6 | 1 | 0 | 5 | 1 | 17% |
| human | 6 | 0 | 0 | 6 | 0 | 0% |

**LLM yield 100% (6/6)** vs random 17% (1/6). But data is too sparse for confident conclusion.

Key findings:
1. LLM correctly identified `first_khat_dominance` with giant instances (n=800+) — unevaluable = adversarial success
2. LLM `asgh_trajectory_conflict` instances show clear budget scaling (front grows 1→3, 5→8, 8→12)
3. Random `random_000` with T=737 took 155s for 1 front point (5× budget)
4. Human not evaluated due to time constraints
5. **Infrastructure bug found**: `run_ehs` doesn't enforce time limit during SGH construction

### Gate: INCONCLUSIVE
- Only 22% of EHS runs completed — insufficient data for arm comparison
- EHS time-limit enforcement needs fixing before rerun
- LLM giant family design is technically successful but the success is a builder bug, not an EHS weakness discovery

## C4-C5: Not started

# Phase C Results

## C0: Protocol Definition — COMPLETED 2026-05-10

Protocol defined in EXPERIMENT_PROTOCOL.md. Four phases:
C0 problem framing, C1 schema, C2 generators, C3 smoke pilot, C4 gate.

## C1: Family Schema — COMPLETED 2026-05-10

Schema defined in `families/family_schema.json`. Includes all required fields,
sanity constraints, and validity/rejection conditions.

## C2: Baseline Generators and LLM Prompt — COMPLETED 2026-05-10

### C2A: Random Family Generator ✅
- Script: `scripts/phaseC_adversarial_family_generation.py --generate-random-families`
- 8 families generated with seed=42
- All families pass validation (0 errors, 0 warnings)
- Saved: `families/random_families.json`

### C2B: Human Sweep Families ✅
- Script: `scripts/phaseC_adversarial_family_generation.py --generate-human-sweep-families`
- 8 designed families:
  - `human_tight_epsilon` → first_khat_dominance
  - `human_loose_epsilon` → epsilon_skip
  - `human_high_price_volatility` → es_exploration_tension
  - `human_low_price_volatility` → res_reinsertion_starvation
  - `human_many_small_jobs` → short_budget_pressure
  - `human_mixed_job_sizes` → asgh_lock_in
  - `human_many_machines_sparse` → front_coverage_gap
  - `human_few_machines_dense` → load_imbalance
- All families pass validation (0 errors, 0 warnings)
- Saved: `families/human_sweep_families.json`

### C2C: Family Validation ✅
- Script: `scripts/phaseC_adversarial_family_generation.py --validate-family-file`
- Validates: required fields, legal ranges, rejection conditions present,
  no degenerate pricing, feasible n/m/T ratios
- Both random and human sweep files pass.

### C2D: LLM Prompt ✅
- `prompts/call1_llm_family_designer.md`
- Includes: problem definition, EHS pipeline description, B6 closure evidence,
  failure mechanisms M1-M8, schema documentation, bad examples to avoid,
  good example, output format instructions.
- DeepSeek NOT called yet.

## C3-C4: Not started

Pending: first DeepSeek family design call, then smoke pilot.

# C3-L2: DeepSeek Counter-Sweep Family Designer — Call 2

You are an expert in combinatorial optimization and adversarial benchmark design.

## Context

In Call 1, you designed 8 BPMSTP instance families targeting EHS failure mechanisms. These were evaluated against human (parameter sweep) and random baselines under C3-Regular protocol (n≤150, T≤200, 30s/90s budgets).

**Your families lost to the human sweep baseline.**

The reason is NOT that your mechanisms were wrong — your A-SGH lock-in and ES exploration-tension targeting was correct. The human sweep won because its simple design (uniform p_j ∈ [1,10] + tight/loose epsilon) is structurally optimized for the front-size-growth yield metric: small uniform jobs allow many rapid khat iterations, producing large Δfs (+7 to +33) without requiring any adversarial mechanism targeting.

## Your Task

Design 8 new second-generation families that **combine** human-sweep front-growth ingredients with your mechanism-specific adversarial targeting.

**Goal**: Beat human sweep on front-size-growth yield (currently 100%) while preserving mechanism specificity.

## Winning Strategy (from analysis)

The human sweep formula:
1. Uniform small processing times → cheap SGH construction → many khats
2. Tight/loose epsilon → many descent steps OR rapid Cmax drop
3. Moderate n/m → assignment diversity without construction cost
4. Simple TOU → energetic tradeoffs without fragile schedules

Your lever: ADD a mechanism-specific stress layer on top of those ingredients.

## Hybrid Mechanism Ideas

- **M1 (first_khat_dominance)**: tight epsilon + heterogeneous machine rates → SGH construction is slow at each tight step, first few khats dominate
- **M2 (asgh_lock_in)**: uniform p_j + step machine rates + tight epsilon → A-SGH retains cheap-machine assignments that break at deeper khats
- **M3 (res_reinsertion_starvation)**: uniform small p_j + narrow rate spread → reinsertion would help but never fires because ES is too efficient
- **M4 (es_exploration_tension)**: uniform p_j + sharp dual-peak TOU → ES finds local energy minima in peak windows, cannot escape to better global assignments
- **M5 (front_coverage_gap)**: loose epsilon + step-function TOU → EHS skips intermediate Cmax values between price steps
- **M6 (short_budget_pressure)**: uniform p_j + high m (parallelism) → many machines create assignment explosion, SGH construction cost dominates
- **M7 (load_imbalance)**: uniform p_j + wide machine-rate spread (5×+) → SGH concentrates on cheap machines, R-ES cannot repair load balance
- **M8 (epsilon_skip)**: loose epsilon + narrow price differences → epsilon spacing is coarser than meaningful energy-rate differences

## Requirements

1. **Every family must target a named mechanism (M1-M8)** AND include a front-growth ingredient
2. **Explain WHY it beats human sweep** (what mechanism layer adds beyond pure epsilon sweep)
3. **Explain WHY it is NOT a clone** of human_tight_epsilon / human_loose_epsilon
4. **Stay within caps**: n ≤ 150, T ≤ 200
5. **Be feasible**: expected sum(p_j) ≤ m·T_min
6. **Expected front growth**: predict Δfs (short→long) for each family
7. **No pure clones**: must differ from human sweep in at least processing distribution, TOU profile, OR machine rates

## Output Format

Output exactly 8 family specs. Use this wrapper:

```json
{
  "generator": "deepseek_v4_pro",
  "generator_call": "call2_counter_sweep",
  "generator_description": "Second-generation LLM-designed adversarial families — counter-sweep design combining human-sweep front-growth ingredients with mechanism-specific EHS stress.",
  "n_families": 8,
  "families": [
    {
      "family_name": "string (unique, descriptive)",
      "hypothesis": "string (why beats human sweep + why not a clone)",
      "description": "string",
      "mechanism_layer": "M1-M8 plus explanation",
      "sweep_layer": "what gives front-size growth",
      "expected_delta_fs": 10,
      "n_jobs_range": {"min": 40, "max": 150},
      "m_machines_range": {"min": 6, "max": 15},
      "horizon_T_range": {"min": 120, "max": 200},
      "processing_time_distribution": {"type": "uniform|bimodal|...", "params": {}},
      "machine_rate_distribution": {"type": "uniform|step|exponential", "params": {}},
      "TOU_price_profile_type": "single_peak|dual_peak|step_function|high_variance|low_variance|...",
      "TOU_price_profile_params": {},
      "epsilon_regime": "tight|medium|loose",
      "expected_EHS_failure_mechanism": "M1-M8",
      "expected_EHS_failure_mechanism_evidence": "why mechanism + sweep > pure sweep",
      "generated_instances_count": 8,
      "validity_constraints": {"feasibility_guarantee": true, "min_total_work": 100, "n_per_machine_min": 3},
      "rejection_conditions": ["sum(p_j) > m*T", "all e[h] equal", "all ct[t] equal", "n < 2*m"],
      "seed_behavior": {"base_seed": 80000, "expected_seed_variance": "medium"}
    }
  ]
}
```

## Constraints (enforced)

- n_min ≥ 40, n_max ≤ 150
- m_min ≥ 6, m_max ≤ 15
- T_min ≥ 80, T_max ≤ 200
- At least 2 distinct machine rates
- TOU NOT all flat
- Processing times NOT all equal (variance required)
- No all-zero prices, no negative values
- generated_instances_count ≥ 3

## What NOT to do

- Do NOT propose `n=150`, `T=200`, "uniform p_j=(1,10)", "tight epsilon" as a complete family — that IS the human sweep
- Do NOT make every family use the same processing distribution
- Do NOT say "make it larger" as the mechanism
- Do NOT reference instance IDs from published benchmarks
- Do NOT propose solver modifications

## Goal Summary

> Design 8 families where each one says: "This family has the front-growth properties of human_tight_epsilon/human_loose_epsilon plus this specific EHS mechanism stress. Here is why the mechanism layer adds diagnostic value, and here is why it is not just a tight/loose epsilon clone."

Output valid JSON only. No markdown, no explanation outside the JSON wrapper.

# C3-L2 DeepSeek Counter-Sweep Evidence Pack

**Date**: 2026-05-10
**For**: DeepSeek V4 Pro — Call 2 counter-sweep family design

## 1. C3-Regular Repaired Results

18 instances (6 per arm), n≤150, T≤200, 30s/90s budgets.
All instances feasible, all 36 EHS runs completed.

### High-Yield Definition
An instance is HIGH-YIELD if ANY of:
- front_size_growth (short→long) ≥ 2 points
- short=0 or ≤1 and long ≥3
- deepest Cmax improves ≥2 from short to long
- short runtime > 1.5× budget producing ≤1 point (very slow single point)

### Results Table

| Arm | Yield Rate | Δfs Range | Mechanism |
|-----|:---:|--------|-----------|
| **human** | **6/6 (100%)** | +7 to +33 | tight/loose epsilon sweep |
| llm | 5/6 (83%) | +3 to +6 | asgh_lock_in, es_exploration_tension |
| random | 4/6 (67%) | +5 to +25 | mixed (small T instances) |

### Per-Instance Data (all arms)

```
arm    family             mech                          n   m   T   fs(short→long)  cmax(short→long)  yield
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
llm    asgh_trajectory    asgh_lock_in                  85   6 200  1→5            200→196           HIGH +4
llm    asgh_trajectory    asgh_lock_in                  91  12 200  2→8            199→193           HIGH +6
llm    asgh_trajectory    asgh_lock_in                  99  12 200  4→7            197→194           HIGH +3
llm    es_local_optima    es_exploration_tension        98   8 200  1→1            200→200           low (saturated)
llm    es_local_optima    es_exploration_tension        93   8 200  2→5            199→196           HIGH +3
llm    es_local_optima    es_exploration_tension        92   8 200  1→2            200→199           HIGH (slow)
human  loose_epsilon      epsilon_skip                  98  10 200  3→11           198→190           HIGH +8
human  loose_epsilon      epsilon_skip                  63   8 159 20→53           140→107           HIGH +33
human  loose_epsilon      epsilon_skip                  88  11 200  4→13           197→188           HIGH +9
human  tight_epsilon      first_khat_dominance          99   8 200  4→17           197→184           HIGH +13
human  tight_epsilon      first_khat_dominance          87   8 196  1→8            196→189           HIGH +7
human  tight_epsilon      first_khat_dominance          81  10 174  6→27           169→148           HIGH +21
random repair_000         short_budget_pressure        110  10  98  3→8             96→91            HIGH +5
random repair_000         short_budget_pressure         38   8 102 56→77            37→16            HIGH +21
random repair_000         short_budget_pressure        105   6 102  1→1            102→102           HIGH (slow)
random repair_001         load_imbalance                21   9 109 27→27            20→20            low (saturated)
random repair_001         load_imbalance                55   8  99 24→49            76→51            HIGH +25
random repair_001         load_imbalance                44   9  86 51→51            36→36            low (saturated)
```

## 2. Why Human Sweep Won

The `human_tight_epsilon` and `human_loose_epsilon` families dominated because:

1. **Uniform processing times** p_j ∈ [1,10]: Every job is small and schedules quickly. SGH construction is O(n·m·T) where each job's enumeration is cheap. Many jobs can be placed, producing diverse khat-level schedules.

2. **Tight/loose epsilon**: The epsilon step controls khat count. Tight epsilon → many small Cmax decrements → many khat iterations → many front points. Loose epsilon → fewer but larger steps → rapid Cmax descent.

3. **Medium n/m ratio** (n=60-120, m=8-12): Enough jobs for assignment diversity, enough machines for optimization, but not so many that SGH dominates budget.

4. **Simple TOU profiles** (single_peak with moderate multiplier 2.0): Creates energetic tradeoffs without making construction fragile.

**Key insight**: The current yield metric rewards front-size growth, and the simplest way to get front-size growth is uniform small jobs + many khat iterations. The human sweep is structurally optimized for this metric, not because it discovers EHS weaknesses.

## 3. LLM Successes (Call 1)

The LLM correctly identified and stressed two EHS mechanisms:

### asgh_trajectory_conflict (3/3 HIGH, Δfs +3 to +6)
- Bimodal job sizes (small 2-5 + large 12-25, 30% small)
- Step machine rates (60% cheap, 40% expensive)
- A-SGH retention locks early assignment decisions; releasing jobs is expensive
- Mechanism confirmed: front growth is mechanism-driven, not just size-driven

### es_local_optima_trap_extreme_rates (2/3 HIGH)
- Truncated normal or exponential processing (mean ~10, variance)
- Wide machine-rate heterogeneity (up to 10× spread)
- ES finds local improvements that prevent escape to better regions
- One instance saturated (all jobs assigned to single cheap machine, no escape path)

## 4. LLM Failure (Call 1)

### es_local_optima_trap_extreme_rates_50000 (1/3 LOW)
- n=98, m=8, T=200
- Extreme machine-rate heterogeneity (7.8× spread)
- SGH assigned ALL jobs to the single cheapest machine
- Result: Cmax=T=200, one front point, no descent possible
- Root cause: extreme rate heterogeneity + no mechanism to force load balancing

## 5. The Challenge — Beat Human Sweep

Your task: design 8 new adversarial families that:

**Must match or beat human sweep on front-size growth** (target: Δfs ≥ 8 on most instances) while preserving mechanism-specific EHS stress.

### Winning strategy (from human sweep analysis):
1. Use processing times that allow rapid SGH construction (uniform small, or bimodal with mostly-small)
2. Use epsilon that creates many khat iterations OR rapid Cmax descent
3. Use TOU profiles that create energetic tradeoffs but don't break construction
4. BUT: add a mechanism-specific adversarial layer beyond pure epsilon sweep

### Suggested hybrid mechanisms:
- uniform p_j ∈ [2,8] + step-function TOU with sharp boundaries → ES exploration trap in price windows
- tight epsilon + machine-rate clusters → A-SGH locks in cheap-machine assignments that break at deeper khats
- loose epsilon + dual-peak TOU → skipped intermediate Pareto points between peaks
- bimodal (p_j small 1-4 + medium 8-14) + heterogeneous rates → trajectory conflict like asgh_trajectory but with faster construction
- uniform p_j ∈ [1,10] + extreme dual-peak TOU (multiplier 4+) → energy cost creates local optima that ES cannot escape

### Constraints:
- n ≤ 150 (hard cap)
- T ≤ 200 (hard cap)
- Instances must be feasible (sum(p) ≤ m·T)
- Must evaluate within 30s/90s budget
- No pure clone of human_tight_epsilon / human_loose_epsilon
- Must name a specific EHS failure mechanism
- Must explain why it beats human sweep AND why it's not just a clone

### Forbidden:
- n > 150
- T > 200
- Giant instances or size-only hardness
- Raw instance IDs
- Families that only say "tight/loose epsilon" without additional mechanism
- "Make it bigger" as strategy

## 6. Expected Per-Family Output

For each of 8 families:

```json
{
  "family_name": "string",
  "hypothesis": "why it beats human sweep + why not a clone",
  "mechanism_layer": "specific EHS mechanism targeted",
  "sweep_layer": "what gives it front-size growth (the human-sweep ingredient)",
  "expected_delta_fs": 10,
  "n_jobs_range": {"min": int, "max": int},
  "m_machines_range": {"min": int, "max": int},
  "horizon_T_range": {"min": int, "max": int},
  "processing_time_distribution": {"type": "...", "params": {...}},
  "machine_rate_distribution": {"type": "...", "params": {...}},
  "TOU_price_profile_type": "...",
  "TOU_price_profile_params": {...},
  "epsilon_regime": "tight|medium|loose",
  "expected_EHS_failure_mechanism": "one of M1-M8",
  "expected_EHS_failure_mechanism_evidence": "why this mechanism + sweep layer > pure sweep",
  "generated_instances_count": 8,
  "validity_constraints": {"feasibility_guarantee": true, "min_total_work": int, "n_per_machine_min": int},
  "rejection_conditions": ["sum(p_j) > m*T", "all e[h] equal", "all ct[t] equal", "n < 2*m"],
  "seed_behavior": {"base_seed": int, "expected_seed_variance": "medium"}
}
```

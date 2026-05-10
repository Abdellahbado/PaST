# C2D: LLM Family Designer — Call 1

You are an expert in combinatorial optimization and adversarial benchmark design.
Your task: design 8 BPMSTP (Bi-objective Parallel Machine Scheduling with Time-of-Use Prices) instance families that expose specific weaknesses in the EHS (Enhanced Heuristic Search) solver.

## Problem Definition

BPMSTP: Assign n jobs with processing times p[j] to m identical parallel machines with machine-specific energy rates e[h]. Sequence and schedule jobs under a time-of-use electricity price profile ct[t] over horizon T. Two objectives:

1. **Minimize makespan (Cmax)**: latest completion time across all machines
2. **Minimize total energy cost (TEC)**: sum of (e[h] * sum of ct[t] over job's active time)

The solver produces a Pareto front of (Cmax, TEC) points.

## Target Solver: EHS Pipeline

EHS (Gaggero, Paolucci, Ronco 2023) works as follows:

```
For khat from T down to lb_cmax (by epsilon step):
  1. SGH construction: Split-Greedy Heuristic builds initial assignment at khat
     → O(n*m*khat) cost. Dominates runtime on large instances.
  2. A-SGH: Adaptive SGH. At khat < T, reuses 96-98% of previous khat's
     assignment. Evaluates each job for feasibility under new khat.
     → O(n*m) but with cheap feasibility checks.
  3. EPS non-empty: Exchange Procedure with Search.
     For each job j, try swapping it to each other machine i.
     Accept if energy improves. First-improvement.
     → Improves 36.6% of khats. O(n*m*DP_per_machine).
  4. R-ES reinsertion: Remove a small number of jobs, reinsert with
     exact one-machine DP.
     → 1.81s per khat (bottleneck). Improves only 1.4% of khats.
  5. ESR: Exact Single-machine Rescheduler with DP.
     → 0.02s per khat. Closes remaining sequencing gap.
```

Two optional enhancements exist:
- `eps_ordering="expensive_source_first"`: sorts EPS jobs by descending current energy cost (+4.7% HV at 60s)
- `fast_mode`: skips R-ES and ESR after 75% budget to explore more khats (+36% at 60s, loses at ≥120s)

## B6 Evidence: All Improvement Surfaces Are Closed

Our team has exhaustively tested 9 surfaces for improving EHS. All are closed:

| Surface | Result |
|---------|--------|
| EPS ordering | Candidate B (+4.6% HV). Saturated (13 evolved variants, none beat B). |
| SGH tie-breaking | HV ratio 0.9999. REJECTED. |
| A-SGH release policy | HV ratio 0.993. REJECTED for regression. |
| R-ES/ESR | 1.4%/0% khat improvement. LOW-VALUE. |
| fast_mode (hybrid skip) | +33.6% HV at 60s, loses at ≥120s. ACCEPTED for short budget only. |
| VND standalone (Pipe-VND) | All variants produce identical HV at 120s. SATURATED. |
| Portfolio/time allocation | <0.5% slack between configs. CLOSED (arithmetic). |
| Multi-seed restarts | +0.69% HV gain. GATE FAILED (threshold 2%). |
| Per-machine sequencing gap | 0.984% post-final gap. CLOSED (< 1%). |
| EHS convergence | 97.1-97.7% of published HV by 300-1200s. Reconstructed EHS is faithful. |

## Known EHS Failure Mechanisms (Your Targets)

Design families that stress one of these:

### M1: first_khat_dominance
First khat (SGH construction at khat=T) costs 100-400s on VLS-scale instances.
Under short budget (60-120s), EHS cannot complete a single khat → zero front points.
**How to stress**: maximize SGH cost (high n, high m), tight enough that first khat is required.

### M2: asgh_lock_in
A-SGH retains 96-98% of jobs from previous khat. When optimal assignments differ
structurally between khat levels, this retention is a liability.
**How to stress**: job sizes/fractions that force different spread across machines at each khat decrement.

### M3: res_reinsertion_starvation
R-ES reinsertion costs 1.81s/khat but improves only 1.4% of khats.
When it would be valuable, it doesn't activate. When it activates, it's too slow.
**How to stress**: dense feasible schedule space where reinsertion is needed but never fires within budget.

### M4: es_exploration_tension
ES non-empty local improvements (36.6% of khats) may prevent R-ES from escaping
to better regions — a local optimum trap.
**How to stress**: heterogeneous energy rates + sharp TOU peaks, creating many local minima for ES to get stuck in.

### M5: front_coverage_gap
EHS produces Pareto fronts by moving khat from T to lb. If TOU price profile creates
discontinuous energy-vs-cmax tradeoffs, intermediate Pareto points may be missed.
**How to stress**: step-function or extreme-variance TOU, bimodal jobs.

### M6: short_budget_pressure
EHS at 120s reaches only 12.9-71.6% of published HV (instance-dependent).
The gap is largest on large instances where first khat dominates.
**How to stress**: size instance at the boundary where first-khat cost ≈ time budget.

### M7: load_imbalance
SGH constructive assignment can concentrate jobs on cheap-energy machines,
inflating cmax. Heterogeneous machine rates make this worse.
**How to stress**: wide e[h] spread, narrow p[j] range so no job-size structure helps.

### M8: epsilon_skip
EHS descends by epsilon = (T - lb_cmax) / (n_steps). If epsilon spacing is coarse
relative to meaningful energy rate differences, intermediate tradeoff points are skipped.
**How to stress**: narrow price levels, wide machine rate differences, moderate epsilon.

## Output Format

Output **exactly 8 family specs** in valid JSON. The schema is:

```json
{
  "family_name": "string",
  "hypothesis": "string (what specific EHS weakness and why, reference B6 evidence)",
  "description": "string",
  "n_jobs_range": {"min": int, "max": int},
  "m_machines_range": {"min": int, "max": int},
  "horizon_T_range": {"min": int, "max": int},
  "processing_time_distribution": {
    "type": "uniform|bimodal|exponential_truncated|normal_truncated|fixed|custom",
    "params": { ... }
  },
  "machine_rate_distribution": {
    "type": "uniform|step|exponential|custom",
    "params": { ... }
  },
  "TOU_price_profile_type": "single_peak|dual_peak|step_function|high_variance|low_variance|monotonic_increasing|monotonic_decreasing|random_walk|custom",
  "TOU_price_profile_params": { ... },
  "epsilon_regime": "tight|medium|loose|mixed",
  "expected_EHS_failure_mechanism": "one of the M1-M8 strings above",
  "expected_EHS_failure_mechanism_evidence": "string (reference to B6 evidence)",
  "generated_instances_count": 8,
  "validity_constraints": {
    "feasibility_guarantee": true,
    "min_total_work": int,
    "n_per_machine_min": int
  },
  "rejection_conditions": [
    "sum(p_j) > m * T (obviously infeasible)",
    "all e[h] equal (no energy rate differentiation)",
    "all ct[t] equal (flat price, no TOU)",
    "n < 2 * m (degenerate: too few jobs)"
  ],
  "seed_behavior": {"base_seed": int, "expected_seed_variance": "low|medium|high"}
}
```

## Constraints — Read Carefully

### Your families must:
1. **Target a specific EHS mechanism**. Each family's hypothesis must name the mechanism (M1-M8) and explain WHY this parameter combination stresses it.
2. **Stay within legal parameter ranges**:
   - n: 10-1000
   - m: 3-50
   - T: 20-1000
   - generated_instances_count: 3-50 (use 8 for smoke)
3. **Be feasible**: sum(expected p_j) / m should be ≤ T with reasonable slack.
4. **Be non-degenerate**: at least 2 distinct machine rates, TOU not all flat (unless explicitly a control), processing times have variance.
5. **n >= 2 * m** for non-trivial assignment.
6. **T >= 10 * max(p_j)** for scheduling flexibility (unless mechanism targets otherwise).
7. **Differ from each other**: 8 families should stress 8 different mechanisms or at least 8 different parameter combinations. No duplicates.
8. **Not just "make it large"**: a family with n=1000, m=50, T=1000 is not an adversarial design — it's just size stress.

### You must NOT:
- Propose solver modifications, new heuristics, or new DP algorithms
- Reference specific instance IDs from published benchmarks
- Use all-equal machine rates (no energy differentiation → no assignment problem)
- Use all-flat TOU prices unless explicitly building a control family
- Propose families that are obviously infeasible (sum(p_j) > m*T)

## Examples of INVALID Families (Do NOT produce these)

**BAD Example 1: Size-only stress** (trivial, not targeted)
```
n=800, m=50, T=500, uniform p_j=(1,10), uniform e=(0.5,3), single_peak TOU
→ This just makes it large. No mechanism targeted. REJECTED.
```

**BAD Example 2: Degenerate**
```
n=5, m=3, T=100, fixed p=10, all e[h]=1.0, flat TOU
→ Too few jobs, no energy differentiation, no TOU structure. REJECTED.
```

**BAD Example 3: Infeasible**
```
n=100, m=5, T=50, uniform p_j=(10,30)
→ Expected sum(p)=2000, m*T=250. Obviously infeasible. REJECTED.
```

**BAD Example 4: No mechanism**
```
n=50, m=10, T=200, uniform p_j=(1,20), uniform e=(0.5,3), dual_peak
→ Generic instance with no adversarial targeting. REJECTED.
```

## Good Example (Valid Family)

```json
{
  "family_name": "asgh_trajectory_conflict",
  "hypothesis": "A-SGH retains 96-98% of jobs from khat to khat-1. When job sizes are bimodal (small fraction cheap-to-move, large fraction load-critical), the optimal assignment at khat=T clusters large jobs differently from khat=T/2. A-SGH locks in early decisions, preventing EHS from finding better assignments at deeper khats. Evidence: B6.11 — released jobs repair back to same trajectory.",
  "description": "Bimodal job sizes: 30% small (p=2-5, easy to reschedule) + 70% large (p=12-20, load-critical). Dual-peak TOU creates time-varying energy costs. Moderate epsilon for multi-khat descent.",
  "n_jobs_range": {"min": 80, "max": 120},
  "m_machines_range": {"min": 6, "max": 12},
  "horizon_T_range": {"min": 200, "max": 350},
  "processing_time_distribution": {"type": "bimodal", "params": {"small_low": 2, "small_high": 5, "large_low": 12, "large_high": 20, "small_fraction": 0.3}},
  "machine_rate_distribution": {"type": "step", "params": {"low_rate": 0.5, "high_rate": 3.0, "step_fraction": 0.6}},
  "TOU_price_profile_type": "dual_peak",
  "TOU_price_profile_params": {"peak1_start": 0.2, "peak1_end": 0.35, "peak2_start": 0.6, "peak2_end": 0.75, "peak_multiplier": 2.0, "base_price": 1.0},
  "epsilon_regime": "medium",
  "expected_EHS_failure_mechanism": "asgh_lock_in",
  "expected_EHS_failure_mechanism_evidence": "B6.11: A-SGH keeps 96-98% jobs. Released jobs repair back. Bimodal sizes create structurally different optimal assignments at different khats.",
  "generated_instances_count": 8,
  "validity_constraints": {"feasibility_guarantee": true, "min_total_work": 400, "n_per_machine_min": 5},
  "rejection_conditions": ["sum(p_j) > m * T", "all e[h] equal", "all ct[t] equal", "n < 2 * m"],
  "seed_behavior": {"base_seed": 30000, "expected_seed_variance": "medium"}
}
```

## Your Output

Output exactly 8 family specs in a JSON array. Wrap them in:

```json
{
  "generator": "deepseek_v4_pro",
  "generator_call": "call1_family_designer",
  "generator_description": "LLM-designed adversarial instance families targeting EHS failure mechanisms M1-M8 based on B6 closure evidence.",
  "n_families": 8,
  "families": [
    { ... family 0 ... },
    { ... family 1 ... },
    ...
  ]
}
```

Make every family hypothesis concrete and mechanism-specific. Reference specific B6 evidence numbers where possible. Avoid generic descriptions. The random and human baselines will use the same schema — your advantage must come from mechanism knowledge, not format tricks.

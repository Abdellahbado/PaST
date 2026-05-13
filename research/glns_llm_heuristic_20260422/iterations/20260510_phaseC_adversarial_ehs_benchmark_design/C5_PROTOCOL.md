# C5 Protocol: Robust Multi-Budget Adversarial Validation

**Date**: 2026-05-13
**Purpose**: Test whether LLM-designed instance families remain difficult for EHS beyond the trivial 30s short-budget regime.

## Core Question

Do LLM-designed families expose persistent EHS stress mechanisms under multiple time budgets, or is their difficulty only a short-budget artifact?

## Arms (frozen families)

All families frozen before C5 — no tuning based on C5 results.

### Main external comparison

| Arm | Families | Instances |
|-----|----------|-----------|
| **LLM Call2** | hybrid_M2_tight_steprates_asghlock, hybrid_M1_tight_hetero_rates_firstkhat_dom | 3 each = 6 |
| **Literature** | wang_mls_generator, anghinolfi_vls_capped | 3 each = 6 |
| **Random** | random_004 (step rates, medium epsilon), random_005 (step rates, tight epsilon) | 3 each = 6 |

### Internal control (reported separately)

| Arm | Families | Instances |
|-----|----------|-----------|
| agent_manual_sweep | human_tight_epsilon, human_mixed_job_sizes | 3 each = 6 |

### Simple stress baseline (non-LLM, for reference)

| Arm | Description |
|-----|-------------|
| simple_stress | Tight epsilon + small uniform jobs (p_j~U[1,4]) + step machine rates (0.5/3.0) + single-peak TOU. 3 instances. |

Total: 9 families × 3 instances = 27 instances × 3 budgets = 81 EHS runs.

## Budgets

Three time budgets per instance:

| Budget | Value | Purpose |
|--------|-------|---------|
| Short | 30s | Continuity with C4; detect initial front growth |
| Medium | 120s | Check if difficulty persists beyond initial budget crush |
| Long | 300s | Test convergence behavior; detect persistent mechanism stress |

## Constraints

- n ≤ 150, T ≤ 200 (same caps as C4)
- Comparable n/m/T ranges across arms where possible
- No size-only hardness
- All instances must be feasible (sum(p_j) ≤ m × T)

## Metrics

### Per-budget metrics
1. `front_size`: number of non-dominated points in Pareto front
2. `deepest_cmax`: maximum makespan in front (worst-case solution)
3. `shallowest_cmax`: minimum makespan in front (best-case solution)
4. `best_energy`: minimum total energy cost in front
5. `hypervolume`: HV against shared reference point (cmax=T+1, energy=worst_energy_in_all_runs × 1.1)

### Delta metrics
6. `delta_fs_30_120` = front_size(120s) - front_size(30s)
7. `delta_fs_120_300` = front_size(300s) - front_size(120s)
8. `delta_fs_30_300` = front_size(300s) - front_size(30s)
9. `hv_gain_30_120` = HV(120s) - HV(30s)
10. `hv_gain_120_300` = HV(300s) - HV(120s)
11. `delta_cmax_30_300` = deepest_cmax(30s) - deepest_cmax(300s)

### Persistent difficulty metrics
```
persistent_hard = 
    front_size_120 > front_size_30 + 3       # significant growth after short budget
    AND front_size_300 >= front_size_120     # no regression at longer budgets
    AND NOT (front_size_30 <= 2 AND front_size_120 <= 2)  # nontrivial
```

Note: An instance that converges at 120s but NOT at 30s IS persistent-hard. 
The definition checks whether EHS needs more than the trivial 30s budget.
Continued growth from 120s→300s is stronger evidence but not required.

### Mechanism confirmation (budget-aware)

For `asgh_lock_in` (hybrid_M2, agent_mixed_job_sizes, simple_stress):
```
mechanism_confirmed =
    persistent_hard
    AND delta_fs_30_120 > 2              # short budget truncates; lock breaks at 120s
    AND (rt_120 > 30 OR rt_300 > 30)    # EHS still needs substantial time at medium/long
```

For `first_khat_dominance` (hybrid_M1, agent_tight_epsilon):
```
mechanism_confirmed =
    runtime_30s >= 28 AND front_size_30 <= 3    # first khat dominates short budget
    AND front_size_120 > front_size_30 + 3      # significant growth when given medium budget
    AND front_size_300 > front_size_120         # continues growing
```

For `mixed` (literature):
```
mechanism_confirmed = persistent_hard  # any non-trivial continued growth
```

### Nontrivial mechanism confirmation
```
nontrivial_mc =
    mechanism_confirmed
    AND persistent_hard
    AND NOT explained only by n > 100 OR T > 150
```

## Decision Gates

### Gate: STRONG
1. LLM arm beats literature on persistent_hard rate (at least 5/6 vs N/6 with N < 5)
2. LLM arm beats random on persistent_hard rate
3. At least one LLM family has mechanism_confirmed on ≥ 2/3 instances
4. LLM persistent_hard rate ≥ simple_stress baseline
5. Result not fully explained by larger n/T

### Gate: MODERATE
1. LLM beats random on persistent_hard rate
2. LLM beats or ties literature on persistent_hard rate
3. At least one LLM family has mechanism_confirmed on ≥ 1/3 instances

### Gate: WEAK
1. LLM beats random on persistent_hard rate
2. LLM families lose to literature on persistent_hard

### Gate: FAIL
1. LLM families only win at 30s and disappear by 120s
2. Random or literature match LLM on persistent difficulty
3. Mechanism confirmation weak or absent

## Artifacts to Produce

- `C5_PROTOCOL.md` (this file)
- `scripts/_c5_robust_validation.py` — full pipeline
- `eval/c5_raw_runs.csv` — all 81 EHS runs
- `eval/c5_instance_summary.csv` — 27 instance-level summary
- `eval/c5_family_summary.csv` — per-family aggregation
- `eval/c5_mechanism_confirmation.csv` — mechanism check per instance
- `families/c5_frozen_families.json` — frozen family selection
- `notes/c5_claims.md` — paper-safe claims
- `C5_RESULTS.md`
- `C5_DECISION.md`

## Paper-Safe Claims

Acceptable claim:
> "LLM-generated families expose persistent EHS stress mechanisms better than random and literature-derived generators under controlled multi-budget evaluation."

Unacceptable claims:
- "LLM found the hardest instances"
- "LLM proves EHS is weak"
- "LLM beats all baselines including expert-designed sweeps"

# C2B Human Sweep Family Generation

**Date**: 2026-05-10
**Families generated**: 8

## Method
8 fixed families designed as simple parameter sweeps over key dimensions:
epsilon tightness, price volatility, job size distribution, and machine density.
Each family targets a specific EHS mechanism based on B6 evidence.

## Design rationale
These are simple, transparent designs — the human baseline should be strong
enough to challenge the LLM, not trivially beatable.

## Families
- **human_tight_epsilon** → first_khat_dominance: Tight epsilon (≈0.5·max(p_j)) forces many khats. SGH cost dominates. EHS explores few khats under short budget.
- **human_loose_epsilon** → epsilon_skip: Loose epsilon (≈2·max(p_j)) gives few khats. Intermediate tradeoff points may be skipped.
- **human_high_price_volatility** → es_exploration_tension: High TOU price variance creates sharp energy tradeoffs. ES may trap solver in local improvements.
- **human_low_price_volatility** → res_reinsertion_starvation: Nearly flat TOU prices. R-ES reinsertion scheduling refinement is wasted effort since any schedule is energy-equivalent.
- **human_many_small_jobs** → short_budget_pressure: High n, small p_j. SGH O(n·m) cost dominates. First khat may exceed short budget.
- **human_mixed_job_sizes** → asgh_lock_in: Bimodal job sizes. Large jobs hard to reinsert. A-SGH locks in early assignment errors.
- **human_many_machines_sparse** → front_coverage_gap: Many machines, few jobs per machine. Sparse front. Combinatorial assignment search.
- **human_few_machines_dense** → load_imbalance: Few machines, dense jobs. Load balance critical. SGH may fail on heterogeneous rates.

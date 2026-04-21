# Phase F Configuration Probe Results

Date: 2026-04-19

## Scope executed

- new method-family probe (configuration decomposition)
- required primary run: instance `46`, `epsilon=77`
- regime-1 full-enumeration master only
- no column generation implementation in this task

## Implemented pipeline

1. C++ enumerator/pricer (`solvers/cpp/phaseF_config_probe.cpp`)
   - loads paper instance data (`Data_p`, `Data_c`, `Data_e`)
   - groups machines into rate classes
   - fully enumerates all feasible type-count configurations under `epsilon`
   - computes per-class exact configuration costs via `solve_sparse_dp`
   - exports machine-readable artifacts

2. Python master (`solvers/phaseF_config_master_probe.py`)
   - builds class-count integer master (CP-SAT)
   - enforces exact class machine counts and exact type coverage
   - minimizes exact priced cost
   - exports solution summary

## Required truth-source comparisons at `46/77`

- exact reference (`solvers/parallel_f2_cp_sat.py`): `103`
- paper EHS (`temp/paper_exact_repo/results/EHS/1/res_46.csv`): `103`
- `greedy_dp`: `118`
- `greedy_dp_local_search_relocate_only`: `109`
- configuration master (this probe): `103`

## Required metrics (`46/77`)

- number of job types: `4`
- number of rate classes: `3`
- number of configurations per class: `4536`
- total priced configurations: `13608`
- pricing runtime: `18.124586 s`
- master solve runtime: `8.657083 s`
- total runtime: `26.782123 s`
- max RSS:
  - pricing binary run: `6,078,464 bytes` (~5.8 MB)
  - master run: `878,444,544 bytes` (~838 MB)
- final TEC: `103`

## Optional post-success checks on instance 46

- `46/73`: `OPTIMAL`, TEC `103` (matches reference near-opt/F2-init)
- `46/120`: `OPTIMAL`, TEC `103`

## Column-generation caution

- `solve_pricing_dp` was reviewed but not declared directly plug-and-play for true branch-and-price.
- reduced-cost mapping details (especially treatment of empty pattern and exact dual/sign mapping) need a dedicated validation pass.
- therefore this phase remains a full-enumeration regime-1 probe only.

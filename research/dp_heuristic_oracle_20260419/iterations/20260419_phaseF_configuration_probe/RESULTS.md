# Results

## Core probe result (`46/77`)

- full-enumeration configuration master solved with CP-SAT status `OPTIMAL`
- TEC = `103`
- exact reference at `46/77` = `103`
- paper EHS at `46/77` = `103`

Therefore first-probe correctness criterion is satisfied.

## Size and runtime (`46/77`)

- job types: `4` (`lengths: 1,2,3,4`)
- rate classes: `3` (`rates: 1,2,3` with machine counts `3,1,4`)
- configurations enumerated per class: `4536` (same pool reused per class)
- total priced `(class,config)` pairs: `3 * 4536 = 13608`
- enumeration runtime: `0.000173 s`
- pricing runtime: `18.124586 s`
- master solve runtime: `8.657083 s`
- total runtime: `26.782123 s`

## Baseline comparisons at `46/77`

- exact (`parallel_f2_cp_sat`): `103`
- paper EHS stored front: `103`
- `greedy_dp`: `118`
- `greedy_dp_local_search_relocate_only`: `109`
- Phase F master: `103`

## Optional extra epsilon checks on instance 46

- `46/73`:
  - configurations: `4534`
  - TEC: `103`
  - matches reference near-opt/F2-init (`103`)
- `46/120`:
  - configurations: `4536`
  - TEC: `103`

## Artifacts

- `temp/phaseF_config_probe/configs_46_77.csv`
- `temp/phaseF_config_probe/meta_46_77.json`
- `temp/phaseF_config_probe/master_46_77.json`
- `temp/phaseF_config_probe/master_solution_46_77.csv`
- `temp/phaseF_config_probe/master_46_73.json`
- `temp/phaseF_config_probe/master_46_120.json`

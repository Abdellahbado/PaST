# Phase G Regime-2 Restricted-CG Probe Results

Date: 2026-04-20

## Scope executed

- regime 2 only (12 job types)
- single primary target: instance `61`, `epsilon=347`
- restricted master + pricing-style column-addition loop
- no frontier run, no broad benchmark

## What was implemented

- restricted-CG probe driver:
  - `solvers/phaseG_regime2_cg_probe.py`
- exact per-column class pricing helper:
  - `solvers/cpp/phaseG_config_pricer.cpp`
- reused pricing search tool for reduced-cost candidates:
  - `solvers/cpp/pricing_compare.cpp`

## Reduced-cost mapping status

- Mapping used: `rc_c(a) = cost_c(a) - alpha_c - sum_k beta_k * a_k`.
- `solve_pricing_dp` matches this for non-empty patterns with rewards=`beta` and sigma=`alpha_c`.
- Probe caveat: empty pattern is not produced by pricing (`jd > 0`), so zero-column is explicitly kept in master.
- Therefore this is a **bounded, approximately complete CG-style probe**, not a full branch-and-price claim.

## Corrected rerun metrics (`61/347`)

- number of job types: `12`
- number of rate classes: `6`
- initial column count: `259`
- pricing / column-addition iterations: `12`
- final column count: `271`
- LP bound (restricted master): `7024.761959`
- best integer TEC found (restricted master IP): `7040`
- gap to paper EHS (`6710`): `+330`
- gap to reference/F2-init (`6643`): `+397`
- pricing runtime: `264.322016 s`
  - initial pricing: `0.896112 s`
  - pricing search loop: `262.985713 s`
  - newly added-column exact pricing: `0.440191 s`
- master runtime:
  - LP total: `0.032592 s`
  - IP solve: `0.277847 s`
- total runtime: `264.750191 s`
- max RSS (`/usr/bin/time -l`): `1,177,485,312` bytes
- stop reason: `max_iter`

## Comparison context at same epsilon (`61/347`)

- paper EHS: `6710` (`temp/paper_exact_repo/results/EHS/1/res_61.csv`)
- reference near-opt/F2-init: `6643` (`temp/paper_exact_repo/results/reference_near_optimal_fronts/res_61.csv`, `temp/paper_exact_repo/results/F2-init/1/res_61.csv`)
- one-shot `greedy_dp`: `7088`
- one-shot `greedy_dp_local_search_relocate_only`: `7081`
- Phase G restricted-CG probe: `7040`

## Duplicate-stop correction and loop behavior

- Bug in prior run: loop stopped when the single best negative candidate was a duplicate.
- Correction: when duplicate-hit occurs, run bounded threshold pricing for duplicate-hit classes, gather a candidate list, and pick best non-duplicate negative column.
- In corrected rerun, the loop no longer stopped on duplicate; it added 12 new columns (`259 -> 271`) and reached iteration cap.
- Added columns were genuinely new (pool-membership guard enforced; all inserted columns were non-duplicates at insertion time).
- Most new columns came from threshold diversification (mainly class index `2`).
- LP bound remained unchanged at `7024.761959` across iterations.

## Comparison with prior (pre-fix) run at same point

- prior run: `2` iterations, `260` columns, stop `duplicate_pricing_column`, TEC `7040`.
- corrected run: `12` iterations, `271` columns, stop `max_iter`, TEC `7040`.
- net improvement in TEC over previous `7040`: `0`.
- net LP-bound improvement over previous `7024.761959`: `0`.

## Interpretation

- The corrected loop removes the artificial duplicate-stop failure mode and confirms the pipeline can keep adding valid new columns.
- Even after that correction, quality does not improve (`TEC` and LP bound unchanged) and remains far from EHS/reference.
- This is direct evidence that current regime-2 restricted-CG path is non-competitive in this bounded form, not just duplicate-stop limited.

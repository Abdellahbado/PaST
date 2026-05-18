# PAPER GROUPS EXTENSION SUMMARY (PLAN 05 + PLAN 11)

This note tracks the paper-family extension ledger after the K=4 generator fix and the next group-by-group PLAN11 frontier pass.

## Baseline Policy Preserved

- Accepted package unchanged: `energy_core + direct + step1_exact_guided`.
- K=4 baseline remains the accepted PLAN10 policy (`PATTERN_DP_K` default to 4 for K=4, K=4 signature dedup default off).
- No baseline replacement was made during PLAN11; any new variant was additive-only.

## Source-of-Truth Cleanup (Phase A)

- Removed duplicated refreshed `g3567` rows at `(n,seed)=(2500,0/1),(3500,0/1),(5000,0/1)` from `research/k_vs_arithmetic_axes_20260412/csv/plan05/PAPER_GROUPS_PLAN05_n_extension.csv`.
- `research/k_vs_arithmetic_axes_20260412/csv/plan05/PAPER_GROUPS_PLAN05_n_extension.csv` is now a clean ledger (no duplicate logical keys by `family_id,n,lambda,seed`).

## PLAN11 Group-by-Group n Frontier (lambda=1.3, seeds 0/1)

| Family | Last exact n | First regime change beyond 5000 | Dominant behavior in PLAN11 rows |
|---|---:|---:|---|
| {3,5,6,7} (`g3567`) | 6000 | 7000 | exact at 6000 (Step 3); timeout at 7000; `length_error` crash at 8000 |
| {2,4} (`g24`) | 10000 | none through 10000 | exact and Step-2-decided through 10000 |
| {1,2,3,5,7} (`g12357`) | 8000 | 10000 | exact/Step-2 through 8000; timeout at 10000 |
| {2,4,6,8,10} (`g246810`) | 6000 | 7000 | exact/Step-2 at 6000; `length_error` crash from 7000 |
| {1,2,3,4,5,6,7,8,9,10} (`g12345678910`) | 3500 | 5000+ | existing timeout at 5000 remains; timeout again at 6000/7000 |
| {8,10} (`g810`) | 5000 | 6000 | `length_error` crash from 6000 |
| {3,7} (`g37`) | 5000 under corrected reroute | 6000+ not retested in corrected route | old ledger exact only through 600 because later rows were misrouted; PLAN13 reroute closes tested rows `750,1000,1500,2500,3500,5000` through Step-3 exact profile realization |

## Experimental Additive Variant (Phase C)

- Ran explicit experiment on stalled group `g810` only:
  - baseline vs `exp_g810_force_beam` (`PAST_PROFILE_REALIZATION_SELECTOR_POLICY=force_beam`), on `n=7000,8000`, seeds `0,1`.
- Outcome: both variants fail identically with early `length_error`; no evidence to promote variant.
- Baseline policy remains unchanged.

## PLAN13 two-track correction (2026-04-21)

- Added separate PLAN13 artifacts for `{1..10}` and `g37` reroute diagnosis:
  - `research/k_vs_arithmetic_axes_20260412/csv/plan13/PLAN13_easyfamily_g12345678910.csv`
  - `research/k_vs_arithmetic_axes_20260412/csv/plan13/PLAN13_g37_k2_reroute.csv`
  - `research/k_vs_arithmetic_axes_20260412/csv/plan13/PLAN13_variant_compare.csv`
  - `research/k_vs_arithmetic_axes_20260412/csv/plan13/PLAN13_g37_k2_variant_compare.csv`

- Track A `{1..10}`:
  - bounded reruns at `n=5000` remain unresolved in this pass;
  - baseline rows hit run-window timeout,
  - additive probes did not recover closure under memory-safe caps,
  - so no new exact extension claim is added.

- Track B `g37`:
  - rerouted rows now use intended K=2 mainline Step-3 exact profile path
    (`selector_decision=exact`, `selector_reason=k2_exact_default`,
    `step3_mode=exact`, `fwd_pack_method=profile_realization_dp_exact`),
  - and close at zero gap through tested rows `n=750..5000` on seeds `0/1`.

- Archive interpretation update:
  - prior unresolved `g37` rows with `non_mainline_solver` should be treated as
    misrouted evidence, not as evidence that intended K=2 Step-3 exact mode
    fails on `g37` up to `n=5000`.

## HPC / code provenance note

For rerunning these paper-group extensions and identifying the responsible
solver code, use:

- `research/k_vs_arithmetic_axes_20260412/PAPER_HPC_REPRODUCIBILITY_MAP.md`

Final paper runtimes should be regenerated on HPC; this local ledger is the
method/provenance source and current frontier guide.

## Artifacts

- PLAN05 source ledger: `research/k_vs_arithmetic_axes_20260412/csv/plan05/PAPER_GROUPS_PLAN05_n_extension.csv`
- PLAN05 summary CSV: `research/k_vs_arithmetic_axes_20260412/csv/plan05/PAPER_GROUPS_PLAN05_final_summary.csv`
- PLAN11 baseline frontier CSV: `research/k_vs_arithmetic_axes_20260412/csv/plan11/PAPER_GROUPS_PLAN11_n_extension.csv`
- PLAN11 additive experiment CSV: `research/k_vs_arithmetic_axes_20260412/csv/plan11/PAPER_GROUPS_PLAN11_variant_compare.csv`

## PLAN16 fixed-n K-scaling pivot (2026-04-22)

- Separate from n-frontier extension, a fixed-`n=1000` campaign was added to
  compare behavior across current family `K` values.
- Artifact:
  - `research/k_vs_arithmetic_axes_20260412/csv/plan16/PLAN16_k_scaling_n1000.csv`
- Main takeaways:
  - large-`K` contiguous dense families (`K=10`, `K=20`) close exactly at Step 2,
    and dense-unit fastpath improves runtime;
  - corrected K=2 reroute rows show `g37` and `g810` also close exactly at
    `n=1000`, via Step 3 `profile_realization_dp_exact`;
  - the earlier unresolved `plan16` K=2 rows were caused by forcing
    `energy_core`, which bypasses the Step-3 selector path for `K=2`.

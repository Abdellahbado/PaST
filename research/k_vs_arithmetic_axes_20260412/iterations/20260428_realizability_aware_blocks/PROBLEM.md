# Problem

This iteration starts a bounded experiment on **realizability-aware recovered blocks** for the hard fixed-`n=1000` K-axis families.

The current hard cases are not primarily blocked because Step 2 or Step 3 cannot construct any incumbent. For hard irregular `K=10/12`, Step 3 `profile_repair_beam` often produces a finite incumbent, but exact closure remains out of reach and some rows time out or produce no incumbent. PLAN26 also showed that the beam's globally feasible chosen count path does not necessarily survive strict block-local evaluation.

## Evidence behind this direction

- Easy unit-contiguous families `{1..K}` close at `n=1000` through large `K`, often at Step 2.
- Hard irregular families degrade around `K=8`, have finite gaps at `K=10`, and become timeout/no-incumbent dominated around `K=12`.
- PLAN20/22/23 showed that Step-3 survivor policy changes can move individual gaps, but did not produce a stable global improvement.
- PLAN24/24B/25/26 showed that Step-4 corridor exact DP is not currently the right lever:
  - global sparse exact DP overflows or skips,
  - local corridor avoids overflow but is invalid under current block-local evaluation,
  - the base beam path itself fails local block validation.
- The old `smart_reconstruct(...)` path was a global count-aware reconstruction over the relaxed DP table. It is useful as a warning, not as the next method: it scales poorly in `prod_i(total_i+1)` and does not repair the recovered block structure.

## Core hypothesis

The hard irregular cases suffer because the Step-1 recovered blocks are good lower-bound objects but not always good **realizable block objects** for the finite multiset of job sizes.

The next method should not globally reconstruct the whole sequence. It should first measure whether individual recovered blocks are locally realizable, then apply very small local repairs to the bad blocks before Step 3.

## Non-goals

- Do not revive `smart_reconstruct(...)` as the main method.
- Do not continue Step-4 corridor exact DP in this iteration.
- Do not implement full column generation, branch-and-price, MIP/SAT, or a new global relaxation.
- Do not silently change accepted baseline defaults.
- Do not delete or overwrite existing PLAN17-PLAN26 artifacts.

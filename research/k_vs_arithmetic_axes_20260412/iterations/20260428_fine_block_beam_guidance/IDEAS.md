# Ideas

## 1. Residual-aware scoring audit

PLAN27's `residual_aware` variant had zero gap effect, but the diagnostics stayed at `score_policy=default`. First determine whether the policy was ever active.

## 2. Existing-policy oracle

Use PLAN27 and PLAN29 artifacts to compute the best achievable result from already-tested policies. This gives a grounded upper bound before new C++ changes.

## 3. Family-aware survivor selection

PLAN27 suggests:

- hardA benefits most from `uniform_mult2`
- hardB benefits more from ambiguity policies such as `ambig_scoreband_mult2` or `late_ambig`

Test this directly instead of forcing one global policy.

## 4. Fine-block coarse-lookahead scoring

Use coarse or price-window information as a score term only. The beam still transitions on fine blocks.

This keeps the useful signal from PLAN29 without inheriting its main failure mode.


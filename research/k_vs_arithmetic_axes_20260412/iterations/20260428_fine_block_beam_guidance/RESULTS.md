# Results

## Decision: A

Phase 2 family-aware survivor selection passes all gate criteria. Promote as recommended K=10 hard irregular beam policy.

## Phase 0: Residual-aware plumbing fixed

Bug identified and fixed: `beam_diag.score_policy` and residual fields were not copied from `ProfileRepairBeamDiag` to `RecoveredBlockPackingResult`. Now `PAST_PROFILE_REPAIR_BEAM_SCORE_POLICY=residual_aware` correctly activates and shows in diagnostics.

## Phase 1: Oracle from PLAN27

### Best global policy
`uniform_mult2`: mean gap 0.0343% (vs 0.0345%), runtime -14.3%, 6/8 not worse.

### Best family-aware (hypothetical)
- hardA: `uniform_mult2` (3W/0L/1T)
- hardB: `ambig_scoreband_mult2` (3W/1L/0T)
- Combined mean gap: ~0.0318%

### Per-row oracle upper bound
Mean gap 0.0304% — improvement ceiling is modest (~0.004%).

## Phase 2: Family-aware survivor selection → PASSES

Two variants pass:

### family_aware_ambig (hardA=uniform_mult2, hardB=ambig_scoreband_mult2)
- 6/8 improved, 7/8 not worse vs standard
- Mean gap: 0.0318% (baseline 0.0345%, Δ=-0.0027%)
- Runtime: faster than baseline (uniform_mult2 -14.3%)

### family_aware_late (hardA=uniform_mult2, hardB=late_ambig)
- 6/8 improved, 7/8 not worse vs standard
- Mean gap: 0.0309% (baseline 0.0345%, Δ=-0.0036%)
- Runtime: slightly slower on hardB (late_ambig is slower)

## Phase 3: Fine-block coarse-lookahead → Not promoted

Single smoke test shows worse gap (0.0417% vs 0.0391%). Combined with modest oracle ceiling, not justified.

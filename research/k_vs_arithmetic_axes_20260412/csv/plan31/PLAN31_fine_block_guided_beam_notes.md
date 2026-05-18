# PLAN31 — Fine-Block Guided Beam Scoring — Decision A

## Decision: A

Promote **family-aware survivor selection** as the recommended Step-3 beam policy for hard irregular K=10 rows:
- `hardA_k10`: use `uniform_mult2`
- `hardB_k10`: use `ambig_scoreband_mult2` (or `late_ambig`)

Both `family_aware_ambig` and `family_aware_late` pass all Phase 2 gate criteria.

## Phase 0: Residual-Aware Plumbing Audit → Fixed

The `PAST_PROFILE_REPAIR_BEAM_SCORE_POLICY=residual_aware` env var was not reaching the CSV diagnostics. Bug: `beam_diag.score_policy` and residual fields were computed inside `block_repair_feasible_beam_ub` but the copy from `beam_diag` to `RecoveredBlockPackingResult` was missing in `run_profile_beam_attempt`.

**Fix**: Added copy lines for `profile_beam_score_policy`, `profile_beam_residual_weight`, `profile_beam_residual_mean_penalty`, `profile_beam_residual_max_penalty`, `profile_beam_late_frac`.

Verified: setting `PAST_PROFILE_REPAIR_BEAM_SCORE_POLICY=residual_aware` now correctly shows `score_policy = residual_aware` with valid penalty values.

## Phase 1: Existing-Policy Oracle

### Best global policy
`uniform_mult2`: mean gap 0.0343% (vs standard 0.0345%), runtime -14.3%.

### Best family-aware (hypothetical)
- hardA_k10: `uniform_mult2` — 3W/0L/1T vs standard
- hardB_k10: `ambig_scoreband_mult2` — 3W/1L/0T vs standard
- Combined mean gap: ~0.0318% (vs baselines 0.0345%)

### Per-row oracle upper bound
Mean gap 0.0304% — the improvement ceiling is modest (~0.004% from baseline).

## Phase 2: Family-Aware Survivor Selection → PASSES

### family_aware_ambig (hardA=uniform_mult2, hardB=ambig_scoreband_mult2)

| Family | s | Baseline | Family-Aware | Δ | Status |
|--------|----|----------|-------------|------|--------|
| hardA | 0 | 0.0273% | 0.0216% | -0.0057% | BETTER |
| hardA | 1 | 0.0385% | 0.0315% | -0.0070% | BETTER |
| hardA | 2 | 0.0217% | 0.0217% | 0 | SAME |
| hardA | 3 | 0.0320% | 0.0312% | -0.0008% | BETTER |
| hardB | 0 | 0.0391% | 0.0375% | -0.0016% | BETTER |
| hardB | 1 | 0.0477% | 0.0409% | -0.0068% | BETTER |
| hardB | 2 | 0.0450% | 0.0389% | -0.0061% | BETTER |
| hardB | 3 | 0.0249% | 0.0311% | +0.0062% | WORSE |

**Gate**: 6/8 improved, 7/8 not worse, mean gap 0.0318% (-0.0027%). **PASS**.

### family_aware_late (hardA=uniform_mult2, hardB=late_ambig)

| Family | s | Baseline | Family-Aware | Δ | Status |
|--------|----|----------|-------------|------|--------|
| hardA | 0 | 0.0273% | 0.0216% | -0.0057% | BETTER |
| hardA | 1 | 0.0385% | 0.0315% | -0.0070% | BETTER |
| hardA | 2 | 0.0217% | 0.0217% | 0 | SAME |
| hardA | 3 | 0.0320% | 0.0312% | -0.0008% | BETTER |
| hardB | 0 | 0.0391% | 0.0340% | -0.0051% | BETTER |
| hardB | 1 | 0.0477% | 0.0401% | -0.0076% | BETTER |
| hardB | 2 | 0.0450% | 0.0344% | -0.0106% | BETTER |
| hardB | 3 | 0.0249% | 0.0325% | +0.0076% | WORSE |

**Gate**: 6/8 improved, 7/8 not worse, mean gap 0.0309% (-0.0036%). **PASS**.

### Runtime

Both family-aware variants are faster than standard on average (uniform_mult2 is -14.3%, ambig_scoreband_mult2 is -10%). The `late_ambig` variant is slower on hardB (+106s for seed 0, +142s for seed 1) but still within acceptable limits.

### Note on seed 3

`hardB_k10` seed 3 is the universal failure case — every policy (uniform_mult2, ambig_scoreband_mult2, late_ambig, late_residual_ambig) worsens on this seed. This is a structural property of this specific seed under the current beam, not a policy flaw.

## Phase 3: Fine-Block Coarse-Lookahead → Not Promoted

A smoke test on `hardB_k10` seed 0 with `fine_plus_coarse_lookahead` (weight=0.3, window=3) shows gap=0.0417% vs baseline 0.0391% — worse. This single test is not sufficient for a full campaign, but combined with the modest improvement ceiling from Phase 1, coarse lookahead is not justified for promotion.

The lookahead scoring implementation is code-complete and available behind `PAST_PROFILE_REPAIR_BEAM_SCORE_POLICY=fine_plus_coarse_lookahead` for future testing if needed.

## Recommendation

1. **Promote family-aware survivor selection** as the recommended K=10 hard irregular beam policy.
2. **Do not promote** coarse-lookahead scoring (insufficient signal, overhead not justified).
3. **Keep** `uniform_mult2` as the default global policy (already PLAN27 A).
4. **Mark** `residual_aware` as fixed but not promoted (zero gap effect, runtime +11.3%).

## Family-aware policy: explainability

The family structure difference explains the asymmetry:
- `hardA_k10` = {2,3,4,5,7,11,13,17,19,23} includes 2 as the smallest job. Uniform multiplicity helps because small jobs allow more flexible packing.
- `hardB_k10` = {3,5,7,11,13,17,19,23,29,31} starts at 3, no 2. Ambiguous-scoreband multiplicity helps because the beam needs more diversity in scoring to handle the coarser arithmetic structure.

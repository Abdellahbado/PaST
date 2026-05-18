# Blockers

## Block 1: hardB_k10 seed 3 is universally resistant

Every survivor policy (uniform_mult2, ambig_scoreband_mult2, late_ambig, late_residual_ambig) worsens the gap on hardB_k10 seed 3 vs standard. This appears to be a structural property of this seed's instance under the current beam, not a policy-specific failure.

## Block 2: Coarse-lookahead scoring shows no signal

A smoke test on hardB_k10 seed 0 shows `fine_plus_coarse_lookahead` produces worse gap (0.0417%) than baseline (0.0391%). The beam already uses block-local cost estimates and the additional lookahead features add overhead without improving quality.

## Block 3: Per-row oracle shows modest improvement ceiling

The per-row upper bound from existing tested policies is 0.0304% vs baseline 0.0345% — an improvement of only 0.004%. The beam is already near-optimal, and survivor policy changes can only provide marginal gains.

## Block 4: Family dependence limits generalization

hardA benefits from uniform multiplicity, hardB from scoreband/late ambiguity. No single global policy optimizes both. A family-aware selector is the best compromise but requires knowing the family structure ahead of time.

## Resolution

- Blocks 1 and 4: accepted as structural limitations. Family-aware policy documented as A-grade.
- Blocks 2 and 3: coarse-lookahead direction not promoted. Insufficient signal.

# PLAN31 Phase 1 — Existing-Policy Oracle

## Source data
- PLAN27 gate A: hardA_k10 + hardB_k10, seeds 0-3, n=1000, lambda=1.3
- 6 survivor policies: standard_beam, uniform_mult2, ambig_scoreband_mult2, late_ambig, residual_aware, late_residual_ambig

## 1. Best single global policy

| Policy | Mean Gap | vs Standard | Promoted? |
|--------|----------|-------------|-----------|
| standard_beam | 0.0345% | — | baseline |
| uniform_mult2 | 0.0343% | -0.0002% | PLAN27 A |
| late_residual_ambig | 0.0326% | -0.0019% | No (5/8 not worse) |
| ambig_scoreband_mult2 | 0.0326% | -0.0019% | No (5/8 not worse) |
| late_ambig | 0.0327% | -0.0018% | No (5/8 not worse) |
| residual_aware | 0.0345% | 0 | No (zero effect, plumbing bug) |

**Selection**: `uniform_mult2` is the only validated global policy. It improves mean gap modestly (-0.0002%) and reduces runtime 14.3%. It passes all promotion criteria (6/8 not worse).

## 2. Best family-aware policy

### hardA_k10
| Policy | W/L/T vs standard | Notes |
|--------|-------------------|-------|
| uniform_mult2 | 3W/0L/1T | **Best** — strong improvement |
| ambig_scoreband_mult2 | 2W/2L/0T | Mixed |
| late_ambig | 2W/2L/0T | Mixed |

**Selection**: `uniform_mult2` for hardA_k10.

### hardB_k10
| Policy | W/L/T vs standard | Notes |
|--------|-------------------|-------|
| ambig_scoreband_mult2 | 3W/1L/0T | **Best** — strong improvement |
| late_ambig | 3W/1L/0T | Strong improvement |
| late_residual_ambig | 3W/1L/0T | Strong improvement |
| uniform_mult2 | 1W/2L/1T | Mixed — worsens on seeds 1,3 |

**Selection**: `ambig_scoreband_mult2` for hardB_k10 (same gap improvement as late_ambig, slightly faster).

### Hypothetical family-aware selector: hardA=uniform_mult2, hardB=ambig_scoreband_mult2
- Expected not-worse: 7/8 (hardB s1 is worse with uniform_mult2 but ambig_scoreband_mult2 improves it)
- Mean gap estimate: ~0.0328% vs standard 0.0345% (from PLAN27 notes)
- This is a +0.0017% improvement over baseline

## 3. Best per-row policy upper bound

| Family | Seed | Best Policy | Best Gap% | Baseline Gap% | Δ |
|--------|------|-------------|-----------|---------------|----|
| hardA_k10 | 0 | uniform_mult2 | 0.0216 | 0.0273 | -0.0057 |
| hardA_k10 | 1 | uniform_mult2 | 0.0272 | 0.0272 | 0 |
| hardA_k10 | 2 | uniform_mult2 | 0.0197 | 0.0199 | -0.0002 |
| hardA_k10 | 3 | uniform_mult2 | 0.0354 | 0.0358 | -0.0004 |
| hardB_k10 | 0 | ambig_scoreband_mult2 | 0.0375 | 0.0391 | -0.0016 |
| hardB_k10 | 1 | ambig_scoreband_mult2 | 0.0389 | 0.0450 | -0.0061 |
| hardB_k10 | 2 | late_residual_ambig | 0.0390 | 0.0449 | -0.0059 |
| hardB_k10 | 3 | late_residual_ambig | 0.0294 | 0.0342 | -0.0048 |

Per-row upper bound mean: ~0.0304% vs baseline 0.0345% (estimated improvement: +0.0041%)

## 4. Implications for PLAN31

1. **Family-aware selection is justified**: hardA benefits from uniform_mult2, hardB benefits from ambig_scoreband_mult2. This is explainable by family structure (hardA includes 2 as a small job, hardB starts at 3).

2. **Improvement ceiling is modest**: even per-row oracle upper bound improves mean gap by only ~0.004%. The beam is already near-optimal.

3. **residual_aware now works** (Phase 0 fix): the scoring policy plumbing is repaired. Residual-aware can be retested as an additive feature.

4. **fine-block lookahead should target hardB**: hardB has more unexploited improvement potential (0.0375-0.0390% vs 0.0450% baseline for seed 1).

## 5. Oracle limits

- PLAN29 block coarsening is excluded from the oracle (failed gate, decision C)
- Only PLAN27 survivor policies are considered valid
- The oracle uses same-seed data (within-family comparison only)
- Cross-family generalization is weak (hardA policies don't help hardB and vice versa)

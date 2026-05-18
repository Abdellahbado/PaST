# PLAN32 Phase 0 — Existing K12 Incumbent Audit

## Key findings

### Which K12 rows already have finite UB?

| Family | Seed | Best UB | Best Gap | Source | Policy |
|--------|------|---------|----------|--------|--------|
| easy_k12 | 0 | 62,967,165 | 0.0000% (exact) | PLAN28 | baseline |
| easy_k12 | 1 | 62,384,424 | 0.0000% (exact) | PLAN28 | baseline |
| hardA_k12 | 0 | 129,770,495 | 0.0232% | PLAN22B | uniform_mult2 |
| hardA_k12 | 1 | 132,462,319 | 0.0444% | PLAN22B | uniform_mult2 |
| hardA_k12 | 2 | 128,534,730 | 0.0399% | PLAN18 | profile_repair_beam |
| hardA_k12 | 3 | — | — | — | **NO INCUMBENT** |
| hardB_k12 | 0 | 187,869,803 | 0.0439% | PLAN22B | ambig_scoreband_mult2 |
| hardB_k12 | 1 | 186,111,159 | 0.0434% | PLAN28 | profile_repair_beam |
| hardB_k12 | 2 | 184,568,251 | 0.0292% | PLAN18 | profile_repair_beam |
| hardB_k12 | 3 | — | — | — | **NO INCUMBENT** |

### True no-incumbent targets (need PLAN32)

- **hardA_k12 seed 3**: Never recovered in any PLAN (18/19/22B/27/28/29)
- **hardB_k12 seed 3**: Never recovered in any PLAN (18/19/22B/27/28/29)

### Which route/policy produced finite UB most reliably?

1. `profile_repair_beam` + `auto_v1` selector (uniform_mult2): 6/8 reсovered incumbents
2. `profile_repair_beam` + ambig_scoreband_mult2: 6/8 recovered
3. Energy-core baseline: 0/8 recovered (always selector_bypass or timeout)

### Why some rows fail

- **hardA_k12 s3**: Beam times out or produces no candidate. No known incumbent.
- **hardB_k12 s3**: Same failure mode. No known incumbent.
- **hardB_k12 s1**: Recovered in PLAN28 but not in earlier plans. Fragile — depends on seed structure.

### Implication for PLAN32

The initial UB safety layer must produce a feasible incumbent for ALL rows, including the two true zero-incumbent seeds. This means:
1. The initial UB layer is mandatory before beam work
2. If the beam/exact times out, we MUST return the initial UB
3. The initial UB quality doesn't need to be excellent — finite > -1

## Sources

- PLAN18: `csv/plan18/PLAN18_k_boundary_refine_failure_signatures.csv`
- PLAN19: `csv/plan19/PLAN19_k10_k12_redesign_raw.csv`
- PLAN22B: `csv/plan22b/PLAN22B_ambig_scoreband_validation_raw.csv`
- PLAN27: `csv/plan27/PLAN27_step3_adaptive_survivor_raw.csv`
- PLAN28: `csv/plan28/PLAN28_block_realizability_diagnostics_raw.csv`

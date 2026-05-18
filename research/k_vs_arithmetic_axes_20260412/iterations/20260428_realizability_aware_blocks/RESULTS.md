# Results

## Decision C

**PLAN28 Phase A diagnostics FAIL to separate easy from hard rows.**

The block-realizability diagnostic gate does not pass because the computed metrics do not cleanly distinguish easy unit-contiguous families (gap=0%) from hard irregular families (gap>0%).

## Key results table

| Family | K | Seed | Gap% | Blocks | Bad Blocks | Bad Rate | Min Fin | Mean Fin | Base Survives | Reject Reason |
|--------|----|------|------|--------|-----------|----------|---------|----------|---------------|---------------|
| easy | 8 | 0 | 0% | 6 | 3 | 0.500 | 36 | 36.7 | 0 | block_0_chosen_infeasible |
| easy | 8 | 1 | 0% | 6 | 3 | 0.500 | 35 | 37.5 | 0 | block_0_chosen_infeasible |
| easy | 10 | 0 | 0% | 8 | 4 | 0.500 | 45 | 48.4 | 0 | block_0_chosen_infeasible |
| easy | 10 | 1 | 0% | 9 | 5 | 0.556 | 45 | 48.0 | 0 | block_0_chosen_infeasible |
| easy | 12 | 0 | 0% | 9 | 5 | 0.556 | 50 | 55.6 | 0 | block_0_chosen_infeasible |
| easy | 12 | 1 | 0% | 9 | 5 | 0.556 | 51 | 55.9 | 0 | block_0_chosen_infeasible |
| hardA | 8 | 0 | 0.005% | 10 | 5 | 0.500 | 36 | 37.0 | 0 | block_0_chosen_infeasible |
| hardA | 8 | 1 | 0.020% | 12 | 9 | 0.750 | 36 | 38.1 | 0 | block_0_chosen_infeasible |
| hardA | 10 | 0 | 0.017% | 14 | 7 | **0.500** | 44 | 47.2 | 0 | block_0_chosen_infeasible |
| hardA | 10 | 1 | 0.009% | 14 | 7 | **0.500** | 45 | 46.9 | 0 | block_0_chosen_infeasible |
| hardA | 12 | 0 | 0.024% | 22 | 17 | 0.773 | 54 | 56.0 | 0 | block_0_chosen_infeasible |
| hardA | 12 | 1 | 0.045% | 24 | 20 | 0.833 | 54 | 55.2 | 0 | block_0_chosen_infeasible |
| hardB | 8 | 0 | 0.030% | 18 | 15 | 0.833 | 36 | 36.8 | 0 | block_0_chosen_infeasible |
| hardB | 8 | 1 | 0.033% | 18 | 15 | 0.833 | 36 | 36.8 | 0 | block_0_chosen_infeasible |
| hardB | 10 | 0 | 0.039% | 24 | 18 | 0.750 | 44 | 46.1 | 0 | block_0_chosen_infeasible |
| hardB | 10 | 1 | 0.048% | 22 | 17 | 0.773 | 44 | 46.2 | 0 | block_0_chosen_infeasible |
| hardB | 12 | 0 | nan | — | — | — | — | — | — | no_incumbent |
| hardB | 12 | 1 | 0.043% | 29 | 20 | 0.690 | 55 | 56.5 | 0 | block_0_chosen_infeasible |

## Four reasons the diagnostics fail

### 1. Universal base-path failure
`base_path_survives=0` for every row. The beam's chosen counts are never locally feasible at block 0. This is not a signal — it's a constant.

### 2. Bad-rate overlap
Easy families have bad_rate 50–56%. HardA_k10 has bad_rate **50%** (same) but gap 0.009–0.017%. A diagnostic that produces the same value for a gap-0 and a gap>0 row does not separate.

### 3. Finite patterns are K-dependent
Mean finite patterns per block: K=8 → 36–38, K=10 → 45–48, K=12 → 55–56. Identical for easy and hard families at the same K.

### 4. Easy families close at Step 2
All easy rows achieve exact closure via Step 2 (FFD/FFI). Their "bad blocks" are never used. Hard families cannot close at Step 2 and must use the beam, whose blocks are locally irreparable.

## Memory safety

All 18 rows ran within the 16 GB cap. Peak RSS: 0.58–5.74 GB. No memory kills.

## Artifacts

- `csv/plan28/PLAN28_block_realizability_diagnostics_raw.csv`
- `csv/plan28/PLAN28_block_realizability_diagnostics_summary.csv`
- `csv/plan28/PLAN28_block_realizability_notes.md`

# PLAN28 Phase A: Block Realizability Diagnostics — Gate Decision C

## Decision: C

**Gate A FAILS.** Block-realizability diagnostics do not clearly separate easy
unit-contiguous rows from hard irregular rows, and do not reliably correlate
with final gap or no-incumbent outcomes.

## Key findings

### 1. Universal base-path failure
- `block_realiz_base_path_survives = 0` for ALL 17 rows that produced beam incumbents
- Beam's chosen counts are never locally feasible at block 0 for any family
- This is structural: `generate_energy_core_patterns` builds patterns by work capacity,
  not by block-local schedulability. The beam validates the full global sequence,
  not individual blocks.

### 2. Bad-block rates overlap between easy and hard
| Family Type | K | Bad Rate Range | Gap Range |
|---------------|---|---------------|-----------|
| Easy | 8-12 | 50.0%–55.6% | 0% (all exact) |
| hardA | 8 | 50.0% (s0), 75.0% (s1) | 0.0047%–0.0202% |
| hardA | 10 | 50.0% (both seeds) | 0.0088%–0.0172% |
| hardA | 12 | 77.3%–83.3% | 0.0239%–0.0448% |
| hardB | 8 | 83.3% (both seeds) | 0.0297%–0.0332% |
| hardB | 10 | 75.0%–77.3% | 0.0391%–0.0477% |
| hardB | 12 | 68.97% (s1 only) | 0.0434% (s1), no incumbent (s0) |

**Critical counterexample:** `hardA_k10` has bad_rate=50.0% (same as easy families)
but gap=0.0088%–0.0172%. A diagnostic that produces the same value for a gap-0 row
and a gap>0 row does not separate.

### 3. Finite pattern counts are K-dependent, not family-dependent
- K=8: mean_finite_patterns = 36–38 across all families
- K=10: mean_finite_patterns = 45–48 across all families
- K=12: mean_finite_patterns = 55–56 across all families
- No material difference between easy and hard families at the same K

### 4. Easy families close at Step 2 despite bad blocks
- All easy rows close at Step 2 (FFD/FFI), gap=0%
- Their bad blocks are irrelevant because the pipeline never relies on the beam
- Hard families cannot close at Step 2 and must use the beam, whose blocks are bad

## Why the diagnostics don't work

The fundamental issue, confirmed by PLAN26, is that **block-local evaluation
(`evaluate_profile_block_counts`) is stricter than the beam's global validation.
The beam's chosen counts are globally feasible (they produce a valid schedule)
but not locally feasible per block. This mismatch is universal — it affects
easy families too, but easy families don't need the beam.

The diagnostics measure whether the beam's blocks are locally realizable.
Since they never are (always fail at block 0), the diagnostic is a constant
(base_path_survives=0) and provides no discriminative signal.

## Phase A rows

18 rows run (9 families × 2 seeds). 17 produced beam incumbents with diagnostics.
1 row (hardB_k12 seed 0) produced no incumbent at all.
All rows memory-safe (peak RSS 0.58–5.74 GB, well under 16 GB cap).

## Next step

This direction is stopped. **Decision C**: Block-realizability diagnostics
do not separate easy from hard. The problem is structural, not diagnostic.

Possible future directions:
1. Replace block-local evaluation with global sequence evaluation (major redesign)
2. Generate only actually schedulable patterns per block (may severely limit pool)
3. Abandon block-layered approach for hard irregular families entirely

## Artifacts
- `csv/plan28/PLAN28_block_realizability_diagnostics_raw.csv`
- `csv/plan28/PLAN28_block_realizability_diagnostics_summary.csv`
- `csv/plan28/PLAN28_block_realizability_notes.md` (this file)
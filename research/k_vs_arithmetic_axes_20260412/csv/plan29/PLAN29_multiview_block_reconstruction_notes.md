# PLAN29 Phase A: Multi-View Block Reconstruction — Gate Decision C

## Decision: C

**Gate A FAILS.** No single coarsening view improves at least 4/8 hard K10 anchor rows.

## Gate A per-variant summary

| Variant | Improved | Not Worse | Gate |
|---------|----------|-----------|------|
| target_B12 | 3/8 | 5/8 | FAIL |
| target_B8 | 2/8 | 3/8 | FAIL |
| price_preserve_B12 | 2/8 | 4/8 | FAIL |
| coarsen2 | 1/8 | 1/8 | FAIL |
| arith_adaptive | 1/8 | 3/8 | FAIL |
| coarsen3 | 0/8 | 0/8 | FAIL |

## Key findings

### 1. Coarsening makes things worse on hardA families
- On hardA_k10, every coarsening variant is either same-as-baseline (target_B12,
  price_preserve_B12 on seed 0) or strictly worse (seeds 1-3)
- `arith_adaptive` is a no-op on hardA (adaptive threshold `2*max_len=46` 
  never triggers — block lengths are all >46 for this family)
- `coarsen2` doubles the gap on seed 2 (0.0379% vs 0.0199%)

### 2. Mixed results on hardB families
- `target_B8` improves seeds 0 and 1 (0.0391→0.0294%, 0.0477→0.0377%) but
  worsens seeds 2 and 3
- `target_B12` improves seeds 1 and 2 (hardB) and seed 2 (hardA)
- `price_preserve_B12` improves hardB seed 1 (0.0477→0.0376%)
- No single view generalizes across both families

### 3. Coarsening consistently degrades on most seeds
- `coarsen2` and `coarsen3` are universally worse on hardA and mostly worse on hardB
- The coarser the blocks, the worse the gap (coarsen3 > coarsen2 > baseline gap)
- This directly contradicts the hypothesis that coarser blocks help the beam

### 4. arith_adaptive is a no-op or harmful
- On hardA: zero boundaries removed (threshold too high)
- On hardB seed 0: improves (9 removed, 15 blocks)
- On hardB seed 1: worsens (9 removed, 13 blocks)
- On hardB seed 2: worsens (8 removed, 15 blocks)
- On hardB seed 3: worsens (9 removed, 15 blocks)
- Net effect: 1 improved, 3 worse, 4 unchanged

## Why coarsening doesn't help

The beam already operates on merged blocks that aggregate touching/overlapping
recovered blocks. Further coarsening reduces the number of beam layers but also:

1. **Losses price-profile fidelity** — wider blocks average over price changes,
   making the beam's cost estimates less accurate
2. **Reduces count-allocation precision** — fewer layers mean the beam commits
   to larger count aggregates, reducing its ability to fine-tune per-window
3. **Pattern explosion** — wider blocks have larger capacity, creating more
   possible count patterns (wider search space), not fewer

The beam's quality depends on the accuracy of its block-local cost estimates.
Wider blocks have less accurate local costs because price varies more within
the block.

## Phase A rows

- 56 rows run (8 base rows × 7 variants)
- All memory-safe (peak RSS well under 16 GB)
- All rows completed within 1200s time limit
- No exact rows (0/56 is_optimal)

## Next step

This direction is stopped. **Decision C**: Adjacent block coarsening does not
improve beam incumbent quality. The beam is already near-optimal with the
standard block structure.

## Artifacts
- `csv/plan29/PLAN29_multiview_block_reconstruction_raw.csv`
- `csv/plan29/PLAN29_multiview_block_reconstruction_compare.csv`
- `csv/plan29/PLAN29_multiview_block_reconstruction_summary.csv`
- `csv/plan29/PLAN29_multiview_block_reconstruction_notes.md` (this file)
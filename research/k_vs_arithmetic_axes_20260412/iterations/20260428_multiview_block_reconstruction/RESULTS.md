# Results

## Decision C

**PLAN29 Phase A Gate A FAILS.** No single adjacent block coarsening view improves at least 4/8 hard K10 anchor rows.

## Per-variant improved counts

| Variant | Improved | Not Worse | Gate |
|---------|----------|-----------|------|
| target_B12 | 3/8 | 4/8 | FAIL |
| target_B8 | 3/8 | 3/8 | FAIL |
| price_preserve_B12 | 3/8 | 4/8 | FAIL |
| coarsen2 | 1/8 | 1/8 | FAIL |
| arith_adaptive | 1/8 | 5/8 | FAIL |
| coarsen3 | 0/8 | 0/8 | FAIL |

Required: ≥ 4/8.

## Per-family patterns

### hardA_k10
All coarsening variants are either same-as-baseline or strictly worse. `arith_adaptive` is a no-op on hardA (threshold `2*max_len=46` never triggers — all block lengths exceed 46).

| Seed | baseline | best variant | best gap | Δ |
|------|----------|-------------|----------|---|
| 0 | 0.0172% | target_B12 | 0.0172% | 0 |
| 1 | 0.0088% | arith_adaptive | 0.0088% | 0 |
| 2 | 0.0199% | target_B12 | 0.0197% | -0.0002% |
| 3 | 0.0091% | arith_adaptive | 0.0091% | 0 |

### hardB_k10
Mixed results. `target_B8` and `target_B12` improve seeds 0-2 but worsen seed 3.

| Seed | baseline | best variant | best gap | Δ |
|------|----------|-------------|----------|---|
| 0 | 0.0391% | target_B8 | 0.0294% | -0.0097% |
| 1 | 0.0477% | price_preserve_B12 | 0.0376% | -0.0101% |
| 2 | 0.0450% | target_B12 | 0.0432% | -0.0018% |
| 3 | 0.0249% | baseline | 0.0249% | 0 |

## Why coarsening doesn't help

The beam already operates on merged blocks that aggregate touching/overlapping recovered blocks. Further coarsening:

1. **Losses price-profile fidelity** — wider blocks average over price changes
2. **Reduces count-allocation precision** — fewer layers mean less fine-tuning
3. **Pattern explosion** — wider blocks have larger capacity, creating MORE candidate patterns, not fewer

The beam's quality depends on the accuracy of its block-local cost estimates. Wider blocks have less accurate local costs because price varies more within the block.

## Memory safety

All 56 rows completed within 16 GB cap. No memory kills. No timeouts.

## Artifacts

- `csv/plan29/PLAN29_multiview_block_reconstruction_raw.csv` (56 rows)
- `csv/plan29/PLAN29_multiview_block_reconstruction_compare.csv`
- `csv/plan29/PLAN29_multiview_block_reconstruction_summary.csv`
- `csv/plan29/PLAN29_multiview_block_reconstruction_notes.md`

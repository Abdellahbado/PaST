# Ideas

## Multi-view adjacent coarsening

Given the current merged block list:

```text
B1 | B2 | B3 | ... | BB
```

create a small set of alternative views by removing selected adjacent boundaries:

- `baseline`: current merged blocks.
- `coarsen2`: `(B1+B2) | (B3+B4) | ...`
- `coarsen3`: `(B1+B2+B3) | (B4+B5+B6) | ...`
- `target_B12`: merge adjacent blocks until about 12 blocks remain.
- `target_B8`: merge adjacent blocks until about 8 blocks remain.
- `price_preserve_B12`: merge to about 12 blocks while preserving boundaries with sharp price changes.
- `arith_adaptive`: coarsen more aggressively around short blocks when the largest job sizes are large relative to block length.

## Why this may help

Step 3 beam is layered by recovered blocks. Too many thin layers force the beam to commit to count allocations early. Coarser blocks reduce the number of decisions and allow counts to move over a wider local time region.

The tradeoff:

- too fine: brittle count flow;
- too coarse: loses price-profile structure;
- price-preserving coarsening: keeps important tariff boundaries while reducing weak boundaries.

## Stop rule

If no coarsening view improves at least half of the K10 anchor rows, do not continue this direction.

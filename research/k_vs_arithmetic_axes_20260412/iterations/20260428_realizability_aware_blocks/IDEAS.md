# Ideas

## Main idea: block realizability diagnostics

For each recovered block produced by Step 1 / profile recovery, compute cheap diagnostics that describe whether the block is realistically fillable by the remaining multitype jobs.

Useful per-block diagnostics:

- block index, start/end/length/work capacity
- chosen count vector from the Step-3 beam when available
- whether the beam's chosen count vector is locally schedulable under the current block evaluator
- number of locally finite candidate count patterns
- minimum and average local pattern cost among retained candidates
- block slack / residual work mismatch
- first rejection reason when local evaluation fails

Useful row-level summaries:

- total recovered blocks
- bad blocks count and bad-block rate
- first bad block index
- minimum finite local patterns over blocks
- mean finite local patterns over blocks
- whether the base beam path survives block-local validation
- final UB/LB/gap and deciding step

## Small repair ideas

Only run repair variants if diagnostics correlate with hard-family failure.

Allowed repairs:

- `merge_bad_prev`: merge each selected bad block with its previous neighbor.
- `merge_bad_next`: merge each selected bad block with its next neighbor.
- `merge_bad_triplet`: merge previous/current/next around the first few bad blocks.
- `small_boundary_shift`: shift one small amount of work from a bad block to a neighbor.
- `best_local_repair`: try the small options above and select the one with the best diagnostic score before Step 3.

The repair should modify only the recovered block list passed to Step 3. It must not claim a better lower bound or exact proof.

## Why this is different from old smart reconstruction

Old `smart_reconstruct(...)` searched a global count-state DP guided by the relaxed table. That repeats the scalable-state-space problem and does not directly fix bad recovered blocks.

This iteration instead keeps the relaxed profile, detects bad local block objects, and changes only a small number of local block boundaries/merges before the existing Step-3 beam.

## Stop rule

If bad-block diagnostics do not separate easy unit-contiguous rows from hard irregular rows, stop this direction and do not implement repairs.

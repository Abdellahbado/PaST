# Blockers

## Block 1: Block-local evaluation is universally stricter than global beam validation

This is the fundamental structural blocker, confirmed by PLAN28.

### What it is
`generate_energy_core_patterns` generates patterns by **work capacity** (total job lengths ≤ block capacity), not by **schedulability** (whether the specific multiset can be scheduled in the block given machine transition constraints). The beam validates the full global sequence using `solve_fixed_sequence` on the entire horizon. Block-local evaluation via `evaluate_profile_block_counts` is much stricter — it requires the chosen counts to be independently schedulable within each block's time window.

### Evidence from PLAN28
- `block_realiz_base_path_survives = 0` for ALL 17 rows with beam incumbents
- The first block (block 0) always fails: the beam's chosen counts for that block cannot be locally scheduled
- This is true for easy families (that close at Step 2 anyway) and hard families (that rely on the beam)
- `block_realiz_bad_rate = 50%` for easy families — but they don't care because they close at Step 2

### Why it blocks this direction
Block-realizability diagnostics cannot separate easy from hard because the beam's blocks are always bad at the beginning. The diagnostic is a constant, not a signal. Any repair that modifies individual blocks will fail because the beam's global count allocation doesn't match block-local schedulability constraints.

### Relation to previous findings
PLAN26 already documented that `base_candidate_not_found_at_layer_0` is the root cause of local corridor failure. PLAN28 reproduces this finding at the raw block level: it's not a corridor issue, it's a fundamental mismatch between beam global construction and block-local evaluation.

### Possible resolution (not pursued under PLAN28)
1. Replace block-local evaluation with global sequence evaluation (major redesign)
2. Generate only actually schedulable patterns per block (would severely limit the candidate pool, likely making beam search infeasible)
3. Accept that the beam is close enough to optimal (<0.05% gap) and invest in different approaches (different relaxation, tighter bounds, or column generation — but these are out of scope for this thread)

## Block 2: Easy families close at Step 2, making block diagnostics irrelevant

Easy unit-contiguous families `{1..K}` close exactly via Step 2 FFD/FFI. They never need the beam. The beam's diagnostic (bad blocks at block 0) is irrelevant for these families. The separation between easy (gap=0) and hard (gap>0) is a pipeline routing difference, not a block-quality difference.

## Resolution status

**This direction is stopped (Decision C).** No further work on block-realizability diagnostics or bounded repair under this iteration.

# Plan 02: Dynamic Pricing Diagnostic + Paper Consolidation

## Status

Prepared on: `2026-04-13`

Depends on: Plan 01 (completed, Recommendation 2 selected)

Archive context:

- [EXPERT_GUIDANCE.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/EXPERT_GUIDANCE.md)
- [RESULTS.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/RESULTS.md)
- [BLOCKERS.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/BLOCKERS.md)
- [csv/two_axis_grid/TWO_AXIS_GRID_20260412_phase2.csv](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260412_phase2.csv)

Primary code:

- [stateful_dp_solver.cpp](/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/stateful_dp_solver.cpp)
- [run_two_axis_grid.py](/Users/mac/Documents/Study/PFE/PaST/scripts/run_two_axis_grid.py)

---

## Objective

This plan has two goals, in strict priority order:

1. **Determine whether dynamic pricing is the RIGHT intervention** — not
   assumed, tested.
2. **If it is, implement it cleanly and validate it.** If it is NOT,
   consolidate the paper story with the current beam-owned baseline.

This plan explicitly forbids spending more than one coding cycle on
dynamic pricing before checking the result. If pricing does not improve
the hard rows, the plan shifts to paper consolidation immediately.

---

## What we know entering this plan

### Completed cross-cell grid (from Plan 01)

| Family | K | n=300 | n=1000 gap | n=1000 winner | merged blocks |
|---|---|---|---|---|---|
| {3,5,7,11} | 4 | exact | 0.0083% | energy_core | 9 |
| {2,3,4,5,7,11} | 6 | exact | 0.0064–0.0130% | beam | 8–14 |
| {4,5,6,7,8,9} | 6 | exact | 0.0138–0.0260% | beam | 9–17 |
| {3,5,7,11,13,17,19,23} | 8 | exact | 0.0157% | beam | 19 |
| {2,3,5,7,11,...,29} | 10 | exact | 0.0221% | beam | 20 |

### Key observations

1. The beam wins Level 2 on ALL K≥6 rows.
2. Gaps grow mildly with K on irregular arithmetic (0.008% → 0.022%).
3. The Lagrangian baseline was NOT recovered — the earlier strong numbers
   came from hidden hybrid behavior.
4. The semigroup relaxation (LB) is already tight — all remaining gap is
   on the UB side.

### The critical unanswered question

Is the beam winning because:

- **(a) Pool gap**: It discovers count-vector assignments that are NOT in
  the Lagrangian's pre-filtered pattern pool? Or
- **(b) Search gap**: It finds BETTER COMBINATIONS of the same patterns
  the Lagrangian has access to?

**This question MUST be answered before implementing dynamic pricing.**

If (a): dynamic pricing will help — it gives the Lagrangian access to
the missing patterns.

If (b): dynamic pricing will NOT help — the Lagrangian needs better
search, not more patterns. In this case, the beam is already the best
Level 2 method, and the paper should use it as is.

**Important architectural insight**: The beam (`block_repair_feasible_beam`)
does NOT operate on a pre-enumerated pattern pool. It builds assignments
incrementally through a forward search, implicitly generating any feasible
count vector on the fly. The Lagrangian operates on a FIXED, FILTERED
pattern pool. This means the beam has inherently more flexibility by
construction — it can use patterns the Lagrangian has never seen.

This strongly suggests the answer is (a), but we must verify empirically.

---

## Non-goals

1. Do NOT build a full branch-and-price solver.
2. Do NOT implement arc-flow.
3. Do NOT change Level 1 (relaxation) or Level 3 (per-block eval).
4. Do NOT spend more than one coding cycle on pricing before measuring.
5. Do NOT block the paper on pricing success — the current gaps are
   already very small (< 0.025%).

---

## Phase A. Diagnose pool gap vs search gap

### Purpose

Answer the critical question before writing any pricing code.

### Method

On ONE representative hard row (e.g., `hard_k6_2345711 n=1000 seed=0`):

1. Run the solver and capture the beam's winning assignment — the exact
   count vector `counts[b][j]` that the beam used for each block `b`.

2. For each block `b`, check whether the beam's count vector
   `counts[b][:]` is present in the Lagrangian's filtered pattern pool
   `patterns[b]`.

3. Count:
   - `n_blocks_beam_in_pool`: number of blocks where the beam's pattern
     IS in the pool
   - `n_blocks_beam_not_in_pool`: number of blocks where it is NOT

### Implementation

This requires adding a small diagnostic inside the solver. In
`block_repair_feasible_beam`, after the beam finds its winning
assignment, log the count vectors. Then in the Lagrangian path (or a
separate diagnostic pass), check pool membership.

Alternatively: after the beam wins, iterate over all blocks, hash the
beam's count vector, and check whether it hashes to any pattern in the
Lagrangian's pool for that block. This is O(B × P) where P is patterns
per block — very cheap.

### Required output

Log one line per run:

```
POOL_DIAG: total_blocks=14, beam_in_pool=11, beam_not_in_pool=3
```

### Decision rule

- If `beam_not_in_pool >= 2` on the representative rows:
  → **Pool gap is real. Proceed to Phase B (dynamic pricing).**

- If `beam_not_in_pool == 0` on all rows:
  → **Pool gap is zero. The bottleneck is search quality. Skip Phase B,
     go directly to Phase D (paper consolidation using beam as Level 2).**

- If `beam_not_in_pool == 1` and it's a marginal block:
  → Run the diagnostic on 2–3 more rows. If pattern persists, proceed
     to Phase B. If it's noise, go to Phase D.

### Success criterion

We have a clear, logged answer to "pool vs search" for at least three
hard-arithmetic rows.

### If the diagnostic is too hard to implement

If instrumenting the beam to log its block-level count vectors is too
invasive, use this approximation instead:

- Count the number of filtered patterns per block in the Lagrangian pool
- Compare against the theoretical number of feasible count vectors
  (bounded representability as fill patterns)
- If the pool contains < 50% of feasible patterns for blocks with gaps,
  pool limitation is likely

---

## Phase B. Implement clean dynamic pricing (only if Phase A says pool gap)

### Purpose

Give the Lagrangian access to patterns it currently cannot see, without
rewriting the assignment search.

### Architecture

The pricing subproblem for block `b` at Lagrangian iteration `t` is:

```
maximize  Σ_j  λ_j^(t) × count_j
subject to  Σ_j  L_j × count_j  ≤  cap_b
            0 ≤ count_j ≤ remaining_j    for all j
```

where `λ_j^(t)` are the current dual multipliers on type-j job totals.

This is a **bounded knapsack** with K item types and capacity `cap_b`.
Solvable by standard DP in O(cap_b × K) time.

### Implementation spec

1. At each Lagrangian iteration, for each block `b`:
   - Solve the bounded knapsack pricing problem
   - If the resulting pattern is NOT already in `patterns[b]`:
     - Add it to the pool
     - Recompute its Level 3 cost via `eval_block_counts(bi, new_counts)`
     - Update `block_pattern_cost[bi]` accordingly

2. Keep the existing Lagrangian search unchanged — it now operates on a
   larger pattern pool but uses the same logic.

3. **Memory safety**: cap the total number of dynamically added patterns
   per block at `PAST_PRICING_MAX_ADDED_PER_BLOCK` (default: 32).

4. **Deduplication**: before adding a pattern, hash the count vector and
   check against existing patterns. Do not add duplicates.

### Important restrictions

- Do NOT change the subgradient step logic.
- Do NOT change the repair handoff logic.
- Do NOT change the Level 3 evaluator.
- Do NOT change the beam path.
- The pricing branch should be behind a flag:
  `PAST_BLOCK_REPAIR_LAGR_DYNAMIC_PRICING` (default: `1` — enabled).

### Previous attempt warning

Coder 1 implemented a first-cut version that caused segfaults on larger
rows. The root cause was likely unbounded pattern pool growth or missing
bounds checks on dynamically added patterns. The new implementation MUST:

- Cap pool growth per block
- Guard all array accesses for dynamically sized pools
- Be tested under ASAN before recording results

### Required validation rows

After implementation, rerun at minimum:

- `hard_k6_2345711 n=1000` (seed 0 AND seed 1)
- `hard_k8_irregular n=1000`
- `hard_k10_irregular n=1000`
- `medium_k6_dense n=1000`

All using `step1_exact_guided` mode.

### Success criterion

**Dynamic pricing is a success if:**

- At least 2 of the above rows show gap improvement > 20% relative
  (e.g., 0.0130% → 0.0104% or better)
- AND the Lagrangian reclaims the winner tag on at least 1 row
- AND no crashes or regressions on easy-arithmetic rows

**Dynamic pricing is a failure if:**

- No improvement on any row, OR
- Regressions (gaps widen or runs crash), OR
- The beam still dominates even with the enlarged pool

### If pricing succeeds

Proceed to Phase C (extended validation).

### If pricing fails

Skip Phase C. Go directly to Phase D (paper consolidation). The beam
is the best Level 2 method. Accept the 0.006–0.022% gaps and write the
paper.

---

## Phase C. Extended pricing validation (only if Phase B succeeds)

### Purpose

Run the expanded grid with pricing enabled to build paper-ready tables.

### Tasks

1. Rerun the full two-axis grid (all families, n=300 and n=1000) with
   pricing enabled, using `run_two_axis_grid.py`.

2. Verify that easy-arithmetic rows remain exact at Step 1 (no
   regression).

3. Record results in:
   - `csv/two_axis_grid/TWO_AXIS_GRID_20260412_phase3_pricing.csv`

4. Compute improvement deltas vs the Phase 2 baseline.

### Required comparison table

For the paper, produce a table like:

| Family | K | n | Baseline gap | Pricing gap | Δ | Winner |
|---|---|---|---|---|---|---|

### Success criterion

Pricing improves the hard/medium rows without regressing easy rows.

---

## Phase D. Paper consolidation (always executed)

### Purpose

Regardless of whether pricing helps, consolidate the paper-ready
evidence. The paper story does NOT depend on pricing success.

### Required deliverables

#### D1. Final two-axis summary table

Combine Phase 1 + Phase 2 (+ Phase 3 if applicable) into one clean
table showing:

| Family | Class | K | n | Gap | Winner | Frobenius | Density | Merged blocks |

#### D2. Non-monotonicity demonstration

Include one explicit comparison for the paper:

- K=10 {1..10} n=1000: exact at Step 1
- K=6 {2,3,4,5,7,11} n=1000: gap 0.0130%

State: "K=10 is easier than K=6 because the semigroup {1..10} is
complete, while {2,3,4,5,7,11} has bounded representability gaps."

#### D3. Level-by-level failure analysis

For each non-exact row, state which level is the bottleneck:

- Level 1 (relaxation): never — LB is always tight
- Level 2 (assignment): always — beam wins, gaps are assignment-driven
- Level 3 (within-block): resolved by exact per-block DP

#### D4. Arithmetic descriptors table

For each family, report:

| Family | K | has_1 | GCD | Multiplicity | Contiguous | Frobenius | Density(100) | Apéry max |

#### D5. Runtime scaling table

For each family class, show runtime vs n:

| Family | n=300 | n=1000 | n=1500 | n=2500 |

---

## Phase E. Update EXPERT_GUIDANCE.md and archive files

### Purpose

Record the outcome of this plan cleanly.

### Required updates

1. `LOG.md`: full chronological record
2. `RESULTS.md`: final summary tables
3. `BLOCKERS.md`: remaining open questions
4. `EXPERT_GUIDANCE.md`: update "latest implementation findings" section

---

## Decision rules at end of plan

### If pricing succeeded AND extended validation passed

The paper can claim:

- Dynamic pricing closes residual Level 2 gaps on hard arithmetic
- The adaptive pipeline is: semigroup relaxation → Lagrangian with
  pricing → per-block exact DP

### If pricing failed OR was skipped (Phase A said search gap)

The paper should claim:

- Beam search is the best available Level 2 method
- Remaining gaps are < 0.025% and assignment-driven
- Dynamic pricing / column generation is a promising future direction

Both stories are publishable. The paper's contribution is the two-axis
framework and the three-level decomposition, not the specific Level 2
solver.

---

## Fallback ladder

1. Pool/search diagnostic (Phase A) — always do this
2. Dynamic pricing (Phase B) — only if Phase A says pool gap
3. Extended validation (Phase C) — only if Phase B succeeds
4. Paper consolidation (Phase D) — always do this
5. Archive cleanup (Phase E) — always do this

---

## Exact files expected to change

Likely:

- `solvers/cpp/stateful_dp_solver.cpp` (diagnostic + pricing)
- `scripts/run_two_axis_grid.py` (new CSV output)
- `research/k_vs_arithmetic_axes_20260412/LOG.md`
- `research/k_vs_arithmetic_axes_20260412/RESULTS.md`
- `research/k_vs_arithmetic_axes_20260412/BLOCKERS.md`

New:

- `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260412_phase3_pricing.csv`

Should NOT change:

- `PROBLEM.md`, `IDEAS.md`, `LITERATURE.md` (framing, not execution)
- Level 3 evaluator code
- Level 1 relaxation code

---

## Reporting format

At the end, the coder should report:

1. Pool/search diagnostic result (counts per row)
2. Whether dynamic pricing was implemented
3. If yes: gap comparison table (baseline vs pricing)
4. Whether pricing changed the Level 2 winner on any row
5. Final two-axis summary table for the paper
6. Recommended paper narrative (pricing-inclusive or beam-only)

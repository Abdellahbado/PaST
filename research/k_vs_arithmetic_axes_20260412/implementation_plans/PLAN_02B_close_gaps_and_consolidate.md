# Plan 02B: Self-Contained Gap Closure + Paper Consolidation

## Status

Prepared on: `2026-04-13`

Replaces: Plan 02 Phases B–C (dynamic pricing skipped after Phase A
diagnostic showed search gap).

Depends on: Plan 01 (completed), Plan 02 Phase A (completed — diagnostic
tautological, see caveat).

---

## Critical correction: Phase A diagnostic was tautological

The beam (`block_repair_feasible_beam_ub`) and the Lagrangian both call
`generate_energy_core_patterns` to build the SAME pattern pool. The
check "is the beam's winning pattern in the Lagrangian's pool?" will
ALWAYS return yes by construction. This diagnostic does not distinguish
pool gap from search gap.

**The real pool-vs-search question** is: does the OPTIMAL Level 2
assignment use patterns outside the shared pool? We cannot answer this
without solving Level 2 exactly — which is what this plan does.

---

## Objective

1. **Solve Level 2 exactly on the hard rows** using our OWN self-contained
   branch-and-bound (no external/commercial solvers).
2. **Determine ground truth**: Where does the remaining gap live — is it
   Level 2 (beam found the wrong combination of existing patterns) or
   Level 1 (block profile from relaxation is slightly suboptimal)?
3. **Consolidate results into paper-ready tables.**

---

## Non-goals

1. Do NOT use any external solver (no Gurobi, CPLEX, CBC, PuLP, GLPK).
2. Do NOT change Level 1 or Level 3.
3. Do NOT implement dynamic pricing (deferred — diagnostic inconclusive).

---

## Phase A. Self-contained exact Level 2 solver

### Why this is tractable

The Level 2 problem after pattern generation is:

```
Select exactly one pattern p_b ∈ Pool_b for each block b = 1..B
Minimize  Σ_b cost[b][p_b]
Subject to  Σ_b counts_j[p_b] = n_j   for each type j = 1..K
```

Problem size:
- B = 8–20 blocks
- |Pool_b| ≈ 10–30 patterns per block
- K = 4–10 type constraints

This can be solved EXACTLY by a block-by-block branch-and-bound
with bounding:

- **Branch**: at block b, try each pattern p ∈ Pool_b
- **State**: residual type counts r_j = n_j - (types assigned so far)
- **Bound**: current cost + Σ_{remaining blocks b'} min_{p∈Pool_b'} cost[b'][p]
- **Prune**: if bound > best_known_UB, skip

The bound is cheap (precomputed min costs per block).

### Search space caveat

The worst-case search tree has |Pool|^B leaves. For B=20 and
|Pool|=20, that is 20^20 ≈ 10^26 — obviously intractable without
pruning. The approach relies on two pruning mechanisms:

1. **Suffix feasibility bounds** (suffix_min, suffix_max per type):
   These prune branches where the remaining blocks CANNOT deliver
   the needed type counts. Already computed in the beam function.
   These are extremely effective because the type constraints are
   tight (each type must reach exactly n_j).

2. **Cost bounding**: Current cost + suffix min cost > best_known
   prunes suboptimal branches early.

**Realistic expectations per row:**
- B=8–9 (K=4, K=6 rows): should complete in seconds
- B=14 (K=6 seed 1): may need 10–30s, likely feasible
- B=19–20 (K=8, K=10): may timeout at 30s — increase limit or
  accept the best incumbent found

**If B=19–20 times out**: this is still useful — record the best
solution found and whether it improved over the beam. The timeout
does NOT invalidate the diagnostic for B=8–14 rows.

### Architecture

Add a new function in `stateful_dp_solver.cpp`:

```cpp
double block_repair_exact_level2_ub(
    const std::vector<RecoveredBlock> &merged,
    const std::vector<int> &lengths,
    const std::vector<int> &totals,
    const std::vector<std::vector<RepairPattern>> &patterns,
    const std::vector<std::vector<double>> &block_pattern_cost,
    int K, int B)
```

Implementation:

1. Precompute `min_remaining_cost[b]` = min cost pattern for block b.
   Accumulate suffix sums.
2. Use recursive DFS with pruning:
   - At block `b`, try each pattern `p` in `Pool_b`
   - Check: for each type j, `r_j - counts_j[p] >= 0` (non-negative residual)
   - Check: for each type j, `r_j - counts_j[p] <= suffix_max[b+1][j]`
     (enough remaining capacity to use the surplus)
   - Check: `r_j - counts_j[p] >= suffix_min[b+1][j]`
     (enough remaining demand to fill minimum requirements)
   - Compute bound: `current_cost + cost[b][p] + suffix_min_cost[b+1]`
   - If bound > best_UB: prune
   - Recurse to block b+1
3. At block B (leaf): if all r_j == 0, update best_UB.

### Feasibility constraints (important)

The suffix_min and suffix_max arrays are already computed in the beam
function (lines 2888–2906 in `stateful_dp_solver.cpp`). Reuse the same
logic. These provide tight feasibility pruning:

- `counts_assigned_j + suffix_min_j <= n_j` (can we still fill minimum?)
- `counts_assigned_j + suffix_max_j >= n_j` (can we still reach target?)

### Time limit

Gate the search with `PAST_BLOCK_REPAIR_EXACT_L2_TIME` (default: 30s).
If the time limit expires, return the best solution found so far (which
may equal the beam's result or may already be better).

### Integration

Call this function AFTER the beam, using the beam's UB as the initial
best_known. The exact search will either:
- Confirm the beam is optimal (no better assignment exists in the pool)
- Find a better assignment
- Timeout (meaning Level 2 is genuinely hard at this size)

Gate behind `PAST_BLOCK_REPAIR_EXACT_L2` (default: `1` — enabled).

### Required validation

Run on:
- `hard_k4_irregular n=1000` (B=9, gap 0.0083%)
- `hard_k6_2345711 n=1000 seed=0` (B=8, gap 0.0130%)
- `hard_k6_2345711 n=1000 seed=1` (B=14, gap 0.0064%)
- `hard_k8_irregular n=1000` (B=19, gap 0.0157%)
- `hard_k10_irregular n=1000` (B=20, gap 0.0221%)
- `medium_k6_dense n=1000 seed=0` (B=9, gap 0.0138%)

Record for each: `exact_l2_ub`, `exact_l2_time`, `exact_l2_nodes`,
`exact_l2_closed` (boolean: did it finish within time limit?).

---

## Phase B. Large Neighborhood Search on the assignment (if Phase A
leaves gaps open)

### Why LNS is the right next step

If the exact Level 2 solver confirms beam = optimal-in-pool but gap
remains (meaning the pool is the ceiling), we need to explore patterns
OUTSIDE the pool. LNS does this without a full pricing framework:

1. **Destroy**: Unfix 2–3 blocks in the current best assignment
2. **Repair**: Re-fill those blocks by solving a small sub-knapsack
   (enumerate all feasible count vectors for 2–3 blocks with capacity
   constraint and residual type demands)
3. **Evaluate**: Use Level 3 exact DP to score the new assignment
4. **Accept**: If total cost improves, keep. Otherwise discard.

### Why this overcomes the pattern pool ceiling

The repair step does NOT use the pre-generated pattern pool. It
enumerates ALL feasible count vectors for the unfixed blocks by
solving a small bounded knapsack. This means:
- New patterns are discovered on the fly
- No pre-filtering limits apply
- The search is guided by the current assignment (warm-started)

### Implementation

```cpp
double block_repair_lns_ub(
    const std::vector<RecoveredBlock> &merged,
    const std::vector<int> &lengths,
    const std::vector<int> &totals,
    int K, int B,
    const std::vector<int> &initial_assignment,  // pattern indices
    double initial_cost,
    const SPACESResult &spaces)
```

Loop:
1. Select 2 blocks to unfix (try all pairs, or random subset)
2. For the 2 unfixed blocks b1, b2:
   - Compute residual type demands: r_j = n_j - (assigned by fixed blocks)
   - Enumerate all (c1, c2) such that:
     - Σ_j L_j × c1_j = cap_{b1}
     - Σ_j L_j × c2_j = cap_{b2}
     - c1_j + c2_j = r_j for all j
   - For each valid (c1, c2), evaluate Level 3 cost for both blocks
   - Keep the best (c1, c2)
3. If improvement found: update assignment, loop
4. If no improvement after trying all pairs: stop

### Enumeration cost for 2 blocks

For 2 blocks with capacity ~500 and K=6:
- The count vector for block 1 is fully determined by the capacity
  constraint and the residual demands: c2_j = r_j - c1_j
- So we only enumerate c1 (one block's count vector)
- This is a bounded knapsack enumeration: O(cap × K) states ≈ 3000
- Times Level 3 evaluation per state: varies (exact DP or sequence)
- Total: ~3000 × K evaluations per block pair ≈ 18K operations

Very cheap. Can try all B×(B-1)/2 pairs easily.

### Gate

`PAST_BLOCK_REPAIR_LNS` (default: `1`).
`PAST_BLOCK_REPAIR_LNS_MAX_PAIRS` (default: B*(B-1)/2, all pairs).
`PAST_BLOCK_REPAIR_LNS_TIME` (default: 30s).

---

## Phase C. Pricing check without external solver (diagnostic only)

### Purpose

Determine whether the pattern pool contains all USEFUL patterns, or
whether there exist beneficial patterns not in the pool.

### Method

After the Lagrangian converges, compute the dual prices λ_j from its
best iterate. For each block b, solve a bounded knapsack:

```
minimize  eval_block_counts(b, counts) - Σ_j λ_j × counts_j
subject to  Σ_j L_j × counts_j = cap_b
            0 ≤ counts_j ≤ remaining_j
```

This is the PRICING SUBPROBLEM. If the minimum reduced cost is
negative, there exists a beneficial pattern not in the pool.

### Implementation

Use a counting DP on the capacity dimension:
- State: remaining capacity c, type index i
- dp[c][i] = min reduced cost achievable using types i..K-1 with
  capacity c
- Transitions: for count_i = 0, 1, ..., min(remaining_i, c/L_i):
  dp[c][i] = min over count_i of:
    dp[c - count_i × L_i][i+1] + (partial eval contribution for type i)

The "partial eval contribution" per type is tricky because Level 3
costs are sequence-dependent. As an approximation, use the Lagrangian
proxy cost (which is separable by type) instead of exact Level 3.

### Output

Log: `PRICING_CHECK: block=b, best_reduced_cost=X, is_in_pool=Y`

If any block shows negative reduced cost for an out-of-pool pattern,
pool gap is confirmed.

### Gate

`PAST_BLOCK_REPAIR_PRICING_CHECK` (default: `0` — diagnostic only).

---

## Phase D. Paper consolidation (always executed)

### D1. Final two-axis summary table

Combine all phases. Include both beam UB and exact Level 2 UB (if
available):

```
| Family           | K  | n    | LB          | UB(beam)    | UB(exact L2)| Gap(beam)| Gap(exact)|
|------------------|----|------|-------------|-------------|-------------|----------|-----------|
| {1..10}          | 10 | 1000 | 16,137,402  | 16,137,402  | —           | 0.0000%  | —         |
| {3,5,7,11}       | 4  | 1000 | 63,952,923  | 63,958,209  | ???         | 0.0083%  | ???       |
| ...              |    |      |             |             |             |          |           |
```

### D2. Gap decomposition table

```
| Family       | K  | Total gap | Search gap         | Pool/structural gap |
|              |    | UB-LB     | UB(beam)-UB(exact) | UB(exact)-LB        |
|--------------|----|-----------|-------------------—|---------------------|
```

### D3. Non-monotonicity + arithmetic descriptors

Already specified in Plan 02. Consolidate.

### D4. Level-by-level bottleneck summary

Already specified. Include the exact Level 2 evidence.

---

## Appendix: Unexplored ideas ranked by promise

These are ideas we identified but have NOT implemented. Ranked by
estimated impact and feasibility:

### Rank 1: Block coordinate descent (HIGH promise, EASY to implement)

Fix all blocks except one. Re-optimize that one block by trying all
patterns in its pool. Iterate over all blocks. Repeat until convergence.

- This is cheaper than exact Level 2 (O(B × |Pool|) per pass vs
  exponential B&B)
- Converges to a local optimum of the assignment
- Starting from the beam's assignment, may find improvements
- VERY easy to implement: ~40 lines of C++

### Rank 2: Multi-block local improvement with k-opt (MEDIUM promise)

Instead of unfixing 2 blocks at a time (LNS), systematically try
all 3-block combinations. For B=14, there are C(14,3) = 364
combinations — each solved by exhaustive enumeration over the pattern
pool (|Pool|^3 ≈ 8000 options). Total: ~3 million evaluations, each
involving a pattern cost lookup. Very fast.

### Rank 3: Alternative block profiles from near-optimal DP states
(HIGH promise but complex)

The semigroup DP may have MULTIPLE optimal or near-optimal block
profiles. Currently we use only one. Explore alternative profiles
(e.g., merge two adjacent small blocks, split one large block)
and run the full Level 2+3 pipeline on each. The best across
profiles is the final UB. This addresses the "Level 1 quality"
concern from the expert guidance.

### Rank 4: Pattern generation by perturbation (MEDIUM promise)

Take existing patterns and generate new ones by small perturbations:
- Transfer 1 job of type j → type k while maintaining capacity
  (requires L_j = L_k or multi-job swaps that are capacity-neutral)
- This creates patterns NOT in the original pool
- Feed them through Level 3 evaluation
- Add the best to the pool and re-run exact Level 2

### Rank 5: Swap-chain local search at job level (LOW-MEDIUM promise)

Instead of pattern-level moves, work at the job level:
- Move groups of jobs between blocks along capacity-neutral chains
  (e.g., swap 5 type-3 jobs for 3 type-5 jobs between blocks b1 and b2,
  since 5×3 = 3×5 = 15 time units)
- Requires: p×L_j = q×L_k where p,q are the swap sizes
- Minimum swap: p = L_k/gcd(L_j,L_k), q = L_j/gcd(L_j,L_k)
- For L=3,L=5: swap 5-for-3 (moves 15 time units)

### Not recommended

- Full n-fold IP: too complex to implement, marginal practical benefit
- Commercial solvers: against project constraints
- Arc-flow: equivalent to pattern enumeration; already captured by pool

---

## Files expected to change

New:
- Self-contained exact Level 2 function in `stateful_dp_solver.cpp`
- Optional LNS function in `stateful_dp_solver.cpp`

Modified:
- `stateful_dp_solver.cpp` (new functions + integration in dispatch)
- Archive markdown files (LOG, RESULTS, BLOCKERS, EXPERT_GUIDANCE)

Should NOT change:
- Level 1 relaxation code
- Level 3 evaluator code
- Pattern generation code
- Beam search code

---

## Reporting format

At the end, the coder should report:

1. Exact Level 2 results: per-row UB and whether it matches the beam
2. Gap decomposition: search gap + structural gap
3. Whether LNS found any patterns outside the pool
4. Whether any row was fully closed (exact L2 UB = LB)
5. All paper-ready tables (D1–D4)
6. Appendix: which unexplored ideas were tested and with what result

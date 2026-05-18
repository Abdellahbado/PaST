# Plan 04B (Corrected): Type-Aware Pruning for Exact DP

## Status

Corrected on: `2026-04-14` after code review feedback.

Original Plan 04B analyzed the WRONG function (`solve_sparse_dp_stateful`
at line 925). The correct exact DP functions are:

- `solve_exact_multiset_dp` (dense, line ~7636)
- `solve_sparse_exact_multiset_dp` (sparse/hash, line ~8257)

The sparse variant ALREADY has four pruning mechanisms:
1. `pruned_bound` — work-only processing cost LB (lb_proc_cost)
2. `pruned_completion` — completion DP table bound
3. `pruned_relaxed` — relaxed DP reachability/cost bound
4. `pruned_dominance` — same-state cost dominance

So the starting point is NOT "one simple bound" as originally written.
The DP is already moderately sophisticated.

---

## What remains genuinely missing

After reviewing the actual exact DP code, BOTH the dense and sparse
variants use a **work-only** processing cost LB. The state `(t, rw)`
tells us the remaining work in time units but NOT which types remain.
All existing bounds (proc cost, completion, relaxed) operate on `(t, rw)`
— none of them use the type-decomposition of the remaining work.

This means one enhancement from the original plan is still valid and
genuinely NEW:

### The type-aware job-cost lower bound

The current `lb_proc_cost(t, rw)` computes: "cheapest way to schedule
`rw` generic time units starting from time `t`."

But we KNOW the remaining work is not generic. It consists of `c_j`
jobs of length `L_j` for each type j. A job of length L=11 CANNOT fit
in a gap smaller than 11 time slots. Its minimum cost is NOT 11 cheap
single-unit slots — it is the cheapest L=11 contiguous block.

**This is the one enhancement worth implementing first.**

---

## Corrected implementation: type-aware bound

### What the coder should implement

In BOTH `solve_exact_multiset_dp` and `solve_sparse_exact_multiset_dp`:

#### Step 1: Precompute per-type minimum costs

Before the main DP loop, for each job type `j` and each time `t`,
compute the minimum cost of scheduling one type-j job starting at
or after time `t`:

```cpp
// min_job_cost[j][t] = min over t_s >= t of
//   (processing_cost[t_s : t_s + L_j])
//   where the slot [t_s, t_s + L_j] is within [t, T]

std::vector<std::vector<double>> min_job_cost(K, std::vector<double>(T + 1, kInf));
for (int j = 0; j < K; ++j) {
    int L = lengths[j];
    // Fill from right to left (suffix minimum)
    for (int t = T - L; t >= 0; --t) {
        double cost_here = prefix_proc[t + L] - prefix_proc[t];
        min_job_cost[j][t] = std::min(cost_here, min_job_cost[j][t + 1]);
    }
}
```

This is O(K × T) precomputation. Negligible.

#### Step 2: Compute the type-aware LB in the pruning check

In the transition loop, after computing `new_s` (the new count state),
extract the remaining counts and compute:

```cpp
// For dense DP: counts are directly available from state_counts
// For sparse DP: decode from mixed-radix state key

double lb_type_aware = 0.0;
for (int j = 0; j < K; ++j) {
    int remaining_j = totals[j] - new_counts[j];
    lb_type_aware += remaining_j * min_job_cost[j][t_e];
}
lb_type_aware += min_c_end_from[earliest_end];

// Take max of existing bound and type-aware bound
double lb = std::max(lb_existing, cost + lb_type_aware);
```

**For the dense DP** (`solve_exact_multiset_dp`):
The counts are already in `state_counts[s * K + j]`, so `new_counts[j]`
is `state_counts[new_s * K + j]`. NO extra decoding needed.

**For the sparse DP** (`solve_sparse_exact_multiset_dp`):
The state key must be decoded to get counts. This is already done
elsewhere (e.g., for the bank restart). Alternatively, precompute
`per_state_lb_type[s]` for all states — but this is O(NC × K) which
may be too large. Instead, decode on the fly for the sparse variant
only when needed (the decode is K divisions, ~6 operations).

#### Step 3: Add diagnostic counter

```cpp
if (lb_type_aware > best + kEps && lb_existing <= best + kEps)
    g_last_exact_dp_diag.pruned_type_aware += 1.0;
```

This tells us how many EXTRA states the type-aware bound prunes that
the existing bounds did not catch.

### Where to insert

In `solve_exact_multiset_dp` (dense), the pruning check is at lines
~8058, ~8132, ~8192. Add the type-aware bound as a `std::max` with
the existing `lb`.

In `solve_sparse_exact_multiset_dp` (sparse), the pruning checks are
at lines ~8446, ~8537, ~8635. Same: add `std::max` with existing `lb`.

---

## Corrected implementation: incumbent-guided expansion order

### What the coder should implement

In the sparse exact DP only (the dense DP processes states in fixed
order by index), sort each time layer's states before processing them.

After collecting all states at time `t_end` in `dp_maps[t_end]`,
extract them into a vector, sort by `cost + lb_proc_cost(t_end, rw)`,
and process in that order.

This causes the DP to find better completions EARLIER, tightening
`best` sooner and enabling more `lb > best` pruning of later states
within the same and subsequent layers.

```cpp
// At the start of processing dp_maps[t_end]:
std::vector<std::pair<int64_t, double>> sorted_states(
    dp_maps[t_end].begin(), dp_maps[t_end].end());
std::sort(sorted_states.begin(), sorted_states.end(),
    [&](const auto &a, const auto &b) {
        int rwa = state_rw(a.first);
        int rwb = state_rw(b.first);
        return (a.second + lb_proc_cost(rwa)) <
               (b.second + lb_proc_cost(rwb));
    });
// Then iterate over sorted_states instead of dp_maps[t_end]
```

**Cost**: O(|layer| log |layer|) per time step. Only worth doing for
large layers. Gate behind: `if (dp_maps[t_end].size() > 1000)`.

---

## What NOT to implement (from original Plan 04B)

### Block-structure suffix LB: WITHDRAWN

The coder correctly identified that the recovered block profile is
NOT a hard constraint on the true optimum. The global exact DP searches
over ALL valid schedules, not just those matching the recovered profile.
Therefore "sum of cheapest future blocks" is NOT an admissible lower
bound for the global exact DP.

**Do not implement this as a pruning bound.**

### Componentwise dominance: DEFERRED

Theoretically valid but complex to implement correctly in the current
state representation. Defer until the type-aware bound is measured.

### Semigroup suffix feasibility: DEFERRED

Too much engineering for uncertain gains. The relaxed DP bound already
captures some of this information.

### Two-pass beam→exact: ALREADY THE PIPELINE

Step 3 (beam) → Step 4 (exact DP) is already the method. Not a new
enhancement.

---

## Success criterion

After implementing the type-aware bound:

1. Run on the hard K=6 rows and report `pruned_type_aware` count
2. Compare total `states_expanded` with and without the bound
3. If ≥10% more states pruned: the bound is valuable
4. If <5%: the existing bounds already capture most of this information

The incumbent-guided ordering should be measured by: does `best` improve
earlier in the time sweep? Compare the time step at which `best` first
reaches its final value, with and without ordering.

---

## Files to modify

- `stateful_dp_solver.cpp`:
  - `solve_exact_multiset_dp` (~line 7636): add min_job_cost precompute
    + type-aware LB check
  - `solve_sparse_exact_multiset_dp` (~line 8257): add min_job_cost
    precompute + type-aware LB check + optional layer sorting
  - `ExactDPDiag` struct: add `pruned_type_aware` field

Should NOT modify:
- Any other solver function
- Step 3 code
- SPACES computation

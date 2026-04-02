#pragma once

#include "dp_solver.hpp"

#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace dp
{

    enum class RelaxationMode
    {
        Unit,
        Gcd,
        Semigroup,
    };

    struct MachineStateConfig
    {
        std::vector<std::string> states;
        int off_idx = -1;
        int proc_idx = -1;
        std::vector<std::vector<int>> t_trans;
        std::vector<std::vector<double>> p_trans;

        static MachineStateConfig paper_nosby();
    };

    struct SPACESResult
    {
        std::vector<double> c_star;
        std::vector<double> c_start;
        std::vector<double> c_end;
        double p_proc = 0.0;
        int early = 0;
        int late = 0;
        int h = 0;
        int max_gap = -1;
        bool banded = false;

        double gap_cost(int t_end, int t_start) const noexcept;
    };

    struct StatefulParent
    {
        int prev_t_end = -1;
        int64_t prev_state = 0;
        int length = 0;
        int t_start = 0;
    };

    MachineStateConfig make_paper_nosby_config();
    MachineStateConfig make_paper_twosby_config();
    std::vector<double> build_proc_prefix(const std::vector<double> &prices, double p_proc);
    SPACESResult compute_spaces(
        const std::vector<double> &prices,
        const MachineStateConfig &config,
        int max_gap = -1);

    DPResult solve_sparse_dp_stateful(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        const DPParams &params = {});

    // Compute a conservative max useful gap between consecutive processing
    // segments. Beyond this gap, shutdown+restart is always cheaper than idling.
    // Uses price range to determine worst-case idle break-even.
    int auto_max_gap(const MachineStateConfig &config, int h,
                     const std::vector<double> &prices);

    // Compute optimal TEC for a fixed job sequence using the job-interval graph DP.
    // O(n * h * max_gap) — very fast.
    double solve_fixed_sequence(
        const std::vector<int> &sequence,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces);

    // Compute a good upper bound by trying heuristic sequences.
    double compute_initial_ub(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        int n_random = 50,
        double known_lb = 0.0);

    // Relaxed DP lower bound: state = (t_end, remaining_work) only.
    // Drops the per-type job count constraint, allowing any job length at each step.
    // Valid LB because it is strictly more permissive than the exact problem.
    // O(T * rw * max_gap * K) ≈ 80M ops for benchmark instances (~1s).
    double solve_relaxed_dp_lb(
        const std::vector<int> &lengths,
        int total_rw,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        RelaxationMode mode = RelaxationMode::Semigroup);

    // Combined: forward relaxed DP LB + bin-packing UB in a single pass.
    // Saves one full relaxed DP computation vs running them separately.
    struct RelaxedDPResult
    {
        double lb;
        double bin_pack_ub;
        int64_t states_reached = 0;
        int64_t states_expanded = 0;
        // Expose relaxed DP cost table for smart reconstruction
        std::vector<double> rdp;  // dp[(T+2) * RW], flattened 2D array
        int RW = 0;               // row width = total_rw + 1
        int block_count = 0;
        int merged_block_count = 0;
        std::string pack_solver = "default";
        std::string pack_external_status = "disabled";
        std::string pack_method = "none";
        std::string pack_outcome = "not_attempted";
        double t_pack_external = 0.0;
        double t_pack_heuristic = 0.0;
        double t_pack_dfs = 0.0;
        double t_pack_block_dp = 0.0;
    };
    RelaxedDPResult solve_relaxed_dp_with_binpack(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces);

    // Same as solve_relaxed_dp_with_binpack, but the relaxed path is computed
    // with the strengthened R_feas transition filter instead of plain
    // semigroup reachability. This is useful when the question is not only
    // "how strong is the lower bound?" but also "does the recovered relaxed
    // block profile pack with the real multiset?".
    RelaxedDPResult solve_relaxed_dp_lb_feas_with_binpack(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces);

    // One-tracked-type version of R_partial with recovered-block packability.
    // The tracked type is chosen explicitly or via the same automatic
    // critical-type selector used by solve_relaxed_dp_lb_partial.
    RelaxedDPResult solve_relaxed_dp_lb_partial_with_binpack(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        std::vector<int> tracked_types = {},
        int max_auto_tracked = 1,
        bool use_remainder_feas = true);

    // Backward relaxed LB: reverse time, take max(forward, backward).
    // Accepts fwd_config to generically reverse any machine (not just NOSBY).
    double solve_relaxed_dp_lb_backward(
        const std::vector<int> &lengths,
        int total_rw,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        const MachineStateConfig &fwd_config);

    // Bin-packing UB: run relaxed DP with schedule tracking, extract blocks,
    // FFD-pack actual jobs into blocks, evaluate the packed sequence.
    double bin_packing_ub(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces);

    // Smart reconstruction: count-aware path search using rdp table for pruning.
    // Searches for count-feasible paths through the relaxed DP cost table.
    // Returns the cost of the best count-feasible schedule, or kInf if none found.
    double smart_reconstruct(
        const std::vector<double> &rdp,
        int RW,
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        double known_ub = kInf,
        double time_limit_sec = 30.0);

    // Local search: pairwise swap hill climbing on a sequence.
    // Modifies best_seq in place. Returns improved cost.
    double local_search_ub(
        std::vector<int> &best_seq,
        double best_cost,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        int max_passes = 3,
        double time_budget_sec = 0.5);

    // Two-class relaxed LB: state = (t_end, rw_small, rw_large).
    // Tighter than single-class by preventing cross-class substitution.
    double solve_relaxed_dp_lb_two_class(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        int threshold = 2);

    // Exact multiset DP for small K: state = (t_end, c0, c1, ..., c_{K-1}).
    // Dense array. Returns the exact optimal cost (both LB and UB).
    // Skips (returns kInf) if state space is too large.
    double solve_exact_multiset_dp(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        double known_ub = kInf,
        double time_limit_sec = 120.0);

    // Sparse exact multiset DP for larger K: uses hash maps instead of dense
    // arrays. Slower per-state but handles much larger state spaces.
    // Falls back when dense approach is infeasible.
    double solve_sparse_exact_multiset_dp(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        double known_ub = kInf,
        double time_limit_sec = 300.0,
        const std::vector<double> *relaxed_dp = nullptr,
        int relaxed_RW = 0,
        double relaxed_lb = kInf);

    // ── New LB hierarchy bounds ──────────────────────────────────────────

    // R_feas: relaxed DP (t, rw) with bounded-work and two-sided
    // transition-feasibility filtering.
    // At state (t, rw), a transition by type j is allowed only if:
    //   1) the already-placed work W-rw is reachable with bounded counts,
    //   2) it can be explained while leaving one type-j job unused, and
    //   3) after taking type j, the remaining work rw-L_j is still achievable
    //      under the residual bounded counts.
    // This remains a lower bound (still merges count vectors), but is
    // strictly tighter than the original one-sided R_feas.
    double solve_relaxed_dp_lb_feas(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces);

    // R_Lagr: Lagrangian relaxation of per-type count constraints.
    // Penalizes over-use of each type via multipliers λ_j ≥ 0.
    // Each iteration solves a modified R_semi with per-type cost offsets.
    // Converges to the tightest bound achievable from (t, rw) state space.
    double solve_relaxed_dp_lb_lagrangian(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        int max_iters = 50,
        double time_limit_sec = 5.0);

    // R_feas+Lagr: Combined bound — Lagrangian relaxation with
    // transition-feasibility filtering. Strictly dominates both R_feas
    // and R_Lagr individually.
    double solve_relaxed_dp_lb_feas_lagrangian(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        int max_iters = 50,
        double time_limit_sec = 5.0);

    // R_partial: Partial count-vector relaxation.
    // Tracks 1-2 critical type counts exactly while relaxing the rest
    // to remaining work. The relaxed remainder is still filtered through
    // bounded/two-sided feasibility checks, yielding a tighter hybrid
    // relaxation than plain R_partial or R_feas alone.
    //
    // State: (t_end, c_tracked[0], [c_tracked[1],] rw_rest)
    //   - c_tracked[i] is the exact count of tracked type i placed so far
    //   - rw_rest is the remaining work from non-tracked types
    //
    // Hierarchy:  R_feas ≤ R_partial(1) ≤ R_partial(2) ≤ Exact
    //
    // tracked_types: indices of types to track exactly.
    //   If empty, auto-selects up to max_auto_tracked critical types using
    //   one relaxed overuse diagnostic pass, then falls back to scarcity.
    //   If size 1: state is 3D (t, c, rw_rest)
    //   If size 2: state is 4D (t, c0, c1, rw_rest)
    //   Max 2 tracked types supported.
    // use_remainder_feas controls whether the untracked remainder uses the
    // bounded/two-sided feasibility filter (true) or plain remaining-work
    // transitions (false).
    double solve_relaxed_dp_lb_partial(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        std::vector<int> tracked_types = {},
        double time_limit_sec = 20.0,
        int max_auto_tracked = 2,
        bool use_remainder_feas = true);

    // Precompute bounded-knapsack feasibility sets A_j^- for each type j.
    // A_j^-[w] = true iff work amount w is achievable with a_j ≤ n_j - 1.
    // Returns K vectors of size (W+1).
    std::vector<std::vector<bool>> compute_feas_sets(
        const std::vector<int> &lengths,
        const std::vector<int> &totals);

    // Precompute the bounded achievable-work set
    // A = {w : ∃ allocation summing to w with a_i ≤ n_i}.
    std::vector<bool> compute_bounded_work_set(
        const std::vector<int> &lengths,
        const std::vector<int> &totals);

} // namespace dp

#pragma once

#include "dp_solver.hpp"

#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace dp
{

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
        const SPACESResult &spaces);

    // Combined: forward relaxed DP LB + bin-packing UB in a single pass.
    // Saves one full relaxed DP computation vs running them separately.
    struct RelaxedDPResult
    {
        double lb;
        double bin_pack_ub;
    };
    RelaxedDPResult solve_relaxed_dp_with_binpack(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces);

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
        double time_limit_sec = 300.0);

} // namespace dp
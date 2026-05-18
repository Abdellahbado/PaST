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
        MachineStateConfig config;
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
    // Stateless special case: free idling, no startup/shutdown costs, processing
    // power = 1 so the prefix is the raw price prefix.
    SPACESResult make_stateless_spaces(int T);
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
        double known_lb = 0.0,
        std::string *out_best_policy = nullptr,
        int *out_finite_candidates = nullptr,
        double *out_time_to_first_ub_sec = nullptr,
        std::vector<int> *out_best_seq = nullptr);

    // PLAN33: polish a given sequence with local search (swap hill climbing).
    double polish_best_sequence_ub(
        std::vector<int> &best_seq,
        double best_ub,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        double time_budget_sec = 2.0);

    // Compute a good upper bound by partitioning jobs across M parallel machines.
    // Each machine independently schedules its subset via solve_fixed_sequence.
    // Tries several load-balancing partition policies (LPT, SPT, alternating,
    // round-robin, random). Returns first feasible total cost, or kInf.
    double compute_parallel_initial_ub(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        int M,
        int n_random = 50,
        double known_lb = 0.0,
        std::string *out_policy = nullptr,
        int *out_machines_used = nullptr,
        int *out_failed_machines = nullptr);

    double guided_completion_ub(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        const std::vector<double> &completion_dp,
        int completion_RW,
        int completion_rw_scale,
        int n_rollouts = 8,
        int top_k = 4);

    double completion_guided_beam_ub(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        const std::vector<double> &completion_dp,
        int completion_RW,
        int completion_rw_scale,
        double known_ub = kInf,
        int beam_width = 256,
        double time_limit_sec = 30.0);

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
        std::vector<Segment> block_profile;
        std::vector<Segment> merged_blocks;
        // Expose relaxed DP cost table for smart reconstruction
        std::vector<double> rdp;  // dp[(T+2) * RW], flattened 2D array
        int RW = 0;               // row width = total_rw + 1
        int block_count = 0;
        int merged_block_count = 0;
        int merged_gcd_bad_count = 0;
        int merged_local_unreachable_count = 0;
        std::string pack_solver = "default";
        std::string pack_external_status = "disabled";
        std::string pack_method = "none";
        std::string pack_outcome = "not_attempted";
        std::string merged_caps_signature;
        std::string merged_bad_caps_signature;
        double t_pack_external = 0.0;
        double t_pack_heuristic = 0.0;
        double t_pack_dfs = 0.0;
        double t_pack_block_dp = 0.0;
        double t_pack_profile_recovery = 0.0;
        double t_pack_merge_blocks = 0.0;
        double t_pack_to_first_candidate = 0.0;
        double t_pack_ffd_only = 0.0;
        int step2_reached = 0;
        int step2_produced_ub = 0;
        // PLAN15 dense-unit diagnostics (forward-relax/profile split)
        double t_dense_spaces_or_lb = 0.0;
        double t_dense_profile_dp = 0.0;
        double t_dense_profile_recovery = 0.0;
        double t_dense_block_build = 0.0;
        double t_dense_job_materialization = 0.0;
        double t_dense_step2_pack = 0.0;
        double t_dense_pre_step2_total = 0.0;
        int pack_profiles_tried = 0;
        int pack_co_optimal_profiles = 0;
        double block_dp_state_space = 0.0;
        double block_dp_total_compositions = 0.0;
        double block_dp_total_comp_estimate = 0.0;
        double block_dp_max_comp_estimate = 0.0;
        double block_dp_max_compositions_per_block = 0.0;
        std::string block_dp_status = "not_attempted";
        int block_dp_timed_out = 0;
        double beam_ub_for_exact_l2 = kInf;
        double exact_l2_ub = kInf;
        double t_exact_l2 = 0.0;
        double exact_l2_nodes = 0.0;
        int exact_l2_closed = 0;
        int exact_l2_improved_over_beam = 0;
        int exact_l2_beam_optimal_in_pool = 0;
        std::string exact_l2_status = "not_attempted";
        double profile_beam_base_width = 0.0;
        double profile_beam_avg_width = 0.0;
        double profile_beam_max_width = 0.0;
        double profile_beam_states_considered = 0.0;
        double profile_beam_states_kept = 0.0;
        double profile_beam_pruned_over = 0.0;
        double profile_beam_pruned_suffix = 0.0;
        double profile_beam_pruned_discrepancy = 0.0;
        int profile_beam_discrepancy_budget = 0;
        int profile_beam_discrepancy_depth = 0;
        std::string profile_beam_status = "not_attempted";
        int profile_beam_timed_out = 0;
        std::string profile_beam_key_multi_policy = "off";
        int profile_beam_key_multi_max = 1;
        double profile_beam_key_multi_score_eps = 0.0;
        double profile_beam_key_multi_diversity_eps = 0.0;
        // PLAN27 residual-aware + late ambiguity diagnostics
        std::string profile_beam_score_policy = "default";
        double profile_beam_residual_weight = 0.0;
        double profile_beam_residual_mean_penalty = 0.0;
        double profile_beam_residual_max_penalty = 0.0;
        double profile_beam_late_frac = 0.0;
        int profile_realization_hardest_first = 0;
        int profile_realization_exact_suffix_prune = 0;
        double t_pack_profile_beam = 0.0;
        double t_pack_block_dp_exact = 0.0;
        double profile_step2_ub = kInf;
        double profile_beam_candidate_ub = kInf;
        double profile_beam_plus_candidate_ub = kInf;
        double profile_exact_candidate_ub = kInf;
        int profile_beam_improved_over_step2 = 0;
        int profile_exact_improved_over_step2 = 0;
        std::string profile_incumbent_source = "auto";
        double profile_incumbent_ub_for_exact = kInf;
        std::string profile_selector_policy = "off";
        std::string profile_selector_decision = "legacy";
        std::string profile_selector_reason = "selector_disabled";
        int profile_selector_has_one = 0;
        int profile_selector_contiguous = 0;
        int profile_selector_multiplicity = 0;
        double profile_selector_semigroup_density = 0.0;
        int profile_selector_hard_alarm = 0;
        int profile_exact_primary_fallback_to_beam = 0;
        std::string profile_exact_primary_status_before_fallback = "not_applicable";
        std::string profile_step3_incumbent_mode = "not_applicable";
        int dense_unit_fastpath_active = 0;
        int count_based_ffd_active = 0;
        int dense_unit_relax_fastpath_active = 0;
        int dense_unit_energy_profile_active = 0;
        int dense_unit_relax_fastpath_fallback = 0;
        int dense_unit_energy_profile_fallback = 0;
        std::string dense_unit_relax_mode = "none";
        double ec_generated_patterns_total = 0.0;
        double ec_generated_patterns_max_block = 0.0;
        double ec_retained_patterns_total = 0.0;
        double ec_retained_patterns_max_block = 0.0;
        std::string ec_retained_patterns_signature;
        double ec_time_completion = 0.0;
        double ec_time_pattern_generation = 0.0;
        double ec_time_exact_core = 0.0;
        double ec_pruned_core_window = 0.0;
        double ec_pruned_suffix = 0.0;
        double ec_pruned_transition = 0.0;
        double ec_pruned_bound = 0.0;
        int ec_delta_used = -1;
        int ec_fixed_blocks = 0;
        int ec_two_phase_used = 0;
        double ec_phase1_feasible_ub = kInf;
        double ec_time_phase1 = 0.0;
        // PLAN24: beam chosen counts for exact corridor
        std::vector<std::vector<int>> profile_beam_chosen_counts;
        std::vector<int> profile_beam_block_order;
        // PLAN28: block-realizability diagnostics
        int block_realiz_diag_active = 0;
        int block_realiz_blocks_total = 0;
        int block_realiz_bad_blocks = 0;
        double block_realiz_bad_rate = 0.0;
        int block_realiz_first_bad_block = -1;
        double block_realiz_min_finite_patterns = 0.0;
        double block_realiz_mean_finite_patterns = 0.0;
        int block_realiz_base_path_survives = 0;
        std::string block_realiz_base_reject_reason = "not_run";
        double block_realiz_diag_time_sec = 0.0;
        int block_realiz_diag_skipped = 0;
        std::string block_realiz_diag_skip_reason;
        // Per-block array (CSV-safe concatenation for raw output)
        std::string block_realiz_per_block_payload;
        // PLAN29: multi-view block reconstruction diagnostics
        std::string block_view_policy = "baseline";
        int block_view_original_blocks = 0;
        int block_view_final_blocks = 0;
        int block_view_removed_boundaries = 0;
        int block_view_target_b = 0;
        int block_view_price_preserve_used = 0;
        int block_view_arith_adaptive_used = 0;
        int block_view_selected = 0;
        int block_view_eval_count = 0;
        double block_view_best_ub = kInf;
        double block_view_time_sec = 0.0;
    };

    struct ExactDPDiagnostics
    {
        std::string mode = "none";
        std::string variant = "p0";
        double initial_ub = kInf;
        double final_ub = kInf;
        double elapsed_sec = 0.0;
        double states_reached = 0.0;
        double states_expanded = 0.0;
        double pruned_bound = 0.0;
        double pruned_relaxed = 0.0;
        double pruned_completion = 0.0;
        double pruned_type_aware = 0.0;
        double pruned_dominance = 0.0;
        int timed_out = 0;
        int exhaustive = 0;
        // PLAN24 corridor diagnostics
        int corridor_enabled = 0;
        int corridor_delta = 0;
        double corridor_pruned = 0.0;
        int corridor_infeasible = 0;
        // PLAN24B corridor force-entry diagnostics
        std::string stop_reason = "none";
    };

    ExactDPDiagnostics consume_last_exact_dp_diagnostics();

    // PLAN24: beam-guided exact corridor
    struct ExactCorridor
    {
        bool enabled = false;
        int delta = 0;
        std::vector<int> prefix_work;                // size = B+1
        std::vector<std::vector<int>> prefix_counts; // size = B+1, each size K
    };
    void set_exact_corridor(const ExactCorridor &corridor);
    void clear_exact_corridor();

    // PLAN25/PLAN26: local corridor DP diagnostics
    struct LocalCorridorDiag
    {
        int enabled = 0;
        int delta = 0;
        std::string status = "not_attempted";
        int layers = 0;
        int64_t states_seen = 0;
        int states_kept_max = 0;
        int64_t states_pruned = 0;
        int64_t transitions_considered = 0;
        int64_t transitions_kept = 0;
        double time_sec = 0.0;
        double best_ub = kInf;
        int closed = 0;
        std::string stop_reason = "none";
        int memory_safe = 1;
        // PLAN26 alignment / validation diagnostics
        int beam_counts_size = 0;
        int merged_blocks = 0;
        int block_profile_blocks = 0;
        int block_count_mismatch = 0;
        int target_offset_l1 = 0;
        int target_in_corridor = 0;
        int base_candidates_finite = 0;
        int empty_candidate_blocks = 0;
        int first_empty_layer = -1;
        int base_path_survives = 0;
        double base_path_cost = kInf;
        std::string base_path_reject_reason = "none";
    };

    double beam_corridor_local_dp(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        const std::vector<Segment> &merged_seg,
        const std::vector<std::vector<int>> &beam_chosen_counts,
        const std::vector<int> &block_order,
        double known_ub,
        double time_limit_sec,
        int delta,
        LocalCorridorDiag &diag);

    struct RelaxedTableResult
    {
        std::vector<double> rdp;
        std::vector<double> off_rdp;
        int RW = 0;
        int rw_scale = 1;
        double lb = kInf;
    };

    RelaxedTableResult compute_relaxed_dp_table(
        const std::vector<int> &lengths,
        int total_rw,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        RelaxationMode mode = RelaxationMode::Semigroup);

    RelaxedTableResult compute_relaxed_completion_table(
        const std::vector<int> &lengths,
        int total_rw,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        RelaxationMode mode = RelaxationMode::Semigroup);

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
        double relaxed_lb = kInf,
        const std::vector<double> *completion_dp = nullptr,
        int completion_RW = 0,
        int completion_rw_scale = 1);

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

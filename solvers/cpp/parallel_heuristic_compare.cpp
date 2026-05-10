#include "dp_solver.hpp"
#include "stateful_dp_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{

constexpr double kInf = dp::kInf;

std::vector<double> read_values_as_double(const std::string &path)
{
    std::ifstream f(path);
    std::vector<double> out;
    double x = 0.0;
    while (f >> x)
        out.push_back(x);
    return out;
}

std::vector<int> read_values_as_int(const std::string &path)
{
    std::ifstream f(path);
    std::vector<int> out;
    double x = 0.0;
    while (f >> x)
        out.push_back(static_cast<int>(std::llround(x)));
    return out;
}

struct PaperInstance
{
    std::vector<int> jobs;
    std::vector<double> prices;
    std::vector<double> rates;
};

PaperInstance load_paper_instance(int instance_id, const std::string &data_dir)
{
    const std::string suffix = std::to_string(instance_id);
    PaperInstance inst;
    inst.jobs = read_values_as_int(data_dir + "/Data_p" + suffix + ".txt");
    inst.prices = read_values_as_double(data_dir + "/Data_c" + suffix + ".txt");
    inst.rates = read_values_as_double(data_dir + "/Data_e" + suffix + ".txt");
    return inst;
}

std::string join_ints(const std::vector<int> &v)
{
    std::ostringstream oss;
    for (std::size_t i = 0; i < v.size(); ++i)
    {
        if (i)
            oss << ';';
        oss << v[i];
    }
    return oss.str();
}

std::string join_doubles(const std::vector<double> &v)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    for (std::size_t i = 0; i < v.size(); ++i)
    {
        if (i)
            oss << ';';
        if (v[i] >= kInf * 0.5)
            oss << "inf";
        else
            oss << v[i];
    }
    return oss.str();
}

std::string join_strings(const std::vector<std::string> &v)
{
    std::ostringstream oss;
    for (std::size_t i = 0; i < v.size(); ++i)
    {
        if (i)
            oss << ';';
        oss << v[i];
    }
    return oss.str();
}

struct Assignment
{
    bool feasible = false;
    std::vector<std::vector<int>> machine_job_lengths;
    std::vector<int> machine_loads;
};

#include "phaseW_assignment_policies.hpp"

Assignment build_lpt_greedy_assignment(
    const std::vector<int> &jobs,
    const std::vector<double> &rates,
    const std::vector<double> &prices,
    int epsilon)
{
    const int m = static_cast<int>(rates.size());
    Assignment out;
    out.feasible = false;
    out.machine_job_lengths.assign(static_cast<std::size_t>(m), {});
    out.machine_loads.assign(static_cast<std::size_t>(m), 0);

    std::vector<int> order(static_cast<std::size_t>(jobs.size()), 0);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b)
              {
                  if (jobs[static_cast<std::size_t>(a)] != jobs[static_cast<std::size_t>(b)])
                      return jobs[static_cast<std::size_t>(a)] > jobs[static_cast<std::size_t>(b)];
                  return a < b;
              });

    for (int jid : order)
    {
        const int p = jobs[static_cast<std::size_t>(jid)];
        int best_h = -1;
        double best_delta = kInf;

        for (int h = 0; h < m; ++h)
        {
            const int load = out.machine_loads[static_cast<std::size_t>(h)];
            if (load + p > epsilon)
                continue;

            double delta = 0.0;
            for (int t = load; t < load + p; ++t)
                delta += rates[static_cast<std::size_t>(h)] * prices[static_cast<std::size_t>(t)];

            if (delta + 1e-12 < best_delta)
            {
                best_delta = delta;
                best_h = h;
            }
            else if (std::fabs(delta - best_delta) <= 1e-12)
            {
                if (best_h < 0 || out.machine_loads[static_cast<std::size_t>(h)] < out.machine_loads[static_cast<std::size_t>(best_h)] ||
                    (out.machine_loads[static_cast<std::size_t>(h)] == out.machine_loads[static_cast<std::size_t>(best_h)] && h < best_h))
                {
                    best_h = h;
                }
            }
        }

        if (best_h < 0)
            return out;

        out.machine_job_lengths[static_cast<std::size_t>(best_h)].push_back(p);
        out.machine_loads[static_cast<std::size_t>(best_h)] += p;
    }

    out.feasible = true;
    return out;
}

Assignment build_lpt_greedy_assignment_randomized(
    const std::vector<int> &jobs,
    const std::vector<double> &rates,
    const std::vector<double> &prices,
    int epsilon,
    std::mt19937_64 &rng,
    int rcl_size)
{
    const int m = static_cast<int>(rates.size());
    Assignment out;
    out.feasible = false;
    out.machine_job_lengths.assign(static_cast<std::size_t>(m), {});
    out.machine_loads.assign(static_cast<std::size_t>(m), 0);

    std::vector<int> order(static_cast<std::size_t>(jobs.size()), 0);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b)
              {
                  if (jobs[static_cast<std::size_t>(a)] != jobs[static_cast<std::size_t>(b)])
                      return jobs[static_cast<std::size_t>(a)] > jobs[static_cast<std::size_t>(b)];
                  return a < b;
              });

    for (int jid : order)
    {
        const int p = jobs[static_cast<std::size_t>(jid)];
        struct Candidate
        {
            int h = -1;
            double delta = kInf;
            int load = std::numeric_limits<int>::max();
        };

        std::vector<Candidate> cand;
        cand.reserve(static_cast<std::size_t>(m));
        for (int h = 0; h < m; ++h)
        {
            const int load = out.machine_loads[static_cast<std::size_t>(h)];
            if (load + p > epsilon)
                continue;

            double delta = 0.0;
            for (int t = load; t < load + p; ++t)
                delta += rates[static_cast<std::size_t>(h)] * prices[static_cast<std::size_t>(t)];

            cand.push_back(Candidate{h, delta, load});
        }

        if (cand.empty())
            return out;

        std::sort(cand.begin(), cand.end(), [](const Candidate &a, const Candidate &b)
                  {
                      if (std::fabs(a.delta - b.delta) > 1e-12)
                          return a.delta < b.delta;
                      if (a.load != b.load)
                          return a.load < b.load;
                      return a.h < b.h;
                  });

        const int keep = std::max(1, std::min(rcl_size, static_cast<int>(cand.size())));
        std::uniform_int_distribution<int> pick(0, keep - 1);
        const int chosen = pick(rng);
        const int best_h = cand[static_cast<std::size_t>(chosen)].h;

        out.machine_job_lengths[static_cast<std::size_t>(best_h)].push_back(p);
        out.machine_loads[static_cast<std::size_t>(best_h)] += p;
    }

    out.feasible = true;
    return out;
}

double sequence_preserving_esr_cost(
    const std::vector<int> &sequence,
    const std::vector<double> &prices,
    int epsilon,
    double rate)
{
    if (sequence.empty())
        return 0.0;

    const int U = static_cast<int>(sequence.size());
    int total_work = 0;
    for (int p : sequence)
        total_work += p;
    if (total_work > epsilon)
        return kInf;

    std::vector<double> prefix(static_cast<std::size_t>(epsilon) + 1, 0.0);
    for (int t = 0; t < epsilon; ++t)
        prefix[static_cast<std::size_t>(t) + 1] = prefix[static_cast<std::size_t>(t)] + rate * prices[static_cast<std::size_t>(t)];

    auto block_cost = [&](int start, int len) -> double
    {
        return prefix[static_cast<std::size_t>(start + len)] - prefix[static_cast<std::size_t>(start)];
    };

    std::vector<int> pref_before(static_cast<std::size_t>(U), 0);
    std::vector<int> suffix_from(static_cast<std::size_t>(U), 0);

    int run = 0;
    for (int i = 0; i < U; ++i)
    {
        pref_before[static_cast<std::size_t>(i)] = run;
        run += sequence[static_cast<std::size_t>(i)];
    }
    run = 0;
    for (int i = U - 1; i >= 0; --i)
    {
        run += sequence[static_cast<std::size_t>(i)];
        suffix_from[static_cast<std::size_t>(i)] = run;
    }

    std::vector<std::vector<double>> V(static_cast<std::size_t>(U + 1), std::vector<double>(static_cast<std::size_t>(epsilon + 2), kInf));
    for (int t = 0; t <= epsilon; ++t)
        V[static_cast<std::size_t>(U)][static_cast<std::size_t>(t)] = 0.0;

    for (int i = U - 1; i >= 0; --i)
    {
        const int len = sequence[static_cast<std::size_t>(i)];
        const int min_t = pref_before[static_cast<std::size_t>(i)];
        const int max_t = epsilon - suffix_from[static_cast<std::size_t>(i)];
        if (min_t > max_t)
            return kInf;

        V[static_cast<std::size_t>(i)][static_cast<std::size_t>(max_t)] =
            block_cost(max_t, len) + V[static_cast<std::size_t>(i + 1)][static_cast<std::size_t>(max_t + len)];

        for (int t = max_t - 1; t >= min_t; --t)
        {
            const double place = block_cost(t, len) + V[static_cast<std::size_t>(i + 1)][static_cast<std::size_t>(t + len)];
            const double skip = V[static_cast<std::size_t>(i)][static_cast<std::size_t>(t + 1)];
            V[static_cast<std::size_t>(i)][static_cast<std::size_t>(t)] = std::min(place, skip);
        }
    }

    return V[0][0];
}

double dp_exact_machine_cost(
    const std::vector<int> &machine_jobs,
    const std::vector<double> &prices,
    int epsilon,
    double rate,
    double per_machine_time_limit_sec)
{
    if (machine_jobs.empty())
        return 0.0;

    std::map<int, int> cnt;
    for (int p : machine_jobs)
        cnt[p]++;
    std::vector<int> lengths;
    std::vector<int> totals;
    for (const auto &[len, c] : cnt)
    {
        lengths.push_back(len);
        totals.push_back(c);
    }

    std::vector<double> prefix(static_cast<std::size_t>(epsilon) + 1, 0.0);
    for (int t = 0; t < epsilon; ++t)
        prefix[static_cast<std::size_t>(t) + 1] = prefix[static_cast<std::size_t>(t)] + rate * prices[static_cast<std::size_t>(t)];

    dp::DPParams params;
    params.time_limit = per_machine_time_limit_sec;
    params.track_schedule = false;

    auto res = dp::solve_sparse_dp(lengths, totals, prefix, epsilon, params);
    if (!res.feasible)
        return kInf;
    return res.cost;
}

double relaxed_machine_lb(
    const std::vector<int> &machine_jobs,
    const std::vector<double> &prices,
    int epsilon,
    double rate)
{
    if (machine_jobs.empty())
        return 0.0;

    std::map<int, int> cnt;
    int total_work = 0;
    for (int p : machine_jobs)
    {
        cnt[p]++;
        total_work += p;
    }

    std::vector<int> lengths;
    std::vector<int> totals;
    for (const auto &[len, c] : cnt)
    {
        lengths.push_back(len);
        totals.push_back(c);
    }

    std::vector<double> prefix(static_cast<std::size_t>(epsilon) + 1, 0.0);
    for (int t = 0; t < epsilon; ++t)
        prefix[static_cast<std::size_t>(t) + 1] = prefix[static_cast<std::size_t>(t)] + rate * prices[static_cast<std::size_t>(t)];

    auto spaces = dp::make_stateless_spaces(epsilon);
    const double lb_feas = dp::solve_relaxed_dp_lb_feas(lengths, totals, prefix, epsilon, spaces);
    if (lb_feas < kInf * 0.5)
        return lb_feas;

    const double lb_semi = dp::solve_relaxed_dp_lb(lengths, total_work, prefix, epsilon, spaces, dp::RelaxationMode::Semigroup);
    return lb_semi;
}

double fallback_slot_lb(
    const std::vector<int> &machine_jobs,
    const std::vector<double> &prices,
    int epsilon,
    double rate)
{
    int total_work = 0;
    for (int p : machine_jobs)
        total_work += p;
    if (total_work <= 0)
        return 0.0;
    if (total_work > epsilon)
        return kInf;

    std::vector<double> clipped(prices.begin(), prices.begin() + epsilon);
    std::sort(clipped.begin(), clipped.end());
    double lb = 0.0;
    for (int i = 0; i < total_work; ++i)
        lb += rate * clipped[static_cast<std::size_t>(i)];
    return lb;
}

struct VariantResult
{
    bool feasible = false;
    std::string variant;
    int epsilon_prev = -1;
    double runtime_sec = 0.0;
    double tec_total = kInf;
    double assignment_conditioned_lb = kInf;
    std::vector<int> machine_job_counts;
    std::vector<double> machine_exact_cost;
    std::vector<double> machine_relaxed_lb;
    std::vector<std::string> machine_lb_source;
    std::vector<int> final_machine_loads;
    std::int64_t accepted_moves = 0;
    std::int64_t accepted_relocate_moves = 0;
    std::int64_t accepted_swap_moves = 0;
    std::int64_t evaluated_relocate_moves = 0;
    std::int64_t evaluated_swap_moves = 0;
    std::int64_t displaced_jobs = 0;
    std::int64_t reinsertion_candidates_scored = 0;
    std::int64_t exact_dp_evals_repair = 0;
    std::int64_t exact_dp_evals_post_repair_local_search = 0;
    int relocate_cleanup_used = 0;
    std::int64_t exact_dp_calls_initial = 0;
    std::int64_t exact_dp_calls_local_search_only = 0;
    std::int64_t accepted_swap_intra_moves = 0;
    std::int64_t accepted_swap_inter_moves = 0;
    std::int64_t accepted_insert_inter_moves = 0;
    std::int64_t evaluated_swap_intra_moves = 0;
    std::int64_t evaluated_swap_inter_moves = 0;
    std::int64_t evaluated_insert_inter_moves = 0;
    std::int64_t exact_dp_cache_hits = 0;
    std::int64_t exact_dp_cache_misses = 0;
    std::string stop_reason = "na";
    std::string dominant_improvement_move = "none";
    double diagnostic_start_tec = -1.0;
    double diagnostic_best_tec = -1.0;
    int diagnostic_improving_move_found = 0;
    std::int64_t diagnostic_exact_evaluated_moves = 0;
    int exception_candidates_considered = 0;
    int exception_candidates_evaluated = 0;
    int exception_budget_used = 0;
    int exception_improvement_count = 0;
    double exception_best_delta = 0.0;
    int outside_pool_distinct_src = 0;
    int outside_pool_distinct_tgt = 0;
    int outside_pool_max_src_share = 0;
    int outside_pool_max_tgt_share = 0;
    int selected_distinct_src = 0;
    int selected_distinct_tgt = 0;
    int selected_max_src_share = 0;
    int selected_max_tgt_share = 0;
    double exception_hit_rate = 0.0;
    double final_machine_load_pressure = 0.0;
    double avg_machine_load_pressure = 0.0;
    int phaseV_score_escape_candidates_considered = 0;
    int phaseV_score_escape_candidates_evaluated = 0;
    int phaseV_score_escape_improvement_count = 0;
    double phaseV_score_escape_best_delta = 0.0;
    int phaseV_score_escape_escape_rounds = 0;
    int phaseV_score_escape_normal_rounds = 0;
    int phaseV_score_escape_distinct_pairs = 0;
    double phaseV_score_escape_max_cheap_lb = 0.0;
    int phaseX_candidates_considered = 0;
    int phaseX_candidates_evaluated = 0;
    int phaseX_improvement_count = 0;
    double phaseX_best_delta = 0.0;
    int phaseX_normal_rounds = 0;
    int phaseX_escape_rounds = 0;
    std::string phaseX_policy_name;
    int phaseY_candidates_generated = 0;
    int phaseY_candidates_selected = 0;
    int phaseY_candidates_evaluated = 0;
    int phaseY_improvements = 0;
    double phaseY_best_delta = 0.0;
    double phaseY_accepted_delta = 0.0;
    int phaseY_fallback_used = 0;
    int phaseY_invalid_ids_dropped = 0;
    int phaseY_sources_used = 0;
    int phaseY_targets_used = 0;
    std::string phaseY_proposal_name;
};

struct RepairMoveItem
{
    int p = 0;
    int source_machine = -1;
    double priority_score = 0.0;
};

enum class RepairMode
{
    DpRanked,
    PriorityDisplaced
};

std::optional<RepairMode> parse_repair_mode(const std::string &variant)
{
    if (variant == "history_repair_dp_ranked" || variant == "history_repair_dp_ranked_relocate")
        return RepairMode::DpRanked;
    if (variant == "history_repair_priority_displaced" || variant == "history_repair_priority_displaced_relocate")
        return RepairMode::PriorityDisplaced;
    return std::nullopt;
}

bool is_history_variant(const std::string &variant)
{
    return parse_repair_mode(variant).has_value();
}

bool history_uses_relocate_cleanup(const std::string &variant)
{
    return variant == "history_repair_dp_ranked_relocate" ||
           variant == "history_repair_priority_displaced_relocate";
}

Assignment assignment_from_jobs(const std::vector<std::vector<int>> &jobs_by_machine)
{
    Assignment a;
    a.feasible = true;
    a.machine_job_lengths = jobs_by_machine;
    a.machine_loads.assign(jobs_by_machine.size(), 0);
    for (std::size_t h = 0; h < jobs_by_machine.size(); ++h)
    {
        int load = 0;
        for (int p : jobs_by_machine[h])
            load += p;
        a.machine_loads[h] = load;
    }
    return a;
}

VariantResult evaluate_history_repair_step(
    const Assignment &prev_assignment,
    const std::vector<double> &rates,
    const std::vector<double> &prices,
    int epsilon_prev,
    int epsilon,
    const std::string &variant,
    double per_machine_dp_limit_sec,
    int ls_max_rounds,
    std::int64_t ls_max_moves_per_round,
    double ls_time_cap_sec,
    Assignment &next_assignment);

std::string machine_multiset_key(const std::vector<int> &machine_jobs, double rate)
{
    std::map<int, int> cnt;
    for (int p : machine_jobs)
        cnt[p]++;
    std::ostringstream oss;
    oss << std::llround(rate * 1000000.0) << "|";
    for (const auto &[len, c] : cnt)
        oss << len << ":" << c << "|";
    return oss.str();
}

struct ExactCostCache
{
    std::unordered_map<std::string, double> cache;
    std::int64_t hits = 0;
    std::int64_t misses = 0;
};

double exact_machine_cost_cached(
    const std::vector<int> &machine_jobs,
    const std::vector<double> &prices,
    int epsilon,
    double rate,
    double per_machine_time_limit_sec,
    ExactCostCache &cache)
{
    const std::string key = machine_multiset_key(machine_jobs, rate);
    auto it = cache.cache.find(key);
    if (it != cache.cache.end())
    {
        ++cache.hits;
        return it->second;
    }

    ++cache.misses;
    const double c = dp_exact_machine_cost(machine_jobs, prices, epsilon, rate, per_machine_time_limit_sec);
    cache.cache.emplace(key, c);
    return c;
}

struct LocalSearchStats
{
    std::int64_t accepted_relocate = 0;
    std::int64_t accepted_swap = 0;
    std::int64_t evaluated_relocate = 0;
    std::int64_t evaluated_swap = 0;
};

struct VndStats
{
    std::int64_t accepted_swap_intra = 0;
    std::int64_t accepted_swap_inter = 0;
    std::int64_t accepted_insert_inter = 0;
    std::int64_t evaluated_swap_intra = 0;
    std::int64_t evaluated_swap_inter = 0;
    std::int64_t evaluated_insert_inter = 0;
    std::string stop_reason = "max_rounds";
    int exception_candidates_considered = 0;
    int exception_candidates_evaluated = 0;
    int exception_budget_used = 0;
    int exception_improvement_count = 0;
    double exception_best_delta = 0.0;
    int outside_pool_distinct_src = 0;
    int outside_pool_distinct_tgt = 0;
    int outside_pool_max_src_share = 0;
    int outside_pool_max_tgt_share = 0;
    int selected_distinct_src = 0;
    int selected_distinct_tgt = 0;
    int selected_max_src_share = 0;
    int selected_max_tgt_share = 0;
    double exception_hit_rate = 0.0;
    double final_machine_load_pressure = 0.0;
    double avg_machine_load_pressure = 0.0;
    int phaseV_score_escape_candidates_considered = 0;
    int phaseV_score_escape_candidates_evaluated = 0;
    int phaseV_score_escape_improvement_count = 0;
    double phaseV_score_escape_best_delta = 0.0;
    int phaseV_score_escape_escape_rounds = 0;
    int phaseV_score_escape_normal_rounds = 0;
    int phaseV_score_escape_distinct_pairs = 0;
    double phaseV_score_escape_max_cheap_lb = 0.0;
    int phaseX_candidates_considered = 0;
    int phaseX_candidates_evaluated = 0;
    int phaseX_improvement_count = 0;
    double phaseX_best_delta = 0.0;
    int phaseX_normal_rounds = 0;
    int phaseX_escape_rounds = 0;
    std::string phaseX_policy_name;
    int phaseY_candidates_generated = 0;
    int phaseY_candidates_selected = 0;
    int phaseY_candidates_evaluated = 0;
    int phaseY_improvements = 0;
    double phaseY_best_delta = 0.0;
    double phaseY_accepted_delta = 0.0;
    int phaseY_fallback_used = 0;
    int phaseY_invalid_ids_dropped = 0;
    int phaseY_sources_used = 0;
    int phaseY_targets_used = 0;
    std::string phaseY_proposal_name;
};

struct NoScreenDiagStats
{
    std::int64_t evaluated_insert_inter = 0;
    std::int64_t evaluated_swap_inter = 0;
    std::int64_t accepted_insert_inter = 0;
    std::int64_t accepted_swap_inter = 0;
    double start_tec = kInf;
    double best_tec = kInf;
    bool improving_move_found = false;
    std::string best_move = "none";
    std::string stop_reason = "diag_no_improving_move";
};

struct StageL1MoveLogger
{
    bool enabled = false;
    std::string source_variant = "vnd_exact_dp_insert_rank_diverse_trimmed";
    std::ofstream broad;
    std::ofstream exact;
    std::int64_t broad_rows = 0;
    std::int64_t exact_rows = 0;
    std::int64_t exact_positive_rows = 0;
    std::int64_t next_record_id = 1;

    std::int64_t allocate_record_id()
    {
        return next_record_id++;
    }

    bool open_with_paths(
        const std::string &dir_path,
        const std::string &broad_name,
        const std::string &exact_name)
    {
        if (!enabled)
            return true;
        std::error_code ec;
        std::filesystem::create_directories(dir_path, ec);
        if (ec)
            return false;

        const std::string broad_path = dir_path + "/" + broad_name;
        const std::string exact_path = dir_path + "/" + exact_name;
        broad.open(broad_path, std::ios::out | std::ios::trunc);
        exact.open(exact_path, std::ios::out | std::ios::trunc);
        if (!broad.is_open() || !exact.is_open())
            return false;

        broad << "record_id,instance_id,epsilon,source_variant,seed_id,ls_round,job_id,job_processing_time,job_type_id,"
                 "source_machine_id,source_rate,source_rate_class,source_exact_cost,source_relaxed_lb,source_exact_minus_lb_gap,source_load,source_utilization,source_job_count,"
                 "target_machine_id,target_rate,target_rate_class,target_exact_cost,target_relaxed_lb,target_exact_minus_lb_gap,target_load,target_utilization,target_residual_slack_before,target_job_count,"
                 "projected_target_load_after,projected_source_load_after,source_to_target_rate_diff,cheap_lb_delta_proxy,source_cost_density,target_cost_density,"
                 "source_top_expensive_flag,target_top_expensive_flag,screen_score_s1,screen_score_s2,"
                 "current_tec,current_tec_minus_start,accepted_improving_moves_so_far,exact_eval_cap,exact_eval_budget_remaining,exact_eval_tier,epsilon_feasible,"
                 "context_id,source_cost_rank_num,source_cost_rank_den,target_slack_rank_num,target_slack_rank_den,epsilon_stress\n";

        exact << "record_id,instance_id,epsilon,source_variant,seed_id,ls_round,job_id,job_processing_time,job_type_id,"
                 "source_machine_id,source_rate,source_rate_class,source_exact_cost,source_relaxed_lb,source_exact_minus_lb_gap,source_load,source_utilization,source_job_count,"
                 "target_machine_id,target_rate,target_rate_class,target_exact_cost,target_relaxed_lb,target_exact_minus_lb_gap,target_load,target_utilization,target_residual_slack_before,target_job_count,"
                 "projected_target_load_after,projected_source_load_after,source_to_target_rate_diff,cheap_lb_delta_proxy,source_cost_density,target_cost_density,"
                 "source_top_expensive_flag,target_top_expensive_flag,screen_score_s1,screen_score_s2,"
                 "current_tec,current_tec_minus_start,accepted_improving_moves_so_far,exact_eval_cap,exact_eval_budget_remaining,exact_eval_tier,epsilon_feasible,"
                 "context_id,source_cost_rank_num,source_cost_rank_den,target_slack_rank_num,target_slack_rank_den,epsilon_stress,"
                 "exact_evaluation_performed,exact_old_touched_cost_sum,exact_new_touched_cost_sum,exact_total_delta,label_improving,label_accepted\n";
        return true;
    }

    bool open(const std::string &dir_path)
    {
        return open_with_paths(dir_path, "moves_broad_61_347.csv", "moves_exact_labeled_61_347.csv");
    }

    void write_broad(
        std::int64_t record_id,
        int instance_id,
        int epsilon,
        int seed_id,
        int ls_round,
        std::int64_t job_id,
        int p,
        int job_type_id,
        int src,
        double src_rate,
        int src_rate_class,
        double src_exact,
        double src_lb,
        double src_gap,
        int src_load,
        double src_util,
        int src_jobs,
        int tgt,
        double tgt_rate,
        int tgt_rate_class,
        double tgt_exact,
        double tgt_lb,
        double tgt_gap,
        int tgt_load,
        double tgt_util,
        int tgt_slack_before,
        int tgt_jobs,
        int tgt_load_after,
        int src_load_after,
        double rate_diff,
        double cheap_lb_delta,
        double src_density,
        double tgt_density,
        int src_top,
        int tgt_top,
        double s1,
        double s2,
        double current_tec,
        double tec_minus_start,
        std::int64_t accepted_so_far,
        int exact_eval_cap,
        int exact_eval_remaining,
        int exact_eval_tier,
        int epsilon_feasible,
        int context_id,
        int source_cost_rank_num,
        int source_cost_rank_den,
        int target_slack_rank_num,
        int target_slack_rank_den,
        double epsilon_stress)
    {
        if (!enabled)
            return;
        broad << record_id << ',' << instance_id << ',' << epsilon << ',' << source_variant << ','
              << seed_id << ',' << ls_round << ',' << job_id << ',' << p << ',' << job_type_id << ','
              << src << ',' << std::fixed << std::setprecision(6) << src_rate << ',' << src_rate_class << ','
              << src_exact << ',' << src_lb << ',' << src_gap << ',' << src_load << ',' << src_util << ',' << src_jobs << ','
              << tgt << ',' << tgt_rate << ',' << tgt_rate_class << ',' << tgt_exact << ',' << tgt_lb << ',' << tgt_gap << ','
              << tgt_load << ',' << tgt_util << ',' << tgt_slack_before << ',' << tgt_jobs << ','
              << tgt_load_after << ',' << src_load_after << ',' << rate_diff << ',' << cheap_lb_delta << ','
              << src_density << ',' << tgt_density << ',' << src_top << ',' << tgt_top << ',' << s1 << ',' << s2 << ','
              << current_tec << ',' << tec_minus_start << ',' << accepted_so_far << ',' << exact_eval_cap << ','
              << exact_eval_remaining << ',' << exact_eval_tier << ',' << epsilon_feasible << ','
              << context_id << ',' << source_cost_rank_num << ',' << source_cost_rank_den << ','
              << target_slack_rank_num << ',' << target_slack_rank_den << ',' << epsilon_stress << '\n';
        ++broad_rows;
    }

    void write_exact(
        std::int64_t record_id,
        int instance_id,
        int epsilon,
        int seed_id,
        int ls_round,
        std::int64_t job_id,
        int p,
        int job_type_id,
        int src,
        double src_rate,
        int src_rate_class,
        double src_exact,
        double src_lb,
        double src_gap,
        int src_load,
        double src_util,
        int src_jobs,
        int tgt,
        double tgt_rate,
        int tgt_rate_class,
        double tgt_exact,
        double tgt_lb,
        double tgt_gap,
        int tgt_load,
        double tgt_util,
        int tgt_slack_before,
        int tgt_jobs,
        int tgt_load_after,
        int src_load_after,
        double rate_diff,
        double cheap_lb_delta,
        double src_density,
        double tgt_density,
        int src_top,
        int tgt_top,
        double s1,
        double s2,
        double current_tec,
        double tec_minus_start,
        std::int64_t accepted_so_far,
        int exact_eval_cap,
        int exact_eval_remaining,
        int exact_eval_tier,
        int epsilon_feasible,
        int context_id,
        int source_cost_rank_num,
        int source_cost_rank_den,
        int target_slack_rank_num,
        int target_slack_rank_den,
        double epsilon_stress,
        double exact_old_touched,
        double exact_new_touched,
        double exact_total_delta,
        int improving,
        int accepted)
    {
        if (!enabled)
            return;
        exact << record_id << ',' << instance_id << ',' << epsilon << ',' << source_variant << ','
              << seed_id << ',' << ls_round << ',' << job_id << ',' << p << ',' << job_type_id << ','
              << src << ',' << std::fixed << std::setprecision(6) << src_rate << ',' << src_rate_class << ','
              << src_exact << ',' << src_lb << ',' << src_gap << ',' << src_load << ',' << src_util << ',' << src_jobs << ','
              << tgt << ',' << tgt_rate << ',' << tgt_rate_class << ',' << tgt_exact << ',' << tgt_lb << ',' << tgt_gap << ','
              << tgt_load << ',' << tgt_util << ',' << tgt_slack_before << ',' << tgt_jobs << ','
              << tgt_load_after << ',' << src_load_after << ',' << rate_diff << ',' << cheap_lb_delta << ','
              << src_density << ',' << tgt_density << ',' << src_top << ',' << tgt_top << ',' << s1 << ',' << s2 << ','
              << current_tec << ',' << tec_minus_start << ',' << accepted_so_far << ',' << exact_eval_cap << ','
              << exact_eval_remaining << ',' << exact_eval_tier << ',' << epsilon_feasible << ','
              << context_id << ',' << source_cost_rank_num << ',' << source_cost_rank_den << ','
              << target_slack_rank_num << ',' << target_slack_rank_den << ',' << epsilon_stress << ','
              << 1 << ',' << exact_old_touched << ',' << exact_new_touched << ',' << exact_total_delta << ',' << improving << ',' << accepted << '\n';
        ++exact_rows;
        if (improving)
            ++exact_positive_rows;
    }
};

struct StageL1LogContext
{
    int instance_id = -1;
    int seed_id = -1;
    double start_tec = 0.0;
    StageL1MoveLogger *logger = nullptr;
    int context_id = -1;
    int source_cost_rank_num = -1;
    int source_cost_rank_den = -1;
    int target_slack_rank_num = -1;
    int target_slack_rank_den = -1;
    double epsilon_stress = 0.0;
};

enum class InsertScreenMode
{
    DualPressureGlobal,
    DiverseTwoStage,
    DiverseTrimmed,
    DiverseBudgeted,
    DenseLabeling,
    MissetAudit,
    ExceptionLaneLLM,
    ExceptionLaneRandom,
    ExceptionLaneRefined1,
    ExceptionLaneRefined2,
    ExceptionLaneRefined3,
    ScoreEscapeSampler,
    PhaseXPolicyJson,
    PhaseYTraceProbe,
    PhaseYExecuteProposal,
    PhaseYRandomProposal
};

static int g_random_exception_seed = 42;
static int g_audit_instance_id = -1;
static std::string g_phaseX_policy_path;

struct PhaseXPolicy
{
    std::string policy_name;
    std::string normal_mode = "llm_score";
    std::string escape_mode = "none";
    int switch_after_no_hit = 2;
    bool switch_back_on_hit = true;
    int initial_budget = 4;
    int max_budget = 12;
    int grow_on_hit = 2;
    int shrink_on_miss = 1;
    int max_per_source = 3;
    int max_per_target = 3;
    bool require_positive_cheap_lb = false;
    double coverage_bonus = 0.0;
    double random_mix = 0.0;
    double cheap_lb_weight = 0.0;
    double s2_weight = 1.0;
    double slack_weight = 0.5;
    int guard_max_budget = 0;
};

static std::string phaseX_trim(const std::string& s)
{
    std::size_t start = 0;
    while (start < s.size() && (s[start] == ' ' || s[start] == '\t' || s[start] == '\n' || s[start] == '\r')) ++start;
    if (start >= s.size()) return "";
    std::size_t end = s.size() - 1;
    while (end > start && (s[end] == ' ' || s[end] == '\t' || s[end] == '\n' || s[end] == '\r')) --end;
    return s.substr(start, end - start + 1);
}

static std::string phaseX_unquote(const std::string& s)
{
    std::string t = phaseX_trim(s);
    if (t.size() >= 2 && t.front() == '"' && t.back() == '"') return t.substr(1, t.size() - 2);
    return t;
}

static PhaseXPolicy phaseX_read_policy(const std::string& path)
{
    PhaseXPolicy p;
    std::ifstream f(path);
    if (!f.is_open()) { std::cerr << "PhaseX: cannot open policy " << path << "\n"; return p; }
    std::stringstream buf; buf << f.rdbuf();
    std::string raw = buf.str();
    std::size_t bo = raw.find('{');
    std::size_t bc = raw.rfind('}');
    if (bo == std::string::npos || bc == std::string::npos || bc <= bo) return p;
    std::string body = raw.substr(bo + 1, bc - bo - 1);
    std::map<std::string, std::string> kv;
    std::size_t pos = 0;
    while (pos < body.size()) {
        std::size_t ks = body.find('"', pos); if (ks == std::string::npos) break;
        std::size_t ke = body.find('"', ks + 1); if (ke == std::string::npos) break;
        std::string key = body.substr(ks + 1, ke - ks - 1);
        std::size_t col = body.find(':', ke + 1); if (col == std::string::npos) break;
        std::size_t vs = col + 1;
        while (vs < body.size() && (body[vs] == ' ' || body[vs] == '\t' || body[vs] == '\n' || body[vs] == '\r')) ++vs;
        if (vs >= body.size()) break;
        std::string val;
        if (body[vs] == '"') { std::size_t ve = vs + 1; while (ve < body.size() && !(body[ve] == '"' && body[ve-1] != '\\')) ++ve; val = body.substr(vs, ve - vs + 1); pos = ve + 1; }
        else { std::size_t ve = vs; while (ve < body.size() && body[ve] != ',' && body[ve] != '}' && body[ve] != '\n') ++ve; val = body.substr(vs, ve - vs); pos = ve; }
        kv[key] = phaseX_trim(val);
    }
    if (kv.count("policy_name")) p.policy_name = phaseX_unquote(kv["policy_name"]);
    if (kv.count("normal_mode")) p.normal_mode = phaseX_unquote(kv["normal_mode"]);
    if (kv.count("escape_mode")) p.escape_mode = phaseX_unquote(kv["escape_mode"]);
    if (kv.count("switch_after_no_hit")) p.switch_after_no_hit = std::stoi(kv["switch_after_no_hit"]);
    if (kv.count("switch_back_on_hit")) { std::string v = kv["switch_back_on_hit"]; p.switch_back_on_hit = (v == "true" || v == "1"); }
    if (kv.count("initial_budget")) p.initial_budget = std::stoi(kv["initial_budget"]);
    if (kv.count("max_budget")) p.max_budget = std::stoi(kv["max_budget"]);
    if (kv.count("grow_on_hit")) p.grow_on_hit = std::stoi(kv["grow_on_hit"]);
    if (kv.count("shrink_on_miss")) p.shrink_on_miss = std::stoi(kv["shrink_on_miss"]);
    if (kv.count("max_per_source")) p.max_per_source = std::stoi(kv["max_per_source"]);
    if (kv.count("max_per_target")) p.max_per_target = std::stoi(kv["max_per_target"]);
    if (kv.count("require_positive_cheap_lb")) { std::string v = kv["require_positive_cheap_lb"]; p.require_positive_cheap_lb = (v == "true" || v == "1"); }
    if (kv.count("coverage_bonus")) p.coverage_bonus = std::stod(kv["coverage_bonus"]);
    if (kv.count("random_mix")) p.random_mix = std::stod(kv["random_mix"]);
    if (kv.count("cheap_lb_weight")) p.cheap_lb_weight = std::stod(kv["cheap_lb_weight"]);
    if (kv.count("s2_weight")) p.s2_weight = std::stod(kv["s2_weight"]);
    if (kv.count("slack_weight")) p.slack_weight = std::stod(kv["slack_weight"]);
    if (kv.count("guard_max_budget")) p.guard_max_budget = std::stoi(kv["guard_max_budget"]);
    return p;
}

struct LocalSearchConfig
{
    bool enable_relocate = true;
    bool enable_swap = true;
    bool screened_swap = false;
    int screened_swap_length_tolerance = 1;
    bool priority_machines = false;
    int priority_top_k = 8;
};

void run_local_search(
    std::vector<std::vector<int>> &machine_jobs,
    std::vector<int> &machine_loads,
    std::vector<double> &machine_exact_cost,
    const std::vector<double> &rates,
    const std::vector<double> &prices,
    int epsilon,
    double per_machine_dp_limit_sec,
    int ls_max_rounds,
    std::int64_t ls_max_moves_per_round,
    double ls_time_cap_sec,
    const LocalSearchConfig &cfg,
    ExactCostCache &cache,
    LocalSearchStats &stats)
{
    const int m = static_cast<int>(machine_jobs.size());
    const auto t0 = std::chrono::steady_clock::now();
    auto ls_elapsed_sec = [&]() -> double
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };

    for (int round = 0; round < ls_max_rounds; ++round)
    {
        if (ls_elapsed_sec() > ls_time_cap_sec)
            break;

        bool improved = false;
        std::int64_t inspected = 0;

        std::vector<int> active_source_machines;
        if (cfg.priority_machines)
        {
            std::vector<std::pair<double, int>> machine_score;
            machine_score.reserve(static_cast<std::size_t>(m));
            for (int h = 0; h < m; ++h)
            {
                const auto &jobs_h = machine_jobs[static_cast<std::size_t>(h)];
                const double safe_lb = fallback_slot_lb(jobs_h, prices, epsilon, rates[static_cast<std::size_t>(h)]);
                const double gap = machine_exact_cost[static_cast<std::size_t>(h)] - safe_lb;
                machine_score.emplace_back(gap, h);
            }
            std::sort(machine_score.begin(), machine_score.end(), [](const auto &a, const auto &b)
                      {
                          if (std::fabs(a.first - b.first) > 1e-12)
                              return a.first > b.first;
                          return a.second < b.second;
                      });

            const int top_k = std::max(1, std::min(cfg.priority_top_k, m));
            for (int i = 0; i < top_k; ++i)
                active_source_machines.push_back(machine_score[static_cast<std::size_t>(i)].second);
        }
        else
        {
            active_source_machines.resize(static_cast<std::size_t>(m));
            std::iota(active_source_machines.begin(), active_source_machines.end(), 0);
        }

        if (cfg.enable_relocate)
        {
            for (int src_idx = 0; src_idx < static_cast<int>(active_source_machines.size()) && !improved; ++src_idx)
            {
                const int a = active_source_machines[static_cast<std::size_t>(src_idx)];
                for (int ia = 0; ia < static_cast<int>(machine_jobs[static_cast<std::size_t>(a)].size()) && !improved; ++ia)
                {
                    const int p = machine_jobs[static_cast<std::size_t>(a)][static_cast<std::size_t>(ia)];
                    for (int b = 0; b < m && !improved; ++b)
                    {
                        if (a == b)
                            continue;
                        if (machine_loads[static_cast<std::size_t>(a)] - p > epsilon || machine_loads[static_cast<std::size_t>(b)] + p > epsilon)
                            continue;

                        if (inspected >= ls_max_moves_per_round || ls_elapsed_sec() > ls_time_cap_sec)
                            break;

                        ++inspected;
                        ++stats.evaluated_relocate;

                        auto jobs_a_new = machine_jobs[static_cast<std::size_t>(a)];
                        auto jobs_b_new = machine_jobs[static_cast<std::size_t>(b)];
                        jobs_a_new.erase(jobs_a_new.begin() + ia);
                        jobs_b_new.push_back(p);

                        const double new_a = exact_machine_cost_cached(jobs_a_new, prices, epsilon, rates[static_cast<std::size_t>(a)], per_machine_dp_limit_sec, cache);
                        const double new_b = exact_machine_cost_cached(jobs_b_new, prices, epsilon, rates[static_cast<std::size_t>(b)], per_machine_dp_limit_sec, cache);
                        if (!(new_a < kInf * 0.5) || !(new_b < kInf * 0.5))
                            continue;

                        const double old_ab = machine_exact_cost[static_cast<std::size_t>(a)] + machine_exact_cost[static_cast<std::size_t>(b)];
                        const double new_ab = new_a + new_b;
                        if (new_ab + 1e-9 < old_ab)
                        {
                            machine_jobs[static_cast<std::size_t>(a)] = std::move(jobs_a_new);
                            machine_jobs[static_cast<std::size_t>(b)] = std::move(jobs_b_new);
                            machine_loads[static_cast<std::size_t>(a)] -= p;
                            machine_loads[static_cast<std::size_t>(b)] += p;
                            machine_exact_cost[static_cast<std::size_t>(a)] = new_a;
                            machine_exact_cost[static_cast<std::size_t>(b)] = new_b;
                            ++stats.accepted_relocate;
                            improved = true;
                        }
                    }
                }
            }
        }

        if (improved)
            continue;

        if (!cfg.enable_swap)
            break;

        inspected = 0;
        for (int a = 0; a < m && !improved; ++a)
        {
            if (cfg.priority_machines)
            {
                bool a_is_active = false;
                for (int h : active_source_machines)
                    if (h == a)
                        a_is_active = true;
                if (!a_is_active)
                    continue;
            }

            for (int b = a + 1; b < m && !improved; ++b)
            {
                for (int ia = 0; ia < static_cast<int>(machine_jobs[static_cast<std::size_t>(a)].size()) && !improved; ++ia)
                {
                    const int pa = machine_jobs[static_cast<std::size_t>(a)][static_cast<std::size_t>(ia)];
                    for (int ib = 0; ib < static_cast<int>(machine_jobs[static_cast<std::size_t>(b)].size()) && !improved; ++ib)
                    {
                        const int pb = machine_jobs[static_cast<std::size_t>(b)][static_cast<std::size_t>(ib)];

                        if (cfg.screened_swap && std::abs(pa - pb) > cfg.screened_swap_length_tolerance)
                            continue;

                        const int load_a_new = machine_loads[static_cast<std::size_t>(a)] - pa + pb;
                        const int load_b_new = machine_loads[static_cast<std::size_t>(b)] - pb + pa;
                        if (load_a_new > epsilon || load_b_new > epsilon)
                            continue;

                        if (inspected >= ls_max_moves_per_round || ls_elapsed_sec() > ls_time_cap_sec)
                            break;

                        ++inspected;
                        ++stats.evaluated_swap;

                        auto jobs_a_new = machine_jobs[static_cast<std::size_t>(a)];
                        auto jobs_b_new = machine_jobs[static_cast<std::size_t>(b)];
                        jobs_a_new[static_cast<std::size_t>(ia)] = pb;
                        jobs_b_new[static_cast<std::size_t>(ib)] = pa;

                        if (cfg.screened_swap)
                        {
                            const double old_ab = machine_exact_cost[static_cast<std::size_t>(a)] + machine_exact_cost[static_cast<std::size_t>(b)];
                            const double lb_a = fallback_slot_lb(jobs_a_new, prices, epsilon, rates[static_cast<std::size_t>(a)]);
                            const double lb_b = fallback_slot_lb(jobs_b_new, prices, epsilon, rates[static_cast<std::size_t>(b)]);
                            if (lb_a + lb_b >= old_ab - 1e-9)
                                continue;
                        }

                        const double new_a = exact_machine_cost_cached(jobs_a_new, prices, epsilon, rates[static_cast<std::size_t>(a)], per_machine_dp_limit_sec, cache);
                        const double new_b = exact_machine_cost_cached(jobs_b_new, prices, epsilon, rates[static_cast<std::size_t>(b)], per_machine_dp_limit_sec, cache);
                        if (!(new_a < kInf * 0.5) || !(new_b < kInf * 0.5))
                            continue;

                        const double old_ab = machine_exact_cost[static_cast<std::size_t>(a)] + machine_exact_cost[static_cast<std::size_t>(b)];
                        const double new_ab = new_a + new_b;
                        if (new_ab + 1e-9 < old_ab)
                        {
                            machine_jobs[static_cast<std::size_t>(a)] = std::move(jobs_a_new);
                            machine_jobs[static_cast<std::size_t>(b)] = std::move(jobs_b_new);
                            machine_loads[static_cast<std::size_t>(a)] = load_a_new;
                            machine_loads[static_cast<std::size_t>(b)] = load_b_new;
                            machine_exact_cost[static_cast<std::size_t>(a)] = new_a;
                            machine_exact_cost[static_cast<std::size_t>(b)] = new_b;
                            ++stats.accepted_swap;
                            improved = true;
                        }
                    }
                }
            }
        }

        if (!improved)
            break;
    }
}

void run_vnd_exact_dp(
    std::vector<std::vector<int>> &machine_jobs,
    std::vector<int> &machine_loads,
    std::vector<double> &machine_exact_cost,
    const std::vector<double> &rates,
    const std::vector<double> &prices,
    int epsilon,
    double per_machine_dp_limit_sec,
    int max_rounds,
    std::int64_t max_screened_moves_per_round,
    double time_cap_sec,
    ExactCostCache &cache,
    VndStats &stats)
{
    const int m = static_cast<int>(machine_jobs.size());
    const auto t0 = std::chrono::steady_clock::now();
    auto elapsed_sec = [&]() -> double
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };

    const int shortlist_cap = 48;
    const int exact_eval_cap = 24;

    struct SwapInterCand
    {
        int a = -1;
        int b = -1;
        int ia = -1;
        int ib = -1;
        double gain_lb = 0.0;
    };

    struct InsertInterCand
    {
        int a = -1;
        int b = -1;
        int ia = -1;
        int p = 0;
        double gain_lb = 0.0;
    };

    for (int round = 0; round < max_rounds; ++round)
    {
        if (elapsed_sec() > time_cap_sec)
        {
            stats.stop_reason = "time_cap";
            return;
        }

        bool improved_this_round = false;
        int neighborhood = 0;
        while (neighborhood < 3)
        {
            if (elapsed_sec() > time_cap_sec)
            {
                stats.stop_reason = "time_cap";
                return;
            }

            bool improved = false;
            if (neighborhood == 0)
            {
                std::int64_t exact_checked = 0;
                for (int h = 0; h < m; ++h)
                {
                    const int n = static_cast<int>(machine_jobs[static_cast<std::size_t>(h)].size());
                    for (int i = 0; i < n; ++i)
                    {
                        for (int j = i + 1; j < n; ++j)
                        {
                            ++stats.evaluated_swap_intra;
                            if (stats.evaluated_swap_intra >= max_screened_moves_per_round)
                                break;

                            if (exact_checked < exact_eval_cap)
                            {
                                ++exact_checked;
                                auto trial_h = machine_jobs[static_cast<std::size_t>(h)];
                                std::swap(trial_h[static_cast<std::size_t>(i)], trial_h[static_cast<std::size_t>(j)]);
                                const double new_h = exact_machine_cost_cached(
                                    trial_h,
                                    prices,
                                    epsilon,
                                    rates[static_cast<std::size_t>(h)],
                                    per_machine_dp_limit_sec,
                                    cache);
                                if (!(new_h < kInf * 0.5))
                                    continue;
                                const double old_h = machine_exact_cost[static_cast<std::size_t>(h)];
                                if (new_h + 1e-9 < old_h)
                                {
                                    machine_jobs[static_cast<std::size_t>(h)] = std::move(trial_h);
                                    machine_exact_cost[static_cast<std::size_t>(h)] = new_h;
                                    ++stats.accepted_swap_intra;
                                    improved = true;
                                    break;
                                }
                            }
                        }
                        if (improved)
                            break;
                        if (stats.evaluated_swap_intra >= max_screened_moves_per_round)
                            break;
                    }
                    if (improved)
                        break;
                    if (stats.evaluated_swap_intra >= max_screened_moves_per_round)
                        break;
                }
            }
            else if (neighborhood == 1)
            {
                std::vector<SwapInterCand> cand;
                cand.reserve(256);
                std::int64_t screened = 0;

                for (int a = 0; a < m; ++a)
                {
                    for (int b = a + 1; b < m; ++b)
                    {
                        const auto &jobs_a = machine_jobs[static_cast<std::size_t>(a)];
                        const auto &jobs_b = machine_jobs[static_cast<std::size_t>(b)];
                        for (int ia = 0; ia < static_cast<int>(jobs_a.size()); ++ia)
                        {
                            const int pa = jobs_a[static_cast<std::size_t>(ia)];
                            for (int ib = 0; ib < static_cast<int>(jobs_b.size()); ++ib)
                            {
                                if (screened >= max_screened_moves_per_round)
                                    break;
                                ++screened;
                                ++stats.evaluated_swap_inter;

                                const int pb = jobs_b[static_cast<std::size_t>(ib)];
                                const int load_a_new = machine_loads[static_cast<std::size_t>(a)] - pa + pb;
                                const int load_b_new = machine_loads[static_cast<std::size_t>(b)] - pb + pa;
                                if (load_a_new > epsilon || load_b_new > epsilon)
                                    continue;

                                auto trial_a = jobs_a;
                                auto trial_b = jobs_b;
                                trial_a[static_cast<std::size_t>(ia)] = pb;
                                trial_b[static_cast<std::size_t>(ib)] = pa;

                                const double old_lb =
                                    fallback_slot_lb(jobs_a, prices, epsilon, rates[static_cast<std::size_t>(a)]) +
                                    fallback_slot_lb(jobs_b, prices, epsilon, rates[static_cast<std::size_t>(b)]);
                                const double new_lb =
                                    fallback_slot_lb(trial_a, prices, epsilon, rates[static_cast<std::size_t>(a)]) +
                                    fallback_slot_lb(trial_b, prices, epsilon, rates[static_cast<std::size_t>(b)]);

                                if (new_lb + 1e-9 < old_lb)
                                    cand.push_back(SwapInterCand{a, b, ia, ib, old_lb - new_lb});
                            }
                        }
                    }
                }

                std::sort(cand.begin(), cand.end(), [](const SwapInterCand &x, const SwapInterCand &y)
                          {
                              if (std::fabs(x.gain_lb - y.gain_lb) > 1e-12)
                                  return x.gain_lb > y.gain_lb;
                              if (x.a != y.a)
                                  return x.a < y.a;
                              if (x.b != y.b)
                                  return x.b < y.b;
                              if (x.ia != y.ia)
                                  return x.ia < y.ia;
                              return x.ib < y.ib;
                          });

                const int lim = std::min<int>(exact_eval_cap, std::min<int>(shortlist_cap, static_cast<int>(cand.size())));
                for (int idx = 0; idx < lim && !improved; ++idx)
                {
                    const auto c = cand[static_cast<std::size_t>(idx)];
                    auto trial_a = machine_jobs[static_cast<std::size_t>(c.a)];
                    auto trial_b = machine_jobs[static_cast<std::size_t>(c.b)];
                    const int pa = trial_a[static_cast<std::size_t>(c.ia)];
                    const int pb = trial_b[static_cast<std::size_t>(c.ib)];
                    trial_a[static_cast<std::size_t>(c.ia)] = pb;
                    trial_b[static_cast<std::size_t>(c.ib)] = pa;

                    const double new_a = exact_machine_cost_cached(
                        trial_a,
                        prices,
                        epsilon,
                        rates[static_cast<std::size_t>(c.a)],
                        per_machine_dp_limit_sec,
                        cache);
                    const double new_b = exact_machine_cost_cached(
                        trial_b,
                        prices,
                        epsilon,
                        rates[static_cast<std::size_t>(c.b)],
                        per_machine_dp_limit_sec,
                        cache);
                    if (!(new_a < kInf * 0.5) || !(new_b < kInf * 0.5))
                        continue;

                    const double old_ab = machine_exact_cost[static_cast<std::size_t>(c.a)] + machine_exact_cost[static_cast<std::size_t>(c.b)];
                    const double new_ab = new_a + new_b;
                    if (new_ab + 1e-9 < old_ab)
                    {
                        machine_jobs[static_cast<std::size_t>(c.a)] = std::move(trial_a);
                        machine_jobs[static_cast<std::size_t>(c.b)] = std::move(trial_b);
                        machine_loads[static_cast<std::size_t>(c.a)] = machine_loads[static_cast<std::size_t>(c.a)] - pa + pb;
                        machine_loads[static_cast<std::size_t>(c.b)] = machine_loads[static_cast<std::size_t>(c.b)] - pb + pa;
                        machine_exact_cost[static_cast<std::size_t>(c.a)] = new_a;
                        machine_exact_cost[static_cast<std::size_t>(c.b)] = new_b;
                        ++stats.accepted_swap_inter;
                        improved = true;
                    }
                }
            }
            else
            {
                std::vector<InsertInterCand> cand;
                cand.reserve(256);
                std::int64_t screened = 0;

                for (int a = 0; a < m; ++a)
                {
                    const auto &jobs_a = machine_jobs[static_cast<std::size_t>(a)];
                    for (int ia = 0; ia < static_cast<int>(jobs_a.size()); ++ia)
                    {
                        const int p = jobs_a[static_cast<std::size_t>(ia)];
                        for (int b = 0; b < m; ++b)
                        {
                            if (a == b)
                                continue;
                            if (screened >= max_screened_moves_per_round)
                                break;
                            ++screened;
                            ++stats.evaluated_insert_inter;

                            if (machine_loads[static_cast<std::size_t>(a)] - p > epsilon || machine_loads[static_cast<std::size_t>(b)] + p > epsilon)
                                continue;

                            auto trial_a = jobs_a;
                            auto trial_b = machine_jobs[static_cast<std::size_t>(b)];
                            trial_a.erase(trial_a.begin() + ia);
                            trial_b.push_back(p);

                            const double old_lb =
                                fallback_slot_lb(jobs_a, prices, epsilon, rates[static_cast<std::size_t>(a)]) +
                                fallback_slot_lb(machine_jobs[static_cast<std::size_t>(b)], prices, epsilon, rates[static_cast<std::size_t>(b)]);
                            const double new_lb =
                                fallback_slot_lb(trial_a, prices, epsilon, rates[static_cast<std::size_t>(a)]) +
                                fallback_slot_lb(trial_b, prices, epsilon, rates[static_cast<std::size_t>(b)]);
                            if (new_lb + 1e-9 < old_lb)
                                cand.push_back(InsertInterCand{a, b, ia, p, old_lb - new_lb});
                        }
                    }
                }

                std::sort(cand.begin(), cand.end(), [](const InsertInterCand &x, const InsertInterCand &y)
                          {
                              if (std::fabs(x.gain_lb - y.gain_lb) > 1e-12)
                                  return x.gain_lb > y.gain_lb;
                              if (x.a != y.a)
                                  return x.a < y.a;
                              if (x.b != y.b)
                                  return x.b < y.b;
                              return x.ia < y.ia;
                          });

                const int lim = std::min<int>(exact_eval_cap, std::min<int>(shortlist_cap, static_cast<int>(cand.size())));
                for (int idx = 0; idx < lim && !improved; ++idx)
                {
                    const auto c = cand[static_cast<std::size_t>(idx)];
                    auto trial_a = machine_jobs[static_cast<std::size_t>(c.a)];
                    auto trial_b = machine_jobs[static_cast<std::size_t>(c.b)];
                    const int p = trial_a[static_cast<std::size_t>(c.ia)];
                    trial_a.erase(trial_a.begin() + c.ia);
                    trial_b.push_back(p);

                    const double new_a = exact_machine_cost_cached(
                        trial_a,
                        prices,
                        epsilon,
                        rates[static_cast<std::size_t>(c.a)],
                        per_machine_dp_limit_sec,
                        cache);
                    const double new_b = exact_machine_cost_cached(
                        trial_b,
                        prices,
                        epsilon,
                        rates[static_cast<std::size_t>(c.b)],
                        per_machine_dp_limit_sec,
                        cache);
                    if (!(new_a < kInf * 0.5) || !(new_b < kInf * 0.5))
                        continue;

                    const double old_ab = machine_exact_cost[static_cast<std::size_t>(c.a)] + machine_exact_cost[static_cast<std::size_t>(c.b)];
                    const double new_ab = new_a + new_b;
                    if (new_ab + 1e-9 < old_ab)
                    {
                        machine_jobs[static_cast<std::size_t>(c.a)] = std::move(trial_a);
                        machine_jobs[static_cast<std::size_t>(c.b)] = std::move(trial_b);
                        machine_loads[static_cast<std::size_t>(c.a)] -= p;
                        machine_loads[static_cast<std::size_t>(c.b)] += p;
                        machine_exact_cost[static_cast<std::size_t>(c.a)] = new_a;
                        machine_exact_cost[static_cast<std::size_t>(c.b)] = new_b;
                        ++stats.accepted_insert_inter;
                        improved = true;
                    }
                }
            }

            if (improved)
            {
                improved_this_round = true;
                neighborhood = 0;
            }
            else
            {
                ++neighborhood;
            }
        }

        if (!improved_this_round)
        {
            stats.stop_reason = "no_improving_move";
            return;
        }
    }

    stats.stop_reason = "max_rounds";
}

void run_noscreen_1move_diagnostic(
    std::vector<std::vector<int>> &machine_jobs,
    std::vector<int> &machine_loads,
    std::vector<double> &machine_exact_cost,
    const std::vector<double> &rates,
    const std::vector<double> &prices,
    int epsilon,
    double per_machine_dp_limit_sec,
    std::int64_t max_exact_evaluated_moves,
    double time_cap_sec,
    ExactCostCache &cache,
    NoScreenDiagStats &stats)
{
    const int m = static_cast<int>(machine_jobs.size());
    const auto t0 = std::chrono::steady_clock::now();
    auto elapsed_sec = [&]() -> double
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };

    double current_tec = 0.0;
    for (double c : machine_exact_cost)
        current_tec += c;
    stats.start_tec = current_tec;
    stats.best_tec = current_tec;

    std::vector<std::pair<double, int>> machine_priority;
    machine_priority.reserve(static_cast<std::size_t>(m));
    for (int h = 0; h < m; ++h)
        machine_priority.emplace_back(machine_exact_cost[static_cast<std::size_t>(h)], h);
    std::sort(machine_priority.begin(), machine_priority.end(), [](const auto &a, const auto &b)
              {
                  if (std::fabs(a.first - b.first) > 1e-12)
                      return a.first > b.first;
                  return a.second < b.second;
              });

    const int source_top_k = std::max(2, std::min(m, 8));
    std::vector<int> active_sources;
    active_sources.reserve(static_cast<std::size_t>(source_top_k));
    for (int i = 0; i < source_top_k; ++i)
        active_sources.push_back(machine_priority[static_cast<std::size_t>(i)].second);

    enum class MoveKind
    {
        None,
        InsertInter,
        SwapInter
    };

    struct BestMove
    {
        MoveKind kind = MoveKind::None;
        int a = -1;
        int b = -1;
        int ia = -1;
        int ib = -1;
        int p = 0;
        int pa = 0;
        int pb = 0;
        double candidate_tec = kInf;
        double new_a = kInf;
        double new_b = kInf;
    };

    BestMove best;

    const std::int64_t min_swap_budget = std::min<std::int64_t>(64, std::max<std::int64_t>(8, max_exact_evaluated_moves / 4));
    const std::int64_t insert_budget = std::max<std::int64_t>(0, max_exact_evaluated_moves - min_swap_budget);

    auto budget_or_time_hit = [&]() -> bool
    {
        return stats.evaluated_insert_inter + stats.evaluated_swap_inter >= max_exact_evaluated_moves || elapsed_sec() > time_cap_sec;
    };

    auto insert_budget_hit = [&]() -> bool
    {
        return stats.evaluated_insert_inter >= insert_budget;
    };

    for (int src_idx = 0; src_idx < static_cast<int>(active_sources.size()); ++src_idx)
    {
        if (budget_or_time_hit() || insert_budget_hit())
            break;
        const int a = active_sources[static_cast<std::size_t>(src_idx)];
        const auto &jobs_a = machine_jobs[static_cast<std::size_t>(a)];

        for (int ia = 0; ia < static_cast<int>(jobs_a.size()); ++ia)
        {
            if (budget_or_time_hit() || insert_budget_hit())
                break;
            const int p = jobs_a[static_cast<std::size_t>(ia)];

            for (int b = 0; b < m; ++b)
            {
                if (a == b)
                    continue;
                if (budget_or_time_hit() || insert_budget_hit())
                    break;

                if (machine_loads[static_cast<std::size_t>(a)] - p > epsilon || machine_loads[static_cast<std::size_t>(b)] + p > epsilon)
                    continue;

                ++stats.evaluated_insert_inter;

                auto trial_a = jobs_a;
                auto trial_b = machine_jobs[static_cast<std::size_t>(b)];
                trial_a.erase(trial_a.begin() + ia);
                trial_b.push_back(p);

                const double new_a = exact_machine_cost_cached(
                    trial_a,
                    prices,
                    epsilon,
                    rates[static_cast<std::size_t>(a)],
                    per_machine_dp_limit_sec,
                    cache);
                const double new_b = exact_machine_cost_cached(
                    trial_b,
                    prices,
                    epsilon,
                    rates[static_cast<std::size_t>(b)],
                    per_machine_dp_limit_sec,
                    cache);
                if (!(new_a < kInf * 0.5) || !(new_b < kInf * 0.5))
                    continue;

                const double old_ab = machine_exact_cost[static_cast<std::size_t>(a)] + machine_exact_cost[static_cast<std::size_t>(b)];
                const double new_ab = new_a + new_b;
                const double cand_tec = current_tec - old_ab + new_ab;
                if (cand_tec + 1e-9 < best.candidate_tec)
                {
                    best.kind = MoveKind::InsertInter;
                    best.a = a;
                    best.b = b;
                    best.ia = ia;
                    best.ib = -1;
                    best.p = p;
                    best.pa = p;
                    best.pb = 0;
                    best.candidate_tec = cand_tec;
                    best.new_a = new_a;
                    best.new_b = new_b;
                }
            }
        }
    }

    for (int a_idx = 0; a_idx < static_cast<int>(active_sources.size()); ++a_idx)
    {
        if (budget_or_time_hit())
            break;
        const int a = active_sources[static_cast<std::size_t>(a_idx)];
        const auto &jobs_a = machine_jobs[static_cast<std::size_t>(a)];

        for (int b_idx = a_idx + 1; b_idx < static_cast<int>(active_sources.size()); ++b_idx)
        {
            if (budget_or_time_hit())
                break;
            const int b = active_sources[static_cast<std::size_t>(b_idx)];
            const auto &jobs_b = machine_jobs[static_cast<std::size_t>(b)];

            for (int ia = 0; ia < static_cast<int>(jobs_a.size()); ++ia)
            {
                if (budget_or_time_hit())
                    break;
                const int pa = jobs_a[static_cast<std::size_t>(ia)];

                for (int ib = 0; ib < static_cast<int>(jobs_b.size()); ++ib)
                {
                    if (budget_or_time_hit())
                        break;
                    const int pb = jobs_b[static_cast<std::size_t>(ib)];
                    const int load_a_new = machine_loads[static_cast<std::size_t>(a)] - pa + pb;
                    const int load_b_new = machine_loads[static_cast<std::size_t>(b)] - pb + pa;
                    if (load_a_new > epsilon || load_b_new > epsilon)
                        continue;

                    ++stats.evaluated_swap_inter;

                    auto trial_a = jobs_a;
                    auto trial_b = jobs_b;
                    trial_a[static_cast<std::size_t>(ia)] = pb;
                    trial_b[static_cast<std::size_t>(ib)] = pa;

                    const double new_a = exact_machine_cost_cached(
                        trial_a,
                        prices,
                        epsilon,
                        rates[static_cast<std::size_t>(a)],
                        per_machine_dp_limit_sec,
                        cache);
                    const double new_b = exact_machine_cost_cached(
                        trial_b,
                        prices,
                        epsilon,
                        rates[static_cast<std::size_t>(b)],
                        per_machine_dp_limit_sec,
                        cache);
                    if (!(new_a < kInf * 0.5) || !(new_b < kInf * 0.5))
                        continue;

                    const double old_ab = machine_exact_cost[static_cast<std::size_t>(a)] + machine_exact_cost[static_cast<std::size_t>(b)];
                    const double new_ab = new_a + new_b;
                    const double cand_tec = current_tec - old_ab + new_ab;
                    if (cand_tec + 1e-9 < best.candidate_tec)
                    {
                        best.kind = MoveKind::SwapInter;
                        best.a = a;
                        best.b = b;
                        best.ia = ia;
                        best.ib = ib;
                        best.p = 0;
                        best.pa = pa;
                        best.pb = pb;
                        best.candidate_tec = cand_tec;
                        best.new_a = new_a;
                        best.new_b = new_b;
                    }
                }
            }
        }
    }

    if (best.kind == MoveKind::InsertInter || best.kind == MoveKind::SwapInter)
    {
        stats.improving_move_found = (best.candidate_tec + 1e-9 < current_tec);
        if (stats.improving_move_found)
        {
            if (best.kind == MoveKind::InsertInter)
            {
                auto trial_a = machine_jobs[static_cast<std::size_t>(best.a)];
                auto trial_b = machine_jobs[static_cast<std::size_t>(best.b)];
                const int p = trial_a[static_cast<std::size_t>(best.ia)];
                trial_a.erase(trial_a.begin() + best.ia);
                trial_b.push_back(p);

                machine_jobs[static_cast<std::size_t>(best.a)] = std::move(trial_a);
                machine_jobs[static_cast<std::size_t>(best.b)] = std::move(trial_b);
                machine_loads[static_cast<std::size_t>(best.a)] -= p;
                machine_loads[static_cast<std::size_t>(best.b)] += p;
                machine_exact_cost[static_cast<std::size_t>(best.a)] = best.new_a;
                machine_exact_cost[static_cast<std::size_t>(best.b)] = best.new_b;

                stats.accepted_insert_inter = 1;
                stats.best_move = "insert_inter";
            }
            else
            {
                auto trial_a = machine_jobs[static_cast<std::size_t>(best.a)];
                auto trial_b = machine_jobs[static_cast<std::size_t>(best.b)];
                std::swap(trial_a[static_cast<std::size_t>(best.ia)], trial_b[static_cast<std::size_t>(best.ib)]);

                machine_jobs[static_cast<std::size_t>(best.a)] = std::move(trial_a);
                machine_jobs[static_cast<std::size_t>(best.b)] = std::move(trial_b);
                machine_loads[static_cast<std::size_t>(best.a)] = machine_loads[static_cast<std::size_t>(best.a)] - best.pa + best.pb;
                machine_loads[static_cast<std::size_t>(best.b)] = machine_loads[static_cast<std::size_t>(best.b)] - best.pb + best.pa;
                machine_exact_cost[static_cast<std::size_t>(best.a)] = best.new_a;
                machine_exact_cost[static_cast<std::size_t>(best.b)] = best.new_b;

                stats.accepted_swap_inter = 1;
                stats.best_move = "swap_inter";
            }
            stats.best_tec = best.candidate_tec;
            stats.stop_reason = "diag_found_improving_move";
        }
    }

    if (!stats.improving_move_found)
    {
        if (elapsed_sec() > time_cap_sec)
            stats.stop_reason = "diag_time_cap";
        else if (stats.evaluated_insert_inter + stats.evaluated_swap_inter >= max_exact_evaluated_moves)
            stats.stop_reason = "diag_exact_eval_cap";
        else
            stats.stop_reason = "diag_no_improving_move";
        stats.best_tec = current_tec;
    }
}

struct PhaseYAcceptedMove {
    int round;
    int source;
    int target;
    int job_size;
    double delta_tec;
    bool was_exception;
};

// ---- Phase Y2 proposal machinery ----

struct PhaseYProposal {
    std::string proposal_name;
    std::vector<int> source_machines;
    std::vector<int> target_machines;
    int size_small = 0;
    int size_medium = 0;
    int size_large = 0;
    int max_candidates = 20;
    std::string ranking_hint;
    std::string diversity_rule;
    std::string fallback_if_empty;
};

static std::string json_extract_string(const std::string &json, const std::string &key)
{
    std::string search = "\"" + key + "\"";
    std::size_t pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return "";
    while (pos < json.size() && (json[pos] == ':' || json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r' || json[pos] == '\t')) ++pos;
    if (pos >= json.size() || json[pos] != '"') return "";
    ++pos;
    std::string out;
    while (pos < json.size() && json[pos] != '"') { out += json[pos]; ++pos; }
    return out;
}

static int json_extract_int(const std::string &json, const std::string &key)
{
    std::string search = "\"" + key + "\"";
    std::size_t pos = json.find(search);
    if (pos == std::string::npos) return 0;
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return 0;
    while (pos < json.size() && (json[pos] == ':' || json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r' || json[pos] == '\t')) ++pos;
    if (pos >= json.size()) return 0;
    std::string num;
    while (pos < json.size() && (isdigit(json[pos]) || json[pos] == '-')) { num += json[pos]; ++pos; }
    return num.empty() ? 0 : std::stoi(num);
}

static std::vector<int> json_extract_M_array(const std::string &json, const std::string &key, int &dropped)
{
    std::vector<int> out;
    std::string search = "\"" + key + "\"";
    std::size_t pos = json.find(search);
    if (pos == std::string::npos) return out;
    pos = json.find('[', pos + search.size());
    if (pos == std::string::npos) return out;
    std::size_t end = json.find(']', pos);
    if (end == std::string::npos) return out;
    std::string arr = json.substr(pos + 1, end - pos - 1);
    std::size_t i = 0;
    while (i < arr.size()) {
        std::size_t q1 = arr.find('"', i);
        if (q1 == std::string::npos) break;
        std::size_t q2 = arr.find('"', q1 + 1);
        if (q2 == std::string::npos) break;
        std::string val = arr.substr(q1 + 1, q2 - q1 - 1);
        if (!val.empty() && val[0] == 'M') {
            int mid = std::stoi(val.substr(1));
            out.push_back(mid);
        } else {
            ++dropped;
        }
        i = q2 + 1;
    }
    return out;
}

static std::vector<std::string> json_extract_string_array(const std::string &json, const std::string &key)
{
    std::vector<std::string> out;
    std::string search = "\"" + key + "\"";
    std::size_t pos = json.find(search);
    if (pos == std::string::npos) return out;
    pos = json.find('[', pos + search.size());
    if (pos == std::string::npos) return out;
    std::size_t end = json.find(']', pos);
    if (end == std::string::npos) return out;
    std::string arr = json.substr(pos + 1, end - pos - 1);
    std::size_t i = 0;
    while (i < arr.size()) {
        std::size_t q1 = arr.find('"', i);
        if (q1 == std::string::npos) break;
        std::size_t q2 = arr.find('"', q1 + 1);
        if (q2 == std::string::npos) break;
        out.push_back(arr.substr(q1 + 1, q2 - q1 - 1));
        i = q2 + 1;
    }
    return out;
}

static PhaseYProposal parse_phaseY_proposal(const std::string &path, int m, int &invalid_dropped)
{
    PhaseYProposal prop;
    std::ifstream f(path);
    if (!f.is_open()) { prop.proposal_name = "FILE_NOT_FOUND"; return prop; }
    std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();

    prop.proposal_name = json_extract_string(json, "proposal_name");
    prop.source_machines = json_extract_M_array(json, "source_machines", invalid_dropped);
    prop.target_machines = json_extract_M_array(json, "target_machines", invalid_dropped);

    auto sz_classes = json_extract_string_array(json, "job_size_classes");
    for (const auto &s : sz_classes) {
        if (s == "small") prop.size_small = 1;
        else if (s == "medium") prop.size_medium = 1;
        else if (s == "large") prop.size_large = 1;
    }

    prop.max_candidates = json_extract_int(json, "max_candidates");
    if (prop.max_candidates < 1) prop.max_candidates = 1;
    if (prop.max_candidates > 30) prop.max_candidates = 30;

    prop.ranking_hint = json_extract_string(json, "ranking_hint");
    prop.diversity_rule = json_extract_string(json, "diversity_rule");
    prop.fallback_if_empty = json_extract_string(json, "fallback_if_empty");

    {
        std::set<int> dedup(prop.source_machines.begin(), prop.source_machines.end());
        prop.source_machines.assign(dedup.begin(), dedup.end());
    }
    {
        std::set<int> dedup(prop.target_machines.begin(), prop.target_machines.end());
        prop.target_machines.assign(dedup.begin(), dedup.end());
    }

    std::vector<int> valid_src, valid_tgt;
    for (int s : prop.source_machines) if (s >= 0 && s < m) valid_src.push_back(s);
    invalid_dropped += static_cast<int>(prop.source_machines.size()) - static_cast<int>(valid_src.size());
    prop.source_machines = valid_src;
    for (int t : prop.target_machines) if (t >= 0 && t < m) valid_tgt.push_back(t);
    invalid_dropped += static_cast<int>(prop.target_machines.size()) - static_cast<int>(valid_tgt.size());
    prop.target_machines = valid_tgt;

    return prop;
}

struct PhaseYCand {
    int a;
    int ia;
    int p;
    int b;
    double score;
};

static void rank_phaseY_candidates(
    std::vector<PhaseYCand> &cands,
    const std::string &ranking_hint,
    const std::vector<std::vector<int>> &machine_jobs,
    const std::vector<int> &machine_loads,
    const std::vector<double> &machine_exact_cost,
    const std::vector<double> &machine_lb_cur,
    const std::vector<double> &source_gap,
    const std::vector<double> &source_density,
    const std::vector<double> &rates,
    const std::vector<double> &prices,
    int epsilon,
    std::mt19937 &rng)
{
    std::map<std::pair<int,int>,double> cheap_lb_cache;
    auto cheap_lb_delta = [&](int a, int ia, int p, int b) {
        auto key = std::make_pair(static_cast<std::int64_t>(a) * 1000000LL + ia, b);
        auto it = cheap_lb_cache.find(key);
        if (it != cheap_lb_cache.end()) return it->second;
        auto trial_a = machine_jobs[static_cast<std::size_t>(a)];
        auto trial_b = machine_jobs[static_cast<std::size_t>(b)];
        trial_a.erase(trial_a.begin() + ia);
        trial_b.push_back(p);
        double old_lb = machine_lb_cur[static_cast<std::size_t>(a)] + machine_lb_cur[static_cast<std::size_t>(b)];
        double new_lb = fallback_slot_lb(trial_a, prices, epsilon, rates[static_cast<std::size_t>(a)]) +
                        fallback_slot_lb(trial_b, prices, epsilon, rates[static_cast<std::size_t>(b)]);
        double d = std::max(0.0, old_lb - new_lb);
        cheap_lb_cache[key] = d;
        return d;
    };

    for (auto &c : cands) {
        if (ranking_hint == "cheap_lb") {
            c.score = cheap_lb_delta(c.a, c.ia, c.p, c.b);
        } else if (ranking_hint == "cost_gap") {
            c.score = source_gap[static_cast<std::size_t>(c.a)];
        } else if (ranking_hint == "slack") {
            c.score = static_cast<double>(epsilon - machine_loads[static_cast<std::size_t>(c.b)]);
        } else if (ranking_hint == "hybrid") {
            double lbd = cheap_lb_delta(c.a, c.ia, c.p, c.b);
            double gap = source_gap[static_cast<std::size_t>(c.a)];
            double slk = static_cast<double>(epsilon - machine_loads[static_cast<std::size_t>(c.b)]);
            c.score = 0.4*lbd + 0.35*gap + 0.25*slk;
        } else if (ranking_hint == "s2") {
            double src_gap_dens = source_gap[static_cast<std::size_t>(c.a)] / static_cast<double>(std::max(1, machine_loads[static_cast<std::size_t>(c.a)]));
            double src_cost_dens = source_density[static_cast<std::size_t>(c.a)];
            double tgt_headroom = std::max(0.0, machine_exact_cost[static_cast<std::size_t>(c.b)] - machine_lb_cur[static_cast<std::size_t>(c.b)]);
            double tgt_headroom_dens = tgt_headroom / static_cast<double>(std::max(1, machine_loads[static_cast<std::size_t>(c.b)]));
            double fullness_after = static_cast<double>(machine_loads[static_cast<std::size_t>(c.b)] + c.p) / static_cast<double>(std::max(1, epsilon));
            double job_norm = static_cast<double>(c.p) / static_cast<double>(std::max(1, epsilon));
            double tgt_cost_dens = machine_exact_cost[static_cast<std::size_t>(c.b)] / static_cast<double>(std::max(1, machine_loads[static_cast<std::size_t>(c.b)]));
            double s1 = 2.2*src_gap_dens + 1.4*src_cost_dens + 0.9*fullness_after + 0.8*tgt_headroom_dens + 0.5*job_norm - 0.3*tgt_cost_dens;
            double lbd = cheap_lb_delta(c.a, c.ia, c.p, c.b);
            double a_gap = source_gap[static_cast<std::size_t>(c.a)] / static_cast<double>(std::max(1, machine_loads[static_cast<std::size_t>(c.a)]));
            c.score = 0.60*s1 + 0.40*lbd + 0.30*a_gap;
        } else {
            c.score = 0.0;
        }
    }

    if (ranking_hint == "random") {
        std::shuffle(cands.begin(), cands.end(), rng);
    } else {
        std::sort(cands.begin(), cands.end(), [](const PhaseYCand &x, const PhaseYCand &y) {
            if (std::fabs(x.score - y.score) > 1e-12) return x.score > y.score;
            if (x.a != y.a) return x.a < y.a;
            if (x.b != y.b) return x.b < y.b;
            return x.ia < y.ia;
        });
    }
}

static std::vector<PhaseYCand> apply_diversity_and_select(
    const std::vector<PhaseYCand> &cands,
    const PhaseYProposal &prop,
    int max_k)
{
    std::vector<PhaseYCand> selected;
    std::map<int,int> src_cnt, tgt_cnt;
    std::set<std::pair<int,int>> pairs_used;

    int src_quota = std::max(1, (max_k + static_cast<int>(prop.source_machines.size()) - 1) / std::max(1, static_cast<int>(prop.source_machines.size())));
    int tgt_quota = std::max(1, (max_k + static_cast<int>(prop.target_machines.size()) - 1) / std::max(1, static_cast<int>(prop.target_machines.size())));

    for (const auto &c : cands) {
        if (static_cast<int>(selected.size()) >= max_k) break;
        if (prop.diversity_rule == "per_source" && src_cnt[c.a] >= src_quota) continue;
        if (prop.diversity_rule == "per_target" && tgt_cnt[c.b] >= tgt_quota) continue;
        if (prop.diversity_rule == "source_target_pair") {
            auto key = std::make_pair(c.a, c.b);
            if (pairs_used.count(key)) continue;
            pairs_used.insert(key);
        }
        selected.push_back(c);
        ++src_cnt[c.a];
        ++tgt_cnt[c.b];
    }
    return selected;
}

static PhaseYProposal generate_random_proposal(
    const std::vector<double> &machine_exact_cost,
    const std::vector<int> &machine_loads,
    int epsilon,
    int m,
    int max_candidates,
    std::mt19937 &rng)
{
    PhaseYProposal prop;
    prop.proposal_name = "random_neighborhood";
    prop.max_candidates = std::max(1, std::min(30, max_candidates));
    prop.ranking_hint = "random";
    prop.fallback_if_empty = "top_s2_same_budget";

    const int wanted = 4;

    std::vector<std::pair<double,int>> by_cost;
    for (int h = 0; h < m; ++h)
        if (machine_loads[static_cast<std::size_t>(h)] > 0 && machine_exact_cost[static_cast<std::size_t>(h)] < kInf * 0.5)
            by_cost.emplace_back(machine_exact_cost[static_cast<std::size_t>(h)], h);
    std::sort(by_cost.begin(), by_cost.end(), std::greater<>());
    int avail_src = static_cast<int>(by_cost.size());
    int ns = std::max(1, std::min(5, std::min(wanted, avail_src)));
    std::vector<int> src_pool;
    {
        std::vector<double> weights;
        double total_w = 0.0;
        for (int i = 0; i < avail_src; ++i) {
            double w = std::max(1.0, std::min(by_cost[static_cast<std::size_t>(i)].first, 10000.0));
            if (!std::isfinite(w) || w <= 0.0) w = 1.0;
            weights.push_back(w);
            total_w += w;
        }
        std::set<int> seen;
        if (total_w > 0.0 && avail_src > 0) {
            std::uniform_real_distribution<double> urd(0.0, total_w);
            for (int tries = 0; tries < ns * 10 && static_cast<int>(src_pool.size()) < ns; ++tries) {
                double dart = urd(rng);
                double acc = 0.0;
                for (int i = 0; i < avail_src; ++i) {
                    acc += weights[static_cast<std::size_t>(i)];
                    if (dart <= acc) {
                        int pick = by_cost[static_cast<std::size_t>(i)].second;
                        if (!seen.count(pick)) {
                            seen.insert(pick);
                            src_pool.push_back(pick);
                        }
                        break;
                    }
                }
            }
        }
        if (src_pool.empty() && !by_cost.empty()) src_pool.push_back(by_cost[0].second);
    }
    prop.source_machines = src_pool;

    std::vector<std::pair<int,int>> by_slack;
    for (int h = 0; h < m; ++h)
        by_slack.emplace_back(std::max(0, epsilon - machine_loads[static_cast<std::size_t>(h)]), h);
    std::sort(by_slack.begin(), by_slack.end(), std::greater<>());
    int nt = std::max(1, std::min(5, std::min(wanted, m)));
    std::vector<int> tgt_pool;
    {
        std::vector<int> weights_slack;
        int64_t total_sw = 0;
        for (int i = 0; i < m; ++i) {
            int w = std::max(1, by_slack[static_cast<std::size_t>(i)].first);
            weights_slack.push_back(w);
            total_sw += w;
        }
        std::set<int> seen;
        if (total_sw > 0 && m > 0) {
            std::uniform_int_distribution<int> uid(0, static_cast<int>(total_sw - 1));
            for (int tries = 0; tries < nt * 10 && static_cast<int>(tgt_pool.size()) < nt; ++tries) {
                int dart = uid(rng);
                int64_t acc = 0;
                for (int i = 0; i < m; ++i) {
                    acc += static_cast<int64_t>(weights_slack[static_cast<std::size_t>(i)]);
                    if (dart < acc) {
                        int pick = by_slack[static_cast<std::size_t>(i)].second;
                        if (!seen.count(pick)) {
                            seen.insert(pick);
                            tgt_pool.push_back(pick);
                        }
                        break;
                    }
                }
            }
        }
        if (tgt_pool.empty()) tgt_pool.push_back(by_slack[0].second);
    }
    prop.target_machines = tgt_pool;

    {
        std::uniform_int_distribution<int> coin(0, 1);
        int tries = 0;
        do {
            prop.size_small = coin(rng);
            prop.size_medium = coin(rng);
            prop.size_large = coin(rng);
            ++tries;
        } while (prop.size_small == 0 && prop.size_medium == 0 && prop.size_large == 0 && tries < 20);
        if (prop.size_small == 0 && prop.size_medium == 0 && prop.size_large == 0)
            prop.size_small = prop.size_medium = prop.size_large = 1;
    }

    std::uniform_int_distribution<int> div_coin(0, 3);
    int dv = div_coin(rng);
    const char *divs[] = {"per_source", "per_target", "source_target_pair", "none"};
    prop.diversity_rule = divs[dv];

    return prop;
}

static void execute_phaseY_proposal(
    const PhaseYProposal &prop,
    std::vector<std::vector<int>> &machine_jobs,
    std::vector<int> &machine_loads,
    std::vector<double> &machine_exact_cost,
    const std::vector<double> &machine_lb_cur,
    const std::vector<double> &source_gap,
    const std::vector<double> &source_density,
    const std::vector<double> &rates,
    const std::vector<double> &prices,
    int epsilon,
    int m,
    double per_machine_dp_limit_sec,
    ExactCostCache &cache,
    VndStats &stats,
    std::mt19937 &rng)
{
    std::vector<PhaseYCand> raw_cands;
    for (int sa : prop.source_machines) {
        if (sa < 0 || sa >= m) continue;
        const auto &jobs = machine_jobs[static_cast<std::size_t>(sa)];
        for (int ia = 0; ia < static_cast<int>(jobs.size()); ++ia) {
            int p = jobs[static_cast<std::size_t>(ia)];
            int sz_class = (p <= 4) ? 1 : (p <= 8) ? 2 : 3;
            bool ok = (sz_class == 1 && prop.size_small) || (sz_class == 2 && prop.size_medium) || (sz_class == 3 && prop.size_large);
            if (!ok) continue;
            for (int tb : prop.target_machines) {
                if (tb < 0 || tb >= m) continue;
                if (sa == tb) continue;
                if (machine_loads[static_cast<std::size_t>(tb)] + p > epsilon) continue;
                PhaseYCand c;
                c.a = sa; c.ia = ia; c.p = p; c.b = tb;
                raw_cands.push_back(c);
            }
        }
    }
    stats.phaseY_candidates_generated = static_cast<int>(raw_cands.size());

    if (raw_cands.empty()) {
        if (prop.fallback_if_empty == "random_same_budget") {
            PhaseYProposal fallback = generate_random_proposal(machine_exact_cost, machine_loads, epsilon, m, prop.max_candidates, rng);
            fallback.proposal_name = prop.proposal_name + "_fallback_random";
            auto fstats = stats;
            fstats.phaseY_candidates_generated = 0;
            fstats.phaseY_candidates_selected = 0;
            fstats.phaseY_candidates_evaluated = 0;
            fstats.phaseY_improvements = 0;
            fstats.phaseY_best_delta = 0.0;
            fstats.phaseY_accepted_delta = 0.0;
            fstats.phaseY_fallback_used = 1;
            fstats.phaseY_invalid_ids_dropped = 0;
            fstats.phaseY_sources_used = 0;
            fstats.phaseY_targets_used = 0;
            execute_phaseY_proposal(fallback, machine_jobs, machine_loads, machine_exact_cost,
                machine_lb_cur, source_gap, source_density, rates, prices, epsilon, m,
                per_machine_dp_limit_sec, cache, fstats, rng);
            stats.phaseY_candidates_generated = fstats.phaseY_candidates_generated;
            stats.phaseY_candidates_selected = fstats.phaseY_candidates_selected;
            stats.phaseY_candidates_evaluated = fstats.phaseY_candidates_evaluated;
            stats.phaseY_improvements = fstats.phaseY_improvements;
            stats.phaseY_best_delta = fstats.phaseY_best_delta;
            stats.phaseY_accepted_delta = fstats.phaseY_accepted_delta;
            stats.phaseY_fallback_used = 1;
            stats.phaseY_proposal_name = fallback.proposal_name;
        }
        return;
    }

    rank_phaseY_candidates(raw_cands, prop.ranking_hint, machine_jobs, machine_loads,
        machine_exact_cost, machine_lb_cur, source_gap, source_density, rates, prices, epsilon, rng);

    auto selected = apply_diversity_and_select(raw_cands, prop, prop.max_candidates);
    stats.phaseY_candidates_selected = static_cast<int>(selected.size());

    std::set<int> src_used, tgt_used;
    int improvements = 0;
    double best_delta = 0.0;
    int evaluated = 0;
    for (const auto &c : selected) {
        auto trial_a = machine_jobs[static_cast<std::size_t>(c.a)];
        auto trial_b = machine_jobs[static_cast<std::size_t>(c.b)];
        const int p_act = trial_a[static_cast<std::size_t>(c.ia)];
        trial_a.erase(trial_a.begin() + c.ia);
        trial_b.push_back(p_act);

        ++evaluated;
        const double new_a = exact_machine_cost_cached(trial_a, prices, epsilon, rates[static_cast<std::size_t>(c.a)], per_machine_dp_limit_sec, cache);
        const double new_b = exact_machine_cost_cached(trial_b, prices, epsilon, rates[static_cast<std::size_t>(c.b)], per_machine_dp_limit_sec, cache);
        if (!(new_a < kInf * 0.5) || !(new_b < kInf * 0.5)) continue;

        const double old_ab = machine_exact_cost[static_cast<std::size_t>(c.a)] + machine_exact_cost[static_cast<std::size_t>(c.b)];
        if (new_a + new_b + 1e-9 >= old_ab) continue;

        double delta = old_ab - (new_a + new_b);
        if (delta > best_delta) best_delta = delta;

        machine_jobs[static_cast<std::size_t>(c.a)] = std::move(trial_a);
        machine_jobs[static_cast<std::size_t>(c.b)] = std::move(trial_b);
        machine_loads[static_cast<std::size_t>(c.a)] -= p_act;
        machine_loads[static_cast<std::size_t>(c.b)] += p_act;
        machine_exact_cost[static_cast<std::size_t>(c.a)] = new_a;
        machine_exact_cost[static_cast<std::size_t>(c.b)] = new_b;
        ++stats.accepted_insert_inter;
        ++improvements;
        src_used.insert(c.a);
        tgt_used.insert(c.b);
    }

    stats.phaseY_candidates_evaluated = evaluated;
    stats.phaseY_improvements = improvements;
    stats.phaseY_best_delta = best_delta;
    stats.phaseY_accepted_delta = best_delta;
    stats.phaseY_sources_used = static_cast<int>(src_used.size());
    stats.phaseY_targets_used = static_cast<int>(tgt_used.size());
    stats.phaseY_proposal_name = prop.proposal_name;
}

static void write_phaseY_trace_json(
    const std::vector<std::vector<int>>& machine_jobs,
    const std::vector<int>& machine_loads,
    const std::vector<double>& machine_exact_cost,
    const std::vector<double>& machine_lb_cur,
    const std::vector<double>& source_gap,
    const std::vector<double>& source_lb,
    const std::vector<double>& source_density,
    const std::vector<double>& rates,
    const std::vector<double>& prices,
    int epsilon,
    int m,
    int round,
    int instance_id,
    double current_tec,
    const VndStats& stats,
    int no_hit_streak,
    bool had_shortlist_improvement,
    const ExactCostCache& cache,
    const std::vector<int>& core_source_hits,
    const std::vector<int>& core_target_hits,
    int evaluated_exact_this_round,
    int no_improving_this_round,
    int ring_count,
    const std::vector<PhaseYAcceptedMove>& last_accepted_moves)
{
    std::string out_dir = "research/learned_move_screening_20260420/iterations/20260510_phaseY_online_llm_neighborhood_proposal/traces/generated";
    std::string mkdir_cmd = "mkdir -p " + out_dir;
    std::system(mkdir_cmd.c_str());

    std::string cell_label = "Cell_unknown";
    if (instance_id == 61) cell_label = "Cell_A";
    else if (instance_id == 62) cell_label = "Cell_B";
    else if (instance_id == 65) cell_label = "Cell_C";
    else cell_label = "Cell_" + std::to_string(instance_id);

    std::string fname = out_dir + "/trace_" + cell_label + "_r" + std::to_string(round) + ".json";
    std::ofstream f(fname);
    if (!f.is_open()) return;

    std::vector<int> small_jobs(static_cast<std::size_t>(m), 0);
    std::vector<int> medium_jobs(static_cast<std::size_t>(m), 0);
    std::vector<int> large_jobs(static_cast<std::size_t>(m), 0);
    std::vector<int> job_counts(static_cast<std::size_t>(m), 0);
    int total_jobs = 0;
    for (int h = 0; h < m; ++h) {
        job_counts[static_cast<std::size_t>(h)] = static_cast<int>(machine_jobs[static_cast<std::size_t>(h)].size());
        total_jobs += job_counts[static_cast<std::size_t>(h)];
        for (int p : machine_jobs[static_cast<std::size_t>(h)]) {
            if (p <= 4) ++small_jobs[static_cast<std::size_t>(h)];
            else if (p <= 8) ++medium_jobs[static_cast<std::size_t>(h)];
            else ++large_jobs[static_cast<std::size_t>(h)];
        }
    }

    std::vector<std::pair<double,int>> cost_sorted;
    for (int h = 0; h < m; ++h)
        cost_sorted.emplace_back(machine_exact_cost[static_cast<std::size_t>(h)], h);
    std::sort(cost_sorted.begin(), cost_sorted.end());
    std::vector<int> cost_quartile(static_cast<std::size_t>(m), 0);
    for (std::size_t i = 0; i < cost_sorted.size(); ++i) {
        double frac = static_cast<double>(i) / static_cast<double>(std::max<std::size_t>(1, cost_sorted.size()-1));
        cost_quartile[static_cast<std::size_t>(cost_sorted[i].second)] = (frac < 0.25) ? 1 : (frac < 0.50) ? 2 : (frac < 0.75) ? 3 : 4;
    }

    std::string regime_str = (epsilon < 250) ? "tight" : (epsilon <= 350) ? "medium" : "loose";

    f << "{\n";
    f << "  \"trace_id\": \"" << cell_label << "_r" << round << "\",\n";
    f << "  \"cell_label\": \"" << cell_label << "\",\n";
    f << "  \"round\": " << round << ",\n";
    f << "  \"timestamp\": \"generated\",\n\n";

    f << "  \"regime\": {\n";
    f << "    \"cell_label\": \"" << cell_label << "\",\n";
    f << "    \"epsilon\": " << epsilon << ",\n";
    f << "    \"num_machines\": " << m << ",\n";
    f << "    \"total_jobs\": " << total_jobs << ",\n";
    f << "    \"epsilon_regime\": \"" << regime_str << "\",\n";
    f << "    \"job_size_range\": [1, 12],\n";
    f << "    \"episode_epsilon_progression\": [" << epsilon << "]\n";
    f << "  },\n\n";

    f << "  \"snapshot\": {\n";
    f << "    \"current_tec\": " << current_tec << ",\n";
    f << "    \"best_tec_episode\": " << current_tec << ",\n";
    f << "    \"tec_improvement_last_n_rounds\": 0.0,\n";
    f << "    \"no_hit_streak\": " << no_hit_streak << ",\n";
    f << "    \"total_rounds_completed\": " << (round+1) << ",\n";
    f << "    \"total_accepted_moves_so_far\": " << stats.accepted_insert_inter << ",\n";
    f << "    \"exact_dp_evals_so_far\": " << (cache.hits + cache.misses) << ",\n";
    f << "    \"core_lane_stagnation_active\": true,\n";
    f << "    \"exception_lane_active\": false,\n";
    f << "    \"stop_reason_guard\": \"none\"\n";
    f << "  },\n\n";

    f << "  \"machines\": [\n";
    for (int h = 0; h < m; ++h) {
        int load = machine_loads[static_cast<std::size_t>(h)];
        int slack_val = epsilon - load;
        double lp = (epsilon > 0) ? static_cast<double>(load) / static_cast<double>(epsilon) : 0.0;
        double ec = machine_exact_cost[static_cast<std::size_t>(h)];
        double rlb = machine_lb_cur[static_cast<std::size_t>(h)];
        double gap = source_gap[static_cast<std::size_t>(h)];
        double cd = (load > 0) ? ec / static_cast<double>(load) : 0.0;
        int jc = job_counts[static_cast<std::size_t>(h)];
        int sm = small_jobs[static_cast<std::size_t>(h)];
        int md = medium_jobs[static_cast<std::size_t>(h)];
        int lg = large_jobs[static_cast<std::size_t>(h)];
        int rate_val = static_cast<int>(rates[static_cast<std::size_t>(h)]);

        int src_hits = (h < static_cast<int>(core_source_hits.size())) ? core_source_hits[static_cast<std::size_t>(h)] : 0;
        int tgt_hits = (h < static_cast<int>(core_target_hits.size())) ? core_target_hits[static_cast<std::size_t>(h)] : 0;
        bool starved = (jc > 0 && src_hits == 0);

        f << "    {\"id\":\"M" << h << "\",\"jobs\":" << jc << ",\"load\":" << load
          << ",\"slack\":" << slack_val << ",\"load_pressure\":" << std::round(lp*1000.0)/1000.0
          << ",\"exact_cost\":" << std::round(ec*10.0)/10.0
          << ",\"relaxed_lb\":" << std::round(rlb*10.0)/10.0
          << ",\"gap\":" << std::round(gap*10.0)/10.0
          << ",\"cost_density\":" << std::round(cd*1000.0)/1000.0
          << ",\"small_jobs\":" << sm << ",\"medium_jobs\":" << md << ",\"large_jobs\":" << lg
          << ",\"core_source_hits\":" << src_hits << ",\"core_target_hits\":" << tgt_hits << ",\"rate\":" << rate_val
          << ",\"starved\":" << (starved ? "true" : "false") << "}";
        if (h < m - 1) f << ",";
        f << "\n";
    }
    f << "  ],\n\n";

    f << "  \"recent\": {\n";
    f << "    \"last_accepted_moves\": [\n";
    {
        int start = std::max(0, ring_count - 10);
        int display_count = std::min(ring_count, 10);
        for (int mi = 0; mi < display_count; ++mi) {
            int idx = (start + mi) % std::max(1, static_cast<int>(last_accepted_moves.size()));
            if (idx < 0 || idx >= static_cast<int>(last_accepted_moves.size())) continue;
            const auto& m = last_accepted_moves[static_cast<std::size_t>(idx)];
            f << "      {\"round\":" << m.round
              << ",\"source\":\"M" << m.source
              << "\",\"target\":\"M" << m.target
              << "\",\"job_size\":" << m.job_size
              << ",\"delta_tec\":" << std::round(m.delta_tec*10.0)/10.0
              << ",\"was_exception\":" << (m.was_exception ? "true" : "false") << "}";
            if (mi < display_count - 1) f << ",";
            f << "\n";
        }
    }
    f << "    ],\n";
    f << "    \"failed_summary\": {\n";
    f << "      \"evaluated_exact_this_round\": " << evaluated_exact_this_round << ",\n";
    f << "      \"no_improving_move_found\": " << (no_improving_this_round > 0 ? "true" : "false") << ",\n";
    f << "      \"note\": \"only Y1.1 signal: whether evaluated candidates produced an improvement\"\n";
    f << "    },\n";
    f << "    \"core_shortlist_composition\": {\n";
    f << "      \"distinct_sources_note\": \"not tracked per-round in trace probe\"\n";
    f << "    },\n";
    f << "    \"outside_pool_composition\": {\n";
    f << "      \"total_candidates\": " << stats.outside_pool_distinct_src << ",\n";
    f << "      \"distinct_sources\": " << stats.outside_pool_distinct_src << ",\n";
    f << "      \"distinct_targets\": " << stats.outside_pool_distinct_tgt << ",\n";
    f << "      \"source_coverage\": " << (m > 0 ? static_cast<double>(stats.outside_pool_distinct_src) / static_cast<double>(m) : 0.0) << "\n";
    f << "    },\n";
    f << "    \"next_round_budget\": {\n";
    f << "      \"core_budget\": 14\n";
    f << "    }\n";
    f << "  },\n\n";

    std::vector<std::pair<double,int>> by_cost;
    for (int h = 0; h < m; ++h)
        by_cost.emplace_back(machine_exact_cost[static_cast<std::size_t>(h)], h);
    std::sort(by_cost.begin(), by_cost.end(), std::greater<>());

    f << "  \"candidate_pools\": {\n";
    f << "    \"top_sources_by_cost\": [\n";
    for (int i = 0; i < std::min(5, m); ++i) {
        int h = by_cost[static_cast<std::size_t>(i)].second;
        f << "      {\"id\":\"M" << h << "\",\"exact_cost\":" << std::round(machine_exact_cost[static_cast<std::size_t>(h)]*10.0)/10.0
          << ",\"gap\":" << std::round(source_gap[static_cast<std::size_t>(h)]*10.0)/10.0
          << ",\"cost_density\":" << std::round(source_density[static_cast<std::size_t>(h)]*1000.0)/1000.0
          << ",\"jobs\":" << job_counts[static_cast<std::size_t>(h)] << "}";
        if (i < std::min(5,m)-1) f << ",";
        f << "\n";
    }
    f << "    ],\n";

    std::vector<std::pair<double,int>> by_gap;
    for (int h = 0; h < m; ++h)
        by_gap.emplace_back(source_gap[static_cast<std::size_t>(h)], h);
    std::sort(by_gap.begin(), by_gap.end(), std::greater<>());

    f << "    \"top_sources_by_gap\": [\n";
    for (int i = 0; i < std::min(5, m); ++i) {
        int h = by_gap[static_cast<std::size_t>(i)].second;
        f << "      {\"id\":\"M" << h << "\",\"gap\":" << std::round(source_gap[static_cast<std::size_t>(h)]*10.0)/10.0
          << ",\"exact_cost\":" << std::round(machine_exact_cost[static_cast<std::size_t>(h)]*10.0)/10.0 << "}";
        if (i < std::min(5,m)-1) f << ",";
        f << "\n";
    }
    f << "    ],\n";

    std::vector<std::pair<int,int>> by_slack;
    for (int h = 0; h < m; ++h)
        by_slack.emplace_back(epsilon - machine_loads[static_cast<std::size_t>(h)], h);
    std::sort(by_slack.begin(), by_slack.end(), std::greater<>());

    f << "    \"top_targets_by_slack\": [\n";
    for (int i = 0; i < std::min(5, m); ++i) {
        int h = by_slack[static_cast<std::size_t>(i)].second;
        int slk = epsilon - machine_loads[static_cast<std::size_t>(h)];
        double lp2 = (epsilon > 0) ? static_cast<double>(machine_loads[static_cast<std::size_t>(h)]) / static_cast<double>(epsilon) : 0.0;
        f << "      {\"id\":\"M" << h << "\",\"slack\":" << slk
          << ",\"load_pressure\":" << std::round(lp2*1000.0)/1000.0
          << ",\"jobs\":" << job_counts[static_cast<std::size_t>(h)] << "}";
        if (i < std::min(5,m)-1) f << ",";
        f << "\n";
    }
    f << "    ],\n";

    f << "    \"underexplored_sources\": [\n";
    {
        std::vector<std::pair<double,int>> ue_src;
        for (int h = 0; h < m; ++h) {
            int hits = (h < static_cast<int>(core_source_hits.size())) ? core_source_hits[static_cast<std::size_t>(h)] : 0;
            if (hits == 0 && job_counts[static_cast<std::size_t>(h)] > 0)
                ue_src.emplace_back(machine_exact_cost[static_cast<std::size_t>(h)], h);
        }
        std::sort(ue_src.begin(), ue_src.end(), std::greater<>());
        int ue_n = std::min(5, static_cast<int>(ue_src.size()));
        for (int i = 0; i < ue_n; ++i) {
            int h = ue_src[static_cast<std::size_t>(i)].second;
            f << "      {\"id\":\"M" << h << "\",\"exact_cost\":" << std::round(machine_exact_cost[static_cast<std::size_t>(h)]*10.0)/10.0
              << ",\"gap\":" << std::round(source_gap[static_cast<std::size_t>(h)]*10.0)/10.0
              << ",\"cost_density\":" << std::round(source_density[static_cast<std::size_t>(h)]*1000.0)/1000.0
              << ",\"core_hits\":0}";
            if (i < ue_n - 1) f << ",";
            f << "\n";
        }
    }
    f << "    ],\n";
    f << "    \"underexplored_targets\": [\n";
    {
        std::vector<std::pair<int,int>> ue_tgt;
        for (int h = 0; h < m; ++h) {
            int hits = (h < static_cast<int>(core_target_hits.size())) ? core_target_hits[static_cast<std::size_t>(h)] : 0;
            if (hits == 0) {
                int slk = epsilon - machine_loads[static_cast<std::size_t>(h)];
                ue_tgt.emplace_back(slk, h);
            }
        }
        std::sort(ue_tgt.begin(), ue_tgt.end(), std::greater<>());
        int ue_n = std::min(5, static_cast<int>(ue_tgt.size()));
        for (int i = 0; i < ue_n; ++i) {
            int h = ue_tgt[static_cast<std::size_t>(i)].second;
            int slk = epsilon - machine_loads[static_cast<std::size_t>(h)];
            double lp2 = (epsilon > 0) ? static_cast<double>(machine_loads[static_cast<std::size_t>(h)]) / static_cast<double>(epsilon) : 0.0;
            f << "      {\"id\":\"M" << h << "\",\"slack\":" << slk
              << ",\"load_pressure\":" << std::round(lp2*1000.0)/1000.0
              << ",\"jobs\":" << job_counts[static_cast<std::size_t>(h)]
              << ",\"core_hits\":0}";
            if (i < ue_n - 1) f << ",";
            f << "\n";
        }
    }
    f << "    ],\n";

    int q1_sm = 0, q1_md = 0, q1_lg = 0, q1_n = 0;
    int q2_sm = 0, q2_md = 0, q2_lg = 0, q2_n = 0;
    int q3_sm = 0, q3_md = 0, q3_lg = 0, q3_n = 0;
    int q4_sm = 0, q4_md = 0, q4_lg = 0, q4_n = 0;
    for (int h = 0; h < m; ++h) {
        int q = cost_quartile[static_cast<std::size_t>(h)];
        int* s = (q==1)?&q1_sm:(q==2)?&q2_sm:(q==3)?&q3_sm:&q4_sm;
        int* md = (q==1)?&q1_md:(q==2)?&q2_md:(q==3)?&q3_md:&q4_md;
        int* l = (q==1)?&q1_lg:(q==2)?&q2_lg:(q==3)?&q3_lg:&q4_lg;
        int* n = (q==1)?&q1_n:(q==2)?&q2_n:(q==3)?&q3_n:&q4_n;
        *s += small_jobs[static_cast<std::size_t>(h)];
        *md += medium_jobs[static_cast<std::size_t>(h)];
        *l += large_jobs[static_cast<std::size_t>(h)];
        *n += job_counts[static_cast<std::size_t>(h)];
    }
    auto pct = [](int part, int total) { return total > 0 ? static_cast<int>(std::round(static_cast<double>(part)*100.0/static_cast<double>(total))) : 0; };
    f << "    \"job_size_by_cost_quartile\": {\n";
    f << "      \"q4_highest\": {\"small_pct\":" << pct(q4_sm, q4_n) << ",\"medium_pct\":" << pct(q4_md, q4_n) << ",\"large_pct\":" << pct(q4_lg, q4_n) << "},\n";
    f << "      \"q3\": {\"small_pct\":" << pct(q3_sm, q3_n) << ",\"medium_pct\":" << pct(q3_md, q3_n) << ",\"large_pct\":" << pct(q3_lg, q3_n) << "},\n";
    f << "      \"q2\": {\"small_pct\":" << pct(q2_sm, q2_n) << ",\"medium_pct\":" << pct(q2_md, q2_n) << ",\"large_pct\":" << pct(q2_lg, q2_n) << "},\n";
    f << "      \"q1_lowest\": {\"small_pct\":" << pct(q1_sm, q1_n) << ",\"medium_pct\":" << pct(q1_md, q1_n) << ",\"large_pct\":" << pct(q1_lg, q1_n) << "}\n";
    f << "    }\n";
    f << "  },\n\n";

    f << "  \"prior_arms\": {\n";
    if (instance_id == 61) {
        f << "    \"trimmed\": 6884,\n";
        f << "    \"llm_exception\": 6869,\n";
        f << "    \"random_best\": 6852,\n";
        f << "    \"score_escape\": 6884,\n";
        f << "    \"phaseX_random_best\": 6884,\n";
        f << "    \"phaseX_llm_best\": 6884\n";
    } else if (instance_id == 62) {
        f << "    \"trimmed\": 9687,\n";
        f << "    \"llm_exception\": 9455,\n";
        f << "    \"random_best\": 9583,\n";
        f << "    \"score_escape\": 9484,\n";
        f << "    \"phaseX_random_best\": 9495,\n";
        f << "    \"phaseX_llm_best\": 9495\n";
    } else if (instance_id == 65) {
        f << "    \"trimmed\": 27031,\n";
        f << "    \"llm_exception\": 26926,\n";
        f << "    \"random_best\": 26262,\n";
        f << "    \"score_escape\": 26470,\n";
        f << "    \"phaseX_random_best\": 26263,\n";
        f << "    \"phaseX_llm_best\": 26478\n";
    } else {
        f << "    \"note\": \"no prior results for this cell\"\n";
    }
    f << "  }\n";
    f << "}\n";
    f.close();

    std::string md_fname = out_dir + "/trace_" + cell_label + "_r" + std::to_string(round) + ".md";
    std::ofstream fm(md_fname);
    if (fm.is_open()) {
        fm << "# Phase Y Trace — " << cell_label << ", Round " << round << "\n\n";
        fm << "| Field | Value |\n";
        fm << "|-------|-------|\n";
        fm << "| trace_id | " << cell_label << "_r" << round << " |\n";
        fm << "| cell_label | " << cell_label << " |\n";
        fm << "| round | " << round << " |\n";
        fm << "| epsilon | " << epsilon << " |\n";
        fm << "| num_machines | " << m << " |\n";
        fm << "| total_jobs | " << total_jobs << " |\n";
        fm << "| epsilon_regime | " << regime_str << " |\n";
        fm << "| current_tec | " << std::round(current_tec*10.0)/10.0 << " |\n";
        fm << "| no_hit_streak | " << no_hit_streak << " |\n";
        fm << "| total_accepted_moves_so_far | " << stats.accepted_insert_inter << " |\n";
        fm << "| core_lane_stagnation_active | true |\n";
        fm << "| exception_lane_active | false |\n\n";

        fm << "## Machine State Table\n\n";
        fm << "| M | J | L | S | LP | EC | RLB | Gap | CD | s | m | l | CS | CT | Rate | SL |\n";
        fm << "|:--|--:|--:|--:|----:|----:|----:|----:|-----:|--:|--:|--:|--:|--:|----:|--:|\n";
        for (int h = 0; h < m; ++h) {
            int load = machine_loads[static_cast<std::size_t>(h)];
            int slack_val = epsilon - load;
            double lp = (epsilon > 0) ? static_cast<double>(load) / static_cast<double>(epsilon) : 0.0;
            double ec = machine_exact_cost[static_cast<std::size_t>(h)];
            double rlb = machine_lb_cur[static_cast<std::size_t>(h)];
            double gap = source_gap[static_cast<std::size_t>(h)];
            double cd = (load > 0) ? ec / static_cast<double>(load) : 0.0;
            int rate_val = static_cast<int>(rates[static_cast<std::size_t>(h)]);
            int src_hits_md = (h < static_cast<int>(core_source_hits.size())) ? core_source_hits[static_cast<std::size_t>(h)] : 0;
            int tgt_hits_md = (h < static_cast<int>(core_target_hits.size())) ? core_target_hits[static_cast<std::size_t>(h)] : 0;
            std::string starved_md = (job_counts[static_cast<std::size_t>(h)] > 0 && src_hits_md == 0) ? "yes" : "no";
            fm << "| M" << h << " | " << job_counts[static_cast<std::size_t>(h)]
               << " | " << load << " | " << slack_val
               << " | " << std::fixed << std::setprecision(2) << lp
               << " | " << std::fixed << std::setprecision(1) << ec
               << " | " << std::fixed << std::setprecision(1) << rlb
               << " | " << std::fixed << std::setprecision(1) << gap
               << " | " << std::fixed << std::setprecision(3) << cd
               << " | " << small_jobs[static_cast<std::size_t>(h)]
               << " | " << medium_jobs[static_cast<std::size_t>(h)]
               << " | " << large_jobs[static_cast<std::size_t>(h)]
               << " | " << src_hits_md << " | " << tgt_hits_md << " | " << rate_val << " | " << starved_md << " |\n";
        }
        fm << "\n## Candidate Pool Summary\n\n";
        fm << "### Top Sources by Cost\n";
        fm << "| M | EC | Gap | CD | Jobs |\n";
        fm << "|:--|----:|----:|-----:|-----:|\n";
        for (int i = 0; i < std::min(5, m); ++i) {
            int h = by_cost[static_cast<std::size_t>(i)].second;
            fm << "| M" << h << " | " << std::fixed << std::setprecision(1) << machine_exact_cost[static_cast<std::size_t>(h)]
               << " | " << std::fixed << std::setprecision(1) << source_gap[static_cast<std::size_t>(h)]
               << " | " << std::fixed << std::setprecision(3) << source_density[static_cast<std::size_t>(h)]
               << " | " << job_counts[static_cast<std::size_t>(h)] << " |\n";
        }
        fm << "\n### Top Targets by Slack\n";
        fm << "| M | Slack | LP | Jobs |\n";
        fm << "|:--|:-----:|----:|-----:|\n";
        for (int i = 0; i < std::min(5, m); ++i) {
            int h = by_slack[static_cast<std::size_t>(i)].second;
            int slk = epsilon - machine_loads[static_cast<std::size_t>(h)];
            double lp2 = (epsilon > 0) ? static_cast<double>(machine_loads[static_cast<std::size_t>(h)]) / static_cast<double>(epsilon) : 0.0;
            fm << "| M" << h << " | " << slk << " | " << std::fixed << std::setprecision(2) << lp2
               << " | " << job_counts[static_cast<std::size_t>(h)] << " |\n";
        }
        fm << "\n### Underexplored Sources (core_hits=0)\n";
        {
            std::vector<std::pair<double,int>> ue_src;
            for (int h = 0; h < m; ++h) {
                int hits = (h < static_cast<int>(core_source_hits.size())) ? core_source_hits[static_cast<std::size_t>(h)] : 0;
                if (hits == 0 && job_counts[static_cast<std::size_t>(h)] > 0)
                    ue_src.emplace_back(machine_exact_cost[static_cast<std::size_t>(h)], h);
            }
            std::sort(ue_src.begin(), ue_src.end(), std::greater<>());
            if (!ue_src.empty()) {
                fm << "| M | EC | Gap | CD | Jobs |\n";
                fm << "|:--|----:|----:|-----:|-----:|\n";
                for (int i = 0; i < std::min(5, static_cast<int>(ue_src.size())); ++i) {
                    int h = ue_src[static_cast<std::size_t>(i)].second;
                    fm << "| M" << h << " | " << std::fixed << std::setprecision(1) << machine_exact_cost[static_cast<std::size_t>(h)]
                       << " | " << std::fixed << std::setprecision(1) << source_gap[static_cast<std::size_t>(h)]
                       << " | " << std::fixed << std::setprecision(3) << source_density[static_cast<std::size_t>(h)]
                       << " | " << job_counts[static_cast<std::size_t>(h)] << " |\n";
                }
            } else {
                fm << "(all non-empty machines reached by current shortlist)\n";
            }
        }
        fm << "\n### Underexplored Targets (core_hits=0)\n";
        {
            std::vector<std::pair<int,int>> ue_tgt;
            for (int h = 0; h < m; ++h) {
                int hits = (h < static_cast<int>(core_target_hits.size())) ? core_target_hits[static_cast<std::size_t>(h)] : 0;
                if (hits == 0) {
                    int slk = epsilon - machine_loads[static_cast<std::size_t>(h)];
                    ue_tgt.emplace_back(slk, h);
                }
            }
            std::sort(ue_tgt.begin(), ue_tgt.end(), std::greater<>());
            if (!ue_tgt.empty()) {
                fm << "| M | Slack | LP | Jobs |\n";
                fm << "|:--|:-----:|----:|-----:|\n";
                for (int i = 0; i < std::min(5, static_cast<int>(ue_tgt.size())); ++i) {
                    int h = ue_tgt[static_cast<std::size_t>(i)].second;
                    int slk = epsilon - machine_loads[static_cast<std::size_t>(h)];
                    double lp2 = (epsilon > 0) ? static_cast<double>(machine_loads[static_cast<std::size_t>(h)]) / static_cast<double>(epsilon) : 0.0;
                    fm << "| M" << h << " | " << slk << " | " << std::fixed << std::setprecision(2) << lp2
                       << " | " << job_counts[static_cast<std::size_t>(h)] << " |\n";
                }
            } else {
                fm << "(all machines targeted by current shortlist)\n";
            }
        }
        fm << "\n### Last Accepted Moves\n";
        {
            int total_pushes = ring_count;
            int size = static_cast<int>(last_accepted_moves.size());
            int display_count = std::min(total_pushes, 10);
            int oldest_logical = total_pushes - display_count;
            if (display_count > 0) {
                fm << "| # | Round | Source | Target | Size | Δ TEC | Exc? |\n";
                fm << "|--:|:-----:|--------|--------|:----:|------:|:----:|\n";
                for (int mi = 0; mi < display_count; ++mi) {
                    int push_order = oldest_logical + mi;
                    int buf_idx = push_order % std::max(1, size);
                    const auto& m = last_accepted_moves[static_cast<std::size_t>(buf_idx)];
                    fm << "| " << (mi+1) << " | " << m.round
                       << " | M" << m.source << " | M" << m.target
                       << " | " << m.job_size
                       << " | " << std::fixed << std::setprecision(1) << m.delta_tec
                       << " | " << (m.was_exception ? "yes" : "no") << " |\n";
                }
            } else {
                fm << "(no moves accepted yet)\n";
            }
        }
        fm << "\n**Note**: Trace generated by phaseY_trace_probe at first DiverseTrimmed stagnation.\n";
        fm.close();
    }

    std::cout << "[phaseY_trace_probe] Trace written to " << fname << "\n";
}

void run_insert_inter_screened_redesign(
    std::vector<std::vector<int>> &machine_jobs,
    std::vector<int> &machine_loads,
    std::vector<double> &machine_exact_cost,
    const std::vector<double> &rates,
    const std::vector<double> &prices,
    int epsilon,
    double per_machine_dp_limit_sec,
    int max_rounds,
    std::int64_t max_screened_moves_per_round,
    double time_cap_sec,
    InsertScreenMode mode,
    ExactCostCache &cache,
    VndStats &stats,
    const StageL1LogContext *log_ctx = nullptr)
{
    const int m = static_cast<int>(machine_jobs.size());
    const auto t0 = std::chrono::steady_clock::now();
    auto elapsed_sec = [&]() -> double
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };

    struct InsertCand
    {
        std::int64_t record_id = -1;
        std::int64_t job_id = -1;
        int job_type_id = -1;
        int a = -1;
        int b = -1;
        int ia = -1;
        int p = 0;
        double src_rate = 0.0;
        double tgt_rate = 0.0;
        int src_rate_class = -1;
        int tgt_rate_class = -1;
        double src_exact = 0.0;
        double tgt_exact = 0.0;
        double src_lb = 0.0;
        double tgt_lb = 0.0;
        double src_gap = 0.0;
        double tgt_gap = 0.0;
        int src_load = 0;
        int tgt_load = 0;
        int src_jobs = 0;
        int tgt_jobs = 0;
        int tgt_slack_before = 0;
        int projected_tgt_load_after = 0;
        int projected_src_load_after = 0;
        double rate_diff = 0.0;
        double cheap_lb_delta = 0.0;
        double src_density = 0.0;
        double tgt_density = 0.0;
        int src_top_expensive = 0;
        int tgt_top_expensive = 0;
        int epsilon_feasible = 0;
        double s1 = -kInf;
        double s2 = -kInf;
    };

    const bool is_diverse_mode =
        (mode == InsertScreenMode::DiverseTwoStage ||
         mode == InsertScreenMode::DiverseTrimmed ||
         mode == InsertScreenMode::DiverseBudgeted ||
         mode == InsertScreenMode::DenseLabeling ||
         mode == InsertScreenMode::ExceptionLaneLLM ||
         mode == InsertScreenMode::ExceptionLaneRandom ||
         mode == InsertScreenMode::ExceptionLaneRefined1 ||
         mode == InsertScreenMode::ExceptionLaneRefined2 ||
         mode == InsertScreenMode::ExceptionLaneRefined3 ||
         mode == InsertScreenMode::ScoreEscapeSampler ||
         mode == InsertScreenMode::PhaseXPolicyJson);
    const bool is_trimmed_mode = (mode == InsertScreenMode::DiverseTrimmed || mode == InsertScreenMode::ExceptionLaneLLM || mode == InsertScreenMode::ExceptionLaneRandom || mode == InsertScreenMode::ExceptionLaneRefined1 || mode == InsertScreenMode::ExceptionLaneRefined2 || mode == InsertScreenMode::ExceptionLaneRefined3 || mode == InsertScreenMode::ScoreEscapeSampler || mode == InsertScreenMode::PhaseXPolicyJson || mode == InsertScreenMode::PhaseYTraceProbe || mode == InsertScreenMode::PhaseYExecuteProposal || mode == InsertScreenMode::PhaseYRandomProposal);
    const bool is_exception_mode = (mode == InsertScreenMode::ExceptionLaneLLM || mode == InsertScreenMode::ExceptionLaneRandom || mode == InsertScreenMode::ExceptionLaneRefined1 || mode == InsertScreenMode::ExceptionLaneRefined2 || mode == InsertScreenMode::ExceptionLaneRefined3 || mode == InsertScreenMode::ScoreEscapeSampler || mode == InsertScreenMode::PhaseXPolicyJson);
    const bool is_exception_random = (mode == InsertScreenMode::ExceptionLaneRandom);
    const bool is_refined1 = (mode == InsertScreenMode::ExceptionLaneRefined1);
    const bool is_refined2 = (mode == InsertScreenMode::ExceptionLaneRefined2);
    const bool is_refined3 = (mode == InsertScreenMode::ExceptionLaneRefined3);
    const bool is_score_escape_sampler = (mode == InsertScreenMode::ScoreEscapeSampler);
    const bool is_phaseX_policy_json = (mode == InsertScreenMode::PhaseXPolicyJson);
    const bool is_phaseY_trace_probe = (mode == InsertScreenMode::PhaseYTraceProbe);
    const bool is_phaseY_execute = (mode == InsertScreenMode::PhaseYExecuteProposal);
    const bool is_phaseY_random = (mode == InsertScreenMode::PhaseYRandomProposal);
    const bool is_phaseY_any = (is_phaseY_trace_probe || is_phaseY_execute || is_phaseY_random);
    const bool is_budgeted_mode = (mode == InsertScreenMode::DiverseBudgeted);
    const bool is_dense_mode = (mode == InsertScreenMode::DenseLabeling);

    const int source_top_k = [&]() -> int
    {
        if (is_trimmed_mode)
            return std::max(3, std::min(m, 5));
        if (is_budgeted_mode)
            return std::max(4, std::min(m, 6));
        if (is_dense_mode)
            return std::max(8, std::min(m, 16));
        return std::max(4, std::min(m, 10));
    }();

    const int per_source_keep = [&]() -> int
    {
        if (mode == InsertScreenMode::DiverseTwoStage)
            return 8;
        if (is_trimmed_mode)
            return 3;
        if (is_budgeted_mode)
            return 4;
        if (is_dense_mode)
            return 14;
        return 1000000;
    }();

    const int shortlist_cap = [&]() -> int
    {
        if (mode == InsertScreenMode::DiverseTwoStage)
            return 96;
        if (is_trimmed_mode)
            return 32;
        if (is_budgeted_mode)
            return 40;
        if (is_dense_mode)
            return 320;
        return 64;
    }();

    const int exact_eval_cap = [&]() -> int
    {
        if (mode == InsertScreenMode::DiverseTwoStage)
            return 40;
        if (is_trimmed_mode)
            return 14;
        if (is_budgeted_mode)
            return 20;
        if (is_dense_mode)
            return 280;
        return 32;
    }();

    static int phaseY_consecutive_no_hit = 0;
    static int phaseY_instance_guard = -1;
    if (is_phaseY_any) {
        if (phaseY_instance_guard != g_audit_instance_id) {
            phaseY_consecutive_no_hit = 0;
            phaseY_instance_guard = g_audit_instance_id;
        }
    }

    static std::vector<int> phaseY_source_hits;
    static std::vector<int> phaseY_target_hits;
    static std::vector<PhaseYAcceptedMove> phaseY_ring;
    static int phaseY_ring_count = 0;
    static int phaseY_last_evaluated_exact = 0;
    static int phaseY_last_no_improving = 0;

    for (int round = 0; round < max_rounds; ++round)
    {
        if (elapsed_sec() > time_cap_sec)
        {
            stats.stop_reason = "time_cap";
            return;
        }

        std::vector<std::pair<double, int>> source_priority;
        source_priority.reserve(static_cast<std::size_t>(m));
        std::vector<double> source_gap(static_cast<std::size_t>(m), 0.0);
        std::vector<double> source_density(static_cast<std::size_t>(m), 0.0);
        std::vector<double> source_lb(static_cast<std::size_t>(m), 0.0);
        std::vector<double> machine_lb_cur(static_cast<std::size_t>(m), 0.0);
        std::vector<double> machine_gap_cur(static_cast<std::size_t>(m), 0.0);

        for (int h = 0; h < m; ++h)
        {
            machine_lb_cur[static_cast<std::size_t>(h)] = fallback_slot_lb(machine_jobs[static_cast<std::size_t>(h)], prices, epsilon, rates[static_cast<std::size_t>(h)]);
            machine_gap_cur[static_cast<std::size_t>(h)] = std::max(0.0, machine_exact_cost[static_cast<std::size_t>(h)] - machine_lb_cur[static_cast<std::size_t>(h)]);
        }

        double current_tec = 0.0;
        for (double c : machine_exact_cost)
            current_tec += c;

        std::set<int> top_expensive;
        {
            std::vector<std::pair<double, int>> expensive_rank;
            expensive_rank.reserve(static_cast<std::size_t>(m));
            for (int h = 0; h < m; ++h)
                expensive_rank.emplace_back(machine_exact_cost[static_cast<std::size_t>(h)], h);
            std::sort(expensive_rank.begin(), expensive_rank.end(), [](const auto &x, const auto &y)
                      {
                          if (std::fabs(x.first - y.first) > 1e-12)
                              return x.first > y.first;
                          return x.second < y.second;
                      });
            const int keep_top = std::max(1, std::min(m, 3));
            for (int i = 0; i < keep_top; ++i)
                top_expensive.insert(expensive_rank[static_cast<std::size_t>(i)].second);
        }

        std::vector<int> cost_rank_num(static_cast<std::size_t>(m), -1);
        {
            std::vector<std::pair<double, int>> cost_rank;
            cost_rank.reserve(static_cast<std::size_t>(m));
            for (int h = 0; h < m; ++h)
                cost_rank.emplace_back(machine_exact_cost[static_cast<std::size_t>(h)], h);
            std::sort(cost_rank.begin(), cost_rank.end(), [](const auto &x, const auto &y)
                      {
                          if (std::fabs(x.first - y.first) > 1e-12)
                              return x.first > y.first;
                          return x.second < y.second;
                      });
            for (int i = 0; i < static_cast<int>(cost_rank.size()); ++i)
                cost_rank_num[static_cast<std::size_t>(cost_rank[static_cast<std::size_t>(i)].second)] = i + 1;
        }

        std::vector<int> slack_rank_num(static_cast<std::size_t>(m), -1);
        {
            std::vector<std::pair<int, int>> slack_rank;
            slack_rank.reserve(static_cast<std::size_t>(m));
            for (int h = 0; h < m; ++h)
                slack_rank.emplace_back(epsilon - machine_loads[static_cast<std::size_t>(h)], h);
            std::sort(slack_rank.begin(), slack_rank.end(), [](const auto &x, const auto &y)
                      {
                          if (x.first != y.first)
                              return x.first < y.first;
                          return x.second < y.second;
                      });
            for (int i = 0; i < static_cast<int>(slack_rank.size()); ++i)
                slack_rank_num[static_cast<std::size_t>(slack_rank[static_cast<std::size_t>(i)].second)] = i + 1;
        }

        for (int h = 0; h < m; ++h)
        {
            if (machine_jobs[static_cast<std::size_t>(h)].empty())
                continue;
            const int load = machine_loads[static_cast<std::size_t>(h)];
            const double lb = machine_lb_cur[static_cast<std::size_t>(h)];
            const double gap = std::max(0.0, machine_exact_cost[static_cast<std::size_t>(h)] - lb);
            const double dens = machine_exact_cost[static_cast<std::size_t>(h)] / static_cast<double>(std::max(1, load));
            source_lb[static_cast<std::size_t>(h)] = lb;
            source_gap[static_cast<std::size_t>(h)] = gap;
            source_density[static_cast<std::size_t>(h)] = dens;
            const double pr = 0.7 * gap + 0.3 * machine_exact_cost[static_cast<std::size_t>(h)];
            source_priority.emplace_back(pr, h);
        }
        std::sort(source_priority.begin(), source_priority.end(), [](const auto &x, const auto &y)
                  {
                      if (std::fabs(x.first - y.first) > 1e-12)
                          return x.first > y.first;
                      return x.second < y.second;
                  });

        if (is_trimmed_mode && static_cast<int>(source_priority.size()) > source_top_k)
        {
            const double keep_threshold = source_priority[static_cast<std::size_t>(source_top_k - 1)].first;
            source_priority.erase(
                std::remove_if(source_priority.begin(), source_priority.end(), [&](const auto &item)
                               { return item.first + 1e-12 < keep_threshold; }),
                source_priority.end());
        }

        std::vector<InsertCand> pool;
        std::vector<InsertCand> outside_pool;
        pool.reserve(1024);
        std::int64_t screened_in_round = 0;

        for (int rank = 0; rank < static_cast<int>(source_priority.size()) && rank < source_top_k; ++rank)
        {
            if (elapsed_sec() > time_cap_sec || screened_in_round >= max_screened_moves_per_round)
                break;

            const int a = source_priority[static_cast<std::size_t>(rank)].second;
            const int load_a = machine_loads[static_cast<std::size_t>(a)];
            const auto &jobs_a = machine_jobs[static_cast<std::size_t>(a)];
            std::vector<InsertCand> source_pool;
            source_pool.reserve(128);

            for (int ia = 0; ia < static_cast<int>(jobs_a.size()); ++ia)
            {
                if (elapsed_sec() > time_cap_sec || screened_in_round >= max_screened_moves_per_round)
                    break;

                const int p = jobs_a[static_cast<std::size_t>(ia)];
                for (int b = 0; b < m; ++b)
                {
                    if (a == b)
                        continue;
                    if (elapsed_sec() > time_cap_sec || screened_in_round >= max_screened_moves_per_round)
                        break;

                    ++screened_in_round;
                    ++stats.evaluated_insert_inter;

                    const int load_b = machine_loads[static_cast<std::size_t>(b)];
                    const int eps_feasible = (load_a - p <= epsilon && load_b + p <= epsilon) ? 1 : 0;
                    std::int64_t record_id = -1;

                    if (log_ctx && log_ctx->logger && log_ctx->logger->enabled)
                    {
                        record_id = log_ctx->logger->allocate_record_id();
                        const double src_rate = rates[static_cast<std::size_t>(a)];
                        const double tgt_rate = rates[static_cast<std::size_t>(b)];
                        const double src_exact = machine_exact_cost[static_cast<std::size_t>(a)];
                        const double tgt_exact = machine_exact_cost[static_cast<std::size_t>(b)];
                        const double src_lb = source_lb[static_cast<std::size_t>(a)];
                        const double tgt_lb = machine_lb_cur[static_cast<std::size_t>(b)];
                        const double src_gap = source_gap[static_cast<std::size_t>(a)];
                        const double tgt_gap = machine_gap_cur[static_cast<std::size_t>(b)];
                        const double src_density = source_density[static_cast<std::size_t>(a)];
                        const double tgt_density = tgt_exact / static_cast<double>(std::max(1, load_b));

                        log_ctx->logger->write_broad(
                            record_id,
                            log_ctx->instance_id,
                            epsilon,
                            log_ctx->seed_id,
                            round,
                            static_cast<std::int64_t>(a) * 1000000LL + ia,
                            p,
                            p,
                            a,
                            src_rate,
                            static_cast<int>(std::llround(src_rate * 1000.0)),
                            src_exact,
                            src_lb,
                            src_gap,
                            load_a,
                            static_cast<double>(load_a) / static_cast<double>(std::max(1, epsilon)),
                            static_cast<int>(jobs_a.size()),
                            b,
                            tgt_rate,
                            static_cast<int>(std::llround(tgt_rate * 1000.0)),
                            tgt_exact,
                            tgt_lb,
                            tgt_gap,
                            load_b,
                            static_cast<double>(load_b) / static_cast<double>(std::max(1, epsilon)),
                            epsilon - load_b,
                            static_cast<int>(machine_jobs[static_cast<std::size_t>(b)].size()),
                            load_b + p,
                            load_a - p,
                            tgt_rate - src_rate,
                            0.0,
                            src_density,
                            tgt_density,
                            top_expensive.count(a) ? 1 : 0,
                            top_expensive.count(b) ? 1 : 0,
                            0.0,
                            0.0,
                            current_tec,
                            current_tec - log_ctx->start_tec,
                            stats.accepted_insert_inter,
                            exact_eval_cap,
                            exact_eval_cap,
                            0,
                            eps_feasible,
                            (log_ctx ? log_ctx->context_id : -1),
                            cost_rank_num[static_cast<std::size_t>(a)],
                            m,
                            slack_rank_num[static_cast<std::size_t>(b)],
                            m,
                            (epsilon > 0) ? static_cast<double>(load_a) / static_cast<double>(epsilon) : 0.0);
                    }

                    if (!eps_feasible)
                        continue;

                    const double src_gap_dens = source_gap[static_cast<std::size_t>(a)] / static_cast<double>(std::max(1, load_a));
                    const double src_cost_dens = source_density[static_cast<std::size_t>(a)];
                    const double tgt_cost_dens = machine_exact_cost[static_cast<std::size_t>(b)] / static_cast<double>(std::max(1, load_b));
                    const double tgt_lb = machine_lb_cur[static_cast<std::size_t>(b)];
                    const double tgt_headroom = std::max(0.0, machine_exact_cost[static_cast<std::size_t>(b)] - tgt_lb);
                    const double tgt_headroom_dens = tgt_headroom / static_cast<double>(std::max(1, load_b));
                    const double fullness_after = static_cast<double>(load_b + p) / static_cast<double>(std::max(1, epsilon));
                    const double job_norm = static_cast<double>(p) / static_cast<double>(std::max(1, epsilon));

                    const double score1 =
                        2.2 * src_gap_dens +
                        1.4 * src_cost_dens +
                        0.9 * fullness_after +
                        0.8 * tgt_headroom_dens +
                        0.5 * job_norm -
                        0.3 * tgt_cost_dens;

                    auto trial_a = jobs_a;
                    auto trial_b = machine_jobs[static_cast<std::size_t>(b)];
                    trial_a.erase(trial_a.begin() + ia);
                    trial_b.push_back(p);
                    const double old_lb =
                        source_lb[static_cast<std::size_t>(a)] +
                        tgt_lb;
                    const double new_lb =
                        fallback_slot_lb(trial_a, prices, epsilon, rates[static_cast<std::size_t>(a)]) +
                        fallback_slot_lb(trial_b, prices, epsilon, rates[static_cast<std::size_t>(b)]);
                    const double cheap_lb_delta = old_lb - new_lb;

                    InsertCand cand;
                    cand.a = a;
                    cand.b = b;
                    cand.ia = ia;
                    cand.p = p;
                    cand.s1 = score1;
                    cand.s2 = -kInf;
                    cand.record_id = record_id;
                    cand.job_id = static_cast<std::int64_t>(a) * 1000000LL + ia;
                    cand.job_type_id = p;
                    cand.src_rate = rates[static_cast<std::size_t>(a)];
                    cand.tgt_rate = rates[static_cast<std::size_t>(b)];
                    cand.src_rate_class = static_cast<int>(std::llround(cand.src_rate * 1000.0));
                    cand.tgt_rate_class = static_cast<int>(std::llround(cand.tgt_rate * 1000.0));
                    cand.src_exact = machine_exact_cost[static_cast<std::size_t>(a)];
                    cand.tgt_exact = machine_exact_cost[static_cast<std::size_t>(b)];
                    cand.src_lb = source_lb[static_cast<std::size_t>(a)];
                    cand.tgt_lb = tgt_lb;
                    cand.src_gap = source_gap[static_cast<std::size_t>(a)];
                    cand.tgt_gap = tgt_headroom;
                    cand.src_load = load_a;
                    cand.tgt_load = load_b;
                    cand.src_jobs = static_cast<int>(jobs_a.size());
                    cand.tgt_jobs = static_cast<int>(machine_jobs[static_cast<std::size_t>(b)].size());
                    cand.tgt_slack_before = epsilon - load_b;
                    cand.projected_tgt_load_after = load_b + p;
                    cand.projected_src_load_after = load_a - p;
                    cand.rate_diff = cand.tgt_rate - cand.src_rate;
                    cand.cheap_lb_delta = cheap_lb_delta;
                    cand.src_density = src_cost_dens;
                    cand.tgt_density = tgt_cost_dens;
                    cand.src_top_expensive = top_expensive.count(a) ? 1 : 0;
                    cand.tgt_top_expensive = top_expensive.count(b) ? 1 : 0;
                    cand.epsilon_feasible = 1;

                    source_pool.push_back(cand);
                }
            }

            std::sort(source_pool.begin(), source_pool.end(), [](const InsertCand &x, const InsertCand &y)
                      {
                          if (std::fabs(x.s1 - y.s1) > 1e-12)
                              return x.s1 > y.s1;
                          if (x.b != y.b)
                              return x.b < y.b;
                          return x.ia < y.ia;
                      });

            const int keep = std::min<int>(per_source_keep, static_cast<int>(source_pool.size()));
            if (is_trimmed_mode || is_budgeted_mode)
            {
                const int per_target_quota = (is_trimmed_mode && !is_exception_mode) ? 1 : (is_budgeted_mode ? 2 : 1);
                std::vector<int> target_count(static_cast<std::size_t>(m), 0);
                int selected = 0;
                for (const auto &cand : source_pool)
                {
                    if (selected >= keep)
                    {
                        if (is_exception_mode)
                            outside_pool.push_back(cand);
                        continue;
                    }
                    if (target_count[static_cast<std::size_t>(cand.b)] >= per_target_quota)
                    {
                        if (is_exception_mode)
                            outside_pool.push_back(cand);
                        continue;
                    }
                    ++target_count[static_cast<std::size_t>(cand.b)];
                    pool.push_back(cand);
                    ++selected;
                }
            }
            else
            {
                for (int i = 0; i < keep; ++i)
                    pool.push_back(source_pool[static_cast<std::size_t>(i)]);
                if (is_exception_mode)
                {
                    for (int i = keep; i < static_cast<int>(source_pool.size()); ++i)
                        outside_pool.push_back(source_pool[static_cast<std::size_t>(i)]);
                }
            }
        }

        if (is_phaseY_any) {
            phaseY_source_hits.assign(static_cast<std::size_t>(m), 0);
            phaseY_target_hits.assign(static_cast<std::size_t>(m), 0);
            for (const auto& c : pool) {
                if (c.a >= 0 && c.a < m) ++phaseY_source_hits[static_cast<std::size_t>(c.a)];
                if (c.b >= 0 && c.b < m) ++phaseY_target_hits[static_cast<std::size_t>(c.b)];
            }
        }

        if (pool.empty())
        {
            stats.stop_reason = "no_improving_move";
            return;
        }

        if (is_diverse_mode)
        {
            for (auto &c : pool)
            {
                const auto &jobs_a = machine_jobs[static_cast<std::size_t>(c.a)];
                const auto &jobs_b = machine_jobs[static_cast<std::size_t>(c.b)];
                auto trial_a = jobs_a;
                auto trial_b = jobs_b;
                const int p = trial_a[static_cast<std::size_t>(c.ia)];
                trial_a.erase(trial_a.begin() + c.ia);
                trial_b.push_back(p);

                const double old_lb =
                    fallback_slot_lb(jobs_a, prices, epsilon, rates[static_cast<std::size_t>(c.a)]) +
                    fallback_slot_lb(jobs_b, prices, epsilon, rates[static_cast<std::size_t>(c.b)]);
                const double new_lb =
                    fallback_slot_lb(trial_a, prices, epsilon, rates[static_cast<std::size_t>(c.a)]) +
                    fallback_slot_lb(trial_b, prices, epsilon, rates[static_cast<std::size_t>(c.b)]);
                const double lb_gain = std::max(0.0, old_lb - new_lb);

                const double a_gap = source_gap[static_cast<std::size_t>(c.a)] / static_cast<double>(std::max(1, machine_loads[static_cast<std::size_t>(c.a)]));
                c.s2 = 0.60 * c.s1 + 0.40 * lb_gain + 0.30 * a_gap;
            }

            std::sort(pool.begin(), pool.end(), [](const InsertCand &x, const InsertCand &y)
                      {
                          if (std::fabs(x.s2 - y.s2) > 1e-12)
                              return x.s2 > y.s2;
                          if (x.a != y.a)
                              return x.a < y.a;
                          if (x.b != y.b)
                              return x.b < y.b;
                          return x.ia < y.ia;
                      });
        }
        else
        {
            std::sort(pool.begin(), pool.end(), [](const InsertCand &x, const InsertCand &y)
                      {
                          if (std::fabs(x.s1 - y.s1) > 1e-12)
                              return x.s1 > y.s1;
                          if (x.a != y.a)
                              return x.a < y.a;
                          if (x.b != y.b)
                              return x.b < y.b;
                          return x.ia < y.ia;
                      });
        }

        bool improved = false;
        const int base_lim = std::min<int>(exact_eval_cap, std::min<int>(shortlist_cap, static_cast<int>(pool.size())));

        int lim = base_lim;
        std::vector<int> staged_caps;
        if (is_budgeted_mode)
        {
            const int c1 = std::max(4, std::min(8, lim));
            const int c2 = std::max(c1, std::min(16, lim));
            staged_caps.push_back(c1);
            if (c2 > c1)
                staged_caps.push_back(c2);
            if (lim > c2)
                staged_caps.push_back(lim);
        }

        int evaluated_exact_this_round = 0;
        int staged_idx = 0;
        int staged_target = is_budgeted_mode && !staged_caps.empty() ? staged_caps[0] : lim;
        const double top_score = (is_diverse_mode && !pool.empty()) ? pool.front().s2 : (!pool.empty() ? pool.front().s1 : -kInf);

        bool dense_has_improving = false;
        double dense_best_delta = 0.0;
        int dense_best_a = -1;
        int dense_best_b = -1;
        int dense_best_p = 0;
        double dense_best_new_a = kInf;
        double dense_best_new_b = kInf;
        std::vector<int> dense_best_jobs_a;
        std::vector<int> dense_best_jobs_b;

        for (int idx = 0; idx < lim && !improved; ++idx)
        {
            if (is_budgeted_mode && evaluated_exact_this_round >= staged_target)
            {
                if (staged_idx + 1 >= static_cast<int>(staged_caps.size()))
                    break;

                const auto &pivot = pool[static_cast<std::size_t>(idx - 1)];
                const double pivot_score = is_diverse_mode ? pivot.s2 : pivot.s1;
                if (top_score > -kInf * 0.5 && pivot_score + 1e-9 < 0.70 * top_score)
                    break;

                ++staged_idx;
                staged_target = staged_caps[static_cast<std::size_t>(staged_idx)];
            }

            if (elapsed_sec() > time_cap_sec)
            {
                stats.stop_reason = "time_cap";
                return;
            }

            ++evaluated_exact_this_round;

            const auto c = pool[static_cast<std::size_t>(idx)];
            auto trial_a = machine_jobs[static_cast<std::size_t>(c.a)];
            auto trial_b = machine_jobs[static_cast<std::size_t>(c.b)];
            const int p = trial_a[static_cast<std::size_t>(c.ia)];
            trial_a.erase(trial_a.begin() + c.ia);
            trial_b.push_back(p);

            const double new_a = exact_machine_cost_cached(
                trial_a,
                prices,
                epsilon,
                rates[static_cast<std::size_t>(c.a)],
                per_machine_dp_limit_sec,
                cache);
            const double new_b = exact_machine_cost_cached(
                trial_b,
                prices,
                epsilon,
                rates[static_cast<std::size_t>(c.b)],
                per_machine_dp_limit_sec,
                cache);
            if (!(new_a < kInf * 0.5) || !(new_b < kInf * 0.5))
                continue;

            const double old_ab = machine_exact_cost[static_cast<std::size_t>(c.a)] + machine_exact_cost[static_cast<std::size_t>(c.b)];
            const double new_ab = new_a + new_b;
            const double exact_delta = new_ab - old_ab;
            const int improving = (new_ab + 1e-9 < old_ab) ? 1 : 0;
            const int accepted = improving;
            const int exact_remaining = std::max(0, staged_target - evaluated_exact_this_round);
            const int exact_tier = is_budgeted_mode ? (staged_idx + 1) : 1;

            if (log_ctx && log_ctx->logger && log_ctx->logger->enabled)
            {
                log_ctx->logger->write_exact(
                    c.record_id,
                    log_ctx->instance_id,
                    epsilon,
                    log_ctx->seed_id,
                    round,
                    c.job_id,
                    c.p,
                    c.job_type_id,
                    c.a,
                    c.src_rate,
                    c.src_rate_class,
                    c.src_exact,
                    c.src_lb,
                    c.src_gap,
                    c.src_load,
                    static_cast<double>(c.src_load) / static_cast<double>(std::max(1, epsilon)),
                    c.src_jobs,
                    c.b,
                    c.tgt_rate,
                    c.tgt_rate_class,
                    c.tgt_exact,
                    c.tgt_lb,
                    c.tgt_gap,
                    c.tgt_load,
                    static_cast<double>(c.tgt_load) / static_cast<double>(std::max(1, epsilon)),
                    c.tgt_slack_before,
                    c.tgt_jobs,
                    c.projected_tgt_load_after,
                    c.projected_src_load_after,
                    c.rate_diff,
                    c.cheap_lb_delta,
                    c.src_density,
                    c.tgt_density,
                    c.src_top_expensive,
                    c.tgt_top_expensive,
                    c.s1,
                    c.s2,
                    current_tec,
                    current_tec - log_ctx->start_tec,
                    stats.accepted_insert_inter,
                    exact_eval_cap,
                    exact_remaining,
                    exact_tier,
                    c.epsilon_feasible,
                    (log_ctx ? log_ctx->context_id : -1),
                    cost_rank_num[static_cast<std::size_t>(c.a)],
                    m,
                    slack_rank_num[static_cast<std::size_t>(c.b)],
                    m,
                    (epsilon > 0) ? static_cast<double>(c.src_load) / static_cast<double>(epsilon) : 0.0,
                    old_ab,
                    new_ab,
                    exact_delta,
                    improving,
                    accepted);
            }

            if (new_ab + 1e-9 < old_ab)
            {
                if (is_dense_mode)
                {
                    const double delta = old_ab - new_ab;
                    if (!dense_has_improving || delta > dense_best_delta + 1e-9)
                    {
                        dense_has_improving = true;
                        dense_best_delta = delta;
                        dense_best_a = c.a;
                        dense_best_b = c.b;
                        dense_best_p = p;
                        dense_best_new_a = new_a;
                        dense_best_new_b = new_b;
                        dense_best_jobs_a = trial_a;
                        dense_best_jobs_b = trial_b;
                    }
                }
                else
                {
                    machine_jobs[static_cast<std::size_t>(c.a)] = std::move(trial_a);
                    machine_jobs[static_cast<std::size_t>(c.b)] = std::move(trial_b);
                    machine_loads[static_cast<std::size_t>(c.a)] -= p;
                    machine_loads[static_cast<std::size_t>(c.b)] += p;
                    machine_exact_cost[static_cast<std::size_t>(c.a)] = new_a;
                    machine_exact_cost[static_cast<std::size_t>(c.b)] = new_b;
                    ++stats.accepted_insert_inter;
                    improved = true;

                    if (is_phaseY_any) {
                        PhaseYAcceptedMove m;
                        m.round = round;
                        m.source = c.a;
                        m.target = c.b;
                        m.job_size = p;
                        m.delta_tec = old_ab - new_ab;
                        m.was_exception = false;
                        const int idx = phaseY_ring_count % 10;
                        if (phaseY_ring_count < 10)
                            phaseY_ring.push_back(m);
                        else
                            phaseY_ring[static_cast<std::size_t>(idx)] = m;
                        ++phaseY_ring_count;
                    }
                }
            }
        }

        bool had_shortlist_improvement = improved;

        if (is_phaseY_any) {
            phaseY_last_evaluated_exact = evaluated_exact_this_round;
            phaseY_last_no_improving = improved ? 0 : 1;
        }

        if (is_exception_mode)
        {
            if (is_score_escape_sampler)
            {
                static int score_escape_budget = 4;
                static int score_escape_no_hit = 0;
                static int score_escape_improved_last = 0;
                static int score_escape_instance_guard = -1;
                static bool score_escape_mode = false;
                static int score_escape_k = 3;

                if (score_escape_instance_guard != g_audit_instance_id)
                {
                    score_escape_budget = 4;
                    score_escape_no_hit = 0;
                    score_escape_improved_last = 0;
                    score_escape_mode = false;
                    score_escape_k = 3;
                    score_escape_instance_guard = g_audit_instance_id;
                }

                stats.phaseV_score_escape_normal_rounds += (score_escape_mode ? 0 : 1);
                stats.phaseV_score_escape_escape_rounds += (score_escape_mode ? 1 : 0);

                const int shortlist_improved = had_shortlist_improvement ? 1 : 0;
                if (score_escape_improved_last > 0) { score_escape_budget = std::min(12, score_escape_budget + 2); score_escape_no_hit = 0; }
                else { ++score_escape_no_hit; if (score_escape_no_hit >= 2 && shortlist_improved > 0) score_escape_budget = std::max(1, score_escape_budget - 1); }
                stats.exception_budget_used = score_escape_budget;

                std::vector<const InsertCand*> exc_cands;
                exc_cands.reserve(static_cast<std::size_t>(score_escape_budget));

                if (!score_escape_mode)
                {
                    struct ScoredCand { const InsertCand* cand; double score; };
                    std::vector<ScoredCand> scored;
                    scored.reserve(outside_pool.size());
                    for (auto& c : outside_pool)
                    {
                        if (!c.epsilon_feasible) continue;
                        double s2 = c.s2;
                        if (s2 < -1e9)
                        {
                            double lb_gain = std::max(0.0, c.cheap_lb_delta);
                            double a_gap = source_gap[static_cast<std::size_t>(c.a)] / static_cast<double>(std::max(1, machine_loads[static_cast<std::size_t>(c.a)]));
                            s2 = 0.60 * c.s1 + 0.40 * lb_gain + 0.30 * a_gap;
                        }
                        double tgt_slack = std::max(0.0, static_cast<double>(epsilon - c.tgt_load));
                        double slack_bonus = (tgt_slack / std::max(1.0, static_cast<double>(epsilon))) * 0.5;
                        double src_tightness = 0.0;
                        if (epsilon > 0) src_tightness = std::max(0.0, 1.0 - (static_cast<double>(c.src_load) / static_cast<double>(epsilon)));
                        double tightness_bonus = src_tightness * 0.2;
                        double sc = s2 + slack_bonus + tightness_bonus;
                        scored.push_back({&c, sc});
                    }
                    std::sort(scored.begin(), scored.end(), [](const ScoredCand& x, const ScoredCand& y) {
                        if (std::fabs(x.score - y.score) > 1e-12) return x.score > y.score;
                        return x.cand < y.cand;
                    });
                    const int exc_cap = std::min(score_escape_budget, static_cast<int>(scored.size()));
                    std::map<int, int> src_cnt, tgt_cnt;
                    for (const auto& sc : scored)
                    {
                        if (static_cast<int>(exc_cands.size()) >= exc_cap) break;
                        if (src_cnt[sc.cand->a] >= 3) continue;
                        if (tgt_cnt[sc.cand->b] >= 3) continue;
                        exc_cands.push_back(sc.cand);
                        ++src_cnt[sc.cand->a];
                        ++tgt_cnt[sc.cand->b];
                    }
                    stats.exception_candidates_considered = static_cast<int>(scored.size());
                }
                else
                {
                    std::map<std::pair<int, int>, double> pair_max_delta;
                    std::map<std::pair<int, int>, const InsertCand*> pair_best;
                    std::vector<const InsertCand*> eligible;
                    for (auto& c : outside_pool)
                    {
                        if (!c.epsilon_feasible) continue;
                        if (c.cheap_lb_delta <= 0.0) continue;
                        eligible.push_back(&c);
                        auto key = std::make_pair(c.a, c.b);
                        if (c.cheap_lb_delta > pair_max_delta[key])
                        {
                            pair_max_delta[key] = c.cheap_lb_delta;
                            pair_best[key] = &c;
                        }
                    }
                    if (!pair_best.empty())
                    {
                        std::vector<std::pair<std::pair<int,int>, double>> pairs(pair_max_delta.begin(), pair_max_delta.end());
                        std::sort(pairs.begin(), pairs.end(), [](const auto& x, const auto& y) { return x.second > y.second; });
                        stats.phaseV_score_escape_distinct_pairs = static_cast<int>(pairs.size());
                        stats.phaseV_score_escape_max_cheap_lb = pairs.empty() ? 0.0 : pairs[0].second;
                        const int K = 3;
                        std::map<int, int> src_cnt, tgt_cnt;
                        for (const auto& pr : pairs)
                        {
                            if (static_cast<int>(exc_cands.size()) >= K) break;
                            const auto* cand = pair_best.at(pr.first);
                            if (src_cnt[cand->a] >= 2) continue;
                            if (tgt_cnt[cand->b] >= 2) continue;
                            exc_cands.push_back(cand);
                            ++src_cnt[cand->a];
                            ++tgt_cnt[cand->b];
                        }
                    }
                    stats.exception_candidates_considered = static_cast<int>(eligible.size());
                }

                stats.phaseV_score_escape_candidates_considered = stats.exception_candidates_considered;

                {
                    std::set<int> op_src, op_tgt;
                    for (const auto& c : outside_pool)
                    {
                        if (!c.epsilon_feasible) continue;
                        op_src.insert(c.a); op_tgt.insert(c.b);
                    }
                    stats.outside_pool_distinct_src = static_cast<int>(op_src.size());
                    stats.outside_pool_distinct_tgt = static_cast<int>(op_tgt.size());
                }

                stats.exception_candidates_evaluated = static_cast<int>(exc_cands.size());
                stats.phaseV_score_escape_candidates_evaluated = static_cast<int>(exc_cands.size());

                {
                    std::map<int, int> sel_src_cnt, sel_tgt_cnt;
                    for (const auto* c : exc_cands) { ++sel_src_cnt[c->a]; ++sel_tgt_cnt[c->b]; }
                    stats.selected_distinct_src = static_cast<int>(sel_src_cnt.size());
                    stats.selected_distinct_tgt = static_cast<int>(sel_tgt_cnt.size());
                }

                int exc_improved_count = 0;
                double exc_best_delta = 0.0;
                for (const auto* c : exc_cands)
                {
                    if (elapsed_sec() > time_cap_sec) break;
                    if (static_cast<std::size_t>(c->a) >= machine_jobs.size()) continue;
                    if (static_cast<std::size_t>(c->b) >= machine_jobs.size()) continue;
                    if (static_cast<std::size_t>(c->ia) >= machine_jobs[static_cast<std::size_t>(c->a)].size()) continue;
                    const int p_actual = machine_jobs[static_cast<std::size_t>(c->a)][static_cast<std::size_t>(c->ia)];
                    if (p_actual != c->p) continue;
                    if (machine_loads[static_cast<std::size_t>(c->b)] + c->p > epsilon) continue;
                    auto trial_a = machine_jobs[static_cast<std::size_t>(c->a)];
                    auto trial_b = machine_jobs[static_cast<std::size_t>(c->b)];
                    trial_a.erase(trial_a.begin() + c->ia);
                    trial_b.push_back(c->p);
                    const double new_a = exact_machine_cost_cached(trial_a, prices, epsilon, rates[static_cast<std::size_t>(c->a)], per_machine_dp_limit_sec, cache);
                    const double new_b = exact_machine_cost_cached(trial_b, prices, epsilon, rates[static_cast<std::size_t>(c->b)], per_machine_dp_limit_sec, cache);
                    if (!(new_a < kInf * 0.5) || !(new_b < kInf * 0.5)) continue;
                    const double old_ab = machine_exact_cost[static_cast<std::size_t>(c->a)] + machine_exact_cost[static_cast<std::size_t>(c->b)];
                    if (new_a + new_b + 1e-9 < old_ab) {
                        double delta = old_ab - (new_a + new_b);
                        if (delta > exc_best_delta) exc_best_delta = delta;
                        ++exc_improved_count;
                        machine_jobs[static_cast<std::size_t>(c->a)] = std::move(trial_a);
                        machine_jobs[static_cast<std::size_t>(c->b)] = std::move(trial_b);
                        machine_loads[static_cast<std::size_t>(c->a)] -= c->p;
                        machine_loads[static_cast<std::size_t>(c->b)] += c->p;
                        machine_exact_cost[static_cast<std::size_t>(c->a)] = new_a;
                        machine_exact_cost[static_cast<std::size_t>(c->b)] = new_b;
                        ++stats.accepted_insert_inter;
                        improved = true;
                    }
                }

                stats.exception_improvement_count = exc_improved_count;
                stats.exception_best_delta = exc_best_delta;
                stats.exception_hit_rate = (exc_cands.size() > 0) ? static_cast<double>(exc_improved_count) / static_cast<double>(exc_cands.size()) : 0.0;
                stats.phaseV_score_escape_improvement_count = exc_improved_count;
                stats.phaseV_score_escape_best_delta = exc_best_delta;

                score_escape_improved_last = exc_improved_count;

                if (exc_improved_count > 0)
                {
                    score_escape_mode = false;
                    score_escape_k = 3;
                }
                else if (!score_escape_mode)
                {
                    if (score_escape_no_hit >= 2)
                    {
                        score_escape_mode = true;
                        score_escape_k = 3;
                    }
                }
                continue;
            }

            if (is_phaseX_policy_json)
            {
                static PhaseXPolicy phaseX_policy;
                static bool phaseX_loaded = false;
                static int phaseX_budget = 4;
                static int phaseX_no_hit = 0;
                static int phaseX_improved_last = 0;
                static int phaseX_inst_guard = -1;

                if (!phaseX_loaded)
                {
                    phaseX_policy = phaseX_read_policy(g_phaseX_policy_path);
                    phaseX_loaded = true;
                    stats.phaseX_policy_name = phaseX_policy.policy_name;
                }
                if (phaseX_inst_guard != g_audit_instance_id)
                {
                    phaseX_budget = phaseX_policy.initial_budget;
                    phaseX_no_hit = 0;
                    phaseX_improved_last = 0;
                    phaseX_inst_guard = g_audit_instance_id;
                    stats.phaseX_policy_name = phaseX_policy.policy_name;
                }

                const int total_jobs = static_cast<int>(std::accumulate(machine_jobs.begin(), machine_jobs.end(), std::size_t{0}, [](std::size_t s, const auto& v) { return s + v.size(); }));
                const double job_dens = m > 0 ? static_cast<double>(total_jobs) / static_cast<double>(m) : 0.0;
                const double eps_per_job = job_dens > 0.0 ? static_cast<double>(epsilon) / job_dens : 0.0;
                const bool is_guard = (eps_per_job <= 3.0);

                const int sl_improved = had_shortlist_improvement ? 1 : 0;
                if (phaseX_improved_last > 0) { phaseX_budget = std::min(phaseX_policy.max_budget, phaseX_budget + phaseX_policy.grow_on_hit); phaseX_no_hit = 0; }
                else { ++phaseX_no_hit; if (phaseX_no_hit >= 2 && sl_improved > 0) phaseX_budget = std::max(1, phaseX_budget - phaseX_policy.shrink_on_miss); }
                int exc_cap = phaseX_budget;
                if (is_guard && phaseX_policy.guard_max_budget > 0) exc_cap = std::min(exc_cap, phaseX_policy.guard_max_budget);
                else if (is_guard && phaseX_policy.guard_max_budget == 0) exc_cap = 0;

                bool in_escape = false;
                if (phaseX_policy.escape_mode != "none" && phaseX_policy.switch_after_no_hit > 0 && phaseX_no_hit >= phaseX_policy.switch_after_no_hit)
                {
                    in_escape = true;
                    if (phaseX_policy.switch_back_on_hit && phaseX_improved_last > 0) in_escape = false;
                }
                stats.phaseX_normal_rounds += (in_escape ? 0 : 1);
                stats.phaseX_escape_rounds += (in_escape ? 1 : 0);
                stats.exception_budget_used = exc_cap;

                struct PXCand { const InsertCand* cand; double score; };
                std::vector<PXCand> px_scored;
                px_scored.reserve(outside_pool.size());

                auto px_slack = [epsilon](const InsertCand& c) { double s = std::max(0.0, static_cast<double>(epsilon - c.tgt_load)); return (s / std::max(1.0, static_cast<double>(epsilon))) * 0.5; };
                auto px_llm = [epsilon,&px_slack](const InsertCand& c) { double s2 = c.s2; if (s2 < -1e9) { s2 = 0.60 * c.s1 + 0.40 * std::max(0.0, c.cheap_lb_delta); } double st = 0.0; if (epsilon>0) st = std::max(0.0, 1.0 - static_cast<double>(c.src_load) / static_cast<double>(epsilon)); return s2 + px_slack(c) + st * 0.2; };
                auto px_rnd = [](std::uint64_t sd) { std::mt19937_64 r(sd); return static_cast<double>(r()) / static_cast<double>(r.max()); };

                for (auto& c : outside_pool)
                {
                    if (!c.epsilon_feasible) continue;
                    if (phaseX_policy.require_positive_cheap_lb && c.cheap_lb_delta <= 0.0) continue;
                    double sc;
                    if (!in_escape) {
                        if (phaseX_policy.normal_mode == "llm_score") sc = px_llm(c);
                        else if (phaseX_policy.normal_mode == "s2") sc = c.s2;
                        else if (phaseX_policy.normal_mode == "random") { std::uint64_t sd = static_cast<std::uint64_t>(g_audit_instance_id)*1000003ULL + static_cast<std::uint64_t>(round)*11003ULL + static_cast<std::uint64_t>(&c-outside_pool.data()); sc = px_rnd(sd); }
                        else if (phaseX_policy.normal_mode == "cheap_lb") sc = c.cheap_lb_delta;
                        else { std::uint64_t sd = static_cast<std::uint64_t>(g_audit_instance_id)*1000003ULL + static_cast<std::uint64_t>(round)*11003ULL + static_cast<std::uint64_t>(&c-outside_pool.data()); double r = px_rnd(sd); sc = phaseX_policy.cheap_lb_weight*c.cheap_lb_delta + phaseX_policy.s2_weight*c.s2 + phaseX_policy.slack_weight*px_slack(c) + phaseX_policy.random_mix*r; }
                    } else {
                        if (phaseX_policy.escape_mode == "cheap_lb_pair") sc = c.cheap_lb_delta;
                        else if (phaseX_policy.escape_mode == "random_pair") { std::uint64_t sd = static_cast<std::uint64_t>(g_audit_instance_id)*1000003ULL + static_cast<std::uint64_t>(round)*11003ULL + static_cast<std::uint64_t>(c.a)*77003ULL + static_cast<std::uint64_t>(c.b); sc = px_rnd(sd); }
                        else if (phaseX_policy.escape_mode == "coverage") { sc = c.s2; std::set<int> us,ut; for (auto& x:px_scored) { us.insert(x.cand->a); ut.insert(x.cand->b); } if (!us.count(c.a)) sc+=phaseX_policy.coverage_bonus; if (!ut.count(c.b)) sc+=phaseX_policy.coverage_bonus; }
                        else sc = std::max(0.0, c.cheap_lb_delta) - c.s2;
                    }
                    px_scored.push_back({&c, sc});
                }
                stats.phaseX_candidates_considered = static_cast<int>(px_scored.size());
                stats.exception_candidates_considered = static_cast<int>(px_scored.size());
                std::sort(px_scored.begin(), px_scored.end(), [](const PXCand& x, const PXCand& y) { if (std::fabs(x.score-y.score)>1e-12) return x.score>y.score; return x.cand<y.cand; });
                const int eff_cap = std::min(exc_cap, static_cast<int>(px_scored.size()));
                std::vector<const InsertCand*> px_sel; px_sel.reserve(static_cast<std::size_t>(eff_cap));
                std::map<int,int> px_sc, px_tc;
                for (auto& sc : px_scored) { if (static_cast<int>(px_sel.size())>=eff_cap) break; if (px_sc[sc.cand->a]>=phaseX_policy.max_per_source) continue; if (px_tc[sc.cand->b]>=phaseX_policy.max_per_target) continue; px_sel.push_back(sc.cand); ++px_sc[sc.cand->a]; ++px_tc[sc.cand->b]; }
                stats.exception_candidates_evaluated = static_cast<int>(px_sel.size());
                stats.phaseX_candidates_evaluated = static_cast<int>(px_sel.size());
                { std::set<int> os,ot; for (auto& sc:px_scored) { os.insert(sc.cand->a); ot.insert(sc.cand->b); } stats.outside_pool_distinct_src = static_cast<int>(os.size()); stats.outside_pool_distinct_tgt = static_cast<int>(ot.size()); }
                { std::map<int,int> ss,st; for (auto* c:px_sel) { ++ss[c->a]; ++st[c->b]; } stats.selected_distinct_src = static_cast<int>(ss.size()); stats.selected_distinct_tgt = static_cast<int>(st.size()); }
                int px_imp = 0; double px_best = 0.0;
                for (auto* c : px_sel) {
                    if (elapsed_sec() > time_cap_sec) break;
                    if (static_cast<std::size_t>(c->a)>=machine_jobs.size() || static_cast<std::size_t>(c->b)>=machine_jobs.size()) continue;
                    if (static_cast<std::size_t>(c->ia)>=machine_jobs[static_cast<std::size_t>(c->a)].size()) continue;
                    if (machine_jobs[static_cast<std::size_t>(c->a)][static_cast<std::size_t>(c->ia)]!=c->p) continue;
                    if (machine_loads[static_cast<std::size_t>(c->b)]+c->p>epsilon) continue;
                    auto ta = machine_jobs[static_cast<std::size_t>(c->a)]; auto tb = machine_jobs[static_cast<std::size_t>(c->b)];
                    ta.erase(ta.begin()+c->ia); tb.push_back(c->p);
                    double na = exact_machine_cost_cached(ta,prices,epsilon,rates[static_cast<std::size_t>(c->a)],per_machine_dp_limit_sec,cache);
                    double nb = exact_machine_cost_cached(tb,prices,epsilon,rates[static_cast<std::size_t>(c->b)],per_machine_dp_limit_sec,cache);
                    if (!(na<kInf*0.5)||!(nb<kInf*0.5)) continue;
                    double oab = machine_exact_cost[static_cast<std::size_t>(c->a)]+machine_exact_cost[static_cast<std::size_t>(c->b)];
                    if (na+nb+1e-9<oab) { double d=oab-(na+nb); if (d>px_best) px_best=d; ++px_imp; machine_jobs[static_cast<std::size_t>(c->a)]=std::move(ta); machine_jobs[static_cast<std::size_t>(c->b)]=std::move(tb); machine_loads[static_cast<std::size_t>(c->a)]-=c->p; machine_loads[static_cast<std::size_t>(c->b)]+=c->p; machine_exact_cost[static_cast<std::size_t>(c->a)]=na; machine_exact_cost[static_cast<std::size_t>(c->b)]=nb; ++stats.accepted_insert_inter; improved=true; }
                }
                stats.exception_improvement_count = px_imp; stats.exception_best_delta = px_best;
                stats.exception_hit_rate = (px_sel.size()>0) ? static_cast<double>(px_imp)/static_cast<double>(px_sel.size()) : 0.0;
                stats.phaseX_improvement_count = px_imp; stats.phaseX_best_delta = px_best;
                phaseX_improved_last = px_imp;
                continue;
            }

            {
                std::set<int> distinct_src_set;
                std::set<int> distinct_tgt_set;
                std::map<int, int> src_share;
                std::map<int, int> tgt_share;
                for (const auto &cand : outside_pool)
                {
                    distinct_src_set.insert(cand.a);
                    distinct_tgt_set.insert(cand.b);
                    ++src_share[cand.a];
                    ++tgt_share[cand.b];
                }
                stats.outside_pool_distinct_src = static_cast<int>(distinct_src_set.size());
                stats.outside_pool_distinct_tgt = static_cast<int>(distinct_tgt_set.size());
                stats.outside_pool_max_src_share = 0;
                stats.outside_pool_max_tgt_share = 0;
                for (const auto &kv : src_share)
                    if (kv.second > stats.outside_pool_max_src_share)
                        stats.outside_pool_max_src_share = kv.second;
                for (const auto &kv : tgt_share)
                    if (kv.second > stats.outside_pool_max_tgt_share)
                        stats.outside_pool_max_tgt_share = kv.second;
            }
            stats.exception_candidates_considered = static_cast<int>(outside_pool.size());

            if (!had_shortlist_improvement && !outside_pool.empty())
            {
                int exception_budget = 10;
                std::vector<int> selected_indices;

                if (is_exception_random)
                {
                    std::vector<int> idx(outside_pool.size());
                    std::iota(idx.begin(), idx.end(), 0);
                    std::mt19937 rng(static_cast<unsigned int>(g_random_exception_seed + round));
                    std::shuffle(idx.begin(), idx.end(), rng);
                    exception_budget = std::min(10, static_cast<int>(idx.size()));
                    selected_indices.assign(idx.begin(), idx.begin() + exception_budget);
                }
                else if (is_refined1)
                {
                    std::map<std::pair<int, int>, std::vector<int>> strata;
                    for (int i = 0; i < static_cast<int>(outside_pool.size()); ++i)
                        strata[{outside_pool[static_cast<std::size_t>(i)].a, outside_pool[static_cast<std::size_t>(i)].b}].push_back(i);
                    std::vector<std::pair<int, int>> stratum_keys;
                    for (const auto &kv : strata)
                        stratum_keys.push_back(kv.first);
                    std::mt19937 rng_s(static_cast<unsigned int>(g_random_exception_seed + round * 1007));
                    std::shuffle(stratum_keys.begin(), stratum_keys.end(), rng_s);
                    exception_budget = std::min(10, static_cast<int>(stratum_keys.size()));
                    for (int k = 0; k < exception_budget; ++k)
                    {
                        const auto &cand_idx = strata[stratum_keys[static_cast<std::size_t>(k)]];
                        auto best_it = std::max_element(cand_idx.begin(), cand_idx.end(),
                            [&](int i1, int i2) { return outside_pool[static_cast<std::size_t>(i1)].s1 < outside_pool[static_cast<std::size_t>(i2)].s1; });
                        selected_indices.push_back(*best_it);
                    }
                }
                else if (is_refined2)
                {
                    std::map<std::pair<int, int>, std::vector<int>> strata;
                    for (int i = 0; i < static_cast<int>(outside_pool.size()); ++i)
                        strata[{outside_pool[static_cast<std::size_t>(i)].a, outside_pool[static_cast<std::size_t>(i)].b}].push_back(i);
                    std::vector<std::pair<std::pair<int, int>, std::vector<int>>> sorted_strata(strata.begin(), strata.end());
                    std::sort(sorted_strata.begin(), sorted_strata.end(),
                        [](const auto &x, const auto &y) { return x.second.size() < y.second.size(); });
                    exception_budget = std::min(5, static_cast<int>(sorted_strata.size()));
                    for (int k = 0; k < exception_budget; ++k)
                    {
                        const auto &cand_idx = sorted_strata[static_cast<std::size_t>(k)].second;
                        auto best_it = std::max_element(cand_idx.begin(), cand_idx.end(),
                            [&](int i1, int i2) { return outside_pool[static_cast<std::size_t>(i1)].s1 < outside_pool[static_cast<std::size_t>(i2)].s1; });
                        selected_indices.push_back(*best_it);
                    }
                }
                else if (is_refined3)
                {
                    std::vector<int> sorted_idx(outside_pool.size());
                    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
                    std::sort(sorted_idx.begin(), sorted_idx.end(),
                        [&](int i1, int i2) { return outside_pool[static_cast<std::size_t>(i1)].s1 > outside_pool[static_cast<std::size_t>(i2)].s1; });
                    std::set<int> covered_src;
                    std::set<int> covered_tgt;
                    exception_budget = 8;
                    for (int i = 0; i < static_cast<int>(sorted_idx.size()) && static_cast<int>(selected_indices.size()) < exception_budget; ++i)
                    {
                        int idx = sorted_idx[static_cast<std::size_t>(i)];
                        int ca = outside_pool[static_cast<std::size_t>(idx)].a;
                        int cb = outside_pool[static_cast<std::size_t>(idx)].b;
                        if (covered_src.count(ca) == 0 || covered_tgt.count(cb) == 0)
                        {
                            selected_indices.push_back(idx);
                            covered_src.insert(ca);
                            covered_tgt.insert(cb);
                        }
                    }
                    for (int i = 0; i < static_cast<int>(sorted_idx.size()) && static_cast<int>(selected_indices.size()) < exception_budget; ++i)
                    {
                        int idx = sorted_idx[static_cast<std::size_t>(i)];
                        if (std::find(selected_indices.begin(), selected_indices.end(), idx) == selected_indices.end())
                            selected_indices.push_back(idx);
                    }
                }
                else
                {
                    std::vector<int> sorted_idx(outside_pool.size());
                    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
                    std::sort(sorted_idx.begin(), sorted_idx.end(),
                        [&](int i1, int i2) { return outside_pool[static_cast<std::size_t>(i1)].s1 > outside_pool[static_cast<std::size_t>(i2)].s1; });
                    exception_budget = std::min(10, static_cast<int>(sorted_idx.size()));
                    selected_indices.assign(sorted_idx.begin(), sorted_idx.begin() + exception_budget);
                }

                {
                    std::set<int> sel_src_set;
                    std::set<int> sel_tgt_set;
                    std::map<int, int> sel_src_share;
                    std::map<int, int> sel_tgt_share;
                    for (int si : selected_indices)
                    {
                        const auto &cand = outside_pool[static_cast<std::size_t>(si)];
                        sel_src_set.insert(cand.a);
                        sel_tgt_set.insert(cand.b);
                        ++sel_src_share[cand.a];
                        ++sel_tgt_share[cand.b];
                    }
                    stats.selected_distinct_src = static_cast<int>(sel_src_set.size());
                    stats.selected_distinct_tgt = static_cast<int>(sel_tgt_set.size());
                    stats.selected_max_src_share = 0;
                    stats.selected_max_tgt_share = 0;
                    for (const auto &kv : sel_src_share)
                        if (kv.second > stats.selected_max_src_share)
                            stats.selected_max_src_share = kv.second;
                    for (const auto &kv : sel_tgt_share)
                        if (kv.second > stats.selected_max_tgt_share)
                            stats.selected_max_tgt_share = kv.second;
                }

                int evaluated = 0;
                double best_delta = 0.0;
                int best_cand_idx = -1;

                for (int si : selected_indices)
                {
                    if (elapsed_sec() > time_cap_sec)
                        break;
                    if (evaluated >= exception_budget)
                        break;

                    const auto &c = outside_pool[static_cast<std::size_t>(si)];
                    auto trial_a = machine_jobs[static_cast<std::size_t>(c.a)];
                    auto trial_b = machine_jobs[static_cast<std::size_t>(c.b)];
                    const int job_p = trial_a[static_cast<std::size_t>(c.ia)];
                    trial_a.erase(trial_a.begin() + c.ia);
                    trial_b.push_back(job_p);

                    ++evaluated;

                    const double new_a = exact_machine_cost_cached(trial_a, prices, epsilon, rates[static_cast<std::size_t>(c.a)], per_machine_dp_limit_sec, cache);
                    const double new_b = exact_machine_cost_cached(trial_b, prices, epsilon, rates[static_cast<std::size_t>(c.b)], per_machine_dp_limit_sec, cache);
                    if (!(new_a < kInf * 0.5) || !(new_b < kInf * 0.5))
                        continue;

                    const double old_ab = machine_exact_cost[static_cast<std::size_t>(c.a)] + machine_exact_cost[static_cast<std::size_t>(c.b)];
                    const double new_ab = new_a + new_b;
                    const double delta = old_ab - new_ab;

                    if (delta > best_delta + 1e-9)
                    {
                        best_delta = delta;
                        best_cand_idx = si;
                    }
                }

                stats.exception_candidates_evaluated = evaluated;
                stats.exception_budget_used = exception_budget;
                stats.exception_best_delta = (best_delta > 1e-9) ? best_delta : 0.0;
                stats.exception_improvement_count = (best_delta > 1e-9) ? 1 : 0;
                stats.exception_hit_rate = (evaluated > 0) ? static_cast<double>(stats.exception_improvement_count) / static_cast<double>(evaluated) : 0.0;

                if (best_delta > 1e-9 && best_cand_idx >= 0)
                {
                    const auto &c = outside_pool[static_cast<std::size_t>(best_cand_idx)];
                    auto trial_a_final = machine_jobs[static_cast<std::size_t>(c.a)];
                    auto trial_b_final = machine_jobs[static_cast<std::size_t>(c.b)];
                    const int p_final = trial_a_final[static_cast<std::size_t>(c.ia)];
                    trial_a_final.erase(trial_a_final.begin() + c.ia);
                    trial_b_final.push_back(p_final);
                    const double na = exact_machine_cost_cached(trial_a_final, prices, epsilon, rates[static_cast<std::size_t>(c.a)], per_machine_dp_limit_sec, cache);
                    const double nb = exact_machine_cost_cached(trial_b_final, prices, epsilon, rates[static_cast<std::size_t>(c.b)], per_machine_dp_limit_sec, cache);
                    machine_jobs[static_cast<std::size_t>(c.a)] = std::move(trial_a_final);
                    machine_jobs[static_cast<std::size_t>(c.b)] = std::move(trial_b_final);
                    machine_loads[static_cast<std::size_t>(c.a)] -= p_final;
                    machine_loads[static_cast<std::size_t>(c.b)] += p_final;
                    machine_exact_cost[static_cast<std::size_t>(c.a)] = na;
                    machine_exact_cost[static_cast<std::size_t>(c.b)] = nb;
                    ++stats.accepted_insert_inter;
                    improved = true;
                }
            }
            else
            {
                stats.exception_candidates_evaluated = 0;
                stats.exception_budget_used = 0;
                stats.exception_improvement_count = 0;
                stats.exception_best_delta = 0.0;
                stats.exception_hit_rate = 0.0;
                stats.selected_distinct_src = 0;
                stats.selected_distinct_tgt = 0;
                stats.selected_max_src_share = 0;
                stats.selected_max_tgt_share = 0;
            }

            {
                double max_pressure = 0.0;
                double sum_pressure = 0.0;
                for (int h = 0; h < m; ++h)
                {
                    double pressure = static_cast<double>(machine_loads[static_cast<std::size_t>(h)]) / static_cast<double>(std::max(1, epsilon));
                    if (pressure > max_pressure)
                        max_pressure = pressure;
                    sum_pressure += pressure;
                }
                stats.final_machine_load_pressure = max_pressure;
                stats.avg_machine_load_pressure = sum_pressure / static_cast<double>(m);
            }

            continue;
        }

        if (is_dense_mode && dense_has_improving)
        {
            machine_jobs[static_cast<std::size_t>(dense_best_a)] = std::move(dense_best_jobs_a);
            machine_jobs[static_cast<std::size_t>(dense_best_b)] = std::move(dense_best_jobs_b);
            machine_loads[static_cast<std::size_t>(dense_best_a)] -= dense_best_p;
            machine_loads[static_cast<std::size_t>(dense_best_b)] += dense_best_p;
            machine_exact_cost[static_cast<std::size_t>(dense_best_a)] = dense_best_new_a;
            machine_exact_cost[static_cast<std::size_t>(dense_best_b)] = dense_best_new_b;
            ++stats.accepted_insert_inter;
            improved = true;
        }

        if (!improved)
        {
            if (is_phaseY_any)
            {
                ++phaseY_consecutive_no_hit;
                if (phaseY_consecutive_no_hit >= 2)
                {
                    if (is_phaseY_trace_probe) {
                        stats.stop_reason = "phaseY_trace_written";
                        write_phaseY_trace_json(machine_jobs, machine_loads, machine_exact_cost,
                            machine_lb_cur, source_gap, source_lb, source_density,
                            rates, prices, epsilon, m, round, g_audit_instance_id,
                            current_tec, stats, phaseY_consecutive_no_hit,
                            had_shortlist_improvement, cache,
                            phaseY_source_hits, phaseY_target_hits,
                            evaluated_exact_this_round, phaseY_last_no_improving,
                            phaseY_ring_count, phaseY_ring);
                    } else if (is_phaseY_execute || is_phaseY_random) {
                        static std::mt19937 phaseY_rng(static_cast<unsigned int>(g_audit_instance_id * 10007 + 29));
                        if (is_phaseY_execute) {
                            const char *prop_path = std::getenv("PHASEY_PROPOSAL_PATH");
                            if (prop_path && prop_path[0]) {
                                int invalid = 0;
                                PhaseYProposal prop = parse_phaseY_proposal(prop_path, m, invalid);
                                stats.phaseY_invalid_ids_dropped = invalid;
                                stats.stop_reason = "phaseY_proposal_executed";
                                execute_phaseY_proposal(prop, machine_jobs, machine_loads, machine_exact_cost,
                                    machine_lb_cur, source_gap, source_density, rates, prices, epsilon, m,
                                    per_machine_dp_limit_sec, cache, stats, phaseY_rng);
                            }
                        } else {
                            int k = 20;
                            const char *k_env = std::getenv("PHASEY_PROPOSAL_K");
                            if (k_env && k_env[0]) k = std::stoi(k_env);
                            const char *seed_env = std::getenv("PHASEY_RANDOM_SEED");
                            unsigned int seed = seed_env ? static_cast<unsigned int>(std::stoi(seed_env)) : (static_cast<unsigned int>(g_audit_instance_id * 10007 + 31));
                            phaseY_rng.seed(seed);
                            PhaseYProposal prop = generate_random_proposal(machine_exact_cost, machine_loads, epsilon, m, k, phaseY_rng);
                            stats.phaseY_proposal_name = prop.proposal_name;
                            stats.stop_reason = "phaseY_random_executed";
                            execute_phaseY_proposal(prop, machine_jobs, machine_loads, machine_exact_cost,
                                machine_lb_cur, source_gap, source_density, rates, prices, epsilon, m,
                                per_machine_dp_limit_sec, cache, stats, phaseY_rng);
                        }
                    }
                    return;
                }
                continue;
            }
            stats.stop_reason = "no_improving_move";
            return;
        }
        else
        {
            if (is_phaseY_any)
                phaseY_consecutive_no_hit = 0;
        }
    }

    stats.stop_reason = "max_rounds";

    if (is_phaseY_any)
    {
        std::vector<double> lb_cur(static_cast<std::size_t>(m), 0.0);
        std::vector<double> gap_cur(static_cast<std::size_t>(m), 0.0);
        std::vector<double> src_lb(static_cast<std::size_t>(m), 0.0);
        std::vector<double> src_dens(static_cast<std::size_t>(m), 0.0);
        double total_ec = 0.0;
        for (int h = 0; h < m; ++h) {
            lb_cur[static_cast<std::size_t>(h)] = fallback_slot_lb(machine_jobs[static_cast<std::size_t>(h)], prices, epsilon, rates[static_cast<std::size_t>(h)]);
            gap_cur[static_cast<std::size_t>(h)] = std::max(0.0, machine_exact_cost[static_cast<std::size_t>(h)] - lb_cur[static_cast<std::size_t>(h)]);
            src_lb[static_cast<std::size_t>(h)] = lb_cur[static_cast<std::size_t>(h)];
            int load_h = machine_loads[static_cast<std::size_t>(h)];
            src_dens[static_cast<std::size_t>(h)] = (load_h > 0) ? machine_exact_cost[static_cast<std::size_t>(h)] / static_cast<double>(load_h) : 0.0;
            total_ec += machine_exact_cost[static_cast<std::size_t>(h)];
        }
        if (is_phaseY_trace_probe) {
        write_phaseY_trace_json(machine_jobs, machine_loads, machine_exact_cost,
            lb_cur, gap_cur, src_lb, src_dens,
            rates, prices, epsilon, m, max_rounds - 1, g_audit_instance_id,
            total_ec, stats, phaseY_consecutive_no_hit,
            false, cache,
            phaseY_source_hits, phaseY_target_hits,
            phaseY_last_evaluated_exact, phaseY_last_no_improving,
            phaseY_ring_count, phaseY_ring);
        }
        else if (is_phaseY_execute || is_phaseY_random) {
            static std::mt19937 phaseY_rng2(static_cast<unsigned int>(g_audit_instance_id * 10007 + 53));
            if (is_phaseY_execute) {
                    const char *prop_path = std::getenv("PHASEY_PROPOSAL_PATH");
                    if (prop_path && prop_path[0]) {
                        int invalid = 0;
                        PhaseYProposal prop = parse_phaseY_proposal(prop_path, m, invalid);
                        stats.phaseY_invalid_ids_dropped = invalid;
                        stats.stop_reason = "phaseY_proposal_executed_max_rounds";
                        execute_phaseY_proposal(prop, machine_jobs, machine_loads, machine_exact_cost,
                            lb_cur, gap_cur, src_dens, rates, prices, epsilon, m,
                            per_machine_dp_limit_sec, cache, stats, phaseY_rng2);
                    }
                } else {
                    int k = 20;
                    const char *k_env = std::getenv("PHASEY_PROPOSAL_K");
                    if (k_env && k_env[0]) k = std::stoi(k_env);
                    const char *seed_env = std::getenv("PHASEY_RANDOM_SEED");
                    unsigned int seed = seed_env ? static_cast<unsigned int>(std::stoi(seed_env)) : (static_cast<unsigned int>(g_audit_instance_id * 10007 + 59));
                    phaseY_rng2.seed(seed);
                    PhaseYProposal prop = generate_random_proposal(machine_exact_cost, machine_loads, epsilon, m, k, phaseY_rng2);
                    stats.phaseY_proposal_name = prop.proposal_name;
                    stats.stop_reason = "phaseY_random_executed_max_rounds";
                    execute_phaseY_proposal(prop, machine_jobs, machine_loads, machine_exact_cost,
                        lb_cur, gap_cur, src_dens, rates, prices, epsilon, m,
                        per_machine_dp_limit_sec, cache, stats, phaseY_rng2);
                }
            }

    }
}

VariantResult evaluate_history_repair_step(
    const Assignment &prev_assignment,
    const std::vector<double> &rates,
    const std::vector<double> &prices,
    int epsilon_prev,
    int epsilon,
    const std::string &variant,
    double per_machine_dp_limit_sec,
    int ls_max_rounds,
    std::int64_t ls_max_moves_per_round,
    double ls_time_cap_sec,
    Assignment &next_assignment)
{
    VariantResult out;
    out.variant = variant;
    out.epsilon_prev = epsilon_prev;
    out.feasible = prev_assignment.feasible;
    if (!out.feasible)
        return out;

    const auto t0 = std::chrono::steady_clock::now();
    const int m = static_cast<int>(rates.size());
    auto mode_opt = parse_repair_mode(variant);
    if (!mode_opt.has_value())
    {
        out.feasible = false;
        return out;
    }
    const RepairMode mode = *mode_opt;
    const bool use_relocate_cleanup = history_uses_relocate_cleanup(variant);
    out.relocate_cleanup_used = use_relocate_cleanup ? 1 : 0;

    std::vector<std::vector<int>> jobs_by_machine = prev_assignment.machine_job_lengths;
    std::vector<int> loads = prev_assignment.machine_loads;

    std::vector<RepairMoveItem> displaced;
    displaced.reserve(64);

    for (int h = 0; h < m; ++h)
    {
        auto &jobs_h = jobs_by_machine[static_cast<std::size_t>(h)];
        while (loads[static_cast<std::size_t>(h)] > epsilon)
        {
            const int overload = loads[static_cast<std::size_t>(h)] - epsilon;
            int best_idx = -1;
            int best_len = std::numeric_limits<int>::max();
            for (int i = 0; i < static_cast<int>(jobs_h.size()); ++i)
            {
                const int p = jobs_h[static_cast<std::size_t>(i)];
                if (p < best_len)
                {
                    best_len = p;
                    best_idx = i;
                }
                if (p == overload)
                {
                    best_len = p;
                    best_idx = i;
                    break;
                }
            }
            if (best_idx < 0)
            {
                out.feasible = false;
                return out;
            }

            const int p = jobs_h[static_cast<std::size_t>(best_idx)];
            RepairMoveItem item;
            item.p = p;
            item.source_machine = h;
            item.priority_score = rates[static_cast<std::size_t>(h)] * static_cast<double>(p);
            displaced.push_back(item);
            jobs_h.erase(jobs_h.begin() + best_idx);
            loads[static_cast<std::size_t>(h)] -= p;
        }
    }

    out.displaced_jobs = static_cast<std::int64_t>(displaced.size());

    if (mode == RepairMode::DpRanked)
    {
        std::sort(displaced.begin(), displaced.end(), [](const RepairMoveItem &a, const RepairMoveItem &b)
                  {
                      if (a.p != b.p)
                          return a.p > b.p;
                      return a.source_machine < b.source_machine;
                  });
    }
    else
    {
        std::sort(displaced.begin(), displaced.end(), [](const RepairMoveItem &a, const RepairMoveItem &b)
                  {
                      if (std::fabs(a.priority_score - b.priority_score) > 1e-12)
                          return a.priority_score > b.priority_score;
                      if (a.p != b.p)
                          return a.p > b.p;
                      return a.source_machine < b.source_machine;
                  });
    }

    ExactCostCache repair_cache;
    for (const auto &item : displaced)
    {
        int best_h = -1;
        double best_score = kInf;
        int best_load = std::numeric_limits<int>::max();

        for (int h = 0; h < m; ++h)
        {
            if (loads[static_cast<std::size_t>(h)] + item.p > epsilon)
                continue;

            ++out.reinsertion_candidates_scored;

            const auto &cur_jobs = jobs_by_machine[static_cast<std::size_t>(h)];
            auto trial_jobs = cur_jobs;
            trial_jobs.push_back(item.p);

            double cur_lb = relaxed_machine_lb(cur_jobs, prices, epsilon, rates[static_cast<std::size_t>(h)]);
            if (!(cur_lb < kInf * 0.5))
                cur_lb = fallback_slot_lb(cur_jobs, prices, epsilon, rates[static_cast<std::size_t>(h)]);

            double trial_lb = relaxed_machine_lb(trial_jobs, prices, epsilon, rates[static_cast<std::size_t>(h)]);
            if (!(trial_lb < kInf * 0.5))
                trial_lb = fallback_slot_lb(trial_jobs, prices, epsilon, rates[static_cast<std::size_t>(h)]);

            const double score = trial_lb - cur_lb;
            const int load_after = loads[static_cast<std::size_t>(h)] + item.p;
            if (score + 1e-9 < best_score)
            {
                best_score = score;
                best_h = h;
                best_load = load_after;
            }
            else if (std::fabs(score - best_score) <= 1e-9)
            {
                if (load_after < best_load || (load_after == best_load && h < best_h))
                {
                    best_h = h;
                    best_load = load_after;
                }
            }
        }

        if (best_h < 0)
        {
            out.feasible = false;
            return out;
        }

        auto trial_jobs = jobs_by_machine[static_cast<std::size_t>(best_h)];
        trial_jobs.push_back(item.p);
        const std::int64_t misses_before = repair_cache.misses;
        const double c = exact_machine_cost_cached(
            trial_jobs,
            prices,
            epsilon,
            rates[static_cast<std::size_t>(best_h)],
            per_machine_dp_limit_sec,
            repair_cache);
        out.exact_dp_evals_repair += (repair_cache.misses - misses_before);
        if (!(c < kInf * 0.5))
        {
            out.feasible = false;
            return out;
        }

        jobs_by_machine[static_cast<std::size_t>(best_h)] = std::move(trial_jobs);
        loads[static_cast<std::size_t>(best_h)] += item.p;
    }

    out.machine_job_counts.assign(static_cast<std::size_t>(m), 0);
    out.machine_exact_cost.assign(static_cast<std::size_t>(m), kInf);
    out.machine_relaxed_lb.assign(static_cast<std::size_t>(m), kInf);
    out.machine_lb_source.assign(static_cast<std::size_t>(m), "none");

    ExactCostCache initial_cache;
    for (int h = 0; h < m; ++h)
    {
        const auto &jobs_h = jobs_by_machine[static_cast<std::size_t>(h)];
        const double c = exact_machine_cost_cached(
            jobs_h,
            prices,
            epsilon,
            rates[static_cast<std::size_t>(h)],
            per_machine_dp_limit_sec,
            initial_cache);
        if (!(c < kInf * 0.5))
        {
            out.feasible = false;
            return out;
        }
        out.machine_exact_cost[static_cast<std::size_t>(h)] = c;
    }
    out.exact_dp_calls_initial = initial_cache.misses;

    if (use_relocate_cleanup)
    {
        LocalSearchConfig cfg;
        cfg.enable_relocate = true;
        cfg.enable_swap = false;

        ExactCostCache ls_cache;
        LocalSearchStats ls_stats;
        run_local_search(
            jobs_by_machine,
            loads,
            out.machine_exact_cost,
            rates,
            prices,
            epsilon,
            per_machine_dp_limit_sec,
            ls_max_rounds,
            ls_max_moves_per_round,
            ls_time_cap_sec,
            cfg,
            ls_cache,
            ls_stats);

        out.accepted_relocate_moves = ls_stats.accepted_relocate;
        out.accepted_swap_moves = ls_stats.accepted_swap;
        out.accepted_moves = ls_stats.accepted_relocate + ls_stats.accepted_swap;
        out.evaluated_relocate_moves = ls_stats.evaluated_relocate;
        out.evaluated_swap_moves = ls_stats.evaluated_swap;
        out.exact_dp_calls_local_search_only = ls_cache.misses;
        out.exact_dp_evals_post_repair_local_search = ls_cache.misses;
        if (ls_stats.accepted_relocate > ls_stats.accepted_swap)
            out.dominant_improvement_move = "relocate";
        else if (ls_stats.accepted_swap > ls_stats.accepted_relocate)
            out.dominant_improvement_move = "swap";
        else if (out.accepted_moves > 0)
            out.dominant_improvement_move = "mixed";
        else
            out.dominant_improvement_move = "none";
    }
    else
    {
        out.accepted_moves = 0;
        out.accepted_relocate_moves = 0;
        out.accepted_swap_moves = 0;
        out.evaluated_relocate_moves = 0;
        out.evaluated_swap_moves = 0;
        out.exact_dp_calls_local_search_only = 0;
        out.exact_dp_evals_post_repair_local_search = 0;
        out.dominant_improvement_move = "none";
    }

    out.tec_total = 0.0;
    out.assignment_conditioned_lb = 0.0;
    for (int h = 0; h < m; ++h)
    {
        const auto &jobs_h = jobs_by_machine[static_cast<std::size_t>(h)];
        out.machine_job_counts[static_cast<std::size_t>(h)] = static_cast<int>(jobs_h.size());

        const double machine_cost = out.machine_exact_cost[static_cast<std::size_t>(h)];
        double machine_lb = relaxed_machine_lb(jobs_h, prices, epsilon, rates[static_cast<std::size_t>(h)]);
        std::string lb_source = "relaxed_dp_feas_or_semigroup";
        if (!(machine_lb < kInf * 0.5))
        {
            machine_lb = fallback_slot_lb(jobs_h, prices, epsilon, rates[static_cast<std::size_t>(h)]);
            lb_source = "fallback_slot_lb";
        }
        if (machine_lb > machine_cost + 1e-6)
        {
            machine_lb = fallback_slot_lb(jobs_h, prices, epsilon, rates[static_cast<std::size_t>(h)]);
            lb_source = "fallback_slot_lb_due_to_lb_violation";
        }
        out.machine_relaxed_lb[static_cast<std::size_t>(h)] = machine_lb;
        out.machine_lb_source[static_cast<std::size_t>(h)] = lb_source;
        out.tec_total += machine_cost;
        out.assignment_conditioned_lb += machine_lb;
    }

    out.final_machine_loads = loads;
    out.runtime_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    next_assignment = assignment_from_jobs(jobs_by_machine);
    return out;
}

Assignment build_dp_guided_assignment(
    const std::vector<int> &jobs,
    const std::vector<double> &rates,
    const std::vector<double> &prices,
    int epsilon)
{
    const int m = static_cast<int>(rates.size());
    Assignment out;
    out.feasible = false;
    out.machine_job_lengths.assign(static_cast<std::size_t>(m), {});
    out.machine_loads.assign(static_cast<std::size_t>(m), 0);

    std::vector<int> order(static_cast<std::size_t>(jobs.size()), 0);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b)
              {
                  if (jobs[static_cast<std::size_t>(a)] != jobs[static_cast<std::size_t>(b)])
                      return jobs[static_cast<std::size_t>(a)] > jobs[static_cast<std::size_t>(b)];
                  return a < b;
              });

    std::map<std::tuple<int, int, std::string>, double> score_cache;
    auto machine_score = [&](const std::vector<int> &jobs_h, int h) -> double
    {
        std::map<int, int> cnt;
        for (int p : jobs_h)
            cnt[p]++;
        std::ostringstream sig;
        for (const auto &[len, c] : cnt)
            sig << len << ':' << c << '|';
        const auto key = std::make_tuple(h, static_cast<int>(jobs_h.size()), sig.str());
        auto it = score_cache.find(key);
        if (it != score_cache.end())
            return it->second;

        double lb = relaxed_machine_lb(jobs_h, prices, epsilon, rates[static_cast<std::size_t>(h)]);
        if (!(lb < kInf * 0.5))
            lb = fallback_slot_lb(jobs_h, prices, epsilon, rates[static_cast<std::size_t>(h)]);
        score_cache.emplace(key, lb);
        return lb;
    };

    for (int jid : order)
    {
        const int p = jobs[static_cast<std::size_t>(jid)];
        int best_h = -1;
        double best_score = kInf;
        double best_delta = kInf;

        for (int h = 0; h < m; ++h)
        {
            const int load = out.machine_loads[static_cast<std::size_t>(h)];
            if (load + p > epsilon)
                continue;

            auto trial_jobs = out.machine_job_lengths[static_cast<std::size_t>(h)];
            trial_jobs.push_back(p);
            const double score = machine_score(trial_jobs, h);

            double delta = 0.0;
            for (int t = load; t < load + p; ++t)
                delta += rates[static_cast<std::size_t>(h)] * prices[static_cast<std::size_t>(t)];

            if (score + 1e-9 < best_score)
            {
                best_score = score;
                best_delta = delta;
                best_h = h;
            }
            else if (std::fabs(score - best_score) <= 1e-9)
            {
                if (delta + 1e-12 < best_delta)
                {
                    best_delta = delta;
                    best_h = h;
                }
                else if (std::fabs(delta - best_delta) <= 1e-12)
                {
                    if (best_h < 0 || out.machine_loads[static_cast<std::size_t>(h)] < out.machine_loads[static_cast<std::size_t>(best_h)] ||
                        (out.machine_loads[static_cast<std::size_t>(h)] == out.machine_loads[static_cast<std::size_t>(best_h)] && h < best_h))
                    {
                        best_h = h;
                    }
                }
            }
        }

        if (best_h < 0)
            return out;

        out.machine_job_lengths[static_cast<std::size_t>(best_h)].push_back(p);
        out.machine_loads[static_cast<std::size_t>(best_h)] += p;
    }

    out.feasible = true;
    return out;
}

VariantResult evaluate_variant(
    const Assignment &assignment,
    const std::vector<double> &rates,
    const std::vector<double> &prices,
    int epsilon,
    const std::string &variant,
    double per_machine_dp_limit_sec,
    int ls_max_rounds,
    std::int64_t ls_max_moves_per_round,
    double ls_time_cap_sec,
    const StageL1LogContext *log_ctx = nullptr)
{
    VariantResult out;
    out.variant = variant;
    out.feasible = assignment.feasible;
    if (!assignment.feasible)
        return out;

    const auto t0 = std::chrono::steady_clock::now();
    const int m = static_cast<int>(rates.size());

    out.machine_job_counts.assign(static_cast<std::size_t>(m), 0);
    out.machine_exact_cost.assign(static_cast<std::size_t>(m), kInf);
    out.machine_relaxed_lb.assign(static_cast<std::size_t>(m), kInf);
    out.machine_lb_source.assign(static_cast<std::size_t>(m), "none");

    out.tec_total = 0.0;
    out.assignment_conditioned_lb = 0.0;

    std::vector<std::vector<int>> jobs_by_machine = assignment.machine_job_lengths;
    std::vector<int> loads = assignment.machine_loads;
    ExactCostCache ls_cache;
    std::int64_t dp_calls_before_local = 0;

    if (variant == "greedy_dp_local_search" ||
        variant == "greedy_dp_local_search_relocate_only" ||
        variant == "greedy_dp_local_search_relocate_multistart" ||
        variant == "greedy_dp_local_search_screened_swap" ||
        variant == "greedy_dp_local_search_priority_machines" ||
        variant == "w4_c1_relocate_only" ||
        variant == "w4_c1_op6_combined" ||
        variant == "vnd_exact_dp" ||
        variant == "vnd_exact_dp_insert_rank_v1" ||
        variant == "vnd_exact_dp_insert_rank_diverse" ||
        variant == "vnd_exact_dp_insert_rank_diverse_trimmed" ||
        variant == "vnd_exact_dp_insert_rank_diverse_budgeted" ||
         variant == "vnd_exact_dp_insert_rank_dense_labeling" ||
         variant == "phaseS_llm_exception_lane" ||
         variant == "phaseS_random_exception_lane" ||
         variant == "phaseS_refined1_stratified" ||
         variant == "phaseS_refined2_anticore" ||
         variant == "phaseS_refined3_coverage" ||
         variant == "phaseV_score_escape_sampler" ||
         variant == "phaseX_policy_json" ||
         variant == "phaseY_trace_probe" ||
         variant == "phaseY_execute_proposal" ||
         variant == "phaseY_random_proposal" ||
         variant == "phaseI_noscreen_diagnostic")
    {
        LocalSearchConfig cfg;
        if (variant == "greedy_dp_local_search_relocate_only" || variant == "w4_c1_relocate_only")
        {
            cfg.enable_relocate = true;
            cfg.enable_swap = false;
        }
        else if (variant == "greedy_dp_local_search_screened_swap")
        {
            cfg.enable_relocate = true;
            cfg.enable_swap = true;
            cfg.screened_swap = true;
            cfg.screened_swap_length_tolerance = 1;
        }
        else if (variant == "greedy_dp_local_search_priority_machines")
        {
            cfg.enable_relocate = true;
            cfg.enable_swap = true;
            cfg.priority_machines = true;
            cfg.priority_top_k = std::max(2, m / 3);
        }

        for (int h = 0; h < m; ++h)
        {
            const auto &jobs_h = jobs_by_machine[static_cast<std::size_t>(h)];
            const double c = exact_machine_cost_cached(
                jobs_h,
                prices,
                epsilon,
                rates[static_cast<std::size_t>(h)],
                per_machine_dp_limit_sec,
                ls_cache);
            if (!(c < kInf * 0.5))
            {
                out.feasible = false;
                return out;
            }
            out.machine_exact_cost[static_cast<std::size_t>(h)] = c;
        }

        dp_calls_before_local = ls_cache.misses;

        if (variant == "vnd_exact_dp")
        {
            VndStats vnd_stats;
            run_vnd_exact_dp(
                jobs_by_machine,
                loads,
                out.machine_exact_cost,
                rates,
                prices,
                epsilon,
                per_machine_dp_limit_sec,
                ls_max_rounds,
                ls_max_moves_per_round,
                ls_time_cap_sec,
                ls_cache,
                vnd_stats);

            out.accepted_swap_intra_moves = vnd_stats.accepted_swap_intra;
            out.accepted_swap_inter_moves = vnd_stats.accepted_swap_inter;
            out.accepted_insert_inter_moves = vnd_stats.accepted_insert_inter;
            out.evaluated_swap_intra_moves = vnd_stats.evaluated_swap_intra;
            out.evaluated_swap_inter_moves = vnd_stats.evaluated_swap_inter;
            out.evaluated_insert_inter_moves = vnd_stats.evaluated_insert_inter;
            out.accepted_swap_moves = vnd_stats.accepted_swap_intra + vnd_stats.accepted_swap_inter;
            out.accepted_relocate_moves = vnd_stats.accepted_insert_inter;
            out.accepted_moves = out.accepted_swap_moves + out.accepted_relocate_moves;
            out.evaluated_swap_moves = vnd_stats.evaluated_swap_intra + vnd_stats.evaluated_swap_inter;
            out.evaluated_relocate_moves = vnd_stats.evaluated_insert_inter;
            out.stop_reason = vnd_stats.stop_reason;

            if (out.accepted_insert_inter_moves >= out.accepted_swap_inter_moves &&
                out.accepted_insert_inter_moves >= out.accepted_swap_intra_moves &&
                out.accepted_insert_inter_moves > 0)
                out.dominant_improvement_move = "insert_inter";
            else if (out.accepted_swap_inter_moves >= out.accepted_swap_intra_moves && out.accepted_swap_inter_moves > 0)
                out.dominant_improvement_move = "swap_inter";
            else if (out.accepted_swap_intra_moves > 0)
                out.dominant_improvement_move = "swap_intra";
            else
                out.dominant_improvement_move = "none";
        }
        else if (variant == "vnd_exact_dp_insert_rank_v1" ||
                 variant == "vnd_exact_dp_insert_rank_diverse" ||
                 variant == "vnd_exact_dp_insert_rank_diverse_trimmed" ||
                 variant == "vnd_exact_dp_insert_rank_diverse_budgeted" ||
                 variant == "vnd_exact_dp_insert_rank_dense_labeling" ||
                 variant == "phaseS_llm_exception_lane" ||
                 variant == "phaseS_random_exception_lane" ||
                 variant == "phaseS_refined1_stratified" ||
                 variant == "phaseS_refined2_anticore" ||
                 variant == "phaseS_refined3_coverage" ||
                 variant == "phaseV_score_escape_sampler" ||
                 variant == "phaseX_policy_json" ||
                 variant == "phaseY_trace_probe" ||
         variant == "phaseY_execute_proposal" ||
         variant == "phaseY_random_proposal" ||
                 variant == "phaseI_noscreen_diagnostic")
        {
             VndStats insert_stats;
            run_insert_inter_screened_redesign(
                jobs_by_machine,
                loads,
                out.machine_exact_cost,
                rates,
                prices,
                epsilon,
                per_machine_dp_limit_sec,
                ls_max_rounds,
                ls_max_moves_per_round,
                ls_time_cap_sec,
                (variant == "vnd_exact_dp_insert_rank_v1") ? InsertScreenMode::DualPressureGlobal :
                (variant == "vnd_exact_dp_insert_rank_diverse") ? InsertScreenMode::DiverseTwoStage :
                (variant == "vnd_exact_dp_insert_rank_diverse_trimmed") ? InsertScreenMode::DiverseTrimmed :
                (variant == "vnd_exact_dp_insert_rank_diverse_budgeted") ? InsertScreenMode::DiverseBudgeted :
                (variant == "phaseS_llm_exception_lane") ? InsertScreenMode::ExceptionLaneLLM :
                (variant == "phaseS_random_exception_lane") ? InsertScreenMode::ExceptionLaneRandom :
                (variant == "phaseS_refined1_stratified") ? InsertScreenMode::ExceptionLaneRefined1 :
                (variant == "phaseS_refined2_anticore") ? InsertScreenMode::ExceptionLaneRefined2 :
                (variant == "phaseS_refined3_coverage") ? InsertScreenMode::ExceptionLaneRefined3 :
                (variant == "phaseV_score_escape_sampler") ? InsertScreenMode::ScoreEscapeSampler :
                (variant == "phaseX_policy_json") ? InsertScreenMode::PhaseXPolicyJson :
                (variant == "phaseY_trace_probe") ? InsertScreenMode::PhaseYTraceProbe :
                (variant == "phaseY_execute_proposal") ? InsertScreenMode::PhaseYExecuteProposal :
                (variant == "phaseY_random_proposal") ? InsertScreenMode::PhaseYRandomProposal :
                InsertScreenMode::DenseLabeling,
                ls_cache,
                insert_stats,
                log_ctx);

            out.accepted_swap_intra_moves = 0;
            out.accepted_swap_inter_moves = 0;
            out.accepted_insert_inter_moves = insert_stats.accepted_insert_inter;
            out.evaluated_swap_intra_moves = 0;
            out.evaluated_swap_inter_moves = 0;
            out.evaluated_insert_inter_moves = insert_stats.evaluated_insert_inter;
            out.accepted_swap_moves = 0;
            out.accepted_relocate_moves = insert_stats.accepted_insert_inter;
            out.accepted_moves = out.accepted_relocate_moves;
            out.evaluated_swap_moves = 0;
            out.evaluated_relocate_moves = insert_stats.evaluated_insert_inter;
            out.stop_reason = insert_stats.stop_reason;
            out.dominant_improvement_move = (out.accepted_insert_inter_moves > 0 ? "insert_inter" : "none");
            out.exception_candidates_considered = insert_stats.exception_candidates_considered;
            out.exception_candidates_evaluated = insert_stats.exception_candidates_evaluated;
            out.exception_budget_used = insert_stats.exception_budget_used;
            out.exception_improvement_count = insert_stats.exception_improvement_count;
            out.exception_best_delta = insert_stats.exception_best_delta;
            out.outside_pool_distinct_src = insert_stats.outside_pool_distinct_src;
            out.outside_pool_distinct_tgt = insert_stats.outside_pool_distinct_tgt;
            out.outside_pool_max_src_share = insert_stats.outside_pool_max_src_share;
            out.outside_pool_max_tgt_share = insert_stats.outside_pool_max_tgt_share;
            out.selected_distinct_src = insert_stats.selected_distinct_src;
            out.selected_distinct_tgt = insert_stats.selected_distinct_tgt;
            out.selected_max_src_share = insert_stats.selected_max_src_share;
            out.selected_max_tgt_share = insert_stats.selected_max_tgt_share;
            out.exception_hit_rate = insert_stats.exception_hit_rate;
            out.final_machine_load_pressure = insert_stats.final_machine_load_pressure;
            out.avg_machine_load_pressure = insert_stats.avg_machine_load_pressure;
            out.phaseV_score_escape_candidates_considered = insert_stats.phaseV_score_escape_candidates_considered;
            out.phaseV_score_escape_candidates_evaluated = insert_stats.phaseV_score_escape_candidates_evaluated;
            out.phaseV_score_escape_improvement_count = insert_stats.phaseV_score_escape_improvement_count;
            out.phaseV_score_escape_best_delta = insert_stats.phaseV_score_escape_best_delta;
            out.phaseV_score_escape_escape_rounds = insert_stats.phaseV_score_escape_escape_rounds;
            out.phaseV_score_escape_normal_rounds = insert_stats.phaseV_score_escape_normal_rounds;
            out.phaseV_score_escape_distinct_pairs = insert_stats.phaseV_score_escape_distinct_pairs;
            out.phaseV_score_escape_max_cheap_lb = insert_stats.phaseV_score_escape_max_cheap_lb;
            out.phaseX_candidates_considered = insert_stats.phaseX_candidates_considered;
            out.phaseX_candidates_evaluated = insert_stats.phaseX_candidates_evaluated;
            out.phaseX_improvement_count = insert_stats.phaseX_improvement_count;
            out.phaseX_best_delta = insert_stats.phaseX_best_delta;
            out.phaseX_normal_rounds = insert_stats.phaseX_normal_rounds;
            out.phaseX_escape_rounds = insert_stats.phaseX_escape_rounds;
            out.phaseX_policy_name = insert_stats.phaseX_policy_name;
            out.phaseY_proposal_name = insert_stats.phaseY_proposal_name;
            out.phaseY_candidates_generated = insert_stats.phaseY_candidates_generated;
            out.phaseY_candidates_selected = insert_stats.phaseY_candidates_selected;
            out.phaseY_candidates_evaluated = insert_stats.phaseY_candidates_evaluated;
            out.phaseY_improvements = insert_stats.phaseY_improvements;
            out.phaseY_best_delta = insert_stats.phaseY_best_delta;
            out.phaseY_accepted_delta = insert_stats.phaseY_accepted_delta;
            out.phaseY_fallback_used = insert_stats.phaseY_fallback_used;
            out.phaseY_invalid_ids_dropped = insert_stats.phaseY_invalid_ids_dropped;
            out.phaseY_sources_used = insert_stats.phaseY_sources_used;
            out.phaseY_targets_used = insert_stats.phaseY_targets_used;
        }
        else if (variant == "phaseI_noscreen_diagnostic")
        {
            NoScreenDiagStats diag_stats;
            run_noscreen_1move_diagnostic(
                jobs_by_machine,
                loads,
                out.machine_exact_cost,
                rates,
                prices,
                epsilon,
                per_machine_dp_limit_sec,
                ls_max_moves_per_round,
                ls_time_cap_sec,
                ls_cache,
                diag_stats);

            out.accepted_swap_intra_moves = 0;
            out.accepted_swap_inter_moves = diag_stats.accepted_swap_inter;
            out.accepted_insert_inter_moves = diag_stats.accepted_insert_inter;
            out.evaluated_swap_intra_moves = 0;
            out.evaluated_swap_inter_moves = diag_stats.evaluated_swap_inter;
            out.evaluated_insert_inter_moves = diag_stats.evaluated_insert_inter;
            out.accepted_swap_moves = diag_stats.accepted_swap_inter;
            out.accepted_relocate_moves = diag_stats.accepted_insert_inter;
            out.accepted_moves = out.accepted_swap_moves + out.accepted_relocate_moves;
            out.evaluated_swap_moves = diag_stats.evaluated_swap_inter;
            out.evaluated_relocate_moves = diag_stats.evaluated_insert_inter;
            out.stop_reason = diag_stats.stop_reason;
            out.diagnostic_start_tec = diag_stats.start_tec;
            out.diagnostic_best_tec = diag_stats.best_tec;
            out.diagnostic_improving_move_found = diag_stats.improving_move_found ? 1 : 0;
            out.diagnostic_exact_evaluated_moves = diag_stats.evaluated_insert_inter + diag_stats.evaluated_swap_inter;
            out.dominant_improvement_move = diag_stats.best_move;
            if (out.accepted_moves <= 0)
                out.dominant_improvement_move = "none";
        }
        else if (variant == "w4_c1_op6_combined")
        {
            // W4 Operator 6: Combined two-step repair (200 DP budget)
            // Step 0: initial DP evaluation of c1 assignment
            for (int h = 0; h < m; ++h) {
                const auto &jobs_h = jobs_by_machine[static_cast<std::size_t>(h)];
                const double c = exact_machine_cost_cached(
                    jobs_h, prices, epsilon, rates[static_cast<std::size_t>(h)],
                    per_machine_dp_limit_sec, ls_cache);
                if (!(c < kInf * 0.5)) {
                    out.feasible = false;
                    return out;
                }
                out.machine_exact_cost[static_cast<std::size_t>(h)] = c;
                out.tec_total += c;
            }
            dp_calls_before_local = ls_cache.misses;

            // Step 1: relieve worst overload (most loaded → underloaded)
            // Step 2: activate worst underuse (overloaded → least loaded)
            long long total_p = 0;
            for (int h = 0; h < m; ++h) total_p += loads[h];
            double L_avg = (double)total_p / m;
            int budget = 200;
            bool improved_step1 = false;

            // Step 1: move job from most-overloaded to underloaded
            int src = 0;
            for (int h = 1; h < m; ++h)
                if (loads[h] > loads[src]) src = h;
            for (int job_idx = (int)jobs_by_machine[src].size() - 1; job_idx >= 0 && budget > 0; --job_idx) {
                int p = jobs_by_machine[src][job_idx];
                for (int dst = 0; dst < m && budget > 0; ++dst) {
                    if (dst == src) continue;
                    if (loads[dst] + p > epsilon) continue;
                    if (loads[dst] >= L_avg) continue;
                    // Try move
                    std::vector<std::vector<int>> new_jbm = jobs_by_machine;
                    std::vector<int> new_loads = loads;
                    new_jbm[src].erase(new_jbm[src].begin() + job_idx);
                    new_jbm[dst].push_back(p);
                    new_loads[src] -= p;
                    new_loads[dst] += p;
                    double new_tec = 0.0;
                    for (int h = 0; h < m; ++h) {
                        double c = exact_machine_cost_cached(
                            new_jbm[h], prices, epsilon, rates[h],
                            per_machine_dp_limit_sec, ls_cache);
                        if (c >= kInf * 0.5) { new_tec = kInf; break; }
                        new_tec += c;
                    }
                    budget--;
                    if (new_tec < out.tec_total) {
                        out.tec_total = new_tec;
                        jobs_by_machine = new_jbm;
                        loads = new_loads;
                        improved_step1 = true;
                        out.accepted_relocate_moves++;
                        out.accepted_moves++;
                        goto step2;
                    }
                }
            }
            step2:
            // Step 2: fill most-underloaded machine from overloaded
            {
                int dst = 0;
                for (int h = 1; h < m; ++h)
                    if (loads[h] < loads[dst] || (loads[h] == loads[dst] && h < dst)) dst = h;
                for (int src2 = 0; src2 < m && budget > 0; ++src2) {
                    if (src2 == dst) continue;
                    if (loads[src2] <= L_avg) continue;
                    for (int job_idx = (int)jobs_by_machine[src2].size() - 1; job_idx >= 0 && budget > 0; --job_idx) {
                        int p = jobs_by_machine[src2][job_idx];
                        if (loads[dst] + p > epsilon) continue;
                        std::vector<std::vector<int>> new_jbm = jobs_by_machine;
                        std::vector<int> new_loads = loads;
                        new_jbm[src2].erase(new_jbm[src2].begin() + job_idx);
                        new_jbm[dst].push_back(p);
                        new_loads[src2] -= p;
                        new_loads[dst] += p;
                        double new_tec = 0.0;
                        for (int h = 0; h < m; ++h) {
                            double c = exact_machine_cost_cached(
                                new_jbm[h], prices, epsilon, rates[h],
                                per_machine_dp_limit_sec, ls_cache);
                            if (c >= kInf * 0.5) { new_tec = kInf; break; }
                            new_tec += c;
                        }
                        budget--;
                        if (new_tec < out.tec_total) {
                            out.tec_total = new_tec;
                            jobs_by_machine = new_jbm;
                            loads = new_loads;
                            out.accepted_relocate_moves++;
                            out.accepted_moves++;
                            break;
                        }
                    }
                }
            }
            out.evaluated_relocate_moves = 200 - budget;
            out.stop_reason = "budget_exhausted";
            out.dominant_improvement_move = (out.accepted_moves > 0) ? "relocate" : "none";
        }
        else
        {
            LocalSearchStats ls_stats;
            run_local_search(
                jobs_by_machine,
                loads,
                out.machine_exact_cost,
                rates,
                prices,
                epsilon,
                per_machine_dp_limit_sec,
                ls_max_rounds,
                ls_max_moves_per_round,
                ls_time_cap_sec,
                cfg,
                ls_cache,
                ls_stats);

            out.accepted_relocate_moves = ls_stats.accepted_relocate;
            out.accepted_swap_moves = ls_stats.accepted_swap;
            out.accepted_moves = ls_stats.accepted_relocate + ls_stats.accepted_swap;
            out.evaluated_relocate_moves = ls_stats.evaluated_relocate;
            out.evaluated_swap_moves = ls_stats.evaluated_swap;
            out.stop_reason = "no_improving_move";
            if (ls_stats.accepted_relocate > ls_stats.accepted_swap)
                out.dominant_improvement_move = "relocate";
            else if (ls_stats.accepted_swap > ls_stats.accepted_relocate)
                out.dominant_improvement_move = "swap";
            else if (out.accepted_moves > 0)
                out.dominant_improvement_move = "mixed";
            else
                out.dominant_improvement_move = "none";
        }

        out.exact_dp_calls_initial = dp_calls_before_local;
        out.exact_dp_calls_local_search_only = ls_cache.misses - dp_calls_before_local;
        out.exact_dp_cache_hits = ls_cache.hits;
        out.exact_dp_cache_misses = ls_cache.misses;
    }

    for (int h = 0; h < m; ++h)
    {
        const auto &jobs_h = jobs_by_machine[static_cast<std::size_t>(h)];
        out.machine_job_counts[static_cast<std::size_t>(h)] = static_cast<int>(jobs_h.size());

        double machine_cost = kInf;
        if (variant == "greedy_esr")
        {
            machine_cost = sequence_preserving_esr_cost(jobs_h, prices, epsilon, rates[static_cast<std::size_t>(h)]);
        }
        else if (variant == "greedy_dp")
        {
            machine_cost = dp_exact_machine_cost(jobs_h, prices, epsilon, rates[static_cast<std::size_t>(h)], per_machine_dp_limit_sec);
        }
        else if (variant == "greedy_dp_local_search" ||
                 variant == "greedy_dp_local_search_relocate_only" ||
                 variant == "greedy_dp_local_search_relocate_multistart" ||
                 variant == "greedy_dp_local_search_screened_swap" ||
                 variant == "greedy_dp_local_search_priority_machines" ||
                 variant == "vnd_exact_dp" ||
                 variant == "vnd_exact_dp_insert_rank_v1" ||
                 variant == "vnd_exact_dp_insert_rank_diverse" ||
                 variant == "vnd_exact_dp_insert_rank_diverse_trimmed" ||
                 variant == "vnd_exact_dp_insert_rank_diverse_budgeted" ||
                 variant == "vnd_exact_dp_insert_rank_dense_labeling" ||
                 variant == "phaseS_llm_exception_lane" ||
                 variant == "phaseS_random_exception_lane" ||
                 variant == "phaseS_refined1_stratified" ||
                 variant == "phaseS_refined2_anticore" ||
                 variant == "phaseS_refined3_coverage" ||
                 variant == "phaseV_score_escape_sampler" ||
                 variant == "phaseX_policy_json" ||
                 variant == "phaseY_trace_probe" ||
         variant == "phaseY_execute_proposal" ||
         variant == "phaseY_random_proposal" ||
                 variant == "phaseI_noscreen_diagnostic")
        {
            machine_cost = out.machine_exact_cost[static_cast<std::size_t>(h)];
        }
        else
        {
            out.feasible = false;
            return out;
        }

        double machine_lb = relaxed_machine_lb(jobs_h, prices, epsilon, rates[static_cast<std::size_t>(h)]);
        std::string lb_source = "relaxed_dp_feas_or_semigroup";
        if (!(machine_lb < kInf * 0.5))
        {
            machine_lb = fallback_slot_lb(jobs_h, prices, epsilon, rates[static_cast<std::size_t>(h)]);
            lb_source = "fallback_slot_lb";
        }

        if (machine_lb > machine_cost + 1e-6)
        {
            machine_lb = fallback_slot_lb(jobs_h, prices, epsilon, rates[static_cast<std::size_t>(h)]);
            lb_source = "fallback_slot_lb_due_to_lb_violation";
        }

        out.machine_exact_cost[static_cast<std::size_t>(h)] = machine_cost;
        out.machine_relaxed_lb[static_cast<std::size_t>(h)] = machine_lb;
        out.machine_lb_source[static_cast<std::size_t>(h)] = lb_source;

        if (!(machine_cost < kInf * 0.5) || !(machine_lb < kInf * 0.5))
        {
            out.feasible = false;
            return out;
        }

        out.tec_total += machine_cost;
        out.assignment_conditioned_lb += machine_lb;
    }

    out.final_machine_loads = loads;
    out.runtime_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

void print_csv_header()
{
    std::cout << "instance_id,epsilon,epsilon_prev,variant,runtime_sec,tec_total,assignment_conditioned_lb,num_machines,"
                 "machine_job_counts,machine_exact_cost,machine_relaxed_lb,machine_lb_source,final_machine_loads,"
                 "accepted_moves,accepted_relocate_moves,accepted_swap_moves,evaluated_relocate_moves,evaluated_swap_moves,displaced_jobs,reinsertion_candidates_scored,"
                 "exact_dp_evals_repair,exact_dp_evals_post_repair_local_search,relocate_cleanup_used,dominant_improvement_move,exact_dp_calls_initial,exact_dp_calls_local_search_only,"
                 "accepted_swap_intra_moves,accepted_swap_inter_moves,accepted_insert_inter_moves,evaluated_swap_intra_moves,evaluated_swap_inter_moves,evaluated_insert_inter_moves,"
                 "exact_dp_cache_hits,exact_dp_cache_misses,stop_reason,diagnostic_start_tec,diagnostic_best_tec,diagnostic_improving_move_found,diagnostic_exact_evaluated_moves,"
                 "exception_candidates_considered,exception_candidates_evaluated,exception_budget_used,exception_improvement_count,exception_best_delta,"
                 "outside_pool_distinct_src,outside_pool_distinct_tgt,outside_pool_max_src_share,outside_pool_max_tgt_share,"
                 "selected_distinct_src,selected_distinct_tgt,selected_max_src_share,selected_max_tgt_share,"
                 "exception_hit_rate,final_machine_load_pressure,avg_machine_load_pressure,"
                 "phaseV_score_escape_candidates_considered,phaseV_score_escape_candidates_evaluated,phaseV_score_escape_improvement_count,phaseV_score_escape_best_delta,"
                 "phaseV_score_escape_escape_rounds,phaseV_score_escape_normal_rounds,phaseV_score_escape_distinct_pairs,phaseV_score_escape_max_cheap_lb,"
                  "phaseX_policy_name,phaseX_candidates_considered,phaseX_candidates_evaluated,phaseX_improvement_count,phaseX_best_delta,phaseX_normal_rounds,phaseX_escape_rounds,"
                  "phaseY_proposal_name,phaseY_candidates_generated,phaseY_candidates_selected,phaseY_candidates_evaluated,phaseY_improvements,phaseY_best_delta,phaseY_accepted_delta,phaseY_fallback_used,phaseY_invalid_ids_dropped,phaseY_sources_used,phaseY_targets_used\n";
}

void print_csv_row(
    int instance_id,
    int epsilon,
    const VariantResult &res,
    const std::vector<int> &loads)
{
    std::cout << "paper_" << instance_id << ","
              << epsilon << ","
              << res.epsilon_prev << ","
              << res.variant << ","
              << std::fixed << std::setprecision(6) << res.runtime_sec << ","
              << (res.feasible ? res.tec_total : -1.0) << ","
              << (res.feasible ? res.assignment_conditioned_lb : -1.0) << ","
              << loads.size() << ","
              << join_ints(res.machine_job_counts) << ","
              << join_doubles(res.machine_exact_cost) << ","
              << join_doubles(res.machine_relaxed_lb) << ","
              << join_strings(res.machine_lb_source) << ","
              << join_ints(res.final_machine_loads.empty() ? loads : res.final_machine_loads) << ","
              << res.accepted_moves << ","
              << res.accepted_relocate_moves << ","
              << res.accepted_swap_moves << ","
              << res.evaluated_relocate_moves << ","
              << res.evaluated_swap_moves << ","
              << res.displaced_jobs << ","
              << res.reinsertion_candidates_scored << ","
              << res.exact_dp_evals_repair << ","
              << res.exact_dp_evals_post_repair_local_search << ","
              << res.relocate_cleanup_used << ","
              << res.dominant_improvement_move << ","
              << res.exact_dp_calls_initial << ","
              << res.exact_dp_calls_local_search_only << ","
              << res.accepted_swap_intra_moves << ","
              << res.accepted_swap_inter_moves << ","
              << res.accepted_insert_inter_moves << ","
              << res.evaluated_swap_intra_moves << ","
              << res.evaluated_swap_inter_moves << ","
              << res.evaluated_insert_inter_moves << ","
              << res.exact_dp_cache_hits << ","
              << res.exact_dp_cache_misses << ","
              << res.stop_reason << ","
              << res.diagnostic_start_tec << ","
              << res.diagnostic_best_tec << ","
              << res.diagnostic_improving_move_found << ","
              << res.diagnostic_exact_evaluated_moves << ","
              << res.exception_candidates_considered << ","
              << res.exception_candidates_evaluated << ","
              << res.exception_budget_used << ","
              << res.exception_improvement_count << ","
              << res.exception_best_delta << ","
              << res.outside_pool_distinct_src << ","
              << res.outside_pool_distinct_tgt << ","
              << res.outside_pool_max_src_share << ","
              << res.outside_pool_max_tgt_share << ","
              << res.selected_distinct_src << ","
              << res.selected_distinct_tgt << ","
              << res.selected_max_src_share << ","
              << res.selected_max_tgt_share << ","
              << res.exception_hit_rate << ","
              << res.final_machine_load_pressure << ","
              << res.avg_machine_load_pressure << ","
              << res.phaseV_score_escape_candidates_considered << ","
              << res.phaseV_score_escape_candidates_evaluated << ","
              << res.phaseV_score_escape_improvement_count << ","
              << res.phaseV_score_escape_best_delta << ","
              << res.phaseV_score_escape_escape_rounds << ","
              << res.phaseV_score_escape_normal_rounds << ","
              << res.phaseV_score_escape_distinct_pairs << ","
              << res.phaseV_score_escape_max_cheap_lb << ","
              << res.phaseX_policy_name << ","
              << res.phaseX_candidates_considered << ","
              << res.phaseX_candidates_evaluated << ","
              << res.phaseX_improvement_count << ","
              << res.phaseX_best_delta << ","
              << res.phaseX_normal_rounds << ","
              << res.phaseX_escape_rounds << ","
              << res.phaseY_proposal_name << ","
              << res.phaseY_candidates_generated << ","
              << res.phaseY_candidates_selected << ","
              << res.phaseY_candidates_evaluated << ","
              << res.phaseY_improvements << ","
              << res.phaseY_best_delta << ","
              << res.phaseY_accepted_delta << ","
              << res.phaseY_fallback_used << ","
              << res.phaseY_invalid_ids_dropped << ","
              << res.phaseY_sources_used << ","
              << res.phaseY_targets_used << "\n";
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 5)
    {
        std::cerr << "Usage:\n"
                  << "  parallel_heuristic_compare paper-instance <instance_id> <epsilon> <variant> [data_dir] [per_machine_dp_limit_sec] [ls_time_cap_sec] [ls_max_rounds] [ls_max_moves_per_round]\n"
                  << "  parallel_heuristic_compare paper-history-chain <instance_id> <epsilon_start> <epsilon_end> <variant> [data_dir] [per_machine_dp_limit_sec] [ls_time_cap_sec] [ls_max_rounds] [ls_max_moves_per_round]\n"
                   << "Variants: greedy_esr | greedy_dp | dp_guided_assignment_dp | greedy_dp_local_search | greedy_dp_local_search_relocate_only | greedy_dp_local_search_relocate_multistart | greedy_dp_local_search_screened_swap | greedy_dp_local_search_priority_machines | vnd_exact_dp | vnd_exact_dp_insert_rank_v1 | vnd_exact_dp_insert_rank_diverse | vnd_exact_dp_insert_rank_diverse_trimmed | vnd_exact_dp_insert_rank_diverse_budgeted | vnd_exact_dp_insert_rank_dense_labeling | phaseS_llm_exception_lane | phaseS_random_exception_lane | phaseS_refined1_stratified | phaseS_refined2_anticore | phaseS_refined3_coverage | phaseV_score_escape_sampler | phaseX_policy_json | phaseY_trace_probe | phaseY_execute_proposal | phaseY_random_proposal | phaseI_noscreen_diagnostic | stageL1_dataset_logging | stageL15_dense_labeling | stageO_synthetic_dense_logging | history_repair_dp_ranked | history_repair_dp_ranked_relocate | history_repair_priority_displaced | history_repair_priority_displaced_relocate | all\n";
        return 1;
    }

    const std::string mode = argv[1];
    if (mode != "paper-instance" && mode != "paper-history-chain")
    {
        std::cerr << "Unsupported mode: " << mode << "\n";
        return 1;
    }

    const int instance_id = std::stoi(argv[2]);

    if (const char* env_seed = std::getenv("PHASES_RANDOM_SEED"))
        g_random_exception_seed = std::stoi(env_seed);

    if (const char* env_phaseX = std::getenv("PHASEX_POLICY_PATH"))
        g_phaseX_policy_path = env_phaseX;

    int epsilon = -1;
    int epsilon_end = -1;
    std::string variant;
    std::string data_dir = "temp/paper_exact_repo/instances";
    double per_machine_dp_limit_sec = 30.0;
    double ls_time_cap_sec = 10.0;
    int ls_max_rounds = 5;
    std::int64_t ls_max_moves_per_round = 20000;

    if (mode == "paper-instance")
    {
        epsilon = std::stoi(argv[3]);
        variant = argv[4];
        data_dir = (argc > 5 ? argv[5] : data_dir);
        per_machine_dp_limit_sec = (argc > 6 ? std::stod(argv[6]) : per_machine_dp_limit_sec);
        ls_time_cap_sec = (argc > 7 ? std::stod(argv[7]) : ls_time_cap_sec);
        ls_max_rounds = (argc > 8 ? std::stoi(argv[8]) : ls_max_rounds);
        ls_max_moves_per_round = (argc > 9 ? std::stoll(argv[9]) : ls_max_moves_per_round);
    }
    else
    {
        if (argc < 6)
        {
            std::cerr << "paper-history-chain requires <instance_id> <epsilon_start> <epsilon_end> <variant>\n";
            return 1;
        }
        epsilon = std::stoi(argv[3]);
        epsilon_end = std::stoi(argv[4]);
        variant = argv[5];
        data_dir = (argc > 6 ? argv[6] : data_dir);
        per_machine_dp_limit_sec = (argc > 7 ? std::stod(argv[7]) : per_machine_dp_limit_sec);
        ls_time_cap_sec = (argc > 8 ? std::stod(argv[8]) : ls_time_cap_sec);
        ls_max_rounds = (argc > 9 ? std::stoi(argv[9]) : ls_max_rounds);
        ls_max_moves_per_round = (argc > 10 ? std::stoll(argv[10]) : ls_max_moves_per_round);
    }

    auto inst = load_paper_instance(instance_id, data_dir);
    if (inst.jobs.empty() || inst.prices.empty() || inst.rates.empty())
    {
        std::cerr << "Could not load jobs/prices/rates for paper instance " << instance_id << " from " << data_dir << "\n";
        return 1;
    }
    if (epsilon <= 0 || epsilon > static_cast<int>(inst.prices.size()))
    {
        std::cerr << "epsilon must be in [1,T]\n";
        return 1;
    }

    print_csv_header();

    if (mode == "paper-history-chain")
    {
        const int epsilon_start = epsilon;
        const std::string history_variant = variant;
        if (!is_history_variant(history_variant))
        {
            std::cerr << "Unsupported history variant: " << history_variant << "\n";
            return 1;
        }
        if (epsilon_start < epsilon_end || epsilon_start <= 0 || epsilon_start > static_cast<int>(inst.prices.size()) || epsilon_end <= 0)
        {
            std::cerr << "For paper-history-chain require 1 <= epsilon_end <= epsilon_start <= T\n";
            return 1;
        }

        std::vector<double> prices_start(inst.prices.begin(), inst.prices.begin() + epsilon_start);
        Assignment prev_assignment = build_lpt_greedy_assignment(inst.jobs, inst.rates, prices_start, epsilon_start);
        if (!prev_assignment.feasible)
        {
            std::cerr << "Could not build seed assignment at epsilon_start\n";
            return 1;
        }

        VariantResult seed = evaluate_variant(
            prev_assignment,
            inst.rates,
            prices_start,
            epsilon_start,
            "greedy_dp_local_search_relocate_only",
            per_machine_dp_limit_sec,
            ls_max_rounds,
            ls_max_moves_per_round,
            ls_time_cap_sec);
        seed.variant = "history_seed_greedy_dp_local_search_relocate_only";
        seed.epsilon_prev = -1;
        print_csv_row(instance_id, epsilon_start, seed, prev_assignment.machine_loads);

        for (int e = epsilon_start - 1; e >= epsilon_end; --e)
        {
            std::vector<double> prices_e(inst.prices.begin(), inst.prices.begin() + e);
            Assignment next_assignment;
            VariantResult r = evaluate_history_repair_step(
                prev_assignment,
                inst.rates,
                prices_e,
                e + 1,
                e,
                history_variant,
                per_machine_dp_limit_sec,
                ls_max_rounds,
                ls_max_moves_per_round,
                ls_time_cap_sec,
                next_assignment);
            print_csv_row(instance_id, e, r, prev_assignment.machine_loads);
            if (!r.feasible)
                break;
            prev_assignment = next_assignment;
        }
        return 0;
    }

    std::vector<double> clipped_prices(inst.prices.begin(), inst.prices.begin() + epsilon);

    auto run_variant = [&](const std::string &name) -> VariantResult
    {
        if (name == "stageL15_dense_labeling")
        {
            struct DenseContext
            {
                int context_id = -1;
                int instance_id = -1;
                int epsilon = -1;
                int seeds = 0;
                std::string label;
            };

            const auto t0 = std::chrono::steady_clock::now();
            const int rcl_size = 3;
            const int dense_rounds = std::max(ls_max_rounds, 8);
            const std::int64_t dense_screen_cap = std::max<std::int64_t>(ls_max_moves_per_round, 120000);
            const double dense_time_cap = std::max(ls_time_cap_sec, 25.0);
            const std::string out_dir = "temp/phaseL15_dense_labeling";

            std::vector<DenseContext> contexts = {
                DenseContext{1, 61, 347, 10, "inst61_eps347"},
                DenseContext{2, 61, 346, 8, "inst61_eps346"},
                DenseContext{3, 61, 345, 8, "inst61_eps345"},
                DenseContext{4, 64, 79, 8, "inst64_eps79"}};

            StageL1MoveLogger logger;
            logger.enabled = true;
            logger.source_variant = "vnd_exact_dp_insert_rank_dense_labeling";
            if (!logger.open_with_paths(out_dir, "moves_broad_aggregate.csv", "moves_exact_labeled_aggregate.csv"))
            {
                VariantResult fail;
                fail.variant = name;
                fail.feasible = false;
                return fail;
            }

            std::ofstream seed_summary(out_dir + "/context_seed_summary.csv", std::ios::out | std::ios::trunc);
            if (seed_summary.is_open())
            {
                seed_summary << "context_id,context_label,instance_id,epsilon,seed_id,start_tec,final_tec,broad_rows,exact_rows,positive_rows,accepted_insert_moves,stop_reason\n";
            }

            VariantResult best_anchor;
            best_anchor.variant = name;
            best_anchor.feasible = false;

            for (const auto &ctx : contexts)
            {
                auto ctx_inst = load_paper_instance(ctx.instance_id, data_dir);
                if (ctx_inst.jobs.empty() || ctx_inst.prices.empty() || ctx_inst.rates.empty())
                    continue;
                if (ctx.epsilon <= 0 || ctx.epsilon > static_cast<int>(ctx_inst.prices.size()))
                    continue;

                std::vector<double> ctx_prices(ctx_inst.prices.begin(), ctx_inst.prices.begin() + ctx.epsilon);

                for (int s = 0; s < ctx.seeds; ++s)
                {
                    const std::int64_t broad_before = logger.broad_rows;
                    const std::int64_t exact_before = logger.exact_rows;
                    const std::int64_t pos_before = logger.exact_positive_rows;

                    std::mt19937_64 rng(static_cast<std::uint64_t>(ctx.instance_id) * 5000011ULL +
                                        static_cast<std::uint64_t>(ctx.epsilon) * 9001ULL +
                                        static_cast<std::uint64_t>(s + 701));
                    Assignment assignment = build_lpt_greedy_assignment_randomized(ctx_inst.jobs, ctx_inst.rates, ctx_prices, ctx.epsilon, rng, rcl_size);
                    if (!assignment.feasible)
                        continue;

                    VariantResult start_eval = evaluate_variant(
                        assignment,
                        ctx_inst.rates,
                        ctx_prices,
                        ctx.epsilon,
                        "greedy_dp",
                        per_machine_dp_limit_sec,
                        dense_rounds,
                        dense_screen_cap,
                        dense_time_cap);
                    if (!start_eval.feasible)
                        continue;

                    StageL1LogContext log_ctx;
                    log_ctx.instance_id = ctx.instance_id;
                    log_ctx.seed_id = s;
                    log_ctx.start_tec = start_eval.tec_total;
                    log_ctx.logger = &logger;
                    log_ctx.context_id = ctx.context_id;

                    VariantResult candidate = evaluate_variant(
                        assignment,
                        ctx_inst.rates,
                        ctx_prices,
                        ctx.epsilon,
                        "vnd_exact_dp_insert_rank_dense_labeling",
                        per_machine_dp_limit_sec,
                        dense_rounds,
                        dense_screen_cap,
                        dense_time_cap,
                        &log_ctx);
                    if (!candidate.feasible)
                        continue;

                    const std::int64_t broad_added = logger.broad_rows - broad_before;
                    const std::int64_t exact_added = logger.exact_rows - exact_before;
                    const std::int64_t pos_added = logger.exact_positive_rows - pos_before;

                    if (seed_summary.is_open())
                    {
                        seed_summary << ctx.context_id << ',' << ctx.label << ',' << ctx.instance_id << ',' << ctx.epsilon << ',' << s << ','
                                     << std::fixed << std::setprecision(6) << start_eval.tec_total << ',' << candidate.tec_total << ','
                                     << broad_added << ',' << exact_added << ',' << pos_added << ','
                                     << candidate.accepted_insert_inter_moves << ',' << candidate.stop_reason << '\n';
                    }

                    if (ctx.context_id == 1)
                    {
                        if (!best_anchor.feasible || candidate.tec_total + 1e-9 < best_anchor.tec_total)
                            best_anchor = candidate;
                    }
                }
            }

            if (best_anchor.feasible)
            {
                best_anchor.variant = name;
                best_anchor.runtime_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            }
            else
            {
                best_anchor.variant = name;
                best_anchor.feasible = false;
            }

            print_csv_row(instance_id, epsilon, best_anchor, best_anchor.final_machine_loads);
            return best_anchor;
        }

        if (name == "stageL1_dataset_logging")
        {
            const auto t0 = std::chrono::steady_clock::now();
            const int seeds = 12;
            const int rcl_size = 3;
            const std::string out_dir = "temp/phaseL1_dataset_logging";

            StageL1MoveLogger logger;
            logger.enabled = true;
            logger.source_variant = "vnd_exact_dp_insert_rank_diverse_trimmed";
            if (!logger.open(out_dir))
            {
                VariantResult fail;
                fail.variant = name;
                fail.feasible = false;
                return fail;
            }

            VariantResult best;
            best.variant = name;
            best.feasible = false;

            for (int s = 0; s < seeds; ++s)
            {
                std::mt19937_64 rng(static_cast<std::uint64_t>(instance_id) * 3000017ULL +
                                    static_cast<std::uint64_t>(epsilon) * 7001ULL +
                                    static_cast<std::uint64_t>(s + 101));
                Assignment assignment = build_lpt_greedy_assignment_randomized(inst.jobs, inst.rates, clipped_prices, epsilon, rng, rcl_size);
                if (!assignment.feasible)
                    continue;

                VariantResult start_eval = evaluate_variant(
                    assignment,
                    inst.rates,
                    clipped_prices,
                    epsilon,
                    "greedy_dp",
                    per_machine_dp_limit_sec,
                    ls_max_rounds,
                    ls_max_moves_per_round,
                    ls_time_cap_sec);
                if (!start_eval.feasible)
                    continue;

                StageL1LogContext ctx;
                ctx.instance_id = instance_id;
                ctx.seed_id = s;
                ctx.start_tec = start_eval.tec_total;
                ctx.logger = &logger;

                VariantResult candidate = evaluate_variant(
                    assignment,
                    inst.rates,
                    clipped_prices,
                    epsilon,
                    "vnd_exact_dp_insert_rank_diverse_trimmed",
                    per_machine_dp_limit_sec,
                    ls_max_rounds,
                    ls_max_moves_per_round,
                    ls_time_cap_sec,
                    &ctx);
                if (!candidate.feasible)
                    continue;

                if (!best.feasible || candidate.tec_total + 1e-9 < best.tec_total)
                    best = candidate;
            }

            if (best.feasible)
            {
                best.variant = name;
                best.runtime_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            }

            const std::string meta_path = out_dir + "/dataset_summary_61_347.json";
            std::ofstream meta(meta_path, std::ios::out | std::ios::trunc);
            if (meta.is_open())
            {
                const double pos_rate = (logger.exact_rows > 0)
                                            ? static_cast<double>(logger.exact_positive_rows) / static_cast<double>(logger.exact_rows)
                                            : 0.0;
                meta << "{\n"
                     << "  \"instance_id\": " << instance_id << ",\n"
                     << "  \"epsilon\": " << epsilon << ",\n"
                     << "  \"source_variant\": \"vnd_exact_dp_insert_rank_diverse_trimmed\",\n"
                     << "  \"seeds_used\": " << seeds << ",\n"
                     << "  \"broad_candidate_records\": " << logger.broad_rows << ",\n"
                     << "  \"exact_labeled_records\": " << logger.exact_rows << ",\n"
                     << "  \"exact_positive_improving_records\": " << logger.exact_positive_rows << ",\n"
                     << "  \"exact_positive_rate\": " << std::fixed << std::setprecision(6) << pos_rate << "\n"
                     << "}\n";
            }

            print_csv_row(instance_id, epsilon, best, best.final_machine_loads);
            return best;
        }

        if (name == "stageO_synthetic_dense_logging")
        {
            const auto t0 = std::chrono::steady_clock::now();
            const int seeds = 4;
            const int rcl_size = 3;
            const int dense_rounds = std::max(ls_max_rounds, 2);
            const std::int64_t dense_screen_cap = std::max<std::int64_t>(ls_max_moves_per_round, 12000);
            const double dense_time_cap = std::max(ls_time_cap_sec, 6.0);
            const std::string out_dir = "temp/phaseO_synthetic_dense_labeling";

            StageL1MoveLogger logger;
            logger.enabled = true;
            logger.source_variant = "vnd_exact_dp_insert_rank_dense_labeling";

            const std::string broad_name = "moves_broad_instance_" + std::to_string(instance_id) +
                                           "_eps_" + std::to_string(epsilon) + ".csv";
            const std::string exact_name = "moves_exact_labeled_instance_" + std::to_string(instance_id) +
                                           "_eps_" + std::to_string(epsilon) + ".csv";

            if (!logger.open_with_paths(out_dir, broad_name, exact_name))
            {
                VariantResult fail;
                fail.variant = name;
                fail.feasible = false;
                return fail;
            }

            VariantResult best;
            best.variant = name;
            best.feasible = false;

            for (int s = 0; s < seeds; ++s)
            {
                std::mt19937_64 rng(static_cast<std::uint64_t>(instance_id) * 5000011ULL +
                                    static_cast<std::uint64_t>(epsilon) * 9001ULL +
                                    static_cast<std::uint64_t>(s + 701));
                Assignment assignment = build_lpt_greedy_assignment_randomized(inst.jobs, inst.rates, clipped_prices, epsilon, rng, rcl_size);
                if (!assignment.feasible)
                    continue;

                VariantResult start_eval = evaluate_variant(
                    assignment,
                    inst.rates,
                    clipped_prices,
                    epsilon,
                    "greedy_dp",
                    per_machine_dp_limit_sec,
                    dense_rounds,
                    dense_screen_cap,
                    dense_time_cap);
                if (!start_eval.feasible)
                    continue;

                StageL1LogContext ctx;
                ctx.instance_id = instance_id;
                ctx.seed_id = s;
                ctx.start_tec = start_eval.tec_total;
                ctx.logger = &logger;
                ctx.context_id = -1;

                VariantResult candidate = evaluate_variant(
                    assignment,
                    inst.rates,
                    clipped_prices,
                    epsilon,
                    "vnd_exact_dp_insert_rank_dense_labeling",
                    per_machine_dp_limit_sec,
                    dense_rounds,
                    dense_screen_cap,
                    dense_time_cap,
                    &ctx);
                if (!candidate.feasible)
                    continue;

                if (!best.feasible || candidate.tec_total + 1e-9 < best.tec_total)
                    best = candidate;
            }

            if (best.feasible)
            {
                best.variant = name;
                best.runtime_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            }

            print_csv_row(instance_id, epsilon, best, best.final_machine_loads);
            return best;
        }

        if (name == "vnd_exact_dp")
        {
            const auto t0 = std::chrono::steady_clock::now();
            const int starts = 4;
            const int rcl_size = 3;
            VariantResult best;
            best.variant = name;
            best.feasible = false;

            for (int s = 0; s < starts; ++s)
            {
                std::mt19937_64 rng(static_cast<std::uint64_t>(instance_id) * 2000003ULL +
                                    static_cast<std::uint64_t>(epsilon) * 11003ULL +
                                    static_cast<std::uint64_t>(s + 17));
                Assignment assignment = build_lpt_greedy_assignment_randomized(inst.jobs, inst.rates, clipped_prices, epsilon, rng, rcl_size);
                if (!assignment.feasible)
                    continue;

                VariantResult candidate = evaluate_variant(
                    assignment,
                    inst.rates,
                    clipped_prices,
                    epsilon,
                    "vnd_exact_dp",
                    per_machine_dp_limit_sec,
                    ls_max_rounds,
                    ls_max_moves_per_round,
                    ls_time_cap_sec);
                if (!candidate.feasible)
                    continue;

                if (!best.feasible || candidate.tec_total + 1e-9 < best.tec_total)
                    best = candidate;
            }

            if (!best.feasible)
            {
                Assignment fallback = build_lpt_greedy_assignment(inst.jobs, inst.rates, clipped_prices, epsilon);
                if (fallback.feasible)
                {
                    VariantResult candidate = evaluate_variant(
                        fallback,
                        inst.rates,
                        clipped_prices,
                        epsilon,
                        "vnd_exact_dp",
                        per_machine_dp_limit_sec,
                        ls_max_rounds,
                        ls_max_moves_per_round,
                        ls_time_cap_sec);
                    if (candidate.feasible)
                        best = candidate;
                }
            }

            best.variant = name;
            best.runtime_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            print_csv_row(instance_id, epsilon, best, best.final_machine_loads);
            return best;
        }

        if (name == "vnd_exact_dp_insert_rank_v1" ||
            name == "vnd_exact_dp_insert_rank_diverse" ||
            name == "vnd_exact_dp_insert_rank_diverse_trimmed" ||
            name == "vnd_exact_dp_insert_rank_diverse_budgeted" ||
            name == "phaseS_llm_exception_lane" ||
            name == "phaseS_random_exception_lane" ||
            name == "phaseS_refined1_stratified" ||
            name == "phaseS_refined2_anticore" ||
            name == "phaseS_refined3_coverage" ||
            name == "phaseV_score_escape_sampler" ||
            name == "phaseX_policy_json" ||
            name == "phaseY_trace_probe" ||
            name == "phaseY_execute_proposal" ||
            name == "phaseY_random_proposal")
        {
            const auto t0 = std::chrono::steady_clock::now();
            const bool is_trace_probe = (name == "phaseY_trace_probe");
            const bool is_1start = is_trace_probe || name == "phaseY_execute_proposal" || name == "phaseY_random_proposal";
            const int starts = is_1start ? 1 : 4;
            const int rcl_size = 3;
            VariantResult best;
            best.variant = name;
            best.feasible = false;

            for (int s = 0; s < starts; ++s)
            {
                g_audit_instance_id = instance_id;
                std::mt19937_64 rng(static_cast<std::uint64_t>(instance_id) * 2000003ULL +
                                    static_cast<std::uint64_t>(epsilon) * 11003ULL +
                                    static_cast<std::uint64_t>(s + 17));
                Assignment assignment = build_lpt_greedy_assignment_randomized(inst.jobs, inst.rates, clipped_prices, epsilon, rng, rcl_size);
                if (!assignment.feasible)
                    continue;

                VariantResult candidate = evaluate_variant(
                    assignment,
                    inst.rates,
                    clipped_prices,
                    epsilon,
                    name,
                    per_machine_dp_limit_sec,
                    ls_max_rounds,
                    ls_max_moves_per_round,
                    ls_time_cap_sec);
                if (!candidate.feasible)
                    continue;

                if (!best.feasible || candidate.tec_total + 1e-9 < best.tec_total)
                    best = candidate;
            }

            if (!best.feasible)
            {
                Assignment fallback = build_lpt_greedy_assignment(inst.jobs, inst.rates, clipped_prices, epsilon);
                if (fallback.feasible)
                {
                    VariantResult candidate = evaluate_variant(
                        fallback,
                        inst.rates,
                        clipped_prices,
                        epsilon,
                        name,
                        per_machine_dp_limit_sec,
                        ls_max_rounds,
                        ls_max_moves_per_round,
                        ls_time_cap_sec);
                    if (candidate.feasible)
                        best = candidate;
                }
            }

            best.variant = name;
            best.runtime_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            print_csv_row(instance_id, epsilon, best, best.final_machine_loads);
            return best;
        }

        if (name == "phaseI_noscreen_diagnostic")
        {
            const auto t0 = std::chrono::steady_clock::now();
            const int starts = 4;
            const int rcl_size = 3;
            VariantResult best_start_eval;
            best_start_eval.variant = "greedy_dp";
            best_start_eval.feasible = false;
            Assignment best_assignment;
            best_assignment.feasible = false;

            for (int s = 0; s < starts; ++s)
            {
                std::mt19937_64 rng(static_cast<std::uint64_t>(instance_id) * 2000003ULL +
                                    static_cast<std::uint64_t>(epsilon) * 11003ULL +
                                    static_cast<std::uint64_t>(s + 17));
                Assignment assignment = build_lpt_greedy_assignment_randomized(inst.jobs, inst.rates, clipped_prices, epsilon, rng, rcl_size);
                if (!assignment.feasible)
                    continue;

                VariantResult start_eval = evaluate_variant(
                    assignment,
                    inst.rates,
                    clipped_prices,
                    epsilon,
                    "greedy_dp",
                    per_machine_dp_limit_sec,
                    ls_max_rounds,
                    ls_max_moves_per_round,
                    ls_time_cap_sec);
                if (!start_eval.feasible)
                    continue;

                if (!best_start_eval.feasible || start_eval.tec_total + 1e-9 < best_start_eval.tec_total)
                {
                    best_start_eval = start_eval;
                    best_assignment = assignment;
                }
            }

            if (!best_assignment.feasible)
            {
                Assignment fallback = build_lpt_greedy_assignment(inst.jobs, inst.rates, clipped_prices, epsilon);
                if (fallback.feasible)
                {
                    VariantResult start_eval = evaluate_variant(
                        fallback,
                        inst.rates,
                        clipped_prices,
                        epsilon,
                        "greedy_dp",
                        per_machine_dp_limit_sec,
                        ls_max_rounds,
                        ls_max_moves_per_round,
                        ls_time_cap_sec);
                    if (start_eval.feasible)
                    {
                        best_start_eval = start_eval;
                        best_assignment = fallback;
                    }
                }
            }

            VariantResult diag;
            if (best_assignment.feasible)
            {
                diag = evaluate_variant(
                    best_assignment,
                    inst.rates,
                    clipped_prices,
                    epsilon,
                    "phaseI_noscreen_diagnostic",
                    per_machine_dp_limit_sec,
                    ls_max_rounds,
                    ls_max_moves_per_round,
                    ls_time_cap_sec);
            }
            else
            {
                diag.variant = "phaseI_noscreen_diagnostic";
                diag.feasible = false;
            }

            if (diag.feasible)
            {
                if (!(diag.diagnostic_start_tec >= 0.0))
                    diag.diagnostic_start_tec = best_start_eval.tec_total;
                if (!(diag.diagnostic_best_tec >= 0.0))
                    diag.diagnostic_best_tec = diag.tec_total;
            }

            diag.variant = name;
            diag.runtime_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            print_csv_row(instance_id, epsilon, diag, best_assignment.machine_loads);
            return diag;
        }

        if (name == "greedy_dp_local_search_relocate_multistart")
        {
            const auto t0 = std::chrono::steady_clock::now();
            const int starts = 8;
            const int rcl_size = 3;
            VariantResult best;
            best.variant = name;
            best.feasible = false;

            for (int s = 0; s < starts; ++s)
            {
                std::mt19937_64 rng(static_cast<std::uint64_t>(instance_id) * 1000003ULL +
                                    static_cast<std::uint64_t>(epsilon) * 10007ULL +
                                    static_cast<std::uint64_t>(s + 1));
                Assignment assignment = build_lpt_greedy_assignment_randomized(inst.jobs, inst.rates, clipped_prices, epsilon, rng, rcl_size);
                if (!assignment.feasible)
                    continue;

                VariantResult candidate = evaluate_variant(
                    assignment,
                    inst.rates,
                    clipped_prices,
                    epsilon,
                    "greedy_dp_local_search_relocate_only",
                    per_machine_dp_limit_sec,
                    ls_max_rounds,
                    ls_max_moves_per_round,
                    ls_time_cap_sec);
                if (!candidate.feasible)
                    continue;

                if (!best.feasible || candidate.tec_total + 1e-9 < best.tec_total)
                    best = candidate;
            }

            best.variant = name;
            best.runtime_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            print_csv_row(instance_id, epsilon, best, best.final_machine_loads);
            return best;
        }

        Assignment assignment;
        if (name == "dp_guided_assignment_dp")
            assignment = build_dp_guided_assignment(inst.jobs, inst.rates, clipped_prices, epsilon);
        else if (name == "lpt_dp")
            assignment = build_pure_lpt_assignment(inst.jobs, inst.rates, clipped_prices, epsilon);
        else if (name == "energy_rate_greedy_dp")
            assignment = build_energy_rate_greedy_assignment(inst.jobs, inst.rates, clipped_prices, epsilon);
        else if (name == "hybrid_load_energy_dp")
            assignment = build_hybrid_load_energy_assignment(inst.jobs, inst.rates, clipped_prices, epsilon);
        else if (name == "w1_rate_class_water_filling_dp")
            assignment = build_w1_rate_class_water_filling(inst.jobs, inst.rates, clipped_prices, epsilon);
        else if (name == "w15_hybrid_pw_tiebreak")
            assignment = build_hybrid_pw_tiebreak(inst.jobs, inst.rates, clipped_prices, epsilon);
        else if (name == "w15_waterfill_pw_correction")
            assignment = build_waterfill_pw_correction(inst.jobs, inst.rates, clipped_prices, epsilon);
        else if (name == "w2_g1_c0_combined")
            assignment = build_w2_g1_c0_combined_waterfill_hybrid_pw(inst.jobs, inst.rates, clipped_prices, epsilon);
        else if (name == "w2_g1_c1_profile_gated")
            assignment = build_w2_g1_c1_profile_gated_selector(inst.jobs, inst.rates, clipped_prices, epsilon);
        else if (name == "w2_g1_c2_multiround_wf_correction")
            assignment = build_w2_g1_c2_multiround_waterfill_correction(inst.jobs, inst.rates, clipped_prices, epsilon);
        else if (name == "w2_g1_c3_slacksafe_hybrid_pwm")
            assignment = build_w2_g1_c3_slacksafe_hybrid_pwm(inst.jobs, inst.rates, clipped_prices, epsilon);
        else if (name == "w2_g1_c4_pwm_adaptive_insert")
            assignment = build_w2_g1_c4_pwm_adaptive_insertion(inst.jobs, inst.rates, clipped_prices, epsilon);
        else if (name == "w2_g1_c5_multipass_portfolio")
            assignment = build_w2_g1_c5_multipass_portfolio(inst.jobs, inst.rates, clipped_prices, epsilon);
        else if (name == "w2_g2_c0_adaptive_tiebreak")
            assignment = build_w2_g2_c0_adaptive_tiebreak(inst.jobs, inst.rates, clipped_prices, epsilon);
        else if (name == "w2_g2_c1_topk_tiebreak")
            assignment = build_w2_g2_c1_topk_tiebreak(inst.jobs, inst.rates, clipped_prices, epsilon);
        else if (name == "w2_g2_c2_energy_gated_pwm")
            assignment = build_w2_g2_c2_energy_gated_pwm(inst.jobs, inst.rates, clipped_prices, epsilon);
        else if (name == "w2_g2_c3_staged_alpha_pwm")
            assignment = build_w2_g2_c3_staged_alpha_pwm(inst.jobs, inst.rates, clipped_prices, epsilon);
        else if (name == "w2_g2_c4_late_pwm_only")
            assignment = build_w2_g2_c4_late_pwm_only(inst.jobs, inst.rates, clipped_prices, epsilon);
        else if (name == "w2_g2_c5_twopass_refine")
            assignment = build_w2_g2_c5_twopass_refine(inst.jobs, inst.rates, clipped_prices, epsilon);
        else if (name == "w4_c1_relocate_only" || name == "w4_c1_op6_combined")
            assignment = build_w2_g2_c1_topk_tiebreak(inst.jobs, inst.rates, clipped_prices, epsilon);
        else if (name == "hybrid_manual_price_tiebreak_dp")
            assignment = build_hybrid_manual_price_tiebreak(inst.jobs, inst.rates, clipped_prices, epsilon);
        else if (name == "a1_rate_stratified_hybrid")
            assignment = build_a1_rate_stratified_hybrid(inst.jobs, inst.rates, clipped_prices, epsilon);
        else if (name == "a2_energy_first_lpt_complete")
            assignment = build_a2_energy_first_lpt_complete(inst.jobs, inst.rates, clipped_prices, epsilon);
        else if (name == "a3_three_strategy_portfolio")
            assignment = build_a3_three_strategy_portfolio(inst.jobs, inst.rates, clipped_prices, epsilon);
        else if (name == "a4_job_size_adaptive_hybrid")
            assignment = build_a4_job_size_adaptive_hybrid(inst.jobs, inst.rates, clipped_prices, epsilon);
        else if (name == "a5_lpt_energy_relocate")
            assignment = build_a5_lpt_energy_relocate(inst.jobs, inst.rates, clipped_prices, epsilon);
        else if (name == "a6_multistart_randomized_hybrid")
            assignment = build_a6_multistart_randomized_hybrid(inst.jobs, inst.rates, clipped_prices, epsilon);
        else if (name == "dsl_config_dp") {
            const char *cfg_path = std::getenv("DSL_CONFIG_PATH");
            if (cfg_path && cfg_path[0] != '\0')
                assignment = build_hybrid_load_energy_assignment(inst.jobs, inst.rates, clipped_prices, epsilon);
            else
                assignment = build_hybrid_load_energy_assignment(inst.jobs, inst.rates, clipped_prices, epsilon);
        }
        else
            assignment = build_lpt_greedy_assignment(inst.jobs, inst.rates, clipped_prices, epsilon);

        VariantResult r;
        if (!assignment.feasible)
        {
            r.variant = name;
            r.feasible = false;
            print_csv_row(instance_id, epsilon, r, assignment.machine_loads);
            return r;
        }

        std::string optimizer = name;
        if (name == "dp_guided_assignment_dp")
            optimizer = "greedy_dp";
        // All LLM assignment policies use fast ESR (DP over fixed sequence)
        if (name.find("lpt_dp") == 0 || name.find("energy_rate_greedy_dp") == 0 ||
            name.find("hybrid_load_energy_dp") == 0 || name.find("hybrid_manual_price_tiebreak_dp") == 0 ||
            name.find("dsl_config_dp") == 0 ||
            name.find("a1_") == 0 || name.find("a2_") == 0 || name.find("a3_") == 0 || name.find("a4_") == 0 || name.find("a5_") == 0 || name.find("a6_") == 0 ||
            name.find("w1_") == 0 ||
            name.find("w15_") == 0 || name.find("w2_") == 0 || name.find("w4_") == 0)
            optimizer = "greedy_esr";
        r = evaluate_variant(
            assignment,
            inst.rates,
            clipped_prices,
            epsilon,
            optimizer,
            per_machine_dp_limit_sec,
            ls_max_rounds,
            ls_max_moves_per_round,
            ls_time_cap_sec);
        r.variant = name;
        print_csv_row(instance_id, epsilon, r, assignment.machine_loads);
        return r;
    };

    if (variant == "all")
    {
        run_variant("greedy_esr");
        run_variant("greedy_dp");
        run_variant("dp_guided_assignment_dp");
        run_variant("greedy_dp_local_search");
        run_variant("greedy_dp_local_search_relocate_only");
        run_variant("greedy_dp_local_search_relocate_multistart");
        run_variant("greedy_dp_local_search_screened_swap");
        run_variant("greedy_dp_local_search_priority_machines");
        run_variant("vnd_exact_dp");
        run_variant("vnd_exact_dp_insert_rank_v1");
        run_variant("vnd_exact_dp_insert_rank_diverse");
        run_variant("vnd_exact_dp_insert_rank_diverse_trimmed");
        run_variant("vnd_exact_dp_insert_rank_diverse_budgeted");
        run_variant("vnd_exact_dp_insert_rank_dense_labeling");
        run_variant("phaseS_llm_exception_lane");
        run_variant("phaseS_random_exception_lane");
        run_variant("phaseS_refined1_stratified");
        run_variant("phaseS_refined2_anticore");
        run_variant("phaseS_refined3_coverage");
        run_variant("phaseV_score_escape_sampler");
        run_variant("phaseX_policy_json");
        run_variant("phaseY_trace_probe");
        run_variant("stageL1_dataset_logging");
        run_variant("stageL15_dense_labeling");
        run_variant("stageO_synthetic_dense_logging");
        return 0;
    }

    if (variant != "greedy_esr" &&
        variant != "greedy_dp" &&
        variant != "dp_guided_assignment_dp" &&
        variant != "greedy_dp_local_search" &&
        variant != "greedy_dp_local_search_relocate_only" &&
        variant != "greedy_dp_local_search_relocate_multistart" &&
        variant != "greedy_dp_local_search_screened_swap" &&
        variant != "greedy_dp_local_search_priority_machines" &&
        variant != "vnd_exact_dp" &&
        variant != "vnd_exact_dp_insert_rank_v1" &&
        variant != "vnd_exact_dp_insert_rank_diverse" &&
        variant != "vnd_exact_dp_insert_rank_diverse_trimmed" &&
        variant != "vnd_exact_dp_insert_rank_diverse_budgeted" &&
        variant != "vnd_exact_dp_insert_rank_dense_labeling" &&
        variant != "phaseS_llm_exception_lane" &&
        variant != "phaseS_random_exception_lane" &&
        variant != "phaseS_refined1_stratified" &&
        variant != "phaseS_refined2_anticore" &&
        variant != "phaseS_refined3_coverage" &&
        variant != "phaseV_score_escape_sampler" &&
        variant != "phaseX_policy_json" &&
        variant != "phaseY_trace_probe" &&
        variant != "phaseY_execute_proposal" &&
        variant != "phaseY_random_proposal" &&
        variant != "phaseI_noscreen_diagnostic" &&
        variant != "stageL1_dataset_logging" &&
        variant != "stageL15_dense_labeling" &&
        variant != "stageO_synthetic_dense_logging" &&
        variant != "history_repair_dp_ranked" &&
        variant != "history_repair_dp_ranked_relocate" &&
        variant != "history_repair_priority_displaced" &&
        variant != "history_repair_priority_displaced_relocate" &&
        // Phase W assignment policies
        variant != "lpt_dp" &&
        variant != "energy_rate_greedy_dp" &&
        variant != "hybrid_load_energy_dp" &&
        variant != "hybrid_manual_price_tiebreak_dp" &&
        variant != "dsl_config_dp" &&
        variant != "a1_rate_stratified_hybrid" &&
        variant != "a2_energy_first_lpt_complete" &&
        variant != "a3_three_strategy_portfolio" &&
        variant != "a4_job_size_adaptive_hybrid" &&
        variant != "a5_lpt_energy_relocate" &&
        variant != "w1_rate_class_water_filling_dp" &&
        variant != "w15_hybrid_pw_tiebreak" &&
        variant != "w15_waterfill_pw_correction" &&
        variant != "w2_g1_c0_combined" &&
        variant != "w2_g1_c1_profile_gated" &&
        variant != "w2_g1_c2_multiround_wf_correction" &&
        variant != "w2_g1_c3_slacksafe_hybrid_pwm" &&
        variant != "w2_g1_c4_pwm_adaptive_insert" &&
        variant != "w2_g1_c5_multipass_portfolio" &&
        variant != "w2_g2_c0_adaptive_tiebreak" &&
        variant != "w2_g2_c1_topk_tiebreak" &&
        variant != "w2_g2_c2_energy_gated_pwm" &&
        variant != "w2_g2_c3_staged_alpha_pwm" &&
        variant != "w2_g2_c4_late_pwm_only" &&
        variant != "w2_g2_c5_twopass_refine" &&
        variant != "w4_c1_relocate_only" &&
        variant != "w4_c1_op6_combined")
    {
        std::cerr << "Unsupported variant: " << variant << "\n";
        return 1;
    }

    run_variant(variant);
    return 0;
}

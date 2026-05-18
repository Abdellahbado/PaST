#include "stateful_dp_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_set>

namespace
{

    // ---------------------------------------------------------------------------
    // Minimal JSON array parsers — no external dependencies.
    // Handles: {"instance_id":"...","prices":[1.0,2.0,...],"jobs":[1,2,...]}
    // ---------------------------------------------------------------------------

    std::vector<double> json_parse_double_array(const std::string &s, const std::string &key)
    {
        std::vector<double> out;
        std::string needle = "\"" + key + "\"";
        auto pos = s.find(needle);
        if (pos == std::string::npos)
            return out;
        auto lb = s.find('[', pos);
        auto rb = s.find(']', lb);
        if (lb == std::string::npos || rb == std::string::npos)
            return out;
        std::istringstream ss(s.substr(lb + 1, rb - lb - 1));
        std::string tok;
        while (std::getline(ss, tok, ','))
        {
            while (!tok.empty() && (tok.front() == ' ' || tok.front() == '\n'))
                tok.erase(tok.begin());
            if (!tok.empty())
                out.push_back(std::stod(tok));
        }
        return out;
    }

    std::vector<int> json_parse_int_array(const std::string &s, const std::string &key)
    {
        std::vector<int> out;
        std::string needle = "\"" + key + "\"";
        auto pos = s.find(needle);
        if (pos == std::string::npos)
            return out;
        auto lb = s.find('[', pos);
        auto rb = s.find(']', lb);
        if (lb == std::string::npos || rb == std::string::npos)
            return out;
        std::istringstream ss(s.substr(lb + 1, rb - lb - 1));
        std::string tok;
        while (std::getline(ss, tok, ','))
        {
            while (!tok.empty() && (tok.front() == ' ' || tok.front() == '\n'))
                tok.erase(tok.begin());
            if (!tok.empty())
                out.push_back(std::stoi(tok));
        }
        return out;
    }

    std::string json_parse_string(const std::string &s, const std::string &key)
    {
        std::string needle = "\"" + key + "\"";
        auto pos = s.find(needle);
        if (pos == std::string::npos)
            return "";
        auto colon = s.find(':', pos + needle.size());
        auto q1 = s.find('"', colon + 1);
        auto q2 = s.find('"', q1 + 1);
        if (q1 == std::string::npos || q2 == std::string::npos)
            return "";
        return s.substr(q1 + 1, q2 - q1 - 1);
    }

    int resolve_max_gap(
        const dp::MachineStateConfig &cfg,
        const std::vector<double> &prices,
        bool use_banded_spaces)
    {
        if (!use_banded_spaces)
            return -1;

        int T = static_cast<int>(prices.size());
        int auto_gap = dp::auto_max_gap(cfg, T, prices);

        if (const char *override_v = std::getenv("PAST_MAX_GAP_OVERRIDE"))
        {
            std::string s(override_v);
            if (s == "full")
                return -1;
            if (s == "auto" || s.empty())
                return auto_gap;
            try
            {
                int v = std::stoi(s);
                return std::min(T, std::max(v, 0));
            }
            catch (const std::exception &)
            {
                return auto_gap;
            }
        }

        if (const char *scale_v = std::getenv("PAST_MAX_GAP_SCALE"))
        {
            try
            {
                double scale = std::stod(scale_v);
                if (scale > 0.0)
                {
                    int scaled = static_cast<int>(std::ceil(auto_gap * scale));
                    return std::min(T, std::max(scaled, 1));
                }
            }
            catch (const std::exception &)
            {
            }
        }

        return auto_gap;
    }

    // ---------------------------------------------------------------------------
    // Ablation configuration — controls which components are active.
    // ---------------------------------------------------------------------------
    struct AblationConfig
    {
        bool use_banded_spaces = true; // false → full O(h²) SPACES
        bool use_heuristics = true;    // false → skip Steps 2-5 (no primal heuristics)
        bool use_relaxation_lb = true; // false → skip Steps 1,4,5 LB computation
        bool use_smart_recon = true;   // false → skip Step 5.5
        bool use_exact_shortcut = true;
        bool use_exact_dp = true;
        bool profile_bounds = false; // true → compute all LB stages even if gap closes
        bool use_exact_guidance = false; // run semigroup DP only to guide sparse exact DP
        bool use_feas_profile_pack = false; // run R_feas + recovered-profile packing before exact
        bool adaptive_pipeline = true; // skip wasteful stages for known instance regimes
        // When both use_heuristics=false and use_relaxation_lb=false,
        // we get exact-DP-only (baseline).
    };

using Clock = std::chrono::steady_clock;
using Dur = std::chrono::duration<double>;

static bool env_flag_exact(const char *name)
{
    const char *raw = std::getenv(name);
    return raw && std::string(raw) == "1";
}

static int env_int_exact(const char *name, int fallback)
{
    const char *raw = std::getenv(name);
    if (!raw || !*raw)
        return fallback;
    char *end = nullptr;
    long v = std::strtol(raw, &end, 10);
    if (end == raw)
        return fallback;
    if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max())
        return fallback;
    return static_cast<int>(v);
}

static double env_double_exact(const char *name, double fallback)
{
    const char *raw = std::getenv(name);
    if (!raw || !*raw)
        return fallback;
    char *end = nullptr;
    double v = std::strtod(raw, &end);
    if (end == raw)
        return fallback;
    return v;
}

static int64_t env_int64_exact(const char *name, int64_t fallback)
{
    const char *raw = std::getenv(name);
    if (!raw || !*raw)
        return fallback;
    char *end = nullptr;
    long long v = std::strtoll(raw, &end, 10);
    if (end == raw)
        return fallback;
    return static_cast<int64_t>(v);
}

static std::string env_str_exact(const char *name)
{
    const char *raw = std::getenv(name);
    return raw ? std::string(raw) : std::string();
}

static std::string to_lower_ascii_copy(std::string s)
{
    for (char &ch : s)
    {
        if (ch >= 'A' && ch <= 'Z')
            ch = static_cast<char>(ch - 'A' + 'a');
    }
    return s;
}

static std::string canonical_exact_variant(std::string v)
{
    v = to_lower_ascii_copy(v);
    if (v.empty() || v == "baseline")
        return "p0";
    if (v == "type" || v == "type_aware")
        return "p1";
    if (v == "ordering" || v == "inc_order")
        return "p2";
    if (v == "type_order" || v == "p1p2")
        return "p3";
    if (v == "p0" || v == "p1" || v == "p2" || v == "p3" || v == "p4")
        return v;
    return "p0";
}

static double completion_lookup(
    const dp::RelaxedTableResult &tab,
    const std::vector<double> &arr,
    int T,
    int t,
    int rw)
{
    if (arr.empty() || tab.RW <= 0)
        return dp::kInf;
    if (t < 0)
        t = 0;
    if (t > T + 1)
        t = T + 1;
    int scaled_rw = tab.rw_scale > 1 ? rw / tab.rw_scale : rw;
    if (scaled_rw < 0 || scaled_rw >= tab.RW)
        return dp::kInf;
    return arr[static_cast<std::size_t>(t) * tab.RW + scaled_rw];
}

static dp::RelaxedDPResult run_partial_binpack_stage(
    const std::vector<int> &lens,
    const std::vector<int> &tots,
    const std::vector<double> &prefix,
    int T,
    const dp::SPACESResult &spaces)
{
    const int max_auto_tracked = env_int_exact("PAST_PARTIAL_MAX_TRACKED", 1);
    const bool use_remainder_feas = !env_flag_exact("PAST_PARTIAL_DISABLE_REMAINDER_FEAS");
    const int K = static_cast<int>(lens.size());
    int trials = std::max(1, env_int_exact("PAST_PARTIAL_TRACKED_TRIALS", 1));

    auto run_one = [&](std::vector<int> tracked) -> dp::RelaxedDPResult
    {
        return dp::solve_relaxed_dp_lb_partial_with_binpack(
            lens, tots, prefix, T, spaces, std::move(tracked), max_auto_tracked, use_remainder_feas);
    };

    dp::RelaxedDPResult best = run_one({});
    if (trials <= 1 || K <= 1)
        return best;

    std::vector<int> order(K);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b)
              {
                  if (tots[a] != tots[b])
                      return tots[a] < tots[b];
                  if (lens[a] != lens[b])
                      return lens[a] > lens[b];
                  return a < b;
              });

    auto better = [](const dp::RelaxedDPResult &cand, const dp::RelaxedDPResult &cur) -> bool
    {
        const bool cand_pack = cand.pack_outcome == "feasible";
        const bool cur_pack = cur.pack_outcome == "feasible";
        if (cand_pack != cur_pack)
            return cand_pack;
        const bool cand_lb = std::isfinite(cand.lb) && cand.lb < dp::kInf * 0.5;
        const bool cur_lb = std::isfinite(cur.lb) && cur.lb < dp::kInf * 0.5;
        if (cand_lb != cur_lb)
            return cand_lb;
        if (cand_lb && cur_lb && std::fabs(cand.lb - cur.lb) > 1e-9)
            return cand.lb > cur.lb;
        const bool cand_ub = std::isfinite(cand.bin_pack_ub) && cand.bin_pack_ub < dp::kInf * 0.5;
        const bool cur_ub = std::isfinite(cur.bin_pack_ub) && cur.bin_pack_ub < dp::kInf * 0.5;
        if (cand_ub != cur_ub)
            return cand_ub;
        if (cand_ub && cur_ub && std::fabs(cand.bin_pack_ub - cur.bin_pack_ub) > 1e-9)
            return cand.bin_pack_ub < cur.bin_pack_ub;
        return false;
    };

    int tried = 0;
    if (max_auto_tracked >= 2 && K >= 2)
    {
        for (int a = 0; a < K; ++a)
        {
            for (int b = a + 1; b < K; ++b)
            {
                if (++tried > trials)
                    break;
                auto cand = run_one({order[a], order[b]});
                if (better(cand, best))
                    best = std::move(cand);
                if (best.pack_outcome == "feasible")
                    break;
            }
            if (tried >= trials || best.pack_outcome == "feasible")
                break;
        }
    }
    else
    {
        for (int j : order)
        {
            if (++tried > trials)
                break;
            auto cand = run_one({j});
            if (better(cand, best))
                best = std::move(cand);
            if (best.pack_outcome == "feasible")
                break;
        }
    }
    return best;
}

    bool is_valid_relax_lb(double v)
    {
        return std::isfinite(v) && v < dp::kInf * 0.5;
    }

    bool use_suffix_completion_guidance()
    {
        const char *v = std::getenv("PAST_DISABLE_SUFFIX_COMPLETION");
        if (!v)
            return true;
        std::string s(v);
        return !(s == "1" || s == "true" || s == "TRUE" || s == "yes" || s == "YES");
    }

    bool use_guided_incumbent_only()
    {
        const char *v = std::getenv("PAST_GUIDED_UB_ONLY");
        if (!v)
            return false;
        std::string s(v);
        return (s == "1" || s == "true" || s == "TRUE" || s == "yes" || s == "YES");
    }

    bool use_beam_incumbent_only()
    {
        const char *v = std::getenv("PAST_BEAM_UB_ONLY");
        if (!v)
            return false;
        std::string s(v);
        return (s == "1" || s == "true" || s == "TRUE" || s == "yes" || s == "YES");
    }

    int beam_width_setting()
    {
        const char *v = std::getenv("PAST_BEAM_WIDTH");
        if (!v)
            return 256;
        try
        {
            return std::max(1, std::stoi(std::string(v)));
        }
        catch (const std::exception &)
        {
            return 256;
        }
    }

    bool adaptive_pipeline_enabled()
    {
        const char *v = std::getenv("PAST_DISABLE_ADAPTIVE_PIPELINE");
        if (!v)
            return true;
        std::string s(v);
        return !(s == "1" || s == "true" || s == "TRUE" || s == "yes" || s == "YES");
    }

    int adaptive_feas_k_min()
    {
        return env_int_exact("PAST_ADAPTIVE_FEAS_K_MIN", 5);
    }

    bool should_adaptive_jump_to_exact(
        int K,
        double ub,
        const std::string &pack_outcome,
        bool use_exact_shortcut,
        bool exact_enabled,
        bool adaptive_enabled,
        bool profile_bounds)
    {
        if (!exact_enabled || use_exact_shortcut || !adaptive_enabled || profile_bounds)
            return false;
        if (K < adaptive_feas_k_min())
            return false;
        if (!(ub < dp::kInf * 0.5))
            return false;
        return pack_outcome == "feasible" || pack_outcome == "exact";
    }

    // ---------------------------------------------------------------------------
    // Ablation-aware solver. Returns a structured CSV row with per-step timing.
    // ---------------------------------------------------------------------------
    std::string solve_one_ablation(
        const std::string &instance_id,
        const std::vector<double> &prices,
        const std::vector<int> &jobs,
        const std::string &machine_type,
        const AblationConfig &ab,
        double time_limit = -1.0)
    {
        auto t0_total = Clock::now();

        std::map<int, int> cnt;
        for (int p : jobs)
            cnt[p]++;
        std::vector<int> lens, tots;
        for (auto &kv : cnt)
        {
            lens.push_back(kv.first);
            tots.push_back(kv.second);
        }

        auto cfg = (machine_type == "twosby") ? dp::make_paper_twosby_config()
                                              : dp::make_paper_nosby_config();

        // --- SPACES preprocessing ---
        auto t0_spaces = Clock::now();
        int mg = resolve_max_gap(cfg, prices, ab.use_banded_spaces);
        auto spaces = dp::compute_spaces(prices, cfg, mg);
        double t_spaces = Dur(Clock::now() - t0_spaces).count();

        auto prefix = dp::build_proc_prefix(prices, spaces.p_proc);
        int T = (int)prices.size();
        int total_rw = 0;
        for (std::size_t i = 0; i < lens.size(); ++i)
            total_rw += lens[i] * tots[i];

        double ub = dp::kInf;
        double lb = 0.0;
        bool timed_out = false;
        auto gap_closed = [&]()
        { return std::fabs(ub - lb) < 0.01; };
        auto elapsed_sec = [&]()
        { return Dur(Clock::now() - t0_total).count(); };
        auto remaining_sec = [&]()
        {
            if (time_limit <= 0.0)
                return 1.0e18;
            return std::max(0.0, time_limit - elapsed_sec());
        };
        auto out_of_time = [&]()
        {
            if (time_limit > 0.0 && elapsed_sec() >= time_limit)
            {
                timed_out = true;
                return true;
            }
            return false;
        };
        auto should_stop = [&]()
        { return gap_closed() && !ab.profile_bounds; };

        int64_t NC_est = 1;
        bool use_exact_shortcut = false;
        for (std::size_t i = 0; i < lens.size(); ++i)
        {
            NC_est *= (tots[i] + 1);
            if (NC_est > 50'000)
                break;
        }
        if (ab.use_exact_shortcut &&
            NC_est < 50'000 &&
            NC_est * static_cast<int64_t>(T + 2) <= 600'000'000LL)
            use_exact_shortcut = true;

        double t_fwd_relax = 0, t_heuristic = 0, t_local_search = 0;
        double t_bwd_relax = 0, t_two_class = 0, t_exact = 0;
        double t_feas_profile = 0;
        dp::ExactDPDiagnostics exact_diag_row;
        dp::LocalCorridorDiag local_corridor_diag;
        auto should_replace_exact_diag = [](const dp::ExactDPDiagnostics &current_diag,
                                            const dp::ExactDPDiagnostics &candidate_diag)
        {
            if (candidate_diag.mode.empty() || candidate_diag.mode == "none")
                return false;
            if (candidate_diag.mode.rfind("dense_skip_", 0) == 0 &&
                current_diag.mode.rfind("sparse", 0) == 0)
                return false;
            return true;
        };
        std::string exact_incumbent_source = to_lower_ascii_copy(env_str_exact("PAST_EXACT_INCUMBENT_SOURCE"));
        if (exact_incumbent_source.empty())
            exact_incumbent_source = "auto";
        std::string exact_variant_env = canonical_exact_variant(env_str_exact("PAST_EXACT_DP_VARIANT"));
        std::string exact_variant_active = exact_variant_env;
        std::string step_reached = "none";

        // Per-step LB/UB tracking (for diagnostics)
        double lb_after_fwd = 0, lb_after_feas = 0, lb_after_fl = 0;
        double lb_after_feas_profile = 0;
        double ub_after_fwd = dp::kInf, ub_after_heur = dp::kInf, ub_after_ls = dp::kInf;
        double ub_after_feas_profile = dp::kInf;
        int64_t states_fwd_reached = 0, states_fwd_expanded = 0;
        dp::RelaxedTableResult suffix_relax;
        bool suffix_relax_ready = false;
        auto ensure_suffix_relax = [&]()
        {
            if (suffix_relax_ready)
                return;
            suffix_relax = dp::compute_relaxed_completion_table(
                lens, total_rw, prefix, T, spaces, dp::RelaxationMode::Semigroup);
            suffix_relax_ready = true;
        };
        double t_smart_recon = 0;
        std::string winner_detail = "none";
        bool guided_incumbent_only = use_guided_incumbent_only();
        bool beam_incumbent_only = use_beam_incumbent_only();
        int beam_width = beam_width_setting();
        bool enable_suffix_completion = use_suffix_completion_guidance();
        bool adaptive_enabled = ab.adaptive_pipeline && adaptive_pipeline_enabled();

        // Declare fwd outside block so we can reuse rdp table in smart_reconstruct
        dp::RelaxedDPResult fwd;
        dp::RelaxedDPResult partial_prof;
        dp::RelaxedDPResult feas_prof;

        // PLAN32: anytime initial UB diagnostics — declared early for goto safety
        double anytime_initial_ub = dp::kInf;
        std::string anytime_initial_ub_source = "none";
        double anytime_time_to_first_ub = 0.0;
        int anytime_initial_ub_valid = 0;
        int anytime_ub_used_on_timeout = 0;

        // PLAN32B: parallel initial UB diagnostics
        double parallel_initial_ub = dp::kInf;
        int parallel_initial_ub_valid = 0;
        std::string parallel_initial_ub_policy = "none";
        double parallel_initial_ub_time_sec = 0.0;
        int parallel_initial_ub_machines_used = 0;
        int parallel_initial_ub_failed_machines = 0;
        int parallel_initial_ub_used_on_timeout = 0;
        int initial_ub_lb_consistent = 1;
        std::string initial_ub_rejected_reason;
        std::string initial_ub_model_note = "single_machine";

        // PLAN33: certified anytime hard-K prepass diagnostics
        int cert_anytime_enabled = 0;
        int cert_anytime_k_min = 0;
        double cert_anytime_gap_stop_pct = 0.0;
        int cert_anytime_triggered = 0;
        int cert_anytime_stopped = 0;
        double cert_anytime_initial_ub = 0.0;
        double cert_anytime_lb = 0.0;
        double cert_anytime_gap_pct = 0.0;
        std::string cert_anytime_best_policy;
        int cert_anytime_finite_candidates = 0;
        double cert_anytime_time_to_first_ub = 0.0;
        double cert_anytime_time_total = 0.0;
        int cert_anytime_polish_used = 0;
        double cert_anytime_ub_before_polish = 0.0;
        double cert_anytime_ub_after_polish = 0.0;

        // ── PLAN32/PLAN32C: anytime initial UB safety layer (runs BEFORE forward DP) ──
        if (env_flag_exact("PAST_ANYTIME_INITIAL_UB"))
        {
            bool hardk_only = env_flag_exact("PAST_ANYTIME_HARDK_ONLY");
            int K_env = static_cast<int>(lens.size());
            if (!hardk_only || K_env >= 10)
            {
                auto t0_any = Clock::now();
                int trials = std::max(1, env_int_exact("PAST_ANYTIME_INITIAL_UB_TRIALS", 75));
                bool use_local_search = env_flag_exact("PAST_ANYTIME_INITIAL_UB_LOCAL_SEARCH");

                double init_ub = dp::compute_initial_ub(lens, tots, prefix, T, spaces, trials, lb);

                // PLAN32C: parallel initial UB — partition jobs across M machines.
                // This is DIAGNOSTIC ONLY by default (changes the model from 1-machine to M-machine).
                // To use as incumbent, set PAST_ANYTIME_PARALLEL_UB_OPT_IN=1.
                bool parallel_opt_in = env_flag_exact("PAST_ANYTIME_PARALLEL_UB_OPT_IN");
                int para_M = (parallel_opt_in || env_flag_exact("PAST_ANYTIME_PARALLEL_DIAGNOSTIC"))
                                 ? env_int_exact("PAST_ANYTIME_PARALLEL_MACHINES", -1) : 0;
                if (para_M <= 0 && (parallel_opt_in || env_flag_exact("PAST_ANYTIME_PARALLEL_DIAGNOSTIC")))
                {
                    int window = spaces.late > spaces.early ? (spaces.late - spaces.early + 1) : T;
                    para_M = std::max(1, (total_rw + window - 1) / window);
                    para_M = std::max(para_M, (int)std::ceil(total_rw / (0.7 * window)));
                    if (para_M > 64) para_M = 64;
                }

                if (para_M >= 2)
                {
                    auto t0_para = Clock::now();
                    std::string para_pol;
                    int para_used = 0, para_failed = 0;
                    double para_ub = dp::compute_parallel_initial_ub(
                        lens, tots, prefix, T, spaces, para_M, trials, lb,
                        &para_pol, &para_used, &para_failed);
                    parallel_initial_ub_time_sec = Dur(Clock::now() - t0_para).count();

                    if (para_ub < dp::kInf * 0.5)
                    {
                        parallel_initial_ub = para_ub;
                        parallel_initial_ub_valid = 1;
                        parallel_initial_ub_policy = para_pol;
                        parallel_initial_ub_machines_used = para_used;
                        parallel_initial_ub_failed_machines = para_failed;
                        if (para_ub < init_ub && parallel_opt_in)
                        {
                            init_ub = para_ub;
                            anytime_initial_ub_source = "parallel_" + para_pol;
                        }
                        // If not opt-in, note model mismatch but record diagnostic
                        if (!parallel_opt_in)
                            initial_ub_model_note = "parallel_diag_only_UB:" + std::to_string(static_cast<long long>(para_ub));
                    }
                }

                if (use_local_search && init_ub < dp::kInf * 0.5)
                {
                    std::vector<int> all_jobs;
                    for (std::size_t ji = 0; ji < lens.size(); ++ji)
                        for (int j = 0; j < tots[ji]; ++j)
                            all_jobs.push_back(lens[ji]);
                    std::vector<int> seq = all_jobs;
                    std::sort(seq.begin(), seq.end());
                    double ls = dp::local_search_ub(seq, dp::solve_fixed_sequence(seq, prefix, T, spaces), prefix, T, spaces, 5, 2.0);
                    if (ls < init_ub) init_ub = ls;
                    std::sort(seq.begin(), seq.end(), std::greater<int>());
                    ls = dp::local_search_ub(seq, dp::solve_fixed_sequence(seq, prefix, T, spaces), prefix, T, spaces, 5, 2.0);
                    if (ls < init_ub) init_ub = ls;
                }

                anytime_time_to_first_ub = Dur(Clock::now() - t0_any).count();

                if (init_ub < dp::kInf * 0.5)
                {
                    anytime_initial_ub = init_ub;
                    if (anytime_initial_ub_source.empty() || anytime_initial_ub_source == "none")
                        anytime_initial_ub_source = use_local_search ? "portfolio_with_ls" : "portfolio";
                    anytime_initial_ub_valid = 1;
                    if (init_ub < ub)
                    {
                        ub = init_ub;
                        if (winner_detail == "none")
                            winner_detail = "anytime_initial_ub";
                    }
                }
            }
        }

        // ── PLAN33: certified anytime hard-K prepass ──
        if (env_flag_exact("PAST_CERT_ANYTIME_PREPASS"))
        {
            int K_env = static_cast<int>(lens.size());
            int k_min = env_int_exact("PAST_CERT_ANYTIME_K_MIN", 10);
            double gap_stop_pct = env_double_exact("PAST_CERT_ANYTIME_GAP_STOP_PCT", 0.1);
            int cert_trials = env_int_exact("PAST_CERT_ANYTIME_TRIALS", 5);
            int polish_en = env_int_exact("PAST_CERT_ANYTIME_POLISH", 1);

            cert_anytime_enabled = 1;
            cert_anytime_k_min = k_min;
            cert_anytime_gap_stop_pct = gap_stop_pct;

            if (K_env >= k_min)
            {
                cert_anytime_triggered = 1;
                auto t0_cert = Clock::now();

                // Phase 1: compute initial UB with enhanced diagnostics
                std::string best_policy;
                int finite_candidates = 0;
                double time_to_first_ub = 0.0;
                std::vector<int> best_seq;
                double cert_ub = dp::compute_initial_ub(lens, tots, prefix, T, spaces,
                    cert_trials, lb, &best_policy, &finite_candidates,
                    &time_to_first_ub, &best_seq);
                cert_anytime_ub_before_polish = cert_ub;
                cert_anytime_best_policy = best_policy;
                cert_anytime_finite_candidates = finite_candidates;
                cert_anytime_time_to_first_ub = time_to_first_ub;

                // Phase 2: polish the best sequence if feasible
                if (polish_en && cert_ub < dp::kInf * 0.5 && !best_seq.empty())
                {
                    double polished = dp::polish_best_sequence_ub(
                        best_seq, cert_ub, prefix, T, spaces, 5.0);
                    if (polished < cert_ub - 1e-6)
                    {
                        cert_ub = polished;
                        cert_anytime_polish_used = 1;
                    }
                }
                cert_anytime_ub_after_polish = cert_ub;

                // Phase 2.5: fallback — if cert prepass found no finite UB but
                // the anytime block did, borrow the anytime UB
                if (!(cert_ub < dp::kInf * 0.5) && anytime_initial_ub_valid &&
                    anytime_initial_ub < dp::kInf * 0.5)
                {
                    cert_ub = anytime_initial_ub;
                    cert_anytime_ub_before_polish = anytime_initial_ub;
                    cert_anytime_ub_after_polish = anytime_initial_ub;
                    cert_anytime_best_policy = "fallback_" + anytime_initial_ub_source;
                    cert_anytime_finite_candidates = 0;
                    cert_anytime_time_to_first_ub = 0.0;
                    cert_anytime_polish_used = 0;
                }

                cert_anytime_initial_ub = cert_ub;
                cert_anytime_time_total = Dur(Clock::now() - t0_cert).count();

                // Set incumbent from prepass for forward DP to use
                if (cert_ub < dp::kInf * 0.5 && cert_ub < ub)
                {
                    ub = cert_ub;
                    if (winner_detail == "none")
                        winner_detail = "cert_anytime_prepass";
                    anytime_initial_ub = cert_ub;
                    anytime_initial_ub_valid = 1;
                    anytime_initial_ub_source = best_policy;
                    anytime_time_to_first_ub = time_to_first_ub;
                }
            }
        }

        // --- Step 1: Forward relaxed DP with bin-packing (LB + UB) ---
        if (ab.use_relaxation_lb || ab.use_heuristics || ab.use_exact_guidance)
        {
            // PLAN32B: if anytime_initial_ub_only is set, skip heavy DP
            // and return the anytime UB immediately (debug/Phase B mode).
            if (env_flag_exact("PAST_ANYTIME_INITIAL_UB_ONLY") &&
                (anytime_initial_ub_valid || parallel_initial_ub_valid))
            {
                if (step_reached == "anytime_initial_ub" || step_reached == "none")
                    step_reached = "anytime_fallback";
                goto done;
            }

            auto t0 = Clock::now();

            // PLAN33: compute semigroup LB first for early gap-stop
            bool plan33_lb_done = false;
            if (cert_anytime_triggered && ub < dp::kInf * 0.5)
            {
                auto fwd_tab = dp::compute_relaxed_dp_table(
                    lens, total_rw, prefix, T, spaces, dp::RelaxationMode::Semigroup);
                double semigroup_lb = fwd_tab.lb;
                fwd.lb = semigroup_lb;
                fwd.rdp = std::move(fwd_tab.rdp);
                fwd.RW = fwd_tab.RW;
                fwd.bin_pack_ub = dp::kInf;
                if (ab.use_relaxation_lb && semigroup_lb > lb)
                    lb = semigroup_lb;
                plan33_lb_done = true;

                // PLAN33 gap-stop check
                if (lb > 0)
                {
                    double cert_gap = 100.0 * (ub - lb) / lb;
                    cert_anytime_gap_pct = cert_gap;
                    cert_anytime_lb = lb;
                    if (cert_gap <= cert_anytime_gap_stop_pct)
                    {
                        cert_anytime_stopped = 1;
                        winner_detail = "cert_anytime_prepass";
                        step_reached = "cert_anytime_prepass";
                        t_fwd_relax = Dur(Clock::now() - t0).count();
                        lb_after_fwd = lb;
                        ub_after_fwd = ub;
                        goto done;
                    }
                }
            }

            // Full forward DP (bin-packing + profiles)
            if (env_flag_exact("PAST_FWD_LB_ONLY") && !plan33_lb_done)
            {
                auto fwd_tab = dp::compute_relaxed_dp_table(
                    lens, total_rw, prefix, T, spaces, dp::RelaxationMode::Semigroup);
                fwd.lb = fwd_tab.lb;
                fwd.rdp = std::move(fwd_tab.rdp);
                fwd.RW = fwd_tab.RW;
                fwd.bin_pack_ub = dp::kInf;
            }
            else if (!plan33_lb_done)
            {
                fwd = dp::solve_relaxed_dp_with_binpack(lens, tots, prefix, T, spaces);
            }
            else
            {
                // PLAN33 computed LB; now compute full pack if gap was too large
                fwd = dp::solve_relaxed_dp_with_binpack(lens, tots, prefix, T, spaces);
            }
            t_fwd_relax = Dur(Clock::now() - t0).count();
            if (ab.use_relaxation_lb)
                lb = fwd.lb;
            if ((ab.use_relaxation_lb || ab.use_heuristics) && fwd.bin_pack_ub < ub)
                ub = fwd.bin_pack_ub;
            states_fwd_reached = fwd.states_reached;
            states_fwd_expanded = fwd.states_expanded;
            step_reached = ab.use_exact_guidance && !ab.use_relaxation_lb && !ab.use_heuristics
                               ? "exact_guidance"
                               : "fwd_relax";
            lb_after_fwd = lb;
            ub_after_fwd = ub;

            if (should_stop())
            {
                winner_detail = (fwd.pack_method != "none")
                                    ? ("fwd_relax:" + fwd.pack_method)
                                    : "fwd_relax";
                goto done;
            }
            if (should_adaptive_jump_to_exact(
                    static_cast<int>(lens.size()), ub, fwd.pack_outcome,
                    use_exact_shortcut, ab.use_exact_dp, adaptive_enabled, ab.profile_bounds))
                goto exact_dp;
            if (out_of_time())
                goto done;

            // PLAN33: certified anytime gap-stop — early exit if gap <= threshold
            if (cert_anytime_triggered && ub < dp::kInf * 0.5 && lb > 0)
            {
                double cert_gap = 100.0 * (ub - lb) / lb;
                cert_anytime_gap_pct = cert_gap;
                cert_anytime_lb = lb;
                if (cert_gap <= cert_anytime_gap_stop_pct)
                {
                    cert_anytime_stopped = 1;
                    winner_detail = "cert_anytime_prepass";
                    step_reached = "cert_anytime_prepass";
                    goto done;
                }
            }
        }

        if (env_flag_exact("PAST_PARTIAL_BINPACK_STAGE"))
        {
            auto t0 = Clock::now();
            partial_prof = run_partial_binpack_stage(lens, tots, prefix, T, spaces);
            t_two_class = Dur(Clock::now() - t0).count();
            if (is_valid_relax_lb(partial_prof.lb) && partial_prof.lb > lb)
                lb = partial_prof.lb;
            if (partial_prof.bin_pack_ub < ub)
                ub = partial_prof.bin_pack_ub;
            step_reached = "partial_profile";
            if (should_stop())
            {
                winner_detail = (partial_prof.pack_method != "none")
                                    ? ("partial_profile:" + partial_prof.pack_method)
                                    : "partial_profile";
                goto done;
            }
            if (should_adaptive_jump_to_exact(
                    static_cast<int>(lens.size()), ub, partial_prof.pack_outcome,
                    use_exact_shortcut, ab.use_exact_dp, adaptive_enabled, ab.profile_bounds))
                goto exact_dp;
            if (out_of_time())
                goto done;
        }

        // --- Optional clean escalation: R_feas + recovered-profile packing ---
        // Used for the policy study that compares:
        //   semi -> fixed-profile certifier -> exact
        // against
        //   semi -> fixed-profile certifier -> feas -> fixed-profile certifier -> exact
        if (ab.use_feas_profile_pack)
        {
            auto t0 = Clock::now();
            feas_prof = dp::solve_relaxed_dp_lb_feas_with_binpack(lens, tots, prefix, T, spaces);
            t_feas_profile = Dur(Clock::now() - t0).count();
            if (is_valid_relax_lb(feas_prof.lb) && feas_prof.lb > lb)
                lb = feas_prof.lb;
            if (feas_prof.bin_pack_ub < ub)
                ub = feas_prof.bin_pack_ub;
            lb_after_feas_profile = lb;
            ub_after_feas_profile = ub;
            step_reached = "feas_profile";
            if (should_stop())
            {
                winner_detail = (feas_prof.pack_method != "none")
                                    ? ("feas_profile:" + feas_prof.pack_method)
                                    : "feas_profile";
                goto done;
            }
            if (should_adaptive_jump_to_exact(
                    static_cast<int>(lens.size()), ub, feas_prof.pack_outcome,
                    use_exact_shortcut, ab.use_exact_dp, adaptive_enabled, ab.profile_bounds))
                goto exact_dp;
            if (out_of_time())
                goto done;
        }

        if (use_exact_shortcut)
            goto exact_dp;

        // --- Step 2: Incumbent builder ---
        if (ab.use_heuristics)
        {
            auto t0 = Clock::now();
            double heur_ub = dp::kInf;
            if (beam_incumbent_only)
            {
                if (enable_suffix_completion)
                    ensure_suffix_relax();
                heur_ub = dp::completion_guided_beam_ub(
                    lens, tots, prefix, T, spaces,
                    suffix_relax.rdp, suffix_relax.RW, suffix_relax.rw_scale,
                    ub, beam_width, std::min(remaining_sec(), 30.0));
            }
            else if (guided_incumbent_only)
            {
                if (enable_suffix_completion)
                    ensure_suffix_relax();
                heur_ub = dp::guided_completion_ub(
                    lens, tots, prefix, T, spaces,
                    suffix_relax.rdp, suffix_relax.RW, suffix_relax.rw_scale);
            }
            else
            {
                heur_ub = dp::compute_initial_ub(lens, tots, prefix, T, spaces, 50, lb);
            }
            t_heuristic = Dur(Clock::now() - t0).count();
            if (heur_ub < ub)
                ub = heur_ub;
            step_reached = "heuristic_ub";
            ub_after_heur = ub;
            if (should_stop())
            {
                winner_detail = "heuristic_ub";
                goto done;
            }
            if (out_of_time())
                goto done;
        }

        // --- Legacy incumbent polish: local search from SPT + LPT ---
        if (ab.use_heuristics && !guided_incumbent_only && !beam_incumbent_only)
        {
            auto t0 = Clock::now();
            std::vector<int> all_jobs;
            for (std::size_t i = 0; i < lens.size(); ++i)
                for (int j = 0; j < tots[i]; ++j)
                    all_jobs.push_back(lens[i]);

            std::vector<int> seq = all_jobs;
            std::sort(seq.begin(), seq.end());
            double spt_cost = dp::solve_fixed_sequence(seq, prefix, T, spaces);
            double ls_cost = dp::local_search_ub(seq, spt_cost, prefix, T, spaces, 3);
            if (ls_cost < ub)
                ub = ls_cost;

            std::sort(seq.begin(), seq.end(), std::greater<int>());
            double lpt_cost = dp::solve_fixed_sequence(seq, prefix, T, spaces);
            ls_cost = dp::local_search_ub(seq, lpt_cost, prefix, T, spaces, 3);
            if (ls_cost < ub)
                ub = ls_cost;
            t_local_search = Dur(Clock::now() - t0).count();
            step_reached = "local_search";
            ub_after_ls = ub;
            if (should_stop())
            {
                winner_detail = "local_search";
                goto done;
            }
            if (out_of_time())
                goto done;
        }

        // --- Step 4: R_feas LB (transition-feasibility filter) ---
        if (ab.use_heuristics && ab.use_relaxation_lb)
        {
            auto t0 = Clock::now();
            double lb_feas = dp::solve_relaxed_dp_lb_feas(lens, tots, prefix, T, spaces);
            t_bwd_relax = Dur(Clock::now() - t0).count(); // reuse column for R_feas
            if (is_valid_relax_lb(lb_feas) && lb_feas > lb)
                lb = lb_feas;
            lb_after_feas = lb;
            step_reached = "r_feas";
            if (should_stop())
            {
                winner_detail = "r_feas";
                goto done;
            }
            if (out_of_time())
                goto done;
        }

        // --- Step 4.5: R_partial LB (partial count-vector tracking) ---
        if (ab.use_heuristics && ab.use_relaxation_lb && (ub - lb > 0.5))
        {
            auto t0 = Clock::now();
            double lb_par = dp::solve_relaxed_dp_lb_partial(
                lens, tots, prefix, T, spaces, {}, 20.0);
            double t_partial = Dur(Clock::now() - t0).count();
            (void)t_partial; // timing captured in overall elapsed
            if (is_valid_relax_lb(lb_par) && lb_par > lb)
                lb = lb_par;
            step_reached = "r_partial";
            if (should_stop())
            {
                winner_detail = "r_partial";
                goto done;
            }
            if (out_of_time())
                goto done;
        }

        // --- Step 5: R_feas+Lagr LB (combined bound) ---
        if (ab.use_heuristics && ab.use_relaxation_lb && (ub - lb > 0.5))
        {
            auto t0 = Clock::now();
            double lb_fl = dp::solve_relaxed_dp_lb_feas_lagrangian(
                lens, tots, prefix, T, spaces, 50, 10.0);
            t_two_class = Dur(Clock::now() - t0).count(); // reuse column for R_feas+Lagr
            if (is_valid_relax_lb(lb_fl) && lb_fl > lb)
                lb = lb_fl;
            lb_after_fl = lb;
            step_reached = "r_feas_lagr";
            if (should_stop())
            {
                winner_detail = "r_feas_lagr";
                goto done;
            }
            if (out_of_time())
                goto done;
        }

        // --- Step 5.5: Smart reconstruction (count-aware search on relaxed DP table) ---
        if (ab.use_smart_recon && !fwd.rdp.empty())
        {
            auto t0 = Clock::now();
            double sr_cost = dp::smart_reconstruct(
                fwd.rdp, fwd.RW,
                lens, tots, prefix, T, spaces,
                ub, 30.0);
            t_smart_recon = Dur(Clock::now() - t0).count();
            if (sr_cost < ub)
                ub = sr_cost;
            if (sr_cost < dp::kInf && std::fabs(sr_cost - lb) < 0.01)
                lb = ub; // proven optimal
            step_reached = "smart_recon";
            if (should_stop())
            {
                winner_detail = "smart_recon";
                goto done;
            }
            if (out_of_time())
                goto done;
        }

        // --- Step 6+7: Exact DP ---
        if (!ab.use_exact_dp)
            goto done;
    exact_dp:
        {
            auto t0 = Clock::now();
            (void)dp::consume_last_exact_dp_diagnostics();
            double exact_budget = remaining_sec();
            if (exact_budget <= 0.0)
            {
                timed_out = true;
                goto done;
            }
            const std::string exact_guide_source = env_str_exact("PAST_EXACT_GUIDE_SOURCE");
            const dp::RelaxedDPResult *guide = nullptr;
            if (exact_guide_source == "feas" && !feas_prof.rdp.empty())
                guide = &feas_prof;
            else if (exact_guide_source == "partial" && !partial_prof.rdp.empty())
                guide = &partial_prof;
            else if (!fwd.rdp.empty() && (ab.use_relaxation_lb || ab.use_heuristics || ab.use_exact_guidance))
                guide = &fwd;

            auto choose_exact_initial_ub = [&](double fallback) -> double
            {
                auto finite = [](double v) -> bool
                { return std::isfinite(v) && v < dp::kInf * 0.5; };
                double i0 = fwd.profile_step2_ub;
                double i1 = fwd.profile_exact_candidate_ub;
                double i2 = fwd.profile_beam_candidate_ub;
                double i3 = fwd.profile_beam_plus_candidate_ub;
                double i4 = std::min(i1, i3);
                double best_any = fallback;
                if (finite(i0)) best_any = std::min(best_any, i0);
                if (finite(i1)) best_any = std::min(best_any, i1);
                if (finite(i2)) best_any = std::min(best_any, i2);
                if (finite(i3)) best_any = std::min(best_any, i3);

                auto pick = [&](const std::string &src) -> double
                {
                    if (src == "i0" || src == "quick") return i0;
                    if (src == "i1" || src == "exact_step3") return i1;
                    if (src == "i2" || src == "beam") return i2;
                    if (src == "i3" || src == "beam_plus") return i3;
                    if (src == "i4" || src == "best_step3") return i4;
                    return best_any;
                };

                double chosen = pick(exact_incumbent_source);
                if (!finite(chosen))
                {
                    if (finite(i0))
                        chosen = i0;
                    else
                        chosen = best_any;
                }
                return chosen;
            };

            ub = choose_exact_initial_ub(ub);
            const std::vector<double> *guided_rdp = guide ? &guide->rdp : nullptr;
            int guided_RW = guide ? guide->RW : 0;
            double guided_lb = guide ? guide->lb : dp::kInf;
            if (enable_suffix_completion)
                ensure_suffix_relax();

            // PLAN24: set up beam-guided exact corridor if enabled
            bool corridor_enabled = env_flag_exact("PAST_EXACT_CORRIDOR_ENABLE");
            int corridor_delta = env_int_exact("PAST_EXACT_CORRIDOR_DELTA", 0);
            std::string corridor_source = env_str_exact("PAST_EXACT_CORRIDOR_SOURCE");
            if (corridor_source.empty()) corridor_source = "profile_beam";
            dp::ExactCorridor corridor;
            if (corridor_enabled && corridor_source == "profile_beam" &&
                !fwd.profile_beam_chosen_counts.empty())
            {
                int B = static_cast<int>(fwd.profile_beam_chosen_counts.size());
                int K = static_cast<int>(lens.size());
                corridor.enabled = true;
                corridor.delta = corridor_delta;
                corridor.prefix_work.assign(B + 1, 0);
                corridor.prefix_counts.assign(B + 1, std::vector<int>(K, 0));
                // Use block_order if available, otherwise assume identity
                const std::vector<int> &order = fwd.profile_beam_block_order;
                bool use_order = (static_cast<int>(order.size()) == B);
                for (int pos = 0; pos < B; ++pos)
                {
                    int bi = use_order ? order[pos] : pos;
                    corridor.prefix_work[pos + 1] = corridor.prefix_work[pos];
                    // merged block lengths are not stored in fwd; approximate using chosen_counts * lengths
                    for (int j = 0; j < K; ++j)
                        corridor.prefix_work[pos + 1] += fwd.profile_beam_chosen_counts[bi][j] * lens[j];
                    for (int j = 0; j < K; ++j)
                        corridor.prefix_counts[pos + 1][j] = corridor.prefix_counts[pos][j] + fwd.profile_beam_chosen_counts[bi][j];
                }
                dp::set_exact_corridor(corridor);
            }
            else
            {
                dp::clear_exact_corridor();
            }

            double exact = dp::solve_sparse_exact_multiset_dp(
                lens, tots, prefix, T, spaces, ub, exact_budget,
                guided_rdp, guided_RW, guided_lb,
                enable_suffix_completion ? &suffix_relax.rdp : nullptr,
                enable_suffix_completion ? suffix_relax.RW : 0,
                enable_suffix_completion ? suffix_relax.rw_scale : 1);
            exact_diag_row = dp::consume_last_exact_dp_diagnostics();
            if (!exact_diag_row.variant.empty())
                exact_variant_active = exact_diag_row.variant;
            if (exact < dp::kInf)
            {
                if (exact < ub)
                    ub = exact;
                lb = ub;
            }
            if (!gap_closed())
            {
                double dense_budget = remaining_sec();
                if (dense_budget <= 0.0)
                {
                    timed_out = true;
                    t_exact = Dur(Clock::now() - t0).count();
                    goto done;
                }
                exact = dp::solve_exact_multiset_dp(lens, tots, prefix, T, spaces, ub, dense_budget);
                dp::ExactDPDiagnostics dense_diag = dp::consume_last_exact_dp_diagnostics();
                if (should_replace_exact_diag(exact_diag_row, dense_diag))
                    exact_diag_row = dense_diag;
                if (!dense_diag.variant.empty())
                    exact_variant_active = dense_diag.variant;
                if (exact < dp::kInf)
                {
                    if (exact < ub)
                        ub = exact;
                    lb = ub;
                }
            }

            // PLAN25/PLAN26: local corridor DP
            if (env_flag_exact("PAST_BEAM_CORRIDOR_LOCAL_DP"))
            {
                auto t0_local = Clock::now();
                int local_delta = env_int_exact("PAST_BEAM_CORRIDOR_LOCAL_DELTA", 1);
                double local_budget = env_int_exact("PAST_BEAM_CORRIDOR_LOCAL_TIME_LIMIT", 300);
                // Use exact merged blocks from pack_recovered_blocks
                std::vector<dp::Segment> merged_seg = fwd.merged_blocks;
                if (merged_seg.empty() && !fwd.block_profile.empty())
                {
                    // Fallback reconstruction if merged_blocks not populated
                    merged_seg.push_back(fwd.block_profile[0]);
                    for (size_t i = 1; i < fwd.block_profile.size(); ++i)
                    {
                        dp::Segment &last = merged_seg.back();
                        if (fwd.block_profile[i].start <= last.start + last.length)
                        {
                            int new_end = std::max(last.start + last.length,
                                                   fwd.block_profile[i].start + fwd.block_profile[i].length);
                            last.length = new_end - last.start;
                        }
                        else
                        {
                            merged_seg.push_back(fwd.block_profile[i]);
                        }
                    }
                }
                double local_ub = dp::beam_corridor_local_dp(
                    lens, tots, prefix, T, spaces, merged_seg,
                    fwd.profile_beam_chosen_counts,
                    fwd.profile_beam_block_order,
                    ub, local_budget, local_delta, local_corridor_diag);
                if (local_ub < ub)
                {
                    ub = local_ub;
                    // Do NOT set lb=ub; local corridor is exact only inside the corridor
                }
                local_corridor_diag.time_sec = Dur(Clock::now() - t0_local).count();
            }

            t_exact = Dur(Clock::now() - t0).count();
            step_reached = "exact";
            winner_detail = "exact";
            dp::clear_exact_corridor();
        }

    done:
        // PLAN32C: anytime fallback on timeout — only use models consistent with current LB
        if (env_flag_exact("PAST_ANYTIME_RETURN_ON_TIMEOUT") &&
            timed_out && !(ub < dp::kInf * 0.5))
        {
            // Prefer parallel UB only if opted in AND model-consistent with LB
            bool parallel_opt_in = env_flag_exact("PAST_ANYTIME_PARALLEL_UB_OPT_IN");
            if (parallel_initial_ub_valid && parallel_opt_in)
            {
                ub = parallel_initial_ub;
                winner_detail = "anytime_parallel_fallback";
                step_reached = "anytime_parallel_fallback";
                parallel_initial_ub_used_on_timeout = 1;
                initial_ub_model_note = "parallel_machines_used_on_timeout";
            }
            else if (anytime_initial_ub_valid)
            {
                ub = anytime_initial_ub;
                winner_detail = "anytime_fallback";
                step_reached = "anytime_fallback";
                anytime_ub_used_on_timeout = 1;
            }
        }

        // PLAN32C: LB-consistency guard — reject any UB that violates known LB
        if (ub < dp::kInf * 0.5 && lb > 0 && ub < lb - 1.0)
        {
            initial_ub_lb_consistent = 0;
            initial_ub_rejected_reason = "UB_below_LB_ub_" + std::to_string(static_cast<long long>(ub))
                                       + "_lb_" + std::to_string(static_cast<long long>(lb));
            ub = dp::kInf;
            if (winner_detail.find("anytime") != std::string::npos)
                winner_detail = "anytime_rejected_ub_below_lb";
        }
        bool feasible = (ub < dp::kInf * 0.5);
        bool proven_optimal = feasible && gap_closed();
        double elapsed = Dur(Clock::now() - t0_total).count();
        double gap_pct = (lb > 0 && feasible) ? 100.0 * (ub - lb) / lb : 0.0;
        if (winner_detail == "none")
            winner_detail = step_reached;

        std::ostringstream row;
            int step1_decided = 0;
            int step2_decided = 0;
            int step3_decided = 0;
            int step4_decided = 0;
            if (winner_detail == "exact" || step_reached == "exact")
            {
                step4_decided = 1;
            }
            else if (fwd.pack_method == "profile_repair_beam")
            {
                step3_decided = 1;
            }
            else if (fwd.pack_method == "ffd" ||
                     fwd.pack_method == "ffd_count" ||
                     fwd.pack_method == "bfd" ||
                     fwd.pack_method == "ffi" ||
                     fwd.pack_method == "bfi" ||
                     fwd.pack_method == "random_ff" ||
                     fwd.pack_method == "random_bf")
            {
                step2_decided = 1;
            }
            else if (fwd.pack_method == "profile_realization_dp_exact" ||
                     fwd.pack_method == "block_dp_exact")
            {
                step3_decided = 1;
            }
            else if (fwd.pack_method != "none")
            {
                step3_decided = 1;
            }
            else
            {
                step1_decided = 1;
            }

            int exact_dp_used = (step4_decided || t_exact > 0.0) ? 1 : 0;
            int exact_l2_mainline_used =
                (fwd.pack_method == "block_repair_exact_level2_archival" ||
                 fwd.pack_method == "block_repair_exact_level2")
                    ? 1
                    : 0;

            dp::ExactDPDiagnostics exact_diag =
                (exact_dp_used ? exact_diag_row : dp::ExactDPDiagnostics{});
            if (exact_dp_used && exact_diag.variant.empty())
                exact_diag.variant = exact_variant_active;

            row << instance_id << ","
                << (int)jobs.size() << ","
                << prices.size() << ","
            << std::fixed << std::setprecision(6) << (feasible ? ub : -1.0) << ","
            << std::fixed << std::setprecision(6) << (feasible ? lb : -1.0) << ","
            << std::fixed << std::setprecision(4) << gap_pct << ","
            << (feasible ? 1 : 0) << ","
            << (proven_optimal ? 1 : 0) << ","
            << (timed_out ? 1 : 0) << ","
            << std::fixed << std::setprecision(4) << elapsed << ","
            << std::fixed << std::setprecision(4) << t_spaces << ","
            << std::fixed << std::setprecision(4) << t_fwd_relax << ","
            << std::fixed << std::setprecision(4) << t_heuristic << ","
            << std::fixed << std::setprecision(4) << t_local_search << ","
            << std::fixed << std::setprecision(4) << t_bwd_relax << ","
            << std::fixed << std::setprecision(4) << t_two_class << ","
            << std::fixed << std::setprecision(4) << t_feas_profile << ","
            << std::fixed << std::setprecision(4) << t_smart_recon << ","
            << std::fixed << std::setprecision(4) << t_exact << ","
            << step_reached << ","
            << (spaces.banded ? spaces.max_gap : -1) << ","
            << std::fixed << std::setprecision(6) << lb_after_fwd << ","
            << std::fixed << std::setprecision(6) << lb_after_feas << ","
            << std::fixed << std::setprecision(6) << lb_after_fl << ","
            << std::fixed << std::setprecision(6) << lb_after_feas_profile << ","
            << std::fixed << std::setprecision(6) << ub_after_fwd << ","
            << std::fixed << std::setprecision(6) << ub_after_heur << ","
            << std::fixed << std::setprecision(6) << ub_after_ls << ","
            << std::fixed << std::setprecision(6) << ub_after_feas_profile << ","
            << states_fwd_reached << ","
            << states_fwd_expanded << ","
            << winner_detail << ","
            << fwd.block_count << ","
            << fwd.merged_block_count << ","
            << fwd.pack_solver << ","
            << fwd.pack_external_status << ","
            << fwd.pack_method << ","
            << fwd.pack_outcome << ","
            << std::fixed << std::setprecision(4) << fwd.t_pack_external << ","
            << std::fixed << std::setprecision(4) << fwd.t_pack_heuristic << ","
            << std::fixed << std::setprecision(4) << fwd.t_pack_dfs << ","
            << std::fixed << std::setprecision(4) << fwd.t_pack_block_dp << ","
            << std::fixed << std::setprecision(4) << fwd.t_pack_profile_recovery << ","
            << std::fixed << std::setprecision(4) << fwd.t_pack_merge_blocks << ","
            << std::fixed << std::setprecision(4) << fwd.t_pack_to_first_candidate << ","
            << std::fixed << std::setprecision(4) << fwd.t_pack_ffd_only << ","
            << fwd.step2_reached << ","
            << fwd.step2_produced_ub << ","
            << std::fixed << std::setprecision(4) << fwd.t_dense_spaces_or_lb << ","
            << std::fixed << std::setprecision(4) << fwd.t_dense_profile_dp << ","
            << std::fixed << std::setprecision(4) << fwd.t_dense_profile_recovery << ","
            << std::fixed << std::setprecision(4) << fwd.t_dense_block_build << ","
            << std::fixed << std::setprecision(4) << fwd.t_dense_job_materialization << ","
            << std::fixed << std::setprecision(4) << fwd.t_dense_step2_pack << ","
            << std::fixed << std::setprecision(4) << fwd.t_dense_pre_step2_total << ","
            << fwd.pack_profiles_tried << ","
            << fwd.pack_co_optimal_profiles << ","
            << std::fixed << std::setprecision(0) << fwd.block_dp_state_space << ","
            << std::fixed << std::setprecision(0) << fwd.block_dp_total_compositions << ","
            << std::fixed << std::setprecision(0) << fwd.block_dp_total_comp_estimate << ","
            << std::fixed << std::setprecision(0) << fwd.block_dp_max_comp_estimate << ","
            << std::fixed << std::setprecision(0) << fwd.block_dp_max_compositions_per_block << ","
            << fwd.block_dp_status << ","
            << fwd.block_dp_timed_out << ","
            << std::fixed << std::setprecision(6) << fwd.beam_ub_for_exact_l2 << ","
            << std::fixed << std::setprecision(6) << fwd.exact_l2_ub << ","
            << std::fixed << std::setprecision(4) << fwd.t_exact_l2 << ","
            << std::fixed << std::setprecision(0) << fwd.exact_l2_nodes << ","
            << fwd.exact_l2_closed << ","
            << fwd.exact_l2_improved_over_beam << ","
            << fwd.exact_l2_beam_optimal_in_pool << ","
            << fwd.exact_l2_status << ","
            << std::fixed << std::setprecision(0) << fwd.profile_beam_base_width << ","
            << std::fixed << std::setprecision(1) << fwd.profile_beam_avg_width << ","
            << std::fixed << std::setprecision(0) << fwd.profile_beam_max_width << ","
            << std::fixed << std::setprecision(0) << fwd.profile_beam_states_considered << ","
            << std::fixed << std::setprecision(0) << fwd.profile_beam_states_kept << ","
            << std::fixed << std::setprecision(0) << fwd.profile_beam_pruned_over << ","
            << std::fixed << std::setprecision(0) << fwd.profile_beam_pruned_suffix << ","
            << std::fixed << std::setprecision(0) << fwd.profile_beam_pruned_discrepancy << ","
            << fwd.profile_beam_discrepancy_budget << ","
            << fwd.profile_beam_discrepancy_depth << ","
            << fwd.profile_beam_status << ","
            << fwd.profile_beam_timed_out << ","
            << fwd.profile_beam_key_multi_policy << ","
            << fwd.profile_beam_key_multi_max << ","
            << std::fixed << std::setprecision(4) << fwd.profile_beam_key_multi_score_eps << ","
            << std::fixed << std::setprecision(4) << fwd.profile_beam_key_multi_diversity_eps << ","
            << fwd.profile_beam_score_policy << ","
            << std::fixed << std::setprecision(4) << fwd.profile_beam_residual_weight << ","
            << std::fixed << std::setprecision(6) << fwd.profile_beam_residual_mean_penalty << ","
            << std::fixed << std::setprecision(6) << fwd.profile_beam_residual_max_penalty << ","
            << std::fixed << std::setprecision(4) << fwd.profile_beam_late_frac << ","
            << fwd.profile_realization_hardest_first << ","
            << fwd.profile_realization_exact_suffix_prune << ","
            << std::fixed << std::setprecision(4) << fwd.t_pack_profile_beam << ","
            << std::fixed << std::setprecision(4) << fwd.t_pack_block_dp_exact << ","
            << std::fixed << std::setprecision(6) << fwd.profile_step2_ub << ","
            << std::fixed << std::setprecision(6) << fwd.profile_beam_candidate_ub << ","
            << std::fixed << std::setprecision(6) << fwd.profile_beam_plus_candidate_ub << ","
            << std::fixed << std::setprecision(6) << fwd.profile_exact_candidate_ub << ","
            << fwd.profile_beam_improved_over_step2 << ","
            << fwd.profile_exact_improved_over_step2 << ","
            << fwd.profile_incumbent_source << ","
            << std::fixed << std::setprecision(6) << fwd.profile_incumbent_ub_for_exact << ","
            << fwd.profile_selector_policy << ","
            << fwd.profile_selector_decision << ","
            << fwd.profile_selector_reason << ","
            << fwd.profile_selector_has_one << ","
            << fwd.profile_selector_contiguous << ","
            << fwd.profile_selector_multiplicity << ","
            << std::fixed << std::setprecision(6) << fwd.profile_selector_semigroup_density << ","
            << fwd.profile_selector_hard_alarm << ","
            << fwd.profile_exact_primary_fallback_to_beam << ","
            << fwd.profile_exact_primary_status_before_fallback << ","
            << fwd.profile_step3_incumbent_mode << ","
            << fwd.dense_unit_fastpath_active << ","
            << fwd.count_based_ffd_active << ","
            << fwd.dense_unit_relax_fastpath_active << ","
            << fwd.dense_unit_energy_profile_active << ","
            << fwd.dense_unit_relax_fastpath_fallback << ","
            << fwd.dense_unit_energy_profile_fallback << ","
            << fwd.dense_unit_relax_mode << ","
            << std::fixed << std::setprecision(0) << fwd.ec_generated_patterns_total << ","
            << std::fixed << std::setprecision(0) << fwd.ec_generated_patterns_max_block << ","
            << std::fixed << std::setprecision(0) << fwd.ec_retained_patterns_total << ","
            << std::fixed << std::setprecision(0) << fwd.ec_retained_patterns_max_block << ","
            << fwd.ec_retained_patterns_signature << ","
            << std::fixed << std::setprecision(4) << fwd.ec_time_completion << ","
            << std::fixed << std::setprecision(4) << fwd.ec_time_pattern_generation << ","
            << std::fixed << std::setprecision(4) << fwd.ec_time_exact_core << ","
            << std::fixed << std::setprecision(0) << fwd.ec_pruned_core_window << ","
            << std::fixed << std::setprecision(0) << fwd.ec_pruned_suffix << ","
            << std::fixed << std::setprecision(0) << fwd.ec_pruned_transition << ","
            << std::fixed << std::setprecision(0) << fwd.ec_pruned_bound << ","
            << fwd.ec_delta_used << ","
            << fwd.ec_fixed_blocks << ","
            << fwd.ec_two_phase_used << ","
            << std::fixed << std::setprecision(6) << fwd.ec_phase1_feasible_ub << ","
            << std::fixed << std::setprecision(4) << fwd.ec_time_phase1 << ","
            << exact_incumbent_source << ","
            << exact_diag.variant << ","
            << exact_diag.mode << ","
            << std::fixed << std::setprecision(6) << exact_diag.initial_ub << ","
            << std::fixed << std::setprecision(6) << exact_diag.final_ub << ","
            << std::fixed << std::setprecision(4) << exact_diag.elapsed_sec << ","
            << std::fixed << std::setprecision(0) << exact_diag.states_reached << ","
            << std::fixed << std::setprecision(0) << exact_diag.states_expanded << ","
            << std::fixed << std::setprecision(0) << exact_diag.pruned_bound << ","
            << std::fixed << std::setprecision(0) << exact_diag.pruned_relaxed << ","
            << std::fixed << std::setprecision(0) << exact_diag.pruned_completion << ","
            << std::fixed << std::setprecision(0) << exact_diag.pruned_type_aware << ","
            << std::fixed << std::setprecision(0) << exact_diag.pruned_dominance << ","
            << exact_diag.timed_out << ","
            << exact_diag.exhaustive << ","
            << exact_diag.corridor_enabled << ","
            << exact_diag.corridor_delta << ","
            << std::fixed << std::setprecision(0) << exact_diag.corridor_pruned << ","
            << exact_diag.corridor_infeasible << ","
            << (env_flag_exact("PAST_EXACT_CORRIDOR_FORCE_ENTRY") ? "1" : "0") << ","
            << env_int64_exact("PAST_EXACT_CORRIDOR_MAX_STATES", 50000000LL) << ","
            << env_int_exact("PAST_EXACT_CORRIDOR_TIME_LIMIT", 300) << ","
            << exact_diag.stop_reason << ","
            << local_corridor_diag.enabled << ","
            << local_corridor_diag.delta << ","
            << local_corridor_diag.status << ","
            << local_corridor_diag.layers << ","
            << std::fixed << std::setprecision(0) << local_corridor_diag.states_seen << ","
            << local_corridor_diag.states_kept_max << ","
            << std::fixed << std::setprecision(0) << local_corridor_diag.states_pruned << ","
            << std::fixed << std::setprecision(0) << local_corridor_diag.transitions_considered << ","
            << std::fixed << std::setprecision(0) << local_corridor_diag.transitions_kept << ","
            << std::fixed << std::setprecision(4) << local_corridor_diag.time_sec << ","
            << std::fixed << std::setprecision(6) << local_corridor_diag.best_ub << ","
            << local_corridor_diag.closed << ","
            << local_corridor_diag.stop_reason << ","
            << local_corridor_diag.memory_safe << ","
            << local_corridor_diag.beam_counts_size << ","
            << local_corridor_diag.merged_blocks << ","
            << local_corridor_diag.block_count_mismatch << ","
            << local_corridor_diag.target_offset_l1 << ","
            << local_corridor_diag.target_in_corridor << ","
            << local_corridor_diag.base_candidates_finite << ","
            << local_corridor_diag.empty_candidate_blocks << ","
            << local_corridor_diag.first_empty_layer << ","
            << local_corridor_diag.base_path_survives << ","
            << std::fixed << std::setprecision(4) << local_corridor_diag.base_path_cost << ","
            << local_corridor_diag.base_path_reject_reason << ","
            << step1_decided << ","
            << step2_decided << ","
            << step3_decided << ","
            << step4_decided << ","
            << exact_dp_used << ","
            << exact_l2_mainline_used << ","
            // PLAN28: block-realizability diagnostics
            << fwd.block_realiz_diag_active << ","
            << fwd.block_realiz_blocks_total << ","
            << fwd.block_realiz_bad_blocks << ","
            << std::fixed << std::setprecision(4) << fwd.block_realiz_bad_rate << ","
            << fwd.block_realiz_first_bad_block << ","
            << std::fixed << std::setprecision(1) << fwd.block_realiz_min_finite_patterns << ","
            << std::fixed << std::setprecision(1) << fwd.block_realiz_mean_finite_patterns << ","
            << fwd.block_realiz_base_path_survives << ","
            << fwd.block_realiz_base_reject_reason << ","
            << std::fixed << std::setprecision(4) << fwd.block_realiz_diag_time_sec << ","
            << fwd.block_realiz_diag_skipped << ","
            << fwd.block_realiz_diag_skip_reason << ","
            << fwd.block_realiz_per_block_payload << ","
            // PLAN29: block-view reconstruction diagnostics
            << fwd.block_view_policy << ","
            << fwd.block_view_original_blocks << ","
            << fwd.block_view_final_blocks << ","
            << fwd.block_view_removed_boundaries << ","
            << fwd.block_view_target_b << ","
            << fwd.block_view_price_preserve_used << ","
            << fwd.block_view_arith_adaptive_used << ","
            << fwd.block_view_selected << ","
            << fwd.block_view_eval_count << ","
            << std::fixed << std::setprecision(6) << fwd.block_view_best_ub << ","
            << std::fixed << std::setprecision(4) << fwd.block_view_time_sec << ","
            // PLAN32: anytime UB diagnostics
            << std::fixed << std::setprecision(6) << anytime_initial_ub << ","
            << anytime_initial_ub_source << ","
            << std::fixed << std::setprecision(4) << anytime_time_to_first_ub << ","
            << anytime_initial_ub_valid << ","
            << anytime_ub_used_on_timeout << ","
            // PLAN32B: parallel initial UB diagnostics
            << std::fixed << std::setprecision(6) << parallel_initial_ub << ","
            << parallel_initial_ub_valid << ","
            << parallel_initial_ub_policy << ","
            << std::fixed << std::setprecision(4) << parallel_initial_ub_time_sec << ","
            << parallel_initial_ub_machines_used << ","
            << parallel_initial_ub_failed_machines << ","
            << parallel_initial_ub_used_on_timeout << ","
            // PLAN32C: LB-consistency guard
            << initial_ub_lb_consistent << ","
            << initial_ub_rejected_reason << ","
            << initial_ub_model_note << ","
            // PLAN33: certified anytime hard-K prepass
            << cert_anytime_enabled << ","
            << cert_anytime_k_min << ","
            << std::fixed << std::setprecision(4) << cert_anytime_gap_stop_pct << ","
            << cert_anytime_triggered << ","
            << cert_anytime_stopped << ","
            << std::fixed << std::setprecision(6) << cert_anytime_initial_ub << ","
            << std::fixed << std::setprecision(6) << cert_anytime_lb << ","
            << std::fixed << std::setprecision(6) << cert_anytime_gap_pct << ","
            << cert_anytime_best_policy << ","
            << cert_anytime_finite_candidates << ","
            << std::fixed << std::setprecision(4) << cert_anytime_time_to_first_ub << ","
            << std::fixed << std::setprecision(4) << cert_anytime_time_total << ","
            << cert_anytime_polish_used << ","
            << std::fixed << std::setprecision(6) << cert_anytime_ub_before_polish << ","
            << std::fixed << std::setprecision(6) << cert_anytime_ub_after_polish << ","
            << feas_prof.block_count << ","
            << feas_prof.merged_block_count << ","
            << feas_prof.pack_solver << ","
            << feas_prof.pack_external_status << ","
            << feas_prof.pack_method << ","
            << feas_prof.pack_outcome << ","
            << std::fixed << std::setprecision(4) << feas_prof.t_pack_external << ","
            << std::fixed << std::setprecision(4) << feas_prof.t_pack_heuristic << ","
            << std::fixed << std::setprecision(4) << feas_prof.t_pack_dfs << ","
            << std::fixed << std::setprecision(4) << feas_prof.t_pack_block_dp << ","
            << std::fixed << std::setprecision(4) << feas_prof.t_pack_profile_recovery << ","
            << feas_prof.pack_profiles_tried << ","
            << feas_prof.pack_co_optimal_profiles << ","
            << std::fixed << std::setprecision(0) << feas_prof.block_dp_state_space << ","
            << std::fixed << std::setprecision(0) << feas_prof.block_dp_total_compositions << ","
            << feas_prof.block_dp_status;
        return row.str();
    }

    // ---------------------------------------------------------------------------
    // Core solver helper — called by all modes.
    // Returns one CSV data row (no header): instance_id,n_jobs,horizon,cost,...
    // ---------------------------------------------------------------------------
    std::string solve_one(
        const std::string &instance_id,
        const std::vector<double> &prices,
        const std::vector<int> &jobs,
        double time_limit,
        const std::string &machine_type = "nosby")
    {
        auto t0_total = std::chrono::steady_clock::now();

        std::map<int, int> cnt;
        for (int p : jobs)
            cnt[p]++;
        std::vector<int> lens, tots;
        for (auto &kv : cnt)
        {
            lens.push_back(kv.first);
            tots.push_back(kv.second);
        }

        auto cfg = (machine_type == "twosby") ? dp::make_paper_twosby_config()
                                              : dp::make_paper_nosby_config();
        int mg = resolve_max_gap(cfg, prices, true);
        auto spaces = dp::compute_spaces(prices, cfg, mg);
        auto prefix = dp::build_proc_prefix(prices, spaces.p_proc);
        int T = (int)prices.size();

        double ub = dp::kInf;
        double lb = 0.0;
        bool timed_out = false;
        auto gap_closed = [&]()
        { return std::fabs(ub - lb) < 0.01; };
        auto elapsed_sec = [&]()
        {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_total).count();
        };
        auto remaining_sec = [&]()
        {
            if (time_limit <= 0.0)
                return 1.0e18;
            return std::max(0.0, time_limit - elapsed_sec());
        };
        auto out_of_time = [&]()
        {
            if (time_limit > 0.0 && elapsed_sec() >= time_limit)
            {
                timed_out = true;
                return true;
            }
            return false;
        };
        auto tiny_gap_for_exact = [&]()
        {
            if (!(ub < dp::kInf * 0.5) || lb <= 0.0)
                return false;
            double rel_gap = (ub - lb) / lb;
            // If the current gap is already tiny, extra heuristic / backup work
            // is unlikely to beat simply giving the remaining budget to exact DP.
            return rel_gap <= 5e-4;
        };

        // Compute NC to decide whether to skip expensive heuristics
        int64_t NC_est = 1;
        bool use_exact_shortcut = false;
        for (std::size_t i = 0; i < lens.size(); ++i)
        {
            NC_est *= (tots[i] + 1);
            if (NC_est > 50'000)
                break;
        }
        if (NC_est < 50'000 && NC_est * (int64_t)(T + 2) <= 600'000'000LL)
            use_exact_shortcut = true;

        int64_t states_fwd_reached = 0, states_fwd_expanded = 0;
        bool enable_suffix_completion = use_suffix_completion_guidance();
        bool guided_incumbent_only = use_guided_incumbent_only();
        bool beam_incumbent_only = use_beam_incumbent_only();
        int beam_width = beam_width_setting();
        bool adaptive_enabled = adaptive_pipeline_enabled();
        int total_rw = 0;
        for (std::size_t i = 0; i < lens.size(); ++i)
            total_rw += lens[i] * tots[i];
        dp::RelaxedTableResult suffix_relax;
        dp::RelaxedDPResult partial_prof;
        dp::RelaxedDPResult feas_guide_prof;
        bool suffix_relax_ready = false;
        bool skip_generic_incumbent = false;
        auto ensure_suffix_relax = [&]()
        {
            if (suffix_relax_ready)
                return;
            suffix_relax = dp::compute_relaxed_completion_table(
                lens, total_rw, prefix, T, spaces, dp::RelaxationMode::Semigroup);
            suffix_relax_ready = true;
        };

        // --- Step 1: Forward relaxed DP with bin-packing (single pass) ---
        // Gets both LB and bin-pack UB from one DP computation.
        dp::RelaxedDPResult fwd;
        if (env_flag_exact("PAST_FWD_LB_ONLY"))
        {
            auto fwd_tab = dp::compute_relaxed_dp_table(
                lens, total_rw, prefix, T, spaces, dp::RelaxationMode::Semigroup);
            fwd.lb = fwd_tab.lb;
            fwd.rdp = std::move(fwd_tab.rdp);
            fwd.RW = fwd_tab.RW;
            fwd.bin_pack_ub = dp::kInf;
        }
        else
        {
            fwd = dp::solve_relaxed_dp_with_binpack(lens, tots, prefix, T, spaces);
        }
        lb = fwd.lb;
        if (fwd.bin_pack_ub < ub)
            ub = fwd.bin_pack_ub;
        states_fwd_reached = fwd.states_reached;
        states_fwd_expanded = fwd.states_expanded;
        if (gap_closed())
            goto done;
        if (should_adaptive_jump_to_exact(
                static_cast<int>(lens.size()), ub, fwd.pack_outcome,
                use_exact_shortcut, true, adaptive_enabled, false))
            goto exact_dp;
        if (!use_exact_shortcut && tiny_gap_for_exact())
            goto exact_dp;
        if (out_of_time())
            goto done;

        if (env_flag_exact("PAST_PARTIAL_BINPACK_STAGE"))
        {
            partial_prof = run_partial_binpack_stage(lens, tots, prefix, T, spaces);
            if (is_valid_relax_lb(partial_prof.lb) && partial_prof.lb > lb)
                lb = partial_prof.lb;
            if (partial_prof.bin_pack_ub < ub)
                ub = partial_prof.bin_pack_ub;
            if (gap_closed())
                goto done;
            if (should_adaptive_jump_to_exact(
                    static_cast<int>(lens.size()), ub, partial_prof.pack_outcome,
                    use_exact_shortcut, true, adaptive_enabled, false))
                goto exact_dp;
            if (!use_exact_shortcut && tiny_gap_for_exact())
                goto exact_dp;
            if (out_of_time())
                goto done;
        }

        skip_generic_incumbent =
            env_flag_exact("PAST_SKIP_HEUR_IF_PARTIAL_UB") &&
            partial_prof.bin_pack_ub < dp::kInf * 0.5;

        // If NC is small, skip Steps 2-5 (expensive heuristics) and go straight
        // to Step 6 (exact DP) which will solve it quickly.
        if (use_exact_shortcut)
            goto exact_dp;
        if (skip_generic_incumbent)
            goto exact_dp;

        // --- Step 2: Incumbent builder ---
        {
            double heur_ub = dp::kInf;
            if (beam_incumbent_only)
            {
                if (enable_suffix_completion)
                    ensure_suffix_relax();
                heur_ub = dp::completion_guided_beam_ub(
                    lens, tots, prefix, T, spaces,
                    suffix_relax.rdp, suffix_relax.RW, suffix_relax.rw_scale,
                    ub, beam_width, std::min(remaining_sec(), 30.0));
            }
            else if (guided_incumbent_only)
            {
                if (enable_suffix_completion)
                    ensure_suffix_relax();
                heur_ub = dp::guided_completion_ub(
                    lens, tots, prefix, T, spaces,
                    suffix_relax.rdp, suffix_relax.RW, suffix_relax.rw_scale);
            }
            else
            {
                heur_ub = dp::compute_initial_ub(lens, tots, prefix, T, spaces, 50, lb);
            }
            if (heur_ub < ub)
                ub = heur_ub;
            if (gap_closed())
                goto done;
            if (!use_exact_shortcut && tiny_gap_for_exact())
                goto exact_dp;
            if (out_of_time())
                goto done;
        }

        // --- Legacy incumbent polish: local search from SPT + LPT ---
        if (!guided_incumbent_only && !beam_incumbent_only)
        {
            std::vector<int> all_jobs;
            for (std::size_t i = 0; i < lens.size(); ++i)
                for (int j = 0; j < tots[i]; ++j)
                    all_jobs.push_back(lens[i]);

            std::vector<int> seq = all_jobs;
            std::sort(seq.begin(), seq.end());
            double spt_cost = dp::solve_fixed_sequence(seq, prefix, T, spaces);
            double ls_cost = dp::local_search_ub(seq, spt_cost, prefix, T, spaces, 3);
            if (ls_cost < ub)
                ub = ls_cost;
            if (gap_closed())
                goto done;

            std::sort(seq.begin(), seq.end(), std::greater<int>());
            double lpt_cost = dp::solve_fixed_sequence(seq, prefix, T, spaces);
            ls_cost = dp::local_search_ub(seq, lpt_cost, prefix, T, spaces, 3);
            if (ls_cost < ub)
                ub = ls_cost;
            if (gap_closed())
                goto done;
            if (!use_exact_shortcut && tiny_gap_for_exact())
                goto exact_dp;
            if (out_of_time())
                goto done;
        }

        // --- Step 4: R_feas LB (transition-feasibility filter) ---
        {
            double lb_feas = dp::solve_relaxed_dp_lb_feas(lens, tots, prefix, T, spaces);
            if (is_valid_relax_lb(lb_feas) && lb_feas > lb)
                lb = lb_feas;
            if (gap_closed())
                goto done;
            if (!use_exact_shortcut && tiny_gap_for_exact())
                goto exact_dp;
            if (out_of_time())
                goto done;
        }

        // --- Step 4.5: R_partial LB (partial count-vector tracking) ---
        // Tracks 1-2 scarcest types exactly; usually 10-30× faster than Lagrangian
        // and often produces tighter or equal bounds.
        {
            double budget = std::min(20.0, remaining_sec());
            if (budget <= 0.0)
            {
                timed_out = true;
                goto done;
            }
            double lb_par = dp::solve_relaxed_dp_lb_partial(
                lens, tots, prefix, T, spaces, {}, budget);
            if (is_valid_relax_lb(lb_par) && lb_par > lb)
                lb = lb_par;
            if (gap_closed())
                goto done;
            if (out_of_time())
                goto done;
        }

        if (env_flag_exact("PAST_FEAS_GUIDE_STAGE"))
        {
            const char *old = std::getenv("PAST_FEAS_LB_ONLY");
            setenv("PAST_FEAS_LB_ONLY", "1", 1);
            feas_guide_prof = dp::solve_relaxed_dp_lb_feas_with_binpack(lens, tots, prefix, T, spaces);
            if (old && *old)
                setenv("PAST_FEAS_LB_ONLY", old, 1);
            else
                unsetenv("PAST_FEAS_LB_ONLY");
            if (is_valid_relax_lb(feas_guide_prof.lb) && feas_guide_prof.lb > lb)
                lb = feas_guide_prof.lb;
            if (gap_closed())
                goto done;
            if (out_of_time())
                goto done;
        }

        // --- Step 5: R_feas+Lagr LB (combined bound) ---
        {
            double budget = std::min(10.0, remaining_sec());
            if (budget > 0.0)
            {
                double lb_fl = dp::solve_relaxed_dp_lb_feas_lagrangian(lens, tots, prefix, T, spaces, 50, budget);
                if (is_valid_relax_lb(lb_fl) && lb_fl > lb)
                    lb = lb_fl;
                if (gap_closed())
                    goto done;
            if (out_of_time())
                goto done;
        }
            else
            {
                timed_out = true;
                goto done;
            }
        }

        // --- Step 5.5: Smart reconstruction (count-aware search on relaxed DP table) ---
        if (!fwd.rdp.empty())
        {
            double budget = std::min(30.0, remaining_sec());
            if (budget <= 0.0)
            {
                timed_out = true;
                goto done;
            }
            double sr_cost = dp::smart_reconstruct(
                fwd.rdp, fwd.RW,
                lens, tots, prefix, T, spaces,
                ub, budget);
            if (sr_cost < ub)
                ub = sr_cost;
            if (sr_cost < dp::kInf && std::fabs(sr_cost - lb) < 0.01)
                lb = ub; // proven optimal
            if (gap_closed())
                goto done;
            if (out_of_time())
                goto done;
        }

        // --- Step 6: Sparse exact DP, guided by the forward semigroup table
        // when available (default exact stage) ---
    exact_dp:
    {
        double budget = remaining_sec();
        if (budget <= 0.0)
        {
            timed_out = true;
            goto done;
        }
        if (enable_suffix_completion)
            ensure_suffix_relax();
        const std::string exact_guide_source = env_str_exact("PAST_EXACT_GUIDE_SOURCE");
        const dp::RelaxedDPResult *guide = &fwd;
        if (exact_guide_source == "feas" && !feas_guide_prof.rdp.empty())
            guide = &feas_guide_prof;
        if (exact_guide_source == "partial" && !partial_prof.rdp.empty())
            guide = &partial_prof;
        double exact = dp::solve_sparse_exact_multiset_dp(
            lens, tots, prefix, T, spaces, ub, budget,
            guide && !guide->rdp.empty() ? &guide->rdp : nullptr,
            guide ? guide->RW : 0,
            guide ? guide->lb : dp::kInf,
            enable_suffix_completion ? &suffix_relax.rdp : nullptr,
            enable_suffix_completion ? suffix_relax.RW : 0,
            enable_suffix_completion ? suffix_relax.rw_scale : 1);
        if (exact < dp::kInf)
        {
            if (exact < ub)
                ub = exact;
            lb = ub;
        }
    }
        if (gap_closed())
            goto done;
        if (out_of_time())
            goto done;

        // --- Step 7: Dense exact multiset DP fallback ---
        {
            double budget = remaining_sec();
            if (budget <= 0.0)
            {
                timed_out = true;
                goto done;
            }
            double exact = dp::solve_exact_multiset_dp(lens, tots, prefix, T, spaces, ub, budget);
            if (exact < dp::kInf)
            {
                if (exact < ub)
                    ub = exact;
                lb = ub; // proven optimal
            }
        }

    done:
        bool feasible = (ub < dp::kInf * 0.5);
        bool proven_optimal = feasible && gap_closed();
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_total).count();
        double gap_pct = (lb > 0 && feasible) ? 100.0 * (ub - lb) / lb : 0.0;

        std::ostringstream row;
        row << instance_id << ","
            << (int)jobs.size() << ","
            << prices.size() << ","
            << std::fixed << std::setprecision(6) << (feasible ? ub : -1.0) << ","
            << std::fixed << std::setprecision(6) << (feasible ? lb : -1.0) << ","
            << std::fixed << std::setprecision(4) << gap_pct << ","
            << (feasible ? 1 : 0) << ","
            << (proven_optimal ? 1 : 0) << ","
            << (timed_out ? 1 : 0) << ","
            << std::fixed << std::setprecision(4) << elapsed << ","
            << states_fwd_reached << ","
            << states_fwd_expanded;
        return row.str();
    }

    std::vector<std::string> split_csv_fields(const std::string &line)
    {
        std::vector<std::string> out;
        std::istringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ','))
            out.push_back(tok);
        return out;
    }

    std::string solve_relaxation_profile(
        const std::string &instance_id,
        const std::vector<double> &prices,
        const std::vector<int> &jobs,
        double time_limit,
        const std::string &machine_type = "nosby")
    {
        std::map<int, int> cnt;
        for (int p : jobs)
            cnt[p]++;
        std::vector<int> lens, tots;
        for (auto &kv : cnt)
        {
            lens.push_back(kv.first);
            tots.push_back(kv.second);
        }

        auto cfg = (machine_type == "twosby") ? dp::make_paper_twosby_config()
                                              : dp::make_paper_nosby_config();
        int mg = resolve_max_gap(cfg, prices, true);
        auto spaces = dp::compute_spaces(prices, cfg, mg);
        auto prefix = dp::build_proc_prefix(prices, spaces.p_proc);
        int T = static_cast<int>(prices.size());

        int total_rw = 0;
        for (std::size_t i = 0; i < lens.size(); ++i)
            total_rw += lens[i] * tots[i];

        auto run_relax = [&](dp::RelaxationMode mode) -> std::pair<double, double>
        {
            auto t0 = Clock::now();
            double lb = dp::solve_relaxed_dp_lb(lens, total_rw, prefix, T, spaces, mode);
            double elapsed = Dur(Clock::now() - t0).count();
            return {lb, elapsed};
        };

        auto [lb_unit, t_unit] = run_relax(dp::RelaxationMode::Unit);
        auto [lb_gcd, t_gcd] = run_relax(dp::RelaxationMode::Gcd);
        auto [lb_semi, t_semi] = run_relax(dp::RelaxationMode::Semigroup);

        std::string full_row = solve_one(instance_id, prices, jobs, time_limit, machine_type);
        auto fields = split_csv_fields(full_row);
        double opt = -1.0;
        int is_optimal = 0;
        double t_opt = 0.0;
        if (fields.size() >= 10)
        {
            opt = std::stod(fields[3]);
            is_optimal = std::stoi(fields[7]);
            t_opt = std::stod(fields[9]);
        }

        std::ostringstream row;
        row << instance_id << ","
            << static_cast<int>(jobs.size()) << ","
            << prices.size() << ","
            << std::fixed << std::setprecision(6) << lb_unit << ","
            << std::fixed << std::setprecision(6) << lb_gcd << ","
            << std::fixed << std::setprecision(6) << lb_semi << ","
            << std::fixed << std::setprecision(6) << opt << ","
            << is_optimal << ","
            << std::fixed << std::setprecision(4) << t_unit << ","
            << std::fixed << std::setprecision(4) << t_gcd << ","
            << std::fixed << std::setprecision(4) << t_semi << ","
            << std::fixed << std::setprecision(4) << t_opt;
        return row.str();
    }

    std::string solve_relaxation_feas_profile(
        const std::string &instance_id,
        const std::vector<double> &prices,
        const std::vector<int> &jobs,
        const std::string &machine_type = "nosby")
    {
        std::map<int, int> cnt;
        for (int p : jobs)
            cnt[p]++;
        std::vector<int> lens, tots;
        for (auto &kv : cnt)
        {
            lens.push_back(kv.first);
            tots.push_back(kv.second);
        }

        auto cfg = (machine_type == "twosby") ? dp::make_paper_twosby_config()
                                              : dp::make_paper_nosby_config();
        int mg = resolve_max_gap(cfg, prices, true);
        auto spaces = dp::compute_spaces(prices, cfg, mg);
        auto prefix = dp::build_proc_prefix(prices, spaces.p_proc);
        int T = static_cast<int>(prices.size());

        int total_rw = 0;
        for (std::size_t i = 0; i < lens.size(); ++i)
            total_rw += lens[i] * tots[i];

        auto time_call = [&](auto &&fn) -> std::pair<double, double>
        {
            auto t0 = Clock::now();
            double val = fn();
            double elapsed = Dur(Clock::now() - t0).count();
            return {val, elapsed};
        };

        auto [lb_semi, t_semi] = time_call([&]()
        {
            return dp::solve_relaxed_dp_lb(
                lens, total_rw, prefix, T, spaces, dp::RelaxationMode::Semigroup);
        });

        auto [lb_feas, t_feas] = time_call([&]()
        {
            return dp::solve_relaxed_dp_lb_feas(lens, tots, prefix, T, spaces);
        });

        std::ostringstream row;
        row << instance_id << ","
            << static_cast<int>(jobs.size()) << ","
            << prices.size() << ","
            << std::fixed << std::setprecision(6) << lb_semi << ","
            << std::fixed << std::setprecision(6) << lb_feas << ","
            << std::fixed << std::setprecision(4) << t_semi << ","
            << std::fixed << std::setprecision(4) << t_feas;
        return row.str();
    }

    std::string solve_relaxation_hierarchy_profile(
        const std::string &instance_id,
        const std::vector<double> &prices,
        const std::vector<int> &jobs,
        double exact_time_limit,
        const std::string &machine_type = "nosby")
    {
        std::map<int, int> cnt;
        for (int p : jobs)
            cnt[p]++;
        std::vector<int> lens, tots;
        for (auto &kv : cnt)
        {
            lens.push_back(kv.first);
            tots.push_back(kv.second);
        }

        auto cfg = (machine_type == "twosby") ? dp::make_paper_twosby_config()
                                              : dp::make_paper_nosby_config();
        int mg = resolve_max_gap(cfg, prices, true);
        auto spaces = dp::compute_spaces(prices, cfg, mg);
        auto prefix = dp::build_proc_prefix(prices, spaces.p_proc);
        int T = static_cast<int>(prices.size());

        int total_rw = 0;
        for (std::size_t i = 0; i < lens.size(); ++i)
            total_rw += lens[i] * tots[i];

        auto time_call = [&](auto &&fn) -> std::pair<double, double>
        {
            auto t0 = Clock::now();
            double val = fn();
            double elapsed = Dur(Clock::now() - t0).count();
            return {val, elapsed};
        };

        auto [lb_semi, t_semi] = time_call([&]()
        {
            return dp::solve_relaxed_dp_lb(
                lens, total_rw, prefix, T, spaces, dp::RelaxationMode::Semigroup);
        });

        auto [lb_feas, t_feas] = time_call([&]()
        {
            return dp::solve_relaxed_dp_lb_feas(lens, tots, prefix, T, spaces);
        });

        auto [lb_lagr, t_lagr] = time_call([&]()
        {
            return dp::solve_relaxed_dp_lb_lagrangian(lens, tots, prefix, T, spaces, 50, 5.0);
        });

        auto [lb_fl, t_fl] = time_call([&]()
        {
            return dp::solve_relaxed_dp_lb_feas_lagrangian(lens, tots, prefix, T, spaces, 50, 5.0);
        });

        int scarce = 0;
        for (int j = 1; j < static_cast<int>(tots.size()); ++j)
            if (tots[j] < tots[scarce])
                scarce = j;

        double partial_hierarchy_time_limit = 10.0;
        if (const char *env = std::getenv("PAST_PARTIAL_LIMIT_SEC"))
        {
            try
            {
                partial_hierarchy_time_limit = std::max(0.0, std::stod(env));
            }
            catch (...)
            {
            }
        }

        auto [lb_par_plain1, t_par_plain1] = time_call([&]()
        {
            return dp::solve_relaxed_dp_lb_partial(
                lens, tots, prefix, T, spaces, std::vector<int>{scarce}, partial_hierarchy_time_limit, 1, false);
        });

        auto [lb_par_feas1, t_par_feas1] = time_call([&]()
        {
            return dp::solve_relaxed_dp_lb_partial(
                lens, tots, prefix, T, spaces, std::vector<int>{scarce}, partial_hierarchy_time_limit, 1, true);
        });

        auto [lb_par_adapt1, t_par_adapt1] = time_call([&]()
        {
            return dp::solve_relaxed_dp_lb_partial(
                lens, tots, prefix, T, spaces, {}, partial_hierarchy_time_limit, 1, true);
        });

        auto [lb_par_adapt2, t_par_adapt2] = time_call([&]()
        {
            return dp::solve_relaxed_dp_lb_partial(
                lens, tots, prefix, T, spaces, {}, partial_hierarchy_time_limit, 2, true);
        });

        auto normalize_lb = [](double v) -> double
        {
            return (std::isfinite(v) && v < dp::kInf * 0.5) ? v : -1.0;
        };

        lb_semi = normalize_lb(lb_semi);
        lb_feas = normalize_lb(lb_feas);
        lb_lagr = normalize_lb(lb_lagr);
        lb_fl = normalize_lb(lb_fl);
        lb_par_plain1 = normalize_lb(lb_par_plain1);
        lb_par_feas1 = normalize_lb(lb_par_feas1);
        lb_par_adapt1 = normalize_lb(lb_par_adapt1);
        lb_par_adapt2 = normalize_lb(lb_par_adapt2);

        double best_relax = -1.0;
        for (double v : {lb_semi, lb_feas, lb_lagr, lb_fl, lb_par_plain1, lb_par_feas1, lb_par_adapt1, lb_par_adapt2})
            best_relax = std::max(best_relax, v);

        double opt = -1.0;
        int is_optimal = 0;
        double t_opt = 0.0;
        if (exact_time_limit > 0.0)
        {
            auto t0 = Clock::now();
            double exact = dp::solve_sparse_exact_multiset_dp(
                lens, tots, prefix, T, spaces, dp::kInf, exact_time_limit);
            if (exact >= dp::kInf * 0.5)
            {
                exact = dp::solve_exact_multiset_dp(
                    lens, tots, prefix, T, spaces, dp::kInf, exact_time_limit);
            }
            t_opt = Dur(Clock::now() - t0).count();
            if (exact < dp::kInf * 0.5)
            {
                opt = exact;
                is_optimal = 1;
            }
        }

        std::ostringstream row;
        row << instance_id << ","
            << static_cast<int>(jobs.size()) << ","
            << prices.size() << ","
            << std::fixed << std::setprecision(6) << lb_semi << ","
            << std::fixed << std::setprecision(6) << lb_feas << ","
            << std::fixed << std::setprecision(6) << lb_lagr << ","
            << std::fixed << std::setprecision(6) << lb_fl << ","
            << std::fixed << std::setprecision(6) << lb_par_plain1 << ","
            << std::fixed << std::setprecision(6) << lb_par_feas1 << ","
            << std::fixed << std::setprecision(6) << lb_par_adapt1 << ","
            << std::fixed << std::setprecision(6) << lb_par_adapt2 << ","
            << std::fixed << std::setprecision(6) << best_relax << ","
            << std::fixed << std::setprecision(6) << opt << ","
            << is_optimal << ","
            << std::fixed << std::setprecision(4) << t_semi << ","
            << std::fixed << std::setprecision(4) << t_feas << ","
            << std::fixed << std::setprecision(4) << t_lagr << ","
            << std::fixed << std::setprecision(4) << t_fl << ","
            << std::fixed << std::setprecision(4) << t_par_plain1 << ","
            << std::fixed << std::setprecision(4) << t_par_feas1 << ","
            << std::fixed << std::setprecision(4) << t_par_adapt1 << ","
            << std::fixed << std::setprecision(4) << t_par_adapt2 << ","
            << std::fixed << std::setprecision(4) << t_opt;
        return row.str();
    }

    std::string solve_relaxation_pack_profile(
        const std::string &instance_id,
        const std::vector<double> &prices,
        const std::vector<int> &jobs,
        const std::string &machine_type = "nosby")
    {
        std::map<int, int> cnt;
        for (int p : jobs)
            cnt[p]++;
        std::vector<int> lens, tots;
        for (auto &kv : cnt)
        {
            lens.push_back(kv.first);
            tots.push_back(kv.second);
        }

        auto cfg = (machine_type == "twosby") ? dp::make_paper_twosby_config()
                                              : dp::make_paper_nosby_config();
        int mg = resolve_max_gap(cfg, prices, true);
        auto spaces = dp::compute_spaces(prices, cfg, mg);
        auto prefix = dp::build_proc_prefix(prices, spaces.p_proc);
        int T = static_cast<int>(prices.size());

        dp::RelaxedDPResult semi, feas, partial;
        double t_semi_total = 0.0, t_feas_total = 0.0, t_partial_total = 0.0;
        if (!env_flag_exact("PAST_RELAX_PACK_SKIP_SEMI"))
        {
            auto t0_semi = Clock::now();
            semi = dp::solve_relaxed_dp_with_binpack(lens, tots, prefix, T, spaces);
            t_semi_total = Dur(Clock::now() - t0_semi).count();
        }
        else
        {
            semi.lb = dp::kInf;
            semi.bin_pack_ub = dp::kInf;
            semi.pack_outcome = "skipped";
        }

        if (!env_flag_exact("PAST_RELAX_PACK_SKIP_FEAS"))
        {
            auto t0_feas = Clock::now();
            feas = dp::solve_relaxed_dp_lb_feas_with_binpack(lens, tots, prefix, T, spaces);
            t_feas_total = Dur(Clock::now() - t0_feas).count();
        }
        else
        {
            feas.lb = dp::kInf;
            feas.bin_pack_ub = dp::kInf;
            feas.pack_outcome = "skipped";
        }

        if (!env_flag_exact("PAST_RELAX_PACK_SKIP_PARTIAL"))
        {
            auto t0_partial = Clock::now();
            partial = run_partial_binpack_stage(lens, tots, prefix, T, spaces);
            t_partial_total = Dur(Clock::now() - t0_partial).count();
        }
        else
        {
            partial.lb = dp::kInf;
            partial.bin_pack_ub = dp::kInf;
            partial.pack_outcome = "skipped";
        }

        auto norm = [](double v) -> double
        {
            return (std::isfinite(v) && v < dp::kInf * 0.5) ? v : -1.0;
        };
        auto packable = [](const dp::RelaxedDPResult &r) -> int
        {
            return r.pack_outcome == "feasible" ? 1 : 0;
        };

        std::ostringstream row;
        row << instance_id << ","
            << static_cast<int>(jobs.size()) << ","
            << prices.size() << ","
            << std::fixed << std::setprecision(6) << norm(semi.lb) << ","
            << std::fixed << std::setprecision(6) << norm(semi.bin_pack_ub) << ","
            << packable(semi) << ","
            << semi.pack_outcome << ","
            << semi.pack_method << ","
            << semi.block_count << ","
            << semi.merged_block_count << ","
            << semi.merged_gcd_bad_count << ","
            << semi.merged_local_unreachable_count << ","
            << semi.merged_bad_caps_signature << ","
            << semi.merged_caps_signature << ","
            << std::fixed << std::setprecision(4) << t_semi_total << ","
            << std::fixed << std::setprecision(4) << semi.t_pack_heuristic << ","
            << std::fixed << std::setprecision(4) << semi.t_pack_dfs << ","
            << std::fixed << std::setprecision(4) << semi.t_pack_block_dp << ","
            << std::fixed << std::setprecision(6) << norm(feas.lb) << ","
            << std::fixed << std::setprecision(6) << norm(feas.bin_pack_ub) << ","
            << packable(feas) << ","
            << feas.pack_outcome << ","
            << feas.pack_method << ","
            << feas.block_count << ","
            << feas.merged_block_count << ","
            << feas.merged_gcd_bad_count << ","
            << feas.merged_local_unreachable_count << ","
            << feas.merged_bad_caps_signature << ","
            << feas.merged_caps_signature << ","
            << std::fixed << std::setprecision(4) << t_feas_total << ","
            << std::fixed << std::setprecision(4) << feas.t_pack_heuristic << ","
            << std::fixed << std::setprecision(4) << feas.t_pack_dfs << ","
            << std::fixed << std::setprecision(4) << feas.t_pack_block_dp << ","
            << std::fixed << std::setprecision(6) << norm(partial.lb) << ","
            << std::fixed << std::setprecision(6) << norm(partial.bin_pack_ub) << ","
            << packable(partial) << ","
            << partial.pack_outcome << ","
            << partial.pack_method << ","
            << partial.block_count << ","
            << partial.merged_block_count << ","
            << partial.merged_gcd_bad_count << ","
            << partial.merged_local_unreachable_count << ","
            << partial.merged_bad_caps_signature << ","
            << partial.merged_caps_signature << ","
            << std::fixed << std::setprecision(4) << t_partial_total << ","
            << std::fixed << std::setprecision(4) << partial.t_pack_heuristic << ","
            << std::fixed << std::setprecision(4) << partial.t_pack_dfs << ","
            << std::fixed << std::setprecision(4) << partial.t_pack_block_dp;
        return row.str();
    }

    std::string solve_completion_gap_profile(
        const std::string &instance_id,
        const std::vector<double> &prices,
        const std::vector<int> &jobs,
        const std::string &machine_type = "nosby")
    {
        std::map<int, int> cnt;
        for (int p : jobs)
            cnt[p]++;
        std::vector<int> lens, tots;
        for (auto &kv : cnt)
        {
            lens.push_back(kv.first);
            tots.push_back(kv.second);
        }

        auto cfg = (machine_type == "twosby") ? dp::make_paper_twosby_config()
                                              : dp::make_paper_nosby_config();
        int mg = resolve_max_gap(cfg, prices, true);
        auto spaces = dp::compute_spaces(prices, cfg, mg);
        auto prefix = dp::build_proc_prefix(prices, spaces.p_proc);
        int T = static_cast<int>(prices.size());
        int total_rw = 0;
        for (std::size_t i = 0; i < lens.size(); ++i)
            total_rw += lens[i] * tots[i];

        const std::string saved_mode = env_str_exact("PAST_BLOCK_REPAIR_COMPLETION_MODE");
        auto restore_mode = [&]()
        {
            if (saved_mode.empty())
                unsetenv("PAST_BLOCK_REPAIR_COMPLETION_MODE");
            else
                setenv("PAST_BLOCK_REPAIR_COMPLETION_MODE", saved_mode.c_str(), 1);
        };

        setenv("PAST_BLOCK_REPAIR_COMPLETION_MODE", "cheap", 1);
        auto t0_cheap = Clock::now();
        dp::RelaxedTableResult cheap = dp::compute_relaxed_completion_table(
            lens, total_rw, prefix, T, spaces, dp::RelaxationMode::Semigroup);
        double t_cheap = Dur(Clock::now() - t0_cheap).count();

        setenv("PAST_BLOCK_REPAIR_COMPLETION_MODE", "direct", 1);
        auto t0_direct = Clock::now();
        dp::RelaxedTableResult direct = dp::compute_relaxed_completion_table(
            lens, total_rw, prefix, T, spaces, dp::RelaxationMode::Semigroup);
        double t_direct = Dur(Clock::now() - t0_direct).count();
        restore_mode();

        std::vector<int> sample_t = {0, spaces.early, T / 4, T / 2, (3 * T) / 4, spaces.late};
        std::sort(sample_t.begin(), sample_t.end());
        sample_t.erase(std::unique(sample_t.begin(), sample_t.end()), sample_t.end());
        sample_t.erase(std::remove_if(sample_t.begin(), sample_t.end(), [&](int t)
                                      { return t < 0 || t > T; }),
                       sample_t.end());

        std::vector<int> sample_rw = {total_rw, (3 * total_rw) / 4, total_rw / 2, total_rw / 4};
        std::sort(sample_rw.begin(), sample_rw.end());
        sample_rw.erase(std::unique(sample_rw.begin(), sample_rw.end()), sample_rw.end());
        sample_rw.erase(std::remove_if(sample_rw.begin(), sample_rw.end(), [&](int rw)
                                       { return rw < 0 || rw > total_rw; }),
                        sample_rw.end());

        double sum_cont_ratio = 0.0, sum_off_ratio = 0.0;
        double max_cont_ratio = 0.0, max_off_ratio = 0.0;
        int n_cont = 0, n_off = 0;
        for (int t : sample_t)
        {
            for (int rw : sample_rw)
            {
                double cheap_cont = completion_lookup(cheap, cheap.rdp, T, t, rw);
                double direct_cont = completion_lookup(direct, direct.rdp, T, t, rw);
                if (cheap_cont > 0.0 && cheap_cont < dp::kInf && direct_cont < dp::kInf)
                {
                    double ratio = direct_cont / cheap_cont;
                    sum_cont_ratio += ratio;
                    max_cont_ratio = std::max(max_cont_ratio, ratio);
                    ++n_cont;
                }

                double cheap_off = completion_lookup(cheap, cheap.off_rdp, T, t, rw);
                double direct_off = completion_lookup(direct, direct.off_rdp, T, t, rw);
                if (cheap_off > 0.0 && cheap_off < dp::kInf && direct_off < dp::kInf)
                {
                    double ratio = direct_off / cheap_off;
                    sum_off_ratio += ratio;
                    max_off_ratio = std::max(max_off_ratio, ratio);
                    ++n_off;
                }
            }
        }

        std::ostringstream row;
        row << instance_id << ","
            << static_cast<int>(jobs.size()) << ","
            << prices.size() << ","
            << lens.size() << ","
            << total_rw << ","
            << mg << ","
            << std::fixed << std::setprecision(6) << cheap.lb << ","
            << std::fixed << std::setprecision(6) << direct.lb << ","
            << (n_cont + n_off) << ","
            << std::fixed << std::setprecision(6) << (n_cont ? (sum_cont_ratio / n_cont) : 0.0) << ","
            << std::fixed << std::setprecision(6) << max_cont_ratio << ","
            << std::fixed << std::setprecision(6) << (n_off ? (sum_off_ratio / n_off) : 0.0) << ","
            << std::fixed << std::setprecision(6) << max_off_ratio << ","
            << std::fixed << std::setprecision(4) << t_cheap << ","
            << std::fixed << std::setprecision(4) << t_direct;
        return row.str();
    }

} // namespace

int main(int argc, char **argv)
{
    std::string mode = (argc > 1 ? argv[1] : "paper-example");

    // -----------------------------------------------------------------------
    // MODE: dump-schedule
    // Reads one JSON object from stdin, solves it with full schedule tracking,
    // and outputs a JSON object containing the schedule (processing segments),
    // the proven optimal cost, and the raw prices/jobs — ready for Layer 2+3
    // verification with  scripts/verify/tec_verifier.py verify <output.json>
    //
    // Input  (stdin, one line):
    //   {"instance_id":"...","prices":[...],"jobs":[...],"machine":"twosby"}
    //
    // Output (stdout):
    //   {
    //     "instance_id": "...",
    //     "machine":     "twosby",
    //     "cost":        12288440.000000,
    //     "lb":          12288440.000000,
    //     "is_optimal":  1,
    //     "n_jobs":      150,
    //     "horizon":     1768,
    //     "jobs":        [8,8,...,10,10,...],
    //     "prices":      [1200.5,...],
    //     "schedule":    [{"start":42,"length":8},...]
    //   }
    //
    // Usage:
    //   echo '{"instance_id":"test","prices":[...],"jobs":[...],"machine":"twosby"}' \
    //     | ./stateful_compare dump-schedule
    //
    //   # Or pipe from the Python instance loader:
    //   python3 hpc/03_run_our_solver.py --dump-jsonl groups/348 \
    //     | ./stateful_compare dump-schedule > schedule_348.json
    // -----------------------------------------------------------------------
    if (mode == "dump-schedule")
    {
        std::string line;
        if (!std::getline(std::cin, line) || line.empty() || line.front() != '{')
        {
            std::cerr << "dump-schedule: expected JSON object on stdin\n";
            return 1;
        }

        auto prices  = json_parse_double_array(line, "prices");
        auto jobs    = json_parse_int_array(line, "jobs");
        auto iid     = json_parse_string(line, "instance_id");
        auto machine = json_parse_string(line, "machine");
        if (machine.empty()) machine = "nosby";
        if (prices.empty() || jobs.empty())
        {
            std::cerr << "dump-schedule: could not parse prices or jobs\n";
            return 1;
        }

        // Build lens/tots
        std::map<int,int> cnt;
        for (int p : jobs) cnt[p]++;
        std::vector<int> lens, tots;
        for (auto &kv : cnt) { lens.push_back(kv.first); tots.push_back(kv.second); }

        // Machine config
        auto cfg = (machine == "twosby") ? dp::make_paper_twosby_config()
                                         : dp::make_paper_nosby_config();
        int mg = resolve_max_gap(cfg, prices, true);
        auto spaces = dp::compute_spaces(prices, cfg, mg);
        auto prefix = dp::build_proc_prefix(prices, spaces.p_proc);
        int T = (int)prices.size();

        // Solve with schedule tracking enabled
        dp::DPParams params;
        params.track_schedule  = true;
        params.early_tie_break = true;

        auto res = dp::solve_sparse_dp_stateful(lens, tots, prefix, T, spaces, params);

        if (!res.feasible)
        {
            std::cerr << "dump-schedule: instance is infeasible\n";
            return 1;
        }

        // ── Emit JSON ──────────────────────────────────────────────────
        std::cout << "{\n";
        std::cout << "  \"instance_id\": \"" << iid << "\",\n";
        std::cout << "  \"machine\": \""     << machine << "\",\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  \"cost\": "          << res.cost  << ",\n";
        std::cout << "  \"lb\": "            << res.cost  << ",\n";
        std::cout << "  \"is_optimal\": "    << (res.timed_out ? 0 : 1) << ",\n";
        std::cout << "  \"n_jobs\": "        << (int)jobs.size()   << ",\n";
        std::cout << "  \"horizon\": "       << T                   << ",\n";

        // jobs array
        std::cout << "  \"jobs\": [";
        for (std::size_t i = 0; i < jobs.size(); ++i)
            std::cout << (i ? "," : "") << jobs[i];
        std::cout << "],\n";

        // prices array (abbreviated — full array needed by verifier)
        std::cout << "  \"prices\": [";
        for (std::size_t i = 0; i < prices.size(); ++i)
            std::cout << (i ? "," : "") << std::setprecision(4) << prices[i];
        std::cout << "],\n";

        // schedule: list of {start, length} objects
        std::cout << "  \"schedule\": [";
        for (std::size_t i = 0; i < res.segments.size(); ++i)
        {
            if (i) std::cout << ",";
            std::cout << "\n    {\"start\":" << res.segments[i].start
                      << ",\"length\":"       << res.segments[i].length << "}";
        }
        std::cout << "\n  ]\n";
        std::cout << "}\n";
        return 0;
    }

    // -----------------------------------------------------------------------
    // MODE: paper-example
    // Reproduces Example 1 from arXiv:2506.10405.  Expected cost = 342.
    // Usage: stateful_compare paper-example
    // -----------------------------------------------------------------------
    if (mode == "paper-example")
    {
        std::vector<double> prices = {9, 7, 9, 13, 3, 11, 3, 13, 6, 7, 60, 4, 10, 6, 9, 3, 14, 0, 4, 6};
        std::vector<int> jobs = {1, 2, 4};
        std::map<int, int> cnt;
        for (int p : jobs)
            cnt[p]++;
        std::vector<int> lens, tots;
        for (auto &kv : cnt)
        {
            lens.push_back(kv.first);
            tots.push_back(kv.second);
        }

        auto cfg = dp::make_paper_nosby_config();
        int mg = resolve_max_gap(cfg, prices, true);
        auto spaces = dp::compute_spaces(prices, cfg, mg);
        auto prefix_proc = dp::build_proc_prefix(prices, spaces.p_proc);

        dp::DPParams params;
        params.track_schedule = true;
        params.early_tie_break = true;

        auto res = dp::solve_sparse_dp_stateful(lens, tots, prefix_proc, (int)prices.size(), spaces, params);

        std::cout << "paper_example"
                  << " cost=" << std::fixed << std::setprecision(6) << res.cost
                  << " finish=" << res.finish_time
                  << " feasible=" << (res.feasible ? 1 : 0)
                  << " timed_out=" << (res.timed_out ? 1 : 0)
                  << " segments=";
        for (std::size_t i = 0; i < res.segments.size(); ++i)
            std::cout << (i ? ";" : "") << res.segments[i].start << ":" << res.segments[i].length;
        std::cout << "\n";
        return 0;
    }

    // -----------------------------------------------------------------------
    // MODE: synthetic
    // Generate one instance from (n, lambda, seed), solve it, print one CSV row.
    // Usage: stateful_compare synthetic [n_jobs] [lambda] [seed] [time_limit_sec]
    // -----------------------------------------------------------------------
    if (mode == "synthetic")
    {
        int n_jobs = (argc > 2 ? std::stoi(argv[2]) : 20);
        double lambda = (argc > 3 ? std::stod(argv[3]) : 1.3);
        uint64_t seed = (argc > 4 ? static_cast<uint64_t>(std::stoull(argv[4])) : 42ULL);
        double time_limit = (argc > 5 ? std::stod(argv[5]) : -1.0);

        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<int> p_dist(1, 5);
        std::vector<int> jobs(n_jobs);
        for (int &p : jobs)
            p = p_dist(rng);

        auto cfg = dp::make_paper_nosby_config();
        int startup = cfg.t_trans[cfg.off_idx][cfg.proc_idx];
        int shutdown = cfg.t_trans[cfg.proc_idx][cfg.off_idx];
        int lower = startup + std::accumulate(jobs.begin(), jobs.end(), 0) + shutdown;
        int horizon = static_cast<int>(std::ceil(lambda * lower));

        std::uniform_int_distribution<int> c_dist(1, 10);
        std::vector<double> prices(horizon);
        for (double &c : prices)
            c = static_cast<double>(c_dist(rng));

        std::string iid = "paper_nosby_n" + std::to_string(n_jobs) + "_lam" + std::to_string(lambda).substr(0, 3) + "_s" + std::to_string(seed);

        std::cout << "instance_id,n_jobs,horizon,ub,lb,gap_pct,feasible,is_optimal,timed_out,runtime_sec\n";
        std::cout << solve_one(iid, prices, jobs, time_limit) << "\n";
        return 0;
    }

    // -----------------------------------------------------------------------
    // MODE: ablation-stdin
    // Like solve-stdin, but accepts an ablation config as argv[2]:
    //   "full"         — default adaptive pipeline (banded SPACES + regime-aware bound-and-refine)
    //   "full_profile" — run the full pipeline without adaptive skipping and
    //                    without the exact-shortcut, to profile stage
    //                    contributions on a fixed policy
    //   "no_smart_recon" — full pipeline without Step 5.5
    //   "full_spaces"  — full O(h²) SPACES + full bound-and-refine
    //   "bounds_only"  — banded SPACES + heuristics + LB strengthening,
    //                    but no smart reconstruction or exact DP
    //   "bounds_profile" — same as bounds_only, but computes all LB stages
    //                      even when the forward stage already closes the gap
    //   "step1_exact_guided" — semigroup recovery + profile realization
    //                      (truncated/exact Step-3 DP modes) then semigroup-guided exact
    //   "semi_feas_exact_guided" — Step 1, then R_feas recovered-profile
    //                      packing/certification, then semigroup-guided exact
    //   "exact_only"   — banded SPACES + exact DP only (no heuristics/relaxations)
    //   "exact_guided_only" — semigroup DP only for sparse exact guidance, then exact DP
    //   "baseline"     — full O(h²) SPACES + exact DP only
    //
    // Output CSV has extra columns for per-step timing.
    // Usage: cat instances.jsonl | stateful_compare ablation-stdin <config> [time_limit_sec]
    // -----------------------------------------------------------------------
    if (mode == "ablation-stdin")
    {
        std::string ab_mode = (argc > 2 ? argv[2] : "full");
        double time_limit = (argc > 3 ? std::stod(argv[3]) : -1.0);
        AblationConfig ab;
        if (ab_mode == "full")
        {
            ab.use_banded_spaces = true;
            ab.use_heuristics = true;
            ab.use_relaxation_lb = true;
        }
        else if (ab_mode == "no_smart_recon")
        {
            ab.use_banded_spaces = true;
            ab.use_heuristics = true;
            ab.use_relaxation_lb = true;
            ab.use_smart_recon = false;
        }
        else if (ab_mode == "full_profile")
        {
            ab.use_banded_spaces = true;
            ab.use_heuristics = true;
            ab.use_relaxation_lb = true;
            ab.profile_bounds = true;
            ab.use_exact_shortcut = false;
            ab.adaptive_pipeline = false;
        }
        else if (ab_mode == "full_spaces")
        {
            ab.use_banded_spaces = false;
            ab.use_heuristics = true;
            ab.use_relaxation_lb = true;
        }
        else if (ab_mode == "step1_only")
        {
            ab.use_banded_spaces = true;
            ab.use_heuristics = false;
            ab.use_relaxation_lb = true;
            ab.use_smart_recon = false;
            ab.use_exact_shortcut = false;
            ab.use_exact_dp = false;
        }
        else if (ab_mode == "bounds_only")
        {
            ab.use_banded_spaces = true;
            ab.use_heuristics = true;
            ab.use_relaxation_lb = true;
            ab.use_smart_recon = false;
            ab.use_exact_shortcut = false;
            ab.use_exact_dp = false;
        }
        else if (ab_mode == "bounds_profile")
        {
            ab.use_banded_spaces = true;
            ab.use_heuristics = true;
            ab.use_relaxation_lb = true;
            ab.use_smart_recon = false;
            ab.use_exact_shortcut = false;
            ab.use_exact_dp = false;
            ab.profile_bounds = true;
        }
        else if (ab_mode == "step1_exact_guided")
        {
            ab.use_banded_spaces = true;
            ab.use_heuristics = false;
            ab.use_relaxation_lb = true;
            ab.use_smart_recon = false;
            ab.use_exact_shortcut = true;
            ab.use_exact_guidance = true;
        }
        else if (ab_mode == "step1_smart_exact_guided")
        {
            ab.use_banded_spaces = true;
            ab.use_heuristics = false;
            ab.use_relaxation_lb = true;
            ab.use_smart_recon = true;
            ab.use_exact_shortcut = false;
            ab.use_exact_guidance = true;
        }
        else if (ab_mode == "semi_feas_exact_guided")
        {
            ab.use_banded_spaces = true;
            ab.use_heuristics = false;
            ab.use_relaxation_lb = true;
            ab.use_smart_recon = false;
            ab.use_exact_shortcut = false;
            ab.use_exact_guidance = true;
            ab.use_feas_profile_pack = true;
        }
        else if (ab_mode == "semi_feas_smart_exact_guided")
        {
            ab.use_banded_spaces = true;
            ab.use_heuristics = false;
            ab.use_relaxation_lb = true;
            ab.use_smart_recon = true;
            ab.use_exact_shortcut = false;
            ab.use_exact_guidance = true;
            ab.use_feas_profile_pack = true;
        }
        else if (ab_mode == "exact_only")
        {
            ab.use_banded_spaces = true;
            ab.use_heuristics = false;
            ab.use_relaxation_lb = false;
        }
        else if (ab_mode == "exact_guided_only")
        {
            ab.use_banded_spaces = true;
            ab.use_heuristics = false;
            ab.use_relaxation_lb = false;
            ab.use_smart_recon = false;
            ab.use_exact_guidance = true;
        }
        else if (ab_mode == "baseline")
        {
            ab.use_banded_spaces = false;
            ab.use_heuristics = false;
            ab.use_relaxation_lb = false;
        }
        else
        {
            std::cerr << "Unknown ablation config: " << ab_mode << "\n";
            return 1;
        }

        std::cout << "instance_id,n_jobs,horizon,ub,lb,gap_pct,feasible,is_optimal,"
                  << "timed_out,runtime_sec,t_spaces,t_fwd_relax,t_heuristic,"
                  << "t_local_search,t_r_feas,t_r_feas_lagr,t_feas_profile,t_smart_recon,t_exact,"
                  << "step_reached,max_gap,lb_after_fwd,lb_after_feas,lb_after_fl,lb_after_feas_profile,ub_after_fwd,ub_after_heur,ub_after_ls,ub_after_feas_profile,states_fwd_reached,states_fwd_expanded,"
                  << "winner_detail,fwd_block_count,fwd_merged_block_count,fwd_pack_solver,fwd_pack_external_status,fwd_pack_method,fwd_pack_outcome,"
                  << "t_fwd_pack_external,t_fwd_pack_heuristic,t_fwd_pack_dfs,t_fwd_pack_block_dp,t_fwd_pack_profile_recovery,t_fwd_pack_merge_blocks,t_fwd_pack_to_first_candidate,t_fwd_pack_ffd_only,fwd_step2_reached,fwd_step2_produced_ub,t_dense_spaces_or_lb,t_dense_profile_dp,t_dense_profile_recovery,t_dense_block_build,t_dense_job_materialization,t_dense_step2_pack,t_dense_pre_step2_total,fwd_pack_profiles_tried,fwd_pack_co_optimal_profiles,fwd_block_dp_state_space,fwd_block_dp_total_compositions,fwd_block_dp_total_comp_estimate,fwd_block_dp_max_comp_estimate,fwd_block_dp_max_compositions_per_block,fwd_block_dp_status,fwd_block_dp_timed_out,"
                  << "fwd_beam_ub_for_exact_l2,fwd_exact_l2_ub,fwd_exact_l2_time,fwd_exact_l2_nodes,fwd_exact_l2_closed,fwd_exact_l2_improved_over_beam,fwd_exact_l2_beam_optimal_in_pool,fwd_exact_l2_status,"
                  << "fwd_profile_beam_base_width,fwd_profile_beam_avg_width,fwd_profile_beam_max_width,fwd_profile_beam_states_considered,fwd_profile_beam_states_kept,fwd_profile_beam_pruned_over,fwd_profile_beam_pruned_suffix,fwd_profile_beam_pruned_discrepancy,fwd_profile_beam_discrepancy_budget,fwd_profile_beam_discrepancy_depth,"
                  << "fwd_profile_beam_status,fwd_profile_beam_timed_out,fwd_profile_beam_key_multi_policy,fwd_profile_beam_key_multi_max,fwd_profile_beam_key_multi_score_eps,fwd_profile_beam_key_multi_diversity_eps,fwd_profile_beam_score_policy,fwd_profile_beam_residual_weight,fwd_profile_beam_residual_mean_penalty,fwd_profile_beam_residual_max_penalty,fwd_profile_beam_late_frac,fwd_profile_realization_hardest_first,fwd_profile_realization_exact_suffix_prune,fwd_t_pack_profile_beam,fwd_t_pack_block_dp_exact,fwd_profile_step2_ub,fwd_profile_beam_candidate_ub,fwd_profile_beam_plus_candidate_ub,fwd_profile_exact_candidate_ub,fwd_profile_beam_improved_over_step2,fwd_profile_exact_improved_over_step2,fwd_profile_incumbent_source,fwd_profile_incumbent_ub_for_exact,fwd_profile_selector_policy,fwd_profile_selector_decision,fwd_profile_selector_reason,fwd_profile_selector_has_one,fwd_profile_selector_contiguous,fwd_profile_selector_multiplicity,fwd_profile_selector_semigroup_density,fwd_profile_selector_hard_alarm,fwd_profile_exact_primary_fallback_to_beam,fwd_profile_exact_primary_status_before_fallback,fwd_profile_step3_incumbent_mode,fwd_dense_unit_fastpath_active,fwd_count_based_ffd_active,fwd_dense_unit_relax_fastpath_active,fwd_dense_unit_energy_profile_active,fwd_dense_unit_relax_fastpath_fallback,fwd_dense_unit_energy_profile_fallback,fwd_dense_unit_relax_mode,"
                  << "fwd_ec_generated_patterns_total,fwd_ec_generated_patterns_max_block,fwd_ec_retained_patterns_total,fwd_ec_retained_patterns_max_block,fwd_ec_retained_patterns_signature,fwd_ec_time_completion,fwd_ec_time_pattern_generation,fwd_ec_time_exact_core,fwd_ec_pruned_core_window,fwd_ec_pruned_suffix,fwd_ec_pruned_transition,fwd_ec_pruned_bound,fwd_ec_delta_used,fwd_ec_fixed_blocks,fwd_ec_two_phase_used,fwd_ec_phase1_feasible_ub,fwd_ec_time_phase1,"
                   << "exact_incumbent_source,exact_diag_variant,exact_diag_mode,exact_diag_initial_ub,exact_diag_final_ub,exact_diag_elapsed,exact_diag_states_reached,exact_diag_states_expanded,exact_diag_pruned_bound,exact_diag_pruned_relaxed,exact_diag_pruned_completion,exact_diag_pruned_type_aware,exact_diag_pruned_dominance,exact_diag_timed_out,exact_diag_exhaustive,exact_diag_corridor_enabled,exact_diag_corridor_delta,exact_diag_corridor_pruned,exact_diag_corridor_infeasible,corridor_force_entry,corridor_max_states,corridor_time_limit,stop_reason,"
                   << "local_corridor_enabled,local_corridor_delta,local_corridor_status,local_corridor_layers,local_corridor_states_seen,local_corridor_states_kept_max,local_corridor_states_pruned,local_corridor_transitions_considered,local_corridor_transitions_kept,local_corridor_time_sec,local_corridor_best_ub,local_corridor_closed,local_corridor_stop_reason,local_corridor_memory_safe,"
                   << "local_corridor_beam_counts_size,local_corridor_merged_blocks,local_corridor_block_count_mismatch,local_corridor_target_offset_l1,local_corridor_target_in_corridor,local_corridor_base_candidates_finite,local_corridor_empty_candidate_blocks,local_corridor_first_empty_layer,local_corridor_base_path_survives,local_corridor_base_path_cost,local_corridor_base_path_reject_reason,"
                   << "diag_step1_decided,diag_step2_decided,diag_step3_decided,diag_step4_decided,diag_exact_dp_used,diag_exact_l2_mainline_used,"
                   // PLAN28: block-realizability diagnostics
                   << "block_realiz_diag_active,block_realiz_blocks_total,block_realiz_bad_blocks,block_realiz_bad_rate,block_realiz_first_bad_block,block_realiz_min_finite_patterns,block_realiz_mean_finite_patterns,block_realiz_base_path_survives,block_realiz_base_reject_reason,block_realiz_diag_time_sec,block_realiz_diag_skipped,block_realiz_diag_skip_reason,block_realiz_per_block_payload,"
                   // PLAN29: block-view reconstruction diagnostics
                   << "block_view_policy,block_view_original_blocks,block_view_final_blocks,block_view_removed_boundaries,block_view_target_b,block_view_price_preserve_used,block_view_arith_adaptive_used,block_view_selected,block_view_eval_count,block_view_best_ub,block_view_time_sec,"
                    // PLAN32: anytime UB diagnostics
                    << "anytime_initial_ub,anytime_initial_ub_source,anytime_time_to_first_ub,anytime_initial_ub_valid,anytime_ub_used_on_timeout,"
                    // PLAN32B: parallel initial UB diagnostics
                    << "parallel_initial_ub,parallel_initial_ub_valid,parallel_initial_ub_policy,parallel_initial_ub_time_sec,parallel_initial_ub_machines_used,parallel_initial_ub_failed_machines,parallel_initial_ub_used_on_timeout,"
                    // PLAN32C: LB-consistency guard
                    << "initial_ub_lb_consistent,initial_ub_rejected_reason,initial_ub_model_note,"
                    // PLAN33: certified anytime hard-K prepass
                    << "cert_anytime_enabled,cert_anytime_k_min,cert_anytime_gap_stop_pct,cert_anytime_triggered,cert_anytime_stopped,cert_anytime_initial_ub,cert_anytime_lb,cert_anytime_gap_pct,cert_anytime_best_policy,cert_anytime_finite_candidates,cert_anytime_time_to_first_ub,cert_anytime_time_total,cert_anytime_polish_used,cert_anytime_ub_before_polish,cert_anytime_ub_after_polish,"
                    << "feas_block_count,feas_merged_block_count,feas_pack_solver,feas_pack_external_status,feas_pack_method,feas_pack_outcome,"
                  << "t_feas_pack_external,t_feas_pack_heuristic,t_feas_pack_dfs,t_feas_pack_block_dp,t_feas_pack_profile_recovery,feas_pack_profiles_tried,feas_pack_co_optimal_profiles,feas_block_dp_state_space,feas_block_dp_total_compositions,feas_block_dp_status\n";
        std::cout.flush();

        std::string line;
        while (std::getline(std::cin, line))
        {
            if (line.empty() || line.front() != '{')
                continue;
            auto prices = json_parse_double_array(line, "prices");
            auto jobs = json_parse_int_array(line, "jobs");
            auto iid = json_parse_string(line, "instance_id");
            auto machine = json_parse_string(line, "machine");
            if (machine.empty())
                machine = "nosby";
            if (prices.empty() || jobs.empty())
            {
                std::cerr << "warn: skipping malformed line: " << line.substr(0, 80) << "\n";
                continue;
            }
            std::cout << solve_one_ablation(iid, prices, jobs, machine, ab, time_limit) << "\n";
            std::cout.flush();
        }
        return 0;
    }

    // -----------------------------------------------------------------------
    // MODE: relaxation-stdin
    // Reads one JSON object per line from stdin and reports the three relaxed
    // lower bounds: unit, gcd, and semigroup, plus the full solver optimum.
    //
    // Usage: cat instances.jsonl | stateful_compare relaxation-stdin [time_limit_sec]
    // -----------------------------------------------------------------------
    if (mode == "relaxation-stdin")
    {
        double time_limit = (argc > 2 ? std::stod(argv[2]) : -1.0);

        std::cout << "instance_id,n_jobs,horizon,lb_unit,lb_gcd,lb_semi,opt,is_optimal,"
                  << "t_unit,t_gcd,t_semi,t_opt\n";
        std::cout.flush();

        std::string line;
        while (std::getline(std::cin, line))
        {
            if (line.empty() || line.front() != '{')
                continue;
            auto prices = json_parse_double_array(line, "prices");
            auto jobs = json_parse_int_array(line, "jobs");
            auto iid = json_parse_string(line, "instance_id");
            auto machine = json_parse_string(line, "machine");
            if (machine.empty())
                machine = "nosby";
            if (prices.empty() || jobs.empty())
            {
                std::cerr << "warn: skipping malformed line: " << line.substr(0, 80) << "\n";
                continue;
            }
            std::cout << solve_relaxation_profile(iid, prices, jobs, time_limit, machine) << "\n";
            std::cout.flush();
        }
        return 0;
    }

    // -----------------------------------------------------------------------
    // MODE: relax-feas-stdin
    // Reads one JSON object per line and reports only semigroup and R_feas
    // lower bounds. This is lighter than the full hierarchy and lighter than
    // packability recovery, and is intended for paper-facing structure studies.
    //
    // Usage: cat instances.jsonl | stateful_compare relax-feas-stdin
    // -----------------------------------------------------------------------
    if (mode == "relax-feas-stdin")
    {
        std::cout << "instance_id,n_jobs,horizon,lb_semi,lb_feas,t_semi,t_feas\n";
        std::cout.flush();

        std::string line;
        while (std::getline(std::cin, line))
        {
            if (line.empty() || line.front() != '{')
                continue;
            auto prices = json_parse_double_array(line, "prices");
            auto jobs = json_parse_int_array(line, "jobs");
            auto iid = json_parse_string(line, "instance_id");
            auto machine = json_parse_string(line, "machine");
            if (machine.empty())
                machine = "nosby";
            if (prices.empty() || jobs.empty())
            {
                std::cerr << "warn: skipping malformed line: " << line.substr(0, 80) << "\n";
                continue;
            }
            std::cout << solve_relaxation_feas_profile(iid, prices, jobs, machine) << "\n";
            std::cout.flush();
        }
        return 0;
    }

    // -----------------------------------------------------------------------
    // MODE: relax-hierarchy-stdin
    // Reads one JSON object per line from stdin and reports the main lower-bound
    // hierarchy used in the project, including the partial variants.
    //
    // Usage: cat instances.jsonl | stateful_compare relax-hierarchy-stdin [exact_time_limit_sec]
    // -----------------------------------------------------------------------
    if (mode == "relax-hierarchy-stdin")
    {
        double exact_time_limit = (argc > 2 ? std::stod(argv[2]) : -1.0);

        std::cout << "instance_id,n_jobs,horizon,lb_semi,lb_feas,lb_lagr,lb_feas_lagr,"
                  << "lb_par_plain1,lb_par_feas1,lb_par_adapt1,lb_par_adapt2,best_relax,opt,is_optimal,"
                  << "t_semi,t_feas,t_lagr,t_feas_lagr,t_par_plain1,t_par_feas1,t_par_adapt1,t_par_adapt2,t_opt\n";
        std::cout.flush();

        std::string line;
        while (std::getline(std::cin, line))
        {
            if (line.empty() || line.front() != '{')
                continue;
            auto prices = json_parse_double_array(line, "prices");
            auto jobs = json_parse_int_array(line, "jobs");
            auto iid = json_parse_string(line, "instance_id");
            auto machine = json_parse_string(line, "machine");
            if (machine.empty())
                machine = "nosby";
            if (prices.empty() || jobs.empty())
            {
                std::cerr << "warn: skipping malformed line: " << line.substr(0, 80) << "\n";
                continue;
            }
            std::cout << solve_relaxation_hierarchy_profile(
                             iid, prices, jobs, exact_time_limit, machine)
                      << "\n";
            std::cout.flush();
        }
        return 0;
    }

    // -----------------------------------------------------------------------
    // MODE: relax-pack-stdin
    // Reads one JSON object per line and reports, for both semigroup and
    // R_feas, the recovered relaxed block profile and whether it can be packed
    // by the Step-3 exact profile-realization DP mode.
    //
    // Usage: cat instances.jsonl | stateful_compare relax-pack-stdin
    // Recommended env for paper-quality packability studies:
    //   PAST_RELAXED_BINPACK_ALLOW_SMALL_NC=1
    //   PAST_RELAXED_BINPACK_NATIVE_FIRST=1
    // -----------------------------------------------------------------------
    if (mode == "relax-pack-stdin")
    {
        std::cout << "instance_id,n_jobs,horizon,"
                  << "lb_semi,ub_semi,semi_packable,semi_pack_outcome,semi_pack_method,semi_blocks,semi_merged_blocks,semi_gcd_bad,semi_local_bad,semi_bad_caps,semi_caps,t_semi_total,t_semi_pack_heur,t_semi_pack_dfs,t_semi_pack_blockdp,"
                  << "lb_feas,ub_feas,feas_packable,feas_pack_outcome,feas_pack_method,feas_blocks,feas_merged_blocks,feas_gcd_bad,feas_local_bad,feas_bad_caps,feas_caps,t_feas_total,t_feas_pack_heur,t_feas_pack_dfs,t_feas_pack_blockdp,"
                  << "lb_partial,ub_partial,partial_packable,partial_pack_outcome,partial_pack_method,partial_blocks,partial_merged_blocks,partial_gcd_bad,partial_local_bad,partial_bad_caps,partial_caps,t_partial_total,t_partial_pack_heur,t_partial_pack_dfs,t_partial_pack_blockdp\n";
        std::cout.flush();

        std::string line;
        while (std::getline(std::cin, line))
        {
            if (line.empty() || line.front() != '{')
                continue;
            auto prices = json_parse_double_array(line, "prices");
            auto jobs = json_parse_int_array(line, "jobs");
            auto iid = json_parse_string(line, "instance_id");
            auto machine = json_parse_string(line, "machine");
            if (machine.empty())
                machine = "nosby";
            if (prices.empty() || jobs.empty())
            {
                std::cerr << "warn: skipping malformed line: " << line.substr(0, 80) << "\n";
                continue;
            }
            std::cout << solve_relaxation_pack_profile(iid, prices, jobs, machine) << "\n";
            std::cout.flush();
        }
        return 0;
    }

    // -----------------------------------------------------------------------
    // MODE: completion-gap-stdin
    // Reads one JSON object per line and compares the cheap completion guide
    // with the direct backward semigroup completion guide on sampled states.
    //
    // Usage: cat instances.jsonl | stateful_compare completion-gap-stdin
    // -----------------------------------------------------------------------
    if (mode == "completion-gap-stdin")
    {
        std::cout << "instance_id,n_jobs,horizon,k_types,total_work,max_gap,"
                  << "cheap_lb,direct_lb,sample_count,mean_cont_ratio,max_cont_ratio,"
                  << "mean_off_ratio,max_off_ratio,t_cheap,t_direct\n";
        std::cout.flush();

        std::string line;
        while (std::getline(std::cin, line))
        {
            if (line.empty() || line.front() != '{')
                continue;
            auto prices = json_parse_double_array(line, "prices");
            auto jobs = json_parse_int_array(line, "jobs");
            auto iid = json_parse_string(line, "instance_id");
            auto machine = json_parse_string(line, "machine");
            if (machine.empty())
                machine = "nosby";
            if (prices.empty() || jobs.empty())
            {
                std::cerr << "warn: skipping malformed line: " << line.substr(0, 80) << "\n";
                continue;
            }
            std::cout << solve_completion_gap_profile(iid, prices, jobs, machine) << "\n";
            std::cout.flush();
        }
        return 0;
    }

    // -----------------------------------------------------------------------
    // MODE: solve-stdin
    // Reads one JSON object per line from stdin, solves each, emits CSV rows.
    // JSON format: {"instance_id":"...","prices":[...],"jobs":[...]}
    //
    // This is the mode used by the Python parallel launcher (run_cpp_benchmark.py).
    // Each C++ process handles its assigned chunk sequentially — no shared
    // memory between workers.
    //
    // Usage: cat instances.jsonl | stateful_compare solve-stdin [time_limit_sec]
    // -----------------------------------------------------------------------
    if (mode == "solve-stdin")
    {
        double time_limit = (argc > 2 ? std::stod(argv[2]) : -1.0);

        std::cout << "instance_id,n_jobs,horizon,ub,lb,gap_pct,feasible,is_optimal,timed_out,runtime_sec,states_fwd_reached,states_fwd_expanded\n";
        std::cout.flush();

        std::string line;
        while (std::getline(std::cin, line))
        {
            if (line.empty() || line.front() != '{')
                continue;
            auto prices = json_parse_double_array(line, "prices");
            auto jobs = json_parse_int_array(line, "jobs");
            auto iid = json_parse_string(line, "instance_id");
            auto machine = json_parse_string(line, "machine");
            if (machine.empty())
                machine = "nosby";
            if (prices.empty() || jobs.empty())
            {
                std::cerr << "warn: skipping malformed line: " << line.substr(0, 80) << "\n";
                continue;
            }
            std::cout << solve_one(iid, prices, jobs, time_limit, machine) << "\n";
            std::cout.flush();
        }
        return 0;
    }

    // -----------------------------------------------------------------------
    // MODE: benchmark  (single-process sweep — no parallelism)
    // Generates instances using the same derived-seed formula as the Python
    // generator, so results are identical to running via the Python path.
    // For parallel execution, prefer the Python launcher + solve-stdin instead.
    //
    // Usage: stateful_compare benchmark [n_csv] [lambda_csv] [seeds_csv] [time_limit_sec]
    // Defaults: n=150,170,190  lambda=1.3,1.6,1.9,2.2  seeds=42  time_limit=3600
    // -----------------------------------------------------------------------
    if (mode == "benchmark")
    {
        auto parse_ints = [](const std::string &s)
        {
            std::vector<int> out;
            std::istringstream ss(s);
            std::string tok;
            while (std::getline(ss, tok, ','))
                if (!tok.empty())
                    out.push_back(std::stoi(tok));
            return out;
        };
        auto parse_doubles = [](const std::string &s)
        {
            std::vector<double> out;
            std::istringstream ss(s);
            std::string tok;
            while (std::getline(ss, tok, ','))
                if (!tok.empty())
                    out.push_back(std::stod(tok));
            return out;
        };

        std::string n_arg = (argc > 2 ? argv[2] : "150,170,190");
        std::string lam_arg = (argc > 3 ? argv[3] : "1.3,1.6,1.9,2.2");
        std::string seed_arg = (argc > 4 ? argv[4] : "42");
        double time_limit = (argc > 5 ? std::stod(argv[5]) : 3600.0);

        auto n_values = parse_ints(n_arg);
        auto lam_values = parse_doubles(lam_arg);
        auto seeds_raw = parse_ints(seed_arg);

        auto cfg = dp::make_paper_nosby_config();
        int startup = cfg.t_trans[cfg.off_idx][cfg.proc_idx];
        int shutdown = cfg.t_trans[cfg.proc_idx][cfg.off_idx];

        std::cout << "instance_id,seed,n_jobs,lambda,horizon,ub,lb,gap_pct,feasible,is_optimal,timed_out,runtime_sec\n";

        for (int seed_i : seeds_raw)
        {
            for (int n_jobs : n_values)
            {
                for (double lam : lam_values)
                {
                    // Derived seed matches Python's generate_stateful_paper_dataset formula:
                    //   derived = seed * 1_000_003 + n * 10_007 + round(lam*10) * 101
                    uint64_t derived = static_cast<uint64_t>(seed_i) * 1000003ULL + static_cast<uint64_t>(n_jobs) * 10007ULL + static_cast<uint64_t>(static_cast<int>(std::round(lam * 10.0))) * 101ULL;
                    std::mt19937_64 rng(derived);
                    std::uniform_int_distribution<int> p_dist(1, 5);
                    std::vector<int> jobs(n_jobs);
                    for (int &p : jobs)
                        p = p_dist(rng);

                    int lower = startup + std::accumulate(jobs.begin(), jobs.end(), 0) + shutdown;
                    int horizon = static_cast<int>(std::ceil(lam * lower));

                    std::uniform_int_distribution<int> c_dist(1, 10);
                    std::vector<double> prices(horizon);
                    for (double &c : prices)
                        c = static_cast<double>(c_dist(rng));

                    std::string iid = "paper_nosby_n" + std::to_string(n_jobs) + "_lam" + std::to_string(lam).substr(0, 3) + "_s" + std::to_string(seed_i);

                    std::string row = solve_one(iid, prices, jobs, time_limit);
                    // row = "instance_id,n_jobs,horizon,ub,lb,gap_pct,feasible,is_optimal,timed_out,runtime_sec"
                    // Skip first 3 fields (iid,n_jobs,horizon) — we emit them ourselves with seed/lambda.
                    auto c1 = row.find(',');
                    auto c2 = row.find(',', c1 + 1);
                    auto c3 = row.find(',', c2 + 1);
                    std::cout << iid << ","
                              << seed_i << ","
                              << n_jobs << ","
                              << lam << ","
                              << horizon << ","
                              << row.substr(c3 + 1) // ub,lb,gap_pct,feasible,is_optimal,timed_out,runtime_sec
                              << "\n";
                    std::cout.flush();
                }
            }
        }
        return 0;
    }

    std::cerr << "Usage:\n"
              << "  stateful_compare paper-example\n"
              << "  stateful_compare synthetic [n_jobs] [lambda] [seed] [time_limit_sec]\n"
              << "  stateful_compare solve-stdin [time_limit_sec]\n"
              << "      reads one JSON line per instance from stdin:\n"
              << "      {\"instance_id\":\"...\",\"prices\":[...],\"jobs\":[...]}\n"
              << "      used by the Python parallel launcher (run_cpp_benchmark.py)\n"
              << "  stateful_compare relaxation-stdin [time_limit_sec]\n"
              << "      reads JSONL from stdin, outputs unit/gcd/semigroup relaxed LBs\n"
              << "  stateful_compare relax-feas-stdin\n"
              << "      reads JSONL from stdin, outputs semigroup and R_feas lower bounds\n"
              << "  stateful_compare relax-hierarchy-stdin [exact_time_limit_sec]\n"
              << "      reads JSONL from stdin, outputs the project lower-bound hierarchy\n"
              << "  stateful_compare relax-pack-stdin\n"
              << "      reads JSONL from stdin, outputs semigroup vs R_feas packability\n"
              << "  stateful_compare completion-gap-stdin\n"
              << "      reads JSONL from stdin, compares cheap vs direct completion guides\n"
              << "  stateful_compare ablation-stdin <config>\n"
              << "      config: full | full_profile | full_spaces | step1_only | step1_exact_guided | semi_feas_exact_guided | bounds_only | bounds_profile | exact_only | exact_guided_only | baseline\n"
              << "      reads JSONL from stdin, outputs CSV with per-step timing\n"
              << "  stateful_compare benchmark [n_csv] [lambda_csv] [seeds_csv] [time_limit_sec]\n"
              << "      single-process sweep (parallel: use Python launcher instead)\n"
              << "      e.g.:  benchmark 150,170,190 1.3,1.6,1.9,2.2 42 3600\n";
    return 1;
}

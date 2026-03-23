#include "stateful_dp_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

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
        // When both use_heuristics=false and use_relaxation_lb=false,
        // we get exact-DP-only (baseline).
    };

    using Clock = std::chrono::steady_clock;
    using Dur = std::chrono::duration<double>;

    // ---------------------------------------------------------------------------
    // Ablation-aware solver. Returns a structured CSV row with per-step timing.
    // ---------------------------------------------------------------------------
    std::string solve_one_ablation(
        const std::string &instance_id,
        const std::vector<double> &prices,
        const std::vector<int> &jobs,
        const std::string &machine_type,
        const AblationConfig &ab)
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

        double ub = dp::kInf;
        double lb = 0.0;
        auto gap_closed = [&]()
        { return std::fabs(ub - lb) < 0.01; };

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
        std::string step_reached = "none";

        // Per-step LB/UB tracking (for diagnostics)
        double lb_after_fwd = 0, lb_after_feas = 0, lb_after_fl = 0;
        double ub_after_fwd = dp::kInf, ub_after_heur = dp::kInf, ub_after_ls = dp::kInf;
        int64_t states_fwd_reached = 0, states_fwd_expanded = 0;
        double t_smart_recon = 0;
        std::string winner_detail = "none";

        // Declare fwd outside block so we can reuse rdp table in smart_reconstruct
        dp::RelaxedDPResult fwd;

        // --- Step 1: Forward relaxed DP with bin-packing (LB + UB) ---
        if (ab.use_relaxation_lb || ab.use_heuristics)
        {
            auto t0 = Clock::now();
            fwd = dp::solve_relaxed_dp_with_binpack(lens, tots, prefix, T, spaces);
            t_fwd_relax = Dur(Clock::now() - t0).count();
            if (ab.use_relaxation_lb)
                lb = fwd.lb;
            if (fwd.bin_pack_ub < ub)
                ub = fwd.bin_pack_ub;
            states_fwd_reached = fwd.states_reached;
            states_fwd_expanded = fwd.states_expanded;
            step_reached = "fwd_relax";
            lb_after_fwd = lb;
            ub_after_fwd = ub;
            if (gap_closed())
            {
                winner_detail = (fwd.pack_method != "none")
                                    ? ("fwd_relax:" + fwd.pack_method)
                                    : "fwd_relax";
                goto done;
            }
        }

        if (use_exact_shortcut)
            goto exact_dp;

        // --- Step 2: Heuristic UB (SPT/LPT/alternating/K!/random) ---
        if (ab.use_heuristics)
        {
            auto t0 = Clock::now();
            double heur_ub = dp::compute_initial_ub(lens, tots, prefix, T, spaces, 50, lb);
            t_heuristic = Dur(Clock::now() - t0).count();
            if (heur_ub < ub)
                ub = heur_ub;
            step_reached = "heuristic_ub";
            ub_after_heur = ub;
            if (gap_closed())
            {
                winner_detail = "heuristic_ub";
                goto done;
            }
        }

        // --- Step 3: Local search from SPT + LPT ---
        if (ab.use_heuristics)
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
            if (gap_closed())
            {
                winner_detail = "local_search";
                goto done;
            }
        }

        // --- Step 4: R_feas LB (transition-feasibility filter) ---
        if (ab.use_heuristics && ab.use_relaxation_lb)
        {
            auto t0 = Clock::now();
            double lb_feas = dp::solve_relaxed_dp_lb_feas(lens, tots, prefix, T, spaces);
            t_bwd_relax = Dur(Clock::now() - t0).count(); // reuse column for R_feas
            if (lb_feas > lb)
                lb = lb_feas;
            lb_after_feas = lb;
            step_reached = "r_feas";
            if (gap_closed())
            {
                winner_detail = "r_feas";
                goto done;
            }
        }

        // --- Step 5: R_feas+Lagr LB (combined bound) ---
        if (ab.use_heuristics && ab.use_relaxation_lb && (ub - lb > 0.5))
        {
            auto t0 = Clock::now();
            double lb_fl = dp::solve_relaxed_dp_lb_feas_lagrangian(
                lens, tots, prefix, T, spaces, 50, 10.0);
            t_two_class = Dur(Clock::now() - t0).count(); // reuse column for R_feas+Lagr
            if (lb_fl > lb)
                lb = lb_fl;
            lb_after_fl = lb;
            step_reached = "r_feas_lagr";
            if (gap_closed())
            {
                winner_detail = "r_feas_lagr";
                goto done;
            }
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
            if (sr_cost < dp::kInf && std::fabs(sr_cost - ub) < 0.01)
                lb = ub; // proven optimal
            step_reached = "smart_recon";
            if (gap_closed())
            {
                winner_detail = "smart_recon";
                goto done;
            }
        }

        // --- Step 6+7: Exact DP ---
        if (!ab.use_exact_dp)
            goto done;
    exact_dp:
        {
            auto t0 = Clock::now();
            double exact = dp::solve_exact_multiset_dp(lens, tots, prefix, T, spaces, ub);
            if (exact < dp::kInf)
            {
                if (exact < ub)
                    ub = exact;
                lb = ub;
            }
            if (!gap_closed())
            {
                exact = dp::solve_sparse_exact_multiset_dp(lens, tots, prefix, T, spaces, ub, 300.0);
                if (exact < dp::kInf)
                {
                    if (exact < ub)
                        ub = exact;
                    lb = ub;
                }
            }
            t_exact = Dur(Clock::now() - t0).count();
            step_reached = "exact";
            winner_detail = "exact";
        }

    done:
        bool feasible = (ub < dp::kInf * 0.5);
        bool proven_optimal = feasible && gap_closed();
        double elapsed = Dur(Clock::now() - t0_total).count();
        double gap_pct = (lb > 0 && feasible) ? 100.0 * (ub - lb) / lb : 0.0;
        if (winner_detail == "none")
            winner_detail = step_reached;

        std::ostringstream row;
        row << instance_id << ","
            << (int)jobs.size() << ","
            << prices.size() << ","
            << std::fixed << std::setprecision(6) << (feasible ? ub : -1.0) << ","
            << std::fixed << std::setprecision(6) << (feasible ? lb : -1.0) << ","
            << std::fixed << std::setprecision(4) << gap_pct << ","
            << (feasible ? 1 : 0) << ","
            << (proven_optimal ? 1 : 0) << ","
            << 0 << ","
            << std::fixed << std::setprecision(4) << elapsed << ","
            << std::fixed << std::setprecision(4) << t_spaces << ","
            << std::fixed << std::setprecision(4) << t_fwd_relax << ","
            << std::fixed << std::setprecision(4) << t_heuristic << ","
            << std::fixed << std::setprecision(4) << t_local_search << ","
            << std::fixed << std::setprecision(4) << t_bwd_relax << ","
            << std::fixed << std::setprecision(4) << t_two_class << ","
            << std::fixed << std::setprecision(4) << t_smart_recon << ","
            << std::fixed << std::setprecision(4) << t_exact << ","
            << step_reached << ","
            << (spaces.banded ? spaces.max_gap : -1) << ","
            << std::fixed << std::setprecision(6) << lb_after_fwd << ","
            << std::fixed << std::setprecision(6) << lb_after_feas << ","
            << std::fixed << std::setprecision(6) << lb_after_fl << ","
            << std::fixed << std::setprecision(6) << ub_after_fwd << ","
            << std::fixed << std::setprecision(6) << ub_after_heur << ","
            << std::fixed << std::setprecision(6) << ub_after_ls << ","
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
            << std::fixed << std::setprecision(4) << fwd.t_pack_block_dp;
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
        double /*time_limit*/,
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
        auto gap_closed = [&]()
        { return std::fabs(ub - lb) < 0.01; };

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

        // --- Step 1: Forward relaxed DP with bin-packing (single pass) ---
        // Gets both LB and bin-pack UB from one DP computation.
        auto fwd = dp::solve_relaxed_dp_with_binpack(lens, tots, prefix, T, spaces);
        lb = fwd.lb;
        if (fwd.bin_pack_ub < ub)
            ub = fwd.bin_pack_ub;
        states_fwd_reached = fwd.states_reached;
        states_fwd_expanded = fwd.states_expanded;
        if (gap_closed())
            goto done;

        // If NC is small, skip Steps 2-5 (expensive heuristics) and go straight
        // to Step 6 (exact DP) which will solve it quickly.
        if (use_exact_shortcut)
            goto exact_dp;

        // --- Step 2: Heuristic UB (SPT/LPT/alternating/K! perms/random) ---
        {
            double heur_ub = dp::compute_initial_ub(lens, tots, prefix, T, spaces, 50, lb);
            if (heur_ub < ub)
                ub = heur_ub;
            if (gap_closed())
                goto done;
        }

        // --- Step 3: Local search from SPT + LPT ---
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
        }

        // --- Step 4: R_feas LB (transition-feasibility filter) ---
        {
            double lb_feas = dp::solve_relaxed_dp_lb_feas(lens, tots, prefix, T, spaces);
            if (lb_feas > lb)
                lb = lb_feas;
            if (gap_closed())
                goto done;
        }

        // --- Step 5: R_feas+Lagr LB (combined bound) ---
        {
            double lb_fl = dp::solve_relaxed_dp_lb_feas_lagrangian(lens, tots, prefix, T, spaces, 50, 10.0);
            if (lb_fl > lb)
                lb = lb_fl;
            if (gap_closed())
                goto done;
        }

        // --- Step 5.5: Smart reconstruction (count-aware search on relaxed DP table) ---
        if (!fwd.rdp.empty())
        {
            double sr_cost = dp::smart_reconstruct(
                fwd.rdp, fwd.RW,
                lens, tots, prefix, T, spaces,
                ub, 30.0);
            if (sr_cost < ub)
                ub = sr_cost;
            if (sr_cost < dp::kInf && std::fabs(sr_cost - ub) < 0.01)
                lb = ub; // proven optimal
            if (gap_closed())
                goto done;
        }

        // --- Step 6: Exact multiset DP (for small K, e.g., K=2 or K=3) ---
        // Dense (t, c0, c1, ...) DP. Gives exact optimal = both LB and UB.
    exact_dp:
    {
        double exact = dp::solve_exact_multiset_dp(lens, tots, prefix, T, spaces, ub);
        if (exact < dp::kInf)
        {
            // Exact DP completed: exact cost is both a valid UB and LB
            if (exact < ub)
                ub = exact;
            lb = ub; // proven optimal
        }
    }
        if (gap_closed())
            goto done;

        // --- Step 7: Sparse exact DP (fallback for larger state spaces) ---
        {
            double exact = dp::solve_sparse_exact_multiset_dp(
                lens, tots, prefix, T, spaces, ub, 300.0);
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
            << 0 << ","
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
    //   "full"         — default (banded SPACES + full bound-and-refine)
    //   "no_smart_recon" — full pipeline without Step 5.5
    //   "full_spaces"  — full O(h²) SPACES + full bound-and-refine
    //   "exact_only"   — banded SPACES + exact DP only (no heuristics/relaxations)
    //   "baseline"     — full O(h²) SPACES + exact DP only
    //
    // Output CSV has extra columns for per-step timing.
    // Usage: cat instances.jsonl | stateful_compare ablation-stdin <config>
    // -----------------------------------------------------------------------
    if (mode == "ablation-stdin")
    {
        std::string ab_mode = (argc > 2 ? argv[2] : "full");
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
        else if (ab_mode == "exact_only")
        {
            ab.use_banded_spaces = true;
            ab.use_heuristics = false;
            ab.use_relaxation_lb = false;
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
                  << "t_local_search,t_r_feas,t_r_feas_lagr,t_smart_recon,t_exact,"
                  << "step_reached,max_gap,lb_after_fwd,lb_after_feas,lb_after_fl,ub_after_fwd,ub_after_heur,ub_after_ls,states_fwd_reached,states_fwd_expanded,"
                  << "winner_detail,fwd_block_count,fwd_merged_block_count,fwd_pack_solver,fwd_pack_external_status,fwd_pack_method,fwd_pack_outcome,"
                  << "t_fwd_pack_external,t_fwd_pack_heuristic,t_fwd_pack_dfs,t_fwd_pack_block_dp\n";
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
            std::cout << solve_one_ablation(iid, prices, jobs, machine, ab) << "\n";
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
              << "  stateful_compare ablation-stdin <config>\n"
              << "      config: full | full_spaces | exact_only | baseline\n"
              << "      reads JSONL from stdin, outputs CSV with per-step timing\n"
              << "  stateful_compare benchmark [n_csv] [lambda_csv] [seeds_csv] [time_limit_sec]\n"
              << "      single-process sweep (parallel: use Python launcher instead)\n"
              << "      e.g.:  benchmark 150,170,190 1.3,1.6,1.9,2.2 42 3600\n";
    return 1;
}

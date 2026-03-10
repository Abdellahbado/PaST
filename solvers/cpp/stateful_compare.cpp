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
        int mg = dp::auto_max_gap(cfg, (int)prices.size(), prices);
        auto spaces = dp::compute_spaces(prices, cfg, mg);
        auto prefix = dp::build_proc_prefix(prices, spaces.p_proc);
        int T = (int)prices.size();

        int total_rw = 0;
        for (std::size_t i = 0; i < lens.size(); ++i)
            total_rw += lens[i] * tots[i];

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

        // --- Step 1: Forward relaxed DP with bin-packing (single pass) ---
        // Gets both LB and bin-pack UB from one DP computation.
        auto fwd = dp::solve_relaxed_dp_with_binpack(lens, tots, prefix, T, spaces);
        lb = fwd.lb;
        if (fwd.bin_pack_ub < ub)
            ub = fwd.bin_pack_ub;
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

        // --- Step 4: Backward relaxed LB ---
        {
            double lb_back = dp::solve_relaxed_dp_lb_backward(lens, total_rw, prefix, T, spaces, cfg);
            if (lb_back > lb)
                lb = lb_back;
            if (gap_closed())
                goto done;
        }

        // --- Step 5: Two-class relaxed LB (if gap still open) ---
        if (ub - lb > 0.5)
        {
            double lb2 = dp::solve_relaxed_dp_lb_two_class(lens, tots, prefix, T, spaces, 2);
            if (lb2 > lb)
                lb = lb2;
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
            << std::fixed << std::setprecision(4) << elapsed;
        return row.str();
    }

} // namespace

int main(int argc, char **argv)
{
    std::string mode = (argc > 1 ? argv[1] : "paper-example");

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
        int mg = dp::auto_max_gap(cfg, (int)prices.size(), prices);
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

        std::cout << "instance_id,n_jobs,horizon,ub,lb,gap_pct,feasible,is_optimal,timed_out,runtime_sec\n";
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
              << "  stateful_compare benchmark [n_csv] [lambda_csv] [seeds_csv] [time_limit_sec]\n"
              << "      single-process sweep (parallel: use Python launcher instead)\n"
              << "      e.g.:  benchmark 150,170,190 1.3,1.6,1.9,2.2 42 3600\n";
    return 1;
}

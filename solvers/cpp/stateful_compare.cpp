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

} // namespace

int main(int argc, char **argv)
{
    std::string mode = (argc > 1 ? argv[1] : "paper-example");

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
        auto spaces = dp::compute_spaces(prices, cfg);
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
        {
            std::cout << (i ? ";" : "") << res.segments[i].start << ":" << res.segments[i].length;
        }
        std::cout << "\n";
        return 0;
    }

    if (mode == "synthetic")
    {
        int n_jobs = (argc > 2 ? std::stoi(argv[2]) : 20);
        double lambda = (argc > 3 ? std::stod(argv[3]) : 1.3);
        uint64_t seed = (argc > 4 ? static_cast<uint64_t>(std::stoull(argv[4])) : 42ULL);
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

        std::map<int, int> cnt;
        for (int p : jobs)
            cnt[p]++;
        std::vector<int> lens, tots;
        for (auto &kv : cnt)
        {
            lens.push_back(kv.first);
            tots.push_back(kv.second);
        }

        auto spaces = dp::compute_spaces(prices, cfg);
        auto prefix_proc = dp::build_proc_prefix(prices, spaces.p_proc);
        dp::DPParams params;
        params.track_schedule = false;
        params.early_tie_break = true;
        auto res = dp::solve_sparse_dp_stateful(lens, tots, prefix_proc, horizon, spaces, params);
        std::cout << "synthetic"
                  << " seed=" << seed
                  << " n=" << n_jobs
                  << " lambda=" << lambda
                  << " h=" << horizon
                  << " cost=" << std::fixed << std::setprecision(6) << res.cost
                  << " feasible=" << (res.feasible ? 1 : 0)
                  << " timed_out=" << (res.timed_out ? 1 : 0)
                  << "\n";
        return 0;
    }

    if (mode == "benchmark")
    {
        // Section 5.1 sweep: n x lambda x seeds, outputs CSV to stdout.
        // Usage: stateful_compare benchmark [n_csv] [lambda_csv] [seeds_csv] [time_limit_sec]
        // Defaults match paper: n=150,170,190  lambda=1.3,1.6,1.9,2.2  seeds=42  time_limit=3600

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

        // CSV header
        std::cout << "instance_id,seed,n_jobs,lambda,horizon,cost,feasible,is_optimal,timed_out,runtime_sec\n";

        for (int seed_i : seeds_raw)
        {
            for (int n_jobs : n_values)
            {
                for (double lam : lam_values)
                {
                    uint64_t seed = static_cast<uint64_t>(seed_i);
                    std::mt19937_64 rng(seed);
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

                    std::map<int, int> cnt;
                    for (int p : jobs)
                        cnt[p]++;
                    std::vector<int> lens, tots;
                    for (auto &kv : cnt)
                    {
                        lens.push_back(kv.first);
                        tots.push_back(kv.second);
                    }

                    auto spaces = dp::compute_spaces(prices, cfg);
                    auto prefix_proc = dp::build_proc_prefix(prices, spaces.p_proc);

                    dp::DPParams params;
                    params.track_schedule = false;
                    params.early_tie_break = true;
                    params.time_limit = time_limit;

                    auto t0 = std::chrono::steady_clock::now();
                    auto res = dp::solve_sparse_dp_stateful(lens, tots, prefix_proc, horizon, spaces, params);
                    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

                    std::string iid = "paper_nosby_n" + std::to_string(n_jobs) + "_lam" + std::to_string(lam).substr(0, 3) + "_s" + std::to_string(seed_i);

                    std::cout << iid << ","
                              << seed_i << ","
                              << n_jobs << ","
                              << lam << ","
                              << horizon << ","
                              << std::fixed << std::setprecision(6) << res.cost << ","
                              << (res.feasible ? 1 : 0) << ","
                              << ((res.feasible && !res.timed_out) ? 1 : 0) << ","
                              << (res.timed_out ? 1 : 0) << ","
                              << std::setprecision(4) << elapsed << "\n";
                    std::cout.flush();
                }
            }
        }
        return 0;
    }

    std::cerr << "Usage:\n"
              << "  stateful_compare paper-example\n"
              << "  stateful_compare synthetic [n_jobs] [lambda] [seed]\n"
              << "  stateful_compare benchmark [n_csv] [lambda_csv] [seeds_csv] [time_limit_sec]\n"
              << "    defaults: n=150,170,190  lambda=1.3,1.6,1.9,2.2  seeds=42  time_limit=3600\n"
              << "    example:  stateful_compare benchmark 150,170,190 1.3,1.6,1.9,2.2 42,7,13 60\n";
    return 1;
}
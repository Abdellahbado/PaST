#include "stateful_dp_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <climits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>

namespace dp
{

    namespace
    {

        struct PairHash
        {
            std::size_t operator()(const std::pair<int, int64_t> &x) const noexcept
            {
                uint64_t a = static_cast<uint64_t>(static_cast<uint32_t>(x.first));
                uint64_t b = static_cast<uint64_t>(x.second);
                return static_cast<std::size_t>((b * 14695981039346656037ULL) ^ (a + 0x9e3779b97f4a7c15ULL));
            }
        };

        double edge_cost(const std::vector<double> &prefix, int start, int dur, double power)
        {
            if (start < 0)
                return kInf;
            int end = start + dur;
            if (end < 0 || end > static_cast<int>(prefix.size()) - 1)
                return kInf;
            return (prefix[end] - prefix[start]) * power;
        }

        void close_same_time(std::vector<double> &dist, const std::vector<double> &zero_closure, int n_s)
        {
            std::vector<double> out = dist;
            for (int s_to = 0; s_to < n_s; ++s_to)
            {
                double best = out[s_to];
                for (int s_from = 0; s_from < n_s; ++s_from)
                {
                    double ds = dist[s_from];
                    double zc = zero_closure[s_from * n_s + s_to];
                    if (ds >= kInf || zc >= kInf)
                        continue;
                    best = std::min(best, ds + zc);
                }
                out[s_to] = best;
            }
            dist.swap(out);
        }

        int compute_earliest_proc(const MachineStateConfig &cfg)
        {
            int n_s = static_cast<int>(cfg.states.size());
            std::vector<int> dist(n_s, static_cast<int>(1e9));
            dist[cfg.off_idx] = 0;
            bool changed = true;
            while (changed)
            {
                changed = false;
                for (int s = 0; s < n_s; ++s)
                {
                    if (dist[s] >= static_cast<int>(1e9))
                        continue;
                    for (int sp = 0; sp < n_s; ++sp)
                    {
                        int dur = cfg.t_trans[s][sp];
                        if (dur < 0 || s == sp)
                            continue;
                        int nd = dist[s] + dur;
                        if (nd < dist[sp])
                        {
                            dist[sp] = nd;
                            changed = true;
                        }
                    }
                }
            }
            return dist[cfg.proc_idx] >= static_cast<int>(1e9) ? 0 : (1 + dist[cfg.proc_idx]);
        }

        int compute_latest_proc(const MachineStateConfig &cfg, int h)
        {
            int n_s = static_cast<int>(cfg.states.size());
            std::vector<int> dist(n_s, static_cast<int>(1e9));
            dist[cfg.off_idx] = 0;
            bool changed = true;
            while (changed)
            {
                changed = false;
                for (int s = 0; s < n_s; ++s)
                {
                    for (int sp = 0; sp < n_s; ++sp)
                    {
                        if (dist[sp] >= static_cast<int>(1e9) || s == sp)
                            continue;
                        int dur = cfg.t_trans[s][sp];
                        if (dur < 0)
                            continue;
                        int nd = dur + dist[sp];
                        if (nd < dist[s])
                        {
                            dist[s] = nd;
                            changed = true;
                        }
                    }
                }
            }
            int shutdown = dist[cfg.proc_idx] >= static_cast<int>(1e9) ? h + 1 : dist[cfg.proc_idx];
            return std::max(0, (h - 2) - shutdown);
        }

        int gcd_all(const std::vector<int> &vals)
        {
            int g = 0;
            for (int v : vals)
                g = std::gcd(g, v);
            return std::max(g, 1);
        }

        std::vector<int> relaxation_chunk_lengths(
            const std::vector<int> &lengths,
            RelaxationMode mode)
        {
            if (mode == RelaxationMode::Unit)
                return {1};

            if (mode == RelaxationMode::Gcd)
                return {gcd_all(lengths)};

            std::vector<int> out = lengths;
            std::sort(out.begin(), out.end());
            out.erase(std::unique(out.begin(), out.end()), out.end());
            return out;
        }

        enum class ExternalPackStatus
        {
            Disabled,
            Feasible,
            Infeasible,
            Error,
        };

        struct ExternalPackResult
        {
            ExternalPackStatus status = ExternalPackStatus::Disabled;
            std::vector<int> sequence;
            std::string solver = "disabled";
            double runtime_sec = 0.0;
        };

        ExternalPackResult exact_pack_via_ortools(
            const std::vector<int> &block_caps,
            const std::vector<int> &lengths,
            const std::vector<int> &totals)
        {
            const char *mode = std::getenv("PAST_RELAXED_BINPACK_SOLVER");
            if (!mode)
                return {};
            std::string solver_mode(mode);
            if (solver_mode != "ortools" &&
                solver_mode != "z3" &&
                solver_mode != "constraint")
            {
                ExternalPackResult out;
                out.solver = solver_mode;
                return out;
            }

            ExternalPackResult out;
            out.status = ExternalPackStatus::Error;
            out.solver = solver_mode;

            std::filesystem::path root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path();
            std::filesystem::path script = root / "scripts" /
                                           (solver_mode == "z3"
                                                ? "exact_block_pack_z3.py"
                                                : (solver_mode == "constraint"
                                                       ? "exact_block_pack_constraint.py"
                                                       : "exact_block_pack_ortools.py"));
            if (!std::filesystem::exists(script))
                return out;

            auto tmp_dir = std::filesystem::temp_directory_path();
            auto unique = std::to_string(::getpid()) + "_" +
                          std::to_string(
                              std::chrono::steady_clock::now().time_since_epoch().count());
            std::filesystem::path input = tmp_dir / ("past_binpack_in_" + unique + ".json");
            std::filesystem::path output = tmp_dir / ("past_binpack_out_" + unique + ".txt");

            {
                std::ofstream f(input);
                if (!f)
                    return out;
                auto write_vec = [&](const char *name, const std::vector<int> &v)
                {
                    f << '"' << name << "\":[";
                    for (size_t i = 0; i < v.size(); ++i)
                    {
                        if (i)
                            f << ',';
                        f << v[i];
                    }
                    f << "]";
                };
                f << "{";
                write_vec("capacities", block_caps);
                f << ",";
                write_vec("lengths", lengths);
                f << ",";
                write_vec("totals", totals);
                f << "}";
            }

            double time_limit_sec = 20.0;
            if (const char *limit_v = std::getenv("PAST_RELAXED_BINPACK_TIME_LIMIT_SEC"))
            {
                try
                {
                    time_limit_sec = std::max(0.0, std::stod(limit_v));
                }
                catch (const std::exception &)
                {
                }
            }

            std::ostringstream cmd;
            cmd << "python3 \"" << script.string() << "\" \"" << input.string()
                << "\" \"" << output.string() << "\" \"" << std::fixed << std::setprecision(3)
                << time_limit_sec << "\"";
            auto t0 = std::chrono::steady_clock::now();
            int rc = std::system(cmd.str().c_str());
            out.runtime_sec =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (rc != 0 || !std::filesystem::exists(output))
            {
                std::error_code ec;
                std::filesystem::remove(input, ec);
                std::filesystem::remove(output, ec);
                return out;
            }

            std::ifstream f(output);
            std::string status_line;
            std::getline(f, status_line);
            if (status_line == "feasible")
            {
                out.status = ExternalPackStatus::Feasible;
                std::string seq_line;
                std::getline(f, seq_line);
                std::istringstream ss(seq_line);
                std::string tok;
                while (std::getline(ss, tok, ','))
                {
                    if (!tok.empty())
                        out.sequence.push_back(std::stoi(tok));
                }
            }
            else if (status_line == "infeasible")
            {
                out.status = ExternalPackStatus::Infeasible;
            }
            else
            {
                out.status = ExternalPackStatus::Error;
            }

            std::error_code ec;
            std::filesystem::remove(input, ec);
            std::filesystem::remove(output, ec);
            return out;
        }

    } // namespace

    MachineStateConfig MachineStateConfig::paper_nosby()
    {
        return make_paper_nosby_config();
    }

    MachineStateConfig make_paper_nosby_config()
    {
        MachineStateConfig cfg;
        cfg.states = {"off", "proc", "idle"};
        cfg.off_idx = 0;
        cfg.proc_idx = 1;
        int n_s = 3;
        cfg.t_trans.assign(n_s, std::vector<int>(n_s, -1));
        cfg.p_trans.assign(n_s, std::vector<double>(n_s, kInf));
        auto set_edge = [&](int s, int sp, int dur, double power)
        {
            cfg.t_trans[s][sp] = dur;
            cfg.p_trans[s][sp] = power;
        };
        set_edge(0, 0, 1, 0.0);
        set_edge(0, 1, 2, 5.0);
        set_edge(1, 1, 1, 4.0);
        set_edge(1, 0, 1, 1.0); // Fig 2: proc->off edge labeled 1/1 (T=1, P=1)
        set_edge(1, 2, 0, 0.0);
        set_edge(2, 2, 1, 2.0);
        set_edge(2, 1, 0, 0.0);
        return cfg;
    }

    // TWOSBY (5-state) machine config from arXiv:2506.10405, Table 1.
    // States: off0(deep off, P=0), sby1(standby1, P=2), sby2(standby2, P=4),
    //         proc(processing, P=10), idle(idle, P=8).
    // Transitions: off_i <-> proc (via startup/shutdown), proc <-> idle (instant).
    // No direct idle<->off or off<->off transitions.
    MachineStateConfig make_paper_twosby_config()
    {
        MachineStateConfig cfg;
        // Indices: off0=0, sby1=1, sby2=2, proc=3, idle=4
        cfg.states = {"off0", "sby1", "sby2", "proc", "idle"};
        cfg.off_idx = 0; // machine starts/ends in full-off (P=0)
        cfg.proc_idx = 3;
        int n_s = 5;
        cfg.t_trans.assign(n_s, std::vector<int>(n_s, -1));
        cfg.p_trans.assign(n_s, std::vector<double>(n_s, kInf));
        auto set_edge = [&](int s, int sp, int dur, double power)
        {
            cfg.t_trans[s][sp] = dur;
            cfg.p_trans[s][sp] = power;
        };
        // Self-loops (holding in each state for 1 time unit)
        set_edge(0, 0, 1, 0.0);  // off0: free
        set_edge(1, 1, 1, 2.0);  // sby1: P=2
        set_edge(2, 2, 1, 4.0);  // sby2: P=4
        set_edge(3, 3, 1, 10.0); // proc: P=10
        set_edge(4, 4, 1, 8.0);  // idle: P=8
        // Startup: off_i -> proc
        set_edge(0, 3, 4, 15.0); // off0->proc: T=4, P=15
        set_edge(1, 3, 3, 13.0); // sby1->proc: T=3, P=13
        set_edge(2, 3, 2, 12.0); // sby2->proc: T=2, P=12
        // Shutdown: proc -> off_i
        set_edge(3, 0, 1, 2.0); // proc->off0: T=1, P=2
        set_edge(3, 1, 1, 2.0); // proc->sby1: T=1, P=2
        set_edge(3, 2, 1, 2.0); // proc->sby2: T=1, P=2
        // Instant: proc <-> idle
        set_edge(3, 4, 0, 0.0); // proc->idle: T=0, P=0
        set_edge(4, 3, 0, 0.0); // idle->proc: T=0, P=0
        return cfg;
    }

    std::vector<double> build_proc_prefix(const std::vector<double> &prices, double p_proc)
    {
        std::vector<double> prefix(prices.size() + 1, 0.0);
        for (std::size_t i = 0; i < prices.size(); ++i)
            prefix[i + 1] = prefix[i] + prices[i] * p_proc;
        return prefix;
    }

    double SPACESResult::gap_cost(int t_end, int t_start) const noexcept
    {
        if (t_start < t_end)
            return kInf;
        if (t_start == t_end)
            return 0.0;
        if (max_gap > 0 && (t_start - t_end) > max_gap)
            return c_end[t_end] + c_start[t_start];
        if (banded)
            return c_star[t_end * (max_gap + 1) + (t_start - t_end)];
        return c_star[t_end * (h + 1) + t_start];
    }

    // =====================================================================
    //  auto_max_gap: compute a conservative bound on useful gap length
    // =====================================================================

    int auto_max_gap(const MachineStateConfig &config, int h,
                     const std::vector<double> &prices)
    {
        int n_s = static_cast<int>(config.states.size());
        if (prices.empty() || h <= 0)
            return 0;

        std::vector<double> prefix(h + 1, 0.0);
        for (int i = 0; i < h; ++i)
            prefix[i + 1] = prefix[i] + prices[i];

        // Compute idle power (cheapest way to stay available per interval)
        double p_idle = kInf;
        for (int s = 0; s < n_s; ++s)
        {
            if (s == config.off_idx)
                continue;
            // Check if we can reach proc from s with 0-time transition
            bool can_proc = (s == config.proc_idx);
            if (!can_proc)
                for (int sp = 0; sp < n_s; ++sp)
                    if (config.t_trans[s][sp] == 0 && sp == config.proc_idx)
                        can_proc = true;
            if (!can_proc)
                continue;
            // Self-loop power for idling in this state
            if (config.t_trans[s][s] >= 0 && config.p_trans[s][s] < p_idle)
                p_idle = config.p_trans[s][s];
        }
        if (p_idle >= kInf || p_idle <= 0)
            return std::min(100, h); // fallback

        auto max_direct_transition_cost = [&](int s_from, int s_to) -> double
        {
            int dur = config.t_trans[s_from][s_to];
            double power = config.p_trans[s_from][s_to];
            if (dur <= 0 || power >= kInf || dur > h)
                return -1.0;
            double best = 0.0;
            for (int t = 0; t + dur <= h; ++t)
            {
                double window_sum = prefix[t + dur] - prefix[t];
                best = std::max(best, power * window_sum);
            }
            return best;
        };

        // Conservative upper bound on full shutdown+restart cost over all positions.
        double max_shutdown = max_direct_transition_cost(config.proc_idx, config.off_idx);
        double max_startup = max_direct_transition_cost(config.off_idx, config.proc_idx);
        double max_restart_cost = -1.0;
        if (max_shutdown >= 0.0 && max_startup >= 0.0)
            max_restart_cost = max_shutdown + max_startup;

        // Legacy coarse bound based on min/max prices.
        double max_price = *std::max_element(prices.begin(), prices.end());
        double min_price = *std::min_element(prices.begin(), prices.end());
        if (min_price <= 0)
            min_price = 0.01;
        double restart_power = 0.0;
        if (config.t_trans[config.proc_idx][config.off_idx] > 0)
            restart_power += config.p_trans[config.proc_idx][config.off_idx] *
                             config.t_trans[config.proc_idx][config.off_idx];
        if (config.t_trans[config.off_idx][config.proc_idx] > 0)
            restart_power += config.p_trans[config.off_idx][config.proc_idx] *
                             config.t_trans[config.off_idx][config.proc_idx];
        int coarse_gap = static_cast<int>(std::ceil(restart_power * max_price / (p_idle * min_price))) + 3;

        // Sharper safe bound: use the minimum actual price-window sum of length g.
        // For any gap of length g, idling/ready cost is at least p_idle * min_window_sum(g).
        // Once that exceeds a conservative upper bound on full restart cost, any longer
        // gap is safely handled by shutdown+startup.
        auto min_window_sum = [&](int len) -> double
        {
            double best = kInf;
            for (int t = 0; t + len <= h; ++t)
                best = std::min(best, prefix[t + len] - prefix[t]);
            return best;
        };

        int sharp_gap = h;
        if (max_restart_cost >= 0.0)
        {
            int lo = 1, hi = h;
            while (lo < hi)
            {
                int mid = lo + (hi - lo) / 2;
                if (p_idle * min_window_sum(mid) >= max_restart_cost)
                    hi = mid;
                else
                    lo = mid + 1;
            }
            sharp_gap = lo + 3;
        }

        int gap = std::min(coarse_gap, sharp_gap);
        return std::min(std::max(gap, 15), h);
    }

    // =====================================================================
    //  solve_fixed_sequence: O(n * h * max_gap) DP for a fixed job order
    // =====================================================================

    double solve_fixed_sequence(
        const std::vector<int> &sequence,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces)
    {
        if (sequence.empty())
            return kInf;

        int n = static_cast<int>(sequence.size());
        int eff_max_gap = spaces.banded ? spaces.max_gap : T;

        // prev[t] = min cost having scheduled jobs 0..k-1, with job k-1 ending at t
        std::vector<double> prev(T + 2, kInf);
        std::vector<double> curr(T + 2, kInf);

        // Base: first job
        int L0 = sequence[0];
        for (int t_s = spaces.early; t_s <= spaces.late; ++t_s)
        {
            double startup = spaces.c_start[t_s];
            if (startup >= kInf)
                continue;
            int t_e = t_s + L0;
            if (t_e > T || t_e > spaces.late + 1)
                continue;
            double cost = startup + (prefix_proc[t_e] - prefix_proc[t_s]);
            prev[t_e] = std::min(prev[t_e], cost);
        }

        // DP: schedule remaining jobs
        for (int k = 1; k < n; ++k)
        {
            std::fill(curr.begin(), curr.end(), kInf);
            int L = sequence[k];

            // Precompute prefix-min of (prev[t] + c_end[t]) for beyond-max-gap.
            std::vector<double> pm(T + 2, kInf);
            if (spaces.banded)
            {
                for (int t = 1; t <= T; ++t)
                {
                    pm[t] = pm[t - 1];
                    if (prev[t] < kInf && spaces.c_end[t] < kInf)
                        pm[t] = std::min(pm[t], prev[t] + spaces.c_end[t]);
                }
            }

            // Within max_gap: c_star depends on both t_prev and gap, so O(T × max_gap).
            for (int t_prev = 1; t_prev <= T; ++t_prev)
            {
                if (prev[t_prev] >= kInf)
                    continue;
                int gap_max = std::min(t_prev + eff_max_gap + 1, spaces.late + 1);
                for (int t_s = t_prev; t_s < gap_max; ++t_s)
                {
                    double gap = spaces.gap_cost(t_prev, t_s);
                    if (gap >= kInf)
                        continue;
                    int t_e = t_s + L;
                    if (t_e > T || t_e > spaces.late + 1)
                        continue;
                    double cost = prev[t_prev] + gap + (prefix_proc[t_e] - prefix_proc[t_s]);
                    curr[t_e] = std::min(curr[t_e], cost);
                }
            }

            // Beyond max_gap: shutdown + startup decomposition (O(T) total)
            if (spaces.banded)
            {
                for (int t_s = spaces.early; t_s <= spaces.late; ++t_s)
                {
                    int t_cutoff = t_s - eff_max_gap - 1;
                    if (t_cutoff < 1)
                        continue;
                    double min_pe = pm[t_cutoff];
                    if (min_pe >= kInf)
                        continue;
                    double c_s = spaces.c_start[t_s];
                    if (c_s >= kInf)
                        continue;
                    int t_e = t_s + L;
                    if (t_e > T || t_e > spaces.late + 1)
                        continue;
                    double cost = min_pe + c_s + (prefix_proc[t_e] - prefix_proc[t_s]);
                    curr[t_e] = std::min(curr[t_e], cost);
                }
            }

            std::swap(prev, curr);
        }

        // Best including shutdown
        double best = kInf;
        for (int t = 1; t <= T; ++t)
            if (prev[t] < kInf && spaces.c_end[t] < kInf)
                best = std::min(best, prev[t] + spaces.c_end[t]);
        return best;
    }

    // =====================================================================
    //  compute_initial_ub: try multiple heuristic sequences, take best
    // =====================================================================

    double compute_initial_ub(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        int n_random,
        double known_lb)
    {
        int K = static_cast<int>(lengths.size());

        // Build multiset of all job lengths
        std::vector<int> all_jobs;
        for (int i = 0; i < K; ++i)
            for (int j = 0; j < totals[i]; ++j)
                all_jobs.push_back(lengths[i]);

        if (all_jobs.empty())
            return 0.0;

        double best = kInf;

        // Helper for early termination
        auto check_and_update = [&](double cost) -> bool
        {
            if (cost < best)
            {
                best = cost;
                // Early termination if gap closed
                if (known_lb > 0 && std::abs(cost - known_lb) < 0.01)
                    return true;
            }
            return false;
        };

        // Try: sorted ascending (SPT)
        std::vector<int> seq = all_jobs;
        std::sort(seq.begin(), seq.end());
        if (check_and_update(solve_fixed_sequence(seq, prefix_proc, T, spaces)))
            return best;

        // Try: sorted descending (LPT)
        std::sort(seq.begin(), seq.end(), std::greater<int>());
        if (check_and_update(solve_fixed_sequence(seq, prefix_proc, T, spaces)))
            return best;

        // Try: alternating short/long
        {
            std::vector<int> asc = all_jobs;
            std::sort(asc.begin(), asc.end());
            std::vector<int> alt;
            alt.reserve(all_jobs.size());
            int lo = 0, hi = static_cast<int>(asc.size()) - 1;
            bool pick_lo = true;
            while (lo <= hi)
            {
                alt.push_back(pick_lo ? asc[lo++] : asc[hi--]);
                pick_lo = !pick_lo;
            }
            if (check_and_update(solve_fixed_sequence(alt, prefix_proc, T, spaces)))
                return best;
        }

        // Try: all K! permutations of type-group orderings
        // E.g., for perm (2,0,4,1,3): all jobs of lengths[2], then lengths[0], etc.
        if (K <= 8) // K! ≤ 40320
        {
            std::vector<int> perm(K);
            std::iota(perm.begin(), perm.end(), 0);
            do
            {
                std::vector<int> typed_seq;
                typed_seq.reserve(all_jobs.size());
                for (int idx : perm)
                    for (int j = 0; j < totals[idx]; ++j)
                        typed_seq.push_back(lengths[idx]);
                if (check_and_update(solve_fixed_sequence(typed_seq, prefix_proc, T, spaces)))
                    return best;
            } while (std::next_permutation(perm.begin(), perm.end()));
        }

        // Try: random shuffles
        std::mt19937_64 rng(42);
        for (int trial = 0; trial < n_random; ++trial)
        {
            std::shuffle(seq.begin(), seq.end(), rng);
            if (check_and_update(solve_fixed_sequence(seq, prefix_proc, T, spaces)))
                return best;
        }

        return best;
    }

    // =====================================================================
    //  compute_spaces: SPACES preprocessing
    // =====================================================================

    SPACESResult compute_spaces(const std::vector<double> &prices, const MachineStateConfig &config, int max_gap)
    {
        int h = static_cast<int>(prices.size());
        int n_s = static_cast<int>(config.states.size());
        SPACESResult out;
        out.h = h;
        out.p_proc = config.p_trans[config.proc_idx][config.proc_idx];
        out.early = compute_earliest_proc(config);
        out.late = compute_latest_proc(config, h);

        std::vector<double> prefix(h + 1, 0.0);
        for (int i = 0; i < h; ++i)
            prefix[i + 1] = prefix[i] + prices[i];

        struct Edge
        {
            int s_from;
            int s_to;
            int dur;
            double power;
        };
        std::vector<Edge> pos_edges;
        std::vector<double> zero_closure(n_s * n_s, kInf);
        for (int s = 0; s < n_s; ++s)
            zero_closure[s * n_s + s] = 0.0;

        for (int s = 0; s < n_s; ++s)
        {
            for (int sp = 0; sp < n_s; ++sp)
            {
                int dur = config.t_trans[s][sp];
                double power = config.p_trans[s][sp];
                if (dur < 0 || power >= kInf)
                    continue;
                if (dur == 0)
                {
                    zero_closure[s * n_s + sp] = std::min(zero_closure[s * n_s + sp], 0.0);
                }
                else
                {
                    pos_edges.push_back({s, sp, dur, power});
                }
            }
        }

        for (int k = 0; k < n_s; ++k)
        {
            for (int i = 0; i < n_s; ++i)
            {
                double ik = zero_closure[i * n_s + k];
                if (ik >= kInf)
                    continue;
                for (int j = 0; j < n_s; ++j)
                {
                    double via = ik + zero_closure[k * n_s + j];
                    if (via < zero_closure[i * n_s + j])
                        zero_closure[i * n_s + j] = via;
                }
            }
        }

        int eff_max_gap = (max_gap > 0 ? max_gap : h);
        out.max_gap = (eff_max_gap < h ? eff_max_gap : -1);
        out.banded = eff_max_gap < h;
        int cstar_cols = out.banded ? (eff_max_gap + 1) : (h + 1);
        out.c_star.assign((h + 1) * cstar_cols, kInf);
        out.c_start.assign(h + 1, kInf);
        out.c_end.assign(h + 1, kInf);
        for (int i = 0; i <= h; ++i)
            out.c_star[i * cstar_cols] = 0.0;

        for (int t_src = 0; t_src <= h; ++t_src)
        {
            int t_max = std::min(t_src + eff_max_gap, h);
            if (t_src >= t_max)
                continue;
            int n_layers = t_max - t_src + 1;
            std::vector<double> dist(n_layers * n_s, kInf);
            dist[config.proc_idx] = 0.0;
            for (int dt = 0; dt < n_layers; ++dt)
            {
                int t = t_src + dt;
                std::vector<double> row(n_s, kInf);
                for (int s = 0; s < n_s; ++s)
                    row[s] = dist[dt * n_s + s];
                close_same_time(row, zero_closure, n_s);
                for (int s = 0; s < n_s; ++s)
                    dist[dt * n_s + s] = row[s];
                if (t >= h)
                    break;
                for (const auto &e : pos_edges)
                {
                    double base = row[e.s_from];
                    if (base >= kInf)
                        continue;
                    int t_next = t + e.dur;
                    if (t_next > t_max || t_next > h)
                        continue;
                    double cand = base + edge_cost(prefix, t, e.dur, e.power);
                    double &ref = dist[(t_next - t_src) * n_s + e.s_to];
                    if (cand < ref)
                        ref = cand;
                }
            }
            for (int dt = 1; dt < n_layers; ++dt)
            {
                double val = dist[dt * n_s + config.proc_idx];
                if (val >= kInf)
                    continue;
                if (out.banded)
                    out.c_star[t_src * cstar_cols + dt] = val;
                else
                    out.c_star[t_src * cstar_cols + (t_src + dt)] = val;
            }
        }

        std::vector<double> dist_start((h + 1) * n_s, kInf);
        if (h > 0)
        {
            double off_hold = config.p_trans[config.off_idx][config.off_idx];
            dist_start[1 * n_s + config.off_idx] = edge_cost(prefix, 0, 1, off_hold);
        }
        for (int t = 1; t < h; ++t)
        {
            std::vector<double> row(n_s, kInf);
            for (int s = 0; s < n_s; ++s)
                row[s] = dist_start[t * n_s + s];
            close_same_time(row, zero_closure, n_s);
            for (int s = 0; s < n_s; ++s)
                dist_start[t * n_s + s] = row[s];
            for (const auto &e : pos_edges)
            {
                double base = row[e.s_from];
                if (base >= kInf)
                    continue;
                int t_next = t + e.dur;
                if (t_next > h)
                    continue;
                double cand = base + edge_cost(prefix, t, e.dur, e.power);
                double &ref = dist_start[t_next * n_s + e.s_to];
                if (cand < ref)
                    ref = cand;
            }
        }
        for (int t = 0; t <= h; ++t)
            out.c_start[t] = dist_start[t * n_s + config.proc_idx];

        std::vector<double> dist_end((h + 1) * n_s, kInf);
        if (h > 0)
        {
            double off_hold = config.p_trans[config.off_idx][config.off_idx];
            dist_end[(h - 1) * n_s + config.off_idx] = edge_cost(prefix, h - 1, 1, off_hold);
        }
        for (int t = h - 1; t > 0; --t)
        {
            std::vector<double> row(n_s, kInf);
            for (int s = 0; s < n_s; ++s)
                row[s] = dist_end[t * n_s + s];
            close_same_time(row, zero_closure, n_s);
            for (int s = 0; s < n_s; ++s)
                dist_end[t * n_s + s] = row[s];
            for (const auto &e : pos_edges)
            {
                int t_from = t - e.dur;
                if (t_from < 0)
                    continue;
                double base = row[e.s_to];
                if (base >= kInf)
                    continue;
                double cand = base + edge_cost(prefix, t_from, e.dur, e.power);
                double &ref = dist_end[t_from * n_s + e.s_from];
                if (cand < ref)
                    ref = cand;
            }
        }
        for (int t = 0; t <= h; ++t)
            out.c_end[t] = dist_end[t * n_s + config.proc_idx];
        return out;
    }

    // =====================================================================
    //  solve_sparse_dp_stateful: the main multiset DP with SPACES
    //  Now using custom StateMap for cache-friendly hashing + aggressive pruning
    // =====================================================================

    DPResult solve_sparse_dp_stateful(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        const DPParams &params)
    {
        using Clock = std::chrono::steady_clock;
        auto t0 = Clock::now();

        int K = static_cast<int>(lengths.size());
        std::vector<int> radices(K);
        std::vector<int64_t> inc(K);
        int64_t mult = 1;
        int64_t final_state = 0;
        int total_rw = 0;
        int max_job_len = 1;
        for (int i = 0; i < K; ++i)
        {
            radices[i] = totals[i] + 1;
            inc[i] = mult;
            final_state += static_cast<int64_t>(totals[i]) * mult;
            mult *= radices[i];
            total_rw += totals[i] * lengths[i];
            max_job_len = std::max(max_job_len, lengths[i]);
        }

        auto elapsed_sec = [&]() -> double
        {
            return std::chrono::duration<double>(Clock::now() - t0).count();
        };

        // Block-based admissible lower bound on remaining processing cost
        constexpr int LB_BLOCK = 20;
        std::vector<double> proc_prices(T, 0.0);
        for (int i = 0; i < T; ++i)
            proc_prices[i] = prefix_proc[i + 1] - prefix_proc[i];

        // For each block boundary b, sorted prefix sums of prices[b..T-1]
        // lb_prefix[b][rw] = sum of rw cheapest proc prices in [b, T)
        int n_blocks = (T / LB_BLOCK) + 1;
        std::vector<std::vector<double>> lb_prefix(n_blocks + 1);
        for (int bi = 0; bi < n_blocks; ++bi)
        {
            int b = bi * LB_BLOCK;
            if (b < T)
            {
                std::vector<double> sp(proc_prices.begin() + b, proc_prices.end());
                std::sort(sp.begin(), sp.end());
                lb_prefix[bi].resize(sp.size() + 1, 0.0);
                for (std::size_t i = 0; i < sp.size(); ++i)
                    lb_prefix[bi][i + 1] = lb_prefix[bi][i] + sp[i];
            }
            else
            {
                lb_prefix[bi] = {0.0};
            }
        }

        auto lb_proc_cost = [&](int t, int rw) -> double
        {
            int bi = t / LB_BLOCK;
            if (bi >= static_cast<int>(lb_prefix.size()))
                return kInf;
            const auto &arr = lb_prefix[bi];
            if (rw >= static_cast<int>(arr.size()))
                return kInf;
            return arr[rw];
        };

        double min_c_end = kInf;
        for (double x : spaces.c_end)
            if (x < min_c_end)
                min_c_end = x;
        if (min_c_end >= kInf)
            min_c_end = 0.0;

        // Suffix-minimum of c_end: min_c_end_from[t] = min over t' >= t of c_end[t']
        // Tighter than global min_c_end for pruning states at time t.
        std::vector<double> min_c_end_from(T + 2, kInf);
        for (int t = T; t >= 0; --t)
        {
            min_c_end_from[t] = min_c_end_from[t + 1];
            if (spaces.c_end[t] < min_c_end_from[t])
                min_c_end_from[t] = spaces.c_end[t];
        }

        // --- Use StateMap (open-addressing hash) for DP layers ---
        std::vector<StateMap *> layers(T + 1, nullptr);
        for (int i = 0; i <= T; ++i)
            layers[i] = new StateMap(64);

        // Parent tracking (only when needed)
        std::unordered_map<std::pair<int, int64_t>, StatefulParent, PairHash> parent;

        double best_final_cost = (params.known_ub > 0) ? params.known_ub : kInf;
        int64_t best_final_pen = std::numeric_limits<int64_t>::max();
        int best_final_time = -1;
        bool timed_out = false;
        int best_partial_jobs = 0;
        double best_partial_cost = kInf;
        int best_partial_time = 0;
        bool use_early = params.early_tie_break;

        int eff_max_gap = spaces.banded ? spaces.max_gap : T;

        // Shutdown bank: for each multiset state, the best (cost + c_end) seen so far.
        // bank_cost[state] = min over all t_end processed so far of (layer[t_end][state].cost + c_end[t_end])
        // Used for beyond-max-gap transitions: after shutting down, restart later.
        StateMap *shutdown_bank = new StateMap(64);
        // deferred_bank stores entries that become eligible at a future time
        // deferred[t] = list of (state, cost+c_end) to add to bank when processing time t
        std::vector<std::vector<std::pair<int64_t, StateEntry>>> deferred(T + 2);

        // --- Seed: schedule the first job from startup ---
        for (int t_s = spaces.early; t_s <= spaces.late; ++t_s)
        {
            double startup = spaces.c_start[t_s];
            if (startup >= kInf)
                continue;
            for (int i = 0; i < K; ++i)
            {
                int L = lengths[i];
                int t_end = t_s + L;
                if (t_end > T || t_end > spaces.late + 1)
                    continue;
                double cost = startup + (prefix_proc[t_end] - prefix_proc[t_s]);
                int new_rw = total_rw - L;
                int earliest_end = std::min(t_end + new_rw, T + 1);
                double lb = cost + lb_proc_cost(t_end, new_rw) + min_c_end_from[earliest_end];
                if (lb > best_final_cost + kEps)
                    continue;

                int64_t new_state = inc[i];
                int64_t pen = use_early ? t_s : 0;
                auto idx = layers[t_end]->lookup(new_state);
                if (idx < 0)
                {
                    layers[t_end]->insert(new_state, {cost, pen, static_cast<int32_t>(new_rw), 1});
                    if (params.track_schedule)
                        parent[{t_end, new_state}] = {-1, 0, L, t_s};
                }
                else
                {
                    auto &sv = layers[t_end]->val_at(idx);
                    if (cost < sv.cost - kEps || (use_early && std::fabs(cost - sv.cost) <= kEps && pen < sv.pen))
                    {
                        sv = {cost, pen, static_cast<int32_t>(new_rw), 1};
                        if (params.track_schedule)
                            parent[{t_end, new_state}] = {-1, 0, L, t_s};
                    }
                }
            }
        }

        // --- Main DP sweep ---
        for (int t_end = 1; t_end <= T; ++t_end)
        {
            if (params.time_limit > 0 && elapsed_sec() > params.time_limit)
            {
                timed_out = true;
                break;
            }

            // ---- Phase A: absorb deferred shutdown entries into bank ----
            // Entries deferred at time t become eligible for restart at t + eff_max_gap + 1.
            // We deferred them to become available here.
            if (spaces.banded && t_end < static_cast<int>(deferred.size()))
            {
                for (auto &[dstate, dentry] : deferred[t_end])
                {
                    auto bidx = shutdown_bank->lookup(dstate);
                    if (bidx < 0)
                    {
                        shutdown_bank->insert(dstate, dentry);
                    }
                    else
                    {
                        auto &bv = shutdown_bank->val_at(bidx);
                        if (dentry.cost < bv.cost - kEps ||
                            (use_early && std::fabs(dentry.cost - bv.cost) <= kEps && dentry.pen < bv.pen))
                            bv = dentry;
                    }
                }
                deferred[t_end].clear();
                deferred[t_end].shrink_to_fit();
            }

            // ---- Phase B: restart from shutdown bank at t_end ----
            // Any state in the bank shut down long enough ago. Restart here.
            if (spaces.banded && shutdown_bank->size() > 0)
            {
                double c_start_here = spaces.c_start[t_end];
                if (c_start_here < kInf)
                {
                    shutdown_bank->for_each([&](int64_t state, const StateEntry &bv)
                                            {
                        if (bv.rw == 0) return;
                        double base_cost = bv.cost + c_start_here;

                        int used[12];
                        {
                            int64_t x = state;
                            for (int i = 0; i < K; ++i)
                            {
                                used[i] = static_cast<int>(x % radices[i]);
                                x /= radices[i];
                            }
                        }

                        for (int i = 0; i < K; ++i)
                        {
                            if (used[i] >= totals[i]) continue;
                            int L = lengths[i];
                            int job_end = t_end + L;
                            if (job_end > T || job_end > spaces.late + 1) continue;
                            int new_rw = bv.rw - L;
                            double cand_cost = base_cost + (prefix_proc[job_end] - prefix_proc[t_end]);
                            int earliest_end = std::min(job_end + new_rw, T + 1);
                            double lb = cand_cost + lb_proc_cost(job_end, new_rw) + min_c_end_from[earliest_end];
                            if (lb > best_final_cost + kEps) continue;

                            int64_t new_state = state + inc[i];
                            int64_t cand_pen = use_early ? (bv.pen + t_end) : bv.pen;
                            int new_jd = bv.jd + 1;
                            auto *target = layers[job_end];
                            auto tidx = target->lookup(new_state);
                            if (tidx < 0)
                            {
                                target->insert(new_state, {cand_cost, cand_pen, static_cast<int32_t>(new_rw), static_cast<int32_t>(new_jd)});
                                if (params.track_schedule)
                                    parent[{job_end, new_state}] = {-1, state, L, t_end}; // -1 means "from bank"
                            }
                            else
                            {
                                auto &tgt = target->val_at(tidx);
                                if (cand_cost < tgt.cost - kEps || (use_early && std::fabs(cand_cost - tgt.cost) <= kEps && cand_pen < tgt.pen))
                                {
                                    tgt = {cand_cost, cand_pen, static_cast<int32_t>(new_rw), static_cast<int32_t>(new_jd)};
                                    if (params.track_schedule)
                                        parent[{job_end, new_state}] = {-1, state, L, t_end};
                                }
                            }
                        } });
                }
            }

            StateMap *layer = layers[t_end];
            if (!layer || layer->size() == 0)
                continue;
            if (params.max_states > 0 && static_cast<int64_t>(layer->size()) > params.max_states)
            {
                timed_out = true;
                break;
            }

            // Track best partial solution
            layer->for_each([&](int64_t /*key*/, const StateEntry &sv)
                            {
                if (sv.jd > best_partial_jobs || (sv.jd == best_partial_jobs && sv.cost < best_partial_cost))
                {
                    best_partial_jobs = sv.jd;
                    best_partial_cost = sv.cost;
                    best_partial_time = t_end;
                } });

            // Check for complete solutions
            auto final_idx = layer->lookup(final_state);
            if (final_idx >= 0)
            {
                const auto &sv = layer->val_at(final_idx);
                double c_with_shutdown = sv.cost + spaces.c_end[t_end];
                int64_t p_final = sv.pen;
                bool better = c_with_shutdown < best_final_cost - kEps;
                if (use_early && !better && std::fabs(c_with_shutdown - best_final_cost) <= kEps)
                    better = (p_final < best_final_pen) || (p_final == best_final_pen && t_end < best_final_time);
                if (better)
                {
                    best_final_cost = c_with_shutdown;
                    best_final_pen = p_final;
                    best_final_time = t_end;
                }
            }

            // ---- Phase C: within-max-gap transitions ----
            layer->for_each([&](int64_t state, const StateEntry &sv)
                            {
                if (sv.rw == 0)
                    return;

                // Decode used counts ONCE per state
                int used[12];
                {
                    int64_t x = state;
                    for (int i = 0; i < K; ++i)
                    {
                        used[i] = static_cast<int>(x % radices[i]);
                        x /= radices[i];
                    }
                }

                // Within max_gap (inclusive)
                int gap_limit = std::min(t_end + eff_max_gap + 1, spaces.late + 1);
                for (int t_s = t_end; t_s < gap_limit; ++t_s)
                {
                    double gap = spaces.gap_cost(t_end, t_s);
                    if (gap >= kInf)
                        continue;
                    double base_cost = sv.cost + gap;
                    for (int i = 0; i < K; ++i)
                    {
                        if (used[i] >= totals[i])
                            continue;
                        int L = lengths[i];
                        int job_end = t_s + L;
                        if (job_end > T || job_end > spaces.late + 1)
                            continue;
                        int new_rw = sv.rw - L;
                        double cand_cost = base_cost + (prefix_proc[job_end] - prefix_proc[t_s]);
                        int earliest_end = std::min(job_end + new_rw, T + 1);
                        double lb = cand_cost + lb_proc_cost(job_end, new_rw) + min_c_end_from[earliest_end];
                        if (lb > best_final_cost + kEps)
                            continue;

                        int64_t new_state = state + inc[i];
                        int64_t cand_pen = use_early ? (sv.pen + t_s) : sv.pen;
                        int new_jd = sv.jd + 1;
                        auto *target = layers[job_end];
                        auto tidx = target->lookup(new_state);
                        if (tidx < 0)
                        {
                            target->insert(new_state, {cand_cost, cand_pen, static_cast<int32_t>(new_rw), static_cast<int32_t>(new_jd)});
                            if (params.track_schedule)
                                parent[{job_end, new_state}] = {t_end, state, L, t_s};
                        }
                        else
                        {
                            auto &tgt = target->val_at(tidx);
                            if (cand_cost < tgt.cost - kEps || (use_early && std::fabs(cand_cost - tgt.cost) <= kEps && cand_pen < tgt.pen))
                            {
                                tgt = {cand_cost, cand_pen, static_cast<int32_t>(new_rw), static_cast<int32_t>(new_jd)};
                                if (params.track_schedule)
                                    parent[{job_end, new_state}] = {t_end, state, L, t_s};
                            }
                        }
                    }
                } }); // end within-max-gap for_each

            // ---- Phase D: defer shutdown entries for future bank absorption ----
            if (spaces.banded)
            {
                double c_end_here = spaces.c_end[t_end];
                if (c_end_here < kInf)
                {
                    layer->for_each([&](int64_t state, const StateEntry &sv)
                                    {
                        if (sv.rw == 0) return;
                        double shutdown_cost = sv.cost + c_end_here;
                        // This entry becomes eligible for restart at t_end + eff_max_gap + 1
                        int eligible_t = t_end + eff_max_gap + 1;
                        if (eligible_t <= T)
                        {
                            StateEntry be = {shutdown_cost, sv.pen, sv.rw, sv.jd};
                            deferred[eligible_t].push_back({state, be});
                        } });
                }
            }

            // Free old layers to bound memory
            int freed_t = t_end - max_job_len - eff_max_gap;
            if (freed_t >= 0 && layers[freed_t])
            {
                layers[freed_t]->clear();
            }
        }

        DPResult out;
        out.timed_out = timed_out;
        if (best_final_time >= 0 && best_final_cost < kInf)
        {
            out.feasible = true;
            out.cost = best_final_cost;
            out.finish_time = best_final_time;
            if (params.track_schedule)
            {
                int t = best_final_time;
                int64_t s = final_state;
                while (true)
                {
                    auto it = parent.find({t, s});
                    if (it == parent.end())
                        break;
                    const StatefulParent &par = it->second;
                    out.segments.push_back({par.t_start, par.length});
                    if (par.prev_t_end < 0)
                        break;
                    t = par.prev_t_end;
                    s = par.prev_state;
                }
                std::reverse(out.segments.begin(), out.segments.end());
            }
        }
        else if (timed_out && best_partial_jobs > 0)
        {
            out.feasible = true;
            out.cost = best_partial_cost;
            out.finish_time = best_partial_time;
        }

        // Cleanup
        delete shutdown_bank;
        for (int i = 0; i <= T; ++i)
            delete layers[i];

        return out;
    }

    // =====================================================================
    //  solve_relaxed_dp_lb: fast lower bound via (t_end, rw) state space
    //
    //  Relaxes per-type job counts: any job length L ∈ {lengths} can be used
    //  as long as L ≤ remaining_work. This is strictly more permissive, so
    //  the optimal cost is ≤ the true optimal → valid lower bound.
    //
    //  State space: T × total_rw ≈ 270K. Per-state work ≈ max_gap × K.
    //  Total: ~80M ops for benchmark instances, runs in ~1s.
    // =====================================================================

    double solve_relaxed_dp_lb(
        const std::vector<int> &lengths,
        int total_rw,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        RelaxationMode mode)
    {
        std::vector<int> allowed_lengths = relaxation_chunk_lengths(lengths, mode);
        int rw_gcd = 0;
        for (int L : allowed_lengths)
            rw_gcd = std::gcd(rw_gcd, L);
        if (rw_gcd <= 0 || (total_rw % rw_gcd) != 0)
            rw_gcd = 1;
        std::vector<int> scaled_lengths = allowed_lengths;
        if (rw_gcd > 1)
            for (int &L : scaled_lengths)
                L /= rw_gcd;
        int K = static_cast<int>(allowed_lengths.size());
        int eff_max_gap = spaces.banded ? spaces.max_gap : T;

        // dp[rw] at time t_end = min cost with (total_rw - rw) work placed, last job ending at t_end
        // We sweep t_end from left to right, using two arrays per rw value.
        // Since we write to future t_end values, use a 2D table: dp[t][rw].
        int total_rw_scaled = total_rw / rw_gcd;
        int RW = total_rw_scaled + 1;
        std::vector<double> dp((T + 2) * RW, kInf);
        std::vector<std::vector<int>> active_rw(T + 2);
        auto idx = [&](int t, int rw) -> int
        { return t * RW + rw; };
        auto relax_cell = [&](int t, int rw, double cost)
        {
            double &ref = dp[idx(t, rw)];
            if (cost < ref)
            {
                if (ref >= kInf)
                    active_rw[t].push_back(rw);
                ref = cost;
            }
        };

        // Seed: first job from startup
        for (int t_s = spaces.early; t_s <= spaces.late; ++t_s)
        {
            double startup = spaces.c_start[t_s];
            if (startup >= kInf)
                continue;
            for (int j = 0; j < K; ++j)
            {
                int L = allowed_lengths[j];
                int Ls = scaled_lengths[j];
                if (L > total_rw)
                    continue;
                int t_e = t_s + L;
                if (t_e > T || t_e > spaces.late + 1)
                    continue;
                double cost = startup + (prefix_proc[t_e] - prefix_proc[t_s]);
                int new_rw = total_rw_scaled - Ls;
                relax_cell(t_e, new_rw, cost);
            }
        }

        // Shutdown bank: for each rw, best (cost + c_end) from past layers
        // Entries are deferred by eff_max_gap + 1 time steps.
        std::vector<double> bank(RW, kInf);
        std::vector<int> bank_active;
        std::vector<uint8_t> bank_seen(RW, 0);
        // deferred[t] = list of (rw, shutdown_cost) to absorb at time t
        std::vector<std::vector<std::pair<int, double>>> deferred(T + 2);

        double best = kInf;

        for (int t_end = 1; t_end <= T; ++t_end)
        {
            // Phase A: absorb deferred shutdown entries into bank
            if (spaces.banded && t_end < static_cast<int>(deferred.size()))
            {
                for (auto &[rw, cost] : deferred[t_end])
                {
                    if (cost < bank[rw])
                    {
                        if (!bank_seen[rw])
                        {
                            bank_seen[rw] = 1;
                            bank_active.push_back(rw);
                        }
                        bank[rw] = cost;
                    }
                }
                deferred[t_end].clear();
            }

            // Phase B: restart from bank at t_end
            if (spaces.banded && spaces.c_start[t_end] < kInf)
            {
                double start_cost = spaces.c_start[t_end];
                for (int rw : bank_active)
                {
                    if (bank[rw] >= kInf)
                        continue;
                    double base = bank[rw] + start_cost;
                    for (int j = 0; j < K; ++j)
                    {
                        int L = allowed_lengths[j];
                        int Ls = scaled_lengths[j];
                        if (Ls > rw)
                            continue;
                        int t_e = t_end + L;
                        if (t_e > T || t_e > spaces.late + 1)
                            continue;
                        double cost = base + (prefix_proc[t_e] - prefix_proc[t_end]);
                        relax_cell(t_e, rw - Ls, cost);
                    }
                }
            }

            // Check for complete solutions at t_end
            double d0 = dp[idx(t_end, 0)];
            if (d0 < kInf && spaces.c_end[t_end] < kInf)
                best = std::min(best, d0 + spaces.c_end[t_end]);

            // Phase C: within-max-gap transitions
            for (int rw : active_rw[t_end])
            {
                double sv_cost = dp[idx(t_end, rw)];
                if (sv_cost >= kInf)
                    continue;
                int gap_limit = std::min(t_end + eff_max_gap + 1, spaces.late + 1);
                for (int t_s = t_end; t_s < gap_limit; ++t_s)
                {
                    double gap = spaces.gap_cost(t_end, t_s);
                    if (gap >= kInf)
                        continue;
                    double base = sv_cost + gap;
                    for (int j = 0; j < K; ++j)
                    {
                        int L = allowed_lengths[j];
                        int Ls = scaled_lengths[j];
                        if (Ls > rw)
                            continue;
                        int t_e = t_s + L;
                        if (t_e > T || t_e > spaces.late + 1)
                            continue;
                        double cost = base + (prefix_proc[t_e] - prefix_proc[t_s]);
                        relax_cell(t_e, rw - Ls, cost);
                    }
                }
            }

            // Phase D: defer shutdown entries
            if (spaces.banded && spaces.c_end[t_end] < kInf)
            {
                double c_end_here = spaces.c_end[t_end];
                for (int rw : active_rw[t_end])
                {
                    double sv_cost = dp[idx(t_end, rw)];
                    if (sv_cost >= kInf)
                        continue;
                    double shutdown_cost = sv_cost + c_end_here;
                    int eligible = t_end + eff_max_gap + 1;
                    if (eligible <= T)
                        deferred[eligible].push_back({rw, shutdown_cost});
                }
            }
        }

        return best;
    }

    struct RecoveredBlock
    {
        int start = 0;
        int length = 0;
    };

    struct RecoveredBlockPackingResult
    {
        double bin_pack_ub = kInf;
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

    RecoveredBlockPackingResult pack_recovered_blocks(
        const std::vector<RecoveredBlock> &blocks,
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces)
    {
        RecoveredBlockPackingResult result;
        result.pack_outcome = blocks.empty() ? "no_blocks" : "failed";
        result.block_count = static_cast<int>(blocks.size());

        if (blocks.empty())
            return result;

        const char *pack_mode = std::getenv("PAST_RELAXED_BINPACK_SOLVER");
        result.pack_solver = pack_mode ? std::string(pack_mode) : "default";

        auto note_pack_candidate = [&](const std::string &method, double cand)
        {
            if (cand < result.bin_pack_ub)
            {
                result.bin_pack_ub = cand;
                result.pack_method = method;
                result.pack_outcome = "feasible";
            }
        };

        std::vector<RecoveredBlock> merged;
        merged.push_back(blocks[0]);
        for (std::size_t i = 1; i < blocks.size(); ++i)
        {
            RecoveredBlock &last = merged.back();
            if (blocks[i].start <= last.start + last.length)
            {
                int new_end = std::max(last.start + last.length,
                                       blocks[i].start + blocks[i].length);
                last.length = new_end - last.start;
            }
            else
            {
                merged.push_back(blocks[i]);
            }
        }
        result.merged_block_count = static_cast<int>(merged.size());

        std::vector<int> all_jobs;
        for (std::size_t i = 0; i < lengths.size(); ++i)
        {
            for (int j = 0; j < totals[i]; ++j)
                all_jobs.push_back(lengths[i]);
        }

        std::size_t nB = merged.size();
        std::vector<int> orig_cap(nB);
        for (std::size_t i = 0; i < nB; ++i)
            orig_cap[i] = merged[i].length;

        bool exact_pack_decided = false;
        // Legacy env vars removed: PAST_RELAXED_BINPACK_NATIVE_FIRST,
        // PAST_RELAXED_BINPACK_ALLOW_SMALL_NC, PAST_RELAXED_BINPACK_DISABLE_DFS_EXACT
        // are no longer needed. Block-DP is the sole packing method and runs unconditionally.


        {
            auto t0_pack = std::chrono::steady_clock::now();

            auto try_pack = [&](const std::vector<int> &jobs, int mode) -> double
            {
                std::vector<int> cap = orig_cap;
                std::vector<std::vector<int>> bj(nB);
                for (int jl : jobs)
                {
                    int best_b = -1;
                    if (mode == 0)
                    {
                        for (std::size_t b = 0; b < nB; ++b)
                        {
                            if (cap[b] >= jl)
                            {
                                best_b = static_cast<int>(b);
                                break;
                            }
                        }
                    }
                    else
                    {
                        int best_rem = INT_MAX;
                        for (std::size_t b = 0; b < nB; ++b)
                        {
                            if (cap[b] >= jl && cap[b] - jl < best_rem)
                            {
                                best_rem = cap[b] - jl;
                                best_b = static_cast<int>(b);
                            }
                        }
                    }
                    if (best_b < 0)
                        return kInf;
                    cap[best_b] -= jl;
                    bj[best_b].push_back(jl);
                }
                std::vector<int> seq;
                for (std::size_t b = 0; b < nB; ++b)
                    for (int j : bj[b])
                        seq.push_back(j);
                return solve_fixed_sequence(seq, prefix_proc, T, spaces);
            };

            {
                std::vector<int> jobs = all_jobs;
                std::sort(jobs.begin(), jobs.end(), std::greater<int>());
                note_pack_candidate("ffd", try_pack(jobs, 0));
            }
            {
                std::vector<int> jobs = all_jobs;
                std::sort(jobs.begin(), jobs.end(), std::greater<int>());
                note_pack_candidate("bfd", try_pack(jobs, 1));
            }
            {
                std::vector<int> jobs = all_jobs;
                std::sort(jobs.begin(), jobs.end());
                note_pack_candidate("ffi", try_pack(jobs, 0));
            }
            {
                std::vector<int> jobs = all_jobs;
                std::sort(jobs.begin(), jobs.end());
                note_pack_candidate("bfi", try_pack(jobs, 1));
            }
            {
                std::mt19937_64 rng(12345);
                std::vector<int> jobs = all_jobs;
                for (int trial = 0; trial < 20; ++trial)
                {
                    std::shuffle(jobs.begin(), jobs.end(), rng);
                    note_pack_candidate(trial & 1 ? "random_bf" : "random_ff",
                                        try_pack(jobs, trial & 1));
                }
            }

            result.t_pack_heuristic =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_pack).count();
        }

        // ---------------------------------------------------------------
        // Block-based DP packing (the sole exact packing method).
        //
        // Given the recovered block capacities orig_cap[0..nB-1] and job
        // type lengths/totals, determine whether there exists a multiset
        // partition of all jobs into the blocks such that each block is
        // exactly filled.  This is solved exactly via reachability in the
        // mixed-radix count-space.
        //
        // DFS packing has been deprecated: block-DP is both faster and
        // provably exact, rendering the ad-hoc capped DFS redundant.
        // ---------------------------------------------------------------
        auto run_block_dp_packing = [&]()
        {
            if (exact_pack_decided || result.bin_pack_ub < kInf * 0.5)
                return;

            int K = static_cast<int>(lengths.size());
            int nBlk = static_cast<int>(nB);

            // --- Enumerate valid compositions per block ---
            struct BComp
            {
                int64_t delta;
                int counts[8];
            };

            std::vector<int64_t> bp_strides(K);
            int64_t bp_NC = 1;
            for (int i = 0; i < K; ++i)
            {
                bp_strides[i] = bp_NC;
                bp_NC *= (totals[i] + 1);
            }

            std::vector<std::vector<BComp>> bcomps(nBlk);
            for (int b = 0; b < nBlk; ++b)
            {
                int cap = orig_cap[b];
                std::vector<int> cc(K, 0);
                std::function<void(int, int)> en = [&](int ti, int r)
                {
                    if (ti == K)
                    {
                        if (r == 0)
                        {
                            BComp bc;
                            bc.delta = 0;
                            for (int i = 0; i < K; ++i)
                            {
                                bc.counts[i] = cc[i];
                                bc.delta += static_cast<int64_t>(cc[i]) * bp_strides[i];
                            }
                            bcomps[b].push_back(bc);
                        }
                        return;
                    }
                    int L = lengths[ti];
                    int mx = std::min(totals[ti], r / L);
                    for (int c = mx; c >= 0; --c)
                    {
                        cc[ti] = c;
                        en(ti + 1, r - c * L);
                    }
                };
                en(0, cap);
            }

            // --- Order blocks by fewest compositions first (prune early) ---
            std::vector<int> border(nBlk);
            std::iota(border.begin(), border.end(), 0);
            std::sort(border.begin(), border.end(), [&](int a, int b)
                      { return bcomps[a].size() < bcomps[b].size(); });

            std::vector<int> suffix_cap(nBlk + 1, 0);
            for (int i = nBlk - 1; i >= 0; --i)
                suffix_cap[i] = suffix_cap[i + 1] + orig_cap[border[i]];

            // --- Helper lambdas ---
            int64_t initial_st = 0;
            for (int i = 0; i < K; ++i)
                initial_st += static_cast<int64_t>(totals[i]) * bp_strides[i];

            auto compute_work = [&](int64_t s) -> int
            {
                int work = 0;
                int64_t tmp = s;
                for (int i = 0; i < K; ++i)
                {
                    int rv = static_cast<int>(tmp % (totals[i] + 1));
                    tmp /= (totals[i] + 1);
                    work += rv * lengths[i];
                }
                return work;
            };

            auto decode_state = [&](int64_t s, int r[8])
            {
                int64_t tmp = s;
                for (int i = 0; i < K; ++i)
                {
                    r[i] = static_cast<int>(tmp % (totals[i] + 1));
                    tmp /= (totals[i] + 1);
                }
            };

            // --- Forward pass: build reachability sets ---
            auto t0_block_dp = std::chrono::steady_clock::now();

            std::vector<std::unordered_set<int64_t>> reach(nBlk + 1);
            reach[0].insert(initial_st);

            for (int bi = 0; bi < nBlk; ++bi)
            {
                int b = border[bi];
                auto &comps_b = bcomps[b];
                int required_work = suffix_cap[bi];
                for (int64_t s : reach[bi])
                {
                    if (compute_work(s) != required_work)
                        continue;
                    int r[8];
                    decode_state(s, r);
                    for (auto &bc : comps_b)
                    {
                        bool ok = true;
                        for (int i = 0; i < K; ++i)
                        {
                            if (bc.counts[i] > r[i])
                            {
                                ok = false;
                                break;
                            }
                        }
                        if (ok)
                            reach[bi + 1].insert(s - bc.delta);
                    }
                }
            }

            // --- Result: reconstruct assignment if feasible ---
            if (reach[nBlk].count(0))
            {
                std::vector<std::vector<int>> asgn(nBlk);
                int64_t s = initial_st;
                bool reconstruction_ok = true;
                for (int bi = 0; bi < nBlk; ++bi)
                {
                    int b = border[bi];
                    if (compute_work(s) != suffix_cap[bi])
                    {
                        reconstruction_ok = false;
                        break;
                    }
                    int r[8];
                    decode_state(s, r);
                    bool found = false;
                    for (auto &bc : bcomps[b])
                    {
                        bool ok = true;
                        for (int i = 0; i < K; ++i)
                        {
                            if (bc.counts[i] > r[i])
                            {
                                ok = false;
                                break;
                            }
                        }
                        if (!ok)
                            continue;
                        int64_t ns = s - bc.delta;
                        if (reach[bi + 1].count(ns))
                        {
                            asgn[b].assign(bc.counts, bc.counts + K);
                            s = ns;
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                    {
                        reconstruction_ok = false;
                        break;
                    }
                }
                if (reconstruction_ok)
                {
                    std::vector<int> seq;
                    for (int b = 0; b < nBlk; ++b)
                        for (int i = 0; i < K; ++i)
                            for (int j = 0; j < asgn[b][i]; ++j)
                                seq.push_back(lengths[i]);
                    note_pack_candidate("block_dp_exact",
                                        solve_fixed_sequence(seq, prefix_proc, T, spaces));
                }
            }
            else
            {
                result.pack_outcome = "infeasible";
            }

            result.t_pack_block_dp =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_block_dp).count();
        };

        run_block_dp_packing();

        return result;
    }

    // =====================================================================
    //  solve_relaxed_dp_with_binpack: combined forward LB + bin-packing UB
    //  in a single relaxed DP pass (with parent tracking).
    //  Returns {lb, bin_pack_ub} — saves one full DP computation.
    // =====================================================================

    RelaxedDPResult solve_relaxed_dp_with_binpack(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces)
    {
        int K = static_cast<int>(lengths.size());
        std::vector<int> job_order(K);
        std::iota(job_order.begin(), job_order.end(), 0);
        std::sort(job_order.begin(), job_order.end(), [&](int a, int b)
                  {
                      if (lengths[a] != lengths[b])
                          return lengths[a] > lengths[b];
                      return a < b;
                  });
        int eff_max_gap = spaces.banded ? spaces.max_gap : T;
        int total_rw = 0;
        for (int i = 0; i < K; ++i)
            total_rw += lengths[i] * totals[i];
        int rw_gcd = 0;
        for (int L : lengths)
            rw_gcd = std::gcd(rw_gcd, L);
        if (rw_gcd <= 0 || (total_rw % rw_gcd) != 0)
            rw_gcd = 1;
        std::vector<int> scaled_lengths = lengths;
        if (rw_gcd > 1)
            for (int &L : scaled_lengths)
                L /= rw_gcd;
        int total_rw_scaled = total_rw / rw_gcd;
        int RW = total_rw_scaled + 1;

        // Relaxed DP with parent tracking (same as bin_packing_ub internal DP)
        struct RPar
        {
            int prev_t;
            int prev_rw;
            int L;
            int t_s;
        };
        std::vector<double> dp((T + 2) * RW, kInf);
        std::vector<RPar> par((T + 2) * RW, {-1, -1, 0, 0});
        std::vector<std::vector<int>> active_rw(T + 2);
        int64_t states_reached = 0;
        int64_t states_expanded = 0;
        auto idx = [&](int t, int rw) -> int
        { return t * RW + rw; };
        auto relax_cell = [&](int t, int rw, double cost, RPar parent)
        {
            int i = idx(t, rw);
            if (cost < dp[i])
            {
                if (dp[i] >= kInf)
                {
                    ++states_reached;
                    active_rw[t].push_back(rw);
                }
                dp[i] = cost;
                par[i] = parent;
            }
        };

        // Seed
        for (int t_s = spaces.early; t_s <= spaces.late; ++t_s)
        {
            double startup = spaces.c_start[t_s];
            if (startup >= kInf)
                continue;
            for (int j : job_order)
            {
                int L = lengths[j];
                int Ls = scaled_lengths[j];
                if (L > total_rw)
                    continue;
                int t_e = t_s + L;
                if (t_e > T || t_e > spaces.late + 1)
                    continue;
                double cost = startup + (prefix_proc[t_e] - prefix_proc[t_s]);
                int new_rw = total_rw_scaled - Ls;
                relax_cell(t_e, new_rw, cost, {-1, total_rw_scaled, L, t_s});
            }
        }

        std::vector<double> bank_cost(RW, kInf);
        std::vector<int> bank_t(RW, -1);
        std::vector<int> bank_active;
        std::vector<uint8_t> bank_seen(RW, 0);
        std::vector<std::vector<std::tuple<int, double, int>>> deferred(T + 2);

        double best = kInf;
        int best_t = -1;

        for (int t_end = 1; t_end <= T; ++t_end)
        {
            if (spaces.banded && t_end < (int)deferred.size())
            {
                for (auto &[rw, cost, t_src] : deferred[t_end])
                    if (cost < bank_cost[rw])
                    {
                        if (!bank_seen[rw])
                        {
                            bank_seen[rw] = 1;
                            bank_active.push_back(rw);
                        }
                        bank_cost[rw] = cost;
                        bank_t[rw] = t_src;
                    }
                deferred[t_end].clear();
            }

            if (spaces.banded && spaces.c_start[t_end] < kInf)
            {
                double start_cost = spaces.c_start[t_end];
                for (int rw : bank_active)
                {
                    if (bank_cost[rw] >= kInf)
                        continue;
                    double base = bank_cost[rw] + start_cost;
                    for (int j : job_order)
                    {
                        int L = lengths[j];
                        int Ls = scaled_lengths[j];
                        if (Ls > rw)
                            continue;
                        int t_e = t_end + L;
                        if (t_e > T || t_e > spaces.late + 1)
                            continue;
                        double cost = base + (prefix_proc[t_e] - prefix_proc[t_end]);
                        relax_cell(t_e, rw - Ls, cost, {bank_t[rw], rw, L, t_end});
                    }
                }
            }

            double d0 = dp[idx(t_end, 0)];
            if (d0 < kInf && spaces.c_end[t_end] < kInf)
            {
                double total = d0 + spaces.c_end[t_end];
                if (total < best)
                {
                    best = total;
                    best_t = t_end;
                }
            }

            for (int rw : active_rw[t_end])
            {
                double sv_cost = dp[idx(t_end, rw)];
                if (sv_cost >= kInf)
                    continue;
                ++states_expanded;
                int gap_limit = std::min(t_end + eff_max_gap + 1, spaces.late + 1);
                for (int t_s = t_end; t_s < gap_limit; ++t_s)
                {
                    double gap = spaces.gap_cost(t_end, t_s);
                    if (gap >= kInf)
                        continue;
                    double base = sv_cost + gap;
                    for (int j : job_order)
                    {
                        int L = lengths[j];
                        int Ls = scaled_lengths[j];
                        if (Ls > rw)
                            continue;
                        int t_e = t_s + L;
                        if (t_e > T || t_e > spaces.late + 1)
                            continue;
                        double cost = base + (prefix_proc[t_e] - prefix_proc[t_s]);
                        relax_cell(t_e, rw - Ls, cost, {t_end, rw, L, t_s});
                    }
                }
            }

            if (spaces.banded && spaces.c_end[t_end] < kInf)
            {
                double c_end_here = spaces.c_end[t_end];
                for (int rw : active_rw[t_end])
                {
                    double sv_cost = dp[idx(t_end, rw)];
                    if (sv_cost >= kInf)
                        continue;
                    int eligible = t_end + eff_max_gap + 1;
                    if (eligible <= T)
                        deferred[eligible].push_back({rw, sv_cost + c_end_here, t_end});
                }
            }
        }

        double lb = best; // This IS the relaxed DP LB

        // ---------------------------------------------------------------
        //  CO-OPTIMAL PROFILE DIVERSITY
        //
        //  The relaxed DP table encodes a DAG of all optimal paths, not
        //  just a single one. By collecting multiple co-optimal terminal
        //  times and backtracking from each, we obtain structurally
        //  diverse block profiles at zero additional DP cost. Each new
        //  profile is an independent packing candidate.
        //
        //  Theorem justification: for any t such that
        //      dp[t][0] + c_end[t] = OPT_relaxed,
        //  the parent chain from (t, 0) yields an optimal block
        //  decomposition. Different t values typically produce different
        //  block structures (different starts, lengths, gaps).
        // ---------------------------------------------------------------
        int max_profiles = (K <= 2) ? std::min(T, 2048) : 256;
        std::vector<int> co_optimal_terminals;
        if (best_t >= 0)
        {
            for (int t = 1; t <= T; ++t)
            {
                double d0 = dp[idx(t, 0)];
                if (d0 < kInf && spaces.c_end[t] < kInf)
                {
                    double total = d0 + spaces.c_end[t];
                    if (std::abs(total - best) < 1e-6)
                        co_optimal_terminals.push_back(t);
                }
            }
            // Ensure best_t is first (highest priority)
            auto it = std::find(co_optimal_terminals.begin(),
                                co_optimal_terminals.end(), best_t);
            if (it != co_optimal_terminals.end() &&
                it != co_optimal_terminals.begin())
            {
                std::swap(*co_optimal_terminals.begin(), *it);
            }
            // Limit to avoid pathological cases
            if ((int)co_optimal_terminals.size() > max_profiles)
                co_optimal_terminals.resize(max_profiles);
        }

        // Try packing each co-optimal profile
        RecoveredBlockPackingResult pack;
        pack.pack_outcome = (best_t >= 0 ? "not_attempted" : "no_relaxed_path");
        for (int term_t : co_optimal_terminals)
        {
            std::vector<RecoveredBlock> blocks;
            {
                int t = term_t;
                int rw = 0;
                while (true)
                {
                    int i = idx(t, rw);
                    const RPar &p = par[i];
                    if (p.L <= 0)
                        break;
                    blocks.push_back({p.t_s, p.L});
                    t = p.prev_t;
                    rw = p.prev_rw;
                    if (t < 0)
                        break;
                }
                std::reverse(blocks.begin(), blocks.end());
            }
            pack = pack_recovered_blocks(blocks, lengths, totals, prefix_proc, T, spaces);
            if (pack.bin_pack_ub < kInf * 0.5)
                break; // found a packable profile!
        }

        RelaxedDPResult result;
        result.lb = lb;
        result.bin_pack_ub = pack.bin_pack_ub;
        result.states_reached = states_reached;
        result.states_expanded = states_expanded;
        result.rdp = std::move(dp);  // zero-copy transfer of the dp table
        result.RW = RW;
        result.block_count = pack.block_count;
        result.merged_block_count = pack.merged_block_count;
        result.pack_solver = pack.pack_solver;
        result.pack_external_status = pack.pack_external_status;
        result.pack_method = pack.pack_method;
        result.pack_outcome = pack.pack_outcome;
        result.t_pack_external = pack.t_pack_external;
        result.t_pack_heuristic = pack.t_pack_heuristic;
        result.t_pack_dfs = pack.t_pack_dfs;
        result.t_pack_block_dp = pack.t_pack_block_dp;
        return result;
    }

    RelaxedDPResult solve_relaxed_dp_lb_feas_with_binpack(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces)
    {
        int K = static_cast<int>(lengths.size());
        std::vector<int> job_order(K);
        std::iota(job_order.begin(), job_order.end(), 0);
        std::sort(job_order.begin(), job_order.end(), [&](int a, int b)
                  {
                      if (lengths[a] != lengths[b])
                          return lengths[a] > lengths[b];
                      return a < b;
                  });
        int eff_max_gap = spaces.banded ? spaces.max_gap : T;
        int total_rw = 0;
        for (int i = 0; i < K; ++i)
            total_rw += lengths[i] * totals[i];
        int RW = total_rw + 1;

        struct RPar
        {
            int prev_t;
            int prev_rw;
            int L;
            int t_s;
        };

        auto bounded_work = compute_bounded_work_set(lengths, totals);
        auto feas = compute_feas_sets(lengths, totals);

        auto can_place = [&](int j, int rw, int next_rw) -> bool
        {
            int placed = total_rw - rw;
            if (placed < 0 || placed > total_rw)
                return false;
            if (next_rw < 0 || next_rw > total_rw)
                return false;
            return bounded_work[placed] && feas[j][placed] && feas[j][next_rw];
        };

        std::vector<double> dp((T + 2) * RW, kInf);
        std::vector<RPar> par((T + 2) * RW, {-1, -1, 0, 0});
        auto idx = [&](int t, int rw) -> int
        { return t * RW + rw; };

        int64_t states_reached = 0;
        int64_t states_expanded = 0;

        for (int t_s = spaces.early; t_s <= spaces.late; ++t_s)
        {
            double startup = spaces.c_start[t_s];
            if (startup >= kInf)
                continue;
            for (int j : job_order)
            {
                int L = lengths[j];
                if (L > total_rw)
                    continue;
                int new_rw = total_rw - L;
                if (!can_place(j, total_rw, new_rw))
                    continue;
                int t_e = t_s + L;
                if (t_e > T || t_e > spaces.late + 1)
                    continue;
                double cost = startup + (prefix_proc[t_e] - prefix_proc[t_s]);
                int i = idx(t_e, new_rw);
                if (cost < dp[i])
                {
                    if (dp[i] >= kInf)
                        ++states_reached;
                    dp[i] = cost;
                    par[i] = {-1, total_rw, L, t_s};
                }
            }
        }

        std::vector<double> bank_cost(RW, kInf);
        std::vector<int> bank_t(RW, -1);
        std::vector<std::vector<std::tuple<int, double, int>>> deferred(T + 2);

        double best = kInf;
        int best_t = -1;

        for (int t_end = 1; t_end <= T; ++t_end)
        {
            if (spaces.banded && t_end < static_cast<int>(deferred.size()))
            {
                for (auto &[rw, cost, t_src] : deferred[t_end])
                {
                    if (cost < bank_cost[rw])
                    {
                        bank_cost[rw] = cost;
                        bank_t[rw] = t_src;
                    }
                }
                deferred[t_end].clear();
            }

            if (spaces.banded && spaces.c_start[t_end] < kInf)
            {
                double start_cost = spaces.c_start[t_end];
                for (int rw = 1; rw < RW; ++rw)
                {
                    if (bank_cost[rw] >= kInf)
                        continue;
                    double base = bank_cost[rw] + start_cost;
                    for (int j : job_order)
                    {
                        int L = lengths[j];
                        if (L > rw)
                            continue;
                        int next_rw = rw - L;
                        if (!can_place(j, rw, next_rw))
                            continue;
                        int t_e = t_end + L;
                        if (t_e > T || t_e > spaces.late + 1)
                            continue;
                        double cost = base + (prefix_proc[t_e] - prefix_proc[t_end]);
                        int i = idx(t_e, next_rw);
                        if (cost < dp[i])
                        {
                            if (dp[i] >= kInf)
                                ++states_reached;
                            dp[i] = cost;
                            par[i] = {bank_t[rw], rw, L, t_end};
                        }
                    }
                }
            }

            double d0 = dp[idx(t_end, 0)];
            if (d0 < kInf && spaces.c_end[t_end] < kInf)
            {
                double total = d0 + spaces.c_end[t_end];
                if (total < best)
                {
                    best = total;
                    best_t = t_end;
                }
            }

            for (int rw = 1; rw < RW; ++rw)
            {
                double sv_cost = dp[idx(t_end, rw)];
                if (sv_cost >= kInf)
                    continue;
                ++states_expanded;
                int gap_limit = std::min(t_end + eff_max_gap + 1, spaces.late + 1);
                for (int t_s = t_end; t_s < gap_limit; ++t_s)
                {
                    double gap = spaces.gap_cost(t_end, t_s);
                    if (gap >= kInf)
                        continue;
                    double base = sv_cost + gap;
                    for (int j : job_order)
                    {
                        int L = lengths[j];
                        if (L > rw)
                            continue;
                        int next_rw = rw - L;
                        if (!can_place(j, rw, next_rw))
                            continue;
                        int t_e = t_s + L;
                        if (t_e > T || t_e > spaces.late + 1)
                            continue;
                        double cost = base + (prefix_proc[t_e] - prefix_proc[t_s]);
                        int i = idx(t_e, next_rw);
                        if (cost < dp[i])
                        {
                            if (dp[i] >= kInf)
                                ++states_reached;
                            dp[i] = cost;
                            par[i] = {t_end, rw, L, t_s};
                        }
                    }
                }
            }

            if (spaces.banded && spaces.c_end[t_end] < kInf)
            {
                double c_end_here = spaces.c_end[t_end];
                for (int rw = 1; rw < RW; ++rw)
                {
                    double sv_cost = dp[idx(t_end, rw)];
                    if (sv_cost >= kInf)
                        continue;
                    int eligible = t_end + eff_max_gap + 1;
                    if (eligible <= T)
                        deferred[eligible].push_back({rw, sv_cost + c_end_here, t_end});
                }
            }
        }

        int max_profiles = (K <= 2) ? std::min(T, 2048) : 256;
        std::vector<int> co_optimal_terminals;
        if (best_t >= 0)
        {
            for (int t = 1; t <= T; ++t)
            {
                double d0 = dp[idx(t, 0)];
                if (d0 < kInf && spaces.c_end[t] < kInf)
                {
                    double total = d0 + spaces.c_end[t];
                    if (std::abs(total - best) < 1e-6)
                        co_optimal_terminals.push_back(t);
                }
            }
            auto it = std::find(co_optimal_terminals.begin(),
                                co_optimal_terminals.end(), best_t);
            if (it != co_optimal_terminals.end() &&
                it != co_optimal_terminals.begin())
            {
                std::swap(*co_optimal_terminals.begin(), *it);
            }
            if ((int)co_optimal_terminals.size() > max_profiles)
                co_optimal_terminals.resize(max_profiles);
        }

        RecoveredBlockPackingResult pack;
        pack.pack_outcome = (best_t >= 0 ? "not_attempted" : "no_relaxed_path");
        for (int term_t : co_optimal_terminals)
        {
            std::vector<RecoveredBlock> blocks;
            int t = term_t;
            int rw = 0;
            while (true)
            {
                int i = idx(t, rw);
                const RPar &p = par[i];
                if (p.L <= 0)
                    break;
                blocks.push_back({p.t_s, p.L});
                t = p.prev_t;
                rw = p.prev_rw;
                if (t < 0)
                    break;
            }
            std::reverse(blocks.begin(), blocks.end());
            pack = pack_recovered_blocks(blocks, lengths, totals, prefix_proc, T, spaces);
            if (pack.bin_pack_ub < kInf * 0.5)
                break;
        }

        RelaxedDPResult result;
        result.lb = best;
        result.bin_pack_ub = pack.bin_pack_ub;
        result.states_reached = states_reached;
        result.states_expanded = states_expanded;
        result.rdp = std::move(dp);
        result.RW = RW;
        result.block_count = pack.block_count;
        result.merged_block_count = pack.merged_block_count;
        result.pack_solver = pack.pack_solver;
        result.pack_external_status = pack.pack_external_status;
        result.pack_method = pack.pack_method;
        result.pack_outcome = pack.pack_outcome;
        result.t_pack_external = pack.t_pack_external;
        result.t_pack_heuristic = pack.t_pack_heuristic;
        result.t_pack_dfs = pack.t_pack_dfs;
        result.t_pack_block_dp = pack.t_pack_block_dp;
        return result;
    }

    // =====================================================================
    //  smart_reconstruct: Count-aware path search using relaxed DP table.
    //  Searches for count-feasible paths through the precomputed rdp table.
    //  Key insight: rdp[t][rw] = min cost to reach (t, rw) in relaxed problem.
    //  A count-feasible path (t, c₁,...,cK) maps to (t, W - Σcⱼ*Lⱼ) = (t, rw).
    //  We can prune count-feasible states whose (t, rw) projection is unreachable
    //  or suboptimal in the relaxed DP.
    // =====================================================================

    double smart_reconstruct(
        const std::vector<double> &rdp,
        int RW,
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        double known_ub,
        double time_limit_sec)
    {
        auto t0_sr = std::chrono::steady_clock::now();
        int K = static_cast<int>(lengths.size());
        if (K == 0)
            return 0.0;

        // Compute strides (mixed-radix encoding) and total state count
        std::vector<int> strides(K);
        int NC = 1;
        for (int i = 0; i < K; ++i)
        {
            strides[i] = NC;
            if (static_cast<int64_t>(NC) * (totals[i] + 1) > 500'000)
                return kInf; // state space too large
            NC *= (totals[i] + 1);
        }

        int final_state = 0;
        int total_rw = 0;
        for (int i = 0; i < K; ++i)
        {
            final_state += totals[i] * strides[i];
            total_rw += totals[i] * lengths[i];
        }

        int eff_max_gap = spaces.banded ? spaces.max_gap : T;

        // Total cells check (memory limit ~4.8GB for doubles)
        int64_t total_cells = static_cast<int64_t>(T + 2) * NC;
        if (total_cells > 600'000'000LL)
            return kInf;

        // Precompute per-state info: remaining work + counts
        std::vector<int> state_rw(NC);
        std::vector<int> state_counts(static_cast<size_t>(NC) * K);
        for (int s = 0; s < NC; ++s)
        {
            int rw = total_rw;
            int tmp = s;
            for (int i = 0; i < K; ++i)
            {
                int ci = tmp % (totals[i] + 1);
                tmp /= (totals[i] + 1);
                state_counts[static_cast<size_t>(s) * K + i] = ci;
                rw -= ci * lengths[i];
            }
            state_rw[s] = rw;
        }

        // Dense DP array for smart reconstruction
        std::vector<double> sr_dp(total_cells, kInf);
        auto sr_idx = [&](int t, int s) -> int64_t
        {
            return static_cast<int64_t>(t) * NC + s;
        };

        // Relaxed DP lookup helper
        auto rdp_cost = [&](int t, int rw) -> double
        {
            if (t < 0 || t > T + 1 || rw < 0 || rw >= RW)
                return kInf;
            return rdp[t * RW + rw];
        };

        // Get the relaxed DP optimal value (LB)
        double rdp_lb = kInf;
        for (int t = 0; t <= T; ++t)
        {
            double d = rdp_cost(t, 0);
            if (d < kInf && spaces.c_end[t] < kInf)
            {
                double total = d + spaces.c_end[t];
                if (total < rdp_lb)
                    rdp_lb = total;
            }
        }

        // Proc-cost LB for pruning: sorted prefix sums
        std::vector<double> proc_prices(T);
        for (int i = 0; i < T; ++i)
            proc_prices[i] = prefix_proc[i + 1] - prefix_proc[i];

        constexpr int LB_BLOCK = 20;
        int n_blocks = (T / LB_BLOCK) + 1;
        std::vector<std::vector<double>> lb_prefix(n_blocks + 1);
        for (int bi = 0; bi < n_blocks; ++bi)
        {
            int b = bi * LB_BLOCK;
            if (b < T)
            {
                std::vector<double> sp(proc_prices.begin() + b, proc_prices.end());
                std::sort(sp.begin(), sp.end());
                lb_prefix[bi].resize(sp.size() + 1, 0.0);
                for (std::size_t i = 0; i < sp.size(); ++i)
                    lb_prefix[bi][i + 1] = lb_prefix[bi][i] + sp[i];
            }
            else
            {
                lb_prefix[bi] = {0.0};
            }
        }
        auto lb_proc_cost = [&](int t, int rw) -> double
        {
            int bi = t / LB_BLOCK;
            if (bi >= static_cast<int>(lb_prefix.size()))
                return kInf;
            const auto &arr = lb_prefix[bi];
            if (rw >= static_cast<int>(arr.size()))
                return kInf;
            return arr[rw];
        };

        // Suffix-minimum of c_end for pruning
        std::vector<double> min_c_end_from(T + 2, kInf);
        for (int t = T; t >= 0; --t)
        {
            min_c_end_from[t] = min_c_end_from[t + 1];
            if (spaces.c_end[t] < min_c_end_from[t])
                min_c_end_from[t] = spaces.c_end[t];
        }

        double best = known_ub;

        // Seed: first job from startup
        for (int t_s = spaces.early; t_s <= spaces.late; ++t_s)
        {
            double startup = spaces.c_start[t_s];
            if (startup >= kInf)
                continue;
            for (int i = 0; i < K; ++i)
            {
                int L = lengths[i];
                int t_e = t_s + L;
                if (t_e > T || t_e > spaces.late + 1)
                    continue;
                double cost = startup + (prefix_proc[t_e] - prefix_proc[t_s]);
                int new_s = strides[i]; // c_i goes from 0 to 1
                int new_rw = state_rw[new_s];

                // Standard LB pruning
                int earliest_end = std::min(t_e + new_rw, T + 1);
                double lb = cost + lb_proc_cost(t_e, new_rw) + min_c_end_from[earliest_end];
                if (lb > best + kEps)
                    continue;

                // rdp pruning: if the relaxed DP says (t_e, new_rw) is unreachable, skip
                if (rdp_cost(t_e, new_rw) >= kInf)
                    continue;

                auto di = sr_idx(t_e, new_s);
                sr_dp[di] = std::min(sr_dp[di], cost);
            }
        }

        // Bank for beyond-max-gap transitions
        std::vector<double> bank(NC, kInf);
        std::vector<std::vector<std::pair<int, double>>> deferred(T + 2);

        for (int t_end = 1; t_end <= T; ++t_end)
        {
            // Time check every 32 steps
            if ((t_end & 31) == 0)
            {
                double elapsed = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - t0_sr)
                                     .count();
                if (elapsed > time_limit_sec)
                    return kInf; // timed out: cannot certify optimality
            }

            // Phase A: absorb deferred shutdown entries
            if (spaces.banded && t_end < static_cast<int>(deferred.size()))
            {
                for (auto &[si, cost] : deferred[t_end])
                    bank[si] = std::min(bank[si], cost);
                deferred[t_end].clear();
            }

            // Phase B: restart from bank
            if (spaces.banded && spaces.c_start[t_end] < kInf)
            {
                double start_cost = spaces.c_start[t_end];
                for (int s = 0; s < NC; ++s)
                {
                    if (state_rw[s] <= 0)
                        continue;
                    if (bank[s] >= kInf)
                        continue;
                    double base = bank[s] + start_cost;
                    const int *counts = &state_counts[static_cast<size_t>(s) * K];
                    for (int i = 0; i < K; ++i)
                    {
                        if (counts[i] >= totals[i])
                            continue;
                        int L = lengths[i];
                        int t_e = t_end + L;
                        if (t_e > T || t_e > spaces.late + 1)
                            continue;
                        double cost = base + (prefix_proc[t_e] - prefix_proc[t_end]);
                        int new_s = s + strides[i];
                        int new_rw = state_rw[new_s];

                        // Standard LB pruning
                        int earliest_end = std::min(t_e + new_rw, T + 1);
                        double lb = cost + lb_proc_cost(t_e, new_rw) + min_c_end_from[earliest_end];
                        if (lb > best + kEps)
                            continue;

                        // rdp pruning: if the relaxed DP says (t_e, new_rw) is unreachable, skip
                        double rdp_val = rdp_cost(t_e, new_rw);
                        if (rdp_val >= kInf)
                            continue;

                        // rdp tightness pruning: if our count-feasible cost at this (t, rw)
                        // projection is much worse than the relaxed optimum, prune
                        // Gap budget = best - rdp_lb; accumulated gap = cost - rdp_val
                        if (rdp_lb < kInf && cost - rdp_val > best - rdp_lb + kEps)
                            continue;

                        auto di = sr_idx(t_e, new_s);
                        sr_dp[di] = std::min(sr_dp[di], cost);
                    }
                }
            }

            // Check for complete solutions
            {
                double d = sr_dp[sr_idx(t_end, final_state)];
                if (d < kInf && spaces.c_end[t_end] < kInf)
                {
                    double total = d + spaces.c_end[t_end];
                    if (total < best)
                        best = total;
                }
            }

            // Phase C: within-max-gap transitions
            int base_offset = t_end * NC;
            for (int s = 0; s < NC; ++s)
            {
                if (state_rw[s] <= 0)
                    continue;
                double sv = sr_dp[base_offset + s];
                if (sv >= kInf)
                    continue;

                const int *counts = &state_counts[static_cast<size_t>(s) * K];
                int gap_limit = std::min(t_end + eff_max_gap + 1, spaces.late + 1);
                for (int t_s = t_end; t_s < gap_limit; ++t_s)
                {
                    double gap = spaces.gap_cost(t_end, t_s);
                    if (gap >= kInf)
                        continue;
                    double base = sv + gap;

                    for (int i = 0; i < K; ++i)
                    {
                        if (counts[i] >= totals[i])
                            continue;
                        int L = lengths[i];
                        int t_e = t_s + L;
                        if (t_e > T || t_e > spaces.late + 1)
                            continue;
                        double cost = base + (prefix_proc[t_e] - prefix_proc[t_s]);
                        int new_s = s + strides[i];
                        int new_rw = state_rw[new_s];

                        // Standard LB pruning
                        int earliest_end = std::min(t_e + new_rw, T + 1);
                        double lb = cost + lb_proc_cost(t_e, new_rw) + min_c_end_from[earliest_end];
                        if (lb > best + kEps)
                            continue;

                        // rdp pruning: if the relaxed DP says (t_e, new_rw) is unreachable, skip
                        double rdp_val = rdp_cost(t_e, new_rw);
                        if (rdp_val >= kInf)
                            continue;

                        // rdp tightness pruning
                        if (rdp_lb < kInf && cost - rdp_val > best - rdp_lb + kEps)
                            continue;

                        auto di = sr_idx(t_e, new_s);
                        sr_dp[di] = std::min(sr_dp[di], cost);
                    }
                }
            }

            // Phase D: defer shutdown entries
            if (spaces.banded && spaces.c_end[t_end] < kInf)
            {
                double c_end_here = spaces.c_end[t_end];
                for (int s = 0; s < NC; ++s)
                {
                    if (state_rw[s] <= 0)
                        continue;
                    double sv = sr_dp[base_offset + s];
                    if (sv >= kInf)
                        continue;
                    double sc = sv + c_end_here;
                    int eligible = t_end + eff_max_gap + 1;
                    if (eligible <= T)
                        deferred[eligible].push_back({s, sc});
                }
            }
        }

        return best;
    }

    // =====================================================================
    //  solve_relaxed_dp_lb_backward: same relaxation run in reverse time.
    //  LB = max(forward, backward) is tighter due to asymmetric prices.
    //  Generic: properly reverses any MachineStateConfig (NOSBY, TWOSBY, etc.).
    // =====================================================================

    double solve_relaxed_dp_lb_backward(
        const std::vector<int> &lengths,
        int total_rw,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        const MachineStateConfig &fwd_config)
    {
        // Extract original prices from prefix_proc
        std::vector<double> prices_rev(T);
        for (int i = 0; i < T; ++i)
            prices_rev[i] = (prefix_proc[T - i] - prefix_proc[T - i - 1]) / spaces.p_proc;

        // Build reversed config: for each edge (s→s') in forward, create (s'→s)
        // Self-loops are unchanged. off_idx and proc_idx stay the same.
        int n_s = static_cast<int>(fwd_config.states.size());
        MachineStateConfig cfg;
        cfg.states = fwd_config.states;
        cfg.off_idx = fwd_config.off_idx;
        cfg.proc_idx = fwd_config.proc_idx;
        cfg.t_trans.assign(n_s, std::vector<int>(n_s, -1));
        cfg.p_trans.assign(n_s, std::vector<double>(n_s, kInf));

        for (int s = 0; s < n_s; ++s)
        {
            for (int sp = 0; sp < n_s; ++sp)
            {
                int dur = fwd_config.t_trans[s][sp];
                double power = fwd_config.p_trans[s][sp];
                if (dur < 0 || power >= kInf)
                    continue;
                if (s == sp)
                {
                    // Self-loops stay the same
                    cfg.t_trans[s][s] = dur;
                    cfg.p_trans[s][s] = power;
                }
                else
                {
                    // Reverse: edge (s→sp) becomes (sp→s) with same T, P
                    cfg.t_trans[sp][s] = dur;
                    cfg.p_trans[sp][s] = power;
                }
            }
        }

        auto spaces_rev = compute_spaces(prices_rev, cfg, spaces.banded ? spaces.max_gap : -1);
        auto prefix_rev = build_proc_prefix(prices_rev, spaces_rev.p_proc);

        return solve_relaxed_dp_lb(lengths, total_rw, prefix_rev, T, spaces_rev);
    }

    // =====================================================================
    //  bin_packing_ub: extract blocks from relaxed schedule, pack real jobs
    //
    //  1. Run relaxed DP with schedule tracking (backtrack parent pointers)
    //  2. Extract contiguous processing blocks [start, start+len)
    //  3. FFD-pack actual jobs into blocks
    //  4. Build a sequence from the packing, evaluate via solve_fixed_sequence
    //  Returns kInf if packing fails.
    // =====================================================================

    double bin_packing_ub(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces)
    {
        int K = static_cast<int>(lengths.size());
        int eff_max_gap = spaces.banded ? spaces.max_gap : T;
        int total_rw = 0;
        for (int i = 0; i < K; ++i)
            total_rw += lengths[i] * totals[i];
        int RW = total_rw + 1;

        // --- Run relaxed DP with parent tracking ---
        // parent[t][rw] = (prev_t_end, prev_rw, job_length, job_start)
        struct RPar
        {
            int prev_t;
            int prev_rw;
            int L;
            int t_s;
        };
        std::vector<double> dp((T + 2) * RW, kInf);
        std::vector<RPar> par((T + 2) * RW, {-1, -1, 0, 0});
        auto idx = [&](int t, int rw) -> int
        { return t * RW + rw; };

        // Seed
        for (int t_s = spaces.early; t_s <= spaces.late; ++t_s)
        {
            double startup = spaces.c_start[t_s];
            if (startup >= kInf)
                continue;
            for (int j = 0; j < K; ++j)
            {
                int L = lengths[j];
                if (L > total_rw)
                    continue;
                int t_e = t_s + L;
                if (t_e > T || t_e > spaces.late + 1)
                    continue;
                double cost = startup + (prefix_proc[t_e] - prefix_proc[t_s]);
                int new_rw = total_rw - L;
                int i = idx(t_e, new_rw);
                if (cost < dp[i])
                {
                    dp[i] = cost;
                    par[i] = {-1, total_rw, L, t_s};
                }
            }
        }

        // Bank for beyond-max-gap (simplified: just track best cost per rw)
        std::vector<double> bank_cost(RW, kInf);
        std::vector<RPar> bank_par(RW, {-1, -1, 0, 0});
        std::vector<int> bank_t(RW, -1);
        std::vector<std::vector<std::tuple<int, double, int>>> deferred(T + 2);
        // tuple: (rw, shutdown_cost, t_end_source)

        double best = kInf;
        int best_t = -1;

        for (int t_end = 1; t_end <= T; ++t_end)
        {
            // Phase A: absorb deferred
            if (spaces.banded && t_end < (int)deferred.size())
            {
                for (auto &[rw, cost, t_src] : deferred[t_end])
                {
                    if (cost < bank_cost[rw])
                    {
                        bank_cost[rw] = cost;
                        bank_t[rw] = t_src;
                    }
                }
                deferred[t_end].clear();
            }

            // Phase B: restart from bank
            if (spaces.banded && spaces.c_start[t_end] < kInf)
            {
                double start_cost = spaces.c_start[t_end];
                for (int rw = 1; rw < RW; ++rw)
                {
                    if (bank_cost[rw] >= kInf)
                        continue;
                    double base = bank_cost[rw] + start_cost;
                    for (int j = 0; j < K; ++j)
                    {
                        int L = lengths[j];
                        if (L > rw)
                            continue;
                        int t_e = t_end + L;
                        if (t_e > T || t_e > spaces.late + 1)
                            continue;
                        double cost = base + (prefix_proc[t_e] - prefix_proc[t_end]);
                        int i = idx(t_e, rw - L);
                        if (cost < dp[i])
                        {
                            dp[i] = cost;
                            par[i] = {bank_t[rw], rw, L, t_end};
                        }
                    }
                }
            }

            // Check complete
            double d0 = dp[idx(t_end, 0)];
            if (d0 < kInf && spaces.c_end[t_end] < kInf)
            {
                double total = d0 + spaces.c_end[t_end];
                if (total < best)
                {
                    best = total;
                    best_t = t_end;
                }
            }

            // Phase C: within-max-gap
            for (int rw = 1; rw < RW; ++rw)
            {
                double sv_cost = dp[idx(t_end, rw)];
                if (sv_cost >= kInf)
                    continue;
                int gap_limit = std::min(t_end + eff_max_gap + 1, spaces.late + 1);
                for (int t_s = t_end; t_s < gap_limit; ++t_s)
                {
                    double gap = spaces.gap_cost(t_end, t_s);
                    if (gap >= kInf)
                        continue;
                    double base = sv_cost + gap;
                    for (int j = 0; j < K; ++j)
                    {
                        int L = lengths[j];
                        if (L > rw)
                            continue;
                        int t_e = t_s + L;
                        if (t_e > T || t_e > spaces.late + 1)
                            continue;
                        double cost = base + (prefix_proc[t_e] - prefix_proc[t_s]);
                        int i = idx(t_e, rw - L);
                        if (cost < dp[i])
                        {
                            dp[i] = cost;
                            par[i] = {t_end, rw, L, t_s};
                        }
                    }
                }
            }

            // Phase D: defer shutdown
            if (spaces.banded && spaces.c_end[t_end] < kInf)
            {
                double c_end_here = spaces.c_end[t_end];
                for (int rw = 1; rw < RW; ++rw)
                {
                    double sv_cost = dp[idx(t_end, rw)];
                    if (sv_cost >= kInf)
                        continue;
                    double shutdown_cost = sv_cost + c_end_here;
                    int eligible = t_end + eff_max_gap + 1;
                    if (eligible <= T)
                        deferred[eligible].push_back({rw, shutdown_cost, t_end});
                }
            }
        }

        if (best_t < 0)
            return kInf;

        // --- Backtrack to extract processing blocks ---
        struct Block
        {
            int start;
            int length;
        };
        std::vector<Block> blocks;
        {
            int t = best_t;
            int rw = 0;
            while (true)
            {
                int i = idx(t, rw);
                const RPar &p = par[i];
                if (p.L <= 0)
                    break;
                blocks.push_back({p.t_s, p.L});
                t = p.prev_t;
                rw = p.prev_rw;
                if (t < 0)
                    break;
            }
            std::reverse(blocks.begin(), blocks.end());
        }

        if (blocks.empty())
            return kInf;

        // --- Merge adjacent/overlapping blocks into contiguous segments ---
        // The relaxed DP may have consecutive jobs with zero gap (t_s == t_prev_end).
        // Merge them into larger blocks.
        std::vector<Block> merged;
        merged.push_back(blocks[0]);
        for (size_t i = 1; i < blocks.size(); ++i)
        {
            Block &last = merged.back();
            if (blocks[i].start <= last.start + last.length)
            {
                // Extend
                int new_end = std::max(last.start + last.length, blocks[i].start + blocks[i].length);
                last.length = new_end - last.start;
            }
            else
            {
                merged.push_back(blocks[i]);
            }
        }

        // --- FFD bin-packing: pack actual jobs into merged blocks ---
        // Build list of all actual jobs, sorted decreasing (FFD)
        std::vector<int> all_jobs;
        for (int i = 0; i < K; ++i)
            for (int j = 0; j < totals[i]; ++j)
                all_jobs.push_back(lengths[i]);
        std::sort(all_jobs.begin(), all_jobs.end(), std::greater<int>());

        // remaining capacity per block
        std::vector<int> cap(merged.size());
        for (size_t i = 0; i < merged.size(); ++i)
            cap[i] = merged[i].length;

        // For each job, find first block with enough capacity (First Fit Decreasing)
        // Track the sequence of jobs per block
        std::vector<std::vector<int>> block_jobs(merged.size());
        bool feasible = true;
        for (int job_len : all_jobs)
        {
            bool placed = false;
            for (size_t b = 0; b < merged.size(); ++b)
            {
                if (cap[b] >= job_len)
                {
                    cap[b] -= job_len;
                    block_jobs[b].push_back(job_len);
                    placed = true;
                    break;
                }
            }
            if (!placed)
            {
                feasible = false;
                break;
            }
        }

        if (!feasible)
            return kInf;

        // --- Build sequence from packing (block order, jobs within each block) ---
        std::vector<int> sequence;
        for (size_t b = 0; b < merged.size(); ++b)
            for (int j : block_jobs[b])
                sequence.push_back(j);

        return solve_fixed_sequence(sequence, prefix_proc, T, spaces);
    }

    // =====================================================================
    //  local_search_ub: improve a sequence via pairwise swap hill climbing
    // =====================================================================

    double local_search_ub(
        std::vector<int> &best_seq,
        double best_cost,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        int max_passes,
        double time_budget_sec)
    {
        int n = static_cast<int>(best_seq.size());
        if (n <= 1)
            return best_cost;

        auto t0 = std::chrono::steady_clock::now();

        for (int pass = 0; pass < max_passes; ++pass)
        {
            bool improved = false;
            for (int i = 0; i < n; ++i)
            {
                // Check time budget periodically (every outer iteration)
                double elapsed = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - t0)
                                     .count();
                if (elapsed > time_budget_sec)
                    goto done;

                for (int j = i + 1; j < n; ++j)
                {
                    if (best_seq[i] == best_seq[j])
                        continue; // same length, no change
                    std::swap(best_seq[i], best_seq[j]);
                    double cost = solve_fixed_sequence(best_seq, prefix_proc, T, spaces);
                    if (cost < best_cost - kEps)
                    {
                        best_cost = cost;
                        improved = true;
                    }
                    else
                    {
                        std::swap(best_seq[i], best_seq[j]); // undo
                    }
                }
            }
            if (!improved)
                break;
        }
    done:
        return best_cost;
    }

    // =====================================================================
    //  solve_relaxed_dp_lb_two_class: tighter LB using (t, rw_small, rw_large)
    //  Split jobs into "small" (length ≤ threshold) and "large" (length > threshold).
    //  State: (t_end, rw_small, rw_large). Still allows any job from each class
    //  but prevents cross-class substitution.
    // =====================================================================

    double solve_relaxed_dp_lb_two_class(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        int threshold)
    {
        int K = static_cast<int>(lengths.size());
        int eff_max_gap = spaces.banded ? spaces.max_gap : T;

        // Classify jobs
        std::vector<int> small_lens, large_lens;
        int total_rw_small = 0, total_rw_large = 0;
        for (int i = 0; i < K; ++i)
        {
            if (lengths[i] <= threshold)
            {
                small_lens.push_back(lengths[i]);
                total_rw_small += lengths[i] * totals[i];
            }
            else
            {
                large_lens.push_back(lengths[i]);
                total_rw_large += lengths[i] * totals[i];
            }
        }

        // Deduplicate within each class
        std::sort(small_lens.begin(), small_lens.end());
        small_lens.erase(std::unique(small_lens.begin(), small_lens.end()), small_lens.end());
        std::sort(large_lens.begin(), large_lens.end());
        large_lens.erase(std::unique(large_lens.begin(), large_lens.end()), large_lens.end());

        int RS = total_rw_small + 1;
        int RL = total_rw_large + 1;

        // Check feasibility of state space
        int64_t state_space = (int64_t)(T + 2) * RS * RL;
        if (state_space > 200'000'000LL)
            return 0.0; // Too large, skip

        // Phase C work estimate: T × RS × RL × max_gap
        int64_t phase_c_work = (int64_t)T * RS * RL * eff_max_gap;
        if (phase_c_work > 500'000'000LL)
            return 0.0; // Too expensive, skip

        // dp[t * RS * RL + rs * RL + rl] = min cost
        std::vector<double> dp(state_space, kInf);
        auto idx = [&](int t, int rs, int rl) -> int64_t
        {
            return (int64_t)t * RS * RL + (int64_t)rs * RL + rl;
        };

        // Seed
        for (int t_s = spaces.early; t_s <= spaces.late; ++t_s)
        {
            double startup = spaces.c_start[t_s];
            if (startup >= kInf)
                continue;

            auto try_job = [&](int L, bool is_small)
            {
                int t_e = t_s + L;
                if (t_e > T || t_e > spaces.late + 1)
                    return;
                double cost = startup + (prefix_proc[t_e] - prefix_proc[t_s]);
                int ns = is_small ? (total_rw_small - L) : total_rw_small;
                int nl = is_small ? total_rw_large : (total_rw_large - L);
                if (ns < 0 || nl < 0)
                    return;
                auto i = idx(t_e, ns, nl);
                dp[i] = std::min(dp[i], cost);
            };

            for (int L : small_lens)
                if (L <= total_rw_small)
                    try_job(L, true);
            for (int L : large_lens)
                if (L <= total_rw_large)
                    try_job(L, false);
        }

        // Banks for beyond-max-gap
        std::vector<double> bank(RS * RL, kInf);
        auto bidx = [&](int rs, int rl) -> int
        { return rs * RL + rl; };
        std::vector<std::vector<std::pair<int, double>>> deferred(T + 2);
        // pair: (rs*RL+rl, shutdown_cost)

        double best = kInf;

        for (int t_end = 1; t_end <= T; ++t_end)
        {
            // Phase A: absorb deferred
            if (spaces.banded && t_end < (int)deferred.size())
            {
                for (auto &[bi, cost] : deferred[t_end])
                    bank[bi] = std::min(bank[bi], cost);
                deferred[t_end].clear();
            }

            // Phase B: restart from bank
            if (spaces.banded && spaces.c_start[t_end] < kInf)
            {
                double start_cost = spaces.c_start[t_end];
                for (int rs = 0; rs < RS; ++rs)
                {
                    for (int rl = 0; rl < RL; ++rl)
                    {
                        if (rs == 0 && rl == 0)
                            continue;
                        double bv = bank[bidx(rs, rl)];
                        if (bv >= kInf)
                            continue;
                        double base = bv + start_cost;

                        for (int L : small_lens)
                        {
                            if (L > rs)
                                continue;
                            int t_e = t_end + L;
                            if (t_e > T || t_e > spaces.late + 1)
                                continue;
                            double cost = base + (prefix_proc[t_e] - prefix_proc[t_end]);
                            auto i = idx(t_e, rs - L, rl);
                            dp[i] = std::min(dp[i], cost);
                        }
                        for (int L : large_lens)
                        {
                            if (L > rl)
                                continue;
                            int t_e = t_end + L;
                            if (t_e > T || t_e > spaces.late + 1)
                                continue;
                            double cost = base + (prefix_proc[t_e] - prefix_proc[t_end]);
                            auto i = idx(t_e, rs, rl - L);
                            dp[i] = std::min(dp[i], cost);
                        }
                    }
                }
            }

            // Check complete
            {
                double d0 = dp[idx(t_end, 0, 0)];
                if (d0 < kInf && spaces.c_end[t_end] < kInf)
                    best = std::min(best, d0 + spaces.c_end[t_end]);
            }

            // Phase C: within-max-gap
            for (int rs = 0; rs < RS; ++rs)
            {
                for (int rl = 0; rl < RL; ++rl)
                {
                    if (rs == 0 && rl == 0)
                        continue;
                    double sv_cost = dp[idx(t_end, rs, rl)];
                    if (sv_cost >= kInf)
                        continue;

                    int gap_limit = std::min(t_end + eff_max_gap + 1, spaces.late + 1);
                    for (int t_s = t_end; t_s < gap_limit; ++t_s)
                    {
                        double gap = spaces.gap_cost(t_end, t_s);
                        if (gap >= kInf)
                            continue;
                        double base = sv_cost + gap;

                        for (int L : small_lens)
                        {
                            if (L > rs)
                                continue;
                            int t_e = t_s + L;
                            if (t_e > T || t_e > spaces.late + 1)
                                continue;
                            double cost = base + (prefix_proc[t_e] - prefix_proc[t_s]);
                            auto i = idx(t_e, rs - L, rl);
                            dp[i] = std::min(dp[i], cost);
                        }
                        for (int L : large_lens)
                        {
                            if (L > rl)
                                continue;
                            int t_e = t_s + L;
                            if (t_e > T || t_e > spaces.late + 1)
                                continue;
                            double cost = base + (prefix_proc[t_e] - prefix_proc[t_s]);
                            auto i = idx(t_e, rs, rl - L);
                            dp[i] = std::min(dp[i], cost);
                        }
                    }
                }
            }

            // Phase D: defer shutdown
            if (spaces.banded && spaces.c_end[t_end] < kInf)
            {
                double c_end_here = spaces.c_end[t_end];
                for (int rs = 0; rs < RS; ++rs)
                {
                    for (int rl = 0; rl < RL; ++rl)
                    {
                        if (rs == 0 && rl == 0)
                            continue;
                        double sv_cost = dp[idx(t_end, rs, rl)];
                        if (sv_cost >= kInf)
                            continue;
                        double sc = sv_cost + c_end_here;
                        int eligible = t_end + eff_max_gap + 1;
                        if (eligible <= T)
                            deferred[eligible].push_back({bidx(rs, rl), sc});
                    }
                }
            }
        }

        return best;
    }

    // =====================================================================
    //  solve_exact_multiset_dp: exact DP for small K using dense
    //  (t_end, c0, c1, ..., c_{K-1}) state space.
    //  Returns the exact optimal cost (valid as both LB and UB).
    //  Skips (returns kInf) if the state space is too large.
    // =====================================================================

    double solve_exact_multiset_dp(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        double known_ub,
        double time_limit_sec)
    {
        auto t0_exact = std::chrono::steady_clock::now();
        int K = static_cast<int>(lengths.size());
        if (K == 0)
            return 0.0;

        // Compute strides (mixed-radix encoding) and total state count
        std::vector<int> strides(K);
        int NC = 1;
        for (int i = 0; i < K; ++i)
        {
            strides[i] = NC;
            if (static_cast<int64_t>(NC) * (totals[i] + 1) > 500'000)
                return kInf; // state space too large
            NC *= (totals[i] + 1);
        }

        int final_state = 0;
        int total_rw = 0;
        for (int i = 0; i < K; ++i)
        {
            final_state += totals[i] * strides[i];
            total_rw += totals[i] * lengths[i];
        }

        int eff_max_gap = spaces.banded ? spaces.max_gap : T;

        // Total cells check (memory limit ~4.8GB for doubles)
        int64_t total_cells = static_cast<int64_t>(T + 2) * NC;
        if (total_cells > 600'000'000LL)
            return kInf;

        // Precompute per-state info: remaining work + which types can be placed
        struct StateInfo
        {
            int rw;
            // For each type: can it be placed? Don't store K_MAX — use inline check
        };
        std::vector<int> state_rw(NC);
        std::vector<int> state_counts(static_cast<size_t>(NC) * K);
        for (int s = 0; s < NC; ++s)
        {
            int rw = total_rw;
            int tmp = s;
            for (int i = 0; i < K; ++i)
            {
                int ci = tmp % (totals[i] + 1);
                tmp /= (totals[i] + 1);
                state_counts[static_cast<size_t>(s) * K + i] = ci;
                rw -= ci * lengths[i];
            }
            state_rw[s] = rw;
        }

        // Dense DP array
        std::vector<double> dp(total_cells, kInf);
        auto idx = [&](int t, int s) -> int64_t
        {
            return static_cast<int64_t>(t) * NC + s;
        };

        // Proc-cost LB for pruning: sorted prefix sums
        std::vector<double> proc_prices(T);
        for (int i = 0; i < T; ++i)
            proc_prices[i] = prefix_proc[i + 1] - prefix_proc[i];

        constexpr int LB_BLOCK = 20;
        int n_blocks = (T / LB_BLOCK) + 1;
        std::vector<std::vector<double>> lb_prefix(n_blocks + 1);
        for (int bi = 0; bi < n_blocks; ++bi)
        {
            int b = bi * LB_BLOCK;
            if (b < T)
            {
                std::vector<double> sp(proc_prices.begin() + b, proc_prices.end());
                std::sort(sp.begin(), sp.end());
                lb_prefix[bi].resize(sp.size() + 1, 0.0);
                for (std::size_t i = 0; i < sp.size(); ++i)
                    lb_prefix[bi][i + 1] = lb_prefix[bi][i] + sp[i];
            }
            else
            {
                lb_prefix[bi] = {0.0};
            }
        }
        auto lb_proc_cost = [&](int t, int rw) -> double
        {
            int bi = t / LB_BLOCK;
            if (bi >= static_cast<int>(lb_prefix.size()))
                return kInf;
            const auto &arr = lb_prefix[bi];
            if (rw >= static_cast<int>(arr.size()))
                return kInf;
            return arr[rw];
        };

        // Suffix-minimum of c_end for pruning
        std::vector<double> min_c_end_from(T + 2, kInf);
        for (int t = T; t >= 0; --t)
        {
            min_c_end_from[t] = min_c_end_from[t + 1];
            if (spaces.c_end[t] < min_c_end_from[t])
                min_c_end_from[t] = spaces.c_end[t];
        }

        double best = known_ub;

        // Seed: first job from startup
        for (int t_s = spaces.early; t_s <= spaces.late; ++t_s)
        {
            double startup = spaces.c_start[t_s];
            if (startup >= kInf)
                continue;
            for (int i = 0; i < K; ++i)
            {
                int L = lengths[i];
                int t_e = t_s + L;
                if (t_e > T || t_e > spaces.late + 1)
                    continue;
                double cost = startup + (prefix_proc[t_e] - prefix_proc[t_s]);
                int new_s = strides[i]; // c_i goes from 0 to 1
                int new_rw = state_rw[new_s];
                int earliest_end = std::min(t_e + new_rw, T + 1);
                double lb = cost + lb_proc_cost(t_e, new_rw) + min_c_end_from[earliest_end];
                if (lb > best + kEps)
                    continue;
                auto di = idx(t_e, new_s);
                dp[di] = std::min(dp[di], cost);
            }
        }

        // Bank for beyond-max-gap transitions
        std::vector<double> bank(NC, kInf);
        std::vector<std::vector<std::pair<int, double>>> deferred(T + 2);

        for (int t_end = 1; t_end <= T; ++t_end)
        {
            // Time check every 32 steps
            if ((t_end & 31) == 0)
            {
                double elapsed = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - t0_exact)
                                     .count();
                if (elapsed > time_limit_sec)
                    return kInf; // timed out: cannot certify optimality
            }

            // Phase A: absorb deferred shutdown entries
            if (spaces.banded && t_end < static_cast<int>(deferred.size()))
            {
                for (auto &[si, cost] : deferred[t_end])
                    bank[si] = std::min(bank[si], cost);
                deferred[t_end].clear();
            }

            // Phase B: restart from bank
            if (spaces.banded && spaces.c_start[t_end] < kInf)
            {
                double start_cost = spaces.c_start[t_end];
                for (int s = 0; s < NC; ++s)
                {
                    if (state_rw[s] <= 0)
                        continue;
                    if (bank[s] >= kInf)
                        continue;
                    double base = bank[s] + start_cost;
                    const int *counts = &state_counts[static_cast<size_t>(s) * K];
                    for (int i = 0; i < K; ++i)
                    {
                        if (counts[i] >= totals[i])
                            continue;
                        int L = lengths[i];
                        int t_e = t_end + L;
                        if (t_e > T || t_e > spaces.late + 1)
                            continue;
                        double cost = base + (prefix_proc[t_e] - prefix_proc[t_end]);
                        int new_s = s + strides[i];
                        int new_rw = state_rw[new_s];
                        int earliest_end = std::min(t_e + new_rw, T + 1);
                        double lb = cost + lb_proc_cost(t_e, new_rw) + min_c_end_from[earliest_end];
                        if (lb > best + kEps)
                            continue;
                        auto di = idx(t_e, new_s);
                        dp[di] = std::min(dp[di], cost);
                    }
                }
            }

            // Check for complete solutions
            {
                double d = dp[idx(t_end, final_state)];
                if (d < kInf && spaces.c_end[t_end] < kInf)
                {
                    double total = d + spaces.c_end[t_end];
                    if (total < best)
                        best = total;
                }
            }

            // Phase C: within-max-gap transitions
            int base_offset = t_end * NC;
            for (int s = 0; s < NC; ++s)
            {
                if (state_rw[s] <= 0)
                    continue;
                double sv = dp[base_offset + s];
                if (sv >= kInf)
                    continue;

                const int *counts = &state_counts[static_cast<size_t>(s) * K];
                int gap_limit = std::min(t_end + eff_max_gap + 1, spaces.late + 1);
                for (int t_s = t_end; t_s < gap_limit; ++t_s)
                {
                    double gap = spaces.gap_cost(t_end, t_s);
                    if (gap >= kInf)
                        continue;
                    double base = sv + gap;

                    for (int i = 0; i < K; ++i)
                    {
                        if (counts[i] >= totals[i])
                            continue;
                        int L = lengths[i];
                        int t_e = t_s + L;
                        if (t_e > T || t_e > spaces.late + 1)
                            continue;
                        double cost = base + (prefix_proc[t_e] - prefix_proc[t_s]);
                        int new_s = s + strides[i];
                        int new_rw = state_rw[new_s];
                        int earliest_end = std::min(t_e + new_rw, T + 1);
                        double lb = cost + lb_proc_cost(t_e, new_rw) + min_c_end_from[earliest_end];
                        if (lb > best + kEps)
                            continue;
                        auto di = idx(t_e, new_s);
                        dp[di] = std::min(dp[di], cost);
                    }
                }
            }

            // Phase D: defer shutdown entries
            if (spaces.banded && spaces.c_end[t_end] < kInf)
            {
                double c_end_here = spaces.c_end[t_end];
                for (int s = 0; s < NC; ++s)
                {
                    if (state_rw[s] <= 0)
                        continue;
                    double sv = dp[base_offset + s];
                    if (sv >= kInf)
                        continue;
                    double sc = sv + c_end_here;
                    int eligible = t_end + eff_max_gap + 1;
                    if (eligible <= T)
                        deferred[eligible].push_back({s, sc});
                }
            }
        }

        return best;
    }

    // =====================================================================
    //  solve_sparse_exact_multiset_dp: exact DP using hash maps for large
    //  state spaces where the dense approach is infeasible.
    //  Uses two rolling hash maps (current time slice + bank) to store
    //  only reachable states.
    //  NC limit: 10M (theoretical), but practical limit is memory/time.
    //  Time limit enforced to avoid runaway computation.
    // =====================================================================

    double solve_sparse_exact_multiset_dp(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        double known_ub,
        double time_limit_sec,
        const std::vector<double> *relaxed_dp,
        int relaxed_RW,
        double relaxed_lb)
    {
        auto t0 = std::chrono::steady_clock::now();
        int K = static_cast<int>(lengths.size());
        if (K == 0)
            return 0.0;

        // Compute strides (mixed-radix encoding) for state indexing
        std::vector<int64_t> strides(K);
        int64_t NC = 1;
        for (int i = 0; i < K; ++i)
        {
            strides[i] = NC;
            NC *= (totals[i] + 1);
            if (NC > 10'000'000LL)
                return kInf; // too many theoretical states
        }

        int64_t final_state = 0;
        int total_rw = 0;
        for (int i = 0; i < K; ++i)
        {
            final_state += static_cast<int64_t>(totals[i]) * strides[i];
            total_rw += totals[i] * lengths[i];
        }

        int eff_max_gap = spaces.banded ? spaces.max_gap : T;

        // Helper: decode state to get remaining work and per-type counts
        auto state_rw = [&](int64_t s) -> int
        {
            int rw = total_rw;
            int64_t tmp = s;
            for (int i = 0; i < K; ++i)
            {
                int ci = static_cast<int>(tmp % (totals[i] + 1));
                tmp /= (totals[i] + 1);
                rw -= ci * lengths[i];
            }
            return rw;
        };

        auto state_count = [&](int64_t s, int i) -> int
        {
            int64_t tmp = s;
            for (int j = 0; j < i; ++j)
                tmp /= (totals[j] + 1);
            return static_cast<int>(tmp % (totals[i] + 1));
        };
        auto relaxed_cost = [&](int t, int rw) -> double
        {
            if (!relaxed_dp)
                return 0.0;
            if (t < 0 || t > T + 1 || rw < 0 || rw >= relaxed_RW)
                return kInf;
            return (*relaxed_dp)[static_cast<size_t>(t) * relaxed_RW + rw];
        };

        // Proc-cost LB for pruning (sorted cheapest slots)
        std::vector<double> proc_prices(T);
        for (int i = 0; i < T; ++i)
            proc_prices[i] = prefix_proc[i + 1] - prefix_proc[i];
        std::vector<double> sorted_prices = proc_prices;
        std::sort(sorted_prices.begin(), sorted_prices.end());
        std::vector<double> sorted_prefix(T + 1, 0.0);
        for (int i = 0; i < T; ++i)
            sorted_prefix[i + 1] = sorted_prefix[i] + sorted_prices[i];
        auto lb_proc_cost = [&](int rw) -> double
        {
            if (rw <= 0)
                return 0.0;
            if (rw > T)
                return kInf;
            return sorted_prefix[rw];
        };

        // Suffix-minimum of c_end for pruning
        std::vector<double> min_c_end_from(T + 2, kInf);
        for (int t = T; t >= 0; --t)
        {
            min_c_end_from[t] = min_c_end_from[t + 1];
            if (spaces.c_end[t] < min_c_end_from[t])
                min_c_end_from[t] = spaces.c_end[t];
        }

        double best = known_ub;

        // Use vector of hash maps: one per time step (only allocated if needed)
        // To control memory, we process one time step at a time and keep a
        // "frontier" of active states at each t_end.
        // dp_at[t_end] = map of state -> cost
        // But storing all T maps is too much. Instead, process forward and
        // propagate into future time steps using a sparse structure.

        // Approach: sweep t_end from 0..T. At each t_end, process all states
        // that END at t_end. For each, try all gap+job extensions.
        // Use a vector of maps indexed by t_end.

        // Memory control: limit total entries across all maps
        constexpr int64_t MAX_TOTAL_ENTRIES = 50'000'000LL;
        int64_t total_entries = 0;

        std::vector<std::unordered_map<int64_t, double>> dp_maps(T + 2);
        std::unordered_map<int64_t, double> bank; // for beyond-max-gap restarts
        std::vector<std::vector<std::pair<int64_t, double>>> deferred(T + 2);

        // Seed: first job from startup
        for (int t_s = spaces.early; t_s <= spaces.late; ++t_s)
        {
            double startup = spaces.c_start[t_s];
            if (startup >= kInf)
                continue;
            for (int i = 0; i < K; ++i)
            {
                int L = lengths[i];
                int t_e = t_s + L;
                if (t_e > T || t_e > spaces.late + 1)
                    continue;
                double cost = startup + (prefix_proc[t_e] - prefix_proc[t_s]);
                int64_t new_s = strides[i];
                int new_rw = state_rw(new_s);
                int earliest_end = std::min(t_e + new_rw, T + 1);
                double lb = cost + lb_proc_cost(new_rw) + min_c_end_from[earliest_end];
                if (lb > best + kEps)
                    continue;
                if (relaxed_dp)
                {
                    double rdp_val = relaxed_cost(t_e, new_rw);
                    if (rdp_val >= kInf)
                        continue;
                    if (relaxed_lb < kInf && cost - rdp_val > best - relaxed_lb + kEps)
                        continue;
                }
                auto &m = dp_maps[t_e];
                auto it = m.find(new_s);
                if (it == m.end())
                {
                    m[new_s] = cost;
                    ++total_entries;
                }
                else if (cost < it->second)
                {
                    it->second = cost;
                }
            }
        }

        bool exhaustive = true;
        for (int t_end = 1; t_end <= T; ++t_end)
        {
            // Time check every 16 steps
            if ((t_end & 15) == 0)
            {
                double elapsed = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - t0)
                                     .count();
                if (elapsed > time_limit_sec)
                { exhaustive = false; break; }
                if (total_entries > MAX_TOTAL_ENTRIES)
                { exhaustive = false; break; }
            }

            // Phase A: absorb deferred shutdown entries into bank
            if (spaces.banded && t_end < static_cast<int>(deferred.size()))
            {
                for (auto &[si, cost] : deferred[t_end])
                {
                    auto it = bank.find(si);
                    if (it == bank.end())
                        bank[si] = cost;
                    else if (cost < it->second)
                        it->second = cost;
                }
                deferred[t_end].clear();
            }

            // Phase B: restart from bank
            if (spaces.banded && spaces.c_start[t_end] < kInf)
            {
                double start_cost = spaces.c_start[t_end];
                for (auto &[s, bcost] : bank)
                {
                    if (state_rw(s) <= 0)
                        continue;
                    double base = bcost + start_cost;
                    for (int i = 0; i < K; ++i)
                    {
                        if (state_count(s, i) >= totals[i])
                            continue;
                        int L = lengths[i];
                        int t_e = t_end + L;
                        if (t_e > T || t_e > spaces.late + 1)
                            continue;
                        double cost = base + (prefix_proc[t_e] - prefix_proc[t_end]);
                        int64_t new_s = s + strides[i];
                        int new_rw = state_rw(new_s);
                        int earliest_end = std::min(t_e + new_rw, T + 1);
                        double lb = cost + lb_proc_cost(new_rw) + min_c_end_from[earliest_end];
                        if (lb > best + kEps)
                            continue;
                        if (relaxed_dp)
                        {
                            double rdp_val = relaxed_cost(t_e, new_rw);
                            if (rdp_val >= kInf)
                                continue;
                            if (relaxed_lb < kInf && cost - rdp_val > best - relaxed_lb + kEps)
                                continue;
                        }
                        auto &m = dp_maps[t_e];
                        auto it = m.find(new_s);
                        if (it == m.end())
                        {
                            m[new_s] = cost;
                            ++total_entries;
                        }
                        else if (cost < it->second)
                        {
                            it->second = cost;
                        }
                    }
                }
            }

            // Check for complete solutions
            {
                auto &m = dp_maps[t_end];
                auto it = m.find(final_state);
                if (it != m.end() && it->second < kInf && spaces.c_end[t_end] < kInf)
                {
                    double total = it->second + spaces.c_end[t_end];
                    if (total < best)
                        best = total;
                }
            }

            // Phase C: within-max-gap transitions
            auto &cur_map = dp_maps[t_end];
            for (auto &[s, sv] : cur_map)
            {
                if (state_rw(s) <= 0)
                    continue;
                if (sv >= kInf)
                    continue;

                int gap_limit = std::min(t_end + eff_max_gap + 1, spaces.late + 1);
                for (int t_s = t_end; t_s < gap_limit; ++t_s)
                {
                    double gap = spaces.gap_cost(t_end, t_s);
                    if (gap >= kInf)
                        continue;
                    double base = sv + gap;

                    for (int i = 0; i < K; ++i)
                    {
                        if (state_count(s, i) >= totals[i])
                            continue;
                        int L = lengths[i];
                        int t_e = t_s + L;
                        if (t_e > T || t_e > spaces.late + 1)
                            continue;
                        double cost = base + (prefix_proc[t_e] - prefix_proc[t_s]);
                        int64_t new_s = s + strides[i];
                        int new_rw = state_rw(new_s);
                        int earliest_end = std::min(t_e + new_rw, T + 1);
                        double lb = cost + lb_proc_cost(new_rw) + min_c_end_from[earliest_end];
                        if (lb > best + kEps)
                            continue;
                        if (relaxed_dp)
                        {
                            double rdp_val = relaxed_cost(t_e, new_rw);
                            if (rdp_val >= kInf)
                                continue;
                            if (relaxed_lb < kInf && cost - rdp_val > best - relaxed_lb + kEps)
                                continue;
                        }
                        auto &m = dp_maps[t_e];
                        auto it2 = m.find(new_s);
                        if (it2 == m.end())
                        {
                            m[new_s] = cost;
                            ++total_entries;
                        }
                        else if (cost < it2->second)
                        {
                            it2->second = cost;
                        }
                    }
                }
            }

            // Phase D: defer shutdown entries for beyond-max-gap restart
            if (spaces.banded && spaces.c_end[t_end] < kInf)
            {
                double c_end_here = spaces.c_end[t_end];
                for (auto &[s, sv] : cur_map)
                {
                    if (state_rw(s) <= 0)
                        continue;
                    if (sv >= kInf)
                        continue;
                    double sc = sv + c_end_here;
                    int eligible = t_end + eff_max_gap + 1;
                    if (eligible <= T)
                        deferred[eligible].push_back({s, sc});
                }
            }

            // Free processed time step to reclaim memory
            cur_map.clear();
            total_entries -= 0; // approximate (we cleared but don't track exactly)
        }

        return exhaustive ? best : kInf;
    }

    // =====================================================================
    //  compute_feas_sets: precompute A_j^- for each type j.
    //  A_j^-[w] = true iff work amount w is achievable using any
    //  allocation (a_1,...,a_K) with Σ a_i*p_i = w and a_j ≤ n_j - 1.
    //
    //  Method: bounded knapsack DP.
    //  For type j, we run a standard bounded subset-sum DP with counts
    //  (n_1,...,n_{j-1}, n_j - 1, n_{j+1},...,n_K).
    //  Complexity: O(K * W * max(n_i)) per type → O(K^2 * W * max(n_i)) total.
    //  In practice W ≤ 500, K ≤ 10, max(n_i) ≤ 200 → fast.
    // =====================================================================

    std::vector<std::vector<bool>> compute_feas_sets(
        const std::vector<int> &lengths,
        const std::vector<int> &totals)
    {
        int K = static_cast<int>(lengths.size());
        int W = 0;
        for (int i = 0; i < K; ++i)
            W += lengths[i] * totals[i];

        std::vector<std::vector<bool>> feas(K, std::vector<bool>(W + 1, false));

        for (int j = 0; j < K; ++j)
        {
            // Bounded subset sum with modified counts: n_j replaced by n_j - 1
            // Use standard DP: dp[w] = true if achievable
            std::vector<bool> dp_reach(W + 1, false);
            dp_reach[0] = true;

            for (int i = 0; i < K; ++i)
            {
                int L = lengths[i];
                int cap = (i == j) ? (totals[i] - 1) : totals[i];
                if (cap <= 0)
                    continue;

                // Process type i with up to cap copies using binary decomposition
                // for efficiency: decompose cap into powers of 2 + remainder
                int remaining = cap;
                int group = 1;
                while (remaining > 0)
                {
                    int take = std::min(group, remaining);
                    int weight = take * L;
                    // Add this grouped item (0-1 knapsack step, backward)
                    for (int w = W; w >= weight; --w)
                    {
                        if (dp_reach[w - weight])
                            dp_reach[w] = true;
                    }
                    remaining -= take;
                    group *= 2;
                }
            }

            feas[j] = dp_reach;
        }

        return feas;
    }

    std::vector<bool> compute_bounded_work_set(
        const std::vector<int> &lengths,
        const std::vector<int> &totals)
    {
        int K = static_cast<int>(lengths.size());
        int W = 0;
        for (int i = 0; i < K; ++i)
            W += lengths[i] * totals[i];

        std::vector<bool> reach(W + 1, false);
        reach[0] = true;

        for (int i = 0; i < K; ++i)
        {
            int L = lengths[i];
            int remaining = totals[i];
            int group = 1;
            while (remaining > 0)
            {
                int take = std::min(group, remaining);
                int weight = take * L;
                for (int w = W; w >= weight; --w)
                {
                    if (reach[w - weight])
                        reach[w] = true;
                }
                remaining -= take;
                group *= 2;
            }
        }

        return reach;
    }

    // =====================================================================
    //  solve_relaxed_dp_lb_feas: R_feas — relaxed DP with bounded-work and
    //  two-sided transition filtering.
    //
    //  A transition by type j from remaining work rw to rw-L_j is allowed only
    //  if:
    //    1) the already-placed work W-rw is globally bounded-reachable,
    //    2) the prefix can still leave one type-j job unused, and
    //    3) the suffix rw-L_j is achievable under the residual bounded counts.
    //
    //  This is stronger than the original one-sided R_feas but still not exact,
    //  because multiple count vectors still project to the same rw value.
    // =====================================================================

    double solve_relaxed_dp_lb_feas(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces)
    {
        int K = static_cast<int>(lengths.size());
        int eff_max_gap = spaces.banded ? spaces.max_gap : T;
        int total_rw = 0;
        for (int i = 0; i < K; ++i)
            total_rw += lengths[i] * totals[i];

        // Precompute bounded-work reachability and per-type reduced-capacity
        // reachability sets.
        auto bounded_work = compute_bounded_work_set(lengths, totals);
        auto feas = compute_feas_sets(lengths, totals);

        auto can_place = [&](int j, int rw, int next_rw) -> bool
        {
            int placed = total_rw - rw;
            if (placed < 0 || placed > total_rw)
                return false;
            if (next_rw < 0 || next_rw > total_rw)
                return false;
            if (!bounded_work[placed])
                return false;
            if (!feas[j][placed])
                return false;
            if (!feas[j][next_rw])
                return false;
            return true;
        };

        int RW = total_rw + 1;
        std::vector<double> dp((T + 2) * RW, kInf);
        auto idx = [&](int t, int rw) -> int
        { return t * RW + rw; };

        // Seed: first job from startup
        for (int t_s = spaces.early; t_s <= spaces.late; ++t_s)
        {
            double startup = spaces.c_start[t_s];
            if (startup >= kInf)
                continue;
            for (int j = 0; j < K; ++j)
            {
                int L = lengths[j];
                if (L > total_rw)
                    continue;
                int new_rw = total_rw - L;
                if (!can_place(j, total_rw, new_rw))
                    continue;
                int t_e = t_s + L;
                if (t_e > T || t_e > spaces.late + 1)
                    continue;
                double cost = startup + (prefix_proc[t_e] - prefix_proc[t_s]);
                double &ref = dp[idx(t_e, new_rw)];
                ref = std::min(ref, cost);
            }
        }

        // Bank for beyond-max-gap transitions
        std::vector<double> bank(RW, kInf);
        std::vector<std::vector<std::pair<int, double>>> deferred(T + 2);

        double best = kInf;

        for (int t_end = 1; t_end <= T; ++t_end)
        {
            // Phase A: absorb deferred shutdown entries into bank
            if (spaces.banded && t_end < static_cast<int>(deferred.size()))
            {
                for (auto &[rw, cost] : deferred[t_end])
                    bank[rw] = std::min(bank[rw], cost);
                deferred[t_end].clear();
            }

            // Phase B: restart from bank at t_end
            if (spaces.banded && spaces.c_start[t_end] < kInf)
            {
                double start_cost = spaces.c_start[t_end];
                for (int rw = 1; rw < RW; ++rw)
                {
                    if (bank[rw] >= kInf)
                        continue;
                    double base = bank[rw] + start_cost;
                    for (int j = 0; j < K; ++j)
                    {
                        int L = lengths[j];
                        if (L > rw)
                            continue;
                        int next_rw = rw - L;
                        if (!can_place(j, rw, next_rw))
                            continue;
                        int t_e = t_end + L;
                        if (t_e > T || t_e > spaces.late + 1)
                            continue;
                        double cost = base + (prefix_proc[t_e] - prefix_proc[t_end]);
                        double &ref = dp[idx(t_e, next_rw)];
                        ref = std::min(ref, cost);
                    }
                }
            }

            // Check for complete solutions at t_end
            double d0 = dp[idx(t_end, 0)];
            if (d0 < kInf && spaces.c_end[t_end] < kInf)
                best = std::min(best, d0 + spaces.c_end[t_end]);

            // Phase C: within-max-gap transitions
            for (int rw = 1; rw < RW; ++rw)
            {
                double sv_cost = dp[idx(t_end, rw)];
                if (sv_cost >= kInf)
                    continue;
                int gap_limit = std::min(t_end + eff_max_gap + 1, spaces.late + 1);
                for (int t_s = t_end; t_s < gap_limit; ++t_s)
                {
                    double gap = spaces.gap_cost(t_end, t_s);
                    if (gap >= kInf)
                        continue;
                    double base = sv_cost + gap;
                    for (int j = 0; j < K; ++j)
                    {
                        int L = lengths[j];
                        if (L > rw)
                            continue;
                        int next_rw = rw - L;
                        if (!can_place(j, rw, next_rw))
                            continue;
                        int t_e = t_s + L;
                        if (t_e > T || t_e > spaces.late + 1)
                            continue;
                        double cost = base + (prefix_proc[t_e] - prefix_proc[t_s]);
                        double &ref = dp[idx(t_e, next_rw)];
                        ref = std::min(ref, cost);
                    }
                }
            }

            // Phase D: defer shutdown entries
            if (spaces.banded && spaces.c_end[t_end] < kInf)
            {
                double c_end_here = spaces.c_end[t_end];
                for (int rw = 1; rw < RW; ++rw)
                {
                    double sv_cost = dp[idx(t_end, rw)];
                    if (sv_cost >= kInf)
                        continue;
                    double shutdown_cost = sv_cost + c_end_here;
                    int eligible = t_end + eff_max_gap + 1;
                    if (eligible <= T)
                        deferred[eligible].push_back({rw, shutdown_cost});
                }
            }
        }

        return best;
    }

    // =====================================================================
    //  solve_relaxed_dp_lb_lagrangian: Lagrangian relaxation of per-type
    //  count constraints over the R_semi relaxed DP.
    //
    //  The exact DP constrains: at most n_j copies of type j.
    //  We dualize this: L(λ) = min_{(t,rw) paths} [path_cost + Σ_j λ_j (count_j - n_j)]
    //  At λ=0 this is R_semi. The dual max_{λ≥0} L(λ) is tighter.
    //
    //  Each iteration:
    //  1. Run modified R_semi with edge costs increased by λ_j per type j
    //  2. Backtrack optimal path to count type usage
    //  3. Update λ via subgradient: λ_j = max(0, λ_j + step*(count_j - n_j))
    //
    //  Returns the best LB found across all iterations.
    // =====================================================================

    double solve_relaxed_dp_lb_lagrangian(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        int max_iters,
        double time_limit_sec)
    {
        using Clock = std::chrono::steady_clock;
        auto t0 = Clock::now();

        int K = static_cast<int>(lengths.size());
        int eff_max_gap = spaces.banded ? spaces.max_gap : T;
        int total_rw = 0;
        for (int i = 0; i < K; ++i)
            total_rw += lengths[i] * totals[i];
        int RW = total_rw + 1;

        // Lagrangian multipliers (one per type)
        std::vector<double> lambda(K, 0.0);
        double best_lb = 0.0;
        double best_ub = kInf; // track best known UB for step size

        // We'll need parent tracking to backtrack and count type usage
        struct RPar
        {
            int prev_t;
            int prev_rw;
            int type_idx; // which type was placed (-1 for seed)
        };

        for (int iter = 0; iter < max_iters; ++iter)
        {
            double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
            if (elapsed > time_limit_sec)
                break;

            // Run modified relaxed DP with per-type cost offsets lambda[j]
            std::vector<double> dp(static_cast<size_t>(T + 2) * RW, kInf);
            std::vector<RPar> par(static_cast<size_t>(T + 2) * RW, {-1, -1, -1});
            auto idx = [&](int t, int rw) -> int
            { return t * RW + rw; };

            // Seed
            for (int t_s = spaces.early; t_s <= spaces.late; ++t_s)
            {
                double startup = spaces.c_start[t_s];
                if (startup >= kInf)
                    continue;
                for (int j = 0; j < K; ++j)
                {
                    int L = lengths[j];
                    if (L > total_rw)
                        continue;
                    int t_e = t_s + L;
                    if (t_e > T || t_e > spaces.late + 1)
                        continue;
                    double cost = startup + (prefix_proc[t_e] - prefix_proc[t_s]) + lambda[j];
                    int new_rw = total_rw - L;
                    int i = idx(t_e, new_rw);
                    if (cost < dp[i])
                    {
                        dp[i] = cost;
                        par[i] = {-1, total_rw, j};
                    }
                }
            }

            // Bank for beyond-max-gap
            std::vector<double> bank_cost(RW, kInf);
            std::vector<int> bank_t(RW, -1);
            std::vector<std::vector<std::tuple<int, double, int>>> deferred(T + 2);

            double iter_best = kInf;
            int iter_best_t = -1;

            for (int t_end = 1; t_end <= T; ++t_end)
            {
                // Phase A
                if (spaces.banded && t_end < (int)deferred.size())
                {
                    for (auto &[rw, cost, t_src] : deferred[t_end])
                        if (cost < bank_cost[rw])
                        {
                            bank_cost[rw] = cost;
                            bank_t[rw] = t_src;
                        }
                    deferred[t_end].clear();
                }

                // Phase B: restart from bank
                if (spaces.banded && spaces.c_start[t_end] < kInf)
                {
                    double start_cost = spaces.c_start[t_end];
                    for (int rw = 1; rw < RW; ++rw)
                    {
                        if (bank_cost[rw] >= kInf)
                            continue;
                        double base = bank_cost[rw] + start_cost;
                        for (int j = 0; j < K; ++j)
                        {
                            int L = lengths[j];
                            if (L > rw)
                                continue;
                            int t_e = t_end + L;
                            if (t_e > T || t_e > spaces.late + 1)
                                continue;
                            double cost = base + (prefix_proc[t_e] - prefix_proc[t_end]) + lambda[j];
                            int i = idx(t_e, rw - L);
                            if (cost < dp[i])
                            {
                                dp[i] = cost;
                                par[i] = {bank_t[rw], rw, j};
                            }
                        }
                    }
                }

                // Check complete
                double d0 = dp[idx(t_end, 0)];
                if (d0 < kInf && spaces.c_end[t_end] < kInf)
                {
                    double total = d0 + spaces.c_end[t_end];
                    if (total < iter_best)
                    {
                        iter_best = total;
                        iter_best_t = t_end;
                    }
                }

                // Phase C
                for (int rw = 1; rw < RW; ++rw)
                {
                    double sv_cost = dp[idx(t_end, rw)];
                    if (sv_cost >= kInf)
                        continue;
                    int gap_limit = std::min(t_end + eff_max_gap + 1, spaces.late + 1);
                    for (int t_s = t_end; t_s < gap_limit; ++t_s)
                    {
                        double gap = spaces.gap_cost(t_end, t_s);
                        if (gap >= kInf)
                            continue;
                        double base = sv_cost + gap;
                        for (int j = 0; j < K; ++j)
                        {
                            int L = lengths[j];
                            if (L > rw)
                                continue;
                            int t_e = t_s + L;
                            if (t_e > T || t_e > spaces.late + 1)
                                continue;
                            double cost = base + (prefix_proc[t_e] - prefix_proc[t_s]) + lambda[j];
                            int i = idx(t_e, rw - L);
                            if (cost < dp[i])
                            {
                                dp[i] = cost;
                                par[i] = {t_end, rw, j};
                            }
                        }
                    }
                }

                // Phase D
                if (spaces.banded && spaces.c_end[t_end] < kInf)
                {
                    double c_end_here = spaces.c_end[t_end];
                    for (int rw = 1; rw < RW; ++rw)
                    {
                        double sv_cost = dp[idx(t_end, rw)];
                        if (sv_cost >= kInf)
                            continue;
                        int eligible = t_end + eff_max_gap + 1;
                        if (eligible <= T)
                            deferred[eligible].push_back({rw, sv_cost + c_end_here, t_end});
                    }
                }
            }

            if (iter_best >= kInf)
                continue; // infeasible under current lambda, skip

            // Compute Lagrangian LB: L(λ) = iter_best - Σ_j λ_j * n_j
            double lagr_lb = iter_best;
            for (int j = 0; j < K; ++j)
                lagr_lb -= lambda[j] * totals[j];

            best_lb = std::max(best_lb, lagr_lb);

            // Backtrack to count type usage
            std::vector<int> type_count(K, 0);
            if (iter_best_t >= 0)
            {
                int t = iter_best_t;
                int rw = 0;
                while (true)
                {
                    int i = idx(t, rw);
                    const RPar &p = par[i];
                    if (p.type_idx < 0)
                        break;
                    type_count[p.type_idx]++;
                    t = p.prev_t;
                    rw = p.prev_rw;
                    if (t < 0)
                        break;
                }
            }

            // Subgradient: g_j = count_j - n_j
            // Step size: Polyak's rule: step = α * (UB - L(λ)) / ||g||^2
            // If no UB known, use a diminishing step
            double sq_norm = 0.0;
            std::vector<double> grad(K);
            bool all_feasible = true;
            for (int j = 0; j < K; ++j)
            {
                grad[j] = static_cast<double>(type_count[j] - totals[j]);
                sq_norm += grad[j] * grad[j];
                if (type_count[j] > totals[j])
                    all_feasible = false;
            }

            if (all_feasible)
            {
                // The relaxed solution is also feasible for the original problem
                // iter_best (minus lambda adjustments) is a valid UB too
                // But the actual cost without lambda is different — we'd need to
                // re-evaluate. For now, the LB is solid.
                // If all counts are exactly n_j, we're at optimum
                bool exact_match = true;
                for (int j = 0; j < K; ++j)
                    if (type_count[j] != totals[j])
                    { exact_match = false; break; }
                if (exact_match)
                    break; // optimal found
            }

            if (sq_norm < 1e-12)
                break; // converged

            // Polyak step size with estimated UB
            double alpha = 1.5 / (1.0 + iter * 0.05); // diminishing factor
            double step;
            if (best_ub < kInf)
                step = alpha * (best_ub - lagr_lb) / sq_norm;
            else
                step = alpha * std::max(1.0, std::abs(lagr_lb) * 0.01) / sq_norm;

            // Update multipliers
            for (int j = 0; j < K; ++j)
                lambda[j] = std::max(0.0, lambda[j] + step * grad[j]);
        }

        return best_lb;
    }

    // =====================================================================
    //  solve_relaxed_dp_lb_feas_lagrangian: Combined bound.
    //  Uses both feasibility filtering (R_feas) AND Lagrangian penalties
    //  (R_Lagr) simultaneously. This is the tightest (t,rw) bound.
    // =====================================================================

    double solve_relaxed_dp_lb_feas_lagrangian(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        int max_iters,
        double time_limit_sec)
    {
        using Clock = std::chrono::steady_clock;
        auto t0 = Clock::now();

        int K = static_cast<int>(lengths.size());
        int eff_max_gap = spaces.banded ? spaces.max_gap : T;
        int total_rw = 0;
        for (int i = 0; i < K; ++i)
            total_rw += lengths[i] * totals[i];
        int RW = total_rw + 1;

        // Precompute bounded-work reachability and reduced-capacity reachability.
        auto bounded_work = compute_bounded_work_set(lengths, totals);
        auto feas = compute_feas_sets(lengths, totals);
        auto can_place = [&](int j, int rw, int next_rw) -> bool {
            int placed = total_rw - rw;
            return placed >= 0 && placed < (int)feas[j].size() &&
                   next_rw >= 0 && next_rw < (int)feas[j].size() &&
                   bounded_work[placed] && feas[j][placed] && feas[j][next_rw];
        };

        std::vector<double> lambda(K, 0.0);
        double best_lb = 0.0;
        double best_ub = kInf;

        struct RPar { int prev_t; int prev_rw; int type_idx; };

        for (int iter = 0; iter < max_iters; ++iter)
        {
            double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
            if (elapsed > time_limit_sec)
                break;

            std::vector<double> dp(static_cast<size_t>(T + 2) * RW, kInf);
            std::vector<RPar> par(static_cast<size_t>(T + 2) * RW, {-1, -1, -1});
            auto idx = [&](int t, int rw) -> int { return t * RW + rw; };

            // Seed
            for (int t_s = spaces.early; t_s <= spaces.late; ++t_s)
            {
                double startup = spaces.c_start[t_s];
                if (startup >= kInf) continue;
                for (int j = 0; j < K; ++j)
                {
                    int L = lengths[j];
                    if (L > total_rw) continue;
                    int new_rw = total_rw - L;
                    if (!can_place(j, total_rw, new_rw)) continue; // feas filter
                    int t_e = t_s + L;
                    if (t_e > T || t_e > spaces.late + 1) continue;
                    double cost = startup + (prefix_proc[t_e] - prefix_proc[t_s]) + lambda[j];
                    int i = idx(t_e, new_rw);
                    if (cost < dp[i]) { dp[i] = cost; par[i] = {-1, total_rw, j}; }
                }
            }

            std::vector<double> bank_cost(RW, kInf);
            std::vector<int> bank_t(RW, -1);
            std::vector<std::vector<std::tuple<int, double, int>>> deferred(T + 2);

            double iter_best = kInf;
            int iter_best_t = -1;

            for (int t_end = 1; t_end <= T; ++t_end)
            {
                // Phase A
                if (spaces.banded && t_end < (int)deferred.size())
                {
                    for (auto &[rw, cost, t_src] : deferred[t_end])
                        if (cost < bank_cost[rw]) { bank_cost[rw] = cost; bank_t[rw] = t_src; }
                    deferred[t_end].clear();
                }

                // Phase B: restart from bank
                if (spaces.banded && spaces.c_start[t_end] < kInf)
                {
                    double start_cost = spaces.c_start[t_end];
                    for (int rw = 1; rw < RW; ++rw)
                    {
                        if (bank_cost[rw] >= kInf) continue;
                        double base = bank_cost[rw] + start_cost;
                        for (int j = 0; j < K; ++j)
                        {
                            int L = lengths[j];
                            if (L > rw) continue;
                            int next_rw = rw - L;
                            if (!can_place(j, rw, next_rw)) continue; // feas filter
                            int t_e = t_end + L;
                            if (t_e > T || t_e > spaces.late + 1) continue;
                            double cost = base + (prefix_proc[t_e] - prefix_proc[t_end]) + lambda[j];
                            int i = idx(t_e, next_rw);
                            if (cost < dp[i]) { dp[i] = cost; par[i] = {bank_t[rw], rw, j}; }
                        }
                    }
                }

                // Check complete
                double d0 = dp[idx(t_end, 0)];
                if (d0 < kInf && spaces.c_end[t_end] < kInf)
                {
                    double total = d0 + spaces.c_end[t_end];
                    if (total < iter_best) { iter_best = total; iter_best_t = t_end; }
                }

                // Phase C
                for (int rw = 1; rw < RW; ++rw)
                {
                    double sv_cost = dp[idx(t_end, rw)];
                    if (sv_cost >= kInf) continue;
                    int gap_limit = std::min(t_end + eff_max_gap + 1, spaces.late + 1);
                    for (int t_s = t_end; t_s < gap_limit; ++t_s)
                    {
                        double gap = spaces.gap_cost(t_end, t_s);
                        if (gap >= kInf) continue;
                        double base = sv_cost + gap;
                        for (int j = 0; j < K; ++j)
                        {
                            int L = lengths[j];
                            if (L > rw) continue;
                            int next_rw = rw - L;
                            if (!can_place(j, rw, next_rw)) continue; // feas filter
                            int t_e = t_s + L;
                            if (t_e > T || t_e > spaces.late + 1) continue;
                            double cost = base + (prefix_proc[t_e] - prefix_proc[t_s]) + lambda[j];
                            int i = idx(t_e, next_rw);
                            if (cost < dp[i]) { dp[i] = cost; par[i] = {t_end, rw, j}; }
                        }
                    }
                }

                // Phase D
                if (spaces.banded && spaces.c_end[t_end] < kInf)
                {
                    double c_end_here = spaces.c_end[t_end];
                    for (int rw = 1; rw < RW; ++rw)
                    {
                        double sv_cost = dp[idx(t_end, rw)];
                        if (sv_cost >= kInf) continue;
                        int eligible = t_end + eff_max_gap + 1;
                        if (eligible <= T)
                            deferred[eligible].push_back({rw, sv_cost + c_end_here, t_end});
                    }
                }
            }

            if (iter_best >= kInf) continue;

            double lagr_lb = iter_best;
            for (int j = 0; j < K; ++j)
                lagr_lb -= lambda[j] * totals[j];
            best_lb = std::max(best_lb, lagr_lb);

            // Backtrack to count type usage
            std::vector<int> type_count(K, 0);
            if (iter_best_t >= 0)
            {
                int t = iter_best_t, rw = 0;
                while (true)
                {
                    int i = idx(t, rw);
                    const RPar &p = par[i];
                    if (p.type_idx < 0) break;
                    type_count[p.type_idx]++;
                    t = p.prev_t; rw = p.prev_rw;
                    if (t < 0) break;
                }
            }

            double sq_norm = 0.0;
            std::vector<double> grad(K);
            bool all_feasible = true;
            for (int j = 0; j < K; ++j)
            {
                grad[j] = static_cast<double>(type_count[j] - totals[j]);
                sq_norm += grad[j] * grad[j];
                if (type_count[j] > totals[j]) all_feasible = false;
            }

            if (all_feasible)
            {
                bool exact_match = true;
                for (int j = 0; j < K; ++j)
                    if (type_count[j] != totals[j]) { exact_match = false; break; }
                if (exact_match) break;
            }

            if (sq_norm < 1e-12) break;

            double alpha = 1.5 / (1.0 + iter * 0.05);
            double step;
            if (best_ub < kInf)
                step = alpha * (best_ub - lagr_lb) / sq_norm;
            else
                step = alpha * std::max(1.0, std::abs(lagr_lb) * 0.01) / sq_norm;

            for (int j = 0; j < K; ++j)
                lambda[j] = std::max(0.0, lambda[j] + step * grad[j]);
        }

        return best_lb;
    }

    static std::vector<int> select_partial_tracked_types(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        int max_auto_tracked)
    {
        int K = static_cast<int>(lengths.size());
        max_auto_tracked = std::max(0, std::min(max_auto_tracked, 2));
        if (K == 0 || max_auto_tracked == 0)
            return {};

        int eff_max_gap = spaces.banded ? spaces.max_gap : T;
        int total_rw = 0;
        for (int i = 0; i < K; ++i)
            total_rw += lengths[i] * totals[i];
        int RW = total_rw + 1;

        struct RPar
        {
            int prev_t;
            int prev_rw;
            int type_idx;
        };

        std::vector<double> dp(static_cast<size_t>(T + 2) * RW, kInf);
        std::vector<RPar> par(static_cast<size_t>(T + 2) * RW, {-1, -1, -1});
        auto idx = [&](int t, int rw) -> int
        { return t * RW + rw; };

        for (int t_s = spaces.early; t_s <= spaces.late; ++t_s)
        {
            double startup = spaces.c_start[t_s];
            if (startup >= kInf)
                continue;
            for (int j = 0; j < K; ++j)
            {
                int L = lengths[j];
                if (L > total_rw)
                    continue;
                int t_e = t_s + L;
                if (t_e > T || t_e > spaces.late + 1)
                    continue;
                double cost = startup + (prefix_proc[t_e] - prefix_proc[t_s]);
                int new_rw = total_rw - L;
                int i = idx(t_e, new_rw);
                if (cost < dp[i])
                {
                    dp[i] = cost;
                    par[i] = {-1, total_rw, j};
                }
            }
        }

        std::vector<double> bank_cost(RW, kInf);
        std::vector<int> bank_t(RW, -1);
        std::vector<std::vector<std::tuple<int, double, int>>> deferred(T + 2);

        double best = kInf;
        int best_t = -1;
        for (int t_end = 1; t_end <= T; ++t_end)
        {
            if (spaces.banded && t_end < static_cast<int>(deferred.size()))
            {
                for (auto &[rw, cost, t_src] : deferred[t_end])
                {
                    if (cost < bank_cost[rw])
                    {
                        bank_cost[rw] = cost;
                        bank_t[rw] = t_src;
                    }
                }
                deferred[t_end].clear();
            }

            if (spaces.banded && spaces.c_start[t_end] < kInf)
            {
                double start_cost = spaces.c_start[t_end];
                for (int rw = 1; rw < RW; ++rw)
                {
                    if (bank_cost[rw] >= kInf)
                        continue;
                    double base = bank_cost[rw] + start_cost;
                    for (int j = 0; j < K; ++j)
                    {
                        int L = lengths[j];
                        if (L > rw)
                            continue;
                        int t_e = t_end + L;
                        if (t_e > T || t_e > spaces.late + 1)
                            continue;
                        double cost = base + (prefix_proc[t_e] - prefix_proc[t_end]);
                        int i = idx(t_e, rw - L);
                        if (cost < dp[i])
                        {
                            dp[i] = cost;
                            par[i] = {bank_t[rw], rw, j};
                        }
                    }
                }
            }

            double d0 = dp[idx(t_end, 0)];
            if (d0 < kInf && spaces.c_end[t_end] < kInf)
            {
                double total = d0 + spaces.c_end[t_end];
                if (total < best)
                {
                    best = total;
                    best_t = t_end;
                }
            }

            for (int rw = 1; rw < RW; ++rw)
            {
                double sv_cost = dp[idx(t_end, rw)];
                if (sv_cost >= kInf)
                    continue;
                int gap_limit = std::min(t_end + eff_max_gap + 1, spaces.late + 1);
                for (int t_s = t_end; t_s < gap_limit; ++t_s)
                {
                    double gap = spaces.gap_cost(t_end, t_s);
                    if (gap >= kInf)
                        continue;
                    double base = sv_cost + gap;
                    for (int j = 0; j < K; ++j)
                    {
                        int L = lengths[j];
                        if (L > rw)
                            continue;
                        int t_e = t_s + L;
                        if (t_e > T || t_e > spaces.late + 1)
                            continue;
                        double cost = base + (prefix_proc[t_e] - prefix_proc[t_s]);
                        int i = idx(t_e, rw - L);
                        if (cost < dp[i])
                        {
                            dp[i] = cost;
                            par[i] = {t_end, rw, j};
                        }
                    }
                }
            }

            if (spaces.banded && spaces.c_end[t_end] < kInf)
            {
                double c_end_here = spaces.c_end[t_end];
                for (int rw = 1; rw < RW; ++rw)
                {
                    double sv_cost = dp[idx(t_end, rw)];
                    if (sv_cost >= kInf)
                        continue;
                    int eligible = t_end + eff_max_gap + 1;
                    if (eligible <= T)
                        deferred[eligible].push_back({rw, sv_cost + c_end_here, t_end});
                }
            }
        }

        std::vector<int> usage(K, 0);
        if (best_t >= 0)
        {
            int t = best_t;
            int rw = 0;
            while (true)
            {
                int i = idx(t, rw);
                const RPar &p = par[i];
                if (p.type_idx < 0)
                    break;
                usage[p.type_idx]++;
                t = p.prev_t;
                rw = p.prev_rw;
                if (t < 0)
                    break;
            }
        }

        std::vector<int> by_overuse(K);
        std::iota(by_overuse.begin(), by_overuse.end(), 0);
        std::sort(by_overuse.begin(), by_overuse.end(), [&](int a, int b)
        {
            int excess_a = usage[a] - totals[a];
            int excess_b = usage[b] - totals[b];
            bool pos_a = excess_a > 0;
            bool pos_b = excess_b > 0;
            if (pos_a != pos_b)
                return pos_a > pos_b;
            double rel_a = pos_a ? static_cast<double>(excess_a) / std::max(1, totals[a]) : 0.0;
            double rel_b = pos_b ? static_cast<double>(excess_b) / std::max(1, totals[b]) : 0.0;
            if (std::fabs(rel_a - rel_b) > 1e-12)
                return rel_a > rel_b;
            if (excess_a != excess_b)
                return excess_a > excess_b;
            if (totals[a] != totals[b])
                return totals[a] < totals[b];
            return lengths[a] < lengths[b];
        });

        std::vector<int> by_scarcity(K);
        std::iota(by_scarcity.begin(), by_scarcity.end(), 0);
        std::sort(by_scarcity.begin(), by_scarcity.end(), [&](int a, int b)
        {
            if (totals[a] != totals[b])
                return totals[a] < totals[b];
            return lengths[a] < lengths[b];
        });

        std::vector<int> chosen;
        for (int j : by_overuse)
        {
            if (usage[j] <= totals[j])
                continue;
            chosen.push_back(j);
            if (static_cast<int>(chosen.size()) >= max_auto_tracked)
                break;
        }
        for (int j : by_scarcity)
        {
            if (static_cast<int>(chosen.size()) >= max_auto_tracked)
                break;
            if (std::find(chosen.begin(), chosen.end(), j) != chosen.end())
                continue;
            chosen.push_back(j);
        }
        return chosen;
    }

    // =====================================================================
    //  solve_relaxed_dp_lb_partial: R_partial — partial count-vector
    //  relaxation.
    //
    //  Tracks 1-2 critical type counts exactly; the rest are pooled into
    //  a single remaining-work scalar. The relaxed remainder is further
    //  filtered through bounded/two-sided feasibility checks.
    //
    //  State with 1 tracked type:
    //    (t_end, c_tracked, rw_rest)
    //    where c_tracked ∈ [0, n_tracked], rw_rest ∈ [0, W_rest]
    //
    //  State with 2 tracked types:
    //    (t_end, c0, c1, rw_rest)
    //    where c0 ∈ [0, n0], c1 ∈ [0, n1], rw_rest ∈ [0, W_rest]
    //
    //  Hierarchy target:  R_feas ≤ R_partial(1) ≤ R_partial(2) ≤ Exact
    //
    //  Single forward sweep after one optional diagnostic pass for automatic
    //  tracked-type selection.
    // =====================================================================

    double solve_relaxed_dp_lb_partial(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        std::vector<int> tracked_types,
        double time_limit_sec,
        int max_auto_tracked,
        bool use_remainder_feas)
    {
        using Clock = std::chrono::steady_clock;
        auto t0 = Clock::now();

        int K = static_cast<int>(lengths.size());
        if (K == 0)
            return 0.0;

        int eff_max_gap = spaces.banded ? spaces.max_gap : T;

        max_auto_tracked = std::max(0, std::min(max_auto_tracked, 2));

        // ── Auto-select tracked types if not specified ──────────────
        if (tracked_types.empty())
        {
            tracked_types = select_partial_tracked_types(
                lengths, totals, prefix_proc, T, spaces, max_auto_tracked);
        }

        // Clamp to at most 2 tracked types
        if (tracked_types.size() > static_cast<size_t>(max_auto_tracked))
            tracked_types.resize(max_auto_tracked);
        if (tracked_types.size() > 2)
            tracked_types.resize(2);

        // ── Set up tracked vs relaxed types ────────────────────────
        int n_tracked = static_cast<int>(tracked_types.size());
        // is_tracked[j] = true if type j is tracked exactly
        std::vector<bool> is_tracked(K, false);
        for (int j : tracked_types)
            is_tracked[j] = true;

        // Compute rest-work (from non-tracked types)
        int rw_rest_total = 0;
        for (int j = 0; j < K; ++j)
            if (!is_tracked[j])
                rw_rest_total += lengths[j] * totals[j];

        // Feasibility structures for the relaxed remainder only.
        std::vector<int> rest_lengths;
        std::vector<int> rest_totals;
        std::vector<int> rest_local_idx(K, -1);
        rest_lengths.reserve(K);
        rest_totals.reserve(K);
        for (int j = 0; j < K; ++j)
        {
            if (is_tracked[j])
                continue;
            rest_local_idx[j] = static_cast<int>(rest_lengths.size());
            rest_lengths.push_back(lengths[j]);
            rest_totals.push_back(totals[j]);
        }
        std::vector<bool> rest_bounded;
        std::vector<std::vector<bool>> rest_feas;
        if (use_remainder_feas)
        {
            rest_bounded = compute_bounded_work_set(rest_lengths, rest_totals);
            rest_feas = compute_feas_sets(rest_lengths, rest_totals);
        }
        auto can_place_rest = [&](int j, int rr, int next_rr) -> bool
        {
            int local = rest_local_idx[j];
            if (local < 0)
                return false;
            int placed_rest = rw_rest_total - rr;
            if (placed_rest < 0 || placed_rest > rw_rest_total)
                return false;
            if (next_rr < 0 || next_rr > rw_rest_total)
                return false;
            if (!use_remainder_feas)
                return true;
            if (!rest_bounded[placed_rest])
                return false;
            if (!rest_feas[local][placed_rest])
                return false;
            if (!rest_feas[local][next_rr])
                return false;
            return true;
        };

        // Dimensions for tracked counts
        int nc0 = (n_tracked >= 1) ? (totals[tracked_types[0]] + 1) : 1;
        int nc1 = (n_tracked >= 2) ? (totals[tracked_types[1]] + 1) : 1;
        int RW = rw_rest_total + 1;

        int64_t cells_per_t = static_cast<int64_t>(nc0) * nc1 * RW;
        int64_t total_cells = static_cast<int64_t>(T + 2) * cells_per_t;

        // Bail out if state space is too large
        if (total_cells > 300'000'000LL)
        {
            std::cerr << "[R_partial] State space too large ("
                      << total_cells << " cells), skipping.\n";
            return kInf;
        }

        // ── Indexing ───────────────────────────────────────────────
        // dp[(t * nc0 * nc1 + c0 * nc1 + c1) * RW + rr]
        auto idx = [&](int t, int c0, int c1, int rr) -> int64_t
        {
            return (static_cast<int64_t>(t) * nc0 * nc1 +
                    static_cast<int64_t>(c0) * nc1 + c1) *
                       RW +
                   rr;
        };

        std::vector<double> dp(total_cells, kInf);

        // ── Proc-cost lower bound for pruning ─────────────────────
        std::vector<double> proc_prices(T);
        for (int i = 0; i < T; ++i)
            proc_prices[i] = prefix_proc[i + 1] - prefix_proc[i];
        std::vector<double> sorted_prices = proc_prices;
        std::sort(sorted_prices.begin(), sorted_prices.end());
        std::vector<double> sorted_prefix(T + 1, 0.0);
        for (int i = 0; i < T; ++i)
            sorted_prefix[i + 1] = sorted_prefix[i] + sorted_prices[i];
        auto lb_proc_cost = [&](int work) -> double
        {
            if (work <= 0) return 0.0;
            if (work > T) return kInf;
            return sorted_prefix[work];
        };

        // Suffix-minimum of c_end for pruning
        std::vector<double> min_c_end_from(T + 2, kInf);
        for (int t = T; t >= 0; --t)
        {
            min_c_end_from[t] = min_c_end_from[t + 1];
            if (spaces.c_end[t] < min_c_end_from[t])
                min_c_end_from[t] = spaces.c_end[t];
        }

        double best = kInf;
        bool exhausted = true;

        // ── Helper to compute total remaining work from state ──────
        auto state_total_rw = [&](int c0, int c1, int rr) -> int
        {
            int rw = rr; // remaining from non-tracked
            if (n_tracked >= 1)
                rw += (totals[tracked_types[0]] - c0) * lengths[tracked_types[0]];
            if (n_tracked >= 2)
                rw += (totals[tracked_types[1]] - c1) * lengths[tracked_types[1]];
            return rw;
        };

        // ── Pruning helper ────────────────────────────────────────
        auto prune = [&](double cost, int t_e, int c0, int c1, int rr) -> bool
        {
            int rw = state_total_rw(c0, c1, rr);
            int earliest_end = std::min(t_e + rw, T + 1);
            double lb = cost + lb_proc_cost(rw) + min_c_end_from[earliest_end];
            return lb > best + kEps;
        };

        // ── Helper to try updating a DP cell ──────────────────────
        auto try_update = [&](int t_e, int c0, int c1, int rr, double cost)
        {
            if (prune(cost, t_e, c0, c1, rr))
                return;
            int64_t i = idx(t_e, c0, c1, rr);
            if (cost < dp[i])
                dp[i] = cost;
        };

        // ── Seed: first job from startup ──────────────────────────
        for (int t_s = spaces.early; t_s <= spaces.late; ++t_s)
        {
            double startup = spaces.c_start[t_s];
            if (startup >= kInf)
                continue;
            for (int j = 0; j < K; ++j)
            {
                int L = lengths[j];
                int t_e = t_s + L;
                if (t_e > T || t_e > spaces.late + 1)
                    continue;
                double cost = startup + (prefix_proc[t_e] - prefix_proc[t_s]);

                int c0 = 0, c1 = 0, rr = rw_rest_total;
                if (n_tracked >= 1 && j == tracked_types[0])
                {
                    c0 = 1;
                    if (c0 > totals[tracked_types[0]])
                        continue;
                }
                else if (n_tracked >= 2 && j == tracked_types[1])
                {
                    c1 = 1;
                    if (c1 > totals[tracked_types[1]])
                        continue;
                }
                                else
                                {
                                    // Non-tracked type: reduce rw_rest
                                    rr = rw_rest_total - L;
                                    if (rr < 0)
                                        continue;
                                    if (!can_place_rest(j, rw_rest_total, rr))
                                        continue;
                                }
                                try_update(t_e, c0, c1, rr, cost);
                            }
        }

        // ── Bank for beyond-max-gap transitions ───────────────────
        // Bank state: (c0, c1, rr) → min cost after shutdown
        int64_t bank_cells = static_cast<int64_t>(nc0) * nc1 * RW;
        std::vector<double> bank(bank_cells, kInf);
        auto bank_idx = [&](int c0, int c1, int rr) -> int64_t
        {
            return (static_cast<int64_t>(c0) * nc1 + c1) * RW + rr;
        };

        // Deferred shutdown entries: eligible_time → list of (c0, c1, rr, cost)
        struct DeferredEntry
        {
            int c0, c1, rr;
            double cost;
        };
        std::vector<std::vector<DeferredEntry>> deferred(T + 2);

        // ── Main forward sweep ────────────────────────────────────
        for (int t_end = 1; t_end <= T; ++t_end)
        {
            // Time check
            if ((t_end & 63) == 0)
            {
                double elapsed = std::chrono::duration<double>(
                                     Clock::now() - t0)
                                     .count();
                if (elapsed > time_limit_sec)
                {
                    exhausted = false;
                    break;
                }
            }

            // Phase A: absorb deferred shutdown entries into bank
            if (spaces.banded && t_end < static_cast<int>(deferred.size()))
            {
                for (auto &e : deferred[t_end])
                {
                    int64_t bi = bank_idx(e.c0, e.c1, e.rr);
                    if (e.cost < bank[bi])
                        bank[bi] = e.cost;
                }
                deferred[t_end].clear();
            }

            // Phase B: restart from bank at t_end
            if (spaces.banded && spaces.c_start[t_end] < kInf)
            {
                double start_cost = spaces.c_start[t_end];
                for (int c0 = 0; c0 < nc0; ++c0)
                    for (int c1 = 0; c1 < nc1; ++c1)
                        for (int rr = 0; rr < RW; ++rr)
                        {
                            int64_t bi = bank_idx(c0, c1, rr);
                            if (bank[bi] >= kInf)
                                continue;
                            // Must have remaining work
                            if (state_total_rw(c0, c1, rr) <= 0)
                                continue;
                            double base = bank[bi] + start_cost;

                            for (int j = 0; j < K; ++j)
                            {
                                int L = lengths[j];
                                int t_e = t_end + L;
                                if (t_e > T || t_e > spaces.late + 1)
                                    continue;
                                double cost = base + (prefix_proc[t_e] - prefix_proc[t_end]);

                                int nc0_ = c0, nc1_ = c1, nrr = rr;
                                if (n_tracked >= 1 && j == tracked_types[0])
                                {
                                    nc0_ = c0 + 1;
                                    if (nc0_ > totals[tracked_types[0]])
                                        continue;
                                }
                                else if (n_tracked >= 2 && j == tracked_types[1])
                                {
                                    nc1_ = c1 + 1;
                                    if (nc1_ > totals[tracked_types[1]])
                                        continue;
                                }
                                else
                                {
                                    nrr = rr - L;
                                    if (nrr < 0)
                                        continue;
                                    if (!can_place_rest(j, rr, nrr))
                                        continue;
                                }
                                try_update(t_e, nc0_, nc1_, nrr, cost);
                            }
                        }
            }

            // Check for complete solutions at t_end
            {
                int final_c0 = (n_tracked >= 1) ? totals[tracked_types[0]] : 0;
                int final_c1 = (n_tracked >= 2) ? totals[tracked_types[1]] : 0;
                int64_t i = idx(t_end, final_c0, final_c1, 0);
                if (dp[i] < kInf && spaces.c_end[t_end] < kInf)
                {
                    double total = dp[i] + spaces.c_end[t_end];
                    if (total < best)
                        best = total;
                }
            }

            // Phase C: within-max-gap transitions
            for (int c0 = 0; c0 < nc0; ++c0)
                for (int c1 = 0; c1 < nc1; ++c1)
                    for (int rr = 0; rr < RW; ++rr)
                    {
                        int64_t si = idx(t_end, c0, c1, rr);
                        double sv_cost = dp[si];
                        if (sv_cost >= kInf)
                            continue;
                        if (state_total_rw(c0, c1, rr) <= 0)
                            continue;

                        int gap_limit = std::min(t_end + eff_max_gap + 1,
                                                 spaces.late + 1);
                        for (int t_s = t_end; t_s < gap_limit; ++t_s)
                        {
                            double gap = spaces.gap_cost(t_end, t_s);
                            if (gap >= kInf)
                                continue;
                            double base = sv_cost + gap;

                            for (int j = 0; j < K; ++j)
                            {
                                int L = lengths[j];
                                int t_e = t_s + L;
                                if (t_e > T || t_e > spaces.late + 1)
                                    continue;
                                double cost = base +
                                              (prefix_proc[t_e] - prefix_proc[t_s]);

                                int nc0_ = c0, nc1_ = c1, nrr = rr;
                                if (n_tracked >= 1 && j == tracked_types[0])
                                {
                                    nc0_ = c0 + 1;
                                    if (nc0_ > totals[tracked_types[0]])
                                        continue;
                                }
                                else if (n_tracked >= 2 && j == tracked_types[1])
                                {
                                    nc1_ = c1 + 1;
                                    if (nc1_ > totals[tracked_types[1]])
                                        continue;
                                }
                                else
                                {
                                    nrr = rr - L;
                                    if (nrr < 0)
                                        continue;
                                    if (!can_place_rest(j, rr, nrr))
                                        continue;
                                }
                                try_update(t_e, nc0_, nc1_, nrr, cost);
                            }
                        }
                    }

            // Phase D: defer shutdown entries for beyond-max-gap restart
            if (spaces.banded && spaces.c_end[t_end] < kInf)
            {
                double c_end_here = spaces.c_end[t_end];
                for (int c0 = 0; c0 < nc0; ++c0)
                    for (int c1 = 0; c1 < nc1; ++c1)
                        for (int rr = 0; rr < RW; ++rr)
                        {
                            int64_t si = idx(t_end, c0, c1, rr);
                            if (dp[si] >= kInf)
                                continue;
                            if (state_total_rw(c0, c1, rr) <= 0)
                                continue;
                            double sc = dp[si] + c_end_here;
                            int eligible = t_end + eff_max_gap + 1;
                            if (eligible <= T)
                                deferred[eligible].push_back(
                                    {c0, c1, rr, sc});
                        }
            }
        }

        return exhausted ? best : kInf;
    }

    RelaxedDPResult solve_relaxed_dp_lb_partial_with_binpack(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        std::vector<int> tracked_types,
        int max_auto_tracked,
        bool use_remainder_feas)
    {
        int K = static_cast<int>(lengths.size());
        if (K == 0)
        {
            RelaxedDPResult out;
            out.lb = 0.0;
            out.bin_pack_ub = 0.0;
            out.pack_outcome = "no_blocks";
            return out;
        }

        max_auto_tracked = std::max(0, std::min(max_auto_tracked, 1));
        if (tracked_types.empty())
            tracked_types = select_partial_tracked_types(
                lengths, totals, prefix_proc, T, spaces, max_auto_tracked);
        if (tracked_types.empty())
        {
            int scarce = 0;
            for (int j = 1; j < K; ++j)
                if (totals[j] < totals[scarce] ||
                    (totals[j] == totals[scarce] && lengths[j] < lengths[scarce]))
                    scarce = j;
            tracked_types.push_back(scarce);
        }
        if (tracked_types.size() > 1)
            tracked_types.resize(1);

        int tracked = tracked_types[0];
        int n_tracked = totals[tracked];
        int nc0 = n_tracked + 1;

        int rw_rest_total = 0;
        std::vector<int> rest_lengths;
        std::vector<int> rest_totals;
        std::vector<int> rest_local_idx(K, -1);
        for (int j = 0; j < K; ++j)
        {
            if (j == tracked)
                continue;
            rest_local_idx[j] = static_cast<int>(rest_lengths.size());
            rest_lengths.push_back(lengths[j]);
            rest_totals.push_back(totals[j]);
            rw_rest_total += lengths[j] * totals[j];
        }
        int RW = rw_rest_total + 1;

        std::vector<bool> rest_bounded;
        std::vector<std::vector<bool>> rest_feas;
        if (use_remainder_feas)
        {
            rest_bounded = compute_bounded_work_set(rest_lengths, rest_totals);
            rest_feas = compute_feas_sets(rest_lengths, rest_totals);
        }
        auto can_place_rest = [&](int j, int rr, int next_rr) -> bool
        {
            int local = rest_local_idx[j];
            if (local < 0)
                return false;
            if (!use_remainder_feas)
                return next_rr >= 0 && next_rr <= rw_rest_total;
            int placed_rest = rw_rest_total - rr;
            if (placed_rest < 0 || placed_rest > rw_rest_total)
                return false;
            if (next_rr < 0 || next_rr > rw_rest_total)
                return false;
            return rest_bounded[placed_rest] &&
                   rest_feas[local][placed_rest] &&
                   rest_feas[local][next_rr];
        };

        int eff_max_gap = spaces.banded ? spaces.max_gap : T;
        int64_t cells_per_t = static_cast<int64_t>(nc0) * RW;
        int64_t total_cells = static_cast<int64_t>(T + 2) * cells_per_t;
        if (total_cells > 300'000'000LL)
        {
            RelaxedDPResult out;
            out.lb = kInf;
            out.bin_pack_ub = kInf;
            out.pack_outcome = "no_relaxed_path";
            return out;
        }

        struct PPar
        {
            int prev_t;
            int prev_c0;
            int prev_rr;
            int L;
            int t_s;
        };

        auto idx = [&](int t, int c0, int rr) -> int64_t
        {
            return (static_cast<int64_t>(t) * nc0 + c0) * RW + rr;
        };
        auto total_rw = [&](int c0, int rr) -> int
        {
            return rr + (n_tracked - c0) * lengths[tracked];
        };

        std::vector<double> dp(total_cells, kInf);
        std::vector<PPar> par(total_cells, {-1, -1, -1, 0, 0});
        int64_t states_reached = 0;
        int64_t states_expanded = 0;

        std::vector<double> proc_prices(T);
        for (int i = 0; i < T; ++i)
            proc_prices[i] = prefix_proc[i + 1] - prefix_proc[i];
        std::vector<double> sorted_prices = proc_prices;
        std::sort(sorted_prices.begin(), sorted_prices.end());
        std::vector<double> sorted_prefix(T + 1, 0.0);
        for (int i = 0; i < T; ++i)
            sorted_prefix[i + 1] = sorted_prefix[i] + sorted_prices[i];
        auto lb_proc_cost = [&](int work) -> double
        {
            if (work <= 0)
                return 0.0;
            if (work > T)
                return kInf;
            return sorted_prefix[work];
        };

        std::vector<double> min_c_end_from(T + 2, kInf);
        for (int t = T; t >= 0; --t)
        {
            min_c_end_from[t] = min_c_end_from[t + 1];
            if (spaces.c_end[t] < min_c_end_from[t])
                min_c_end_from[t] = spaces.c_end[t];
        }

        double best = kInf;
        int best_t = -1;

        auto prune = [&](double cost, int t_e, int c0, int rr) -> bool
        {
            int rw = total_rw(c0, rr);
            int earliest_end = std::min(t_e + rw, T + 1);
            double lb = cost + lb_proc_cost(rw) + min_c_end_from[earliest_end];
            return lb > best + kEps;
        };

        auto try_update = [&](int t_e, int c0, int rr, double cost, int prev_t, int prev_c0, int prev_rr, int L, int t_s)
        {
            if (prune(cost, t_e, c0, rr))
                return;
            int64_t i = idx(t_e, c0, rr);
            if (cost < dp[i])
            {
                if (dp[i] >= kInf)
                    ++states_reached;
                dp[i] = cost;
                par[i] = {prev_t, prev_c0, prev_rr, L, t_s};
            }
        };

        for (int t_s = spaces.early; t_s <= spaces.late; ++t_s)
        {
            double startup = spaces.c_start[t_s];
            if (startup >= kInf)
                continue;
            for (int j = 0; j < K; ++j)
            {
                int L = lengths[j];
                int t_e = t_s + L;
                if (t_e > T || t_e > spaces.late + 1)
                    continue;
                double cost = startup + (prefix_proc[t_e] - prefix_proc[t_s]);
                if (j == tracked)
                {
                    if (n_tracked <= 0)
                        continue;
                    try_update(t_e, 1, rw_rest_total, cost, -1, n_tracked, rw_rest_total, L, t_s);
                }
                else
                {
                    int rr = rw_rest_total - L;
                    if (rr < 0 || !can_place_rest(j, rw_rest_total, rr))
                        continue;
                    try_update(t_e, 0, rr, cost, -1, n_tracked, rw_rest_total, L, t_s);
                }
            }
        }

        int64_t bank_cells = static_cast<int64_t>(nc0) * RW;
        std::vector<double> bank(bank_cells, kInf);
        std::vector<int> bank_t(bank_cells, -1);
        auto bank_idx = [&](int c0, int rr) -> int64_t
        {
            return static_cast<int64_t>(c0) * RW + rr;
        };
        std::vector<std::vector<std::tuple<int, int, double, int>>> deferred(T + 2);

        for (int t_end = 1; t_end <= T; ++t_end)
        {
            if (spaces.banded && t_end < static_cast<int>(deferred.size()))
            {
                for (auto &[c0, rr, cost, src_t] : deferred[t_end])
                {
                    int64_t bi = bank_idx(c0, rr);
                    if (cost < bank[bi])
                    {
                        bank[bi] = cost;
                        bank_t[bi] = src_t;
                    }
                }
                deferred[t_end].clear();
            }

            if (spaces.banded && spaces.c_start[t_end] < kInf)
            {
                double start_cost = spaces.c_start[t_end];
                for (int c0 = 0; c0 < nc0; ++c0)
                    for (int rr = 0; rr < RW; ++rr)
                    {
                        int64_t bi = bank_idx(c0, rr);
                        if (bank[bi] >= kInf || total_rw(c0, rr) <= 0)
                            continue;
                        double base = bank[bi] + start_cost;
                        for (int j = 0; j < K; ++j)
                        {
                            int L = lengths[j];
                            int t_e = t_end + L;
                            if (t_e > T || t_e > spaces.late + 1)
                                continue;
                            double cost = base + (prefix_proc[t_e] - prefix_proc[t_end]);
                            if (j == tracked)
                            {
                                if (c0 + 1 > n_tracked)
                                    continue;
                                try_update(t_e, c0 + 1, rr, cost, bank_t[bi], c0, rr, L, t_end);
                            }
                            else
                            {
                                int nrr = rr - L;
                                if (nrr < 0 || !can_place_rest(j, rr, nrr))
                                    continue;
                                try_update(t_e, c0, nrr, cost, bank_t[bi], c0, rr, L, t_end);
                            }
                        }
                    }
            }

            int64_t fi = idx(t_end, n_tracked, 0);
            if (dp[fi] < kInf && spaces.c_end[t_end] < kInf)
            {
                double total = dp[fi] + spaces.c_end[t_end];
                if (total < best)
                {
                    best = total;
                    best_t = t_end;
                }
            }

            for (int c0 = 0; c0 < nc0; ++c0)
                for (int rr = 0; rr < RW; ++rr)
                {
                    int64_t si = idx(t_end, c0, rr);
                    double sv_cost = dp[si];
                    if (sv_cost >= kInf || total_rw(c0, rr) <= 0)
                        continue;
                    ++states_expanded;
                    int gap_limit = std::min(t_end + eff_max_gap + 1, spaces.late + 1);
                    for (int t_s = t_end; t_s < gap_limit; ++t_s)
                    {
                        double gap = spaces.gap_cost(t_end, t_s);
                        if (gap >= kInf)
                            continue;
                        double base = sv_cost + gap;
                        for (int j = 0; j < K; ++j)
                        {
                            int L = lengths[j];
                            int t_e = t_s + L;
                            if (t_e > T || t_e > spaces.late + 1)
                                continue;
                            double cost = base + (prefix_proc[t_e] - prefix_proc[t_s]);
                            if (j == tracked)
                            {
                                if (c0 + 1 > n_tracked)
                                    continue;
                                try_update(t_e, c0 + 1, rr, cost, t_end, c0, rr, L, t_s);
                            }
                            else
                            {
                                int nrr = rr - L;
                                if (nrr < 0 || !can_place_rest(j, rr, nrr))
                                    continue;
                                try_update(t_e, c0, nrr, cost, t_end, c0, rr, L, t_s);
                            }
                        }
                    }
                }

            if (spaces.banded && spaces.c_end[t_end] < kInf)
            {
                double c_end_here = spaces.c_end[t_end];
                for (int c0 = 0; c0 < nc0; ++c0)
                    for (int rr = 0; rr < RW; ++rr)
                    {
                        int64_t si = idx(t_end, c0, rr);
                        if (dp[si] >= kInf || total_rw(c0, rr) <= 0)
                            continue;
                        int eligible = t_end + eff_max_gap + 1;
                        if (eligible <= T)
                            deferred[eligible].push_back({c0, rr, dp[si] + c_end_here, t_end});
                    }
            }
        }

        RecoveredBlockPackingResult pack;
        pack.pack_outcome = (best_t >= 0 ? "not_attempted" : "no_relaxed_path");
        if (best_t >= 0)
        {
            std::vector<RecoveredBlock> blocks;
            int t = best_t;
            int c0 = n_tracked;
            int rr = 0;
            while (true)
            {
                int64_t i = idx(t, c0, rr);
                const PPar &p = par[i];
                if (p.L <= 0)
                    break;
                blocks.push_back({p.t_s, p.L});
                t = p.prev_t;
                c0 = p.prev_c0;
                rr = p.prev_rr;
                if (t < 0)
                    break;
            }
            std::reverse(blocks.begin(), blocks.end());
            pack = pack_recovered_blocks(blocks, lengths, totals, prefix_proc, T, spaces);
        }

        RelaxedDPResult result;
        result.lb = best;
        result.bin_pack_ub = pack.bin_pack_ub;
        result.states_reached = states_reached;
        result.states_expanded = states_expanded;
        result.rdp = std::move(dp);
        result.RW = RW;
        result.block_count = pack.block_count;
        result.merged_block_count = pack.merged_block_count;
        result.pack_solver = pack.pack_solver;
        result.pack_external_status = pack.pack_external_status;
        result.pack_method = pack.pack_method;
        result.pack_outcome = pack.pack_outcome;
        result.t_pack_external = pack.t_pack_external;
        result.t_pack_heuristic = pack.t_pack_heuristic;
        result.t_pack_dfs = pack.t_pack_dfs;
        result.t_pack_block_dp = pack.t_pack_block_dp;
        return result;
    }

} // namespace dp

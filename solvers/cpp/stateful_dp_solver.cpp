#include "stateful_dp_solver.hpp"

#include <array>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
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
        ExactDPDiagnostics g_last_exact_dp_diag;
    }

    ExactDPDiagnostics consume_last_exact_dp_diagnostics()
    {
        ExactDPDiagnostics out = g_last_exact_dp_diag;
        g_last_exact_dp_diag = ExactDPDiagnostics{};
        return out;
    }

    // PLAN24: beam-guided exact corridor
    namespace
    {
        ExactCorridor g_exact_corridor;
    }

    void set_exact_corridor(const ExactCorridor &corridor)
    {
        g_exact_corridor = corridor;
    }

    void clear_exact_corridor()
    {
        g_exact_corridor = ExactCorridor{};
    }

    namespace
    {
        bool check_exact_corridor_counts(const int *counts, int placed_work, int K)
        {
            if (!g_exact_corridor.enabled)
                return true;
            const auto &cor = g_exact_corridor;
            int b = 0;
            int PB = static_cast<int>(cor.prefix_work.size());
            while (b + 1 < PB && placed_work >= cor.prefix_work[b + 1])
                ++b;
            for (int j = 0; j < K; ++j)
            {
                int lo = cor.prefix_counts[b][j] - cor.delta;
                int hi = (b + 1 < PB) ? cor.prefix_counts[b + 1][j] + cor.delta
                                      : cor.prefix_counts.back()[j] + cor.delta;
                if (counts[j] < lo || counts[j] > hi)
                {
                    g_last_exact_dp_diag.corridor_pruned += 1.0;
                    return false;
                }
            }
            return true;
        }

        bool check_exact_corridor_sparse(int64_t state_key, int total_rw,
                                         const std::vector<int> &totals,
                                         const std::vector<int> &lengths, int K)
        {
            if (!g_exact_corridor.enabled)
                return true;
            int placed_work = total_rw;
            int64_t tmp = state_key;
            int counts[16] = {0};
            for (int j = 0; j < K; ++j)
            {
                int cj = static_cast<int>(tmp % (totals[j] + 1));
                tmp /= (totals[j] + 1);
                counts[j] = cj;
                placed_work -= cj * lengths[j];
            }
            return check_exact_corridor_counts(counts, placed_work, K);
        }
    } // namespace

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

    SPACESResult make_stateless_spaces(int T)
    {
        MachineStateConfig cfg;
        cfg.states = {"off", "proc"};
        cfg.off_idx = 0;
        cfg.proc_idx = 1;
        cfg.t_trans.assign(2, std::vector<int>(2, -1));
        cfg.p_trans.assign(2, std::vector<double>(2, kInf));

        auto set_edge = [&](int s, int sp, int dur, double power)
        {
            cfg.t_trans[s][sp] = dur;
            cfg.p_trans[s][sp] = power;
        };

        set_edge(0, 0, 1, 0.0);
        set_edge(1, 1, 1, 1.0);
        set_edge(0, 1, 0, 0.0);
        set_edge(1, 0, 0, 0.0);

        std::vector<double> unit_prices(static_cast<std::size_t>(std::max(0, T)), 1.0);
        return compute_spaces(unit_prices, cfg, -1);
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
        double known_lb,
        std::string *out_best_policy,
        int *out_finite_candidates,
        double *out_time_to_first_ub_sec,
        std::vector<int> *out_best_seq)
    {
        int K = static_cast<int>(lengths.size());
        auto t0 = (out_time_to_first_ub_sec) ? std::chrono::steady_clock::now()
                                             : std::chrono::steady_clock::now();

        // Build multiset of all job lengths
        std::vector<int> all_jobs;
        for (int i = 0; i < K; ++i)
            for (int j = 0; j < totals[i]; ++j)
                all_jobs.push_back(lengths[i]);

        if (all_jobs.empty())
        {
            if (out_best_policy) *out_best_policy = "empty";
            if (out_finite_candidates) *out_finite_candidates = 0;
            return 0.0;
        }

        double best = kInf;
        std::string best_policy = "none";
        int finite_candidates = 0;
        bool first_finite_recorded = false;

        // Helper for tracking diagnostics
        auto record_candidate = [&](double cost, const std::string &policy,
                                     const std::vector<int> &seq)
        {
            if (cost < kInf * 0.5)
            {
                ++finite_candidates;
                if (!first_finite_recorded && out_time_to_first_ub_sec)
                {
                    *out_time_to_first_ub_sec = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0).count();
                    first_finite_recorded = true;
                }
                if (cost < best)
                {
                    best = cost;
                    best_policy = policy;
                    if (out_best_seq) *out_best_seq = seq;
                }
            }
        };

        // Helper for early termination
        auto check_and_update = [&](double cost) -> bool
        {
            if (cost < best)
            {
                best = cost;
                if (known_lb > 0 && std::abs(cost - known_lb) < 0.01)
                    return true;
            }
            return false;
        };

        // Try: sorted ascending (SPT)
        {
            std::vector<int> seq = all_jobs;
            std::sort(seq.begin(), seq.end());
            double cost = solve_fixed_sequence(seq, prefix_proc, T, spaces);
            record_candidate(cost, "spt", seq);
            if (check_and_update(cost))
            {
                if (out_best_policy) *out_best_policy = best_policy;
                if (out_finite_candidates) *out_finite_candidates = finite_candidates;
                return best;
            }
        }

        // Try: sorted descending (LPT)
        {
            std::vector<int> seq = all_jobs;
            std::sort(seq.begin(), seq.end(), std::greater<int>());
            double cost = solve_fixed_sequence(seq, prefix_proc, T, spaces);
            record_candidate(cost, "lpt", seq);
            if (check_and_update(cost))
            {
                if (out_best_policy) *out_best_policy = best_policy;
                if (out_finite_candidates) *out_finite_candidates = finite_candidates;
                return best;
            }
        }

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
            double cost = solve_fixed_sequence(alt, prefix_proc, T, spaces);
            record_candidate(cost, "alternating", alt);
            if (check_and_update(cost))
            {
                if (out_best_policy) *out_best_policy = best_policy;
                if (out_finite_candidates) *out_finite_candidates = finite_candidates;
                return best;
            }
        }

        // Try: all K! permutations of type-group orderings
        if (K <= 8)
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
                double cost = solve_fixed_sequence(typed_seq, prefix_proc, T, spaces);
                record_candidate(cost, "perm", typed_seq);
                if (check_and_update(cost))
                {
                    if (out_best_policy) *out_best_policy = best_policy;
                    if (out_finite_candidates) *out_finite_candidates = finite_candidates;
                    return best;
                }
            } while (std::next_permutation(perm.begin(), perm.end()));
        }

        // Try: random shuffles
        {
            std::vector<int> seq = all_jobs;
            std::sort(seq.begin(), seq.end()); // start from sorted for shuffle
            std::mt19937_64 rng(42);
            for (int trial = 0; trial < n_random; ++trial)
            {
                std::shuffle(seq.begin(), seq.end(), rng);
                double cost = solve_fixed_sequence(seq, prefix_proc, T, spaces);
                record_candidate(cost, "random_" + std::to_string(trial), seq);
                if (check_and_update(cost))
                {
                    if (out_best_policy) *out_best_policy = best_policy;
                    if (out_finite_candidates) *out_finite_candidates = finite_candidates;
                    return best;
                }
            }
        }

        if (out_best_policy) *out_best_policy = best_policy;
        if (out_finite_candidates) *out_finite_candidates = finite_candidates;
        return best;
    }

    // =====================================================================
    //  polish_best_sequence_ub: polish a given sequence via local search
    // =====================================================================

    double polish_best_sequence_ub(
        std::vector<int> &best_seq,
        double best_ub,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        double time_budget_sec)
    {
        if (best_seq.empty() || best_ub >= kInf * 0.5)
            return best_ub;
        return local_search_ub(best_seq, best_ub, prefix_proc, T, spaces, 5, time_budget_sec);
    }

    double compute_parallel_initial_ub(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        int M,
        int n_random,
        double known_lb,
        std::string *out_policy,
        int *out_machines_used,
        int *out_failed_machines)
    {
        std::vector<int> all_jobs;
        for (std::size_t i = 0; i < lengths.size(); ++i)
            for (int j = 0; j < totals[i]; ++j)
                all_jobs.push_back(lengths[i]);

        if (all_jobs.empty())
        {
            if (out_policy) *out_policy = "empty";
            if (out_machines_used) *out_machines_used = 0;
            if (out_failed_machines) *out_failed_machines = 0;
            return 0.0;
        }

        int real_M = std::max(1, std::min(M, static_cast<int>(all_jobs.size())));

        double best = kInf;

        auto set_out = [&](const std::string &pol, int used, int failed)
        {
            if (out_policy) *out_policy = pol;
            if (out_machines_used) *out_machines_used = used;
            if (out_failed_machines) *out_failed_machines = failed;
        };

        // Evaluate a machine partition.  On each machine the DP is run
        // for both ascending and descending order; the minimum per machine
        // is taken.
        auto try_partition = [&](const std::vector<std::vector<int>> &machines,
                                 const std::string &pol) -> bool
        {
            double total = 0.0;
            int used = 0;
            for (const auto &mjobs : machines)
            {
                if (mjobs.empty()) continue;
                used++;
                double best_m = kInf;

                std::vector<int> seq = mjobs;
                std::sort(seq.begin(), seq.end());
                double c = solve_fixed_sequence(seq, prefix_proc, T, spaces);
                if (c < best_m) best_m = c;

                if (best_m >= kInf * 0.5)
                {
                    std::sort(seq.begin(), seq.end(), std::greater<int>());
                    c = solve_fixed_sequence(seq, prefix_proc, T, spaces);
                    if (c < best_m) best_m = c;
                }

                if (best_m >= kInf * 0.5)
                    return false; // one infeasible machine → whole partition fails

                total += best_m;
            }

            if (total < best)
            {
                best = total;
                set_out(pol, used, 0);
                if (known_lb > 0 && std::abs(total - known_lb) < 0.01)
                    return true; // gap closed — stop early
            }
            return false;
        };

        // ── Partitioning policies ──

        // Helper: build partition from a job ordering using least-loaded assignment.
        auto least_loaded_partition = [&](const std::vector<int> &job_order) -> std::vector<std::vector<int>>
        {
            std::vector<std::vector<int>> mach(static_cast<std::size_t>(real_M));
            std::vector<int> loads(static_cast<std::size_t>(real_M), 0);
            for (int p : job_order)
            {
                int best_m = 0;
                for (int m = 1; m < real_M; ++m)
                    if (loads[static_cast<std::size_t>(m)] < loads[static_cast<std::size_t>(best_m)])
                        best_m = m;
                mach[static_cast<std::size_t>(best_m)].push_back(p);
                loads[static_cast<std::size_t>(best_m)] += p;
            }
            return mach;
        };

        // Policy 1: LPT load-balanced (descending jobs)
        {
            std::vector<int> desc = all_jobs;
            std::sort(desc.begin(), desc.end(), std::greater<int>());
            if (try_partition(least_loaded_partition(desc), "lpt"))
                return best;
        }

        // Policy 2: SPT load-balanced (ascending jobs)
        {
            std::vector<int> asc = all_jobs;
            std::sort(asc.begin(), asc.end());
            if (try_partition(least_loaded_partition(asc), "spt"))
                return best;
        }

        // Policy 3: Alternating load-balanced (outer ends inward)
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
            if (try_partition(least_loaded_partition(alt), "alternating"))
                return best;
        }

        // Policy 4: Round-robin by type (keep same-type jobs together)
        {
            std::vector<std::vector<int>> mach(static_cast<std::size_t>(real_M));
            for (std::size_t i = 0; i < lengths.size(); ++i)
            {
                int target = static_cast<int>(i) % real_M;
                for (int j = 0; j < totals[i]; ++j)
                    mach[static_cast<std::size_t>(target)].push_back(lengths[i]);
            }
            if (try_partition(mach, "round_robin_type"))
                return best;
        }

        // Policy 5: Random load-balanced
        {
            std::mt19937_64 rng(42);
            for (int trial = 0; trial < n_random; ++trial)
            {
                std::vector<int> shuf = all_jobs;
                std::shuffle(shuf.begin(), shuf.end(), rng);
                if (try_partition(least_loaded_partition(shuf), "random"))
                    return best;
            }
        }

        // No feasible partition found — reset output
        if (best >= kInf * 0.5)
            set_out("none", 0, 0);

        return best;
    }

    double guided_completion_ub(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        const std::vector<double> & /*completion_dp*/,
        int /*completion_RW*/,
        int /*completion_rw_scale*/,
        int n_rollouts,
        int /*top_k*/)
    {
        return compute_initial_ub(lengths, totals, prefix_proc, T, spaces,
                                  std::max(8, n_rollouts), 0.0);
    }

    double completion_guided_beam_ub(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        const std::vector<double> & /*completion_dp*/,
        int /*completion_RW*/,
        int /*completion_rw_scale*/,
        double known_ub,
        int beam_width,
        double /*time_limit_sec*/)
    {
        int n_random = std::max(16, std::min(4 * beam_width, 256));
        return compute_initial_ub(lengths, totals, prefix_proc, T, spaces,
                                  n_random, known_ub > 0.0 ? known_ub : 0.0);
    }

    // =====================================================================
    //  compute_spaces: SPACES preprocessing
    // =====================================================================

    SPACESResult compute_spaces(const std::vector<double> &prices, const MachineStateConfig &config, int max_gap)
    {
        int h = static_cast<int>(prices.size());
        int n_s = static_cast<int>(config.states.size());
        SPACESResult out;
        out.config = config;
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

    RelaxedTableResult compute_relaxed_dp_table(
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
        int total_rw_scaled = total_rw / rw_gcd;
        int RW = total_rw_scaled + 1;

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
        auto idx = [&](int t, int rw) -> int
        { return t * RW + rw; };
        auto relax_cell = [&](int t, int rw, double cost, RPar parent)
        {
            int i = idx(t, rw);
            if (cost < dp[i])
            {
                if (dp[i] >= kInf)
                    active_rw[t].push_back(rw);
                dp[i] = cost;
                par[i] = parent;
            }
        };

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
                relax_cell(t_e, total_rw_scaled - Ls, cost, {-1, total_rw_scaled, L, t_s});
            }
        }

        std::vector<double> bank_cost(RW, kInf);
        std::vector<int> bank_t(RW, -1);
        std::vector<int> bank_active;
        std::vector<uint8_t> bank_seen(RW, 0);
        std::vector<std::vector<std::tuple<int, double, int>>> deferred(T + 2);

        double best = kInf;
        for (int t_end = 1; t_end <= T; ++t_end)
        {
            if (spaces.banded && t_end < static_cast<int>(deferred.size()))
            {
                for (auto &[rw, cost, t_src] : deferred[t_end])
                {
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
                        relax_cell(t_e, rw - Ls, cost, {bank_t[rw], rw, L, t_end});
                    }
                }
            }

            double d0 = dp[idx(t_end, 0)];
            if (d0 < kInf && spaces.c_end[t_end] < kInf)
                best = std::min(best, d0 + spaces.c_end[t_end]);

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

        RelaxedTableResult out;
        out.rdp = std::move(dp);
        out.RW = RW;
        out.rw_scale = rw_gcd;
        out.lb = best;
        return out;
    }

    RelaxedTableResult compute_relaxed_completion_table(
        const std::vector<int> &lengths,
        int total_rw,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        RelaxationMode mode)
    {
        auto suffix_completion = [&]() -> RelaxedTableResult
        {
            std::vector<double> unit_prices(T, 0.0);
            for (int i = 0; i < T; ++i)
                unit_prices[i] = prefix_proc[i + 1] - prefix_proc[i];

            std::vector<double> min_c_end_from(T + 2, kInf);
            for (int t = T; t >= 0; --t)
            {
                min_c_end_from[t] = min_c_end_from[t + 1];
                if (spaces.c_end[t] < min_c_end_from[t])
                    min_c_end_from[t] = spaces.c_end[t];
            }

            constexpr int LB_BLOCK = 20;
            int n_blocks = (T / LB_BLOCK) + 1;
            std::vector<std::vector<double>> suffix_sorted_prefix(n_blocks + 1);
            for (int bi = 0; bi < n_blocks; ++bi)
            {
                int b = bi * LB_BLOCK;
                std::vector<double> vals;
                if (b < T)
                    vals.assign(unit_prices.begin() + b, unit_prices.end());
                std::sort(vals.begin(), vals.end());
                suffix_sorted_prefix[bi].resize(vals.size() + 1, 0.0);
                for (std::size_t i = 0; i < vals.size(); ++i)
                    suffix_sorted_prefix[bi][i + 1] = suffix_sorted_prefix[bi][i] + vals[i];
            }
            suffix_sorted_prefix[n_blocks] = {0.0};

            int RW = total_rw + 1;
            std::vector<double> table((T + 2) * RW, kInf);
            for (int t = 0; t <= T + 1; ++t)
            {
                int bi = std::min(t / LB_BLOCK, n_blocks);
                const auto &pref = suffix_sorted_prefix[bi];
                int max_rw = static_cast<int>(pref.size()) - 1;
                table[static_cast<std::size_t>(t) * RW + 0] = (t <= T ? spaces.c_end[t] : 0.0);
                for (int rw = 1; rw <= total_rw; ++rw)
                {
                    double val = (rw <= max_rw ? pref[rw] : kInf);
                    if (val < kInf)
                    {
                        int earliest_end = std::min(T + 1, t + rw);
                        if (earliest_end <= T)
                            val += min_c_end_from[earliest_end];
                    }
                    table[static_cast<std::size_t>(t) * RW + rw] = val;
                }
            }

            RelaxedTableResult out;
            out.rdp = table;
            out.off_rdp = std::move(table);
            out.RW = RW;
            out.rw_scale = 1;
            out.lb = out.off_rdp[total_rw];
            return out;
        };

        auto direct_completion = [&]() -> RelaxedTableResult
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
            int total_rw_scaled = total_rw / rw_gcd;
            int RW = total_rw_scaled + 1;
            int eff_max_gap = spaces.banded ? spaces.max_gap : T;

            int direct_max_cells = 500000000;
            if (const char *raw = std::getenv("PAST_BLOCK_REPAIR_COMPLETION_DIRECT_MAX_CELLS"))
            {
                char *end = nullptr;
                long v = std::strtol(raw, &end, 10);
                if (end && *end == '\0')
                    direct_max_cells = static_cast<int>(v);
            }
            int64_t direct_cells = static_cast<int64_t>(T + 2) * static_cast<int64_t>(RW);
            if (direct_max_cells > 0 && direct_cells > static_cast<int64_t>(direct_max_cells))
            {
                return RelaxedTableResult{};
            }

            std::vector<double> start_exact((T + 2) * RW, kInf);
            std::vector<double> cont((T + 2) * RW, kInf);
            std::vector<double> off((T + 2) * RW, kInf);

            auto idx = [&](int t, int rw) -> std::size_t
            {
                return static_cast<std::size_t>(t) * RW + rw;
            };

            cont[idx(T + 1, 0)] = 0.0;
            off[idx(T + 1, 0)] = 0.0;

            for (int t = T; t >= 0; --t)
            {
                cont[idx(t, 0)] = spaces.c_end[t];
                off[idx(t, 0)] = 0.0;

                for (int rw = 1; rw <= total_rw_scaled; ++rw)
                {
                    double start_here = kInf;
                    if (t <= spaces.late)
                    {
                        for (int j = 0; j < K; ++j)
                        {
                            int L = allowed_lengths[j];
                            int Ls = scaled_lengths[j];
                            if (Ls > rw)
                                continue;
                            int t_e = t + L;
                            if (t_e > T || t_e > spaces.late + 1)
                                continue;
                            double tail = cont[idx(t_e, rw - Ls)];
                            if (tail >= kInf)
                                continue;
                            double cand = (prefix_proc[t_e] - prefix_proc[t]) + tail;
                            if (cand < start_here)
                                start_here = cand;
                        }
                    }
                    start_exact[idx(t, rw)] = start_here;

                    double off_best = off[idx(t + 1, rw)];
                    if (spaces.c_start[t] < kInf && start_here < kInf)
                        off_best = std::min(off_best, spaces.c_start[t] + start_here);
                    off[idx(t, rw)] = off_best;

                    double best = kInf;
                    int gap_limit = std::min(t + eff_max_gap, spaces.late);
                    for (int t_s = t; t_s <= gap_limit; ++t_s)
                    {
                        double tail = start_exact[idx(t_s, rw)];
                        if (tail >= kInf)
                            continue;
                        double gap = spaces.gap_cost(t, t_s);
                        if (gap >= kInf)
                            continue;
                        best = std::min(best, gap + tail);
                    }
                    if (spaces.banded && spaces.c_end[t] < kInf)
                    {
                        int eligible = std::min(T + 1, t + eff_max_gap + 1);
                        double restart = off[idx(eligible, rw)];
                        if (restart < kInf)
                            best = std::min(best, spaces.c_end[t] + restart);
                    }
                    cont[idx(t, rw)] = best;
                }
            }

            RelaxedTableResult out;
            out.rdp = std::move(cont);
            out.off_rdp = std::move(off);
            out.RW = RW;
            out.rw_scale = rw_gcd;
            out.lb = out.off_rdp[total_rw_scaled];
            return out;
        };

        auto env_int_local = [](const char *name, int fallback) -> int
        {
            const char *raw = std::getenv(name);
            if (!raw || !*raw)
                return fallback;
            char *end = nullptr;
            long v = std::strtol(raw, &end, 10);
            if (!end || *end != '\0')
                return fallback;
            return static_cast<int>(v);
        };

        auto mode_str = []() -> std::string
        {
            const char *raw = std::getenv("PAST_BLOCK_REPAIR_COMPLETION_MODE");
            return raw ? std::string(raw) : std::string("auto");
        }();
        int direct_k_threshold = env_int_local("PAST_BLOCK_REPAIR_COMPLETION_DIRECT_K", 5);
        bool want_direct = false;
        if (mode_str == "direct")
            want_direct = true;
        else if (mode_str == "cheap")
            want_direct = false;
        else
            want_direct = static_cast<int>(lengths.size()) >= direct_k_threshold;

        bool want_diag = env_int_local("PAST_BLOCK_REPAIR_COMPLETION_DIAG", 0) != 0;
        RelaxedTableResult cheap;
        RelaxedTableResult direct;

        if (want_diag || !want_direct)
            cheap = suffix_completion();
        if (want_diag || want_direct)
            direct = direct_completion();

        if (want_diag)
        {
            auto lookup = [&](const RelaxedTableResult &tab, const std::vector<double> &arr,
                              int t, int rw) -> double
            {
                if (arr.empty() || tab.RW <= 0)
                    return kInf;
                int scaled_rw = tab.rw_scale > 1 ? rw / tab.rw_scale : rw;
                if (t < 0)
                    t = 0;
                if (t > T + 1 || scaled_rw < 0 || scaled_rw >= tab.RW)
                    return kInf;
                return arr[static_cast<std::size_t>(t) * tab.RW + scaled_rw];
            };

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

            std::cerr << "completion_diag mode=" << mode_str
                      << " K=" << lengths.size()
                      << " T=" << T
                      << " total_rw=" << total_rw
                      << " cheap_lb=" << cheap.lb
                      << " direct_lb=" << direct.lb
                      << "\n";
            for (int t : sample_t)
            {
                for (int rw : sample_rw)
                {
                    double cheap_cont = lookup(cheap, cheap.rdp, t, rw);
                    double direct_cont = lookup(direct, direct.rdp, t, rw);
                    double cheap_off = lookup(cheap, cheap.off_rdp, t, rw);
                    double direct_off = lookup(direct, direct.off_rdp, t, rw);
                    std::cerr << "completion_diag_sample"
                              << " t=" << t
                              << " rw=" << rw
                              << " cheap_cont=" << cheap_cont
                              << " direct_cont=" << direct_cont
                              << " cheap_off=" << cheap_off
                              << " direct_off=" << direct_off
                              << "\n";
                }
            }
        }

        if (want_direct)
            return direct.rdp.empty() ? suffix_completion() : direct;
        return cheap;
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
        double t_pack_profile_recovery = 0.0;
        double t_pack_merge_blocks = 0.0;
        double t_pack_to_first_candidate = 0.0;
        double t_pack_ffd_only = 0.0;
        int step2_reached = 0;
        int step2_produced_ub = 0;
        // PLAN15 dense-unit forward-relax/profile split diagnostics
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
        // PLAN26: exact merged blocks used by profile_repair_beam
        std::vector<Segment> merged_blocks;
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
        double block_view_best_ub = 0.0;
        double block_view_time_sec = 0.0;
    };

    struct ProfileRepairBeamDiag
    {
        double base_width = 0.0;
        double avg_width = 0.0;
        double max_width = 0.0;
        double states_considered = 0.0;
        double states_kept = 0.0;
        double pruned_over = 0.0;
        double pruned_suffix = 0.0;
        double pruned_discrepancy = 0.0;
        int discrepancy_budget = 0;
        int discrepancy_depth = 0;
        std::string status = "not_attempted";
        int timed_out = 0;
        std::string key_multi_policy = "off";
        int key_multi_max = 1;
        double key_multi_score_eps = 0.0;
        double key_multi_diversity_eps = 0.0;
        // PLAN27 residual-aware + late ambiguity diagnostics
        std::string score_policy = "default";
        double residual_weight = 0.0;
        double residual_mean_penalty = 0.0;
        double residual_max_penalty = 0.0;
        double late_frac = 0.0;
    };

    namespace
    {
        constexpr int kMaxRepairTypes = 16;

        struct RepairPattern
        {
            std::vector<int> counts;
            int work = 0;
            int local_dev = 0;
            double center_dev = 0.0;
        };

        struct EnergyCorePatternPoolDiag
        {
            std::vector<int> generated_per_block;
            std::vector<int> retained_per_block;
            int fixed_blocks = 0;
            double generated_total = 0.0;
            double generated_max_block = 0.0;
            double retained_total = 0.0;
            double retained_max_block = 0.0;
            std::string retained_signature;
        };

        struct EnergyCoreRunDiag
        {
            EnergyCorePatternPoolDiag pool;
            double t_completion = 0.0;
            double t_patterns = 0.0;
            double t_exact_core = 0.0;
            double pruned_core_window = 0.0;
            double pruned_suffix = 0.0;
            double pruned_transition = 0.0;
            double pruned_bound = 0.0;
            int delta_used = -1;
            int retained_fixed_blocks = 0;
            int two_phase_used = 0;
            double phase1_feasible_ub = kInf;
            double t_phase1 = 0.0;
        };

        struct EnergyCoreNode
        {
            std::array<int, kMaxRepairTypes> counts{};
            int prev_end = -1;
            int prefix_work = 0;
            double g = kInf;
            double f = kInf;
        };

        struct EnergyCoreKey
        {
            std::array<uint16_t, kMaxRepairTypes> counts{};
            uint8_t used = 0;
            int prev_end = -1;

            bool operator==(const EnergyCoreKey &other) const noexcept
            {
                if (prev_end != other.prev_end || used != other.used)
                    return false;
                for (uint8_t i = 0; i < used; ++i)
                {
                    if (counts[i] != other.counts[i])
                        return false;
                }
                return true;
            }
        };

        struct EnergyCoreKeyHash
        {
            std::size_t operator()(const EnergyCoreKey &key) const noexcept
            {
                std::size_t h = 1469598103934665603ULL;
                h ^= static_cast<std::size_t>(key.used);
                h *= 1099511628211ULL;
                for (uint8_t i = 0; i < key.used; ++i)
                {
                    h ^= static_cast<std::size_t>(key.counts[i]);
                    h *= 1099511628211ULL;
                }
                h ^= static_cast<std::size_t>(static_cast<uint32_t>(key.prev_end));
                h *= 1099511628211ULL;
                return h;
            }
        };

        struct CountKey
        {
            std::array<uint16_t, kMaxRepairTypes> counts{};
            uint8_t used = 0;

            bool operator==(const CountKey &other) const noexcept
            {
                if (used != other.used)
                    return false;
                for (uint8_t i = 0; i < used; ++i)
                {
                    if (counts[i] != other.counts[i])
                        return false;
                }
                return true;
            }
        };

        struct CountKeyHash
        {
            std::size_t operator()(const CountKey &key) const noexcept
            {
                std::size_t h = 1469598103934665603ULL;
                h ^= static_cast<std::size_t>(key.used);
                h *= 1099511628211ULL;
                for (uint8_t i = 0; i < key.used; ++i)
                {
                    h ^= static_cast<std::size_t>(key.counts[i]);
                    h *= 1099511628211ULL;
                }
                return h;
            }
        };

        int env_int_or(const char *name, int fallback)
        {
            const char *raw = std::getenv(name);
            if (!raw || !*raw)
                return fallback;
            char *end = nullptr;
            long v = std::strtol(raw, &end, 10);
            if (!end || *end != '\0')
                return fallback;
            return static_cast<int>(v);
        }

        int64_t env_int64_or(const char *name, int64_t fallback)
        {
            const char *raw = std::getenv(name);
            if (!raw || !*raw)
                return fallback;
            char *end = nullptr;
            long long v = std::strtoll(raw, &end, 10);
            if (!end || *end != '\0')
                return fallback;
            return static_cast<int64_t>(v);
        }

        double env_double_or(const char *name, double fallback)
        {
            const char *raw = std::getenv(name);
            if (!raw || !*raw)
                return fallback;
            char *end = nullptr;
            double v = std::strtod(raw, &end);
            if (!end || *end != '\0')
                return fallback;
            return v;
        }

        std::string env_str_or(const char *name, const std::string &fallback)
        {
            const char *raw = std::getenv(name);
            if (!raw || !*raw)
                return fallback;
            return std::string(raw);
        }

        std::string to_lower_ascii(std::string s)
        {
            for (char &ch : s)
            {
                if (ch >= 'A' && ch <= 'Z')
                    ch = static_cast<char>(ch - 'A' + 'a');
            }
            return s;
        }

        double semigroup_density_prefix(const std::vector<int> &lengths, int cap)
        {
            if (lengths.empty() || cap < 0)
                return 0.0;
            std::vector<uint8_t> reachable(static_cast<std::size_t>(cap + 1), 0);
            reachable[0] = 1;
            for (int x = 0; x <= cap; ++x)
            {
                if (!reachable[static_cast<std::size_t>(x)])
                    continue;
                for (int L : lengths)
                {
                    int nx = x + L;
                    if (nx <= cap)
                        reachable[static_cast<std::size_t>(nx)] = 1;
                }
            }
            int hit = 0;
            for (uint8_t v : reachable)
                hit += (v != 0) ? 1 : 0;
            return static_cast<double>(hit) / std::max(1, cap + 1);
        }

        EnergyCoreKey energy_core_key(const EnergyCoreNode &node, int K)
        {
            EnergyCoreKey key;
            key.used = static_cast<uint8_t>(K);
            for (int j = 0; j < K; ++j)
                key.counts[j] = static_cast<uint16_t>(node.counts[j]);
            key.prev_end = node.prev_end;
            return key;
        }

        CountKey count_key_from_counts(const std::array<int, kMaxRepairTypes> &counts, int K)
        {
            CountKey key;
            key.used = static_cast<uint8_t>(K);
            for (int j = 0; j < K; ++j)
                key.counts[j] = static_cast<uint16_t>(counts[j]);
            return key;
        }

        std::string pattern_counts_key(const std::vector<int> &counts)
        {
            std::string key(sizeof(int) * counts.size(), '\0');
            if (!counts.empty())
                std::memcpy(key.data(), counts.data(), sizeof(int) * counts.size());
            return key;
        }

        bool profile_realization_hardest_first_enabled()
        {
            return env_int_or("PAST_PROFILE_REALIZATION_HARDEST_FIRST", 0) != 0;
        }

        std::string exact_dp_variant_name()
        {
            const char *raw = std::getenv("PAST_EXACT_DP_VARIANT");
            if (!raw || !*raw)
                return "p0";
            std::string v = to_lower_ascii(std::string(raw));
            if (v == "baseline")
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

        bool exact_dp_type_aware_lb_enabled(const std::string &variant)
        {
            int fallback = (variant == "p1" || variant == "p3" || variant == "p4") ? 1 : 0;
            return env_int_or("PAST_EXACT_DP_TYPE_AWARE_LB", fallback) != 0;
        }

        bool exact_dp_incumbent_ordering_enabled(const std::string &variant)
        {
            int fallback = (variant == "p2" || variant == "p3" || variant == "p4") ? 1 : 0;
            return env_int_or("PAST_EXACT_DP_INCUMBENT_ORDERING", fallback) != 0;
        }

        std::vector<int> profile_realization_block_order(
            const std::vector<RecoveredBlock> &merged,
            const std::vector<std::vector<RepairPattern>> &patterns)
        {
            int B = static_cast<int>(merged.size());
            std::vector<int> order(B);
            std::iota(order.begin(), order.end(), 0);
            if (!profile_realization_hardest_first_enabled())
                return order;
            std::sort(order.begin(), order.end(), [&](int a, int b)
                      {
                          if (patterns[a].size() != patterns[b].size())
                              return patterns[a].size() < patterns[b].size();
                          if (merged[a].length != merged[b].length)
                              return merged[a].length > merged[b].length;
                          return a < b;
                      });
            return order;
        }

        double lookup_completion_lb(
            const std::vector<double> &table,
            int table_RW,
            int rw_scale,
            int T,
            int t,
            int rw)
        {
            if (table.empty())
                return 0.0;
            if (rw < 0)
                return kInf;
            int scaled_rw = rw_scale > 1 ? rw / rw_scale : rw;
            if (t < 0)
                t = 0;
            if (t > T + 1)
                t = T + 1;
            if (scaled_rw < 0 || scaled_rw >= table_RW)
                return kInf;
            return table[static_cast<std::size_t>(t) * table_RW + scaled_rw];
        }

        std::vector<std::vector<RepairPattern>> generate_energy_core_patterns(
            const std::vector<RecoveredBlock> &merged,
            const std::vector<int> &lengths,
            const std::vector<int> &totals,
            int T,
            EnergyCorePatternPoolDiag *pool_diag = nullptr)
        {
            if (pool_diag)
                *pool_diag = EnergyCorePatternPoolDiag{};
            int K = static_cast<int>(lengths.size());
            int total_work = 0;
            for (int i = 0; i < K; ++i)
                total_work += lengths[i] * totals[i];

            const int per_work_keep = env_int_or("PAST_BLOCK_REPAIR_PER_WORK_KEEP", std::max(2, K));
            const int global_keep = env_int_or("PAST_BLOCK_REPAIR_GLOBAL_KEEP", std::max(32, 8 * K));
            const int diversify_per_type = env_int_or("PAST_BLOCK_REPAIR_DIVERSIFY_PER_TYPE", K >= 6 ? 4 : 2);
            const int max_len = lengths.empty() ? 0 : *std::max_element(lengths.begin(), lengths.end());
            const int diversify_dev_window = env_int_or("PAST_BLOCK_REPAIR_DIVERSIFY_DEV_WINDOW", std::max(2, max_len));
            const int dp_generator_k = env_int_or("PAST_BLOCK_REPAIR_PATTERN_DP_K", (K == 4) ? 4 : 5);
            const bool enable_signature_dedup = env_int_or("PAST_BLOCK_REPAIR_EC_SIGNATURE_DEDUP", (K == 4) ? 0 : 1) != 0;
            const bool enable_diversify = env_int_or("PAST_BLOCK_REPAIR_EC_DIVERSIFY", 1) != 0;

            auto better_center_then_local = [](const RepairPattern &a, const RepairPattern &b)
            {
                if (a.center_dev != b.center_dev)
                    return a.center_dev < b.center_dev;
                return a.local_dev < b.local_dev;
            };

            auto better_local_then_center = [](const RepairPattern &a, const RepairPattern &b)
            {
                if (a.local_dev != b.local_dev)
                    return a.local_dev < b.local_dev;
                return a.center_dev < b.center_dev;
            };

            auto trim_bucket = [&](std::vector<RepairPattern> &bucket, int keep,
                                   const auto &better)
            {
                if (static_cast<int>(bucket.size()) <= keep)
                {
                    std::sort(bucket.begin(), bucket.end(), better);
                    return;
                }
                auto nth = bucket.begin() + keep;
                std::nth_element(bucket.begin(), nth, bucket.end(),
                                 [&](const RepairPattern &x, const RepairPattern &y)
                                 { return better(x, y); });
                bucket.resize(keep);
                std::sort(bucket.begin(), bucket.end(), better);
            };

            std::vector<std::vector<RepairPattern>> out(merged.size());
            std::vector<int> order(K);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](int a, int b)
                      {
                          if (lengths[a] != lengths[b])
                              return lengths[a] > lengths[b];
                          if (totals[a] != totals[b])
                              return totals[a] < totals[b];
                          return a < b;
                      });

            for (std::size_t bi = 0; bi < merged.size(); ++bi)
            {
                int cap = merged[bi].length;
                int start = merged[bi].start;
                int high = (bi + 1 < merged.size())
                               ? std::max(0, merged[bi + 1].start - start)
                               : std::max(0, std::min(T, T) - start);
                high = std::max(high, cap);

                std::vector<double> local_center(K, 0.0);
                for (int j = 0; j < K; ++j)
                    local_center[j] = total_work > 0 ? (static_cast<double>(totals[j]) * cap / total_work) : 0.0;

                std::vector<RepairPattern> flat;
                if (K >= dp_generator_k)
                {
                    std::vector<std::vector<RepairPattern>> cur(high + 1), nxt(high + 1);
                    RepairPattern seed;
                    seed.counts.assign(K, 0);
                    seed.work = 0;
                    seed.local_dev = cap;
                    seed.center_dev = 0.0;
                    cur[0].push_back(seed);

                    for (int oi = 0; oi < K; ++oi)
                    {
                        int j = order[oi];
                        int L = lengths[j];
                        for (auto &bucket : nxt)
                            bucket.clear();
                        for (int work = 0; work <= high; ++work)
                        {
                            if (cur[work].empty())
                                continue;
                            for (const auto &pat : cur[work])
                            {
                                int maxc = std::min(totals[j], (high - work) / std::max(1, L));
                                for (int c = 0; c <= maxc; ++c)
                                {
                                    RepairPattern npat = pat;
                                    npat.counts[j] = c;
                                    npat.work = work + c * L;
                                    npat.center_dev = pat.center_dev + std::abs(static_cast<double>(c) - local_center[j]);
                                    npat.local_dev = std::abs(npat.work - cap);
                                    auto &bucket = nxt[npat.work];
                                    bucket.push_back(std::move(npat));
                                }
                            }
                        }
                        for (auto &bucket : nxt)
                        {
                            if (bucket.empty())
                                continue;
                            trim_bucket(bucket, per_work_keep, better_center_then_local);
                        }
                        cur.swap(nxt);
                    }

                    for (auto &bucket : cur)
                        for (auto &pat : bucket)
                            flat.push_back(std::move(pat));
                }
                else
                {
                    std::vector<std::vector<RepairPattern>> per_work(high + 1);
                    std::vector<int> counts(K, 0);
                    std::vector<int> suffix_max_work(K + 1, 0);
                    for (int oi = K - 1; oi >= 0; --oi)
                    {
                        int j = order[oi];
                        suffix_max_work[oi] = suffix_max_work[oi + 1] + totals[j] * lengths[j];
                    }

                    std::function<void(int, int)> dfs = [&](int oi, int work)
                    {
                        if (work > high)
                            return;
                        if (oi == K)
                        {
                            RepairPattern pat;
                            pat.counts = counts;
                            pat.work = work;
                            pat.local_dev = std::abs(work - cap);
                            pat.center_dev = 0.0;
                            for (int j = 0; j < K; ++j)
                                pat.center_dev += std::abs(static_cast<double>(counts[j]) - local_center[j]);
                            auto &bucket = per_work[work];
                            bucket.push_back(std::move(pat));
                            trim_bucket(bucket, per_work_keep, better_local_then_center);
                            return;
                        }

                        int j = order[oi];
                        int L = lengths[j];
                        int maxc = std::min(totals[j], (high - work) / std::max(1, L));
                        for (int c = 0; c <= maxc; ++c)
                        {
                            counts[j] = c;
                            int next_work = work + c * L;
                            if (next_work > high)
                                break;
                            if (next_work + suffix_max_work[oi + 1] < 0)
                                continue;
                            dfs(oi + 1, next_work);
                        }
                        counts[j] = 0;
                    };
                    dfs(0, 0);

                    for (auto &bucket : per_work)
                        for (auto &pat : bucket)
                            flat.push_back(std::move(pat));
                }

                int generated_count = static_cast<int>(flat.size());

                // Safe dominance reduction: exact count-signature duplicates only.
                // Keep the best representative per count vector.
                if (enable_signature_dedup && !flat.empty())
                {
                    std::unordered_map<std::string, RepairPattern> uniq;
                    uniq.reserve(flat.size() * 2 + 1);
                    for (auto &pat : flat)
                    {
                        std::string key = pattern_counts_key(pat.counts);
                        auto it = uniq.find(key);
                        if (it == uniq.end())
                        {
                            uniq.emplace(std::move(key), std::move(pat));
                            continue;
                        }
                        const RepairPattern &cur = it->second;
                        bool better = (pat.local_dev < cur.local_dev) ||
                                      (pat.local_dev == cur.local_dev && pat.center_dev < cur.center_dev);
                        if (better)
                            it->second = std::move(pat);
                    }
                    flat.clear();
                    flat.reserve(uniq.size());
                    for (auto &kv : uniq)
                        flat.push_back(std::move(kv.second));
                }

                auto better_final = [&](const RepairPattern &a, const RepairPattern &b)
                {
                    if (a.local_dev != b.local_dev)
                        return a.local_dev < b.local_dev;
                    return a.center_dev < b.center_dev;
                };
                if (static_cast<int>(flat.size()) > global_keep)
                {
                    auto nth = flat.begin() + global_keep;
                    std::nth_element(flat.begin(), nth, flat.end(),
                                     [&](const RepairPattern &x, const RepairPattern &y)
                                     { return better_final(x, y); });
                    flat.resize(global_keep);
                }
                std::sort(flat.begin(), flat.end(), better_final);
                if (enable_diversify && !flat.empty() && diversify_per_type > 0 && static_cast<int>(flat.size()) > global_keep)
                {
                    std::vector<RepairPattern> diversified;
                    diversified.reserve(std::min<int>(static_cast<int>(flat.size()), global_keep + 2 * K * diversify_per_type));
                    std::unordered_set<std::string> seen;
                    seen.reserve(diversified.capacity() * 2 + 1);

                    auto try_add = [&](const RepairPattern &pat)
                    {
                        std::string key = pattern_counts_key(pat.counts);
                        if (seen.insert(key).second)
                            diversified.push_back(pat);
                    };

                    int reserve_for_extremes = 2 * K * diversify_per_type;
                    int base_keep = std::max(1, global_keep - reserve_for_extremes);
                    for (int idx = 0; idx < static_cast<int>(flat.size()) && static_cast<int>(diversified.size()) < base_keep; ++idx)
                        try_add(flat[idx]);

                    int best_local = flat.front().local_dev;
                    std::vector<const RepairPattern *> window;
                    window.reserve(flat.size());
                    for (const auto &pat : flat)
                    {
                        if (pat.local_dev <= best_local + diversify_dev_window)
                            window.push_back(&pat);
                    }
                    if (window.empty())
                    {
                        for (const auto &pat : flat)
                            window.push_back(&pat);
                    }

                    for (int j = 0; j < K; ++j)
                    {
                        auto high = window;
                        std::sort(high.begin(), high.end(), [j](const RepairPattern *a, const RepairPattern *b)
                                  {
                                      if (a->counts[j] != b->counts[j])
                                          return a->counts[j] > b->counts[j];
                                      if (a->local_dev != b->local_dev)
                                          return a->local_dev < b->local_dev;
                                      return a->center_dev < b->center_dev;
                                  });
                        for (int take = 0; take < static_cast<int>(high.size()) && take < diversify_per_type; ++take)
                            try_add(*high[take]);

                        auto low = window;
                        std::sort(low.begin(), low.end(), [j](const RepairPattern *a, const RepairPattern *b)
                                  {
                                      if (a->counts[j] != b->counts[j])
                                          return a->counts[j] < b->counts[j];
                                      if (a->local_dev != b->local_dev)
                                          return a->local_dev < b->local_dev;
                                      return a->center_dev < b->center_dev;
                                  });
                        for (int take = 0; take < static_cast<int>(low.size()) && take < diversify_per_type; ++take)
                            try_add(*low[take]);
                    }

                    for (int idx = base_keep; idx < static_cast<int>(flat.size()) && static_cast<int>(diversified.size()) < global_keep; ++idx)
                        try_add(flat[idx]);
                    while (static_cast<int>(diversified.size()) > global_keep)
                        diversified.pop_back();
                    flat = std::move(diversified);
                }
                // No-op else branch: when diversification is disabled, flat is already
                // trimmed with partial selection above.

                bool have_zero = false;
                for (const auto &pat : flat)
                    if (pat.work == 0)
                        have_zero = true;
                if (!have_zero)
                {
                    RepairPattern zero;
                    zero.counts.assign(K, 0);
                    zero.work = 0;
                    zero.local_dev = cap;
                    flat.push_back(std::move(zero));
                }
                out[bi] = std::move(flat);

                if (pool_diag)
                {
                    pool_diag->generated_per_block.push_back(generated_count);
                    pool_diag->retained_per_block.push_back(static_cast<int>(out[bi].size()));
                    pool_diag->generated_total += generated_count;
                    pool_diag->generated_max_block = std::max(pool_diag->generated_max_block, static_cast<double>(generated_count));
                    pool_diag->retained_total += static_cast<int>(out[bi].size());
                    pool_diag->retained_max_block = std::max(pool_diag->retained_max_block, static_cast<double>(out[bi].size()));
                    if (out[bi].size() <= 1)
                        pool_diag->fixed_blocks += 1;
                }
            }

            if (pool_diag)
            {
                std::ostringstream sig;
                for (std::size_t bi = 0; bi < pool_diag->retained_per_block.size(); ++bi)
                {
                    if (bi)
                        sig << "|";
                    sig << bi << ":" << pool_diag->retained_per_block[bi];
                }
                pool_diag->retained_signature = sig.str();
            }

            return out;
        }

        void build_profile_block_local_views(
            const std::vector<RecoveredBlock> &merged,
            const std::vector<double> &prefix_proc,
            int T,
            const SPACESResult &spaces,
            std::vector<SPACESResult> *block_spaces,
            std::vector<std::vector<double>> *block_prefix_proc)
        {
            int B = static_cast<int>(merged.size());
            block_spaces->assign(B, SPACESResult{});
            block_prefix_proc->assign(B, {});
            for (int bi = 0; bi < B; ++bi)
            {
                const int block_start = merged[bi].start;
                const int block_len = merged[bi].length;
                SPACESResult local = spaces;
                local.early = 0;
                local.late = std::max(0, block_len - 1);
                local.h = block_len;
                if (local.banded)
                    local.max_gap = std::min(local.max_gap, block_len);

                local.c_start.assign(block_len + 1, kInf);
                local.c_end.assign(block_len + 1, kInf);
                for (int t = 0; t <= block_len; ++t)
                {
                    local.c_start[t] = spaces.c_start[block_start + t];
                    local.c_end[t] = spaces.c_end[block_start + t];
                }

                if (local.banded)
                {
                    int stride = local.max_gap + 1;
                    local.c_star.assign(static_cast<std::size_t>(block_len + 1) * stride, kInf);
                    for (int t_end = 0; t_end <= block_len; ++t_end)
                    {
                        int max_delta = std::min(local.max_gap, block_len - t_end);
                        for (int delta = 0; delta <= max_delta; ++delta)
                        {
                            local.c_star[static_cast<std::size_t>(t_end) * stride + delta] =
                                spaces.gap_cost(block_start + t_end, block_start + t_end + delta);
                        }
                    }
                }
                else
                {
                    int stride = block_len + 1;
                    local.c_star.assign(static_cast<std::size_t>(stride) * stride, kInf);
                    for (int t_end = 0; t_end <= block_len; ++t_end)
                    {
                        for (int t_start = t_end; t_start <= block_len; ++t_start)
                        {
                            local.c_star[static_cast<std::size_t>(t_end) * stride + t_start] =
                                spaces.gap_cost(block_start + t_end, block_start + t_start);
                        }
                    }
                }

                (*block_spaces)[bi] = std::move(local);
                (*block_prefix_proc)[bi].assign(block_len + 1, 0.0);
                for (int t = 0; t <= block_len; ++t)
                    (*block_prefix_proc)[bi][t] = prefix_proc[block_start + t] - prefix_proc[block_start];
            }
        }

        double evaluate_profile_block_counts(
            int bi,
            const std::vector<int> &counts,
            const std::vector<int> &lengths,
            const std::vector<RecoveredBlock> &merged,
            const std::vector<SPACESResult> &block_spaces,
            const std::vector<std::vector<double>> &block_prefix_proc,
            int l3_max_cells,
            double l3_time_limit)
        {
            int K = static_cast<int>(lengths.size());
            int total_jobs = 0;
            int theoretical_states = 1;
            for (int j = 0; j < K; ++j)
            {
                total_jobs += counts[j];
                if (counts[j] < 0)
                    return kInf;
                if (counts[j] == 0)
                    continue;
                if (theoretical_states > l3_max_cells / std::max(1, counts[j] + 1))
                {
                    theoretical_states = l3_max_cells + 1;
                    break;
                }
                theoretical_states *= (counts[j] + 1);
            }
            if (total_jobs == 0)
                return 0.0;

            const auto &local_prefix = block_prefix_proc[bi];
            const auto &local_spaces = block_spaces[bi];
            int local_T = merged[bi].length;

            std::vector<int> seq_desc;
            std::vector<int> seq_asc;
            seq_desc.reserve(total_jobs);
            seq_asc.reserve(total_jobs);
            for (int j = K - 1; j >= 0; --j)
                for (int c = 0; c < counts[j]; ++c)
                    seq_desc.push_back(lengths[j]);
            for (int j = 0; j < K; ++j)
                for (int c = 0; c < counts[j]; ++c)
                    seq_asc.push_back(lengths[j]);

            double heuristic = kInf;
            if (!seq_desc.empty())
                heuristic = std::min(heuristic, solve_fixed_sequence(seq_desc, local_prefix, local_T, local_spaces));
            if (!seq_asc.empty())
                heuristic = std::min(heuristic, solve_fixed_sequence(seq_asc, local_prefix, local_T, local_spaces));

            int64_t total_cells = static_cast<int64_t>(local_T + 2) * theoretical_states;
            if (total_cells > l3_max_cells)
                return heuristic;

            double exact = solve_exact_multiset_dp(
                lengths,
                counts,
                local_prefix,
                local_T,
                local_spaces,
                heuristic,
                l3_time_limit);
            if (exact < kInf * 0.5)
                return exact;
            return heuristic;
        }

        double block_repair_feasible_beam_ub(
            const std::vector<RecoveredBlock> &merged,
            const std::vector<int> &lengths,
            const std::vector<int> &totals,
            const std::vector<double> &prefix_proc,
            int T,
            const SPACESResult &spaces,
            std::vector<std::vector<int>> *chosen_counts_out,
            ProfileRepairBeamDiag *diag_out,
            double known_ub,
            bool strengthened);

        double block_repair_energy_core_ub(
            const std::vector<RecoveredBlock> &merged,
            const std::vector<int> &lengths,
            const std::vector<int> &totals,
            const std::vector<double> &prefix_proc,
            int T,
            const SPACESResult &spaces,
            double known_ub,
            EnergyCoreRunDiag *diag_out = nullptr)
        {
            if (diag_out)
                *diag_out = EnergyCoreRunDiag{};
            int K = static_cast<int>(lengths.size());
            if (merged.empty() || K <= 2 || K > kMaxRepairTypes)
                return kInf;
            bool trace = env_int_or("PAST_BLOCK_REPAIR_TRACE", 0) != 0;

            int total_work = 0;
            for (int i = 0; i < K; ++i)
                total_work += lengths[i] * totals[i];

            int total_jobs = 0;
            for (int x : totals)
                total_jobs += x;

            auto t0_completion = std::chrono::steady_clock::now();
            auto completion = compute_relaxed_completion_table(
                lengths, total_work, prefix_proc, T, spaces, RelaxationMode::Semigroup);
            double t_completion =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_completion).count();
            auto t0_patterns = std::chrono::steady_clock::now();

            EnergyCorePatternPoolDiag pool_diag;
            auto patterns = generate_energy_core_patterns(merged, lengths, totals, T, &pool_diag);
            double t_patterns =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_patterns).count();

            if (diag_out)
            {
                diag_out->pool = pool_diag;
                diag_out->retained_fixed_blocks = pool_diag.fixed_blocks;
                diag_out->t_completion = t_completion;
                diag_out->t_patterns = t_patterns;
            }

            if (trace)
            {
                std::cerr << "block_repair_trace method=energy_core_prepare"
                          << " merged_blocks=" << merged.size()
                          << " t_completion=" << t_completion
                          << " t_patterns=" << t_patterns
                          << " generated_total=" << pool_diag.generated_total
                          << " generated_max=" << pool_diag.generated_max_block
                          << " retained_total=" << pool_diag.retained_total
                          << " retained_max=" << pool_diag.retained_max_block
                          << " fixed_blocks=" << pool_diag.fixed_blocks
                          << "\n";
            }
            for (const auto &vec : patterns)
                if (vec.empty())
                    return kInf;

            int B = static_cast<int>(merged.size());
            std::vector<int> prefix_target(B + 1, 0);
            for (int bi = 0; bi < B; ++bi)
                prefix_target[bi + 1] = prefix_target[bi] + merged[bi].length;

            std::vector<std::vector<int>> suffix_min(B + 1, std::vector<int>(K, 0));
            std::vector<std::vector<int>> suffix_max(B + 1, std::vector<int>(K, 0));
            std::vector<std::vector<int>> block_spread(B, std::vector<int>(K, 0));
            for (int bi = B - 1; bi >= 0; --bi)
            {
                for (int j = 0; j < K; ++j)
                {
                    int mn = INT_MAX;
                    int mx = 0;
                    for (const auto &pat : patterns[bi])
                    {
                        mn = std::min(mn, pat.counts[j]);
                        mx = std::max(mx, pat.counts[j]);
                    }
                    if (mn == INT_MAX)
                        mn = 0;
                    block_spread[bi][j] = std::max(0, mx - mn);
                    suffix_min[bi][j] = suffix_min[bi + 1][j] + mn;
                    suffix_max[bi][j] = suffix_max[bi + 1][j] + mx;
                }
            }

            std::vector<std::vector<double>> block_center(B, std::vector<double>(K, 0.0));
            bool stronger_center = env_int_or("PAST_BLOCK_REPAIR_EC_STRONGER_CENTER", 1) != 0;
            int center_topk = std::max(1, env_int_or("PAST_BLOCK_REPAIR_CORE_CENTER_TOPK", 6));
            double center_pattern_w = stronger_center
                                          ? std::min(1.0, std::max(0.0, env_double_or("PAST_BLOCK_REPAIR_CORE_PATTERN_CENTER_W", 0.60)))
                                          : 0.0;
            for (int bi = 0; bi < B; ++bi)
            {
                int cap = merged[bi].length;
                for (int j = 0; j < K; ++j)
                {
                    double cap_mu = (total_work > 0)
                                        ? (static_cast<double>(totals[j]) * cap / total_work)
                                        : 0.0;
                    int take = std::min(center_topk, static_cast<int>(patterns[bi].size()));
                    double num = 0.0;
                    double den = 0.0;
                    for (int pi = 0; pi < take; ++pi)
                    {
                        const auto &pat = patterns[bi][pi];
                        double w = 1.0 / (1.0 + pat.local_dev + 0.25 * pat.center_dev);
                        num += w * pat.counts[j];
                        den += w;
                    }
                    double pat_mu = (den > 0.0) ? (num / den) : cap_mu;
                    block_center[bi][j] = (1.0 - center_pattern_w) * cap_mu + center_pattern_w * pat_mu;
                }
            }

            std::vector<std::vector<double>> prefix_center(B + 1, std::vector<double>(K, 0.0));
            for (int bi = 0; bi < B; ++bi)
            {
                for (int j = 0; j < K; ++j)
                    prefix_center[bi + 1][j] = prefix_center[bi][j] + block_center[bi][j];
            }

            bool adaptive_delta = env_int_or("PAST_BLOCK_REPAIR_EC_ADAPTIVE_DELTA", 1) != 0;
            std::vector<int> type_bonus(K, 0);
            int scarce_cut = std::max(2, total_jobs / std::max(4, 3 * K));
            int max_len = 0;
            for (int L : lengths)
                max_len = std::max(max_len, L);
            if (adaptive_delta)
            {
                for (int j = 0; j < K; ++j)
                {
                    if (totals[j] <= scarce_cut)
                        type_bonus[j] += 1;
                    if (lengths[j] >= max_len - 1)
                        type_bonus[j] += 1;
                }
            }

            std::vector<int> pat_sizes(B, 0);
            for (int bi = 0; bi < B; ++bi)
                pat_sizes[bi] = static_cast<int>(patterns[bi].size());
            std::vector<int> pat_sizes_sorted = pat_sizes;
            std::sort(pat_sizes_sorted.begin(), pat_sizes_sorted.end());
            int pat_median = pat_sizes_sorted.empty() ? 0 : pat_sizes_sorted[pat_sizes_sorted.size() / 2];
            std::vector<int> block_bonus(B, 0);
            if (adaptive_delta)
            {
                for (int bi = 0; bi < B; ++bi)
                {
                    int spread_sum = 0;
                    for (int j = 0; j < K; ++j)
                        spread_sum += block_spread[bi][j];
                    int avg_spread = (K > 0) ? (spread_sum / K) : 0;
                    if (pat_sizes[bi] > pat_median)
                        block_bonus[bi] += 1;
                    if (avg_spread >= 2)
                        block_bonus[bi] += 1;
                }
            }

            int delta_init = env_int_or("PAST_BLOCK_REPAIR_CORE_DELTA", K <= 3 ? 4 : (K <= 5 ? 3 : 2));
            int delta_max = env_int_or("PAST_BLOCK_REPAIR_CORE_MAX_DELTA", K <= 3 ? 10 : (K <= 5 ? 8 : 7));
            int delta_step = std::max(1, env_int_or("PAST_BLOCK_REPAIR_CORE_DELTA_STEP", 1));
            int state_keep = env_int_or("PAST_BLOCK_REPAIR_EG_STATE_KEEP", K <= 3 ? 30000 : (K <= 5 ? 15000 : 6000));
            double best = known_ub;
            bool two_phase = env_int_or("PAST_BLOCK_REPAIR_EC_TWO_PHASE", 1) != 0;
            if (diag_out)
                diag_out->two_phase_used = two_phase ? 1 : 0;

            if (two_phase)
            {
                auto t0_phase1 = std::chrono::steady_clock::now();
                double phase1_ub = block_repair_feasible_beam_ub(
                    merged, lengths, totals, prefix_proc, T, spaces, nullptr, nullptr, best, false);
                double t_phase1 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_phase1).count();
                if (diag_out)
                {
                    diag_out->phase1_feasible_ub = phase1_ub;
                    diag_out->t_phase1 = t_phase1;
                }
                if (phase1_ub < best)
                    best = phase1_ub;

                double close_eps = env_double_or("PAST_BLOCK_REPAIR_EC_PHASE1_CLOSE_EPS", 0.01);
                if (phase1_ub < kInf * 0.5 && completion.lb < kInf * 0.5 && phase1_ub <= completion.lb + close_eps)
                {
                    if (diag_out)
                    {
                        diag_out->delta_used = 0;
                        diag_out->t_exact_core = 0.0;
                    }
                    return phase1_ub;
                }
            }

            auto t0_exact_core = std::chrono::steady_clock::now();
            double total_core_pruned = 0.0;
            double total_suffix_pruned = 0.0;
            double total_transition_pruned = 0.0;
            double total_bound_pruned = 0.0;
            int used_delta = -1;

            for (int delta = delta_init; delta <= delta_max; delta += delta_step)
            {
                std::vector<std::vector<int>> core_lo(B + 1, std::vector<int>(K, 0));
                std::vector<std::vector<int>> core_hi(B + 1, std::vector<int>(K, 0));
                for (int bi = 0; bi < B; ++bi)
                {
                    for (int j = 0; j < K; ++j)
                    {
                        int adaptive_delta = delta;
                        if (adaptive_delta < 0)
                            adaptive_delta = 0;
                        if (adaptive_delta > 0 && adaptive_delta + type_bonus[j] > adaptive_delta)
                            adaptive_delta += type_bonus[j];
                        if (adaptive_delta > 0 && delta > delta_init && adaptive_delta + block_bonus[bi] > adaptive_delta)
                            adaptive_delta += block_bonus[bi];
                        if (adaptive_delta > 0 && block_spread[bi][j] > 2 * adaptive_delta)
                            adaptive_delta += 1;
                        double mu = prefix_center[bi + 1][j];
                        core_lo[bi + 1][j] = std::max(0, static_cast<int>(std::floor(mu)) - adaptive_delta);
                        core_hi[bi + 1][j] = std::min(totals[j], static_cast<int>(std::ceil(mu)) + adaptive_delta);
                    }
                }
                for (int j = 0; j < K; ++j)
                    core_lo[B][j] = core_hi[B][j] = totals[j];

                std::vector<EnergyCoreNode> layer;
                EnergyCoreNode seed;
                seed.counts.fill(0);
                seed.prev_end = -1;
                seed.prefix_work = 0;
                seed.g = 0.0;
                seed.f = lookup_completion_lb(completion.off_rdp, completion.RW, completion.rw_scale, T, 0, total_work);
                layer.push_back(seed);

                bool expanded_all = true;
                for (int bi = 0; bi < B; ++bi)
                {
                    std::size_t n_considered = 0;
                    std::size_t n_core_pruned = 0;
                    std::size_t n_suffix_pruned = 0;
                    std::size_t n_transition_pruned = 0;
                    std::size_t n_h_pruned = 0;
                    std::unordered_map<EnergyCoreKey, EnergyCoreNode, EnergyCoreKeyHash> next_best;
                    next_best.reserve(static_cast<std::size_t>(std::max(64, state_keep * 2)));
                    for (const auto &state : layer)
                    {
                        if (state.g >= best)
                            continue;
                        for (const auto &pat : patterns[bi])
                        {
                            ++n_considered;
                            EnergyCoreNode nxt = state;
                            bool counts_ok = true;
                            for (int j = 0; j < K; ++j)
                            {
                                nxt.counts[j] += pat.counts[j];
                                if (nxt.counts[j] < core_lo[bi + 1][j] || nxt.counts[j] > core_hi[bi + 1][j])
                                {
                                    counts_ok = false;
                                    break;
                                }
                            }
                            if (!counts_ok)
                            {
                                ++n_core_pruned;
                                total_core_pruned += 1.0;
                                continue;
                            }

                            bool suffix_ok = true;
                            for (int j = 0; j < K; ++j)
                            {
                                int rem = totals[j] - nxt.counts[j];
                                if (rem < suffix_min[bi + 1][j] || rem > suffix_max[bi + 1][j])
                                {
                                    suffix_ok = false;
                                    break;
                                }
                            }
                            if (!suffix_ok)
                            {
                                ++n_suffix_pruned;
                                total_suffix_pruned += 1.0;
                                continue;
                            }

                            double incr = 0.0;
                            int new_prev_end = state.prev_end;
                            if (pat.work > 0)
                            {
                                int start = merged[bi].start;
                                if (state.prev_end < 0)
                                {
                                    if (spaces.c_start[start] >= kInf)
                                    {
                                        ++n_transition_pruned;
                                        total_transition_pruned += 1.0;
                                        continue;
                                    }
                                    incr += spaces.c_start[start];
                                }
                                else
                                {
                                    double gap = spaces.gap_cost(state.prev_end, start);
                                    if (gap >= kInf)
                                    {
                                        ++n_transition_pruned;
                                        total_transition_pruned += 1.0;
                                        continue;
                                    }
                                    incr += gap;
                                }
                                int block_end = start + pat.work;
                                if (block_end > T || block_end > spaces.late + 1)
                                {
                                    ++n_transition_pruned;
                                    total_transition_pruned += 1.0;
                                    continue;
                                }
                                incr += prefix_proc[block_end] - prefix_proc[start];
                                new_prev_end = block_end;
                            }

                            nxt.prefix_work += pat.work;
                            nxt.g += incr;
                            nxt.prev_end = new_prev_end;
                            int time_anchor = (new_prev_end >= 0) ? new_prev_end : merged[bi].start;
                            int rw_remaining = total_work - nxt.prefix_work;
                            const auto &h_table = (new_prev_end >= 0) ? completion.rdp : completion.off_rdp;
                            double h = lookup_completion_lb(h_table, completion.RW, completion.rw_scale,
                                                            T, time_anchor, rw_remaining);
                            nxt.f = nxt.g + h;
                            if (nxt.f >= best)
                            {
                                ++n_h_pruned;
                                total_bound_pruned += 1.0;
                                continue;
                            }

                            EnergyCoreKey key = energy_core_key(nxt, K);
                            auto it = next_best.find(key);
                            if (it == next_best.end() || nxt.g < it->second.g - kEps || (std::abs(nxt.g - it->second.g) <= kEps && nxt.f < it->second.f))
                                next_best[key] = std::move(nxt);
                        }
                    }

                    std::vector<EnergyCoreNode> next_layer;
                    next_layer.reserve(next_best.size());
                    for (auto &kv : next_best)
                        next_layer.push_back(std::move(kv.second));
                    std::sort(next_layer.begin(), next_layer.end(), [](const EnergyCoreNode &a, const EnergyCoreNode &b)
                              {
                                  if (std::abs(a.f - b.f) > kEps)
                                      return a.f < b.f;
                                  return a.g < b.g;
                              });
                    if (static_cast<int>(next_layer.size()) > state_keep)
                        next_layer.resize(state_keep);
                    if (trace)
                    {
                        std::cerr << "block_repair_trace method=energy_core"
                                  << " delta=" << delta
                                  << " block=" << bi
                                  << " layer_in=" << layer.size()
                                  << " patterns=" << patterns[bi].size()
                                  << " considered=" << n_considered
                                  << " core_pruned=" << n_core_pruned
                                  << " suffix_pruned=" << n_suffix_pruned
                                  << " transition_pruned=" << n_transition_pruned
                                  << " h_pruned=" << n_h_pruned
                                  << " kept=" << next_layer.size()
                                  << " best=" << best
                                  << "\n";
                    }
                    if (next_layer.empty())
                    {
                        expanded_all = false;
                        break;
                    }
                    layer = std::move(next_layer);
                }

                if (!expanded_all)
                    continue;

                for (const auto &state : layer)
                {
                    bool target = true;
                    for (int j = 0; j < K; ++j)
                    {
                        if (state.counts[j] != totals[j])
                        {
                            target = false;
                            break;
                        }
                    }
                    if (!target)
                        continue;
                    double total = state.g;
                    if (state.prev_end >= 0)
                    {
                        if (spaces.c_end[state.prev_end] >= kInf)
                            continue;
                        total += spaces.c_end[state.prev_end];
                    }
                    if (total < best)
                    {
                        best = total;
                        if (used_delta < 0)
                            used_delta = delta;
                    }
                }

                if (best < known_ub)
                    break;
            }

            double t_exact_core =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_exact_core).count();

            if (diag_out)
            {
                diag_out->t_exact_core = t_exact_core;
                diag_out->pruned_core_window = total_core_pruned;
                diag_out->pruned_suffix = total_suffix_pruned;
                diag_out->pruned_transition = total_transition_pruned;
                diag_out->pruned_bound = total_bound_pruned;
                diag_out->delta_used = used_delta;
            }

            return best;
        }

        double block_repair_feasible_counts_ub(
            const std::vector<RecoveredBlock> &merged,
            const std::vector<int> &lengths,
            const std::vector<int> &totals,
            const std::vector<double> &prefix_proc,
            int T,
            const SPACESResult &spaces)
        {
            int K = static_cast<int>(lengths.size());
            if (merged.empty() || K <= 2 || K > kMaxRepairTypes)
                return kInf;

            bool trace = env_int_or("PAST_BLOCK_REPAIR_TRACE", 0) != 0;
            int max_layer_states = env_int_or("PAST_BLOCK_REPAIR_FEAS_MAX_LAYER", K >= 6 ? 500000 : 1000000);
            auto patterns = generate_energy_core_patterns(merged, lengths, totals, T);
            for (const auto &vec : patterns)
                if (vec.empty())
                    return kInf;

            int B = static_cast<int>(merged.size());
            std::vector<std::vector<int>> suffix_min(B + 1, std::vector<int>(K, 0));
            std::vector<std::vector<int>> suffix_max(B + 1, std::vector<int>(K, 0));
            for (int bi = B - 1; bi >= 0; --bi)
            {
                for (int j = 0; j < K; ++j)
                {
                    int mn = INT_MAX;
                    int mx = 0;
                    for (const auto &pat : patterns[bi])
                    {
                        mn = std::min(mn, pat.counts[j]);
                        mx = std::max(mx, pat.counts[j]);
                    }
                    if (mn == INT_MAX)
                        mn = 0;
                    suffix_min[bi][j] = suffix_min[bi + 1][j] + mn;
                    suffix_max[bi][j] = suffix_max[bi + 1][j] + mx;
                }
            }

            struct FeasNode
            {
                std::array<int, kMaxRepairTypes> counts{};
                int parent_idx = -1;
                int pat_idx = -1;
            };

            std::vector<std::vector<FeasNode>> layers(B + 1);
            layers[0].push_back(FeasNode{});

            for (int bi = 0; bi < B; ++bi)
            {
                std::unordered_map<CountKey, int, CountKeyHash> next_index;
                next_index.reserve(static_cast<std::size_t>(std::max(256, static_cast<int>(layers[bi].size()) * 4)));
                std::vector<FeasNode> next_layer;

                std::size_t n_considered = 0;
                std::size_t n_over_pruned = 0;
                std::size_t n_suffix_pruned = 0;

                for (int si = 0; si < static_cast<int>(layers[bi].size()); ++si)
                {
                    const auto &state = layers[bi][si];
                    for (int pi = 0; pi < static_cast<int>(patterns[bi].size()); ++pi)
                    {
                        ++n_considered;
                        const auto &pat = patterns[bi][pi];
                        FeasNode nxt;
                        nxt.counts = state.counts;
                        nxt.parent_idx = si;
                        nxt.pat_idx = pi;

                        bool ok = true;
                        for (int j = 0; j < K; ++j)
                        {
                            nxt.counts[j] += pat.counts[j];
                            if (nxt.counts[j] > totals[j])
                            {
                                ok = false;
                                break;
                            }
                        }
                        if (!ok)
                        {
                            ++n_over_pruned;
                            continue;
                        }
                        for (int j = 0; j < K; ++j)
                        {
                            int rem = totals[j] - nxt.counts[j];
                            if (rem < suffix_min[bi + 1][j] || rem > suffix_max[bi + 1][j])
                            {
                                ok = false;
                                break;
                            }
                        }
                        if (!ok)
                        {
                            ++n_suffix_pruned;
                            continue;
                        }

                        CountKey key = count_key_from_counts(nxt.counts, K);
                        if (next_index.find(key) == next_index.end())
                        {
                            int idx = static_cast<int>(next_layer.size());
                            next_index.emplace(key, idx);
                            next_layer.push_back(std::move(nxt));
                            if (static_cast<int>(next_layer.size()) > max_layer_states)
                            {
                                if (trace)
                                {
                                    std::cerr << "block_repair_trace method=feasible_counts_abort"
                                              << " block=" << bi
                                              << " reached_layer_states=" << next_layer.size()
                                              << " max_layer_states=" << max_layer_states
                                              << "\n";
                                }
                                return kInf;
                            }
                        }
                    }
                }

                if (trace)
                {
                    std::cerr << "block_repair_trace method=feasible_counts"
                              << " block=" << bi
                              << " layer_in=" << layers[bi].size()
                              << " patterns=" << patterns[bi].size()
                              << " considered=" << n_considered
                              << " over_pruned=" << n_over_pruned
                              << " suffix_pruned=" << n_suffix_pruned
                              << " kept=" << next_layer.size()
                              << "\n";
                }

                if (next_layer.empty())
                    return kInf;
                layers[bi + 1] = std::move(next_layer);
            }

            int target_idx = -1;
            for (int si = 0; si < static_cast<int>(layers[B].size()); ++si)
            {
                bool target = true;
                for (int j = 0; j < K; ++j)
                {
                    if (layers[B][si].counts[j] != totals[j])
                    {
                        target = false;
                        break;
                    }
                }
                if (target)
                {
                    target_idx = si;
                    break;
                }
            }
            if (target_idx < 0)
                return kInf;

            std::vector<int> chosen_pat(B, -1);
            int cur_idx = target_idx;
            for (int bi = B; bi >= 1; --bi)
            {
                const auto &node = layers[bi][cur_idx];
                chosen_pat[bi - 1] = node.pat_idx;
                cur_idx = node.parent_idx;
            }

            std::vector<int> seq_desc;
            std::vector<int> seq_asc;
            for (int bi = 0; bi < B; ++bi)
            {
                const auto &pat = patterns[bi][chosen_pat[bi]];
                for (int j = K - 1; j >= 0; --j)
                    for (int c = 0; c < pat.counts[j]; ++c)
                        seq_desc.push_back(lengths[j]);
                for (int j = 0; j < K; ++j)
                    for (int c = 0; c < pat.counts[j]; ++c)
                        seq_asc.push_back(lengths[j]);
            }

            double best = kInf;
            if (!seq_desc.empty())
                best = std::min(best, solve_fixed_sequence(seq_desc, prefix_proc, T, spaces));
            if (!seq_asc.empty())
                best = std::min(best, solve_fixed_sequence(seq_asc, prefix_proc, T, spaces));
            return best;
        }

        double block_repair_feasible_beam_ub(
            const std::vector<RecoveredBlock> &merged,
            const std::vector<int> &lengths,
            const std::vector<int> &totals,
            const std::vector<double> &prefix_proc,
            int T,
            const SPACESResult &spaces,
            std::vector<std::vector<int>> *chosen_counts_out = nullptr,
            ProfileRepairBeamDiag *diag_out = nullptr,
            double known_ub = kInf,
            bool strengthened = false)
        {
            int K = static_cast<int>(lengths.size());
            if (chosen_counts_out)
                chosen_counts_out->clear();
            if (diag_out)
                *diag_out = ProfileRepairBeamDiag{};
            if (merged.empty() || K <= 2 || K > kMaxRepairTypes)
            {
                if (diag_out)
                    diag_out->status = "skipped_shape";
                return kInf;
            }

            bool trace = env_int_or("PAST_BLOCK_REPAIR_TRACE", 0) != 0;
            int base_beam_width = env_int_or("PAST_BLOCK_REPAIR_FEAS_BEAM_WIDTH", K >= 6 ? 200000 : 100000);
            int adaptive_width_min = env_int_or("PAST_PROFILE_REPAIR_BEAM_WIDTH_MIN", std::max(64, base_beam_width / 2));
            int adaptive_width_max = env_int_or("PAST_PROFILE_REPAIR_BEAM_WIDTH_MAX", std::max(base_beam_width, base_beam_width * 3));
            int discrepancy_budget = env_int_or("PAST_PROFILE_REPAIR_BEAM_DISC_BUDGET", K <= 4 ? 0 : 1);
            int discrepancy_depth = env_int_or("PAST_PROFILE_REPAIR_BEAM_DISC_DEPTH", std::min(K <= 4 ? 2 : 4, static_cast<int>(merged.size())));
            int discrepancy_topk = env_int_or("PAST_PROFILE_REPAIR_BEAM_DISC_TOPK", 1);
            if (strengthened)
            {
                int width_pct = std::max(100, env_int_or("PAST_PROFILE_REPAIR_BEAM_STRONG_WIDTH_PCT", 160));
                base_beam_width = std::max(base_beam_width, static_cast<int>((static_cast<long long>(base_beam_width) * width_pct) / 100LL));
                adaptive_width_max = std::max(adaptive_width_max, base_beam_width * 2);
                discrepancy_budget += std::max(0, env_int_or("PAST_PROFILE_REPAIR_BEAM_STRONG_DISC_BONUS", 1));
                discrepancy_topk = std::max(discrepancy_topk, env_int_or("PAST_PROFILE_REPAIR_BEAM_STRONG_TOPK", 2));
            }
            discrepancy_budget = std::max(0, discrepancy_budget);
            discrepancy_depth = std::max(0, discrepancy_depth);
            discrepancy_topk = std::max(1, discrepancy_topk);
            double beam_time_limit = std::max(0.0, env_double_or("PAST_PROFILE_REPAIR_BEAM_TIME_LIMIT", 0.0));
            auto t0_beam = std::chrono::steady_clock::now();
            auto beam_out_of_time = [&]() -> bool
            {
                if (beam_time_limit <= 0.0)
                    return false;
                double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_beam).count();
                return elapsed >= beam_time_limit;
            };

            double w_center = env_double_or("PAST_PROFILE_REPAIR_BEAM_W_CENTER", 1.0);
            double w_feas = env_double_or("PAST_PROFILE_REPAIR_BEAM_W_FEAS", 0.75);
            double w_local = env_double_or("PAST_PROFILE_REPAIR_BEAM_W_LOCAL", 0.6);
            double w_arith = env_double_or("PAST_PROFILE_REPAIR_BEAM_W_ARITH", 1.0);
            double filler_target = env_double_or("PAST_PROFILE_REPAIR_BEAM_FILLER_TARGET", 0.08);
            int local_rank_l3_max_cells = env_int_or("PAST_PROFILE_REPAIR_BEAM_RANK_L3_MAX_CELLS", 20'000);
            double local_rank_l3_time = env_double_or("PAST_PROFILE_REPAIR_BEAM_RANK_L3_TIME", 0.01);

            // PLAN20B additive beam-survivor experiments
            int key_multiplicity = std::max(1, env_int_or("PAST_PROFILE_REPAIR_BEAM_KEY_MULTI", 1));
            std::string bucket_split_str = env_str_or("PAST_PROFILE_REPAIR_BEAM_BUCKET_SPLIT", "");
            std::string bucket_metric_str = env_str_or("PAST_PROFILE_REPAIR_BEAM_BUCKET_METRIC", "score");

            // PLAN22 adaptive node evaluation / survivor policy
            std::string key_multi_policy = env_str_or("PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_POLICY", "off");
            int key_multi_max = std::max(1, env_int_or("PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_MAX", 1));
            double key_multi_early_frac = env_double_or("PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_EARLY_FRAC", 0.4);
            double key_multi_score_eps = env_double_or("PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_SCORE_EPS", 0.05);
            double key_multi_diversity_eps = env_double_or("PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_DIVERSITY_EPS", 0.05);

            // PLAN23 role-based survivor policy
            double role_score_band = env_double_or("PAST_PROFILE_REPAIR_BEAM_ROLE_SCORE_BAND", 0.08);
            int role_keep_feas = env_int_or("PAST_PROFILE_REPAIR_BEAM_ROLE_KEEP_FEAS", 0);
            int role_max = std::max(1, env_int_or("PAST_PROFILE_REPAIR_BEAM_ROLE_MAX", 3));
            if (key_multi_policy == "role")
                key_multi_max = role_max;

            // PLAN27 adaptive survivor policy: late ambiguity + residual-aware scoring
            double late_frac = env_double_or("PAST_PROFILE_REPAIR_BEAM_LATE_FRAC", 0.35);
            std::string score_policy = env_str_or("PAST_PROFILE_REPAIR_BEAM_SCORE_POLICY", "default");
            double residual_weight = env_double_or("PAST_PROFILE_REPAIR_BEAM_RESIDUAL_WEIGHT", 0.2);
            if (residual_weight < 0.0)
                residual_weight = 0.0;
            if (residual_weight > 2.0)
                residual_weight = 2.0;

            auto raw_patterns = generate_energy_core_patterns(merged, lengths, totals, T);
            for (const auto &vec : raw_patterns)
                if (vec.empty())
                    return kInf;

            int B = static_cast<int>(merged.size());
            std::vector<int> block_order = profile_realization_block_order(merged, raw_patterns);

            std::vector<RecoveredBlock> merged_ordered(B);
            std::vector<std::vector<RepairPattern>> patterns(B);
            std::vector<int> ordered_to_orig(B, -1);
            for (int pos = 0; pos < B; ++pos)
            {
                int bi = block_order[pos];
                ordered_to_orig[pos] = bi;
                merged_ordered[pos] = merged[bi];
                patterns[pos] = raw_patterns[bi];
            }

            int total_work = 0;
            for (int j = 0; j < K; ++j)
                total_work += lengths[j] * totals[j];

            std::vector<int> prefix_target(B + 1, 0);
            for (int bi = 0; bi < B; ++bi)
                prefix_target[bi + 1] = prefix_target[bi] + merged_ordered[bi].length;

            std::vector<int> suffix_cap(B + 1, 0);
            for (int bi = B - 1; bi >= 0; --bi)
                suffix_cap[bi] = suffix_cap[bi + 1] + merged_ordered[bi].length;

            std::vector<std::vector<int>> suffix_min(B + 1, std::vector<int>(K, 0));
            std::vector<std::vector<int>> suffix_max(B + 1, std::vector<int>(K, 0));
            for (int bi = B - 1; bi >= 0; --bi)
            {
                for (int j = 0; j < K; ++j)
                {
                    int mn = INT_MAX;
                    int mx = 0;
                    for (const auto &pat : patterns[bi])
                    {
                        mn = std::min(mn, pat.counts[j]);
                        mx = std::max(mx, pat.counts[j]);
                    }
                    if (mn == INT_MAX)
                        mn = 0;
                    suffix_min[bi][j] = suffix_min[bi + 1][j] + mn;
                    suffix_max[bi][j] = suffix_max[bi + 1][j] + mx;
                }
            }

            std::vector<int> type_order_small(K);
            std::iota(type_order_small.begin(), type_order_small.end(), 0);
            std::sort(type_order_small.begin(), type_order_small.end(), [&](int a, int b)
                      {
                          if (lengths[a] != lengths[b])
                              return lengths[a] < lengths[b];
                          return a < b;
                      });

            std::vector<SPACESResult> block_spaces;
            std::vector<std::vector<double>> block_prefix_proc;
            build_profile_block_local_views(
                merged_ordered,
                prefix_proc,
                T,
                spaces,
                &block_spaces,
                &block_prefix_proc);

            auto eval_local_pattern_cost = [&](int bi, const std::vector<int> &counts) -> double
            {
                return evaluate_profile_block_counts(
                    bi,
                    counts,
                    lengths,
                    merged_ordered,
                    block_spaces,
                    block_prefix_proc,
                    local_rank_l3_max_cells,
                    local_rank_l3_time);
            };

            std::vector<std::vector<double>> pattern_local_rank(B);
            std::vector<std::vector<int>> pattern_pref_rank(B);
            for (int bi = 0; bi < B; ++bi)
            {
                int P = static_cast<int>(patterns[bi].size());
                std::vector<double> local_cost(P, kInf);
                for (int pi = 0; pi < P; ++pi)
                    local_cost[pi] = eval_local_pattern_cost(bi, patterns[bi][pi].counts);

                std::vector<int> order(P);
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(), [&](int a, int b)
                          {
                              if (local_cost[a] != local_cost[b])
                                  return local_cost[a] < local_cost[b];
                              if (patterns[bi][a].local_dev != patterns[bi][b].local_dev)
                                  return patterns[bi][a].local_dev < patterns[bi][b].local_dev;
                              if (patterns[bi][a].center_dev != patterns[bi][b].center_dev)
                                  return patterns[bi][a].center_dev < patterns[bi][b].center_dev;
                              return a < b;
                          });
                pattern_local_rank[bi].assign(P, 1.0);
                pattern_pref_rank[bi].assign(P, P + 1);
                double denom = std::max(1, P - 1);
                for (int rank = 0; rank < P; ++rank)
                {
                    int pi = order[rank];
                    pattern_local_rank[bi][pi] = static_cast<double>(rank) / denom;
                    pattern_pref_rank[bi][pi] = rank;
                }
            }

            struct FeasBeamNode
            {
                std::array<int, kMaxRepairTypes> counts{};
                int parent_idx = -1;
                int pat_idx = -1;
                double score = 0.0;
                double s_center = 0.0;
                double s_feas = 0.0;
                double s_local = 0.0;
                double s_arith = 0.0;
                int discrepancy = 0;
                int prefix_work = 0;
            };

            struct FeasBeamKey
            {
                CountKey key;
                uint8_t disc = 0;
                bool operator==(const FeasBeamKey &other) const noexcept
                {
                    return disc == other.disc && key == other.key;
                }
            };

            struct FeasBeamKeyHash
            {
                std::size_t operator()(const FeasBeamKey &k) const noexcept
                {
                    std::size_t h = CountKeyHash{}(k.key);
                    h ^= static_cast<std::size_t>(k.disc) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
                    return h;
                }
            };

            auto center_score = [&](const std::array<int, kMaxRepairTypes> &counts, int upto_block) -> double
            {
                double score = 0.0;
                double pref_cap = static_cast<double>(prefix_target[upto_block]);
                for (int j = 0; j < K; ++j)
                {
                    double mu = total_work > 0 ? static_cast<double>(totals[j]) * pref_cap / total_work : 0.0;
                    score += std::abs(static_cast<double>(counts[j]) - mu);
                }
                return score;
            };

            auto feasibility_pressure = [&](const std::array<int, kMaxRepairTypes> &counts, int next_block) -> double
            {
                double pressure = 0.0;
                for (int j = 0; j < K; ++j)
                {
                    int rem = totals[j] - counts[j];
                    int slack_lo = rem - suffix_min[next_block][j];
                    int slack_hi = suffix_max[next_block][j] - rem;
                    if (slack_lo < 0 || slack_hi < 0)
                        return 1.0e9;
                    pressure += 1.0 / (1.0 + static_cast<double>(slack_lo));
                    pressure += 1.0 / (1.0 + static_cast<double>(slack_hi));
                }
                return pressure;
            };

            auto arithmetic_pressure = [&](const std::array<int, kMaxRepairTypes> &counts, int next_block, int prefix_work) -> double
            {
                int rem_work = std::max(0, total_work - prefix_work);
                if (rem_work <= 0)
                    return 0.0;

                int small_work = 0;
                for (int idx = 0; idx < std::min(2, K); ++idx)
                {
                    int j = type_order_small[idx];
                    int rem = std::max(0, totals[j] - counts[j]);
                    small_work += rem * lengths[j];
                }
                double filler_ratio = static_cast<double>(small_work) / std::max(1, rem_work);
                double filler_pen = std::max(0.0, filler_target - filler_ratio);

                int mod_base = (K >= 2) ? std::max(2, lengths[type_order_small[1]]) : std::max(2, lengths[type_order_small[0]]);
                int residue = rem_work % mod_base;
                residue = std::min(residue, mod_base - residue);
                double residue_pen = static_cast<double>(residue) / mod_base;

                double rw_gap_pen = std::abs(rem_work - suffix_cap[next_block]) /
                                    static_cast<double>(std::max(1, suffix_cap[next_block]));
                return filler_pen + residue_pen + rw_gap_pen;
            };

            auto residual_penalty = [&](const std::array<int, kMaxRepairTypes> &counts, int next_block, int prefix_work) -> double
            {
                int rem_work = std::max(0, total_work - prefix_work);
                if (rem_work <= 0)
                    return 0.0;

                // remaining large-job pressure: fraction of remaining work in largest 3 types
                int large_work = 0;
                for (int idx = std::max(0, K - 3); idx < K; ++idx)
                {
                    int j = type_order_small[idx];
                    int rem = std::max(0, totals[j] - counts[j]);
                    large_work += rem * lengths[j];
                }
                double large_pressure = static_cast<double>(large_work) / std::max(1, rem_work);

                // type count imbalance: coefficient of variation of remaining counts weighted by length
                double rem_mean = 0.0, rem_var = 0.0;
                int rem_types = 0;
                for (int j = 0; j < K; ++j)
                {
                    int rem = std::max(0, totals[j] - counts[j]);
                    if (rem > 0)
                    {
                        rem_mean += rem;
                        rem_types++;
                    }
                }
                if (rem_types > 1)
                {
                    rem_mean /= rem_types;
                    for (int j = 0; j < K; ++j)
                    {
                        int rem = std::max(0, totals[j] - counts[j]);
                        if (rem > 0)
                        {
                            double d = static_cast<double>(rem) - rem_mean;
                            rem_var += d * d;
                        }
                    }
                    rem_var /= rem_types;
                }
                double imbalance = std::sqrt(rem_var) / std::max(1.0, rem_mean);

                // capacity slack relative to remaining work
                double slack = static_cast<double>(suffix_cap[next_block] - rem_work) / std::max(1, rem_work);

                return large_pressure + 0.5 * imbalance + 1.0 / (1.0 + 10.0 * slack);
            };

            // PLAN31: coarse-lookahead scoring — uses relaxed profile targets
            // without replacing fine blocks. Precompute per-block target work
            // proportional to block capacity.
            double lookahead_weight = 0.0;
            std::vector<double> block_target_work(B, 0.0);
            std::vector<double> block_price_vol(B, 0.0);
            int lookahead_window = 0;
            if (score_policy == "fine_plus_coarse_lookahead")
            {
                lookahead_weight = residual_weight; // reuse residual_weight as lookahead strength
                lookahead_window = env_int_or("PAST_LOOKAHEAD_WINDOW", 3);
                // Target work per block: proportional to block capacity
                for (int bi = 0; bi < B; ++bi)
                    block_target_work[bi] = static_cast<double>(merged_ordered[bi].length) *
                                            static_cast<double>(total_work) / static_cast<double>(prefix_target[B]);
                // Price volatility per block (std dev of prices within block)
                for (int bi = 0; bi < B; ++bi)
                {
                    int bstart = merged_ordered[bi].start;
                    int blen = merged_ordered[bi].length;
                    if (blen <= 1) { block_price_vol[bi] = 0.0; continue; }
                    double mean_price = 0.0;
                    for (int t = bstart; t < bstart + blen && t < T; ++t)
                        mean_price += (prefix_proc[t + 1] - prefix_proc[t]);
                    mean_price /= std::min(blen, T - bstart);
                    double var = 0.0;
                    for (int t = bstart; t < bstart + blen && t < T; ++t)
                    {
                        double p = prefix_proc[t + 1] - prefix_proc[t];
                        double d = p - mean_price;
                        var += d * d;
                    }
                    block_price_vol[bi] = std::sqrt(var / std::max(1, std::min(blen, T - bstart)));
                }
            }

            auto lookahead_penalty = [&](const std::array<int, kMaxRepairTypes> &counts, int next_block, int prefix_work) -> double
            {
                if (lookahead_window <= 0 || lookahead_weight <= 0.0)
                    return 0.0;
                // Compute remaining work and counts
                int rem_work = std::max(0, total_work - prefix_work);
                if (rem_work <= 0)
                    return 0.0;
                std::vector<int> rem_counts(K, 0);
                for (int j = 0; j < K; ++j)
                    rem_counts[j] = std::max(0, totals[j] - counts[j]);

                double penalty = 0.0;
                int window = std::min(lookahead_window, B - next_block);
                for (int w = 0; w < window; ++w)
                {
                    int bi = next_block + w;
                    double target = block_target_work[bi];
                    // Estimate work achievable in this block from remaining counts
                    double est_work = 0.0;
                    for (int j = 0; j < K; ++j)
                    {
                        int maxc = std::min(rem_counts[j], merged_ordered[bi].length / std::max(1, lengths[j]));
                        // Distribute remaining counts proportionally across remaining blocks
                        int share = std::min(maxc, (rem_counts[j] + window - w) / (window - w + 1));
                        est_work += share * lengths[j];
                    }
                    // Penalize deviation from target
                    double dev = est_work - target;
                    penalty += std::abs(dev) / std::max(1.0, target);
                    // Add price volatility penalty: if block has high price variance, penalize large allocations
                    double vol_pen = block_price_vol[bi] * est_work / std::max(1.0, target) * 0.01;
                    penalty += vol_pen;
                }
                return penalty;
            };

            std::vector<std::vector<FeasBeamNode>> layers(B + 1);
            layers[0].push_back(FeasBeamNode{});
            layers[0][0].prefix_work = 0;

            double width_sum = 0.0;
            double width_max = 0.0;
            double total_considered = 0.0;
            double total_kept = 0.0;
            double total_pruned_over = 0.0;
            double total_pruned_suffix = 0.0;
            double total_pruned_disc = 0.0;
            double residual_pen_sum = 0.0;
            double residual_pen_max = 0.0;
            int residual_pen_count = 0;

            for (int bi = 0; bi < B; ++bi)
            {
                if (beam_out_of_time())
                {
                    if (diag_out)
                    {
                        diag_out->status = "timeout";
                        diag_out->timed_out = 1;
                    }
                    return kInf;
                }
                std::unordered_map<FeasBeamKey, int, FeasBeamKeyHash> next_index;
                std::unordered_map<FeasBeamKey, std::vector<int>, FeasBeamKeyHash> next_multi_index;
                std::unordered_map<FeasBeamKey, std::vector<int>, FeasBeamKeyHash> next_role_candidates;
                int reserve_est = std::max(256, base_beam_width * std::max(1, discrepancy_budget + 1));
                if (key_multiplicity <= 1)
                    next_index.reserve(static_cast<std::size_t>(reserve_est * 2));
                else
                    next_multi_index.reserve(static_cast<std::size_t>(reserve_est * 2));
                std::vector<FeasBeamNode> next_layer;

                std::size_t n_considered = 0;
                std::size_t n_over_pruned = 0;
                std::size_t n_suffix_pruned = 0;
                std::size_t n_disc_pruned = 0;

                for (int si = 0; si < static_cast<int>(layers[bi].size()); ++si)
                {
                    const auto &state = layers[bi][si];
                    for (int pi = 0; pi < static_cast<int>(patterns[bi].size()); ++pi)
                    {
                        ++n_considered;
                        if ((n_considered & 4095ULL) == 0ULL && beam_out_of_time())
                        {
                            if (diag_out)
                            {
                                diag_out->status = "timeout";
                                diag_out->timed_out = 1;
                            }
                            return kInf;
                        }
                        const auto &pat = patterns[bi][pi];
                        FeasBeamNode nxt;
                        nxt.counts = state.counts;
                        nxt.parent_idx = si;
                        nxt.pat_idx = pi;
                        nxt.discrepancy = state.discrepancy;
                        nxt.prefix_work = state.prefix_work + patterns[bi][pi].work;

                        bool ok = true;
                        for (int j = 0; j < K; ++j)
                        {
                            nxt.counts[j] += pat.counts[j];
                            if (nxt.counts[j] > totals[j])
                            {
                                ok = false;
                                break;
                            }
                        }
                        if (!ok)
                        {
                            ++n_over_pruned;
                            continue;
                        }
                        for (int j = 0; j < K; ++j)
                        {
                            int rem = totals[j] - nxt.counts[j];
                            if (rem < suffix_min[bi + 1][j] || rem > suffix_max[bi + 1][j])
                            {
                                ok = false;
                                break;
                            }
                        }
                        if (!ok)
                        {
                            ++n_suffix_pruned;
                            continue;
                        }

                        int disc_excess = 0;
                        if (bi < discrepancy_depth && pattern_pref_rank[bi][pi] >= discrepancy_topk)
                        {
                            int raw_disc = nxt.discrepancy + 1;
                            if (raw_disc > discrepancy_budget)
                            {
                                disc_excess = raw_disc - discrepancy_budget;
                                ++n_disc_pruned;
                            }
                            nxt.discrepancy = std::min(raw_disc, discrepancy_budget);
                        }

                        double s_center = center_score(nxt.counts, bi + 1);
                        double s_feas = feasibility_pressure(nxt.counts, bi + 1);
                        double s_arith = arithmetic_pressure(nxt.counts, bi + 1, nxt.prefix_work);
                        double s_local = pattern_local_rank[bi][pi];
                        nxt.s_center = s_center;
                        nxt.s_feas = s_feas;
                        nxt.s_local = s_local;
                        nxt.s_arith = s_arith;
                        double s_residual = 0.0;
                        if (score_policy == "residual_aware")
                        {
                            s_residual = residual_penalty(nxt.counts, bi + 1, nxt.prefix_work);
                            residual_pen_sum += s_residual;
                            residual_pen_max = std::max(residual_pen_max, s_residual);
                            ++residual_pen_count;
                        }
                        double s_lookahead = 0.0;
                        if (score_policy == "fine_plus_coarse_lookahead")
                        {
                            s_lookahead = lookahead_penalty(nxt.counts, bi + 1, nxt.prefix_work);
                        }
                        nxt.score = w_center * s_center +
                                    w_feas * s_feas +
                                    w_local * s_local +
                                    w_arith * s_arith +
                                    residual_weight * s_residual +
                                    lookahead_weight * s_lookahead +
                                    0.05 * static_cast<double>(nxt.discrepancy) +
                                    0.10 * static_cast<double>(disc_excess);

                        FeasBeamKey key;
                        key.key = count_key_from_counts(nxt.counts, K);
                        key.disc = static_cast<uint8_t>(std::min(255, nxt.discrepancy));

                        // PLAN22 / PLAN23: adaptive per-key multiplicity / survivor policy
                        bool use_adaptive_multi = (key_multi_policy != "off" && key_multi_max > 1);
                        int effective_multi = 1;
                        if (use_adaptive_multi)
                        {
                            bool is_early = (static_cast<double>(bi) < key_multi_early_frac * static_cast<double>(B));
                            bool is_late = (static_cast<double>(bi) >= (1.0 - late_frac) * static_cast<double>(B));
                            if (key_multi_policy == "uniform")
                                effective_multi = key_multi_max;
                            else if (key_multi_policy == "early")
                                effective_multi = is_early ? key_multi_max : 1;
                            else if (key_multi_policy == "late_ambig")
                                effective_multi = is_late ? key_multi_max : 1;
                            else if (key_multi_policy == "ambig_scoreband")
                                effective_multi = key_multi_max;
                            else if (key_multi_policy == "hybrid")
                                effective_multi = is_early ? key_multi_max : 1;
                            else if (key_multi_policy == "role")
                                effective_multi = key_multi_max;
                        }

                        if (key_multi_policy == "role" && key_multi_max > 1)
                        {
                            int idx = static_cast<int>(next_layer.size());
                            next_layer.push_back(std::move(nxt));
                            auto &vec = next_role_candidates[key];
                            vec.push_back(idx);
                            int candidate_cap = key_multi_max * 6;
                            if (static_cast<int>(vec.size()) > candidate_cap)
                            {
                                int worst_pos = 0;
                                double worst_score = next_layer[vec[0]].score;
                                for (int vi = 1; vi < static_cast<int>(vec.size()); ++vi)
                                {
                                    if (next_layer[vec[vi]].score > worst_score)
                                    {
                                        worst_score = next_layer[vec[vi]].score;
                                        worst_pos = vi;
                                    }
                                }
                                vec.erase(vec.begin() + worst_pos);
                            }
                        }
                        else if (!use_adaptive_multi || effective_multi <= 1)
                        {
                            auto it = next_index.find(key);
                            if (it == next_index.end())
                            {
                                int idx = static_cast<int>(next_layer.size());
                                next_index.emplace(key, idx);
                                next_layer.push_back(std::move(nxt));
                            }
                            else if (nxt.score < next_layer[it->second].score)
                            {
                                next_layer[it->second] = std::move(nxt);
                            }
                        }
                        else
                        {
                            auto it = next_multi_index.find(key);
                            if (it == next_multi_index.end())
                            {
                                int idx = static_cast<int>(next_layer.size());
                                next_multi_index.emplace(key, std::vector<int>{idx});
                                next_layer.push_back(std::move(nxt));
                            }
                            else
                            {
                                auto &vec = it->second;
                                double best_score = next_layer[vec[0]].score;
                                for (int mi = 1; mi < static_cast<int>(vec.size()); ++mi)
                                    best_score = std::min(best_score, next_layer[vec[mi]].score);

                                bool should_insert = false;
                                int replace_pos = -1;
                                if (static_cast<int>(vec.size()) < effective_multi)
                                {
                                    should_insert = true;
                                }
                                else
                                {
                                    int worst_idx = 0;
                                    double worst_score = next_layer[vec[0]].score;
                                    for (int mi = 1; mi < static_cast<int>(vec.size()); ++mi)
                                    {
                                        if (next_layer[vec[mi]].score > worst_score)
                                        {
                                            worst_score = next_layer[vec[mi]].score;
                                            worst_idx = mi;
                                        }
                                    }
                                    if (nxt.score < worst_score)
                                    {
                                        should_insert = true;
                                        replace_pos = worst_idx;
                                    }
                                }

                                if (should_insert)
                                {
                                    bool pass_filters = true;
                                    if (key_multi_policy == "ambig_scoreband" || key_multi_policy == "hybrid" || key_multi_policy == "late_ambig")
                                    {
                                        double score_band = key_multi_score_eps * std::max(1.0, std::abs(best_score));
                                        if (nxt.score > best_score + score_band)
                                            pass_filters = false;
                                        if (pass_filters && !vec.empty())
                                        {
                                            bool diverse = false;
                                            for (int kept_idx : vec)
                                            {
                                                const auto &kept = next_layer[kept_idx];
                                                if (std::abs(nxt.s_local - kept.s_local) > key_multi_diversity_eps ||
                                                    std::abs(nxt.s_arith - kept.s_arith) > key_multi_diversity_eps ||
                                                    std::abs(nxt.s_feas - kept.s_feas) > key_multi_diversity_eps ||
                                                    std::abs(nxt.s_center - kept.s_center) > key_multi_diversity_eps ||
                                                    std::abs(static_cast<double>(nxt.prefix_work - kept.prefix_work)) / std::max(1.0, static_cast<double>(kept.prefix_work)) > key_multi_diversity_eps)
                                                {
                                                    diverse = true;
                                                    break;
                                                }
                                            }
                                            if (!diverse)
                                                pass_filters = false;
                                        }
                                    }
                                    if (pass_filters)
                                    {
                                        if (replace_pos < 0)
                                        {
                                            int idx = static_cast<int>(next_layer.size());
                                            vec.push_back(idx);
                                            next_layer.push_back(std::move(nxt));
                                        }
                                        else
                                        {
                                            next_layer[vec[replace_pos]] = std::move(nxt);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // PLAN23: role-based survivor selection
                if (key_multi_policy == "role" && key_multi_max > 1 && !next_role_candidates.empty())
                {
                    std::vector<FeasBeamNode> selected_layer;
                    selected_layer.reserve(next_layer.size());
                    for (auto &kv : next_role_candidates)
                    {
                        const auto &cand_indices = kv.second;
                        if (cand_indices.empty())
                            continue;
                        // rep_score: lowest total score
                        int rep_score_idx = cand_indices[0];
                        double best_score = next_layer[rep_score_idx].score;
                        for (int ci = 1; ci < static_cast<int>(cand_indices.size()); ++ci)
                        {
                            int idx = cand_indices[ci];
                            if (next_layer[idx].score < best_score)
                            {
                                best_score = next_layer[idx].score;
                                rep_score_idx = idx;
                            }
                        }
                        double score_limit = best_score + role_score_band * std::max(1.0, std::abs(best_score));
                        // rep_local: lowest s_local within score band
                        int rep_local_idx = -1;
                        double best_local = kInf;
                        // rep_arith: lowest s_arith within score band
                        int rep_arith_idx = -1;
                        double best_arith = kInf;
                        // rep_feas: lowest s_feas within score band (if enabled)
                        int rep_feas_idx = -1;
                        double best_feas = kInf;
                        for (int ci = 0; ci < static_cast<int>(cand_indices.size()); ++ci)
                        {
                            int idx = cand_indices[ci];
                            const auto &node = next_layer[idx];
                            if (node.score > score_limit)
                                continue;
                            if (node.s_local < best_local)
                            {
                                best_local = node.s_local;
                                rep_local_idx = idx;
                            }
                            if (node.s_arith < best_arith)
                            {
                                best_arith = node.s_arith;
                                rep_arith_idx = idx;
                            }
                            if (role_keep_feas && node.s_feas < best_feas)
                            {
                                best_feas = node.s_feas;
                                rep_feas_idx = idx;
                            }
                        }
                        // Collect unique representatives
                        std::vector<int> reps;
                        reps.push_back(rep_score_idx);
                        if (rep_local_idx >= 0 && rep_local_idx != rep_score_idx)
                            reps.push_back(rep_local_idx);
                        if (rep_arith_idx >= 0 && rep_arith_idx != rep_score_idx && rep_arith_idx != rep_local_idx)
                            reps.push_back(rep_arith_idx);
                        if (rep_feas_idx >= 0 && rep_feas_idx != rep_score_idx && rep_feas_idx != rep_local_idx && rep_feas_idx != rep_arith_idx)
                            reps.push_back(rep_feas_idx);
                        // If too many, keep best by score, then local, then arith, then feas
                        if (static_cast<int>(reps.size()) > key_multi_max)
                        {
                            std::sort(reps.begin(), reps.end(), [&](int a, int b)
                                      {
                                          const auto &na = next_layer[a];
                                          const auto &nb = next_layer[b];
                                          if (na.score != nb.score)
                                              return na.score < nb.score;
                                          if (na.s_local != nb.s_local)
                                              return na.s_local < nb.s_local;
                                          if (na.s_arith != nb.s_arith)
                                              return na.s_arith < nb.s_arith;
                                          return na.s_feas < nb.s_feas;
                                      });
                            reps.resize(key_multi_max);
                        }
                        for (int idx : reps)
                            selected_layer.push_back(next_layer[idx]);
                    }
                    next_layer = std::move(selected_layer);
                }

                auto better_beam_node = [](const FeasBeamNode &a, const FeasBeamNode &b)
                {
                    if (a.score != b.score)
                        return a.score < b.score;
                    if (a.discrepancy != b.discrepancy)
                        return a.discrepancy < b.discrepancy;
                    return a.parent_idx < b.parent_idx;
                };

                double considered_scale = static_cast<double>(n_considered) /
                                          std::max(1.0, static_cast<double>(base_beam_width * 4));
                double candidate_pressure = std::min(1.0, considered_scale);
                double pattern_diversity = std::min(1.0, static_cast<double>(patterns[bi].size()) / 48.0);
                double early_bonus = (bi < discrepancy_depth) ? 0.2 : 0.0;
                double incumbent_tightness = (known_ub < kInf * 0.5) ? 0.12 : 0.0;
                double cand_coeff = (K <= 4) ? 0.35 : 0.6;
                double div_coeff = (K <= 4) ? 0.25 : 0.4;
                double width_mult = 1.0 + cand_coeff * candidate_pressure + div_coeff * pattern_diversity + early_bonus - incumbent_tightness;
                int layer_width = static_cast<int>(std::round(base_beam_width * width_mult));
                layer_width = std::max(adaptive_width_min, std::min(adaptive_width_max, layer_width));

                // PLAN20B: diversity bucket split
                bool use_buckets = !bucket_split_str.empty();
                int bucket_a_pct = 100;
                int bucket_b_pct = 0;
                if (use_buckets)
                {
                    // Parse simple two-bucket split like "70,30"
                    size_t comma = bucket_split_str.find(',');
                    if (comma != std::string::npos)
                    {
                        try { bucket_a_pct = std::stoi(bucket_split_str.substr(0, comma)); } catch (...) {}
                        try { bucket_b_pct = std::stoi(bucket_split_str.substr(comma + 1)); } catch (...) {}
                    }
                    bucket_a_pct = std::max(0, std::min(100, bucket_a_pct));
                    bucket_b_pct = std::max(0, std::min(100 - bucket_a_pct, bucket_b_pct));
                }

                if (static_cast<int>(next_layer.size()) > layer_width)
                {
                    if (!use_buckets || bucket_b_pct == 0)
                    {
                        auto nth = next_layer.begin() + layer_width;
                        std::nth_element(next_layer.begin(), nth, next_layer.end(), better_beam_node);
                        next_layer.resize(layer_width);
                    }
                    else
                    {
                        int a_width = static_cast<int>(layer_width * bucket_a_pct / 100.0);
                        int b_width = layer_width - a_width;
                        a_width = std::max(1, a_width);
                        b_width = std::max(1, b_width);

                        // Bucket A: best by score
                        auto nth_a = next_layer.begin() + std::min(a_width, static_cast<int>(next_layer.size()));
                        std::nth_element(next_layer.begin(), nth_a, next_layer.end(), better_beam_node);
                        std::vector<FeasBeamNode> bucket_a(next_layer.begin(), nth_a);

                        // Bucket B: from the rest, best by secondary metric
                        std::vector<FeasBeamNode> rest(nth_a, next_layer.end());
                        auto better_by_metric = [&](const FeasBeamNode &a, const FeasBeamNode &b)
                        {
                            if (bucket_metric_str == "feas")
                            {
                                if (a.s_feas != b.s_feas)
                                    return a.s_feas < b.s_feas;
                            }
                            else if (bucket_metric_str == "local")
                            {
                                if (a.s_local != b.s_local)
                                    return a.s_local < b.s_local;
                            }
                            else
                            {
                                if (a.score != b.score)
                                    return a.score < b.score;
                            }
                            if (a.discrepancy != b.discrepancy)
                                return a.discrepancy < b.discrepancy;
                            return a.parent_idx < b.parent_idx;
                        };
                        int take_b = std::min(b_width, static_cast<int>(rest.size()));
                        auto nth_b = rest.begin() + take_b;
                        std::nth_element(rest.begin(), nth_b, rest.end(), better_by_metric);
                        rest.resize(take_b);

                        // Merge
                        next_layer.clear();
                        next_layer.reserve(bucket_a.size() + rest.size());
                        next_layer.insert(next_layer.end(), bucket_a.begin(), bucket_a.end());
                        next_layer.insert(next_layer.end(), rest.begin(), rest.end());
                    }
                }
                std::sort(next_layer.begin(), next_layer.end(), better_beam_node);

                width_sum += layer_width;
                width_max = std::max(width_max, static_cast<double>(layer_width));
                total_considered += static_cast<double>(n_considered);
                total_kept += static_cast<double>(next_layer.size());
                total_pruned_over += static_cast<double>(n_over_pruned);
                total_pruned_suffix += static_cast<double>(n_suffix_pruned);
                total_pruned_disc += static_cast<double>(n_disc_pruned);

                if (trace)
                {
                    std::cerr << "block_repair_trace method=feasible_beam"
                              << " block=" << bi
                              << " layer_in=" << layers[bi].size()
                              << " patterns=" << patterns[bi].size()
                              << " considered=" << n_considered
                              << " over_pruned=" << n_over_pruned
                              << " suffix_pruned=" << n_suffix_pruned
                              << " discrepancy_pruned=" << n_disc_pruned
                              << " width=" << layer_width
                              << " kept=" << next_layer.size()
                              << "\n";
                }

                if (next_layer.empty())
                {
                    if (diag_out)
                        diag_out->status = "infeasible_frontier";
                    return kInf;
                }
                layers[bi + 1] = std::move(next_layer);
            }

            int target_idx = -1;
            for (int si = 0; si < static_cast<int>(layers[B].size()); ++si)
            {
                bool target = true;
                for (int j = 0; j < K; ++j)
                {
                    if (layers[B][si].counts[j] != totals[j])
                    {
                        target = false;
                        break;
                    }
                }
                if (target)
                {
                    target_idx = si;
                    break;
                }
            }
            if (target_idx < 0)
            {
                if (diag_out)
                    diag_out->status = "infeasible_counts";
                return kInf;
            }

            if (diag_out)
            {
                diag_out->base_width = static_cast<double>(base_beam_width);
                diag_out->avg_width = width_sum / std::max(1, B);
                diag_out->max_width = width_max;
                diag_out->states_considered = total_considered;
                diag_out->states_kept = total_kept;
                diag_out->pruned_over = total_pruned_over;
                diag_out->pruned_suffix = total_pruned_suffix;
                diag_out->pruned_discrepancy = total_pruned_disc;
                diag_out->discrepancy_budget = discrepancy_budget;
                diag_out->discrepancy_depth = discrepancy_depth;
                diag_out->key_multi_policy = key_multi_policy;
                diag_out->key_multi_max = key_multi_max;
                diag_out->key_multi_score_eps = key_multi_score_eps;
                diag_out->key_multi_diversity_eps = key_multi_diversity_eps;
                diag_out->score_policy = score_policy;
                diag_out->residual_weight = residual_weight;
                diag_out->late_frac = late_frac;
                if (residual_pen_count > 0)
                {
                    diag_out->residual_mean_penalty = residual_pen_sum / static_cast<double>(residual_pen_count);
                    diag_out->residual_max_penalty = residual_pen_max;
                }
                diag_out->status = "feasible";
            }

            std::vector<int> chosen_pat(B, -1);
            int cur_idx = target_idx;
            for (int bi = B; bi >= 1; --bi)
            {
                const auto &node = layers[bi][cur_idx];
                chosen_pat[bi - 1] = node.pat_idx;
                cur_idx = node.parent_idx;
            }

            if (chosen_counts_out)
            {
                chosen_counts_out->assign(B, {});
                for (int bi = 0; bi < B; ++bi)
                    (*chosen_counts_out)[ordered_to_orig[bi]] = patterns[bi][chosen_pat[bi]].counts;
            }

            std::vector<std::vector<int>> chosen_counts_orig(B);
            for (int bi = 0; bi < B; ++bi)
                chosen_counts_orig[ordered_to_orig[bi]] = patterns[bi][chosen_pat[bi]].counts;

            std::vector<int> seq_desc;
            std::vector<int> seq_asc;
            for (int bi = 0; bi < B; ++bi)
            {
                const auto &counts = chosen_counts_orig[bi];
                for (int j = K - 1; j >= 0; --j)
                    for (int c = 0; c < counts[j]; ++c)
                        seq_desc.push_back(lengths[j]);
                for (int j = 0; j < K; ++j)
                    for (int c = 0; c < counts[j]; ++c)
                        seq_asc.push_back(lengths[j]);
            }

            double best = kInf;
            if (!seq_desc.empty())
                best = std::min(best, solve_fixed_sequence(seq_desc, prefix_proc, T, spaces));
            if (!seq_asc.empty())
                best = std::min(best, solve_fixed_sequence(seq_asc, prefix_proc, T, spaces));
            return best;
        }

        double block_repair_profile_repair_beam_ub(
            const std::vector<RecoveredBlock> &merged,
            const std::vector<int> &lengths,
            const std::vector<int> &totals,
            const std::vector<double> &prefix_proc,
            int T,
            const SPACESResult &spaces,
            double known_ub = kInf,
            ProfileRepairBeamDiag *diag_out = nullptr,
            bool strengthened = false,
            std::vector<std::vector<int>> *chosen_counts_out = nullptr)
        {
            int K = static_cast<int>(lengths.size());
            int B = static_cast<int>(merged.size());
            if (merged.empty() || K <= 2 || K > kMaxRepairTypes)
                return kInf;

            std::vector<std::vector<int>> chosen_counts;
            ProfileRepairBeamDiag beam_diag;
            double beam_cost = block_repair_feasible_beam_ub(
                merged, lengths, totals, prefix_proc, T, spaces, &chosen_counts, &beam_diag, known_ub, strengthened);
            if (diag_out)
                *diag_out = beam_diag;
            double best = std::min(known_ub, beam_cost);
            if (chosen_counts.size() != merged.size())
                return best;
            if (chosen_counts_out)
                *chosen_counts_out = chosen_counts;

            int l3_max_cells = env_int_or("PAST_PROFILE_REPAIR_BEAM_L3_MAX_CELLS", 50'000);
            double l3_time_limit = env_double_or("PAST_PROFILE_REPAIR_BEAM_L3_TIME_LIMIT", 0.05);

            auto patterns = generate_energy_core_patterns(merged, lengths, totals, T);
            for (const auto &vec : patterns)
            {
                if (vec.empty())
                    return best;
            }

            std::vector<SPACESResult> block_spaces;
            std::vector<std::vector<double>> block_prefix_proc;
            build_profile_block_local_views(
                merged,
                prefix_proc,
                T,
                spaces,
                &block_spaces,
                &block_prefix_proc);

            auto eval_block_counts = [&](int bi, const std::vector<int> &counts) -> double
            {
                return evaluate_profile_block_counts(
                    bi,
                    counts,
                    lengths,
                    merged,
                    block_spaces,
                    block_prefix_proc,
                    l3_max_cells,
                    l3_time_limit);
            };

            std::vector<std::vector<double>> eval_cost(B);
            std::vector<int> chosen_pat(B, -1);
            for (int bi = 0; bi < B; ++bi)
            {
                eval_cost[bi].assign(patterns[bi].size(), kInf);
                std::string chosen_key = pattern_counts_key(chosen_counts[bi]);
                double best_match_cost = kInf;
                for (int pi = 0; pi < static_cast<int>(patterns[bi].size()); ++pi)
                {
                    double c = eval_block_counts(bi, patterns[bi][pi].counts);
                    eval_cost[bi][pi] = c;
                    if (c < kInf * 0.5 && pattern_counts_key(patterns[bi][pi].counts) == chosen_key)
                    {
                        if (c < best_match_cost)
                        {
                            best_match_cost = c;
                            chosen_pat[bi] = pi;
                        }
                    }
                }
                if (chosen_pat[bi] < 0)
                    return best;
            }

            auto total_cost = [&](const std::vector<int> &pat_idx) -> double
            {
                double c = 0.0;
                for (int bi = 0; bi < B; ++bi)
                {
                    int pi = pat_idx[bi];
                    if (pi < 0 || pi >= static_cast<int>(eval_cost[bi].size()))
                        return kInf;
                    if (!(eval_cost[bi][pi] < kInf * 0.5))
                        return kInf;
                    c += eval_cost[bi][pi];
                }
                return c;
            };

            std::vector<int> cur_pat = chosen_pat;
            double cur_cost = total_cost(cur_pat);
            if (cur_cost < best)
                best = cur_cost;

            int local_passes = env_int_or("PAST_PROFILE_REPAIR_BEAM_LOCAL_PASSES", 0);
            int local_max_merged = env_int_or("PAST_PROFILE_REPAIR_BEAM_LOCAL_MAX_MERGED", 32);
            if (strengthened)
            {
                local_passes = std::max(local_passes, env_int_or("PAST_PROFILE_REPAIR_BEAM_STRONG_LOCAL_PASSES", 1));
                local_max_merged = std::max(local_max_merged, env_int_or("PAST_PROFILE_REPAIR_BEAM_STRONG_LOCAL_MAX_MERGED", 48));
            }
            if (!(cur_cost < kInf * 0.5) || local_passes <= 0 || B < 2 || B > local_max_merged)
                return best;

            for (int pass = 0; pass < local_passes; ++pass)
            {
                bool improved = false;
                for (int bi = 0; bi + 1 < B; ++bi)
                {
                    int base_p1 = cur_pat[bi];
                    int base_p2 = cur_pat[bi + 1];
                    if (!(eval_cost[bi][base_p1] < kInf * 0.5) ||
                        !(eval_cost[bi + 1][base_p2] < kInf * 0.5))
                        continue;

                    std::array<int, kMaxRepairTypes> pair_counts{};
                    for (int j = 0; j < K; ++j)
                    {
                        pair_counts[j] = patterns[bi][base_p1].counts[j] +
                                         patterns[bi + 1][base_p2].counts[j];
                    }

                    double cur_pair_cost = eval_cost[bi][base_p1] + eval_cost[bi + 1][base_p2];
                    double best_pair_cost = cur_pair_cost;
                    int best_p1 = base_p1;
                    int best_p2 = base_p2;

                    for (int p1 = 0; p1 < static_cast<int>(patterns[bi].size()); ++p1)
                    {
                        if (!(eval_cost[bi][p1] < kInf * 0.5))
                            continue;
                        for (int p2 = 0; p2 < static_cast<int>(patterns[bi + 1].size()); ++p2)
                        {
                            if (!(eval_cost[bi + 1][p2] < kInf * 0.5))
                                continue;
                            bool ok = true;
                            for (int j = 0; j < K; ++j)
                            {
                                if (patterns[bi][p1].counts[j] + patterns[bi + 1][p2].counts[j] != pair_counts[j])
                                {
                                    ok = false;
                                    break;
                                }
                            }
                            if (!ok)
                                continue;

                            double cand_pair = eval_cost[bi][p1] + eval_cost[bi + 1][p2];
                            if (cand_pair + 1e-9 < best_pair_cost)
                            {
                                best_pair_cost = cand_pair;
                                best_p1 = p1;
                                best_p2 = p2;
                            }
                        }
                    }

                    if (best_p1 != base_p1 || best_p2 != base_p2)
                    {
                        cur_pat[bi] = best_p1;
                        cur_pat[bi + 1] = best_p2;
                        cur_cost += best_pair_cost - cur_pair_cost;
                        improved = true;
                    }
                }
                if (!improved)
                    break;
            }

            if (cur_cost < best)
                best = cur_cost;
            return best;
        }

        struct ExactLevel2Result
        {
            double ub = kInf;
            double time_sec = 0.0;
            double nodes = 0.0;
            bool closed = false;
            std::string status = "not_attempted";
        };

        ExactLevel2Result block_repair_exact_level2_ub(
            const std::vector<RecoveredBlock> &merged,
            const std::vector<int> &lengths,
            const std::vector<int> &totals,
            const std::vector<double> &prefix_proc,
            int T,
            const SPACESResult &spaces,
            double initial_ub)
        {
            ExactLevel2Result out;
            int K = static_cast<int>(lengths.size());
            int B = static_cast<int>(merged.size());
            if (merged.empty() || K <= 2 || K > kMaxRepairTypes)
            {
                out.status = "skipped_shape";
                return out;
            }

            auto t0 = std::chrono::steady_clock::now();
            const double time_limit = std::max(0.01, env_double_or("PAST_BLOCK_REPAIR_EXACT_L2_TIME", 30.0));
            int l3_max_cells = env_int_or("PAST_BLOCK_REPAIR_L3_MAX_CELLS", 50'000);
            double l3_time_limit = env_double_or("PAST_BLOCK_REPAIR_L3_TIME_LIMIT", 0.05);

            auto patterns = generate_energy_core_patterns(merged, lengths, totals, T);
            for (const auto &vec : patterns)
            {
                if (vec.empty())
                {
                    out.status = "no_patterns";
                    return out;
                }
            }

            std::vector<SPACESResult> block_spaces;
            std::vector<std::vector<double>> block_prefix_proc;
            build_profile_block_local_views(
                merged,
                prefix_proc,
                T,
                spaces,
                &block_spaces,
                &block_prefix_proc);

            auto eval_block_counts = [&](int bi, const std::vector<int> &counts) -> double
            {
                return evaluate_profile_block_counts(
                    bi,
                    counts,
                    lengths,
                    merged,
                    block_spaces,
                    block_prefix_proc,
                    l3_max_cells,
                    l3_time_limit);
            };

            std::vector<std::vector<double>> block_pattern_cost(B);
            std::vector<std::vector<int>> active_idx(B);
            std::vector<double> block_min_cost(B, kInf);
            for (int bi = 0; bi < B; ++bi)
            {
                block_pattern_cost[bi].assign(patterns[bi].size(), kInf);
                for (int pi = 0; pi < static_cast<int>(patterns[bi].size()); ++pi)
                {
                    double c = eval_block_counts(bi, patterns[bi][pi].counts);
                    block_pattern_cost[bi][pi] = c;
                    if (c < kInf * 0.5)
                    {
                        active_idx[bi].push_back(pi);
                        block_min_cost[bi] = std::min(block_min_cost[bi], c);
                    }
                }
                if (active_idx[bi].empty())
                {
                    out.status = "no_finite_pattern_cost";
                    return out;
                }
                std::sort(active_idx[bi].begin(), active_idx[bi].end(), [&](int a, int b)
                          { return block_pattern_cost[bi][a] < block_pattern_cost[bi][b]; });
            }

            std::vector<double> block_min_active_cost(B, kInf);
            for (int bi = 0; bi < B; ++bi)
            {
                if (!active_idx[bi].empty())
                    block_min_active_cost[bi] = block_pattern_cost[bi][active_idx[bi].front()];
            }

            std::vector<int> block_order = profile_realization_block_order(merged, patterns);

            std::vector<std::vector<int>> suffix_min(B + 1, std::vector<int>(K, 0));
            std::vector<std::vector<int>> suffix_max(B + 1, std::vector<int>(K, 0));
            for (int pos = B - 1; pos >= 0; --pos)
            {
                int bi = block_order[pos];
                for (int j = 0; j < K; ++j)
                {
                    int mn = INT_MAX;
                    int mx = 0;
                    for (int pi : active_idx[bi])
                    {
                        mn = std::min(mn, patterns[bi][pi].counts[j]);
                        mx = std::max(mx, patterns[bi][pi].counts[j]);
                    }
                    if (mn == INT_MAX)
                    {
                        out.status = "no_suffix_bounds";
                        return out;
                    }
                    suffix_min[pos][j] = suffix_min[pos + 1][j] + mn;
                    suffix_max[pos][j] = suffix_max[pos + 1][j] + mx;
                }
            }

            std::vector<double> suffix_min_cost(B + 1, 0.0);
            for (int pos = B - 1; pos >= 0; --pos)
            {
                int bi = block_order[pos];
                if (!(block_min_active_cost[bi] < kInf * 0.5) || !(suffix_min_cost[pos + 1] < kInf * 0.5))
                    suffix_min_cost[pos] = kInf;
                else
                    suffix_min_cost[pos] = block_min_active_cost[bi] + suffix_min_cost[pos + 1];
            }

            double best = (initial_ub < kInf * 0.5) ? initial_ub : kInf;
            std::vector<int> rem = totals;
            bool timed_out = false;
            std::vector<std::unordered_map<CountKey, double, CountKeyHash>> best_seen(B + 1);
            for (auto &mp : best_seen)
                mp.reserve(4096);

            auto out_of_time = [&]() -> bool
            {
                if (timed_out)
                    return true;
                double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (elapsed >= time_limit)
                {
                    timed_out = true;
                    return true;
                }
                return false;
            };

            std::function<void(int, double)> dfs = [&](int bi, double cur_cost)
            {
                if ((static_cast<int64_t>(out.nodes) & 255LL) == 0 && out_of_time())
                    return;
                out.nodes += 1.0;

                CountKey key;
                key.used = static_cast<uint8_t>(K);
                for (int j = 0; j < K; ++j)
                    key.counts[j] = static_cast<uint16_t>(rem[j]);
                auto &seen_here = best_seen[bi];
                auto it_seen = seen_here.find(key);
                if (it_seen != seen_here.end() && cur_cost >= it_seen->second - 1e-9)
                    return;
                if (it_seen == seen_here.end() || cur_cost + 1e-9 < it_seen->second)
                    seen_here[key] = cur_cost;

                if (bi == B)
                {
                    for (int j = 0; j < K; ++j)
                        if (rem[j] != 0)
                            return;
                    if (cur_cost < best)
                        best = cur_cost;
                    return;
                }

                if (best < kInf * 0.5)
                {
                    double lb = cur_cost + suffix_min_cost[bi];
                    if (lb >= best - 1e-9)
                        return;
                }

                int block = block_order[bi];

                for (int pi : active_idx[block])
                {
                    const auto &pat = patterns[block][pi];
                    bool ok = true;
                    for (int j = 0; j < K; ++j)
                    {
                        int nxt = rem[j] - pat.counts[j];
                        if (nxt < 0)
                        {
                            ok = false;
                            break;
                        }
                        if (nxt < suffix_min[bi + 1][j] || nxt > suffix_max[bi + 1][j])
                        {
                            ok = false;
                            break;
                        }
                    }
                    if (!ok)
                        continue;

                    double next_cost = cur_cost + block_pattern_cost[block][pi];
                    if (best < kInf * 0.5)
                    {
                        double lb = next_cost + suffix_min_cost[bi + 1];
                        if (lb >= best - 1e-9)
                            continue;
                    }

                    for (int j = 0; j < K; ++j)
                        rem[j] -= pat.counts[j];
                    dfs(bi + 1, next_cost);
                    for (int j = 0; j < K; ++j)
                        rem[j] += pat.counts[j];
                    if (timed_out)
                        return;
                }
            };

            dfs(0, 0.0);

            out.time_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            out.closed = !timed_out;
            out.ub = best;
            if (timed_out)
                out.status = "timeout";
            else if (best < kInf * 0.5)
                out.status = "closed";
            else
                out.status = "infeasible";
            return out;
        }

        double block_repair_lagrangian_assign_ub(
            const std::vector<RecoveredBlock> &merged,
            const std::vector<int> &lengths,
            const std::vector<int> &totals,
            const std::vector<double> &prefix_proc,
            int T,
            const SPACESResult &spaces,
            double known_ub = kInf)
        {
            int K = static_cast<int>(lengths.size());
            if (merged.empty() || K <= 2 || K > kMaxRepairTypes)
                return kInf;

            bool trace = env_int_or("PAST_BLOCK_REPAIR_TRACE", 0) != 0;
            int max_merged = env_int_or("PAST_BLOCK_REPAIR_LAGR_MAX_MERGED", 56);
            if (static_cast<int>(merged.size()) > max_merged)
            {
                if (trace)
                {
                    std::cerr << "block_repair_trace method=lagrangian_assign_skip"
                              << " merged_blocks=" << merged.size()
                              << " max_merged=" << max_merged
                              << "\n";
                }
                return kInf;
            }
            int max_iters = env_int_or("PAST_BLOCK_REPAIR_LAGR_MAX_ITERS", K >= 6 ? 160 : 96);
            int restarts = env_int_or("PAST_BLOCK_REPAIR_LAGR_RESTARTS", 2);
            int B = static_cast<int>(merged.size());
            int total_work = 0;
            for (int j = 0; j < K; ++j)
                total_work += lengths[j] * totals[j];
            int default_repair_l1 = (K >= 6 ? (B > 18 ? 96 : 48) : 16);
            double default_center_weight = (B > 18 ? 0.08 : 0.02);
            double default_local_weight = (B > 18 ? 0.04 : 0.01);
            int repair_l1 = env_int_or("PAST_BLOCK_REPAIR_LAGR_REPAIR_L1", default_repair_l1);
            int stall_iters = env_int_or("PAST_BLOCK_REPAIR_LAGR_STALL_ITERS", 10);
            double alpha0 = env_double_or("PAST_BLOCK_REPAIR_LAGR_ALPHA0", 2.0);
            double alpha_min = env_double_or("PAST_BLOCK_REPAIR_LAGR_ALPHA_MIN", 0.01);
            double center_weight = env_double_or("PAST_BLOCK_REPAIR_LAGR_CENTER_W", default_center_weight);
            double local_weight = env_double_or("PAST_BLOCK_REPAIR_LAGR_LOCAL_W", default_local_weight);

            auto patterns = generate_energy_core_patterns(merged, lengths, totals, T);
            for (const auto &vec : patterns)
                if (vec.empty())
                    return kInf;

            std::vector<int> order(K);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](int a, int b)
                      { return lengths[a] > lengths[b]; });

            std::vector<int> prefix_target(B + 1, 0);
            for (int bi = 0; bi < B; ++bi)
                prefix_target[bi + 1] = prefix_target[bi] + merged[bi].length;

            std::vector<int> high_by_block(B, 0);
            std::vector<std::vector<double>> local_centers(B, std::vector<double>(K, 0.0));
            for (int bi = 0; bi < B; ++bi)
            {
                int cap = merged[bi].length;
                int start = merged[bi].start;
                int high = (bi + 1 < B)
                               ? std::max(0, merged[bi + 1].start - start)
                               : std::max(0, T - start);
                high_by_block[bi] = std::max(high, cap);
                for (int j = 0; j < K; ++j)
                    local_centers[bi][j] = total_work > 0 ? (static_cast<double>(totals[j]) * cap / total_work) : 0.0;
            }

            std::vector<std::vector<int>> suffix_min(B + 1, std::vector<int>(K, 0));
            std::vector<std::vector<int>> suffix_max(B + 1, std::vector<int>(K, 0));
            auto refresh_suffix_bounds = [&]()
            {
                std::fill(suffix_min[B].begin(), suffix_min[B].end(), 0);
                std::fill(suffix_max[B].begin(), suffix_max[B].end(), 0);
                for (int bi = B - 1; bi >= 0; --bi)
                {
                    for (int j = 0; j < K; ++j)
                    {
                        int mn = INT_MAX;
                        int mx = 0;
                        for (const auto &pat : patterns[bi])
                        {
                            mn = std::min(mn, pat.counts[j]);
                            mx = std::max(mx, pat.counts[j]);
                        }
                        if (mn == INT_MAX)
                            mn = 0;
                        suffix_min[bi][j] = suffix_min[bi + 1][j] + mn;
                        suffix_max[bi][j] = suffix_max[bi + 1][j] + mx;
                    }
                }
            };
            refresh_suffix_bounds();

            int l3_max_cells = env_int_or("PAST_BLOCK_REPAIR_L3_MAX_CELLS", 50'000);
            double l3_time_limit = env_double_or("PAST_BLOCK_REPAIR_L3_TIME_LIMIT", 0.05);

            std::vector<SPACESResult> block_spaces(B);
            std::vector<std::vector<double>> block_prefix_proc(B);
            for (int bi = 0; bi < B; ++bi)
            {
                const int block_start = merged[bi].start;
                const int block_len = merged[bi].length;
                SPACESResult local = spaces;
                local.early = 0;
                local.late = std::max(0, block_len - 1);
                local.h = block_len;
                if (local.banded)
                    local.max_gap = std::min(local.max_gap, block_len);

                local.c_start.assign(block_len + 1, kInf);
                local.c_end.assign(block_len + 1, kInf);
                for (int t = 0; t <= block_len; ++t)
                {
                    local.c_start[t] = spaces.c_start[block_start + t];
                    local.c_end[t] = spaces.c_end[block_start + t];
                }

                if (local.banded)
                {
                    int stride = local.max_gap + 1;
                    local.c_star.assign(static_cast<std::size_t>(block_len + 1) * stride, kInf);
                    for (int t_end = 0; t_end <= block_len; ++t_end)
                    {
                        int max_delta = std::min(local.max_gap, block_len - t_end);
                        for (int delta = 0; delta <= max_delta; ++delta)
                        {
                            local.c_star[static_cast<std::size_t>(t_end) * stride + delta] =
                                spaces.gap_cost(block_start + t_end, block_start + t_end + delta);
                        }
                    }
                }
                else
                {
                    int stride = block_len + 1;
                    local.c_star.assign(static_cast<std::size_t>(stride) * stride, kInf);
                    for (int t_end = 0; t_end <= block_len; ++t_end)
                    {
                        for (int t_start = t_end; t_start <= block_len; ++t_start)
                        {
                            local.c_star[static_cast<std::size_t>(t_end) * stride + t_start] =
                                spaces.gap_cost(block_start + t_end, block_start + t_start);
                        }
                    }
                }

                block_spaces[bi] = std::move(local);
                block_prefix_proc[bi].assign(block_len + 1, 0.0);
                for (int t = 0; t <= block_len; ++t)
                    block_prefix_proc[bi][t] = prefix_proc[block_start + t] - prefix_proc[block_start];
            }

            auto eval_block_counts = [&](int bi, const std::vector<int> &counts) -> double
            {
                int total_jobs = 0;
                int theoretical_states = 1;
                for (int j = 0; j < K; ++j)
                {
                    total_jobs += counts[j];
                    if (counts[j] < 0)
                        return kInf;
                    if (counts[j] == 0)
                        continue;
                    if (theoretical_states > l3_max_cells / std::max(1, counts[j] + 1))
                    {
                        theoretical_states = l3_max_cells + 1;
                        break;
                    }
                    theoretical_states *= (counts[j] + 1);
                }
                if (total_jobs == 0)
                    return 0.0;

                const auto &local_prefix = block_prefix_proc[bi];
                const auto &local_spaces = block_spaces[bi];
                int local_T = merged[bi].length;

                std::vector<int> seq_desc;
                std::vector<int> seq_asc;
                seq_desc.reserve(total_jobs);
                seq_asc.reserve(total_jobs);
                for (int j = K - 1; j >= 0; --j)
                    for (int c = 0; c < counts[j]; ++c)
                        seq_desc.push_back(lengths[j]);
                for (int j = 0; j < K; ++j)
                    for (int c = 0; c < counts[j]; ++c)
                        seq_asc.push_back(lengths[j]);

                double heuristic = kInf;
                if (!seq_desc.empty())
                    heuristic = std::min(heuristic, solve_fixed_sequence(seq_desc, local_prefix, local_T, local_spaces));
                if (!seq_asc.empty())
                    heuristic = std::min(heuristic, solve_fixed_sequence(seq_asc, local_prefix, local_T, local_spaces));

                int64_t total_cells = static_cast<int64_t>(local_T + 2) * theoretical_states;
                if (total_cells > l3_max_cells)
                    return heuristic;

                double exact = solve_exact_multiset_dp(
                    lengths,
                    counts,
                    local_prefix,
                    local_T,
                    local_spaces,
                    heuristic,
                    l3_time_limit);
                if (exact < kInf * 0.5)
                    return exact;
                return heuristic;
            };

            std::vector<std::vector<double>> block_pattern_search_cost(B);
            std::vector<std::vector<double>> block_pattern_eval_cost(B);
            std::vector<std::unordered_set<std::string>> pattern_seen(B);
            std::vector<int> base_pattern_sizes(B, 0);
            for (int bi = 0; bi < B; ++bi)
            {
                block_pattern_search_cost[bi].assign(patterns[bi].size(), kInf);
                block_pattern_eval_cost[bi].assign(patterns[bi].size(), kInf);
                pattern_seen[bi].reserve(patterns[bi].size() * 2 + 1);
                int start = merged[bi].start;
                for (int pi = 0; pi < static_cast<int>(patterns[bi].size()); ++pi)
                {
                    const auto &pat = patterns[bi][pi];
                    pattern_seen[bi].insert(pattern_counts_key(pat.counts));
                    if (pat.work <= 0)
                    {
                        block_pattern_search_cost[bi][pi] = 0.0;
                        block_pattern_eval_cost[bi][pi] = 0.0;
                        continue;
                    }
                    double eval_cost = eval_block_counts(bi, pat.counts);
                    block_pattern_eval_cost[bi][pi] = eval_cost;
                    int end = start + pat.work;
                    if (start < 0 || start > spaces.late || end > T || end > spaces.late + 1)
                        continue;
                    if (spaces.c_start[start] >= kInf || spaces.c_end[end] >= kInf)
                        continue;
                    // Keep the corrected Lagrangian search proxy separate from the
                    // stronger Level-3 evaluation. The dual search chooses count
                    // assignments; final candidate evaluation is delegated to the
                    // per-block exact/heuristic evaluator above.
                    if (eval_cost < kInf * 0.5)
                    {
                        block_pattern_search_cost[bi][pi] =
                            spaces.c_start[start] + (prefix_proc[end] - prefix_proc[start]) + spaces.c_end[end];
                    }
                }
                base_pattern_sizes[bi] = static_cast<int>(patterns[bi].size());
            }

            auto eval_choice = [&](const std::vector<int> &chosen_pat) -> double
            {
                double total = 0.0;
                for (int bi = 0; bi < B; ++bi)
                {
                    int pi = chosen_pat[bi];
                    if (pi < 0 || pi >= static_cast<int>(block_pattern_eval_cost[bi].size()))
                        return kInf;
                    double cost = block_pattern_eval_cost[bi][pi];
                    if (!(cost < kInf * 0.5))
                        return kInf;
                    total += cost;
                }
                return total;
            };

            auto improve_exact_choice = [&](const std::vector<int> &chosen_pat) -> double
            {
                int local_improve_max_merged = env_int_or("PAST_BLOCK_REPAIR_LAGR_LOCAL_IMPROVE_MAX_MERGED", 24);
                int local_improve_passes = env_int_or("PAST_BLOCK_REPAIR_LAGR_LOCAL_IMPROVE_PASSES", 1);
                if (B <= 1 || B > local_improve_max_merged || local_improve_passes <= 0)
                    return eval_choice(chosen_pat);

                std::vector<int> cur = chosen_pat;
                double cur_cost = eval_choice(cur);
                if (!(cur_cost < kInf * 0.5))
                    return cur_cost;

                for (int pass = 0; pass < local_improve_passes; ++pass)
                {
                    bool improved = false;
                    for (int b = 0; b + 1 < B; ++b)
                    {
                        const auto &base1 = patterns[b][cur[b]];
                        const auto &base2 = patterns[b + 1][cur[b + 1]];
                        std::array<int, kMaxRepairTypes> pair_counts{};
                        for (int j = 0; j < K; ++j)
                            pair_counts[j] = base1.counts[j] + base2.counts[j];

                        double best_pair_cost = cur_cost;
                        int best_p1 = -1;
                        int best_p2 = -1;
                        for (int p1 = 0; p1 < static_cast<int>(patterns[b].size()); ++p1)
                        {
                            const auto &alt1 = patterns[b][p1];
                            for (int p2 = 0; p2 < static_cast<int>(patterns[b + 1].size()); ++p2)
                            {
                                if (p1 == cur[b] && p2 == cur[b + 1])
                                    continue;
                                const auto &alt2 = patterns[b + 1][p2];
                                bool ok = true;
                                for (int j = 0; j < K; ++j)
                                {
                                    if (alt1.counts[j] + alt2.counts[j] != pair_counts[j])
                                    {
                                        ok = false;
                                        break;
                                    }
                                }
                                if (!ok)
                                    continue;
                                auto cand = cur;
                                cand[b] = p1;
                                cand[b + 1] = p2;
                                double cand_cost = eval_choice(cand);
                                if (cand_cost + 1e-9 < best_pair_cost)
                                {
                                    best_pair_cost = cand_cost;
                                    best_p1 = p1;
                                    best_p2 = p2;
                                }
                            }
                        }

                        if (best_p1 >= 0)
                        {
                            cur[b] = best_p1;
                            cur[b + 1] = best_p2;
                            cur_cost = best_pair_cost;
                            improved = true;
                        }
                    }
                    if (!improved)
                        break;
                }
                return cur_cost;
            };

            auto exact_from_counts = [&](const std::vector<int> &counts) -> bool
            {
                for (int j = 0; j < K; ++j)
                    if (counts[j] != totals[j])
                        return false;
                return true;
            };

            auto center_score = [&](const std::array<int, kMaxRepairTypes> &counts, int upto_block) -> double
            {
                double score = 0.0;
                double pref_cap = static_cast<double>(prefix_target[upto_block]);
                for (int j = 0; j < K; ++j)
                {
                    double mu = total_work > 0 ? static_cast<double>(totals[j]) * pref_cap / total_work : 0.0;
                    score += std::abs(static_cast<double>(counts[j]) - mu);
                }
                return score;
            };

            auto add_priced_pattern = [&](int bi, const std::vector<double> &lambda) -> bool
            {
                int max_extra = env_int_or("PAST_BLOCK_REPAIR_LAGR_PRICING_MAX_EXTRA", 24);
                double price_center_weight = env_double_or("PAST_BLOCK_REPAIR_LAGR_PRICING_CENTER_W", center_weight);
                double price_local_weight = env_double_or("PAST_BLOCK_REPAIR_LAGR_PRICING_LOCAL_W", local_weight);
                if (static_cast<int>(patterns[bi].size()) >= base_pattern_sizes[bi] + max_extra)
                    return false;

                int high = high_by_block[bi];
                int cap = merged[bi].length;
                const auto &local_center = local_centers[bi];
                std::vector<double> cur(high + 1, -kInf);
                std::vector<double> nxt(high + 1, -kInf);
                std::vector<std::vector<int>> prev_work(K + 1, std::vector<int>(high + 1, -1));
                std::vector<std::vector<int>> choice_count(K + 1, std::vector<int>(high + 1, 0));
                cur[0] = 0.0;

                for (int oi = 0; oi < K; ++oi)
                {
                    int j = order[oi];
                    int L = lengths[j];
                    std::fill(nxt.begin(), nxt.end(), -kInf);
                    for (int work = 0; work <= high; ++work)
                    {
                        if (!(cur[work] > -kInf * 0.5))
                            continue;
                        int maxc = std::min(totals[j], (high - work) / std::max(1, L));
                        for (int c = 0; c <= maxc; ++c)
                        {
                            int nw = work + c * L;
                            double val = cur[work] + lambda[j] * static_cast<double>(c) -
                                         price_center_weight * std::abs(static_cast<double>(c) - local_center[j]);
                            if (val > nxt[nw] + 1e-9)
                            {
                                nxt[nw] = val;
                                prev_work[oi + 1][nw] = work;
                                choice_count[oi + 1][nw] = c;
                            }
                        }
                    }
                    cur.swap(nxt);
                }

                int best_work = -1;
                double best_score = kInf;
                for (int work = 0; work <= high; ++work)
                {
                    if (!(cur[work] > -kInf * 0.5))
                        continue;
                    int start = merged[bi].start;
                    int end = start + work;
                    double base_cost = 0.0;
                    if (work > 0)
                    {
                        if (start < 0 || start > spaces.late || end > T || end > spaces.late + 1)
                            continue;
                        if (spaces.c_start[start] >= kInf || spaces.c_end[end] >= kInf)
                            continue;
                        base_cost = spaces.c_start[start] + (prefix_proc[end] - prefix_proc[start]) + spaces.c_end[end];
                    }
                    double score = base_cost + price_local_weight * std::abs(work - cap) - cur[work];
                    if (score < best_score)
                    {
                        best_score = score;
                        best_work = work;
                    }
                }
                if (best_work < 0)
                    return false;

                RepairPattern pat;
                pat.counts.assign(K, 0);
                pat.work = best_work;
                pat.local_dev = std::abs(best_work - cap);
                pat.center_dev = 0.0;
                int work = best_work;
                for (int oi = K; oi >= 1; --oi)
                {
                    int c = choice_count[oi][work];
                    int j = order[oi - 1];
                    pat.counts[j] = c;
                    pat.center_dev += std::abs(static_cast<double>(c) - local_center[j]);
                    work = prev_work[oi][work];
                    if (work < 0 && oi > 1)
                        return false;
                }

                std::string key = pattern_counts_key(pat.counts);
                if (!pattern_seen[bi].insert(key).second)
                    return false;

                double eval_cost = eval_block_counts(bi, pat.counts);
                if (!(eval_cost < kInf * 0.5))
                {
                    pattern_seen[bi].erase(key);
                    return false;
                }

                double search_cost = 0.0;
                if (pat.work > 0)
                {
                    int start = merged[bi].start;
                    int end = start + pat.work;
                    if (start < 0 || start > spaces.late || end > T || end > spaces.late + 1)
                    {
                        pattern_seen[bi].erase(key);
                        return false;
                    }
                    if (spaces.c_start[start] >= kInf || spaces.c_end[end] >= kInf)
                    {
                        pattern_seen[bi].erase(key);
                        return false;
                    }
                    search_cost = spaces.c_start[start] + (prefix_proc[end] - prefix_proc[start]) + spaces.c_end[end];
                }

                patterns[bi].push_back(std::move(pat));
                block_pattern_eval_cost[bi].push_back(eval_cost);
                block_pattern_search_cost[bi].push_back(search_cost);
                return true;
            };

            auto try_local_repair = [&](const std::vector<int> &chosen_pat,
                                        const std::vector<int> &sum_counts) -> double
            {
                if (exact_from_counts(sum_counts))
                    return improve_exact_choice(chosen_pat);

                for (int bi = 0; bi < B; ++bi)
                {
                    const auto &base = patterns[bi][chosen_pat[bi]];
                    for (int pi = 0; pi < static_cast<int>(patterns[bi].size()); ++pi)
                    {
                        if (pi == chosen_pat[bi])
                            continue;
                        const auto &alt = patterns[bi][pi];
                        bool ok = true;
                        for (int j = 0; j < K; ++j)
                        {
                            int v = sum_counts[j] - base.counts[j] + alt.counts[j];
                            if (v != totals[j])
                            {
                                ok = false;
                                break;
                            }
                        }
                        if (ok)
                        {
                            auto fixed = chosen_pat;
                            fixed[bi] = pi;
                            return eval_choice(fixed);
                        }
                    }
                }

                for (int b1 = 0; b1 < B; ++b1)
                {
                    const auto &base1 = patterns[b1][chosen_pat[b1]];
                    for (int b2 = b1 + 1; b2 < B; ++b2)
                    {
                        const auto &base2 = patterns[b2][chosen_pat[b2]];
                        for (int p1 = 0; p1 < static_cast<int>(patterns[b1].size()); ++p1)
                        {
                            const auto &alt1 = patterns[b1][p1];
                            for (int p2 = 0; p2 < static_cast<int>(patterns[b2].size()); ++p2)
                            {
                                const auto &alt2 = patterns[b2][p2];
                                bool ok = true;
                                for (int j = 0; j < K; ++j)
                                {
                                    int v = sum_counts[j] - base1.counts[j] - base2.counts[j] + alt1.counts[j] + alt2.counts[j];
                                    if (v != totals[j])
                                    {
                                        ok = false;
                                        break;
                                    }
                                }
                                if (ok)
                                {
                                    auto fixed = chosen_pat;
                                    fixed[b1] = p1;
                                    fixed[b2] = p2;
                                    return improve_exact_choice(fixed);
                                }
                            }
                        }
                    }
                }

                if (repair_l1 > 0)
                {
                    std::vector<int> cur_choice = chosen_pat;
                    std::vector<int> cur_counts = sum_counts;
                    int max_greedy = std::max(4, repair_l1 * 3);
                    for (int iter = 0; iter < max_greedy; ++iter)
                    {
                        int best_b = -1;
                        int best_p = -1;
                        int best_l1 = INT_MAX;
                        double best_obj = kInf;
                        int cur_l1 = 0;
                        for (int j = 0; j < K; ++j)
                            cur_l1 += std::abs(totals[j] - cur_counts[j]);
                        if (cur_l1 == 0)
                            return eval_choice(cur_choice);

                        for (int bi = 0; bi < B; ++bi)
                        {
                            const auto &base = patterns[bi][cur_choice[bi]];
                            for (int pi = 0; pi < static_cast<int>(patterns[bi].size()); ++pi)
                            {
                                if (pi == cur_choice[bi])
                                    continue;
                                const auto &alt = patterns[bi][pi];
                                int next_l1 = 0;
                                for (int j = 0; j < K; ++j)
                                {
                                    int v = cur_counts[j] - base.counts[j] + alt.counts[j];
                                    next_l1 += std::abs(totals[j] - v);
                                }
                                if (next_l1 >= cur_l1)
                                    continue;
                                double obj = block_pattern_search_cost[bi][pi] +
                                             center_weight * alt.center_dev +
                                             local_weight * alt.local_dev;
                                if (next_l1 < best_l1 || (next_l1 == best_l1 && obj < best_obj))
                                {
                                    best_l1 = next_l1;
                                    best_obj = obj;
                                    best_b = bi;
                                    best_p = pi;
                                }
                            }
                        }

                        if (best_b < 0)
                            break;
                        const auto &base = patterns[best_b][cur_choice[best_b]];
                        const auto &alt = patterns[best_b][best_p];
                        for (int j = 0; j < K; ++j)
                            cur_counts[j] = cur_counts[j] - base.counts[j] + alt.counts[j];
                        cur_choice[best_b] = best_p;
                    }
                    if (exact_from_counts(cur_counts))
                        return improve_exact_choice(cur_choice);
                }
                return kInf;
            };

            auto seeded_feasible_beam = [&](const std::vector<int> &seed_choice, int seed_l1) -> double
            {
                int max_merged = env_int_or("PAST_BLOCK_REPAIR_LAGR_SEEDED_BEAM_MAX_MERGED", 24);
                int max_l1 = env_int_or("PAST_BLOCK_REPAIR_LAGR_SEEDED_BEAM_MAX_L1", 64);
                int beam_width = env_int_or("PAST_BLOCK_REPAIR_LAGR_SEEDED_BEAM_WIDTH", K >= 6 ? 30000 : 12000);
                double seed_weight = env_double_or("PAST_BLOCK_REPAIR_LAGR_SEEDED_BEAM_SEED_W", 1.0);
                double center_bias = env_double_or("PAST_BLOCK_REPAIR_LAGR_SEEDED_BEAM_CENTER_W", 0.15);
                if (B <= 0 || B > max_merged || seed_l1 > max_l1 || static_cast<int>(seed_choice.size()) != B)
                    return kInf;

                std::vector<std::array<int, kMaxRepairTypes>> seed_prefix_counts(B + 1);
                for (auto &arr : seed_prefix_counts)
                    arr.fill(0);
                for (int bi = 0; bi < B; ++bi)
                {
                    seed_prefix_counts[bi + 1] = seed_prefix_counts[bi];
                    int pi = seed_choice[bi];
                    if (pi < 0 || pi >= static_cast<int>(patterns[bi].size()))
                        return kInf;
                    const auto &pat = patterns[bi][pi];
                    for (int j = 0; j < K; ++j)
                        seed_prefix_counts[bi + 1][j] += pat.counts[j];
                }

                struct SeedBeamNode
                {
                    std::array<int, kMaxRepairTypes> counts{};
                    int parent_idx = -1;
                    int pat_idx = -1;
                    double score = 0.0;
                };

                auto seed_score = [&](const std::array<int, kMaxRepairTypes> &counts, int upto_block) -> double
                {
                    double score = center_bias * center_score(counts, upto_block);
                    for (int j = 0; j < K; ++j)
                        score += seed_weight * std::abs(counts[j] - seed_prefix_counts[upto_block][j]);
                    return score;
                };

                std::vector<std::vector<SeedBeamNode>> layers(B + 1);
                layers[0].push_back(SeedBeamNode{});

                for (int bi = 0; bi < B; ++bi)
                {
                    std::unordered_map<CountKey, int, CountKeyHash> next_index;
                    next_index.reserve(static_cast<std::size_t>(std::max(256, beam_width * 2)));
                    std::vector<SeedBeamNode> next_layer;

                    for (int si = 0; si < static_cast<int>(layers[bi].size()); ++si)
                    {
                        const auto &state = layers[bi][si];
                        for (int pi = 0; pi < static_cast<int>(patterns[bi].size()); ++pi)
                        {
                            const auto &pat = patterns[bi][pi];
                            SeedBeamNode nxt;
                            nxt.counts = state.counts;
                            nxt.parent_idx = si;
                            nxt.pat_idx = pi;

                            bool ok = true;
                            for (int j = 0; j < K; ++j)
                            {
                                nxt.counts[j] += pat.counts[j];
                                if (nxt.counts[j] > totals[j])
                                {
                                    ok = false;
                                    break;
                                }
                            }
                            if (!ok)
                                continue;
                            for (int j = 0; j < K; ++j)
                            {
                                int rem = totals[j] - nxt.counts[j];
                                if (rem < suffix_min[bi + 1][j] || rem > suffix_max[bi + 1][j])
                                {
                                    ok = false;
                                    break;
                                }
                            }
                            if (!ok)
                                continue;

                            nxt.score = seed_score(nxt.counts, bi + 1);
                            CountKey key = count_key_from_counts(nxt.counts, K);
                            auto it = next_index.find(key);
                            if (it == next_index.end())
                            {
                                int idx = static_cast<int>(next_layer.size());
                                next_index.emplace(key, idx);
                                next_layer.push_back(std::move(nxt));
                            }
                            else if (nxt.score < next_layer[it->second].score)
                            {
                                next_layer[it->second] = std::move(nxt);
                            }
                        }
                    }

                    std::sort(next_layer.begin(), next_layer.end(), [](const SeedBeamNode &a, const SeedBeamNode &b)
                              {
                                  if (a.score != b.score)
                                      return a.score < b.score;
                                  return a.parent_idx < b.parent_idx;
                              });
                    if (static_cast<int>(next_layer.size()) > beam_width)
                        next_layer.resize(beam_width);
                    if (next_layer.empty())
                        return kInf;
                    layers[bi + 1] = std::move(next_layer);
                }

                int target_idx = -1;
                for (int si = 0; si < static_cast<int>(layers[B].size()); ++si)
                {
                    bool target = true;
                    for (int j = 0; j < K; ++j)
                    {
                        if (layers[B][si].counts[j] != totals[j])
                        {
                            target = false;
                            break;
                        }
                    }
                    if (target)
                    {
                        target_idx = si;
                        break;
                    }
                }
                if (target_idx < 0)
                    return kInf;

                std::vector<int> chosen_pat(B, -1);
                int cur_idx = target_idx;
                for (int bi = B; bi >= 1; --bi)
                {
                    const auto &node = layers[bi][cur_idx];
                    chosen_pat[bi - 1] = node.pat_idx;
                    cur_idx = node.parent_idx;
                }
                return improve_exact_choice(chosen_pat);
            };

            double best = known_ub;
            int best_l1 = INT_MAX;
            std::vector<int> best_choice;
            std::vector<int> best_counts;

            for (int restart = 0; restart < std::max(1, restarts); ++restart)
            {
                std::vector<double> lambda(K, 0.0);
                double alpha = alpha0;
                double best_dual = -kInf;
                int stall = 0;
                if (restart > 0)
                {
                    for (int j = 0; j < K; ++j)
                        lambda[j] = (restart & 1 ? 0.25 : -0.25) * (j + 1);
                }

                for (int it = 0; it < std::max(1, max_iters); ++it)
                {
                    bool pricing_enabled = env_int_or("PAST_BLOCK_REPAIR_LAGR_PRICING", 0) != 0;
                    if (pricing_enabled)
                    {
                        bool added_any = false;
                        for (int bi = 0; bi < B; ++bi)
                            added_any = add_priced_pattern(bi, lambda) || added_any;
                        if (added_any)
                            refresh_suffix_bounds();
                    }

                    std::vector<int> chosen_pat(B, 0);
                    std::vector<int> sum_counts(K, 0);
                    double lagrangian_sum = 0.0;

                    for (int bi = 0; bi < B; ++bi)
                    {
                        double best_score = kInf;
                        int best_pi = 0;
                        for (int pi = 0; pi < static_cast<int>(patterns[bi].size()); ++pi)
                        {
                            const auto &pat = patterns[bi][pi];
                            double base_cost = block_pattern_search_cost[bi][pi];
                            if (!(base_cost < kInf * 0.5))
                                continue;
                            double dual = base_cost +
                                          center_weight * pat.center_dev +
                                          local_weight * pat.local_dev;
                            for (int j = 0; j < K; ++j)
                                dual -= lambda[j] * static_cast<double>(pat.counts[j]);
                            if (dual < best_score)
                            {
                                best_score = dual;
                                best_pi = pi;
                            }
                        }
                        chosen_pat[bi] = best_pi;
                        const auto &pick = patterns[bi][best_pi];
                        lagrangian_sum += best_score;
                        for (int j = 0; j < K; ++j)
                            sum_counts[j] += pick.counts[j];
                    }

                    int l1 = 0;
                    double g_norm_sq = 0.0;
                    std::vector<int> diff(K, 0);
                    for (int j = 0; j < K; ++j)
                    {
                        diff[j] = totals[j] - sum_counts[j];
                        l1 += std::abs(diff[j]);
                        g_norm_sq += static_cast<double>(diff[j]) * static_cast<double>(diff[j]);
                    }
                    double dual_value = lagrangian_sum;
                    for (int j = 0; j < K; ++j)
                        dual_value += lambda[j] * static_cast<double>(totals[j]);

                    if (trace)
                    {
                        std::cerr << "block_repair_trace method=lagrangian_assign"
                                  << " restart=" << restart
                                  << " iter=" << it
                                  << " l1=" << l1
                                  << " best_l1=" << best_l1
                                  << " dual=" << dual_value
                                  << " alpha=" << alpha
                                  << " incumbent=" << best
                                  << "\n";
                    }

                    if (l1 < best_l1)
                    {
                        best_l1 = l1;
                        best_choice = chosen_pat;
                        best_counts = sum_counts;
                    }

                    if (l1 == 0)
                    {
                        best = std::min(best, improve_exact_choice(chosen_pat));
                        break;
                    }

                    if (l1 <= repair_l1)
                    {
                        best = std::min(best, try_local_repair(chosen_pat, sum_counts));
                        if (best < kInf * 0.5)
                            break;
                    }

                    if (dual_value > best_dual + 1e-9)
                    {
                        best_dual = dual_value;
                        stall = 0;
                    }
                    else
                    {
                        ++stall;
                        if (stall >= stall_iters)
                        {
                            alpha *= 0.5;
                            stall = 0;
                            if (alpha < alpha_min)
                                alpha = alpha_min;
                        }
                    }

                    double ub_est = best;
                    if (!(ub_est < kInf * 0.5))
                        ub_est = dual_value * 1.05;
                    double gap_est = std::max(1e-6, ub_est - dual_value);
                    double step = alpha * gap_est / std::max(1.0, g_norm_sq);
                    for (int j = 0; j < K; ++j)
                    {
                        lambda[j] += step * static_cast<double>(diff[j]);
                    }
                }

                if (best >= kInf * 0.5 && !best_choice.empty() && best_l1 <= repair_l1 && static_cast<int>(best_counts.size()) == K)
                    best = std::min(best, try_local_repair(best_choice, best_counts));
                bool use_seeded_beam =
                    env_int_or("PAST_BLOCK_REPAIR_LAGR_USE_SEEDED_BEAM", 0) != 0;
                if (use_seeded_beam && best >= kInf * 0.5 && !best_choice.empty())
                    best = std::min(best, seeded_feasible_beam(best_choice, best_l1));

                if (best < kInf * 0.5)
                    break;
            }

            if (trace)
            {
                std::cerr << "block_repair_trace method=lagrangian_assign_done"
                          << " best_l1=" << best_l1
                          << " incumbent=" << best
                          << "\n";
            }
            return best;
        }

        double block_repair_rg_beam_ub(
            const std::vector<RecoveredBlock> &merged,
            const std::vector<int> &lengths,
            const std::vector<int> &totals,
            const std::vector<double> &prefix_proc,
            int T,
            const SPACESResult &spaces,
            double known_ub)
        {
            int K = static_cast<int>(lengths.size());
            if (merged.empty() || K <= 2 || K > kMaxRepairTypes)
                return kInf;
            bool trace = env_int_or("PAST_BLOCK_REPAIR_TRACE", 0) != 0;

            int total_work = 0;
            for (int i = 0; i < K; ++i)
                total_work += lengths[i] * totals[i];
            auto t0_completion = std::chrono::steady_clock::now();
            auto completion = compute_relaxed_completion_table(
                lengths, total_work, prefix_proc, T, spaces, RelaxationMode::Semigroup);
            double t_completion =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_completion).count();
            auto t0_patterns = std::chrono::steady_clock::now();
            auto patterns = generate_energy_core_patterns(merged, lengths, totals, T);
            double t_patterns =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_patterns).count();
            if (trace)
            {
                std::size_t total_patterns = 0;
                std::size_t max_patterns = 0;
                for (const auto &vec : patterns)
                {
                    total_patterns += vec.size();
                    max_patterns = std::max(max_patterns, vec.size());
                }
                std::cerr << "block_repair_trace method=rg_beam_prepare"
                          << " merged_blocks=" << merged.size()
                          << " t_completion=" << t_completion
                          << " t_patterns=" << t_patterns
                          << " total_patterns=" << total_patterns
                          << " max_patterns=" << max_patterns
                          << "\n";
            }
            for (const auto &vec : patterns)
                if (vec.empty())
                    return kInf;

            int B = static_cast<int>(merged.size());
            std::vector<std::vector<int>> suffix_min(B + 1, std::vector<int>(K, 0));
            std::vector<std::vector<int>> suffix_max(B + 1, std::vector<int>(K, 0));
            for (int bi = B - 1; bi >= 0; --bi)
            {
                for (int j = 0; j < K; ++j)
                {
                    int mn = INT_MAX;
                    int mx = 0;
                    for (const auto &pat : patterns[bi])
                    {
                        mn = std::min(mn, pat.counts[j]);
                        mx = std::max(mx, pat.counts[j]);
                    }
                    if (mn == INT_MAX)
                        mn = 0;
                    suffix_min[bi][j] = suffix_min[bi + 1][j] + mn;
                    suffix_max[bi][j] = suffix_max[bi + 1][j] + mx;
                }
            }

            int beam_width = env_int_or("PAST_BLOCK_REPAIR_RG_BEAM_WIDTH", K <= 4 ? 20000 : 12000);
            double best = known_ub;
            std::vector<EnergyCoreNode> layer;
            EnergyCoreNode seed;
            seed.counts.fill(0);
            seed.prev_end = -1;
            seed.prefix_work = 0;
            seed.g = 0.0;
            seed.f = lookup_completion_lb(completion.off_rdp, completion.RW, completion.rw_scale, T, 0, total_work);
            layer.push_back(seed);

            for (int bi = 0; bi < B; ++bi)
            {
                std::size_t n_considered = 0;
                std::size_t n_over_pruned = 0;
                std::size_t n_suffix_pruned = 0;
                std::size_t n_transition_pruned = 0;
                std::size_t n_h_pruned = 0;
                std::unordered_map<EnergyCoreKey, EnergyCoreNode, EnergyCoreKeyHash> next_best;
                next_best.reserve(static_cast<std::size_t>(std::max(64, beam_width * 2)));
                for (const auto &state : layer)
                {
                    if (state.g >= best)
                        continue;
                    for (const auto &pat : patterns[bi])
                    {
                        ++n_considered;
                        EnergyCoreNode nxt = state;
                        bool suffix_ok = true;
                        for (int j = 0; j < K; ++j)
                        {
                            nxt.counts[j] += pat.counts[j];
                            if (nxt.counts[j] > totals[j])
                            {
                                suffix_ok = false;
                                break;
                            }
                        }
                        if (!suffix_ok)
                        {
                            ++n_over_pruned;
                            continue;
                        }
                        for (int j = 0; j < K; ++j)
                        {
                            int rem = totals[j] - nxt.counts[j];
                            if (rem < suffix_min[bi + 1][j] || rem > suffix_max[bi + 1][j])
                            {
                                suffix_ok = false;
                                break;
                            }
                        }
                        if (!suffix_ok)
                        {
                            ++n_suffix_pruned;
                            continue;
                        }

                        double incr = 0.0;
                        int new_prev_end = state.prev_end;
                        if (pat.work > 0)
                        {
                            int start = merged[bi].start;
                            if (state.prev_end < 0)
                            {
                                if (spaces.c_start[start] >= kInf)
                                {
                                    ++n_transition_pruned;
                                    continue;
                                }
                                incr += spaces.c_start[start];
                            }
                            else
                            {
                                double gap = spaces.gap_cost(state.prev_end, start);
                                if (gap >= kInf)
                                {
                                    ++n_transition_pruned;
                                    continue;
                                }
                                incr += gap;
                            }
                            int block_end = start + pat.work;
                            if (block_end > T || block_end > spaces.late + 1)
                            {
                                ++n_transition_pruned;
                                continue;
                            }
                            incr += prefix_proc[block_end] - prefix_proc[start];
                            new_prev_end = block_end;
                        }

                        nxt.prefix_work += pat.work;
                        nxt.g += incr;
                        nxt.prev_end = new_prev_end;
                        int time_anchor = (new_prev_end >= 0) ? new_prev_end : merged[bi].start;
                        int rw_remaining = total_work - nxt.prefix_work;
                        const auto &h_table = (new_prev_end >= 0) ? completion.rdp : completion.off_rdp;
                        double h = lookup_completion_lb(h_table, completion.RW, completion.rw_scale,
                                                        T, time_anchor, rw_remaining);
                        nxt.f = nxt.g + h;
                        if (nxt.f >= best)
                        {
                            ++n_h_pruned;
                            continue;
                        }

                        EnergyCoreKey key = energy_core_key(nxt, K);
                        auto it = next_best.find(key);
                        if (it == next_best.end() || nxt.g < it->second.g - kEps || (std::abs(nxt.g - it->second.g) <= kEps && nxt.f < it->second.f))
                            next_best[key] = std::move(nxt);
                    }
                }

                std::vector<EnergyCoreNode> next_layer;
                next_layer.reserve(next_best.size());
                for (auto &kv : next_best)
                    next_layer.push_back(std::move(kv.second));
                std::sort(next_layer.begin(), next_layer.end(), [](const EnergyCoreNode &a, const EnergyCoreNode &b)
                          {
                              if (std::abs(a.f - b.f) > kEps)
                                  return a.f < b.f;
                              return a.g < b.g;
                          });
                if (static_cast<int>(next_layer.size()) > beam_width)
                    next_layer.resize(beam_width);
                if (trace)
                {
                    std::cerr << "block_repair_trace method=rg_beam"
                              << " block=" << bi
                              << " layer_in=" << layer.size()
                              << " patterns=" << patterns[bi].size()
                              << " considered=" << n_considered
                              << " over_pruned=" << n_over_pruned
                              << " suffix_pruned=" << n_suffix_pruned
                              << " transition_pruned=" << n_transition_pruned
                              << " h_pruned=" << n_h_pruned
                              << " kept=" << next_layer.size()
                              << " best=" << best
                              << "\n";
                }
                if (next_layer.empty())
                    return best;
                layer = std::move(next_layer);
            }

            for (const auto &state : layer)
            {
                bool target = true;
                for (int j = 0; j < K; ++j)
                {
                    if (state.counts[j] != totals[j])
                    {
                        target = false;
                        break;
                    }
                }
                if (!target)
                    continue;
                double total = state.g;
                if (state.prev_end >= 0)
                {
                    if (spaces.c_end[state.prev_end] >= kInf)
                        continue;
                    total += spaces.c_end[state.prev_end];
                }
                if (total < best)
                    best = total;
            }
            return best;
        }
    }

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
        bool pack_trace = env_int_or("PAST_RELAXED_PACK_TRACE", 0) != 0;

        if (blocks.empty())
            return result;

        const char *pack_mode = std::getenv("PAST_RELAXED_BINPACK_SOLVER");
        result.pack_solver = pack_mode ? std::string(pack_mode) : "profile_repair_beam";

        const std::string incumbent_source = to_lower_ascii(std::string(
            std::getenv("PAST_EXACT_INCUMBENT_SOURCE") ? std::getenv("PAST_EXACT_INCUMBENT_SOURCE") : "auto"));
        result.profile_incumbent_source = incumbent_source;

        auto note_pack_candidate = [&](const std::string &method, double cand)
        {
            if (cand < result.bin_pack_ub)
            {
                result.bin_pack_ub = cand;
                result.pack_method = method;
                result.pack_outcome = "feasible";
            }
        };

        auto t0_merge_blocks = std::chrono::steady_clock::now();
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
        result.merged_blocks.clear();
        result.merged_blocks.reserve(merged.size());
        for (const auto &rb : merged)
            result.merged_blocks.push_back({rb.start, rb.length});
        result.t_pack_merge_blocks =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_merge_blocks).count();

        // ── PLAN29: block-view adjacent coarsening ─────────────────────
        {
            std::string view_policy = env_str_or("PAST_BLOCK_VIEW_POLICY", "baseline");
            result.block_view_policy = view_policy;
            result.block_view_original_blocks = static_cast<int>(merged.size());

            if (view_policy != "baseline" && merged.size() >= 2)
            {
                auto t0_view = std::chrono::steady_clock::now();
                int target_b = env_int_or("PAST_BLOCK_VIEW_TARGET_B", 12);

                auto merge_two = [&](std::vector<RecoveredBlock> &v, int a, int b) {
                    int new_start = std::min(v[a].start, v[b].start);
                    int new_len = std::max(v[a].start + v[a].length, v[b].start + v[b].length) - new_start;
                    v[a].start = new_start;
                    v[a].length = new_len;
                };

                auto remove_boundary = [&](std::vector<RecoveredBlock> &v, int idx) {
                    // Merge v[idx] into v[idx+1], keep v[idx] as merged result
                    merge_two(v, idx, idx + 1);
                    v.erase(v.begin() + idx + 1);
                };

                if (view_policy == "coarsen2")
                {
                    std::vector<RecoveredBlock> out;
                    for (int i = 0; i < static_cast<int>(merged.size()); i += 2)
                    {
                        if (i + 1 < static_cast<int>(merged.size()))
                        {
                            RecoveredBlock rb = merged[i];
                            int new_end = std::max(rb.start + rb.length,
                                                   merged[i + 1].start + merged[i + 1].length);
                            rb.length = new_end - rb.start;
                            out.push_back(rb);
                        }
                        else
                        {
                            out.push_back(merged[i]);
                        }
                    }
                    result.block_view_removed_boundaries = static_cast<int>(merged.size()) - static_cast<int>(out.size());
                    merged = std::move(out);
                }
                else if (view_policy == "coarsen3")
                {
                    std::vector<RecoveredBlock> out;
                    for (int i = 0; i < static_cast<int>(merged.size()); i += 3)
                    {
                        RecoveredBlock rb = merged[i];
                        int max_end = rb.start + rb.length;
                        for (int j = i + 1; j < static_cast<int>(merged.size()) && j < i + 3; ++j)
                            max_end = std::max(max_end, merged[j].start + merged[j].length);
                        rb.length = max_end - rb.start;
                        out.push_back(rb);
                    }
                    result.block_view_removed_boundaries = static_cast<int>(merged.size()) - static_cast<int>(out.size());
                    merged = std::move(out);
                }
                else if (view_policy == "target_b")
                {
                    result.block_view_target_b = target_b;
                    std::vector<RecoveredBlock> v = merged;
                    int removed = 0;
                    while (static_cast<int>(v.size()) > target_b)
                    {
                        // Find index of narrowest boundary to remove
                        int best_i = 0;
                        int best_cost = std::numeric_limits<int>::max();
                        for (int i = 0; i + 1 < static_cast<int>(v.size()); ++i)
                        {
                            int width1 = v[i].length;
                            int width2 = v[i + 1].length;
                            int cost = width1 + width2; // simpler boundary = smaller total
                            if (cost < best_cost)
                            {
                                best_cost = cost;
                                best_i = i;
                            }
                        }
                        remove_boundary(v, best_i);
                        ++removed;
                    }
                    result.block_view_removed_boundaries = removed;
                    merged = std::move(v);
                }
                else if (view_policy == "price_preserve")
                {
                    result.block_view_target_b = target_b;
                    result.block_view_price_preserve_used = 1;
                    // Compute price jump at each boundary using prefix_proc
                    std::vector<double> boundary_price_jump(merged.size() - 1, 0.0);
                    for (std::size_t bi = 0; bi + 1 < merged.size(); ++bi)
                    {
                        int bound_t = merged[bi].start + merged[bi].length;
                        // Local price difference across boundary
                        double price_before = 0.0;
                        double price_after = 0.0;
                        if (bound_t > 0 && bound_t < T)
                        {
                            price_before = prefix_proc[bound_t] - prefix_proc[std::max(0, bound_t - 5)];
                            price_after = prefix_proc[std::min(T, bound_t + 5)] - prefix_proc[bound_t];
                            price_before /= std::min(5, bound_t);
                            price_after /= std::min(5, T - bound_t);
                        }
                        boundary_price_jump[bi] = std::abs(price_before - price_after);
                    }
                    // Find max price jump for normalization
                    double max_jump = 0.0;
                    for (double j : boundary_price_jump)
                        max_jump = std::max(max_jump, j);
                    if (max_jump <= 0.0)
                        max_jump = 1.0;

                    std::vector<RecoveredBlock> v = merged;
                    int removed = 0;
                    while (static_cast<int>(v.size()) > target_b)
                    {
                        int best_i = 0;
                        double best_cost = 1e100;
                        for (int i = 0; i + 1 < static_cast<int>(v.size()); ++i)
                        {
                            int width1 = v[i].length;
                            int width2 = v[i + 1].length;
                            double width_cost = static_cast<double>(width1 + width2);
                            // Penalize boundaries with large price jumps
                            double price_factor = 1.0 + 2.0 * (boundary_price_jump[static_cast<std::size_t>(i)] / max_jump);
                            double cost = width_cost * price_factor;
                            if (cost < best_cost)
                            {
                                best_cost = cost;
                                best_i = i;
                            }
                        }
                        remove_boundary(v, best_i);
                        ++removed;
                    }
                    result.block_view_removed_boundaries = removed;
                    merged = std::move(v);
                }
                else if (view_policy == "arith_adaptive")
                {
                    result.block_view_arith_adaptive_used = 1;
                    int max_len = 1;
                    for (int L : lengths)
                        max_len = std::max(max_len, L);
                    // Coarsen blocks shorter than 2 * max_len
                    int short_threshold = 2 * max_len;
                    std::vector<RecoveredBlock> v = merged;
                    int removed = 0;
                    // Merge each short block with its shorter neighbor
                    bool changed = true;
                    int max_passes = 3;
                    for (int pass = 0; pass < max_passes && changed; ++pass)
                    {
                        changed = false;
                        for (int i = 0; i + 1 < static_cast<int>(v.size()); )
                        {
                            bool left_short = v[i].length < short_threshold;
                            bool right_short = v[i + 1].length < short_threshold;
                            if (left_short || right_short)
                            {
                                remove_boundary(v, i);
                                ++removed;
                                changed = true;
                                // Stay at same i (now pointing to merged block)
                            }
                            else
                            {
                                ++i;
                            }
                        }
                    }
                    result.block_view_removed_boundaries = removed;
                    merged = std::move(v);
                }

                result.block_view_time_sec =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_view).count();
            }

            result.block_view_final_blocks = static_cast<int>(merged.size());
            result.block_view_eval_count = 1;

            // Update downstream structures after coarsening
            result.merged_block_count = static_cast<int>(merged.size());
            result.merged_blocks.clear();
            result.merged_blocks.reserve(merged.size());
            for (const auto &rb : merged)
                result.merged_blocks.push_back({rb.start, rb.length});
        }

        result.profile_realization_hardest_first =
            profile_realization_hardest_first_enabled() ? 1 : 0;
        result.profile_realization_exact_suffix_prune =
            env_int_or("PAST_PROFILE_REALIZATION_EXACT_SUFFIX_PRUNE", 1) != 0 ? 1 : 0;

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

        const int K = static_cast<int>(lengths.size());
        const int nBlk = static_cast<int>(nB);

        std::vector<int> sorted_lengths = lengths;
        std::sort(sorted_lengths.begin(), sorted_lengths.end());
        sorted_lengths.erase(std::unique(sorted_lengths.begin(), sorted_lengths.end()), sorted_lengths.end());
        int has_one = (std::find(sorted_lengths.begin(), sorted_lengths.end(), 1) != sorted_lengths.end()) ? 1 : 0;
        int contiguous = 1;
        for (int i = 1; i < static_cast<int>(sorted_lengths.size()); ++i)
        {
            if (sorted_lengths[i] != sorted_lengths[i - 1] + 1)
            {
                contiguous = 0;
                break;
            }
        }
        int multiplicity = sorted_lengths.empty() ? 0 : sorted_lengths.front();
        int density_cap = std::max(16, env_int_or("PAST_PROFILE_REALIZATION_SELECTOR_DENSITY_CAP", 100));
        double semigroup_density = semigroup_density_prefix(sorted_lengths, density_cap);

        result.profile_selector_has_one = has_one;
        result.profile_selector_contiguous = contiguous;
        result.profile_selector_multiplicity = multiplicity;
        result.profile_selector_semigroup_density = semigroup_density;

        auto sat_mul = [](double a, double b, double lim) -> double
        {
            if (a <= 0.0 || b <= 0.0)
                return 0.0;
            if (a >= lim || b >= lim)
                return lim;
            if (a > lim / b)
                return lim;
            return a * b;
        };

        double selector_state_space = 1.0;
        const double selector_saturate = 1.0e18;
        for (int i = 0; i < K; ++i)
            selector_state_space = sat_mul(selector_state_space, static_cast<double>(totals[i] + 1), selector_saturate);
        result.block_dp_state_space = selector_state_space;

        double selector_total_comp_est = 0.0;
        double selector_max_comp_est = 0.0;
        for (int b = 0; b < nBlk; ++b)
        {
            double comp_est = 1.0;
            int cap = orig_cap[static_cast<std::size_t>(b)];
            for (int i = 0; i < K; ++i)
            {
                int maxc = std::min(totals[i], cap / std::max(1, lengths[i]));
                comp_est = sat_mul(comp_est, static_cast<double>(maxc + 1), selector_saturate);
            }
            selector_total_comp_est = std::min(selector_saturate, selector_total_comp_est + comp_est);
            selector_max_comp_est = std::max(selector_max_comp_est, comp_est);
        }
        result.block_dp_total_comp_estimate = selector_total_comp_est;
        result.block_dp_max_comp_estimate = selector_max_comp_est;

        double selector_avg_branch_est = 0.0;
        double selector_max_branch_est = 0.0;
        for (int b = 0; b < nBlk; ++b)
        {
            double branch_est = 0.0;
            int cap = orig_cap[static_cast<std::size_t>(b)];
            for (int i = 0; i < K; ++i)
            {
                int maxc = std::min(totals[i], cap / std::max(1, lengths[i]));
                branch_est += static_cast<double>(maxc + 1);
            }
            selector_avg_branch_est += branch_est;
            selector_max_branch_est = std::max(selector_max_branch_est, branch_est);
        }
        if (nBlk > 0)
            selector_avg_branch_est /= static_cast<double>(nBlk);

        const bool k2_profile_mode = (K == 2);
        const bool k4plus_profile_mode = (K >= 4);
        const bool beam_profile_supported = (K >= 3 && K <= kMaxRepairTypes);

        bool run_mainline_profile_repair =
            (result.pack_solver == "default" || result.pack_solver == "profile_repair_beam");
        bool selector_in_scope = run_mainline_profile_repair && (k2_profile_mode || k4plus_profile_mode);
        bool run_profile_beam_mode = selector_in_scope;
        bool run_profile_exact_mode = selector_in_scope;
        bool selector_exact_primary = false;

        const char *selector_policy_raw = std::getenv("PAST_PROFILE_REALIZATION_SELECTOR_POLICY");
        std::string selector_policy = selector_policy_raw ? to_lower_ascii(std::string(selector_policy_raw)) : std::string();
        if (selector_policy.empty())
            selector_policy = "auto_v1";

        result.profile_selector_policy = selector_policy;
        result.profile_selector_decision = "legacy";
        result.profile_selector_reason = "selector_disabled";
        result.profile_exact_primary_fallback_to_beam = 0;
        result.profile_exact_primary_status_before_fallback = "not_applicable";
        result.profile_step3_incumbent_mode = "not_attempted";

        if (selector_in_scope)
        {
            if (selector_policy == "0" || selector_policy == "off" || selector_policy == "disabled")
            {
                run_profile_beam_mode = true;
                run_profile_exact_mode = true;
                result.profile_selector_decision = "legacy_both";
                result.profile_selector_reason = "policy_off";
            }
            else if (selector_policy == "force_exact")
            {
                run_profile_beam_mode = false;
                run_profile_exact_mode = true;
                selector_exact_primary = true;
                result.profile_selector_decision = "exact";
                result.profile_selector_reason = "forced_exact";
            }
            else if (selector_policy == "force_beam")
            {
                run_profile_beam_mode = true;
                run_profile_exact_mode = false;
                result.profile_selector_decision = "beam";
                result.profile_selector_reason = "forced_beam";
            }
            else
            {
                if (k2_profile_mode)
                {
                    // Mode A (K=2): exact profile realization by default.
                    // Keep explicit safety gates so policy remains structural.
                    double k2_state_space_max = std::max(1.0, env_double_or("PAST_PROFILE_REALIZATION_SELECTOR_K2_MAX_STATE_SPACE", 1.0e8));
                    double k2_total_comp_est_max = std::max(1.0, env_double_or("PAST_PROFILE_REALIZATION_SELECTOR_K2_MAX_TOTAL_COMP_EST", 2.0e8));
                    double k2_max_comp_est_max = std::max(1.0, env_double_or("PAST_PROFILE_REALIZATION_SELECTOR_K2_MAX_BLOCK_COMP_EST", 1.5e8));

                    bool exact_ok = true;
                    std::string reason = "k2_exact_default";
                    if (selector_state_space > k2_state_space_max)
                    {
                        exact_ok = false;
                        reason = "k2_state_space";
                    }
                    else if (selector_total_comp_est > k2_total_comp_est_max)
                    {
                        exact_ok = false;
                        reason = "k2_total_comp_est";
                    }
                    else if (selector_max_comp_est > k2_max_comp_est_max)
                    {
                        exact_ok = false;
                        reason = "k2_max_block_comp_est";
                    }

                    run_profile_exact_mode = exact_ok;
                    run_profile_beam_mode = false;
                    selector_exact_primary = exact_ok;
                    result.profile_selector_decision = exact_ok ? "exact" : "skip";
                    result.profile_selector_reason = reason;
                    result.profile_selector_hard_alarm = 0;
                }
                else
                {
                    // Mode B/C (K>=4): structural exact-vs-beam selector.
                    int merged_max = env_int_or("PAST_PROFILE_REALIZATION_SELECTOR_MAX_MERGED", 4);
                    double state_space_max = std::max(1.0, env_double_or("PAST_PROFILE_REALIZATION_SELECTOR_MAX_STATE_SPACE", 1.0e8));
                    double total_comp_est_max = std::max(1.0, env_double_or("PAST_PROFILE_REALIZATION_SELECTOR_MAX_TOTAL_COMP_EST", 1.0e8));
                    double max_comp_est_max = std::max(1.0, env_double_or("PAST_PROFILE_REALIZATION_SELECTOR_MAX_BLOCK_COMP_EST", 8.0e7));
                    double avg_branch_est_max = std::max(1.0, env_double_or("PAST_PROFILE_REALIZATION_SELECTOR_MAX_AVG_BRANCH_EST", 400.0));
                    double max_branch_est_max = std::max(1.0, env_double_or("PAST_PROFILE_REALIZATION_SELECTOR_MAX_BLOCK_BRANCH_EST", 600.0));
                    int hard_alarm_merged_min = env_int_or("PAST_PROFILE_REALIZATION_SELECTOR_HARD_ALARM_MIN_MERGED", 10);
                    double hard_alarm_density_max = env_double_or("PAST_PROFILE_REALIZATION_SELECTOR_HARD_ALARM_DENSITY_MAX", 0.975);
                    int hard_alarm =
                        (has_one == 0) &&
                        (contiguous == 0) &&
                        (nBlk >= hard_alarm_merged_min) &&
                        (semigroup_density <= hard_alarm_density_max)
                            ? 1
                            : 0;
                    result.profile_selector_hard_alarm = hard_alarm;

                    bool exact_ok = true;
                    std::string reason = "ok";
                    if (nBlk > merged_max)
                    {
                        exact_ok = false;
                        reason = "merged_blocks";
                    }
                    else if (selector_state_space > state_space_max)
                    {
                        exact_ok = false;
                        reason = "state_space";
                    }
                    else if (selector_total_comp_est > total_comp_est_max)
                    {
                        exact_ok = false;
                        reason = "total_comp_est";
                    }
                    else if (selector_max_comp_est > max_comp_est_max)
                    {
                        exact_ok = false;
                        reason = "max_block_comp_est";
                    }
                    else if (selector_avg_branch_est > avg_branch_est_max)
                    {
                        exact_ok = false;
                        reason = "avg_branch_est";
                    }
                    else if (selector_max_branch_est > max_branch_est_max)
                    {
                        exact_ok = false;
                        reason = "max_block_branch_est";
                    }
                    else if (hard_alarm)
                    {
                        exact_ok = false;
                        reason = "hard_alarm";
                    }

                    run_profile_exact_mode = exact_ok;
                    run_profile_beam_mode = !exact_ok;
                    selector_exact_primary = exact_ok;
                    result.profile_selector_decision = exact_ok ? "exact" : "beam";
                    result.profile_selector_reason = reason;
                }
            }
        }
        else
        {
            run_profile_beam_mode = false;
            run_profile_exact_mode = (result.pack_solver == "block_dp_exact");
            if (run_profile_exact_mode)
            {
                selector_exact_primary = true;
                result.profile_selector_decision = "exact";
                result.profile_selector_reason = "explicit_block_dp_exact";
            }
            else
            {
                result.profile_selector_decision = "not_applicable";
                result.profile_selector_reason = "non_mainline_solver";
            }
        }

        bool inc_src_i0 = (incumbent_source == "i0" || incumbent_source == "quick");
        bool inc_src_i1 = (incumbent_source == "i1" || incumbent_source == "exact_step3");
        bool inc_src_i2 = (incumbent_source == "i2" || incumbent_source == "beam");
        bool inc_src_i3 = (incumbent_source == "i3" || incumbent_source == "beam_plus");
        bool inc_src_i4 = (incumbent_source == "i4" || incumbent_source == "best_step3");
        bool explicit_inc_source = !(incumbent_source.empty() || incumbent_source == "auto");
        if (selector_in_scope && explicit_inc_source)
        {
            if (inc_src_i0)
            {
                run_profile_exact_mode = false;
                run_profile_beam_mode = false;
                result.profile_selector_reason = "incumbent_i0";
            }
            else
            {
                if (inc_src_i1 || inc_src_i4)
                    run_profile_exact_mode = true;
                if (inc_src_i2 || inc_src_i3 || inc_src_i4)
                {
                    if (beam_profile_supported)
                        run_profile_beam_mode = true;
                    else if (run_profile_exact_mode)
                        result.profile_selector_reason = "beam_unsupported";
                }
            }
        }

        if (!beam_profile_supported)
            run_profile_beam_mode = false;

        bool pool_diag_enabled = env_int_or("PAST_BLOCK_REPAIR_POOL_DIAG", 0) != 0;
        if (pool_diag_enabled && lengths.size() >= 5)
        {
            auto lagr_patterns = generate_energy_core_patterns(merged, lengths, totals, T);
            bool have_pool = true;
            for (const auto &vec : lagr_patterns)
            {
                if (vec.empty())
                {
                    have_pool = false;
                    break;
                }
            }

            std::vector<std::vector<int>> beam_counts;
            if (have_pool)
            {
                (void)block_repair_feasible_beam_ub(
                    merged, lengths, totals, prefix_proc, T, spaces, &beam_counts);
            }

            int beam_in_pool = 0;
            int beam_not_in_pool = 0;
            if (have_pool && beam_counts.size() == merged.size())
            {
                for (std::size_t bi = 0; bi < merged.size(); ++bi)
                {
                    std::unordered_set<std::string> pool;
                    pool.reserve(lagr_patterns[bi].size() * 2 + 1);
                    for (const auto &pat : lagr_patterns[bi])
                        pool.insert(pattern_counts_key(pat.counts));
                    if (pool.count(pattern_counts_key(beam_counts[bi])) > 0)
                        ++beam_in_pool;
                    else
                        ++beam_not_in_pool;
                }
            }

            std::cerr << "POOL_DIAG: total_blocks=" << merged.size()
                      << ", beam_in_pool=" << beam_in_pool
                      << ", beam_not_in_pool=" << beam_not_in_pool
                      << "\n";
        }

        bool exact_pack_decided = false;
        double beam_ub_for_exact_l2 = kInf;
        bool dense_unit_fastpath =
            (env_int_or("PAST_DENSE_UNIT_STEP2_FASTPATH", 0) != 0) &&
            (has_one == 1) &&
            (contiguous == 1) &&
            (K >= env_int_or("PAST_DENSE_UNIT_FASTPATH_K_MIN", 8));
        bool count_based_ffd = (env_int_or("PAST_COUNT_BASED_FFD", 0) != 0);
        result.dense_unit_fastpath_active = dense_unit_fastpath ? 1 : 0;
        result.count_based_ffd_active = count_based_ffd ? 1 : 0;

        // Legacy env vars removed: PAST_RELAXED_BINPACK_NATIVE_FIRST,
        // PAST_RELAXED_BINPACK_ALLOW_SMALL_NC, PAST_RELAXED_BINPACK_DISABLE_DFS_EXACT.
        // Step-3 profile realization now has two modes over the same recovered
        // blocks: exact fixed-block DP and truncated profile-repair beam.


        {
            auto t0_pack = std::chrono::steady_clock::now();
            if (pack_trace)
            {
                std::cerr << "pack_trace stage=heuristic_start"
                          << " jobs=" << all_jobs.size()
                          << " merged_blocks=" << nB
                          << "\n";
            }

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

            auto try_pack_count_based = [&](int mode) -> double
            {
                std::vector<int> cap = orig_cap;
                std::vector<std::vector<int>> bj(nB);
                std::vector<int> idx_desc(K);
                std::iota(idx_desc.begin(), idx_desc.end(), 0);
                std::sort(idx_desc.begin(), idx_desc.end(), [&](int a, int b)
                          {
                              if (lengths[a] != lengths[b])
                                  return lengths[a] > lengths[b];
                              return a < b;
                          });
                for (int ti : idx_desc)
                {
                    int jl = lengths[ti];
                    int need = totals[ti];
                    for (int c = 0; c < need; ++c)
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
                }
                std::vector<int> seq;
                for (std::size_t b = 0; b < nB; ++b)
                    for (int j : bj[b])
                        seq.push_back(j);
                return solve_fixed_sequence(seq, prefix_proc, T, spaces);
            };

            auto step2_start = std::chrono::steady_clock::now();
            result.step2_reached = 1;
            bool first_candidate_recorded = false;
            auto note_step2_candidate = [&](const std::string &method, double cand)
            {
                if (cand < kInf * 0.5)
                {
                    result.step2_produced_ub = 1;
                    if (!first_candidate_recorded)
                    {
                        result.t_pack_to_first_candidate =
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - step2_start).count();
                        first_candidate_recorded = true;
                    }
                }
                note_pack_candidate(method, cand);
            };

            if (dense_unit_fastpath)
            {
                auto t0_ffd_only = std::chrono::steady_clock::now();
                if (count_based_ffd)
                    note_step2_candidate("ffd_count", try_pack_count_based(0));
                else
                {
                    std::vector<int> jobs = all_jobs;
                    std::sort(jobs.begin(), jobs.end(), std::greater<int>());
                    note_step2_candidate("ffd", try_pack(jobs, 0));
                }
                result.t_pack_ffd_only =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_ffd_only).count();
            }
            else
            {
                {
                    std::vector<int> jobs = all_jobs;
                    std::sort(jobs.begin(), jobs.end(), std::greater<int>());
                    note_step2_candidate("ffd", try_pack(jobs, 0));
                }
                {
                    std::vector<int> jobs = all_jobs;
                    std::sort(jobs.begin(), jobs.end(), std::greater<int>());
                    note_step2_candidate("bfd", try_pack(jobs, 1));
                }
                {
                    std::vector<int> jobs = all_jobs;
                    std::sort(jobs.begin(), jobs.end());
                    note_step2_candidate("ffi", try_pack(jobs, 0));
                }
                {
                    std::vector<int> jobs = all_jobs;
                    std::sort(jobs.begin(), jobs.end());
                    note_step2_candidate("bfi", try_pack(jobs, 1));
                }
                {
                    std::mt19937_64 rng(12345);
                    std::vector<int> jobs = all_jobs;
                    for (int trial = 0; trial < 20; ++trial)
                    {
                        std::shuffle(jobs.begin(), jobs.end(), rng);
                        note_step2_candidate(trial & 1 ? "random_bf" : "random_ff",
                                             try_pack(jobs, trial & 1));
                    }
                }
            }

            if (!first_candidate_recorded)
                result.t_pack_to_first_candidate =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - step2_start).count();


            result.t_pack_heuristic =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_pack).count();
            if (!dense_unit_fastpath)
                result.t_pack_ffd_only = result.t_pack_heuristic;
            if (pack_trace)
            {
                std::cerr << "pack_trace stage=heuristic_done"
                          << " t_heuristic=" << result.t_pack_heuristic
                          << " incumbent=" << result.bin_pack_ub
                          << " method=" << result.pack_method
                          << "\n";
            }
        }

        result.profile_step2_ub = result.bin_pack_ub;
        if (dense_unit_fastpath && result.profile_step2_ub < kInf * 0.5)
        {
            result.pack_method = count_based_ffd ? "ffd_count" : "ffd";
            result.pack_outcome = "feasible";
            result.profile_beam_status = "skipped_fastpath";
            result.profile_beam_timed_out = 0;
            result.profile_beam_candidate_ub = kInf;
            result.profile_beam_plus_candidate_ub = kInf;
            result.profile_exact_candidate_ub = kInf;
            result.profile_beam_improved_over_step2 = 0;
            result.profile_exact_improved_over_step2 = 0;
            result.block_dp_status = "skipped_fastpath";
            result.profile_step3_incumbent_mode = "none";
            result.profile_incumbent_ub_for_exact = result.profile_step2_ub;
            return result;
        }

        auto run_profile_beam_attempt = [&](const std::string &attempt_reason)
        {
            auto t0_repair = std::chrono::steady_clock::now();
            ProfileRepairBeamDiag beam_diag;
            result.profile_beam_status = "running";
            if (pack_trace)
            {
                std::cerr << "pack_trace stage=profile_repair_beam_start"
                          << " reason=" << attempt_reason
                          << " incumbent=" << result.bin_pack_ub
                          << "\n";
            }
            std::vector<std::vector<int>> beam_chosen_counts;
            double cand = block_repair_profile_repair_beam_ub(
                merged, lengths, totals, prefix_proc, T, spaces, result.bin_pack_ub, &beam_diag, false, &beam_chosen_counts);
            note_pack_candidate("profile_repair_beam", cand);
            result.profile_beam_chosen_counts = std::move(beam_chosen_counts);
            result.profile_beam_block_order.resize(merged.size());
            std::iota(result.profile_beam_block_order.begin(), result.profile_beam_block_order.end(), 0);
            result.profile_beam_base_width = beam_diag.base_width;
            result.profile_beam_avg_width = beam_diag.avg_width;
            result.profile_beam_max_width = beam_diag.max_width;
            result.profile_beam_states_considered = beam_diag.states_considered;
            result.profile_beam_states_kept = beam_diag.states_kept;
            result.profile_beam_pruned_over = beam_diag.pruned_over;
            result.profile_beam_pruned_suffix = beam_diag.pruned_suffix;
            result.profile_beam_pruned_discrepancy = beam_diag.pruned_discrepancy;
            result.profile_beam_discrepancy_budget = beam_diag.discrepancy_budget;
            result.profile_beam_discrepancy_depth = beam_diag.discrepancy_depth;
            result.profile_beam_status = beam_diag.status;
            result.profile_beam_timed_out = beam_diag.timed_out;
            result.profile_beam_key_multi_policy = beam_diag.key_multi_policy;
            result.profile_beam_key_multi_max = beam_diag.key_multi_max;
            result.profile_beam_key_multi_score_eps = beam_diag.key_multi_score_eps;
            result.profile_beam_key_multi_diversity_eps = beam_diag.key_multi_diversity_eps;
            // PLAN27: scoring policy and residual diagnostics
            result.profile_beam_score_policy = beam_diag.score_policy;
            result.profile_beam_residual_weight = beam_diag.residual_weight;
            result.profile_beam_residual_mean_penalty = beam_diag.residual_mean_penalty;
            result.profile_beam_residual_max_penalty = beam_diag.residual_max_penalty;
            result.profile_beam_late_frac = beam_diag.late_frac;
            // PLAN28: block-realizability diagnostics (Phase A: diagnostics only)
            {
                int br_diag_enabled = env_int_or("PAST_BLOCK_REALIZ_DIAG", 0);
                result.block_realiz_diag_active = br_diag_enabled;
                if (br_diag_enabled && !result.profile_beam_chosen_counts.empty() && !merged.empty())
                {
                    auto t0_br = std::chrono::steady_clock::now();
                    int B = static_cast<int>(merged.size());
                    int max_patterns = env_int_or("PAST_BLOCK_REALIZ_DIAG_MAX_PATTERNS", 200000);
                    result.block_realiz_blocks_total = B;

                    // Build block local views (needed for evaluate_profile_block_counts)
                    std::vector<SPACESResult> br_block_spaces;
                    std::vector<std::vector<double>> br_block_prefix_proc;
                    build_profile_block_local_views(
                        merged, prefix_proc, T, spaces, &br_block_spaces, &br_block_prefix_proc);

                    int l3_max_cells = env_int_or("PAST_PROFILE_REPAIR_BEAM_L3_MAX_CELLS", 50000);
                    double l3_time_limit = env_double_or("PAST_PROFILE_REPAIR_BEAM_L3_TIME_LIMIT", 0.05);

                    int bad_blocks = 0;
                    int first_bad = -1;
                    int min_finite = std::numeric_limits<int>::max();
                    double sum_finite = 0.0;
                    int blocks_with_data = 0;
                    int base_path_ok = 1;
                    std::string base_reject_reason = "none";
                    int diag_skipped = 0;
                    std::string diag_skip_reason;
                    std::ostringstream per_block_oss;
                    per_block_oss << std::fixed << std::setprecision(6);

                    // Generate patterns for all blocks (capped)
                    auto patterns = generate_energy_core_patterns(merged, lengths, totals, T);

                    for (int bi = 0; bi < B; ++bi)
                    {
                        // Evaluate beam chosen counts for this block
                        double chosen_cost = evaluate_profile_block_counts(
                            bi, result.profile_beam_chosen_counts[bi], lengths, merged,
                            br_block_spaces, br_block_prefix_proc, l3_max_cells, l3_time_limit);

                        bool chosen_feasible = (chosen_cost < kInf * 0.5);
                        if (!chosen_feasible)
                        {
                            bad_blocks++;
                            if (first_bad < 0)
                                first_bad = bi;
                            if (base_path_ok)
                            {
                                base_path_ok = 0;
                                base_reject_reason = "block_" + std::to_string(bi) + "_chosen_infeasible";
                            }
                        }

                        // Count finite patterns for this block
                        int finite_count = 0;
                        double best_cost = kInf;
                        std::string reject_detail = chosen_feasible ? "ok" : "chosen_infeasible";
                        int n_pats = std::min(static_cast<int>(patterns[bi].size()), max_patterns);
                        for (int pi = 0; pi < n_pats; ++pi)
                        {
                            double c = evaluate_profile_block_counts(
                                bi, patterns[bi][pi].counts, lengths, merged,
                                br_block_spaces, br_block_prefix_proc, l3_max_cells, l3_time_limit);
                            if (c < kInf * 0.5)
                            {
                                finite_count++;
                                if (c < best_cost)
                                    best_cost = c;
                            }
                        }
                        if (static_cast<int>(patterns[bi].size()) > max_patterns)
                        {
                            diag_skipped = 1;
                            diag_skip_reason = "pattern_cap_" + std::to_string(max_patterns);
                        }
                        if (finite_count == 0)
                        {
                            if (reject_detail == "ok")
                                reject_detail = "no_finite_patterns";
                            // Track as bad if no patterns survive (even if chosen was ok)
                            if (chosen_feasible)
                            {
                                bad_blocks++;
                                if (first_bad < 0)
                                    first_bad = bi;
                            }
                        }

                        blocks_with_data++;
                        min_finite = std::min(min_finite, finite_count);
                        sum_finite += static_cast<double>(finite_count);

                        // Per-block payload
                        if (bi > 0) per_block_oss << "|";
                        per_block_oss << bi << ";"
                                      << merged[bi].start << ";"
                                      << merged[bi].length << ";"
                                      << (chosen_feasible ? 1 : 0) << ";"
                                      << finite_count << ";"
                                      << (best_cost < kInf * 0.5 ? best_cost : -1.0) << ";"
                                      << reject_detail;
                    }

                    result.block_realiz_bad_blocks = bad_blocks;
                    result.block_realiz_first_bad_block = first_bad;
                    result.block_realiz_bad_rate = B > 0 ? static_cast<double>(bad_blocks) / static_cast<double>(B) : 0.0;
                    result.block_realiz_min_finite_patterns = blocks_with_data > 0 ? static_cast<double>(min_finite) : 0.0;
                    result.block_realiz_mean_finite_patterns = blocks_with_data > 0 ? sum_finite / static_cast<double>(blocks_with_data) : 0.0;
                    result.block_realiz_base_path_survives = base_path_ok;
                    result.block_realiz_base_reject_reason = base_reject_reason;
                    result.block_realiz_diag_skipped = diag_skipped;
                    result.block_realiz_diag_skip_reason = diag_skip_reason;
                    result.block_realiz_per_block_payload = per_block_oss.str();
                    result.block_realiz_diag_time_sec =
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_br).count();
                }
            }
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_repair).count();
            result.t_pack_block_dp += dt;
            result.t_pack_profile_beam += dt;
            if (cand < result.profile_beam_candidate_ub)
                result.profile_beam_candidate_ub = cand;
            result.profile_beam_improved_over_step2 =
                (result.profile_beam_candidate_ub < kInf * 0.5 && result.profile_beam_candidate_ub + 1e-9 < result.profile_step2_ub) ? 1 : 0;
            if (pack_trace)
            {
                std::cerr << "pack_trace stage=profile_repair_beam_done"
                          << " reason=" << attempt_reason
                          << " t_profile_repair_beam=" << dt
                          << " cand=" << cand
                          << " base_width=" << result.profile_beam_base_width
                          << " avg_width=" << result.profile_beam_avg_width
                          << " max_width=" << result.profile_beam_max_width
                          << " considered=" << result.profile_beam_states_considered
                          << " kept=" << result.profile_beam_states_kept
                          << " pruned_disc=" << result.profile_beam_pruned_discrepancy
                          << " incumbent=" << result.bin_pack_ub
                          << " method=" << result.pack_method
                          << "\n";
            }
        };

        if (run_profile_beam_mode)
        {
            run_profile_beam_attempt("selector_primary");
        }
        else
        {
            result.profile_beam_status = "skipped_selector";
            result.profile_beam_timed_out = 0;
            result.profile_beam_candidate_ub = kInf;
            result.profile_beam_improved_over_step2 = 0;
        }

        // PLAN19 additive redesign: bounded exact closure after beam incumbent.
        bool exact_after_beam = env_int_or("PAST_PROFILE_REALIZATION_EXACT_AFTER_BEAM_ENABLE", 0) != 0;
        if (exact_after_beam && !run_profile_exact_mode && result.profile_beam_candidate_ub < kInf * 0.5)
        {
            int relaxed_merged_max = env_int_or("PAST_PROFILE_REALIZATION_EXACT_AFTER_BEAM_MAX_MERGED", 24);
            double relaxed_state_max = env_double_or("PAST_PROFILE_REALIZATION_EXACT_AFTER_BEAM_MAX_STATE", 1.0e12);
            double relaxed_comp_max = env_double_or("PAST_PROFILE_REALIZATION_EXACT_AFTER_BEAM_MAX_COMP", 1.0e12);
            if (static_cast<int>(merged.size()) <= relaxed_merged_max &&
                selector_state_space <= relaxed_state_max &&
                selector_total_comp_est <= relaxed_comp_max)
            {
                run_profile_exact_mode = true;
                result.profile_selector_decision = "exact";
                result.profile_selector_reason = "exact_after_beam_relaxed";
            }
        }

        auto run_profile_beam_plus_attempt = [&](const std::string &attempt_reason)
        {
            auto t0_repair = std::chrono::steady_clock::now();
            ProfileRepairBeamDiag beam_diag;
            double cand = block_repair_profile_repair_beam_ub(
                merged, lengths, totals, prefix_proc, T, spaces, result.bin_pack_ub, &beam_diag, true);
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_repair).count();
            result.profile_beam_plus_candidate_ub = std::min(result.profile_beam_plus_candidate_ub, cand);
            if (pack_trace)
            {
                std::cerr << "pack_trace stage=profile_repair_beam_plus_done"
                          << " reason=" << attempt_reason
                          << " t_profile_repair_beam_plus=" << dt
                          << " cand=" << cand
                          << " incumbent=" << result.bin_pack_ub
                          << "\n";
            }
        };

        bool need_beam_plus =
            (incumbent_source == "i3") ||
            (incumbent_source == "i4") ||
            (incumbent_source == "beam_plus") ||
            (incumbent_source == "best_step3");
        if (need_beam_plus)
            run_profile_beam_plus_attempt("incumbent_matrix");

        if (result.pack_solver == "lagrangian_assign" &&
            lengths.size() >= 5 &&
            !(result.bin_pack_ub < kInf * 0.5))
        {
            auto t0_lagr = std::chrono::steady_clock::now();
            if (pack_trace)
            {
                std::cerr << "pack_trace stage=lagrangian_assign_start"
                          << " incumbent=" << result.bin_pack_ub
                          << "\n";
            }
            double cand = block_repair_lagrangian_assign_ub(
                merged, lengths, totals, prefix_proc, T, spaces, result.bin_pack_ub);
            note_pack_candidate("block_repair_lagrangian_assign", cand);
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_lagr).count();
            result.t_pack_block_dp += dt;
            if (pack_trace)
            {
                std::cerr << "pack_trace stage=lagrangian_assign_done"
                          << " t_lagrangian_assign=" << dt
                          << " cand=" << cand
                          << " incumbent=" << result.bin_pack_ub
                          << " method=" << result.pack_method
                          << "\n";
            }
        }

        bool allow_lagrangian_beam_polish =
            (result.pack_solver == "lagrangian_beam_polish") &&
            lengths.size() >= 5 &&
            result.pack_method == "block_repair_lagrangian_assign" &&
            (result.bin_pack_ub < kInf * 0.5);
        int lagr_beam_polish_max_merged = env_int_or("PAST_BLOCK_REPAIR_LAGR_BEAM_POLISH_MAX_MERGED", 0);
        if (allow_lagrangian_beam_polish &&
            static_cast<int>(merged.size()) <= lagr_beam_polish_max_merged)
        {
            auto t0_polish = std::chrono::steady_clock::now();
            if (pack_trace)
            {
                std::cerr << "pack_trace stage=lagrangian_beam_polish_start"
                          << " incumbent=" << result.bin_pack_ub
                          << " merged_blocks=" << merged.size()
                          << "\n";
            }
            double cand = block_repair_feasible_beam_ub(
                merged, lengths, totals, prefix_proc, T, spaces);
            if (cand < beam_ub_for_exact_l2)
                beam_ub_for_exact_l2 = cand;
            note_pack_candidate("block_repair_feasible_beam_after_lagrangian", cand);
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_polish).count();
            result.t_pack_block_dp += dt;
            if (pack_trace)
            {
                std::cerr << "pack_trace stage=lagrangian_beam_polish_done"
                          << " t_lagrangian_beam_polish=" << dt
                          << " cand=" << cand
                          << " incumbent=" << result.bin_pack_ub
                          << " method=" << result.pack_method
                          << "\n";
            }
        }

        if (result.pack_solver == "feasible_counts" &&
            lengths.size() >= 5 &&
            !(result.bin_pack_ub < kInf * 0.5))
        {
            auto t0_feas = std::chrono::steady_clock::now();
            if (pack_trace)
            {
                std::cerr << "pack_trace stage=feasible_counts_start"
                          << " incumbent=" << result.bin_pack_ub
                          << "\n";
            }
            double cand = block_repair_feasible_counts_ub(
                merged, lengths, totals, prefix_proc, T, spaces);
            note_pack_candidate("block_repair_feasible_counts", cand);
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_feas).count();
            result.t_pack_block_dp += dt;
            if (pack_trace)
            {
                std::cerr << "pack_trace stage=feasible_counts_done"
                          << " t_feasible_counts=" << dt
                          << " cand=" << cand
                          << " incumbent=" << result.bin_pack_ub
                          << " method=" << result.pack_method
                          << "\n";
            }
        }

        if (result.pack_solver == "feasible_beam" &&
            lengths.size() >= 5 &&
            !(result.bin_pack_ub < kInf * 0.5))
        {
            auto t0_feas = std::chrono::steady_clock::now();
            if (pack_trace)
            {
                std::cerr << "pack_trace stage=feasible_beam_start"
                          << " incumbent=" << result.bin_pack_ub
                          << "\n";
            }
            double cand = block_repair_feasible_beam_ub(
                merged, lengths, totals, prefix_proc, T, spaces);
            if (cand < beam_ub_for_exact_l2)
                beam_ub_for_exact_l2 = cand;
            note_pack_candidate("block_repair_feasible_beam", cand);
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_feas).count();
            result.t_pack_block_dp += dt;
            if (pack_trace)
            {
                std::cerr << "pack_trace stage=feasible_beam_done"
                          << " t_feasible_beam=" << dt
                          << " cand=" << cand
                          << " incumbent=" << result.bin_pack_ub
                          << " method=" << result.pack_method
                          << "\n";
            }
        }

        if ((result.pack_solver == "energy_guided" ||
             result.pack_solver == "energy_core") &&
            lengths.size() >= 3)
        {
            auto t0_energy = std::chrono::steady_clock::now();
            EnergyCoreRunDiag ec_diag;
            if (pack_trace)
            {
                std::cerr << "pack_trace stage=energy_core_start"
                          << " incumbent=" << result.bin_pack_ub
                          << "\n";
            }
            double cand = block_repair_energy_core_ub(
                merged, lengths, totals, prefix_proc, T, spaces, result.bin_pack_ub, &ec_diag);

            result.ec_generated_patterns_total = ec_diag.pool.generated_total;
            result.ec_generated_patterns_max_block = ec_diag.pool.generated_max_block;
            result.ec_retained_patterns_total = ec_diag.pool.retained_total;
            result.ec_retained_patterns_max_block = ec_diag.pool.retained_max_block;
            result.ec_retained_patterns_signature = ec_diag.pool.retained_signature;
            result.ec_time_completion = ec_diag.t_completion;
            result.ec_time_pattern_generation = ec_diag.t_patterns;
            result.ec_time_exact_core = ec_diag.t_exact_core;
            result.ec_pruned_core_window = ec_diag.pruned_core_window;
            result.ec_pruned_suffix = ec_diag.pruned_suffix;
            result.ec_pruned_transition = ec_diag.pruned_transition;
            result.ec_pruned_bound = ec_diag.pruned_bound;
            result.ec_delta_used = ec_diag.delta_used;
            result.ec_fixed_blocks = ec_diag.retained_fixed_blocks;
            result.ec_two_phase_used = ec_diag.two_phase_used;
            result.ec_phase1_feasible_ub = ec_diag.phase1_feasible_ub;
            result.ec_time_phase1 = ec_diag.t_phase1;

            note_pack_candidate("block_repair_energy_core", cand);
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_energy).count();
            result.t_pack_block_dp += dt;
            if (pack_trace)
            {
                std::cerr << "pack_trace stage=energy_core_done"
                          << " t_energy_core=" << dt
                          << " cand=" << cand
                          << " incumbent=" << result.bin_pack_ub
                          << " method=" << result.pack_method
                          << "\n";
            }
        }

        if (result.pack_solver == "rg_beam" &&
            lengths.size() >= 4 &&
            !(result.bin_pack_ub < kInf * 0.5))
        {
            auto t0_beam = std::chrono::steady_clock::now();
            if (pack_trace)
            {
                std::cerr << "pack_trace stage=rg_beam_start"
                          << " incumbent=" << result.bin_pack_ub
                          << "\n";
            }
            double cand = block_repair_rg_beam_ub(
                merged, lengths, totals, prefix_proc, T, spaces, result.bin_pack_ub);
            note_pack_candidate("block_repair_rg_beam", cand);
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_beam).count();
            result.t_pack_block_dp += dt;
            if (pack_trace)
            {
                std::cerr << "pack_trace stage=rg_beam_done"
                          << " t_rg_beam=" << dt
                          << " cand=" << cand
                          << " incumbent=" << result.bin_pack_ub
                          << " method=" << result.pack_method
                          << "\n";
            }
        }

        bool exact_l2_enabled = env_int_or("PAST_BLOCK_REPAIR_EXACT_L2", 0) != 0;
        if (!exact_l2_enabled)
        {
            result.beam_ub_for_exact_l2 = beam_ub_for_exact_l2;
            result.exact_l2_status = "disabled";
        }
        else if (lengths.size() < 4)
        {
            result.beam_ub_for_exact_l2 = beam_ub_for_exact_l2;
            result.exact_l2_status = "skipped_small_k";
        }
        else
        {
            if (!(beam_ub_for_exact_l2 < kInf * 0.5))
                beam_ub_for_exact_l2 = block_repair_feasible_beam_ub(
                    merged, lengths, totals, prefix_proc, T, spaces);
            result.beam_ub_for_exact_l2 = beam_ub_for_exact_l2;

            double initial_ub = beam_ub_for_exact_l2;
            if (!(initial_ub < kInf * 0.5))
                initial_ub = result.bin_pack_ub;
            auto exact_l2 = block_repair_exact_level2_ub(
                merged, lengths, totals, prefix_proc, T, spaces, initial_ub);
            result.exact_l2_ub = exact_l2.ub;
            result.t_exact_l2 = exact_l2.time_sec;
            result.exact_l2_nodes = exact_l2.nodes;
            result.exact_l2_closed = exact_l2.closed ? 1 : 0;
            result.exact_l2_status = exact_l2.status;
            if (beam_ub_for_exact_l2 < kInf * 0.5 && exact_l2.ub < kInf * 0.5)
            {
                if (exact_l2.ub + 1e-9 < beam_ub_for_exact_l2)
                    result.exact_l2_improved_over_beam = 1;
                if (exact_l2.closed && std::abs(exact_l2.ub - beam_ub_for_exact_l2) <= 1e-9)
                    result.exact_l2_beam_optimal_in_pool = 1;
            }
            bool exact_l2_apply = env_int_or("PAST_BLOCK_REPAIR_EXACT_L2_APPLY", 0) != 0;
            if (exact_l2_apply)
                note_pack_candidate("block_repair_exact_level2_archival", exact_l2.ub);
            else
                result.exact_l2_status = "diag_" + result.exact_l2_status;
        }

        // ---------------------------------------------------------------
        // Step-3 exact mode: fixed-block profile-realization DP.
        //
        // Given recovered block capacities and bounded type multiplicities,
        // solve profile realization exactly in count-state space, with the
        // same per-block count semantics as profile_repair_beam.
        //
        // This remains separate from Step 4 global exact DP fallback.
        // ---------------------------------------------------------------
        auto run_block_dp_packing = [&]()
        {
            bool force_exact_dp = env_int_or("PAST_RELAXED_BINPACK_FORCE_EXACT_DP", 0) != 0;
            bool mainline_exact_mode =
                (result.pack_solver == "default" || result.pack_solver == "profile_repair_beam") &&
                (env_int_or("PAST_PROFILE_REALIZATION_DP_EXACT_ENABLE", 1) != 0);
            if (exact_pack_decided || (!force_exact_dp && !mainline_exact_mode && result.bin_pack_ub < kInf * 0.5))
            {
                result.block_dp_status =
                    (result.bin_pack_ub < kInf * 0.5) ? "skipped_have_incumbent" : "skipped_decided";
                if (pack_trace)
                {
                    std::cerr << "pack_trace stage=block_dp_skip"
                              << " reason=" << result.block_dp_status
                              << " incumbent=" << result.bin_pack_ub
                              << "\n";
                }
                return;
            }

            if (!run_profile_exact_mode)
            {
                result.block_dp_status = "skipped_selector";
                if (pack_trace)
                {
                    std::cerr << "pack_trace stage=block_dp_skip"
                              << " reason=selector"
                              << " decision=" << result.profile_selector_decision
                              << " selector_reason=" << result.profile_selector_reason
                              << "\n";
                }
                return;
            }

            double exact_time_limit = std::max(0.0, env_double_or("PAST_PROFILE_REALIZATION_EXACT_TIME_LIMIT", 0.0));
            auto t0_block_dp = std::chrono::steady_clock::now();
            auto exact_out_of_time = [&]() -> bool
            {
                if (exact_time_limit <= 0.0)
                    return false;
                double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_block_dp).count();
                return elapsed >= exact_time_limit;
            };

            int K = static_cast<int>(lengths.size());
            int nBlk = static_cast<int>(nB);
            double nc_limit = static_cast<double>(env_int_or("PAST_RELAXED_BINPACK_MAX_NC", 100000000));
            double comp_est_limit = static_cast<double>(env_int_or("PAST_RELAXED_BINPACK_MAX_COMP_EST", 100000000));
            bool use_suffix_prune = env_int_or("PAST_PROFILE_REALIZATION_EXACT_SUFFIX_PRUNE", 1) != 0;
            result.block_dp_timed_out = 0;

            // --- Enumerate valid compositions per block ---
            struct BComp
            {
                int64_t delta;
                int counts[kMaxRepairTypes];
            };

            std::vector<int64_t> bp_strides(K);
            int64_t bp_NC = 1;
            for (int i = 0; i < K; ++i)
            {
                bp_strides[i] = bp_NC;
                bp_NC *= (totals[i] + 1);
            }
            result.block_dp_state_space = static_cast<double>(bp_NC);

            double total_comp_est = 0.0;
            double max_block_comp_est = 0.0;
            for (int b = 0; b < nBlk; ++b)
            {
                double comp_est = 1.0;
                int cap = orig_cap[b];
                for (int i = 0; i < K; ++i)
                {
                    int maxc = std::min(totals[i], cap / std::max(1, lengths[i]));
                    comp_est = sat_mul(comp_est, static_cast<double>(maxc + 1), comp_est_limit);
                }
                total_comp_est = std::min(comp_est_limit, total_comp_est + comp_est);
                max_block_comp_est = std::max(max_block_comp_est, comp_est);
            }
            result.block_dp_total_comp_estimate = total_comp_est;
            result.block_dp_max_comp_estimate = max_block_comp_est;

            if (pack_trace)
            {
                std::cerr << "pack_trace stage=block_dp_estimate"
                          << " state_space=" << result.block_dp_state_space
                          << " total_comp_est=" << total_comp_est
                          << " max_block_comp_est=" << max_block_comp_est
                          << " nc_limit=" << nc_limit
                          << " comp_est_limit=" << comp_est_limit
                          << "\n";
            }

            if (total_comp_est >= comp_est_limit)
            {
                result.block_dp_status = "skipped_comp_est";
                if (!(result.bin_pack_ub < kInf * 0.5))
                    result.pack_outcome = "exact_skipped_comp_est";
                if (pack_trace)
                {
                    std::cerr << "pack_trace stage=block_dp_skip"
                              << " reason=comp_est_limit"
                              << " total_comp_est=" << total_comp_est
                              << " comp_est_limit=" << comp_est_limit
                              << "\n";
                }
                return;
            }

            std::vector<std::vector<BComp>> bcomps(nBlk);
            double total_comps = 0.0;
            std::size_t max_block_comps = 0;
            for (int b = 0; b < nBlk; ++b)
            {
                if (exact_out_of_time())
                {
                    result.block_dp_status = "timeout";
                    result.block_dp_timed_out = 1;
                    result.t_pack_block_dp = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_block_dp).count();
                    result.t_pack_block_dp_exact = result.t_pack_block_dp;
                    return;
                }
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
                total_comps += static_cast<double>(bcomps[b].size());
                max_block_comps = std::max(max_block_comps, bcomps[b].size());
            }
            result.block_dp_total_compositions = total_comps;
            result.block_dp_max_compositions_per_block = static_cast<double>(max_block_comps);

            if (pack_trace)
            {
                std::cerr << "pack_trace stage=block_dp_prepare"
                          << " blocks=" << nBlk
                          << " K=" << K
                          << " state_space=" << result.block_dp_state_space
                          << " total_comps=" << result.block_dp_total_compositions
                          << " max_block_comps=" << max_block_comps
                          << " nc_limit=" << nc_limit
                          << "\n";
            }

            if (result.block_dp_state_space > nc_limit)
            {
                result.block_dp_status = "skipped_nc";
                if (!(result.bin_pack_ub < kInf * 0.5))
                    result.pack_outcome = "exact_skipped_nc";
                if (pack_trace)
                {
                    std::cerr << "pack_trace stage=block_dp_skip"
                              << " reason=nc_limit"
                              << " state_space=" << result.block_dp_state_space
                              << " nc_limit=" << nc_limit
                              << "\n";
                }
                return;
            }

            // --- Shared Step-3 block ordering (hardest-first when enabled) ---
            std::vector<int> border(nBlk);
            std::iota(border.begin(), border.end(), 0);
            if (profile_realization_hardest_first_enabled())
            {
                std::sort(border.begin(), border.end(), [&](int a, int b)
                          {
                              if (bcomps[a].size() != bcomps[b].size())
                                  return bcomps[a].size() < bcomps[b].size();
                              if (orig_cap[a] != orig_cap[b])
                                  return orig_cap[a] > orig_cap[b];
                              return a < b;
                          });
            }

            std::vector<int> suffix_cap(nBlk + 1, 0);
            for (int i = nBlk - 1; i >= 0; --i)
                suffix_cap[i] = suffix_cap[i + 1] + orig_cap[border[i]];

            std::vector<std::vector<int>> suffix_min_counts(nBlk + 1, std::vector<int>(K, 0));
            std::vector<std::vector<int>> suffix_max_counts(nBlk + 1, std::vector<int>(K, 0));
            for (int bi = nBlk - 1; bi >= 0; --bi)
            {
                int b = border[bi];
                for (int i = 0; i < K; ++i)
                {
                    int mn = INT_MAX;
                    int mx = 0;
                    for (const auto &bc : bcomps[b])
                    {
                        mn = std::min(mn, bc.counts[i]);
                        mx = std::max(mx, bc.counts[i]);
                    }
                    if (mn == INT_MAX)
                        mn = 0;
                    suffix_min_counts[bi][i] = suffix_min_counts[bi + 1][i] + mn;
                    suffix_max_counts[bi][i] = suffix_max_counts[bi + 1][i] + mx;
                }
            }

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

            auto decode_state = [&](int64_t s, int r[kMaxRepairTypes])
            {
                int64_t tmp = s;
                for (int i = 0; i < K; ++i)
                {
                    r[i] = static_cast<int>(tmp % (totals[i] + 1));
                    tmp /= (totals[i] + 1);
                }
            };

            // --- Forward pass: build reachability sets ---
            result.block_dp_status = "running";

            std::vector<std::unordered_set<int64_t>> reach(nBlk + 1);
            reach[0].insert(initial_st);

            for (int bi = 0; bi < nBlk; ++bi)
            {
                if (exact_out_of_time())
                {
                    result.block_dp_status = "timeout";
                    result.block_dp_timed_out = 1;
                    break;
                }
                int b = border[bi];
                auto &comps_b = bcomps[b];
                int required_work = suffix_cap[bi];
                for (int64_t s : reach[bi])
                {
                    if (((static_cast<uint64_t>(s) + static_cast<uint64_t>(bi)) & 4095ULL) == 0ULL && exact_out_of_time())
                    {
                        result.block_dp_status = "timeout";
                        result.block_dp_timed_out = 1;
                        break;
                    }
                    if (compute_work(s) != required_work)
                        continue;
                    int r[kMaxRepairTypes];
                    decode_state(s, r);
                    if (use_suffix_prune)
                    {
                        bool layer_feasible = true;
                        for (int i = 0; i < K; ++i)
                        {
                            if (r[i] < suffix_min_counts[bi][i] || r[i] > suffix_max_counts[bi][i])
                            {
                                layer_feasible = false;
                                break;
                            }
                        }
                        if (!layer_feasible)
                            continue;
                    }
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
                            int rem = r[i] - bc.counts[i];
                            if (use_suffix_prune &&
                                (rem < suffix_min_counts[bi + 1][i] || rem > suffix_max_counts[bi + 1][i]))
                            {
                                ok = false;
                                break;
                            }
                        }
                        if (ok)
                            reach[bi + 1].insert(s - bc.delta);
                    }
                }
                if (pack_trace)
                {
                    std::cerr << "pack_trace stage=block_dp_layer"
                              << " layer=" << bi
                              << " reach_in=" << reach[bi].size()
                              << " reach_out=" << reach[bi + 1].size()
                              << " comps=" << bcomps[border[bi]].size()
                              << "\n";
                }
                if (result.block_dp_timed_out)
                    break;
            }

            // --- Result: reconstruct assignment if feasible ---
            if (result.block_dp_timed_out)
            {
                // keep timeout status
            }
            else if (reach[nBlk].count(0))
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
                    int r[kMaxRepairTypes];
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
                            int rem = r[i] - bc.counts[i];
                            if (use_suffix_prune &&
                                (rem < suffix_min_counts[bi + 1][i] || rem > suffix_max_counts[bi + 1][i]))
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
                    int l3_max_cells = env_int_or("PAST_BLOCK_REPAIR_L3_MAX_CELLS", 50'000);
                    double l3_time_limit = env_double_or("PAST_BLOCK_REPAIR_L3_TIME_LIMIT", 0.05);
                    std::vector<SPACESResult> block_spaces;
                    std::vector<std::vector<double>> block_prefix_proc;
                    build_profile_block_local_views(
                        merged,
                        prefix_proc,
                        T,
                        spaces,
                        &block_spaces,
                        &block_prefix_proc);
                    double exact_mode_cost = 0.0;
                    bool exact_mode_finite = true;
                    for (int b = 0; b < nBlk; ++b)
                    {
                        double c = evaluate_profile_block_counts(
                            b,
                            asgn[b],
                            lengths,
                            merged,
                            block_spaces,
                            block_prefix_proc,
                            l3_max_cells,
                            l3_time_limit);
                        if (!(c < kInf * 0.5))
                        {
                            exact_mode_finite = false;
                            break;
                        }
                        exact_mode_cost += c;
                    }
                    if (exact_mode_finite)
                    {
                        result.profile_exact_candidate_ub = exact_mode_cost;
                        result.profile_exact_improved_over_step2 =
                            (exact_mode_cost + 1e-9 < result.profile_step2_ub) ? 1 : 0;
                        note_pack_candidate("profile_realization_dp_exact", exact_mode_cost);
                        if (pack_trace)
                        {
                            std::cerr << "pack_trace stage=profile_realization_dp_exact_done"
                                      << " hardest_first=" << (profile_realization_hardest_first_enabled() ? 1 : 0)
                                      << " suffix_prune=" << (use_suffix_prune ? 1 : 0)
                                      << " cand=" << exact_mode_cost
                                      << " incumbent=" << result.bin_pack_ub
                                      << "\n";
                        }
                    }
                    result.block_dp_status = "feasible";
                }
                else
                    result.block_dp_status = "reconstruct_failed";
            }
            else
            {
                result.pack_outcome = "infeasible";
                result.block_dp_status = "infeasible";
            }

            result.t_pack_block_dp =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_block_dp).count();
            result.t_pack_block_dp_exact = result.t_pack_block_dp;
            if (pack_trace)
            {
                std::cerr << "pack_trace stage=block_dp_done"
                          << " status=" << result.block_dp_status
                          << " t_block_dp=" << result.t_pack_block_dp
                          << " incumbent=" << result.bin_pack_ub
                          << "\n";
            }
        };

        // Step-3 exact mode is now part of default/profile-repair flow and can
        // be explicitly enabled for diagnostics.
        bool run_block_dp =
            run_profile_exact_mode ||
            (env_int_or("PAST_RELAXED_BINPACK_BLOCK_DP_DIAG", 0) != 0);
        if (run_block_dp)
            run_block_dp_packing();
        else
            result.block_dp_status = (!run_profile_exact_mode && selector_in_scope) ? "skipped_selector" : "disabled";

        bool exact_primary_auto_v1 =
            selector_in_scope && selector_exact_primary && (selector_policy == "auto_v1");
        if (selector_exact_primary)
            result.profile_exact_primary_status_before_fallback = result.block_dp_status;
        if (exact_primary_auto_v1)
        {
            bool exact_has_candidate = (result.profile_exact_candidate_ub < kInf * 0.5);
            if (!exact_has_candidate)
            {
                result.profile_exact_primary_fallback_to_beam = 1;
                if (pack_trace)
                {
                    std::cerr << "pack_trace stage=profile_exact_primary_fallback"
                              << " exact_status=" << result.block_dp_status
                              << " exact_candidate=" << result.profile_exact_candidate_ub
                              << " step2_ub=" << result.profile_step2_ub
                              << "\n";
                }
                run_profile_beam_attempt("exact_primary_fallback");
            }
        }

        if (selector_exact_primary && result.profile_beam_status == "skipped_selector")
            result.profile_beam_status = "skipped_selector_exact_primary";

        if (!run_profile_exact_mode && result.profile_exact_candidate_ub >= kInf * 0.5)
        {
            result.profile_exact_improved_over_step2 = 0;
            if (result.block_dp_status == "not_attempted")
                result.block_dp_status = "skipped_selector";
        }

        {
            double ub_i0 = result.profile_step2_ub;
            double ub_i1 = result.profile_exact_candidate_ub;
            double ub_i2 = result.profile_beam_candidate_ub;
            double ub_i3 = result.profile_beam_plus_candidate_ub;
            double ub_i4 = std::min(ub_i1, ub_i3);

            auto finite_ub = [](double v) -> bool
            { return v < kInf * 0.5; };

            auto choose_best = [&](double fallback) -> double
            {
                double best = fallback;
                if (finite_ub(ub_i0)) best = std::min(best, ub_i0);
                if (finite_ub(ub_i1)) best = std::min(best, ub_i1);
                if (finite_ub(ub_i2)) best = std::min(best, ub_i2);
                if (finite_ub(ub_i3)) best = std::min(best, ub_i3);
                return best;
            };

            double chosen = result.bin_pack_ub;
            if (inc_src_i0)
                chosen = ub_i0;
            else if (inc_src_i1)
                chosen = ub_i1;
            else if (inc_src_i2)
                chosen = ub_i2;
            else if (inc_src_i3)
                chosen = ub_i3;
            else if (inc_src_i4)
                chosen = ub_i4;
            else
                chosen = choose_best(result.bin_pack_ub);

            if (!(chosen < kInf * 0.5))
                chosen = finite_ub(ub_i0) ? ub_i0 : choose_best(result.bin_pack_ub);

            if (chosen < kInf * 0.5)
            {
                result.bin_pack_ub = chosen;
                result.pack_outcome = "feasible";
            }
            result.profile_incumbent_ub_for_exact = chosen;
        }

        bool step3_exact_finite = (result.profile_exact_candidate_ub < kInf * 0.5);
        bool step3_beam_finite = (result.profile_beam_candidate_ub < kInf * 0.5);
        if (step3_exact_finite || step3_beam_finite)
        {
            if (step3_exact_finite && (!step3_beam_finite || result.profile_exact_candidate_ub <= result.profile_beam_candidate_ub + 1e-9))
                result.profile_step3_incumbent_mode = "exact";
            else
                result.profile_step3_incumbent_mode = "beam";
        }
        else
        {
            result.profile_step3_incumbent_mode = "none";
        }

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
        auto t_dense_begin = std::chrono::steady_clock::now();
        std::vector<int> sorted_lengths = lengths;
        std::sort(sorted_lengths.begin(), sorted_lengths.end());
        sorted_lengths.erase(std::unique(sorted_lengths.begin(), sorted_lengths.end()), sorted_lengths.end());
        bool dense_unit_candidate = !sorted_lengths.empty() && sorted_lengths.front() == 1;
        for (int i = 1; i < static_cast<int>(sorted_lengths.size()) && dense_unit_candidate; ++i)
            if (sorted_lengths[i] != sorted_lengths[i - 1] + 1)
                dense_unit_candidate = false;
        const bool dense_relax_fastpath =
            dense_unit_candidate &&
            (K >= env_int_or("PAST_DENSE_UNIT_RELAX_FASTPATH_K_MIN", 8)) &&
            (env_int_or("PAST_DENSE_UNIT_RELAX_FASTPATH", 0) != 0);
        const bool dense_energy_profile =
            dense_unit_candidate &&
            (K >= env_int_or("PAST_DENSE_UNIT_RELAX_FASTPATH_K_MIN", 8)) &&
            (env_int_or("PAST_DENSE_UNIT_ENERGY_PROFILE", 0) != 0);

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
        auto t_dense_after_spaces_or_lb = std::chrono::steady_clock::now();
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
        auto t_dense_after_profile_dp = std::chrono::steady_clock::now();

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
        bool pack_trace = env_int_or("PAST_RELAXED_PACK_TRACE", 0) != 0;
        double total_profile_recovery = 0.0;
        int profiles_tried = 0;
        std::vector<Segment> chosen_block_profile;
        pack.pack_co_optimal_profiles = static_cast<int>(co_optimal_terminals.size());
        if (pack_trace)
        {
            std::cerr << "pack_trace stage=co_optimal_profiles"
                      << " count=" << co_optimal_terminals.size()
                      << " best_t=" << best_t
                      << "\n";
        }
        for (int term_t : co_optimal_terminals)
        {
            auto t0_profile = std::chrono::steady_clock::now();
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
            double t_profile = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_profile).count();
            total_profile_recovery += t_profile;
            profiles_tried += 1;
            if (pack_trace)
            {
                std::cerr << "pack_trace stage=profile"
                          << " idx=" << profiles_tried
                          << " term_t=" << term_t
                          << " blocks=" << blocks.size()
                          << " t_profile=" << t_profile
                          << "\n";
            }
            pack = pack_recovered_blocks(blocks, lengths, totals, prefix_proc, T, spaces);
            pack.t_pack_profile_recovery = total_profile_recovery;
            pack.pack_profiles_tried = profiles_tried;
            pack.pack_co_optimal_profiles = static_cast<int>(co_optimal_terminals.size());
            if (pack.bin_pack_ub < kInf * 0.5)
            {
                chosen_block_profile.clear();
                chosen_block_profile.reserve(blocks.size());
                for (const auto &b : blocks)
                    chosen_block_profile.push_back(Segment{b.start, b.length});
                break; // found a packable profile!
            }
        }

        auto t_dense_after_profile_recovery = std::chrono::steady_clock::now();

        RelaxedDPResult result;
        result.lb = lb;
        result.bin_pack_ub = pack.bin_pack_ub;
        result.states_reached = states_reached;
        result.states_expanded = states_expanded;
        result.block_profile = std::move(chosen_block_profile);
        result.rdp = std::move(dp);  // zero-copy transfer of the dp table
        result.RW = RW;
        result.block_count = pack.block_count;
        result.merged_block_count = pack.merged_block_count;
        result.merged_blocks = pack.merged_blocks;
        result.pack_solver = pack.pack_solver;
        result.pack_external_status = pack.pack_external_status;
        result.pack_method = pack.pack_method;
        result.pack_outcome = pack.pack_outcome;
        result.t_pack_external = pack.t_pack_external;
        result.t_pack_heuristic = pack.t_pack_heuristic;
        result.t_pack_dfs = pack.t_pack_dfs;
        result.t_pack_block_dp = pack.t_pack_block_dp;
        result.t_pack_profile_recovery = pack.t_pack_profile_recovery;
        result.t_pack_merge_blocks = pack.t_pack_merge_blocks;
        result.t_pack_to_first_candidate = pack.t_pack_to_first_candidate;
        result.t_pack_ffd_only = pack.t_pack_ffd_only;
        result.step2_reached = pack.step2_reached;
        result.step2_produced_ub = pack.step2_produced_ub;

        result.t_dense_spaces_or_lb =
            std::chrono::duration<double>(t_dense_after_spaces_or_lb - t_dense_begin).count();
        result.t_dense_profile_dp =
            std::chrono::duration<double>(t_dense_after_profile_dp - t_dense_after_spaces_or_lb).count();
        result.t_dense_profile_recovery =
            std::chrono::duration<double>(t_dense_after_profile_recovery - t_dense_after_profile_dp).count();
        result.t_dense_block_build = pack.t_pack_merge_blocks;
        result.t_dense_job_materialization = std::max(0.0, pack.t_pack_heuristic - pack.t_pack_ffd_only);
        result.t_dense_step2_pack = pack.t_pack_ffd_only;
        result.t_dense_pre_step2_total =
            result.t_dense_spaces_or_lb + result.t_dense_profile_dp + result.t_dense_profile_recovery +
            result.t_dense_block_build + result.t_dense_job_materialization + result.t_dense_step2_pack;
        result.pack_profiles_tried = pack.pack_profiles_tried;
        result.pack_co_optimal_profiles = pack.pack_co_optimal_profiles;
        result.block_dp_state_space = pack.block_dp_state_space;
        result.block_dp_total_compositions = pack.block_dp_total_compositions;
        result.block_dp_total_comp_estimate = pack.block_dp_total_comp_estimate;
        result.block_dp_max_comp_estimate = pack.block_dp_max_comp_estimate;
        result.block_dp_max_compositions_per_block = pack.block_dp_max_compositions_per_block;
        result.block_dp_status = pack.block_dp_status;
        result.block_dp_timed_out = pack.block_dp_timed_out;
        result.beam_ub_for_exact_l2 = pack.beam_ub_for_exact_l2;
        result.exact_l2_ub = pack.exact_l2_ub;
        result.t_exact_l2 = pack.t_exact_l2;
        result.exact_l2_nodes = pack.exact_l2_nodes;
        result.exact_l2_closed = pack.exact_l2_closed;
        result.exact_l2_improved_over_beam = pack.exact_l2_improved_over_beam;
        result.exact_l2_beam_optimal_in_pool = pack.exact_l2_beam_optimal_in_pool;
        result.exact_l2_status = pack.exact_l2_status;
        result.profile_beam_base_width = pack.profile_beam_base_width;
        result.profile_beam_avg_width = pack.profile_beam_avg_width;
        result.profile_beam_max_width = pack.profile_beam_max_width;
        result.profile_beam_states_considered = pack.profile_beam_states_considered;
        result.profile_beam_states_kept = pack.profile_beam_states_kept;
        result.profile_beam_pruned_over = pack.profile_beam_pruned_over;
        result.profile_beam_pruned_suffix = pack.profile_beam_pruned_suffix;
        result.profile_beam_pruned_discrepancy = pack.profile_beam_pruned_discrepancy;
        result.profile_beam_discrepancy_budget = pack.profile_beam_discrepancy_budget;
        result.profile_beam_discrepancy_depth = pack.profile_beam_discrepancy_depth;
        result.profile_beam_status = pack.profile_beam_status;
        result.profile_beam_timed_out = pack.profile_beam_timed_out;
        result.profile_beam_key_multi_policy = pack.profile_beam_key_multi_policy;
        result.profile_beam_key_multi_max = pack.profile_beam_key_multi_max;
        result.profile_beam_key_multi_score_eps = pack.profile_beam_key_multi_score_eps;
        result.profile_beam_key_multi_diversity_eps = pack.profile_beam_key_multi_diversity_eps;
        result.profile_beam_score_policy = pack.profile_beam_score_policy;
        result.profile_beam_residual_weight = pack.profile_beam_residual_weight;
        result.profile_beam_residual_mean_penalty = pack.profile_beam_residual_mean_penalty;
        result.profile_beam_residual_max_penalty = pack.profile_beam_residual_max_penalty;
        result.profile_beam_late_frac = pack.profile_beam_late_frac;
        result.profile_realization_hardest_first = pack.profile_realization_hardest_first;
        result.profile_realization_exact_suffix_prune = pack.profile_realization_exact_suffix_prune;
        result.t_pack_profile_beam = pack.t_pack_profile_beam;
        result.t_pack_block_dp_exact = pack.t_pack_block_dp_exact;
        result.profile_step2_ub = pack.profile_step2_ub;
        result.profile_beam_candidate_ub = pack.profile_beam_candidate_ub;
        result.profile_beam_plus_candidate_ub = pack.profile_beam_plus_candidate_ub;
        result.profile_exact_candidate_ub = pack.profile_exact_candidate_ub;
        result.profile_beam_improved_over_step2 = pack.profile_beam_improved_over_step2;
        result.profile_exact_improved_over_step2 = pack.profile_exact_improved_over_step2;
        result.profile_incumbent_source = pack.profile_incumbent_source;
        result.profile_incumbent_ub_for_exact = pack.profile_incumbent_ub_for_exact;
        result.profile_selector_policy = pack.profile_selector_policy;
        result.profile_selector_decision = pack.profile_selector_decision;
        result.profile_selector_reason = pack.profile_selector_reason;
        result.profile_selector_has_one = pack.profile_selector_has_one;
        result.profile_selector_contiguous = pack.profile_selector_contiguous;
        result.profile_selector_multiplicity = pack.profile_selector_multiplicity;
        result.profile_selector_semigroup_density = pack.profile_selector_semigroup_density;
        result.profile_selector_hard_alarm = pack.profile_selector_hard_alarm;
        result.profile_exact_primary_fallback_to_beam = pack.profile_exact_primary_fallback_to_beam;
        result.profile_exact_primary_status_before_fallback = pack.profile_exact_primary_status_before_fallback;
        result.profile_step3_incumbent_mode = pack.profile_step3_incumbent_mode;
        result.profile_beam_chosen_counts = std::move(pack.profile_beam_chosen_counts);
        result.profile_beam_block_order = std::move(pack.profile_beam_block_order);
        // PLAN28: block-realizability diagnostics
        result.block_realiz_diag_active = pack.block_realiz_diag_active;
        result.block_realiz_blocks_total = pack.block_realiz_blocks_total;
        result.block_realiz_bad_blocks = pack.block_realiz_bad_blocks;
        result.block_realiz_bad_rate = pack.block_realiz_bad_rate;
        result.block_realiz_first_bad_block = pack.block_realiz_first_bad_block;
        result.block_realiz_min_finite_patterns = pack.block_realiz_min_finite_patterns;
        result.block_realiz_mean_finite_patterns = pack.block_realiz_mean_finite_patterns;
        result.block_realiz_base_path_survives = pack.block_realiz_base_path_survives;
        result.block_realiz_base_reject_reason = pack.block_realiz_base_reject_reason;
        result.block_realiz_diag_time_sec = pack.block_realiz_diag_time_sec;
        result.block_realiz_diag_skipped = pack.block_realiz_diag_skipped;
        result.block_realiz_diag_skip_reason = pack.block_realiz_diag_skip_reason;
        result.block_realiz_per_block_payload = pack.block_realiz_per_block_payload;
        // PLAN29: block-view diagnostics
        result.block_view_policy = pack.block_view_policy;
        result.block_view_original_blocks = pack.block_view_original_blocks;
        result.block_view_final_blocks = pack.block_view_final_blocks;
        result.block_view_removed_boundaries = pack.block_view_removed_boundaries;
        result.block_view_target_b = pack.block_view_target_b;
        result.block_view_price_preserve_used = pack.block_view_price_preserve_used;
        result.block_view_arith_adaptive_used = pack.block_view_arith_adaptive_used;
        result.block_view_selected = pack.block_view_selected;
        result.block_view_eval_count = pack.block_view_eval_count;
        result.block_view_best_ub = pack.block_view_best_ub;
        result.block_view_time_sec = pack.block_view_time_sec;
        result.dense_unit_fastpath_active = pack.dense_unit_fastpath_active;
        result.count_based_ffd_active = pack.count_based_ffd_active;
        result.dense_unit_relax_fastpath_active = dense_relax_fastpath ? 1 : 0;
        result.dense_unit_energy_profile_active = dense_energy_profile ? 1 : 0;
        result.dense_unit_relax_fastpath_fallback =
            (dense_relax_fastpath && !(pack.dense_unit_fastpath_active && pack.profile_step2_ub < kInf * 0.5)) ? 1 : 0;
        result.dense_unit_energy_profile_fallback =
            (dense_energy_profile && !(pack.dense_unit_fastpath_active && pack.profile_step2_ub < kInf * 0.5)) ? 1 : 0;
        if (dense_energy_profile)
            result.dense_unit_relax_mode = "energy_profile";
        else if (dense_relax_fastpath)
            result.dense_unit_relax_mode = "relax_fastpath";
        else
            result.dense_unit_relax_mode = "none";
        result.ec_generated_patterns_total = pack.ec_generated_patterns_total;
        result.ec_generated_patterns_max_block = pack.ec_generated_patterns_max_block;
        result.ec_retained_patterns_total = pack.ec_retained_patterns_total;
        result.ec_retained_patterns_max_block = pack.ec_retained_patterns_max_block;
        result.ec_retained_patterns_signature = pack.ec_retained_patterns_signature;
        result.ec_time_completion = pack.ec_time_completion;
        result.ec_time_pattern_generation = pack.ec_time_pattern_generation;
        result.ec_time_exact_core = pack.ec_time_exact_core;
        result.ec_pruned_core_window = pack.ec_pruned_core_window;
        result.ec_pruned_suffix = pack.ec_pruned_suffix;
        result.ec_pruned_transition = pack.ec_pruned_transition;
        result.ec_pruned_bound = pack.ec_pruned_bound;
        result.ec_delta_used = pack.ec_delta_used;
        result.ec_fixed_blocks = pack.ec_fixed_blocks;
        result.ec_two_phase_used = pack.ec_two_phase_used;
        result.ec_phase1_feasible_ub = pack.ec_phase1_feasible_ub;
        result.ec_time_phase1 = pack.ec_time_phase1;
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
        bool pack_trace = env_int_or("PAST_RELAXED_PACK_TRACE", 0) != 0;
        double total_profile_recovery = 0.0;
        int profiles_tried = 0;
        std::vector<Segment> chosen_block_profile;
        pack.pack_co_optimal_profiles = static_cast<int>(co_optimal_terminals.size());
        if (pack_trace)
        {
            std::cerr << "pack_trace stage=co_optimal_profiles"
                      << " count=" << co_optimal_terminals.size()
                      << " best_t=" << best_t
                      << "\n";
        }
        for (int term_t : co_optimal_terminals)
        {
            auto t0_profile = std::chrono::steady_clock::now();
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
            double t_profile = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_profile).count();
            total_profile_recovery += t_profile;
            profiles_tried += 1;
            if (pack_trace)
            {
                std::cerr << "pack_trace stage=profile"
                          << " idx=" << profiles_tried
                          << " term_t=" << term_t
                          << " blocks=" << blocks.size()
                          << " t_profile=" << t_profile
                          << "\n";
            }
            pack = pack_recovered_blocks(blocks, lengths, totals, prefix_proc, T, spaces);
            pack.t_pack_profile_recovery = total_profile_recovery;
            pack.pack_profiles_tried = profiles_tried;
            pack.pack_co_optimal_profiles = static_cast<int>(co_optimal_terminals.size());
            if (pack.bin_pack_ub < kInf * 0.5)
            {
                chosen_block_profile.clear();
                chosen_block_profile.reserve(blocks.size());
                for (const auto &b : blocks)
                    chosen_block_profile.push_back(Segment{b.start, b.length});
                break;
            }
        }

        RelaxedDPResult result;
        result.lb = best;
        result.bin_pack_ub = pack.bin_pack_ub;
        result.states_reached = states_reached;
        result.states_expanded = states_expanded;
        result.block_profile = std::move(chosen_block_profile);
        result.rdp = std::move(dp);
        result.RW = RW;
        result.block_count = pack.block_count;
        result.merged_block_count = pack.merged_block_count;
        result.merged_blocks = pack.merged_blocks;
        result.pack_solver = pack.pack_solver;
        result.pack_external_status = pack.pack_external_status;
        result.pack_method = pack.pack_method;
        result.pack_outcome = pack.pack_outcome;
        result.t_pack_external = pack.t_pack_external;
        result.t_pack_heuristic = pack.t_pack_heuristic;
        result.t_pack_dfs = pack.t_pack_dfs;
        result.t_pack_block_dp = pack.t_pack_block_dp;
        result.t_pack_profile_recovery = pack.t_pack_profile_recovery;
        result.t_pack_merge_blocks = pack.t_pack_merge_blocks;
        result.t_pack_to_first_candidate = pack.t_pack_to_first_candidate;
        result.t_pack_ffd_only = pack.t_pack_ffd_only;
        result.step2_reached = pack.step2_reached;
        result.step2_produced_ub = pack.step2_produced_ub;
        result.t_dense_spaces_or_lb = 0.0;
        result.t_dense_profile_dp = 0.0;
        result.t_dense_profile_recovery = 0.0;
        result.t_dense_block_build = pack.t_pack_merge_blocks;
        result.t_dense_job_materialization = std::max(0.0, pack.t_pack_heuristic - pack.t_pack_ffd_only);
        result.t_dense_step2_pack = pack.t_pack_ffd_only;
        result.t_dense_pre_step2_total =
            result.t_dense_spaces_or_lb + result.t_dense_profile_dp + result.t_dense_profile_recovery +
            result.t_dense_block_build + result.t_dense_job_materialization + result.t_dense_step2_pack;
        result.pack_profiles_tried = pack.pack_profiles_tried;
        result.pack_co_optimal_profiles = pack.pack_co_optimal_profiles;
        result.block_dp_state_space = pack.block_dp_state_space;
        result.block_dp_total_compositions = pack.block_dp_total_compositions;
        result.block_dp_total_comp_estimate = pack.block_dp_total_comp_estimate;
        result.block_dp_max_comp_estimate = pack.block_dp_max_comp_estimate;
        result.block_dp_max_compositions_per_block = pack.block_dp_max_compositions_per_block;
        result.block_dp_status = pack.block_dp_status;
        result.block_dp_timed_out = pack.block_dp_timed_out;
        result.beam_ub_for_exact_l2 = pack.beam_ub_for_exact_l2;
        result.exact_l2_ub = pack.exact_l2_ub;
        result.t_exact_l2 = pack.t_exact_l2;
        result.exact_l2_nodes = pack.exact_l2_nodes;
        result.exact_l2_closed = pack.exact_l2_closed;
        result.exact_l2_improved_over_beam = pack.exact_l2_improved_over_beam;
        result.exact_l2_beam_optimal_in_pool = pack.exact_l2_beam_optimal_in_pool;
        result.exact_l2_status = pack.exact_l2_status;
        result.profile_beam_base_width = pack.profile_beam_base_width;
        result.profile_beam_avg_width = pack.profile_beam_avg_width;
        result.profile_beam_max_width = pack.profile_beam_max_width;
        result.profile_beam_states_considered = pack.profile_beam_states_considered;
        result.profile_beam_states_kept = pack.profile_beam_states_kept;
        result.profile_beam_pruned_over = pack.profile_beam_pruned_over;
        result.profile_beam_pruned_suffix = pack.profile_beam_pruned_suffix;
        result.profile_beam_pruned_discrepancy = pack.profile_beam_pruned_discrepancy;
        result.profile_beam_discrepancy_budget = pack.profile_beam_discrepancy_budget;
        result.profile_beam_discrepancy_depth = pack.profile_beam_discrepancy_depth;
        result.profile_beam_status = pack.profile_beam_status;
        result.profile_beam_timed_out = pack.profile_beam_timed_out;
        result.profile_beam_key_multi_policy = pack.profile_beam_key_multi_policy;
        result.profile_beam_key_multi_max = pack.profile_beam_key_multi_max;
        result.profile_beam_key_multi_score_eps = pack.profile_beam_key_multi_score_eps;
        result.profile_beam_key_multi_diversity_eps = pack.profile_beam_key_multi_diversity_eps;
        result.profile_beam_score_policy = pack.profile_beam_score_policy;
        result.profile_beam_residual_weight = pack.profile_beam_residual_weight;
        result.profile_beam_residual_mean_penalty = pack.profile_beam_residual_mean_penalty;
        result.profile_beam_residual_max_penalty = pack.profile_beam_residual_max_penalty;
        result.profile_beam_late_frac = pack.profile_beam_late_frac;
        result.profile_realization_hardest_first = pack.profile_realization_hardest_first;
        result.profile_realization_exact_suffix_prune = pack.profile_realization_exact_suffix_prune;
        result.t_pack_profile_beam = pack.t_pack_profile_beam;
        result.t_pack_block_dp_exact = pack.t_pack_block_dp_exact;
        result.profile_step2_ub = pack.profile_step2_ub;
        result.profile_beam_candidate_ub = pack.profile_beam_candidate_ub;
        result.profile_beam_plus_candidate_ub = pack.profile_beam_plus_candidate_ub;
        result.profile_exact_candidate_ub = pack.profile_exact_candidate_ub;
        result.profile_beam_improved_over_step2 = pack.profile_beam_improved_over_step2;
        result.profile_exact_improved_over_step2 = pack.profile_exact_improved_over_step2;
        result.profile_incumbent_source = pack.profile_incumbent_source;
        result.profile_incumbent_ub_for_exact = pack.profile_incumbent_ub_for_exact;
        result.profile_selector_policy = pack.profile_selector_policy;
        result.profile_selector_decision = pack.profile_selector_decision;
        result.profile_selector_reason = pack.profile_selector_reason;
        result.profile_selector_has_one = pack.profile_selector_has_one;
        result.profile_selector_contiguous = pack.profile_selector_contiguous;
        result.profile_selector_multiplicity = pack.profile_selector_multiplicity;
        result.profile_selector_semigroup_density = pack.profile_selector_semigroup_density;
        result.profile_selector_hard_alarm = pack.profile_selector_hard_alarm;
        result.profile_exact_primary_fallback_to_beam = pack.profile_exact_primary_fallback_to_beam;
        result.profile_exact_primary_status_before_fallback = pack.profile_exact_primary_status_before_fallback;
        result.profile_step3_incumbent_mode = pack.profile_step3_incumbent_mode;
        result.dense_unit_fastpath_active = pack.dense_unit_fastpath_active;
        result.count_based_ffd_active = pack.count_based_ffd_active;
        result.dense_unit_relax_fastpath_active = 0;
        result.dense_unit_energy_profile_active = 0;
        result.dense_unit_relax_fastpath_fallback = 0;
        result.dense_unit_energy_profile_fallback = 0;
        result.dense_unit_relax_mode = "none";
        result.ec_generated_patterns_total = pack.ec_generated_patterns_total;
        result.ec_generated_patterns_max_block = pack.ec_generated_patterns_max_block;
        result.ec_retained_patterns_total = pack.ec_retained_patterns_total;
        result.ec_retained_patterns_max_block = pack.ec_retained_patterns_max_block;
        result.ec_retained_patterns_signature = pack.ec_retained_patterns_signature;
        result.ec_time_completion = pack.ec_time_completion;
        result.ec_time_pattern_generation = pack.ec_time_pattern_generation;
        result.ec_time_exact_core = pack.ec_time_exact_core;
        result.ec_pruned_core_window = pack.ec_pruned_core_window;
        result.ec_pruned_suffix = pack.ec_pruned_suffix;
        result.ec_pruned_transition = pack.ec_pruned_transition;
        result.ec_pruned_bound = pack.ec_pruned_bound;
        result.ec_delta_used = pack.ec_delta_used;
        result.ec_fixed_blocks = pack.ec_fixed_blocks;
        result.ec_two_phase_used = pack.ec_two_phase_used;
        result.ec_phase1_feasible_ub = pack.ec_phase1_feasible_ub;
        result.ec_time_phase1 = pack.ec_time_phase1;
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
                if (totals[i] <= 0)
                    continue;
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
        g_last_exact_dp_diag = ExactDPDiagnostics{};
        g_last_exact_dp_diag.mode = "dense";
        const std::string exact_variant = exact_dp_variant_name();
        g_last_exact_dp_diag.variant = exact_variant;
        const bool use_type_aware_lb = exact_dp_type_aware_lb_enabled(exact_variant);
        g_last_exact_dp_diag.initial_ub = known_ub;
        g_last_exact_dp_diag.corridor_enabled = g_exact_corridor.enabled ? 1 : 0;
        g_last_exact_dp_diag.corridor_delta = g_exact_corridor.delta;
        auto mark_dense_non_exhaustive = [&](double final_ub, const std::string &mode)
        {
            g_last_exact_dp_diag.mode = mode;
            g_last_exact_dp_diag.final_ub = final_ub;
            g_last_exact_dp_diag.elapsed_sec =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_exact).count();
            g_last_exact_dp_diag.exhaustive = 0;
        };
        int K = static_cast<int>(lengths.size());
        if (K == 0)
        {
            g_last_exact_dp_diag.final_ub = 0.0;
            g_last_exact_dp_diag.exhaustive = 1;
            return 0.0;
        }

        // Compute strides (mixed-radix encoding) and total state count
        std::vector<int> strides(K);
        int NC = 1;
        for (int i = 0; i < K; ++i)
        {
            strides[i] = NC;
            if (static_cast<int64_t>(NC) * (totals[i] + 1) > 500'000)
            {
                mark_dense_non_exhaustive(known_ub, "dense_skip_state_space");
                return kInf; // state space too large
            }
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
        {
            mark_dense_non_exhaustive(known_ub, "dense_skip_memory");
            return kInf;
        }

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
        int64_t dense_states_reached = 0;
        int64_t dense_states_expanded = 0;
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

        std::vector<std::vector<double>> min_job_cost;
        if (use_type_aware_lb)
        {
            min_job_cost.assign(K, std::vector<double>(T + 2, kInf));
            for (int j = 0; j < K; ++j)
            {
                int L = lengths[j];
                for (int t = T; t >= 0; --t)
                {
                    double best_here = min_job_cost[j][t + 1];
                    if (t + L <= T)
                    {
                        double c = prefix_proc[t + L] - prefix_proc[t];
                        if (c < best_here)
                            best_here = c;
                    }
                    min_job_cost[j][t] = best_here;
                }
            }
        }

        auto lb_type_aware_proc = [&](int t, const int *new_counts) -> double
        {
            if (!use_type_aware_lb)
                return 0.0;
            if (t < 0)
                t = 0;
            if (t > T + 1)
                t = T + 1;
            double lb = 0.0;
            for (int j = 0; j < K; ++j)
            {
                int rem_j = totals[j] - new_counts[j];
                if (rem_j <= 0)
                    continue;
                double c = min_job_cost[j][t];
                if (!(c < kInf * 0.5))
                    return kInf;
                lb += static_cast<double>(rem_j) * c;
            }
            return lb;
        };

        double best = known_ub;

        // Seed: first job from startup
        for (int t_s = spaces.early; t_s <= spaces.late; ++t_s)
        {
            double startup = spaces.c_start[t_s];
            if (startup >= kInf)
                continue;
            for (int i = 0; i < K; ++i)
            {
                if (totals[i] <= 0)
                    continue;
                int L = lengths[i];
                int t_e = t_s + L;
                if (t_e > T || t_e > spaces.late + 1)
                    continue;
                double cost = startup + (prefix_proc[t_e] - prefix_proc[t_s]);
                int new_s = strides[i]; // c_i goes from 0 to 1
                int new_rw = state_rw[new_s];
                int earliest_end = std::min(t_e + new_rw, T + 1);
                double lb_base = cost + lb_proc_cost(t_e, new_rw) + min_c_end_from[earliest_end];
                double lb = lb_base;
                if (use_type_aware_lb)
                {
                    const int *new_counts = &state_counts[static_cast<size_t>(new_s) * K];
                    double lb_ta = cost + lb_type_aware_proc(t_e, new_counts) + min_c_end_from[earliest_end];
                    lb = std::max(lb, lb_ta);
                    if (lb_ta > best + kEps && lb_base <= best + kEps)
                        g_last_exact_dp_diag.pruned_type_aware += 1.0;
                }
                if (lb > best + kEps)
                    continue;
                // PLAN24: corridor check on new state
                if (g_exact_corridor.enabled)
                {
                    const int *new_counts = &state_counts[static_cast<size_t>(new_s) * K];
                    int placed_work = total_rw - new_rw;
                    if (!check_exact_corridor_counts(new_counts, placed_work, K))
                        continue;
                }
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
                {
                    g_last_exact_dp_diag.timed_out = 1;
                    g_last_exact_dp_diag.final_ub = best;
                    g_last_exact_dp_diag.elapsed_sec = elapsed;
                    g_last_exact_dp_diag.states_reached = static_cast<double>(dense_states_reached);
                    g_last_exact_dp_diag.states_expanded = static_cast<double>(dense_states_expanded);
                    g_last_exact_dp_diag.exhaustive = 0;
                    return kInf; // timed out: cannot certify optimality
                }
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
                        double lb_base = cost + lb_proc_cost(t_e, new_rw) + min_c_end_from[earliest_end];
                        double lb = lb_base;
                        if (use_type_aware_lb)
                        {
                            const int *new_counts = &state_counts[static_cast<size_t>(new_s) * K];
                            double lb_ta = cost + lb_type_aware_proc(t_e, new_counts) + min_c_end_from[earliest_end];
                            lb = std::max(lb, lb_ta);
                            if (lb_ta > best + kEps && lb_base <= best + kEps)
                                g_last_exact_dp_diag.pruned_type_aware += 1.0;
                        }
                if (lb > best + kEps)
                {
                    g_last_exact_dp_diag.pruned_bound += 1.0;
                    continue;
                }
                // PLAN24: corridor check on new state
                if (g_exact_corridor.enabled)
                {
                    const int *new_counts = &state_counts[static_cast<size_t>(new_s) * K];
                    int placed_work = total_rw - new_rw;
                    if (!check_exact_corridor_counts(new_counts, placed_work, K))
                        continue;
                }
                auto di = idx(t_e, new_s);
                if (cost < dp[di])
                {
                    if (dp[di] >= kInf)
                        ++dense_states_reached;
                    dp[di] = cost;
                }
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
                ++dense_states_expanded;

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
                        double lb_base = cost + lb_proc_cost(t_e, new_rw) + min_c_end_from[earliest_end];
                        double lb = lb_base;
                        if (use_type_aware_lb)
                        {
                            const int *new_counts = &state_counts[static_cast<size_t>(new_s) * K];
                            double lb_ta = cost + lb_type_aware_proc(t_e, new_counts) + min_c_end_from[earliest_end];
                            lb = std::max(lb, lb_ta);
                            if (lb_ta > best + kEps && lb_base <= best + kEps)
                                g_last_exact_dp_diag.pruned_type_aware += 1.0;
                        }
                        if (lb > best + kEps)
                        {
                            g_last_exact_dp_diag.pruned_bound += 1.0;
                            continue;
                        }
                        // PLAN24: corridor check on new state
                        if (g_exact_corridor.enabled)
                        {
                            const int *new_counts = &state_counts[static_cast<size_t>(new_s) * K];
                            int placed_work = total_rw - new_rw;
                            if (!check_exact_corridor_counts(new_counts, placed_work, K))
                                continue;
                        }
                        auto di = idx(t_e, new_s);
                        if (cost < dp[di])
                        {
                            if (dp[di] >= kInf)
                                ++dense_states_reached;
                            dp[di] = cost;
                        }
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

        g_last_exact_dp_diag.final_ub = best;
        g_last_exact_dp_diag.elapsed_sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_exact).count();
        g_last_exact_dp_diag.states_reached = static_cast<double>(dense_states_reached);
        g_last_exact_dp_diag.states_expanded = static_cast<double>(dense_states_expanded);
        g_last_exact_dp_diag.exhaustive = 1;
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
        double relaxed_lb,
        const std::vector<double> *completion_dp,
        int completion_RW,
        int completion_rw_scale)
    {
        auto t0 = std::chrono::steady_clock::now();
        g_last_exact_dp_diag = ExactDPDiagnostics{};
        g_last_exact_dp_diag.mode = "sparse";
        const std::string exact_variant = exact_dp_variant_name();
        g_last_exact_dp_diag.variant = exact_variant;
        const bool use_type_aware_lb = exact_dp_type_aware_lb_enabled(exact_variant);
        const bool use_incumbent_ordering = exact_dp_incumbent_ordering_enabled(exact_variant);
        const int ordering_min_layer = std::max(2, env_int_or("PAST_EXACT_DP_ORDERING_MIN_LAYER", 2));
        g_last_exact_dp_diag.initial_ub = known_ub;
        g_last_exact_dp_diag.corridor_enabled = g_exact_corridor.enabled ? 1 : 0;
        g_last_exact_dp_diag.corridor_delta = g_exact_corridor.delta;
        auto mark_sparse_non_exhaustive = [&](double final_ub, const std::string &mode)
        {
            g_last_exact_dp_diag.mode = mode;
            g_last_exact_dp_diag.final_ub = final_ub;
            g_last_exact_dp_diag.elapsed_sec =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            g_last_exact_dp_diag.exhaustive = 0;
        };
        int K = static_cast<int>(lengths.size());
        if (K == 0)
        {
            g_last_exact_dp_diag.final_ub = 0.0;
            g_last_exact_dp_diag.exhaustive = 1;
            return 0.0;
        }

        // Compute strides (mixed-radix encoding) for state indexing
        std::vector<int64_t> strides(K);
        int64_t NC = 1;
        int64_t max_theoretical_states = env_int64_or("PAST_SPARSE_EXACT_MAX_THEORETICAL", 1000000000LL);
        // PLAN24B: corridor force-entry bypass
        bool corridor_force_entry = (env_int_or("PAST_EXACT_CORRIDOR_FORCE_ENTRY", 0) != 0);
        bool corridor_active = g_exact_corridor.enabled;
        bool force_entry = corridor_force_entry && corridor_active;
        if (force_entry)
        {
            max_theoretical_states = std::numeric_limits<int64_t>::max();
            // Clamp the exact DP time budget to the corridor time limit
            int corridor_tlim = env_int_or("PAST_EXACT_CORRIDOR_TIME_LIMIT", 300);
            if (corridor_tlim > 0 && corridor_tlim < time_limit_sec)
                time_limit_sec = corridor_tlim;
        }
        for (int i = 0; i < K; ++i)
        {
            strides[i] = NC;
            if (totals[i] < 0)
            {
                mark_sparse_non_exhaustive(known_ub, "sparse_invalid_totals");
                if (force_entry) g_last_exact_dp_diag.stop_reason = "invalid_totals";
                return kInf;
            }
            if (NC > std::numeric_limits<int64_t>::max() / static_cast<int64_t>(totals[i] + 1))
            {
                mark_sparse_non_exhaustive(known_ub, "sparse_skip_overflow");
                if (force_entry) g_last_exact_dp_diag.stop_reason = "overflow";
                return kInf; // encoding overflow
            }
            NC *= (totals[i] + 1);
            if (NC > max_theoretical_states)
            {
                if (force_entry)
                {
                    // Bypass: theoretical NC is huge, but we gate on reachable states instead
                    g_last_exact_dp_diag.stop_reason = "force_entry_active";
                    break; // allow entry, guardrail on actual states reached later
                }
                mark_sparse_non_exhaustive(known_ub, "sparse_skip_theoretical");
                g_last_exact_dp_diag.stop_reason = "sparse_skip_theoretical";
                return kInf; // guardrail on theoretical lattice size
            }
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
        auto completion_cost = [&](int t, int rw) -> double
        {
            if (!completion_dp)
                return 0.0;
            if (rw < 0)
                return kInf;
            int scaled_rw = rw;
            if (completion_rw_scale > 1)
                scaled_rw = rw / completion_rw_scale;
            if (t < 0 || t > T + 1 || scaled_rw < 0 || scaled_rw >= completion_RW)
                return kInf;
            return (*completion_dp)[static_cast<size_t>(t) * completion_RW + scaled_rw];
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

        std::vector<std::vector<double>> min_job_cost;
        if (use_type_aware_lb)
        {
            min_job_cost.assign(K, std::vector<double>(T + 2, kInf));
            for (int j = 0; j < K; ++j)
            {
                int L = lengths[j];
                for (int t = T; t >= 0; --t)
                {
                    double best_here = min_job_cost[j][t + 1];
                    if (t + L <= T)
                    {
                        double c = prefix_proc[t + L] - prefix_proc[t];
                        if (c < best_here)
                            best_here = c;
                    }
                    min_job_cost[j][t] = best_here;
                }
            }
        }

        auto lb_type_aware_proc = [&](int t, int64_t state_key) -> double
        {
            if (!use_type_aware_lb)
                return 0.0;
            if (t < 0)
                t = 0;
            if (t > T + 1)
                t = T + 1;
            int64_t tmp = state_key;
            double lb = 0.0;
            for (int j = 0; j < K; ++j)
            {
                int used_j = static_cast<int>(tmp % (totals[j] + 1));
                tmp /= (totals[j] + 1);
                int rem_j = totals[j] - used_j;
                if (rem_j <= 0)
                    continue;
                double c = min_job_cost[j][t];
                if (!(c < kInf * 0.5))
                    return kInf;
                lb += static_cast<double>(rem_j) * c;
            }
            return lb;
        };

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

        auto relax_insert = [&](std::unordered_map<int64_t, double> &m, int64_t st, double cost)
        {
            auto it = m.find(st);
            if (it == m.end())
            {
                m[st] = cost;
                ++total_entries;
                g_last_exact_dp_diag.states_reached += 1.0;
                return;
            }
            if (cost + kEps < it->second)
            {
                it->second = cost;
                return;
            }
            g_last_exact_dp_diag.pruned_dominance += 1.0;
        };

        // Seed: first job from startup
        for (int t_s = spaces.early; t_s <= spaces.late; ++t_s)
        {
            double startup = spaces.c_start[t_s];
            if (startup >= kInf)
                continue;
            for (int i = 0; i < K; ++i)
            {
                if (totals[i] <= 0)
                    continue;
                int L = lengths[i];
                int t_e = t_s + L;
                if (t_e > T || t_e > spaces.late + 1)
                    continue;
                double cost = startup + (prefix_proc[t_e] - prefix_proc[t_s]);
                int64_t new_s = strides[i];
                int new_rw = state_rw(new_s);
                int earliest_end = std::min(t_e + new_rw, T + 1);
                double lb_base = cost + lb_proc_cost(new_rw) + min_c_end_from[earliest_end];
                double lb = lb_base;
                if (completion_dp)
                {
                    double lb_completion = cost + completion_cost(t_e, new_rw);
                    lb = std::max(lb, lb_completion);
                    if (lb_completion > best + kEps && lb_base <= best + kEps)
                        g_last_exact_dp_diag.pruned_completion += 1.0;
                }
                if (use_type_aware_lb)
                {
                    double lb_before_type = lb;
                    double lb_type = cost + lb_type_aware_proc(t_e, new_s) + min_c_end_from[earliest_end];
                    lb = std::max(lb, lb_type);
                    if (lb_type > best + kEps && lb_before_type <= best + kEps)
                        g_last_exact_dp_diag.pruned_type_aware += 1.0;
                }
                if (lb > best + kEps)
                {
                    g_last_exact_dp_diag.pruned_bound += 1.0;
                    continue;
                }
                if (relaxed_dp)
                {
                    double rdp_val = relaxed_cost(t_e, new_rw);
                    if (rdp_val >= kInf)
                    {
                        g_last_exact_dp_diag.pruned_relaxed += 1.0;
                        continue;
                    }
                    if (relaxed_lb < kInf && cost - rdp_val > best - relaxed_lb + kEps)
                    {
                        g_last_exact_dp_diag.pruned_relaxed += 1.0;
                        continue;
                    }
                }
                // PLAN24: corridor check on new state
                if (!check_exact_corridor_sparse(new_s, total_rw, totals, lengths, K))
                    continue;
                auto &m = dp_maps[t_e];
                relax_insert(m, new_s, cost);
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
                if (elapsed >= time_limit_sec)
                {
                    exhaustive = false;
                    g_last_exact_dp_diag.timed_out = 1;
                    if (force_entry) g_last_exact_dp_diag.stop_reason = "timeout";
                    break;
                }
                int64_t corridor_max_states = force_entry ? env_int64_or("PAST_EXACT_CORRIDOR_MAX_STATES", 50'000'000LL) : MAX_TOTAL_ENTRIES;
                if (total_entries > corridor_max_states)
                {
                    exhaustive = false;
                    if (force_entry) g_last_exact_dp_diag.stop_reason = "states_exceeded";
                    break;
                }
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
                        double lb_base = cost + lb_proc_cost(new_rw) + min_c_end_from[earliest_end];
                        double lb = lb_base;
                        if (completion_dp)
                        {
                            double lb_completion = cost + completion_cost(t_e, new_rw);
                            lb = std::max(lb, lb_completion);
                            if (lb_completion > best + kEps && lb_base <= best + kEps)
                                g_last_exact_dp_diag.pruned_completion += 1.0;
                        }
                        if (use_type_aware_lb)
                        {
                            double lb_before_type = lb;
                            double lb_type = cost + lb_type_aware_proc(t_e, new_s) + min_c_end_from[earliest_end];
                            lb = std::max(lb, lb_type);
                            if (lb_type > best + kEps && lb_before_type <= best + kEps)
                                g_last_exact_dp_diag.pruned_type_aware += 1.0;
                        }
                        if (lb > best + kEps)
                        {
                            g_last_exact_dp_diag.pruned_bound += 1.0;
                            continue;
                        }
                        if (relaxed_dp)
                        {
                            double rdp_val = relaxed_cost(t_e, new_rw);
                            if (rdp_val >= kInf)
                            {
                                g_last_exact_dp_diag.pruned_relaxed += 1.0;
                                continue;
                            }
                            if (relaxed_lb < kInf && cost - rdp_val > best - relaxed_lb + kEps)
                            {
                                g_last_exact_dp_diag.pruned_relaxed += 1.0;
                                continue;
                            }
                        }
                        // PLAN24: corridor check on new state
                        if (!check_exact_corridor_sparse(new_s, total_rw, totals, lengths, K))
                            continue;
                        auto &m = dp_maps[t_e];
                        relax_insert(m, new_s, cost);
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
            std::vector<std::pair<int64_t, double>> ordered_states;
            ordered_states.reserve(cur_map.size());
            for (auto &kv : cur_map)
                ordered_states.push_back(kv);
            bool slack_tiebreak =
                use_incumbent_ordering &&
                static_cast<int>(ordered_states.size()) >= ordering_min_layer;
            std::sort(ordered_states.begin(), ordered_states.end(), [&](const auto &a, const auto &b)
                      {
                          int rwa = state_rw(a.first);
                          int rwb = state_rw(b.first);
                          int ea = std::min(t_end + std::max(0, rwa), T + 1);
                          int eb = std::min(t_end + std::max(0, rwb), T + 1);
                          double lba = a.second + lb_proc_cost(rwa) + min_c_end_from[ea];
                          double lbb = b.second + lb_proc_cost(rwb) + min_c_end_from[eb];
                          if (completion_dp)
                          {
                              lba = std::max(lba, a.second + completion_cost(t_end, rwa));
                              lbb = std::max(lbb, b.second + completion_cost(t_end, rwb));
                          }
                          if (use_type_aware_lb)
                          {
                              lba = std::max(lba, a.second + lb_type_aware_proc(t_end, a.first) + min_c_end_from[ea]);
                              lbb = std::max(lbb, b.second + lb_type_aware_proc(t_end, b.first) + min_c_end_from[eb]);
                          }
                          if (std::abs(lba - lbb) > kEps)
                              return lba < lbb;
                          if (slack_tiebreak && best < kInf * 0.5)
                          {
                              double sa = best - lba;
                              double sb = best - lbb;
                              if (std::abs(sa - sb) > kEps)
                                  return sa < sb;
                          }
                          return a.second < b.second;
                      });

            for (const auto &[s, sv] : ordered_states)
            {
                if (state_rw(s) <= 0)
                    continue;
                if (sv >= kInf)
                    continue;
                g_last_exact_dp_diag.states_expanded += 1.0;

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
                        double lb_base = cost + lb_proc_cost(new_rw) + min_c_end_from[earliest_end];
                        double lb = lb_base;
                        if (completion_dp)
                        {
                            double lb_completion = cost + completion_cost(t_e, new_rw);
                            lb = std::max(lb, lb_completion);
                            if (lb_completion > best + kEps && lb_base <= best + kEps)
                                g_last_exact_dp_diag.pruned_completion += 1.0;
                        }
                        if (use_type_aware_lb)
                        {
                            double lb_before_type = lb;
                            double lb_type = cost + lb_type_aware_proc(t_e, new_s) + min_c_end_from[earliest_end];
                            lb = std::max(lb, lb_type);
                            if (lb_type > best + kEps && lb_before_type <= best + kEps)
                                g_last_exact_dp_diag.pruned_type_aware += 1.0;
                        }
                        if (lb > best + kEps)
                        {
                            g_last_exact_dp_diag.pruned_bound += 1.0;
                            continue;
                        }
                        if (relaxed_dp)
                        {
                            double rdp_val = relaxed_cost(t_e, new_rw);
                            if (rdp_val >= kInf)
                            {
                                g_last_exact_dp_diag.pruned_relaxed += 1.0;
                                continue;
                            }
                            if (relaxed_lb < kInf && cost - rdp_val > best - relaxed_lb + kEps)
                            {
                                g_last_exact_dp_diag.pruned_relaxed += 1.0;
                                continue;
                            }
                        }
                        // PLAN24: corridor check on new state
                        if (!check_exact_corridor_sparse(new_s, total_rw, totals, lengths, K))
                            continue;
                        auto &m = dp_maps[t_e];
                        relax_insert(m, new_s, cost);
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

        g_last_exact_dp_diag.final_ub = best;
        g_last_exact_dp_diag.elapsed_sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        g_last_exact_dp_diag.exhaustive = exhaustive ? 1 : 0;
        if (force_entry && exhaustive)
            g_last_exact_dp_diag.stop_reason = "exhaustive";
        else if (force_entry && g_last_exact_dp_diag.stop_reason == "none")
            g_last_exact_dp_diag.stop_reason = "force_entry_active";
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
            auto t0_profile = std::chrono::steady_clock::now();
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
            double t_profile =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_profile).count();
            pack = pack_recovered_blocks(blocks, lengths, totals, prefix_proc, T, spaces);
            pack.t_pack_profile_recovery = t_profile;
            pack.pack_profiles_tried = 1;
            pack.pack_co_optimal_profiles = 1;
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
        result.merged_blocks = pack.merged_blocks;
        result.pack_solver = pack.pack_solver;
        result.pack_external_status = pack.pack_external_status;
        result.pack_method = pack.pack_method;
        result.pack_outcome = pack.pack_outcome;
        result.t_pack_external = pack.t_pack_external;
        result.t_pack_heuristic = pack.t_pack_heuristic;
        result.t_pack_dfs = pack.t_pack_dfs;
        result.t_pack_block_dp = pack.t_pack_block_dp;
        result.t_pack_profile_recovery = pack.t_pack_profile_recovery;
        result.t_pack_merge_blocks = pack.t_pack_merge_blocks;
        result.t_pack_to_first_candidate = pack.t_pack_to_first_candidate;
        result.t_pack_ffd_only = pack.t_pack_ffd_only;
        result.step2_reached = pack.step2_reached;
        result.step2_produced_ub = pack.step2_produced_ub;
        result.t_dense_spaces_or_lb = 0.0;
        result.t_dense_profile_dp = 0.0;
        result.t_dense_profile_recovery = 0.0;
        result.t_dense_block_build = pack.t_pack_merge_blocks;
        result.t_dense_job_materialization = std::max(0.0, pack.t_pack_heuristic - pack.t_pack_ffd_only);
        result.t_dense_step2_pack = pack.t_pack_ffd_only;
        result.t_dense_pre_step2_total =
            result.t_dense_spaces_or_lb + result.t_dense_profile_dp + result.t_dense_profile_recovery +
            result.t_dense_block_build + result.t_dense_job_materialization + result.t_dense_step2_pack;
        result.pack_profiles_tried = pack.pack_profiles_tried;
        result.pack_co_optimal_profiles = pack.pack_co_optimal_profiles;
        result.block_dp_state_space = pack.block_dp_state_space;
        result.block_dp_total_compositions = pack.block_dp_total_compositions;
        result.block_dp_total_comp_estimate = pack.block_dp_total_comp_estimate;
        result.block_dp_max_comp_estimate = pack.block_dp_max_comp_estimate;
        result.block_dp_max_compositions_per_block = pack.block_dp_max_compositions_per_block;
        result.block_dp_status = pack.block_dp_status;
        result.block_dp_timed_out = pack.block_dp_timed_out;
        result.beam_ub_for_exact_l2 = pack.beam_ub_for_exact_l2;
        result.exact_l2_ub = pack.exact_l2_ub;
        result.t_exact_l2 = pack.t_exact_l2;
        result.exact_l2_nodes = pack.exact_l2_nodes;
        result.exact_l2_closed = pack.exact_l2_closed;
        result.exact_l2_improved_over_beam = pack.exact_l2_improved_over_beam;
        result.exact_l2_beam_optimal_in_pool = pack.exact_l2_beam_optimal_in_pool;
        result.exact_l2_status = pack.exact_l2_status;
        result.profile_beam_base_width = pack.profile_beam_base_width;
        result.profile_beam_avg_width = pack.profile_beam_avg_width;
        result.profile_beam_max_width = pack.profile_beam_max_width;
        result.profile_beam_states_considered = pack.profile_beam_states_considered;
        result.profile_beam_states_kept = pack.profile_beam_states_kept;
        result.profile_beam_pruned_over = pack.profile_beam_pruned_over;
        result.profile_beam_pruned_suffix = pack.profile_beam_pruned_suffix;
        result.profile_beam_pruned_discrepancy = pack.profile_beam_pruned_discrepancy;
        result.profile_beam_discrepancy_budget = pack.profile_beam_discrepancy_budget;
        result.profile_beam_discrepancy_depth = pack.profile_beam_discrepancy_depth;
        result.profile_beam_status = pack.profile_beam_status;
        result.profile_beam_timed_out = pack.profile_beam_timed_out;
        result.profile_beam_key_multi_policy = pack.profile_beam_key_multi_policy;
        result.profile_beam_key_multi_max = pack.profile_beam_key_multi_max;
        result.profile_beam_key_multi_score_eps = pack.profile_beam_key_multi_score_eps;
        result.profile_beam_key_multi_diversity_eps = pack.profile_beam_key_multi_diversity_eps;
        result.profile_beam_score_policy = pack.profile_beam_score_policy;
        result.profile_beam_residual_weight = pack.profile_beam_residual_weight;
        result.profile_beam_residual_mean_penalty = pack.profile_beam_residual_mean_penalty;
        result.profile_beam_residual_max_penalty = pack.profile_beam_residual_max_penalty;
        result.profile_beam_late_frac = pack.profile_beam_late_frac;
        result.profile_realization_hardest_first = pack.profile_realization_hardest_first;
        result.profile_realization_exact_suffix_prune = pack.profile_realization_exact_suffix_prune;
        result.t_pack_profile_beam = pack.t_pack_profile_beam;
        result.t_pack_block_dp_exact = pack.t_pack_block_dp_exact;
        result.profile_step2_ub = pack.profile_step2_ub;
        result.profile_beam_candidate_ub = pack.profile_beam_candidate_ub;
        result.profile_beam_plus_candidate_ub = pack.profile_beam_plus_candidate_ub;
        result.profile_exact_candidate_ub = pack.profile_exact_candidate_ub;
        result.profile_beam_improved_over_step2 = pack.profile_beam_improved_over_step2;
        result.profile_exact_improved_over_step2 = pack.profile_exact_improved_over_step2;
        result.profile_incumbent_source = pack.profile_incumbent_source;
        result.profile_incumbent_ub_for_exact = pack.profile_incumbent_ub_for_exact;
        result.profile_selector_policy = pack.profile_selector_policy;
        result.profile_selector_decision = pack.profile_selector_decision;
        result.profile_selector_reason = pack.profile_selector_reason;
        result.profile_selector_has_one = pack.profile_selector_has_one;
        result.profile_selector_contiguous = pack.profile_selector_contiguous;
        result.profile_selector_multiplicity = pack.profile_selector_multiplicity;
        result.profile_selector_semigroup_density = pack.profile_selector_semigroup_density;
        result.profile_selector_hard_alarm = pack.profile_selector_hard_alarm;
        result.profile_exact_primary_fallback_to_beam = pack.profile_exact_primary_fallback_to_beam;
        result.profile_exact_primary_status_before_fallback = pack.profile_exact_primary_status_before_fallback;
        result.profile_step3_incumbent_mode = pack.profile_step3_incumbent_mode;
        result.dense_unit_fastpath_active = pack.dense_unit_fastpath_active;
        result.count_based_ffd_active = pack.count_based_ffd_active;
        result.dense_unit_relax_fastpath_active = 0;
        result.dense_unit_energy_profile_active = 0;
        result.dense_unit_relax_fastpath_fallback = 0;
        result.dense_unit_energy_profile_fallback = 0;
        result.dense_unit_relax_mode = "none";
        result.ec_generated_patterns_total = pack.ec_generated_patterns_total;
        result.ec_generated_patterns_max_block = pack.ec_generated_patterns_max_block;
        result.ec_retained_patterns_total = pack.ec_retained_patterns_total;
        result.ec_retained_patterns_max_block = pack.ec_retained_patterns_max_block;
        result.ec_retained_patterns_signature = pack.ec_retained_patterns_signature;
        result.ec_time_completion = pack.ec_time_completion;
        result.ec_time_pattern_generation = pack.ec_time_pattern_generation;
        result.ec_time_exact_core = pack.ec_time_exact_core;
        result.ec_pruned_core_window = pack.ec_pruned_core_window;
        result.ec_pruned_suffix = pack.ec_pruned_suffix;
        result.ec_pruned_transition = pack.ec_pruned_transition;
        result.ec_pruned_bound = pack.ec_pruned_bound;
        result.ec_delta_used = pack.ec_delta_used;
        result.ec_fixed_blocks = pack.ec_fixed_blocks;
        result.ec_two_phase_used = pack.ec_two_phase_used;
        result.ec_phase1_feasible_ub = pack.ec_phase1_feasible_ub;
        result.ec_time_phase1 = pack.ec_time_phase1;
        return result;
    }

    // =====================================================================
    //  PLAN25: beam-corridor local exact DP
    //  Uses local offset encoding around beam prefix trajectory.
    //  Avoids global mixed-radix int64 overflow.
    // =====================================================================

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
        LocalCorridorDiag &diag)
    {
        auto t0 = std::chrono::steady_clock::now();
        diag.enabled = 1;
        diag.delta = delta;
        diag.status = "running";

        int K = static_cast<int>(lengths.size());
        int B = static_cast<int>(merged_seg.size());
        if (B == 0 || K == 0 || beam_chosen_counts.empty())
        {
            diag.status = "no_beam_path";
            diag.stop_reason = "no_beam_path";
            diag.time_sec = 0.0;
            return kInf;
        }

        // Convert Segment to RecoveredBlock for existing helpers
        std::vector<RecoveredBlock> merged;
        merged.reserve(B);
        for (const auto &s : merged_seg)
            merged.push_back({s.start, s.length});

        // Build block local views
        std::vector<SPACESResult> block_spaces;
        std::vector<std::vector<double>> block_prefix_proc;
        build_profile_block_local_views(merged, prefix_proc, T, spaces, &block_spaces, &block_prefix_proc);

        // Compute prefix counts along beam trajectory
        std::vector<std::vector<int>> prefix_counts(B + 1, std::vector<int>(K, 0));
        for (int pos = 0; pos < B; ++pos)
        {
            int bi = (static_cast<int>(block_order.size()) == B) ? block_order[pos] : pos;
            for (int j = 0; j < K; ++j)
                prefix_counts[pos + 1][j] = prefix_counts[pos][j] + beam_chosen_counts[bi][j];
        }

        // Target final offset
        std::vector<int> target_offset(K, 0);
        for (int j = 0; j < K; ++j)
            target_offset[j] = totals[j] - prefix_counts[B][j];

        // Check if target is within corridor
        bool target_in_corridor = true;
        for (int j = 0; j < K; ++j)
        {
            if (target_offset[j] < -delta || target_offset[j] > delta)
            {
                target_in_corridor = false;
                break;
            }
        }
        // PLAN26: populate alignment diagnostics
        diag.beam_counts_size = static_cast<int>(beam_chosen_counts.size());
        diag.merged_blocks = B;
        diag.block_count_mismatch = (static_cast<int>(beam_chosen_counts.size()) != B) ? 1 : 0;
        int target_l1 = 0;
        for (int j = 0; j < K; ++j)
            target_l1 += std::abs(target_offset[j]);
        diag.target_offset_l1 = target_l1;
        diag.target_in_corridor = target_in_corridor ? 1 : 0;

        if (!target_in_corridor)
        {
            diag.status = "infeasible_corridor";
            diag.stop_reason = "infeasible_corridor";
            diag.time_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            return kInf;
        }

        // Generate candidate count vectors per block
        struct Candidate
        {
            std::vector<int> counts;
            double cost;
        };
        std::vector<std::vector<Candidate>> candidates(B);
        const int max_candidates_per_block = 50;
        const int l3_max_cells = 1000000;
        const double l3_time_limit = 0.05;

        for (int pos = 0; pos < B; ++pos)
        {
            int bi = (static_cast<int>(block_order.size()) == B) ? block_order[pos] : pos;
            const std::vector<int> &beam_c = beam_chosen_counts[bi];

            std::vector<Candidate> cands;
            auto add_candidate = [&](const std::vector<int> &c)
            {
                double cost = evaluate_profile_block_counts(
                    bi, c, lengths, merged, block_spaces, block_prefix_proc,
                    l3_max_cells, l3_time_limit);
                if (cost < kInf * 0.5)
                    cands.push_back({c, cost});
            };

            // Base: beam's counts
            add_candidate(beam_c);

            // Single perturbations
            for (int j = 0; j < K; ++j)
            {
                if (beam_c[j] > 0)
                {
                    std::vector<int> c = beam_c;
                    c[j] -= 1;
                    add_candidate(c);
                }
                {
                    std::vector<int> c = beam_c;
                    c[j] += 1;
                    add_candidate(c);
                }
            }

            // Pair perturbations: move one job between types
            for (int j1 = 0; j1 < K; ++j1)
            {
                for (int j2 = j1 + 1; j2 < K; ++j2)
                {
                    if (beam_c[j1] > 0)
                    {
                        std::vector<int> c = beam_c;
                        c[j1] -= 1; c[j2] += 1;
                        add_candidate(c);
                    }
                    if (beam_c[j2] > 0)
                    {
                        std::vector<int> c = beam_c;
                        c[j1] += 1; c[j2] -= 1;
                        add_candidate(c);
                    }
                }
            }

            // Sort by cost and keep top N
            std::sort(cands.begin(), cands.end(),
                      [](const Candidate &a, const Candidate &b)
                      { return a.cost < b.cost; });
            if ((int)cands.size() > max_candidates_per_block)
                cands.resize(max_candidates_per_block);
            candidates[pos] = std::move(cands);

            // PLAN26: track empty candidates and base-candidate finiteness
            if (candidates[pos].empty())
            {
                diag.empty_candidate_blocks += 1;
                if (diag.first_empty_layer < 0)
                    diag.first_empty_layer = pos;
            }
            else
            {
                // Check if base beam candidate survived (first candidate is beam_c)
                bool base_finite = false;
                for (const auto &c : candidates[pos])
                {
                    if (c.counts == beam_c)
                    {
                        base_finite = true;
                        break;
                    }
                }
                if (base_finite)
                    diag.base_candidates_finite += 1;
            }
        }

        // DP over layers
        int base_enc = 2 * delta + 1;
        auto encode_offset = [&](const std::vector<int> &offset) -> int
        {
            int id = 0;
            int mult = 1;
            for (int j = 0; j < K; ++j)
            {
                id += (offset[j] + delta) * mult;
                mult *= base_enc;
            }
            return id;
        };

        auto decode_offset = [&](int id) -> std::vector<int>
        {
            std::vector<int> offset(K);
            for (int j = 0; j < K; ++j)
            {
                offset[j] = (id % base_enc) - delta;
                id /= base_enc;
            }
            return offset;
        };

        int target_id = encode_offset(target_offset);

        // PLAN26: quick base-path survival simulation
        {
            std::vector<int> offset(K, 0);
            double cost = 0.0;
            bool survives = true;
            for (int pos = 0; pos < B; ++pos)
            {
                int bi = (static_cast<int>(block_order.size()) == B) ? block_order[pos] : pos;
                bool found_base = false;
                for (const auto &cand : candidates[pos])
                {
                    if (cand.counts == beam_chosen_counts[bi])
                    {
                        cost += cand.cost;
                        found_base = true;
                        break;
                    }
                }
                if (!found_base)
                {
                    diag.base_path_reject_reason = "base_candidate_not_found_at_layer_" + std::to_string(pos);
                    survives = false;
                    break;
                }
                for (int j = 0; j < K; ++j)
                {
                    int actual = prefix_counts[pos][j] + offset[j] + beam_chosen_counts[bi][j];
                    int next_off = actual - prefix_counts[pos + 1][j];
                    if (next_off < -delta || next_off > delta || actual < 0 || actual > totals[j])
                    {
                        diag.base_path_reject_reason = "base_offset_out_of_bounds_at_layer_" + std::to_string(pos);
                        survives = false;
                        break;
                    }
                }
                if (!survives)
                    break;
            }
            if (survives)
            {
                diag.base_path_survives = 1;
                diag.base_path_cost = cost;
            }
        }

        // Use unordered_map for sparse state representation
        std::unordered_map<int, double> prev, cur;
        prev.reserve(100000);
        cur.reserve(100000);
        std::vector<int> zero_offset(K, 0);
        prev[encode_offset(zero_offset)] = 0.0;

        int64_t total_states_seen = 0;
        int max_states_kept = 0;
        int64_t total_trans_considered = 0;
        int64_t total_trans_kept = 0;
        int64_t total_states_pruned = 0;
        double best_ub = kInf;

        int64_t max_total_states = env_int64_or("PAST_BEAM_CORRIDOR_LOCAL_MAX_STATES", 5000000LL);
        bool state_cap_hit = false;
        bool time_cap_hit = false;

        for (int pos = 0; pos < B; ++pos)
        {
            // Time check
            double elapsed = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - t0)
                                 .count();
            if (elapsed >= time_limit_sec)
            {
                time_cap_hit = true;
                break;
            }

            int bi = (static_cast<int>(block_order.size()) == B) ? block_order[pos] : pos;
            const auto &cands = candidates[pos];
            int n_prev = static_cast<int>(prev.size());
            total_states_seen += n_prev;
            max_states_kept = std::max(max_states_kept, n_prev);

            for (const auto &[state_id, cost] : prev)
            {
                auto offset = decode_offset(state_id);
                for (const auto &cand : cands)
                {
                    total_trans_considered++;
                    bool valid = true;
                    std::vector<int> next_offset(K);
                    for (int j = 0; j < K; ++j)
                    {
                        int actual = prefix_counts[pos][j] + offset[j] + cand.counts[j];
                        next_offset[j] = actual - prefix_counts[pos + 1][j];
                        if (next_offset[j] < -delta || next_offset[j] > delta)
                        {
                            valid = false;
                            total_states_pruned++;
                            break;
                        }
                        if (actual < 0 || actual > totals[j])
                        {
                            valid = false;
                            total_states_pruned++;
                            break;
                        }
                    }
                    if (!valid)
                        continue;

                    double new_cost = cost + cand.cost;
                    if (new_cost >= best_ub)
                        continue;

                    int next_id = encode_offset(next_offset);
                    auto it = cur.find(next_id);
                    if (it == cur.end() || new_cost < it->second)
                    {
                        if (it == cur.end())
                            total_trans_kept++;
                        cur[next_id] = new_cost;
                    }
                }
            }

            // Hard state cap
            if ((int64_t)cur.size() > max_total_states)
            {
                state_cap_hit = true;
                // Keep only the best states
                std::vector<std::pair<int, double>> vec(cur.begin(), cur.end());
                std::partial_sort(vec.begin(), vec.begin() + max_total_states, vec.end(),
                                  [](const auto &a, const auto &b)
                                  { return a.second < b.second; });
                vec.resize(max_total_states);
                cur.clear();
                for (auto &p : vec)
                    cur[p.first] = p.second;
            }

            prev.swap(cur);
            cur.clear();

            if (prev.empty())
            {
                diag.status = "infeasible_corridor";
                diag.stop_reason = "infeasible_corridor";
                diag.time_sec = elapsed;
                diag.states_seen = total_states_seen;
                diag.states_kept_max = max_states_kept;
                diag.states_pruned = total_states_pruned;
                diag.transitions_considered = total_trans_considered;
                diag.transitions_kept = total_trans_kept;
                return kInf;
            }
        }

        double elapsed = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t0)
                             .count();

        auto it = prev.find(target_id);
        if (it != prev.end())
        {
            best_ub = it->second;
            diag.status = "feasible";
            diag.stop_reason = time_cap_hit ? "time_limit" : (state_cap_hit ? "state_cap" : "feasible");
            diag.closed = 1;
        }
        else
        {
            diag.status = "infeasible_corridor";
            diag.stop_reason = "infeasible_corridor";
        }

        diag.time_sec = elapsed;
        diag.states_seen = total_states_seen;
        diag.states_kept_max = max_states_kept;
        diag.states_pruned = total_states_pruned;
        diag.transitions_considered = total_trans_considered;
        diag.transitions_kept = total_trans_kept;
        diag.best_ub = best_ub;
        diag.layers = B;
        diag.memory_safe = 1;

        return best_ub;
    }

} // namespace dp

#include "stateful_dp_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace dp
{

    namespace
    {

        struct StatefulEntry
        {
            double cost;
            int64_t pen;
            int rw;
            int jd;
        };

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
        std::vector<int> radices(K), inc(K);
        int64_t mult = 1;
        int64_t final_state = 0;
        int total_rw = 0;
        int max_job_len = 1;
        for (int i = 0; i < K; ++i)
        {
            radices[i] = totals[i] + 1;
            inc[i] = static_cast<int>(mult);
            final_state += static_cast<int64_t>(totals[i]) * mult;
            mult *= radices[i];
            total_rw += totals[i] * lengths[i];
            max_job_len = std::max(max_job_len, lengths[i]);
        }

        auto elapsed_sec = [&]() -> double
        {
            return std::chrono::duration<double>(Clock::now() - t0).count();
        };

        constexpr int LB_BLOCK = 20;
        std::unordered_map<int, std::vector<double>> lb_blocks;
        std::vector<double> proc_prices(T, 0.0);
        for (int i = 0; i < T; ++i)
            proc_prices[i] = prefix_proc[i + 1] - prefix_proc[i];
        for (int b = 0; b <= T; b += LB_BLOCK)
        {
            if (b < T)
            {
                std::vector<double> sp(proc_prices.begin() + b, proc_prices.end());
                std::sort(sp.begin(), sp.end());
                std::vector<double> cs(sp.size() + 1, 0.0);
                for (std::size_t i = 0; i < sp.size(); ++i)
                    cs[i + 1] = cs[i] + sp[i];
                lb_blocks.emplace(b, std::move(cs));
            }
            else
            {
                lb_blocks.emplace(b, std::vector<double>{0.0});
            }
        }

        auto lb_proc_cost = [&](int t, int rw) -> double
        {
            int b = (t / LB_BLOCK) * LB_BLOCK;
            auto it = lb_blocks.find(b);
            if (it == lb_blocks.end())
                return kInf;
            const auto &arr = it->second;
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

        std::vector<std::unordered_map<int64_t, StatefulEntry>> layers(T + 1);
        std::unordered_map<std::pair<int, int64_t>, StatefulParent, PairHash> parent;
        double best_final_cost = kInf;
        int64_t best_final_pen = std::numeric_limits<int64_t>::max();
        int best_final_time = -1;
        bool timed_out = false;
        int best_partial_jobs = 0;
        double best_partial_cost = kInf;
        int best_partial_time = 0;
        bool use_early = params.early_tie_break;

        int eff_max_gap = spaces.banded ? spaces.max_gap : T;

        for (int t_s = spaces.early; t_s <= spaces.late; ++t_s)
        {
            double startup = spaces.c_start[t_s];
            if (startup >= kInf)
                continue;
            for (int i = 0; i < K; ++i)
            {
                int L = lengths[i];
                int t_end = t_s + L;
                if (t_end > T || t_s + L > spaces.late + 1)
                    continue;
                double cost = startup + (prefix_proc[t_end] - prefix_proc[t_s]);
                int64_t new_state = inc[i];
                int new_rw = total_rw - L;
                int64_t pen = use_early ? t_s : 0;
                double lb = cost + lb_proc_cost(t_end, new_rw) + min_c_end;
                if (params.known_ub > 0 && lb > params.known_ub + kEps)
                    continue;
                if (lb > best_final_cost + kEps)
                    continue;
                auto it = layers[t_end].find(new_state);
                if (it == layers[t_end].end() || cost < it->second.cost || (use_early && std::fabs(cost - it->second.cost) <= kEps && pen < it->second.pen))
                {
                    layers[t_end][new_state] = {cost, pen, new_rw, 1};
                    if (params.track_schedule)
                        parent[{t_end, new_state}] = {-1, 0, L, t_s};
                }
            }
        }

        for (int t_end = 1; t_end <= T; ++t_end)
        {
            if (params.time_limit > 0 && elapsed_sec() > params.time_limit)
            {
                timed_out = true;
                break;
            }
            auto &layer = layers[t_end];
            if (layer.empty())
                continue;
            if (params.max_states > 0 && static_cast<int64_t>(layer.size()) > params.max_states)
            {
                timed_out = true;
                break;
            }

            for (const auto &kv : layer)
            {
                const auto &sv = kv.second;
                if (sv.jd > best_partial_jobs || (sv.jd == best_partial_jobs && sv.cost < best_partial_cost))
                {
                    best_partial_jobs = sv.jd;
                    best_partial_cost = sv.cost;
                    best_partial_time = t_end;
                }
            }

            auto it_final = layer.find(final_state);
            if (it_final != layer.end())
            {
                double c_with_shutdown = it_final->second.cost + spaces.c_end[t_end];
                int64_t p_final = it_final->second.pen;
                bool better = c_with_shutdown < best_final_cost;
                if (use_early && !better && std::fabs(c_with_shutdown - best_final_cost) <= kEps)
                {
                    better = (p_final < best_final_pen) || (p_final == best_final_pen && t_end < best_final_time);
                }
                if (better)
                {
                    best_final_cost = c_with_shutdown;
                    best_final_pen = p_final;
                    best_final_time = t_end;
                }
            }

            for (const auto &kv : layer)
            {
                int64_t state = kv.first;
                const auto &sv = kv.second;
                if (sv.rw == 0)
                    continue;

                int gap_limit = std::min(t_end + eff_max_gap, spaces.late + 1);
                for (int t_s = t_end; t_s < gap_limit; ++t_s)
                {
                    double gap = spaces.gap_cost(t_end, t_s);
                    if (gap >= kInf)
                        continue;
                    double base_cost = sv.cost + gap;
                    int64_t x = state;
                    for (int i = 0; i < K; ++i)
                    {
                        int used_i = static_cast<int>(x % radices[i]);
                        x /= radices[i];
                        if (used_i >= totals[i])
                            continue;
                        int L = lengths[i];
                        int job_end = t_s + L;
                        if (job_end > T || t_s + L > spaces.late + 1)
                            continue;
                        int64_t new_state = state + inc[i];
                        int new_rw = sv.rw - L;
                        int new_jd = sv.jd + 1;
                        double cand_cost = base_cost + (prefix_proc[job_end] - prefix_proc[t_s]);
                        int64_t cand_pen = use_early ? (sv.pen + t_s) : sv.pen;
                        double lb = cand_cost + lb_proc_cost(job_end, new_rw) + min_c_end;
                        if (params.known_ub > 0 && lb > params.known_ub + kEps)
                            continue;
                        if (lb > best_final_cost + kEps)
                            continue;
                        auto &target = layers[job_end];
                        auto it = target.find(new_state);
                        if (it == target.end() || cand_cost < it->second.cost || (use_early && std::fabs(cand_cost - it->second.cost) <= kEps && cand_pen < it->second.pen))
                        {
                            target[new_state] = {cand_cost, cand_pen, new_rw, new_jd};
                            if (params.track_schedule)
                                parent[{job_end, new_state}] = {t_end, state, L, t_s};
                        }
                    }
                }

                if (spaces.banded && gap_limit < spaces.late + 1)
                {
                    double c_end_here = spaces.c_end[t_end];
                    if (c_end_here < kInf)
                    {
                        for (int t_s = gap_limit; t_s <= spaces.late; ++t_s)
                        {
                            double startup = spaces.c_start[t_s];
                            if (startup >= kInf)
                                continue;
                            double base_cost = sv.cost + c_end_here + startup;
                            int64_t x = state;
                            for (int i = 0; i < K; ++i)
                            {
                                int used_i = static_cast<int>(x % radices[i]);
                                x /= radices[i];
                                if (used_i >= totals[i])
                                    continue;
                                int L = lengths[i];
                                int job_end = t_s + L;
                                if (job_end > T || t_s + L > spaces.late + 1)
                                    continue;
                                int64_t new_state = state + inc[i];
                                int new_rw = sv.rw - L;
                                int new_jd = sv.jd + 1;
                                double cand_cost = base_cost + (prefix_proc[job_end] - prefix_proc[t_s]);
                                int64_t cand_pen = use_early ? (sv.pen + t_s) : sv.pen;
                                double lb = cand_cost + lb_proc_cost(job_end, new_rw) + min_c_end;
                                if (params.known_ub > 0 && lb > params.known_ub + kEps)
                                    continue;
                                if (lb > best_final_cost + kEps)
                                    continue;
                                auto &target = layers[job_end];
                                auto it = target.find(new_state);
                                if (it == target.end() || cand_cost < it->second.cost || (use_early && std::fabs(cand_cost - it->second.cost) <= kEps && cand_pen < it->second.pen))
                                {
                                    target[new_state] = {cand_cost, cand_pen, new_rw, new_jd};
                                    if (params.track_schedule)
                                        parent[{job_end, new_state}] = {t_end, state, L, t_s};
                                }
                            }
                        }
                    }
                }
            }

            int freed_t = t_end - max_job_len - eff_max_gap;
            if (freed_t >= 0)
                layers[freed_t].clear();
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
            return out;
        }

        if (timed_out && best_partial_jobs > 0)
        {
            out.feasible = true;
            out.cost = best_partial_cost;
            out.finish_time = best_partial_time;
        }
        return out;
    }

} // namespace dp
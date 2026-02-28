/**
 * bnb_solver.cpp — Branch-and-Bound implementation.
 *
 * Port of BranchAndBoundSolver (bnb_solver_custom.py):
 *   - SPT / LPT initialization
 *   - DFS with GCD relaxation LB + FFD bin-packing prune
 *   - Sequence DP for cost evaluation (O(T·J), rolling prefix-min/argmin)
 *   - Time limit via steady_clock
 */

#include "bnb_solver.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <unordered_set>

namespace bnb {

// ─────────────────────────────────────────────────────────────────────────────
//  Sequence DP
//  For J jobs in fixed order (proc_times), find optimal start times.
//  Returns (cost, starts[0..J-1]).
//
//  TEC[i][t] = min cost for first i jobs with job i-1 starting at t.
//  Uses rolling prefix-min with argmin tracking (O(T) per job).
// ─────────────────────────────────────────────────────────────────────────────
std::pair<double, std::vector<int>> evaluate_sequence(
    const std::vector<int>&    proc_times,
    const std::vector<double>& prefix,
    int                        T)
{
    const int J = static_cast<int>(proc_times.size());
    if (J == 0) return {0.0, {}};

    // Compute earliest/latest start for each job
    std::vector<int> ES(J), LS(J);
    {
        int cum_before = 0;
        for (int i = 0; i < J; ++i) {
            ES[i] = cum_before;
            cum_before += proc_times[i];
        }
        int cum_after = 0;
        for (int i = J - 1; i >= 0; --i) {
            cum_after += proc_times[i];
            LS[i] = T - cum_after;
        }
    }

    // TEC[i][t] — use two 1D vectors (rolling)
    // prev_layer = TEC[i-1], curr_layer = TEC[i]
    const double INF_VAL = kInf;
    std::vector<double> prev(T, INF_VAL), curr(T, INF_VAL);
    std::vector<int>    prev_par(T, -1), curr_par(T, -1);
    // parent[i][t] stores the previous-job start that achieved TEC[i][t]
    // We flatten to a (J+1) × T matrix for backtracking.
    // To save memory we store just the starts per row.
    std::vector<std::vector<int>> all_par(J + 1, std::vector<int>(T, -1));

    // Base: dummy job 0 at t=0 (TEC[0][0]=0, rest=inf)
    prev.assign(T, INF_VAL);
    if (T > 0) prev[0] = 0.0;
    all_par[0][0] = 0;

    // Running prefix min/argmin over prev
    std::vector<double> pfx_min(T + 1, INF_VAL);
    std::vector<int>    pfx_arg(T + 1, -1);

    for (int i = 1; i <= J; ++i) {
        int p = proc_times[i - 1];
        int p_prev = (i >= 2) ? proc_times[i - 2] : 0;

        // Build prefix min/argmin for prev (over indices 0..T-1)
        {
            double cur_min = INF_VAL;
            int    cur_arg = -1;
            for (int t = 0; t < T; ++t) {
                if (prev[t] < cur_min) { cur_min = prev[t]; cur_arg = t; }
                pfx_min[t] = cur_min;
                pfx_arg[t] = cur_arg;
            }
        }

        curr.assign(T, INF_VAL);
        curr_par.assign(T, -1);

        int max_s = std::min(LS[i - 1], T - p);
        for (int s = ES[i - 1]; s <= max_s; ++s) {
            // Previous job must end by s: start_prev <= s - p_prev
            int limit = s - (i == 1 ? 0 : p_prev);
            if (i == 1) limit = 0;  // first job: must come from dummy at 0
            if (limit < 0) continue;
            if (limit >= T) limit = T - 1;

            double best_prev = (i == 1) ? prev[0] : pfx_min[limit];
            if (best_prev >= INF_VAL) continue;

            double cost_here = prefix[s + p] - prefix[s];
            double total     = best_prev + cost_here;
            if (total < curr[s]) {
                curr[s] = total;
                curr_par[s] = (i == 1) ? 0 : pfx_arg[limit];
            }
        }

        all_par[i] = curr_par;
        std::swap(prev, curr);
    }

    // Backtrack
    double best_cost = kInf;
    int best_s = -1;
    for (int t = 0; t < T; ++t) {
        if (prev[t] < best_cost) { best_cost = prev[t]; best_s = t; }
    }
    if (best_s < 0 || best_cost >= kInf) return {kInf, {}};

    std::vector<int> starts(J);
    int cur_s = best_s;
    for (int i = J; i >= 1; --i) {
        starts[i - 1] = cur_s;
        cur_s = all_par[i][cur_s];
    }

    return {best_cost, starts};
}

// ─────────────────────────────────────────────────────────────────────────────
//  Internal BnB state
// ─────────────────────────────────────────────────────────────────────────────
struct Solver {
    const Instance&    inst;
    const BnBParams&   params;
    std::vector<double> prefix;  // prefix sums, length T+1

    double           best_cost  = kInf;
    std::vector<int> best_seq;
    std::vector<int> best_starts;

    int  nodes       = 0;
    int  pruned_bp   = 0;
    bool timed_out   = false;

    using Clock = std::chrono::steady_clock;
    Clock::time_point t_start;

    Solver(const Instance& inst_, const BnBParams& p_)
        : inst(inst_), params(p_)
    {
        prefix.resize(inst.T + 1);
        prefix[0] = 0.0;
        for (int t = 0; t < inst.T; ++t)
            prefix[t + 1] = prefix[t] + inst.energy_costs[t];
        t_start = Clock::now();
    }

    bool check_timeout() {
        if (params.time_limit <= 0) return false;
        double elapsed = std::chrono::duration<double>(Clock::now() - t_start).count();
        if (elapsed > params.time_limit) { timed_out = true; return true; }
        return false;
    }

    double evaluate_seq(const std::vector<int>& seq) {
        if (seq.empty()) return 0.0;
        std::vector<int> pts;
        pts.reserve(seq.size());
        for (int j : seq) pts.push_back(inst.processing_times[j]);
        auto [cost, _] = evaluate_sequence(pts, prefix, inst.T);
        return cost;
    }

    // LPT heuristic
    std::vector<int> lpt() {
        std::vector<int> jobs(inst.n_jobs);
        std::iota(jobs.begin(), jobs.end(), 0);
        std::sort(jobs.begin(), jobs.end(), [&](int a, int b) {
            return inst.processing_times[a] > inst.processing_times[b];
        });
        return jobs;
    }

    // SPT heuristic
    std::vector<int> spt() {
        std::vector<int> jobs(inst.n_jobs);
        std::iota(jobs.begin(), jobs.end(), 0);
        std::sort(jobs.begin(), jobs.end(), [&](int a, int b) {
            return inst.processing_times[a] < inst.processing_times[b];
        });
        return jobs;
    }

    // GCD of ints in a container
    template<class It>
    static int gcd_range(It begin, It end) {
        if (begin == end) return 1;
        int g = *begin++;
        for (; begin != end; ++begin) g = std::gcd(g, *begin);
        return g == 0 ? 1 : g;
    }

    // Compute lower bound with blocks.
    // Returns (lb_cost, starts_of_relaxed_unfixed_pieces, relaxed_proc_times)
    struct LBResult {
        double            lb;
        std::vector<int>  relaxed_pts;  // full relaxed sequence (fixed + pieces)
        std::vector<int>  starts;       // start times for all jobs in relaxed_pts
        int               n_fixed;      // how many are from partial_sequence
        int               gcd_val;
    };

    LBResult compute_lb(
        const std::vector<int>& partial_seq,
        const std::vector<int>& remaining)   // job indices
    {
        if (remaining.empty()) {
            double c = evaluate_seq(partial_seq);
            return {c, {}, {}, (int)partial_seq.size(), 1};
        }

        // Gather remaining processing times
        std::vector<int> rem_pts;
        rem_pts.reserve(remaining.size());
        for (int j : remaining) rem_pts.push_back(inst.processing_times[j]);

        int total_rem = std::accumulate(rem_pts.begin(), rem_pts.end(), 0);
        if (total_rem == 0) {
            double c = evaluate_seq(partial_seq);
            return {c, {}, {}, (int)partial_seq.size(), 1};
        }

        int g = gcd_range(rem_pts.begin(), rem_pts.end());
        int n_pieces = total_rem / g;

        // Build relaxed sequence
        std::vector<int> relaxed;
        relaxed.reserve(partial_seq.size() + n_pieces);
        for (int j : partial_seq) relaxed.push_back(inst.processing_times[j]);
        int n_fixed = (int)relaxed.size();
        for (int k = 0; k < n_pieces; ++k) relaxed.push_back(g);

        auto [lb, starts] = evaluate_sequence(relaxed, prefix, inst.T);
        return {lb, relaxed, starts, n_fixed, g};
    }

    // Extract contiguous blocks from the unfixed portion of the schedule
    std::vector<int> extract_blocks(
        const std::vector<int>& relaxed_pts,
        const std::vector<int>& starts,
        int n_fixed)
    {
        if (n_fixed >= (int)starts.size()) return {};

        // Build intervals for unfixed jobs
        std::vector<std::pair<int,int>> intervals;
        intervals.reserve(starts.size() - n_fixed);
        for (int i = n_fixed; i < (int)starts.size(); ++i) {
            int s = starts[i];
            int e = s + relaxed_pts[i];
            intervals.emplace_back(s, e);
        }
        if (intervals.empty()) return {};

        std::sort(intervals.begin(), intervals.end());

        // Merge overlapping intervals → block sizes
        std::vector<int> blocks;
        int cs = intervals[0].first, ce = intervals[0].second;
        for (int k = 1; k < (int)intervals.size(); ++k) {
            if (intervals[k].first <= ce) {
                ce = std::max(ce, intervals[k].second);
            } else {
                blocks.push_back(ce - cs);
                cs = intervals[k].first;
                ce = intervals[k].second;
            }
        }
        blocks.push_back(ce - cs);
        return blocks;
    }

    // FFD bin packing: try to fit remaining jobs (sorted by p desc) into blocks.
    // Returns packed sequence of job indices, or empty if infeasible.
    std::vector<int> try_bin_packing(
        const std::vector<int>& remaining,
        const std::vector<int>& blocks)
    {
        if (blocks.empty() || remaining.empty()) return {};

        // Sort jobs by processing time desc
        std::vector<int> jobs = remaining;
        std::sort(jobs.begin(), jobs.end(), [&](int a, int b) {
            return inst.processing_times[a] > inst.processing_times[b];
        });

        int max_block = *std::max_element(blocks.begin(), blocks.end());
        if (inst.processing_times[jobs[0]] > max_block) return {};

        std::vector<int> bins = blocks;  // remaining capacity
        std::vector<std::vector<int>> bin_contents(bins.size());

        for (int j : jobs) {
            int pt = inst.processing_times[j];
            // First fit (sorted bins try largest first for best fit — but FFD is fine)
            bool placed = false;
            for (int b = 0; b < (int)bins.size(); ++b) {
                if (bins[b] >= pt) {
                    bins[b] -= pt;
                    bin_contents[b].push_back(j);
                    placed = true;
                    break;
                }
            }
            if (!placed) return {};
        }

        // Flatten
        std::vector<int> packed;
        for (auto& bc : bin_contents)
            for (int j : bc) packed.push_back(j);
        return packed;
    }

    // DFS
    void dfs(std::vector<int>& partial, std::vector<int> remaining_vec) {
        if (check_timeout()) return;
        ++nodes;

        if (remaining_vec.empty()) {
            double cost = evaluate_seq(partial);
            if (cost < best_cost) {
                best_cost   = cost;
                best_seq    = partial;
                // Also compute starts for the final result
                std::vector<int> pts;
                for (int j : partial) pts.push_back(inst.processing_times[j]);
                auto [_, st] = evaluate_sequence(pts, prefix, inst.T);
                best_starts = st;
            }
            return;
        }

        if (check_timeout()) return;

        // Compute LB
        auto lb_res = compute_lb(partial, remaining_vec);
        double lb   = lb_res.lb;

        if (check_timeout()) return;
        if (lb >= best_cost) return;  // prune by bound

        // Bin-packing heuristic
        if (!lb_res.starts.empty()) {
            auto blocks = extract_blocks(lb_res.relaxed_pts, lb_res.starts, lb_res.n_fixed);
            if (!blocks.empty()) {
                if (check_timeout()) return;
                auto packed = try_bin_packing(remaining_vec, blocks);
                if (!packed.empty()) {
                    // Build full candidate
                    std::vector<int> candidate = partial;
                    candidate.insert(candidate.end(), packed.begin(), packed.end());
                    // Cost should equal lb (exactly — relaxation is tight)
                    // But verify:
                    double cand_cost = evaluate_seq(candidate);
                    if (cand_cost < best_cost) {
                        best_cost = cand_cost;
                        best_seq  = candidate;
                        std::vector<int> pts;
                        for (int j : candidate) pts.push_back(inst.processing_times[j]);
                        auto [_, st] = evaluate_sequence(pts, prefix, inst.T);
                        best_starts = st;
                    }
                    ++pruned_bp;
                    return;
                }
            }
        }

        // Branching: symmetry break — one representative per unique processing time
        std::map<int, int> pt_to_rep;  // pt → first job in remaining with that pt
        for (int j : remaining_vec) {
            int pt = inst.processing_times[j];
            if (pt_to_rep.find(pt) == pt_to_rep.end())
                pt_to_rep[pt] = j;
        }

        // Branch in sorted order of processing times
        std::vector<std::pair<int,int>> branches(pt_to_rep.begin(), pt_to_rep.end());
        // Default: ascending order (SPT-style)
        std::sort(branches.begin(), branches.end());

        for (auto& [pt, job] : branches) {
            if (check_timeout()) return;

            // Build new remaining
            std::vector<int> new_remaining;
            new_remaining.reserve(remaining_vec.size() - 1);
            bool removed = false;
            for (int j : remaining_vec) {
                if (!removed && j == job) { removed = true; continue; }
                new_remaining.push_back(j);
            }

            partial.push_back(job);
            dfs(partial, std::move(new_remaining));
            partial.pop_back();

            // Alpha-beta: if current best is at most lb, stop
            if (best_cost <= lb) break;
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Main entry point
// ─────────────────────────────────────────────────────────────────────────────
BnBResult solve_bnb(const Instance& inst, const BnBParams& params) {
    Solver slv(inst, params);

    // Warm-start with SPT and LPT
    {
        auto spt_seq = slv.spt();
        double sc    = slv.evaluate_seq(spt_seq);
        auto lpt_seq = slv.lpt();
        double lc    = slv.evaluate_seq(lpt_seq);
        if (sc <= lc) {
            slv.best_cost = sc; slv.best_seq = spt_seq;
        } else {
            slv.best_cost = lc; slv.best_seq = lpt_seq;
        }
        // Compute starts for best
        std::vector<int> pts;
        for (int j : slv.best_seq) pts.push_back(inst.processing_times[j]);
        auto [_, st] = evaluate_sequence(pts, slv.prefix, inst.T);
        slv.best_starts = st;

        if (params.verbose) {
            double better = std::min(sc, lc);
            const char* which = (sc <= lc) ? "SPT" : "LPT";
            // (can't print here without iostream — handled in compare.cpp)
            (void)better; (void)which;
        }
    }

    // DFS
    std::vector<int> all_jobs(inst.n_jobs);
    std::iota(all_jobs.begin(), all_jobs.end(), 0);
    std::vector<int> partial;
    partial.reserve(inst.n_jobs);
    slv.dfs(partial, all_jobs);

    BnBResult res;
    res.sequence   = slv.best_seq;
    res.starts     = slv.best_starts;
    res.cost       = slv.best_cost;
    res.nodes      = slv.nodes;
    res.pruned_bp  = slv.pruned_bp;
    res.timed_out  = slv.timed_out;
    res.solve_time = std::chrono::duration<double>(
        Solver::Clock::now() - slv.t_start).count();

    return res;
}

}  // namespace bnb

#pragma once
/**
 * bnb_solver.hpp — Branch-and-Bound for single-machine TOU scheduling.
 *
 * Port of BranchAndBoundSolver (bnb_solver_custom.py).
 *
 * Features:
 *  - SPT / LPT warm-start heuristics
 *  - GCD relaxation lower bound  (chop remaining jobs into GCD pieces, solve with DP)
 *  - Bin-packing primal heuristic (FFD into relaxed blocks)
 *  - Symmetry breaking (one representative per unique processing time)
 *  - Alpha-beta style early-exit when best_cost ≤ lb
 *  - O(T·J) sequence DP for exact cost evaluation (rolling prefix-min)
 *  - Time limit via steady_clock
 */

#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace bnb {

static constexpr double kInf = 1e300;  // large sentinel, safe under -ffast-math

// ─────────────────────────────────────────────────────────────────────────────
//  Problem instance
// ─────────────────────────────────────────────────────────────────────────────
struct Instance {
    int                 n_jobs;
    std::vector<int>    processing_times;  // p[j] = processing time of job j
    int                 T;
    std::vector<double> energy_costs;      // prices[t], length T
};

// ─────────────────────────────────────────────────────────────────────────────
//  Solver result
// ─────────────────────────────────────────────────────────────────────────────
struct BnBResult {
    std::vector<int> sequence;    // optimal job ordering (indices into p[])
    std::vector<int> starts;      // start time for each job in sequence
    double           cost        = kInf;
    double           solve_time  = 0.0;   // seconds
    int              nodes       = 0;
    int              pruned_bp   = 0;     // pruned by bin-packing heuristic
    bool             timed_out   = false;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Solver parameters
// ─────────────────────────────────────────────────────────────────────────────
struct BnBParams {
    double time_limit = 300.0;  // seconds; ≤0 means no limit
    bool   verbose    = false;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Sequence DP helper
//  Given a fixed sequence of J jobs (specified by their processing times),
//  find optimal start times minimizing total energy cost.
//  Returns (cost, starts_vector).
// ─────────────────────────────────────────────────────────────────────────────
std::pair<double, std::vector<int>> evaluate_sequence(
    const std::vector<int>&    proc_times,   // J entries
    const std::vector<double>& prefix,       // prefix sums, length T+1
    int                        T
);

// ─────────────────────────────────────────────────────────────────────────────
//  Main entry point
// ─────────────────────────────────────────────────────────────────────────────
BnBResult solve_bnb(const Instance& inst, const BnBParams& params = {});

}  // namespace bnb

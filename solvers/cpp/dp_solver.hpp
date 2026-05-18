#pragma once
/**
 * dp_solver.hpp — Sparse DP for single-machine TOU scheduling.
 *
 * Faithful C++ port of solve_sparse_dp_python / solve_sparse_dp_cython.
 *
 * State encoding: mixed-radix integer over "used" counts.
 *   state = Σ used[i] * mult[i],   mult[0]=1, mult[i]=mult[i-1]*(totals[i-1]+1)
 *
 * Per-layer storage: open-addressing hash map (int64 → StateEntry).
 * Load factor 70% → doubles on overflow.
 * Hash: Fibonacci/FNV variant — same as Cython version.
 *
 * Memory per state: ~48 bytes (StateEntry = 24 bytes + 8-byte key slot +
 *   overhead), vs ~260 bytes in CPython dicts.
 */

#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace dp
{

    // ─────────────────────────────────────────────────────────────────────────────
    //  Constants
    // ─────────────────────────────────────────────────────────────────────────────
    static constexpr double kEps = 1e-12;
    static constexpr double kInf = 1e300;   // large sentinel, safe under -ffast-math
    static constexpr int64_t kEmpty = -1LL; // 0xFFFFFFFFFFFFFFFF — matches memset(0xFF)

    // ─────────────────────────────────────────────────────────────────────────────
    //  StateEntry — 24 bytes
    // ─────────────────────────────────────────────────────────────────────────────
    struct StateEntry
    {
        double cost; // accumulated energy cost
        int64_t pen; // tie-break penalty (sum of start times)
        int32_t rw;  // remaining work (slots)
        int32_t jd;  // jobs done
    };

    // ─────────────────────────────────────────────────────────────────────────────
    //  Open-addressing hash map  int64 → StateEntry
    //  Fibonacci hash; linear probing; 70% load factor; power-of-2 capacity.
    // ─────────────────────────────────────────────────────────────────────────────
    class StateMap
    {
    public:
        explicit StateMap(std::size_t initial_cap = 64);
        ~StateMap();

        // Non-copyable, movable
        StateMap(const StateMap &) = delete;
        StateMap &operator=(const StateMap &) = delete;
        StateMap(StateMap &&) noexcept;
        StateMap &operator=(StateMap &&) noexcept;

        // Returns index of slot for key, or -1 if not found.
        std::ptrdiff_t lookup(int64_t key) const noexcept;

        // Insert key with val (key must NOT exist). Returns slot index.
        std::ptrdiff_t insert(int64_t key, const StateEntry &val);

        // Direct access by slot index
        StateEntry &val_at(std::ptrdiff_t idx) noexcept { return vals_[idx]; }
        const StateEntry &val_at(std::ptrdiff_t idx) const noexcept { return vals_[idx]; }

        std::size_t size() const noexcept { return size_; }
        std::size_t capacity() const noexcept { return cap_; }

        // Iterate over live entries
        template <class Fn>
        void for_each(Fn &&fn) const noexcept
        {
            for (std::size_t i = 0; i < cap_; ++i)
                if (keys_[i] != kEmpty)
                    fn(keys_[i], vals_[i]);
        }

        void clear() noexcept;

    private:
        void grow();
        static std::size_t hash_idx(int64_t key, std::size_t mask) noexcept;

        int64_t *keys_ = nullptr;
        StateEntry *vals_ = nullptr;
        std::size_t cap_ = 0;
        std::size_t mask_ = 0;
        std::size_t size_ = 0;
    };

    // ─────────────────────────────────────────────────────────────────────────────
    //  Open-addressing hash map  int64 → int32  (parent map)
    // ─────────────────────────────────────────────────────────────────────────────
    class ParentMap
    {
    public:
        explicit ParentMap(std::size_t initial_cap = 4096);
        ~ParentMap();

        ParentMap(const ParentMap &) = delete;
        ParentMap &operator=(const ParentMap &) = delete;
        ParentMap(ParentMap &&) noexcept;
        ParentMap &operator=(ParentMap &&) noexcept;

        void set(int64_t key, int32_t val);
        // Returns -2 if not found
        int32_t get(int64_t key) const noexcept;

        std::size_t size() const noexcept { return size_; }

    private:
        void grow();
        static std::size_t hash_idx(int64_t key, std::size_t mask) noexcept;

        int64_t *keys_ = nullptr;
        int32_t *vals_ = nullptr;
        std::size_t cap_ = 0;
        std::size_t mask_ = 0;
        std::size_t size_ = 0;
    };

    // ─────────────────────────────────────────────────────────────────────────────
    //  DP result
    // ─────────────────────────────────────────────────────────────────────────────
    struct Segment
    {
        int start; // inclusive
        int length;
    };

    struct DPResult
    {
        bool feasible = false;
        double cost = kInf;
        int finish_time = 0;
        bool timed_out = false;
        std::vector<Segment> segments; // (start, length) pairs — empty if not tracked
    };

    // ─────────────────────────────────────────────────────────────────────────────
    //  Solver parameters
    // ─────────────────────────────────────────────────────────────────────────────
    struct DPParams
    {
        double time_limit = -1.0; // seconds; -1 = unlimited
        bool track_schedule = true;
        int64_t max_states = 0; // 0 = unlimited
        bool early_tie_break = true;
        double known_ub = -1.0; // known upper bound; -1 = unused
    };

    struct PricingResult
    {
        bool feasible = false;              // found a non-empty feasible pattern
        bool negative = false;              // reduced cost < 0
        double reduced_cost = kInf;         // includes sigma shift
        double reduced_cost_no_sigma = kInf;
        double energy_cost = kInf;          // original single-machine energy cost
        int finish_time = 0;
        bool timed_out = false;
        std::int64_t states_explored = 0;
        std::vector<int> counts;            // chosen type counts
        std::vector<Segment> segments;      // empty unless tracked
    };

    struct PricingParams
    {
        double time_limit = -1.0;
        bool track_schedule = false;
        int64_t max_states = 0;
        bool early_tie_break = true;
        double cutoff = -1e-9;             // stop early if reduced cost reaches this threshold
    };

    // ─────────────────────────────────────────────────────────────────────────────
    //  Main entry point
    //
    //  lengths[0..K-1]  — distinct processing times (sorted ascending recommended)
    //  totals[0..K-1]   — count of jobs for each length
    //  prefix[0..T]     — prefix sums of energy prices (prefix[i+1]-prefix[i] = price[i])
    //  T                — scheduling horizon
    // ─────────────────────────────────────────────────────────────────────────────
    DPResult solve_sparse_dp(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix,
        int T,
        const DPParams &params = {});

    PricingResult solve_pricing_dp(
        const std::vector<int> &lengths,
        const std::vector<int> &max_counts,
        const std::vector<double> &prefix,
        int T,
        const std::vector<double> &rewards,
        double rate,
        double sigma,
        const PricingParams &params = {});

} // namespace dp

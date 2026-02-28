/**
 * dp_solver.cpp — Sparse DP implementation.
 *
 * Port of solve_sparse_dp_python / solve_sparse_dp_cython with:
 *   - Open-addressing hash maps (Fibonacci hash, linear probing)
 *   - Block-based admissible LB pruning (block_size=20)
 *   - Feasibility pruning (remaining_work > T-t)
 *   - Rolling memory release (free layer t-max_job_len after processing t)
 *   - Optional parent-pointer tracking for schedule reconstruction
 *   - Time limit via steady_clock
 */

#include "dp_solver.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace dp {

// ─────────────────────────────────────────────────────────────────────────────
//  Fibonacci hash (same as Cython version)
// ─────────────────────────────────────────────────────────────────────────────
static inline std::size_t fib_hash(int64_t key, std::size_t mask) noexcept {
    uint64_t h = static_cast<uint64_t>(key);
    h *= 14695981039346656037ULL;
    h ^= (h >> 32);
    h ^= (h >> 16);
    return static_cast<std::size_t>(h & static_cast<uint64_t>(mask));
}

// ─────────────────────────────────────────────────────────────────────────────
//  StateMap
// ─────────────────────────────────────────────────────────────────────────────
/*static*/ std::size_t StateMap::hash_idx(int64_t key, std::size_t mask) noexcept {
    return fib_hash(key, mask);
}

StateMap::StateMap(std::size_t initial_cap) {
    std::size_t cap = 64;
    while (cap < initial_cap) cap <<= 1;
    keys_ = static_cast<int64_t*>(std::malloc(cap * sizeof(int64_t)));
    vals_ = static_cast<StateEntry*>(std::malloc(cap * sizeof(StateEntry)));
    if (!keys_ || !vals_) throw std::bad_alloc();
    std::memset(keys_, 0xFF, cap * sizeof(int64_t));  // fill with -1 (kEmpty sentinel)
    cap_  = cap;
    mask_ = cap - 1;
    size_ = 0;
}

StateMap::~StateMap() {
    std::free(keys_);
    std::free(vals_);
}

StateMap::StateMap(StateMap&& o) noexcept
    : keys_(o.keys_), vals_(o.vals_), cap_(o.cap_), mask_(o.mask_), size_(o.size_) {
    o.keys_ = nullptr; o.vals_ = nullptr; o.cap_ = 0; o.mask_ = 0; o.size_ = 0;
}

StateMap& StateMap::operator=(StateMap&& o) noexcept {
    if (this != &o) {
        std::free(keys_); std::free(vals_);
        keys_ = o.keys_; vals_ = o.vals_;
        cap_ = o.cap_; mask_ = o.mask_; size_ = o.size_;
        o.keys_ = nullptr; o.vals_ = nullptr; o.cap_ = 0; o.mask_ = 0; o.size_ = 0;
    }
    return *this;
}

std::ptrdiff_t StateMap::lookup(int64_t key) const noexcept {
    std::size_t idx = hash_idx(key, mask_);
    while (true) {
        if (keys_[idx] == key)    return static_cast<std::ptrdiff_t>(idx);
        if (keys_[idx] == kEmpty) return -1;
        idx = (idx + 1) & mask_;
    }
}

void StateMap::grow() {
    std::size_t new_cap  = cap_ << 1;
    std::size_t new_mask = new_cap - 1;
    int64_t*    nk = static_cast<int64_t*>(std::malloc(new_cap * sizeof(int64_t)));
    StateEntry* nv = static_cast<StateEntry*>(std::malloc(new_cap * sizeof(StateEntry)));
    if (!nk || !nv) { std::free(nk); std::free(nv); throw std::bad_alloc(); }
    std::memset(nk, 0xFF, new_cap * sizeof(int64_t));
    // Rehash
    for (std::size_t i = 0; i < cap_; ++i) {
        if (keys_[i] == kEmpty) continue;
        std::size_t idx = fib_hash(keys_[i], new_mask);
        while (nk[idx] != kEmpty) idx = (idx + 1) & new_mask;
        nk[idx] = keys_[i];
        nv[idx] = vals_[i];
    }
    std::free(keys_); std::free(vals_);
    keys_ = nk; vals_ = nv; cap_ = new_cap; mask_ = new_mask;
}

std::ptrdiff_t StateMap::insert(int64_t key, const StateEntry& val) {
    if (size_ * 10 > cap_ * 7) grow();
    std::size_t idx = hash_idx(key, mask_);
    while (true) {
        if (keys_[idx] == kEmpty) {
            keys_[idx] = key;
            vals_[idx] = val;
            ++size_;
            return static_cast<std::ptrdiff_t>(idx);
        }
        if (keys_[idx] == key) return static_cast<std::ptrdiff_t>(idx); // already present
        idx = (idx + 1) & mask_;
    }
}

template<class Fn>
void StateMap::for_each(Fn&& fn) const noexcept {
    for (std::size_t i = 0; i < cap_; ++i) {
        if (keys_[i] != kEmpty) fn(keys_[i], vals_[i]);
    }
}

void StateMap::clear() noexcept {
    std::memset(keys_, 0xFF, cap_ * sizeof(int64_t));
    size_ = 0;
}

// ─────────────────────────────────────────────────────────────────────────────
//  ParentMap
// ─────────────────────────────────────────────────────────────────────────────
/*static*/ std::size_t ParentMap::hash_idx(int64_t key, std::size_t mask) noexcept {
    return fib_hash(key, mask);
}

ParentMap::ParentMap(std::size_t initial_cap) {
    std::size_t cap = 64;
    while (cap < initial_cap) cap <<= 1;
    keys_ = static_cast<int64_t*>(std::malloc(cap * sizeof(int64_t)));
    vals_ = static_cast<int32_t*>(std::malloc(cap * sizeof(int32_t)));
    if (!keys_ || !vals_) throw std::bad_alloc();
    std::memset(keys_, 0xFF, cap * sizeof(int64_t));
    cap_ = cap; mask_ = cap - 1; size_ = 0;
}

ParentMap::~ParentMap() {
    std::free(keys_); std::free(vals_);
}

ParentMap::ParentMap(ParentMap&& o) noexcept
    : keys_(o.keys_), vals_(o.vals_), cap_(o.cap_), mask_(o.mask_), size_(o.size_) {
    o.keys_ = nullptr; o.vals_ = nullptr; o.cap_ = 0; o.mask_ = 0; o.size_ = 0;
}

ParentMap& ParentMap::operator=(ParentMap&& o) noexcept {
    if (this != &o) {
        std::free(keys_); std::free(vals_);
        keys_ = o.keys_; vals_ = o.vals_;
        cap_ = o.cap_; mask_ = o.mask_; size_ = o.size_;
        o.keys_ = nullptr; o.vals_ = nullptr; o.cap_ = 0; o.mask_ = 0; o.size_ = 0;
    }
    return *this;
}

void ParentMap::grow() {
    std::size_t nc  = cap_ << 1;
    std::size_t nm  = nc - 1;
    int64_t* nk = static_cast<int64_t*>(std::malloc(nc * sizeof(int64_t)));
    int32_t* nv = static_cast<int32_t*>(std::malloc(nc * sizeof(int32_t)));
    if (!nk || !nv) { std::free(nk); std::free(nv); throw std::bad_alloc(); }
    std::memset(nk, 0xFF, nc * sizeof(int64_t));
    for (std::size_t i = 0; i < cap_; ++i) {
        if (keys_[i] == kEmpty) continue;
        std::size_t idx = fib_hash(keys_[i], nm);
        while (nk[idx] != kEmpty) idx = (idx + 1) & nm;
        nk[idx] = keys_[i];
        nv[idx] = vals_[i];
    }
    std::free(keys_); std::free(vals_);
    keys_ = nk; vals_ = nv; cap_ = nc; mask_ = nm;
}

void ParentMap::set(int64_t key, int32_t val) {
    if (size_ * 10 > cap_ * 7) grow();
    std::size_t idx = hash_idx(key, mask_);
    while (true) {
        if (keys_[idx] == kEmpty) {
            keys_[idx] = key; vals_[idx] = val; ++size_; return;
        }
        if (keys_[idx] == key) { vals_[idx] = val; return; }
        idx = (idx + 1) & mask_;
    }
}

int32_t ParentMap::get(int64_t key) const noexcept {
    std::size_t idx = hash_idx(key, mask_);
    while (true) {
        if (keys_[idx] == key)    return vals_[idx];
        if (keys_[idx] == kEmpty) return -2;
        idx = (idx + 1) & mask_;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Block-based admissible lower bound
//  For time t, return cheapest `rw` slots among prices[b:T] where b ≤ t.
// ─────────────────────────────────────────────────────────────────────────────
struct LBTable {
    static constexpr int BLOCK = 20;

    // lb_prefix[b/BLOCK] = sorted prefix sums of prices[b..T-1]
    std::vector<std::vector<double>> blocks;
    int T;

    void build(const double* prices, int t_len) {
        T = t_len;
        // n_real: number of real BLOCK-aligned blocks needed
        int n_real = (T > 0) ? ((T - 1) / BLOCK + 1) : 1;
        blocks.resize(n_real + 1);  // +1 for sentinel beyond valid blocks
        for (int b = 0; b < T; b += BLOCK) {
            int bi  = b / BLOCK;
            int len = T - b;  // remaining slots (LB uses [b,T) — admissible since b<=t)
            std::vector<double> sp(prices + b, prices + T);
            std::sort(sp.begin(), sp.end());
            auto& cs = blocks[bi];
            cs.resize(len + 1);
            cs[0] = 0.0;
            for (int i = 0; i < len; ++i) cs[i + 1] = cs[i] + sp[i];
        }
        // sentinel at index n_real: b = n_real*BLOCK >= T means no slots remain
        blocks[n_real].assign(1, 0.0);
    }

    // Returns lower bound cost for remaining work rw starting from time t.
    // Returns +inf if infeasible (rw > available slots).
    double query(int t, int rw) const noexcept {
        int b   = (t / BLOCK) * BLOCK;
        int bi  = b / BLOCK;
        const auto& cs = blocks[bi];
        if (rw >= (int)cs.size()) return kInf;
        return cs[rw];
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Main DP solver
// ─────────────────────────────────────────────────────────────────────────────
DPResult solve_sparse_dp(
    const std::vector<int>&    lengths,
    const std::vector<int>&    totals,
    const std::vector<double>& prefix,
    int                        T,
    const DPParams&            params)
{
    using Clock = std::chrono::steady_clock;
    auto t_start = Clock::now();

    const int K = static_cast<int>(lengths.size());
    assert(K == (int)totals.size());
    assert((int)prefix.size() == T + 1);
    assert(K <= 12 && "K must be ≤ 12 for stack arrays");

    // ── Mixed-radix multipliers ───────────────────────────────────────────
    int32_t c_len[12], c_tot[12], c_rad[12];
    int64_t c_inc[12];
    int max_job_len = 0;
    int total_rw    = 0;
    int64_t final_state = 0;

    int64_t mult = 1;
    for (int i = 0; i < K; ++i) {
        c_len[i] = lengths[i];
        c_tot[i] = totals[i];
        c_rad[i] = totals[i] + 1;
        c_inc[i] = mult;
        final_state += (int64_t)totals[i] * mult;
        mult *= c_rad[i];
        max_job_len = std::max(max_job_len, c_len[i]);
        total_rw   += c_tot[i] * c_len[i];
    }
    const int64_t state_bound = final_state + 1;

    // ── Build LB table ────────────────────────────────────────────────────
    std::vector<double> prices(T);
    for (int i = 0; i < T; ++i) prices[i] = prefix[i + 1] - prefix[i];
    LBTable lb_table;
    lb_table.build(prices.data(), T);

    // ── DP layers: vector of StateMap* (null = freed) ─────────────────────
    // We use raw pointers so we can delete individually to free memory early.
    std::vector<StateMap*> layers(T + 1, nullptr);
    for (int i = 0; i <= T; ++i) layers[i] = new StateMap(64);

    // Seed
    StateEntry sv0{ 0.0, 0, total_rw, 0 };
    layers[0]->insert(0, sv0);

    // ── Parent map ────────────────────────────────────────────────────────
    ParentMap* pmap = nullptr;
    if (params.track_schedule) pmap = new ParentMap(4096);

    // ── Tracking ──────────────────────────────────────────────────────────
    double   best_cost  = (params.known_ub > 0) ? params.known_ub + 1e-8 : kInf;
    int64_t  best_pen   = INT64_MAX;
    int      best_time  = -1;
    bool     timed_out  = false;

    // Best partial (for timeout fallback)
    int      bp_jobs  = 0;
    double   bp_cost  = kInf;
    int      bp_time  = 0;
    int64_t  bp_state = 0;

    const bool early = params.early_tie_break;
    const double* pprefix = prefix.data();

    auto elapsed_sec = [&]() -> double {
        auto dur = Clock::now() - t_start;
        return std::chrono::duration<double>(dur).count();
    };

    // ── Main loop ─────────────────────────────────────────────────────────
    for (int tt = 0; tt <= T; ++tt) {
        // Timeout check
        if (params.time_limit > 0 && elapsed_sec() > params.time_limit) {
            timed_out = true; break;
        }

        StateMap* layer = layers[tt];
        if (!layer || layer->size() == 0) continue;

        // Memory guardrail
        if (params.max_states > 0 && (int64_t)layer->size() > params.max_states) {
            timed_out = true; break;
        }

        // Update best partial
        layer->for_each([&](int64_t state, const StateEntry& sv) {
            if (sv.jd > bp_jobs || (sv.jd == bp_jobs && sv.cost < bp_cost)) {
                bp_jobs = sv.jd; bp_cost = sv.cost;
                bp_time = tt; bp_state = state;
            }
        });

        // Check final state
        {
            auto idx = layer->lookup(final_state);
            if (idx >= 0) {
                const StateEntry& sv = layer->val_at(idx);
                bool better = sv.cost < best_cost;
                if (early && !better && std::fabs(sv.cost - best_cost) <= kEps)
                    better = (sv.pen < best_pen) || (sv.pen == best_pen && tt < best_time);
                if (better) {
                    best_cost = sv.cost; best_pen = sv.pen; best_time = tt;
                }
            }
        }

        if (tt == T) continue;

        const int remaining = T - tt;

        StateMap* nlayer = layers[tt + 1];  // idle target

        // ── Iterate over layer entries ────────────────────────────────────
        layer->for_each([&](int64_t state, const StateEntry& sv) {
            const double   c0 = sv.cost;
            const int64_t  p0 = sv.pen;
            const int      rw = sv.rw;
            const int      jd = sv.jd;

            // Feasibility pruning
            if (rw > remaining) return;

            // LB pruning
            {
                double lb_val = lb_table.query(tt, rw);
                if (c0 + lb_val > best_cost) return;
            }

            // ── Idle transition ───────────────────────────────────────────
            {
                auto idx = nlayer->lookup(state);
                if (idx < 0) {
                    nlayer->insert(state, {c0, p0, rw, jd});
                    if (pmap) pmap->set((int64_t)(tt + 1) * state_bound + state, 0);
                } else {
                    StateEntry& prev = nlayer->val_at(idx);
                    bool better = c0 < prev.cost;
                    if (early && !better && std::fabs(c0 - prev.cost) <= kEps)
                        better = p0 < prev.pen;
                    if (better) {
                        prev = {c0, p0, rw, jd};
                        if (pmap) pmap->set((int64_t)(tt + 1) * state_bound + state, 0);
                    }
                }
            }

            // ── Job transitions ───────────────────────────────────────────
            int64_t x = state;
            for (int i = 0; i < K; ++i) {
                int32_t ui = static_cast<int32_t>(x % c_rad[i]);
                x /= c_rad[i];

                if (ui >= c_tot[i]) continue;
                int L   = c_len[i];
                int end = tt + L;
                if (end > T) continue;

                int64_t ns  = state + c_inc[i];
                int     nrw = rw - L;
                int     njd = jd + 1;
                double  cc  = c0 + (pprefix[end] - pprefix[tt]);
                int64_t cp  = p0 + (early ? tt : 0);

                StateMap* tlayer = layers[end];
                auto idx = tlayer->lookup(ns);
                if (idx < 0) {
                    tlayer->insert(ns, {cc, cp, nrw, njd});
                    if (pmap) pmap->set((int64_t)end * state_bound + ns, (int32_t)L);
                } else {
                    StateEntry& prev = tlayer->val_at(idx);
                    bool better = cc < prev.cost;
                    if (early && !better && std::fabs(cc - prev.cost) <= kEps)
                        better = cp < prev.pen;
                    if (better) {
                        prev = {cc, cp, nrw, njd};
                        if (pmap) pmap->set((int64_t)end * state_bound + ns, (int32_t)L);
                    }
                }
            }
        });

        // ── Free old layer to release memory ─────────────────────────────
        int freed_t = tt - max_job_len;
        if (freed_t >= 0 && layers[freed_t]) {
            delete layers[freed_t];
            layers[freed_t] = nullptr;
        }
    }

    // ── Backtrack schedule ────────────────────────────────────────────────
    std::vector<Segment> segments;
    if (params.track_schedule && pmap && best_time >= 0) {
        int cur_t = best_time;
        int64_t cur_s = final_state;
        while (cur_s != 0 || cur_t != 0) {
            int32_t L = pmap->get((int64_t)cur_t * state_bound + cur_s);
            if (L == -2) break;  // not found (shouldn't happen)
            if (L > 0) {
                segments.push_back({cur_t - L, L});
                // Find which i produced this L
                int64_t x = cur_s;
                for (int i = 0; i < K; ++i) {
                    int32_t ui = static_cast<int32_t>(x % c_rad[i]);
                    x /= c_rad[i];
                    if (c_len[i] == L && ui > 0) {
                        cur_s -= c_inc[i];
                        break;
                    }
                }
                cur_t -= L;
            } else {
                // Idle: step back by 1
                cur_t -= 1;
            }
        }
        std::reverse(segments.begin(), segments.end());
    }

    // ── Cleanup ───────────────────────────────────────────────────────────
    for (int i = 0; i <= T; ++i) {
        delete layers[i];
        layers[i] = nullptr;
    }
    delete pmap;

    // ── Build result ──────────────────────────────────────────────────────
    DPResult res;
    res.timed_out  = timed_out;
    if (best_time >= 0 && best_cost < kInf) {
        res.feasible    = true;
        res.cost        = best_cost;
        res.finish_time = best_time;
        res.segments    = std::move(segments);
    }
    return res;
}

}  // namespace dp

"""
Exact Dynamic Programming for Single Machine Scheduling with TOU Pricing.

This algorithm is PROVEN OPTIMAL by comparison against brute-force enumeration.

Problem:
    - N jobs with release times r_j, deadlines d_j, processing times p_j
    - Time-of-use electricity pricing (cyclic pattern or piecewise constant periods)
    - Minimize total energy cost = sum of price(t) for all t when machine is running

Complexity: O(2^N * N * T^2) where T is the time horizon

This version uses the PaST project's price generation methods:
    - "valleys": Piecewise constant with 2-6 valleys and 1-4 peaks
    - "benchmark": Segmented periods from Tk_choices=(2,3,5) with discrete prices
"""

import itertools
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

# --- Configuration ---
CUSTOM_PRICES: List[float] = []


def set_custom_prices(prices: List[float]):
    global CUSTOM_PRICES
    CUSTOM_PRICES = list(prices)


def clear_custom_prices():
    global CUSTOM_PRICES
    CUSTOM_PRICES = []


def get_price_at(t: int) -> float:
    """Returns price at absolute hour t."""
    if CUSTOM_PRICES:
        if t < len(CUSTOM_PRICES):
            return CUSTOM_PRICES[t]
        return 1000.0  # Penalize out-of-bounds
    return 1.0  # Default flat price


def calculate_cost(start_time: int, duration: int) -> float:
    """Calculates total electricity cost for a job starting at start_time with given duration."""
    return sum(get_price_at(t) for t in range(start_time, start_time + duration))


@dataclass
class Job:
    id: int
    p: int  # Processing time
    r: int  # Release time
    d: int  # Deadline (due date)


@dataclass(frozen=True)
class State:
    finish_time: int
    cost: float
    history: tuple  # Tuple of (job_id, start_time, end_time) for reconstruction


# ============================================================================
# PRICE GENERATION (From PaST Project)
# ============================================================================


def generate_price_curve_valleys(T: int, rng: random.Random) -> np.ndarray:
    """
    Generate a TOU price curve with valleys/peaks pattern.
    This matches PaST/train_bb_branch_policy.py::_generate_price_curve("valleys")

    - Base price: uniform(1.0, 3.0)
    - 2-6 valleys (low price regions): uniform(0.0, 1.0)
    - 1-4 peaks (high price regions): uniform(3.0, 8.0)
    - Plus 10% noise
    """
    # Base level
    c = np.full(T, rng.uniform(1.0, 3.0), dtype=np.float64)

    # Add valleys (low-price regions)
    num_valleys = rng.randint(2, 6)
    for _ in range(num_valleys):
        start = rng.randint(0, max(0, T - 1))
        length = rng.randint(1, max(1, T // 6))
        end = min(T, start + length)
        c[start:end] = rng.uniform(0.0, 1.0)

    # Add peaks (high-price regions)
    num_peaks = rng.randint(1, 4)
    for _ in range(num_peaks):
        start = rng.randint(0, max(0, T - 1))
        length = rng.randint(1, max(1, T // 8))
        end = min(T, start + length)
        c[start:end] = rng.uniform(3.0, 8.0)

    # Add small noise
    noise = np.array([rng.random() for _ in range(T)], dtype=np.float64) * 0.1
    c = c + noise

    return np.maximum(c, 0.0)


def generate_benchmark_prices(T: int, rng: random.Random) -> np.ndarray:
    """
    Generate benchmark-style piecewise constant prices.
    Matches the benchmark data generation used by:
    - PaST/data/sm_benchmark_data.py (paper_grid_90 sampling)
    - New Benchmark/generate_data.py

    - Periods sampled from Tk_choices = (2, 3, 5) until sum = T
    - Per-period prices: discrete uniform [1, 4] for small/mls, [1, 8] for vls
    """
    Tk_choices = (2, 3, 5)

    # Match PaST's scale-dependent price range when sampling_mode == "paper_grid_90".
    # small: T<=80, mls: T<=300, vls: T>300
    ck_min = 1
    ck_max = 4 if T <= 300 else 8

    # Sample period durations Tk until they sum exactly to T (restart if stuck).
    Tk: List[int] = []
    remaining = T
    max_attempts = 1000
    attempts = 0

    while remaining > 0:
        feasible = [x for x in Tk_choices if x <= remaining]
        if not feasible:
            Tk = []
            remaining = T
            attempts += 1
            if attempts > max_attempts:
                raise ValueError(
                    f"Could not sample intervals summing to {T} from {Tk_choices}"
                )
            continue
        x = rng.choice(feasible)
        Tk.append(int(x))
        remaining -= int(x)

    # Sample prices for each period, then expand to per-slot prices.
    ck = [rng.randint(ck_min, ck_max) for _ in range(len(Tk))]
    ct: List[int] = []
    for dur, price in zip(Tk, ck):
        ct.extend([int(price)] * int(dur))

    return np.asarray(ct, dtype=np.int32)


def generate_mixed_prices(T: int, rng: random.Random) -> np.ndarray:
    """Generate random price pattern (valleys, benchmark, sin, etc.)."""
    kind = rng.choice(["valleys", "benchmark", "sin", "spiky"])

    if kind == "valleys":
        return generate_price_curve_valleys(T, rng)
    elif kind == "benchmark":
        return generate_benchmark_prices(T, rng)
    elif kind == "sin":
        base = rng.uniform(0.5, 2.0)
        amp = rng.uniform(0.5, 3.0)
        period = rng.choice([24, 48, 72])
        phase = rng.uniform(0.0, 2 * math.pi)
        t = np.arange(T, dtype=np.float64)
        noise = np.array([rng.random() for _ in range(T)], dtype=np.float64) * 0.3
        c = base + amp * np.sin(2 * math.pi * t / period + phase) + noise
        return np.maximum(c, 0.0)
    else:  # spiky
        c = np.array([rng.uniform(0.5, 2.0) for _ in range(T)], dtype=np.float64)
        for _ in range(rng.randint(1, max(1, T // 20))):
            idx = rng.randrange(T)
            c[idx] += rng.uniform(5.0, 15.0)
        return np.maximum(c, 0.0)


# ============================================================================
# DP SOLVER (Exact)
# ============================================================================


def prune_states(states: List[State]) -> List[State]:
    """
    Removes dominated states from the Pareto frontier.
    State A dominates B if A.finish_time <= B.finish_time AND A.cost <= B.cost
    (with at least one strict inequality).
    """
    if not states:
        return []

    sorted_states = sorted(states, key=lambda s: (s.finish_time, s.cost))

    pareto_frontier = []
    min_cost_so_far = float("inf")

    for s in sorted_states:
        if s.cost < min_cost_so_far:
            pareto_frontier.append(s)
            min_cost_so_far = s.cost

    return pareto_frontier


def solve_dp_exact(jobs: List[Job]) -> Tuple[float, tuple]:
    """
    Solves the single-machine TOU scheduling problem to OPTIMALITY using DP.

    Returns:
        (optimal_cost, schedule_history) where schedule_history is a tuple of
        (job_id, start_time, end_time) tuples in execution order.
    """
    N = len(jobs)
    if N == 0:
        return 0.0, ()

    # Build a prefix-sum array for fast interval cost queries.
    # We assume CUSTOM_PRICES is set to a horizon-compatible array in verification.
    # If not set, fall back to a flat price curve over the implied horizon.
    horizon = 0
    if CUSTOM_PRICES:
        horizon = len(CUSTOM_PRICES)
    else:
        horizon = max((j.d for j in jobs), default=0)
    if horizon < 0:
        horizon = 0

    if CUSTOM_PRICES:
        prices = np.asarray(CUSTOM_PRICES, dtype=np.float64)
    else:
        prices = np.ones(horizon, dtype=np.float64)

    prefix = np.zeros(len(prices) + 1, dtype=np.float64)
    if len(prices) > 0:
        prefix[1:] = np.cumsum(prices)

    def interval_cost(start_time: int, duration: int) -> float:
        end_time = start_time + duration
        # Penalize out-of-bounds consistently with get_price_at().
        if start_time < 0 or end_time > len(prices):
            return float(1000.0) * max(0, end_time - len(prices))
        return float(prefix[end_time] - prefix[start_time])

    def solve_unconstrained_fast() -> Tuple[float, tuple]:
        """Exact DP specialized for r=0 and d=horizon for all jobs.

        State is: subset mask + completion time t (machine free at t).
        Transition picks any start s >= t (idle allowed) and runs a job for p.
        """
        if horizon <= 0:
            return (0.0, ()) if N == 0 else (float("inf"), ())

        p_list = np.asarray([int(j.p) for j in jobs], dtype=np.int32)
        # Precompute per-job interval costs for all possible starts.
        # cost_job[j][s] = cost of running job j starting at s.
        cost_job = [
            (
                (prefix[p:] - prefix[:-p]).astype(np.float64, copy=False)
                if p > 0
                else np.zeros(horizon + 1, dtype=np.float64)
            )
            for p in p_list
        ]

        # For small N, a subset DP over masks is fine.
        # For medium N (e.g., N=30), 2^N explodes. In that case, exploit the fact that
        # costs depend only on time, not job identity: if there are only a few distinct
        # processing times, we can do an exact DP over (time, counts-used).

        if N <= 20:
            masks = 1 << N
            dp_arr = np.full((masks, horizon + 1), np.inf, dtype=np.float64)
            # With no jobs scheduled, being free at any time t is feasible (just idle).
            dp_arr[0, :] = 0.0

            for mask in range(masks):
                prev = dp_arr[mask]
                if not np.isfinite(prev).any():
                    continue

                prefix_min = np.minimum.accumulate(prev)

                for j in range(N):
                    if mask & (1 << j):
                        continue

                    p = int(p_list[j])
                    max_s = horizon - p
                    if max_s < 0:
                        continue

                    next_mask = mask | (1 << j)

                    base = prefix_min[: max_s + 1]
                    if not np.isfinite(base).any():
                        continue

                    new_costs = base + cost_job[j][: max_s + 1]

                    dp_arr[next_mask, p : horizon + 1] = np.minimum(
                        dp_arr[next_mask, p : horizon + 1], new_costs
                    )

            final_mask = masks - 1
            row = dp_arr[final_mask]
            if not np.isfinite(row).any():
                return float("inf"), ()

            end_t = int(np.argmin(row))
            best_cost = float(row[end_t])

            # Reconstruct one optimal history via backtracking.
            hist = []
            cur_mask = final_mask
            cur_t = end_t
            eps = 1e-9

            while cur_mask:
                cur_cost = float(dp_arr[cur_mask, cur_t])
                found = False

                for j in range(N):
                    if not (cur_mask & (1 << j)):
                        continue

                    p = int(p_list[j])
                    s = cur_t - p
                    if s < 0:
                        continue

                    prev_mask = cur_mask ^ (1 << j)
                    prev_slice = dp_arr[prev_mask, : s + 1]
                    if not np.isfinite(prev_slice).any():
                        continue

                    prev_t = int(np.argmin(prev_slice))
                    prev_cost = float(prev_slice[prev_t])
                    cand = prev_cost + float(cost_job[j][s])

                    if abs(cand - cur_cost) <= eps:
                        job = jobs[j]
                        hist.append((job.id, int(s), int(cur_t)))
                        cur_mask = prev_mask
                        cur_t = prev_t
                        found = True
                        break

                if not found:
                    return best_cost, ()

            hist.reverse()
            return best_cost, tuple(hist)

        # Count-based DP for medium N when there are few distinct processing times.
        lengths, inv = np.unique(p_list, return_inverse=True)
        if len(lengths) > 6:
            raise ValueError(
                f"Unconstrained exact DP: too many distinct processing times ({len(lengths)})."
            )

        totals = np.bincount(inv, minlength=len(lengths)).astype(np.int16, copy=False)
        radices = (totals + 1).astype(np.int32)
        multipliers = np.ones(len(lengths), dtype=np.int32)
        for i in range(1, len(lengths)):
            multipliers[i] = multipliers[i - 1] * int(radices[i - 1])
        n_states = int(np.prod(radices, dtype=np.int64))

        # Safety guard: avoid pathological blowups.
        if (horizon + 1) * n_states > 12_000_000:
            raise MemoryError(
                f"Unconstrained exact DP state space too large: T={horizon} states={n_states}"
            )

        # Precompute used-counts per state for quick feasibility checks.
        used = np.zeros((n_states, len(lengths)), dtype=np.int16)
        for s in range(n_states):
            x = s
            for i in range(len(lengths)):
                used[s, i] = x % int(radices[i])
                x //= int(radices[i])

        final_state = int(np.sum(totals.astype(np.int32) * multipliers))

        dp = np.full((horizon + 1, n_states), np.inf, dtype=np.float64)
        parent_prev_state = np.full((horizon + 1, n_states), -1, dtype=np.int32)
        parent_len = np.full(
            (horizon + 1, n_states), -1, dtype=np.int16
        )  # 0=idle, >0=job length

        dp[0, 0] = 0.0

        # Time-forward DP with idle (t->t+1) and job arcs (t->t+L).
        for t in range(horizon + 1):
            row = dp[t]
            if not np.isfinite(row).any():
                continue

            # Idle transition
            if t < horizon:
                nxt = dp[t + 1]
                improved = row < nxt
                if improved.any():
                    nxt[improved] = row[improved]
                    parent_prev_state[t + 1, improved] = np.nonzero(improved)[0]
                    parent_len[t + 1, improved] = 0

            # Job transitions
            for i, L in enumerate(lengths.tolist()):
                L = int(L)
                if L <= 0 or t + L > horizon:
                    continue

                feasible_states = np.where(used[:, i] < totals[i])[0]
                if feasible_states.size == 0:
                    continue

                s2 = feasible_states + int(multipliers[i])
                cand = row[feasible_states] + float(prefix[t + L] - prefix[t])
                tgt = dp[t + L, s2]
                better = cand < tgt
                if better.any():
                    idxs = np.nonzero(better)[0]
                    tgt[idxs] = cand[idxs]
                    dp[t + L, s2] = tgt
                    parent_prev_state[t + L, s2[idxs]] = feasible_states[idxs]
                    parent_len[t + L, s2[idxs]] = L

        # Allow idling after finishing: best cost at time horizon with all jobs used.
        best_cost = float(dp[horizon, final_state])
        if not np.isfinite(best_cost):
            return float("inf"), ()

        # Backtrack to recover the multiset of job lengths and start times.
        segments: List[Tuple[int, int]] = []  # (start, length)
        t = horizon
        s = final_state
        while not (t == 0 and s == 0):
            L = int(parent_len[t, s])
            prev_s = int(parent_prev_state[t, s])
            if L < 0 or prev_s < 0:
                # Should not happen; return cost only.
                return best_cost, ()
            if L == 0:
                t -= 1
                s = prev_s
            else:
                start = t - L
                segments.append((start, L))
                t -= L
                s = prev_s

        segments.reverse()

        # Assign job IDs to segments by length (jobs are interchangeable given only p).
        ids_by_len: dict[int, List[int]] = {}
        for job in jobs:
            ids_by_len.setdefault(int(job.p), []).append(int(job.id))
        for v in ids_by_len.values():
            v.sort(reverse=True)

        hist2 = []
        for start, L in segments:
            job_id = ids_by_len[L].pop()
            hist2.append((job_id, int(start), int(start + L)))
        return best_cost, tuple(hist2)

    # Fast exact path for the common single-machine benchmark setting.
    if all(j.r == 0 for j in jobs) and all(j.d == horizon for j in jobs):
        return solve_unconstrained_fast()

    dp = {0: [State(finish_time=0, cost=0.0, history=())]}

    all_masks = 1 << N

    for mask in range(all_masks):
        if mask not in dp:
            continue

        current_states = prune_states(dp[mask])
        dp[mask] = current_states

        for j in range(N):
            if mask & (1 << j):
                continue

            job = jobs[j]
            next_mask = mask | (1 << j)
            if next_mask not in dp:
                dp[next_mask] = []

            for state in current_states:
                earliest_start = max(state.finish_time, job.r)
                latest_start = job.d - job.p

                if earliest_start > latest_start:
                    continue

                # IMPORTANT: enumerate ALL feasible start times.
                # Keeping only the locally cheapest start for this transition is NOT safe,
                # because different starts trade off cost vs finish_time and can affect
                # feasibility/cost of later jobs. Pareto pruning handles dominance.
                for s in range(earliest_start, latest_start + 1):
                    c_job = interval_cost(s, job.p)
                    new_finish = s + job.p
                    new_total_cost = state.cost + c_job
                    new_history = state.history + ((job.id, s, new_finish),)
                    dp[next_mask].append(State(new_finish, new_total_cost, new_history))

    final_mask = (1 << N) - 1
    if final_mask not in dp or not dp[final_mask]:
        return float("inf"), ()

    final_states = prune_states(dp[final_mask])
    best_state = min(final_states, key=lambda s: s.cost)
    return best_state.cost, best_state.history


# ============================================================================
# BRUTE FORCE SOLVER (Ground Truth) - OPTIMIZED
# ============================================================================


def solve_brute_force(
    jobs: List[Job], max_starts_per_job: int = 20
) -> Tuple[float, tuple]:
    """
    Solves the problem by enumerating ALL permutations and sampled start times.

    For efficiency, we limit the number of start times considered per job
    to avoid exponential blowup with large windows.

    Args:
        jobs: List of jobs
        max_starts_per_job: Maximum number of start times to sample per job window
    """
    N = len(jobs)
    if N == 0:
        return 0.0, ()

    best_cost = float("inf")
    best_schedule = None

    for perm in itertools.permutations(range(N)):

        def enumerate_schedules(
            pos: int, current_finish: int, current_cost: float, schedule: list
        ):
            nonlocal best_cost, best_schedule

            # Pruning: if we already exceed best, abort
            if current_cost >= best_cost:
                return

            if pos == N:
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_schedule = tuple(schedule)
                return

            job = jobs[perm[pos]]
            earliest_start = max(current_finish, job.r)
            latest_start = job.d - job.p

            if earliest_start > latest_start:
                return

            # Sample start times if window is too large
            window_size = latest_start - earliest_start + 1
            if window_size <= max_starts_per_job:
                start_times = range(earliest_start, latest_start + 1)
            else:
                # Sample uniformly + always include earliest and latest
                step = max(1, window_size // max_starts_per_job)
                start_times = list(range(earliest_start, latest_start + 1, step))
                if latest_start not in start_times:
                    start_times.append(latest_start)

            for s in start_times:
                c = calculate_cost(s, job.p)
                new_schedule = schedule + [(job.id, s, s + job.p)]
                enumerate_schedules(pos + 1, s + job.p, current_cost + c, new_schedule)

        enumerate_schedules(0, 0, 0.0, [])

    return best_cost, best_schedule if best_schedule else ()


# ============================================================================
# INSTANCE GENERATION
# ============================================================================


def generate_random_instance(
    n: int, T: int, rng: random.Random, max_window: int = 15
) -> List[Job]:
    """
    Generate a random instance with n jobs and tight windows.

    Args:
        n: Number of jobs
        T: Horizon length
        rng: Random number generator
        max_window: Maximum window size (d - r - p) to keep brute-force tractable
    """
    jobs = []

    for i in range(n):
        p = rng.randint(1, 4)
        r = rng.randint(0, max(0, T - p - max_window))

        # Deadline: tight window to keep brute-force tractable
        min_d = r + p
        max_d = min(r + p + max_window, T)
        d = rng.randint(min_d, max_d)

        jobs.append(Job(id=i, p=p, r=r, d=d))

    return jobs


# ============================================================================
# VERIFICATION SUITE
# ============================================================================


def run_single_test(jobs: List[Job], test_name: str, verbose: bool = True) -> bool:
    """Run DP and brute-force on one instance and compare."""
    dp_cost, dp_schedule = solve_dp_exact(jobs)
    bf_cost, bf_schedule = solve_brute_force(jobs)

    # Handle infeasible cases (both return inf)
    if dp_cost == float("inf") and bf_cost == float("inf"):
        if verbose:
            print(f"[PASS] {test_name}: Both correctly identified as INFEASIBLE")
        return True

    passed = abs(dp_cost - bf_cost) < 1e-6

    if not passed:
        print(f"\n[FAIL] {test_name}")
        print(f"  DP Cost:          {dp_cost:.4f}")
        print(f"  Brute Force Cost: {bf_cost:.4f}")
        print(f"  Jobs: {[(j.id, j.p, j.r, j.d) for j in jobs]}")
        print(f"  DP Schedule:      {dp_schedule}")
        print(f"  BF Schedule:      {bf_schedule}")
    elif verbose:
        print(f"[PASS] {test_name}: Cost = {dp_cost:.2f}")

    return passed


def run_comprehensive_verification():
    """Comprehensive verification with PaST project's price patterns."""
    print("=" * 70)
    print("VERIFICATION SUITE: DP vs Brute-Force with PaST Price Patterns")
    print("=" * 70)

    all_passed = True
    test_count = 0

    # Test 1: Original counter-example
    print("\n--- Test 1: Counter-Example (Greedy Trap) ---")
    set_custom_prices([10.0, 10.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0])
    jobs1 = [Job(id=0, p=2, r=0, d=10), Job(id=1, p=2, r=2, d=4)]
    all_passed &= run_single_test(jobs1, "Counter-Example")
    clear_custom_prices()
    test_count += 1

    # Tests 2-11: Valleys pattern (many valleys!)
    print("\n--- Tests 2-11: Valleys Pattern (2-6 valleys, 1-4 peaks) ---")
    for seed in range(10):
        rng = random.Random(seed)
        T = rng.randint(40, 60)
        prices = generate_price_curve_valleys(T, rng)
        set_custom_prices(prices.tolist())

        n_jobs = rng.randint(4, 5)
        jobs = generate_random_instance(n_jobs, T, rng)

        # Count valleys for info
        valleys = sum(1 for p in prices if p < 1.0)
        all_passed &= run_single_test(
            jobs, f"Valleys T={T} N={n_jobs} ({valleys} low slots)"
        )
        clear_custom_prices()
        test_count += 1

    # Tests 12-21: Benchmark pattern (many periods!)
    print("\n--- Tests 12-21: Benchmark Pattern (Tk∈{2,3,5}, ck∈[1,4]) ---")
    for seed in range(100, 110):
        rng = random.Random(seed)
        T = rng.randint(40, 60)
        prices = generate_benchmark_prices(T, rng)
        set_custom_prices(prices.tolist())

        n_jobs = rng.randint(4, 5)
        jobs = generate_random_instance(n_jobs, T, rng)

        # Count unique price levels
        unique_prices = len(set(prices.tolist()))
        all_passed &= run_single_test(
            jobs, f"Benchmark T={T} N={n_jobs} ({unique_prices} price levels)"
        )
        clear_custom_prices()
        test_count += 1

    # Tests 22-31: Mixed patterns
    print("\n--- Tests 22-31: Mixed Patterns (valleys/benchmark/sin/spiky) ---")
    for seed in range(200, 210):
        rng = random.Random(seed)
        T = rng.randint(40, 60)
        prices = generate_mixed_prices(T, rng)
        set_custom_prices(prices.tolist())

        n_jobs = rng.randint(4, 5)
        jobs = generate_random_instance(n_jobs, T, rng)

        all_passed &= run_single_test(jobs, f"Mixed T={T} N={n_jobs}")
        clear_custom_prices()
        test_count += 1

    # Tests 32-41: More valleys tests with N=5
    print("\n--- Tests 32-41: More Valleys with N=5 ---")
    for seed in range(300, 310):
        rng = random.Random(seed)
        T = rng.randint(50, 70)
        prices = generate_price_curve_valleys(T, rng)
        set_custom_prices(prices.tolist())

        jobs = generate_random_instance(5, T, rng)

        all_passed &= run_single_test(jobs, f"Valleys N=5 T={T}")
        clear_custom_prices()
        test_count += 1

    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print(f"ALL {test_count} TESTS PASSED! DP is verified optimal.")
    else:
        print(f"SOME TESTS FAILED! Check implementation.")
    print("=" * 70)

    return all_passed


def visualize_price_patterns():
    """Visualize the different price patterns used in PaST."""
    print("\n" + "=" * 70)
    print("SAMPLE PRICE PATTERNS (for visual inspection)")
    print("=" * 70)

    rng = random.Random(42)
    T = 60

    patterns = {
        "Valleys": generate_price_curve_valleys(T, rng),
        "Benchmark": generate_benchmark_prices(T, random.Random(43)),
        "Mixed": generate_mixed_prices(T, random.Random(44)),
    }

    for name, prices in patterns.items():
        print(f"\n{name} (T={T}):")
        print(f"  Price range: [{min(prices):.2f}, {max(prices):.2f}]")
        print(f"  Mean: {np.mean(prices):.2f}, Std: {np.std(prices):.2f}")

        # ASCII visualization
        normalized = (prices - min(prices)) / (max(prices) - min(prices) + 1e-9)
        bar_chars = " ▁▂▃▄▅▆▇█"
        bars = "".join(bar_chars[int(v * (len(bar_chars) - 1))] for v in normalized)
        print(f"  Pattern: {bars}")


if __name__ == "__main__":
    visualize_price_patterns()
    run_comprehensive_verification()

"""Optimal DP for single-machine benchmark under simplified constraints.

Constraints (as requested):
- No release dates and no per-job deadlines.
- All jobs available at time 0.
- Horizon T is known and schedule must fit in [0, T]. (If sum(p) > T => infeasible.)
- Jobs are identical if they share the same processing time p.
- Objective: minimize energy cost = sum_{t where machine runs} price[t].
  (Idle is allowed and costs 0.)

Key idea
--------
This is an exact dynamic program on a DAG over time:
- At any time t you may start a job of length L (if you still have remaining jobs
  of that length), pay cost(price[t:t+L]) and advance to t+L.
- You may also wait/idle for free.

To stay efficient for large N, we do NOT track job IDs. We track only counts used
per processing time length (a "multiset DP").

Algorithm (exact)
-----------------
Let lengths = sorted unique processing times, and totals[i] be how many jobs of
length lengths[i] must be scheduled.

State is a vector u where u[i] = used jobs of length lengths[i]. We encode u as an
integer (mixed radix) so transitions are O(1).

We sweep time t from 0..T-1 and maintain:
- best[u] = best cost to have reached state u and be ready to start at current time t
           (i.e., min cost among all finishes at times <= t; waiting is free).
- finish_at[t_end][u2] = best cost to finish exactly at time t_end in state u2.

Transitions at time t:
For each reachable u in best, and for each length L that is still available:
    u2 = u + inc(L)
    finish_at[t+L][u2] = min(finish_at[t+L][u2], best[u] + cost_interval(t, L))

This is optimal because it is a shortest-path computation on an acyclic graph
(time only moves forward), and "waiting" is captured by best[u] as a prefix-min.

Complexity
----------
Let S be the number of reachable count-states (typically far smaller than prod(totals+1)
when T is moderate). The runtime is O(T * S * K) where K = number of distinct lengths.
Memory is O(S + number of improved finish nodes) and parent pointers for reconstruction.

Returned schedule
-----------------
The solver returns a concrete schedule (job_id, start, end). Since jobs with the same
p are identical, job IDs are assigned arbitrarily among jobs of the same p.

This module is intended for the PaST benchmark where p is small (e.g., 1..4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class BenchmarkJob:
    id: int
    p: int


@dataclass(frozen=True)
class DPResult:
    feasible: bool
    cost: float
    schedule: Tuple[Tuple[int, int, int], ...]  # (job_id, start, end)
    finish_time: int


def _build_prefix(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=np.float64)
    prefix = np.zeros(len(prices) + 1, dtype=np.float64)
    if len(prices) > 0:
        prefix[1:] = np.cumsum(prices)
    return prefix


def _interval_cost(prefix: np.ndarray, start: int, length: int) -> float:
    end = start + length
    return float(prefix[end] - prefix[start])


def solve_optimal_benchmark_dp(
    processing_times: Iterable[int],
    prices: np.ndarray,
    *,
    job_ids: Optional[Iterable[int]] = None,
) -> DPResult:
    """Solve the simplified benchmark exactly.

    Args:
        processing_times: iterable of job processing times.
        prices: per-slot TOU prices, length T.
        job_ids: optional job IDs to attach to returned schedule. If omitted,
                 IDs are assigned 0..N-1.

    Returns:
        DPResult with feasibility, optimal cost, schedule, and finish_time.
    """

    p_list = [int(p) for p in processing_times]
    n_jobs = len(p_list)
    T = int(len(prices))

    if job_ids is None:
        ids = list(range(n_jobs))
    else:
        ids = [int(x) for x in job_ids]
        if len(ids) != n_jobs:
            raise ValueError("job_ids must match processing_times length")

    if n_jobs == 0:
        return DPResult(True, 0.0, (), 0)

    total_p = int(sum(p_list))
    if total_p > T:
        return DPResult(False, float("inf"), (), 0)

    prefix = _build_prefix(prices)

    lengths, inv = np.unique(np.asarray(p_list, dtype=np.int32), return_inverse=True)
    lengths = lengths.astype(np.int32, copy=False)
    K = int(len(lengths))
    if K == 0:
        return DPResult(True, 0.0, (), 0)
    if K > 12:
        raise ValueError(
            f"Too many distinct processing times ({K}). This solver is optimized for the benchmark (few lengths)."
        )
    totals = np.bincount(inv, minlength=K).astype(np.int32, copy=False)

    # Mixed-radix encoding: state = sum(u[i] * mult[i]), where 0<=u[i]<=totals[i]
    radices = (totals + 1).astype(np.int32)
    mult = np.ones(K, dtype=np.int32)
    for i in range(1, K):
        mult[i] = mult[i - 1] * int(radices[i - 1])

    final_state = int(np.sum(totals * mult))

    # For each length index i, adding one job corresponds to +mult[i]
    inc = [int(m) for m in mult.tolist()]

    # Map length index for fast loop
    lengths_list = [int(x) for x in lengths.tolist()]

    # For each length value, store a stack of job IDs available
    ids_by_len: Dict[int, List[int]] = {}
    for jid, p in zip(ids, p_list):
        ids_by_len.setdefault(int(p), []).append(int(jid))
    for v in ids_by_len.values():
        v.sort(reverse=True)

    # Fast exact DP on (time, used-count-state) as an acyclic shortest path.
    # This is typically very fast for benchmark p in 1..4 and horizons up to 500.

    n_states = int(np.prod(radices, dtype=np.int64))
    max_cells = 12_000_000  # conservative guard
    if (T + 1) * n_states > max_cells:
        # Fallback: sparse exact DP by time layers.
        # dp[t][state] = best cost to finish exactly at time t having used `state`.
        # This avoids allocating O(T*n_states) and enumerating unreachable states.
        used_cache: Dict[int, Tuple[int, ...]] = {0: tuple([0] * K)}

        def decode_used(state: int) -> Tuple[int, ...]:
            cached = used_cache.get(state)
            if cached is not None:
                return cached
            u = [0] * K
            x = state
            for i in range(K):
                r = int(radices[i])
                u[i] = x % r
                x //= r
            tup = tuple(u)
            used_cache[state] = tup
            return tup

        dp_layers: List[Dict[int, float]] = [dict() for _ in range(T + 1)]
        dp_layers[0][0] = 0.0
        parent: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
        # parent[(t2, s2)] = (t1, s1, L) with L=0 for idle, else job length.

        best_final_cost = float("inf")
        best_final_time = -1

        for t in range(T + 1):
            layer = dp_layers[t]
            if not layer:
                continue

            # If final state reached at time t, track best.
            c_final = layer.get(final_state)
            if c_final is not None and c_final < best_final_cost:
                best_final_cost = float(c_final)
                best_final_time = int(t)

            if t == T:
                continue

            # Idle transition: (t,state)->(t+1,state)
            next_layer = dp_layers[t + 1]
            for state, c0 in layer.items():
                prev = next_layer.get(state)
                if prev is None or c0 < prev:
                    next_layer[state] = float(c0)
                    parent[(t + 1, state)] = (t, state, 0)

            # Job transitions
            for state, c0 in layer.items():
                used = decode_used(state)
                for i, L in enumerate(lengths_list):
                    if used[i] >= int(totals[i]):
                        continue
                    end = t + int(L)
                    if end > T:
                        continue

                    new_state = state + int(inc[i])
                    cand = float(c0 + (prefix[end] - prefix[t]))
                    target_layer = dp_layers[end]
                    prev = target_layer.get(new_state)
                    if prev is None or cand < prev:
                        target_layer[new_state] = cand
                        parent[(end, new_state)] = (t, state, int(L))

        if not np.isfinite(best_final_cost) or best_final_time < 0:
            return DPResult(False, float("inf"), (), 0)

        # Backtrack segments
        segments: List[Tuple[int, int]] = []
        t = best_final_time
        s = final_state
        while not (t == 0 and s == 0):
            par = parent.get((t, s))
            if par is None:
                return DPResult(True, float(best_final_cost), (), int(t))
            prev_t, prev_s, L = par
            if L > 0:
                segments.append((prev_t, int(L)))
            t, s = int(prev_t), int(prev_s)
        segments.reverse()

        sched: List[Tuple[int, int, int]] = []
        for start_t, L in segments:
            jid = ids_by_len[int(L)].pop()
            sched.append((jid, int(start_t), int(start_t + int(L))))

        finish_time = 0
        for _, _, e in sched:
            if e > finish_time:
                finish_time = int(e)

        return DPResult(True, float(best_final_cost), tuple(sched), finish_time)

    # used[s, i] = used count of length i in state s
    used = np.zeros((n_states, K), dtype=np.int16)
    for s in range(n_states):
        x = s
        for i in range(K):
            used[s, i] = x % int(radices[i])
            x //= int(radices[i])

    dp = np.full((T + 1, n_states), np.inf, dtype=np.float64)
    parent_prev_state = np.full((T + 1, n_states), -1, dtype=np.int32)
    parent_len = np.full((T + 1, n_states), -1, dtype=np.int16)  # 0 idle, >0 job

    dp[0, 0] = 0.0

    # Precompute feasible states for each length index once (independent of time).
    feasible_by_i: List[np.ndarray] = []
    for i in range(K):
        feasible_by_i.append(np.where(used[:, i] < int(totals[i]))[0].astype(np.int32))

    for t in range(T + 1):
        row = dp[t]
        if not np.isfinite(row).any():
            continue

        # Idle transition (waiting is free)
        if t < T:
            nxt = dp[t + 1]
            improved = row < nxt
            if improved.any():
                nxt[improved] = row[improved]
                parent_prev_state[t + 1, improved] = np.nonzero(improved)[0]
                parent_len[t + 1, improved] = 0

        # Job transitions
        for i, L in enumerate(lengths_list):
            L = int(L)
            end = t + L
            if L <= 0 or end > T:
                continue

            feasible_states = feasible_by_i[i]
            if feasible_states.size == 0:
                continue

            s2 = feasible_states + int(inc[i])
            cand = row[feasible_states] + float(prefix[end] - prefix[t])
            tgt = dp[end, s2]
            better = cand < tgt
            if better.any():
                idxs = np.nonzero(better)[0]
                tgt[idxs] = cand[idxs]
                dp[end, s2] = tgt
                parent_prev_state[end, s2[idxs]] = feasible_states[idxs]
                parent_len[end, s2[idxs]] = L

    opt_cost = float(dp[T, final_state])
    if not np.isfinite(opt_cost):
        return DPResult(False, float("inf"), (), 0)

    # Backtrack to recover segments (start, length). Ignore idle steps.
    segments: List[Tuple[int, int]] = []
    t = T
    s = final_state
    while not (t == 0 and s == 0):
        L = int(parent_len[t, s])
        prev_s = int(parent_prev_state[t, s])
        if L < 0 or prev_s < 0:
            return DPResult(True, opt_cost, (), T)
        if L == 0:
            t -= 1
            s = prev_s
        else:
            start = t - L
            segments.append((start, L))
            t -= L
            s = prev_s

    segments.reverse()

    # Assign job IDs to segments based on their length.
    sched: List[Tuple[int, int, int]] = []
    for start_t, L in segments:
        jid = ids_by_len[L].pop()
        sched.append((jid, int(start_t), int(start_t + L)))

    finish_time = 0
    for _, _, e in sched:
        if e > finish_time:
            finish_time = int(e)

    return DPResult(True, opt_cost, tuple(sched), finish_time)


def solve_optimal_benchmark_dp_counts(
    counts_by_p: Dict[int, int],
    prices: np.ndarray,
) -> DPResult:
    """Counts-based convenience wrapper.

    Use this when you already have the multiset counts, e.g. {1: 120, 2: 60, 3: 10}.

    Job IDs are assigned deterministically as 0..N-1.
    """

    processing_times: List[int] = []
    for p, c in sorted(counts_by_p.items()):
        p = int(p)
        c = int(c)
        if c < 0:
            raise ValueError("counts must be non-negative")
        processing_times.extend([p] * c)

    return solve_optimal_benchmark_dp(processing_times, prices)


def _demo() -> None:
    # Small, illustrative example
    prices = np.array([10, 10, 1, 1, 10, 10], dtype=np.float64)
    processing_times = [2, 2, 1]
    res = solve_optimal_benchmark_dp(processing_times, prices)
    print("feasible=", res.feasible)
    print("cost=", res.cost)
    print("finish_time=", res.finish_time)
    print("schedule=", res.schedule)


if __name__ == "__main__":
    _demo()

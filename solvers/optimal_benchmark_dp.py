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

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# Import accelerated solver (with fallback)
try:
    from PaST.solvers.optimal_benchmark_dp_numba import (
        solve_sparse_dp_python,
        solve_sparse_dp_python_beam,
    )

    _ACCEL_AVAILABLE = True
except ImportError:
    _ACCEL_AVAILABLE = False


_EPS = 1e-12


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
    is_optimal: bool = True  # False if solution is from greedy completion on timeout
    timed_out: bool = False


def _build_prefix(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=np.float64)
    prefix = np.zeros(len(prices) + 1, dtype=np.float64)
    if len(prices) > 0:
        prefix[1:] = np.cumsum(prices)
    return prefix


def _interval_cost(prefix: np.ndarray, start: int, length: int) -> float:
    end = start + length
    return float(prefix[end] - prefix[start])


def _greedy_complete_schedule(
    remaining_counts: Dict[int, int],  # length -> count remaining
    prices: np.ndarray,
    T: int,
    occupied_slots: Optional[
        List[Tuple[int, int]]
    ] = None,  # (start, end) of already scheduled
    ids_by_len: Optional[Dict[int, List[int]]] = None,  # length -> available job IDs
) -> Tuple[float, List[Tuple[int, int, int]]]:
    """Greedily schedule remaining jobs in cheapest available slots.

    Args:
        remaining_counts: How many jobs of each length still need to be scheduled.
        prices: Price array of length T.
        T: Time horizon.
        occupied_slots: Already scheduled (start, end) pairs to avoid.
        ids_by_len: Job IDs available for each length (will pop from these).

    Returns:
        (total_cost, schedule) where schedule is list of (job_id, start, end).
    """
    if not remaining_counts or all(c == 0 for c in remaining_counts.values()):
        return 0.0, []

    # Build prefix for cost computation
    prefix = _build_prefix(prices)

    # Find free time slots
    occupied = set()
    if occupied_slots:
        for s, e in occupied_slots:
            for t in range(s, e):
                occupied.add(t)

    free_slots = [t for t in range(T) if t not in occupied]

    # Sort free slots by price (ascending) to greedily pick cheapest
    free_slots.sort(key=lambda t: prices[t])

    # Create job list: expand counts to individual jobs
    jobs_to_schedule: List[Tuple[int, int]] = []  # (length, job_id or -1)
    for length, count in remaining_counts.items():
        for _ in range(count):
            if ids_by_len and length in ids_by_len and ids_by_len[length]:
                jid = ids_by_len[length].pop()
            else:
                jid = -1  # placeholder
            jobs_to_schedule.append((length, jid))

    # Sort jobs by length descending (larger jobs first - harder to place)
    jobs_to_schedule.sort(key=lambda x: -x[0])

    total_cost = 0.0
    schedule: List[Tuple[int, int, int]] = []

    for length, jid in jobs_to_schedule:
        # Find first window of 'length' consecutive free slots
        placed = False
        best_start = -1
        best_cost = float("inf")

        # Try each starting position in free slots
        for start_t in range(T - length + 1):
            # Check if all slots [start_t, start_t + length) are free
            if all(t not in occupied for t in range(start_t, start_t + length)):
                cost = float(prefix[start_t + length] - prefix[start_t])
                if cost < best_cost:
                    best_cost = cost
                    best_start = start_t

        if best_start >= 0:
            # Place the job
            for t in range(best_start, best_start + length):
                occupied.add(t)
            total_cost += best_cost
            schedule.append((jid, best_start, best_start + length))
            placed = True

        if not placed:
            # Cannot place this job - infeasible greedy completion
            return float("inf"), []

    return total_cost, schedule


def _solve_sparse_dp_inline(
    lengths_list: List[int],
    totals: np.ndarray,
    prefix: np.ndarray,
    T: int,
    radices: np.ndarray,
    mult: np.ndarray,
    K: int,
    final_state: int,
    tie_break: str = "early",
    time_limit: float = -1.0,
) -> Tuple[float, int, Dict, bool, Optional[Tuple[int, int, float]]]:
    """Inline sparse DP fallback with early pruning and time limit.

    Returns:
        (best_final_cost, best_final_time, parent, timed_out, best_partial)

    best_partial is (time, state, cost) of the best partial state found,
    or None if a complete solution was found.
    """
    import time as time_module

    start_time = time_module.perf_counter()

    totals_arr = totals
    inc = [int(m) for m in mult]
    total_jobs = sum(int(t) for t in totals_arr)

    # State decoding cache
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

    def remaining_work(used: Tuple[int, ...]) -> int:
        work = 0
        for i in range(K):
            work += (int(totals_arr[i]) - used[i]) * lengths_list[i]
        return work

    def count_scheduled(used: Tuple[int, ...]) -> int:
        return sum(used)

    # DP layers
    dp_layers: List[Dict[int, Tuple[float, int]]] = [dict() for _ in range(T + 1)]
    dp_layers[0][0] = (0.0, 0)
    parent: Dict[Tuple[int, int], Tuple[int, int, int]] = {}

    best_final_cost = float("inf")
    best_final_pen = 2**63 - 1
    best_final_time = -1
    timed_out = False

    # Track best partial state: (num_scheduled, cost, time, state)
    # We want max jobs scheduled, then min cost
    best_partial_jobs = 0
    best_partial_cost = float("inf")
    best_partial_time = 0
    best_partial_state = 0

    for t in range(T + 1):
        # Check timeout
        if time_limit > 0 and (time_module.perf_counter() - start_time) > time_limit:
            timed_out = True
            break

        layer = dp_layers[t]
        if not layer:
            continue

        # Update best partial state from this layer
        for state, (cost, pen) in layer.items():
            used = decode_used(state)
            n_scheduled = count_scheduled(used)
            # Better if: more jobs scheduled, or same jobs but lower cost
            if n_scheduled > best_partial_jobs or (
                n_scheduled == best_partial_jobs and cost < best_partial_cost
            ):
                best_partial_jobs = n_scheduled
                best_partial_cost = cost
                best_partial_time = t
                best_partial_state = state

        # Check for final state
        v_final = layer.get(final_state)
        if v_final is not None:
            c_final, p_final = float(v_final[0]), int(v_final[1])
            better = c_final < best_final_cost
            if (
                tie_break == "early"
                and not better
                and abs(c_final - best_final_cost) <= _EPS
            ):
                better = p_final < best_final_pen or (
                    p_final == best_final_pen and t < best_final_time
                )
            if better:
                best_final_cost = c_final
                best_final_pen = p_final
                best_final_time = t

        if t == T:
            continue

        # Idle + job transitions
        next_layer = dp_layers[t + 1]
        for state, (c0, p0) in layer.items():
            # Early pruning
            used = decode_used(state)
            if remaining_work(used) > T - t:
                continue

            # Idle transition
            prev = next_layer.get(state)
            if prev is None:
                next_layer[state] = (float(c0), int(p0))
                parent[(t + 1, state)] = (t, state, 0)
            else:
                c_prev, p_prev = float(prev[0]), int(prev[1])
                better = c0 < c_prev
                if (
                    tie_break == "early"
                    and not better
                    and abs(float(c0) - c_prev) <= _EPS
                ):
                    better = p0 < p_prev
                if better:
                    next_layer[state] = (float(c0), int(p0))
                    parent[(t + 1, state)] = (t, state, 0)

            # Job transitions
            for i, L in enumerate(lengths_list):
                if used[i] >= int(totals_arr[i]):
                    continue
                end = t + L
                if end > T:
                    continue

                new_state = state + inc[i]
                cand_cost = float(c0 + (prefix[end] - prefix[t]))
                cand_pen = int(p0)
                if tie_break == "early":
                    cand_pen += t

                target_layer = dp_layers[end]
                prev = target_layer.get(new_state)
                if prev is None:
                    target_layer[new_state] = (cand_cost, cand_pen)
                    parent[(end, new_state)] = (t, state, L)
                else:
                    prev_cost, prev_pen = float(prev[0]), int(prev[1])
                    better = cand_cost < prev_cost
                    if (
                        tie_break == "early"
                        and not better
                        and abs(cand_cost - prev_cost) <= _EPS
                    ):
                        better = cand_pen < prev_pen
                    if better:
                        target_layer[new_state] = (cand_cost, cand_pen)
                        parent[(end, new_state)] = (t, state, L)

    # Return best partial info if we timed out without finding complete solution
    best_partial = None
    if timed_out and best_final_time < 0 and best_partial_jobs > 0:
        best_partial = (best_partial_time, best_partial_state, best_partial_cost)

    return best_final_cost, best_final_time, parent, timed_out, best_partial


def solve_optimal_benchmark_dp(
    processing_times: Iterable[int],
    prices: np.ndarray,
    *,
    job_ids: Optional[Iterable[int]] = None,
    tie_break: str = "cost",
    time_limit: float = -1.0,
    guided: bool = False,
    beam_width: int = 2000,
    prune_factor: float = 2.0,
    vhat: Optional[callable] = None,
) -> DPResult:
    """Solve the simplified benchmark exactly.

    Args:
        processing_times: iterable of job processing times.
        prices: per-slot TOU prices, length T.
        job_ids: optional job IDs to attach to returned schedule. If omitted,
                 IDs are assigned 0..N-1.
        tie_break: tie-breaking rule among equal-cost optima.
            - "cost": cost only (any optimal schedule)
            - "early": among equal-cost schedules, minimize sum of job start times;
              then minimize finish time.

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

    tie_break = str(tie_break).strip().lower()
    if tie_break not in {"cost", "early"}:
        raise ValueError("tie_break must be 'cost' or 'early'")

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

    # Optional guided approximate solver: beam-pruned sparse DP.
    # Uses lexicographic pruning on (g_cost + vhat, pen) and preserves idle moves.
    if bool(guided):
        if vhat is None:
            raise ValueError(
                "guided=True requires providing vhat(t:int, state:int)->float"
            )
        if not _ACCEL_AVAILABLE:
            raise RuntimeError(
                "guided beam DP requires PaST.solvers.optimal_benchmark_dp_numba to be importable"
            )

        best_final_cost, best_final_time, parent, timed_out, best_partial = (
            solve_sparse_dp_python_beam(
                lengths=lengths,
                totals=totals,
                prefix=prefix,
                T=T,
                radices=radices,
                mult=mult,
                K=K,
                final_state=final_state,
                vhat=vhat,
                beam_width=int(beam_width),
                prune_factor=float(prune_factor),
                time_limit=float(time_limit),
                tie_break=tie_break,
            )
        )

        if np.isfinite(best_final_cost) and best_final_time >= 0:
            segments: List[Tuple[int, int]] = []
            t = best_final_time
            s = final_state
            while not (t == 0 and s == 0):
                par = parent.get((t, s))
                if par is None:
                    return DPResult(
                        True,
                        float(best_final_cost),
                        (),
                        int(t),
                        is_optimal=False,
                        timed_out=bool(timed_out),
                    )
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

            return DPResult(
                True,
                float(best_final_cost),
                tuple(sched),
                int(finish_time),
                is_optimal=False,
                timed_out=bool(timed_out),
            )

        # If beam solver didn't find a complete solution, fall back to the regular exact solver.

    # Fast exact DP on (time, used-count-state) as an acyclic shortest path.
    # This is typically very fast for benchmark p in 1..4 and horizons up to 500.

    n_states = int(np.prod(radices, dtype=np.int64))
    # Dense DP threshold: balance speed vs memory
    # For K <= 6, use higher threshold since vectorized NumPy is much faster
    if K <= 6:
        max_cells = 200_000_000  # 200M cells ≈ 1.6GB (acceptable for modern machines)
    else:
        max_cells = (
            12_000_000  # 12M cells ≈ 96MB (conservative for many unique lengths)
        )

    # Use sparse DP only if K > 6 (many unique lengths) AND state space exceeds limit
    use_sparse = K > 6 and (T + 1) * n_states > max_cells
    if use_sparse:
        # Fallback: use optimized sparse DP with early pruning and time limit.
        # This is much faster than the inline version due to early feasibility checks.
        if _ACCEL_AVAILABLE:
            best_final_cost, best_final_time, parent, timed_out, best_partial = (
                solve_sparse_dp_python(
                    lengths=lengths,
                    totals=totals,
                    prefix=prefix,
                    T=T,
                    radices=radices,
                    mult=mult,
                    K=K,
                    final_state=final_state,
                    time_limit=time_limit,
                    tie_break=tie_break,
                )
            )
        else:
            # Inline fallback if accelerated module not available
            best_final_cost, best_final_time, parent, timed_out, best_partial = (
                _solve_sparse_dp_inline(
                    lengths_list=[int(x) for x in lengths.tolist()],
                    totals=totals,
                    prefix=prefix,
                    T=T,
                    radices=radices,
                    mult=mult,
                    K=K,
                    final_state=final_state,
                    tie_break=tie_break,
                    time_limit=time_limit,
                )
            )

        if np.isfinite(best_final_cost) and best_final_time >= 0:
            # Backtrack segments
            segments: List[Tuple[int, int]] = []
            t = best_final_time
            s = final_state
            while not (t == 0 and s == 0):
                par = parent.get((t, s))
                if par is None:
                    return DPResult(
                        True,
                        float(best_final_cost),
                        (),
                        int(t),
                        is_optimal=True,
                        timed_out=False,
                    )
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

            return DPResult(
                True,
                float(best_final_cost),
                tuple(sched),
                finish_time,
                is_optimal=True,
                timed_out=False,
            )

        # Handle timeout with greedy completion for sparse DP
        if timed_out and best_partial is not None:
            partial_time, partial_state, partial_cost = best_partial

            # Decode partial state to get used counts
            def decode_state(state: int) -> Tuple[int, ...]:
                u = [0] * K
                x = state
                for i in range(K):
                    r = int(radices[i])
                    u[i] = x % r
                    x //= r
                return tuple(u)

            used_counts = decode_state(partial_state)

            # Backtrack to get partial schedule
            segments = []
            t = partial_time
            s = partial_state
            while not (t == 0 and s == 0):
                par = parent.get((t, s))
                if par is None:
                    break
                prev_t, prev_s, L = par
                if L > 0:
                    segments.append((prev_t, int(L)))
                t, s = int(prev_t), int(prev_s)
            segments.reverse()

            # Build partial schedule
            partial_sched: List[Tuple[int, int, int]] = []
            for start_t, L in segments:
                if L in ids_by_len and ids_by_len[L]:
                    jid = ids_by_len[L].pop()
                    partial_sched.append((jid, int(start_t), int(start_t + int(L))))

            # Compute remaining jobs
            remaining_counts: Dict[int, int] = {}
            for i, length in enumerate(lengths_list):
                remaining = int(totals[i]) - int(used_counts[i])
                if remaining > 0:
                    remaining_counts[length] = remaining

            # Get occupied slots from partial schedule
            occupied_slots = [(s_t, e_t) for _, s_t, e_t in partial_sched]

            # Greedy completion
            greedy_cost, greedy_sched = _greedy_complete_schedule(
                remaining_counts,
                prices,
                T,
                occupied_slots,
                ids_by_len,
            )

            if np.isfinite(greedy_cost):
                # Combine schedules
                full_sched = partial_sched + greedy_sched
                total_cost = partial_cost + greedy_cost
                finish_time = max((e for _, _, e in full_sched), default=0)

                return DPResult(
                    True,
                    total_cost,
                    tuple(full_sched),
                    finish_time,
                    is_optimal=False,
                    timed_out=True,
                )

        return DPResult(
            False, float("inf"), (), 0, is_optimal=False, timed_out=timed_out
        )

    # used[s, i] = used count of length i in state s
    used = np.zeros((n_states, K), dtype=np.int16)
    for s in range(n_states):
        x = s
        for i in range(K):
            used[s, i] = x % int(radices[i])
            x //= int(radices[i])

    dp = np.full((T + 1, n_states), np.inf, dtype=np.float64)
    dp_pen = np.full((T + 1, n_states), np.iinfo(np.int32).max, dtype=np.int32)
    parent_prev_state = np.full((T + 1, n_states), -1, dtype=np.int32)
    parent_len = np.full((T + 1, n_states), -1, dtype=np.int16)  # 0 idle, >0 job

    dp[0, 0] = 0.0
    dp_pen[0, 0] = 0

    # Precompute feasible states for each length index once (independent of time).
    feasible_by_i: List[np.ndarray] = []
    for i in range(K):
        feasible_by_i.append(np.where(used[:, i] < int(totals[i]))[0].astype(np.int32))

    # Time limit support
    import time as time_module

    dp_start_time = time_module.perf_counter()
    timed_out = False
    last_processed_t = -1

    for t in range(T + 1):
        # Check time limit
        if time_limit > 0 and (time_module.perf_counter() - dp_start_time) > time_limit:
            timed_out = True
            break
        last_processed_t = t

        row = dp[t]
        row_pen = dp_pen[t]
        if not np.isfinite(row).any():
            continue

        # Idle transition (waiting is free)
        if t < T:
            nxt = dp[t + 1]
            nxt_pen = dp_pen[t + 1]

            improved = row < nxt
            if tie_break == "early":
                eq = np.isclose(row, nxt, rtol=0.0, atol=_EPS)
                improved = improved | (eq & (row_pen < nxt_pen))

            if improved.any():
                idxs = np.nonzero(improved)[0]
                nxt[idxs] = row[idxs]
                nxt_pen[idxs] = row_pen[idxs]
                parent_prev_state[t + 1, idxs] = idxs
                parent_len[t + 1, idxs] = 0

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
            cand_pen = row_pen[feasible_states]
            if tie_break == "early":
                cand_pen = cand_pen + int(t)

            tgt = dp[end, s2]
            tgt_pen = dp_pen[end, s2]
            better = cand < tgt
            if tie_break == "early":
                eq = np.isclose(cand, tgt, rtol=0.0, atol=_EPS)
                better = better | (eq & (cand_pen < tgt_pen))

            if better.any():
                idxs = np.nonzero(better)[0]
                tgt[idxs] = cand[idxs]
                dp[end, s2] = tgt
                dp_pen[end, s2[idxs]] = cand_pen[idxs]
                parent_prev_state[end, s2[idxs]] = feasible_states[idxs]
                parent_len[end, s2[idxs]] = L

    # Choose best finishing time for the final state.
    col = dp[:, final_state]

    # Helper function to count scheduled jobs from state
    def count_jobs_in_state(state: int) -> int:
        return int(np.sum(used[state, :]))

    if np.isfinite(col).any() and not timed_out:
        # Found complete solution
        opt_cost = float(np.min(col))
        best_t_candidates = np.where(np.isclose(col, opt_cost, rtol=0.0, atol=_EPS))[0]
        if tie_break == "early":
            pens = dp_pen[best_t_candidates, final_state]
            min_pen = int(np.min(pens))
            best_t_candidates = best_t_candidates[pens == min_pen]
        best_t = int(np.min(best_t_candidates))

        # Backtrack to recover segments (start, length). Ignore idle steps.
        segments: List[Tuple[int, int]] = []
        t = best_t
        s = final_state
        while not (t == 0 and s == 0):
            L = int(parent_len[t, s])
            prev_s = int(parent_prev_state[t, s])
            if L < 0 or prev_s < 0:
                return DPResult(
                    True, opt_cost, (), int(best_t), is_optimal=True, timed_out=False
                )
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

        return DPResult(
            True, opt_cost, tuple(sched), finish_time, is_optimal=True, timed_out=False
        )

    # Timed out or no complete solution found - try greedy completion
    if timed_out:
        # Find best partial state: most jobs scheduled at lowest cost
        best_partial_jobs = 0
        best_partial_cost = float("inf")
        best_partial_time = 0
        best_partial_state = 0

        # Search through all explored layers up to last_processed_t
        for t_search in range(min(last_processed_t + 1, T + 1)):
            row_search = dp[t_search]
            finite_mask = np.isfinite(row_search)
            if not finite_mask.any():
                continue
            finite_states = np.where(finite_mask)[0]
            for s in finite_states:
                n_jobs = count_jobs_in_state(s)
                cost = float(row_search[s])
                if n_jobs > best_partial_jobs or (
                    n_jobs == best_partial_jobs and cost < best_partial_cost
                ):
                    best_partial_jobs = n_jobs
                    best_partial_cost = cost
                    best_partial_time = t_search
                    best_partial_state = s

        if best_partial_jobs > 0:
            # Backtrack to get partial schedule
            segments = []
            t = best_partial_time
            s = best_partial_state
            while not (t == 0 and s == 0):
                L = int(parent_len[t, s])
                prev_s = int(parent_prev_state[t, s])
                if L < 0 or prev_s < 0:
                    break
                if L == 0:
                    t -= 1
                    s = prev_s
                else:
                    start = t - L
                    segments.append((start, L))
                    t -= L
                    s = prev_s
            segments.reverse()

            # Build partial schedule
            partial_sched: List[Tuple[int, int, int]] = []
            scheduled_used = used[best_partial_state, :].copy()

            for start_t, L in segments:
                if L in ids_by_len and ids_by_len[L]:
                    jid = ids_by_len[L].pop()
                    partial_sched.append((jid, int(start_t), int(start_t + L)))

            # Compute remaining jobs
            remaining_counts: Dict[int, int] = {}
            for i, length in enumerate(lengths_list):
                remaining = int(totals[i]) - int(scheduled_used[i])
                if remaining > 0:
                    remaining_counts[length] = remaining

            # Get occupied slots from partial schedule
            occupied_slots = [(s, e) for _, s, e in partial_sched]

            # Greedy completion
            greedy_cost, greedy_sched = _greedy_complete_schedule(
                remaining_counts,
                prices,
                T,
                occupied_slots,
                ids_by_len,
            )

            if np.isfinite(greedy_cost):
                # Combine schedules
                full_sched = partial_sched + greedy_sched
                total_cost = best_partial_cost + greedy_cost
                finish_time = max((e for _, _, e in full_sched), default=0)

                return DPResult(
                    True,
                    total_cost,
                    tuple(full_sched),
                    finish_time,
                    is_optimal=False,
                    timed_out=True,
                )

    return DPResult(False, float("inf"), (), 0, is_optimal=False, timed_out=timed_out)


def solve_optimal_benchmark_dp_counts(
    counts_by_p: Dict[int, int],
    prices: np.ndarray,
    *,
    tie_break: str = "cost",
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

    return solve_optimal_benchmark_dp(processing_times, prices, tie_break=tie_break)


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

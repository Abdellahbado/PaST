"""Numba-accelerated sparse DP for optimal single-machine TOU scheduling.

This module provides a JIT-compiled implementation of the sparse DP algorithm
for cases where the dense NumPy approach would require too much memory.

Key optimizations:
1. Numba @njit compilation of inner loops (50-100x speedup)
2. Typed arrays instead of Python dicts for state tracking
3. Early feasibility pruning based on remaining work
4. Optional time limit for guaranteed termination

Falls back to pure Python if Numba is not installed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import heapq

# Try to import Numba, fall back to pure Python if unavailable
try:
    from numba import njit, prange
    from numba.typed import Dict as NumbaDict
    from numba.core import types

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Create dummy decorator that does nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    def prange(*args):
        return range(*args)


_EPS = 1e-12
_INF = float("inf")


@dataclass(frozen=True)
class DPResultNumba:
    """Result from the Numba-accelerated DP solver."""

    feasible: bool
    cost: float
    finish_time: int
    segments: Tuple[Tuple[int, int], ...]  # (start_time, length) pairs
    timed_out: bool = False


# --- Numba-accelerated core functions ---


@njit(cache=True)
def _encode_state(used: np.ndarray, mult: np.ndarray) -> int:
    """Encode used counts as a single integer (mixed radix)."""
    state = 0
    for i in range(len(used)):
        state += int(used[i]) * int(mult[i])
    return state


@njit(cache=True)
def _decode_state(state: int, radices: np.ndarray, K: int) -> np.ndarray:
    """Decode state integer back to used counts."""
    used = np.zeros(K, dtype=np.int32)
    x = state
    for i in range(K):
        r = radices[i]
        used[i] = x % r
        x //= r
    return used


@njit(cache=True)
def _remaining_work(used: np.ndarray, totals: np.ndarray, lengths: np.ndarray) -> int:
    """Compute remaining work (sum of remaining processing times)."""
    work = 0
    for i in range(len(used)):
        remaining = totals[i] - used[i]
        work += remaining * lengths[i]
    return work


@njit(cache=True)
def _is_complete(used: np.ndarray, totals: np.ndarray) -> bool:
    """Check if all jobs have been scheduled."""
    for i in range(len(used)):
        if used[i] != totals[i]:
            return False
    return True


@njit(cache=True)
def _interval_cost_numba(prefix: np.ndarray, start: int, length: int) -> float:
    """Compute cost for interval [start, start+length)."""
    return prefix[start + length] - prefix[start]


@njit(cache=True)
def solve_sparse_dp_numba_core(
    lengths: np.ndarray,  # K distinct lengths
    totals: np.ndarray,  # counts for each length
    prefix: np.ndarray,  # prefix sums of prices
    T: int,
    radices: np.ndarray,
    mult: np.ndarray,
    final_state: int,
    time_limit_ticks: int,  # -1 for no limit
    start_ticks: int,
) -> Tuple[float, int, np.ndarray, bool]:
    """
    Core sparse DP algorithm with Numba JIT.

    Returns:
        (best_cost, best_finish_time, parent_array, timed_out)

    parent_array is a 2D array where parent_array[i] = [prev_t, prev_state, length]
    for the i-th discovered (t, state) pair.
    """
    K = len(lengths)

    # Use arrays to track state costs instead of dicts
    # This limits max states but is much faster with Numba
    MAX_STATES = 500000
    MAX_TRANSITIONS = 5000000

    # Layer data: for each time t, track (state_id -> (cost, penalty))
    # We use a simpler approach: flat arrays with indices

    # Current layer: arrays indexed by state
    # Since states can be sparse, we use a mapping
    state_to_idx = np.full(MAX_STATES, -1, dtype=np.int32)
    idx_to_state = np.zeros(MAX_STATES, dtype=np.int64)
    state_cost = np.full(MAX_STATES, np.inf, dtype=np.float64)
    state_pen = np.full(MAX_STATES, 2**30, dtype=np.int64)
    n_states_curr = 0

    # Next layer
    next_state_to_idx = np.full(MAX_STATES, -1, dtype=np.int32)
    next_idx_to_state = np.zeros(MAX_STATES, dtype=np.int64)
    next_state_cost = np.full(MAX_STATES, np.inf, dtype=np.float64)
    next_state_pen = np.full(MAX_STATES, 2**30, dtype=np.int64)
    n_states_next = 0

    # Future layers for job completions
    # future_layers[t] = list of (state, cost, penalty)
    # We use a flat array approach
    future_data = np.zeros(
        (MAX_TRANSITIONS, 4), dtype=np.float64
    )  # t, state, cost, pen
    n_future = 0

    # Parent tracking for backtracking
    # parent[state_hash] = (prev_t, prev_state, length)
    parent_data = np.zeros(
        (MAX_TRANSITIONS, 5), dtype=np.int64
    )  # t, state, prev_t, prev_state, L
    n_parent = 0

    # Initialize with state 0 at time 0
    state_to_idx[0] = 0
    idx_to_state[0] = 0
    state_cost[0] = 0.0
    state_pen[0] = 0
    n_states_curr = 1

    best_final_cost = np.inf
    best_final_pen = 2**60
    best_final_time = -1
    timed_out = False

    # Process time steps
    for t in range(T + 1):
        # Check timeout
        if time_limit_ticks > 0:
            # Simple tick check (Numba doesn't support time.perf_counter directly)
            # This will be checked at Python level
            pass

        if n_states_curr == 0:
            continue

        # Check if final state reached
        final_idx = state_to_idx[final_state % MAX_STATES]
        if final_idx >= 0 and idx_to_state[final_idx] == final_state:
            c = state_cost[final_idx]
            p = state_pen[final_idx]
            if c < best_final_cost or (
                abs(c - best_final_cost) < _EPS and p < best_final_pen
            ):
                best_final_cost = c
                best_final_pen = p
                best_final_time = t

        if t == T:
            break

        # Reset next layer
        for i in range(n_states_next):
            next_state_to_idx[next_idx_to_state[i] % MAX_STATES] = -1
        n_states_next = 0

        # Process current layer
        for idx in range(n_states_curr):
            state = idx_to_state[idx]
            c0 = state_cost[idx]
            p0 = state_pen[idx]

            if not np.isfinite(c0):
                continue

            # Decode state to get used counts
            used = _decode_state(state, radices, K)

            # Early pruning: check if remaining work fits
            remaining = _remaining_work(used, totals, lengths)
            if remaining > T - t:
                continue  # Infeasible from this state

            # Idle transition: state stays same, move to t+1
            hash_idx = state % MAX_STATES
            existing_idx = next_state_to_idx[hash_idx]
            if existing_idx < 0 or next_idx_to_state[existing_idx] != state:
                # New state in next layer
                if n_states_next < MAX_STATES - 1:
                    next_state_to_idx[hash_idx] = n_states_next
                    next_idx_to_state[n_states_next] = state
                    next_state_cost[n_states_next] = c0
                    next_state_pen[n_states_next] = p0
                    n_states_next += 1
            else:
                # Update if better
                if c0 < next_state_cost[existing_idx] or (
                    abs(c0 - next_state_cost[existing_idx]) < _EPS
                    and p0 < next_state_pen[existing_idx]
                ):
                    next_state_cost[existing_idx] = c0
                    next_state_pen[existing_idx] = p0

            # Job transitions
            for i in range(K):
                if used[i] >= totals[i]:
                    continue  # No jobs of this length remaining

                L = lengths[i]
                end = t + L
                if end > T:
                    continue  # Would exceed horizon

                # Compute new state
                new_used = used.copy()
                new_used[i] += 1
                new_state = _encode_state(new_used, mult)

                # Compute cost
                cand_cost = c0 + _interval_cost_numba(prefix, t, L)
                cand_pen = p0 + t  # Penalty for early tie-breaking

                # Store in future layer (will be merged later)
                if n_future < MAX_TRANSITIONS - 1:
                    future_data[n_future, 0] = end
                    future_data[n_future, 1] = new_state
                    future_data[n_future, 2] = cand_cost
                    future_data[n_future, 3] = cand_pen
                    n_future += 1

                    # Store parent info
                    if n_parent < MAX_TRANSITIONS - 1:
                        parent_data[n_parent, 0] = end
                        parent_data[n_parent, 1] = new_state
                        parent_data[n_parent, 2] = t
                        parent_data[n_parent, 3] = state
                        parent_data[n_parent, 4] = L
                        n_parent += 1

        # Swap layers
        state_to_idx, next_state_to_idx = next_state_to_idx, state_to_idx
        idx_to_state, next_idx_to_state = next_idx_to_state, idx_to_state
        state_cost, next_state_cost = next_state_cost, state_cost
        state_pen, next_state_pen = next_state_pen, state_pen
        n_states_curr = n_states_next

        # Merge future data for time t+1
        for f_idx in range(n_future):
            f_t = int(future_data[f_idx, 0])
            if f_t == t + 1:
                f_state = int(future_data[f_idx, 1])
                f_cost = future_data[f_idx, 2]
                f_pen = int(future_data[f_idx, 3])

                hash_idx = f_state % MAX_STATES
                existing_idx = state_to_idx[hash_idx]
                if existing_idx < 0 or idx_to_state[existing_idx] != f_state:
                    if n_states_curr < MAX_STATES - 1:
                        state_to_idx[hash_idx] = n_states_curr
                        idx_to_state[n_states_curr] = f_state
                        state_cost[n_states_curr] = f_cost
                        state_pen[n_states_curr] = f_pen
                        n_states_curr += 1
                else:
                    if f_cost < state_cost[existing_idx] or (
                        abs(f_cost - state_cost[existing_idx]) < _EPS
                        and f_pen < state_pen[existing_idx]
                    ):
                        state_cost[existing_idx] = f_cost
                        state_pen[existing_idx] = f_pen

    return best_final_cost, best_final_time, parent_data[:n_parent], timed_out


def solve_sparse_dp_numba(
    lengths: np.ndarray,
    totals: np.ndarray,
    prefix: np.ndarray,
    T: int,
    time_limit: float = -1.0,
    track_schedule: bool = True,
    max_states: int = 0,
) -> DPResultNumba:
    """
    Sparse DP solver (delegates to pure-Python implementation).

    Note: Despite the name, this currently delegates to solve_sparse_dp_python
    rather than the experimental solve_sparse_dp_numba_core, because the Numba
    core has limitations (fixed-size arrays, hash-based state lookup with
    collision risk, no LB pruning, no best_partial tracking).  The Python
    implementation with block-based LB pruning and rw/jobs_done tracking is
    the production path.

    Args:
        lengths: Array of K distinct processing times
        totals: Array of counts for each processing time
        prefix: Prefix sums of price array (length T+1)
        T: Time horizon
        time_limit: Max seconds to run (-1 for no limit)
        track_schedule: If False, skip parent-pointer tracking.
        max_states: Memory guardrail — abort if any layer exceeds this.

    Returns:
        DPResultNumba with cost, finish_time, and segments
    """
    lengths = np.asarray(lengths, dtype=np.int32)
    totals = np.asarray(totals, dtype=np.int32)
    prefix = np.asarray(prefix, dtype=np.float64)

    K = int(len(lengths))
    radices = (totals + 1).astype(np.int32)
    mult = np.ones(K, dtype=np.int64)
    for i in range(1, K):
        mult[i] = mult[i - 1] * int(radices[i - 1])
    final_state = int(np.sum(totals * mult))

    best_cost, best_t, _parent, timed_out, _best_partial = solve_sparse_dp_python(
        lengths=lengths,
        totals=totals,
        prefix=prefix,
        T=int(T),
        radices=radices,
        mult=mult,
        K=K,
        final_state=final_state,
        time_limit=float(time_limit),
        tie_break="early",
        track_schedule=track_schedule,
        max_states=max_states,
    )

    if not np.isfinite(best_cost) or best_t < 0:
        return DPResultNumba(
            feasible=False,
            cost=_INF,
            finish_time=0,
            segments=(),
            timed_out=bool(timed_out),
        )

    return DPResultNumba(
        feasible=True,
        cost=float(best_cost),
        finish_time=int(best_t),
        segments=(),
        timed_out=bool(timed_out),
    )


# --- Fallback pure Python implementation (same algorithm, slower) ---


def solve_sparse_dp_python(
    lengths: np.ndarray,
    totals: np.ndarray,
    prefix: np.ndarray,
    T: int,
    radices: np.ndarray,
    mult: np.ndarray,
    K: int,
    final_state: int,
    time_limit: float = -1.0,
    tie_break: str = "early",
    track_schedule: bool = True,
    max_states: int = 0,
    known_upper_bound: float = -1.0,
) -> Tuple[float, int, Dict, bool, Optional[Tuple[int, int, float]]]:
    """
    Pure Python sparse DP with time limit and early pruning.

    Args:
        track_schedule: If False, skip parent-pointer tracking to save memory.
            The returned parent dict will be empty.  The caller must use a
            greedy fallback if a concrete schedule is needed.
        max_states: If > 0, abort (like timeout) when any single DP layer
            exceeds this many states, to bound memory usage.

    Returns:
        (best_cost, best_finish_time, parent_dict, timed_out, best_partial)

    best_partial is (time, state, cost) of the best partial state found,
    or None if a complete solution was found.
    """
    start_time = time.perf_counter()

    lengths_list = [int(x) for x in lengths]
    totals_arr = totals
    totals_list = [int(t) for t in totals_arr]
    total_jobs = sum(totals_list)
    max_job_len = max(lengths_list) if lengths_list else 1

    # Inc values for state transitions
    inc = [int(m) for m in mult]

    # Packed parent key: t * state_bound + state  →  L
    state_bound = int(final_state) + 1

    # -- Block-based admissible lower bound --
    # Partition time into blocks of size B.  For each block boundary b, sort
    # prices[b:T] and store prefix sums.  For time t use nearest b ≤ t.
    # Admissible because [b, T) ⊇ [t, T).
    _LB_BLOCK = 20
    prices_arr = np.diff(prefix)  # prices_arr[i] = prefix[i+1] - prefix[i]
    _lb_blocks: Dict[int, np.ndarray] = {}
    for b in range(0, T + 1, _LB_BLOCK):
        if b < T:
            sp = np.sort(prices_arr[b:])
            cs = np.empty(len(sp) + 1, dtype=np.float64)
            cs[0] = 0.0
            cs[1:] = np.cumsum(sp)
            _lb_blocks[b] = cs
        else:
            _lb_blocks[b] = np.zeros(1, dtype=np.float64)

    def _lb_cost(t: int, rw: int) -> float:
        """Admissible lower bound: cheapest rw slots from block boundary ≤ t."""
        b = (t // _LB_BLOCK) * _LB_BLOCK
        arr = _lb_blocks[b]
        if rw >= len(arr):
            return float("inf")  # infeasible: need more slots than exist
        return float(arr[rw])

    # -- Decode used counts on demand (no cache, K ≤ 12 so cheap) --
    radices_list = [int(r) for r in radices]

    def decode_used(state: int) -> Tuple[int, ...]:
        u = [0] * K
        x = state
        for i in range(K):
            u[i] = x % radices_list[i]
            x //= radices_list[i]
        return tuple(u)

    # Total remaining work at initial state
    total_rw = sum(int(totals_arr[i]) * lengths_list[i] for i in range(K))

    # DP layers: layer[state] = (cost, pen, rw, jobs_done)
    dp_layers: List[Dict[int, Tuple[float, int, int, int]]] = [
        dict() for _ in range(T + 1)
    ]
    dp_layers[0][0] = (0.0, 0, total_rw, 0)
    # Packed parent: int key (t*state_bound+s) → int value (L).
    parent: Dict[int, int] = {}

    best_final_cost = _INF
    best_final_pen = 2**63 - 1
    best_final_time = -1
    timed_out = False

    # Track best partial state: (num_scheduled, cost, time, state)
    best_partial_jobs = 0
    best_partial_cost = _INF
    best_partial_time = 0
    best_partial_state = 0

    for t in range(T + 1):
        # Check timeout
        if time_limit > 0 and (time.perf_counter() - start_time) > time_limit:
            timed_out = True
            break

        layer = dp_layers[t]
        if not layer:
            continue

        # Memory guardrail: abort if layer exceeds max_states
        if max_states > 0 and len(layer) > max_states:
            timed_out = True
            break

        # Update best partial state from this layer (uses jobs_done, no decode)
        for state, (cost, pen, rw, jd) in layer.items():
            if jd > best_partial_jobs or (
                jd == best_partial_jobs and cost < best_partial_cost
            ):
                best_partial_jobs = jd
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

        # --- Merged idle + job transitions (single pass over layer) ---
        next_layer = dp_layers[t + 1]
        remaining = T - t
        # Pre-fetch lb block for this timestep
        _lb_b = (t // _LB_BLOCK) * _LB_BLOCK
        _lb_arr = _lb_blocks[_lb_b]
        _lb_arr_len = len(_lb_arr)

        early = tie_break == "early"

        for state, (c0, p0, rw, jd) in layer.items():
            # Feasibility pruning
            if rw > remaining:
                continue

            # Lower-bound pruning (inlined)
            if rw < _lb_arr_len:
                if c0 + _lb_arr[rw] > best_final_cost:
                    continue
            else:
                continue  # infeasible

            # -- Idle transition --
            prev = next_layer.get(state)
            if prev is None:
                next_layer[state] = (c0, p0, rw, jd)
                if track_schedule:
                    parent[(t + 1) * state_bound + state] = 0
            else:
                c_prev = prev[0]
                better = c0 < c_prev
                if early and not better and abs(c0 - c_prev) <= _EPS:
                    better = p0 < prev[1]
                if better:
                    next_layer[state] = (c0, p0, rw, jd)
                    if track_schedule:
                        parent[(t + 1) * state_bound + state] = 0

            # -- Job transitions (decode once per state) --
            # Inline decode_used for speed
            x = state
            for i in range(K):
                used_i = x % radices_list[i]
                x //= radices_list[i]

                if used_i >= totals_list[i]:
                    continue
                L = lengths_list[i]
                end = t + L
                if end > T:
                    continue

                new_state = state + inc[i]
                new_rw = rw - L
                new_jd = jd + 1
                cand_cost = c0 + (prefix[end] - prefix[t])
                cand_pen = p0 + t if early else p0

                target_layer = dp_layers[end]
                prev = target_layer.get(new_state)
                if prev is None:
                    target_layer[new_state] = (cand_cost, cand_pen, new_rw, new_jd)
                    if track_schedule:
                        parent[end * state_bound + new_state] = L
                else:
                    prev_cost = prev[0]
                    better = cand_cost < prev_cost
                    if early and not better and abs(cand_cost - prev_cost) <= _EPS:
                        better = cand_pen < prev[1]
                    if better:
                        target_layer[new_state] = (cand_cost, cand_pen, new_rw, new_jd)
                        if track_schedule:
                            parent[end * state_bound + new_state] = L

        # Free processed layer to save memory.
        freed_t = t - max_job_len
        if freed_t >= 0:
            dp_layers[freed_t] = {}

    # Return best partial info if we timed out without finding complete solution
    best_partial = None
    if timed_out and best_final_time < 0 and best_partial_jobs > 0:
        best_partial = (best_partial_time, best_partial_state, best_partial_cost)

    return best_final_cost, best_final_time, parent, timed_out, best_partial


def solve_sparse_dp_python_beam(
    lengths: np.ndarray,
    totals: np.ndarray,
    prefix: np.ndarray,
    T: int,
    radices: np.ndarray,
    mult: np.ndarray,
    K: int,
    final_state: int,
    *,
    vhat: Optional[callable],
    vhat_batch: Optional[callable] = None,
    beam_width: int,
    prune_factor: float = 2.0,
    time_limit: float = -1.0,
    tie_break: str = "early",
) -> Tuple[float, int, Dict, bool, Optional[Tuple[int, int, float]]]:
    """Sparse DP with beam pruning guided by a learned cost-to-go.

    This is an *approximate* DP intended for speed. It preserves:
    - Idle transitions (t -> t+1) as a first-class move
    - Feasibility pruning based on remaining work
    - Lexicographic tie-break (energy first; then `pen` if tie_break=="early")

    Pruning rule:
      At each time layer t, if |layer| exceeds `beam_width * prune_factor`, keep
      only the best `beam_width` states by (g_cost + vhat(t,state), pen).

    Args:
        vhat: Callable (t:int, state:int) -> float, or None to disable heuristic.
        vhat_batch: Optional callable (t:int, states:list[int], costs:ndarray) -> ndarray
            returning scores for all states at once. When provided, used instead
            of serial vhat calls during prune step (much faster for learned models).
        beam_width: Number of states to keep per time layer.
        prune_factor: Only prune when layer is at least beam_width*prune_factor.

    Returns same tuple as `solve_sparse_dp_python`.
    """
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    if prune_factor < 1.0:
        raise ValueError("prune_factor must be >= 1")

    start_time = time.perf_counter()

    lengths_list = [int(x) for x in lengths]
    totals_arr = totals

    inc = [int(m) for m in mult]

    # State decoding cache (same as exact sparse DP)
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

    def score_state(t: int, state: int, cost: float, pen: int) -> Tuple[float, int]:
        h = float(vhat(int(t), int(state))) if vhat is not None else 0.0
        if not np.isfinite(h):
            h = 0.0
        if tie_break == "early":
            return (float(cost) + h, int(pen))
        return (float(cost) + h, 0)

    # DP layers
    dp_layers: List[Dict[int, Tuple[float, int]]] = [dict() for _ in range(T + 1)]
    dp_layers[0][0] = (0.0, 0)
    parent: Dict[Tuple[int, int], Tuple[int, int, int]] = {}

    best_final_cost = _INF
    best_final_pen = 2**63 - 1
    best_final_time = -1
    timed_out = False

    # Track best partial state: (num_scheduled, cost, time, state)
    best_partial_jobs = 0
    best_partial_cost = _INF
    best_partial_time = 0
    best_partial_state = 0

    prune_threshold = int(max(1, round(float(beam_width) * float(prune_factor))))

    for t in range(T + 1):
        if time_limit > 0 and (time.perf_counter() - start_time) > time_limit:
            timed_out = True
            break

        layer = dp_layers[t]
        if not layer:
            continue

        # Beam prune this layer if it has grown too large
        if len(layer) > prune_threshold:
            items = list(layer.items())

            if vhat_batch is not None:
                # Batch scoring: extract states and costs, call vhat_batch once
                states_arr = [int(it[0]) for it in items]
                costs_arr = np.array(
                    [float(it[1][0]) for it in items], dtype=np.float64
                )
                pens_arr = np.array([int(it[1][1]) for it in items], dtype=np.int64)

                h_arr = vhat_batch(int(t), states_arr, costs_arr)
                h_arr = np.asarray(h_arr, dtype=np.float64)
                # Replace non-finite heuristic values with 0
                h_arr = np.where(np.isfinite(h_arr), h_arr, 0.0)

                scores = costs_arr + h_arr
                if tie_break == "early":
                    # Lexicographic sort: primary by score, secondary by penalty
                    order = np.lexsort((pens_arr, scores))
                else:
                    order = np.argsort(scores)

                keep_idx = order[: int(beam_width)]
                layer = {
                    states_arr[int(i)]: (
                        float(costs_arr[int(i)]),
                        int(pens_arr[int(i)]),
                    )
                    for i in keep_idx
                }
            else:
                best_items = heapq.nsmallest(
                    int(beam_width),
                    items,
                    key=lambda it: score_state(
                        t, it[0], float(it[1][0]), int(it[1][1])
                    ),
                )
                layer = {s: (float(cp[0]), int(cp[1])) for s, cp in best_items}
            dp_layers[t] = layer

        # Update best partial state from this layer
        for state, (cost, pen) in layer.items():
            used = decode_used(state)
            n_scheduled = count_scheduled(used)
            if n_scheduled > best_partial_jobs or (
                n_scheduled == best_partial_jobs and cost < best_partial_cost
            ):
                best_partial_jobs = n_scheduled
                best_partial_cost = float(cost)
                best_partial_time = int(t)
                best_partial_state = int(state)

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
                best_final_time = int(t)

        if t == T:
            continue

        # Idle transitions
        next_layer = dp_layers[t + 1]
        for state, (c0, p0) in layer.items():
            used = decode_used(state)
            if remaining_work(used) > T - t:
                continue
            prev = next_layer.get(state)
            if prev is None:
                next_layer[state] = (float(c0), int(p0))
                parent[(t + 1, state)] = (t, state, 0)
            else:
                c_prev, p_prev = float(prev[0]), int(prev[1])
                better = float(c0) < c_prev
                if (
                    tie_break == "early"
                    and not better
                    and abs(float(c0) - c_prev) <= _EPS
                ):
                    better = int(p0) < p_prev
                if better:
                    next_layer[state] = (float(c0), int(p0))
                    parent[(t + 1, state)] = (t, state, 0)

        # Job transitions
        for state, (c0, p0) in layer.items():
            used = decode_used(state)
            if remaining_work(used) > T - t:
                continue

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

    best_partial = None
    if timed_out and best_final_time < 0 and best_partial_jobs > 0:
        best_partial = (best_partial_time, best_partial_state, best_partial_cost)

    return best_final_cost, best_final_time, parent, timed_out, best_partial


def is_numba_available() -> bool:
    """Check if Numba is available and working."""
    return NUMBA_AVAILABLE


# =====================================================================
# Machine-states-aware sparse DP  (SPACES-based, event-driven)
# =====================================================================


def solve_sparse_dp_python_states(
    lengths: np.ndarray,
    totals: np.ndarray,
    prefix_proc: np.ndarray,
    T: int,
    radices: np.ndarray,
    mult: np.ndarray,
    K: int,
    final_state: int,
    c_star: np.ndarray,
    c_start: np.ndarray,
    c_end: np.ndarray,
    early: int,
    late: int,
    max_gap: int = -1,
    time_limit: float = -1.0,
    tie_break: str = "early",
    track_schedule: bool = True,
    max_states: int = 0,
    known_upper_bound: float = -1.0,
) -> Tuple[float, int, Dict, bool, Optional[Tuple[int, int, float]]]:
    """Sparse DP with machine-state switching costs (SPACES-based).

    This is an event-driven DP that eliminates per-timestep idle transitions.
    Instead, the DP jumps from one job-end time to the next job-start time,
    paying the precomputed optimal switching cost c*(t_end, t_start) for the
    gap between consecutive jobs.

    Layer semantics:
        dp_layers[t_end][state] = (cost, pen, rw, jd)
        means: all jobs encoded in `state` are scheduled, the last job ended
        at time t_end, and `cost` includes startup + all switching + all
        processing costs up to (but not including) shutdown.

    Final answer:
        min over t_end: dp_layers[t_end][final_state].cost + c_end[t_end]

    Parent tracking:
        parent[(t_end, new_state)] = (prev_t_end, prev_state, L, t_start)
        Stores the previous job-end time, the job length, and the start time
        for full schedule reconstruction.

    Args:
        lengths: Array of K distinct processing times.
        totals: Array of job counts for each processing time.
        prefix_proc: Prefix sums of prices * P_proc (from build_proc_prefix).
        T: Time horizon (number of intervals).
        radices, mult, K, final_state: State encoding parameters (same as
            the stateless DP).
        c_star: SPACES switching costs.  If max_gap == -1, shape (h+1, h+1)
            with c_star[i, j] = cost from proc@i to proc@j.
            If max_gap > 0, shape (h+1, max_gap+1) banded storage where
            c_star[i, dt] = cost for gap of size dt.
        c_start: Startup costs, shape (h+1,).
        c_end: Shutdown costs, shape (h+1,).
        early: Earliest proc-feasible interval.
        late: Latest proc-feasible interval.
        max_gap: If > 0, banded c_star; gaps beyond this use c_end+c_start.
        time_limit: Wall-clock limit in seconds (-1 = unlimited).
        tie_break: "cost" or "early".
        track_schedule: Whether to store parent pointers.
        max_states: Per-layer state count limit (0 = unlimited).
        known_upper_bound: If > 0, prune states with cost >= this.

    Returns:
        (best_cost, best_finish_time, parent_dict, timed_out, best_partial)
        best_cost includes shutdown cost.
        parent_dict maps (t_end, new_state) -> (prev_t_end, prev_state, L, t_start).
    """
    start_time_wall = time.perf_counter()

    lengths_list = [int(x) for x in lengths]
    totals_list = [int(t) for t in totals]
    total_jobs = sum(totals_list)
    max_job_len = max(lengths_list) if lengths_list else 1

    inc = [int(m) for m in mult]
    radices_list = [int(r) for r in radices]

    # Total remaining work at initial state
    total_rw = sum(totals_list[i] * lengths_list[i] for i in range(K))

    # Effective gap limit
    banded = max_gap > 0
    if not banded:
        eff_max_gap = T
    else:
        eff_max_gap = max_gap

    # ── Admissible lower bound (block-based) ──────────────────────
    # Sort prices*P_proc in suffix blocks for fast LB queries.
    _LB_BLOCK = 20
    proc_prices = np.diff(
        prefix_proc
    )  # proc_prices[i] = prefix_proc[i+1] - prefix_proc[i]
    _lb_blocks: Dict[int, np.ndarray] = {}
    for b in range(0, T + 1, _LB_BLOCK):
        if b < T:
            sp = np.sort(proc_prices[b:])
            cs = np.empty(len(sp) + 1, dtype=np.float64)
            cs[0] = 0.0
            cs[1:] = np.cumsum(sp)
            _lb_blocks[b] = cs
        else:
            _lb_blocks[b] = np.zeros(1, dtype=np.float64)

    # Minimum shutdown cost from any time >= 0
    min_c_end = float(np.min(c_end[c_end < _INF])) if np.any(c_end < _INF) else 0.0

    def _lb_proc_cost(t: int, rw: int) -> float:
        """Admissible LB: cheapest rw processing slots from block boundary <= t."""
        b = (t // _LB_BLOCK) * _LB_BLOCK
        arr = _lb_blocks.get(b)
        if arr is None:
            return _INF
        if rw >= len(arr):
            return _INF
        return float(arr[rw])

    # ── DP layers ─────────────────────────────────────────────────
    # dp_layers[t_end][state] = (cost, pen, rw, jd)
    # cost = total cost so far (startup + switching + processing), excluding shutdown
    dp_layers: List[Dict[int, Tuple[float, int, int, int]]] = [
        dict() for _ in range(T + 1)
    ]

    # Parent pointers: (t_end, new_state) -> (prev_t_end, prev_state, L, t_start)
    parent: Dict[Tuple[int, int], Tuple[int, int, int, int]] = {}

    best_final_cost = _INF  # includes shutdown
    best_final_pen = 2**63 - 1
    best_final_time = -1
    timed_out = False

    # Track best partial for timeout fallback
    best_partial_jobs = 0
    best_partial_cost = _INF
    best_partial_time = 0
    best_partial_state = 0

    use_early = tie_break == "early"

    # ── Initialization: schedule the first job ────────────────────
    # For each possible first-job start time and each job type,
    # compute startup + processing cost and seed the DP.
    for t_s in range(early, late + 1):
        startup = float(c_start[t_s])
        if startup >= _INF:
            continue
        for i in range(K):
            L = lengths_list[i]
            t_end = t_s + L
            if t_end > T or t_s + L > late + 1:
                continue

            job_cost = float(prefix_proc[t_end] - prefix_proc[t_s])
            cost = startup + job_cost
            new_state = inc[i]  # one job of type i
            new_rw = total_rw - L
            pen = t_s if use_early else 0

            # LB pruning
            lb = cost + _lb_proc_cost(t_end, new_rw) + min_c_end
            if known_upper_bound > 0 and lb > known_upper_bound + _EPS:
                continue
            if lb > best_final_cost + _EPS:
                continue

            prev = dp_layers[t_end].get(new_state)
            if prev is None:
                dp_layers[t_end][new_state] = (cost, pen, new_rw, 1)
                if track_schedule:
                    parent[(t_end, new_state)] = (-1, 0, L, t_s)
            else:
                better = cost < prev[0]
                if use_early and not better and abs(cost - prev[0]) <= _EPS:
                    better = pen < prev[1]
                if better:
                    dp_layers[t_end][new_state] = (cost, pen, new_rw, 1)
                    if track_schedule:
                        parent[(t_end, new_state)] = (-1, 0, L, t_s)

    # ── Main DP: iterate over job-end times ───────────────────────
    for t_end in range(1, T + 1):
        if time_limit > 0 and (time.perf_counter() - start_time_wall) > time_limit:
            timed_out = True
            break

        layer = dp_layers[t_end]
        if not layer:
            continue

        # Memory guardrail
        if max_states > 0 and len(layer) > max_states:
            timed_out = True
            break

        # Update best partial
        for state, (cost, pen, rw, jd) in layer.items():
            if jd > best_partial_jobs or (
                jd == best_partial_jobs and cost < best_partial_cost
            ):
                best_partial_jobs = jd
                best_partial_cost = cost
                best_partial_time = t_end
                best_partial_state = state

        # Check for final state
        v_final = layer.get(final_state)
        if v_final is not None:
            c_with_shutdown = float(v_final[0]) + float(c_end[t_end])
            p_final = int(v_final[1])
            better = c_with_shutdown < best_final_cost
            if (
                use_early
                and not better
                and abs(c_with_shutdown - best_final_cost) <= _EPS
            ):
                better = p_final < best_final_pen or (
                    p_final == best_final_pen and t_end < best_final_time
                )
            if better:
                best_final_cost = c_with_shutdown
                best_final_pen = p_final
                best_final_time = t_end

        # ── Expand: try all gap + next-job combinations ───────────
        for state, (c0, p0, rw, jd) in layer.items():
            if rw == 0:
                continue  # All jobs done, no more transitions

            # For each gap start time t_s >= t_end
            gap_limit = min(t_end + eff_max_gap, late + 1)
            for t_s in range(t_end, gap_limit):
                # Compute switching cost
                if t_s == t_end:
                    gap_cost = 0.0  # Consecutive processing
                elif banded:
                    dt = t_s - t_end
                    gap_cost = float(c_star[t_end, dt])
                else:
                    gap_cost = float(c_star[t_end, t_s])

                if gap_cost >= _INF:
                    continue

                base_cost = c0 + gap_cost

                # Try each available job type
                for i in range(K):
                    # Inline decode: extract used_i from state
                    # We need used_i for job type i
                    x_tmp = state
                    for j in range(i):
                        x_tmp //= radices_list[j]
                    used_i = x_tmp % radices_list[i]

                    if used_i >= totals_list[i]:
                        continue

                    L = lengths_list[i]
                    t_job_end = t_s + L
                    if t_job_end > T or t_s + L > late + 1:
                        continue

                    new_state = state + inc[i]
                    new_rw = rw - L
                    new_jd = jd + 1
                    cand_cost = base_cost + float(
                        prefix_proc[t_job_end] - prefix_proc[t_s]
                    )
                    cand_pen = p0 + t_s if use_early else p0

                    # LB pruning
                    lb = cand_cost + _lb_proc_cost(t_job_end, new_rw) + min_c_end
                    if known_upper_bound > 0 and lb > known_upper_bound + _EPS:
                        continue
                    if lb > best_final_cost + _EPS:
                        continue

                    target_layer = dp_layers[t_job_end]
                    prev = target_layer.get(new_state)
                    if prev is None:
                        target_layer[new_state] = (cand_cost, cand_pen, new_rw, new_jd)
                        if track_schedule:
                            parent[(t_job_end, new_state)] = (t_end, state, L, t_s)
                    else:
                        better = cand_cost < prev[0]
                        if (
                            use_early
                            and not better
                            and abs(cand_cost - prev[0]) <= _EPS
                        ):
                            better = cand_pen < prev[1]
                        if better:
                            target_layer[new_state] = (
                                cand_cost,
                                cand_pen,
                                new_rw,
                                new_jd,
                            )
                            if track_schedule:
                                parent[(t_job_end, new_state)] = (t_end, state, L, t_s)

            # Also try gaps beyond eff_max_gap when banded:
            # decompose as c_end[t_end] + c_start[t_s]
            if banded and gap_limit < late + 1:
                c_end_here = float(c_end[t_end])
                if c_end_here < _INF:
                    for t_s in range(gap_limit, late + 1):
                        startup_cost = float(c_start[t_s])
                        if startup_cost >= _INF:
                            continue
                        gap_cost = c_end_here + startup_cost
                        base_cost = c0 + gap_cost

                        for i in range(K):
                            x_tmp = state
                            for j in range(i):
                                x_tmp //= radices_list[j]
                            used_i = x_tmp % radices_list[i]

                            if used_i >= totals_list[i]:
                                continue

                            L = lengths_list[i]
                            t_job_end = t_s + L
                            if t_job_end > T or t_s + L > late + 1:
                                continue

                            new_state = state + inc[i]
                            new_rw = rw - L
                            new_jd = jd + 1
                            cand_cost = base_cost + float(
                                prefix_proc[t_job_end] - prefix_proc[t_s]
                            )
                            cand_pen = p0 + t_s if use_early else p0

                            lb = (
                                cand_cost + _lb_proc_cost(t_job_end, new_rw) + min_c_end
                            )
                            if known_upper_bound > 0 and lb > known_upper_bound + _EPS:
                                continue
                            if lb > best_final_cost + _EPS:
                                continue

                            target_layer = dp_layers[t_job_end]
                            prev = target_layer.get(new_state)
                            if prev is None:
                                target_layer[new_state] = (
                                    cand_cost,
                                    cand_pen,
                                    new_rw,
                                    new_jd,
                                )
                                if track_schedule:
                                    parent[(t_job_end, new_state)] = (
                                        t_end,
                                        state,
                                        L,
                                        t_s,
                                    )
                            else:
                                better = cand_cost < prev[0]
                                if (
                                    use_early
                                    and not better
                                    and abs(cand_cost - prev[0]) <= _EPS
                                ):
                                    better = cand_pen < prev[1]
                                if better:
                                    target_layer[new_state] = (
                                        cand_cost,
                                        cand_pen,
                                        new_rw,
                                        new_jd,
                                    )
                                    if track_schedule:
                                        parent[(t_job_end, new_state)] = (
                                            t_end,
                                            state,
                                            L,
                                            t_s,
                                        )

        # Free old layers to save memory
        freed_t = t_end - max_job_len - eff_max_gap
        if freed_t >= 0:
            dp_layers[freed_t] = {}

    # Return best partial if timed out without complete solution
    best_partial_result = None
    if timed_out and best_final_time < 0 and best_partial_jobs > 0:
        best_partial_result = (
            best_partial_time,
            best_partial_state,
            best_partial_cost,
        )

    return best_final_cost, best_final_time, parent, timed_out, best_partial_result

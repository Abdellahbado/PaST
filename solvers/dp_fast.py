"""Fast DP utilities for fixed-order TOU scheduling.

This module is intentionally dependency-light (numpy only) and is meant for
profiling/benchmarking and for potential integration into ALNS evaluation.

Semantics match `PaST.solvers.baselines_sequence_dp._dp_schedule_fixed_order`:
- Fixed job order (processing times list)
- Non-overlap, respect order
- Finish by T_limit
- Objective: minimize energy = e_single * sum(ct[u] for processing slots)

Key speed levers:
- Use O(T) rolling DP instead of storing TEC for all stages.
- Allow passing precomputed ct prefix sums to avoid repeated cumsum.

Returns cost and the optimal start time of the *last* job (sufficient for
makespan = last_start + last_proc).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def prefix_costs_from_ct(ct: np.ndarray, T_limit: int) -> np.ndarray:
    """prefix_costs[t] = sum_{u < t} ct[u] for t=0..T_limit."""
    T = int(T_limit)
    ct = np.asarray(ct, dtype=np.int32)
    if T < 0:
        raise ValueError("T_limit must be >= 0")
    if ct.shape[0] < T:
        raise ValueError("ct shorter than T_limit")
    prefix = np.zeros(T + 1, dtype=np.float64)
    if T > 0:
        prefix[1:] = np.cumsum(ct[:T], dtype=np.float64)
    return prefix


def dp_cost_fixed_order_rolling(
    processing_times: Sequence[int],
    ct: np.ndarray,
    e_single: int,
    T_limit: int,
) -> Tuple[float, int]:
    """Cost-only DP using rolling arrays; computes prefix sums internally."""
    prefix = prefix_costs_from_ct(ct, T_limit)
    return dp_cost_fixed_order_rolling_precomputed(
        processing_times=processing_times,
        prefix_costs=prefix,
        e_single=e_single,
        T_limit=T_limit,
    )


def dp_cost_fixed_order_rolling_precomputed(
    processing_times: Sequence[int],
    prefix_costs: np.ndarray,
    e_single: int,
    T_limit: int,
) -> Tuple[float, int]:
    """Cost-only DP using rolling arrays.

    Args:
        processing_times: fixed job order durations.
        prefix_costs: prefix sums of ct for the same T_limit.
        e_single: machine energy rate.
        T_limit: deadline.

    Returns:
        (min_cost, best_last_start). If infeasible, cost=inf and best_last_start=0.
    """
    J = int(len(processing_times))
    if J == 0:
        return 0.0, 0

    T = int(T_limit)
    if T <= 0:
        return float("inf"), 0

    if prefix_costs.shape[0] < T + 1:
        raise ValueError("prefix_costs shorter than T_limit")

    PT = np.asarray(processing_times, dtype=np.int32)

    # Earliest/latest start times.
    ES = np.zeros(J, dtype=np.int32)
    LS = np.zeros(J, dtype=np.int32)
    running = 0
    total = int(PT.sum())
    for i in range(J):
        ES[i] = int(running)
        total -= int(PT[i])
        LS[i] = int(T - total - int(PT[i]))
        running += int(PT[i])

    # Quick feasibility check (under fixed order, necessary): last job must have LS>=ES.
    if int(LS[-1]) < int(ES[-1]):
        return float("inf"), 0

    def job_cost(start: int, p: int) -> float:
        return float(e_single) * float(prefix_costs[start + p] - prefix_costs[start])

    dp_prev = np.full(T, np.inf, dtype=np.float64)
    dp_curr = np.full(T, np.inf, dtype=np.float64)

    # First job.
    p0 = int(PT[0])
    es0 = int(ES[0])
    ls0 = int(LS[0])
    if ls0 >= es0:
        latest0 = min(ls0, T - p0)
        for t in range(es0, latest0 + 1):
            dp_prev[t] = job_cost(t, p0)

    # Subsequent jobs.
    for i in range(2, J + 1):
        j = i - 1
        p = int(PT[j])
        es = int(ES[j])
        ls = int(LS[j])
        if ls < es:
            return float("inf"), 0

        p_prev = int(PT[j - 1])
        es_prev = int(ES[j - 1])
        ls_prev = int(LS[j - 1])

        # Prefix min of dp_prev over feasible previous start times.
        prefix_min = np.full(T, np.inf, dtype=np.float64)
        best = np.inf
        for s in range(es_prev, min(ls_prev, T - 1) + 1):
            v = dp_prev[s]
            if v < best:
                best = v
            prefix_min[s] = best

        latest = min(ls, T - p)
        for t in range(es, latest + 1):
            r = min(ls_prev, t - p_prev)
            if r < es_prev:
                continue
            prev_best = prefix_min[r]
            if not np.isfinite(prev_best):
                continue
            dp_curr[t] = prev_best + job_cost(t, p)

        dp_prev, dp_curr = dp_curr, dp_prev
        dp_curr.fill(np.inf)

    best_t = int(np.argmin(dp_prev))
    best_cost = float(dp_prev[best_t])
    if not np.isfinite(best_cost):
        return float("inf"), 0
    return best_cost, best_t

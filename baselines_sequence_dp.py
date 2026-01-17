"""Baselines for PaST-SM: SPT/LPT sequences + DP scheduling for that sequence.

We use:
- SPT/LPT to produce a job *sequence* (order).
- Dynamic Programming (DP) to compute the optimal *schedule* (job start times)
    under the fixed order, minimizing total energy.

This matches the intent of the classic baseline you described (and your provided
solver): sequencing is separate from scheduling.

Cost/feasibility semantics are consistent with `SingleMachinePeriodEnv.step()`:
- Each job i with duration p runs on [start_i, start_i + p)
- Jobs are non-overlapping and respect the given order
- Completion must be <= T_limit
- Energy = e_single * sum(ct[u] for u in processing slots)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class DPResult:
    total_energy: float
    total_return: float
    job_sequence: List[int]
    start_times: List[int]


def dp_schedule_for_job_sequence(
    batch_data_single: Dict[str, Any],
    job_sequence: List[int],
    dp_time_penalty: float = 0.0,
) -> DPResult:
    """DP-schedule a provided job order for one single-machine episode.

    Args:
        batch_data_single: single instance dict with shape (1, ...)
        job_sequence: list of job indices (0..n_jobs-1) describing the fixed order

    Returns:
        DPResult with `job_sequence` and per-position `start_times`.
    """
    n_jobs = int(batch_data_single["n_jobs"][0])
    p_subset = batch_data_single["p_subset"][0][:n_jobs]
    ct = batch_data_single["ct"][0]
    T_limit = int(batch_data_single["T_limit"][0])
    e_single = int(batch_data_single["e_single"][0])

    seq = [int(j) for j in job_sequence[:n_jobs]]
    proc = [int(p_subset[j]) for j in seq]
    total_energy, start_times = _dp_schedule_fixed_order(
        processing_times=proc,
        ct=ct,
        e_single=e_single,
        T_limit=T_limit,
        dp_time_penalty=dp_time_penalty,
    )

    return DPResult(
        total_energy=float(total_energy),
        total_return=(
            float(-total_energy) if np.isfinite(total_energy) else -float("inf")
        ),
        job_sequence=seq,
        start_times=[int(t) for t in start_times],
    )


def spt_sequence(p_subset: np.ndarray, n_jobs: int) -> List[int]:
    """Indices [0..n_jobs-1] sorted by increasing processing time."""
    p = p_subset[:n_jobs].astype(int)
    idx = list(range(int(n_jobs)))
    idx.sort(key=lambda i: (p[i], i))
    return idx


def lpt_sequence(p_subset: np.ndarray, n_jobs: int) -> List[int]:
    """Indices [0..n_jobs-1] sorted by decreasing processing time."""
    p = p_subset[:n_jobs].astype(int)
    idx = list(range(int(n_jobs)))
    idx.sort(key=lambda i: (-p[i], i))
    return idx


def _slice_single_instance(batch_data: Dict[str, Any], index: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch_data.items():
        if isinstance(v, np.ndarray):
            out[k] = v[index : index + 1]
        else:
            raise TypeError(f"Unsupported batch_data type for key={k}: {type(v)}")
    return out


def _dp_schedule_fixed_order(
    processing_times: List[int],
    ct: np.ndarray,
    e_single: int,
    T_limit: int,
    dp_time_penalty: float = 0.0,
) -> Tuple[float, List[int]]:
    """Compute optimal start times for a *fixed order* of jobs.

    This is the same DP idea as in your `solver_improved.py`:
    - state: (i, start_time_i)
    - transition: choose start_time_{i} >= start_time_{i-1} + p_{i-1}
    - objective: minimize sum energy over processing intervals
    """
    J = int(len(processing_times))
    if J == 0:
        return 0.0, []

    T = int(T_limit)
    if T <= 0:
        return float("inf"), [0] * J

    PT = np.array(processing_times, dtype=np.int32)
    ct = np.asarray(ct, dtype=np.int32)
    if ct.shape[0] < T:
        # ct is padded in the batch, but should always cover T_limit.
        return float("inf"), [0] * J

    # prefix_costs[t] = sum_{u < t} ct[u]
    prefix_costs = np.zeros(T + 1, dtype=np.float64)
    prefix_costs[1:] = np.cumsum(ct[:T], dtype=np.float64)

    # Earliest/latest start times to ensure feasibility under order + deadline.
    ES = np.zeros(J, dtype=np.int32)
    LS = np.zeros(J, dtype=np.int32)
    for i in range(J):
        ES[i] = int(np.sum(PT[:i]))
        LS[i] = int(T - np.sum(PT[i:]))

    # TEC[i, t] = best cost for first i jobs, with job i-1 starting at time t.
    TEC = np.full((J + 1, T), np.inf, dtype=np.float64)
    parent = np.full((J + 1, T), -1, dtype=np.int32)
    TEC[0, 0] = 0.0

    def job_cost(start: int, p: int) -> float:
        # Primary objective: energy.
        energy = float(e_single) * float(prefix_costs[start + p] - prefix_costs[start])
        # Optional regularizer: discourage late starts (lexicographic-ish when small).
        # This is an inference-time knob; default keeps the original objective.
        if dp_time_penalty > 0:
            energy += float(dp_time_penalty) * float(start)
        return energy

    for i in range(1, J + 1):
        job_idx = i - 1
        p = int(PT[job_idx])
        es = int(ES[job_idx])
        ls = int(LS[job_idx])
        if ls < es:
            continue

        if i == 1:
            # First job has no predecessor constraint beyond feasibility.
            latest = min(ls, T - p)
            for t in range(es, latest + 1):
                TEC[i, t] = job_cost(t, p)
                parent[i, t] = 0
            continue

        prev_idx = job_idx - 1
        p_prev = int(PT[prev_idx])
        es_prev = int(ES[prev_idx])
        ls_prev = int(LS[prev_idx])

        # Prefix minimum over feasible previous start times.
        prefix_min = np.full(T, np.inf, dtype=np.float64)
        prefix_argmin = np.full(T, -1, dtype=np.int32)
        best_val = np.inf
        best_arg = -1
        for s in range(es_prev, ls_prev + 1):
            v = TEC[i - 1, s]
            if v < best_val:
                best_val = v
                best_arg = s
            prefix_min[s] = best_val
            prefix_argmin[s] = best_arg

        latest = min(ls, T - p)
        for t in range(es, latest + 1):
            r = min(ls_prev, t - p_prev)
            if r < es_prev:
                continue
            prev_best = prefix_min[r]
            if not np.isfinite(prev_best):
                continue
            TEC[i, t] = prev_best + job_cost(t, p)
            parent[i, t] = int(prefix_argmin[r])

    best_t = int(np.argmin(TEC[J, :]))
    min_cost = float(TEC[J, best_t])
    if not np.isfinite(min_cost):
        return float("inf"), [0] * J

    # Backtrack start times.
    start_times = [0] * J
    curr_t = best_t
    for i in range(J, 0, -1):
        start_times[i - 1] = int(curr_t)
        prev_t = int(parent[i, curr_t])
        curr_t = prev_t

    return float(min_cost), start_times


def spt_lpt_with_dp(
    env_config,
    device,
    batch_data: Dict[str, Any],
    which: str,
    dp_time_penalty: float = 0.0,
) -> List[DPResult]:
    """Run SPT+DP or LPT+DP for a batch.

    Note: `env_config` and `device` are accepted for API compatibility with the
    evaluation runner, but the DP baseline itself uses only `batch_data`.
    """
    some_key = next(iter(batch_data.keys()))
    B = int(batch_data[some_key].shape[0])

    results: List[DPResult] = []
    for i in range(B):
        single = _slice_single_instance(batch_data, i)
        n_jobs = int(single["n_jobs"][0])
        p_subset = single["p_subset"][0][:n_jobs]
        ct = single["ct"][0]
        T_limit = int(single["T_limit"][0])
        e_single = int(single["e_single"][0])

        if which.lower() == "spt":
            seq = spt_sequence(p_subset, n_jobs)
        elif which.lower() == "lpt":
            seq = lpt_sequence(p_subset, n_jobs)
        else:
            raise ValueError("which must be 'spt' or 'lpt'")

        proc = [int(p_subset[j]) for j in seq]
        total_energy, start_times = _dp_schedule_fixed_order(
            processing_times=proc,
            ct=ct,
            e_single=e_single,
            T_limit=T_limit,
            dp_time_penalty=dp_time_penalty,
        )

        results.append(
            DPResult(
                total_energy=float(total_energy),
                total_return=(
                    float(-total_energy) if np.isfinite(total_energy) else -float("inf")
                ),
                job_sequence=[int(j) for j in seq],
                start_times=[int(t) for t in start_times],
            )
        )

    return results

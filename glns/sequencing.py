"""Per-machine optimal sequencing layer.

Given a job-to-machine assignment, this module finds the optimal (or near-optimal)
job ordering and start times for each machine using the Cython sparse DP or a
fast heuristic fallback.

This decouples the problem:
    LLM operators  →  decide ASSIGNMENT (which machine gets which job)
    This module    →  decide SEQUENCING  (job order + start times per machine)

The Cython sparse DP (solve_sparse_dp_cython) finds *optimal* sequencing for a
single machine when the number of distinct processing times K ≤ 12. For the
Wang2018 benchmark, K ∈ {3..5} (small/medium) or K = 12 (large), so the sparse
DP is always applicable.

When the sparse DP is too slow (estimated state space > threshold), a fast
heuristic fallback is used: try SPT, LPT, and cheapest-slot orderings, then
pick the best via fixed-order DP.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import the Cython sparse DP
# ---------------------------------------------------------------------------
_CYTHON_AVAILABLE = False
try:
    from solvers._sparse_dp_cython import solve_sparse_dp_cython  # type: ignore

    _CYTHON_AVAILABLE = True
except ImportError:
    try:
        from PaST.solvers._sparse_dp_cython import solve_sparse_dp_cython  # type: ignore

        _CYTHON_AVAILABLE = True
    except ImportError:
        pass

if not _CYTHON_AVAILABLE:
    logger.warning(
        "Cython sparse DP not available — falling back to heuristic sequencing only."
    )


# ---------------------------------------------------------------------------
# Fixed-order DP (pure Python, inlined for zero-dependency usage)
# ---------------------------------------------------------------------------


def _dp_schedule_fixed_order(
    processing_times: List[int],
    ct: List[int],
    e_single: int,
    T_limit: int,
) -> Tuple[float, List[int]]:
    """Optimal start times for a fixed job order on one machine.

    Returns (total_energy_cost, start_times).
    """
    J = len(processing_times)
    if J == 0:
        return 0.0, []
    T = T_limit
    if T <= 0 or len(ct) < T:
        return float("inf"), [0] * J

    prefix = [0.0] * (T + 1)
    for t in range(T):
        prefix[t + 1] = prefix[t] + ct[t]

    PT = processing_times
    ES = [0] * J
    LS = [0] * J
    cum = 0
    for i in range(J):
        ES[i] = cum
        cum += PT[i]
    cum = 0
    for i in range(J - 1, -1, -1):
        cum += PT[i]
        LS[i] = T - cum

    INF = float("inf")
    prev_row = [INF] * T
    parent = [[-1] * T for _ in range(J + 1)]

    p0 = PT[0]
    es0, ls0 = ES[0], LS[0]
    cur_row = [INF] * T
    for t in range(es0, min(ls0, T - p0) + 1):
        cost = float(e_single) * (prefix[t + p0] - prefix[t])
        cur_row[t] = cost
        parent[1][t] = 0

    for i in range(2, J + 1):
        job_idx = i - 1
        p_cur = PT[job_idx]
        es_cur, ls_cur = ES[job_idx], LS[job_idx]
        prev_idx = job_idx - 1
        p_prev = PT[prev_idx]
        es_prev, ls_prev = ES[prev_idx], LS[prev_idx]

        prev_row = cur_row
        cur_row = [INF] * T

        pmin_val = [INF] * T
        pmin_arg = [-1] * T
        best_v = INF
        best_a = -1
        for s in range(es_prev, ls_prev + 1):
            v = prev_row[s]
            if v < best_v:
                best_v = v
                best_a = s
            pmin_val[s] = best_v
            pmin_arg[s] = best_a

        latest = min(ls_cur, T - p_cur)
        for t in range(es_cur, latest + 1):
            r = min(ls_prev, t - p_prev)
            if r < es_prev:
                continue
            pb = pmin_val[r]
            if pb == INF:
                continue
            cost = float(e_single) * (prefix[t + p_cur] - prefix[t])
            cur_row[t] = pb + cost
            parent[i][t] = pmin_arg[r]

    best_t = -1
    best_cost = INF
    for t in range(T):
        if cur_row[t] < best_cost:
            best_cost = cur_row[t]
            best_t = t

    if best_cost == INF:
        return INF, [0] * J

    starts = [0] * J
    ct_t = best_t
    for i in range(J, 0, -1):
        starts[i - 1] = ct_t
        ct_t = parent[i][ct_t]
    return best_cost, starts


# ---------------------------------------------------------------------------
# Sparse DP wrapper for single-machine optimal sequencing
# ---------------------------------------------------------------------------


def _solve_machine_optimal(
    job_processing_times: List[int],
    ct: List[int],
    e_single: int,
    T_limit: int,
    time_limit: float = 5.0,
) -> Tuple[float, int, List[int], List[int]]:
    """Solve a single-machine sequencing + scheduling problem optimally.

    Uses the Cython sparse DP to find the optimal job ordering AND start times.

    Returns:
        (energy_cost, finish_time, job_order, start_times)
        job_order: indices into the input job_processing_times list
        start_times: optimal start time for each job in the returned order
    """
    import numpy as np

    n = len(job_processing_times)
    if n == 0:
        return 0.0, 0, [], []

    total_p = sum(job_processing_times)
    if total_p > T_limit:
        return float("inf"), T_limit + 1, list(range(n)), [0] * n

    prices = np.array(ct[:T_limit], dtype=np.float64)
    T = len(prices)
    prefix = np.zeros(T + 1, dtype=np.float64)
    if T > 0:
        prefix[1:] = np.cumsum(prices)

    # Group jobs by processing time (sparse DP operates on counts, not IDs)
    from collections import Counter

    p_counts = Counter(job_processing_times)
    lengths_list = sorted(p_counts.keys())
    K = len(lengths_list)

    if K > 12:
        # Too many distinct processing times — fall back to heuristic
        return _solve_machine_heuristic(job_processing_times, ct, e_single, T_limit)

    lengths = np.array(lengths_list, dtype=np.int64)
    totals = np.array([p_counts[l] for l in lengths_list], dtype=np.int64)
    radices = (totals + 1).astype(np.int64)
    mult = np.ones(K, dtype=np.int64)
    for i in range(1, K):
        mult[i] = mult[i - 1] * int(radices[i - 1])
    final_state = int(np.sum(totals * mult))

    # Estimate state space size for feasibility check
    state_space = 1
    for t in totals:
        state_space *= int(t) + 1
        if state_space > 2_000_000:  # 2M states threshold
            return _solve_machine_heuristic(job_processing_times, ct, e_single, T_limit)

    best_cost, best_time, parent_dict, timed_out, best_partial = solve_sparse_dp_cython(
        lengths,
        totals,
        prefix,
        T,
        radices,
        mult,
        K,
        final_state,
        time_limit=time_limit,
        tie_break="early",
        track_schedule=True,
        max_states=5_000_000,
        known_upper_bound=-1.0,
    )

    if timed_out or best_cost >= 1e290:
        return _solve_machine_heuristic(job_processing_times, ct, e_single, T_limit)

    # Reconstruct schedule from parent dict
    schedule = _reconstruct_schedule(
        parent_dict,
        best_time,
        final_state,
        lengths_list,
        totals,
        mult,
        K,
        job_processing_times,
        prefix,
    )

    if schedule is None:
        return _solve_machine_heuristic(job_processing_times, ct, e_single, T_limit)

    job_order, start_times = schedule
    # Compute actual energy cost with machine energy rate
    total_energy = 0.0
    for idx, jidx in enumerate(job_order):
        p = job_processing_times[jidx]
        s = start_times[idx]
        total_energy += e_single * float(prefix[s + p] - prefix[s])

    finish = max(
        (
            start_times[i] + job_processing_times[job_order[i]]
            for i in range(len(job_order))
        ),
        default=0,
    )
    return total_energy, finish, job_order, start_times


def _reconstruct_schedule(
    parent_dict: dict,
    best_time: int,
    final_state: int,
    lengths_list: List[int],
    totals,
    mult,
    K: int,
    original_p: List[int],
    prefix,
) -> Optional[Tuple[List[int], List[int]]]:
    """Reconstruct job order and start times from sparse DP parent pointers.

    Returns (job_order, start_times) or None on failure.
    """
    import numpy as np

    # Build segments: (start_time, length)
    segments: List[Tuple[int, int]] = []
    cur_t = best_time
    cur_s = final_state

    state_bound = final_state + 1

    radices = np.array([int(t) + 1 for t in totals], dtype=np.int64)

    while cur_s != 0 or cur_t != 0:
        key = cur_t * state_bound + cur_s
        L = parent_dict.get(key, -2)
        if L == -2:
            break
        if L > 0:
            segments.append((cur_t - L, L))
            # Decode state to find which length group was decremented
            x = cur_s
            for i in range(K):
                ui = int(x % radices[i])
                x //= radices[i]
                if lengths_list[i] == L and ui > 0:
                    cur_s -= int(mult[i])
                    break
            cur_t -= L
        else:
            cur_t -= 1

    segments.reverse()

    if not segments:
        return None

    # Map segments back to original job indices
    # Group original jobs by processing time
    from collections import defaultdict

    jobs_by_p: Dict[int, List[int]] = defaultdict(list)
    for idx, p in enumerate(original_p):
        jobs_by_p[p].append(idx)
    # Sort each group so assignment is deterministic
    for v in jobs_by_p.values():
        v.sort()

    job_order: List[int] = []
    start_times: List[int] = []
    for start, length in segments:
        if length in jobs_by_p and jobs_by_p[length]:
            jidx = jobs_by_p[length].pop(0)
            job_order.append(jidx)
            start_times.append(start)
        else:
            return None  # shouldn't happen

    return job_order, start_times


# ---------------------------------------------------------------------------
# Heuristic fallback: try multiple orderings, pick best
# ---------------------------------------------------------------------------


def _solve_machine_heuristic(
    job_processing_times: List[int],
    ct: List[int],
    e_single: int,
    T_limit: int,
) -> Tuple[float, int, List[int], List[int]]:
    """Heuristic sequencing: try SPT, LPT, energy-sorted orderings.

    Returns (energy_cost, finish_time, job_order, start_times).
    """
    n = len(job_processing_times)
    if n == 0:
        return 0.0, 0, [], []

    candidates: List[List[int]] = []

    # SPT (Shortest Processing Time)
    spt = sorted(range(n), key=lambda j: (job_processing_times[j], j))
    candidates.append(spt)

    # LPT (Longest Processing Time)
    lpt = sorted(range(n), key=lambda j: (-job_processing_times[j], j))
    candidates.append(lpt)

    # Cheapest-slot ordering: sort by processing time ascending, then
    # try to place shorter jobs in expensive slots and longer jobs in cheap slots
    # (reversed SPT — sometimes better for TEC)
    candidates.append(list(reversed(spt)))

    # Evaluate each ordering with fixed-order DP
    best_cost = float("inf")
    best_order: List[int] = spt
    best_starts: List[int] = [0] * n
    best_finish = T_limit

    for order in candidates:
        procs = [job_processing_times[j] for j in order]
        cost, starts = _dp_schedule_fixed_order(procs, ct, e_single, T_limit)
        if cost < best_cost:
            best_cost = cost
            best_order = order
            best_starts = starts
            if starts:
                best_finish = starts[-1] + procs[-1]
            else:
                best_finish = 0

    return best_cost, best_finish, best_order, best_starts


# ---------------------------------------------------------------------------
# Full assignment evaluation (all machines)
# ---------------------------------------------------------------------------


def evaluate_assignment(
    assignment: List[int],
    inst: dict,
    sequencing_mode: str = "auto",
    time_limit_per_machine: float = 2.0,
) -> Tuple[float, int, List[List[int]], List[List[int]]]:
    """Evaluate a complete job-to-machine assignment.

    Uses optimal or heuristic sequencing per machine.

    Args:
        assignment: length-n list, assignment[j] = machine index (0..m-1)
        inst: instance dict {m, n, T, p, e, ct}
        sequencing_mode: "optimal", "heuristic", or "auto"
        time_limit_per_machine: seconds per machine for optimal DP

    Returns:
        (total_energy, makespan, sequences, start_times)
        sequences[h] = ordered job indices for machine h
        start_times[h] = start times for each job in sequences[h]
    """
    m = inst["m"]
    n = inst["n"]
    T = inst["T"]
    p = inst["p"]
    e = inst["e"]
    ct = inst["ct"]

    # Group jobs by machine
    machine_jobs: List[List[int]] = [[] for _ in range(m)]
    for j in range(n):
        mi = assignment[j]
        if 0 <= mi < m:
            machine_jobs[mi].append(j)

    total_energy = 0.0
    makespan = 0
    sequences: List[List[int]] = [[] for _ in range(m)]
    all_starts: List[List[int]] = [[] for _ in range(m)]

    for mi in range(m):
        jobs = machine_jobs[mi]
        if not jobs:
            continue

        job_p = [p[j] for j in jobs]
        e_single = e[mi]

        # Choose sequencing method
        use_optimal = False
        if sequencing_mode == "optimal":
            use_optimal = _CYTHON_AVAILABLE
        elif sequencing_mode == "auto":
            # Estimate state space
            from collections import Counter

            p_counts = Counter(job_p)
            K = len(p_counts)
            state_space = 1
            for cnt in p_counts.values():
                state_space *= cnt + 1
            use_optimal = _CYTHON_AVAILABLE and K <= 12 and state_space <= 500_000
        # else: heuristic

        if use_optimal:
            energy, finish, order, starts = _solve_machine_optimal(
                job_p,
                ct,
                e_single,
                T,
                time_limit=time_limit_per_machine,
            )
        else:
            energy, finish, order, starts = _solve_machine_heuristic(
                job_p,
                ct,
                e_single,
                T,
            )

        if energy == float("inf"):
            return float("inf"), T + 1, [[] for _ in range(m)], [[] for _ in range(m)]

        # Map back to global job indices
        ordered_jobs = [jobs[k] for k in order]
        sequences[mi] = ordered_jobs
        all_starts[mi] = starts
        total_energy += energy
        makespan = max(makespan, finish)

    return total_energy, makespan, sequences, all_starts


# ---------------------------------------------------------------------------
# Parallel evaluation across machines (for HPC)
# ---------------------------------------------------------------------------


def _eval_single_machine_task(args):
    """Worker function for parallel machine evaluation."""
    jobs, job_p, ct, e_single, T, sequencing_mode, time_limit = args
    if not jobs:
        return 0.0, 0, [], [], []

    use_optimal = False
    if sequencing_mode == "optimal":
        use_optimal = _CYTHON_AVAILABLE
    elif sequencing_mode == "auto":
        from collections import Counter

        p_counts = Counter(job_p)
        K = len(p_counts)
        state_space = 1
        for cnt in p_counts.values():
            state_space *= cnt + 1
        use_optimal = _CYTHON_AVAILABLE and K <= 12 and state_space <= 500_000

    if use_optimal:
        energy, finish, order, starts = _solve_machine_optimal(
            job_p,
            ct,
            e_single,
            T,
            time_limit=time_limit,
        )
    else:
        energy, finish, order, starts = _solve_machine_heuristic(
            job_p,
            ct,
            e_single,
            T,
        )

    ordered_jobs = [jobs[k] for k in order]
    return energy, finish, ordered_jobs, starts, jobs


def evaluate_assignment_parallel(
    assignment: List[int],
    inst: dict,
    sequencing_mode: str = "auto",
    time_limit_per_machine: float = 2.0,
    max_workers: Optional[int] = None,
) -> Tuple[float, int, List[List[int]], List[List[int]]]:
    """Parallel version of evaluate_assignment — evaluates machines concurrently.

    Uses ProcessPoolExecutor to distribute per-machine DP across cores.
    Falls back to sequential evaluation for small instances (<= 4 machines).
    """
    m = inst["m"]
    n = inst["n"]
    T = inst["T"]
    p = inst["p"]
    e = inst["e"]
    ct = inst["ct"]

    machine_jobs: List[List[int]] = [[] for _ in range(m)]
    for j in range(n):
        mi = assignment[j]
        if 0 <= mi < m:
            machine_jobs[mi].append(j)

    # For small instances, sequential is faster (no fork overhead)
    if m <= 4:
        return evaluate_assignment(
            assignment, inst, sequencing_mode, time_limit_per_machine
        )

    if max_workers is None:
        max_workers = min(m, os.cpu_count() or 4)

    tasks = []
    for mi in range(m):
        jobs = machine_jobs[mi]
        job_p = [p[j] for j in jobs]
        tasks.append(
            (jobs, job_p, ct, e[mi], T, sequencing_mode, time_limit_per_machine)
        )

    total_energy = 0.0
    makespan = 0
    sequences: List[List[int]] = [[] for _ in range(m)]
    all_starts: List[List[int]] = [[] for _ in range(m)]

    # Use fork context for speed (Cython module is already loaded in parent)
    import multiprocessing as _mp

    try:
        ctx = _mp.get_context("fork")
    except ValueError:
        ctx = _mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = {
            executor.submit(_eval_single_machine_task, t): mi
            for mi, t in enumerate(tasks)
        }
        for future in as_completed(futures):
            mi = futures[future]
            try:
                energy, finish, ordered_jobs, starts, _ = future.result()
                if energy == float("inf"):
                    return (
                        float("inf"),
                        T + 1,
                        [[] for _ in range(m)],
                        [[] for _ in range(m)],
                    )
                sequences[mi] = ordered_jobs
                all_starts[mi] = starts
                total_energy += energy
                makespan = max(makespan, finish)
            except Exception as exc:
                logger.error("Machine %d evaluation failed: %s", mi, exc)
                return (
                    float("inf"),
                    T + 1,
                    [[] for _ in range(m)],
                    [[] for _ in range(m)],
                )

    return total_energy, makespan, sequences, all_starts


# ---------------------------------------------------------------------------
# Assignment ↔ Sequences conversion utilities
# ---------------------------------------------------------------------------


def assignment_to_sequences(assignment: List[int], m: int) -> List[List[int]]:
    """Convert an assignment vector to sequences (arbitrary order within machines)."""
    seqs: List[List[int]] = [[] for _ in range(m)]
    for j, mi in enumerate(assignment):
        if 0 <= mi < m:
            seqs[mi].append(j)
    return seqs


def sequences_to_assignment(sequences: List[List[int]], n: int) -> List[int]:
    """Convert sequences back to an assignment vector."""
    assignment = [-1] * n
    for mi, seq in enumerate(sequences):
        for j in seq:
            if 0 <= j < n:
                assignment[j] = mi
    return assignment


def make_initial_assignment(inst: dict) -> List[int]:
    """LPT round-robin assignment (same as existing make_initial_solution, but returns assignment)."""
    n, m = inst["n"], inst["m"]
    p = inst["p"]
    jobs = sorted(range(n), key=lambda j: -p[j])
    loads = [0] * m
    assignment = [0] * n
    for j in jobs:
        mi = min(range(m), key=lambda i: loads[i])
        assignment[j] = mi
        loads[mi] += p[j]
    return assignment

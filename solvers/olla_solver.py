"""
MAPA — CWP-Seeded Marginal Assignment with DP Scheduling

A novel near-optimal heuristic for assigning N jobs to M identical parallel machines
to minimize TEC (Total Energy Cost) under TOU electricity prices.

Algorithm Overview
------------------
Phase 1  CWP Greedy Initialization:
           - Run the CWP heuristic's assignment logic.
           - Extract the exact per-machine job sequences. This guarantees a highly
             competitive starting point and sets a strict upper bound on TEC.
Phase 2  Exact Per-Machine DP Optimization:
           - For each machine, evaluate the exact optimal timing for the CWP sequence
             using sequence DP.
           - If the exact state space is tractable (≤ 80K states), run sparse DP to jointly
             optimize the sequence and timing.
           - Adopt the better of the CWP sequence or the Sparse DP sequence.
Phase 3  Pipe-VND local search (VND 2024):
           - N2: inter-machine single-job insertion.
           - N3: inter-machine pairwise job swap.
           - Candidate evaluations use exact sequence DP (or sparse DP if states ≤ 30K)
             to ensure monotonic TEC improvement.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── DP imports ────────────────────────────────────────────────────────────────
try:
    from PaST.solvers.baselines_sequence_dp import _dp_schedule_fixed_order
    from PaST.solvers.optimal_benchmark_dp import solve_optimal_benchmark_dp as _sparse_dp
    from PaST.solvers.cwp_solver import construct_schedule_CWP
except ImportError:
    from solvers.baselines_sequence_dp import _dp_schedule_fixed_order  # type: ignore
    from solvers.optimal_benchmark_dp import solve_optimal_benchmark_dp as _sparse_dp  # type: ignore
    from solvers.cwp_solver import construct_schedule_CWP  # type: ignore

# ── Sparse DP limits ──────────────────────────────────────────────────────────
_SPARSE_MAX_STATES       = 80_000   # for Phase 2 (per-machine DP optimization)
_SPARSE_MAX_STATES_LS    = 30_000   # for Phase 3 (local search evaluations)
_SPARSE_TIME_LIMIT       = 0.5      # seconds per machine
_LS_DP_BUDGET            = 300_000  # fast sequence DP budget check


@dataclass
class MAPAResult:
    assignment: Dict[int, int]
    schedules: Dict[int, List[Tuple[int, int, int]]]
    total_cost: float
    machine_costs: List[float]
    machine_orders: Dict[int, List[int]]
    makespan: int
    feasible: bool


@dataclass
class ParetoResult:
    front: List[Tuple[int, float]] = field(default_factory=list)
    solutions: List[MAPAResult] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 2 & 3 — DP Scheduling Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _dp_budget_ok(job_ids: List[int], p: List[int], T: int) -> bool:
    """Return True if running a DP on this job set is expected to be fast."""
    return (len(job_ids) * T * T) <= _LS_DP_BUDGET


def _evaluate_machine_schedule(
    order: List[int],
    processing_times: List[int],
    ct: np.ndarray,
    e_i: float,
    epsilon: int,
    allow_sparse_dp: bool = False,
    state_cap: int = _SPARSE_MAX_STATES
) -> Tuple[float, List[Tuple[int, int, int]], List[int], int]:
    """Evaluate a machine load.

    If allow_sparse_dp is True and state space <= state_cap, use sparse_dp to find
    the optimal sequence. Otherwise, just use sequence DP on the provided order.
    Returns: (cost, schedule, best_order, finish_time)
    """
    if not order:
        return 0.0, [], [], 0

    p_sub = [processing_times[j] for j in order]
    if sum(p_sub) > epsilon:
        return math.inf, [], [], 0

    best_cost = math.inf
    best_sched: List[Tuple[int, int, int]] = []
    best_order: List[int] = order

    # 1. Try exact Sparse DP if allowed & tractable
    if allow_sparse_dp:
        counts = Counter(p_sub)
        n_states = 1
        for c in counts.values():
            n_states *= (c + 1)
        
        if n_states <= state_cap:
            prices = (ct[:epsilon] * float(e_i)).astype(np.float64)
            res = _sparse_dp(
                processing_times=p_sub,
                prices=prices,
                job_ids=order,
                tie_break="early",
                time_limit=_SPARSE_TIME_LIMIT,
                track_schedule=True,
                max_states=state_cap,
            )
            if res.feasible and math.isfinite(res.cost) and not res.timed_out:
                sched = list(res.schedule)
                b_order = [j for j, _, _ in sched]
                ft = max((e for _, _, e in sched), default=0)
                return float(res.cost), sched, b_order, ft

    # 2. Sequence DP on the provided order
    ct_int = ct[:epsilon].astype(np.int32)
    e_int = int(round(e_i))
    cost, starts = _dp_schedule_fixed_order(
        processing_times=p_sub,
        ct=ct_int,
        e_single=e_int,
        T_limit=epsilon,
        dp_time_penalty=0.0,
    )

    if math.isfinite(cost):
        best_cost = cost
        best_sched = [
            (j, int(s), int(s + processing_times[j]))
            for j, s in zip(order, starts)
        ]

    ft = max((e for _, _, e in best_sched), default=0)
    return float(best_cost), best_sched, best_order, ft


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 3 — Pipe-VND Local Search
# ══════════════════════════════════════════════════════════════════════════════

def _vnd_insert(
    assignment: Dict[int, int],
    machine_costs: List[float],
    machine_schedules: Dict[int, List[Tuple[int, int, int]]],
    machine_orders: Dict[int, List[int]],
    processing_times: List[int],
    energy_rates: List[float],
    ct: np.ndarray,
    epsilon: int,
    n_iter: int,
    top_k: int,
) -> bool:
    """N2: inter-machine single-job insertion. Runs in-place."""
    n, m = len(processing_times), len(energy_rates)
    improved_any = False

    for _ in range(n_iter):
        improved = False
        loads = [sum(processing_times[j] for j in machine_orders[i]) for i in range(m)]

        candidates: List[Tuple[float, int, int, int]] = []
        for j in range(n):
            src = assignment[j]
            d = processing_times[j]
            for dst in range(m):
                if dst == src or loads[dst] + d > epsilon:
                    continue
                score = abs(energy_rates[src] - energy_rates[dst]) * d
                candidates.append((score, j, src, dst))

        candidates.sort(key=lambda x: -x[0])

        for _, j, src, dst in candidates[:top_k]:
            if loads[dst] + processing_times[j] > epsilon:
                continue
            
            # The new ordering: preserve relative order, append new job at the end
            # Sequence DP will find its optimal insertion slot automatically.
            src_order = [jj for jj in machine_orders[src] if jj != j]
            dst_order = machine_orders[dst] + [j]

            # Fast budget check
            if not _dp_budget_ok(src_order, processing_times, epsilon) or \
               not _dp_budget_ok(dst_order, processing_times, epsilon):
                continue

            src_c, src_s, src_o, _ = _evaluate_machine_schedule(
                src_order, processing_times, ct, energy_rates[src], epsilon,
                allow_sparse_dp=True, state_cap=_SPARSE_MAX_STATES_LS
            )
            if not math.isfinite(src_c):
                continue

            dst_c, dst_s, dst_o, _ = _evaluate_machine_schedule(
                dst_order, processing_times, ct, energy_rates[dst], epsilon,
                allow_sparse_dp=True, state_cap=_SPARSE_MAX_STATES_LS
            )
            if not math.isfinite(dst_c):
                continue

            if src_c + dst_c < machine_costs[src] + machine_costs[dst] - 1e-9:
                assignment[j] = dst
                loads[src] -= processing_times[j]
                loads[dst] += processing_times[j]
                machine_costs[src] = src_c
                machine_costs[dst] = dst_c
                machine_schedules[src] = src_s
                machine_schedules[dst] = dst_s
                machine_orders[src] = src_o
                machine_orders[dst] = dst_o
                improved = True
                improved_any = True
                break

        if not improved:
            break

    return improved_any


def _vnd_swap(
    assignment: Dict[int, int],
    machine_costs: List[float],
    machine_schedules: Dict[int, List[Tuple[int, int, int]]],
    machine_orders: Dict[int, List[int]],
    processing_times: List[int],
    energy_rates: List[float],
    ct: np.ndarray,
    epsilon: int,
    n_iter: int,
    top_k: int,
) -> bool:
    """N3: inter-machine pairwise job swap. Runs in-place."""
    n, m = len(processing_times), len(energy_rates)
    improved_any = False

    for _ in range(n_iter):
        improved = False
        loads = [sum(processing_times[j] for j in machine_orders[i]) for i in range(m)]

        candidates: List[Tuple[float, int, int]] = []
        for j in range(n):
            for k in range(j + 1, n):
                src, dst = assignment[j], assignment[k]
                if src == dst:
                    continue
                dj, dk = processing_times[j], processing_times[k]
                if dj == dk:
                    continue
                score = abs(energy_rates[src] - energy_rates[dst]) * abs(dj - dk)
                candidates.append((score, j, k))

        candidates.sort(key=lambda x: -x[0])

        for _, j, k in candidates[:top_k]:
            src, dst = assignment[j], assignment[k]
            dj, dk = processing_times[j], processing_times[k]
            
            if loads[src] - dj + dk > epsilon or loads[dst] - dk + dj > epsilon:
                continue

            src_order = [jj for jj in machine_orders[src] if jj != j] + [k]
            dst_order = [jj for jj in machine_orders[dst] if jj != k] + [j]

            if not _dp_budget_ok(src_order, processing_times, epsilon) or \
               not _dp_budget_ok(dst_order, processing_times, epsilon):
                continue

            src_c, src_s, src_o, _ = _evaluate_machine_schedule(
                src_order, processing_times, ct, energy_rates[src], epsilon,
                allow_sparse_dp=True, state_cap=_SPARSE_MAX_STATES_LS
            )
            if not math.isfinite(src_c):
                continue

            dst_c, dst_s, dst_o, _ = _evaluate_machine_schedule(
                dst_order, processing_times, ct, energy_rates[dst], epsilon,
                allow_sparse_dp=True, state_cap=_SPARSE_MAX_STATES_LS
            )
            if not math.isfinite(dst_c):
                continue

            if src_c + dst_c < machine_costs[src] + machine_costs[dst] - 1e-9:
                assignment[j], assignment[k] = dst, src
                loads[src] = loads[src] - dj + dk
                loads[dst] = loads[dst] - dk + dj
                machine_costs[src] = src_c
                machine_costs[dst] = dst_c
                machine_schedules[src] = src_s
                machine_schedules[dst] = dst_s
                machine_orders[src] = src_o
                machine_orders[dst] = dst_o
                improved = True
                improved_any = True
                break

        if not improved:
            break

    return improved_any


# ══════════════════════════════════════════════════════════════════════════════
#  Main solver
# ══════════════════════════════════════════════════════════════════════════════

def solve_olla(
    processing_times: List[int],
    energy_rates: List[float],
    ct: np.ndarray,
    epsilon: Optional[int] = None,
    *,
    local_search: bool = True,
    local_search_iters: int = 4,
    local_search_top_k: int = 40,
) -> MAPAResult:
    n, m = len(processing_times), len(energy_rates)
    T = len(ct)
    if epsilon is None:
        epsilon = T
    epsilon = min(epsilon, T)

    if n == 0:
        return MAPAResult({}, {i: [] for i in range(m)}, 0.0, [0.0]*m, {i: [] for i in range(m)}, 0, True)

    # ── Phase 1: CWP Greedy Initialization ────────────────────────────────
    # Runs the CWP sequence exactly. Guaranteed physically feasible starting point.
    cwp_res = construct_schedule_CWP(processing_times, energy_rates, ct, epsilon)
    
    assignment = cwp_res.assignment
    machine_orders = {i: [] for i in range(m)}
    for j, mi in assignment.items():
        machine_orders[mi].append(j)
    for i in range(m):
        machine_orders[i].sort(key=lambda j: cwp_res.start_times[j])

    # ── Phase 2: Exact Per-Machine DP Optimization ────────────────────────
    machine_costs = [0.0] * m
    machine_schedules = {i: [] for i in range(m)}
    feasible = True

    for i in range(m):
        c, s, o, _ = _evaluate_machine_schedule(
            machine_orders[i], processing_times, ct, energy_rates[i], epsilon,
            allow_sparse_dp=True, state_cap=_SPARSE_MAX_STATES
        )
        if math.isfinite(c):
            machine_costs[i] = c
            machine_schedules[i] = s
            machine_orders[i] = o
        else:
            machine_costs[i] = math.inf
            feasible = False

    # ── Phase 3: Pipe-VND Local Search ────────────────────────────────────
    if feasible and local_search:
        _vnd_insert(
            assignment, machine_costs, machine_schedules, machine_orders,
            processing_times, energy_rates, ct, epsilon,
            n_iter=local_search_iters, top_k=local_search_top_k
        )
        _vnd_swap(
            assignment, machine_costs, machine_schedules, machine_orders,
            processing_times, energy_rates, ct, epsilon,
            n_iter=local_search_iters, top_k=max(1, local_search_top_k // 2)
        )

    makespan = max(
        (max(e for _, _, e in s) for s in machine_schedules.values() if s),
        default=0,
    )

    return MAPAResult(
        assignment=assignment,
        schedules=machine_schedules,
        total_cost=sum(machine_costs),
        machine_costs=machine_costs,
        machine_orders=machine_orders,
        makespan=makespan,
        feasible=feasible
    )


# ── Pareto sweep ──────────────────────────────────────────────────────────────

def solve_olla_pareto(
    processing_times: List[int],
    energy_rates: List[float],
    ct: np.ndarray,
    *,
    local_search: bool = True,
    local_search_iters: int = 4,
    local_search_top_k: int = 40,
) -> ParetoResult:
    T = len(ct)
    lb = max(
        math.ceil(sum(processing_times) / len(energy_rates)) if energy_rates else 0,
        max(processing_times) if processing_times else 0,
    )
    result = ParetoResult()
    pareto_pts: List[Tuple[int, float]] = []
    epsilon = T

    while epsilon >= lb:
        sol = solve_olla(
            processing_times, energy_rates, ct, epsilon,
            local_search=local_search,
            local_search_iters=local_search_iters,
            local_search_top_k=local_search_top_k,
        )
        if sol.feasible and math.isfinite(sol.total_cost):
            dominated = any(cm <= sol.makespan and ec <= sol.total_cost for cm, ec in pareto_pts)
            if not dominated:
                pareto_pts = [(cm, ec) for cm, ec in pareto_pts
                              if not (sol.makespan <= cm and sol.total_cost <= ec)]
                pareto_pts.append((sol.makespan, sol.total_cost))
                result.front.append((sol.makespan, sol.total_cost))
                result.solutions.append(sol)
            epsilon = sol.makespan - 1
        else:
            epsilon -= 1

    idx = sorted(range(len(result.front)), key=lambda i: result.front[i][0])
    result.front = [result.front[i] for i in idx]
    result.solutions = [result.solutions[i] for i in idx]
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  __main__: sanity + full benchmark
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json, os, time as _time

    print("=" * 64)
    print("MAPA Solver — Sanity + Benchmark (MAPA vs CWP)")
    print("=" * 64)

    bench = os.path.join(os.path.dirname(__file__), "..", "New Benchmark", "instances_90.json")
    if not os.path.exists(bench):
        print("instances_90.json not found — skipping benchmark.")
    else:
        with open(bench) as f:
            instances = json.load(f)

        wins = ties = losses = 0
        print(f"\n{'ID':>3} {'n':>4} {'m':>2} {'T':>4} | {'CWP':>10} | {'MAPA':>10} | {'Δ%':>7} | {'t':>6}")
        print("-" * 66)

        for inst in instances:
            p  = inst["p"]
            e  = [float(x) for x in inst["e"]]
            ct = np.array(inst["ct"], dtype=np.float64)
            
            # CWP Baseline
            rc = construct_schedule_CWP(p, e, ct)
            cwp_c  = rc.total_cost
            
            # MAPA Solver
            t0 = _time.perf_counter()
            ro = solve_olla(p, e, ct, local_search=True)
            dt = _time.perf_counter() - t0
            
            mapa_c = ro.total_cost if ro.feasible else math.inf
            pct = (mapa_c - cwp_c) / cwp_c * 100 if cwp_c > 0 else 0.0
            
            if   mapa_c < cwp_c - 1e-6: wins  += 1
            elif mapa_c > cwp_c + 1e-6: losses += 1
            else:                        ties  += 1
            mark = "✓" if mapa_c <= cwp_c + 1e-6 else "✗"
            print(f"{mark} {inst['instance_id']:>3} {inst['n']:>4} {inst['m']:>2} {inst['T']:>4}"
                  f" | {cwp_c:>10.1f} | {mapa_c:>10.1f} | {pct:>+7.2f}% | {dt:>6.2f}s")

        print(f"\nResult: wins={wins}  ties={ties}  losses={losses} out of {len(instances)}")

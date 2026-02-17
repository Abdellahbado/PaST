from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class FeasibleAssignmentResult:
    assignment: List[int]
    loads: Tuple[int, ...]


def greedy_feasible_assignment(
    *,
    processing_times: Sequence[int],
    n_machines: int,
    K: int,
    rng: np.random.Generator,
) -> Optional[FeasibleAssignmentResult]:
    """Create a feasible assignment (sum p per machine <= K) via greedy bin packing.

    Deterministic given rng state.
    """
    n = int(len(processing_times))
    m = int(n_machines)
    K = int(K)
    if m <= 0 or K <= 0:
        return None

    p = np.asarray(processing_times, dtype=np.int64)

    # Order jobs by decreasing p, shuffle ties for diversity.
    idx = np.arange(n, dtype=np.int64)
    tie = rng.random(n)
    order = sorted(idx.tolist(), key=lambda j: (-int(p[j]), float(tie[int(j)]), int(j)))

    loads = [0] * m
    assignment = [-1] * n

    for j in order:
        pj = int(p[int(j)])
        candidates = [mi for mi in range(m) if loads[mi] + pj <= K]
        if not candidates:
            return None

        min_load = min(loads[mi] for mi in candidates)
        best = [mi for mi in candidates if loads[mi] == min_load]
        mi = int(best[int(rng.integers(len(best)))])
        assignment[int(j)] = mi
        loads[mi] += pj

    return FeasibleAssignmentResult(
        assignment=[int(x) for x in assignment],
        loads=tuple(int(x) for x in loads),
    )


def mutate_assignment_relocate(
    *,
    assignment: Sequence[int],
    processing_times: Sequence[int],
    n_machines: int,
    K: int,
    rng: np.random.Generator,
    n_attempts: int = 50,
) -> Optional[FeasibleAssignmentResult]:
    """Random feasible mutation: relocate one job to another machine."""
    n = int(len(assignment))
    m = int(n_machines)
    p = np.asarray(processing_times, dtype=np.int64)

    loads = [0] * m
    for j in range(n):
        loads[int(assignment[j])] += int(p[j])

    for _ in range(int(n_attempts)):
        j = int(rng.integers(n))
        from_m = int(assignment[j])
        pj = int(p[j])

        to_m = int(rng.integers(m))
        if to_m == from_m:
            continue
        if loads[to_m] + pj > int(K):
            continue

        new_assignment = list(int(x) for x in assignment)
        new_assignment[j] = to_m
        new_loads = list(loads)
        new_loads[from_m] -= pj
        new_loads[to_m] += pj
        return FeasibleAssignmentResult(
            assignment=new_assignment,
            loads=tuple(int(x) for x in new_loads),
        )

    return None


def mutate_assignment_swap(
    *,
    assignment: Sequence[int],
    n_machines: int,
    rng: np.random.Generator,
    n_attempts: int = 50,
) -> Optional[List[int]]:
    """Swap the machines of two jobs (always preserves per-machine loads).

    This mutation increases diversity without affecting feasibility.
    """
    n = int(len(assignment))
    if n <= 1:
        return None

    for _ in range(int(n_attempts)):
        j1 = int(rng.integers(n))
        j2 = int(rng.integers(n))
        if j2 == j1:
            continue
        m1 = int(assignment[j1])
        m2 = int(assignment[j2])
        if m1 == m2:
            continue
        if m1 < 0 or m1 >= int(n_machines) or m2 < 0 or m2 >= int(n_machines):
            continue

        new_assignment = list(int(x) for x in assignment)
        new_assignment[j1] = m2
        new_assignment[j2] = m1
        return new_assignment

    return None


def _hard_negative_perturbation(
    *,
    assignment: Sequence[int],
    processing_times: Sequence[int],
    n_machines: int,
    K: int,
    rng: np.random.Generator,
    machine_energy_rates: Optional[Sequence[float]] = None,
) -> Optional[FeasibleAssignmentResult]:
    """Create a worse-but-feasible neighbor to enrich ranking groups.

    Strategy: try moving one of the longest jobs to the highest-rate machine.
    """
    n = int(len(assignment))
    m = int(n_machines)
    if n == 0 or m == 0:
        return None

    p = np.asarray(processing_times, dtype=np.int64)
    loads = [0] * m
    for j in range(n):
        loads[int(assignment[j])] += int(p[j])

    if machine_energy_rates is not None:
        rates = np.asarray(machine_energy_rates, dtype=np.float64)
        worst_m = int(np.argmax(rates))
    else:
        worst_m = int(rng.integers(m))

    # Candidate long jobs
    order = np.argsort(-p, kind="stable")
    for j in order[: min(10, n)]:
        j = int(j)
        from_m = int(assignment[j])
        if from_m == worst_m:
            continue
        pj = int(p[j])
        if loads[worst_m] + pj > int(K):
            continue
        new_assignment = list(int(x) for x in assignment)
        new_assignment[j] = int(worst_m)
        new_loads = list(loads)
        new_loads[from_m] -= pj
        new_loads[worst_m] += pj
        return FeasibleAssignmentResult(
            assignment=new_assignment,
            loads=tuple(int(x) for x in new_loads),
        )

    return None


def build_candidate_assignment_pool(
    *,
    processing_times: Sequence[int],
    machine_energy_rates: Optional[Sequence[float]] = None,
    n_machines: int,
    K: int,
    pool_size: int,
    seed: int,
    n_init_tries: int = 200,
) -> List[List[int]]:
    """Generate a pool of diverse feasible assignments for a fixed (instance, K).

    Produces:
    - multiple feasible restarts (diversity)
    - a random walk with relocate + swap mutations
    - a few hard negatives (still feasible)
    """
    rng = np.random.default_rng(int(seed))

    # Multiple feasible bases
    bases: List[FeasibleAssignmentResult] = []
    for _ in range(int(n_init_tries)):
        base = greedy_feasible_assignment(
            processing_times=processing_times,
            n_machines=n_machines,
            K=K,
            rng=rng,
        )
        if base is None:
            continue
        bases.append(base)
        if len(bases) >= 5:
            break

    if not bases:
        return []

    pool: List[List[int]] = []
    for b in bases:
        pool.append(b.assignment)

    # Random walks from each base.
    max_total_attempts = int(pool_size) * 400
    attempts = 0
    base_idx = 0
    cur = bases[0]

    while len(pool) < int(pool_size) and attempts < max_total_attempts:
        attempts += 1

        # Occasionally restart from a different base.
        if attempts % 50 == 0:
            base_idx = (base_idx + 1) % len(bases)
            cur = bases[base_idx]

        # Mix of relocate and swap.
        if float(rng.random()) < 0.7:
            mutated = mutate_assignment_relocate(
                assignment=cur.assignment,
                processing_times=processing_times,
                n_machines=n_machines,
                K=K,
                rng=rng,
                n_attempts=50,
            )
            if mutated is None:
                continue
            cur = mutated
            pool.append(cur.assignment)
        else:
            swapped = mutate_assignment_swap(
                assignment=cur.assignment,
                n_machines=n_machines,
                rng=rng,
                n_attempts=50,
            )
            if swapped is None:
                continue
            # swap doesn't change loads, so keep loads from cur
            cur = FeasibleAssignmentResult(
                assignment=swapped,
                loads=cur.loads,
            )
            pool.append(cur.assignment)

        # Inject occasional hard negatives derived from current.
        if len(pool) < int(pool_size) and attempts % 40 == 0:
            neg = _hard_negative_perturbation(
                assignment=cur.assignment,
                processing_times=processing_times,
                n_machines=n_machines,
                K=K,
                rng=rng,
                machine_energy_rates=machine_energy_rates,
            )
            if neg is not None:
                pool.append(neg.assignment)

    # Deduplicate (important for ranking groups)
    seen = set()
    uniq: List[List[int]] = []
    for a in pool:
        key = tuple(a)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(a)

    return uniq

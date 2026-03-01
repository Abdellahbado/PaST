"""Assignment-only seed operators for the decomposed G-LNS.

In the assignment-only framework:
- A solution is `assignment: List[int]` of length n, where assignment[j] = machine index
- Destroy operators set some assignment[j] = -1 (unassigned)
- Repair operators assign all unassigned jobs to machines (just pick a machine, no ordering)
- Sequencing is handled separately by the optimal DP solver

This dramatically simplifies the operator design space: repair only decides
WHICH MACHINE, not where in the sequence.
"""

from __future__ import annotations

from glns.schemas import OperatorRecord, OperatorSpec


# ---------------------------------------------------------------------------
# Destroy operators (assignment-only)
# ---------------------------------------------------------------------------

RANDOM_REMOVAL_CODE = '''\
def destroy(assignment, destroy_cnt, instance, rng):
    """Remove destroy_cnt random jobs from the assignment."""
    n = instance["n"]
    assigned = [j for j in range(n) if assignment[j] >= 0]
    if not assigned:
        return [], list(assignment)
    k = min(destroy_cnt, len(assigned))
    removed = rng.sample(assigned, k)
    partial = list(assignment)
    for j in removed:
        partial[j] = -1
    return removed, partial
'''

WORST_ENERGY_REMOVAL_CODE = '''\
def destroy(assignment, destroy_cnt, instance, rng):
    """Remove jobs contributing most to estimated energy cost.

    Proxy: e[machine] * p[job] * avg_price_over_horizon.
    The actual cost depends on DP scheduling, but this is a cheap proxy.
    """
    p = instance["p"]
    e = instance["e"]
    ct = instance["ct"]
    T = instance["T"]
    avg_price = sum(ct) / max(T, 1)
    n = instance["n"]
    costs = []
    for j in range(n):
        mi = assignment[j]
        if mi >= 0:
            est = e[mi] * p[j] * avg_price
            costs.append((est, j))
    costs.sort(key=lambda x: -x[0])
    k = min(destroy_cnt, len(costs))
    removed = [j for _, j in costs[:k]]
    partial = list(assignment)
    for j in removed:
        partial[j] = -1
    return removed, partial
'''

MOST_LOADED_MACHINE_REMOVAL_CODE = '''\
def destroy(assignment, destroy_cnt, instance, rng):
    """Remove jobs from the most loaded machine (reduces makespan)."""
    p = instance["p"]
    m = instance["m"]
    n = instance["n"]
    loads = [0] * m
    machine_jobs = [[] for _ in range(m)]
    for j in range(n):
        mi = assignment[j]
        if mi >= 0:
            loads[mi] += p[j]
            machine_jobs[mi].append(j)
    target = max(range(m), key=lambda i: loads[i])
    jobs = machine_jobs[target]
    k = min(destroy_cnt, len(jobs))
    removed = rng.sample(jobs, k) if k > 0 else []
    partial = list(assignment)
    for j in removed:
        partial[j] = -1
    return removed, partial
'''

PEAK_PERIOD_REMOVAL_CODE = '''\
def destroy(assignment, destroy_cnt, instance, rng):
    """Remove longest jobs from high-energy-rate machines.

    These are the jobs most likely to incur high TEC.
    Prioritises (p[j] * e[machine]) to capture both dimensions.
    """
    p = instance["p"]
    e = instance["e"]
    n = instance["n"]
    items = []
    for j in range(n):
        mi = assignment[j]
        if mi >= 0:
            items.append((p[j] * e[mi], j))
    items.sort(key=lambda x: -x[0])
    k = min(destroy_cnt, len(items))
    removed = [j for _, j in items[:k]]
    partial = list(assignment)
    for j in removed:
        partial[j] = -1
    return removed, partial
'''


# ---------------------------------------------------------------------------
# Repair operators (assignment-only)
# ---------------------------------------------------------------------------

GREEDY_LOAD_BALANCE_CODE = '''\
def repair(partial_assignment, removed_jobs, instance, rng):
    """Assign each removed job to the least-loaded machine (minimise makespan)."""
    p = instance["p"]
    m = instance["m"]
    n = instance["n"]
    assignment = list(partial_assignment)
    loads = [0] * m
    for j in range(n):
        if assignment[j] >= 0:
            loads[assignment[j]] += p[j]
    # Sort removed jobs by decreasing processing time (LPT-like)
    jobs = sorted(removed_jobs, key=lambda j: -p[j])
    for j in jobs:
        mi = min(range(m), key=lambda i: loads[i])
        assignment[j] = mi
        loads[mi] += p[j]
    return assignment
'''

ENERGY_AWARE_ASSIGNMENT_CODE = '''\
def repair(partial_assignment, removed_jobs, instance, rng):
    """Assign each removed job to the machine minimising estimated energy.

    Considers both the machine energy rate e[h] and the current load
    (proxy for where in the TOU schedule the job will land).
    """
    p = instance["p"]
    e = instance["e"]
    ct = instance["ct"]
    m = instance["m"]
    n = instance["n"]
    T = instance["T"]
    assignment = list(partial_assignment)
    loads = [0] * m
    for j in range(n):
        if assignment[j] >= 0:
            loads[assignment[j]] += p[j]
    # Precompute avg price per block of T/m time slots
    # (crude proxy for TOU cost at different time positions)
    block = max(1, T // max(m, 1))
    block_prices = []
    for b in range(0, T, block):
        end = min(b + block, T)
        block_prices.append(sum(ct[b:end]) / max(end - b, 1))
    if not block_prices:
        block_prices = [1.0]
    jobs = sorted(removed_jobs, key=lambda j: -p[j])
    for j in jobs:
        best_mi = 0
        best_cost = float("inf")
        for mi in range(m):
            # Estimate energy: e[mi] * p[j] * avg_price_at_load_position
            load_pos = min(loads[mi] // max(block, 1), len(block_prices) - 1)
            est = e[mi] * p[j] * block_prices[load_pos]
            if est < best_cost:
                best_cost = est
                best_mi = mi
        assignment[j] = best_mi
        loads[best_mi] += p[j]
    return assignment
'''

RANDOM_ASSIGNMENT_CODE = '''\
def repair(partial_assignment, removed_jobs, instance, rng):
    """Assign removed jobs to random machines (pure diversification)."""
    m = instance["m"]
    assignment = list(partial_assignment)
    for j in removed_jobs:
        assignment[j] = rng.randrange(m)
    return assignment
'''


# ---------------------------------------------------------------------------
# Build OperatorRecord objects
# ---------------------------------------------------------------------------


def build_seed_destroy_operators_v2() -> list[OperatorRecord]:
    """Return 4 hand-coded assignment-only destroy operators."""
    specs = [
        OperatorSpec(
            type="destroy",
            idea="Remove random jobs uniformly (assignment-only)",
            code=RANDOM_REMOVAL_CODE,
        ),
        OperatorSpec(
            type="destroy",
            idea="Remove jobs with highest estimated energy contribution",
            code=WORST_ENERGY_REMOVAL_CODE,
        ),
        OperatorSpec(
            type="destroy",
            idea="Remove jobs from the most-loaded machine to reduce Cmax",
            code=MOST_LOADED_MACHINE_REMOVAL_CODE,
        ),
        OperatorSpec(
            type="destroy",
            idea="Remove high p*e jobs (energy-rate × processing time)",
            code=PEAK_PERIOD_REMOVAL_CODE,
        ),
    ]
    return [
        OperatorRecord(id=f"seed_d{i}", spec=s, generation_born=0)
        for i, s in enumerate(specs)
    ]


def build_seed_repair_operators_v2() -> list[OperatorRecord]:
    """Return 3 hand-coded assignment-only repair operators."""
    specs = [
        OperatorSpec(
            type="repair",
            idea="LPT load-balance assignment (minimise makespan)",
            code=GREEDY_LOAD_BALANCE_CODE,
        ),
        OperatorSpec(
            type="repair",
            idea="Energy-aware TOU-position assignment (minimise TEC)",
            code=ENERGY_AWARE_ASSIGNMENT_CODE,
        ),
        OperatorSpec(
            type="repair",
            idea="Random machine assignment (diversification)",
            code=RANDOM_ASSIGNMENT_CODE,
        ),
    ]
    return [
        OperatorRecord(id=f"seed_r{i}", spec=s, generation_born=0)
        for i, s in enumerate(specs)
    ]

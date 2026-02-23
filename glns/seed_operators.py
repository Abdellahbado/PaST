"""Hand-coded seed operators that bootstrap the destroy/repair populations.

These serve two purposes:
1. Provide working operators from generation 0 (before the LLM contributes).
2. Appear as in-context examples in the system prompt so the LLM understands
   the exact interface contract.

All operators follow the same signature contract as LLM-generated ones and
are fully self-contained (no external imports beyond stdlib).
"""

from __future__ import annotations

from glns.schemas import OperatorRecord, OperatorSpec


# ---------------------------------------------------------------------------
# Destroy operators
# ---------------------------------------------------------------------------

RANDOM_REMOVAL_CODE = '''\
def destroy(solution, destroy_cnt, instance, rng):
    """Remove destroy_cnt random jobs from the solution."""
    import copy
    flat = []
    for mi, seq in enumerate(solution):
        for pos, job in enumerate(seq):
            flat.append((mi, pos, job))
    if not flat:
        return [], [list(s) for s in solution]
    k = min(destroy_cnt, len(flat))
    picked = rng.sample(flat, k)
    removed = [job for (_, _, job) in picked]
    # Build partial by removing picked jobs.
    to_remove_per_machine = {}
    for mi, pos, job in picked:
        to_remove_per_machine.setdefault(mi, set()).add(job)
    partial = []
    for mi, seq in enumerate(solution):
        rm = to_remove_per_machine.get(mi, set())
        partial.append([j for j in seq if j not in rm])
    return removed, partial
'''

WORST_ENERGY_REMOVAL_CODE = '''\
def destroy(solution, destroy_cnt, instance, rng):
    """Remove jobs with the highest estimated energy contribution."""
    ct = instance["ct"]
    e = instance["e"]
    p = instance["p"]
    # Estimate each job\'s energy as e[machine] * p[j] * avg_price.
    # (Rough proxy; true cost depends on DP placement.)
    job_info = []
    for mi, seq in enumerate(solution):
        t_cursor = 0
        for job in seq:
            pj = p[job]
            end = min(t_cursor + pj, len(ct))
            cost = e[mi] * sum(ct[t_cursor:end])
            job_info.append((cost, mi, job))
            t_cursor = end
    job_info.sort(key=lambda x: -x[0])
    k = min(destroy_cnt, len(job_info))
    to_remove = set()
    removed = []
    for cost, mi, job in job_info[:k]:
        to_remove.add(job)
        removed.append(job)
    partial = []
    for seq in solution:
        partial.append([j for j in seq if j not in to_remove])
    return removed, partial
'''

MOST_LOADED_MACHINE_REMOVAL_CODE = '''\
def destroy(solution, destroy_cnt, instance, rng):
    """Remove jobs from the machine with the largest total processing time."""
    p = instance["p"]
    loads = []
    for mi, seq in enumerate(solution):
        loads.append((sum(p[j] for j in seq), mi))
    loads.sort(key=lambda x: -x[0])
    target_mi = loads[0][1] if loads else 0
    seq = solution[target_mi]
    k = min(destroy_cnt, len(seq))
    removed = rng.sample(seq, k) if k > 0 else []
    removed_set = set(removed)
    partial = []
    for mi, s in enumerate(solution):
        if mi == target_mi:
            partial.append([j for j in s if j not in removed_set])
        else:
            partial.append(list(s))
    return removed, partial
'''


# ---------------------------------------------------------------------------
# Repair operators
# ---------------------------------------------------------------------------

GREEDY_INSERTION_CODE = '''\
def repair(partial_solution, removed_jobs, instance, rng):
    """Greedily insert jobs into the position that minimises estimated energy."""
    import copy
    ct = instance["ct"]
    e_rates = instance["e"]
    p = instance["p"]
    m = instance["m"]

    def machine_cost(seq, mi):
        """Estimate TEC for a machine sequence (no DP, just sequential placement)."""
        t = 0
        total = 0
        for job in seq:
            pj = p[job]
            end = min(t + pj, len(ct))
            total += e_rates[mi] * sum(ct[t:end])
            t = end
        return total

    sol = [list(s) for s in partial_solution]
    # Insert longer jobs first.
    jobs = sorted(removed_jobs, key=lambda j: -p[j])
    for job in jobs:
        best_cost = float("inf")
        best_mi = 0
        best_pos = 0
        for mi in range(m):
            for pos in range(len(sol[mi]) + 1):
                trial = list(sol[mi])
                trial.insert(pos, job)
                c = machine_cost(trial, mi)
                if c < best_cost:
                    best_cost = c
                    best_mi = mi
                    best_pos = pos
        sol[best_mi].insert(best_pos, job)
    return sol
'''

RANDOM_INSERTION_CODE = '''\
def repair(partial_solution, removed_jobs, instance, rng):
    """Insert removed jobs into random machines at random positions."""
    sol = [list(s) for s in partial_solution]
    m = instance["m"]
    jobs = list(removed_jobs)
    rng.shuffle(jobs)
    for job in jobs:
        mi = rng.randrange(m)
        pos = rng.randrange(len(sol[mi]) + 1)
        sol[mi].insert(pos, job)
    return sol
'''


# ---------------------------------------------------------------------------
# Build OperatorRecord objects
# ---------------------------------------------------------------------------


def build_seed_destroy_operators() -> list[OperatorRecord]:
    """Return 3 hand-coded destroy operators as OperatorRecord objects."""
    specs = [
        OperatorSpec(
            type="destroy",
            idea="Remove random jobs uniformly across machines",
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
    ]
    return [
        OperatorRecord(id=f"seed_d{i}", spec=s, generation_born=0)
        for i, s in enumerate(specs)
    ]


def build_seed_repair_operators() -> list[OperatorRecord]:
    """Return 2 hand-coded repair operators as OperatorRecord objects."""
    specs = [
        OperatorSpec(
            type="repair",
            idea="Greedy insertion minimising estimated energy cost",
            code=GREEDY_INSERTION_CODE,
        ),
        OperatorSpec(
            type="repair",
            idea="Random insertion for diversification",
            code=RANDOM_INSERTION_CODE,
        ),
    ]
    return [
        OperatorRecord(id=f"seed_r{i}", spec=s, generation_born=0)
        for i, s in enumerate(specs)
    ]

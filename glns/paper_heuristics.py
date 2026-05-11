"""Paper-faithful heuristic reconstructions for BPMSTP.

This module is intentionally separate from the B5 reflective hybrid branch.
It reconstructs the algorithmic structure described in:

* Gaggero, Paolucci, Ronco (2023): SGH, A-SGH, R-ES, ESR, EHS.
* Jarboui, Masmoudi, Eddaly (2024): EOA with Pipe-VND and fixed-sequence DP.

The original authors' executable source code is not present in this workspace.
Therefore this file is a transparent reconstruction from the paper
pseudocode, not a claim of byte-level reproduction.  Ambiguous choices are
kept local and documented by function names/comments.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import random
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from glns.sequencing import _dp_schedule_fixed_order


INF = float("inf")

# Cooperative time-limit enforcement for adversarial benchmark evaluation.
# run_ehs() sets this before the khat loop; inner construction functions
# check it periodically so long SGH/A-SGH calls yield within budget.
_EHS_DEADLINE: Optional[float] = None


def _set_ehs_deadline(limit: Optional[float]) -> None:
    global _EHS_DEADLINE
    _EHS_DEADLINE = (time.time() + limit) if limit is not None else None


def _clear_ehs_deadline() -> None:
    global _EHS_DEADLINE
    _EHS_DEADLINE = None


def _ehs_deadline_expired() -> bool:
    return _EHS_DEADLINE is not None and time.time() > _EHS_DEADLINE


@dataclass(frozen=True)
class ScheduledJob:
    job: int
    machine: int
    start: int
    duration: int

    @property
    def end(self) -> int:
        return self.start + self.duration


@dataclass
class PaperSchedule:
    """Concrete schedule with per-machine job sequences and start times."""

    inst: dict
    machine_jobs: List[List[ScheduledJob]]

    def copy(self) -> "PaperSchedule":
        return PaperSchedule(
            self.inst,
            [[ScheduledJob(j.job, j.machine, j.start, j.duration) for j in seq] for seq in self.machine_jobs],
        )

    @property
    def energy(self) -> float:
        ct = self.inst["ct"]
        e = self.inst["e"]
        total = 0.0
        for h, seq in enumerate(self.machine_jobs):
            for sj in seq:
                if sj.end > len(ct):
                    return INF
                total += float(e[h]) * sum(ct[sj.start : sj.end])
        return total

    @property
    def cmax(self) -> int:
        return max((sj.end for seq in self.machine_jobs for sj in seq), default=0)

    @property
    def feasible(self) -> bool:
        n = self.inst["n"]
        seen = set()
        for h, seq in enumerate(self.machine_jobs):
            occ: Dict[int, int] = {}
            for sj in seq:
                if sj.machine != h or sj.start < 0 or sj.end > self.inst["T"]:
                    return False
                if sj.job in seen:
                    return False
                seen.add(sj.job)
                for t in range(sj.start, sj.end):
                    if t in occ:
                        return False
                    occ[t] = sj.job
        return len(seen) == n

    def sequences(self) -> List[List[int]]:
        return [[sj.job for sj in sorted(seq, key=lambda x: (x.start, x.job))] for seq in self.machine_jobs]

    def as_points(self) -> Tuple[int, float]:
        return (self.cmax, self.energy)


def pareto_filter_schedules(schedules: Iterable[PaperSchedule]) -> List[PaperSchedule]:
    """Return strictly non-dominated schedules sorted by makespan, then energy."""
    kept: List[PaperSchedule] = []
    seen_points: set[Tuple[int, float]] = set()
    for sched in schedules:
        if not sched.feasible or sched.energy == INF:
            continue
        p = sched.as_points()
        if p in seen_points:
            continue
        dominated = False
        survivors: List[PaperSchedule] = []
        for other in kept:
            q = other.as_points()
            if _dominates(q, p):
                dominated = True
                break
            if not _dominates(p, q):
                survivors.append(other)
        if not dominated:
            survivors.append(sched)
            kept = survivors
            seen_points = {s.as_points() for s in kept}
    return sorted(kept, key=lambda s: (s.cmax, s.energy))


def _dominates(a: Tuple[int, float], b: Tuple[int, float]) -> bool:
    return a[0] <= b[0] and a[1] <= b[1] and (a[0] < b[0] or a[1] < b[1])


def _lower_bound_cmax(inst: dict) -> int:
    return max(max(inst["p"], default=0), math.ceil(sum(inst["p"]) / max(inst["m"], 1)))


def _jobs_by_processing_time(inst: dict, jobs: Optional[Iterable[int]] = None) -> List[Tuple[int, List[int]]]:
    chosen = list(range(inst["n"])) if jobs is None else list(jobs)
    by_d: Dict[int, List[int]] = {}
    for j in chosen:
        by_d.setdefault(inst["p"][j], []).append(j)
    return [(d, sorted(js)) for d, js in sorted(by_d.items(), reverse=True)]


def schedule_from_sequences_dp(inst: dict, sequences: Sequence[Sequence[int]], epsilon: int) -> Optional[PaperSchedule]:
    """Evaluate fixed machine sequences using the fixed-sequence DP from the papers."""
    machine_jobs: List[List[ScheduledJob]] = []
    for h, seq in enumerate(sequences):
        durations = [inst["p"][j] for j in seq]
        if sum(durations) > epsilon:
            return None
        energy, starts = _dp_schedule_fixed_order(durations, inst["ct"], inst["e"][h], epsilon)
        if energy == INF:
            return None
        machine_jobs.append(
            [ScheduledJob(job=j, machine=h, start=starts[i], duration=inst["p"][j]) for i, j in enumerate(seq)]
        )
    sched = PaperSchedule(inst, machine_jobs)
    return sched if sched.feasible and sched.cmax <= epsilon else None


# ---------------------------------------------------------------------------
# SGH / A-SGH / EHS reconstruction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Location:
    machine: int
    slots: Tuple[int, ...]
    cost: float

    @property
    def first(self) -> int:
        return self.slots[0]


def _empty_slots_from_jobs(inst: dict, machine_jobs: Sequence[Sequence[ScheduledJob]], khat: int) -> List[List[int]]:
    slots = [[-1 for _ in range(khat)] for _ in range(inst["m"])]
    for h, seq in enumerate(machine_jobs):
        for sj in seq:
            if sj.end > khat:
                continue
            for t in range(sj.start, sj.end):
                slots[h][t] = sj.job
    return slots


def _enumerate_free_locations(slots_h: Sequence[int], machine: int, d: int, inst: dict) -> List[_Location]:
    """Enumerate free locations and free split-locations as in SGH.

    A free split-location is represented as all free slots inside a window
    whose occupied gaps separate the selected free sub-blocks.  This matches
    the paper definition: non-consecutive selected free slots may be separated
    only by slots already assigned to other jobs.
    """
    khat = len(slots_h)
    out: List[_Location] = []
    for left in range(khat):
        free: List[int] = []
        for right in range(left, khat):
            if slots_h[right] < 0:
                free.append(right)
            if len(free) == d:
                cost = float(inst["e"][machine]) * sum(inst["ct"][t] for t in free)
                out.append(_Location(machine, tuple(free), cost))
                break
            # Once the window has more than d free slots, any larger window
            # would include an unselected free slot between selected ones.
            if len(free) > d:
                break
    return out


def _assign_split_job(slots: List[List[int]], job: int, loc: _Location) -> None:
    for t in loc.slots:
        slots[loc.machine][t] = job


def _schedule_from_slots(inst: dict, slots: Sequence[Sequence[int]], khat: int) -> Optional[PaperSchedule]:
    """Convert a split schedule into a feasible schedule as described by SGH."""
    p = inst["p"]
    machine_jobs: List[List[ScheduledJob]] = []
    seen = set()
    for h, slots_h in enumerate(slots):
        jobs = sorted({j for j in slots_h if j >= 0}, key=lambda j: (slots_h.index(j), j))
        seq: List[ScheduledJob] = []
        current = 0
        for j in jobs:
            if j in seen:
                return None
            seen.add(j)
            original_start = slots_h.index(j)
            start = max(current, original_start)
            end = start + p[j]
            if end > khat:
                return None
            seq.append(ScheduledJob(j, h, start, p[j]))
            current = end
        machine_jobs.append(seq)
    if len(seen) != inst["n"]:
        # Partial schedules are allowed internally, but final schedules are not.
        return None
    sched = PaperSchedule(inst, machine_jobs)
    return sched if sched.feasible and sched.cmax <= khat else None


def _partial_schedule_from_slots(inst: dict, slots: Sequence[Sequence[int]], khat: int) -> Tuple[List[List[ScheduledJob]], set[int]]:
    p = inst["p"]
    machine_jobs: List[List[ScheduledJob]] = []
    seen: set[int] = set()
    for h, slots_h in enumerate(slots):
        jobs = sorted({j for j in slots_h if j >= 0}, key=lambda j: (slots_h.index(j), j))
        seq: List[ScheduledJob] = []
        current = 0
        for j in jobs:
            if j in seen:
                continue
            seen.add(j)
            start = max(current, slots_h.index(j))
            if start + p[j] <= khat:
                seq.append(ScheduledJob(j, h, start, p[j]))
                current = start + p[j]
        machine_jobs.append(seq)
    return machine_jobs, seen


def _machine_loads_from_slots(slots: Sequence[Sequence[int]], inst: dict) -> List[int]:
    """Return total processing time currently assigned to each machine."""
    m = inst["m"]
    loads = [0] * m
    p = inst["p"]
    for h in range(m):
        seen: set[int] = set()
        for t in range(len(slots[h])):
            j = slots[h][t]
            if j >= 0 and j not in seen:
                loads[h] += p[j]
                seen.add(j)
    return loads


def split_greedy_heuristic(
    inst: dict,
    khat: int,
    rng: random.Random,
    jobs: Optional[Iterable[int]] = None,
    fixed_jobs: Optional[Sequence[Sequence[ScheduledJob]]] = None,
    use_improved_tiebreaking: bool = False,
) -> Optional[PaperSchedule]:
    """Algorithm 5.2 SGH, with optional fixed assignments for A-SGH.

    ``use_improved_tiebreaking`` refines rng tie-breaking among equal-cost
    locations by preferring (1) machines with lower current load, then
    (2) machines with lower energy rate ``e[h]``.  Default ``False`` preserves
    exact baseline behavior.
    """
    selected_jobs = set(range(inst["n"]) if jobs is None else jobs)
    slots = (
        _empty_slots_from_jobs(inst, fixed_jobs, khat)
        if fixed_jobs is not None
        else [[-1 for _ in range(khat)] for _ in range(inst["m"])]
    )
    fixed_seen = {sj.job for seq in fixed_jobs or [] for sj in seq}
    remaining = sorted(selected_jobs - fixed_seen)

    for d, js in _jobs_by_processing_time(inst, remaining):
        if _ehs_deadline_expired():
            return None
        for idx_j, j in enumerate(js):
            if idx_j > 0 and idx_j % 5 == 0 and _ehs_deadline_expired():
                return None
            locations: List[_Location] = []
            for h in range(inst["m"]):
                locations.extend(_enumerate_free_locations(slots[h], h, d, inst))
            if not locations:
                return None
            best_cost = min(loc.cost for loc in locations)
            best = [loc for loc in locations if abs(loc.cost - best_cost) <= 1e-9]

            if use_improved_tiebreaking and len(best) > 1:
                loads = _machine_loads_from_slots(slots, inst)
                min_load = min(loads[loc.machine] for loc in best)
                best = [loc for loc in best if loads[loc.machine] <= min_load + 1e-9]
                if len(best) > 1:
                    min_e = min(inst["e"][loc.machine] for loc in best)
                    best = [loc for loc in best if inst["e"][loc.machine] <= min_e + 1e-9]

            loc = rng.choice(best)
            _assign_split_job(slots, j, loc)

    machine_jobs, seen = _partial_schedule_from_slots(inst, slots, khat)
    if seen != selected_jobs:
        return None
    sched = PaperSchedule(inst, machine_jobs)
    return sched if sched.feasible and sched.cmax <= khat else None


_RELEASE_SCORERS: dict = {}  # {policy_name: callable(job_id, sj, inst, khat, lb) -> (job_id, float)}


def register_release_scorer(name: str, scorer) -> None:
    """Register an external A-SGH release scorer under a policy name.

    ``_score_jobs_for_release`` checks this registry before falling back to
    its built-in manual policies.
    """
    _RELEASE_SCORERS[name] = scorer


def clear_release_scorers() -> None:
    """Remove all externally registered release scorers."""
    _RELEASE_SCORERS.clear()


def _score_jobs_for_release(
    feasible_jobs: List[Tuple[int, "ScheduledJob"]],
    inst: dict,
    khat: int,
    previous: "PaperSchedule",
    policy: str,
    rng: random.Random,
) -> List[int]:
    """Score keepable jobs for release and return job ids ordered by priority.

    Returns job ids sorted by release priority (highest first).
    The caller selects the first ``n`` as released.

    These are MANUAL / TEMPLATE baseline policies, not LLM-generated.

    Policies:
      - cost_pressure: highest current energy cost
      - high_rate_machine: highest machine energy rate
      - boundary_slack: closest to khat boundary (largest end = lowest slack)
      - duration_class: longest jobs first
      - mixed_conservative: composite score (cost + 10 * machine_rate)
      - adaptive_tightness: adapts BOTH scoring and budget:
          scoring flips from cost-on-loose to rate-on-tight as khat shrinks;
          budget is scaled down on tight khats.
    """
    # Check externally registered scorers first (DeepSeek candidates)
    scorer = _RELEASE_SCORERS.get(policy)
    if scorer is not None:
        scored: List[Tuple[int, float]] = []
        for j, sj in feasible_jobs:
            try:
                result = scorer(j, sj, inst, khat)
                if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], (int, float)):
                    scored.append((result[0], float(result[1])))
                else:
                    scored.append((j, 0.0))
            except Exception:
                scored.append((j, 0.0))
        scored.sort(key=lambda x: -x[1])
        return [j for j, _ in scored]

    ct = inst["ct"]
    e = inst["e"]
    lb = _lower_bound_cmax(inst)
    T = inst["T"]
    tightness = max(0.0, min(1.0, 1.0 - (khat - lb) / max(T - lb, 1)))

    scored: List[Tuple[int, float]] = []
    for j, sj in feasible_jobs:
        cost = float(e[sj.machine]) * sum(ct[t] for t in range(sj.start, sj.end))
        if policy == "cost_pressure":
            score = cost
        elif policy == "high_rate_machine":
            score = float(e[sj.machine])
        elif policy == "boundary_slack":
            score = float(sj.end)  # larger end = closer to khat = higher release priority
        elif policy == "duration_class":
            score = float(sj.duration)
        elif policy == "mixed_conservative":
            score = cost + float(e[sj.machine]) * 10.0
        elif policy == "adaptive_tightness":
            weight_cost = cost
            weight_rate = float(e[sj.machine]) * 10.0
            score = weight_cost * (1.0 - tightness) + weight_rate * tightness
        else:
            score = 0.0
        scored.append((j, score))

    scored.sort(key=lambda x: -x[1])
    return [j for j, _ in scored]


def assignment_history_sgh(
    inst: dict,
    khat: int,
    previous: PaperSchedule,
    rng: random.Random,
    use_improved_tiebreaking: bool = False,
    asgh_release_policy: str = "none",
    asgh_release_budget: float = 0.0,
    trace: Optional[List[dict]] = None,
) -> Optional[PaperSchedule]:
    """Algorithm 5.4 A-SGH reconstruction.

    ``asgh_release_policy`` and ``asgh_release_budget`` enable optional
    adaptive release of feasible previous assignments before SGH repair.
    Default ``"none"`` / ``0.0`` preserves exact baseline behaviour.

    If ``trace`` is provided, per-khat diagnostics are appended.
    """
    prev_by_job = {sj.job: sj for seq in previous.machine_jobs for sj in seq}

    # --- Pass 1: classify jobs as feasible-to-keep or infeasible ----------
    occupied_pass1 = [[False for _ in range(khat)] for _ in range(inst["m"])]
    feasible_to_keep: List[Tuple[int, ScheduledJob]] = []
    infeasible: List[int] = []

    for d, js in _jobs_by_processing_time(inst):
        for j in js:
            sj = prev_by_job[j]
            if sj.end > khat:
                infeasible.append(j)
                continue
            if any(occupied_pass1[sj.machine][t] for t in range(sj.start, sj.end)):
                infeasible.append(j)
                continue
            feasible_to_keep.append((j, sj))
            for t in range(sj.start, sj.end):
                occupied_pass1[sj.machine][t] = True

    # --- Apply release policy -----------------------------------------------
    released_from_keep: set[int] = set()
    if asgh_release_policy != "none" and asgh_release_budget > 0 and feasible_to_keep:
        effective_budget = asgh_release_budget
        if asgh_release_policy == "adaptive_tightness":
            lb = _lower_bound_cmax(inst)
            T = inst["T"]
            tightness = max(0.0, min(1.0, 1.0 - (khat - lb) / max(T - lb, 1)))
            effective_budget = asgh_release_budget * max(0.2, 1.0 - tightness)
        n_release = max(1, int(len(feasible_to_keep) * effective_budget))
        release_order = _score_jobs_for_release(
            feasible_to_keep, inst, khat, previous, asgh_release_policy, rng
        )
        released_from_keep = set(release_order[:n_release])

    # --- Pass 2: actually keep non-released jobs, SGH repair the rest ----
    kept: List[List[ScheduledJob]] = [[] for _ in range(inst["m"])]
    occupied = [[False for _ in range(khat)] for _ in range(inst["m"])]
    rejected: set[int] = set(infeasible) | released_from_keep

    for j, sj in feasible_to_keep:
        if j in released_from_keep:
            continue
        kept[sj.machine].append(ScheduledJob(j, sj.machine, sj.start, sj.duration))
        for t in range(sj.start, sj.end):
            occupied[sj.machine][t] = True

    if trace is not None:
        trace.append({
            "khat": khat,
            "n_feasible_keepable": len(feasible_to_keep),
            "n_infeasible": len(infeasible),
            "n_released": len(released_from_keep),
            "n_kept": len(feasible_to_keep) - len(released_from_keep),
            "n_rejected_total": len(rejected),
        })

    if not rejected:
        sched = PaperSchedule(inst, [sorted(seq, key=lambda x: x.start) for seq in kept])
        return sched if sched.feasible and sched.cmax <= khat else None

    return split_greedy_heuristic(
        inst, khat, rng, jobs=range(inst["n"]), fixed_jobs=kept,
        use_improved_tiebreaking=use_improved_tiebreaking,
    )


def exact_single_machine_rescheduler(schedule: PaperSchedule, khat: int) -> Optional[PaperSchedule]:
    """Algorithm 5.6 ESR: fixed sequence, optimal timing per machine."""
    return schedule_from_sequences_dp(schedule.inst, schedule.sequences(), khat)


def _schedule_to_slots(schedule: PaperSchedule, khat: int) -> List[List[int]]:
    slots = [[-1 for _ in range(khat)] for _ in range(schedule.inst["m"])]
    for h, seq in enumerate(schedule.machine_jobs):
        for sj in seq:
            if sj.end > khat:
                continue
            for t in range(sj.start, sj.end):
                slots[h][t] = sj.job
    return slots


def _schedule_from_concrete_slots(inst: dict, slots: Sequence[Sequence[int]], khat: int) -> Optional[PaperSchedule]:
    """Build a feasible non-preemptive schedule from concrete slot contents."""
    machine_jobs: List[List[ScheduledJob]] = []
    seen: set[int] = set()
    for h, slots_h in enumerate(slots):
        seq: List[ScheduledJob] = []
        t = 0
        while t < khat:
            j = slots_h[t]
            if j < 0:
                t += 1
                continue
            start = t
            while t < khat and slots_h[t] == j:
                t += 1
            duration = t - start
            if j in seen or duration != inst["p"][j]:
                return None
            seen.add(j)
            seq.append(ScheduledJob(j, h, start, duration))
        machine_jobs.append(seq)
    if seen != set(range(inst["n"])):
        return None
    sched = PaperSchedule(inst, machine_jobs)
    return sched if sched.feasible and sched.cmax <= khat else None


def _jobs_touching_interval(slots_h: Sequence[int], left: int, length: int) -> set[int]:
    return {j for j in slots_h[left : left + length] if j >= 0}


def _interval_contains_whole_jobs(slots_h: Sequence[int], left: int, length: int, jobs: Iterable[int]) -> bool:
    right = left + length
    for j in jobs:
        positions = [t for t, x in enumerate(slots_h) if x == j]
        if not positions:
            continue
        if min(positions) < left or max(positions) >= right:
            return False
    return True


def _eps_i_intervals(
    slots_h: Sequence[int],
    machine: int,
    length: int,
    include_empty: bool,
) -> List[Tuple[int, int, Tuple[int, ...]]]:
    """Return EPS-I candidates as (machine, left, jobs).

    R-ES calls ES while disregarding empty EPS-Is, so the default caller uses
    ``include_empty=False``.  The interval cannot cut through a job: any job
    appearing in the interval must be fully contained in it.
    """
    out: List[Tuple[int, int, Tuple[int, ...]]] = []
    for left in range(0, len(slots_h) - length + 1):
        window = slots_h[left : left + length]
        has_idle = any(x < 0 for x in window)
        if not has_idle:
            continue
        jobs = _jobs_touching_interval(slots_h, left, length)
        if not include_empty and not jobs:
            continue
        if _interval_contains_whole_jobs(slots_h, left, length, jobs):
            out.append((machine, left, tuple(sorted(jobs))))
    return out


def _best_interval_slots(
    inst: dict,
    machine: int,
    jobs: Sequence[int],
    left: int,
    length: int,
    fallback_order: Sequence[int],
) -> Optional[List[int]]:
    """EPS rearrangement: reschedule jobs inside one EPS to minimize TEC."""
    jobs = list(dict.fromkeys(jobs))
    if not jobs:
        return [-1 for _ in range(length)]
    if sum(inst["p"][j] for j in jobs) > length:
        return None

    if len(jobs) <= 7:
        orders = itertools.permutations(jobs)
    else:
        # The paper does not specify the internal rearrangement solver.  For
        # large EPSs, avoid factorial explosion and keep an LPT-style order.
        seeded = [j for j in fallback_order if j in jobs]
        seeded.extend(j for j in jobs if j not in seeded)
        orders = [tuple(seeded)]

    best_energy = INF
    best_slots: Optional[List[int]] = None
    ct = inst["ct"][left : left + length]
    for order in orders:
        durations = [inst["p"][j] for j in order]
        energy, starts = _dp_schedule_fixed_order(durations, ct, inst["e"][machine], length)
        if energy == INF:
            continue
        local = [-1 for _ in range(length)]
        ok = True
        for idx, j in enumerate(order):
            start = starts[idx]
            for t in range(start, start + inst["p"][j]):
                if local[t] >= 0:
                    ok = False
                    break
                local[t] = j
            if not ok:
                break
        if ok and energy < best_energy:
            best_energy = energy
            best_slots = local
    return best_slots


def _ordered_jobs_in_interval(slots_h: Sequence[int], left: int, length: int) -> List[int]:
    seen: List[int] = []
    for j in slots_h[left : left + length]:
        if j >= 0 and j not in seen:
            seen.append(j)
    return seen


def _apply_eps_move(
    schedule: PaperSchedule,
    khat: int,
    source_machine: int,
    source_left: int,
    target_machine: int,
    target_left: int,
    length: int,
) -> Optional[PaperSchedule]:
    slots = _schedule_to_slots(schedule, khat)
    source_jobs = _ordered_jobs_in_interval(slots[source_machine], source_left, length)
    target_jobs = _ordered_jobs_in_interval(slots[target_machine], target_left, length)

    if not source_jobs:
        return None
    if source_machine == target_machine:
        source_range = set(range(source_left, source_left + length))
        target_range = set(range(target_left, target_left + length))
        if source_range & target_range:
            return None

    # EPS swap transfers the jobs contained in each EPS to the other EPS.
    new_source = _best_interval_slots(
        schedule.inst, source_machine, target_jobs, source_left, length, target_jobs
    )
    new_target = _best_interval_slots(
        schedule.inst, target_machine, source_jobs, target_left, length, source_jobs
    )
    if new_source is None or new_target is None:
        return None

    slots[source_machine][source_left : source_left + length] = new_source
    slots[target_machine][target_left : target_left + length] = new_target
    return _schedule_from_concrete_slots(schedule.inst, slots, khat)


def _exchange_search_non_empty_eps(
    schedule: PaperSchedule,
    khat: int,
    max_passes: int = 20,
    max_moves_checked: int = 20000,
    eps_ordering: str = "default",
) -> PaperSchedule:
    """Algorithm 5.3 ES restricted to non-empty EPS-Is for R-ES line 3.

    ``eps_ordering`` controls the move-scan order inside ES non-empty:
    - ``"default"``: original machine/start/job order.
    - ``"expensive_source_first"``: process jobs with highest current energy
      cost first, then cheapest target intervals first.
    """
    cur = schedule
    for _ in range(max_passes):
        improved = False
        checked = 0
        for d, _ in _jobs_by_processing_time(cur.inst):
            eps_jobs = [
                (sj.machine, sj.start, sj.job)
                for seq in cur.machine_jobs
                for sj in seq
                if sj.duration == d
            ]
            if eps_ordering == "expensive_source_first":
                eps_jobs.sort(
                    key=lambda x: -(
                        cur.inst["e"][x[0]]
                        * sum(cur.inst["ct"][t] for t in range(x[1], x[1] + d))
                    )
                )
            else:
                eps_jobs.sort(key=lambda x: (x[0], x[1], x[2]))
            slots = _schedule_to_slots(cur, khat)
            eps_i: List[Tuple[int, int, Tuple[int, ...]]] = []
            for h in range(cur.inst["m"]):
                eps_i.extend(_eps_i_intervals(slots[h], h, d, include_empty=False))
            eps_i.sort(
                key=lambda item: (
                    cur.inst["e"][item[0]] * sum(cur.inst["ct"][t] for t in range(item[1], item[1] + d)),
                    item[0],
                    item[1],
                )
            )

            for h_j, left_j, _job in eps_jobs:
                for h_i, left_i, _jobs in eps_i:
                    checked += 1
                    if checked > max_moves_checked:
                        return cur
                    candidate = _apply_eps_move(cur, khat, h_j, left_j, h_i, left_i, d)
                    if candidate is not None and candidate.energy + 1e-9 < cur.energy:
                        cur = candidate
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return cur


def exchange_search_with_rescheduling(
    schedule: PaperSchedule,
    khat: int,
    rng: random.Random,
    max_passes: int = 2,
    max_reinsert_checks: int = 5000,
    max_moves_checked: int = 20000,
    eps_ordering: str = "default",
) -> PaperSchedule:
    """Algorithm 5.5 R-ES reconstruction.

    This reconstruction follows Algorithm 5.5 at the policy level:

    1. apply ES while disregarding empty EPS-Is;
    2. account for empty EPS-Is through LPT job removal and smallest-cost
       free/free-split reinsertion;
    3. repeat while TEC improves.

    The paper does not specify all tie-breaking and EPS rearrangement ordering
    details, so this implementation uses deterministic interval ordering and a
    local exact rearrangement for jobs contained in an EPS.
    """
    best = schedule
    for _ in range(max_passes):
        before = best.energy
        cur = _exchange_search_non_empty_eps(
            best.copy(), khat, eps_ordering=eps_ordering,
            max_moves_checked=max_moves_checked,
        )
        checks = 0
        for d, js in _jobs_by_processing_time(cur.inst):
            for j in js:
                checks += 1
                if checks > max_reinsert_checks:
                    break
                machine_jobs = [[sj for sj in seq if sj.job != j] for seq in cur.machine_jobs]
                slots = _empty_slots_from_jobs(cur.inst, machine_jobs, khat)
                locations: List[_Location] = []
                for h in range(cur.inst["m"]):
                    locations.extend(_enumerate_free_locations(slots[h], h, d, cur.inst))
                if not locations:
                    continue
                best_cost = min(loc.cost for loc in locations)
                candidates = [loc for loc in locations if abs(loc.cost - best_cost) <= 1e-9]
                loc = rng.choice(candidates)
                _assign_split_job(slots, j, loc)
                partial_jobs, seen = _partial_schedule_from_slots(cur.inst, slots, khat)
                if len(seen) == cur.inst["n"]:
                    candidate = PaperSchedule(cur.inst, partial_jobs)
                    if candidate.feasible and candidate.cmax <= khat:
                        cur = candidate
            if checks > max_reinsert_checks:
                break
        cur = exact_single_machine_rescheduler(cur, khat) or cur
        if cur.energy + 1e-9 < before:
            best = cur
        else:
            break
    return best


def run_ehs(
    inst: dict,
    rng: Optional[random.Random] = None,
    khat_start: Optional[int] = None,
    khat_stop: Optional[int] = None,
    time_limit_seconds: Optional[float] = None,
    use_history: bool = True,
    use_exchange: bool = True,
    use_esr: bool = True,
    fast_mode: bool = False,
    use_improved_tiebreaking: bool = False,
    eps_ordering: str = "default",
    es_max_moves: int = 20000,
    es_max_reinsert: int = 5000,
    asgh_release_policy: str = "none",
    asgh_release_budget: float = 0.0,
    asgh_trace: Optional[List[dict]] = None,
) -> List[PaperSchedule]:
    """Algorithm 5.7 EHS.

    Set ``use_history=False`` and ``use_esr=False`` to approximate SGS-ES.

    ``fast_mode`` runs baseline full EHS for the first 75% of the time budget,
    then skips exchange and ESR to explore more khats quickly.  It is only
    active when ``time_limit_seconds`` is not None and <= 60.  This mode was
    validated in Phase B6.5c (hybrid 75/25); it improves HV at tight budgets
    but loses to baseline at 120 s and longer.

    ``use_improved_tiebreaking`` refines SGH tie-breaking among equal-cost
    locations by preferring less-loaded machines and lower energy rates.
    Default ``False`` preserves exact baseline behavior.

    ``eps_ordering`` controls the move-scan order inside ES non-empty.
    ``"expensive_source_first"`` was validated in Phase B6.6; it improves
    mean HV by ~4% on synthetic VLS validation.  Default ``"default"``
    preserves exact baseline behavior.

    ``es_max_moves`` and ``es_max_reinsert`` bound the exchange-search and
    reinsertion checks to speed up R-ES on large instances.  Defaults
    preserve the original unbounded behaviour.

    ``asgh_release_policy`` and ``asgh_release_budget`` enable optional
    adaptive release of feasible previous assignments inside A-SGH.
    Default ``"none"`` / ``0.0`` preserves exact baseline behaviour.
    Policies: ``"cost_pressure"``, ``"high_rate_machine"``,
    ``"boundary_slack"``, ``"duration_class"``, ``"mixed_conservative"``,
    ``"adaptive_tightness"``.

    If ``asgh_trace`` is provided, per-khat A-SGH diagnostics are appended.
    """
    rng = rng or random.Random(0)
    t0 = time.time()
    lb = _lower_bound_cmax(inst)
    khat = inst["T"] if khat_start is None else min(khat_start, inst["T"])
    stop = lb if khat_stop is None else max(khat_stop, lb)
    archive: List[PaperSchedule] = []
    previous: Optional[PaperSchedule] = None

    _set_ehs_deadline(time_limit_seconds)

    # fast_mode is time-budget-only and validated for <= 60 s
    use_fast = (
        fast_mode
        and time_limit_seconds is not None
        and time_limit_seconds <= 60.0
    )

    while khat >= stop:
        elapsed = time.time() - t0
        if time_limit_seconds is not None and elapsed > time_limit_seconds:
            break

        # In fast_mode, switch to construction-only after 75% of budget
        in_fast_phase = use_fast and elapsed >= time_limit_seconds * 0.75
        do_exchange = use_exchange and not in_fast_phase
        do_esr = use_esr and not in_fast_phase

        if previous is None or not use_history:
            sched = split_greedy_heuristic(
                inst, khat, rng,
                use_improved_tiebreaking=use_improved_tiebreaking,
            )
        else:
            sched = assignment_history_sgh(
                inst, khat, previous, rng,
                use_improved_tiebreaking=use_improved_tiebreaking,
                asgh_release_policy=asgh_release_policy,
                asgh_release_budget=asgh_release_budget,
                trace=asgh_trace,
            )
        if sched is None:
            break
        if do_exchange and not _ehs_deadline_expired():
            sched = exchange_search_with_rescheduling(
                sched, khat, rng,
                eps_ordering=eps_ordering,
                max_moves_checked=es_max_moves,
                max_reinsert_checks=es_max_reinsert,
            )
        if do_esr and not _ehs_deadline_expired():
            sched = exact_single_machine_rescheduler(sched, khat) or sched
        archive.append(sched)
        previous = sched
        khat -= 1
    _clear_ehs_deadline()
    return pareto_filter_schedules(archive)


def run_sgs_es(
    inst: dict,
    rng: Optional[random.Random] = None,
    khat_start: Optional[int] = None,
    khat_stop: Optional[int] = None,
    time_limit_seconds: Optional[float] = None,
) -> List[PaperSchedule]:
    """Anghinolfi et al. SGS-ES style baseline: SGH plus ES-like improvement."""
    return run_ehs(
        inst,
        rng=rng,
        khat_start=khat_start,
        khat_stop=khat_stop,
        time_limit_seconds=time_limit_seconds,
        use_history=False,
        use_exchange=True,
        use_esr=False,
    )


# ---------------------------------------------------------------------------
# EOA / Pipe-VND reconstruction
# ---------------------------------------------------------------------------


def _initial_random_sequences(inst: dict, rng: random.Random, epsilon: int) -> List[List[int]]:
    sequences: List[List[int]] = [[] for _ in range(inst["m"])]
    loads = [0 for _ in range(inst["m"])]
    jobs = list(range(inst["n"]))
    rng.shuffle(jobs)
    for j in jobs:
        feasible = [h for h in range(inst["m"]) if loads[h] + inst["p"][j] <= epsilon]
        if not feasible:
            feasible = list(range(inst["m"]))
        h = rng.choice(feasible)
        sequences[h].append(j)
        loads[h] += inst["p"][j]
    return sequences


def _lpt_sequences(inst: dict, epsilon: int) -> List[List[int]]:
    sequences: List[List[int]] = [[] for _ in range(inst["m"])]
    loads = [0 for _ in range(inst["m"])]
    for j in sorted(range(inst["n"]), key=lambda x: (-inst["p"][x], x)):
        feasible = [h for h in range(inst["m"]) if loads[h] + inst["p"][j] <= epsilon]
        if not feasible:
            h = min(range(inst["m"]), key=lambda mi: loads[mi])
        else:
            h = min(feasible, key=lambda mi: loads[mi])
        sequences[h].append(j)
        loads[h] += inst["p"][j]
    return sequences


def _perturb_sequences(sequences: List[List[int]], rng: random.Random, repetitions: int = 3) -> List[List[int]]:
    out = [list(seq) for seq in sequences]
    for _ in range(repetitions):
        non_empty = [h for h, seq in enumerate(out) if seq]
        if not non_empty:
            break
        src = rng.choice(non_empty)
        pos = rng.randrange(len(out[src]))
        job = out[src].pop(pos)
        dst = rng.randrange(len(out))
        insert_pos = rng.randrange(len(out[dst]) + 1)
        out[dst].insert(insert_pos, job)
    return out


def _neighbors(sequences: List[List[int]], nbh: int) -> Iterable[List[List[int]]]:
    m = len(sequences)
    if nbh == 1:  # swap-intra
        for h in range(m):
            seq = sequences[h]
            for i in range(len(seq)):
                for k in range(i + 1, len(seq)):
                    cand = [list(s) for s in sequences]
                    cand[h][i], cand[h][k] = cand[h][k], cand[h][i]
                    yield cand
    elif nbh == 2:  # swap-inter
        for h1 in range(m):
            for h2 in range(h1 + 1, m):
                for i in range(len(sequences[h1])):
                    for k in range(len(sequences[h2])):
                        cand = [list(s) for s in sequences]
                        cand[h1][i], cand[h2][k] = cand[h2][k], cand[h1][i]
                        yield cand
    elif nbh == 3:  # insert-inter
        for h1 in range(m):
            for i in range(len(sequences[h1])):
                for h2 in range(m):
                    if h1 == h2:
                        continue
                    for pos in range(len(sequences[h2]) + 1):
                        cand = [list(s) for s in sequences]
                        job = cand[h1].pop(i)
                        cand[h2].insert(pos, job)
                        yield cand


def pipe_vnd(
    inst: dict,
    initial_sequences: Sequence[Sequence[int]],
    epsilon: int,
    archive: Optional[List[PaperSchedule]] = None,
    max_neighbors_per_neighborhood: Optional[int] = None,
    time_limit_seconds: Optional[float] = None,
) -> Tuple[List[List[int]], Optional[PaperSchedule], List[PaperSchedule]]:
    """Algorithm 2 Pipe-VND with first-improvement and fixed-sequence DP."""
    t0 = time.time()
    archive = archive if archive is not None else []
    current_sequences = [list(s) for s in initial_sequences]
    current = schedule_from_sequences_dp(inst, current_sequences, epsilon)
    if current is None:
        return current_sequences, None, archive
    archive.append(current)
    k = 0
    count = 0
    while count < 3:
        if time_limit_seconds is not None and time.time() - t0 > time_limit_seconds:
            break
        improved = False
        checked = 0
        for cand_seq in _neighbors(current_sequences, k + 1):
            checked += 1
            if max_neighbors_per_neighborhood and checked > max_neighbors_per_neighborhood:
                break
            cand = schedule_from_sequences_dp(inst, cand_seq, epsilon)
            if cand is None:
                continue
            archive.append(cand)
            if cand.energy + 1e-9 < current.energy:
                current_sequences = cand_seq
                current = cand
                count = 0
                improved = True
                break
        if not improved:
            count += 1
        k += 1
        if k >= 3:
            k = 0
    return current_sequences, current, pareto_filter_schedules(archive)


def run_eoa(
    inst: dict,
    rng: Optional[random.Random] = None,
    iter_max: int = 1,
    delta: int = 1,
    max_neighbors_per_neighborhood: Optional[int] = None,
    time_limit_seconds: Optional[float] = None,
) -> List[PaperSchedule]:
    """Algorithm 1 Epsilon Oscillation Algorithm reconstruction."""
    rng = rng or random.Random(0)
    t0 = time.time()
    lb = _lower_bound_cmax(inst)
    ub = inst["T"]
    archive: List[PaperSchedule] = []
    sequences = _initial_random_sequences(inst, rng, ub)
    if schedule_from_sequences_dp(inst, sequences, ub) is None:
        sequences = _lpt_sequences(inst, ub)

    for iteration in range(1, iter_max + 1):
        w = 1
        epsilon = lb
        while epsilon >= lb:
            if time_limit_seconds is not None and time.time() - t0 > time_limit_seconds:
                return pareto_filter_schedules(archive)
            if iteration > 1:
                sequences = _perturb_sequences(sequences, rng, repetitions=3)
            remaining = None if time_limit_seconds is None else max(0.0, time_limit_seconds - (time.time() - t0))
            sequences, _, archive = pipe_vnd(
                inst,
                sequences,
                epsilon,
                archive=archive,
                max_neighbors_per_neighborhood=max_neighbors_per_neighborhood,
                time_limit_seconds=remaining,
            )
            epsilon += w * delta
            if epsilon > ub:
                w = -1
                epsilon = ub + w * delta
    return pareto_filter_schedules(archive)

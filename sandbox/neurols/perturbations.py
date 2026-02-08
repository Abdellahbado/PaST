"""Deterministic perturbation operators for escaping local optima.

Perturbations are larger-scale modifications that help the search escape
persistent local optima. Unlike operators which are evaluated and the best
move selected, perturbations are applied directly.

All perturbations are deterministic: same state -> same perturbation result.

Perturbations:
- NONE: No perturbation (continue with operator moves)
- SHAKE_SMALL: Relocate q worst-exposed jobs to better positions
- SHAKE_PEAK: Relocate q jobs with highest peak-slot exposure
- RESTART: Rebuild solution with deterministic construction heuristic
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, TYPE_CHECKING
import numpy as np

from sandbox.neurols.solution import PMALNSSolution
from sandbox.neurols.price_embedding import PriceFeatureExtractor

if TYPE_CHECKING:
    from sandbox.neurols.move_evaluator import FullEvaluation


class PerturbationID(IntEnum):
    """Perturbation identifiers."""

    NONE = 0
    SHAKE_SMALL = 1
    RESTART = 2
    SHAKE_PEAK = 3


@dataclass
class Perturbation:
    """Perturbation operator definition."""

    id: PerturbationID
    name: str
    description: str

    # Perturbation strength (0 = none, 1 = full restart)
    intensity: float

    def apply(
        self,
        solution: PMALNSSolution,
        evaluation: Optional["FullEvaluation"],
        processing_times: np.ndarray,
        ct: np.ndarray,
        machine_energy_rates: np.ndarray,
        K: int,
    ) -> PMALNSSolution:
        """Apply perturbation to solution.

        Args:
            solution: Current solution
            evaluation: Current solution evaluation (with per-job costs)
            processing_times: Job processing times
            ct: Per-slot prices
            machine_energy_rates: Energy rate per machine
            K: Horizon constraint

        Returns:
            New solution after perturbation
        """
        if self.id == PerturbationID.NONE:
            return solution.clone()

        elif self.id == PerturbationID.SHAKE_SMALL:
            return self._apply_shake_small(
                solution, evaluation, processing_times, ct, machine_energy_rates, K
            )

        elif self.id == PerturbationID.SHAKE_PEAK:
            return self._apply_shake_peak(
                solution, evaluation, processing_times, ct, machine_energy_rates, K
            )

        elif self.id == PerturbationID.RESTART:
            return self._apply_restart(solution, processing_times)

        else:
            raise ValueError(f"Unknown perturbation {self.id}")

    def _apply_shake_small(
        self,
        solution: PMALNSSolution,
        evaluation: Optional["FullEvaluation"],
        processing_times: np.ndarray,
        ct: np.ndarray,
        machine_energy_rates: np.ndarray,
        K: int,
        q: int = 3,
    ) -> PMALNSSolution:
        """Shake by relocating q worst-exposed jobs.

        Worst-exposed = highest per-job energy cost from current schedule.
        Jobs are relocated to the front of the lowest-load machine (cheap heuristic).
        """
        new_sol = solution.clone()

        if evaluation is None or not hasattr(evaluation, "per_job_costs"):
            # Fallback: use proxy (longest jobs on highest-energy machines)
            return self._shake_by_proxy(
                new_sol, processing_times, machine_energy_rates, q
            )

        # Compute per-job energy cost and rank
        job_costs = getattr(evaluation, "per_job_costs", None)
        if job_costs is None:
            return self._shake_by_proxy(
                new_sol, processing_times, machine_energy_rates, q
            )

        # Sort jobs by cost descending, deterministic tie-break by job index
        ranked_jobs = sorted(range(solution.n_jobs), key=lambda j: (-job_costs[j], j))

        # Take top-q worst jobs
        worst_jobs = ranked_jobs[:q]

        # Remove worst jobs from their current positions
        removed_data = []
        for job in worst_jobs:
            mi = new_sol.get_assignment(job)
            pos = new_sol.get_position(job)
            removed_data.append((job, mi))

        # Sort by position descending to avoid index shifting issues
        # First, collect (job, machine, position) before any removal
        removal_info = []
        for job in worst_jobs:
            removal_info.append(
                (job, new_sol.get_assignment(job), new_sol.get_position(job))
            )

        # Sort by machine, then by position descending
        removal_info.sort(key=lambda x: (x[1], -x[2]))

        # Remove jobs (from back to front within each machine)
        for job, mi, pos in removal_info:
            # Find current position (may have shifted)
            current_pos = new_sol.sequences[mi].index(job)
            new_sol.sequences[mi].pop(current_pos)

        # Reinsert at front of lowest-load machines
        loads = new_sol.get_machine_loads(processing_times)

        for job, _, _ in removal_info:
            # Find machine with minimum load (deterministic tie-break)
            min_machine = min(range(new_sol.n_machines), key=lambda m: (loads[m], m))
            new_sol.sequences[min_machine].insert(0, job)
            loads[min_machine] += processing_times[job]

        new_sol._invalidate_caches()
        return new_sol

    def _apply_shake_peak(
        self,
        solution: PMALNSSolution,
        evaluation: Optional["FullEvaluation"],
        processing_times: np.ndarray,
        ct: np.ndarray,
        machine_energy_rates: np.ndarray,
        K: int,
        q: int = 3,
    ) -> PMALNSSolution:
        """Shake by relocating q jobs with highest peak-slot exposure.

        We approximate "peak exposure" by how much of a job's scheduled interval
        overlaps peak-level price slots (level=2), weighted by machine energy rate.
        This uses the *current* DP schedule start_times from evaluation.
        """
        new_sol = solution.clone()

        if evaluation is None:
            return self._shake_by_proxy(
                new_sol, processing_times, machine_energy_rates, q
            )

        extractor = PriceFeatureExtractor(ct=np.asarray(ct, dtype=np.float64), K=int(K))
        job_scores: List[tuple] = []

        # Build per-job scores from current schedule
        for mi in range(new_sol.n_machines):
            seq = new_sol.sequences[mi]
            if not seq:
                continue

            if mi >= len(evaluation.per_machine):
                continue
            me = evaluation.per_machine[mi]
            if (not me.start_times) or (len(me.start_times) != len(seq)):
                continue

            e_rate = float(machine_energy_rates[mi])
            for pos, job in enumerate(seq):
                start = int(me.start_times[pos])
                dur = int(processing_times[job])
                if dur <= 0:
                    continue
                # Peak is level index 2
                peak_price_sum = float(
                    extractor.get_interval_level_price_sums(start, dur)[2]
                )
                score = e_rate * peak_price_sum
                job_scores.append((score, int(job)))

        if not job_scores:
            return self._shake_by_proxy(
                new_sol, processing_times, machine_energy_rates, q
            )

        # Sort jobs by score descending, deterministic tie-break by job id
        job_scores.sort(key=lambda x: (-x[0], x[1]))
        worst_jobs = [job for _, job in job_scores[:q]]

        # Collect removal info (job, mi, pos) from the current solution
        removal_info = []
        for job in worst_jobs:
            removal_info.append(
                (job, new_sol.get_assignment(job), new_sol.get_position(job))
            )

        # Sort by machine, then position descending (deterministic, avoids shifting)
        removal_info.sort(key=lambda x: (x[1], -x[2], x[0]))

        for job, mi, _ in removal_info:
            current_pos = new_sol.sequences[mi].index(job)
            new_sol.sequences[mi].pop(current_pos)

        # Choose destination machines: prioritize low energy rate, tie by machine index.
        # Reinsert at the front (position 0). This is cheap and lets DP re-time.
        loads = new_sol.get_machine_loads(processing_times)
        for job, _, _ in removal_info:
            dest = min(
                range(new_sol.n_machines),
                key=lambda m: (float(machine_energy_rates[m]), float(loads[m]), m),
            )
            new_sol.sequences[dest].insert(0, job)
            loads[dest] += processing_times[job]

        new_sol._invalidate_caches()
        return new_sol

    def _shake_by_proxy(
        self,
        solution: PMALNSSolution,
        processing_times: np.ndarray,
        machine_energy_rates: np.ndarray,
        q: int,
    ) -> PMALNSSolution:
        """Fallback shake using proxy: longest jobs on high-energy machines."""
        new_sol = solution.clone()

        # Score: p_j * e_m (proxy for energy contribution)
        job_scores = []
        for job in range(new_sol.n_jobs):
            mi = new_sol.get_assignment(job)
            score = processing_times[job] * machine_energy_rates[mi]
            job_scores.append((score, job))

        # Sort by score descending, deterministic tie-break
        job_scores.sort(key=lambda x: (-x[0], x[1]))

        worst_jobs = [job for _, job in job_scores[:q]]

        # Remove and reinsert at lightest machines
        removal_info = []
        for job in worst_jobs:
            removal_info.append(
                (job, new_sol.get_assignment(job), new_sol.get_position(job))
            )

        removal_info.sort(key=lambda x: (x[1], -x[2]))

        for job, mi, _ in removal_info:
            current_pos = new_sol.sequences[mi].index(job)
            new_sol.sequences[mi].pop(current_pos)

        loads = new_sol.get_machine_loads(processing_times)

        for job, _, _ in removal_info:
            min_machine = min(range(new_sol.n_machines), key=lambda m: (loads[m], m))
            new_sol.sequences[min_machine].insert(0, job)
            loads[min_machine] += processing_times[job]

        new_sol._invalidate_caches()
        return new_sol

    def _apply_restart(
        self,
        solution: PMALNSSolution,
        processing_times: np.ndarray,
    ) -> PMALNSSolution:
        """Restart: rebuild solution with LPT assignment + SPT ordering."""
        return PMALNSSolution.from_load_balanced(
            n_jobs=solution.n_jobs,
            n_machines=solution.n_machines,
            processing_times=processing_times,
        )


# Perturbation instances
NONE = Perturbation(
    id=PerturbationID.NONE,
    name="None",
    description="No perturbation",
    intensity=0.0,
)

SHAKE_SMALL = Perturbation(
    id=PerturbationID.SHAKE_SMALL,
    name="Shake-Small",
    description="Relocate 3 worst-exposed jobs",
    intensity=0.15,
)

SHAKE_PEAK = Perturbation(
    id=PerturbationID.SHAKE_PEAK,
    name="Shake-Peak",
    description="Relocate 3 jobs with highest peak exposure",
    intensity=0.2,
)

RESTART = Perturbation(
    id=PerturbationID.RESTART,
    name="Restart",
    description="Rebuild with deterministic heuristic",
    intensity=1.0,
)

# Perturbation registry
PERTURBATIONS: List[Perturbation] = [NONE, SHAKE_SMALL, RESTART, SHAKE_PEAK]

PERTURBATION_BY_ID = {p.id: p for p in PERTURBATIONS}


def get_perturbation(perturbation_id: PerturbationID) -> Perturbation:
    """Get perturbation by ID."""
    return PERTURBATION_BY_ID[perturbation_id]


# ═══════════════════════════════════════════════════════════════════
# Destroy / Repair operators (for Tripartite-B learned-destroy)
# ═══════════════════════════════════════════════════════════════════


class DestroyID(IntEnum):
    """Destroy operator identifiers for learned-destroy variant."""

    RANDOM_3 = 0  # Remove 3 random jobs, reinsert greedily
    RANDOM_5 = 1  # Remove 5 random jobs
    WORST_MACHINE = 2  # Remove all jobs from worst (most expensive) machine
    EXPENSIVE_3 = 3  # Remove 3 most expensive jobs (by per-job cost proxy)
    BLOCK_2 = 4  # Remove 2 contiguous blocks from random machines


@dataclass
class DestroyOperator:
    """Destroy-repair operator definition.

    Each operator removes a set of jobs then reinserts them with a greedy
    heuristic (lowest-load machine, front of sequence).
    """

    id: DestroyID
    name: str
    description: str

    def apply(
        self,
        solution: PMALNSSolution,
        processing_times: np.ndarray,
        machine_energy_rates: np.ndarray,
        K: int,
    ) -> PMALNSSolution:
        """Apply destroy + greedy repair.

        Returns a new solution (original is not mutated).
        """
        new_sol = solution.clone()
        n = new_sol.n_jobs
        m = new_sol.n_machines

        if self.id == DestroyID.RANDOM_3:
            jobs = self._select_random(new_sol, processing_times, 3)
        elif self.id == DestroyID.RANDOM_5:
            jobs = self._select_random(new_sol, processing_times, 5)
        elif self.id == DestroyID.WORST_MACHINE:
            jobs = self._select_worst_machine(
                new_sol, processing_times, machine_energy_rates
            )
        elif self.id == DestroyID.EXPENSIVE_3:
            jobs = self._select_expensive(
                new_sol, processing_times, machine_energy_rates, 3
            )
        elif self.id == DestroyID.BLOCK_2:
            jobs = self._select_blocks(new_sol, processing_times, 2)
        else:
            raise ValueError(f"Unknown destroy id {self.id}")

        if not jobs:
            return new_sol

        # Remove jobs (back-to-front per machine)
        removal = [
            (j, new_sol.get_assignment(j), new_sol.get_position(j)) for j in jobs
        ]
        removal.sort(key=lambda x: (x[1], -x[2]))
        for j, mi, _ in removal:
            pos = new_sol.sequences[mi].index(j)
            new_sol.sequences[mi].pop(pos)

        # Greedy repair: reinsert at front of lightest machine
        loads = new_sol.get_machine_loads(processing_times)
        for j, _, _ in removal:
            best_m = min(range(m), key=lambda mi: (loads[mi], mi))
            new_sol.sequences[best_m].insert(0, j)
            loads[best_m] += processing_times[j]

        new_sol._invalidate_caches()
        return new_sol

    # ── selection helpers (deterministic, seeded by job indices) ──

    @staticmethod
    def _select_random(sol: PMALNSSolution, p: np.ndarray, q: int) -> List[int]:
        """Deterministic pseudo-random selection based on current solution hash."""
        n = sol.n_jobs
        q = min(q, n)
        # Use hash of current assignment as seed for reproducibility
        h = hash(sol.hash_key()) % (2**31)
        rng = np.random.RandomState(h)
        return list(rng.choice(n, size=q, replace=False))

    @staticmethod
    def _select_worst_machine(
        sol: PMALNSSolution,
        p: np.ndarray,
        e: np.ndarray,
    ) -> List[int]:
        """Remove all jobs from the most expensive machine (load * rate)."""
        m = sol.n_machines
        loads = sol.get_machine_loads(p)
        costs = loads * e
        worst_mi = int(np.argmax(costs))
        return list(sol.sequences[worst_mi])

    @staticmethod
    def _select_expensive(
        sol: PMALNSSolution,
        p: np.ndarray,
        e: np.ndarray,
        q: int,
    ) -> List[int]:
        """Remove q jobs with highest proxy cost (p_j * e_m)."""
        scores = []
        for j in range(sol.n_jobs):
            mi = sol.get_assignment(j)
            scores.append((p[j] * e[mi], j))
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [j for _, j in scores[:q]]

    @staticmethod
    def _select_blocks(sol: PMALNSSolution, p: np.ndarray, n_blocks: int) -> List[int]:
        """Remove n_blocks contiguous pairs from different machines."""
        jobs = []
        h = hash(sol.hash_key()) % (2**31)
        rng = np.random.RandomState(h)
        machines = list(range(sol.n_machines))
        rng.shuffle(machines)
        for mi in machines[:n_blocks]:
            seq = sol.sequences[mi]
            if len(seq) >= 2:
                start = rng.randint(0, len(seq) - 1)
                jobs.extend(seq[start : start + 2])
            elif seq:
                jobs.extend(seq)
        return jobs


# Destroy operator instances
DESTROY_OPERATORS: List[DestroyOperator] = [
    DestroyOperator(DestroyID.RANDOM_3, "Random-3", "Remove 3 random jobs"),
    DestroyOperator(DestroyID.RANDOM_5, "Random-5", "Remove 5 random jobs"),
    DestroyOperator(
        DestroyID.WORST_MACHINE, "Worst-Machine", "Remove worst machine jobs"
    ),
    DestroyOperator(
        DestroyID.EXPENSIVE_3, "Expensive-3", "Remove 3 most expensive jobs"
    ),
    DestroyOperator(DestroyID.BLOCK_2, "Block-2", "Remove 2 contiguous blocks"),
]

DESTROY_BY_ID = {d.id: d for d in DESTROY_OPERATORS}

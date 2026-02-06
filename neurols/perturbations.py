"""Deterministic perturbation operators for escaping local optima.

Perturbations are larger-scale modifications that help the search escape
persistent local optima. Unlike operators which are evaluated and the best
move selected, perturbations are applied directly.

All perturbations are deterministic: same state -> same perturbation result.

Perturbations:
- NONE: No perturbation (continue with operator moves)
- SHAKE_SMALL: Relocate q worst-exposed jobs to better positions
- RESTART: Rebuild solution with deterministic construction heuristic
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, TYPE_CHECKING
import numpy as np

from PaST.neurols.solution import PMALNSSolution

if TYPE_CHECKING:
    from PaST.neurols.move_evaluator import FullEvaluation


class PerturbationID(IntEnum):
    """Perturbation identifiers."""

    NONE = 0
    SHAKE_SMALL = 1
    RESTART = 2


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

RESTART = Perturbation(
    id=PerturbationID.RESTART,
    name="Restart",
    description="Rebuild with deterministic heuristic",
    intensity=1.0,
)

# Perturbation registry
PERTURBATIONS: List[Perturbation] = [NONE, SHAKE_SMALL, RESTART]

PERTURBATION_BY_ID = {p.id: p for p in PERTURBATIONS}


def get_perturbation(perturbation_id: PerturbationID) -> Perturbation:
    """Get perturbation by ID."""
    return PERTURBATION_BY_ID[perturbation_id]

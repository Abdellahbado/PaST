"""Solution representation for parallel machine TOU scheduling.

A solution consists of:
1. Assignment: which machine each job is assigned to
2. Sequences: per-machine ordered list of jobs

The timing/scheduling for each machine sequence is computed optimally by DP
and is NOT part of the solution representation (it's derived).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


@dataclass
class PMALNSSolution:
    """Solution representation: assignment + per-machine job sequences.

    Attributes:
        sequences: List of job sequences, one per machine. sequences[m] contains
                   the ordered list of job indices assigned to machine m.
        n_jobs: Total number of jobs in the problem.
        n_machines: Number of machines.

    Invariants:
        - Every job 0..n_jobs-1 appears exactly once across all sequences
        - len(sequences) == n_machines
    """

    sequences: List[List[int]]
    n_jobs: int
    n_machines: int

    # Cached assignment mapping (job -> machine)
    _assignment_cache: Optional[Dict[int, int]] = field(default=None, repr=False)
    # Cached position mapping (job -> position in its machine sequence)
    _position_cache: Optional[Dict[int, int]] = field(default=None, repr=False)

    def __post_init__(self):
        """Validate solution and initialize caches."""
        self._validate()
        self._rebuild_caches()

    def _validate(self) -> None:
        """Validate solution invariants."""
        if len(self.sequences) != self.n_machines:
            raise ValueError(
                f"Expected {self.n_machines} machine sequences, got {len(self.sequences)}"
            )

        seen = set()
        for mi, seq in enumerate(self.sequences):
            for job in seq:
                if job < 0 or job >= self.n_jobs:
                    raise ValueError(f"Invalid job index {job} (n_jobs={self.n_jobs})")
                if job in seen:
                    raise ValueError(f"Duplicate job {job} in solution")
                seen.add(job)

        if len(seen) != self.n_jobs:
            missing = set(range(self.n_jobs)) - seen
            raise ValueError(f"Missing jobs: {sorted(missing)}")

    def _rebuild_caches(self) -> None:
        """Rebuild assignment and position caches."""
        self._assignment_cache = {}
        self._position_cache = {}
        for mi, seq in enumerate(self.sequences):
            for pos, job in enumerate(seq):
                self._assignment_cache[job] = mi
                self._position_cache[job] = pos

    def _invalidate_caches(self) -> None:
        """Invalidate caches (call after modification)."""
        self._assignment_cache = None
        self._position_cache = None

    def get_assignment(self, job: int) -> int:
        """Get the machine index that job is assigned to."""
        if self._assignment_cache is None:
            self._rebuild_caches()
        return self._assignment_cache[job]

    def get_position(self, job: int) -> int:
        """Get the position of job within its machine sequence."""
        if self._position_cache is None:
            self._rebuild_caches()
        return self._position_cache[job]

    def get_job_at(self, machine: int, position: int) -> int:
        """Get job at given machine and position."""
        return self.sequences[machine][position]

    def clone(self) -> PMALNSSolution:
        """Create a deep copy of this solution."""
        return PMALNSSolution(
            sequences=[list(seq) for seq in self.sequences],
            n_jobs=self.n_jobs,
            n_machines=self.n_machines,
        )

    def hash_key(self) -> Tuple[Tuple[int, ...], ...]:
        """Return a hashable key for this solution (for dedup/caching)."""
        return tuple(tuple(seq) for seq in self.sequences)

    def get_machine_loads(self, processing_times: np.ndarray) -> np.ndarray:
        """Compute total processing time per machine."""
        loads = np.zeros(self.n_machines, dtype=np.int64)
        for mi, seq in enumerate(self.sequences):
            if seq:
                loads[mi] = sum(int(processing_times[j]) for j in seq)
        return loads

    def get_load_imbalance(self, processing_times: np.ndarray) -> float:
        """Compute load imbalance as coefficient of variation."""
        loads = self.get_machine_loads(processing_times)
        if loads.size == 0:
            return 0.0
        mean_load = float(np.mean(loads))
        if mean_load < 1e-9:
            return 0.0
        std_load = float(np.std(loads))
        return std_load / mean_load

    # -------------------------------------------------------------------------
    # Modification operations (used by operators)
    # -------------------------------------------------------------------------

    def remove_job(self, machine: int, position: int) -> int:
        """Remove job at given position, return the removed job."""
        job = self.sequences[machine].pop(position)
        self._invalidate_caches()
        return job

    def insert_job(self, job: int, machine: int, position: int) -> None:
        """Insert job at given machine and position."""
        self.sequences[machine].insert(position, job)
        self._invalidate_caches()

    def swap_jobs(
        self,
        m1: int,
        pos1: int,
        m2: int,
        pos2: int,
    ) -> None:
        """Swap jobs at two positions (same or different machines)."""
        if m1 == m2 and pos1 == pos2:
            return  # No-op

        job1 = self.sequences[m1][pos1]
        job2 = self.sequences[m2][pos2]
        self.sequences[m1][pos1] = job2
        self.sequences[m2][pos2] = job1
        self._invalidate_caches()

    def relocate_job(
        self,
        from_machine: int,
        from_pos: int,
        to_machine: int,
        to_pos: int,
    ) -> int:
        """Relocate job from one position to another, return the moved job.

        Handles position adjustment when moving within same machine.
        """
        job = self.sequences[from_machine].pop(from_pos)

        # Adjust target position if same machine and removing shifts indices
        if from_machine == to_machine and from_pos < to_pos:
            to_pos -= 1

        self.sequences[to_machine].insert(to_pos, job)
        self._invalidate_caches()
        return job

    def relocate_block(
        self,
        from_machine: int,
        from_start: int,
        block_size: int,
        to_machine: int,
        to_pos: int,
    ) -> List[int]:
        """Relocate a contiguous block of jobs, return the moved jobs."""
        # Extract block
        block = self.sequences[from_machine][from_start : from_start + block_size]
        del self.sequences[from_machine][from_start : from_start + block_size]

        # Adjust target position if same machine
        if from_machine == to_machine and from_start < to_pos:
            to_pos -= block_size

        # Insert block at target
        for i, job in enumerate(block):
            self.sequences[to_machine].insert(to_pos + i, job)

        self._invalidate_caches()
        return block

    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_assignment(
        cls,
        assignment: List[int],
        n_machines: int,
        processing_times: Optional[np.ndarray] = None,
        sort_by: str = "spt",
    ) -> PMALNSSolution:
        """Create solution from job-to-machine assignment.

        Args:
            assignment: List where assignment[j] = machine for job j
            n_machines: Number of machines
            processing_times: Optional processing times for sorting
            sort_by: Sorting method for jobs within each machine:
                     "spt" = shortest processing time first
                     "lpt" = longest processing time first
                     "none" = job index order
        """
        n_jobs = len(assignment)
        sequences: List[List[int]] = [[] for _ in range(n_machines)]

        for job, machine in enumerate(assignment):
            sequences[machine].append(job)

        # Sort jobs within each machine
        if processing_times is not None and sort_by != "none":
            for mi in range(n_machines):
                if sort_by == "spt":
                    sequences[mi].sort(key=lambda j: (processing_times[j], j))
                elif sort_by == "lpt":
                    sequences[mi].sort(key=lambda j: (-processing_times[j], j))
                else:
                    sequences[mi].sort()  # by job index
        else:
            # Sort by job index for determinism
            for mi in range(n_machines):
                sequences[mi].sort()

        return cls(sequences=sequences, n_jobs=n_jobs, n_machines=n_machines)

    @classmethod
    def from_random(
        cls,
        n_jobs: int,
        n_machines: int,
        processing_times: np.ndarray,
        rng: np.random.Generator,
    ) -> PMALNSSolution:
        """Create random feasible solution (for testing)."""
        assignment = rng.integers(0, n_machines, size=n_jobs).tolist()
        return cls.from_assignment(
            assignment=assignment,
            n_machines=n_machines,
            processing_times=processing_times,
            sort_by="spt",
        )

    @classmethod
    def from_load_balanced(
        cls,
        n_jobs: int,
        n_machines: int,
        processing_times: np.ndarray,
    ) -> PMALNSSolution:
        """Create load-balanced solution (LPT greedy assignment + SPT ordering).

        Uses Longest Processing Time first for assignment (good load balance),
        then Shortest Processing Time for per-machine ordering.
        """
        # Sort jobs by processing time descending (LPT for assignment)
        sorted_jobs = sorted(range(n_jobs), key=lambda j: (-processing_times[j], j))

        # Greedy assignment: assign each job to least loaded machine
        loads = [0] * n_machines
        assignment = [0] * n_jobs

        for job in sorted_jobs:
            # Find machine with minimum load (deterministic tie-break by index)
            min_machine = min(range(n_machines), key=lambda m: (loads[m], m))
            assignment[job] = min_machine
            loads[min_machine] += processing_times[job]

        return cls.from_assignment(
            assignment=assignment,
            n_machines=n_machines,
            processing_times=processing_times,
            sort_by="spt",
        )


def solution_to_dict(sol: PMALNSSolution) -> Dict[str, Any]:
    """Serialize solution to dictionary."""
    return {
        "sequences": sol.sequences,
        "n_jobs": sol.n_jobs,
        "n_machines": sol.n_machines,
    }


def solution_from_dict(d: Dict[str, Any]) -> PMALNSSolution:
    """Deserialize solution from dictionary."""
    return PMALNSSolution(
        sequences=d["sequences"],
        n_jobs=d["n_jobs"],
        n_machines=d["n_machines"],
    )

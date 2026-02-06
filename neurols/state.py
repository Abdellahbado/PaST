"""MDP state representation for NeuroLS following the paper's design.

State includes:
1. Current solution (assignment + sequences)
2. Current cost f(s)
3. Best cost f(ŝ) found so far
4. Last acceptance decision (boolean)
5. Last operator used
6. Current time step t
7. Number of steps without improvement
8. Number of perturbations/restarts applied

Additional scheduling-specific features:
- Slack to horizon K
- Load imbalance across machines
- Price embedding vector z_price
- Per-machine price-exposure statistics

References: NeuroLS paper Section 4.2 (States)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

from PaST.neurols.solution import PMALNSSolution
from PaST.neurols.operators import OperatorID


@dataclass
class NeuroLSState:
    """Complete MDP state for NeuroLS controller.

    This encapsulates all information the controller needs to make decisions.
    """

    # Core solution state
    solution: PMALNSSolution
    current_cost: float  # f(s): current energy cost
    best_cost: float  # f(ŝ): best energy cost found so far
    best_solution: PMALNSSolution

    # Search trajectory state
    last_acceptance: bool  # Did we accept the last move?
    last_operator: Optional[OperatorID]  # Which operator was last used
    step_t: int  # Current iteration number
    no_improve_steps: int  # Steps since last improvement
    perturbation_count: int  # Number of perturbations applied

    # Problem parameters (needed for feature computation)
    K: int  # Horizon constraint
    processing_times: np.ndarray = field(repr=False)
    machine_energy_rates: np.ndarray = field(repr=False)

    # Optional: per-machine evaluation cache
    per_machine_energy: Optional[np.ndarray] = field(default=None, repr=False)
    per_machine_makespan: Optional[np.ndarray] = field(default=None, repr=False)
    makespan: int = 0

    # Configuration
    max_iterations: int = 200

    def clone(self) -> NeuroLSState:
        """Create a deep copy of the state."""
        return NeuroLSState(
            solution=self.solution.clone(),
            current_cost=self.current_cost,
            best_cost=self.best_cost,
            best_solution=self.best_solution.clone(),
            last_acceptance=self.last_acceptance,
            last_operator=self.last_operator,
            step_t=self.step_t,
            no_improve_steps=self.no_improve_steps,
            perturbation_count=self.perturbation_count,
            K=self.K,
            processing_times=self.processing_times.copy(),
            machine_energy_rates=self.machine_energy_rates.copy(),
            per_machine_energy=(
                self.per_machine_energy.copy()
                if self.per_machine_energy is not None
                else None
            ),
            per_machine_makespan=(
                self.per_machine_makespan.copy()
                if self.per_machine_makespan is not None
                else None
            ),
            makespan=self.makespan,
            max_iterations=self.max_iterations,
        )

    def update_from_move(
        self,
        new_solution: PMALNSSolution,
        new_cost: float,
        new_per_machine_energy: Optional[np.ndarray],
        new_per_machine_makespan: Optional[np.ndarray],
        new_makespan: int,
        accepted: bool,
        operator: Optional[OperatorID],
    ) -> None:
        """Update state after applying a move.

        Args:
            new_solution: Solution after move
            new_cost: Energy cost after move
            new_per_machine_energy: Per-machine energy costs
            new_per_machine_makespan: Per-machine makespans
            new_makespan: Overall makespan
            accepted: Whether the move was accepted
            operator: Operator that was used (None for perturbation)
        """
        if accepted:
            self.solution = new_solution
            self.current_cost = new_cost
            self.per_machine_energy = new_per_machine_energy
            self.per_machine_makespan = new_per_machine_makespan
            self.makespan = new_makespan

            # Check for improvement
            if new_cost < self.best_cost - 1e-9:
                self.best_cost = new_cost
                self.best_solution = new_solution.clone()
                self.no_improve_steps = 0
            else:
                self.no_improve_steps += 1
        else:
            self.no_improve_steps += 1

        self.last_acceptance = accepted
        self.last_operator = operator
        self.step_t += 1

    def update_from_perturbation(
        self,
        new_solution: PMALNSSolution,
        new_cost: float,
        new_per_machine_energy: Optional[np.ndarray],
        new_per_machine_makespan: Optional[np.ndarray],
        new_makespan: int,
    ) -> None:
        """Update state after applying a perturbation.

        Perturbations are always applied (no accept/reject).
        """
        self.solution = new_solution
        self.current_cost = new_cost
        self.per_machine_energy = new_per_machine_energy
        self.per_machine_makespan = new_per_machine_makespan
        self.makespan = new_makespan
        self.perturbation_count += 1
        self.last_operator = None  # Perturbation, not operator
        self.step_t += 1

        if new_cost < self.best_cost - 1e-9:
            self.best_cost = new_cost
            self.best_solution = new_solution.clone()
            self.no_improve_steps = 0
        else:
            self.no_improve_steps += 1

    def is_terminal(self, no_improve_limit: int = 50) -> bool:
        """Check if search should terminate."""
        if self.step_t >= self.max_iterations:
            return True
        if self.no_improve_steps >= no_improve_limit:
            return True
        return False

    # -------------------------------------------------------------------------
    # Feature extraction for model input
    # -------------------------------------------------------------------------

    def get_scalar_features(self) -> np.ndarray:
        """Extract scalar features for decoder input.

        Returns feature vector following NeuroLS state representation.
        """
        # Normalize costs by initial/best estimate
        cost_scale = max(1.0, abs(self.best_cost))

        features = [
            # Cost features
            self.current_cost / cost_scale,  # Normalized current cost
            self.best_cost / cost_scale,  # Normalized best cost
            (self.current_cost - self.best_cost) / cost_scale,  # Gap to best
            # Search progress features
            float(self.last_acceptance),  # Last acceptance (0 or 1)
            float(self.last_operator if self.last_operator is not None else -1)
            / 4.0,  # Last operator normalized
            float(self.step_t) / float(self.max_iterations),  # Progress ratio
            float(self.no_improve_steps)
            / float(max(1, self.step_t)),  # No-improve ratio
            float(self.perturbation_count)
            / float(max(1, self.step_t)),  # Perturbation ratio
            # Makespan slack
            float(self.K - self.makespan) / float(max(1, self.K)),  # Slack to horizon
        ]

        # Load balance features
        loads = self.solution.get_machine_loads(self.processing_times)
        total_load = float(np.sum(loads))
        if total_load > 0:
            features.append(float(np.std(loads)) / float(np.mean(loads) + 1e-9))  # CV
            features.append(float(np.max(loads) - np.min(loads)) / total_load)  # Range
        else:
            features.extend([0.0, 0.0])

        # Per-machine energy features
        if self.per_machine_energy is not None:
            e = self.per_machine_energy
            features.append(float(np.std(e)) / float(np.mean(e) + 1e-9))  # Energy CV
            features.append(float(np.max(e)) / float(np.sum(e) + 1e-9))  # Max share
        else:
            features.extend([0.0, 0.0])

        return np.array(features, dtype=np.float32)

    def get_job_features(self) -> np.ndarray:
        """Extract per-job features for GNN input.

        Returns:
            (n_jobs, F_job) feature matrix
        """
        n_jobs = self.solution.n_jobs

        # Normalize processing times
        p = self.processing_times.astype(np.float32)
        p_max = float(np.max(p)) if p.size > 0 else 1.0
        p_norm = p / p_max

        # Position features
        positions = np.zeros(n_jobs, dtype=np.float32)
        relative_positions = np.zeros(n_jobs, dtype=np.float32)
        machine_assignments = np.zeros(n_jobs, dtype=np.float32)
        machine_loads = self.solution.get_machine_loads(self.processing_times)
        machine_load_norm = np.zeros(n_jobs, dtype=np.float32)

        for job in range(n_jobs):
            mi = self.solution.get_assignment(job)
            pos = self.solution.get_position(job)
            seq_len = len(self.solution.sequences[mi])

            positions[job] = float(pos)
            relative_positions[job] = float(pos) / float(max(1, seq_len - 1))
            machine_assignments[job] = float(mi) / float(
                max(1, self.solution.n_machines - 1)
            )
            machine_load_norm[job] = float(machine_loads[mi]) / float(self.K)

        features = np.stack(
            [
                p_norm,
                positions / float(n_jobs),
                relative_positions,
                machine_assignments,
                machine_load_norm,
            ],
            axis=1,
        )

        return features

    def get_machine_features(self) -> np.ndarray:
        """Extract per-machine features for GNN input.

        Returns:
            (n_machines, F_machine) feature matrix
        """
        n_machines = self.solution.n_machines

        # Load features
        loads = self.solution.get_machine_loads(self.processing_times).astype(
            np.float32
        )
        loads_norm = loads / float(self.K)

        # Energy rate features
        e = self.machine_energy_rates.astype(np.float32)
        e_norm = e / float(np.max(e) + 1e-9)

        # Number of jobs per machine
        n_jobs_per_machine = np.array(
            [len(self.solution.sequences[mi]) for mi in range(n_machines)],
            dtype=np.float32,
        )
        n_jobs_norm = n_jobs_per_machine / float(self.solution.n_jobs)

        # Makespan features
        if self.per_machine_makespan is not None:
            ms = self.per_machine_makespan.astype(np.float32)
            ms_norm = ms / float(self.K)
        else:
            ms_norm = loads_norm  # Proxy

        # Energy features
        if self.per_machine_energy is not None:
            energy = self.per_machine_energy.astype(np.float32)
            energy_norm = energy / float(np.sum(energy) + 1e-9)
        else:
            energy_norm = np.zeros(n_machines, dtype=np.float32)

        features = np.stack(
            [
                loads_norm,
                e_norm,
                n_jobs_norm,
                ms_norm,
                energy_norm,
            ],
            axis=1,
        )

        return features

    def get_assignment_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get dynamic assignment edges for GNN.

        Returns:
            (src_jobs, dst_machines) edge arrays for bipartite graph
        """
        src = []
        dst = []

        for job in range(self.solution.n_jobs):
            mi = self.solution.get_assignment(job)
            src.append(job)
            dst.append(mi)

        return np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary (for logging/debugging)."""
        return {
            "step_t": self.step_t,
            "current_cost": self.current_cost,
            "best_cost": self.best_cost,
            "makespan": self.makespan,
            "no_improve_steps": self.no_improve_steps,
            "perturbation_count": self.perturbation_count,
            "last_acceptance": self.last_acceptance,
            "last_operator": self.last_operator.name if self.last_operator else None,
        }


@dataclass
class StateBuilder:
    """Builder for creating initial NeuroLS state from problem instance."""

    @staticmethod
    def from_instance(
        processing_times: np.ndarray,
        machine_energy_rates: np.ndarray,
        ct: np.ndarray,
        K: int,
        n_machines: int,
        initial_solution: Optional[PMALNSSolution] = None,
        max_iterations: int = 200,
    ) -> NeuroLSState:
        """Create initial state from problem instance.

        Args:
            processing_times: Job processing times, shape (n_jobs,)
            machine_energy_rates: Energy rate per machine, shape (n_machines,)
            ct: Per-slot prices, shape (T_max,)
            K: Horizon constraint (deadline)
            n_machines: Number of machines
            initial_solution: Optional starting solution
            max_iterations: Maximum LS iterations

        Returns:
            Initial NeuroLSState
        """
        n_jobs = len(processing_times)

        # Create initial solution if not provided
        if initial_solution is None:
            initial_solution = PMALNSSolution.from_load_balanced(
                n_jobs=n_jobs,
                n_machines=n_machines,
                processing_times=processing_times,
            )

        # We'll need to evaluate the initial solution to get cost
        # This is done by the caller using MoveEvaluator
        return NeuroLSState(
            solution=initial_solution,
            current_cost=float("inf"),  # Will be set by evaluator
            best_cost=float("inf"),
            best_solution=initial_solution.clone(),
            last_acceptance=True,  # Initial solution is "accepted"
            last_operator=None,
            step_t=0,
            no_improve_steps=0,
            perturbation_count=0,
            K=K,
            processing_times=np.asarray(processing_times, dtype=np.int64),
            machine_energy_rates=np.asarray(machine_energy_rates, dtype=np.float64),
            per_machine_energy=None,
            per_machine_makespan=None,
            makespan=0,
            max_iterations=max_iterations,
        )

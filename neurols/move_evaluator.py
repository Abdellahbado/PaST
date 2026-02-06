"""Move evaluator using incremental DP for fast cost computation.

This module wraps the IncrementalDPSolver to efficiently evaluate LS moves.
Key optimizations:
1. Checkpoint-based incremental DP (restart from nearest checkpoint after modification)
2. Parallel evaluation of multiple candidate moves
3. Per-machine evaluation caching

The evaluator computes optimal timing for each machine's job sequence,
returning total energy cost under the horizon constraint K.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Sequence
import numpy as np

from PaST.neurols.solution import PMALNSSolution
from PaST.neurols.operators import Move, OperatorID, OPERATOR_BY_ID
from PaST.solvers.incremental_dp_solver import (
    IncrementalDPSolver,
    DPCheckpoint,
    IncrementalDPResult,
)


@dataclass
class MachineEval:
    """Evaluation result for a single machine."""

    energy: float
    makespan: int
    start_times: List[int]
    feasible: bool
    checkpoints: List[DPCheckpoint] = field(default_factory=list)


@dataclass
class FullEvaluation:
    """Full solution evaluation across all machines."""

    total_energy: float
    makespan: int  # Max across machines
    feasible: bool
    per_machine: List[MachineEval]
    per_job_costs: Optional[np.ndarray] = None  # Per-job energy contribution


@dataclass
class MoveEvaluation:
    """Evaluation of a specific move."""

    move: Move
    new_cost: float
    delta_cost: float  # new_cost - current_cost
    new_makespan: int
    feasible: bool
    affected_machines: List[int]
    # Per-machine evaluations for affected machines only
    affected_machine_evals: Dict[int, MachineEval] = field(default_factory=dict)


class MoveEvaluator:
    """Evaluator for LS moves using incremental DP.

    This class manages DP solvers for each machine and provides efficient
    incremental evaluation of candidate moves.
    """

    def __init__(
        self,
        processing_times: np.ndarray,
        ct: np.ndarray,
        machine_energy_rates: np.ndarray,
        K: int,
        n_machines: int,
        checkpoint_interval: int = 5,
    ):
        """Initialize the move evaluator.

        Args:
            processing_times: Job processing times, shape (n_jobs,)
            ct: Per-slot prices, shape (T_max,)
            machine_energy_rates: Energy rate per machine, shape (n_machines,)
            K: Horizon constraint (deadline)
            n_machines: Number of machines
            checkpoint_interval: Checkpoint interval for incremental DP
        """
        self.processing_times = np.asarray(processing_times, dtype=np.int64)
        self.ct = np.asarray(ct[:K], dtype=np.float64)  # Truncate to K
        self.machine_energy_rates = np.asarray(machine_energy_rates, dtype=np.float64)
        self.K = int(K)
        self.n_machines = int(n_machines)
        self.checkpoint_interval = int(checkpoint_interval)

        # Create DP solver for each machine (different energy rates)
        self._solvers: List[IncrementalDPSolver] = []
        for mi in range(n_machines):
            solver = IncrementalDPSolver(
                processing_times=self.processing_times,
                ct=self.ct,
                e_single=float(self.machine_energy_rates[mi]),
                T_limit=self.K,
                checkpoint_interval=self.checkpoint_interval,
            )
            self._solvers.append(solver)

        # Cache for per-machine checkpoints (keyed by sequence hash)
        self._checkpoint_cache: Dict[
            Tuple[int, Tuple[int, ...]], List[DPCheckpoint]
        ] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def evaluate_solution(
        self,
        solution: PMALNSSolution,
        compute_per_job_costs: bool = False,
    ) -> FullEvaluation:
        """Evaluate a complete solution.

        Args:
            solution: Solution to evaluate
            compute_per_job_costs: Whether to compute per-job energy costs

        Returns:
            FullEvaluation with total cost and per-machine breakdown
        """
        total_energy = 0.0
        makespan = 0
        feasible = True
        per_machine: List[MachineEval] = []

        per_job_costs = (
            np.zeros(solution.n_jobs, dtype=np.float64)
            if compute_per_job_costs
            else None
        )

        for mi in range(self.n_machines):
            seq = solution.sequences[mi]

            if not seq:
                # Empty machine
                per_machine.append(
                    MachineEval(
                        energy=0.0,
                        makespan=0,
                        start_times=[],
                        feasible=True,
                        checkpoints=[
                            DPCheckpoint(position=0, min_prev=np.zeros(self.K + 1))
                        ],
                    )
                )
                continue

            # Solve with checkpoints
            result = self._solvers[mi].solve_with_checkpoints(seq)

            # Cache checkpoints
            seq_key = (mi, tuple(seq))
            self._checkpoint_cache[seq_key] = result.checkpoints

            if not np.isfinite(result.total_cost) or result.makespan > self.K:
                feasible = False
                per_machine.append(
                    MachineEval(
                        energy=float("inf"),
                        makespan=self.K + 1,
                        start_times=[],
                        feasible=False,
                        checkpoints=result.checkpoints,
                    )
                )
            else:
                per_machine.append(
                    MachineEval(
                        energy=result.total_cost,
                        makespan=result.makespan,
                        start_times=result.start_times,
                        feasible=True,
                        checkpoints=result.checkpoints,
                    )
                )
                total_energy += result.total_cost
                makespan = max(makespan, result.makespan)

                # Compute per-job costs if requested
                if compute_per_job_costs and result.start_times:
                    e_rate = float(self.machine_energy_rates[mi])
                    for idx, job in enumerate(seq):
                        if idx < len(result.start_times):
                            start = result.start_times[idx]
                            p = int(self.processing_times[job])
                            end = min(start + p, len(self.ct))
                            if start >= 0 and end > start:
                                cost = e_rate * float(np.sum(self.ct[start:end]))
                                per_job_costs[job] = cost

        return FullEvaluation(
            total_energy=total_energy if feasible else float("inf"),
            makespan=makespan,
            feasible=feasible,
            per_machine=per_machine,
            per_job_costs=per_job_costs,
        )

    def evaluate_move(
        self,
        solution: PMALNSSolution,
        current_eval: FullEvaluation,
        move: Move,
    ) -> MoveEvaluation:
        """Evaluate a proposed move using incremental DP.

        Args:
            solution: Current solution (before move)
            current_eval: Current solution evaluation
            move: Move to evaluate

        Returns:
            MoveEvaluation with new cost and delta
        """
        # Clone solution and apply move
        new_sol = solution.clone()
        operator = OPERATOR_BY_ID[move.operator]
        operator.apply_move(new_sol, move)

        # Get affected machines
        affected = operator.get_affected_machines(move)

        # Recompute only affected machines
        new_per_machine = list(current_eval.per_machine)
        affected_evals: Dict[int, MachineEval] = {}

        for mi in affected:
            seq = new_sol.sequences[mi]

            if not seq:
                new_eval = MachineEval(
                    energy=0.0,
                    makespan=0,
                    start_times=[],
                    feasible=True,
                    checkpoints=[
                        DPCheckpoint(position=0, min_prev=np.zeros(self.K + 1))
                    ],
                )
            else:
                # Try to use cached checkpoints for incremental computation
                modified_pos = operator.get_affected_position(move, mi)
                cached = self._get_cached_checkpoints(mi, solution.sequences[mi])

                if cached is not None:
                    self._cache_hits += 1
                    cost, ms = self._solvers[mi].recompute_from_position(
                        seq, modified_pos, cached
                    )
                    # For full eval with start times, we'd need to recompute fully
                    # for now, just get cost and makespan
                    if np.isfinite(cost) and ms <= self.K:
                        new_eval = MachineEval(
                            energy=cost,
                            makespan=ms,
                            start_times=[],  # Not computed incrementally
                            feasible=True,
                            checkpoints=[],  # Will be computed if move accepted
                        )
                    else:
                        new_eval = MachineEval(
                            energy=float("inf"),
                            makespan=self.K + 1,
                            start_times=[],
                            feasible=False,
                            checkpoints=[],
                        )
                else:
                    self._cache_misses += 1
                    # Full recomputation
                    result = self._solvers[mi].solve_with_checkpoints(seq)
                    if np.isfinite(result.total_cost) and result.makespan <= self.K:
                        new_eval = MachineEval(
                            energy=result.total_cost,
                            makespan=result.makespan,
                            start_times=result.start_times,
                            feasible=True,
                            checkpoints=result.checkpoints,
                        )
                    else:
                        new_eval = MachineEval(
                            energy=float("inf"),
                            makespan=self.K + 1,
                            start_times=[],
                            feasible=False,
                            checkpoints=[],
                        )

            new_per_machine[mi] = new_eval
            affected_evals[mi] = new_eval

        # Compute new total
        new_total = 0.0
        new_makespan = 0
        new_feasible = True

        for m_eval in new_per_machine:
            if not m_eval.feasible:
                new_feasible = False
                break
            new_total += m_eval.energy
            new_makespan = max(new_makespan, m_eval.makespan)

        if not new_feasible:
            new_total = float("inf")

        return MoveEvaluation(
            move=move,
            new_cost=new_total,
            delta_cost=new_total - current_eval.total_energy,
            new_makespan=new_makespan,
            feasible=new_feasible,
            affected_machines=affected,
            affected_machine_evals=affected_evals,
        )

    def evaluate_moves_batch(
        self,
        solution: PMALNSSolution,
        current_eval: FullEvaluation,
        moves: List[Move],
    ) -> List[MoveEvaluation]:
        """Evaluate multiple moves.

        TODO: Implement true batched evaluation with parallel DP.
        For now, just evaluates sequentially.

        Args:
            solution: Current solution
            current_eval: Current evaluation
            moves: List of moves to evaluate

        Returns:
            List of MoveEvaluation results
        """
        results = []
        for move in moves:
            results.append(self.evaluate_move(solution, current_eval, move))
        return results

    def _get_cached_checkpoints(
        self, machine: int, sequence: List[int]
    ) -> Optional[List[DPCheckpoint]]:
        """Get cached checkpoints for a machine sequence."""
        key = (machine, tuple(sequence))
        return self._checkpoint_cache.get(key)

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache hit/miss statistics."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "cache_size": len(self._checkpoint_cache),
        }

    def clear_cache(self) -> None:
        """Clear the checkpoint cache."""
        self._checkpoint_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


class BatchMoveEvaluator:
    """Batch move evaluator for efficient parallel evaluation.

    This class is designed for evaluating many candidate moves efficiently,
    potentially using GPU acceleration via PyTorch.
    """

    def __init__(
        self,
        processing_times: np.ndarray,
        ct: np.ndarray,
        machine_energy_rates: np.ndarray,
        K: int,
        n_machines: int,
        use_torch: bool = True,
        device: str = "auto",
    ):
        """Initialize batch evaluator.

        Args:
            processing_times: Job processing times
            ct: Per-slot prices
            machine_energy_rates: Energy rate per machine
            K: Horizon constraint
            n_machines: Number of machines
            use_torch: Whether to use PyTorch for batched DP
            device: Device for PyTorch ("auto", "cpu", "cuda")
        """
        self.processing_times = np.asarray(processing_times, dtype=np.int64)
        self.ct = np.asarray(ct[:K], dtype=np.float64)
        self.machine_energy_rates = np.asarray(machine_energy_rates, dtype=np.float64)
        self.K = int(K)
        self.n_machines = int(n_machines)
        self.use_torch = use_torch

        # Determine device
        if device == "auto":
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
                self.use_torch = False
        else:
            self.device = device

        # Load torch components if available
        self._torch_available = False
        if self.use_torch:
            try:
                import torch
                from PaST.solvers.batch_dp_solver import BatchSequenceDPSolver

                self._torch = torch
                self._BatchDPSolver = BatchSequenceDPSolver
                self._torch_available = True
            except ImportError:
                self._torch_available = False

    def evaluate_sequences_batch(
        self,
        machine_sequences: List[Tuple[int, List[int]]],
    ) -> List[Tuple[float, int]]:
        """Evaluate multiple (machine, sequence) pairs in batch.

        Args:
            machine_sequences: List of (machine_idx, job_sequence) tuples

        Returns:
            List of (energy, makespan) tuples
        """
        if not machine_sequences:
            return []

        if self._torch_available and len(machine_sequences) >= 8:
            return self._evaluate_batch_torch(machine_sequences)
        else:
            return self._evaluate_batch_numpy(machine_sequences)

    def _evaluate_batch_numpy(
        self,
        machine_sequences: List[Tuple[int, List[int]]],
    ) -> List[Tuple[float, int]]:
        """Evaluate using numpy (fallback)."""
        results = []

        for mi, seq in machine_sequences:
            if not seq:
                results.append((0.0, 0))
                continue

            solver = IncrementalDPSolver(
                processing_times=self.processing_times,
                ct=self.ct,
                e_single=float(self.machine_energy_rates[mi]),
                T_limit=self.K,
            )
            cost, makespan, _ = solver.solve_full(seq)

            if np.isfinite(cost) and makespan <= self.K:
                results.append((cost, makespan))
            else:
                results.append((float("inf"), self.K + 1))

        return results

    def _evaluate_batch_torch(
        self,
        machine_sequences: List[Tuple[int, List[int]]],
    ) -> List[Tuple[float, int]]:
        """Evaluate using PyTorch batched DP."""
        torch = self._torch
        B = len(machine_sequences)

        # Find max sequence length
        lengths = [len(seq) for _, seq in machine_sequences]
        max_len = max(lengths) if lengths else 1

        # Pad sequences
        proc_batch = np.zeros((B, max_len), dtype=np.int64)
        seq_lens = np.array(lengths, dtype=np.int64)
        e_rates = np.zeros((B,), dtype=np.int64)

        for i, (mi, seq) in enumerate(machine_sequences):
            for j, job in enumerate(seq):
                proc_batch[i, j] = int(self.processing_times[job])
            e_rates[i] = int(self.machine_energy_rates[mi])

        # Convert to torch
        device = torch.device(self.device)
        job_sequences = (
            torch.arange(max_len, dtype=torch.long, device=device)
            .unsqueeze(0)
            .repeat(B, 1)
        )
        processing_times = torch.tensor(proc_batch, dtype=torch.long, device=device)
        ct_t = (
            torch.tensor(self.ct, dtype=torch.int32, device=device)
            .unsqueeze(0)
            .repeat(B, 1)
        )
        e_t = torch.tensor(e_rates, dtype=torch.long, device=device)
        T_t = torch.full((B,), self.K, dtype=torch.long, device=device)
        seq_len_t = torch.tensor(seq_lens, dtype=torch.long, device=device)

        # Run batch DP
        costs_t, end_t = self._BatchDPSolver.solve_with_end_time(
            job_sequences=job_sequences,
            processing_times=processing_times,
            ct=ct_t,
            e_single=e_t,
            T_limit=T_t,
            sequence_lengths=seq_len_t,
        )

        costs = costs_t.detach().cpu().numpy()
        makespans = end_t.detach().cpu().numpy()

        results = []
        for i in range(B):
            if np.isfinite(costs[i]) and makespans[i] <= self.K:
                results.append((float(costs[i]), int(makespans[i])))
            else:
                results.append((float("inf"), self.K + 1))

        return results

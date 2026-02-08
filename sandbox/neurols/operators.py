"""Deterministic local search operators for parallel machine scheduling.

Each operator defines a neighborhood over solutions. Given an operator and
the current solution, the candidate generator enumerates all valid moves
in a fixed deterministic order.

Operators:
- RELOCATE_1: Remove job from (m1, pos1), insert at (m2, pos2)
- SWAP_1: Exchange jobs at (m1, pos1) and (m2, pos2)
- INTRA_INSERT: Move job within same machine (special case of relocate)
- BLOCK_RELOCATE: Move contiguous block of 2-3 jobs

Reachability: This set guarantees any solution can reach any other:
- RELOCATE_1 can change assignment (cross-machine) and any permutation
- SWAP_1 provides alternative exploration path
- INTRA_INSERT optimizes per-machine ordering
- BLOCK_RELOCATE enables larger structure changes
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import List, Tuple, Optional, NamedTuple

from sandbox.neurols.solution import PMALNSSolution


class OperatorID(IntEnum):
    """Operator identifiers (deterministic integer indices)."""

    RELOCATE_1 = 0
    SWAP_1 = 1
    INTRA_INSERT = 2
    BLOCK_RELOCATE = 3
    EVACUATE_PEAK = 4
    SWAP_PEAK_VALLEY = 5  # [NEW]


class Move(NamedTuple):
    """A specific move instance to be applied to a solution.

    Attributes:
        operator: Which operator this move belongs to
        params: Operator-specific parameters (tuple of ints)

    Parameter format by operator:
        RELOCATE_1: (from_m, from_pos, to_m, to_pos)
        SWAP_1: (m1, pos1, m2, pos2)
        INTRA_INSERT: (machine, from_pos, to_pos)
        BLOCK_RELOCATE: (from_m, from_start, block_size, to_m, to_pos)
        EVACUATE_PEAK: (from_m, from_pos, to_m, to_pos)
        SWAP_PEAK_VALLEY: (m1, pos1, m2, pos2)  # Same as SWAP_1
    """

    operator: OperatorID
    params: Tuple[int, ...]
    
    



@dataclass
class Operator:
    """Operator definition with metadata for learning."""

    id: OperatorID
    name: str
    description: str

    # Operator characteristics (useful features for the model)
    is_intra_machine: bool  # Only moves within same machine
    is_cross_machine: bool  # Can move between machines
    preserves_sequence: bool  # Doesn't change relative order of other jobs

    def enumerate_moves(
        self,
        solution: PMALNSSolution,
        max_moves: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Move]:
        """Enumerate all valid moves for this operator in deterministic order.

        Enumeration order: lexicographic by (m1, pos1, m2, pos2, ...) ensuring
        reproducibility.

        Args:
            solution: Current solution
            max_moves: Optional limit on number of moves to enumerate
            context: Optional dictionary with 'evaluation' and 'price_analyzer'

        Returns:
            List of Move objects in deterministic order
        """
        if self.id == OperatorID.RELOCATE_1:
            return self._enum_relocate(solution, max_moves)
        elif self.id == OperatorID.SWAP_1:
            return self._enum_swap(solution, max_moves)
        elif self.id == OperatorID.INTRA_INSERT:
            return self._enum_intra_insert(solution, max_moves)
        elif self.id == OperatorID.BLOCK_RELOCATE:
            return self._enum_block_relocate(solution, max_moves)
        elif self.id == OperatorID.EVACUATE_PEAK:
            return self._enum_evacuate_peak(solution, max_moves, context)
        elif self.id == OperatorID.SWAP_PEAK_VALLEY:
            return self._enum_swap_peak_valley(solution, max_moves, context)
        else:
            raise ValueError(f"Unknown operator {self.id}")

    def _enum_evacuate_peak(
        self,
        solution: PMALNSSolution,
        max_moves: Optional[int],
        context: Optional[Dict[str, Any]],
    ) -> List[Move]:
        """Identify jobs in PEAK windows and generate relocations for them."""
        if not context or "evaluation" not in context or "price_analyzer" not in context:
            return []  # Can't operate without physics context

        ev = context["evaluation"]
        analyzer = context["price_analyzer"]
        
        # We need processing times to know when a job ends.
        # But 'evaluation' typically has start times if it's the right object.
        # Actually FullEvaluation (move_evaluator.py) has 'per_machine' 
        # which is list of MachineEvaluation.
        # MachineEvaluation has 'start_times'.
        
        moves: List[Move] = []
        n_machines = solution.n_machines
        
        # 1. Identify "Bad Jobs" (those in Peak)
        bad_jobs = [] # List of (m, pos)
        
        for m in range(n_machines):
            machine_eval = ev.per_machine[m]
            seq = solution.sequences[m]
            start_times = machine_eval.start_times
            # machine_eval might store job indices differently or just start times matching the sequence?
            # Usually start_times corresponds to jobs in sequence order.
            
            for pos, job_id in enumerate(seq):
                if pos >= len(start_times): break
                t_start = start_times[pos]
                # We need duration. 'processing_times' should be in context too.
                p_j = context["processing_times"][job_id] if "processing_times" in context else 1
                t_end = t_start + p_j
                
                # Check overlap with PEAK
                # Heuristic: if midpoint is in PEAK, or mostly in PEAK
                t_mid = t_start + p_j // 2
                if analyzer.get_tier_at(t_mid) == "PEAK":
                   bad_jobs.append((m, pos))
        
        # 2. Generate Relocate moves for bad jobs
        # Target: preferably VALLEY.
        # For simplicity, let's try to relocate them to ANY other legal position 
        # (the reward function will pick the "good" ones).
        # Or, strictly target Valleylands? The user said "design better moves".
        # Let's filter targets to be likely Valleys? No, start times at target are dynamic.
        # So just proposing "Relocate BAD job" is already a strong heuristic filter.
        
        for (from_m, from_pos) in bad_jobs:
            for to_m in range(n_machines):
                to_seq = solution.sequences[to_m]
                
                limit = len(to_seq)
                if from_m == to_m: limit -= 1
                
                # We can't really predict where a Valley is without looking at target machine load.
                # But let's just enumerate all positions for these select jobs.
                # This significantly reduces the search space compared to full Relocate (N*M*N moves).
                # Here we only do BadJobs * M * N/M = BadJobs * N moves.
                
                n_target = len(to_seq) + 1 if from_m != to_m else len(to_seq)
                
                for to_pos in range(n_target):
                    if from_m == to_m:
                         if to_pos == from_pos or to_pos == from_pos + 1: continue
                    
                    moves.append(Move(OperatorID.EVACUATE_PEAK, (from_m, from_pos, to_m, to_pos)))
                    if max_moves and len(moves) >= max_moves: return moves
                    
        return moves

    def _enum_swap_peak_valley(
        self,
        solution: PMALNSSolution,
        max_moves: Optional[int],
        context: Optional[Dict[str, Any]],
    ) -> List[Move]:
        """Generate swaps between jobs in PEAK and jobs in VALLEY."""
        if not context or "evaluation" not in context or "price_analyzer" not in context:
            return []

        ev = context["evaluation"]
        analyzer = context["price_analyzer"]
        
        peak_jobs = [] 
        valley_jobs = []
        
        # 1. Classify all jobs
        for m in range(solution.n_machines):
            machine_eval = ev.per_machine[m]
            seq = solution.sequences[m]
            start_times = machine_eval.start_times
            
            for pos, job_id in enumerate(seq):
                if pos >= len(start_times): break
                t_start = start_times[pos]
                p_j = context["processing_times"][job_id] if "processing_times" in context else 1
                t_mid = t_start + p_j // 2
                
                tier = analyzer.get_tier_at(t_mid)
                if tier == "PEAK":
                    peak_jobs.append((m, pos))
                elif tier == "VALLEY":
                    valley_jobs.append((m, pos))
                    
        # 2. Generate Swaps
        moves: List[Move] = []
        for (m1, pos1) in peak_jobs:
            for (m2, pos2) in valley_jobs:
                if m1 == m2 and pos1 == pos2: continue
                
                moves.append(Move(OperatorID.SWAP_PEAK_VALLEY, (m1, pos1, m2, pos2)))
                if max_moves and len(moves) >= max_moves: return moves
                
        return moves

    def _enum_relocate(
        self, solution: PMALNSSolution, max_moves: Optional[int]
    ) -> List[Move]:
        """Enumerate RELOCATE_1 moves: (from_m, from_pos) -> (to_m, to_pos)."""
        moves: List[Move] = []
        n_machines = solution.n_machines

        for from_m in range(n_machines):
            from_seq = solution.sequences[from_m]
            for from_pos in range(len(from_seq)):
                for to_m in range(n_machines):
                    to_seq = solution.sequences[to_m]
                    # Target positions: 0 to len(to_seq) inclusive
                    # But if same machine, we need len-1 positions (after removal)
                    if from_m == to_m:
                        # Skip same position and adjacent (no-op or equivalent)
                        n_target = len(
                            to_seq
                        )  # After removal: len-1 + 1 = len positions
                        for to_pos in range(n_target):
                            # Skip no-op: reinserting at same spot or adjacent
                            # After removing from_pos, positions shift
                            # to_pos < from_pos: insert before, actual stays same
                            # to_pos >= from_pos: insert after removal point
                            if to_pos == from_pos or to_pos == from_pos + 1:
                                continue  # No actual change
                            moves.append(
                                Move(
                                    operator=OperatorID.RELOCATE_1,
                                    params=(from_m, from_pos, to_m, to_pos),
                                )
                            )
                            if max_moves and len(moves) >= max_moves:
                                return moves
                    else:
                        # Cross-machine: all target positions valid
                        n_target = len(to_seq) + 1
                        for to_pos in range(n_target):
                            moves.append(
                                Move(
                                    operator=OperatorID.RELOCATE_1,
                                    params=(from_m, from_pos, to_m, to_pos),
                                )
                            )
                            if max_moves and len(moves) >= max_moves:
                                return moves
        return moves

    def _enum_swap(
        self, solution: PMALNSSolution, max_moves: Optional[int]
    ) -> List[Move]:
        """Enumerate SWAP_1 moves: exchange jobs at (m1,p1) and (m2,p2)."""
        moves: List[Move] = []
        n_machines = solution.n_machines

        # Enumerate unordered pairs to avoid duplicates
        for m1 in range(n_machines):
            seq1 = solution.sequences[m1]
            for pos1 in range(len(seq1)):
                # For same machine: pos2 > pos1 to avoid duplicates
                # For different machines: m2 > m1 to avoid duplicates
                for m2 in range(m1, n_machines):
                    seq2 = solution.sequences[m2]
                    start_pos2 = pos1 + 1 if m1 == m2 else 0
                    for pos2 in range(start_pos2, len(seq2)):
                        moves.append(
                            Move(
                                operator=OperatorID.SWAP_1, params=(m1, pos1, m2, pos2)
                            )
                        )
                        if max_moves and len(moves) >= max_moves:
                            return moves
        return moves

    def _enum_intra_insert(
        self, solution: PMALNSSolution, max_moves: Optional[int]
    ) -> List[Move]:
        """Enumerate INTRA_INSERT moves: move job within same machine."""
        moves: List[Move] = []

        for m in range(solution.n_machines):
            seq = solution.sequences[m]
            n = len(seq)
            for from_pos in range(n):
                # After removing from_pos, there are n-1 jobs
                # Insert positions: 0 to n-1 inclusive
                for to_pos in range(n):
                    # Skip no-ops
                    if to_pos == from_pos or to_pos == from_pos + 1:
                        continue
                    moves.append(
                        Move(
                            operator=OperatorID.INTRA_INSERT,
                            params=(m, from_pos, to_pos),
                        )
                    )
                    if max_moves and len(moves) >= max_moves:
                        return moves
        return moves

    def _enum_block_relocate(
        self, solution: PMALNSSolution, max_moves: Optional[int]
    ) -> List[Move]:
        """Enumerate BLOCK_RELOCATE moves: move block of 2-3 jobs."""
        moves: List[Move] = []
        n_machines = solution.n_machines

        for block_size in [2, 3]:  # Fixed block sizes
            for from_m in range(n_machines):
                from_seq = solution.sequences[from_m]
                # Block starting positions
                for from_start in range(len(from_seq) - block_size + 1):
                    for to_m in range(n_machines):
                        to_seq = solution.sequences[to_m]

                        if from_m == to_m:
                            # Same machine: adjust for removal
                            n_remaining = len(to_seq) - block_size
                            for to_pos in range(n_remaining + 1):
                                # Skip if effectively same position
                                # Block at [from_start, from_start+block_size)
                                # After removal, to_pos maps to actual position
                                if from_start <= to_pos < from_start + block_size:
                                    continue
                                # Adjust: if to_pos >= from_start, the indices shift
                                actual_to = to_pos
                                if to_pos > from_start:
                                    actual_to = to_pos + block_size
                                if actual_to == from_start:
                                    continue  # No-op
                                moves.append(
                                    Move(
                                        operator=OperatorID.BLOCK_RELOCATE,
                                        params=(
                                            from_m,
                                            from_start,
                                            block_size,
                                            to_m,
                                            to_pos,
                                        ),
                                    )
                                )
                                if max_moves and len(moves) >= max_moves:
                                    return moves
                        else:
                            # Cross-machine: all positions valid
                            for to_pos in range(len(to_seq) + 1):
                                moves.append(
                                    Move(
                                        operator=OperatorID.BLOCK_RELOCATE,
                                        params=(
                                            from_m,
                                            from_start,
                                            block_size,
                                            to_m,
                                            to_pos,
                                        ),
                                    )
                                )
                                if max_moves and len(moves) >= max_moves:
                                    return moves
        return moves

    def apply_move(self, solution: PMALNSSolution, move: Move) -> None:
        """Apply a move to the solution (in-place modification).

        Args:
            solution: Solution to modify
            move: Move to apply (must match this operator)
        """
        if move.operator != self.id:
            raise ValueError(f"Move operator {move.operator} doesn't match {self.id}")

        if self.id == OperatorID.RELOCATE_1:
            from_m, from_pos, to_m, to_pos = move.params
            solution.relocate_job(from_m, from_pos, to_m, to_pos)

        elif self.id == OperatorID.SWAP_1:
            m1, pos1, m2, pos2 = move.params
            solution.swap_jobs(m1, pos1, m2, pos2)

        elif self.id == OperatorID.INTRA_INSERT:
            m, from_pos, to_pos = move.params
            solution.relocate_job(m, from_pos, m, to_pos)

        elif self.id == OperatorID.BLOCK_RELOCATE:
            from_m, from_start, block_size, to_m, to_pos = move.params
            solution.relocate_block(from_m, from_start, block_size, to_m, to_pos)
            
        elif self.id == OperatorID.EVACUATE_PEAK:
            from_m, from_pos, to_m, to_pos = move.params
            solution.relocate_job(from_m, from_pos, to_m, to_pos)
            
        elif self.id == OperatorID.SWAP_PEAK_VALLEY:
            m1, pos1, m2, pos2 = move.params
            solution.swap_jobs(m1, pos1, m2, pos2)

        else:
            raise ValueError(f"Unknown operator {self.id}")

    def get_affected_machines(self, move: Move) -> List[int]:
        """Get list of machine indices affected by this move.

        Used for incremental DP: only re-evaluate affected machines.
        """
        if self.id == OperatorID.RELOCATE_1:
            from_m, _, to_m, _ = move.params
            return sorted(set([from_m, to_m]))

        elif self.id == OperatorID.SWAP_1:
            m1, _, m2, _ = move.params
            return sorted(set([m1, m2]))
            
        elif self.id == OperatorID.EVACUATE_PEAK:
            from_m, _, to_m, _ = move.params
            return sorted(set([from_m, to_m]))
            
        elif self.id == OperatorID.SWAP_PEAK_VALLEY:
            m1, _, m2, _ = move.params
            return sorted(set([m1, m2]))

        elif self.id == OperatorID.INTRA_INSERT:
            m, _, _ = move.params
            return [m]

        elif self.id == OperatorID.BLOCK_RELOCATE:
            from_m, _, _, to_m, _ = move.params
            return sorted(set([from_m, to_m]))

        else:
            raise ValueError(f"Unknown operator {self.id}")

    def get_affected_position(self, move: Move, machine: int) -> int:
        """Get the earliest affected position in a machine sequence.

        Used for checkpoint-based incremental DP.
        """
        if self.id == OperatorID.RELOCATE_1:
            from_m, from_pos, to_m, to_pos = move.params
            if machine == from_m and machine == to_m:
                return min(from_pos, to_pos)
            elif machine == from_m:
                return from_pos
            else:
                return to_pos

        elif self.id == OperatorID.SWAP_1:
            m1, pos1, m2, pos2 = move.params
            if machine == m1:
                return pos1
            else:
                return pos2

        elif self.id == OperatorID.INTRA_INSERT:
            _, from_pos, to_pos = move.params
            return min(from_pos, to_pos)

        elif self.id == OperatorID.BLOCK_RELOCATE:
            from_m, from_start, _, to_m, to_pos = move.params
            if machine == from_m and machine == to_m:
                return min(from_start, to_pos)
            elif machine == from_m:
                return from_start
            else:
                return to_pos

        else:
            return 0


# Create operator instances
RELOCATE_1 = Operator(
    id=OperatorID.RELOCATE_1,
    name="Relocate-1",
    description="Remove one job and insert at different position",
    is_intra_machine=False,
    is_cross_machine=True,
    preserves_sequence=False,
)

SWAP_1 = Operator(
    id=OperatorID.SWAP_1,
    name="Swap-1",
    description="Exchange two jobs (same or cross machine)",
    is_intra_machine=True,
    is_cross_machine=True,
    preserves_sequence=True,
)

INTRA_INSERT = Operator(
    id=OperatorID.INTRA_INSERT,
    name="Intra-Insert",
    description="Move job within same machine sequence",
    is_intra_machine=True,
    is_cross_machine=False,
    preserves_sequence=False,
)

BLOCK_RELOCATE = Operator(
    id=OperatorID.BLOCK_RELOCATE,
    name="Block-Relocate",
    description="Move contiguous block of 2-3 jobs",
    is_intra_machine=False,
    is_cross_machine=True,
    preserves_sequence=True,  # Within block
)

EVACUATE_PEAK = Operator(
    id=OperatorID.EVACUATE_PEAK,
    name="Evacuate-Peak",
    description="Evacuate jobs from Peak price windows",
    is_intra_machine=False,
    is_cross_machine=True,
    preserves_sequence=False,
)

SWAP_PEAK_VALLEY = Operator(
    id=OperatorID.SWAP_PEAK_VALLEY,
    name="Swap-Peak-Valley",
    description="Swap jobs between Peak and Valley windows",
    is_intra_machine=True, # Actually depends, but let's say True/False doesn't matter much for non-standard
    is_cross_machine=True,
    preserves_sequence=True, # Swaps preserve sequence structure (mostly)
)

# Operator registry (indexed by OperatorID)
OPERATORS: List[Operator] = [
    RELOCATE_1,
    SWAP_1,
    INTRA_INSERT,
    BLOCK_RELOCATE,
    EVACUATE_PEAK,
    SWAP_PEAK_VALLEY,
]

# Convenience lookup
OPERATOR_BY_ID = {op.id: op for op in OPERATORS}


def get_operator(operator_id: OperatorID) -> Operator:
    """Get operator by ID."""
    return OPERATOR_BY_ID[operator_id]

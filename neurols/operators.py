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

from PaST.neurols.solution import PMALNSSolution


class OperatorID(IntEnum):
    """Operator identifiers (deterministic integer indices)."""

    RELOCATE_1 = 0
    SWAP_1 = 1
    INTRA_INSERT = 2
    BLOCK_RELOCATE = 3


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
    ) -> List[Move]:
        """Enumerate all valid moves for this operator in deterministic order.

        Enumeration order: lexicographic by (m1, pos1, m2, pos2, ...) ensuring
        reproducibility.

        Args:
            solution: Current solution
            max_moves: Optional limit on number of moves to enumerate

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
        else:
            raise ValueError(f"Unknown operator {self.id}")

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

# Operator registry (indexed by OperatorID)
OPERATORS: List[Operator] = [
    RELOCATE_1,
    SWAP_1,
    INTRA_INSERT,
    BLOCK_RELOCATE,
]

# Convenience lookup
OPERATOR_BY_ID = {op.id: op for op in OPERATORS}


def get_operator(operator_id: OperatorID) -> Operator:
    """Get operator by ID."""
    return OPERATOR_BY_ID[operator_id]

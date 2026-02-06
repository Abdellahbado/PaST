"""Action space definitions for NeuroLS following the paper's AA/AAN/AANP framework.

NeuroLS defines three action spaces:
- AA: Acceptance only (binary: accept or reject proposed move)
- AAN: Acceptance × Operator (which operator + accept/reject)
- AANP: Acceptance × (Operators ∪ Perturbations) (full control)

Actions are deterministic: selecting an action leads to a specific, reproducible
outcome (operator → best move from that neighborhood, evaluated via DP).

Action encoding (AANP, primary):
- Actions 0 to 2*|Operators|-1: operator moves with accept/reject
- Actions 2*|Operators| to end: perturbations with accept/reject

For AANP with 4 operators and 3 perturbations:
Total actions = 2 * (4 + 3) = 14
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple, NamedTuple, Optional

from PaST.neurols.operators import OperatorID, OPERATORS
from PaST.neurols.perturbations import PerturbationID, PERTURBATIONS


class ActionType(IntEnum):
    """Type of action being taken."""

    OPERATOR = 0  # Apply an operator move
    PERTURBATION = 1  # Apply a perturbation


class DecodedAction(NamedTuple):
    """Decoded action components."""

    accept: bool  # Accept/reject the resulting move
    action_type: ActionType  # Operator or perturbation
    operator_id: Optional[OperatorID]  # If operator
    perturbation_id: Optional[PerturbationID]  # If perturbation


@dataclass
class ActionSpace:
    """Action space definition with encoding/decoding utilities.

    Attributes:
        name: Action space name (AA, AAN, AANP)
        n_actions: Total number of discrete actions
        operators: List of included operators (empty for AA)
        perturbations: List of included perturbations (empty for AA, AAN)
    """

    name: str
    n_actions: int
    operators: List[OperatorID]
    perturbations: List[PerturbationID]

    # Derived
    n_operators: int = 0
    n_perturbations: int = 0

    def __post_init__(self):
        self.n_operators = len(self.operators)
        self.n_perturbations = len(self.perturbations)

        # Verify action count
        if self.name == "AA":
            expected = 2  # accept, reject
        elif self.name == "AAN":
            expected = 2 * self.n_operators
        elif self.name == "AANP":
            expected = 2 * (self.n_operators + self.n_perturbations)
        else:
            expected = self.n_actions

        if self.n_actions != expected:
            raise ValueError(f"Action count mismatch: {self.n_actions} != {expected}")

    def encode_action(
        self,
        accept: bool,
        operator_id: Optional[OperatorID] = None,
        perturbation_id: Optional[PerturbationID] = None,
    ) -> int:
        """Encode action components to discrete action index.

        For AA: just accept (1) or reject (0)
        For AAN: accept + operator
        For AANP: accept + operator/perturbation
        """
        accept_bit = 1 if accept else 0

        if self.name == "AA":
            return accept_bit

        elif self.name == "AAN":
            if operator_id is None:
                raise ValueError("AAN requires operator_id")
            op_idx = self.operators.index(operator_id)
            return accept_bit + 2 * op_idx

        elif self.name == "AANP":
            if operator_id is not None:
                op_idx = self.operators.index(operator_id)
                return accept_bit + 2 * op_idx
            elif perturbation_id is not None:
                # Perturbations come after operators
                pert_idx = self.perturbations.index(perturbation_id)
                return accept_bit + 2 * (self.n_operators + pert_idx)
            else:
                raise ValueError("AANP requires operator_id or perturbation_id")

        else:
            raise ValueError(f"Unknown action space: {self.name}")

    def decode_action(self, action: int) -> DecodedAction:
        """Decode discrete action index to components."""
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"Invalid action {action} for {self.n_actions} actions")

        if self.name == "AA":
            return DecodedAction(
                accept=bool(action == 1),
                action_type=ActionType.OPERATOR,
                operator_id=None,  # AA doesn't select operator
                perturbation_id=None,
            )

        elif self.name == "AAN":
            accept = bool(action % 2)
            op_idx = action // 2
            return DecodedAction(
                accept=accept,
                action_type=ActionType.OPERATOR,
                operator_id=self.operators[op_idx],
                perturbation_id=None,
            )

        elif self.name == "AANP":
            accept = bool(action % 2)
            item_idx = action // 2

            if item_idx < self.n_operators:
                return DecodedAction(
                    accept=accept,
                    action_type=ActionType.OPERATOR,
                    operator_id=self.operators[item_idx],
                    perturbation_id=None,
                )
            else:
                pert_idx = item_idx - self.n_operators
                return DecodedAction(
                    accept=accept,
                    action_type=ActionType.PERTURBATION,
                    operator_id=None,
                    perturbation_id=self.perturbations[pert_idx],
                )

        else:
            raise ValueError(f"Unknown action space: {self.name}")

    def get_action_name(self, action: int) -> str:
        """Get human-readable action name."""
        decoded = self.decode_action(action)
        accept_str = "Accept" if decoded.accept else "Reject"

        if decoded.action_type == ActionType.OPERATOR:
            if decoded.operator_id is not None:
                from PaST.neurols.operators import OPERATOR_BY_ID

                op = OPERATOR_BY_ID[decoded.operator_id]
                return f"{accept_str}+{op.name}"
            else:
                return accept_str
        else:
            from PaST.neurols.perturbations import PERTURBATION_BY_ID

            pert = PERTURBATION_BY_ID[decoded.perturbation_id]
            return f"{accept_str}+{pert.name}"

    def get_operator_actions(self) -> List[int]:
        """Get list of action indices that correspond to operators."""
        if self.name == "AA":
            return [0, 1]

        actions = []
        for i in range(self.n_operators):
            actions.append(2 * i)  # reject + operator i
            actions.append(2 * i + 1)  # accept + operator i
        return actions

    def get_perturbation_actions(self) -> List[int]:
        """Get list of action indices that correspond to perturbations."""
        if self.name in ("AA", "AAN"):
            return []

        actions = []
        offset = 2 * self.n_operators
        for i in range(self.n_perturbations):
            actions.append(offset + 2 * i)  # reject + perturbation i
            actions.append(offset + 2 * i + 1)  # accept + perturbation i
        return actions


# Standard action spaces following NeuroLS paper

# AA: Acceptance only (2 actions)
# Used with a fixed operator (e.g., 2-opt)
AA_SPACE = ActionSpace(
    name="AA",
    n_actions=2,
    operators=[],
    perturbations=[],
)

# AAN: Acceptance + Operator (8 actions = 2 × 4 operators)
# Model selects which neighborhood to explore
AAN_SPACE = ActionSpace(
    name="AAN",
    n_actions=8,
    operators=[
        OperatorID.RELOCATE_1,
        OperatorID.SWAP_1,
        OperatorID.INTRA_INSERT,
        OperatorID.BLOCK_RELOCATE,
    ],
    perturbations=[],
)

# AANP: Acceptance + Operator/Perturbation (14 actions = 2 × 7)
# Full control over LS behavior
AANP_SPACE = ActionSpace(
    name="AANP",
    n_actions=14,
    operators=[
        OperatorID.RELOCATE_1,
        OperatorID.SWAP_1,
        OperatorID.INTRA_INSERT,
        OperatorID.BLOCK_RELOCATE,
    ],
    perturbations=[
        PerturbationID.NONE,
        PerturbationID.SHAKE_SMALL,
        PerturbationID.RESTART,
    ],
)


def get_action_space(name: str) -> ActionSpace:
    """Get action space by name."""
    name = name.upper()
    if name == "AA":
        return AA_SPACE
    elif name == "AAN":
        return AAN_SPACE
    elif name == "AANP":
        return AANP_SPACE
    else:
        raise ValueError(f"Unknown action space: {name}")

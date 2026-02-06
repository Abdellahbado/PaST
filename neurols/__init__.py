"""NeuroLS: Learned Local Search Controller for Parallel Machine TOU Scheduling.

This package adapts the NeuroLS framework (Falkner et al., 2022) to identical
parallel-machine scheduling with TOU electricity prices and energy minimization.

The controller learns to choose among intervention points:
- Acceptance: whether to accept a non-improving move
- Neighborhood/Operator: which LS operator to apply (Relocate, Swap, Insert, Block)
- Perturbation: when to apply shake/restart to escape local optima

Action spaces follow NeuroLS paper:
- AA: Acceptance only (2 actions)
- AAN: Acceptance × Operator (2 × |Operators| actions)
- AANP: Acceptance × (Operators ∪ Perturbations) (2 × |Ops+Perturbs| actions)

Key design constraints:
- Deterministic: same state + action → same move (fixed enumeration, no RNG at inference)
- DP-backed: optimal timing per machine sequence computed by IncrementalDPSolver
- Reachable: operator set can reach any solution from any other
"""

from PaST.neurols.solution import PMALNSSolution
from PaST.neurols.state import NeuroLSState
from PaST.neurols.action_space import ActionSpace, AA_SPACE, AAN_SPACE, AANP_SPACE
from PaST.neurols.operators import Operator, OPERATORS, OperatorID
from PaST.neurols.perturbations import Perturbation, PERTURBATIONS, PerturbationID
from PaST.neurols.candidate_generator import (
    CandidateGenerator,
    DeterministicMoveSelector,
)
from PaST.neurols.move_evaluator import MoveEvaluator, BatchMoveEvaluator
from PaST.neurols.env import NeuroLSEnv, EnvConfig, VectorizedNeuroLSEnv
from PaST.neurols.train import NeuroLSTrainer, TrainConfig

# Lazy imports for torch-dependent modules
try:
    from PaST.neurols.price_embedding import PriceEmbedding, PriceCNNEncoder
    from PaST.neurols.gnn_encoder import BipartiteGNNEncoder, NeuroLSEncoder
    from PaST.neurols.decoder import NeuroLSDecoder, NeuroLSPolicy, IQNQHead

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    PriceEmbedding = None
    PriceCNNEncoder = None
    BipartiteGNNEncoder = None
    NeuroLSEncoder = None
    NeuroLSDecoder = None
    NeuroLSPolicy = None
    IQNQHead = None

__all__ = [
    # Core components (no torch required)
    "PMALNSSolution",
    "NeuroLSState",
    "ActionSpace",
    "AA_SPACE",
    "AAN_SPACE",
    "AANP_SPACE",
    "Operator",
    "OPERATORS",
    "OperatorID",
    "Perturbation",
    "PERTURBATIONS",
    "PerturbationID",
    "CandidateGenerator",
    "DeterministicMoveSelector",
    "MoveEvaluator",
    "BatchMoveEvaluator",
    "NeuroLSEnv",
    "EnvConfig",
    "VectorizedNeuroLSEnv",
    "NeuroLSTrainer",
    "TrainConfig",
    # Torch-dependent (may be None if torch unavailable)
    "PriceEmbedding",
    "PriceCNNEncoder",
    "BipartiteGNNEncoder",
    "NeuroLSEncoder",
    "NeuroLSDecoder",
    "NeuroLSPolicy",
    "IQNQHead",
]

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

from sandbox.neurols.solution import PMALNSSolution
from sandbox.neurols.state import NeuroLSState
from sandbox.neurols.action_space import (
    ActionSpace,
    AA_SPACE,
    AAN_SPACE,
    AANP_SPACE,
    AANPD_SPACE,
    get_action_space,
)
from sandbox.neurols.operators import Operator, OPERATORS, OperatorID
from sandbox.neurols.perturbations import (
    Perturbation,
    PERTURBATIONS,
    PerturbationID,
    DestroyOperator,
    DESTROY_OPERATORS,
    DestroyID,
)
from sandbox.neurols.candidate_generator import (
    CandidateGenerator,
    DeterministicMoveSelector,
)
from sandbox.neurols.move_evaluator import MoveEvaluator, BatchMoveEvaluator
from sandbox.neurols.env import NeuroLSEnv, EnvConfig, VectorizedNeuroLSEnv
from sandbox.neurols.train import NeuroLSTrainer, TrainConfig

# Lazy imports for torch-dependent modules
try:
    from sandbox.neurols.price_embedding import PriceEmbedding, PriceCNNEncoder
    from sandbox.neurols.gnn_encoder import (
        BipartiteGNNEncoder,
        NeuroLSEncoder,
        TripartiteGNNEncoder,
        TripartiteNeuroLSEncoder,
    )
    from sandbox.neurols.decoder import NeuroLSDecoder, NeuroLSPolicy, IQNQHead

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    PriceEmbedding = None
    PriceCNNEncoder = None
    BipartiteGNNEncoder = None
    NeuroLSEncoder = None
    TripartiteGNNEncoder = None
    TripartiteNeuroLSEncoder = None
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

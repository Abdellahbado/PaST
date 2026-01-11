"""
PaST-SM: Period-aware Scheduler Transformer - Single Machine

This package implements an RL-based single-machine scheduler for the
parallel machine scheduling problem with energy consideration under
time-of-use (TOU) electricity pricing.

The agent decides:
1. The ORDER of jobs to schedule
2. WHEN to insert idle time (slack) before running a chosen job

The agent does NOT assign jobs to machines (done by meta-heuristic outside).

Optimization mode: Energy minimization with hard deadline constraint (epsilon-constraint).

Supports 6 ablation variants:
- ppo_short_base: PPO + Short slack + Local periods (K=48) + Base model
- ppo_short_large: PPO + Short slack + Local periods (K=150) + Large model
- ppo_c2f: PPO + Coarse-to-fine slack + Local periods (K=48) + Base model
- ppo_full_tokens: PPO + Full slack + Full periods (K=250) + Base model (no pooling)
- ppo_full_global: PPO + Full slack + Full periods (K=250) + Base model + Global pooling
- reinforce_short_sc: REINFORCE + Short slack + Local periods (K=48) + Base model + Self-critic
"""

__version__ = "2.0.0"

# =============================================================================
# Configuration
# =============================================================================

from .config import (
    # Enums
    SlackType,
    RLAlgorithm,
    ModelSize,
    VariantID,
    # Slack specifications
    ShortSlackSpec,
    CoarseToFineSlackSpec,
    FullSlackSpec,
    # Config classes
    ModelConfig,
    EnvConfig,
    DataConfig,
    TrainingConfig,
    VariantConfig,
    # Variant factories
    get_variant_config,
    list_variants,
    get_ppo_short_base,
    get_ppo_short_large,
    get_ppo_c2f,
    get_ppo_full_tokens,
    get_ppo_full_global,
    get_reinforce_short_sc,
    VARIANT_FACTORIES,
)

# =============================================================================
# Data Generation
# =============================================================================

from .sm_benchmark_data import (
    generate_raw_instance,
    make_single_machine_episode,
    generate_episode_batch,
    generate_episode_batch_variable,
    generate_single_machine_episode,
    sample_intervals_sum_to_T,
    expand_ck_to_ct,
    compute_period_start_slots,
    RawInstance,
    SingleMachineEpisode,
)

# =============================================================================
# Environment
# =============================================================================

from .sm_env import (
    SingleMachinePeriodEnv,
    GPUBatchSingleMachinePeriodEnv,
    slack_to_start_time,
    find_period_at_time,
    ENV_VERSION,
)

# =============================================================================
# Model
# =============================================================================

from .past_sm_model import (
    # Main model
    PaSTSMNet,
    build_model,
    # Components
    PaSTEncoder,
    FactoredActionHead,
    SimpleActionHead,
    ValueHead,
    # Transformer blocks
    PreLNBlock,
    PostLNBlock,
    MultiHeadAttention,
    FeedForward,
    make_transformer_block,
    # Embeddings
    JobEmbedding,
    PeriodEmbedding,
    ContextEmbedding,
    GlobalHorizonEmbedding,
)

# =============================================================================
# PPO Training
# =============================================================================

from .ppo_runner import (
    PPOConfig,
    RolloutBuffer,
    PPORunner,
)

from .eval import (
    EvalResult,
    Evaluator,
    compare_variants,
)

from .train_ppo import (
    RunConfig,
    MetricsLogger,
    CheckpointManager,
    TrainingEnv,
    train,
    get_p100_smoke_config,
    get_a100_full_config,
    set_seed,
    TRAIN_VERSION,
)

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Config enums
    "SlackType",
    "RLAlgorithm",
    "ModelSize",
    "VariantID",
    # Slack specs
    "ShortSlackSpec",
    "CoarseToFineSlackSpec",
    "FullSlackSpec",
    # Config classes
    "ModelConfig",
    "EnvConfig",
    "DataConfig",
    "TrainingConfig",
    "VariantConfig",
    # Variant factories
    "get_variant_config",
    "list_variants",
    "get_ppo_short_base",
    "get_ppo_short_large",
    "get_ppo_c2f",
    "get_ppo_full_tokens",
    "get_ppo_full_global",
    "get_reinforce_short_sc",
    "VARIANT_FACTORIES",
    # Data
    "generate_raw_instance",
    "make_single_machine_episode",
    "generate_episode_batch",
    "generate_episode_batch_variable",
    "generate_single_machine_episode",
    "sample_intervals_sum_to_T",
    "expand_ck_to_ct",
    "compute_period_start_slots",
    "RawInstance",
    "SingleMachineEpisode",
    # Environment
    "SingleMachinePeriodEnv",
    "GPUBatchSingleMachinePeriodEnv",
    "slack_to_start_time",
    "find_period_at_time",
    "ENV_VERSION",
    # Model
    "PaSTSMNet",
    "build_model",
    "PaSTEncoder",
    "FactoredActionHead",
    "SimpleActionHead",
    "ValueHead",
    "PreLNBlock",
    "PostLNBlock",
    "MultiHeadAttention",
    "FeedForward",
    "make_transformer_block",
    "JobEmbedding",
    "PeriodEmbedding",
    "ContextEmbedding",
    "GlobalHorizonEmbedding",
    # PPO Training
    "PPOConfig",
    "RolloutBuffer",
    "PPORunner",
    "EvalResult",
    "Evaluator",
    "compare_variants",
    "RunConfig",
    "MetricsLogger",
    "CheckpointManager",
    "TrainingEnv",
    "train",
    "get_p100_smoke_config",
    "get_a100_full_config",
    "set_seed",
    "TRAIN_VERSION",
]

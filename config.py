"""
Configuration for PaST-SM (Period-aware Scheduler Transformer - Single Machine).

Supports 6 ablation variants:
- ppo_short_base: PPO + Short slack + Local periods (K=48) + Base model
- ppo_short_large: PPO + Short slack + Local periods (K=150) + Large model
- ppo_c2f: PPO + Coarse-to-fine slack + Local periods (K=48) + Base model
- ppo_full_tokens: PPO + Full slack + Full periods (K=250) + Base model (no pooling)
- ppo_full_global: PPO + Full slack + Full periods (K=250) + Base model + Global pooling
- reinforce_short_sc: REINFORCE + Short slack + Local periods (K=48) + Base model + Self-critic
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple


# =============================================================================
# Enums for variant specification
# =============================================================================


class SlackType(Enum):
    """Slack action space type (new naming for 6-variant ablation)."""

    SHORT = "short"  # Fixed discrete slack options (K_short)
    COARSE_TO_FINE = "c2f"  # Two-stage: coarse period bucket + fine offset
    FULL = "full"  # Full period-aligned slack (up to K_full_max periods)


# Legacy SlackVariant enum for backward compatibility with sm_env.py
class SlackVariant(Enum):
    """Legacy slack variant enum (for backward compatibility)."""

    NO_SLACK = "no_slack"
    SHORT_SLACK = "short_slack"
    PERIOD_ALIGNED = "period_aligned"
    COARSE_TO_FINE = "coarse_to_fine"
    FULL_SLACK = "full_slack"
    LEARNED_SLACK = "learned_slack"


class RLAlgorithm(Enum):
    """RL training algorithm."""

    PPO = "ppo"
    REINFORCE = "reinforce"


class ModelSize(Enum):
    """Model capacity preset."""

    BASE = "base"
    LARGE = "large"


class VariantID(Enum):
    """Ablation variant identifiers."""

    PPO_SHORT_BASE = "ppo_short_base"
    PPO_SHORT_LARGE = "ppo_short_large"
    PPO_C2F = "ppo_c2f"
    PPO_FULL_TOKENS = "ppo_full_tokens"
    PPO_FULL_GLOBAL = "ppo_full_global"
    REINFORCE_SHORT_SC = "reinforce_short_sc"
    PPO_FAMILY_Q4 = "ppo_family_q4"  # Job × price-family (4 quartiles)
    PPO_FAMILY_Q4_CTX13 = (
        "ppo_family_q4_ctx13"  # Price-family + identified ctx (q's + next-slot deltas)
    )
    PPO_FAMILY_Q4_BESTSTART = (
        "ppo_family_q4_beststart"  # Price-family + best feasible start within family
    )
    PPO_FAMILY_Q4_CTX13_BESTSTART = (
        "ppo_family_q4_ctx13_beststart"  # Price-family + ctx13 + best feasible start
    )
    PPO_FAMILY_Q4_CTX18_BESTSTART = "ppo_family_q4_ctx18_beststart"  # Price-family + ctx13 + capacity ctx + best feasible start
    PPO_SEQUENCE = "ppo_sequence"  # RL sequences jobs, Batch DP times them
    PPO_DURATION_AWARE_FAMILY = "ppo_duration_aware_family"  # Duration-aware families: families based on avg window cost
    PPO_DURATION_AWARE_FAMILY_CTX13 = "ppo_duration_aware_family_ctx13"  # Duration-aware families + identified ctx (q's + next-start deltas)
    PPO_FAMILY_Q4_BESTSTART_CNN = "ppo_family_q4_beststart_cnn"  # family_beststart with lightweight CNN+DeepSets backbone
    PPO_FAMILY_Q4_CTX13_BESTSTART_CNN = "ppo_family_q4_ctx13_beststart_cnn"  # family_ctx13_beststart with lightweight CNN+DeepSets backbone
    PPO_DURATION_AWARE_FAMILY_CNN = "ppo_duration_aware_family_cnn"  # duration-aware family with lightweight CNN+DeepSets backbone
    PPO_DURATION_AWARE_FAMILY_CTX13_CNN = "ppo_duration_aware_family_ctx13_cnn"  # duration-aware family ctx13 with lightweight CNN+DeepSets backbone
    Q_SEQUENCE = "q_sequence"  # Q-learning for sequences: supervised on DP costs
    Q_SEQUENCE_CTX13 = "q_sequence_ctx13"  # Q-sequence + identified ctx features
    Q_SEQUENCE_CNN = (
        "q_sequence_cnn"  # Q-sequence with lightweight CNN(periods)+DeepSets(jobs)
    )
    Q_SEQUENCE_CNN_CTX13 = (
        "q_sequence_cnn_ctx13"  # Same as Q_SEQUENCE_CNN with ctx13 enabled
    )


# =============================================================================
# Slack Specifications
# =============================================================================


@dataclass
class ShortSlackSpec:
    """
    Specification for SHORT slack variant.

    Defines K_short discrete slack options as period offsets.
    Example: [0, 1, 2, 3, 5, 8, 13, 21] means slack can delay job start
    by 0, 1, 2, 3, 5, 8, 13, or 21 periods from current time.
    """

    slack_options: List[int] = None  # Period offsets, must include 0

    def __post_init__(self):
        if self.slack_options is None:
            # Default: dense short-range offsets.
            # We keep all integers to avoid "missed optimal offset" edge cases,
            # while the policy can still be biased toward small offsets via a
            # learned/external prior.
            self.slack_options = list(range(31))  # 0..30
        if not all(isinstance(s, int) and s >= 0 for s in self.slack_options):
            raise ValueError("All slack_options must be non-negative integers")
        if 0 not in self.slack_options:
            raise ValueError("slack_options must include 0 (no delay option)")
        # Sort for consistency
        self.slack_options = sorted(list(set(self.slack_options)))

    @property
    def K_short(self) -> int:
        """Number of discrete slack choices."""
        return len(self.slack_options)

    def slack_id_to_period_offset(self, slack_id: int) -> int:
        """Convert slack_id to period offset."""
        return self.slack_options[slack_id]


@dataclass
class CoarseToFineSlackSpec:
    """
    Specification for COARSE_TO_FINE slack variant.

    Two-stage selection:
    1. Coarse: Select which "bucket" of periods (e.g., periods 0-4, 5-9, 10-19, ...)
    2. Fine: Select exact period within that bucket

    Total slack choices = num_coarse_buckets * fine_resolution
    """

    # Coarse buckets: list of (start_period, end_period) inclusive
    # Example: [(0, 4), (5, 9), (10, 19), (20, 34), (35, 54)]
    coarse_buckets: List[Tuple[int, int]] = None

    # Fine resolution within each bucket
    # If bucket spans 5 periods and fine_resolution=5, we get exact period selection
    # If bucket spans 10 periods and fine_resolution=5, we get every 2nd period
    fine_resolution: int = 5

    def __post_init__(self):
        if self.coarse_buckets is None:
            # Default: exponentially growing buckets
            self.coarse_buckets = [
                (0, 4),  # 5 periods
                (5, 9),  # 5 periods
                (10, 19),  # 10 periods
                (20, 34),  # 15 periods
                (35, 54),  # 20 periods
            ]

    @property
    def num_coarse_buckets(self) -> int:
        return len(self.coarse_buckets)

    @property
    def K_c2f(self) -> int:
        """Total number of (coarse, fine) combinations."""
        return self.num_coarse_buckets * self.fine_resolution

    def decode_slack_id(self, slack_id: int) -> Tuple[int, int]:
        """Decode slack_id to (coarse_bucket_idx, fine_idx)."""
        coarse_idx = slack_id // self.fine_resolution
        fine_idx = slack_id % self.fine_resolution
        return coarse_idx, fine_idx

    def slack_id_to_period_offset(self, slack_id: int) -> int:
        """Convert slack_id to target period offset."""
        coarse_idx, fine_idx = self.decode_slack_id(slack_id)
        if coarse_idx >= len(self.coarse_buckets):
            return 0  # Fallback

        start, end = self.coarse_buckets[coarse_idx]
        bucket_size = end - start + 1

        # Map fine_idx to position within bucket
        if self.fine_resolution >= bucket_size:
            # Direct mapping
            period_in_bucket = min(fine_idx, bucket_size - 1)
        else:
            # Strided mapping
            stride = bucket_size / self.fine_resolution
            period_in_bucket = int(fine_idx * stride)

        return start + period_in_bucket


@dataclass
class FullSlackSpec:
    """
    Specification for FULL slack variant.

    Agent can choose to start at any of the next K_full_max periods.
    """

    K_full_max: int = 250  # Maximum period offset (covers T_max=500 / min_period=2)

    @property
    def K_full(self) -> int:
        return self.K_full_max


# Legacy spec for backward compatibility with sm_env.py
@dataclass
class PeriodAlignedSlackSpec:
    """
    Legacy specification for period-aligned slack (backward compatibility).

    Replaced by FullSlackSpec in new config system.
    """

    max_periods_lookahead: int = 48

    @property
    def num_choices(self) -> int:
        return self.max_periods_lookahead + 1  # +1 for "start now" option


# =============================================================================
# Model Configuration
# =============================================================================


@dataclass
class ModelConfig:
    """
    Model architecture configuration.

    Supports Base and Large presets with configurable overrides.
    """

    # Transformer dimensions
    d_model: int = 128
    num_heads: int = 8
    num_blocks: int = 2
    d_ff: int = 512  # Feed-forward hidden dim (if using FF layers)
    dropout: float = 0.0

    # Backbone selection
    # - "transformer": original PaST cross-attn/self-attn encoder
    # - "cnn_deepsets": lightweight encoder (CNN over periods + DeepSets over jobs)
    backbone: str = "transformer"

    # Lightweight backbone hyperparameters (used when backbone == "cnn_deepsets")
    period_cnn_channels: int = 128
    period_cnn_layers: int = 2
    period_cnn_kernel_size: int = 3
    deepsets_hidden: int = 256

    # Transformer variant
    use_pre_ln: bool = True  # Pre-LayerNorm (more stable for RL)

    # Input embedding dimensions (all project to d_model)
    job_input_dim: int = 2  # [processing_time, remaining_count] per bin
    period_input_dim: int = 4  # [duration, price, start_offset, is_current]
    ctx_input_dim: int = (
        6  # [t, T_limit, remaining_work, e_single, avg_price_beyond, min_price_beyond]
    )

    # Action head configuration
    use_factored_head: bool = True  # job_logits + slack_logits -> joint
    slack_head_hidden: int = 64  # Hidden dim for slack MLP

    # Optional exponential prior over SHORT slack offsets.
    # If enabled, the model adds a fixed bias to slack logits:
    #   bias(s) = -strength * (offset(s) / tau)
    # where offset(s) is the period offset value from ShortSlackSpec.
    # Set tau to something like 10-20 for a mild bias.
    short_slack_prior_tau: Optional[float] = None
    short_slack_prior_strength: float = 1.0

    # Global horizon embedding (only for FULL_GLOBAL variant)
    use_global_horizon: bool = False  # Add duration-weighted pooling of full periods

    @staticmethod
    def base() -> "ModelConfig":
        """Base model preset."""
        return ModelConfig(
            d_model=128,
            num_heads=8,
            num_blocks=2,
            d_ff=512,
            dropout=0.0,
            use_pre_ln=True,
        )

    @staticmethod
    def large() -> "ModelConfig":
        """Large model preset (more capacity for ablation)."""
        return ModelConfig(
            d_model=256,
            num_heads=8,
            num_blocks=3,
            d_ff=1024,
            dropout=0.1,
            use_pre_ln=True,
        )


# =============================================================================
# Environment Configuration
# =============================================================================


@dataclass
class EnvConfig:
    """Environment and observation space configuration."""

    # Job bins
    M_job_bins: int = 50  # Number of job processing-time bins

    # Period horizon configuration (CONFIGURABLE)
    K_period_local: int = (
        48  # Local lookahead window (default 48, can be 150 for Large)
    )
    K_period_full_max: int = (
        250  # Maximum periods in full horizon (T_max=500 / min_period=2)
    )

    # Feature dimensions
    F_job: int = 2  # [processing_time, count] per bin
    F_period: int = 4  # [duration, price, start_offset, is_current]
    F_ctx: int = (
        6  # [t, T_limit, remaining_work, e_single, avg_price_beyond, min_price_beyond]
    )

    # NEW: Slack configuration for 6-variant ablation
    slack_type: SlackType = SlackType.SHORT
    short_slack_spec: Optional[ShortSlackSpec] = None
    c2f_slack_spec: Optional[CoarseToFineSlackSpec] = None
    full_slack_spec: Optional[FullSlackSpec] = None

    # Whether to include full period tokens (for FULL variants)
    use_periods_full: bool = False

    # Price-family variant: predict (job, family) instead of (job, slack)
    # Families are computed per-episode from price quantiles
    use_price_families: bool = False
    num_price_families: int = 4  # Number of price families (quartiles by default)

    # Price-family decoding: pick the best feasible start within chosen family
    # (minimizes window energy cost) instead of the earliest slot.
    use_best_family_start: bool = False

    # Optional BEST-START tie-breaker / regularizer.
    # Adds (best_family_start_wait_penalty * wait_slots) to the window energy objective.
    # This discourages jumping far into the future just to hit a cheap slot.
    best_family_start_wait_penalty: float = 0.0

    # Optional BEST-START tie-breaker: among near-optimal energy starts, pick the earliest.
    # If > 0, we find the minimum window energy cost and then select the earliest start
    # whose cost is within (min_cost + best_family_start_energy_tolerance).
    # This is scale-sensitive (same units as window_cost = sum(ct) over job duration).
    best_family_start_energy_tolerance: float = 0.0

    # Optional hard cap on waiting for price-family variants.
    # If set (>=0), an action (job,family) is valid only if there exists a feasible
    # start in that family with (start - t_now) <= max_wait_slots, and BEST-START
    # will only search within that wait window.
    # Default None keeps historical behavior.
    best_family_start_max_wait_slots: Optional[int] = None

    # Duration-aware families: families based on average window cost w(s,p)/p
    # instead of slot price. This makes "family 0 = cheap" mean full-job cost.
    use_duration_aware_families: bool = False

    # LEGACY: Fields for backward compatibility with sm_env.py
    slack_variant: SlackVariant = SlackVariant.SHORT_SLACK
    period_aligned_spec: Optional[PeriodAlignedSlackSpec] = None

    def __post_init__(self):
        # Initialize default specs if not provided
        if self.short_slack_spec is None:
            self.short_slack_spec = ShortSlackSpec()
        if self.c2f_slack_spec is None:
            self.c2f_slack_spec = CoarseToFineSlackSpec()
        if self.full_slack_spec is None:
            self.full_slack_spec = FullSlackSpec(K_full_max=self.K_period_full_max)

    def get_num_slack_choices(self) -> int:
        """Get K_slack based on slack type (or num_price_families if using price families)."""
        # Price-family variant overrides slack choices
        if self.use_price_families:
            return self.num_price_families

        if self.slack_type == SlackType.SHORT:
            return self.short_slack_spec.K_short
        elif self.slack_type == SlackType.COARSE_TO_FINE:
            return self.c2f_slack_spec.K_c2f
        elif self.slack_type == SlackType.FULL:
            return self.full_slack_spec.K_full
        else:
            raise ValueError(f"Unknown slack type: {self.slack_type}")

    @property
    def action_dim(self) -> int:
        """Total action space size = M_job_bins * K_slack."""
        return self.M_job_bins * self.get_num_slack_choices()

    # ---------------------------------------------------------------------
    # Backward-compatible aliases (sm_env.py expects these names)
    # ---------------------------------------------------------------------

    @property
    def N_job_pad(self) -> int:
        """Legacy alias for M_job_bins."""
        return self.M_job_bins

    @N_job_pad.setter
    def N_job_pad(self, value: int):
        self.M_job_bins = int(value)

    @property
    def K_period_lookahead(self) -> int:
        """Legacy alias for K_period_local."""
        return self.K_period_local

    @K_period_lookahead.setter
    def K_period_lookahead(self, value: int):
        self.K_period_local = int(value)


# =============================================================================
# Data Configuration
# =============================================================================


@dataclass
class DataConfig:
    """Data generation configuration."""

    # Sampling mode:
    # - "uniform_range": sample (m, n) uniformly from [min,max] ranges and
    #   sample p/e/ck from fixed ranges.
    # - "paper_grid_90": match the 90-instance benchmark distribution used by
    #   New Benchmark/generate_data.py (Wang2018 small+mls, Anghinolfi2021 vls).
    sampling_mode: str = "paper_grid_90"

    # Horizon choices for training
    T_max_choices: List[int] = field(
        default_factory=lambda: [50, 80, 100, 300, 350, 500]
    )

    # Period duration choices (always 2, 3, or 5 as per benchmark)
    Tk_choices: Tuple[int, ...] = (2, 3, 5)

    # Price range for periods
    ck_min: int = 1
    ck_max: int = 4

    # Job processing time range
    p_min: int = 1
    p_max: int = 4

    # Machine energy rate range
    e_min: int = 1
    e_max: int = 3

    # Whether to randomize machine energy rate
    randomize_machine_rate: bool = True

    # Number of machines/jobs for simulating meta-heuristic split
    m_min: int = 3
    m_max: int = 7
    n_min: int = 6
    n_max: int = 25

    # Epsilon-constraint deadline slack
    deadline_slack_ratio_min: float = 0.0
    deadline_slack_ratio_max: float = 0.5


# =============================================================================
# Training Configuration (structure only, not used yet)
# =============================================================================


@dataclass
class TrainingConfig:
    """Training hyperparameters (placeholder for future use)."""

    # RL algorithm
    algorithm: RLAlgorithm = RLAlgorithm.PPO

    # For REINFORCE
    use_self_critic: bool = False  # Self-critic baseline (greedy rollout)

    # Common parameters
    total_env_steps: int = 10_000_000  # Same budget across variants
    batch_size: int = 2048

    # PPO-specific
    ppo_epochs: int = 4
    ppo_clip: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # Learning rate
    lr: float = 3e-4
    lr_schedule: str = "linear"  # "linear", "cosine", "constant"

    # Checkpointing
    save_every_steps: int = 100_000
    eval_every_steps: int = 50_000
    num_eval_episodes: int = 100


# =============================================================================
# Variant Configuration (combines all settings for one ablation)
# =============================================================================


@dataclass
class VariantConfig:
    """
    Complete configuration for one ablation variant.

    Combines model, env, and training configs with variant-specific flags.
    """

    variant_id: VariantID
    model: ModelConfig = field(default_factory=ModelConfig.base)
    env: EnvConfig = field(default_factory=EnvConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Seed
    seed: int = 42
    device: str = "cuda"

    def validate(self) -> bool:
        """Validate configuration consistency."""
        # Check slack type matches spec
        _ = self.env.get_num_slack_choices()

        # Check global horizon flag consistency
        if self.model.use_global_horizon and not self.env.use_periods_full:
            raise ValueError("use_global_horizon=True requires use_periods_full=True")

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "variant_id": self.variant_id.value,
            "model": {
                "d_model": self.model.d_model,
                "num_heads": self.model.num_heads,
                "num_blocks": self.model.num_blocks,
                "d_ff": self.model.d_ff,
                "use_pre_ln": self.model.use_pre_ln,
                "use_global_horizon": self.model.use_global_horizon,
                "backbone": getattr(self.model, "backbone", "transformer"),
            },
            "env": {
                "M_job_bins": self.env.M_job_bins,
                "K_period_local": self.env.K_period_local,
                "K_period_full_max": self.env.K_period_full_max,
                "slack_type": self.env.slack_type.value,
                "use_periods_full": self.env.use_periods_full,
                "use_price_families": self.env.use_price_families,
                "num_price_families": self.env.num_price_families,
                "use_best_family_start": self.env.use_best_family_start,
                "best_family_start_wait_penalty": getattr(
                    self.env, "best_family_start_wait_penalty", 0.0
                ),
                "best_family_start_energy_tolerance": getattr(
                    self.env, "best_family_start_energy_tolerance", 0.0
                ),
                "K_slack": self.env.get_num_slack_choices(),
                "action_dim": self.env.action_dim,
            },
            "training": {
                "algorithm": self.training.algorithm.value,
                "use_self_critic": self.training.use_self_critic,
            },
            "data": {
                "sampling_mode": self.data.sampling_mode,
                "T_max_choices": list(self.data.T_max_choices),
                "Tk_choices": list(self.data.Tk_choices),
                "ck_min": int(self.data.ck_min),
                "ck_max": int(self.data.ck_max),
                "p_min": int(self.data.p_min),
                "p_max": int(self.data.p_max),
                "e_min": int(self.data.e_min),
                "e_max": int(self.data.e_max),
                "randomize_machine_rate": bool(self.data.randomize_machine_rate),
                "m_min": int(self.data.m_min),
                "m_max": int(self.data.m_max),
                "n_min": int(self.data.n_min),
                "n_max": int(self.data.n_max),
                "deadline_slack_ratio_min": float(self.data.deadline_slack_ratio_min),
                "deadline_slack_ratio_max": float(self.data.deadline_slack_ratio_max),
            },
            "seed": self.seed,
        }


# =============================================================================
# Variant Presets (the 6 ablation configurations)
# =============================================================================


def get_ppo_short_base() -> VariantConfig:
    """
    PPO + Short slack + Local periods (K=48) + Base model.

    This is the DEFAULT variant.
    """
    model = ModelConfig.base()
    model.use_global_horizon = False
    # Mild exponential bias toward small SHORT offsets (can still choose 21, 27, etc.).
    model.short_slack_prior_tau = 12.0
    model.short_slack_prior_strength = 1.0

    env = EnvConfig()
    env.K_period_local = 48
    env.slack_type = SlackType.SHORT
    env.use_periods_full = False

    training = TrainingConfig()
    training.algorithm = RLAlgorithm.PPO

    return VariantConfig(
        variant_id=VariantID.PPO_SHORT_BASE,
        model=model,
        env=env,
        training=training,
    )


def get_ppo_short_large() -> VariantConfig:
    """
    PPO + Short slack + Local periods (K=150) + Large model.

    Capacity ablation: more model capacity + larger local window.
    """
    model = ModelConfig.large()
    model.use_global_horizon = False
    model.short_slack_prior_tau = 12.0
    model.short_slack_prior_strength = 1.0

    env = EnvConfig()
    env.K_period_local = 150  # Larger local window
    env.slack_type = SlackType.SHORT
    env.use_periods_full = False

    training = TrainingConfig()
    training.algorithm = RLAlgorithm.PPO

    return VariantConfig(
        variant_id=VariantID.PPO_SHORT_LARGE,
        model=model,
        env=env,
        training=training,
    )


def get_ppo_c2f() -> VariantConfig:
    """
    PPO + Coarse-to-fine slack + Local periods (K=48) + Base model.

    Tests whether structured slack discretization helps.
    """
    model = ModelConfig.base()
    model.use_global_horizon = False

    env = EnvConfig()
    env.K_period_local = 48
    env.slack_type = SlackType.COARSE_TO_FINE
    env.use_periods_full = False

    training = TrainingConfig()
    training.algorithm = RLAlgorithm.PPO

    return VariantConfig(
        variant_id=VariantID.PPO_C2F,
        model=model,
        env=env,
        training=training,
    )


def get_ppo_full_tokens() -> VariantConfig:
    """
    PPO + Full slack (K=250) + Full period tokens + Base model (NO global pooling).

    Full horizon visibility and full action flexibility, but no pooled summary.
    """
    model = ModelConfig.base()
    model.use_global_horizon = False  # Key difference from ppo_full_global

    env = EnvConfig()
    env.K_period_local = env.K_period_full_max  # See all periods
    env.slack_type = SlackType.FULL
    env.use_periods_full = True

    training = TrainingConfig()
    training.algorithm = RLAlgorithm.PPO

    return VariantConfig(
        variant_id=VariantID.PPO_FULL_TOKENS,
        model=model,
        env=env,
        training=training,
    )


def get_ppo_full_global() -> VariantConfig:
    """
    PPO + Full slack (K=250) + Full period tokens + Base model + Global pooling.

    Same as ppo_full_tokens but WITH duration-weighted global horizon embedding.
    Tests: "given full-horizon token access, does a compressed global summary help?"
    """
    model = ModelConfig.base()
    model.use_global_horizon = True  # Key difference from ppo_full_tokens

    env = EnvConfig()
    env.K_period_local = env.K_period_full_max  # See all periods
    env.slack_type = SlackType.FULL
    env.use_periods_full = True

    training = TrainingConfig()
    training.algorithm = RLAlgorithm.PPO

    return VariantConfig(
        variant_id=VariantID.PPO_FULL_GLOBAL,
        model=model,
        env=env,
        training=training,
    )


def get_reinforce_short_sc() -> VariantConfig:
    """
    REINFORCE + Short slack + Local periods (K=48) + Base model + Self-critic.

    Baseline algorithm comparison: REINFORCE with self-critic (greedy) baseline.
    """
    model = ModelConfig.base()
    model.use_global_horizon = False
    model.short_slack_prior_tau = 12.0
    model.short_slack_prior_strength = 1.0

    env = EnvConfig()
    env.K_period_local = 48
    env.slack_type = SlackType.SHORT
    env.use_periods_full = False

    training = TrainingConfig()
    training.algorithm = RLAlgorithm.REINFORCE
    training.use_self_critic = True

    return VariantConfig(
        variant_id=VariantID.REINFORCE_SHORT_SC,
        model=model,
        env=env,
        training=training,
    )


def get_ppo_family_q4() -> VariantConfig:
    """
    PPO + Price-family (4 quartiles) + Local periods (K=48) + Base model.

    Action = (job_id, family_id) where family_id ∈ {0,1,2,3} based on
    per-episode price quantiles. Decoding: earliest feasible start time
    whose starting slot price belongs to the chosen family.

    This variant is agnostic to interval structure and focuses on price levels.
    """
    model = ModelConfig.base()
    model.use_global_horizon = False
    # No slack prior needed - we're predicting price families, not offsets

    env = EnvConfig()
    env.K_period_local = 48
    env.slack_type = SlackType.SHORT  # Keep SHORT for shape compatibility
    # Override slack choices to use price families
    env.use_price_families = True
    env.num_price_families = 4  # 4 quartiles
    env.use_periods_full = False

    training = TrainingConfig()
    training.algorithm = RLAlgorithm.PPO

    return VariantConfig(
        variant_id=VariantID.PPO_FAMILY_Q4,
        model=model,
        env=env,
        training=training,
    )


def get_ppo_family_q4_beststart() -> VariantConfig:
    """
    PPO + Price-family (4 quartiles) + Best-start decoding + Local periods (K=48).

    Same action space as `ppo_family_q4`, but decoding selects the feasible start
    within the chosen family that minimizes window energy cost for the job.
    """
    cfg = get_ppo_family_q4()
    cfg.variant_id = VariantID.PPO_FAMILY_Q4_BESTSTART
    cfg.env.use_best_family_start = True
    return cfg


def get_ppo_family_q4_beststart_cnn() -> VariantConfig:
    """Price-family + best-start decoding, using CNN+DeepSets backbone."""
    cfg = get_ppo_family_q4_beststart()
    cfg.variant_id = VariantID.PPO_FAMILY_Q4_BESTSTART_CNN
    cfg.model.backbone = "cnn_deepsets"
    return cfg


def get_ppo_family_q4_ctx13() -> VariantConfig:
    """Enhanced PPO family variant with identified family semantics in ctx.

    Extends ctx from 6 -> 13 by appending:
        - price quantiles [q25, q50, q75]
        - delta_to_next_slot_in_family[0..3]

    This keeps the same action space as `ppo_family_q4` (job × family), but makes
    family IDs identifiable per episode and improves stability/search guidance.
    """
    cfg = get_ppo_family_q4()
    cfg.variant_id = VariantID.PPO_FAMILY_Q4_CTX13

    # Observation dims
    cfg.env.F_ctx = 13
    # Keep ModelConfig consistent (even though the model uses env.F_ctx).
    cfg.model.ctx_input_dim = 13

    return cfg


def get_ppo_family_q4_ctx13_beststart() -> VariantConfig:
    """Price-family + ctx13 + best-start decoding within chosen family."""
    cfg = get_ppo_family_q4_ctx13()
    cfg.variant_id = VariantID.PPO_FAMILY_Q4_CTX13_BESTSTART
    cfg.env.use_best_family_start = True
    return cfg


def get_ppo_family_q4_ctx13_beststart_cnn() -> VariantConfig:
    """Price-family + ctx13 + best-start decoding, using CNN+DeepSets backbone."""
    cfg = get_ppo_family_q4_ctx13_beststart()
    cfg.variant_id = VariantID.PPO_FAMILY_Q4_CTX13_BESTSTART_CNN
    cfg.model.backbone = "cnn_deepsets"
    return cfg


def get_ppo_family_q4_ctx18_beststart() -> VariantConfig:
    """Price-family + ctx18 (adds family capacity signals) + best-start decoding.

    Extends ctx13 by appending:
        - slots_remaining_in_family[0..3] in [t, min(T_limit, T_max)) (4)
        - cheap_capacity_deficit = (remaining_work - slots_remaining_family0)/(remaining_work+1) (1)

    Total ctx dims: 18.
    """
    cfg = get_ppo_family_q4_ctx13_beststart()
    cfg.variant_id = VariantID.PPO_FAMILY_Q4_CTX18_BESTSTART

    cfg.env.F_ctx = 18
    cfg.model.ctx_input_dim = 18

    # Leave this at 0 by default to keep decoder semantics unchanged unless you tune it.
    # A small value (e.g. 0.05) can reduce pathological "wait for cheap" behavior.
    cfg.env.best_family_start_wait_penalty = 0.0

    return cfg


def get_ppo_sequence() -> VariantConfig:
    """
    PPO + Sequence-Only (RL) + Batch DP (Time).

    Action Space: N_jobs (Discrete).
    - We achieve this by setting SlackType.SHORT with a single option [0].
    - The environment wrapper will interpret action `j` as "append job `j` to sequence".

    Model:
    - Base model.
    - No specialized slack inputs needed.

    Env:
    - K_period_local: irrelevant for timing (DP uses full), but useful for transformer observations.
      Keep it reasonable (e.g. 48 or 50) so the agent sees "local price context".
    """
    model = ModelConfig.base()
    # Ensure action dim is N_jobs * 1
    # We must set this implicitly via the env config having 1 slack choice.

    env = EnvConfig()
    env.slack_type = SlackType.SHORT
    env.short_slack_spec = ShortSlackSpec(slack_options=[0])  # Only 1 choice => dim = N
    env.use_periods_full = False
    env.K_period_local = 48  # Provide some local price context

    training = TrainingConfig()
    training.algorithm = RLAlgorithm.PPO

    return VariantConfig(
        variant_id=VariantID.PPO_SEQUENCE,
        model=model,
        env=env,
        training=training,
    )


def get_ppo_duration_aware_family() -> VariantConfig:
    """
    PPO + Duration-aware families (4 quartiles) + Local periods (K=48) + Base model.

    This is an improved version of the price-family variant. Instead of assigning
    families based on start-slot price, families are based on **average window cost**
    w(s,p)/p = (sum of prices over job duration) / job_length.

    This fixes the "start cheap but spill expensive" problem where a job starts in
    a cheap slot but extends into expensive slots, making "cheap family" misleading.

    Key design choices:
    1. Scalar: average window cost w(s,p)/p (not raw w(s,p)), so family semantics
       are consistent across job lengths.
    2. Quantiles: computed per episode over all feasible (job, start) pairs.
    3. Decoding: earliest feasible start whose avg-window-cost is in the chosen family.
    4. Empty-family fallback: if a family has no feasible starts for a job, that
       (job, family) action is masked out.

    Action = (job_id, family_id) where family_id ∈ {0,1,2,3} based on per-episode
    quantiles of average window cost.
    """
    model = ModelConfig.base()
    model.use_global_horizon = False

    env = EnvConfig()
    env.K_period_local = 48
    env.slack_type = SlackType.SHORT  # Keep SHORT for shape compatibility
    env.use_price_families = True  # Enable family-based action space
    env.use_duration_aware_families = (
        True  # Use duration-aware (avg window cost) families
    )
    env.num_price_families = 4  # 4 quartiles
    env.use_periods_full = False

    training = TrainingConfig()
    training.algorithm = RLAlgorithm.PPO

    return VariantConfig(
        variant_id=VariantID.PPO_DURATION_AWARE_FAMILY,
        model=model,
        env=env,
        training=training,
    )


def get_ppo_duration_aware_family_cnn() -> VariantConfig:
    """Duration-aware families + CNN+DeepSets backbone."""
    cfg = get_ppo_duration_aware_family()
    cfg.variant_id = VariantID.PPO_DURATION_AWARE_FAMILY_CNN
    cfg.model.backbone = "cnn_deepsets"
    return cfg


def get_ppo_duration_aware_family_ctx13() -> VariantConfig:
    """Enhanced duration-aware family variant with identified family semantics in ctx.

    Extends ctx from 6 -> 13 by appending:
        - duration-aware quantiles [q25, q50, q75] over avg window costs
        - delta_to_next_feasible_start_in_family[0..3]

    Keeps the same action space as `ppo_duration_aware_family` (job × family), but
    makes family IDs identifiable per episode and provides search guidance.
    """
    cfg = get_ppo_duration_aware_family()
    cfg.variant_id = VariantID.PPO_DURATION_AWARE_FAMILY_CTX13

    cfg.env.F_ctx = 13
    cfg.model.ctx_input_dim = 13

    return cfg


def get_ppo_duration_aware_family_ctx13_cnn() -> VariantConfig:
    """Duration-aware families + ctx13 + CNN+DeepSets backbone."""
    cfg = get_ppo_duration_aware_family_ctx13()
    cfg.variant_id = VariantID.PPO_DURATION_AWARE_FAMILY_CTX13_CNN
    cfg.model.backbone = "cnn_deepsets"
    return cfg


def get_q_sequence() -> VariantConfig:
    """
    Q-learning for Sequences: supervised regression on DP costs.

    This variant learns Q(s, j) = expected DP cost if job j is chosen next,
    then sequence is completed with a rollout policy.

    Key differences from PPO sequence:
    - Training: supervised regression on DP-labeled counterfactuals
    - Model: dueling Q-head outputting per-job Q-values
    - Inference: argmin Q for greedy, or -Q/tau for SGBS logits

    Uses same environment as PPO_SEQUENCE but different training pipeline.
    """
    model = ModelConfig.base()

    env = EnvConfig()
    env.slack_type = SlackType.SHORT
    env.short_slack_spec = ShortSlackSpec(slack_options=[0])  # Only 1 choice => dim = N
    env.use_periods_full = False
    env.K_period_local = 48  # Local price context

    training = TrainingConfig()
    training.algorithm = RLAlgorithm.PPO  # Placeholder - actual training is supervised

    return VariantConfig(
        variant_id=VariantID.Q_SEQUENCE,
        model=model,
        env=env,
        training=training,
    )


def get_q_sequence_ctx13() -> VariantConfig:
    """
    Q-learning for Sequences with ctx13 features.

    Same as Q_SEQUENCE but with 13-dim identified context features
    (price quartiles + next-slot deltas).
    """
    cfg = get_q_sequence()
    cfg.variant_id = VariantID.Q_SEQUENCE_CTX13

    cfg.env.F_ctx = 13
    cfg.model.ctx_input_dim = 13

    # IMPORTANT: ctx13 semantics (quantiles + delta-to-next-slot) are only populated
    # when price-family features are enabled in the env observation builder.
    # This does NOT change the action space for Q-sequence (still job index), it
    # only enriches ctx for the model.
    cfg.env.use_price_families = True

    return cfg


def get_q_sequence_cnn() -> VariantConfig:
    """Q-learning for Sequences with a lightweight CNN+DeepSets encoder.

    Keeps the same environment as `get_q_sequence()` (sequencing + DP timing),
    but swaps the Q-network backbone to a cheaper architecture:
      - 1D CNN over period tokens (detects cheap regions/patterns)
      - DeepSets-style MLP over jobs + pooling
      - Fusion MLP + dueling Q head
    """
    cfg = get_q_sequence()
    cfg.variant_id = VariantID.Q_SEQUENCE_CNN
    return cfg


def get_q_sequence_cnn_ctx13() -> VariantConfig:
    """CNN+DeepSets Q-sequence variant with ctx13 identification features."""
    cfg = get_q_sequence_cnn()
    cfg.variant_id = VariantID.Q_SEQUENCE_CNN_CTX13

    cfg.env.F_ctx = 13
    cfg.model.ctx_input_dim = 13
    cfg.env.use_price_families = True

    return cfg


# Mapping from variant ID to factory function
VARIANT_FACTORIES = {
    VariantID.PPO_SHORT_BASE: get_ppo_short_base,
    VariantID.PPO_SHORT_LARGE: get_ppo_short_large,
    VariantID.PPO_C2F: get_ppo_c2f,
    VariantID.PPO_FULL_TOKENS: get_ppo_full_tokens,
    VariantID.PPO_FULL_GLOBAL: get_ppo_full_global,
    VariantID.REINFORCE_SHORT_SC: get_reinforce_short_sc,
    VariantID.PPO_FAMILY_Q4: get_ppo_family_q4,
    VariantID.PPO_FAMILY_Q4_CTX13: get_ppo_family_q4_ctx13,
    VariantID.PPO_FAMILY_Q4_BESTSTART: get_ppo_family_q4_beststart,
    VariantID.PPO_FAMILY_Q4_BESTSTART_CNN: get_ppo_family_q4_beststart_cnn,
    VariantID.PPO_FAMILY_Q4_CTX13_BESTSTART: get_ppo_family_q4_ctx13_beststart,
    VariantID.PPO_FAMILY_Q4_CTX13_BESTSTART_CNN: get_ppo_family_q4_ctx13_beststart_cnn,
    VariantID.PPO_FAMILY_Q4_CTX18_BESTSTART: get_ppo_family_q4_ctx18_beststart,
    VariantID.PPO_SEQUENCE: get_ppo_sequence,
    VariantID.PPO_DURATION_AWARE_FAMILY: get_ppo_duration_aware_family,
    VariantID.PPO_DURATION_AWARE_FAMILY_CNN: get_ppo_duration_aware_family_cnn,
    VariantID.PPO_DURATION_AWARE_FAMILY_CTX13: get_ppo_duration_aware_family_ctx13,
    VariantID.PPO_DURATION_AWARE_FAMILY_CTX13_CNN: get_ppo_duration_aware_family_ctx13_cnn,
    VariantID.Q_SEQUENCE: get_q_sequence,
    VariantID.Q_SEQUENCE_CTX13: get_q_sequence_ctx13,
    VariantID.Q_SEQUENCE_CNN: get_q_sequence_cnn,
    VariantID.Q_SEQUENCE_CNN_CTX13: get_q_sequence_cnn_ctx13,
}


def get_variant_config(variant_id: VariantID, seed: int = 42) -> VariantConfig:
    """Get configuration for a specific variant."""
    config = VARIANT_FACTORIES[variant_id]()
    config.seed = seed
    config.validate()
    return config


def list_variants() -> List[VariantID]:
    """List all available variant IDs."""
    return list(VariantID)


# =============================================================================
# Testing
# =============================================================================


if __name__ == "__main__":
    print("=" * 70)
    print("PaST-SM Configuration Test")
    print("=" * 70)

    for variant_id in list_variants():
        config = get_variant_config(variant_id)
        print(f"\n{variant_id.value}:")
        print(
            f"  Model: d_model={config.model.d_model}, blocks={config.model.num_blocks}"
        )
        print(
            f"  Env: K_local={config.env.K_period_local}, K_full_max={config.env.K_period_full_max}"
        )
        print(
            f"  Slack: type={config.env.slack_type.value}, K_slack={config.env.get_num_slack_choices()}"
        )
        print(f"  Action dim: {config.env.action_dim}")
        print(f"  Global horizon: {config.model.use_global_horizon}")
        print(f"  Algorithm: {config.training.algorithm.value}")
        if config.training.use_self_critic:
            print(f"  Self-critic baseline: True")
        if config.env.use_price_families:
            print(f"  Price families: {config.env.num_price_families} (quartile-based)")

    print("\n" + "=" * 70)
    print("All variants validated!")
    print("=" * 70)

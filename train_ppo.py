"""
Main PPO Training Script for PaST-SM.

Features:
- CLI for variant selection, seeds, configs
- Deterministic seeding (torch, numpy, random)
- GPU-native batched training
- GAE(λ) advantage estimation
- Proper checkpointing (latest, best, milestones)
- Console + JSONL logging
- Hard assertions for debugging

Usage:
    python -m PaST.train_ppo --variant_id ppo_short_base --seed 0 --smoke_test
    python -m PaST.train_ppo --variant_id ppo_full_global --config configs/a100_full.yaml
"""

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
import copy
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch
import yaml

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from PaST.config import (
    VariantID,
    VariantConfig,
    get_variant_config,
    list_variants,
    DataConfig,
    EnvConfig,
    SlackType,
)
from PaST.past_sm_model import build_model, PaSTSMNet
from PaST.ppo_runner import PPORunner, PPOConfig, RolloutBuffer
from PaST.eval import Evaluator, EvalResult
from PaST.sm_benchmark_data import generate_episode_batch
from PaST.sm_env import GPUBatchSingleMachinePeriodEnv

# Version
TRAIN_VERSION = "2.0-PPO"


# =============================================================================
# Training Configuration
# =============================================================================


@dataclass
class RunConfig:
    """Complete training run configuration."""

    # Variant and identity
    variant_id: str = "ppo_short_base"
    seed: int = 0
    run_name: Optional[str] = None

    # Environment
    num_envs: int = 256  # B: parallel environments
    rollout_length: int = 128  # T: steps per rollout

    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    ppo_epochs: int = 4
    num_minibatches: int = 8
    target_kl: Optional[float] = 0.02
    normalize_advantages: bool = True

    # Learning rate schedule: "constant", "linear", "cosine"
    lr_schedule: str = "constant"
    lr_end_factor: float = (
        0.1  # Final LR = learning_rate * lr_end_factor (for linear/cosine)
    )

    # Entropy schedule: "constant", "linear", "cosine"
    # Starts at entropy_coef, decays to entropy_coef_end over entropy_decay_fraction of training
    entropy_schedule: str = "constant"
    entropy_coef_start: Optional[float] = (
        None  # If set, overrides entropy_coef as start value
    )
    entropy_coef_end: float = 0.001  # Final entropy coefficient
    entropy_decay_fraction: float = (
        0.8  # Fraction of training over which to decay entropy
    )

    # Training budget
    total_env_steps: int = 10_000_000

    # Curriculum learning (opt-in): start with easier instances, anneal to target.
    # Implemented by increasing deadline slack early (larger T_limit).
    curriculum: bool = False
    curriculum_fraction: float = 0.3  # fraction of updates used for annealing

    # Evaluation
    eval_every_updates: int = 20
    num_eval_instances: int = 256

    # Evaluation determinism / fairness across variants.
    # If set, evaluation instances are generated from this seed (or derived from it)
    # rather than the training seed.
    eval_seed: Optional[int] = None
    # How to derive per-eval seeds from eval_seed:
    # - "fixed": always use eval_seed
    # - "per_update": use eval_seed + update
    eval_seed_mode: str = "per_update"

    # Checkpointing
    save_latest_every_updates: int = 10
    save_latest_every_minutes: float = 15.0
    num_milestone_checkpoints: int = 4  # 25%, 50%, 75%, 100%

    # Logging
    log_every_updates: int = 1

    # Device
    device: str = "cuda"

    # Output
    output_dir: str = "runs"

    # Debug
    anomaly_check_every: int = 50  # Check for NaNs, etc.

    @property
    def steps_per_update(self) -> int:
        return self.num_envs * self.rollout_length

    @property
    def num_updates(self) -> int:
        return self.total_env_steps // self.steps_per_update

    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str):
        """Save config to YAML file."""
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    def to_ppo_config(self) -> PPOConfig:
        """Convert to PPOConfig."""
        return PPOConfig(
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_eps=self.clip_eps,
            value_coef=self.value_coef,
            entropy_coef=self.entropy_coef,
            learning_rate=self.learning_rate,
            max_grad_norm=self.max_grad_norm,
            ppo_epochs=self.ppo_epochs,
            num_minibatches=self.num_minibatches,
            target_kl=self.target_kl,
            normalize_advantages=self.normalize_advantages,
        )


def get_p100_smoke_config() -> RunConfig:
    """P100 smoke test config (~30-60 minutes, catch late errors)."""
    return RunConfig(
        num_envs=256,
        rollout_length=128,
        total_env_steps=256 * 128 * 200,  # ~6.5M steps
        ppo_epochs=4,
        num_minibatches=8,
        eval_every_updates=20,
        save_latest_every_updates=20,
        anomaly_check_every=10,
    )


def get_a100_full_config() -> RunConfig:
    """A100 full training config."""
    return RunConfig(
        num_envs=1024,
        rollout_length=128,
        total_env_steps=100_000_000,  # 100M steps
        ppo_epochs=4,
        num_minibatches=16,
        eval_every_updates=50,
        save_latest_every_updates=10,
        save_latest_every_minutes=15.0,
        anomaly_check_every=100,
    )


# =============================================================================
# Learning Rate and Entropy Scheduling
# =============================================================================


def get_schedule_value(
    schedule_type: str,
    progress: float,
    start_value: float,
    end_value: float,
) -> float:
    """
    Compute scheduled value based on training progress.

    Args:
        schedule_type: "constant", "linear", or "cosine"
        progress: Training progress in [0, 1]
        start_value: Initial value
        end_value: Final value

    Returns:
        Scheduled value for current progress
    """
    progress = max(0.0, min(1.0, progress))

    if schedule_type == "constant":
        return start_value
    elif schedule_type == "linear":
        return start_value + (end_value - start_value) * progress
    elif schedule_type == "cosine":
        # Cosine annealing: smooth decay from start to end
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return end_value + (start_value - end_value) * cosine_decay
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def get_lr_for_update(run_config: "RunConfig", update: int) -> float:
    """Get learning rate for given update based on schedule."""
    progress = update / max(1, run_config.num_updates - 1)
    lr_end = run_config.learning_rate * run_config.lr_end_factor
    return get_schedule_value(
        run_config.lr_schedule,
        progress,
        run_config.learning_rate,
        lr_end,
    )


def get_entropy_coef_for_update(run_config: "RunConfig", update: int) -> float:
    """Get entropy coefficient for given update based on schedule."""
    # Use entropy_coef_start if set, otherwise entropy_coef
    start = (
        run_config.entropy_coef_start
        if run_config.entropy_coef_start is not None
        else run_config.entropy_coef
    )
    end = run_config.entropy_coef_end

    # Decay over entropy_decay_fraction of training
    decay_updates = int(run_config.entropy_decay_fraction * run_config.num_updates)
    if decay_updates <= 0:
        return start

    progress = min(1.0, update / decay_updates)
    return get_schedule_value(run_config.entropy_schedule, progress, start, end)


# =============================================================================
# Seeding
# =============================================================================


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For full determinism (may slow down)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_rng_states() -> Dict[str, Any]:
    """Get current RNG states for checkpointing."""
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: Dict[str, Any]):
    """Restore RNG states from checkpoint."""
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    if torch.cuda.is_available() and "cuda" in states:
        torch.cuda.set_rng_state_all(states["cuda"])


# =============================================================================
# Logging
# =============================================================================


class MetricsLogger:
    """Logger for training metrics (console + JSONL)."""

    def __init__(self, log_dir: Path, run_name: str):
        self.log_dir = log_dir
        self.run_name = run_name
        self.jsonl_path = log_dir / "metrics.jsonl"
        self.start_time = time.time()

        # Ensure directory exists
        log_dir.mkdir(parents=True, exist_ok=True)

    def log(self, update: int, env_steps: int, metrics: Dict[str, float]):
        """Log metrics to console and file."""
        elapsed = time.time() - self.start_time

        # Add timing info
        metrics["time/elapsed_seconds"] = elapsed
        metrics["time/env_steps"] = env_steps
        metrics["time/update"] = update
        metrics["time/sps"] = env_steps / elapsed if elapsed > 0 else 0

        # GPU memory
        if torch.cuda.is_available():
            metrics["gpu/memory_allocated_mb"] = torch.cuda.memory_allocated() / 1e6
            metrics["gpu/memory_reserved_mb"] = torch.cuda.memory_reserved() / 1e6

        # Console output (compact)
        self._print_metrics(update, env_steps, metrics)

        # JSONL output (full)
        self._write_jsonl(metrics)

    def _print_metrics(self, update: int, env_steps: int, metrics: Dict[str, float]):
        """Print compact metrics line to console."""
        elapsed = metrics.get("time/elapsed_seconds", 0)
        sps = metrics.get("time/sps", 0)

        # Core training metrics
        policy_loss = metrics.get("train/policy_loss", 0)
        value_loss = metrics.get("train/value_loss", 0)
        entropy = metrics.get("train/entropy", 0)
        approx_kl = metrics.get("train/approx_kl", 0)
        clip_frac = metrics.get("train/clip_frac", 0)
        grad_norm = metrics.get("train/grad_norm", 0)
        lr = metrics.get("train/lr", 0)

        # Rollout metrics
        reward_mean = metrics.get("rollout/rewards_mean", 0)
        reward_std = metrics.get("rollout/rewards_std", 0)

        # GPU memory
        gpu_mem = metrics.get("gpu/memory_allocated_mb", 0)

        print(
            f"[{update:5d}] steps={env_steps:>10,} | "
            f"ret={reward_mean:>8.2f}±{reward_std:<6.2f} | "
            f"π={policy_loss:>7.4f} V={value_loss:>7.4f} H={entropy:>6.3f} | "
            f"kl={approx_kl:.4f} clip={clip_frac:.3f} | "
            f"∇={grad_norm:>6.3f} lr={lr:.2e} | "
            f"mem={gpu_mem:>6.0f}MB | "
            f"sps={sps:>6.0f} t={elapsed/60:>5.1f}m"
        )

    def _write_jsonl(self, metrics: Dict[str, float]):
        """Append metrics to JSONL file."""
        metrics["timestamp"] = datetime.now().isoformat()
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")


# =============================================================================
# Checkpointing
# =============================================================================


class CheckpointManager:
    """Manages checkpoint saving and loading."""

    def __init__(
        self,
        checkpoint_dir: Path,
        num_milestones: int = 4,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.num_milestones = num_milestones
        self.milestone_updates = set()

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_energy = float("inf")
        self.last_save_time = time.time()

    def should_save_latest(
        self,
        update: int,
        save_every_updates: int,
        save_every_minutes: float,
    ) -> bool:
        """Check if we should save latest checkpoint."""
        if update % save_every_updates == 0:
            return True

        elapsed_minutes = (time.time() - self.last_save_time) / 60
        if elapsed_minutes >= save_every_minutes:
            return True

        return False

    def should_save_milestone(
        self,
        update: int,
        num_updates: int,
    ) -> bool:
        """Check if we should save milestone checkpoint."""
        if self.num_milestones <= 0:
            return False

        # Calculate milestone points
        for i in range(1, self.num_milestones + 1):
            milestone = int(num_updates * i / self.num_milestones)
            if update == milestone and milestone not in self.milestone_updates:
                self.milestone_updates.add(milestone)
                return True

        return False

    def save_latest(
        self,
        runner,
        rng_states: Dict[str, Any],
        run_config: RunConfig,
        variant_config: VariantConfig,
    ):
        """Save latest checkpoint (atomic write)."""
        state = {
            "runner": runner.state_dict(),
            "rng_states": rng_states,
            "run_config": asdict(run_config),
            "variant_id": variant_config.variant_id.value,
        }

        # Atomic save: write temp then rename
        temp_path = self.checkpoint_dir / "latest.pt.tmp"
        final_path = self.checkpoint_dir / "latest.pt"

        torch.save(state, temp_path)
        shutil.move(str(temp_path), str(final_path))

        self.last_save_time = time.time()
        print(f"  [Checkpoint] Saved latest.pt")

    def save_best(
        self,
        runner,
        rng_states: Dict[str, Any],
        run_config: RunConfig,
        variant_config: VariantConfig,
        eval_result: EvalResult,
    ) -> bool:
        """Save best checkpoint if energy improved. Returns True if saved."""
        if eval_result.energy_mean < self.best_energy:
            self.best_energy = eval_result.energy_mean

            state = {
                "runner": runner.state_dict(),
                "rng_states": rng_states,
                "run_config": asdict(run_config),
                "variant_id": variant_config.variant_id.value,
                "eval_result": eval_result.to_dict(),
            }

            torch.save(state, self.checkpoint_dir / "best.pt")
            print(
                f"  [Checkpoint] Saved best.pt (energy={eval_result.energy_mean:.2f})"
            )
            return True

        return False

    def save_milestone(
        self,
        runner,
        rng_states: Dict[str, Any],
        run_config: RunConfig,
        variant_config: VariantConfig,
        update: int,
        num_updates: int,
    ):
        """Save milestone checkpoint."""
        progress = int(100 * update / num_updates)

        state = {
            "runner": runner.state_dict(),
            "rng_states": rng_states,
            "run_config": asdict(run_config),
            "variant_id": variant_config.variant_id.value,
            "update": update,
            "progress_percent": progress,
        }

        path = self.checkpoint_dir / f"milestone_{progress:03d}pct.pt"
        torch.save(state, path)
        print(f"  [Checkpoint] Saved milestone_{progress:03d}pct.pt")

        # Keep only last few milestones
        self._cleanup_milestones()

    def _cleanup_milestones(self, keep: int = 5):
        """Remove old milestone checkpoints."""
        milestones = sorted(self.checkpoint_dir.glob("milestone_*.pt"))
        for old in milestones[:-keep]:
            old.unlink()

    def load_latest(self, device: torch.device) -> Optional[Dict[str, Any]]:
        """Load latest checkpoint if exists."""
        path = self.checkpoint_dir / "latest.pt"
        if path.exists():
            return torch.load(path, map_location=device)
        return None

    def load_best_energy_from_checkpoint(self, device: torch.device):
        """
        Load best_energy from best.pt if it exists.

        Call this after resume to ensure best.pt isn't overwritten
        with worse results.
        """
        path = self.checkpoint_dir / "best.pt"
        if path.exists():
            checkpoint = torch.load(path, map_location=device)
            if "eval_result" in checkpoint:
                self.best_energy = checkpoint["eval_result"].get(
                    "eval/energy_mean", float("inf")
                )
                print(
                    f"  [Checkpoint] Restored best_energy={self.best_energy:.2f} from best.pt"
                )


# =============================================================================
# GPU Environment Wrapper
# =============================================================================


class TrainingEnv:
    """
    Wrapper around GPUBatchSingleMachinePeriodEnv for training.

    Handles:
    - Episode auto-reset
    - Data generation
    - Observation shape reporting
    """

    def __init__(
        self,
        variant_config: VariantConfig,
        num_envs: int,
        device: torch.device,
    ):
        self.variant_config = variant_config
        self.num_envs = num_envs
        self.device = device

        # Create data config
        # Keep a private copy so training-time tweaks (e.g., curriculum) do not
        # mutate variant_config.data (which is also used for evaluation batches).
        self.base_data_config = copy.deepcopy(variant_config.data)
        self.data_config = copy.deepcopy(variant_config.data)

        # Create environment config from the variant env config.
        # This is critical so action dimensions match across variants
        # (slack_type determines K_slack via EnvConfig.get_num_slack_choices()).
        from PaST.config import SlackVariant

        env_config = copy.deepcopy(variant_config.env)

        # sm_env uses legacy slack_variant in its slack timing logic.
        # Keep it consistent with the new slack_type to avoid shape mismatches.
        if env_config.slack_type == SlackType.SHORT:
            env_config.slack_variant = SlackVariant.SHORT_SLACK
        elif env_config.slack_type == SlackType.COARSE_TO_FINE:
            env_config.slack_variant = SlackVariant.COARSE_TO_FINE
        elif env_config.slack_type == SlackType.FULL:
            env_config.slack_variant = SlackVariant.FULL_SLACK

        self.env_config = env_config

        # Create GPU environment
        self.env = GPUBatchSingleMachinePeriodEnv(
            batch_size=num_envs,
            env_config=env_config,
            device=device,
        )

        self.batch_size = num_envs

        # Track done mask for auto-reset
        self.done_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def set_deadline_slack_ratio_range(self, ratio_min: float, ratio_max: float):
        """Update data generation difficulty via deadline slack ratio range."""
        ratio_min_f = float(ratio_min)
        ratio_max_f = float(ratio_max)
        # Keep in [0, 1] and ordered.
        ratio_min_f = max(0.0, min(1.0, ratio_min_f))
        ratio_max_f = max(0.0, min(1.0, ratio_max_f))
        if ratio_min_f > ratio_max_f:
            ratio_min_f, ratio_max_f = ratio_max_f, ratio_min_f

        self.data_config.deadline_slack_ratio_min = ratio_min_f
        self.data_config.deadline_slack_ratio_max = ratio_max_f

    def reset(self, seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Reset all environments with new data."""
        batch_data = generate_episode_batch(
            batch_size=self.num_envs,
            config=self.data_config,
            seed=seed,
        )
        obs = self.env.reset(batch_data)
        self.done_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        return obs

    def step(
        self,
        actions: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """
        Take a step in all environments with auto-reset.

        Auto-resets only the envs that finished (standard vectorized RL behavior).
        Done flags are returned per-env; observations for done envs are from the
        reset episode (matching common VecEnv semantics).

        Returns:
            obs: Next observation (post-reset if any env finished)
            rewards: Rewards for this step
            dones: Done flags - ALL TRUE if any env finished (episode boundary)
            info: Additional info
        """
        obs, rewards, dones, info = self.env.step(actions)

        # Track which envs just finished (before updating done_mask)
        newly_done = dones & ~self.done_mask

        # Auto-reset only the newly done indices.
        if newly_done.any():
            reset_idx = torch.nonzero(newly_done, as_tuple=False).squeeze(1)
            batch_data = generate_episode_batch(
                batch_size=int(reset_idx.numel()),
                config=self.data_config,
                seed=None,
            )
            obs = self.env.reset_indices(batch_data, reset_idx)

        # Keep wrapper done_mask in sync with the underlying env (after any resets).
        # Note: env.step returns cumulative done_mask; PPO should receive per-step terminals.
        if hasattr(self.env, "done_mask") and isinstance(
            self.env.done_mask, torch.Tensor
        ):
            self.done_mask = self.env.done_mask.clone()
        else:
            # Fallback: track cumulatively ourselves
            self.done_mask = self.done_mask | dones

        # PPO terminal signal: only newly finished envs.
        return obs, rewards, newly_done, info

    def get_obs_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Get observation tensor shapes (without batch dimension)."""
        shapes = {}

        # Jobs: (M_job_bins, F_job) - but sm_env uses N_job_pad
        shapes["jobs"] = (self.env_config.N_job_pad, self.env_config.F_job)

        # Periods: (K_period_lookahead, F_period)
        shapes["periods"] = (
            self.env_config.K_period_lookahead,
            self.env_config.F_period,
        )

        # Period mask: (K_period_lookahead,)
        # 1.0 = valid period token, 0.0 = invalid/padding
        shapes["period_mask"] = (self.env_config.K_period_lookahead,)

        # Context: (F_ctx,)
        shapes["ctx"] = (self.env_config.F_ctx,)

        # Job mask: (N_job_pad,)
        shapes["job_mask"] = (self.env_config.N_job_pad,)

        # Action mask: (action_dim,)
        action_dim = self.env_config.N_job_pad * self.env_config.get_num_slack_choices()
        shapes["action_mask"] = (action_dim,)

        return shapes


# =============================================================================
# Main Training Loop
# =============================================================================


def train(
    run_config: RunConfig,
    variant_config: VariantConfig,
    resume_path: Optional[str] = None,
):
    """
    Main training function.

    Args:
        run_config: Training run configuration
        variant_config: Model variant configuration
        resume_path: Path to checkpoint to resume from
    """
    # Setup
    device = torch.device(run_config.device)
    set_seed(run_config.seed)

    # Create run directory
    run_name = run_config.run_name or f"{run_config.variant_id}_seed{run_config.seed}"
    run_dir = (
        Path(run_config.output_dir) / run_config.variant_id / f"seed_{run_config.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    run_config.to_yaml(run_dir / "run_config.yaml")
    with open(run_dir / "variant_config.json", "w") as f:
        json.dump(variant_config.to_dict(), f, indent=2)

    # Initialize logging
    logger = MetricsLogger(run_dir, run_name)
    checkpoint_mgr = CheckpointManager(
        run_dir / "checkpoints",
        num_milestones=run_config.num_milestone_checkpoints,
    )

    print("=" * 80)
    print(f"PaST-SM PPO Training v{TRAIN_VERSION}")
    print("=" * 80)
    print(f"Variant: {run_config.variant_id}")
    print(f"Seed: {run_config.seed}")
    print(f"Device: {device}")
    print(f"Num envs: {run_config.num_envs}")
    print(f"Rollout length: {run_config.rollout_length}")
    print(f"Steps per update: {run_config.steps_per_update:,}")
    print(f"Total updates: {run_config.num_updates:,}")
    print(f"Total env steps: {run_config.total_env_steps:,}")
    print(f"Output: {run_dir}")
    print("=" * 80)

    # Create environment
    print("\nCreating environment...")
    env = TrainingEnv(
        variant_config=variant_config,
        num_envs=run_config.num_envs,
        device=device,
    )
    obs_shapes = env.get_obs_shapes()
    print(f"Observation shapes: {obs_shapes}")

    # Create model
    print("\nCreating model...")
    model = build_model(variant_config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Action dim: {model.action_dim}")

    # Create PPO runner
    print("\nCreating PPO runner...")
    ppo_config = run_config.to_ppo_config()
    runner = PPORunner(
        model=model,
        env=env,
        ppo_config=ppo_config,
        device=device,
        obs_shapes=obs_shapes,
        rollout_length=run_config.rollout_length,
    )

    # Create evaluator
    eval_env = TrainingEnv(
        variant_config=variant_config,
        num_envs=run_config.num_eval_instances,
        device=device,
    )
    evaluator = Evaluator(
        model=model,
        env=eval_env.env,
        device=device,
    )

    # Resume from checkpoint if provided
    start_update = 0
    if resume_path:
        print(f"\nResuming from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device)
        runner.load_state_dict(checkpoint["runner"])
        set_rng_states(checkpoint["rng_states"])
        start_update = runner.update_count
        # Restore best_energy to avoid overwriting best.pt with worse results
        checkpoint_mgr.load_best_energy_from_checkpoint(device)
        print(f"Resumed at update {start_update}, step {runner.global_step}")

    # Curriculum: start with easier instances (looser deadlines) and anneal.
    base_slack_min = float(variant_config.data.deadline_slack_ratio_min)
    base_slack_max = float(variant_config.data.deadline_slack_ratio_max)

    def _curriculum_progress(update_idx: int) -> float:
        if not run_config.curriculum:
            return 1.0
        frac = float(run_config.curriculum_fraction)
        frac = max(0.0, min(1.0, frac))
        denom = max(1, int(round(frac * max(1, run_config.num_updates))))
        return min(1.0, float(update_idx) / float(denom))

    def _apply_curriculum(update_idx: int) -> Tuple[float, float]:
        p = _curriculum_progress(update_idx)
        # Start at the easiest setting: fixed slack_ratio = base_slack_max.
        start_min = base_slack_max
        start_max = base_slack_max

        slack_min = start_min + (base_slack_min - start_min) * p
        slack_max = start_max + (base_slack_max - start_max) * p
        env.set_deadline_slack_ratio_range(slack_min, slack_max)
        return slack_min, slack_max

    # Initial reset
    print("\nStarting training...")
    cur_slack_min, cur_slack_max = _apply_curriculum(start_update)
    obs = env.reset(seed=run_config.seed)

    # Track the most recently used eval seed so the final evaluation can be
    # comparable to the eval curve (especially when eval_seed_mode="fixed").
    last_eval_seed: Optional[int] = None

    # Training loop
    for update in range(start_update, run_config.num_updates):
        update_start = time.time()

        # Update curriculum before collecting the next rollout.
        cur_slack_min, cur_slack_max = _apply_curriculum(update)

        # Apply learning rate schedule
        current_lr = get_lr_for_update(run_config, update)
        runner.set_learning_rate(current_lr)

        # Apply entropy coefficient schedule
        current_entropy_coef = get_entropy_coef_for_update(run_config, update)
        runner.config.entropy_coef = current_entropy_coef

        # Collect rollout
        obs, rollout_metrics = runner.collect_rollout(obs)

        # PPO update
        train_metrics = runner.update()

        # Combine metrics
        metrics = {**rollout_metrics, **train_metrics}
        metrics["train/lr"] = runner.get_learning_rate()
        metrics["train/entropy_coef"] = current_entropy_coef
        metrics["curriculum/enabled"] = float(run_config.curriculum)
        metrics["curriculum/deadline_slack_ratio_min"] = float(cur_slack_min)
        metrics["curriculum/deadline_slack_ratio_max"] = float(cur_slack_max)

        # Anomaly checking
        if update % run_config.anomaly_check_every == 0:
            # Check for NaNs in model
            for name, param in model.named_parameters():
                assert torch.isfinite(param).all(), f"NaN/Inf in {name}"
                if param.grad is not None:
                    assert torch.isfinite(param.grad).all(), f"NaN/Inf in {name}.grad"

        # Evaluation
        is_eval_step = update % run_config.eval_every_updates == 0 and update > 0
        if is_eval_step:
            print(f"\n  [Eval] Running evaluation...")
            if run_config.eval_seed is None:
                eval_seed = run_config.seed + update
            else:
                if run_config.eval_seed_mode == "fixed":
                    eval_seed = run_config.eval_seed
                elif run_config.eval_seed_mode == "per_update":
                    eval_seed = run_config.eval_seed + update
                else:
                    raise ValueError(
                        f"Unknown eval_seed_mode: {run_config.eval_seed_mode}"
                    )
            last_eval_seed = int(eval_seed)
            eval_batch = generate_episode_batch(
                batch_size=run_config.num_eval_instances,
                config=variant_config.data,
                seed=eval_seed,
            )
            eval_result, _ = evaluator.evaluate(eval_batch, deterministic=True)
            eval_metrics = eval_result.to_dict()
            # Merge eval metrics into main metrics (logged once below)
            metrics.update(eval_metrics)

            print(
                f"  [Eval] energy={eval_result.energy_mean:.2f}±{eval_result.energy_std:.2f} "
                f"infeas={eval_result.infeasible_rate*100:.1f}%"
            )

            # Save best checkpoint
            rng_states = get_rng_states()
            checkpoint_mgr.save_best(
                runner, rng_states, run_config, variant_config, eval_result
            )

        # Logging (single log call per update, includes eval metrics if eval step)
        if update % run_config.log_every_updates == 0:
            logger.log(update, runner.global_step, metrics)

        # Save latest checkpoint
        if checkpoint_mgr.should_save_latest(
            update,
            run_config.save_latest_every_updates,
            run_config.save_latest_every_minutes,
        ):
            rng_states = get_rng_states()
            checkpoint_mgr.save_latest(runner, rng_states, run_config, variant_config)

        # Save milestone checkpoint
        if checkpoint_mgr.should_save_milestone(update, run_config.num_updates):
            rng_states = get_rng_states()
            checkpoint_mgr.save_milestone(
                runner,
                rng_states,
                run_config,
                variant_config,
                update,
                run_config.num_updates,
            )

    # Final checkpoint
    print("\n" + "=" * 80)
    print("Training complete!")
    rng_states = get_rng_states()
    checkpoint_mgr.save_latest(runner, rng_states, run_config, variant_config)

    # Final evaluation
    print("\nFinal evaluation...")

    # 1) Evaluate on the SAME eval distribution as the eval curve.
    # If eval_seed_mode is fixed, this is always the same eval set.
    # If eval_seed_mode is per_update, this evaluates on the most recent eval seed.
    if last_eval_seed is not None:
        final_eval_seed = int(last_eval_seed)
    else:
        # No eval ever ran (e.g., eval_every_updates > num_updates). Fall back.
        if run_config.eval_seed is None:
            final_eval_seed = int(run_config.seed)
        else:
            final_eval_seed = int(run_config.eval_seed)

    eval_batch_main = generate_episode_batch(
        batch_size=run_config.num_eval_instances,
        config=variant_config.data,
        seed=final_eval_seed,
    )
    final_latest_main, _ = evaluator.evaluate(eval_batch_main, deterministic=True)
    print(
        f"Final (latest, eval_seed={final_eval_seed}): "
        f"energy={final_latest_main.energy_mean:.2f}±{final_latest_main.energy_std:.2f} "
        f"infeas={final_latest_main.infeasible_rate*100:.1f}%"
    )

    # Write the legacy file name as the comparable-to-curve result.
    with open(run_dir / "final_result.json", "w") as f:
        payload = final_latest_main.to_dict()
        payload["eval/seed"] = int(final_eval_seed)
        payload["eval/kind"] = "latest_on_eval_seed"
        json.dump(payload, f, indent=2)

    # 2) Also evaluate BEST checkpoint on the same eval batch (if it exists).
    best_path = run_dir / "checkpoints" / "best.pt"
    if best_path.exists():
        best_checkpoint = torch.load(best_path, map_location=device)
        runner.load_state_dict(best_checkpoint["runner"])
        best_main, _ = evaluator.evaluate(eval_batch_main, deterministic=True)
        print(
            f"Final (best,   eval_seed={final_eval_seed}): "
            f"energy={best_main.energy_mean:.2f}±{best_main.energy_std:.2f} "
            f"infeas={best_main.infeasible_rate*100:.1f}%"
        )
        with open(run_dir / "final_best_on_eval_seed.json", "w") as f:
            payload = best_main.to_dict()
            payload["eval/seed"] = int(final_eval_seed)
            payload["eval/kind"] = "best_on_eval_seed"
            json.dump(payload, f, indent=2)

        # Restore latest weights to avoid surprising state for any downstream code.
        latest_checkpoint = checkpoint_mgr.load_latest(device)
        if latest_checkpoint is not None:
            runner.load_state_dict(latest_checkpoint["runner"])

    # 3) Optional generalization eval on a held-out seed (stable across variants).
    if run_config.eval_seed is None:
        generalization_seed = int(run_config.seed) + 999999
    else:
        generalization_seed = int(run_config.eval_seed) + 999999
    eval_batch_gen = generate_episode_batch(
        batch_size=run_config.num_eval_instances,
        config=variant_config.data,
        seed=generalization_seed,
    )
    final_latest_gen, _ = evaluator.evaluate(eval_batch_gen, deterministic=True)
    print(
        f"Final (latest, gen_seed={generalization_seed}): "
        f"energy={final_latest_gen.energy_mean:.2f}±{final_latest_gen.energy_std:.2f} "
        f"infeas={final_latest_gen.infeasible_rate*100:.1f}%"
    )
    with open(run_dir / "final_latest_generalization.json", "w") as f:
        payload = final_latest_gen.to_dict()
        payload["eval/seed"] = int(generalization_seed)
        payload["eval/kind"] = "latest_generalization"
        json.dump(payload, f, indent=2)

    print("=" * 80)
    return runner, final_latest_main


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PaST-SM PPO Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Variant selection
    variant_choices = [v.value for v in list_variants()]
    parser.add_argument(
        "--variant_id",
        type=str,
        default="ppo_short_base",
        choices=variant_choices,
        help="Model variant to train",
    )

    # Seeds (supports multiple)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (single seed mode)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Multiple seeds to run (overrides --seed)",
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )

    # Resume
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Quick test mode
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Use P100 smoke test config",
    )

    # Full training mode
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use A100 full training config",
    )

    # Output directory
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs",
        help="Output directory for runs",
    )

    # Evaluation seeding (for fair cross-variant comparison)
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=None,
        help="Base seed for evaluation instance generation (overrides training-seed-derived eval seeds)",
    )
    parser.add_argument(
        "--eval_seed_mode",
        type=str,
        default=None,
        choices=["fixed", "per_update"],
        help="How to vary evaluation seeds across eval calls when --eval_seed is set",
    )

    # Override hyperparameters
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--rollout_length", type=int, default=None)
    parser.add_argument("--total_env_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--ppo_epochs", type=int, default=None)

    # Curriculum learning (easier instances early)
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable curriculum learning by starting with looser deadlines and annealing to the target distribution.",
    )
    parser.add_argument(
        "--curriculum_fraction",
        type=float,
        default=None,
        help="Fraction of total updates used to anneal from easy to target (e.g., 0.3).",
    )

    # Learning rate schedule
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default=None,
        choices=["constant", "linear", "cosine"],
        help="Learning rate schedule type.",
    )
    parser.add_argument(
        "--lr_end_factor",
        type=float,
        default=None,
        help="Final LR = learning_rate * lr_end_factor (for linear/cosine schedules).",
    )

    # Entropy schedule
    parser.add_argument(
        "--entropy_schedule",
        type=str,
        default=None,
        choices=["constant", "linear", "cosine"],
        help="Entropy coefficient schedule type.",
    )
    parser.add_argument(
        "--entropy_coef_start",
        type=float,
        default=None,
        help="Starting entropy coefficient (overrides --entropy_coef if set).",
    )
    parser.add_argument(
        "--entropy_coef_end",
        type=float,
        default=None,
        help="Final entropy coefficient for scheduled decay.",
    )
    parser.add_argument(
        "--entropy_decay_fraction",
        type=float,
        default=None,
        help="Fraction of training over which entropy decays (e.g., 0.8).",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Determine which seeds to run
    seeds = args.seeds if args.seeds else [args.seed]

    # Load or create run config
    if args.config:
        run_config = RunConfig.from_yaml(args.config)
    elif args.smoke_test:
        run_config = get_p100_smoke_config()
    elif args.full:
        run_config = get_a100_full_config()
    else:
        run_config = RunConfig()

    # Apply CLI overrides
    run_config.variant_id = args.variant_id
    run_config.device = args.device
    run_config.output_dir = args.output_dir

    if args.eval_seed is not None:
        run_config.eval_seed = int(args.eval_seed)
    if args.eval_seed_mode is not None:
        run_config.eval_seed_mode = str(args.eval_seed_mode)

    if args.num_envs is not None:
        run_config.num_envs = args.num_envs
    if args.rollout_length is not None:
        run_config.rollout_length = args.rollout_length
    if args.total_env_steps is not None:
        run_config.total_env_steps = args.total_env_steps
    if args.learning_rate is not None:
        run_config.learning_rate = args.learning_rate
    if args.ppo_epochs is not None:
        run_config.ppo_epochs = args.ppo_epochs

    # Curriculum overrides
    if args.curriculum:
        run_config.curriculum = True
    if args.curriculum_fraction is not None:
        run_config.curriculum_fraction = args.curriculum_fraction

    # LR schedule overrides
    if args.lr_schedule is not None:
        run_config.lr_schedule = args.lr_schedule
    if args.lr_end_factor is not None:
        run_config.lr_end_factor = args.lr_end_factor

    # Entropy schedule overrides
    if args.entropy_schedule is not None:
        run_config.entropy_schedule = args.entropy_schedule
    if args.entropy_coef_start is not None:
        run_config.entropy_coef_start = args.entropy_coef_start
    if args.entropy_coef_end is not None:
        run_config.entropy_coef_end = args.entropy_coef_end
    if args.entropy_decay_fraction is not None:
        run_config.entropy_decay_fraction = args.entropy_decay_fraction

    # Get variant config
    variant_id = VariantID(args.variant_id)
    variant_config = get_variant_config(variant_id)

    # Run training for each seed
    for seed in seeds:
        print(f"\n{'#' * 80}")
        print(f"# Running seed {seed}")
        print(f"{'#' * 80}\n")

        run_config.seed = seed
        train(run_config, variant_config, resume_path=args.resume)


if __name__ == "__main__":
    main()

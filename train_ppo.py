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
import os
import random
import shutil
import sys
import time
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

    # Training budget
    total_env_steps: int = 10_000_000

    # Evaluation
    eval_every_updates: int = 20
    num_eval_instances: int = 256

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
        runner: PPORunner,
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
        runner: PPORunner,
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
        runner: PPORunner,
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
        self.data_config = variant_config.data

        # Create environment config (use legacy fields for sm_env compatibility)
        from PaST.config import SlackVariant, ShortSlackSpec, PeriodAlignedSlackSpec

        env_config = EnvConfig()
        env_config.M_job_bins = variant_config.env.M_job_bins
        env_config.K_period_local = variant_config.env.K_period_local
        env_config.K_period_full_max = variant_config.env.K_period_full_max

        # Map new SlackType to legacy SlackVariant for sm_env
        slack_type = variant_config.env.slack_type
        if slack_type == SlackType.SHORT:
            env_config.slack_variant = SlackVariant.SHORT_SLACK
            env_config.short_slack_spec = variant_config.env.short_slack_spec
        elif slack_type == SlackType.COARSE_TO_FINE:
            env_config.slack_variant = SlackVariant.COARSE_TO_FINE
            env_config.c2f_slack_spec = variant_config.env.c2f_slack_spec
        elif slack_type == SlackType.FULL:
            env_config.slack_variant = SlackVariant.FULL_SLACK
            env_config.full_slack_spec = variant_config.env.full_slack_spec

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

        CRITICAL: When ANY env is done, we reset the ENTIRE batch.
        To maintain trajectory integrity, we mark ALL envs as done on that step
        so PPO treats it as an episode boundary for EVERY env (not just the
        ones that actually finished). This is wasteful but correct.

        Returns:
            obs: Next observation (post-reset if any env finished)
            rewards: Rewards for this step
            dones: Done flags - ALL TRUE if any env finished (episode boundary)
            info: Additional info
        """
        obs, rewards, dones, info = self.env.step(actions)

        # Track which envs just finished (before updating done_mask)
        newly_done = dones & ~self.done_mask

        # Auto-reset when ANY episode is done to prevent all-masked action spaces
        if newly_done.any():
            obs = self.reset()
            # CRITICAL: Mark ALL envs as done so PPO treats this as episode boundary
            # for the entire batch. This ensures obs (from new episodes) pairs with
            # done=True, preventing invalid transitions where done=False but next_obs
            # is from a different episode.
            all_done = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            self.done_mask = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            return obs, rewards, all_done, info
        else:
            self.done_mask = self.done_mask | dones
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

    # Initial reset
    print("\nStarting training...")
    obs = env.reset(seed=run_config.seed)

    # Training loop
    for update in range(start_update, run_config.num_updates):
        update_start = time.time()

        # Collect rollout
        obs, rollout_metrics = runner.collect_rollout(obs)

        # PPO update
        train_metrics = runner.update()

        # Combine metrics
        metrics = {**rollout_metrics, **train_metrics}
        metrics["train/lr"] = runner.get_learning_rate()

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
            eval_batch = generate_episode_batch(
                batch_size=run_config.num_eval_instances,
                config=variant_config.data,
                seed=run_config.seed + update,  # Different seed for eval
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
    eval_batch = generate_episode_batch(
        batch_size=run_config.num_eval_instances,
        config=variant_config.data,
        seed=run_config.seed + 999999,
    )
    final_result, _ = evaluator.evaluate(eval_batch, deterministic=True)
    print(
        f"Final: energy={final_result.energy_mean:.2f}±{final_result.energy_std:.2f} "
        f"infeas={final_result.infeasible_rate*100:.1f}%"
    )

    # Save final result
    with open(run_dir / "final_result.json", "w") as f:
        json.dump(final_result.to_dict(), f, indent=2)

    print("=" * 80)
    return runner, final_result


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

    # Override hyperparameters
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--rollout_length", type=int, default=None)
    parser.add_argument("--total_env_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--ppo_epochs", type=int, default=None)

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

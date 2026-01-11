"""
Training module for PaST-SM (Period-aware Scheduler Transformer - Single Machine).

Uses REINFORCE with baseline, following the paper's Algorithm 1.
Key differences from ECSP training:
- Episodes have variable length (loop until done, not fixed N steps)
- Async prefetch generates episode dicts, not task arrays
- Dense energy-only reward (not terminal Tchebycheff reward)
"""

import os
import json
import time
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from threading import Thread
from queue import Queue
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .config import (
    PaSTConfig,
    TrainingConfig,
    DataConfig,
    EnvConfig,
    ModelConfig,
    SlackVariant,
)
from .sm_benchmark_data import (
    generate_episode_batch,
    generate_single_machine_episode,
    SingleMachineEpisode,
)
from .sm_env import (
    SingleMachinePeriodEnv,
    GPUBatchSingleMachinePeriodEnv,
)
from .past_sm_model import PaSTSMNet, obs_dict_to_tensors


# Version identifier
TRAIN_VERSION = "1.0-SM"
print(f"[PaST-SM Trainer v{TRAIN_VERSION}] Loading training module...")


class AsyncEpisodePrefetcher:
    """
    Async data prefetcher that generates episode batches on CPU in background threads.
    Episode dicts are generated in background, GPU transfer happens in main thread.
    """

    def __init__(
        self,
        data_config: DataConfig,
        env_config: EnvConfig,
        batch_size: int,
        device: torch.device,
        prefetch_count: int = 3,
        num_workers: int = 4,
    ):
        self.data_config = data_config
        self.env_config = env_config
        self.batch_size = batch_size
        self.device = device
        self.prefetch_count = prefetch_count
        self.num_workers = num_workers

        # Queue for prefetched batches
        self.queue = Queue(maxsize=prefetch_count)
        self.stop_flag = False
        self.workers = []

        # Padding dimensions from env config
        self.N_job_pad = env_config.N_job_pad
        self.K_period_pad = 250  # Max periods: T_max=500 / min_period=2 = 250
        self.T_max_pad = 500  # Max slots

    def _prefetch_worker(self, worker_id: int):
        """Worker thread that generates episode batches."""
        while not self.stop_flag:
            try:
                # Generate random seed for this batch
                seed = random.randint(0, 2**31 - 1)

                # Generate batch on CPU
                batch = generate_episode_batch(
                    batch_size=self.batch_size,
                    config=self.data_config,
                    seed=seed,
                    N_job_pad=self.N_job_pad,
                    K_period_pad=self.K_period_pad,
                    T_max_pad=self.T_max_pad,
                )

                # Put in queue (blocks if full)
                self.queue.put(batch, timeout=1.0)
            except Exception as e:
                if not self.stop_flag:
                    print(f"Worker {worker_id} error: {e}")
                    continue

    def start(self):
        """Start prefetch workers."""
        self.stop_flag = False
        self.workers = []
        for i in range(self.num_workers):
            worker = Thread(target=self._prefetch_worker, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)

    def get_batch(self) -> Dict[str, np.ndarray]:
        """Get next prefetched batch (CPU numpy arrays)."""
        return self.queue.get(timeout=30.0)

    def stop(self):
        """Stop prefetch workers."""
        self.stop_flag = True
        for worker in self.workers:
            worker.join(timeout=1.0)
        self.workers = []


class Trainer:
    """
    PaST-SM Training using REINFORCE with baseline.

    Key differences from ECSP Trainer:
    - Episodes have variable length (loop until done)
    - Dense reward (negative energy per step)
    - No Tchebycheff decomposition (single objective: minimize energy)
    """

    def __init__(self, config: PaSTConfig):
        """
        Initialize trainer.

        Args:
            config: Master configuration
        """
        self.config = config

        # Validate config
        config.validate()

        # Device
        self.device = torch.device(config.device)

        # Dimensions
        self.batch_size = config.training.batch_size
        self.batches_per_epoch = config.training.batches_per_epoch
        self.num_epochs = config.training.num_epochs

        # Create model
        self.model = PaSTSMNet(config.model, config.env).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.training.initial_lr,
        )

        # Learning rate scheduler
        self.lr_decay_epochs = config.training.lr_decay_epochs
        self.lr_decay_factor = config.training.lr_decay_factor

        # Entropy coefficient
        self.entropy_coef = config.training.entropy_coef

        # Baseline bins for variance reduction
        self.num_baseline_bins = config.training.num_baseline_bins

        # Create batched GPU environment
        self.env = GPUBatchSingleMachinePeriodEnv(
            batch_size=self.batch_size,
            env_config=config.env,
            device=self.device,
        )

        # Create async prefetcher
        self.prefetcher = AsyncEpisodePrefetcher(
            data_config=config.data,
            env_config=config.env,
            batch_size=self.batch_size,
            device=self.device,
            prefetch_count=config.training.prefetch_count,
            num_workers=config.training.num_workers,
        )

        # Training history
        self.history = {
            "epoch": [],
            "loss": [],
            "energy_mean": [],
            "energy_std": [],
            "entropy": [],
            "lr": [],
        }

        # Save directory
        self.save_dir = config.training.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def compute_baseline(
        self,
        rewards: torch.Tensor,  # (B,) total episode rewards
        energies: torch.Tensor,  # (B,) total episode energies
        T_limits: torch.Tensor,  # (B,) deadline constraints
    ) -> torch.Tensor:
        """
        Compute baseline for variance reduction using binning by T_limit.

        Args:
            rewards: Total episode rewards (B,)
            energies: Total episode energies (B,)
            T_limits: Deadline constraints (B,)

        Returns:
            Baselines for each instance (B,)
        """
        B = rewards.shape[0]
        baselines = torch.zeros(B, device=self.device)

        # Bin by T_limit
        T_min = T_limits.min().item()
        T_max = T_limits.max().item()

        if T_max == T_min:
            # All same T_limit - use global mean
            baselines[:] = rewards.mean()
        else:
            # Create bins
            bin_edges = torch.linspace(
                T_min, T_max + 1, self.num_baseline_bins + 1, device=self.device
            )
            bin_indices = torch.bucketize(T_limits.float(), bin_edges) - 1
            bin_indices = torch.clamp(bin_indices, 0, self.num_baseline_bins - 1)

            # Compute mean reward per bin
            for bin_idx in range(self.num_baseline_bins):
                mask = bin_indices == bin_idx
                if mask.sum() > 0:
                    bin_mean = rewards[mask].mean()
                    baselines[mask] = bin_mean

        return baselines

    def rollout_batch(
        self,
        batch_data: Dict[str, np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Rollout a batch of episodes until completion.

        Args:
            batch_data: Episode batch from generate_episode_batch

        Returns:
            total_energies: (B,) total energy for each episode
            total_log_probs: (B,) sum of log probabilities
            mean_entropy: scalar mean entropy
        """
        B = self.batch_size

        # Reset environment with batch data
        obs = self.env.reset(batch_data)

        # Track cumulative values
        total_log_probs = torch.zeros(B, device=self.device)
        total_entropy = 0.0
        step_count = 0
        max_steps = self.config.env.N_job_pad  # Safety limit

        # Track which episodes are done
        done_mask = torch.zeros(B, dtype=torch.bool, device=self.device)

        # Rollout until all episodes are done
        while not done_mask.all() and step_count < max_steps:
            # Forward pass
            probs, logits = self.model.forward_from_obs(obs)

            # Sample actions
            actions, log_probs = self.model.sample_action(probs)

            # Compute entropy (only for active episodes)
            entropy = self.model.compute_entropy(probs)
            total_entropy += entropy.item()

            # Accumulate log probs (only for active episodes)
            active_mask = ~done_mask
            total_log_probs[active_mask] += log_probs[active_mask]

            # Step environment
            obs, rewards, dones, info = self.env.step(actions)

            # Update done mask
            done_mask = dones
            step_count += 1

        # Get final energies
        total_energies = self.env.total_energy.clone()
        mean_entropy = total_entropy / max(step_count, 1)

        return (
            total_energies,
            total_log_probs,
            torch.tensor(mean_entropy, device=self.device),
        )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()

        epoch_losses = []
        epoch_energies = []
        epoch_entropies = []

        for batch_idx in range(self.batches_per_epoch):
            # Get prefetched batch
            batch_data = self.prefetcher.get_batch()

            # Rollout batch
            total_energies, total_log_probs, mean_entropy = self.rollout_batch(
                batch_data
            )

            # Compute rewards (negative energy - we want to maximize reward = minimize energy)
            rewards = -total_energies

            # Get T_limits for baseline computation
            T_limits = torch.tensor(batch_data["T_limit"], device=self.device)

            # Compute baseline
            baselines = self.compute_baseline(rewards, total_energies, T_limits)

            # Compute advantage
            advantages = rewards - baselines

            # Policy gradient loss
            # loss = -E[advantage * log_prob] - entropy_coef * entropy
            pg_loss = -(advantages * total_log_probs).mean()
            entropy_loss = -self.entropy_coef * mean_entropy
            loss = pg_loss + entropy_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            self.optimizer.step()

            # Track metrics
            epoch_losses.append(loss.item())
            epoch_energies.extend(total_energies.cpu().numpy().tolist())
            epoch_entropies.append(mean_entropy.item())

        # Compute epoch statistics
        metrics = {
            "loss": np.mean(epoch_losses),
            "energy_mean": np.mean(epoch_energies),
            "energy_std": np.std(epoch_energies),
            "entropy": np.mean(epoch_entropies),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

        return metrics

    def adjust_learning_rate(self, epoch: int):
        """Adjust learning rate based on epoch."""
        for decay_epoch in self.lr_decay_epochs:
            if epoch == decay_epoch:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] *= self.lr_decay_factor
                print(
                    f"Learning rate decayed to {self.optimizer.param_groups[0]['lr']:.2e}"
                )

    def train(
        self,
        resume_from: Optional[str] = None,
        save_every: int = None,
    ):
        """
        Train the model.

        Args:
            resume_from: Path to checkpoint to resume from
            save_every: Save checkpoint every N epochs (default from config)
        """
        if save_every is None:
            save_every = self.config.training.save_every

        start_epoch = 0

        # Resume from checkpoint if provided
        if resume_from is not None and os.path.exists(resume_from):
            start_epoch = self.load_checkpoint(resume_from) + 1
            print(f"Resumed from epoch {start_epoch}")

        # Start prefetcher
        self.prefetcher.start()

        print(f"\n{'=' * 60}")
        print(f"PaST-SM Training")
        print(f"{'=' * 60}")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Slack variant: {self.config.env.slack_variant.value}")
        print(f"{'=' * 60}\n")

        try:
            pbar = tqdm(range(start_epoch, self.num_epochs), desc="Training")

            for epoch in pbar:
                # Adjust learning rate
                self.adjust_learning_rate(epoch)

                # Train epoch
                metrics = self.train_epoch(epoch)

                # Update history
                self.history["epoch"].append(epoch)
                self.history["loss"].append(metrics["loss"])
                self.history["energy_mean"].append(metrics["energy_mean"])
                self.history["energy_std"].append(metrics["energy_std"])
                self.history["entropy"].append(metrics["entropy"])
                self.history["lr"].append(metrics["lr"])

                # Update progress bar
                pbar.set_postfix(
                    {
                        "loss": f"{metrics['loss']:.4f}",
                        "energy": f"{metrics['energy_mean']:.2f}±{metrics['energy_std']:.2f}",
                        "H": f"{metrics['entropy']:.3f}",
                    }
                )

                # Save checkpoint
                if (epoch + 1) % save_every == 0:
                    self.save_checkpoint(epoch)

        except KeyboardInterrupt:
            print("\nTraining interrupted!")
        finally:
            # Stop prefetcher
            self.prefetcher.stop()

            # Save final checkpoint
            self.save_checkpoint(self.num_epochs - 1, is_final=True)

    def save_checkpoint(self, epoch: int, is_final: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "history": self.history,
        }

        if is_final:
            path = os.path.join(self.save_dir, "past_sm_final.pt")
        else:
            path = os.path.join(self.save_dir, f"past_sm_epoch{epoch}.pt")

        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str) -> int:
        """Load model checkpoint. Returns the epoch number."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)
        return checkpoint["epoch"]

    def save_history(self, filename: str = None):
        """Save training history to JSON."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"history_{timestamp}.json"

        path = os.path.join(self.save_dir, filename)
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Saved history: {path}")


def train_model(
    config: PaSTConfig = None,
    num_epochs: int = None,
    batch_size: int = None,
    device: str = None,
    save_dir: str = None,
    resume_from: str = None,
) -> Trainer:
    """
    Convenience function to train a PaST-SM model.

    Args:
        config: Configuration (creates default if None)
        num_epochs: Override number of epochs
        batch_size: Override batch size
        device: Override device
        save_dir: Override save directory
        resume_from: Path to resume from

    Returns:
        Trained Trainer instance
    """
    if config is None:
        config = PaSTConfig()

    # Apply overrides
    if num_epochs is not None:
        config.training.num_epochs = num_epochs
    if batch_size is not None:
        config.training.batch_size = batch_size
    if device is not None:
        config.device = device
    if save_dir is not None:
        config.training.save_dir = save_dir

    # Create trainer
    trainer = Trainer(config)

    # Train
    trainer.train(resume_from=resume_from)
    trainer.save_history()

    return trainer


if __name__ == "__main__":
    # Test training
    print("Testing PaST-SM training module...")
    print("=" * 60)

    # Create small test config
    config = PaSTConfig()
    config.training.num_epochs = 5
    config.training.batch_size = 32
    config.training.batches_per_epoch = 2
    config.training.save_dir = "test_checkpoints"
    config.device = "cpu"

    # Use NO_SLACK for simplicity
    config.env.slack_variant = SlackVariant.NO_SLACK

    # Validate config
    config.validate()
    print(f"Config validated: slack_variant={config.env.slack_variant.value}")

    # Create trainer
    trainer = Trainer(config)
    print(f"Trainer created")
    print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")

    # Test single batch rollout
    print("\nTesting single batch rollout...")
    from .sm_benchmark_data import generate_episode_batch

    batch_data = generate_episode_batch(
        batch_size=32,
        config=config.data,
        seed=42,
    )

    trainer.env.reset(batch_data)
    obs = trainer.env._get_obs()

    print(f"Observation shapes:")
    for k, v in obs.items():
        print(f"  {k}: {v.shape}")

    # Run short training test
    print("\nRunning short training test...")
    trainer.train()

    print("\n" + "=" * 60)
    print("Training module test complete!")

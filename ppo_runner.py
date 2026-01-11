"""
PPO Runner for PaST-SM (Period-aware Scheduler Transformer - Single Machine).

Implements:
- GPU-native rollout collection (on-policy)
- GAE(λ) advantage estimation
- Clipped PPO objective with value loss and entropy bonus
- Minibatch updates with gradient clipping

All operations stay on GPU to eliminate CPU↔GPU transfer bottleneck.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class PPOConfig:
    """PPO hyperparameters."""

    # GAE parameters
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # PPO clipping
    clip_eps: float = 0.2
    clip_value: bool = True  # Whether to clip value loss
    clip_value_eps: float = 0.2

    # Loss coefficients
    value_coef: float = 0.5
    entropy_coef: float = 0.01

    # Optimization
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0

    # PPO epochs and minibatches
    ppo_epochs: int = 4
    num_minibatches: int = 8

    # Early stopping on KL divergence
    target_kl: Optional[float] = 0.02

    # Advantage normalization
    normalize_advantages: bool = True


class RolloutBuffer:
    """
    GPU-native rollout buffer for PPO.

    Stores transitions from T_rollout steps across B parallel environments.
    All tensors remain on GPU throughout collection and optimization.
    """

    def __init__(
        self,
        num_envs: int,
        rollout_length: int,
        obs_shapes: Dict[str, Tuple[int, ...]],
        action_dim: int,
        device: torch.device,
    ):
        """
        Initialize rollout buffer.

        Args:
            num_envs: Number of parallel environments (B)
            rollout_length: Steps per rollout (T)
            obs_shapes: Dictionary of observation tensor shapes (without batch dim)
            action_dim: Size of action space
            device: GPU device
        """
        self.num_envs = num_envs
        self.rollout_length = rollout_length
        self.device = device
        self.action_dim = action_dim

        # Allocate observation buffers
        self.obs_buffers = {}
        for name, shape in obs_shapes.items():
            self.obs_buffers[name] = torch.zeros(
                (rollout_length, num_envs) + shape,
                dtype=torch.float32,
                device=device,
            )

        # Allocate other buffers
        self.actions = torch.zeros(
            (rollout_length, num_envs), dtype=torch.long, device=device
        )
        self.log_probs = torch.zeros(
            (rollout_length, num_envs), dtype=torch.float32, device=device
        )
        self.rewards = torch.zeros(
            (rollout_length, num_envs), dtype=torch.float32, device=device
        )
        self.dones = torch.zeros(
            (rollout_length, num_envs), dtype=torch.float32, device=device
        )
        self.values = torch.zeros(
            (rollout_length, num_envs), dtype=torch.float32, device=device
        )

        # Computed after rollout
        self.advantages = torch.zeros(
            (rollout_length, num_envs), dtype=torch.float32, device=device
        )
        self.returns = torch.zeros(
            (rollout_length, num_envs), dtype=torch.float32, device=device
        )

        # Position tracking
        self.pos = 0
        self.full = False

    def reset(self):
        """Reset buffer position for new rollout."""
        self.pos = 0
        self.full = False

    def add(
        self,
        obs: Dict[str, Tensor],
        actions: Tensor,
        log_probs: Tensor,
        rewards: Tensor,
        dones: Tensor,
        values: Tensor,
    ):
        """
        Add a transition to the buffer.

        Args:
            obs: Dictionary of observation tensors [B, ...]
            actions: Actions taken [B]
            log_probs: Log probabilities of actions [B]
            rewards: Rewards received [B]
            dones: Done flags [B]
            values: Value estimates [B]
        """
        for name, tensor in obs.items():
            self.obs_buffers[name][self.pos] = tensor

        self.actions[self.pos] = actions
        self.log_probs[self.pos] = log_probs
        self.rewards[self.pos] = rewards
        self.dones[self.pos] = dones.float()
        self.values[self.pos] = values

        self.pos += 1
        if self.pos == self.rollout_length:
            self.full = True

    def compute_gae(
        self,
        last_values: Tensor,
        last_dones: Tensor,
        gamma: float,
        gae_lambda: float,
    ):
        """
        Compute GAE(λ) advantages and returns.

        Args:
            last_values: Value estimates for final state [B]
            last_dones: Done flags for final state [B]
            gamma: Discount factor
            gae_lambda: GAE lambda parameter

        Note on indexing:
            - dones[t] indicates whether episode terminated AFTER taking action at step t
            - So when computing δ_t = r_t + γ(1-done_t)V_{t+1} - V_t, we use dones[t]
            - This is because if done_t=1, there is no V_{t+1} to bootstrap from
        """
        # Bootstrap from last value
        last_gae = torch.zeros(self.num_envs, device=self.device)

        for t in reversed(range(self.rollout_length)):
            if t == self.rollout_length - 1:
                next_non_terminal = 1.0 - last_dones.float()
                next_values = last_values
            else:
                next_non_terminal = (
                    1.0 - self.dones[t].float()
                )  # Use dones[t], not dones[t+1]
                next_values = self.values[t + 1]

            # TD error: δ_t = r_t + γ(1-done_t)V_{t+1} - V_t
            delta = (
                self.rewards[t]
                + gamma * next_non_terminal * next_values
                - self.values[t]
            )

            # GAE: A_t = δ_t + γλ(1-done_t)A_{t+1}
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        # Returns = advantages + values
        self.returns = self.advantages + self.values

    def get_batches(
        self,
        num_minibatches: int,
        normalize_advantages: bool = True,
    ):
        """
        Generator that yields minibatches for PPO updates.

        Args:
            num_minibatches: Number of minibatches to split data into
            normalize_advantages: Whether to normalize advantages per minibatch

        Yields:
            Dictionary of batched tensors for each minibatch
        """
        batch_size = self.num_envs * self.rollout_length
        minibatch_size = batch_size // num_minibatches

        # Flatten time and env dimensions
        flat_obs = {
            name: tensor.reshape(batch_size, *tensor.shape[2:])
            for name, tensor in self.obs_buffers.items()
        }
        flat_actions = self.actions.reshape(batch_size)
        flat_log_probs = self.log_probs.reshape(batch_size)
        flat_values = self.values.reshape(batch_size)
        flat_advantages = self.advantages.reshape(batch_size)
        flat_returns = self.returns.reshape(batch_size)

        # Random permutation for each epoch
        indices = torch.randperm(batch_size, device=self.device)

        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_indices = indices[start:end]

            mb_obs = {name: tensor[mb_indices] for name, tensor in flat_obs.items()}
            mb_actions = flat_actions[mb_indices]
            mb_log_probs = flat_log_probs[mb_indices]
            mb_values = flat_values[mb_indices]
            mb_advantages = flat_advantages[mb_indices]
            mb_returns = flat_returns[mb_indices]

            # Normalize advantages per minibatch
            if normalize_advantages:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

            yield {
                "obs": mb_obs,
                "actions": mb_actions,
                "log_probs_old": mb_log_probs,
                "values_old": mb_values,
                "advantages": mb_advantages,
                "returns": mb_returns,
            }


class PPORunner:
    """
    PPO training runner with GPU-native rollouts.

    Handles:
    - Rollout collection from batched GPU environment
    - PPO loss computation and optimization
    - Metric tracking and logging
    """

    def __init__(
        self,
        model: nn.Module,
        env,  # GPUBatchSingleMachinePeriodEnv
        ppo_config: PPOConfig,
        device: torch.device,
        obs_shapes: Dict[str, Tuple[int, ...]],
        rollout_length: int = 128,
    ):
        """
        Initialize PPO runner.

        Args:
            model: PaSTSMNet model (outputs logits and values)
            env: GPU batched environment
            ppo_config: PPO hyperparameters
            device: GPU device
            obs_shapes: Dictionary of observation shapes
            rollout_length: Steps per rollout (T_rollout)
        """
        self.model = model
        self.env = env
        self.config = ppo_config
        self.device = device
        self.rollout_length = rollout_length

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=ppo_config.learning_rate,
            eps=1e-5,
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(
            num_envs=env.batch_size,
            rollout_length=rollout_length,
            obs_shapes=obs_shapes,
            action_dim=model.action_dim,
            device=device,
        )

        # Tracking
        self.global_step = 0
        self.update_count = 0

    def collect_rollout(
        self,
        obs: Dict[str, Tensor],
    ) -> Tuple[Dict[str, Tensor], Dict[str, float]]:
        """
        Collect T_rollout steps of experience.

        Args:
            obs: Current observation dictionary

        Returns:
            next_obs: Observation after rollout
            metrics: Dictionary of rollout statistics
        """
        self.buffer.reset()

        episode_rewards = []
        episode_lengths = []
        current_ep_rewards = torch.zeros(self.env.batch_size, device=self.device)
        current_ep_lengths = torch.zeros(self.env.batch_size, device=self.device)

        with torch.no_grad():
            for step in range(self.rollout_length):
                # Get action and value from model
                logits, values = self.model(
                    jobs=obs["jobs"],
                    periods_local=obs["periods"],
                    ctx=obs["ctx"],
                    job_mask=obs.get("job_mask"),
                    period_mask=obs.get("period_mask"),
                    periods_full=obs.get("periods_full"),
                    period_full_mask=obs.get("period_full_mask"),
                )

                # Apply action mask (set invalid to -inf)
                if "action_mask" in obs:
                    action_mask = obs["action_mask"]
                    # action_mask: 1 for valid, 0 for invalid
                    logits = logits.masked_fill(action_mask == 0, float("-inf"))

                # ASSERTION: at least one valid action per env
                assert (
                    torch.isfinite(logits).any(dim=-1).all()
                ), "All actions masked for some environment!"

                # Sample action
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

                # Store transition
                self.buffer.add(
                    obs=obs,
                    actions=actions,
                    log_probs=log_probs,
                    rewards=torch.zeros(
                        self.env.batch_size, device=self.device
                    ),  # Placeholder
                    dones=torch.zeros(self.env.batch_size, device=self.device),
                    values=values.squeeze(-1),
                )

                # Environment step
                next_obs, rewards, dones, info = self.env.step(actions)

                # Update buffer with actual rewards and dones
                self.buffer.rewards[step] = rewards
                self.buffer.dones[step] = dones.float()

                # Track episode statistics
                current_ep_rewards += rewards
                current_ep_lengths += 1

                # Record completed episodes
                done_indices = dones.nonzero(as_tuple=True)[0]
                for idx in done_indices:
                    episode_rewards.append(current_ep_rewards[idx].item())
                    episode_lengths.append(current_ep_lengths[idx].item())

                # Reset tracking for done environments
                current_ep_rewards = current_ep_rewards * (~dones).float()
                current_ep_lengths = current_ep_lengths * (~dones).float()

                obs = next_obs
                self.global_step += self.env.batch_size

        # Compute final values for GAE bootstrap
        with torch.no_grad():
            final_logits, final_values = self.model(
                jobs=obs["jobs"],
                periods_local=obs["periods"],
                ctx=obs["ctx"],
                job_mask=obs.get("job_mask"),
                period_mask=obs.get("period_mask"),
                periods_full=obs.get("periods_full"),
                period_full_mask=obs.get("period_full_mask"),
            )

        # Get done flags from the LAST step of the rollout buffer
        # This is more reliable than env.done_mask which may have been reset
        # by the training wrapper's auto-reset logic
        final_dones = self.buffer.dones[self.rollout_length - 1].bool()

        # Compute GAE
        self.buffer.compute_gae(
            last_values=final_values.squeeze(-1),
            last_dones=final_dones,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        # Compute metrics
        metrics = {
            "rollout/rewards_mean": (
                np.mean(episode_rewards) if episode_rewards else 0.0
            ),
            "rollout/rewards_std": np.std(episode_rewards) if episode_rewards else 0.0,
            "rollout/ep_len_mean": np.mean(episode_lengths) if episode_lengths else 0.0,
            "rollout/num_episodes": len(episode_rewards),
        }

        return obs, metrics

    def update(self) -> Dict[str, float]:
        """
        Perform PPO update using collected rollout.

        Returns:
            metrics: Dictionary of training statistics
        """
        # Tracking for metrics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        approx_kls = []
        clip_fracs = []

        for epoch in range(self.config.ppo_epochs):
            # Check for early stopping on KL
            if self.config.target_kl is not None and len(approx_kls) > 0:
                if (
                    np.mean(approx_kls[-self.config.num_minibatches :])
                    > self.config.target_kl
                ):
                    break

            for batch in self.buffer.get_batches(
                num_minibatches=self.config.num_minibatches,
                normalize_advantages=self.config.normalize_advantages,
            ):
                obs = batch["obs"]
                actions = batch["actions"]
                log_probs_old = batch["log_probs_old"]
                values_old = batch["values_old"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                # Forward pass
                logits, values = self.model(
                    jobs=obs["jobs"],
                    periods_local=obs["periods"],
                    ctx=obs["ctx"],
                    job_mask=obs.get("job_mask"),
                    period_mask=obs.get("period_mask"),
                    periods_full=obs.get("periods_full"),
                    period_full_mask=obs.get("period_full_mask"),
                )

                # Apply action mask
                if "action_mask" in obs:
                    action_mask = obs["action_mask"]
                    logits = logits.masked_fill(action_mask == 0, float("-inf"))

                # ASSERTION: logits should be finite (except -inf for masked)
                assert (
                    torch.isfinite(logits).any(dim=-1).all()
                ), "All actions masked in minibatch!"

                # Compute new log probs and entropy
                dist = torch.distributions.Categorical(logits=logits)
                log_probs_new = dist.log_prob(actions)
                entropy = dist.entropy()

                values = values.squeeze(-1)

                # Policy loss (clipped surrogate)
                ratio = torch.exp(log_probs_new - log_probs_old)
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.config.clip_eps,
                        1.0 + self.config.clip_eps,
                    )
                    * advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (optionally clipped)
                if self.config.clip_value:
                    values_clipped = values_old + torch.clamp(
                        values - values_old,
                        -self.config.clip_value_eps,
                        self.config.clip_value_eps,
                    )
                    value_loss_unclipped = (values - returns) ** 2
                    value_loss_clipped = (values_clipped - returns) ** 2
                    value_loss = (
                        0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * ((values - returns) ** 2).mean()

                # Entropy loss (negative because we want to maximize entropy)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                # ASSERTION: loss should be finite
                assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

                self.optimizer.step()

                # Track metrics (no .item() here, will do batch at end)
                policy_losses.append(policy_loss.detach())
                value_losses.append(value_loss.detach())
                entropy_losses.append(entropy.mean().detach())

                # Approximate KL divergence
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
                    clip_frac = (
                        ((ratio - 1.0).abs() > self.config.clip_eps).float().mean()
                    )
                approx_kls.append(approx_kl.detach())
                clip_fracs.append(clip_frac.detach())

        self.update_count += 1

        # Compute metrics (single sync point)
        metrics = {
            "train/policy_loss": torch.stack(policy_losses).mean().item(),
            "train/value_loss": torch.stack(value_losses).mean().item(),
            "train/entropy": torch.stack(entropy_losses).mean().item(),
            "train/approx_kl": torch.stack(approx_kls).mean().item(),
            "train/clip_frac": torch.stack(clip_fracs).mean().item(),
            "train/grad_norm": (
                grad_norm.item() if isinstance(grad_norm, Tensor) else grad_norm
            ),
            "train/ppo_epochs_actual": epoch + 1,
        }

        return metrics

    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def set_learning_rate(self, lr: float):
        """Set learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def state_dict(self) -> Dict[str, Any]:
        """Get state for checkpointing."""
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "update_count": self.update_count,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Load state from checkpoint."""
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.global_step = state["global_step"]
        self.update_count = state["update_count"]

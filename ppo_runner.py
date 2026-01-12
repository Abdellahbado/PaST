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

        # Transition validity mask (1.0 = real transition, 0.0 = placeholder/ignore)
        self.valid = torch.ones(
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
        valid: Optional[Tensor] = None,
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
        # Store only the observation keys the buffer was configured for.
        # This makes the rollout code robust to extra keys in the env obs dict
        # without silently changing training inputs.
        for name, buffer in self.obs_buffers.items():
            buffer[self.pos] = obs[name]

        self.actions[self.pos] = actions
        self.log_probs[self.pos] = log_probs
        self.rewards[self.pos] = rewards
        self.dones[self.pos] = dones.float()
        self.values[self.pos] = values
        if valid is None:
            self.valid[self.pos] = 1.0
        else:
            self.valid[self.pos] = valid.float()

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
            valid_t = (self.valid[t] > 0.5).float()
            if t == self.rollout_length - 1:
                next_non_terminal = (1.0 - last_dones.float()) * valid_t
                next_values = last_values
            else:
                next_non_terminal = (
                    1.0 - self.dones[t].float()
                ) * valid_t  # Use dones[t], not dones[t+1]
                next_values = self.values[t + 1]

            # TD error: δ_t = r_t + γ(1-done_t)V_{t+1} - V_t
            delta = (
                self.rewards[t]
                + gamma * next_non_terminal * next_values
                - self.values[t]
            ) * valid_t

            # GAE: A_t = δ_t + γλ(1-done_t)A_{t+1}
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            # For invalid/placeholder transitions, force advantage to 0 and break propagation.
            last_gae = torch.where(valid_t > 0.5, last_gae, torch.zeros_like(last_gae))
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
        if batch_size <= 0:
            return

        # Clamp to avoid minibatch_size==0 and to ensure we yield <= num_minibatches.
        # We will yield exactly actual_num_minibatches minibatches.
        actual_num_minibatches = int(min(max(1, num_minibatches), batch_size))

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
        flat_valid = self.valid.reshape(batch_size)

        # Random permutation for each epoch
        indices = torch.randperm(batch_size, device=self.device)
        index_chunks = torch.chunk(indices, actual_num_minibatches)

        for mb_indices in index_chunks:
            if mb_indices.numel() == 0:
                continue

            mb_obs = {name: tensor[mb_indices] for name, tensor in flat_obs.items()}
            mb_actions = flat_actions[mb_indices]
            mb_log_probs = flat_log_probs[mb_indices]
            mb_values = flat_values[mb_indices]
            mb_advantages = flat_advantages[mb_indices]
            mb_returns = flat_returns[mb_indices]
            mb_valid = flat_valid[mb_indices]

            # Normalize advantages per minibatch
            if normalize_advantages:
                valid_rows = mb_valid > 0.5
                if valid_rows.any():
                    v = mb_advantages[valid_rows]
                    v_mean = v.mean()
                    v_std = v.std(unbiased=False)
                    mb_advantages = torch.where(
                        valid_rows,
                        (mb_advantages - v_mean) / (v_std + 1e-8),
                        mb_advantages,
                    )

            yield {
                "obs": mb_obs,
                "actions": mb_actions,
                "log_probs_old": mb_log_probs,
                "values_old": mb_values,
                "advantages": mb_advantages,
                "returns": mb_returns,
                "valid": mb_valid,
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

        # For environments that return cumulative dones (done_mask), we must only count
        # newly done envs once. We detect cumulative behavior per step.
        prev_cumulative_done = torch.zeros(
            self.env.batch_size, dtype=torch.bool, device=self.device
        )

        # Track persistent all-masked condition so we don't repeatedly count a forced terminal
        # state when using pulse-style dones.
        prev_all_masked = torch.zeros(
            self.env.batch_size, dtype=torch.bool, device=self.device
        )

        # Diagnostics: how often a wrapper's returned dones disagree with an exposed done_mask.
        # Count per-(step, env) disagreements (not just per-step).
        dones_done_mask_disagree = 0

        with torch.no_grad():
            for step in range(self.rollout_length):
                job_mask = obs.get("job_mask")
                period_mask = obs.get("period_mask")
                period_full_mask = obs.get("period_full_mask")

                # Model expects boolean masks where True means INVALID/masked-out.
                # Env obs may contain float 0/1 with 1 meaning valid.
                if job_mask is not None and job_mask.dtype is not torch.bool:
                    job_mask = job_mask < 0.5
                if period_mask is not None and period_mask.dtype is not torch.bool:
                    period_mask = period_mask < 0.5
                if (
                    period_full_mask is not None
                    and period_full_mask.dtype is not torch.bool
                ):
                    period_full_mask = period_full_mask < 0.5

                periods_full = obs.get("periods_full")
                if (
                    getattr(
                        getattr(self.model, "model_config", None),
                        "use_global_horizon",
                        False,
                    )
                    and periods_full is None
                ):
                    periods_full = obs["periods"]
                    period_full_mask = period_mask

                # Get action and value from model
                logits, values = self.model(
                    jobs=obs["jobs"],
                    periods_local=obs["periods"],
                    ctx=obs["ctx"],
                    job_mask=job_mask,
                    period_mask=period_mask,
                    periods_full=periods_full,
                    period_full_mask=period_full_mask,
                )

                # Apply action mask (set invalid to -inf)
                if "action_mask" in obs:
                    action_mask = obs["action_mask"]
                    # action_mask: 1 for valid, 0 for invalid
                    logits = logits.masked_fill(action_mask == 0, float("-inf"))

                # Handle infeasible/all-masked states robustly.
                # These can occur if an env is done and not reset, or if the deadline makes
                # every (job, slack) action invalid. Treat as terminal for PPO bookkeeping.
                all_masked = ~torch.isfinite(logits).any(dim=-1)  # (B,)
                if all_masked.any():
                    import warnings

                    warnings.warn(
                        f"All actions masked for {all_masked.sum().item()} envs in rollout step {step}; "
                        "treating as terminal to keep PPO stable."
                    )
                    # Best-effort: prevent stepping those envs if we can reach an underlying env.done_mask.
                    # Avoid mutating a wrapper's own done_mask bookkeeping when it exposes an inner env.
                    if hasattr(self.env, "env") and hasattr(self.env.env, "done_mask"):
                        self.env.env.done_mask = self.env.env.done_mask | all_masked
                    elif hasattr(self.env, "done_mask"):
                        self.env.done_mask = self.env.done_mask | all_masked

                # Sample action safely
                safe_logits = torch.where(
                    all_masked.unsqueeze(-1),
                    torch.zeros_like(logits),
                    logits,
                )

                # Sample action
                dist = torch.distributions.Categorical(logits=safe_logits)
                actions = dist.sample()
                # For all-masked envs, force a placeholder action (won't affect done envs)
                actions = actions.masked_fill(all_masked, 0)
                log_probs = dist.log_prob(actions)
                # Placeholder transitions should not contribute policy data
                log_probs = torch.where(
                    all_masked, torch.zeros_like(log_probs), log_probs
                )

                # Valid transition mask: ignore placeholder rows in update/normalization
                valid = (~all_masked).float()
                valid_bool = valid > 0.5

                # Store transition
                self.buffer.add(
                    obs=obs,
                    actions=actions,
                    log_probs=log_probs,
                    rewards=torch.zeros(
                        self.env.batch_size, device=self.device
                    ),  # Placeholder
                    dones=torch.zeros(self.env.batch_size, device=self.device),
                    values=values.squeeze(-1) * valid,
                    valid=valid,
                )

                # Environment step
                next_obs, rewards, dones_step, info = self.env.step(actions)

                # Dones used for PPO/GAE termination. Ensure all-masked rows terminate even if
                # wrappers don't report done for them.
                dones_for_gae = dones_step | all_masked

                # Prefer a real cumulative done signal if the env exposes done_mask.
                # This avoids heuristic equality checks and lets us detect resets (done clearing).
                cumulative_dones = None
                if (
                    hasattr(self.env, "done_mask")
                    and isinstance(self.env.done_mask, torch.Tensor)
                    and self.env.done_mask.shape == dones_step.shape
                ):
                    cumulative_dones = self.env.done_mask
                elif (
                    hasattr(self.env, "env")
                    and hasattr(self.env.env, "done_mask")
                    and isinstance(self.env.env.done_mask, torch.Tensor)
                    and self.env.env.done_mask.shape == dones_step.shape
                ):
                    cumulative_dones = self.env.env.done_mask

                if cumulative_dones is not None:
                    # Compare against dones_step (before OR-ing all_masked) as suggested.
                    dones_done_mask_disagree += int(
                        (dones_step ^ cumulative_dones).sum().item()
                    )

                # Decide whether to use cumulative done_mask semantics for episode counting.
                # Some wrappers (e.g. full-batch auto-reset) expose a done_mask that resets/clears
                # and may not match the dones returned to PPO.
                use_cumulative = False
                if cumulative_dones is not None:
                    # Cumulative done_mask should be monotone non-decreasing during an episode.
                    nondecreasing = (
                        cumulative_dones | prev_cumulative_done
                    ) == cumulative_dones
                    consistent = torch.equal(dones_for_gae, cumulative_dones)
                    use_cumulative = bool(nondecreasing.all().item() and consistent)

                if use_cumulative:
                    active_before = ~prev_cumulative_done
                    newly_done_base = cumulative_dones & ~prev_cumulative_done
                    prev_cumulative_done.copy_(cumulative_dones)
                else:
                    newly_done_base = dones_step
                    active_before = torch.ones_like(dones_step, dtype=torch.bool)

                # Detect wrappers that reset/clear done_mask before returning (done pulse but done_mask is False).
                # In that case, the next observation is already from a new episode, so we must break
                # the all-masked contiguous-run latch.
                reset_like = torch.zeros_like(dones_step, dtype=torch.bool)
                if cumulative_dones is not None:
                    reset_like = dones_step & (~cumulative_dones)

                # If dones are pulse-style (or wrapper-dependent), all_masked could otherwise be
                # re-counted every step. Only count an all-masked terminal once per contiguous run.
                forced_newly_done = all_masked & active_before & ~prev_all_masked
                newly_done = newly_done_base | forced_newly_done

                # Exclude placeholder (all_masked) steps from episode reward/length stats.
                track_mask = active_before & valid_bool

                # Update buffer with actual rewards and dones
                self.buffer.rewards[step] = torch.where(
                    valid_bool, rewards, torch.zeros_like(rewards)
                )
                self.buffer.dones[step] = dones_for_gae.float()

                # Track episode statistics
                current_ep_rewards[track_mask] += rewards[track_mask]
                current_ep_lengths[track_mask] += 1

                # Record completed episodes
                done_indices = newly_done.nonzero(as_tuple=True)[0]
                for idx in done_indices:
                    episode_rewards.append(current_ep_rewards[idx].item())
                    episode_lengths.append(current_ep_lengths[idx].item())

                # Reset tracking for done environments
                current_ep_rewards = current_ep_rewards * (~newly_done).float()
                current_ep_lengths = current_ep_lengths * (~newly_done).float()

                # Update all-masked run tracker.
                # - Keep it True after a forced terminal so we don't recount on the next step.
                # - Break it when a wrapper has already reset the episode in this step.
                prev_all_masked = all_masked & ~reset_like

                obs = next_obs
                self.global_step += self.env.batch_size

        # Compute final values for GAE bootstrap
        with torch.no_grad():
            job_mask = obs.get("job_mask")
            period_mask = obs.get("period_mask")
            period_full_mask = obs.get("period_full_mask")

            if job_mask is not None and job_mask.dtype is not torch.bool:
                job_mask = job_mask < 0.5
            if period_mask is not None and period_mask.dtype is not torch.bool:
                period_mask = period_mask < 0.5
            if (
                period_full_mask is not None
                and period_full_mask.dtype is not torch.bool
            ):
                period_full_mask = period_full_mask < 0.5

            periods_full = obs.get("periods_full")
            # FULL_GLOBAL expects periods_full for global horizon embedding.
            # In our env, "periods" may already contain the full horizon (K=K_full_max).
            if (
                getattr(
                    getattr(self.model, "model_config", None),
                    "use_global_horizon",
                    False,
                )
                and periods_full is None
            ):
                periods_full = obs["periods"]
                period_full_mask = period_mask

            final_logits, final_values = self.model(
                jobs=obs["jobs"],
                periods_local=obs["periods"],
                ctx=obs["ctx"],
                job_mask=job_mask,
                period_mask=period_mask,
                periods_full=periods_full,
                period_full_mask=period_full_mask,
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
            "rollout/dones_done_mask_disagree": float(dones_done_mask_disagree),
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

        prev_epoch_minibatches = self.config.num_minibatches

        for epoch in range(self.config.ppo_epochs):
            # Check for early stopping on KL
            if self.config.target_kl is not None and len(approx_kls) > 0:
                window = min(int(prev_epoch_minibatches), len(approx_kls))
                if window > 0:
                    recent = approx_kls[-window:]
                    if isinstance(recent[0], Tensor):
                        recent_mean = torch.stack(recent).mean().item()
                    else:
                        recent_mean = float(np.mean(recent))
                    if recent_mean > self.config.target_kl:
                        break

            minibatches_this_epoch = 0

            for batch in self.buffer.get_batches(
                num_minibatches=self.config.num_minibatches,
                normalize_advantages=self.config.normalize_advantages,
            ):
                minibatches_this_epoch += 1
                obs = batch["obs"]
                actions = batch["actions"]
                log_probs_old = batch["log_probs_old"]
                values_old = batch["values_old"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                valid = batch.get("valid", None)

                job_mask = obs.get("job_mask")
                period_mask = obs.get("period_mask")
                period_full_mask = obs.get("period_full_mask")

                if job_mask is not None and job_mask.dtype is not torch.bool:
                    job_mask = job_mask < 0.5
                if period_mask is not None and period_mask.dtype is not torch.bool:
                    period_mask = period_mask < 0.5
                if (
                    period_full_mask is not None
                    and period_full_mask.dtype is not torch.bool
                ):
                    period_full_mask = period_full_mask < 0.5

                # Forward pass
                logits, values = self.model(
                    jobs=obs["jobs"],
                    periods_local=obs["periods"],
                    ctx=obs["ctx"],
                    job_mask=job_mask,
                    period_mask=period_mask,
                    periods_full=obs.get("periods_full"),
                    period_full_mask=period_full_mask,
                )

                # Apply action mask
                if "action_mask" in obs:
                    action_mask = obs["action_mask"]
                    logits = logits.masked_fill(action_mask == 0, float("-inf"))

                # Determine which rows are usable for learning.
                # 1) Must not be a placeholder transition from rollout
                # 2) Must have at least one valid action under the current mask
                has_any_action = torch.isfinite(logits).any(dim=-1)
                if valid is not None:
                    usable = (valid > 0.5) & has_any_action
                else:
                    usable = has_any_action

                if usable.sum().item() == 0:
                    continue

                logits = logits[usable]
                values = values[usable]
                actions = actions[usable]
                log_probs_old = log_probs_old[usable]
                values_old = values_old[usable]
                advantages = advantages[usable]
                returns = returns[usable]

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

            prev_epoch_minibatches = max(1, minibatches_this_epoch)

        self.update_count += 1

        # Compute metrics (single sync point)
        if len(policy_losses) == 0:
            return {
                "train/policy_loss": 0.0,
                "train/value_loss": 0.0,
                "train/entropy": 0.0,
                "train/approx_kl": 0.0,
                "train/clip_frac": 0.0,
                "train/grad_norm": 0.0,
                "train/ppo_epochs_actual": 0,
            }

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

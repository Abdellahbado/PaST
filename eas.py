"""
Efficient Active Search (EAS-Lay) for PaST-SM.

Implements EAS-Lay from:
  "Efficient Active Search for Combinatorial Optimization Problems"
  (Hottung, Kwon, Tierney - ICLR 2022)

Combined with SGBS from:
  "Simulation-guided Beam Search for Neural Combinatorial Optimization"
  (Choo et al. - NeurIPS 2022)

EAS-Lay adds instance-specific residual layers (ψ) to a frozen pre-trained
model (θ), enabling fast test-time adaptation via gradient descent on:
    J = J_RL + λ * J_IL

Where:
    J_RL = REINFORCE gradient with incumbent baseline
    J_IL = Imitation learning on best-known solution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from PaST.past_sm_model import PaSTSMNet
from PaST.sm_env import GPUBatchSingleMachinePeriodEnv


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class EASConfig:
    """Configuration for EAS-Lay algorithm."""

    # EAS Layer architecture
    layer_hidden_dim: int = 64
    """Hidden dimension of the residual MLP."""

    # Optimization
    learning_rate: float = 0.003
    """Learning rate for EAS parameter updates."""

    il_weight: float = 0.01
    """Weight λ for imitation learning loss (J_IL)."""

    # Sampling
    samples_per_iter: int = 32
    """Number of solutions sampled per iteration (M)."""

    max_iterations: int = 100
    """Maximum number of EAS iterations."""

    # Baseline
    baseline_type: str = "incumbent"
    """Baseline for REINFORCE: 'incumbent' or 'mean'."""

    # Convergence
    patience: int = 20
    """Early stopping patience (iterations without improvement)."""

    min_improvement: float = 1e-6
    """Minimum improvement to reset patience counter."""


@dataclass
class EASResult:
    """Result from EAS search."""

    best_energy: float
    """Best total energy found."""

    best_return: float
    """Best return (negative energy) found."""

    best_actions: List[int]
    """Action sequence for best solution."""

    iterations: int
    """Number of iterations performed."""

    improvement_history: List[float] = field(default_factory=list)
    """Energy at each iteration (for convergence analysis)."""

    time_seconds: float = 0.0
    """Wall-clock time for search."""


# =============================================================================
# Helper Functions
# =============================================================================


def _prepare_model_masks(
    obs: Dict[str, Tensor],
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """Convert env float masks (1=valid) into model bool masks (True=invalid)."""
    job_mask = obs.get("job_mask")
    period_mask = obs.get("period_mask")
    period_full_mask = obs.get("period_full_mask")

    if job_mask is not None and job_mask.dtype is not torch.bool:
        job_mask = job_mask < 0.5
    if period_mask is not None and period_mask.dtype is not torch.bool:
        period_mask = period_mask < 0.5
    if period_full_mask is not None and period_full_mask.dtype is not torch.bool:
        period_full_mask = period_full_mask < 0.5

    return job_mask, period_mask, period_full_mask


# =============================================================================
# EAS Residual Layer
# =============================================================================


class EASResidualLayer(nn.Module):
    """
    Residual layer for EAS-Lay.

    Adds a learned perturbation to job embeddings:
        job_emb' = job_emb + MLP(job_emb)

    Architecture:
        Linear(d_model, hidden_dim) -> ReLU -> Linear(hidden_dim, d_model)

    Initialized with zeros so the layer starts as identity.
    """

    def __init__(self, d_model: int, hidden_dim: int = 64):
        super().__init__()

        self.d_model = d_model
        self.hidden_dim = hidden_dim

        # Two-layer MLP
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)

        # Initialize to zeros (identity at start)
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply residual transformation.

        Args:
            x: [batch, M, d_model] or [batch, d_model]

        Returns:
            x + MLP(x)
        """
        residual = F.relu(self.fc1(x))
        residual = self.fc2(residual)
        return x + residual


# =============================================================================
# EAS Model Wrapper
# =============================================================================


class EASModelWrapper(nn.Module):
    """
    Wraps a pre-trained PaSTSMNet with EAS residual layers.

    The original model parameters (θ) are frozen.
    Only the EAS layer parameters (ψ) are trainable.

    Architecture:
        Encoder -> EAS Layer (job_emb) -> Action/Value Heads
    """

    def __init__(
        self,
        base_model: PaSTSMNet,
        eas_config: EASConfig,
    ):
        super().__init__()

        self.base_model = base_model
        self.eas_config = eas_config

        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Get model dimensions
        d_model = base_model.model_config.d_model

        # Create EAS residual layer for job embeddings
        self.eas_job_layer = EASResidualLayer(
            d_model=d_model,
            hidden_dim=eas_config.layer_hidden_dim,
        )

        # Optional: EAS layer for context embedding
        self.eas_ctx_layer = EASResidualLayer(
            d_model=d_model,
            hidden_dim=eas_config.layer_hidden_dim,
        )

        # Copy useful attributes from base model
        self.M_job_bins = base_model.M_job_bins
        self.K_slack = base_model.K_slack
        self.action_dim = base_model.action_dim
        self.config = base_model.config
        self.model_config = base_model.model_config
        self.env_config = base_model.env_config

    def get_eas_parameters(self) -> List[nn.Parameter]:
        """Get only the trainable EAS parameters (ψ)."""
        params = []
        params.extend(self.eas_job_layer.parameters())
        params.extend(self.eas_ctx_layer.parameters())
        return params

    def reset_eas_parameters(self):
        """Reset EAS layers to identity (for new instance)."""
        for layer in [self.eas_job_layer, self.eas_ctx_layer]:
            nn.init.zeros_(layer.fc1.weight)
            nn.init.zeros_(layer.fc1.bias)
            nn.init.zeros_(layer.fc2.weight)
            nn.init.zeros_(layer.fc2.bias)

    def forward(
        self,
        jobs: Tensor,
        periods_local: Tensor,
        ctx: Tensor,
        job_mask: Optional[Tensor] = None,
        period_mask: Optional[Tensor] = None,
        periods_full: Optional[Tensor] = None,
        period_full_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass with EAS residual layers.

        Returns:
            logits: [batch, action_dim]
            value: [batch, 1]
        """
        # Get embeddings from frozen encoder
        job_emb, ctx_emb, global_emb = self.base_model.get_embeddings(
            jobs=jobs,
            periods_local=periods_local,
            ctx=ctx,
            job_mask=job_mask,
            period_mask=period_mask,
            periods_full=periods_full,
            period_full_mask=period_full_mask,
        )

        # Apply EAS residual layers
        job_emb = self.eas_job_layer(job_emb)
        ctx_emb = self.eas_ctx_layer(ctx_emb)

        # Complete forward pass through heads
        logits, value = self.base_model.forward_from_embeddings(
            job_emb=job_emb,
            ctx_emb=ctx_emb,
            global_emb=global_emb,
            job_mask=job_mask,
        )

        return logits, value

    def get_policy(
        self,
        jobs: Tensor,
        periods_local: Tensor,
        ctx: Tensor,
        job_mask: Optional[Tensor] = None,
        period_mask: Optional[Tensor] = None,
        periods_full: Optional[Tensor] = None,
        period_full_mask: Optional[Tensor] = None,
    ) -> torch.distributions.Categorical:
        """Get action distribution."""
        logits, _ = self.forward(
            jobs, periods_local, ctx, job_mask, period_mask,
            periods_full, period_full_mask
        )
        return torch.distributions.Categorical(logits=logits)

    def decode_action(self, action: int) -> Tuple[int, int]:
        """Decode action to (job_idx, slack_idx)."""
        return self.base_model.decode_action(action)


# =============================================================================
# EAS Search Runner
# =============================================================================


class EASRunner:
    """
    Runs Efficient Active Search (EAS-Lay) for a single instance.

    Implements the main EAS loop:
    1. Sample M solutions from current policy
    2. Update incumbent if better solution found
    3. Compute RL loss (REINFORCE) and IL loss (imitation)
    4. Update EAS parameters via gradient descent
    5. Repeat until convergence or max iterations
    """

    def __init__(
        self,
        base_model: PaSTSMNet,
        env_config,
        device: torch.device,
        eas_config: Optional[EASConfig] = None,
    ):
        self.base_model = base_model
        self.env_config = env_config
        self.device = device
        self.eas_config = eas_config or EASConfig()

        # Create wrapped model
        self.eas_model = EASModelWrapper(base_model, self.eas_config)
        self.eas_model.to(device)

    def search(
        self,
        batch_data: Dict[str, Any],
        max_time: Optional[float] = None,
    ) -> EASResult:
        """
        Run EAS search for a single instance.

        Args:
            batch_data: Single-instance batch data (batch_size=1)
            max_time: Optional time limit in seconds

        Returns:
            EASResult with best solution found
        """
        import time
        start_time = time.time()

        config = self.eas_config

        # Reset EAS layers for new instance
        self.eas_model.reset_eas_parameters()

        # Create optimizer for EAS parameters
        optimizer = torch.optim.Adam(
            self.eas_model.get_eas_parameters(),
            lr=config.learning_rate,
        )

        # Create environment
        env = GPUBatchSingleMachinePeriodEnv(
            batch_size=1,
            env_config=self.env_config,
            device=self.device,
        )

        # Get initial incumbent via greedy
        env.reset(batch_data)
        incumbent_return, incumbent_energy, incumbent_actions = self._greedy_rollout(
            env, self.eas_model, record_actions=True
        )

        improvement_history = [float(incumbent_energy)]
        best_return = incumbent_return
        best_energy = incumbent_energy
        best_actions = incumbent_actions

        patience_counter = 0
        iteration = 0

        # Main EAS loop
        for iteration in range(config.max_iterations):
            # Check time limit
            if max_time is not None and (time.time() - start_time) > max_time:
                break

            # Sample M solutions
            all_log_probs = []
            all_returns = []

            for _ in range(config.samples_per_iter):
                env.reset(batch_data)
                ret, energy, actions, step_log_probs = self._sample_rollout(
                    env, self.eas_model
                )

                all_log_probs.append(step_log_probs)
                all_returns.append(ret)

                # Update incumbent if better
                if ret > best_return:
                    best_return = ret
                    best_energy = energy
                    best_actions = actions
                    incumbent_return = ret
                    incumbent_actions = actions

            # Compute RL loss (REINFORCE with baseline)
            if config.baseline_type == "incumbent":
                baseline = incumbent_return
            else:  # mean
                baseline = sum(all_returns) / len(all_returns)

            rl_loss = torch.tensor(0.0, device=self.device)
            for step_lps, ret in zip(all_log_probs, all_returns):
                advantage = ret - baseline
                # Sum log probs for the episode, multiply by advantage
                episode_log_prob = sum(step_lps)
                rl_loss = rl_loss - advantage * episode_log_prob

            rl_loss = rl_loss / len(all_log_probs)

            # Compute IL loss (imitation on incumbent)
            env.reset(batch_data)
            il_loss = self._compute_il_loss(env, self.eas_model, incumbent_actions)

            # Combined loss
            total_loss = rl_loss + config.il_weight * il_loss

            # Gradient update
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Track improvement
            improvement_history.append(float(best_energy))

            # Early stopping check
            if len(improvement_history) >= 2:
                improvement = improvement_history[-2] - improvement_history[-1]
                if improvement < config.min_improvement:
                    patience_counter += 1
                else:
                    patience_counter = 0

            if patience_counter >= config.patience:
                break

        elapsed = time.time() - start_time

        return EASResult(
            best_energy=float(best_energy),
            best_return=float(best_return),
            best_actions=best_actions,
            iterations=iteration + 1,
            improvement_history=improvement_history,
            time_seconds=elapsed,
        )

    def _greedy_rollout(
        self,
        env: GPUBatchSingleMachinePeriodEnv,
        model: nn.Module,
        record_actions: bool = False,
        max_steps: Optional[int] = None,
    ) -> Tuple[float, float, Optional[List[int]]]:
        """
        Greedy rollout (argmax actions).

        Returns:
            return, energy, actions (if recorded)
        """
        if max_steps is None:
            max_steps = int(env.N_job_pad) + 5

        actions_list = [] if record_actions else None
        obs = env._get_obs()

        for _ in range(max_steps):
            if env.done_mask.all():
                break

            job_mask, period_mask, period_full_mask = _prepare_model_masks(obs)

            with torch.no_grad():
                logits, _ = model(
                    jobs=obs["jobs"],
                    periods_local=obs["periods"],
                    ctx=obs["ctx"],
                    job_mask=job_mask,
                    period_mask=period_mask,
                    periods_full=obs.get("periods_full"),
                    period_full_mask=period_full_mask,
                )

                # Apply action mask
                action_mask = obs.get("action_mask")
                if action_mask is not None:
                    logits = logits.masked_fill(action_mask == 0, float("-inf"))

                action = logits.argmax(dim=-1)

            if record_actions:
                actions_list.append(int(action[0].item()))

            obs, reward, done, info = env.step(action)

        total_return = -float(env.total_energy[0].item())
        total_energy = float(env.total_energy[0].item())

        return total_return, total_energy, actions_list

    def _sample_rollout(
        self,
        env: GPUBatchSingleMachinePeriodEnv,
        model: nn.Module,
        max_steps: Optional[int] = None,
    ) -> Tuple[float, float, List[int], List[Tensor]]:
        """
        Sample rollout (stochastic actions).

        Returns:
            return, energy, actions, step_log_probs (for gradient)
        """
        if max_steps is None:
            max_steps = int(env.N_job_pad) + 5

        actions_list = []
        step_log_probs = []
        obs = env._get_obs()

        for _ in range(max_steps):
            if env.done_mask.all():
                break

            job_mask, period_mask, period_full_mask = _prepare_model_masks(obs)

            logits, _ = model(
                jobs=obs["jobs"],
                periods_local=obs["periods"],
                ctx=obs["ctx"],
                job_mask=job_mask,
                period_mask=period_mask,
                periods_full=obs.get("periods_full"),
                period_full_mask=period_full_mask,
            )

            # Apply action mask
            action_mask = obs.get("action_mask")
            if action_mask is not None:
                logits = logits.masked_fill(action_mask == 0, float("-inf"))

            # Sample action
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            actions_list.append(int(action[0].item()))
            step_log_probs.append(log_prob[0])

            obs, reward, done, info = env.step(action)

        total_return = -float(env.total_energy[0].item())
        total_energy = float(env.total_energy[0].item())

        return total_return, total_energy, actions_list, step_log_probs

    def _compute_il_loss(
        self,
        env: GPUBatchSingleMachinePeriodEnv,
        model: nn.Module,
        target_actions: List[int],
    ) -> Tensor:
        """
        Compute imitation learning loss.

        Maximize log-likelihood of target actions (incumbent solution).

        Returns:
            Negative log-likelihood (to minimize)
        """
        obs = env._get_obs()
        total_nll = torch.tensor(0.0, device=self.device)

        for target_action in target_actions:
            if env.done_mask.all():
                break

            job_mask, period_mask, period_full_mask = _prepare_model_masks(obs)

            logits, _ = model(
                jobs=obs["jobs"],
                periods_local=obs["periods"],
                ctx=obs["ctx"],
                job_mask=job_mask,
                period_mask=period_mask,
                periods_full=obs.get("periods_full"),
                period_full_mask=period_full_mask,
            )

            # Apply action mask
            action_mask = obs.get("action_mask")
            if action_mask is not None:
                logits = logits.masked_fill(action_mask == 0, float("-inf"))

            # Log probability of target action
            dist = torch.distributions.Categorical(logits=logits)
            action_tensor = torch.tensor([target_action], device=self.device)
            log_prob = dist.log_prob(action_tensor)

            total_nll = total_nll - log_prob[0]

            # Step environment with target action
            obs, _, _, _ = env.step(action_tensor)

        return total_nll


# =============================================================================
# Batch EAS Runner
# =============================================================================


def eas_batch(
    base_model: PaSTSMNet,
    env_config,
    device: torch.device,
    batch_data: Dict[str, Any],
    eas_config: Optional[EASConfig] = None,
    max_time_per_instance: Optional[float] = None,
) -> List[EASResult]:
    """
    Run EAS on a batch of instances (sequentially).

    Each instance gets its own EAS layer parameters.

    Args:
        base_model: Pre-trained PaSTSMNet model
        env_config: Environment configuration
        device: Torch device
        batch_data: Batched instance data
        eas_config: EAS configuration
        max_time_per_instance: Time limit per instance

    Returns:
        List of EASResult for each instance
    """
    batch_size = batch_data["n_jobs"].shape[0]
    results = []

    runner = EASRunner(base_model, env_config, device, eas_config)

    for i in range(batch_size):
        # Slice single instance
        single_data = {}
        for key, val in batch_data.items():
            if isinstance(val, np.ndarray):
                single_data[key] = val[i:i+1]
            elif torch.is_tensor(val):
                single_data[key] = val[i:i+1]
            else:
                single_data[key] = val

        result = runner.search(single_data, max_time=max_time_per_instance)
        results.append(result)

    return results


# =============================================================================
# Utility Functions
# =============================================================================


def create_eas_model(
    base_model: PaSTSMNet,
    eas_config: Optional[EASConfig] = None,
) -> EASModelWrapper:
    """
    Create an EAS-wrapped model from a pre-trained base model.

    Args:
        base_model: Pre-trained PaSTSMNet
        eas_config: EAS configuration

    Returns:
        EASModelWrapper with frozen base and trainable EAS layers
    """
    config = eas_config or EASConfig()
    return EASModelWrapper(base_model, config)

"""
SGBS + EAS Hybrid Algorithm for PaST-SM.

Implements Algorithm 2 from the SGBS paper:
  - SGBS provides exploration and solution generation
  - EAS updates the policy to favor promising regions
  - Solutions found by SGBS guide EAS via imitation learning

The synergistic combination:
  - SGBS explores promising regions identified by rollouts
  - EAS fine-tunes the policy towards best solutions found
  - Incumbent solution shared between both methods
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from PaST.past_sm_model import PaSTSMNet
from PaST.sm_env import GPUBatchSingleMachinePeriodEnv
from PaST.eas import (
    EASConfig,
    EASModelWrapper,
    EASResult,
)
from PaST.sgbs import (
    sgbs_single_instance,
    greedy_rollout,
    DecodeResult,
)


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
# Configuration
# =============================================================================


@dataclass
class SGBSEASConfig:
    """Configuration for SGBS+EAS hybrid algorithm."""

    # SGBS parameters
    sgbs_beta: int = 4
    """Beam width for SGBS."""

    sgbs_gamma: int = 4
    """Expansion factor for SGBS."""

    # EAS parameters
    eas_learning_rate: float = 0.003
    """Learning rate for EAS parameter updates."""

    eas_il_weight: float = 0.01
    """Weight λ for imitation learning loss."""

    eas_layer_hidden_dim: int = 64
    """Hidden dimension of EAS residual layers."""

    # Sampling
    samples_per_iter: int = 32
    """Number of solutions sampled per EAS iteration (M)."""

    # Search budget
    max_iterations: int = 100
    """Maximum number of SGBS+EAS iterations."""

    # SGBS frequency
    sgbs_frequency: int = 1
    """Run SGBS every N iterations (1 = every iteration)."""

    # Convergence
    patience: int = 20
    """Early stopping patience."""

    # Decoding options (passed to SGBS)
    max_wait_slots: Optional[int] = None
    wait_logit_penalty: float = 0.0
    makespan_penalty: float = 0.0


@dataclass
class SGBSEASResult:
    """Result from SGBS+EAS hybrid search."""

    best_energy: float
    """Best total energy found."""

    best_return: float
    """Best return found."""

    best_actions: List[int]
    """Action sequence for best solution."""

    iterations: int
    """Number of iterations performed."""

    sgbs_calls: int = 0
    """Number of SGBS calls made."""

    improvement_history: List[float] = field(default_factory=list)
    """Energy at each iteration."""

    time_seconds: float = 0.0
    """Total search time."""


# =============================================================================
# SGBS+EAS Hybrid Runner
# =============================================================================


class SGBSEASRunner:
    """
    Runs SGBS+EAS hybrid search for a single instance.

    Implements Algorithm 2:
        1. Initialize incumbent via greedy
        2. For each iteration:
           a. Run SGBS to explore (Line 5)
           b. Sample M solutions (Line 6)
           c. Update incumbent
           d. Compute RL + IL loss (Lines 8-9)
           e. Update EAS parameters (Line 10)
        3. Return best solution found
    """

    def __init__(
        self,
        base_model: PaSTSMNet,
        env_config,
        device: torch.device,
        config: Optional[SGBSEASConfig] = None,
    ):
        self.base_model = base_model
        self.env_config = env_config
        self.device = device
        self.config = config or SGBSEASConfig()

        # Create EAS config from SGBS+EAS config
        self.eas_config = EASConfig(
            layer_hidden_dim=self.config.eas_layer_hidden_dim,
            learning_rate=self.config.eas_learning_rate,
            il_weight=self.config.eas_il_weight,
            samples_per_iter=self.config.samples_per_iter,
            max_iterations=self.config.max_iterations,
            patience=self.config.patience,
        )

        # Create wrapped model
        self.eas_model = EASModelWrapper(base_model, self.eas_config)
        self.eas_model.to(device)

    def search(
        self,
        batch_data: Dict[str, Any],
        max_time: Optional[float] = None,
    ) -> SGBSEASResult:
        """
        Run SGBS+EAS hybrid search for a single instance.

        Args:
            batch_data: Single-instance batch data (batch_size=1)
            max_time: Optional time limit in seconds

        Returns:
            SGBSEASResult with best solution found
        """
        start_time = time.time()
        config = self.config

        # Reset EAS layers for new instance
        self.eas_model.reset_eas_parameters()

        # Create optimizer for EAS parameters
        optimizer = torch.optim.Adam(
            self.eas_model.get_eas_parameters(),
            lr=config.eas_learning_rate,
        )

        # Create environment
        env = GPUBatchSingleMachinePeriodEnv(
            batch_size=1,
            env_config=self.env_config,
            device=self.device,
        )

        # Get initial incumbent via greedy
        env.reset(batch_data)
        inc_return, inc_energy, inc_actions = self._greedy_rollout(
            env, self.eas_model, record_actions=True
        )

        best_return = inc_return
        best_energy = inc_energy
        best_actions = inc_actions

        improvement_history = [float(best_energy)]
        sgbs_calls = 0
        patience_counter = 0

        # Main SGBS+EAS loop (Algorithm 2)
        for iteration in range(config.max_iterations):
            # Check time limit
            if max_time is not None and (time.time() - start_time) > max_time:
                break

            # --- Line 5: SGBS exploration ---
            if iteration % config.sgbs_frequency == 0:
                sgbs_result = self._run_sgbs(batch_data)
                sgbs_calls += 1

                if sgbs_result.total_return > best_return:
                    best_return = sgbs_result.total_return
                    best_energy = sgbs_result.total_energy
                    best_actions = sgbs_result.actions
                    inc_return = best_return
                    inc_actions = best_actions

            # --- Line 6: Sample M solutions ---
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
                    inc_return = ret
                    inc_actions = actions

            # --- Line 7: Update incumbent (already done above) ---

            # --- Line 8: Compute RL loss (REINFORCE) ---
            baseline = inc_return  # Use incumbent as baseline

            rl_loss = torch.tensor(0.0, device=self.device)
            for step_lps, ret in zip(all_log_probs, all_returns):
                advantage = ret - baseline
                episode_log_prob = sum(step_lps)
                rl_loss = rl_loss - advantage * episode_log_prob

            rl_loss = rl_loss / len(all_log_probs)

            # --- Line 9: Compute IL loss (imitation on incumbent) ---
            env.reset(batch_data)
            il_loss = self._compute_il_loss(env, self.eas_model, inc_actions)

            # --- Line 10: Policy gradient update ---
            total_loss = rl_loss + config.eas_il_weight * il_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Track improvement
            improvement_history.append(float(best_energy))

            # Early stopping check
            if len(improvement_history) >= 2:
                improvement = improvement_history[-2] - improvement_history[-1]
                if improvement < 1e-6:
                    patience_counter += 1
                else:
                    patience_counter = 0

            if patience_counter >= config.patience:
                break

        elapsed = time.time() - start_time

        return SGBSEASResult(
            best_energy=float(best_energy),
            best_return=float(best_return),
            best_actions=best_actions,
            iterations=iteration + 1,
            sgbs_calls=sgbs_calls,
            improvement_history=improvement_history,
            time_seconds=elapsed,
        )

    def _run_sgbs(self, batch_data: Dict[str, Any]) -> DecodeResult:
        """Run SGBS with current EAS model."""
        # Temporarily switch to eval mode for SGBS
        self.eas_model.eval()

        with torch.no_grad():
            result = sgbs_single_instance(
                model=self.eas_model,
                env_config=self.env_config,
                device=self.device,
                batch_data_single=batch_data,
                beta=self.config.sgbs_beta,
                gamma=self.config.sgbs_gamma,
                max_wait_slots=self.config.max_wait_slots,
                wait_logit_penalty=self.config.wait_logit_penalty,
                makespan_penalty=self.config.makespan_penalty,
            )

        # Back to train mode
        self.eas_model.train()

        return result

    def _greedy_rollout(
        self,
        env: GPUBatchSingleMachinePeriodEnv,
        model: nn.Module,
        record_actions: bool = False,
        max_steps: Optional[int] = None,
    ) -> Tuple[float, float, Optional[List[int]]]:
        """Greedy rollout (argmax actions)."""
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
        """Sample rollout (stochastic actions)."""
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

            action_mask = obs.get("action_mask")
            if action_mask is not None:
                logits = logits.masked_fill(action_mask == 0, float("-inf"))

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
        """Compute imitation learning loss."""
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

            action_mask = obs.get("action_mask")
            if action_mask is not None:
                logits = logits.masked_fill(action_mask == 0, float("-inf"))

            dist = torch.distributions.Categorical(logits=logits)
            action_tensor = torch.tensor([target_action], device=self.device)
            log_prob = dist.log_prob(action_tensor)

            total_nll = total_nll - log_prob[0]

            obs, _, _, _ = env.step(action_tensor)

        return total_nll


# =============================================================================
# Batch Functions
# =============================================================================


def sgbs_eas_batch(
    base_model: PaSTSMNet,
    env_config,
    device: torch.device,
    batch_data: Dict[str, Any],
    config: Optional[SGBSEASConfig] = None,
    max_time_per_instance: Optional[float] = None,
) -> List[SGBSEASResult]:
    """
    Run SGBS+EAS hybrid on a batch of instances (sequentially).

    Each instance gets its own EAS layer parameters.

    Args:
        base_model: Pre-trained PaSTSMNet model
        env_config: Environment configuration
        device: Torch device
        batch_data: Batched instance data
        config: SGBS+EAS configuration
        max_time_per_instance: Time limit per instance

    Returns:
        List of SGBSEASResult for each instance
    """
    batch_size = batch_data["n_jobs"].shape[0]
    results = []

    runner = SGBSEASRunner(base_model, env_config, device, config)

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


def sgbs_eas_single(
    base_model: PaSTSMNet,
    env_config,
    device: torch.device,
    batch_data_single: Dict[str, Any],
    config: Optional[SGBSEASConfig] = None,
    max_time: Optional[float] = None,
) -> SGBSEASResult:
    """
    Run SGBS+EAS hybrid on a single instance.

    Convenience function for single-instance search.

    Args:
        base_model: Pre-trained PaSTSMNet model
        env_config: Environment configuration
        device: Torch device
        batch_data_single: Single-instance batch data
        config: SGBS+EAS configuration
        max_time: Time limit in seconds

    Returns:
        SGBSEASResult with best solution
    """
    runner = SGBSEASRunner(base_model, env_config, device, config)
    return runner.search(batch_data_single, max_time=max_time)

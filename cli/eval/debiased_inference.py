"""Inference-time model debiasing techniques for Q-Sequence models.

When a trained model is worse than random (due to bias), these techniques
can help at inference time without retraining:

1. **Temperature scaling**: Soften Q-values to encourage exploration
2. **Epsilon-mixing**: Mix model actions with random/SPT with probability ε  
3. **Ensemble with random**: Average Q-values with random noise
4. **Multi-restart sampling**: Sample N solutions and take best
5. **Diversity SGBS**: Force diverse candidates in beam search

Usage:
    from PaST.cli.eval.debiased_inference import (
        DebiasedQModel,
        multi_restart_decode,
        ensemble_q_model,
    )
    
    # Wrap model with debiasing
    debiased = DebiasedQModel(model, temperature=2.0, epsilon=0.3)
    
    # Or use multi-restart sampling
    results = multi_restart_decode(model, data, n_restarts=16)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import random

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from PaST.baselines_sequence_dp import dp_schedule_for_job_sequence, DPResult
from PaST.sequence_env import GPUBatchSequenceEnv
from PaST.batch_dp_solver import BatchSequenceDPSolver


class DebiasedQModel(nn.Module):
    """Wrapper that applies debiasing techniques to Q-values at inference time.
    
    Techniques:
    - temperature: Scale Q-values before argmin (higher = more exploration)
    - epsilon: Mix with random/SPT actions with this probability
    - noise_scale: Add Gaussian noise to Q-values
    - entropy_bonus: Add entropy term to encourage diversity
    
    Args:
        base_model: The trained Q-sequence model
        temperature: Temperature for softening Q-values (default=1.0, higher=more random)
        epsilon: Probability of using alternative action (0=always model, 1=always alternative)
        epsilon_policy: What to use when epsilon triggers ("random", "spt", "lpt")
        noise_scale: Std of Gaussian noise to add to Q-values
        random_weight: Weight for mixing with uniform random Q-values (0-1)
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        temperature: float = 1.0,
        epsilon: float = 0.0,
        epsilon_policy: str = "random",
        noise_scale: float = 0.0,
        random_weight: float = 0.0,
        seed: int = 42,
    ):
        super().__init__()
        self.base_model = base_model
        self.temperature = float(temperature)
        self.epsilon = float(epsilon)
        self.epsilon_policy = epsilon_policy.lower()
        self.noise_scale = float(noise_scale)
        self.random_weight = float(random_weight)
        self.rng = np.random.RandomState(seed)
        
        # Freeze base model
        for p in self.base_model.parameters():
            p.requires_grad = False
    
    def forward(
        self,
        jobs: Tensor,
        periods: Tensor,
        ctx: Tensor,
        job_mask: Optional[Tensor] = None,
        processing_times: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """Forward pass with debiasing applied.
        
        Args:
            jobs: (B, N, F) job features
            periods: (B, K, F) period features
            ctx: (B, F_ctx) context
            job_mask: (B, N) True=invalid
            processing_times: (B, N) optional, for SPT/LPT epsilon policy
            
        Returns:
            q_values: (B, N) debiased Q-values
        """
        # Get base Q-values
        with torch.no_grad():
            q_values = self.base_model(jobs, periods, ctx, job_mask, **kwargs)
        
        B, N = q_values.shape
        device = q_values.device
        
        # Apply temperature scaling
        if self.temperature != 1.0:
            q_values = q_values / self.temperature
        
        # Add noise
        if self.noise_scale > 0:
            noise = torch.from_numpy(
                self.rng.randn(B, N).astype(np.float32) * self.noise_scale
            ).to(device)
            q_values = q_values + noise
        
        # Mix with random
        if self.random_weight > 0:
            random_q = torch.from_numpy(
                self.rng.randn(B, N).astype(np.float32)
            ).to(device)
            q_values = (1 - self.random_weight) * q_values + self.random_weight * random_q
        
        # Apply mask to invalid jobs
        if job_mask is not None:
            q_values = q_values.masked_fill(job_mask, float("inf"))
        
        return q_values
    
    def get_action(
        self,
        q_values: Tensor,
        job_mask: Optional[Tensor] = None,
        processing_times: Optional[Tensor] = None,
    ) -> Tensor:
        """Get action with epsilon-greedy exploration.
        
        Args:
            q_values: (B, N) Q-values (lower is better)
            job_mask: (B, N) True=invalid
            processing_times: (B, N) for SPT/LPT policy
            
        Returns:
            actions: (B,) selected job indices
        """
        B, N = q_values.shape
        device = q_values.device
        
        # Model action: argmin Q
        model_action = q_values.argmin(dim=-1)
        
        # If no epsilon, just return model action
        if self.epsilon <= 0:
            return model_action
        
        # Determine alternative actions
        if self.epsilon_policy == "random":
            # Uniform random among valid
            valid_mask = ~job_mask if job_mask is not None else torch.ones(B, N, dtype=torch.bool, device=device)
            random_scores = torch.from_numpy(
                self.rng.rand(B, N).astype(np.float32)
            ).to(device)
            random_scores = random_scores.masked_fill(~valid_mask, float("-inf"))
            alt_action = random_scores.argmax(dim=-1)
            
        elif self.epsilon_policy == "spt" and processing_times is not None:
            # Shortest processing time
            spt_scores = processing_times.float()
            if job_mask is not None:
                spt_scores = spt_scores.masked_fill(job_mask, float("inf"))
            alt_action = spt_scores.argmin(dim=-1)
            
        elif self.epsilon_policy == "lpt" and processing_times is not None:
            # Longest processing time
            lpt_scores = processing_times.float()
            if job_mask is not None:
                lpt_scores = lpt_scores.masked_fill(job_mask, float("-inf"))
            alt_action = lpt_scores.argmax(dim=-1)
            
        else:
            # Fallback to random
            valid_mask = ~job_mask if job_mask is not None else torch.ones(B, N, dtype=torch.bool, device=device)
            random_scores = torch.from_numpy(
                self.rng.rand(B, N).astype(np.float32)
            ).to(device)
            random_scores = random_scores.masked_fill(~valid_mask, float("-inf"))
            alt_action = random_scores.argmax(dim=-1)
        
        # Decide per-sample whether to use model or alternative
        use_alt = torch.from_numpy(
            (self.rng.rand(B) < self.epsilon).astype(np.float32)
        ).to(device).bool()
        
        action = torch.where(use_alt, alt_action, model_action)
        return action
    
    def to(self, device):
        self.base_model = self.base_model.to(device)
        return self
    
    def eval(self):
        self.base_model.eval()
        return self


def multi_restart_decode(
    model: nn.Module,
    variant_config,
    single_data: Dict[str, Any],
    device: torch.device,
    n_restarts: int = 16,
    temperature: float = 1.5,
    seed: int = 42,
) -> DPResult:
    """Sample multiple solutions with stochastic decoding and return best.
    
    This is a simple but effective approach when the model is biased:
    sample many diverse solutions and pick the one with lowest DP cost.
    
    Args:
        model: Q-sequence model
        variant_config: Variant configuration
        single_data: Single instance data
        device: Torch device
        n_restarts: Number of solutions to sample
        temperature: Temperature for softmax sampling (higher = more random)
        seed: Random seed
        
    Returns:
        DPResult for the best solution found
    """
    env_config = variant_config.env
    n_jobs = int(single_data["n_jobs"][0])
    N_pad = env_config.N_job_pad
    
    rng = np.random.RandomState(seed)
    
    # Repeat data for batched sampling
    batch_data = {}
    for k, v in single_data.items():
        if isinstance(v, np.ndarray):
            batch_data[k] = np.repeat(v, n_restarts, axis=0)
        elif torch.is_tensor(v):
            batch_data[k] = v.repeat(n_restarts, *([1] * (v.dim() - 1)))
        else:
            batch_data[k] = v
    
    # Create batched env
    env = GPUBatchSequenceEnv(
        batch_size=n_restarts,
        env_config=env_config,
        device=device,
    )
    env.reset(batch_data)
    
    model.eval()
    with torch.no_grad():
        for step in range(n_jobs):
            obs = env._get_obs()
            job_mask = obs.get("job_mask", env.job_available)
            job_mask = job_mask < 0.5 if job_mask.dtype != torch.bool else ~job_mask
            
            # Get Q-values
            q_values = model(
                obs["jobs"],
                obs["periods"],
                obs["ctx"],
                job_mask,
            )
            q_values = q_values.masked_fill(job_mask, float("inf"))
            
            # Convert to logits (negative Q / temperature)
            logits = -q_values / temperature
            logits = logits.masked_fill(job_mask, float("-inf"))
            
            # Sample actions stochastically
            probs = torch.softmax(logits, dim=-1)
            # Add small uniform noise to break ties and ensure diversity
            noise = torch.from_numpy(
                rng.rand(n_restarts, probs.shape[1]).astype(np.float32) * 1e-6
            ).to(device)
            probs = probs + noise
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            dist = torch.distributions.Categorical(probs=probs)
            actions = dist.sample()
            
            _, _, done, _ = env.step(actions)
            if done.all():
                break
    
    # Score all sequences with batch DP
    energies = BatchSequenceDPSolver.solve(
        job_sequences=env.job_sequences,
        processing_times=env.p_subset,
        ct=env.ct,
        e_single=env.e_single,
        T_limit=env.T_limit,
        sequence_lengths=env.n_jobs.long(),
    )
    
    # Return best
    best_idx = int(energies.argmin().item())
    best_seq = env.job_sequences[best_idx, :n_jobs].cpu().tolist()
    
    return dp_schedule_for_job_sequence(single_data, [int(x) for x in best_seq])


def hybrid_model_random_decode(
    model: nn.Module,
    variant_config,
    single_data: Dict[str, Any],
    device: torch.device,
    n_samples: int = 32,
    model_prob: float = 0.5,
    seed: int = 42,
) -> DPResult:
    """Hybrid decoding: each step picks model or random with probability.
    
    This can help when model is biased but has some useful signal.
    
    Args:
        model: Q-sequence model
        variant_config: Variant configuration  
        single_data: Single instance data
        device: Torch device
        n_samples: Number of parallel samples
        model_prob: Probability of using model action (vs random)
        seed: Random seed
        
    Returns:
        DPResult for best solution
    """
    env_config = variant_config.env
    n_jobs = int(single_data["n_jobs"][0])
    
    rng = np.random.RandomState(seed)
    
    # Repeat data
    batch_data = {}
    for k, v in single_data.items():
        if isinstance(v, np.ndarray):
            batch_data[k] = np.repeat(v, n_samples, axis=0)
        elif torch.is_tensor(v):
            batch_data[k] = v.repeat(n_samples, *([1] * (v.dim() - 1)))
        else:
            batch_data[k] = v
    
    env = GPUBatchSequenceEnv(
        batch_size=n_samples,
        env_config=env_config,
        device=device,
    )
    env.reset(batch_data)
    
    model.eval()
    with torch.no_grad():
        for step in range(n_jobs):
            obs = env._get_obs()
            job_mask = obs.get("job_mask", env.job_available)
            job_mask = job_mask < 0.5 if job_mask.dtype != torch.bool else ~job_mask
            
            # Get Q-values from model
            q_values = model(
                obs["jobs"],
                obs["periods"],
                obs["ctx"],
                job_mask,
            )
            q_values = q_values.masked_fill(job_mask, float("inf"))
            model_actions = q_values.argmin(dim=-1)  # [B]
            
            # Get random actions
            valid_mask = ~job_mask
            random_scores = torch.from_numpy(
                rng.rand(n_samples, q_values.shape[1]).astype(np.float32)
            ).to(device)
            random_scores = random_scores.masked_fill(job_mask, float("-inf"))
            random_actions = random_scores.argmax(dim=-1)  # [B]
            
            # Decide per-sample
            use_model = torch.from_numpy(
                (rng.rand(n_samples) < model_prob).astype(np.float32)
            ).to(device).bool()
            
            actions = torch.where(use_model, model_actions, random_actions)
            
            _, _, done, _ = env.step(actions)
            if done.all():
                break
    
    # Score and return best
    energies = BatchSequenceDPSolver.solve(
        job_sequences=env.job_sequences,
        processing_times=env.p_subset,
        ct=env.ct,
        e_single=env.e_single,
        T_limit=env.T_limit,
        sequence_lengths=env.n_jobs.long(),
    )
    
    best_idx = int(energies.argmin().item())
    best_seq = env.job_sequences[best_idx, :n_jobs].cpu().tolist()
    
    return dp_schedule_for_job_sequence(single_data, [int(x) for x in best_seq])


def sgbs_with_random_restarts(
    model: nn.Module,
    variant_config,
    single_data: Dict[str, Any],
    device: torch.device,
    beta: int = 4,
    gamma: int = 4,
    n_restarts: int = 4,
    seed: int = 42,
) -> DPResult:
    """Run SGBS multiple times with different random completions, return best.
    
    Uses mix_model_random rollout policy with different seeds.
    """
    from PaST.cli.eval.run_eval_q_sequence import sgbs_q_sequence
    
    best_result = None
    best_energy = float("inf")
    
    for i in range(n_restarts):
        # Run SGBS with mixed rollout
        results = sgbs_q_sequence(
            model=model,
            variant_config=variant_config,
            batch_data=single_data,
            device=device,
            beta=beta,
            gamma=gamma,
            rollout_policy="mix_model_random",
            rollout_mix_prob=0.5,
            rollout_seed=seed + i * 1000,
        )
        
        result = results[0]
        if result.total_energy < best_energy:
            best_energy = result.total_energy
            best_result = result
    
    return best_result


# Convenience function to get best debiasing config
def auto_debias_config(model_gap_vs_random: float) -> Dict[str, Any]:
    """Suggest debiasing parameters based on model's gap vs random.
    
    Args:
        model_gap_vs_random: How much worse model is vs random (positive = worse)
        
    Returns:
        Dict of suggested parameters
    """
    if model_gap_vs_random <= 0:
        # Model is better than random, light debiasing
        return {
            "temperature": 1.0,
            "epsilon": 0.0,
            "n_restarts": 1,
            "use_multi_restart": False,
        }
    elif model_gap_vs_random < 0.05:
        # Slightly worse, moderate debiasing
        return {
            "temperature": 1.5,
            "epsilon": 0.1,
            "n_restarts": 8,
            "use_multi_restart": True,
        }
    elif model_gap_vs_random < 0.15:
        # Noticeably worse, aggressive debiasing
        return {
            "temperature": 2.0,
            "epsilon": 0.3,
            "n_restarts": 16,
            "use_multi_restart": True,
        }
    else:
        # Much worse, very aggressive
        return {
            "temperature": 3.0,
            "epsilon": 0.5,
            "n_restarts": 32,
            "use_multi_restart": True,
            "random_weight": 0.3,  # Heavily mix with random
        }

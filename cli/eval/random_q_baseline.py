"""Random Q-value baseline for Q-sequence comparison.

This module provides a baseline that uses random Q-values for job ordering,
serving as a sanity check and lower bound for learned Q-sequence models.
The random Q baseline can be used with:
- Greedy decoding (argmin Q)
- SGBS (beam search with random Q-values)

Usage:
    from PaST.cli.eval.random_q_baseline import RandomQModel
    
    random_model = RandomQModel(seed=42)
    # Use with greedy_decode_q_sequence or sgbs_q_sequence
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


class RandomQModel:
    """Q-sequence model that returns random Q-values.
    
    This baseline helps verify that trained models actually learn something
    useful beyond random ordering. Since Q-values represent "cost-to-go"
    (lower is better), we sample from a standard normal distribution.
    
    Args:
        seed: Random seed for reproducibility
        scale: Standard deviation for random Q-values (default 1.0)
    """
    
    def __init__(self, seed: int = 42, scale: float = 1.0):
        self.seed = int(seed)
        self.scale = float(scale)
        self.rng = np.random.RandomState(self.seed)
        
    def __call__(
        self,
        jobs: Tensor,
        periods: Tensor,
        ctx: Tensor,
        job_mask: Optional[Tensor] = None,
        **kwargs
    ) -> Tuple[Tensor, None]:
        """Forward pass returning random Q-values.
        
        Args:
            jobs: (B, N, job_dim) job features
            periods: (B, K, period_dim) period features (ignored for random Q)
            ctx: (B, ctx_dim) context features  
            job_mask: (B, N) bool mask (True = invalid job)
            **kwargs: Ignored (for compatibility with QSequenceNet)
            
        Returns:
            q_values: (B, N) random Q-values (lower is better)
            aux: None (no auxiliary outputs)
        """
        batch_size, n_jobs, _ = jobs.shape
        device = jobs.device
        
        # Generate random Q-values
        q_values = torch.from_numpy(
            self.rng.randn(batch_size, n_jobs).astype(np.float32) * self.scale
        ).to(device)
        
        # Ensure masked jobs have high Q-values (won't be selected)
        if job_mask is not None:
            q_values = q_values.masked_fill(job_mask, float('inf'))
        
        return q_values
    
    def eval(self):
        """Compatibility method (no-op for random model)."""
        return self
    
    def to(self, device):
        """Compatibility method (random model has no parameters)."""
        return self
    
    def __repr__(self) -> str:
        return f"RandomQModel(seed={self.seed}, scale={self.scale})"


def create_random_q_model(seed: int = 42, scale: float = 1.0) -> RandomQModel:
    """Factory function to create a random Q model.
    
    Args:
        seed: Random seed for reproducibility
        scale: Standard deviation for random Q-values
        
    Returns:
        RandomQModel instance
    """
    return RandomQModel(seed=seed, scale=scale)

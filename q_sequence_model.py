"""
Q-value Model for Sequence Scheduling with Dueling Architecture.

This module extends the PaST encoder with a dueling Q-head for learning
policy-evaluation Q-values: Q(s, j) = expected DP cost if job j is chosen
next in state s, then the sequence is completed with a fixed rollout policy.

Key features:
- Dueling decomposition: Q(s,j) = V(s) + A(s,j) - mean(A(s,:))
- Per-job Q-scores for ranking (used in SGBS expansion)
- Temperature-scaled conversion to logits for action sampling
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from PaST.config import VariantConfig, ModelConfig, EnvConfig
from PaST.past_sm_model import PaSTEncoder


class DuelingQHead(nn.Module):
    """
    Dueling Q-head: Q(s,j) = V(s) + A(s,j) - mean(A(s,:))

    This decomposition stabilizes learning when action advantages are small
    (which is common when DP makes many sequences similar).

    Outputs per-job Q-values where lower Q = better (since Q = expected cost).
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 256,
        use_global_horizon: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.use_global_horizon = use_global_horizon

        # Value stream: V(s) scalar
        # Input: pooled job embeddings + ctx [+ global]
        v_input_dim = d_model * 2
        if use_global_horizon:
            v_input_dim += d_model

        self.value_stream = nn.Sequential(
            nn.Linear(v_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Advantage stream: A(s,j) per job
        # Input: per-job embedding + ctx [+ global]
        a_input_dim = d_model * 2
        if use_global_horizon:
            a_input_dim += d_model

        self.advantage_stream = nn.Sequential(
            nn.Linear(a_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Projection for pooled jobs
        self.pool_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        job_emb: Tensor,  # [B, N, d_model]
        ctx_emb: Tensor,  # [B, d_model]
        global_emb: Optional[Tensor] = None,  # [B, d_model]
        job_mask: Optional[Tensor] = None,  # [B, N] True for invalid
    ) -> Tensor:
        """
        Compute Q-values for each job.

        Returns:
            q_values: [B, N] Q-values (lower = better cost)
        """
        B, N, d = job_emb.shape

        # === Value stream ===
        # Pool job embeddings (masked mean)
        if job_mask is not None:
            valid_mask = (~job_mask).float().unsqueeze(-1)  # [B, N, 1]
            job_pooled = (job_emb * valid_mask).sum(dim=1) / valid_mask.sum(
                dim=1
            ).clamp(min=1)
        else:
            job_pooled = job_emb.mean(dim=1)

        job_pooled = self.pool_proj(job_pooled)  # [B, d_model]

        # Build value input
        if self.use_global_horizon and global_emb is not None:
            v_input = torch.cat([job_pooled, ctx_emb, global_emb], dim=-1)
        else:
            v_input = torch.cat([job_pooled, ctx_emb], dim=-1)

        value = self.value_stream(v_input)  # [B, 1]

        # === Advantage stream ===
        # Per-job advantages
        ctx_expanded = ctx_emb.unsqueeze(1).expand(-1, N, -1)  # [B, N, d_model]

        if self.use_global_horizon and global_emb is not None:
            global_expanded = global_emb.unsqueeze(1).expand(-1, N, -1)
            a_input = torch.cat([job_emb, ctx_expanded, global_expanded], dim=-1)
        else:
            a_input = torch.cat([job_emb, ctx_expanded], dim=-1)

        advantages = self.advantage_stream(a_input).squeeze(-1)  # [B, N]

        # === Dueling combination ===
        # Q(s,j) = V(s) + A(s,j) - mean(A(s,:))
        # Only average over valid jobs
        if job_mask is not None:
            valid_mask_2d = (~job_mask).float()  # [B, N]
            adv_mean = (advantages * valid_mask_2d).sum(
                dim=1, keepdim=True
            ) / valid_mask_2d.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            adv_mean = advantages.mean(dim=1, keepdim=True)

        q_values = value + advantages - adv_mean  # [B, N]

        # Mask invalid jobs with large cost
        if job_mask is not None:
            q_values = q_values.masked_fill(job_mask, float("inf"))

        return q_values


class QSequenceNet(nn.Module):
    """
    Q-value network for sequence scheduling.

    Uses the PaST encoder backbone with a dueling Q-head.
    Outputs per-job Q-values representing expected DP cost.
    """

    def __init__(self, config: VariantConfig):
        super().__init__()

        self.config = config
        self.model_config = config.model
        self.env_config = config.env

        d_model = self.model_config.d_model
        N = self.env_config.M_job_bins  # Max jobs

        # Encoder (shared backbone)
        self.encoder = PaSTEncoder(self.model_config, self.env_config)

        # Dueling Q-head
        self.q_head = DuelingQHead(
            d_model=d_model,
            hidden_dim=256,
            use_global_horizon=self.model_config.use_global_horizon,
        )

        # Store dimensions
        self.N_jobs = N
        self.action_dim = N  # For sequencing: action = job index

    def forward(
        self,
        jobs: Tensor,
        periods_local: Tensor,
        ctx: Tensor,
        job_mask: Optional[Tensor] = None,
        period_mask: Optional[Tensor] = None,
        periods_full: Optional[Tensor] = None,
        period_full_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass to compute Q-values.

        Returns:
            q_values: [B, N] per-job Q-values (lower = better)
        """
        # Encode
        job_emb, ctx_emb, global_emb = self.encoder(
            jobs=jobs,
            periods_local=periods_local,
            ctx=ctx,
            job_mask=job_mask,
            period_mask=period_mask,
            periods_full=periods_full,
            period_full_mask=period_full_mask,
        )

        # Q-values
        q_values = self.q_head(job_emb, ctx_emb, global_emb, job_mask)

        return q_values

    def get_logits(
        self,
        jobs: Tensor,
        periods_local: Tensor,
        ctx: Tensor,
        job_mask: Optional[Tensor] = None,
        period_mask: Optional[Tensor] = None,
        periods_full: Optional[Tensor] = None,
        period_full_mask: Optional[Tensor] = None,
        temperature: float = 1.0,
    ) -> Tensor:
        """
        Convert Q-values to logits for action sampling/SGBS.

        logits = -Q / temperature  (lower Q = higher probability)

        Returns:
            logits: [B, N] action logits
        """
        q_values = self.forward(
            jobs,
            periods_local,
            ctx,
            job_mask,
            period_mask,
            periods_full,
            period_full_mask,
        )

        # Convert Q to logits: lower Q = better = higher logit
        logits = -q_values / temperature

        # Mask invalid jobs
        if job_mask is not None:
            logits = logits.masked_fill(job_mask, float("-inf"))

        return logits

    def get_greedy_action(
        self,
        jobs: Tensor,
        periods_local: Tensor,
        ctx: Tensor,
        job_mask: Optional[Tensor] = None,
        period_mask: Optional[Tensor] = None,
        periods_full: Optional[Tensor] = None,
        period_full_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Get greedy action (job with lowest Q-value).

        Returns:
            actions: [B] job indices
        """
        q_values = self.forward(
            jobs,
            periods_local,
            ctx,
            job_mask,
            period_mask,
            periods_full,
            period_full_mask,
        )

        # argmin over Q (lower Q = better)
        return q_values.argmin(dim=-1)

    def get_value(
        self,
        jobs: Tensor,
        periods_local: Tensor,
        ctx: Tensor,
        job_mask: Optional[Tensor] = None,
        period_mask: Optional[Tensor] = None,
        periods_full: Optional[Tensor] = None,
        period_full_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Get state value V(s) = min_j Q(s,j).

        Returns:
            value: [B] state values
        """
        q_values = self.forward(
            jobs,
            periods_local,
            ctx,
            job_mask,
            period_mask,
            periods_full,
            period_full_mask,
        )

        # V(s) = min_j Q(s,j) over valid jobs
        if job_mask is not None:
            q_values = q_values.masked_fill(job_mask, float("inf"))

        return q_values.min(dim=-1).values


def build_q_model(config: VariantConfig) -> QSequenceNet:
    """Build Q-value model from variant configuration."""
    return QSequenceNet(config)


# =============================================================================
# Compatibility wrapper for SGBS
# =============================================================================


class QModelWrapper(nn.Module):
    """
    Wrapper that makes QSequenceNet compatible with SGBS/greedy_rollout.

    SGBS expects a model with forward() returning (logits, values).
    This wrapper converts Q-values to that interface.
    """

    def __init__(self, q_model: QSequenceNet, temperature: float = 1.0):
        super().__init__()
        self.q_model = q_model
        self.temperature = temperature
        self.action_dim = q_model.action_dim

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
        Returns:
            logits: [B, N] action logits (from -Q/tau)
            values: [B, 1] state values (min Q, negated for reward convention)
        """
        q_values = self.q_model.forward(
            jobs,
            periods_local,
            ctx,
            job_mask,
            period_mask,
            periods_full,
            period_full_mask,
        )

        # Logits from Q
        logits = -q_values / self.temperature
        if job_mask is not None:
            logits = logits.masked_fill(job_mask, float("-inf"))

        # Value = -min_Q (reward convention: higher = better)
        q_masked = q_values.clone()
        if job_mask is not None:
            q_masked = q_masked.masked_fill(job_mask, float("inf"))
        value = -q_masked.min(dim=-1, keepdim=True).values

        return logits, value

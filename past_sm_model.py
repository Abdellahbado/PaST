"""
PaST-SM Neural Network Model (Period-aware Scheduler Transformer - Single Machine).

Implements:
- Cross-attention: jobs attend to local periods for context
- Self-attention: jobs attend to each other for competition
- Pre-LN or Post-LN transformer blocks (configurable)
- Factored action head: job_logits + slack_logits -> joint categorical
- Global horizon embedding: duration-weighted pooling (for FULL_GLOBAL variant)

Supports all 6 ablation variants through configuration.
"""

import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import (
    VariantConfig,
    ModelConfig,
    EnvConfig,
    SlackType,
)


# =============================================================================
# Input Embedding Layers
# =============================================================================


class JobEmbedding(nn.Module):
    """
    Embeds job bin features into d_model dimensions.

    Input: [batch, M_job_bins, F_job] where F_job = [processing_time, count]
    Output: [batch, M_job_bins, d_model]
    """

    def __init__(self, F_job: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(F_job, d_model)

    def forward(self, jobs: Tensor) -> Tensor:
        return self.proj(jobs)


class PeriodEmbedding(nn.Module):
    """
    Embeds period features into d_model dimensions.

    Input: [batch, K_period, F_period] where F_period = [duration, price, start_offset, is_current]
    Output: [batch, K_period, d_model]
    """

    def __init__(self, F_period: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(F_period, d_model)

    def forward(self, periods: Tensor) -> Tensor:
        return self.proj(periods)


class ContextEmbedding(nn.Module):
    """
    Embeds global context features into d_model dimensions.

    Input: [batch, F_ctx] where F_ctx = [t, T_limit, remaining_work, e_single, avg_price, min_price]
    Output: [batch, d_model]
    """

    def __init__(self, F_ctx: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(F_ctx, d_model)

    def forward(self, ctx: Tensor) -> Tensor:
        return self.proj(ctx)


class GlobalHorizonEmbedding(nn.Module):
    """
    Duration-weighted pooling of full period embeddings for global horizon summary.

    Computes: sum(dur_k * E_k) / sum(dur_k)

    This gives a compressed representation of the entire horizon that emphasizes
    longer periods more heavily (since they have more scheduling opportunity).

    Input:
        - periods_full_emb: [batch, K_full_max, d_model] embedded full period tokens
        - periods_full_raw: [batch, K_full_max, F_period] raw features (need duration)
        - period_mask: [batch, K_full_max] boolean mask for valid periods
    Output: [batch, d_model]
    """

    def __init__(self, d_model: int):
        super().__init__()
        # Optional projection after pooling (identity by default)
        self.proj = nn.Identity()

    def forward(
        self,
        periods_full_emb: Tensor,
        periods_full_raw: Tensor,
        period_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            periods_full_emb: [batch, K, d_model]
            periods_full_raw: [batch, K, F_period] - we need duration from index 0
            period_mask: [batch, K] - True for invalid periods (masked out)
        """
        # Extract durations from raw features (index 0)
        durations = periods_full_raw[:, :, 0]  # [batch, K]

        # Apply mask if provided
        # Convention in this codebase: masks are boolean with True meaning INVALID / masked out.
        if period_mask is not None:
            durations = durations * (~period_mask).float()

        # Normalize durations to get weights
        dur_sum = durations.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # [batch, 1]
        weights = durations / dur_sum  # [batch, K]

        # Weighted sum of embeddings
        weights = weights.unsqueeze(-1)  # [batch, K, 1]
        pooled = (periods_full_emb * weights).sum(dim=1)  # [batch, d_model]

        return self.proj(pooled)


# =============================================================================
# Transformer Blocks (Pre-LN and Post-LN variants)
# =============================================================================


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head attention with optional mask support.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = math.sqrt(self.d_k)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            query: [batch, L_q, d_model]
            key: [batch, L_k, d_model]
            value: [batch, L_k, d_model]
            mask: [batch, L_q, L_k] or [batch, 1, L_k] - True for positions to mask OUT
        Returns:
            out: [batch, L_q, d_model]
        """
        B, L_q, _ = query.shape
        _, L_k, _ = key.shape

        # Project to Q, K, V
        Q = self.W_q(query).view(B, L_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, L_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, L_k, self.num_heads, self.d_k).transpose(1, 2)
        # Q, K, V: [batch, heads, L, d_k]

        # Attention scores
        scores = (
            torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        )  # [batch, heads, L_q, L_k]

        # Apply mask (set masked positions to -inf before softmax)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch, 1, L_q, L_k]
            scores = scores.masked_fill(mask, float("-inf"))

        # If a query row is fully masked, softmax would produce NaNs.
        # We zero those rows so attention output becomes zeros.
        fully_masked = None
        if mask is not None:
            fully_masked = mask.all(dim=-1, keepdim=True)  # [batch, 1, L_q, 1]
            scores = torch.where(fully_masked, torch.zeros_like(scores), scores)

        # Softmax and apply to values
        attn_weights = F.softmax(scores, dim=-1)
        if fully_masked is not None:
            attn_weights = torch.where(
                fully_masked, torch.zeros_like(attn_weights), attn_weights
            )
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)  # [batch, heads, L_q, d_k]
        out = out.transpose(1, 2).contiguous().view(B, L_q, self.d_model)

        return self.W_o(out)


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class PreLNBlock(nn.Module):
    """
    Pre-LayerNorm transformer block (more stable for RL training).

    Structure:
        x -> LN -> Attn -> + -> LN -> FF -> +
           |______________|    |__________|
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        is_cross_attention: bool = False,
    ):
        super().__init__()
        self.is_cross_attention = is_cross_attention

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

        if is_cross_attention:
            self.ln_kv = nn.LayerNorm(d_model)

    def forward(
        self, x: Tensor, kv: Optional[Tensor] = None, mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            x: Query tensor [batch, L_q, d_model]
            kv: Key/Value tensor for cross-attention [batch, L_kv, d_model]
            mask: Attention mask
        """
        # Pre-LN Attention
        x_norm = self.ln1(x)
        if self.is_cross_attention and kv is not None:
            kv_norm = self.ln_kv(kv)
            attn_out = self.attn(x_norm, kv_norm, kv_norm, mask)
        else:
            attn_out = self.attn(x_norm, x_norm, x_norm, mask)
        x = x + attn_out

        # Pre-LN Feed-Forward
        x = x + self.ff(self.ln2(x))

        return x


class PostLNBlock(nn.Module):
    """
    Post-LayerNorm transformer block (original Transformer architecture).

    Structure:
        x -> Attn -> + -> LN -> FF -> + -> LN
           |________|      |________|
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        is_cross_attention: bool = False,
    ):
        super().__init__()
        self.is_cross_attention = is_cross_attention

        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(
        self, x: Tensor, kv: Optional[Tensor] = None, mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            x: Query tensor [batch, L_q, d_model]
            kv: Key/Value tensor for cross-attention [batch, L_kv, d_model]
            mask: Attention mask
        """
        # Post-LN Attention
        if self.is_cross_attention and kv is not None:
            attn_out = self.attn(x, kv, kv, mask)
        else:
            attn_out = self.attn(x, x, x, mask)
        x = self.ln1(x + attn_out)

        # Post-LN Feed-Forward
        x = self.ln2(x + self.ff(x))

        return x


def make_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    dropout: float = 0.0,
    is_cross_attention: bool = False,
    use_pre_ln: bool = True,
) -> nn.Module:
    """Factory function to create transformer block based on config."""
    if use_pre_ln:
        return PreLNBlock(d_model, num_heads, d_ff, dropout, is_cross_attention)
    else:
        return PostLNBlock(d_model, num_heads, d_ff, dropout, is_cross_attention)


# =============================================================================
# PaST Encoder: Cross-attention (jobs -> periods) + Self-attention (jobs -> jobs)
# =============================================================================


class PaSTEncoder(nn.Module):
    """
    Period-aware Scheduler Transformer Encoder.

    Architecture:
    1. Embed jobs, periods, and context
    2. For each block:
       - Cross-attention: jobs attend to periods (with period mask)
       - Self-attention: jobs attend to jobs (with job mask)
    3. Output: job embeddings enriched with period and competition context

    Optionally includes global horizon embedding for FULL_GLOBAL variant.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        env_config: EnvConfig,
    ):
        super().__init__()

        self.model_config = model_config
        self.env_config = env_config

        d_model = model_config.d_model
        num_heads = model_config.num_heads
        num_blocks = model_config.num_blocks
        d_ff = model_config.d_ff
        dropout = model_config.dropout
        use_pre_ln = model_config.use_pre_ln

        # Input embeddings
        self.job_embed = JobEmbedding(env_config.F_job, d_model)
        self.period_embed = PeriodEmbedding(env_config.F_period, d_model)
        self.ctx_embed = ContextEmbedding(env_config.F_ctx, d_model)

        # Global horizon embedding (only for FULL_GLOBAL)
        self.use_global_horizon = model_config.use_global_horizon
        if self.use_global_horizon:
            self.global_horizon_embed = GlobalHorizonEmbedding(d_model)
            # Separate embedding for full periods (may be different from local)
            self.period_full_embed = PeriodEmbedding(env_config.F_period, d_model)

        # Transformer blocks
        # Each "block" contains one cross-attention and one self-attention
        self.cross_attn_blocks = nn.ModuleList(
            [
                make_transformer_block(
                    d_model,
                    num_heads,
                    d_ff,
                    dropout,
                    is_cross_attention=True,
                    use_pre_ln=use_pre_ln,
                )
                for _ in range(num_blocks)
            ]
        )

        self.self_attn_blocks = nn.ModuleList(
            [
                make_transformer_block(
                    d_model,
                    num_heads,
                    d_ff,
                    dropout,
                    is_cross_attention=False,
                    use_pre_ln=use_pre_ln,
                )
                for _ in range(num_blocks)
            ]
        )

        # Final layer norm (especially important for Pre-LN)
        self.final_ln = nn.LayerNorm(d_model)

    def forward(
        self,
        jobs: Tensor,  # [batch, M, F_job]
        periods_local: Tensor,  # [batch, K_local, F_period]
        ctx: Tensor,  # [batch, F_ctx]
        job_mask: Optional[Tensor] = None,  # [batch, M] True for empty bins
        period_mask: Optional[
            Tensor
        ] = None,  # [batch, K_local] True for invalid periods
        periods_full: Optional[
            Tensor
        ] = None,  # [batch, K_full, F_period] for global horizon
        period_full_mask: Optional[Tensor] = None,  # [batch, K_full]
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Forward pass through encoder.

        Returns:
            job_emb: [batch, M, d_model] - job embeddings after transformer
            ctx_emb: [batch, d_model] - context embedding
            global_emb: [batch, d_model] or None - global horizon embedding
        """
        # Embed inputs
        job_emb = self.job_embed(jobs)  # [batch, M, d_model]
        period_emb = self.period_embed(periods_local)  # [batch, K, d_model]
        ctx_emb = self.ctx_embed(ctx)  # [batch, d_model]

        # Compute global horizon embedding if enabled
        global_emb = None
        if self.use_global_horizon:
            # FULL_GLOBAL is intended to use a full-horizon token sequence.
            # In practice, some callers/environments may only provide `periods_local`.
            # For robustness (and because many variants set K_local == K_full), we fall back
            # to computing the global summary from local periods when `periods_full` is absent.
            periods_for_global = (
                periods_full if periods_full is not None else periods_local
            )
            mask_for_global = (
                period_full_mask if period_full_mask is not None else period_mask
            )
            periods_for_global_emb = self.period_full_embed(periods_for_global)
            global_emb = self.global_horizon_embed(
                periods_for_global_emb,
                periods_for_global,
                mask_for_global,
            )

        # Convert masks to attention format
        # Cross-attention: jobs [M] query, periods [K] key/value
        # Mask shape: [batch, M, K] or [batch, 1, K]
        cross_attn_mask = None
        if period_mask is not None:
            cross_attn_mask = period_mask.unsqueeze(1)  # [batch, 1, K]

        # Self-attention: jobs [M] attend to jobs [M]
        # Mask shape: [batch, M, M] or [batch, 1, M]
        self_attn_mask = None
        if job_mask is not None:
            self_attn_mask = job_mask.unsqueeze(1)  # [batch, 1, M]

        # Apply transformer blocks
        for cross_block, self_block in zip(
            self.cross_attn_blocks, self.self_attn_blocks
        ):
            # Cross-attention: jobs attend to periods
            job_emb = cross_block(job_emb, period_emb, cross_attn_mask)

            # Self-attention: jobs attend to jobs
            job_emb = self_block(job_emb, None, self_attn_mask)

        # Final layer norm
        job_emb = self.final_ln(job_emb)

        return job_emb, ctx_emb, global_emb


# =============================================================================
# Factored Action Head
# =============================================================================


class FactoredActionHead(nn.Module):
    """
    Factored action head that produces joint (job, slack) logits.

    Architecture:
    1. job_logits = Linear(job_emb)  -> [batch, M]
    2. slack_logits = MLP(job_emb + ctx_emb [+ global_emb]) -> [batch, M, K_slack]
    3. joint_logits = job_logits.unsqueeze(-1) + slack_logits -> [batch, M, K_slack]

    The factorization captures:
    - Job selection based on job features (which job to schedule)
    - Slack selection based on job + context (when to start given the job)

    Final distribution: softmax over [M * K_slack] joint actions.
    """

    def __init__(
        self,
        d_model: int,
        M_job_bins: int,
        K_slack: int,
        slack_head_hidden: int = 64,
        use_global_horizon: bool = False,
    ):
        super().__init__()

        self.M_job_bins = M_job_bins
        self.K_slack = K_slack
        self.use_global_horizon = use_global_horizon

        # Job head: single linear projection
        self.job_head = nn.Linear(d_model, 1)

        # Context dimension for slack head
        # Input: job_emb concatenated with expanded ctx_emb (and optionally global_emb)
        ctx_multiplier = 2 if use_global_horizon else 1
        slack_input_dim = d_model + d_model * ctx_multiplier

        # Slack head: MLP for each job position
        # Input: [job_emb || ctx_emb || global_emb] for each job
        self.slack_head = nn.Sequential(
            nn.Linear(slack_input_dim, slack_head_hidden),
            nn.ReLU(),
            nn.Linear(slack_head_hidden, K_slack),
        )

    def forward(
        self,
        job_emb: Tensor,  # [batch, M, d_model]
        ctx_emb: Tensor,  # [batch, d_model]
        global_emb: Optional[Tensor] = None,  # [batch, d_model]
        job_mask: Optional[Tensor] = None,  # [batch, M] True for invalid
    ) -> Tensor:
        """
        Compute joint logits for (job, slack) actions.

        Returns:
            joint_logits: [batch, M * K_slack] flattened logits
        """
        B, M, d = job_emb.shape

        # Job logits: [batch, M, 1] -> [batch, M]
        job_logits = self.job_head(job_emb).squeeze(-1)  # [batch, M]

        # Prepare slack head input by expanding context
        # ctx_emb: [batch, d] -> [batch, M, d]
        ctx_expanded = ctx_emb.unsqueeze(1).expand(-1, M, -1)

        # Concatenate job embedding with context
        if self.use_global_horizon and global_emb is not None:
            # global_emb: [batch, d] -> [batch, M, d]
            global_expanded = global_emb.unsqueeze(1).expand(-1, M, -1)
            slack_input = torch.cat([job_emb, ctx_expanded, global_expanded], dim=-1)
        else:
            slack_input = torch.cat([job_emb, ctx_expanded], dim=-1)

        # Slack logits: [batch, M, K_slack]
        slack_logits = self.slack_head(slack_input)

        # Joint logits: broadcasting addition
        # job_logits: [batch, M] -> [batch, M, 1]
        # slack_logits: [batch, M, K_slack]
        # joint: [batch, M, K_slack]
        joint_logits = job_logits.unsqueeze(-1) + slack_logits

        # Apply job mask to invalid bins
        if job_mask is not None:
            # job_mask: [batch, M] True for invalid -> [batch, M, 1]
            mask_expanded = job_mask.unsqueeze(-1).expand(-1, -1, self.K_slack)
            joint_logits = joint_logits.masked_fill(mask_expanded, float("-inf"))

        # Flatten: [batch, M, K_slack] -> [batch, M * K_slack]
        joint_logits = joint_logits.view(B, M * self.K_slack)

        return joint_logits


class SimpleActionHead(nn.Module):
    """
    Simple action head (non-factored) for comparison.

    Single MLP that outputs joint logits directly.
    """

    def __init__(
        self,
        d_model: int,
        M_job_bins: int,
        K_slack: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.M_job_bins = M_job_bins
        self.K_slack = K_slack
        action_dim = M_job_bins * K_slack

        # Pool job embeddings and combine with context
        self.pool_proj = nn.Linear(d_model, d_model)

        # MLP head
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        job_emb: Tensor,  # [batch, M, d_model]
        ctx_emb: Tensor,  # [batch, d_model]
        global_emb: Optional[Tensor] = None,
        job_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute joint logits directly (non-factored).
        """
        B, M, d = job_emb.shape

        # Pool job embeddings (mean pooling with mask)
        if job_mask is not None:
            mask = (~job_mask).float().unsqueeze(-1)  # [batch, M, 1]
            job_pooled = (job_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            job_pooled = job_emb.mean(dim=1)

        job_pooled = self.pool_proj(job_pooled)  # [batch, d_model]

        # Combine with context
        combined = torch.cat([job_pooled, ctx_emb], dim=-1)  # [batch, 2*d_model]

        # Get logits
        logits = self.head(combined)  # [batch, M * K_slack]

        # Apply job mask
        if job_mask is not None:
            # Reshape to [batch, M, K_slack]
            logits = logits.view(B, M, self.K_slack)
            mask_expanded = job_mask.unsqueeze(-1).expand(-1, -1, self.K_slack)
            logits = logits.masked_fill(mask_expanded, float("-inf"))
            logits = logits.view(B, M * self.K_slack)

        return logits


# =============================================================================
# Value Head (for PPO)
# =============================================================================


class ValueHead(nn.Module):
    """
    Value function head for PPO.

    Estimates V(s) from job embeddings and context.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 128,
        use_global_horizon: bool = False,
    ):
        super().__init__()

        self.use_global_horizon = use_global_horizon

        # Pool job embeddings
        self.pool_proj = nn.Linear(d_model, d_model)

        # Input dimension: pooled_jobs + ctx [+ global]
        input_dim = d_model * 2
        if use_global_horizon:
            input_dim += d_model

        # Value MLP
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        job_emb: Tensor,  # [batch, M, d_model]
        ctx_emb: Tensor,  # [batch, d_model]
        global_emb: Optional[Tensor] = None,
        job_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute state value estimate.

        Returns:
            value: [batch, 1]
        """
        # Pool job embeddings
        if job_mask is not None:
            mask = (~job_mask).float().unsqueeze(-1)
            job_pooled = (job_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            job_pooled = job_emb.mean(dim=1)

        job_pooled = self.pool_proj(job_pooled)

        # Combine features
        if self.use_global_horizon and global_emb is not None:
            combined = torch.cat([job_pooled, ctx_emb, global_emb], dim=-1)
        else:
            combined = torch.cat([job_pooled, ctx_emb], dim=-1)

        return self.value_net(combined)


# =============================================================================
# Full PaST-SM Model
# =============================================================================


class PaSTSMNet(nn.Module):
    """
    Full PaST-SM Network: Encoder + Action Head + Value Head.

    Configurable via VariantConfig to support all 6 ablation variants.
    """

    def __init__(self, config: VariantConfig):
        super().__init__()

        self.config = config
        self.model_config = config.model
        self.env_config = config.env

        d_model = self.model_config.d_model
        M = self.env_config.M_job_bins
        K_slack = self.env_config.get_num_slack_choices()

        # Encoder
        self.encoder = PaSTEncoder(self.model_config, self.env_config)

        # Action head (factored or simple)
        if self.model_config.use_factored_head:
            self.action_head = FactoredActionHead(
                d_model=d_model,
                M_job_bins=M,
                K_slack=K_slack,
                slack_head_hidden=self.model_config.slack_head_hidden,
                use_global_horizon=self.model_config.use_global_horizon,
            )
        else:
            self.action_head = SimpleActionHead(
                d_model=d_model,
                M_job_bins=M,
                K_slack=K_slack,
            )

        # Value head (for PPO)
        self.value_head = ValueHead(
            d_model=d_model,
            use_global_horizon=self.model_config.use_global_horizon,
        )

        # Store dimensions for external use
        self.M_job_bins = M
        self.K_slack = K_slack
        self.action_dim = M * K_slack

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
        Full forward pass.

        Args:
            jobs: [batch, M, F_job]
            periods_local: [batch, K_local, F_period]
            ctx: [batch, F_ctx]
            job_mask: [batch, M] True for invalid bins
            period_mask: [batch, K_local] True for invalid periods
            periods_full: [batch, K_full, F_period] (optional, for FULL_GLOBAL)
            period_full_mask: [batch, K_full] (optional)

        Returns:
            logits: [batch, M * K_slack] action logits
            value: [batch, 1] state value estimate
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

        # Action logits
        logits = self.action_head(job_emb, ctx_emb, global_emb, job_mask)

        # Value estimate
        value = self.value_head(job_emb, ctx_emb, global_emb, job_mask)

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
        """
        Get action distribution (for sampling actions).
        """
        logits, _ = self.forward(
            jobs,
            periods_local,
            ctx,
            job_mask,
            period_mask,
            periods_full,
            period_full_mask,
        )
        return torch.distributions.Categorical(logits=logits)

    def evaluate_actions(
        self,
        jobs: Tensor,
        periods_local: Tensor,
        ctx: Tensor,
        actions: Tensor,
        job_mask: Optional[Tensor] = None,
        period_mask: Optional[Tensor] = None,
        periods_full: Optional[Tensor] = None,
        period_full_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Evaluate given actions (for PPO update).

        Returns:
            log_probs: [batch] log probabilities of actions
            values: [batch] state values
            entropy: [batch] policy entropy
        """
        logits, values = self.forward(
            jobs,
            periods_local,
            ctx,
            job_mask,
            period_mask,
            periods_full,
            period_full_mask,
        )

        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values.squeeze(-1), entropy

    def decode_action(self, action: int) -> Tuple[int, int]:
        """
        Decode flat action index to (job_bin_idx, slack_idx).

        action = job_bin_idx * K_slack + slack_idx
        """
        job_bin_idx = action // self.K_slack
        slack_idx = action % self.K_slack
        return job_bin_idx, slack_idx


# =============================================================================
# Factory Function
# =============================================================================


def build_model(config: VariantConfig) -> PaSTSMNet:
    """
    Build PaST-SM model from variant configuration.
    """
    model = PaSTSMNet(config)
    return model


# =============================================================================
# Testing
# =============================================================================


if __name__ == "__main__":
    from .config import get_variant_config, list_variants, VariantID

    print("=" * 70)
    print("PaST-SM Model Test")
    print("=" * 70)

    device = "cpu"
    batch_size = 4

    for variant_id in list_variants():
        print(f"\nTesting {variant_id.value}...")

        config = get_variant_config(variant_id)
        model = build_model(config).to(device)

        # Create dummy inputs
        M = config.env.M_job_bins
        K_local = config.env.K_period_local
        K_full = config.env.K_period_full_max
        F_job = config.env.F_job
        F_period = config.env.F_period
        F_ctx = config.env.F_ctx

        jobs = torch.randn(batch_size, M, F_job, device=device)
        periods_local = torch.randn(batch_size, K_local, F_period, device=device)
        ctx = torch.randn(batch_size, F_ctx, device=device)

        # Create masks (some invalid entries)
        job_mask = torch.zeros(batch_size, M, dtype=torch.bool, device=device)
        job_mask[:, -5:] = True  # Last 5 bins invalid

        period_mask = torch.zeros(batch_size, K_local, dtype=torch.bool, device=device)
        period_mask[:, -3:] = True  # Last 3 periods invalid

        # Full periods (only for variants that use them)
        periods_full = None
        period_full_mask = None
        if config.env.use_periods_full:
            periods_full = torch.randn(batch_size, K_full, F_period, device=device)
            period_full_mask = torch.zeros(
                batch_size, K_full, dtype=torch.bool, device=device
            )
            period_full_mask[:, -20:] = True

        # Forward pass
        logits, value = model(
            jobs,
            periods_local,
            ctx,
            job_mask,
            period_mask,
            periods_full,
            period_full_mask,
        )

        # Sample action
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        # Decode action
        job_idx, slack_idx = model.decode_action(action[0].item())

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())

        print(f"  Logits shape: {logits.shape}")
        print(f"  Value shape: {value.shape}")
        print(f"  Action dim: {model.action_dim}")
        print(
            f"  Sample action: {action[0].item()} -> job={job_idx}, slack={slack_idx}"
        )
        print(f"  Parameters: {num_params:,}")
        print(f"  ✓ {variant_id.value} passed!")

    print("\n" + "=" * 70)
    print("All model variants tested successfully!")
    print("=" * 70)

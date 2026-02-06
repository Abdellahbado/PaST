"""Decoder and Q-value head for NeuroLS following the paper's architecture.

This module implements:
1. Aggregation of encoder outputs (node, group, feature embeddings)
2. Q-value regression head (MLP)
3. Implicit Quantile Networks (IQN) for distributional Q-learning

Following NeuroLS Figure 1:
- Concatenate [omega_node; omega_group; omega_feat]
- 2-layer MLP -> |A| Q-values

For IQN (Section 4.4):
- Sample quantile fractions tau ~ Uniform(0, 1)
- Embed tau using cosine features
- Combine with state features for distributional Q-values
"""

from __future__ import annotations

from typing import Tuple, Optional
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:

    class QValueHead(nn.Module):
        """Q-value regression head (standard DQN).

        Input: Concatenated representation [omega_node; omega_group; omega_feat]
        Output: Q-values for each action
        """

        def __init__(
            self,
            input_dim: int,
            n_actions: int,
            hidden_dim: int = 128,
            n_hidden: int = 2,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.input_dim = input_dim
            self.n_actions = n_actions

            layers = []
            in_d = input_dim
            for i in range(n_hidden):
                layers.append(nn.Linear(in_d, hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
                in_d = hidden_dim

            layers.append(nn.Linear(hidden_dim, n_actions))

            self.mlp = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (batch, input_dim) or (input_dim,)

            Returns:
                (batch, n_actions) or (n_actions,) Q-values
            """
            return self.mlp(x)

    class DuelingQHead(nn.Module):
        """Dueling DQN architecture.

        Separates value V(s) and advantage A(s, a):
        Q(s, a) = V(s) + A(s, a) - mean(A(s, .))
        """

        def __init__(
            self,
            input_dim: int,
            n_actions: int,
            hidden_dim: int = 128,
            dropout: float = 0.1,
        ):
            super().__init__()

            # Value stream
            self.value_stream = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

            # Advantage stream
            self.advantage_stream = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_actions),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (batch, input_dim)

            Returns:
                (batch, n_actions) Q-values
            """
            value = self.value_stream(x)  # (B, 1)
            advantage = self.advantage_stream(x)  # (B, A)

            # Combine: Q = V + A - mean(A)
            q = value + advantage - advantage.mean(dim=-1, keepdim=True)
            return q

    class IQNEmbedding(nn.Module):
        """Quantile embedding for Implicit Quantile Networks.

        Following the IQN paper, embed quantile fractions using cosine features:
        phi(tau) = ReLU(sum_i cos(pi * i * tau))
        """

        def __init__(
            self,
            embedding_dim: int = 64,
            n_cosines: int = 64,
        ):
            super().__init__()
            self.n_cosines = n_cosines
            self.embedding_dim = embedding_dim

            # Register buffer for cosine indices
            self.register_buffer(
                "cosine_indices",
                torch.arange(1, n_cosines + 1, dtype=torch.float32).reshape(1, 1, -1),
            )

            # Linear projection after cosine embedding
            self.proj = nn.Linear(n_cosines, embedding_dim)

        def forward(self, tau: torch.Tensor) -> torch.Tensor:
            """
            Args:
                tau: (batch, n_samples) quantile fractions in [0, 1]

            Returns:
                (batch, n_samples, embedding_dim) quantile embeddings
            """
            # tau: (B, N) -> (B, N, 1)
            tau = tau.unsqueeze(-1)

            # Cosine features: cos(pi * i * tau) for i = 1..n_cosines
            # (B, N, 1) * (1, 1, n_cosines) -> (B, N, n_cosines)
            cos_features = torch.cos(math.pi * tau * self.cosine_indices)

            # Project
            return F.relu(self.proj(cos_features))

    class IQNQHead(nn.Module):
        """Implicit Quantile Network Q-value head.

        For distributional RL: outputs quantile values of Q distribution.
        """

        def __init__(
            self,
            state_dim: int,
            n_actions: int,
            hidden_dim: int = 128,
            n_cosines: int = 64,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.n_actions = n_actions

            # Quantile embedding
            self.tau_embed = IQNEmbedding(
                embedding_dim=state_dim,
                n_cosines=n_cosines,
            )

            # State embedding
            self.state_embed = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.GELU(),
            )

            # Combined Q-value head
            self.q_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_actions),
            )

        def forward(
            self,
            state: torch.Tensor,
            tau: torch.Tensor,
        ) -> torch.Tensor:
            """
            Args:
                state: (batch, state_dim) state features
                tau: (batch, n_samples) quantile fractions

            Returns:
                (batch, n_samples, n_actions) quantile Q-values
            """
            batch_size = state.size(0)
            n_samples = tau.size(1)

            # State embedding: (B, hidden_dim)
            state_emb = self.state_embed(state)

            # Quantile embedding: (B, N, state_dim)
            tau_emb = self.tau_embed(tau)

            # Combine: multiply element-wise after broadcasting
            # state_emb: (B, 1, hidden_dim) * (B, N, state_dim) after projection
            # We need to align dimensions - expand state for each sample
            state_emb = state_emb.unsqueeze(1).expand(
                -1, n_samples, -1
            )  # (B, N, hidden)

            # Note: tau_emb is (B, N, state_dim) but we projected to hidden_dim in state_embed
            # Let's use element-wise product then sum (Hadamard product style)
            # Actually, IQN typically uses element-wise multiplication
            combined = state_emb * tau_emb  # (B, N, hidden_dim) if dims match

            # Q-values: (B, N, A)
            q_values = self.q_head(combined)

            return q_values

        def get_q_values(
            self,
            state: torch.Tensor,
            n_samples: int = 32,
        ) -> torch.Tensor:
            """Get mean Q-values by sampling quantiles.

            Args:
                state: (batch, state_dim)
                n_samples: Number of quantile samples

            Returns:
                (batch, n_actions) mean Q-values
            """
            batch_size = state.size(0)

            # Sample quantiles uniformly
            tau = torch.rand(batch_size, n_samples, device=state.device)

            # Get quantile Q-values
            q_samples = self.forward(state, tau)  # (B, N, A)

            # Mean over samples
            return q_samples.mean(dim=1)  # (B, A)

    class NeuroLSDecoder(nn.Module):
        """Full decoder following NeuroLS architecture.

        Components:
        1. Aggregation of encoder outputs
        2. Optional IQN quantile embedding
        3. Q-value head (standard, dueling, or IQN)
        """

        def __init__(
            self,
            d_emb: int = 64,
            n_actions: int = 14,
            hidden_dim: int = 128,
            use_iqn: bool = True,
            n_cosines: int = 64,
            use_dueling: bool = False,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.d_emb = d_emb
            self.n_actions = n_actions
            self.use_iqn = use_iqn

            # Input: 3 * d_emb (node, group, feat)
            input_dim = 3 * d_emb

            if use_iqn:
                self.q_head = IQNQHead(
                    state_dim=input_dim,
                    n_actions=n_actions,
                    hidden_dim=hidden_dim,
                    n_cosines=n_cosines,
                    dropout=dropout,
                )
            elif use_dueling:
                self.q_head = DuelingQHead(
                    input_dim=input_dim,
                    n_actions=n_actions,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
            else:
                self.q_head = QValueHead(
                    input_dim=input_dim,
                    n_actions=n_actions,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )

        def forward(
            self,
            omega_node: torch.Tensor,
            omega_group: torch.Tensor,
            omega_feat: torch.Tensor,
            tau: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Args:
                omega_node: (d_emb,) or (batch, d_emb)
                omega_group: (d_emb,) or (batch, d_emb)
                omega_feat: (d_emb,) or (batch, d_emb)
                tau: Optional (batch, n_samples) for IQN

            Returns:
                Q-values: (n_actions,), (batch, n_actions),
                         or (batch, n_samples, n_actions) for IQN
            """
            # Handle single sample vs batch
            if omega_node.dim() == 1:
                omega_node = omega_node.unsqueeze(0)
                omega_group = omega_group.unsqueeze(0)
                omega_feat = omega_feat.unsqueeze(0)
                squeeze_output = True
            else:
                squeeze_output = False

            # Concatenate
            x = torch.cat([omega_node, omega_group, omega_feat], dim=-1)

            if self.use_iqn:
                if tau is None:
                    # Default: get mean Q-values
                    q = self.q_head.get_q_values(x)
                else:
                    q = self.q_head(x, tau)
            else:
                q = self.q_head(x)

            if squeeze_output and q.dim() > 1:
                q = q.squeeze(0)

            return q

        def get_action(
            self,
            omega_node: torch.Tensor,
            omega_group: torch.Tensor,
            omega_feat: torch.Tensor,
            greedy: bool = True,
        ) -> int:
            """Get action from Q-values.

            Args:
                omega_*: Encoder outputs
                greedy: If True, return argmax; else sample from softmax

            Returns:
                Action index
            """
            with torch.no_grad():
                q = self.forward(omega_node, omega_group, omega_feat)

                if greedy:
                    return q.argmax(dim=-1).item()
                else:
                    probs = F.softmax(q, dim=-1)
                    return torch.multinomial(probs, 1).item()

    class NeuroLSPolicy(nn.Module):
        """Complete NeuroLS policy model.

        Combines:
        - GNN encoder
        - Price embedding (optional)
        - Q-value decoder

        This is the main model class for training and inference.
        """

        def __init__(
            self,
            d_job_in: int = 5,
            d_machine_in: int = 5,
            d_state_in: int = 13,
            d_price_in: int = 64,
            d_emb: int = 64,
            n_actions: int = 14,
            n_layers_static: int = 2,
            n_layers_dynamic: int = 2,
            use_iqn: bool = True,
            use_dueling: bool = False,
            dropout: float = 0.1,
            price_mode: str = "full",
        ):
            super().__init__()

            from PaST.neurols.gnn_encoder import NeuroLSEncoder
            from PaST.neurols.price_embedding import PriceEmbedding

            self.d_emb = d_emb
            self.n_actions = n_actions
            self.price_mode = price_mode

            # Encoder
            self.encoder = NeuroLSEncoder(
                d_job_in=d_job_in,
                d_machine_in=d_machine_in,
                d_state_in=d_state_in,
                d_price_in=d_price_in if price_mode != "none" else 0,
                d_emb=d_emb,
                n_layers_static=n_layers_static,
                n_layers_dynamic=n_layers_dynamic,
                dropout=dropout,
            )

            # Price embedding
            if price_mode != "none":
                self.price_embed = PriceEmbedding(
                    d_emb=d_price_in,
                    mode=price_mode,
                )
            else:
                self.price_embed = None

            # Decoder
            self.decoder = NeuroLSDecoder(
                d_emb=d_emb,
                n_actions=n_actions,
                use_iqn=use_iqn,
                use_dueling=use_dueling,
                dropout=dropout,
            )

        def forward(
            self,
            job_features: torch.Tensor,
            machine_features: torch.Tensor,
            state_features: torch.Tensor,
            job_to_machine: torch.Tensor,
            static_edge_index: torch.Tensor,
            dynamic_edge_index: torch.Tensor,
            price_features: Optional[torch.Tensor] = None,
            machine_exposure: Optional[torch.Tensor] = None,
            tau: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Forward pass for Q-value computation.

            Returns:
                Q-values for all actions
            """
            # Price embedding
            if self.price_embed is not None and price_features is not None:
                price_emb = self.price_embed(price_features, machine_exposure)
            else:
                price_emb = None

            # Encode
            omega_node, omega_group, omega_feat = self.encoder(
                job_features=job_features,
                machine_features=machine_features,
                state_features=state_features,
                job_to_machine=job_to_machine,
                static_edge_index=static_edge_index,
                dynamic_edge_index=dynamic_edge_index,
                price_embedding=price_emb,
            )

            # Decode
            return self.decoder(omega_node, omega_group, omega_feat, tau)

        def get_action(
            self,
            job_features: torch.Tensor,
            machine_features: torch.Tensor,
            state_features: torch.Tensor,
            job_to_machine: torch.Tensor,
            static_edge_index: torch.Tensor,
            dynamic_edge_index: torch.Tensor,
            price_features: Optional[torch.Tensor] = None,
            machine_exposure: Optional[torch.Tensor] = None,
            greedy: bool = True,
            epsilon: float = 0.0,
        ) -> int:
            """Get action using epsilon-greedy policy.

            Args:
                ...: Model inputs
                greedy: If True (and epsilon=0), use argmax
                epsilon: Probability of random action

            Returns:
                Action index
            """
            import random

            if random.random() < epsilon:
                return random.randrange(self.n_actions)

            with torch.no_grad():
                q = self.forward(
                    job_features,
                    machine_features,
                    state_features,
                    job_to_machine,
                    static_edge_index,
                    dynamic_edge_index,
                    price_features,
                    machine_exposure,
                )

                if greedy:
                    return q.argmax(dim=-1).item()
                else:
                    probs = F.softmax(q, dim=-1)
                    return torch.multinomial(probs.unsqueeze(0), 1).item()

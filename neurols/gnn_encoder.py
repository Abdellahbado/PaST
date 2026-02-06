"""GNN Encoder for NeuroLS following the paper's architecture.

This module implements the graph neural network encoder from NeuroLS Section 4.3:
- 3-stage GNN with static/dynamic/consolidation phases
- Bipartite graph: Jobs + Machines nodes
- Static edges: full compatibility (identical machines)
- Dynamic edges: current assignment (job -> machine)
- Group pooling for machine-level aggregation

Architecture:
Stage 1: L_stat GNN layers over static edges
Stage 2: L_dyna GNN layers over dynamic (assignment) edges
Stage 3: 1 GNN layer over static to consolidate

The encoder produces:
- Per-node embeddings: (n_jobs + n_machines, d_emb)
- Group embeddings: (n_machines, d_emb) via pooling
- Global embedding: (d_emb,) via aggregation
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

if _TORCH_AVAILABLE:

    class GNNLayer(nn.Module):
        """Single GNN layer following NeuroLS equation (6).

        h_i^(l) = sigma(MLP1(h_i^(l-1)) + MLP2(sum_j e_ji * h_j^(l-1)))

        With residual connections and layer normalization.
        """

        def __init__(
            self,
            d_emb: int,
            d_hidden: int = None,
            use_edge_weights: bool = True,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.d_emb = d_emb
            d_hidden = d_hidden or d_emb * 2
            self.use_edge_weights = use_edge_weights

            # Self-transform MLP
            self.mlp_self = nn.Sequential(
                nn.Linear(d_emb, d_hidden),
                nn.GELU(),
                nn.Linear(d_hidden, d_emb),
            )

            # Neighbor aggregation MLP
            self.mlp_neigh = nn.Sequential(
                nn.Linear(d_emb, d_hidden),
                nn.GELU(),
                nn.Linear(d_hidden, d_emb),
            )

            self.norm = nn.LayerNorm(d_emb)
            self.dropout = nn.Dropout(dropout)

        def forward(
            self,
            h: torch.Tensor,
            edge_index: torch.Tensor,
            edge_weight: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Args:
                h: Node embeddings, (N, d_emb)
                edge_index: Edge indices, (2, E) where [0] = source, [1] = target
                edge_weight: Optional edge weights, (E,)

            Returns:
                Updated node embeddings, (N, d_emb)
            """
            N = h.size(0)

            # Self-transform
            h_self = self.mlp_self(h)

            # Neighbor aggregation
            src, dst = edge_index[0], edge_index[1]

            # Gather source node features
            h_src = h[src]  # (E, d_emb)

            # Apply edge weights if provided
            if edge_weight is not None and self.use_edge_weights:
                h_src = h_src * edge_weight.unsqueeze(-1)

            # Aggregate by destination (scatter_add)
            h_agg = torch.zeros(N, self.d_emb, device=h.device, dtype=h.dtype)
            h_agg.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.d_emb), h_src)

            h_neigh = self.mlp_neigh(h_agg)

            # Combine with residual
            h_out = h + self.dropout(h_self + h_neigh)
            h_out = self.norm(h_out)

            return h_out

    class BipartiteGNNEncoder(nn.Module):
        """Bipartite GNN encoder for parallel machine scheduling.

        Graph structure:
        - Nodes: Jobs (0..n_jobs-1) + Machines (n_jobs..n_jobs+n_machines-1)
        - Static edges: Full bipartite (all jobs can go to all machines)
        - Dynamic edges: Current assignment (job_i -> machine_m if assigned)

        Following NeuroLS 3-stage architecture:
        1. Static GNN layers (L_stat)
        2. Dynamic GNN layers (L_dyna)
        3. Consolidation GNN layer (1 layer over static)
        """

        def __init__(
            self,
            d_job_in: int,
            d_machine_in: int,
            d_emb: int = 64,
            n_layers_static: int = 2,
            n_layers_dynamic: int = 2,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.d_emb = d_emb
            self.n_layers_static = n_layers_static
            self.n_layers_dynamic = n_layers_dynamic

            # Input projections
            self.job_embed = nn.Sequential(
                nn.Linear(d_job_in, d_emb),
                nn.GELU(),
                nn.Linear(d_emb, d_emb),
            )
            self.machine_embed = nn.Sequential(
                nn.Linear(d_machine_in, d_emb),
                nn.GELU(),
                nn.Linear(d_emb, d_emb),
            )

            # Stage 1: Static GNN layers
            self.static_layers = nn.ModuleList(
                [GNNLayer(d_emb, dropout=dropout) for _ in range(n_layers_static)]
            )

            # Stage 2: Dynamic GNN layers
            self.dynamic_layers = nn.ModuleList(
                [GNNLayer(d_emb, dropout=dropout) for _ in range(n_layers_dynamic)]
            )

            # Stage 3: Consolidation layer
            self.consolidation_layer = GNNLayer(d_emb, dropout=dropout)

            # Output MLP
            self.output_mlp = nn.Sequential(
                nn.Linear(d_emb, d_emb),
                nn.GELU(),
            )

        def forward(
            self,
            job_features: torch.Tensor,
            machine_features: torch.Tensor,
            static_edge_index: torch.Tensor,
            dynamic_edge_index: torch.Tensor,
            static_edge_weight: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Args:
                job_features: (n_jobs, d_job_in)
                machine_features: (n_machines, d_machine_in)
                static_edge_index: (2, E_static) full bipartite edges
                dynamic_edge_index: (2, E_dynamic) current assignment edges
                static_edge_weight: Optional weights for static edges

            Returns:
                job_embeddings: (n_jobs, d_emb)
                machine_embeddings: (n_machines, d_emb)
                global_embedding: (d_emb,)
            """
            n_jobs = job_features.size(0)
            n_machines = machine_features.size(0)

            # Initial embeddings
            h_jobs = self.job_embed(job_features)
            h_machines = self.machine_embed(machine_features)

            # Concatenate to single node tensor
            h = torch.cat([h_jobs, h_machines], dim=0)  # (N, d_emb)

            # Stage 1: Static layers
            for layer in self.static_layers:
                h = layer(h, static_edge_index, static_edge_weight)

            # Stage 2: Dynamic layers
            for layer in self.dynamic_layers:
                h = layer(h, dynamic_edge_index)

            # Stage 3: Consolidation
            h = self.consolidation_layer(h, static_edge_index, static_edge_weight)

            # Output projection
            h = self.output_mlp(h)

            # Split back
            h_jobs = h[:n_jobs]
            h_machines = h[n_jobs:]

            # Global embedding via mean pooling
            h_global = h.mean(dim=0)

            return h_jobs, h_machines, h_global

    class NeuroLSEncoder(nn.Module):
        """Complete NeuroLS encoder with group pooling.

        Following NeuroLS Figure 1:
        1. GNN encoder produces node embeddings
        2. Group pooling for machine-level aggregation (max + mean)
        3. Feature projection for scalar state features
        4. Concatenation for final representation
        """

        def __init__(
            self,
            d_job_in: int = 5,
            d_machine_in: int = 5,
            d_state_in: int = 13,
            d_price_in: int = 64,
            d_emb: int = 64,
            n_layers_static: int = 2,
            n_layers_dynamic: int = 2,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.d_emb = d_emb

            # GNN encoder
            self.gnn = BipartiteGNNEncoder(
                d_job_in=d_job_in,
                d_machine_in=d_machine_in,
                d_emb=d_emb,
                n_layers_static=n_layers_static,
                n_layers_dynamic=n_layers_dynamic,
                dropout=dropout,
            )

            # Group pooling projection (equation 8)
            # Input: max + mean over group = 2 * d_emb
            self.group_mlp = nn.Sequential(
                nn.Linear(2 * d_emb, d_emb),
                nn.GELU(),
            )

            # State feature projection
            self.state_proj = nn.Sequential(
                nn.Linear(d_state_in, d_emb),
                nn.GELU(),
                nn.Linear(d_emb, d_emb),
            )

            # Price embedding projection (if used)
            self.price_proj = (
                nn.Sequential(
                    nn.Linear(d_price_in, d_emb),
                    nn.GELU(),
                )
                if d_price_in > 0
                else None
            )

        def forward(
            self,
            job_features: torch.Tensor,
            machine_features: torch.Tensor,
            state_features: torch.Tensor,
            job_to_machine: torch.Tensor,
            static_edge_index: torch.Tensor,
            dynamic_edge_index: torch.Tensor,
            price_embedding: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Args:
                job_features: (n_jobs, d_job_in)
                machine_features: (n_machines, d_machine_in)
                state_features: (d_state_in,) scalar state features
                job_to_machine: (n_jobs,) machine assignment per job
                static_edge_index: (2, E_static)
                dynamic_edge_index: (2, E_dynamic)
                price_embedding: Optional (d_price_in,) price embedding

            Returns:
                omega_node: (d_emb,) mean node embedding
                omega_group: (d_emb,) group-pooled embedding
                omega_feat: (d_emb,) state + price features
            """
            n_jobs = job_features.size(0)
            n_machines = machine_features.size(0)

            # GNN encoding
            h_jobs, h_machines, h_global = self.gnn(
                job_features,
                machine_features,
                static_edge_index,
                dynamic_edge_index,
            )

            # Group pooling by machine (equation 8)
            # Pool jobs by their assigned machine
            group_max = torch.zeros(n_machines, self.d_emb, device=h_jobs.device)
            group_sum = torch.zeros(n_machines, self.d_emb, device=h_jobs.device)
            group_count = torch.zeros(n_machines, 1, device=h_jobs.device)

            for j in range(n_jobs):
                mi = job_to_machine[j].item()
                group_max[mi] = torch.max(group_max[mi], h_jobs[j])
                group_sum[mi] = group_sum[mi] + h_jobs[j]
                group_count[mi] += 1

            group_mean = group_sum / (group_count + 1e-9)
            group_combined = torch.cat([group_max, group_mean], dim=-1)  # (M, 2*d_emb)
            group_emb = self.group_mlp(group_combined)  # (M, d_emb)

            # Aggregate embeddings
            omega_node = h_jobs.mean(dim=0)  # (d_emb,)
            omega_group = group_emb.mean(dim=0)  # (d_emb,)

            # State features
            omega_state = self.state_proj(state_features)  # (d_emb,)

            # Add price embedding if available
            if price_embedding is not None and self.price_proj is not None:
                omega_price = self.price_proj(price_embedding)
                omega_feat = omega_state + omega_price
            else:
                omega_feat = omega_state

            return omega_node, omega_group, omega_feat

        def get_output_dim(self) -> int:
            """Total output dimension (3 * d_emb)."""
            return 3 * self.d_emb


def build_static_edges(n_jobs: int, n_machines: int) -> torch.Tensor:
    """Build static bipartite edges (all jobs can go to all machines).

    Returns:
        (2, n_jobs * n_machines * 2) edge index tensor
        Edges go both directions: job -> machine and machine -> job
    """
    edges_src = []
    edges_dst = []

    # Job -> Machine edges
    for j in range(n_jobs):
        for m in range(n_machines):
            edges_src.append(j)
            edges_dst.append(n_jobs + m)

    # Machine -> Job edges (reverse)
    for j in range(n_jobs):
        for m in range(n_machines):
            edges_src.append(n_jobs + m)
            edges_dst.append(j)

    return torch.tensor([edges_src, edges_dst], dtype=torch.long)


def build_dynamic_edges(
    job_to_machine: torch.Tensor,
    n_jobs: int,
    n_machines: int,
) -> torch.Tensor:
    """Build dynamic edges from current assignment.

    Args:
        job_to_machine: (n_jobs,) machine index for each job
        n_jobs: Number of jobs
        n_machines: Number of machines

    Returns:
        (2, n_jobs * 2) edge index tensor (bidirectional)
    """
    edges_src = []
    edges_dst = []

    for j in range(n_jobs):
        m = job_to_machine[j].item()
        # Job -> assigned machine
        edges_src.append(j)
        edges_dst.append(n_jobs + m)
        # Machine -> assigned job
        edges_src.append(n_jobs + m)
        edges_dst.append(j)

    return torch.tensor([edges_src, edges_dst], dtype=torch.long)


class TripartiteGNNEncoder(nn.Module):
    """Tripartite GNN encoder with Jobs + Machines + Periods nodes.

    For ablation study: explicit period nodes vs pure embedding.

    Nodes:
    - Jobs: 0..n_jobs-1
    - Machines: n_jobs..n_jobs+n_machines-1
    - Periods: n_jobs+n_machines..n_jobs+n_machines+n_periods-1

    Edges:
    - Job -> Machine (assignment)
    - Job -> Period (schedule placement)
    - Period -> Machine (occupancy)
    """

    def __init__(
        self,
        d_job_in: int,
        d_machine_in: int,
        d_period_in: int,
        d_emb: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_emb = d_emb

        # Input projections
        self.job_embed = nn.Linear(d_job_in, d_emb)
        self.machine_embed = nn.Linear(d_machine_in, d_emb)
        self.period_embed = nn.Linear(d_period_in, d_emb)

        # GNN layers
        self.layers = nn.ModuleList(
            [GNNLayer(d_emb, dropout=dropout) for _ in range(n_layers)]
        )

        # Output
        self.output_mlp = nn.Sequential(
            nn.Linear(d_emb, d_emb),
            nn.GELU(),
        )

    def forward(
        self,
        job_features: torch.Tensor,
        machine_features: torch.Tensor,
        period_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            job_emb, machine_emb, period_emb, global_emb
        """
        n_jobs = job_features.size(0)
        n_machines = machine_features.size(0)
        n_periods = period_features.size(0)

        # Embed
        h = torch.cat(
            [
                self.job_embed(job_features),
                self.machine_embed(machine_features),
                self.period_embed(period_features),
            ],
            dim=0,
        )

        # GNN
        for layer in self.layers:
            h = layer(h, edge_index)

        h = self.output_mlp(h)

        # Split
        h_jobs = h[:n_jobs]
        h_machines = h[n_jobs : n_jobs + n_machines]
        h_periods = h[n_jobs + n_machines :]
        h_global = h.mean(dim=0)

        return h_jobs, h_machines, h_periods, h_global

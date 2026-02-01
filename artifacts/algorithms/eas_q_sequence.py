"""
EAS (Efficient Active Search) for Q-Sequence models.

Implements SGBS+EAS hybrid algorithm for test-time fine-tuning of Q-sequence
models. Based on Algorithm 2 from the SGBS paper.

Key components:
- EASQResidualLayer: Trainable residual layer added to Q-model embeddings
- EASQModelWrapper: Wraps QSequenceNet with trainable EAS parameters
- EASQSequenceRunner: Runs SGBS+EAS hybrid search with DP-based scoring
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from PaST.sequence_env import GPUBatchSequenceEnv
from PaST.batch_dp_solver import BatchSequenceDPSolver
from PaST.baselines_sequence_dp import dp_schedule_for_job_sequence, DPResult


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class EASQConfig:
    """Configuration for EAS on Q-sequence models."""
    
    # EAS Layer architecture
    layer_hidden_dim: int = 64
    """Hidden dimension of the residual MLP."""
    
    # Optimization
    learning_rate: float = 0.005
    """Learning rate for EAS parameter updates."""
    
    il_weight: float = 0.02
    """Weight λ for imitation learning loss (J_IL)."""
    
    # SGBS parameters
    sgbs_beta: int = 4
    """Beam width for SGBS."""
    
    sgbs_gamma: int = 4
    """Expansion factor for SGBS."""
    
    # Sampling per iteration
    samples_per_iter: int = 16
    """Number of solutions sampled per iteration (M)."""
    
    # Search budget
    max_iterations: int = 100
    """Maximum number of SGBS+EAS iterations."""
    
    # Baseline type
    baseline_type: str = "incumbent"
    """Baseline for REINFORCE: 'incumbent' or 'mean'."""
    
    # Early stopping
    patience: int = 20
    """Early stopping after N iterations without improvement."""
    
    min_improvement: float = 1e-6
    """Minimum improvement to reset patience counter."""
    
    # Temperature for sampling
    sample_temperature: float = 1.0
    """Temperature for stochastic sampling during EAS."""


@dataclass
class EASQResult:
    """Result from EAS Q-sequence search."""
    
    best_energy: float
    """Best total energy found."""
    
    best_sequence: List[int]
    """Job sequence for best solution."""
    
    iterations: int
    """Number of iterations performed."""
    
    sgbs_calls: int = 0
    """Number of SGBS calls made."""
    
    improvement_history: List[float] = field(default_factory=list)
    """Energy at each iteration."""
    
    time_seconds: float = 0.0
    """Total search time."""
    
    greedy_energy: float = float("inf")
    """Initial greedy energy for comparison."""


# =============================================================================
# EAS Residual Layer
# =============================================================================


class EASQResidualLayer(nn.Module):
    """
    Residual layer for EAS on Q-sequence models.
    
    Adds a learned perturbation to job embeddings:
        job_emb' = job_emb + MLP(job_emb)
    
    Initialized with zeros so the layer starts as identity.
    """
    
    def __init__(self, d_model: int, hidden_dim: int = 64):
        super().__init__()
        
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        
        # Initialize to zeros (identity at start)
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply residual transformation: x + MLP(x)."""
        residual = F.relu(self.fc1(x))
        residual = self.fc2(residual)
        return x + residual


# =============================================================================
# EAS Q-Model Wrapper
# =============================================================================


class EASQModelWrapper(nn.Module):
    """
    Wraps a Q-sequence model with trainable EAS residual layers.
    
    The original model parameters (θ) are frozen.
    Only the EAS layer parameters (ψ) are trainable.
    """
    
    def __init__(self, q_model: nn.Module, config: EASQConfig):
        super().__init__()
        
        self.q_model = q_model
        self.config = config
        
        # Freeze base model parameters
        for param in self.q_model.parameters():
            param.requires_grad = False
        
        # Get model dimensions from config or encoder
        if hasattr(q_model, 'model_config') and hasattr(q_model.model_config, 'd_model'):
            d_model = q_model.model_config.d_model
        elif hasattr(q_model, 'config') and hasattr(q_model.config, 'model'):
            d_model = q_model.config.model.d_model
        elif hasattr(q_model, 'encoder') and hasattr(q_model.encoder, 'model_config'):
            d_model = q_model.encoder.model_config.d_model
        else:
            d_model = 128  # fallback
        
        self.d_model = d_model
        
        # Create EAS residual layers
        self.eas_job_layer = EASQResidualLayer(d_model, config.layer_hidden_dim)
        self.eas_ctx_layer = EASQResidualLayer(d_model, config.layer_hidden_dim)
    
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
        **kwargs
    ) -> Tensor:
        """
        Forward pass with EAS residual layers applied.
        
        Returns:
            q_values: [B, N] Q-values (lower = better)
        """
        # Get encoder to access embeddings
        if hasattr(self.q_model, 'encoder'):
            # Full QSequenceNet with encoder
            job_emb, ctx_emb, global_emb = self.q_model.encoder(
                jobs=jobs,
                periods_local=periods_local,
                ctx=ctx,
                job_mask=job_mask,
                **kwargs
            )
            
            # Apply EAS residual layers
            job_emb = self.eas_job_layer(job_emb)
            ctx_emb = self.eas_ctx_layer(ctx_emb)
            
            # Forward through Q-head
            q_values = self.q_model.q_head(job_emb, ctx_emb, global_emb, job_mask)
            
        else:
            # Fallback: just call forward (no embedding modification)
            q_values = self.q_model(jobs, periods_local, ctx, job_mask, **kwargs)
        
        return q_values
    
    def get_logits(
        self,
        jobs: Tensor,
        periods_local: Tensor,
        ctx: Tensor,
        job_mask: Optional[Tensor] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> Tensor:
        """Convert Q-values to logits for sampling."""
        q_values = self.forward(jobs, periods_local, ctx, job_mask, **kwargs)
        logits = -q_values / temperature
        if job_mask is not None:
            logits = logits.masked_fill(job_mask, float("-inf"))
        return logits


# =============================================================================
# SGBS for Q-Sequence (Integrated with EAS)
# =============================================================================


def _slice_single_instance(batch_data: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Extract a single instance from batch data."""
    out: Dict[str, Any] = {}
    for k, v in batch_data.items():
        if isinstance(v, np.ndarray):
            out[k] = v[index:index + 1]
        elif torch.is_tensor(v):
            out[k] = v[index:index + 1]
        else:
            out[k] = v
    return out


def sgbs_q_sequence_with_eas(
    model: EASQModelWrapper,
    variant_config,
    single_data: Dict[str, Any],
    device: torch.device,
    beta: int,
    gamma: int,
    temperature: float = 1.0,
) -> DPResult:
    """
    SGBS for Q-sequence model with EAS wrapper.
    
    Returns:
        DPResult with best sequence and energy found
    """
    env_config = variant_config.env
    n = int(single_data["n_jobs"][0])
    
    if n <= 2 or (beta == 1 and gamma == 1):
        # Small instance: just do greedy
        return _greedy_decode(model, variant_config, single_data, device)
    
    N_max = env_config.N_job_pad
    
    # Initialize beam
    beam_sequences = torch.zeros((1, N_max), dtype=torch.long, device=device)
    beam_avail = torch.zeros((1, N_max), dtype=torch.float32, device=device)
    beam_avail[:, :n] = 1.0
    
    best_energy = float("inf")
    best_sequence = list(range(n))
    
    def _repeat_single(data: Dict, reps: int) -> Dict:
        out = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                out[k] = np.repeat(v, reps, axis=0)
            elif torch.is_tensor(v):
                out[k] = v.repeat(reps, *([1] * (v.dim() - 1)))
            else:
                out[k] = v
        return out
    
    for step in range(n):
        beam_size = int(beam_sequences.shape[0])
        
        # Create env for beam
        env_beam = GPUBatchSequenceEnv(
            batch_size=beam_size,
            env_config=env_config,
            device=device,
        )
        env_beam.reset(_repeat_single(single_data, beam_size))
        env_beam.job_sequences = beam_sequences.clone()
        env_beam.job_available = beam_avail.clone()
        env_beam.step_indices = torch.full((beam_size,), step, dtype=torch.long, device=device)
        env_beam.done = torch.zeros(beam_size, dtype=torch.bool, device=device)
        
        obs = env_beam._get_obs()
        
        # Get job mask
        job_mask_float = obs.get("job_mask", env_beam.job_available)
        job_mask = job_mask_float < 0.5 if job_mask_float.dtype != torch.bool else ~job_mask_float
        
        # Get Q-values from EAS model
        with torch.no_grad():
            q_values = model(
                jobs=obs["jobs"],
                periods_local=obs["periods"],
                ctx=obs["ctx"],
                job_mask=job_mask,
            )
        q_values = q_values.masked_fill(job_mask, float("inf"))
        
        # Expansion: top-γ jobs with lowest Q
        k = min(gamma, n - step)
        neg_q = -q_values[:, :n]
        neg_q = neg_q.masked_fill(job_mask[:, :n], float("-inf"))
        top_jobs = torch.topk(neg_q, k=k, dim=-1).indices  # [beam, k]
        
        # Materialize candidates
        cand_sequences = beam_sequences.unsqueeze(1).repeat(1, k, 1).reshape(-1, N_max)
        cand_avail = beam_avail.unsqueeze(1).repeat(1, k, 1).reshape(-1, N_max)
        chosen = top_jobs.reshape(-1)
        
        # Filter invalid
        C = int(chosen.shape[0])
        idx = torch.arange(C, device=device)
        choice_valid = cand_avail[idx, chosen] > 0.5
        if not choice_valid.any():
            break
        
        cand_sequences = cand_sequences[choice_valid].clone()
        cand_avail = cand_avail[choice_valid].clone()
        chosen = chosen[choice_valid]
        
        # Apply choices
        C_valid = int(chosen.shape[0])
        idx2 = torch.arange(C_valid, device=device)
        cand_sequences[idx2, step] = chosen
        cand_avail[idx2, chosen] = 0.0
        
        # Simulation: complete with greedy rollout
        env_roll = GPUBatchSequenceEnv(
            batch_size=C_valid,
            env_config=env_config,
            device=device,
        )
        env_roll.reset(_repeat_single(single_data, C_valid))
        env_roll.job_sequences = cand_sequences.clone()
        env_roll.job_available = cand_avail.clone()
        env_roll.step_indices = torch.full((C_valid,), step + 1, dtype=torch.long, device=device)
        env_roll.done = torch.zeros(C_valid, dtype=torch.bool, device=device)
        
        # Complete sequences with greedy
        for rollout_step in range(step + 1, n):
            if env_roll.done.all():
                break
            
            obs_r = env_roll._get_obs()
            jm_r = obs_r.get("job_mask", env_roll.job_available)
            jm_r = jm_r < 0.5 if jm_r.dtype != torch.bool else ~jm_r
            
            with torch.no_grad():
                q_r = model(
                    jobs=obs_r["jobs"],
                    periods_local=obs_r["periods"],
                    ctx=obs_r["ctx"],
                    job_mask=jm_r,
                )
            q_r = q_r.masked_fill(jm_r, float("inf"))
            action = q_r.argmin(dim=-1)
            env_roll.step(action)
        
        # Score with batch DP
        costs = BatchSequenceDPSolver.solve(
            job_sequences=env_roll.job_sequences,
            processing_times=env_roll.p_subset,
            ct=env_roll.ct,
            e_single=env_roll.e_single,
            T_limit=env_roll.T_limit,
            sequence_lengths=env_roll.n_jobs.long(),
        )
        
        # Track best
        best_idx = int(costs.argmin().item())
        best_cost = float(costs[best_idx].item())
        if best_cost < best_energy:
            best_energy = best_cost
            best_seq_t = env_roll.job_sequences[best_idx, :n].detach().cpu().long()
            best_sequence = [int(x) for x in best_seq_t.tolist()]
        
        # Prune: keep top-β
        keep = min(beta, C_valid)
        keep_idx = torch.topk(-costs, k=keep).indices
        beam_sequences = cand_sequences[keep_idx].clone()
        beam_avail = cand_avail[keep_idx].clone()
    
    return dp_schedule_for_job_sequence(single_data, best_sequence)


def _greedy_decode(
    model: nn.Module,
    variant_config,
    single_data: Dict[str, Any],
    device: torch.device,
) -> DPResult:
    """Greedy decode for Q-sequence model."""
    env_config = variant_config.env
    n = int(single_data["n_jobs"][0])
    
    env = GPUBatchSequenceEnv(batch_size=1, env_config=env_config, device=device)
    env.reset(single_data)
    
    sequence = []
    
    for _ in range(n):
        obs = env._get_obs()
        job_mask = obs.get("job_mask", env.job_available)
        job_mask = job_mask < 0.5 if job_mask.dtype != torch.bool else ~job_mask
        
        with torch.no_grad():
            q_values = model(
                jobs=obs["jobs"],
                periods_local=obs["periods"],
                ctx=obs["ctx"],
                job_mask=job_mask,
            )
        q_values = q_values.masked_fill(job_mask, float("inf"))
        action = q_values.argmin(dim=-1)
        sequence.append(int(action.item()))
        
        _, _, done, _ = env.step(action)
        if done.all():
            break
    
    return dp_schedule_for_job_sequence(single_data, sequence)


# =============================================================================
# EAS Q-Sequence Runner
# =============================================================================


class EASQSequenceRunner:
    """
    Runs SGBS+EAS hybrid search for Q-sequence models.
    
    Implements Algorithm 2 from the SGBS paper:
    1. Initialize incumbent via greedy
    2. For each iteration:
       a. Run SGBS to explore
       b. Sample M solutions
       c. Update incumbent if better found
       d. Compute RL + IL gradients
       e. Update EAS parameters
    """
    
    def __init__(
        self,
        q_model: nn.Module,
        variant_config,
        device: torch.device,
        config: Optional[EASQConfig] = None,
    ):
        self.q_model = q_model
        self.variant_config = variant_config
        self.device = device
        self.config = config or EASQConfig()
        
        # Create wrapped model
        self.eas_model = EASQModelWrapper(q_model, self.config)
        self.eas_model.to(device)
    
    def search(
        self,
        single_data: Dict[str, Any],
        max_time: Optional[float] = None,
    ) -> EASQResult:
        """
        Run SGBS+EAS search for a single instance.
        
        Args:
            single_data: Single-instance data (batch_size=1)
            max_time: Optional time limit in seconds
        
        Returns:
            EASQResult with best solution found
        """
        start_time = time.time()
        config = self.config
        
        # Reset EAS layers
        self.eas_model.reset_eas_parameters()
        
        # Create optimizer for EAS parameters
        optimizer = torch.optim.Adam(
            self.eas_model.get_eas_parameters(),
            lr=config.learning_rate,
        )
        
        # Get initial greedy incumbent
        greedy_result = _greedy_decode(
            self.eas_model, self.variant_config, single_data, self.device
        )
        
        best_energy = float(greedy_result.total_energy)
        best_sequence = list(greedy_result.job_sequence)
        incumbent_sequence = best_sequence.copy()
        
        improvement_history = [best_energy]
        sgbs_calls = 0
        patience_counter = 0
        
        n_jobs = int(single_data["n_jobs"][0])
        
        for iteration in range(config.max_iterations):
            # Check time limit
            if max_time is not None and (time.time() - start_time) > max_time:
                break
            
            # === SGBS exploration ===
            sgbs_result = sgbs_q_sequence_with_eas(
                model=self.eas_model,
                variant_config=self.variant_config,
                single_data=single_data,
                device=self.device,
                beta=config.sgbs_beta,
                gamma=config.sgbs_gamma,
            )
            sgbs_calls += 1
            
            if sgbs_result.total_energy < best_energy:
                best_energy = float(sgbs_result.total_energy)
                best_sequence = list(sgbs_result.job_sequence)
                incumbent_sequence = best_sequence.copy()
            
            # === Sample M solutions ===
            all_log_probs = []
            all_energies = []
            
            env = GPUBatchSequenceEnv(
                batch_size=1,
                env_config=self.variant_config.env,
                device=self.device,
            )
            
            for _ in range(config.samples_per_iter):
                env.reset(single_data)
                
                step_log_probs = []
                seq = []
                
                for step in range(n_jobs):
                    obs = env._get_obs()
                    job_mask = obs.get("job_mask", env.job_available)
                    job_mask = job_mask < 0.5 if job_mask.dtype != torch.bool else ~job_mask
                    
                    logits = self.eas_model.get_logits(
                        jobs=obs["jobs"],
                        periods_local=obs["periods"],
                        ctx=obs["ctx"],
                        job_mask=job_mask,
                        temperature=config.sample_temperature,
                    )
                    
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    
                    step_log_probs.append(log_prob[0])
                    seq.append(int(action.item()))
                    
                    _, _, done, _ = env.step(action)
                    if done.all():
                        break
                
                # Compute energy with DP
                result = dp_schedule_for_job_sequence(single_data, seq)
                energy = float(result.total_energy)
                
                all_log_probs.append(step_log_probs)
                all_energies.append(energy)
                
                if energy < best_energy:
                    best_energy = energy
                    best_sequence = seq.copy()
                    incumbent_sequence = seq.copy()
            
            # === Compute losses ===
            # Baseline
            if config.baseline_type == "incumbent":
                baseline = -best_energy  # Return convention
            else:
                baseline = -sum(all_energies) / len(all_energies)
            
            # RL loss (REINFORCE)
            rl_loss = torch.tensor(0.0, device=self.device)
            for step_lps, energy in zip(all_log_probs, all_energies):
                ret = -energy  # Return = -cost
                advantage = ret - baseline
                episode_log_prob = sum(step_lps)
                rl_loss = rl_loss - advantage * episode_log_prob
            rl_loss = rl_loss / len(all_log_probs)
            
            # IL loss (imitation on incumbent)
            env.reset(single_data)
            il_loss = torch.tensor(0.0, device=self.device)
            
            for target_action in incumbent_sequence:
                obs = env._get_obs()
                job_mask = obs.get("job_mask", env.job_available)
                job_mask = job_mask < 0.5 if job_mask.dtype != torch.bool else ~job_mask
                
                logits = self.eas_model.get_logits(
                    jobs=obs["jobs"],
                    periods_local=obs["periods"],
                    ctx=obs["ctx"],
                    job_mask=job_mask,
                )
                
                dist = torch.distributions.Categorical(logits=logits)
                action_t = torch.tensor([target_action], device=self.device)
                log_prob = dist.log_prob(action_t)
                il_loss = il_loss - log_prob[0]
                
                env.step(action_t)
            
            # Combined loss
            total_loss = rl_loss + config.il_weight * il_loss
            
            # Gradient update
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Track improvement
            improvement_history.append(best_energy)
            
            # Early stopping
            if len(improvement_history) >= 2:
                improvement = improvement_history[-2] - improvement_history[-1]
                if improvement < config.min_improvement:
                    patience_counter += 1
                else:
                    patience_counter = 0
            
            if patience_counter >= config.patience:
                break
        
        elapsed = time.time() - start_time
        
        return EASQResult(
            best_energy=best_energy,
            best_sequence=best_sequence,
            iterations=iteration + 1,
            sgbs_calls=sgbs_calls,
            improvement_history=improvement_history,
            time_seconds=elapsed,
            greedy_energy=float(greedy_result.total_energy),
        )


# =============================================================================
# Batch Runner (parallel across instances)
# =============================================================================


def eas_q_batch(
    q_model: nn.Module,
    variant_config,
    device: torch.device,
    batch_data: Dict[str, Any],
    config: Optional[EASQConfig] = None,
    max_time_per_instance: Optional[float] = None,
    n_workers: int = 1,
) -> List[EASQResult]:
    """
    Run EAS on a batch of instances.
    
    Currently runs sequentially; each instance gets its own EAS parameters.
    Future: Add multiprocessing for CPU parallelism.
    
    Args:
        q_model: Pre-trained Q-sequence model
        variant_config: Variant configuration
        device: Torch device
        batch_data: Batched instance data
        config: EAS configuration
        max_time_per_instance: Time limit per instance
        n_workers: Number of parallel workers (currently unused)
    
    Returns:
        List of EASQResult for each instance
    """
    batch_size = batch_data["n_jobs"].shape[0]
    results = []
    
    runner = EASQSequenceRunner(q_model, variant_config, device, config)
    
    for i in range(batch_size):
        single_data = _slice_single_instance(batch_data, i)
        result = runner.search(single_data, max_time=max_time_per_instance)
        results.append(result)
    
    return results

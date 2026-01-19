"""
Q-Sequence Training Script for PaST-SM.

Train a Q-value model for job sequencing using supervised regression on DP costs.
Uses DAgger-style iterative data collection with counterfactual rollouts.

Key features:
- Uses real GPUBatchSequenceEnv for observation generation (matching inference)
- Batched DP evaluation for counterfactuals (efficient GPU/CPU)
- Model-based completion after warmup rounds (true DAgger)
- Dueling Q-head: V(s) + A(s,j) decomposition for stability

Usage:
    python -m PaST.train_q_sequence --variant_id q_sequence --seed 0 --smoke_test
    python -m PaST.train_q_sequence --variant_id q_sequence_ctx13 --num_workers 8
"""

import argparse
import json
import os
import random
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from PaST.config import (
    VariantID,
    VariantConfig,
    get_variant_config,
    DataConfig,
)
from PaST.q_sequence_model import build_q_model, QSequenceNet
from PaST.batch_dp_solver import BatchSequenceDPSolver
from PaST.sm_benchmark_data import generate_episode_batch
from PaST.sequence_env import GPUBatchSequenceEnv


TRAIN_VERSION = "2.0-QSEQ"


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class QRunConfig:
    """Training configuration for Q-sequence."""

    # Variant
    variant_id: str = "q_sequence"
    seed: int = 0
    run_name: Optional[str] = None

    # Data generation
    batch_size: int = 128  # Training batch size
    episodes_per_round: int = 1024  # Episodes to collect per DAgger round
    num_rounds: int = 100  # Total DAgger rounds
    buffer_size: int = 100_000  # Max transitions to keep in buffer
    collection_batch_size: int = 64  # Episodes per collection batch
    num_collection_workers: int = 0  # 0 = auto (CPU only), 1 = no multiprocessing
    num_dataloader_workers: int = 0  # 0 = auto
    num_cpu_threads: int = 0  # 0 = auto (all cores)

    # Counterfactual exploration
    num_counterfactuals: int = 8  # Number of jobs to try at each step
    exploration_eps: float = 0.2  # Epsilon for random exploration in data collection
    warmup_rounds: int = 5  # Initial rounds use SPT completion (bootstrap)

    # Model training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    num_epochs_per_round: int = 10

    # Loss
    loss_type: str = "huber"  # "huber", "mse"
    huber_delta: float = 1.0

    # Temperature for Q->logits conversion
    temperature: float = 1.0

    # Evaluation
    eval_every_rounds: int = 5
    num_eval_instances: int = 256
    eval_seed: int = 12345

    # Checkpointing
    save_every_rounds: int = 10

    # Device
    device: str = "cuda"

    # Output
    output_dir: str = "runs_q"

    # Debug
    smoke_test: bool = False


# =============================================================================
# Dataset
# =============================================================================


@dataclass
class QTransition:
    """A single Q-learning transition with env-produced observations."""

    # Observation tensors (from env._get_obs())
    jobs: np.ndarray  # (N, F_job)
    periods: np.ndarray  # (K, F_period)
    ctx: np.ndarray  # (F_ctx,)
    job_avail: np.ndarray  # (N,) float - 1=available, 0=unavailable (env convention)

    # Target
    action: int  # Job index chosen
    q_target: float  # DP cost of completed sequence


class QTransitionBuffer:
    """Replay buffer for Q-transitions with fixed capacity."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[QTransition] = []
        self.position = 0

    def push(self, transition: QTransition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def extend(self, transitions: List[QTransition]):
        for t in transitions:
            self.push(t)

    def sample(self, batch_size: int) -> List[QTransition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class QTransitionDataset(Dataset):
    """PyTorch Dataset wrapping Q-transitions."""

    def __init__(self, transitions: List[QTransition]):
        self.transitions = transitions

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        t = self.transitions[idx]
        return {
            "jobs": torch.from_numpy(t.jobs).float(),
            "periods": torch.from_numpy(t.periods).float(),
            "ctx": torch.from_numpy(t.ctx).float(),
            "job_avail": torch.from_numpy(t.job_avail).float(),
            "action": torch.tensor(t.action, dtype=torch.long),
            "q_target": torch.tensor(t.q_target, dtype=torch.float32),
        }


def collate_q_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate for fixed-size padded job/period tensors."""
    B = len(batch)

    # Since we store padded observations, all should have same size
    # But still handle variable sizes for safety
    max_jobs = max(b["jobs"].shape[0] for b in batch)
    max_periods = max(b["periods"].shape[0] for b in batch)
    job_feat = batch[0]["jobs"].shape[1]  # F_job
    period_feat = batch[0]["periods"].shape[1]  # F_period
    ctx_feat = batch[0]["ctx"].shape[0]  # F_ctx

    # Allocate padded tensors
    jobs = torch.zeros(B, max_jobs, job_feat)
    periods = torch.zeros(B, max_periods, period_feat)
    ctx = torch.zeros(B, ctx_feat)
    job_mask = torch.ones(B, max_jobs, dtype=torch.bool)  # True = INVALID for model
    period_mask = torch.ones(B, max_periods, dtype=torch.bool)
    actions = torch.zeros(B, dtype=torch.long)
    q_targets = torch.zeros(B)

    for i, b in enumerate(batch):
        n_jobs = b["jobs"].shape[0]
        n_periods = b["periods"].shape[0]
        avail_len = b["job_avail"].shape[0]

        jobs[i, :n_jobs] = b["jobs"]
        periods[i, :n_periods] = b["periods"]
        ctx[i] = b["ctx"]

        # Convert env availability (1=available) to model mask (True=invalid)
        # job_avail from env is float with 1=available, 0=unavailable
        # Handle potential size mismatch between jobs and availability
        min_len = min(n_jobs, avail_len, max_jobs)
        job_mask[i, :min_len] = b["job_avail"][:min_len] < 0.5  # True where unavailable
        # Positions beyond min_len remain True (invalid) as initialized
        period_mask[i, :n_periods] = False  # Valid periods

        actions[i] = b["action"]
        q_targets[i] = b["q_target"]

    return {
        "jobs": jobs,
        "periods": periods,
        "ctx": ctx,
        "job_mask": job_mask,
        "period_mask": period_mask,
        "actions": actions,
        "q_targets": q_targets,
    }


# =============================================================================
# Batched DP Evaluation
# =============================================================================


def batch_evaluate_sequences(
    sequences: List[List[int]],
    processing_times: torch.Tensor,  # (B_orig, N_pad)
    ct: torch.Tensor,  # (B_orig, T)
    e_single: torch.Tensor,  # (B_orig,)
    T_limit: torch.Tensor,  # (B_orig,)
    instance_indices: List[int],  # Which original instance each sequence belongs to
    device: torch.device,
) -> torch.Tensor:
    """
    Batch evaluate multiple sequences using the DP solver.

    Args:
        sequences: List of job sequences (each is list of job indices)
        processing_times: Original instance processing times (padded)
        ct: Original instance price curves
        e_single: Original instance energy rates
        T_limit: Original instance deadlines
        instance_indices: Maps each sequence to its source instance
        device: Device for computation

    Returns:
        costs: (len(sequences),) DP costs
    """
    if not sequences:
        return torch.tensor([], device=device)

    B_seq = len(sequences)
    N_pad = processing_times.shape[1]  # Original padded width

    # Create sequence tensor with same width as processing_times
    # (solver requires job_sequences.shape[1] == processing_times.shape[1])
    job_sequences = torch.zeros((B_seq, N_pad), dtype=torch.long, device=device)
    sequence_lengths = torch.zeros(B_seq, dtype=torch.long, device=device)

    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        job_sequences[i, :seq_len] = torch.tensor(seq, dtype=torch.long, device=device)
        sequence_lengths[i] = seq_len

    # Gather instance data for each sequence
    idx = torch.tensor(instance_indices, dtype=torch.long, device=device)
    p_batch = processing_times[idx]  # (B_seq, N_pad)
    ct_batch = ct[idx]  # (B_seq, T)
    e_batch = e_single[idx]  # (B_seq,)
    T_batch = T_limit[idx]  # (B_seq,)

    # Call batched DP solver
    costs = BatchSequenceDPSolver.solve(
        job_sequences=job_sequences,
        processing_times=p_batch,
        ct=ct_batch,
        e_single=e_batch,
        T_limit=T_batch,
        sequence_lengths=sequence_lengths,
    )

    return costs


# =============================================================================
# Data Collection using Real Environment
# =============================================================================


def complete_sequence_spt(
    partial_sequence: List[int],
    remaining_jobs: List[int],
    processing_times: np.ndarray,
) -> List[int]:
    """Complete sequence using SPT (shortest processing time) heuristic."""
    if not remaining_jobs:
        return partial_sequence

    remaining_sorted = sorted(remaining_jobs, key=lambda j: processing_times[j])
    return partial_sequence + remaining_sorted


def complete_sequence_model(
    partial_sequence: List[int],
    remaining_jobs: List[int],
    model: QSequenceNet,
    obs_template: Dict[str, torch.Tensor],
    job_available: np.ndarray,
    processing_times: np.ndarray,
    F_job: int,
    device: torch.device,
) -> List[int]:
    """
    Complete sequence using model's greedy policy (argmin Q).

    Updates observations to be consistent with availability:
    - jobs[..., 1] = availability channel (when F_job >= 2)
    - ctx[2] = remaining_work (sum of p for available jobs)
    """
    if not remaining_jobs:
        return partial_sequence

    sequence = list(partial_sequence)
    available = job_available.copy()
    remaining = list(remaining_jobs)

    # Clone obs_template so we can modify it
    jobs = obs_template["jobs"].clone()
    periods = obs_template["periods"].clone()
    ctx = obs_template["ctx"].clone()

    model.eval()
    with torch.no_grad():
        while remaining:
            # Update availability-dependent observation features
            if F_job >= 2:
                jobs[:, 1] = torch.from_numpy(available).float()

            # Update remaining_work in ctx (ctx[2])
            p_tensor = torch.from_numpy(processing_times).float()
            avail_tensor = torch.from_numpy(available).float()
            remaining_work = (p_tensor * avail_tensor).sum()
            ctx[2] = remaining_work

            # Build batched input
            jobs_t = jobs.unsqueeze(0).to(device)
            periods_t = periods.unsqueeze(0).to(device)
            ctx_t = ctx.unsqueeze(0).to(device)

            # Convert availability to model mask (True = invalid)
            mask = torch.from_numpy(available < 0.5).unsqueeze(0).to(device)

            # Get Q-values
            q_values = model(jobs_t, periods_t, ctx_t, mask)

            # Select argmin Q over available jobs
            q_masked = q_values.clone()
            q_masked[0, mask[0]] = float("inf")
            action = q_masked.argmin(dim=-1).item()

            sequence.append(action)
            remaining.remove(action)
            available[action] = 0.0

    return sequence


def _collect_round_batch(
    *,
    env_config,
    data_config: DataConfig,
    variant_config: VariantConfig,
    model_state: Optional[Dict[str, torch.Tensor]],
    batch_size: int,
    num_counterfactuals: int,
    exploration_eps: float,
    use_model_completion: bool,
    device_str: str,
    seed: int,
    num_cpu_threads: int,
) -> List[QTransition]:
    """Collect transitions for a single batch (worker-safe)."""
    if num_cpu_threads > 0:
        try:
            torch.set_num_threads(num_cpu_threads)
            torch.set_num_interop_threads(min(4, num_cpu_threads))
        except Exception:
            pass

    rng = random.Random(seed)

    device = torch.device(device_str)
    model: Optional[QSequenceNet] = None
    if model_state is not None:
        model = build_q_model(variant_config)
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()

    env = GPUBatchSequenceEnv(
        batch_size=batch_size,
        env_config=env_config,
        device=device,
    )

    batch_data = generate_episode_batch(
        batch_size=batch_size,
        config=data_config,
        seed=seed,
    )

    obs = env.reset(batch_data)

    p_all = env.p_subset.clone()
    ct_all = env.ct.clone()
    e_all = env.e_single.clone()
    T_limit_all = env.T_limit.clone()
    n_jobs_all = env.n_jobs.clone()

    transitions: List[QTransition] = []
    F_job = env_config.F_job
    N_pad = obs["jobs"].shape[1]

    for inst_idx in range(batch_size):
        n_jobs = int(n_jobs_all[inst_idx].item())
        p_np = p_all[inst_idx].cpu().numpy()

        obs_torch = {
            "jobs": obs["jobs"][inst_idx].clone(),
            "periods": obs["periods"][inst_idx].clone(),
            "ctx": obs["ctx"][inst_idx].clone(),
        }

        partial_sequence = []
        remaining_jobs = list(range(n_jobs))
        job_available = np.zeros(N_pad, dtype=np.float32)
        job_available[:n_jobs] = 1.0

        p_inst = p_all[inst_idx : inst_idx + 1]
        ct_inst = ct_all[inst_idx : inst_idx + 1]
        e_inst = e_all[inst_idx : inst_idx + 1]
        T_inst = T_limit_all[inst_idx : inst_idx + 1]

        for _ in range(n_jobs):
            if not remaining_jobs:
                break

            if F_job >= 2:
                obs_torch["jobs"][:, 1] = torch.from_numpy(job_available).float()

            remaining_work = (p_np * job_available).sum()
            obs_torch["ctx"][2] = remaining_work

            obs_single = {
                "jobs": obs_torch["jobs"].cpu().numpy(),
                "periods": obs_torch["periods"].cpu().numpy(),
                "ctx": obs_torch["ctx"].cpu().numpy(),
            }

            if rng.random() < exploration_eps or model is None:
                action = rng.choice(remaining_jobs)
            else:
                with torch.no_grad():
                    jobs_t = obs_torch["jobs"].unsqueeze(0).to(device)
                    periods_t = obs_torch["periods"].unsqueeze(0).to(device)
                    ctx_t = obs_torch["ctx"].unsqueeze(0).to(device)
                    mask_t = (
                        torch.from_numpy(job_available < 0.5).unsqueeze(0).to(device)
                    )

                    q = model(jobs_t, periods_t, ctx_t, mask_t)
                    q[0, mask_t[0]] = float("inf")
                    action = q.argmin(dim=-1).item()

            if len(remaining_jobs) <= num_counterfactuals:
                candidates = remaining_jobs.copy()
            else:
                candidates = [action]
                others = [j for j in remaining_jobs if j != action]
                candidates.extend(rng.sample(others, num_counterfactuals - 1))

            cf_sequences = []
            cf_instance_indices = []

            for candidate_job in candidates:
                cf_partial = partial_sequence + [candidate_job]
                cf_remaining = [j for j in remaining_jobs if j != candidate_job]

                if use_model_completion and model is not None:
                    cf_available = job_available.copy()
                    cf_available[candidate_job] = 0.0
                    full_seq = complete_sequence_model(
                        cf_partial,
                        cf_remaining,
                        model,
                        obs_torch,
                        cf_available,
                        p_np,
                        F_job,
                        device,
                    )
                else:
                    full_seq = complete_sequence_spt(cf_partial, cf_remaining, p_np)

                cf_sequences.append(full_seq)
                cf_instance_indices.append(0)

            costs = batch_evaluate_sequences(
                sequences=cf_sequences,
                processing_times=p_inst,
                ct=ct_inst,
                e_single=e_inst,
                T_limit=T_inst,
                instance_indices=cf_instance_indices,
                device=device,
            )

            for cand_idx, candidate_job in enumerate(candidates):
                cf_cost = costs[cand_idx].item()
                if np.isfinite(cf_cost):
                    transitions.append(
                        QTransition(
                            jobs=obs_single["jobs"].copy(),
                            periods=obs_single["periods"].copy(),
                            ctx=obs_single["ctx"].copy(),
                            job_avail=job_available.copy(),
                            action=candidate_job,
                            q_target=cf_cost,
                        )
                    )

            partial_sequence.append(action)
            remaining_jobs.remove(action)
            job_available[action] = 0.0

    return transitions


def collect_round_data(
    env_config,  # EnvConfig, not env instance
    model: Optional[QSequenceNet],
    variant_config: VariantConfig,
    data_config: DataConfig,
    num_episodes: int,
    num_counterfactuals: int,
    exploration_eps: float,
    use_model_completion: bool,
    device: torch.device,
    seed: int,
    collection_batch_size: int = 64,  # Fixed batch size for collection
    num_collection_workers: int = 0,
    num_cpu_threads: int = 0,
) -> List[QTransition]:
    """
    Collect Q-transitions for one round using the real environment.

    Creates env with proper batch size for each collection batch.
    Updates observations to be consistent with availability state:
    - jobs[..., 1] = availability (when F_job >= 2)
    - ctx[2] = remaining_work
    """
    transitions: List[QTransition] = []

    batch_size = min(num_episodes, collection_batch_size)
    num_batches = (num_episodes + batch_size - 1) // batch_size

    device_str = str(device)
    model_state = model.state_dict() if model is not None else None

    if device.type != "cpu":
        num_collection_workers = 1

    if num_collection_workers <= 0:
        cpu_count = os.cpu_count() or 1
        num_collection_workers = min(cpu_count, num_batches) if num_batches > 0 else 1

    cpu_count = os.cpu_count() or 1
    if num_cpu_threads <= 0:
        num_cpu_threads = cpu_count

    threads_per_worker = max(1, num_cpu_threads // max(1, num_collection_workers))

    batch_specs = []
    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_episodes - batch_idx * batch_size)
        if current_batch_size <= 0:
            continue
        batch_seed = seed + batch_idx * 10000
        batch_specs.append((current_batch_size, batch_seed))

    if num_collection_workers <= 1:
        for current_batch_size, batch_seed in batch_specs:
            transitions.extend(
                _collect_round_batch(
                    env_config=env_config,
                    data_config=data_config,
                    variant_config=variant_config,
                    model_state=model_state,
                    batch_size=current_batch_size,
                    num_counterfactuals=num_counterfactuals,
                    exploration_eps=exploration_eps,
                    use_model_completion=use_model_completion,
                    device_str=device_str,
                    seed=batch_seed,
                    num_cpu_threads=threads_per_worker,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=num_collection_workers) as executor:
            futures = []
            for current_batch_size, batch_seed in batch_specs:
                futures.append(
                    executor.submit(
                        _collect_round_batch,
                        env_config=env_config,
                        data_config=data_config,
                        variant_config=variant_config,
                        model_state=model_state,
                        batch_size=current_batch_size,
                        num_counterfactuals=num_counterfactuals,
                        exploration_eps=exploration_eps,
                        use_model_completion=use_model_completion,
                        device_str=device_str,
                        seed=batch_seed,
                        num_cpu_threads=threads_per_worker,
                    )
                )

            for future in as_completed(futures):
                transitions.extend(future.result())

    return transitions


# =============================================================================
# Training
# =============================================================================


def train_epoch(
    model: QSequenceNet,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: QRunConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_q_mse = 0.0
    total_q_mae = 0.0
    n_batches = 0

    for batch in dataloader:
        # Move to device
        jobs = batch["jobs"].to(device)
        periods = batch["periods"].to(device)
        ctx = batch["ctx"].to(device)
        job_mask = batch["job_mask"].to(device)
        period_mask = batch["period_mask"].to(device)
        actions = batch["actions"].to(device)
        q_targets = batch["q_targets"].to(device)

        # Forward pass
        q_values = model(
            jobs=jobs,
            periods_local=periods,
            ctx=ctx,
            job_mask=job_mask,
            period_mask=period_mask,
        )

        # Get Q-values for taken actions
        q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Loss
        if config.loss_type == "huber":
            loss = F.huber_loss(q_pred, q_targets, delta=config.huber_delta)
        else:
            loss = F.mse_loss(q_pred, q_targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        optimizer.step()

        # Metrics
        total_loss += loss.item()
        with torch.no_grad():
            total_q_mse += F.mse_loss(q_pred, q_targets).item()
            total_q_mae += (q_pred - q_targets).abs().mean().item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "q_mse": total_q_mse / max(n_batches, 1),
        "q_mae": total_q_mae / max(n_batches, 1),
    }


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_greedy(
    model: QSequenceNet,
    env_config,
    data_config: DataConfig,
    num_instances: int,
    seed: int,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model using greedy action selection on real environment."""
    model.eval()

    total_cost_model = 0.0
    total_cost_random = 0.0
    total_cost_spt = 0.0
    n_valid = 0

    # Create a fresh env with the exact batch size we need
    eval_env = GPUBatchSequenceEnv(
        batch_size=num_instances,
        env_config=env_config,
        device=device,
    )

    # Generate all eval instances at once
    batch_data = generate_episode_batch(
        batch_size=num_instances,
        config=data_config,
        seed=seed,
    )

    # Reset env
    obs = eval_env.reset(batch_data)

    B = num_instances
    n_jobs_all = eval_env.n_jobs.clone()
    p_all = eval_env.p_subset.clone()
    ct_all = eval_env.ct.clone()
    e_all = eval_env.e_single.clone()
    T_limit_all = eval_env.T_limit.clone()
    F_job = env_config.F_job

    with torch.no_grad():
        for inst_idx in range(B):
            n_jobs = int(n_jobs_all[inst_idx].item())
            N_pad = obs["jobs"].shape[1]  # Padded dimension
            p_np = p_all[inst_idx].cpu().numpy()  # Processing times for this instance

            # Model greedy rollout
            # job_available is full padded size; padding positions start as unavailable
            job_available = np.zeros(N_pad, dtype=np.float32)
            job_available[:n_jobs] = 1.0  # Only actual jobs are available
            model_sequence = []

            obs_torch = {
                "jobs": obs["jobs"][inst_idx].clone(),
                "periods": obs["periods"][inst_idx].clone(),
                "ctx": obs["ctx"][inst_idx].clone(),
            }

            for _ in range(n_jobs):
                # Update observations to be consistent with availability
                if F_job >= 2:
                    obs_torch["jobs"][:, 1] = torch.from_numpy(job_available).float()
                remaining_work = (p_np * job_available).sum()
                obs_torch["ctx"][2] = remaining_work

                jobs_t = obs_torch["jobs"].unsqueeze(0).to(device)
                periods_t = obs_torch["periods"].unsqueeze(0).to(device)
                ctx_t = obs_torch["ctx"].unsqueeze(0).to(device)
                # Mask: True = INVALID (unavailable or padding)
                mask_t = torch.from_numpy(job_available < 0.5).unsqueeze(0).to(device)

                q = model(jobs_t, periods_t, ctx_t, mask_t)
                q[0, mask_t[0]] = float("inf")
                action = q.argmin(dim=-1).item()

                model_sequence.append(action)
                job_available[action] = 0.0

            # Evaluate all sequences
            p_inst = p_all[inst_idx : inst_idx + 1]
            ct_inst = ct_all[inst_idx : inst_idx + 1]
            e_inst = e_all[inst_idx : inst_idx + 1]
            T_inst = T_limit_all[inst_idx : inst_idx + 1]

            # Random sequence
            random_seq = list(range(n_jobs))
            random.shuffle(random_seq)

            # SPT sequence
            p_np = p_all[inst_idx].cpu().numpy()
            spt_seq = sorted(range(n_jobs), key=lambda j: p_np[j])

            # Batch evaluate
            costs = batch_evaluate_sequences(
                sequences=[model_sequence, random_seq, spt_seq],
                processing_times=p_inst,
                ct=ct_inst,
                e_single=e_inst,
                T_limit=T_inst,
                instance_indices=[0, 0, 0],
                device=device,
            )

            model_cost = costs[0].item()
            random_cost = costs[1].item()
            spt_cost = costs[2].item()

            if np.isfinite(model_cost) and np.isfinite(spt_cost):
                total_cost_model += model_cost
                total_cost_random += random_cost
                total_cost_spt += spt_cost
                n_valid += 1

    if n_valid == 0:
        return {"cost": float("inf"), "vs_random": 0.0, "vs_spt": 0.0}

    avg_model = total_cost_model / n_valid
    avg_random = total_cost_random / n_valid
    avg_spt = total_cost_spt / n_valid

    return {
        "cost": avg_model,
        "cost_random": avg_random,
        "cost_spt": avg_spt,
        "vs_random": (
            (avg_random - avg_model) / avg_random * 100 if avg_random > 0 else 0
        ),
        "vs_spt": (avg_spt - avg_model) / avg_spt * 100 if avg_spt > 0 else 0,
        "n_valid": n_valid,
    }


# =============================================================================
# Main
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Train Q-Sequence model")

    parser.add_argument("--variant_id", type=str, default="q_sequence")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--episodes_per_round", type=int, default=1024)
    parser.add_argument("--num_rounds", type=int, default=100)
    parser.add_argument("--collection_batch_size", type=int, default=64)
    parser.add_argument("--num_collection_workers", type=int, default=0)
    parser.add_argument("--num_dataloader_workers", type=int, default=0)
    parser.add_argument("--num_cpu_threads", type=int, default=0)

    parser.add_argument("--num_counterfactuals", type=int, default=8)
    parser.add_argument("--exploration_eps", type=float, default=0.2)
    parser.add_argument("--warmup_rounds", type=int, default=5)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs_per_round", type=int, default=10)

    parser.add_argument("--eval_every_rounds", type=int, default=5)
    parser.add_argument("--num_eval_instances", type=int, default=256)
    parser.add_argument("--save_every_rounds", type=int, default=10)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="runs_q")
    parser.add_argument("--smoke_test", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    # Build config
    config = QRunConfig(
        variant_id=args.variant_id,
        seed=args.seed,
        run_name=args.run_name,
        batch_size=args.batch_size,
        episodes_per_round=args.episodes_per_round,
        num_rounds=args.num_rounds,
        collection_batch_size=args.collection_batch_size,
        num_collection_workers=args.num_collection_workers,
        num_dataloader_workers=args.num_dataloader_workers,
        num_cpu_threads=args.num_cpu_threads,
        num_counterfactuals=args.num_counterfactuals,
        exploration_eps=args.exploration_eps,
        warmup_rounds=args.warmup_rounds,
        learning_rate=args.learning_rate,
        num_epochs_per_round=args.num_epochs_per_round,
        eval_every_rounds=args.eval_every_rounds,
        num_eval_instances=args.num_eval_instances,
        save_every_rounds=args.save_every_rounds,
        device=args.device,
        output_dir=args.output_dir,
        smoke_test=args.smoke_test,
    )

    # Smoke test overrides
    if config.smoke_test:
        config.episodes_per_round = 64
        config.num_rounds = 5
        config.num_epochs_per_round = 2
        config.eval_every_rounds = 2
        config.num_eval_instances = 32
        config.warmup_rounds = 2
        config.collection_batch_size = 16
        config.num_collection_workers = 1
        config.num_dataloader_workers = 0

    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cpu":
        cpu_count = os.cpu_count() or 1
        if config.num_cpu_threads <= 0:
            config.num_cpu_threads = cpu_count
        if config.num_dataloader_workers <= 0:
            config.num_dataloader_workers = min(cpu_count, 8)
        if config.num_collection_workers <= 0:
            config.num_collection_workers = min(cpu_count, 8)
        try:
            torch.set_num_threads(config.num_cpu_threads)
            torch.set_num_interop_threads(min(4, config.num_cpu_threads))
        except Exception:
            pass

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config.run_name or f"{config.variant_id}_s{config.seed}"
    run_dir = Path(config.output_dir) / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Load variant config
    try:
        variant_id = VariantID(config.variant_id)
        variant_config = get_variant_config(variant_id, seed=config.seed)
    except ValueError:
        print(f"Warning: Unknown variant {config.variant_id}, using q_sequence")
        variant_config = get_variant_config(VariantID.Q_SEQUENCE, seed=config.seed)

    data_config = DataConfig()
    env_config = variant_config.env

    # Build model
    model = build_q_model(variant_config)
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Replay buffer
    buffer = QTransitionBuffer(config.buffer_size)

    # Logging
    log_file = open(run_dir / "log.jsonl", "w")

    print(f"\n{'='*60}")
    print(f"Q-Sequence Training: {config.variant_id}")
    print(f"Output: {run_dir}")
    print(f"Warmup rounds (SPT completion): {config.warmup_rounds}")
    print(f"{'='*60}\n")

    best_cost = float("inf")

    for round_idx in range(config.num_rounds):
        round_start = time.time()

        # Determine if using model-based completion
        use_model_completion = round_idx >= config.warmup_rounds
        completion_policy = "model" if use_model_completion else "SPT"

        # === Data Collection ===
        print(
            f"Round {round_idx + 1}/{config.num_rounds} [{completion_policy}]: Collecting...",
            end=" ",
            flush=True,
        )

        # Decay exploration over rounds
        eps = config.exploration_eps * (1 - round_idx / config.num_rounds)

        transitions = collect_round_data(
            env_config=env_config,  # Pass config, not env instance
            model=model if use_model_completion else None,
            variant_config=variant_config,
            data_config=data_config,
            num_episodes=config.episodes_per_round,
            num_counterfactuals=config.num_counterfactuals,
            exploration_eps=eps,
            use_model_completion=use_model_completion,
            device=device,
            seed=config.seed + round_idx * 100000,
            collection_batch_size=config.collection_batch_size,
            num_collection_workers=config.num_collection_workers,
            num_cpu_threads=config.num_cpu_threads,
        )

        buffer.extend(transitions)
        collect_time = time.time() - round_start
        print(f"{len(transitions)} transitions in {collect_time:.1f}s")

        # === Training ===
        print(f"  Training ({len(buffer)} in buffer)...", end=" ", flush=True)
        train_start = time.time()

        # Create dataset from buffer
        dataset = QTransitionDataset(buffer.buffer.copy())
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_q_batch,
            num_workers=config.num_dataloader_workers,
            persistent_workers=config.num_dataloader_workers > 0,
        )

        epoch_metrics = []
        for epoch in range(config.num_epochs_per_round):
            metrics = train_epoch(model, dataloader, optimizer, config, device)
            epoch_metrics.append(metrics)

        avg_loss = np.mean([m["loss"] for m in epoch_metrics])
        avg_mae = np.mean([m["q_mae"] for m in epoch_metrics])
        train_time = time.time() - train_start
        print(f"loss={avg_loss:.4f}, mae={avg_mae:.2f} in {train_time:.1f}s")

        # === Evaluation ===
        eval_results = None
        if (round_idx + 1) % config.eval_every_rounds == 0:
            print(f"  Evaluating...", end=" ", flush=True)
            eval_results = evaluate_greedy(
                model=model,
                env_config=env_config,
                data_config=data_config,
                num_instances=config.num_eval_instances,
                seed=config.eval_seed,
                device=device,
            )
            print(
                f"cost={eval_results['cost']:.2f}, vs_spt={eval_results['vs_spt']:.1f}%"
            )

            # Save best
            if eval_results["cost"] < best_cost:
                best_cost = eval_results["cost"]
                torch.save(model.state_dict(), run_dir / "best_model.pt")

        # === Logging ===
        log_entry = {
            "round": round_idx + 1,
            "time": time.time() - round_start,
            "buffer_size": len(buffer),
            "transitions_collected": len(transitions),
            "completion_policy": completion_policy,
            "exploration_eps": eps,
            "loss": avg_loss,
            "q_mae": avg_mae,
        }
        if eval_results:
            log_entry.update({f"eval_{k}": v for k, v in eval_results.items()})

        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()

        # === Checkpointing ===
        if (round_idx + 1) % config.save_every_rounds == 0:
            torch.save(
                {
                    "round": round_idx + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": asdict(config),
                },
                run_dir / f"checkpoint_{round_idx + 1}.pt",
            )

    # Final save
    torch.save(model.state_dict(), run_dir / "final_model.pt")
    log_file.close()

    print(f"\n{'='*60}")
    print(f"Training complete! Best cost: {best_cost:.2f}")
    print(f"Models saved to: {run_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

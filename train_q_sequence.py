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
import copy
import json
import os
import random
import re
import sys
import time
import warnings
import io
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Suppress warnings globally for training runs (including Gym deprecation noise)
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")
warnings.filterwarnings("ignore")


class _FilteredTextIO(io.TextIOBase):
    """Line-filtering wrapper for stdout/stderr to drop noisy warning prints."""

    def __init__(self, base: io.TextIOBase, *, drop_substrings: List[str]):
        self._base = base
        self._drop = drop_substrings
        self._buf = ""

    def write(self, s: str) -> int:
        if not s:
            return 0

        self._buf += s
        out = []
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line_with_nl = line + "\n"

            # Never suppress tracebacks/exceptions
            if line.startswith("Traceback") or line.startswith("Exception"):
                out.append(line_with_nl)
                continue

            # Drop known noisy messages and generic warning lines
            if any(sub in line for sub in self._drop):
                continue
            if (
                "Warning:" in line
                or "DeprecationWarning" in line
                or "UserWarning" in line
            ):
                continue

            out.append(line_with_nl)

        if out:
            return self._base.write("".join(out))
        return len(s)

    def flush(self) -> None:
        try:
            # Flush any remaining partial line (unless it matches filters)
            if self._buf:
                line = self._buf
                self._buf = ""
                if not any(sub in line for sub in self._drop) and "Warning" not in line:
                    self._base.write(line)
            self._base.flush()
        except Exception:
            pass


_DROP_SUBSTRINGS = [
    "Gym has been unmaintained since 2022",
    "Please upgrade to Gymnasium",
    "See the migration guide at https://gymnasium",
]

# Filter both streams so we also silence messages emitted by worker processes.
try:
    sys.stderr = _FilteredTextIO(sys.stderr, drop_substrings=_DROP_SUBSTRINGS)
    sys.stdout = _FilteredTextIO(sys.stdout, drop_substrings=_DROP_SUBSTRINGS)
except Exception:
    pass

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
    allow_gpu_collection_multiprocessing: bool = (
        False  # opt-in; default keeps 1 worker on CUDA
    )
    num_dataloader_workers: int = 0  # 0 = auto
    num_cpu_threads: int = 0  # 0 = auto (all cores)

    # Counterfactual exploration
    num_counterfactuals: int = 8  # Number of jobs to try at each step
    exploration_eps_start: float = 0.2  # Start epsilon for random exploration
    exploration_eps_end: float = 0.2  # End epsilon (kept equal to start by default)
    exploration_eps_decay_rounds: int = (
        0  # 0 = no decay; else linear over this many rounds
    )
    warmup_rounds: int = 5  # Initial rounds use SPT completion (bootstrap)

    # Completion policy schedule
    completion_policy: str = "model"  # "spt", "model", "mix"
    completion_prob_start: float = 1.0  # For "mix": prob(model) at start
    completion_prob_end: float = 1.0  # For "mix": prob(model) at end
    completion_prob_decay_rounds: int = (
        0  # 0 = no decay; else linear over this many rounds
    )
    # Heuristic used when not using model completion
    # Options: "spt", "lpt", "random", "mixed" (randomly picks spt/lpt/random each time)
    heuristic_policy: str = "mixed"

    # Model training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    num_epochs_per_round: int = 10

    # Loss
    loss_type: str = "huber"  # "huber", "mse"
    huber_delta: float = 1.0

    # Optional listwise ranking loss (off by default)
    listwise_weight: float = 0.0
    listwise_temperature: float = 1.0

    # ---------------------------------------------------------------------
    # New features (opt-in): target aggregation + policy-focused training
    # ---------------------------------------------------------------------

    # Target generation: multiple completions per (s,j) and aggregate.
    # - auto: keep legacy behavior (single completion determined by completion_policy)
    # - otherwise: comma-separated list from {model,spt,lpt,random,mixed}
    target_rollouts: str = "auto"
    # Aggregate rollout costs into a single label per (s,j)
    # - single: legacy (use first/only rollout)
    # - min: min over rollouts
    # - softmin: smooth minimum (temperature-controlled)
    target_rollout_aggregation: str = "single"
    target_num_random_rollouts: int = 2
    target_softmin_tau: float = 1.0

    # Training objective
    # - regression: regress Q(s,j) to aggregated DP cost (legacy)
    # - policy: ranking/classification objective is primary
    train_objective: str = "regression"
    regression_weight: float = 1.0
    policy_weight: float = 0.0
    policy_temperature: float = 1.0
    policy_target: str = "soft"  # soft|hard

    # Top-γ aligned loss (encourage true best action in top-γ)
    top_gamma: int = 0
    top_gamma_weight: float = 0.0
    top_gamma_margin: float = 0.0

    # Slow / checkpointed teacher model for model-based completions
    teacher_update_every_rounds: int = 1
    teacher_update_on_save: bool = True

    # Target shaping (optional)
    # - none: regress to absolute DP cost of the completed sequence
    # - state_min: subtract min DP cost among candidate actions from the same state
    #   (keeps rankings identical, reduces scale/variance)
    target_normalization: str = "none"

    # Candidate set shaping (optional): ensure heuristic candidates are present
    # when sampling counterfactual actions (helps avoid purely random negatives).
    include_heuristic_candidates: bool = False

    # Temperature for Q->logits conversion
    temperature: float = 1.0

    # Evaluation
    eval_every_rounds: int = 5
    num_eval_instances: int = 256
    eval_seed: int = 12345

    # SGBS evaluation (beam search - this is the real metric for the model)
    eval_sgbs: bool = True  # Enable SGBS evaluation
    eval_sgbs_beta: int = 4  # Beam width for SGBS
    eval_sgbs_gamma: int = 4  # Expansion factor for SGBS
    eval_sgbs_num_instances: int = 64  # Fewer instances for SGBS (expensive)

    # Checkpointing
    save_every_rounds: int = 10

    # Device
    device: str = "cuda"

    # Output
    output_dir: str = "runs_q"

    # Debug
    smoke_test: bool = False

    # Curriculum (optional): start with looser deadlines, anneal to target
    curriculum: bool = False
    curriculum_fraction: float = 0.3  # fraction of rounds used for annealing
    curriculum_slack_min: Optional[float] = None
    curriculum_slack_max: Optional[float] = None

    # Resume from checkpoint
    resume_from: Optional[str] = None  # Path to checkpoint file or run directory


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
    state_id: int  # Groups candidates from the same state (for listwise loss)


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
            "state_id": torch.tensor(t.state_id, dtype=torch.long),
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
    state_ids = torch.zeros(B, dtype=torch.long)

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
        state_ids[i] = b["state_id"]

    return {
        "jobs": jobs,
        "periods": periods,
        "ctx": ctx,
        "job_mask": job_mask,
        "period_mask": period_mask,
        "actions": actions,
        "q_targets": q_targets,
        "state_ids": state_ids,
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


def complete_sequence_lpt(
    partial_sequence: List[int],
    remaining_jobs: List[int],
    processing_times: np.ndarray,
) -> List[int]:
    """Complete sequence using LPT (longest processing time) heuristic."""
    if not remaining_jobs:
        return partial_sequence

    remaining_sorted = sorted(remaining_jobs, key=lambda j: -processing_times[j])
    return partial_sequence + remaining_sorted


def complete_sequence_random(
    partial_sequence: List[int],
    remaining_jobs: List[int],
    rng: random.Random,
) -> List[int]:
    """Complete sequence using random shuffle."""
    if not remaining_jobs:
        return partial_sequence

    shuffled = list(remaining_jobs)
    rng.shuffle(shuffled)
    return partial_sequence + shuffled


def _parse_target_rollouts(spec: str) -> List[str]:
    s = (spec or "").strip().lower()
    if not s or s == "auto":
        return ["auto"]
    items = [x.strip() for x in s.split(",") if x.strip()]
    allowed = {"model", "spt", "lpt", "random", "mixed"}
    for it in items:
        if it not in allowed:
            raise ValueError(
                f"Invalid target rollout '{it}'. Allowed: {sorted(allowed)}"
            )
    # de-dup while preserving order
    out: List[str] = []
    seen = set()
    for it in items:
        if it in seen:
            continue
        out.append(it)
        seen.add(it)
    return out


def _aggregate_rollout_costs(
    costs: List[float], aggregation: str, softmin_tau: float
) -> float:
    if not costs:
        return float("nan")
    mode = (aggregation or "single").strip().lower()
    if mode == "single":
        return float(costs[0])
    if mode == "min":
        return float(min(costs))
    if mode == "softmin":
        tau = float(softmin_tau)
        if tau <= 0:
            return float(min(costs))
        c = torch.tensor(costs, dtype=torch.float64)
        # softmin(c) = -tau * logsumexp(-c/tau)
        return float((-tau * torch.logsumexp(-c / tau, dim=0)).item())
    raise ValueError(f"Unknown target_rollout_aggregation: {aggregation}")


def complete_sequence_heuristic(
    partial_sequence: List[int],
    remaining_jobs: List[int],
    processing_times: np.ndarray,
    heuristic: str,
    rng: random.Random,
) -> List[int]:
    """Complete sequence using specified heuristic.

    Args:
        heuristic: One of 'spt', 'lpt', 'random', or 'mixed' (randomly picks one)
    """
    if heuristic == "mixed":
        heuristic = rng.choice(["spt", "lpt", "random"])

    if heuristic == "spt":
        return complete_sequence_spt(partial_sequence, remaining_jobs, processing_times)
    elif heuristic == "lpt":
        return complete_sequence_lpt(partial_sequence, remaining_jobs, processing_times)
    elif heuristic == "random":
        return complete_sequence_random(partial_sequence, remaining_jobs, rng)
    else:
        # Default to SPT
        return complete_sequence_spt(partial_sequence, remaining_jobs, processing_times)


def select_action_heuristic(
    remaining_jobs: List[int],
    processing_times: np.ndarray,
    heuristic: str,
    rng: random.Random,
) -> int:
    """Select next action using specified heuristic.

    Args:
        heuristic: One of 'spt', 'lpt', 'random', or 'mixed' (randomly picks one)
    """
    if not remaining_jobs:
        raise ValueError("No remaining jobs to select from")

    if heuristic == "mixed":
        heuristic = rng.choice(["spt", "lpt", "random"])

    if heuristic == "spt":
        return int(min(remaining_jobs, key=lambda j: processing_times[j]))
    if heuristic == "lpt":
        return int(max(remaining_jobs, key=lambda j: processing_times[j]))
    if heuristic == "random":
        return int(rng.choice(remaining_jobs))

    # Default fallback
    return int(min(remaining_jobs, key=lambda j: processing_times[j]))


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


def complete_sequences_model_batch(
    partial_sequences: List[List[int]],
    available_masks: np.ndarray,  # (B_cand, N_pad) float 1=available
    model: QSequenceNet,
    obs_template: Dict[str, torch.Tensor],
    processing_times: np.ndarray,
    F_job: int,
    device: torch.device,
) -> List[List[int]]:
    """Batched greedy completion for multiple candidate rollouts (same instance state).

    This preserves the exact greedy policy (argmin Q) used by complete_sequence_model,
    but runs candidates in parallel to reduce Python overhead and improve GPU utilization.
    """
    if available_masks.size == 0:
        return [list(seq) for seq in partial_sequences]

    B_cand, N_pad = available_masks.shape
    jobs_base = obs_template["jobs"].to(device)
    periods_base = obs_template["periods"].to(device)
    ctx_base = obs_template["ctx"].to(device)

    jobs = jobs_base.unsqueeze(0).expand(B_cand, -1, -1).clone()
    periods = periods_base.unsqueeze(0).expand(B_cand, -1, -1)
    ctx = ctx_base.unsqueeze(0).expand(B_cand, -1).clone()

    p_t = torch.from_numpy(processing_times).float().to(device)
    avail = torch.from_numpy(available_masks).float().to(device)

    out_sequences: List[List[int]] = [list(seq) for seq in partial_sequences]

    model.eval()
    with torch.no_grad():
        # All candidates share the same job count/padding; stop when none have available jobs.
        while bool((avail > 0.5).any()):
            if F_job >= 2:
                jobs[:, :, 1] = avail

            # ctx[2] = remaining_work
            ctx[:, 2] = (p_t.unsqueeze(0) * avail).sum(dim=1)

            mask = avail < 0.5  # True = invalid
            q = model(jobs, periods, ctx, mask)
            q = q.masked_fill(mask, float("inf"))
            action = q.argmin(dim=-1)  # (B_cand,)

            # Mark chosen as unavailable
            avail.scatter_(1, action.unsqueeze(1), 0.0)

            # Append chosen action to each candidate's sequence
            act_cpu = action.detach().cpu().tolist()
            for i, a in enumerate(act_cpu):
                out_sequences[i].append(int(a))

    return out_sequences


def _collect_round_batch(
    *,
    env_config,
    data_config: DataConfig,
    variant_config: VariantConfig,
    model_state: Optional[Dict[str, torch.Tensor]],
    teacher_model_state: Optional[Dict[str, torch.Tensor]],
    batch_size: int,
    num_counterfactuals: int,
    exploration_eps: float,
    use_model_completion: bool,
    heuristic_policy: str,
    target_normalization: str,
    include_heuristic_candidates: bool,
    target_rollouts: str,
    target_rollout_aggregation: str,
    target_num_random_rollouts: int,
    target_softmin_tau: float,
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

    teacher_model: Optional[QSequenceNet] = None
    if teacher_model_state is not None:
        teacher_model = build_q_model(variant_config)
        teacher_model.load_state_dict(teacher_model_state)
        teacher_model.to(device)
        teacher_model.eval()

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

    # IMPORTANT: state_id must be globally unique across collection calls.
    # It is used to group candidate actions from the *same state* for listwise loss.
    # If state_id collides across batches/rounds/workers, unrelated states get grouped
    # together and the listwise targets/gradients become meaningless.
    #
    # We derive a stable 64-bit base from the per-batch seed (already unique in
    # collect_round_data via seed + batch_idx*10000) and add a local counter.
    # Mask seed to keep the value safely within int64.
    state_id_base = (int(seed) & 0x3FFFFFFF) << 32

    # Accumulate DP evaluations and aggregate them into per-(state,action) labels.
    pending_sequences: List[List[int]] = []
    pending_instance_indices: List[int] = []
    pending_candidate_uid: List[int] = []

    candidate_meta: Dict[int, Dict[str, Any]] = {}
    candidate_costs: Dict[int, List[float]] = {}
    candidate_expected: Dict[int, int] = {}

    state_meta: Dict[int, Dict[str, Any]] = {}
    # state_meta[state_id] = {"expected": int, "candidates": [uid...], "done": {uid: cost}}

    next_candidate_uid = 0
    DP_FLUSH_THRESHOLD = 8192  # sequences per DP call (tunable)

    def _emit_state_if_ready(state_id: int) -> None:
        s = state_meta.get(state_id)
        if not s:
            return
        done: Dict[int, float] = s.get("done", {})
        if len(done) < int(s["expected"]):
            return

        # Optional normalization across candidates in this state.
        state_min_cost = min(done.values()) if done else 0.0
        do_norm = (target_normalization or "none").strip().lower() == "state_min"

        for uid in s["candidates"]:
            meta = candidate_meta.get(uid)
            if meta is None:
                continue
            cf_cost = float(done[uid])
            q_target = float(cf_cost - state_min_cost) if do_norm else float(cf_cost)
            transitions.append(
                QTransition(
                    jobs=meta["obs"]["jobs"].copy(),
                    periods=meta["obs"]["periods"].copy(),
                    ctx=meta["obs"]["ctx"].copy(),
                    job_avail=meta["job_avail"].copy(),
                    action=int(meta["action"]),
                    q_target=float(q_target),
                    state_id=int(state_id),
                )
            )

            # cleanup per-candidate
            candidate_meta.pop(uid, None)
            candidate_costs.pop(uid, None)
            candidate_expected.pop(uid, None)

        state_meta.pop(state_id, None)

    def flush_pending():
        nonlocal pending_sequences, pending_instance_indices, pending_candidate_uid
        if not pending_sequences:
            return
        costs = batch_evaluate_sequences(
            sequences=pending_sequences,
            processing_times=p_all,
            ct=ct_all,
            e_single=e_all,
            T_limit=T_limit_all,
            instance_indices=pending_instance_indices,
            device=device,
        )
        costs_cpu = costs.detach().cpu().numpy().tolist()

        for uid, cf_cost in zip(pending_candidate_uid, costs_cpu):
            if not np.isfinite(cf_cost):
                continue
            uid_i = int(uid)
            candidate_costs.setdefault(uid_i, []).append(float(cf_cost))
            exp = int(candidate_expected.get(uid_i, 0))
            if exp > 0 and len(candidate_costs[uid_i]) >= exp:
                # Candidate complete => aggregate and attach to its state.
                meta = candidate_meta.get(uid_i)
                if meta is None:
                    continue
                state_id = int(meta["state_id"])
                agg = _aggregate_rollout_costs(
                    candidate_costs[uid_i],
                    aggregation=target_rollout_aggregation,
                    softmin_tau=target_softmin_tau,
                )
                s = state_meta.get(state_id)
                if s is not None:
                    s.setdefault("done", {})[uid_i] = float(agg)
                    _emit_state_if_ready(state_id)

        pending_sequences = []
        pending_instance_indices = []
        pending_candidate_uid = []

    # Process each instance sequentially (original working logic)
    state_id_counter = 0
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

            state_id = int(state_id_base + state_id_counter)
            state_id_counter += 1

            if F_job >= 2:
                obs_torch["jobs"][:, 1] = torch.from_numpy(job_available).float()

            remaining_work = (p_np * job_available).sum()
            obs_torch["ctx"][2] = remaining_work

            obs_single = {
                "jobs": obs_torch["jobs"].cpu().numpy(),
                "periods": obs_torch["periods"].cpu().numpy(),
                "ctx": obs_torch["ctx"].cpu().numpy(),
            }

            if rng.random() < exploration_eps:
                action = rng.choice(remaining_jobs)
            elif model is not None:
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
            else:
                action = select_action_heuristic(
                    remaining_jobs=remaining_jobs,
                    processing_times=p_np,
                    heuristic=heuristic_policy,
                    rng=rng,
                )

            # Candidate action set for counterfactual evaluation.
            # Default: chosen action + random negatives.
            # Optional: always include SPT/LPT candidates for more informative comparisons.
            if len(remaining_jobs) <= num_counterfactuals:
                candidates = remaining_jobs.copy()
            else:
                cand_set = {int(action)}
                if include_heuristic_candidates and remaining_jobs:
                    spt_j = min(remaining_jobs, key=lambda j: p_np[j])
                    lpt_j = max(remaining_jobs, key=lambda j: p_np[j])
                    cand_set.add(int(spt_j))
                    cand_set.add(int(lpt_j))

                others = [j for j in remaining_jobs if j not in cand_set]
                need = max(0, int(num_counterfactuals) - len(cand_set))
                if need > 0 and others:
                    cand_set.update(rng.sample(others, min(need, len(others))))
                candidates = list(cand_set)

            # Build full sequences for each candidate.
            rollout_tokens = _parse_target_rollouts(target_rollouts)
            use_auto = len(rollout_tokens) == 1 and rollout_tokens[0] == "auto"
            rollouts_per_candidate: Dict[int, List[List[int]]] = {
                int(j): [] for j in candidates
            }

            model_for_completion = teacher_model if teacher_model is not None else model

            # Legacy/auto behavior: one completion decided by use_model_completion + heuristic
            if use_auto:
                if use_model_completion and model_for_completion is not None:
                    partials = []
                    avail_batch = np.zeros((len(candidates), N_pad), dtype=np.float32)
                    for j, candidate_job in enumerate(candidates):
                        cf_partial = partial_sequence + [candidate_job]
                        partials.append(cf_partial)
                        cf_avail = job_available.copy()
                        cf_avail[candidate_job] = 0.0
                        avail_batch[j] = cf_avail
                    full_seqs = complete_sequences_model_batch(
                        partial_sequences=partials,
                        available_masks=avail_batch,
                        model=model_for_completion,
                        obs_template=obs_torch,
                        processing_times=p_np,
                        F_job=F_job,
                        device=device,
                    )
                    for candidate_job, seq in zip(candidates, full_seqs):
                        rollouts_per_candidate[int(candidate_job)].append(list(seq))
                else:
                    for candidate_job in candidates:
                        cf_partial = partial_sequence + [candidate_job]
                        cf_remaining = [j for j in remaining_jobs if j != candidate_job]
                        rollouts_per_candidate[int(candidate_job)].append(
                            complete_sequence_heuristic(
                                cf_partial, cf_remaining, p_np, heuristic_policy, rng
                            )
                        )
            else:
                # Multi-rollout targets: evaluate several completion styles per (s,j)
                if "model" in rollout_tokens and model_for_completion is not None:
                    partials = []
                    avail_batch = np.zeros((len(candidates), N_pad), dtype=np.float32)
                    for j, candidate_job in enumerate(candidates):
                        cf_partial = partial_sequence + [candidate_job]
                        partials.append(cf_partial)
                        cf_avail = job_available.copy()
                        cf_avail[candidate_job] = 0.0
                        avail_batch[j] = cf_avail
                    model_seqs = complete_sequences_model_batch(
                        partial_sequences=partials,
                        available_masks=avail_batch,
                        model=model_for_completion,
                        obs_template=obs_torch,
                        processing_times=p_np,
                        F_job=F_job,
                        device=device,
                    )
                    for candidate_job, seq in zip(candidates, model_seqs):
                        rollouts_per_candidate[int(candidate_job)].append(list(seq))

                for candidate_job in candidates:
                    cf_partial = partial_sequence + [candidate_job]
                    cf_remaining = [j for j in remaining_jobs if j != candidate_job]

                    if "spt" in rollout_tokens:
                        rollouts_per_candidate[int(candidate_job)].append(
                            complete_sequence_spt(cf_partial, cf_remaining, p_np)
                        )
                    if "lpt" in rollout_tokens:
                        rollouts_per_candidate[int(candidate_job)].append(
                            complete_sequence_lpt(cf_partial, cf_remaining, p_np)
                        )
                    if "mixed" in rollout_tokens:
                        rollouts_per_candidate[int(candidate_job)].append(
                            complete_sequence_heuristic(
                                cf_partial, cf_remaining, p_np, "mixed", rng
                            )
                        )
                    if "random" in rollout_tokens:
                        for _r in range(int(target_num_random_rollouts)):
                            rr = random.Random(rng.randint(0, 2**31 - 1))
                            rollouts_per_candidate[int(candidate_job)].append(
                                complete_sequence_random(cf_partial, cf_remaining, rr)
                            )

                # Safety: ensure every candidate has at least one rollout
                for candidate_job in candidates:
                    if not rollouts_per_candidate[int(candidate_job)]:
                        cf_partial = partial_sequence + [candidate_job]
                        cf_remaining = [j for j in remaining_jobs if j != candidate_job]
                        rollouts_per_candidate[int(candidate_job)].append(
                            complete_sequence_heuristic(
                                cf_partial, cf_remaining, p_np, heuristic_policy, rng
                            )
                        )

            # Register this state's candidate set so we can normalize per-state and emit transitions.
            state_meta[int(state_id)] = {
                "expected": int(len(candidates)),
                "candidates": [],
                "done": {},
            }

            # Queue DP evals for batching.
            for candidate_job in candidates:
                # Allocate candidate uid
                uid = next_candidate_uid
                next_candidate_uid += 1
                state_meta[int(state_id)]["candidates"].append(int(uid))
                candidate_meta[int(uid)] = {
                    "obs": obs_single,
                    "job_avail": job_available.copy(),
                    "action": int(candidate_job),
                    "state_id": int(state_id),
                }
                r_seqs = rollouts_per_candidate[int(candidate_job)]
                candidate_expected[int(uid)] = int(len(r_seqs))
                for full_seq in r_seqs:
                    pending_sequences.append(full_seq)
                    pending_instance_indices.append(inst_idx)
                    pending_candidate_uid.append(int(uid))

            if len(pending_sequences) >= DP_FLUSH_THRESHOLD:
                flush_pending()

            partial_sequence.append(action)
            remaining_jobs.remove(action)
            job_available[action] = 0.0

        # Flush any leftover DP work for this instance.
        flush_pending()

    # Final flush at end of batch.
    flush_pending()

    # Emit any completed states that may still be buffered (should be none).
    for sid in list(state_meta.keys()):
        _emit_state_if_ready(int(sid))

    return transitions


def collect_round_data(
    env_config,  # EnvConfig, not env instance
    model: Optional[QSequenceNet],
    teacher_model: Optional[QSequenceNet],
    variant_config: VariantConfig,
    data_config: DataConfig,
    num_episodes: int,
    num_counterfactuals: int,
    exploration_eps: float,
    use_model_completion: bool,
    heuristic_policy: str,
    target_normalization: str,
    include_heuristic_candidates: bool,
    target_rollouts: str,
    target_rollout_aggregation: str,
    target_num_random_rollouts: int,
    target_softmin_tau: float,
    device: torch.device,
    seed: int,
    collection_batch_size: int = 64,  # Fixed batch size for collection
    num_collection_workers: int = 0,
    allow_gpu_collection_multiprocessing: bool = False,
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
    teacher_state = teacher_model.state_dict() if teacher_model is not None else None

    # Default behavior: keep GPU collection single-process for stability.
    # Opt-in override is available for experimentation.
    if device.type != "cpu" and not allow_gpu_collection_multiprocessing:
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
                    teacher_model_state=teacher_state,
                    batch_size=current_batch_size,
                    num_counterfactuals=num_counterfactuals,
                    exploration_eps=exploration_eps,
                    use_model_completion=use_model_completion,
                    heuristic_policy=heuristic_policy,
                    target_normalization=target_normalization,
                    include_heuristic_candidates=include_heuristic_candidates,
                    target_rollouts=target_rollouts,
                    target_rollout_aggregation=target_rollout_aggregation,
                    target_num_random_rollouts=target_num_random_rollouts,
                    target_softmin_tau=target_softmin_tau,
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
                        teacher_model_state=teacher_state,
                        batch_size=current_batch_size,
                        num_counterfactuals=num_counterfactuals,
                        exploration_eps=exploration_eps,
                        use_model_completion=use_model_completion,
                        heuristic_policy=heuristic_policy,
                        target_normalization=target_normalization,
                        include_heuristic_candidates=include_heuristic_candidates,
                        target_rollouts=target_rollouts,
                        target_rollout_aggregation=target_rollout_aggregation,
                        target_num_random_rollouts=target_num_random_rollouts,
                        target_softmin_tau=target_softmin_tau,
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

    # Numerical safety: if we ever generate non-finite values, skip that batch
    # rather than poisoning the optimizer state with NaNs/Infs.
    warned_nonfinite = False

    total_loss = 0.0
    total_q_mse = 0.0
    total_q_mae = 0.0
    total_listwise = 0.0
    total_policy = 0.0
    total_top_gamma = 0.0
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
        state_ids = batch["state_ids"].to(device)

        # Forward pass
        q_values = model(
            jobs=jobs,
            periods_local=periods,
            ctx=ctx,
            job_mask=job_mask,
            period_mask=period_mask,
        )

        if not torch.isfinite(q_values).all():
            if not warned_nonfinite:
                print(
                    "WARNING: non-finite q_values detected; skipping a batch (consider lowering lr / threads).",
                    flush=True,
                )
                warned_nonfinite = True
            continue

        # Get Q-values for taken actions
        q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # --- Regression loss (optional / auxiliary) ---
        reg_loss = torch.tensor(0.0, device=device)
        if float(getattr(config, "regression_weight", 1.0)) > 0:
            if config.loss_type == "huber":
                reg_loss = F.huber_loss(q_pred, q_targets, delta=config.huber_delta)
            else:
                reg_loss = F.mse_loss(q_pred, q_targets)

        # --- Policy / ranking loss (primary when train_objective=policy) ---
        policy_loss = torch.tensor(0.0, device=device)
        top_gamma_loss = torch.tensor(0.0, device=device)

        objective = (
            (getattr(config, "train_objective", "regression") or "regression")
            .strip()
            .lower()
        )
        policy_w = float(getattr(config, "policy_weight", 0.0))
        topg_w = float(getattr(config, "top_gamma_weight", 0.0))
        topg = int(getattr(config, "top_gamma", 0) or 0)
        topg_margin = float(getattr(config, "top_gamma_margin", 0.0))
        pol_temp = float(getattr(config, "policy_temperature", 1.0))
        pol_target = (
            (getattr(config, "policy_target", "soft") or "soft").strip().lower()
        )

        if objective == "policy" or policy_w > 0 or topg_w > 0:
            unique_states = torch.unique(state_ids)
            pl_sum = 0.0
            tg_sum = 0.0
            count = 0
            for sid in unique_states:
                idx = (state_ids == sid).nonzero(as_tuple=False).squeeze(1)
                if idx.numel() < 2:
                    continue

                # Candidate actions and targets for this state
                cand_actions = actions.index_select(0, idx)  # (M,)
                cand_targets = q_targets.index_select(0, idx)  # (M,)

                # Use one representative forward pass row (observations identical per state)
                rep = int(idx[0].item())
                q_state = q_values[rep]  # (N_pad,)
                q_cand = q_state.index_select(0, cand_actions)  # (M,)

                # Robustify against inf/NaN from exploding weights.
                q_cand = torch.nan_to_num(q_cand, nan=0.0, posinf=1e6, neginf=-1e6)
                cand_targets = torch.nan_to_num(
                    cand_targets, nan=0.0, posinf=1e12, neginf=-1e12
                )

                logits = (-q_cand / max(pol_temp, 1e-8)).clamp(-50.0, 50.0)

                # Target distribution from aggregated costs (lower cost => higher prob)
                if pol_target == "hard":
                    best_pos = int(torch.argmin(cand_targets).item())
                    pl = F.cross_entropy(
                        logits.unsqueeze(0),
                        torch.tensor([best_pos], device=device, dtype=torch.long),
                    )
                else:
                    target_logits = (-cand_targets / max(pol_temp, 1e-8)).clamp(
                        -50.0, 50.0
                    )
                    p_t = F.softmax(target_logits, dim=0)
                    log_p = F.log_softmax(logits, dim=0)
                    pl = -(p_t * log_p).sum()

                pl_sum += pl

                if topg_w > 0 and topg > 0:
                    k = min(int(topg), int(cand_actions.numel()))
                    # Boundary is k-th smallest predicted cost among candidates
                    boundary = torch.kthvalue(q_cand, k).values
                    best_pos = int(torch.argmin(cand_targets).item())
                    q_best = q_cand[best_pos]
                    tg = F.relu(q_best - boundary + float(topg_margin))
                    tg_sum += tg

                count += 1

            if count > 0:
                policy_loss = pl_sum / count
                if topg_w > 0 and topg > 0:
                    top_gamma_loss = tg_sum / count

        # Optional legacy listwise ranking loss (kept for backward compatibility)
        listwise_loss = torch.tensor(0.0, device=device)
        if config.listwise_weight > 0:
            unique_states = torch.unique(state_ids)
            lw_sum = 0.0
            lw_count = 0
            for sid in unique_states:
                idx = (state_ids == sid).nonzero(as_tuple=False).squeeze(1)
                if idx.numel() < 2:
                    continue
                q_pred_g = q_pred.index_select(0, idx)
                q_t_g = q_targets.index_select(0, idx)

                # Target distribution from DP costs (lower cost => higher prob)
                log_p_pred = F.log_softmax(
                    -q_pred_g / config.listwise_temperature, dim=0
                )
                p_t = F.softmax(-q_t_g / config.listwise_temperature, dim=0)
                lw = -(p_t * log_p_pred).sum()
                lw_sum += lw
                lw_count += 1

            if lw_count > 0:
                listwise_loss = lw_sum / lw_count

        # Combine losses
        loss = (
            float(getattr(config, "regression_weight", 1.0)) * reg_loss
            + float(getattr(config, "policy_weight", 0.0)) * policy_loss
            + float(getattr(config, "top_gamma_weight", 0.0)) * top_gamma_loss
            + float(getattr(config, "listwise_weight", 0.0)) * listwise_loss
        )

        if not torch.isfinite(loss):
            if not warned_nonfinite:
                print(
                    "WARNING: non-finite loss detected; skipping a batch (try lower --learning_rate / higher --policy_temperature).",
                    flush=True,
                )
                warned_nonfinite = True
            continue

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
            total_listwise += float(listwise_loss.item())
            total_policy += float(policy_loss.item())
            total_top_gamma += float(top_gamma_loss.item())
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "q_mse": total_q_mse / max(n_batches, 1),
        "q_mae": total_q_mae / max(n_batches, 1),
        "listwise": total_listwise / max(n_batches, 1),
        "policy": total_policy / max(n_batches, 1),
        "top_gamma": total_top_gamma / max(n_batches, 1),
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


def evaluate_sgbs(
    model: QSequenceNet,
    variant_config,
    data_config: DataConfig,
    num_instances: int,
    seed: int,
    device: torch.device,
    beta: int = 4,
    gamma: int = 4,
) -> Dict[str, float]:
    """Evaluate model using SGBS beam search - the real metric for Q-sequence.

    This evaluates whether the Q-value landscape supports good beam search,
    which is the intended use case (not greedy decoding).
    """
    from PaST.cli.eval.run_eval_q_sequence import (
        sgbs_q_sequence,
        greedy_decode_q_sequence,
    )
    from PaST.baselines_sequence_dp import spt_lpt_with_dp

    model.eval()

    # Generate eval instances
    batch_data = generate_episode_batch(
        batch_size=num_instances,
        config=data_config,
        seed=seed,
    )

    # Run SGBS
    sgbs_results = sgbs_q_sequence(
        model=model,
        variant_config=variant_config,
        batch_data=batch_data,
        device=device,
        beta=beta,
        gamma=gamma,
    )

    # Run greedy for comparison
    greedy_results = greedy_decode_q_sequence(
        model=model,
        variant_config=variant_config,
        batch_data=batch_data,
        device=device,
    )

    # Run SPT baseline
    spt_results = spt_lpt_with_dp(
        env_config=variant_config.env,
        device=device,
        batch_data=batch_data,
        which="spt",
    )

    # Compute metrics
    sgbs_costs = [r.total_energy for r in sgbs_results]
    greedy_costs = [r.total_energy for r in greedy_results]
    spt_costs = [r.total_energy for r in spt_results]

    avg_sgbs = np.mean(sgbs_costs)
    avg_greedy = np.mean(greedy_costs)
    avg_spt = np.mean(spt_costs)

    return {
        "sgbs_cost": avg_sgbs,
        "greedy_cost": avg_greedy,
        "spt_cost": avg_spt,
        "sgbs_vs_spt": (avg_spt - avg_sgbs) / avg_spt * 100 if avg_spt > 0 else 0,
        "sgbs_vs_greedy": (
            (avg_greedy - avg_sgbs) / avg_greedy * 100 if avg_greedy > 0 else 0
        ),
        "greedy_vs_spt": (avg_spt - avg_greedy) / avg_spt * 100 if avg_spt > 0 else 0,
        "n_instances": num_instances,
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
    parser.add_argument("--buffer_size", type=int, default=100_000)
    parser.add_argument("--collection_batch_size", type=int, default=64)
    parser.add_argument("--num_collection_workers", type=int, default=0)
    parser.add_argument(
        "--allow_gpu_collection_multiprocessing",
        action="store_true",
        help="DANGEROUS/EXPERIMENTAL: allow multiple collection worker processes even when device=cuda",
    )
    parser.add_argument("--num_dataloader_workers", type=int, default=0)
    parser.add_argument("--num_cpu_threads", type=int, default=0)

    parser.add_argument("--num_counterfactuals", type=int, default=8)
    # Backward-compatible: --exploration_eps sets both start/end
    parser.add_argument("--exploration_eps", type=float, default=0.2)
    parser.add_argument("--exploration_eps_start", type=float, default=None)
    parser.add_argument("--exploration_eps_end", type=float, default=None)
    parser.add_argument("--exploration_eps_decay_rounds", type=int, default=0)
    parser.add_argument("--warmup_rounds", type=int, default=5)
    parser.add_argument(
        "--completion_policy",
        type=str,
        default="model",
        choices=["spt", "model", "mix"],
    )
    parser.add_argument("--completion_prob_start", type=float, default=1.0)
    parser.add_argument("--completion_prob_end", type=float, default=1.0)
    parser.add_argument("--completion_prob_decay_rounds", type=int, default=0)
    parser.add_argument(
        "--heuristic_policy",
        type=str,
        default="mixed",
        choices=["spt", "lpt", "random", "mixed"],
        help="Heuristic for completion when not using model. 'mixed' randomly picks spt/lpt/random.",
    )

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_epochs_per_round", type=int, default=10)
    parser.add_argument(
        "--loss_type", type=str, default="huber", choices=["huber", "mse"]
    )
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--listwise_weight", type=float, default=0.0)
    parser.add_argument("--listwise_temperature", type=float, default=1.0)

    parser.add_argument(
        "--target_normalization",
        type=str,
        default="none",
        choices=["none", "state_min"],
        help="Optional shaping of q_target values. 'state_min' subtracts the minimum candidate DP cost per state.",
    )
    parser.add_argument(
        "--include_heuristic_candidates",
        action="store_true",
        help="Ensure SPT/LPT candidates are included in counterfactual candidate sets (when sampling).",
    )

    # ------------------------------------------------------------------
    # New features (opt-in)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--target_rollouts",
        type=str,
        default="auto",
        help=(
            "Target rollout set for labels. 'auto' keeps legacy single-completion behavior. "
            "Otherwise comma-separated from {model,spt,lpt,random,mixed}."
        ),
    )
    parser.add_argument(
        "--target_rollout_aggregation",
        type=str,
        default="single",
        choices=["single", "min", "softmin"],
        help="How to aggregate multiple rollout DP costs into a single label per (s,j).",
    )
    parser.add_argument(
        "--target_num_random_rollouts",
        type=int,
        default=2,
        help="When target_rollouts includes 'random', number of random completions per (s,j).",
    )
    parser.add_argument(
        "--target_softmin_tau",
        type=float,
        default=1.0,
        help="Softmin temperature tau for target_rollout_aggregation=softmin.",
    )

    parser.add_argument(
        "--train_objective",
        type=str,
        default="regression",
        choices=["regression", "policy"],
        help="Training objective. 'policy' makes ranking/classification primary.",
    )
    parser.add_argument(
        "--regression_weight",
        type=float,
        default=1.0,
        help="Weight for regression loss on DP costs (auxiliary if train_objective=policy).",
    )
    parser.add_argument(
        "--policy_weight",
        type=float,
        default=0.0,
        help="Weight for policy/ranking loss (set >0 or use train_objective=policy).",
    )
    parser.add_argument(
        "--policy_temperature",
        type=float,
        default=1.0,
        help="Temperature for policy target distribution and predicted logits.",
    )
    parser.add_argument(
        "--policy_target",
        type=str,
        default="soft",
        choices=["soft", "hard"],
        help="Policy targets: 'soft' distribution from costs, or 'hard' argmin label.",
    )

    parser.add_argument(
        "--top_gamma",
        type=int,
        default=0,
        help="Top-γ aligned loss: encourage true best action to be within top-γ (0 disables).",
    )
    parser.add_argument(
        "--top_gamma_weight",
        type=float,
        default=0.0,
        help="Weight for top-γ aligned hinge loss.",
    )
    parser.add_argument(
        "--top_gamma_margin",
        type=float,
        default=0.0,
        help="Margin for top-γ hinge loss (default 0).",
    )

    parser.add_argument(
        "--teacher_update_every_rounds",
        type=int,
        default=1,
        help="Update frozen teacher model every N rounds (1 = every round, 0 = never).",
    )
    parser.add_argument(
        "--teacher_update_on_save",
        action="store_true",
        default=True,
        help="Also update teacher when a checkpoint is saved.",
    )
    parser.add_argument(
        "--no_teacher_update_on_save",
        dest="teacher_update_on_save",
        action="store_false",
        help="Disable teacher update on checkpoint save.",
    )

    parser.add_argument("--eval_every_rounds", type=int, default=5)
    parser.add_argument("--num_eval_instances", type=int, default=256)
    parser.add_argument("--save_every_rounds", type=int, default=10)

    # SGBS evaluation
    parser.add_argument(
        "--eval_sgbs",
        action="store_true",
        default=True,
        help="Enable SGBS beam search evaluation (default: True)",
    )
    parser.add_argument(
        "--no_eval_sgbs",
        dest="eval_sgbs",
        action="store_false",
        help="Disable SGBS evaluation for faster training",
    )
    parser.add_argument(
        "--eval_sgbs_beta", type=int, default=4, help="Beam width for SGBS evaluation"
    )
    parser.add_argument(
        "--eval_sgbs_gamma",
        type=int,
        default=4,
        help="Expansion factor for SGBS evaluation",
    )
    parser.add_argument(
        "--eval_sgbs_num_instances",
        type=int,
        default=64,
        help="Number of instances for SGBS eval (fewer than greedy)",
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("PAST_OUTPUT_DIR", "runs_q"),
    )
    parser.add_argument("--smoke_test", action="store_true")

    # Curriculum (deadline slack)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--curriculum_fraction", type=float, default=0.3)
    parser.add_argument("--curriculum_slack_min", type=float, default=None)
    parser.add_argument("--curriculum_slack_max", type=float, default=None)

    # Resume from checkpoint
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint file (.pt) or run directory to resume from. "
        "If a directory is given, uses the latest checkpoint_*.pt file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Avoid unsafe fork() warnings/deadlocks on Linux (e.g., Kaggle) when using
    # DataLoader workers or process pools in a multi-threaded runtime.
    if sys.platform.startswith("linux"):
        try:
            mp.set_start_method("spawn", force=True)
        except Exception:
            pass

    # Build config
    config = QRunConfig(
        variant_id=args.variant_id,
        seed=args.seed,
        run_name=args.run_name,
        batch_size=args.batch_size,
        episodes_per_round=args.episodes_per_round,
        num_rounds=args.num_rounds,
        buffer_size=args.buffer_size,
        collection_batch_size=args.collection_batch_size,
        num_collection_workers=args.num_collection_workers,
        allow_gpu_collection_multiprocessing=args.allow_gpu_collection_multiprocessing,
        num_dataloader_workers=args.num_dataloader_workers,
        num_cpu_threads=args.num_cpu_threads,
        num_counterfactuals=args.num_counterfactuals,
        exploration_eps_start=(
            args.exploration_eps_start
            if args.exploration_eps_start is not None
            else args.exploration_eps
        ),
        exploration_eps_end=(
            args.exploration_eps_end
            if args.exploration_eps_end is not None
            else (
                args.exploration_eps_start
                if args.exploration_eps_start is not None
                else args.exploration_eps
            )
        ),
        exploration_eps_decay_rounds=args.exploration_eps_decay_rounds,
        warmup_rounds=args.warmup_rounds,
        completion_policy=args.completion_policy,
        completion_prob_start=args.completion_prob_start,
        completion_prob_end=args.completion_prob_end,
        completion_prob_decay_rounds=args.completion_prob_decay_rounds,
        heuristic_policy=args.heuristic_policy,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        num_epochs_per_round=args.num_epochs_per_round,
        loss_type=args.loss_type,
        huber_delta=args.huber_delta,
        listwise_weight=args.listwise_weight,
        listwise_temperature=args.listwise_temperature,
        target_normalization=args.target_normalization,
        include_heuristic_candidates=args.include_heuristic_candidates,
        target_rollouts=args.target_rollouts,
        target_rollout_aggregation=args.target_rollout_aggregation,
        target_num_random_rollouts=args.target_num_random_rollouts,
        target_softmin_tau=args.target_softmin_tau,
        train_objective=args.train_objective,
        regression_weight=args.regression_weight,
        policy_weight=args.policy_weight,
        policy_temperature=args.policy_temperature,
        policy_target=args.policy_target,
        top_gamma=args.top_gamma,
        top_gamma_weight=args.top_gamma_weight,
        top_gamma_margin=args.top_gamma_margin,
        teacher_update_every_rounds=args.teacher_update_every_rounds,
        teacher_update_on_save=args.teacher_update_on_save,
        eval_every_rounds=args.eval_every_rounds,
        num_eval_instances=args.num_eval_instances,
        save_every_rounds=args.save_every_rounds,
        eval_sgbs=args.eval_sgbs,
        eval_sgbs_beta=args.eval_sgbs_beta,
        eval_sgbs_gamma=args.eval_sgbs_gamma,
        eval_sgbs_num_instances=args.eval_sgbs_num_instances,
        device=args.device,
        output_dir=args.output_dir,
        smoke_test=args.smoke_test,
        curriculum=args.curriculum,
        curriculum_fraction=args.curriculum_fraction,
        curriculum_slack_min=args.curriculum_slack_min,
        curriculum_slack_max=args.curriculum_slack_max,
        resume_from=args.resume_from,
    )

    # Smoke test overrides
    if config.smoke_test:
        config.episodes_per_round = 64
        config.num_rounds = 5
        config.num_epochs_per_round = 2
        config.eval_every_rounds = 2
        config.num_eval_instances = 32
        config.warmup_rounds = 2
        config.exploration_eps_decay_rounds = 0
        config.completion_prob_decay_rounds = 0
        config.collection_batch_size = 16
        config.num_collection_workers = 1
        config.allow_gpu_collection_multiprocessing = False
        config.num_dataloader_workers = 0

    # Sensible defaults for policy-based training mode
    if (config.train_objective or "regression").strip().lower() == "policy":
        if config.policy_weight <= 0:
            config.policy_weight = 1.0
        # Keep regression as a small stabilizer unless user explicitly disabled it.
        if config.regression_weight >= 1.0:
            config.regression_weight = 0.1

    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cpu_count = os.cpu_count() or 1
    if config.num_cpu_threads <= 0:
        config.num_cpu_threads = cpu_count
    else:
        # Avoid severe oversubscription on small machines (e.g., Kaggle 4 cores).
        if int(config.num_cpu_threads) > int(cpu_count):
            print(
                f"Capping --num_cpu_threads from {config.num_cpu_threads} to os.cpu_count={cpu_count} for stability/perf.",
                flush=True,
            )
            config.num_cpu_threads = int(cpu_count)

    # Configure torch threads even for CUDA runs (preprocessing / DP setup / dataloader)
    try:
        torch.set_num_threads(int(config.num_cpu_threads))
        torch.set_num_interop_threads(min(4, int(config.num_cpu_threads)))
    except Exception:
        pass

    # Cap dataloader workers to CPU count to avoid overload.
    if config.num_dataloader_workers > int(cpu_count):
        print(
            f"Capping --num_dataloader_workers from {config.num_dataloader_workers} to os.cpu_count={cpu_count}.",
            flush=True,
        )
        config.num_dataloader_workers = int(cpu_count)

    if device.type == "cpu":
        cpu_count = os.cpu_count() or 1
        if config.num_dataloader_workers <= 0:
            config.num_dataloader_workers = min(cpu_count, 8)
        if config.num_collection_workers <= 0:
            config.num_collection_workers = min(cpu_count, 8)

    try:
        print(
            f"CPU setup: os.cpu_count={os.cpu_count()} | torch_threads={torch.get_num_threads()} | "
            f"collection_workers={config.num_collection_workers} | dataloader_workers={config.num_dataloader_workers}"
        )
    except Exception:
        pass

    # === Resume handling ===
    start_round = 0
    checkpoint_data = None
    resume_run_dir = None

    if config.resume_from:
        resume_path = Path(config.resume_from)
        if resume_path.is_dir():
            # Find latest checkpoint in directory (could be run dir or checkpoints dir)
            # NOTE: Don't rely on lexicographic sorting (checkpoint_10.pt < checkpoint_5.pt).
            def _ckpt_num(path: Path) -> int:
                m = re.search(r"checkpoint_(\\d+)\\.pt$", path.name)
                return int(m.group(1)) if m else -1

            # Consider BOTH locations; older runs sometimes wrote to the run root,
            # while newer runs write to run_root/checkpoints/.
            checkpoints = list(resume_path.glob("checkpoint_*.pt")) + list(
                resume_path.glob("checkpoints/checkpoint_*.pt")
            )
            if not checkpoints:
                raise FileNotFoundError(
                    f"No checkpoint_*.pt files found in {resume_path} or {resume_path}/checkpoints"
                )
            # Pick newest by numeric suffix.
            checkpoints_sorted = sorted(checkpoints, key=_ckpt_num)
            resume_path = checkpoints_sorted[-1]
            try:
                tail = checkpoints_sorted[-5:]
                tail_str = ", ".join(p.name for p in tail)
                print(
                    f"Found {len(checkpoints)} checkpoints under {resume_path.parent.parent if resume_path.parent.name == 'checkpoints' else resume_path.parent}. "
                    f"Picking: {resume_path.name}. Recent: [{tail_str}]"
                )
            except Exception:
                pass
            resume_run_dir = Path(config.resume_from)
            # If it was the checkpoints dir, go up one level
            if resume_run_dir.name == "checkpoints":
                resume_run_dir = resume_run_dir.parent
        else:
            # File path given - go up to run directory
            resume_run_dir = resume_path.parent
            if resume_run_dir.name == "checkpoints":
                resume_run_dir = resume_run_dir.parent

        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint_data = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
        start_round = checkpoint_data["round"]
        print(
            f"  Resuming from round {start_round + 1} (completed {start_round} rounds)"
        )

    # Setup output directory
    if resume_run_dir is not None:
        # Continue in the same run directory
        run_dir = resume_run_dir
        print(f"  Continuing in existing run directory: {run_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = config.run_name or f"{config.variant_id}_s{config.seed}"
        run_dir = Path(config.output_dir) / f"{run_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

    # Save config (append resume info if resuming)
    config_path = run_dir / "config.json"
    if config.resume_from:
        # Try to append resume info to existing config, or create new one
        if config_path.exists():
            with open(config_path, "r") as f:
                saved_config = json.load(f)
            # Keep the original requested resume path for provenance (can be a directory).
            saved_config["resume_from"] = config.resume_from
            saved_config["resumed_from"] = str(resume_path)
            saved_config["resumed_at_round"] = start_round
            with open(config_path, "w") as f:
                json.dump(saved_config, f, indent=2)
        else:
            # No existing config - create one with current settings + resume info
            new_config = asdict(config)
            new_config["resumed_from"] = str(resume_path)
            new_config["resumed_at_round"] = start_round
            with open(config_path, "w") as f:
                json.dump(new_config, f, indent=2)
    else:
        with open(config_path, "w") as f:
            json.dump(asdict(config), f, indent=2)

    # Load variant config
    try:
        variant_id = VariantID(config.variant_id)
        variant_config = get_variant_config(variant_id, seed=config.seed)
    except ValueError:
        print(f"Warning: Unknown variant {config.variant_id}, using q_sequence")
        variant_config = get_variant_config(VariantID.Q_SEQUENCE, seed=config.seed)

    # Use the variant's DataConfig so generator distribution matches PPO and the suite.
    data_config = copy.deepcopy(variant_config.data)
    env_config = variant_config.env

    # Safety check: ctx13 semantics require price-family features to be enabled
    if env_config.F_ctx >= 13 and not env_config.use_price_families:
        print(
            "Warning: F_ctx >= 13 but use_price_families=False. "
            "Price-family ctx features (quantiles/deltas) will be zeros. "
            "Use q_sequence_ctx13 or enable env.use_price_families."
        )

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

    # Load checkpoint state if resuming
    if checkpoint_data is not None:
        model.load_state_dict(checkpoint_data["model_state"])
        optimizer.load_state_dict(checkpoint_data["optimizer_state"])
        print(f"  Loaded model and optimizer state from checkpoint")

    # Frozen teacher model for completion/labeling (updated slowly)
    teacher_model = build_q_model(variant_config).to(device)
    teacher_model.load_state_dict(model.state_dict())
    teacher_model.eval()

    # Replay buffer (note: buffer is NOT restored - we start fresh data collection)
    buffer = QTransitionBuffer(config.buffer_size)

    # Logging (append mode if resuming)
    log_file = open(run_dir / "log.jsonl", "a" if config.resume_from else "w")

    print(f"\n{'='*60}")
    print(f"Q-Sequence Training: {config.variant_id}")
    print(f"Output: {run_dir}")
    print(f"Warmup rounds (heuristic completion): {config.warmup_rounds}")
    print(f"Heuristic policy (non-model): {config.heuristic_policy}")
    print(
        f"Completion policy: {config.completion_policy} | "
        f"model_prob_start={config.completion_prob_start} | model_prob_end={config.completion_prob_end}"
    )
    print(
        f"Targets: rollouts={config.target_rollouts} | agg={config.target_rollout_aggregation} | "
        f"randK={config.target_num_random_rollouts}"
    )
    print(
        f"Objective: {config.train_objective} | reg_w={config.regression_weight} | "
        f"pol_w={config.policy_weight} | topγ={config.top_gamma} (w={config.top_gamma_weight}) | "
        f"teacher_every={config.teacher_update_every_rounds} | teacher_on_save={config.teacher_update_on_save}"
    )
    print(f"{'='*60}\n")

    best_cost: Optional[float] = None

    for round_idx in range(start_round, config.num_rounds):
        round_start = time.time()

        # Determine exploration epsilon for this round
        if (
            config.exploration_eps_decay_rounds
            and config.exploration_eps_decay_rounds > 0
        ):
            frac = min(1.0, round_idx / max(1, config.exploration_eps_decay_rounds))
            eps = config.exploration_eps_start + frac * (
                config.exploration_eps_end - config.exploration_eps_start
            )
        else:
            eps = config.exploration_eps_start

        # Determine completion policy for this round
        if round_idx < config.warmup_rounds:
            use_model_completion = False
            completion_policy = config.heuristic_policy  # Show actual heuristic used
        else:
            if config.completion_policy == "spt":
                use_model_completion = False
                completion_policy = config.heuristic_policy
            elif config.completion_policy == "model":
                use_model_completion = True
                completion_policy = "model"
            else:
                # mix
                if (
                    config.completion_prob_decay_rounds
                    and config.completion_prob_decay_rounds > 0
                ):
                    frac = min(
                        1.0,
                        (round_idx - config.warmup_rounds)
                        / max(1, config.completion_prob_decay_rounds),
                    )
                    model_prob = config.completion_prob_start + frac * (
                        config.completion_prob_end - config.completion_prob_start
                    )
                else:
                    model_prob = config.completion_prob_start

                use_model_completion = random.random() < model_prob
                completion_policy = (
                    "model" if use_model_completion else config.heuristic_policy
                )

        # === Data Collection ===
        print(
            f"Round {round_idx + 1}/{config.num_rounds} [{completion_policy}]: Collecting...",
            end=" ",
            flush=True,
        )

        # Note: exploration epsilon already computed above

        # Curriculum: adjust deadline slack early, anneal to target
        data_config_round = data_config
        if config.curriculum:
            decay_rounds = max(1, int(config.curriculum_fraction * config.num_rounds))
            frac = min(1.0, round_idx / decay_rounds)

            target_min = data_config.deadline_slack_ratio_min
            target_max = data_config.deadline_slack_ratio_max
            start_min = (
                config.curriculum_slack_min
                if config.curriculum_slack_min is not None
                else min(0.9, target_min + 0.3)
            )
            start_max = (
                config.curriculum_slack_max
                if config.curriculum_slack_max is not None
                else min(0.9, target_max + 0.3)
            )

            slack_min = start_min + frac * (target_min - start_min)
            slack_max = start_max + frac * (target_max - start_max)

            data_config_round = copy.deepcopy(data_config)
            data_config_round.deadline_slack_ratio_min = float(slack_min)
            data_config_round.deadline_slack_ratio_max = float(slack_max)

        transitions = collect_round_data(
            env_config=env_config,  # Pass config, not env instance
            model=model if use_model_completion else None,
            teacher_model=teacher_model,
            variant_config=variant_config,
            data_config=data_config_round,
            num_episodes=config.episodes_per_round,
            num_counterfactuals=config.num_counterfactuals,
            exploration_eps=eps,
            use_model_completion=use_model_completion,
            heuristic_policy=config.heuristic_policy,
            target_normalization=config.target_normalization,
            include_heuristic_candidates=config.include_heuristic_candidates,
            target_rollouts=config.target_rollouts,
            target_rollout_aggregation=config.target_rollout_aggregation,
            target_num_random_rollouts=config.target_num_random_rollouts,
            target_softmin_tau=config.target_softmin_tau,
            device=device,
            seed=config.seed + round_idx * 100000,
            collection_batch_size=config.collection_batch_size,
            num_collection_workers=config.num_collection_workers,
            allow_gpu_collection_multiprocessing=config.allow_gpu_collection_multiprocessing,
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
        avg_listwise = np.mean([m.get("listwise", 0.0) for m in epoch_metrics])
        avg_policy = np.mean([m.get("policy", 0.0) for m in epoch_metrics])
        avg_topg = np.mean([m.get("top_gamma", 0.0) for m in epoch_metrics])
        train_time = time.time() - train_start
        print(
            f"loss={avg_loss:.4f}, mae={avg_mae:.2f}, policy={avg_policy:.4f}, topγ={avg_topg:.4f}, "
            f"listwise={avg_listwise:.4f} in {train_time:.1f}s"
        )

        # Proactively tear down DataLoader workers before evaluation/logging to
        # avoid expensive/shaky multiprocessing shutdown during interpreter exit.
        if config.num_dataloader_workers > 0:
            try:
                import gc

                del dataloader
                del dataset
                gc.collect()
            except Exception:
                pass

        # === Evaluation ===
        eval_results = None
        sgbs_results = None
        if (round_idx + 1) % config.eval_every_rounds == 0:
            # Greedy evaluation
            print(f"  Evaluating (greedy)...", end=" ", flush=True)
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

            # SGBS evaluation (the real metric for beam search usage)
            if config.eval_sgbs:
                print(
                    f"  Evaluating (SGBS β={config.eval_sgbs_beta} γ={config.eval_sgbs_gamma})...",
                    end=" ",
                    flush=True,
                )
                sgbs_results = evaluate_sgbs(
                    model=model,
                    variant_config=variant_config,
                    data_config=data_config,
                    num_instances=config.eval_sgbs_num_instances,
                    seed=config.eval_seed,
                    device=device,
                    beta=config.eval_sgbs_beta,
                    gamma=config.eval_sgbs_gamma,
                )
                print(
                    f"sgbs_cost={sgbs_results['sgbs_cost']:.2f}, "
                    f"sgbs_vs_spt={sgbs_results['sgbs_vs_spt']:.1f}%, "
                    f"sgbs_vs_greedy={sgbs_results['sgbs_vs_greedy']:.1f}%"
                )

            # Save best (based on SGBS if available, otherwise greedy)
            current_cost = (
                sgbs_results["sgbs_cost"] if sgbs_results else eval_results["cost"]
            )
            if best_cost is None or current_cost < best_cost:
                best_cost = current_cost
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
            "policy": avg_policy,
            "top_gamma": avg_topg,
            "listwise": avg_listwise,
        }
        if eval_results:
            log_entry.update({f"eval_{k}": v for k, v in eval_results.items()})
        if sgbs_results:
            log_entry.update({f"sgbs_{k}": v for k, v in sgbs_results.items()})

        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()

        # === Checkpointing ===
        saved_ckpt = False
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
            saved_ckpt = True

        # === Teacher update (slow / checkpointed teacher) ===
        update_teacher = False
        if int(config.teacher_update_every_rounds) > 0:
            if (round_idx + 1) % int(config.teacher_update_every_rounds) == 0:
                update_teacher = True
        if bool(config.teacher_update_on_save) and saved_ckpt:
            update_teacher = True
        if update_teacher:
            try:
                teacher_model.load_state_dict(model.state_dict())
                teacher_model.eval()
            except Exception:
                pass

    # Final save
    torch.save(model.state_dict(), run_dir / "final_model.pt")
    log_file.close()

    print(f"\n{'='*60}")
    if best_cost is None:
        print(
            "Training complete! Best cost: N/A (evaluation never ran; "
            "set --eval_every_rounds to a smaller value to compute it)"
        )
    else:
        print(f"Training complete! Best cost: {best_cost:.2f}")
    print(f"Models saved to: {run_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

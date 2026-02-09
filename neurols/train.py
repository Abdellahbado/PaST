"""Training script for NeuroLS with Double DQN and IQN.

Implements training following NeuroLS paper Section 4.4:
- Double DQN for stable Q-learning
- n-step returns for better credit assignment
- Implicit Quantile Networks (IQN) for distributional RL
- Epsilon-greedy exploration with decay
- Target network soft updates

Usage:
    python -m PaST.neurols.train --config configs/neurols_small.yaml
"""

from __future__ import annotations

import os
import sys
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import random
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class _NoOpSummaryWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_histogram(self, *args, **kwargs):
        pass

    def add_text(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


if _TORCH_AVAILABLE:
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore
    except Exception:
        SummaryWriter = _NoOpSummaryWriter  # type: ignore


@dataclass
class TrainConfig:
    """Training configuration."""

    # Experiment
    exp_name: str = "neurols_default"
    seed: int = 42
    device: str = "cuda" if _TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    log_dir: str = "logs/neurols"
    checkpoint_dir: str = "checkpoints/neurols"

    # Checkpoint retention (helpful on storage-limited HPC)
    # Keep only the most recent periodic checkpoint_{episode}.pt files.
    checkpoint_keep_last: int = 1
    # Additionally keep a separate best checkpoint file (best.pt) updated on new best-ever.
    checkpoint_keep_best: bool = True
    checkpoint_best_name: str = "best.pt"

    # Data
    n_jobs_train: List[int] = field(default_factory=lambda: [10, 15, 20])
    n_machines_train: List[int] = field(default_factory=lambda: [2, 3, 4])
    K_range: Tuple[int, int] = (40, 120)  # Range for horizon K
    K_fixed: Optional[int] = 40  # Use a fixed K to keep tensor shapes consistent
    n_instances_per_reset: int = 100  # Train instances to cycle through

    # Environment
    max_steps: int = 500
    stagnation_limit: int = 100
    action_space: str = "AANP"  # AA, AAN, AANP, AANPD
    reward_mode: str = "dense_best"
    improvement_scale: float = 1.0
    reward_eps: float = 1e-8
    best_bonus_lambda: float = 0.3
    step_penalty: float = 0.0
    graph_type: str = "bipartite"  # bipartite or tripartite

    # Candidate generation
    top_k: int = 10
    use_proxy: bool = True
    proxy_mode: str = "load"  # load | price_aware

    # Optional: append per-job price/exposure features to job_features.
    # Must be consistent with d_job_in.
    job_price_features: str = "none"  # none | basic

    # Optional exposure-based shaping (reward_mode == dense_best_exposure)
    exposure_bonus_lambda: float = 0.0
    exposure_eps: float = 1e-8

    # Model — aligned with NeuroLS paper Appendix B
    # Input feature dimensions (keep defaults for backward compatibility)
    d_job_in: int = 5
    d_machine_in: int = 5
    d_state_in: int = 13
    d_emb: int = 128
    n_layers_static: int = 3  # Paper: L_stat = 3
    n_layers_dynamic: int = 2  # Paper: L_dyna = 2
    use_iqn: bool = True
    use_dueling: bool = False
    price_mode: str = "full"  # none, z_price, full
    dropout: float = 0.1

    # Training — aligned with paper
    n_episodes: int = 10000
    batch_size: int = 32
    learning_rate: float = 5e-4  # Paper: 0.0005
    weight_decay: float = 1e-5
    gamma: float = 0.99
    n_step: int = 3  # n-step returns

    # Exploration — aligned with paper
    epsilon_start: float = 0.95  # Paper: 0.95
    epsilon_end: float = 0.05
    # If None, compute a smooth exponential decay so that epsilon reaches
    # ~epsilon_end near the end of training (n_episodes).
    epsilon_decay: Optional[float] = None

    # Replay buffer — aligned with paper
    buffer_size: int = 32000  # Paper: 32000
    min_buffer_size: int = 1000  # Start training when buffer has this many transitions

    # Target network — aligned with paper
    target_update_freq: int = 500  # Paper: every 500 steps
    tau: float = 0.005  # Soft update (if used)
    use_soft_update: bool = False  # Paper uses hard update every 500 steps

    # IQN
    n_quantile_samples: int = 32
    n_target_quantile_samples: int = 32
    kappa: float = 1.0  # Huber loss threshold

    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    n_eval_episodes: int = 10

    # Parallelism
    n_parallel_envs: int = 0  # 0 = auto (cpu_count // 4, capped 16); 1 = serial

    def __post_init__(self) -> None:
        # Checkpoint retention validation
        if int(self.checkpoint_keep_last) < 1:
            self.checkpoint_keep_last = 1

        # Auto-compute epsilon decay if not provided.
        # We apply decay once per episode, so choose decay such that:
        #   epsilon_start * decay**n_episodes ~= epsilon_end
        if self.epsilon_decay is None:
            if self.n_episodes <= 0:
                self.epsilon_decay = 1.0
                return
            # Degenerate / safety cases
            if self.epsilon_start <= 0 or self.epsilon_end <= 0:
                self.epsilon_decay = 1.0
                return
            if self.epsilon_end >= self.epsilon_start:
                self.epsilon_decay = 1.0
                return

            self.epsilon_decay = float(
                np.exp(
                    np.log(self.epsilon_end / self.epsilon_start)
                    / float(max(1, self.n_episodes))
                )
            )

        # Final validation
        if not (0.0 < float(self.epsilon_decay) <= 1.0):
            raise ValueError(
                f"epsilon_decay must be in (0, 1], got {self.epsilon_decay}"
            )


class ReplayBuffer:
    """Experience replay buffer with n-step returns."""

    def __init__(
        self,
        capacity: int,
        n_step: int = 1,
        gamma: float = 0.99,
    ):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma

        self.buffer = deque(maxlen=capacity)
        self.n_step_buffer = deque(maxlen=n_step)

    def __len__(self) -> int:
        return len(self.buffer)

    def _compute_n_step_return(self) -> Tuple:
        """Compute n-step return from n_step_buffer."""
        # Get first transition
        state, action, _, _, _ = self.n_step_buffer[0]

        # Compute n-step reward
        n_step_reward = 0.0
        for i, (_, _, r, _, _) in enumerate(self.n_step_buffer):
            n_step_reward += (self.gamma**i) * r

        # Get last transition's next_state and done
        _, _, _, next_state, done = self.n_step_buffer[-1]

        return (state, action, n_step_reward, next_state, done)

    def push(
        self,
        state: Dict[str, np.ndarray],
        action: int,
        reward: float,
        next_state: Dict[str, np.ndarray],
        done: bool,
    ):
        """Add transition to buffer."""
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # Wait until we have n transitions
        if len(self.n_step_buffer) < self.n_step:
            return

        # Compute n-step return
        transition = self._compute_n_step_return()
        self.buffer.append(transition)

        # If done, flush remaining transitions
        if done:
            while len(self.n_step_buffer) > 1:
                self.n_step_buffer.popleft()
                if self.n_step_buffer:
                    transition = self._compute_n_step_return()
                    self.buffer.append(transition)
            self.n_step_buffer.clear()

    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Sample batch of transitions."""
        batch = random.sample(self.buffer, batch_size)

        # Unzip
        states, actions, rewards, next_states, dones = zip(*batch)

        return {
            "states": states,
            "actions": np.array(actions),
            "rewards": np.array(rewards, dtype=np.float32),
            "next_states": next_states,
            "dones": np.array(dones, dtype=np.float32),
        }

    def push_episode(self, transitions):
        """Push a complete episode's transitions (handles n-step internally).

        This is used by the parallel collector where transitions arrive
        as a batch after the episode finishes.  A *separate* n-step buffer
        is used so the main self.n_step_buffer is not corrupted.
        """
        temp = deque(maxlen=self.n_step)

        for state, action, reward, next_state, done in transitions:
            temp.append((state, action, reward, next_state, done))

            if len(temp) < self.n_step:
                continue

            s0, a0, _, _, _ = temp[0]
            n_reward = sum(self.gamma**i * r for i, (_, _, r, _, _) in enumerate(temp))
            _, _, _, ns, d = temp[-1]
            self.buffer.append((s0, a0, n_reward, ns, d))

        # Flush remaining (episode is done)
        while len(temp) > 1:
            temp.popleft()
            if temp:
                s0, a0, _, _, _ = temp[0]
                n_reward = sum(
                    self.gamma**i * r for i, (_, _, r, _, _) in enumerate(temp)
                )
                _, _, _, ns, d = temp[-1]
                self.buffer.append((s0, a0, n_reward, ns, d))
        temp.clear()


def collate_states(states: List[Dict], device: str) -> Dict[str, torch.Tensor]:
    """Collate list of state dicts into batched tensors.

    Handles variable-length fields (e.g. period_features, tripartite_edge_index)
    by zero-padding to the maximum size in the batch.
    """
    keys = states[0].keys()

    # Keys that may have variable lengths across batch items
    _VAR_LEN_KEYS = {"period_features", "tripartite_edge_index"}
    _LONG_KEYS = {
        "job_to_machine",
        "assignment_edges",
        "static_edge_index",
        "dynamic_edge_index",
        "tripartite_edge_index",
    }
    # Edge index keys must be padded with -1 (not 0) to avoid spurious self-loops
    _EDGE_INDEX_KEYS = {
        "static_edge_index",
        "dynamic_edge_index",
        "tripartite_edge_index",
    }

    batched = {}
    for key in keys:
        arrays = [s[key] for s in states]

        # Check if shapes vary (for variable-length keys)
        shapes = [a.shape for a in arrays]
        if key in _VAR_LEN_KEYS and len(set(shapes)) > 1:
            # Pad to max shape along each axis
            max_shape = tuple(max(s[i] for s in shapes) for i in range(len(shapes[0])))
            pad_val = -1 if key in _EDGE_INDEX_KEYS else 0
            padded = []
            for a in arrays:
                pad_widths = [
                    (0, max_shape[i] - a.shape[i]) for i in range(len(max_shape))
                ]
                padded.append(
                    np.pad(a, pad_widths, mode="constant", constant_values=pad_val)
                )
            stacked = np.stack(padded, axis=0)
        else:
            stacked = np.stack(arrays, axis=0)

        batched[key] = torch.tensor(stacked, device=device)

        # Set appropriate dtype
        if key in _LONG_KEYS:
            batched[key] = batched[key].long()
        else:
            batched[key] = batched[key].float()

    return batched


class NeuroLSTrainer:
    """Trainer for NeuroLS with Double DQN + IQN."""

    def __init__(self, config: TrainConfig):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for training")

        self.config = config
        self.device = config.device

        # Set seeds
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)

        # Create directories
        self.log_dir = Path(config.log_dir) / config.exp_name
        self.checkpoint_dir = Path(config.checkpoint_dir) / config.exp_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.log_dir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)

        # Initialize model
        self._init_model()

        # Replay buffer
        self.buffer = ReplayBuffer(
            capacity=config.buffer_size,
            n_step=config.n_step,
            gamma=config.gamma,
        )

        # Training state
        self.epsilon = config.epsilon_start
        self.global_step = 0
        self.episode = 0

        # Metrics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_improvements = deque(maxlen=100)
        self.episode_losses = deque(maxlen=100)
        self.episode_accept_rates = deque(maxlen=100)
        self.episode_improve_rates = deque(maxlen=100)
        self.episode_times = deque(maxlen=100)
        self.episode_grad_norms = deque(maxlen=100)
        self.episode_q_values = deque(maxlen=100)  # avg Q for monitoring

        # Best-ever tracking
        self.best_ever_cost = float("inf")
        self.best_ever_improvement = 0.0
        self.best_ever_episode = -1

        # Timing
        self.train_start_time = None

        # Logging
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def _init_model(self):
        """Initialize policy and target networks."""
        from PaST.neurols.decoder import NeuroLSPolicy
        from PaST.neurols.action_space import get_action_space

        # Get action space size
        action_space = get_action_space(self.config.action_space)
        n_actions = action_space.n_actions

        model_kwargs = dict(
            d_job_in=getattr(self.config, "d_job_in", 5),
            d_machine_in=getattr(self.config, "d_machine_in", 5),
            d_state_in=getattr(self.config, "d_state_in", 13),
            d_emb=self.config.d_emb,
            n_actions=n_actions,
            n_layers_static=self.config.n_layers_static,
            n_layers_dynamic=self.config.n_layers_dynamic,
            use_iqn=self.config.use_iqn,
            use_dueling=self.config.use_dueling,
            price_mode=self.config.price_mode,
            dropout=self.config.dropout,
            graph_type=self.config.graph_type,
        )

        # Create policy network
        self.policy_net = NeuroLSPolicy(**model_kwargs).to(self.device)

        # Create target network
        self.target_net = NeuroLSPolicy(**model_kwargs).to(self.device)

        # Copy weights to target
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler (optional)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.n_episodes,
            eta_min=self.config.learning_rate * 0.1,
        )

    def _cleanup_periodic_checkpoints(self) -> None:
        """Delete older periodic checkpoints to limit disk usage.

        Keeps only the newest `checkpoint_keep_last` files matching `checkpoint_*.pt`.
        Does not touch `final.pt` or the best checkpoint (e.g. best.pt).
        """
        keep_last = int(getattr(self.config, "checkpoint_keep_last", 1))
        if keep_last < 1:
            keep_last = 1

        periodic = []
        for p in self.checkpoint_dir.glob("checkpoint_*.pt"):
            # Parse episode from filename `checkpoint_{episode}.pt`
            stem = p.stem
            try:
                ep = int(stem.split("_")[-1])
            except Exception:
                continue
            periodic.append((ep, p))

        if len(periodic) <= keep_last:
            return

        periodic.sort(key=lambda t: t[0])
        to_delete = periodic[:-keep_last]
        for _, path in to_delete:
            try:
                path.unlink()
            except Exception:
                # Best-effort cleanup; ignore filesystem hiccups.
                pass

    def _prune_periodic_checkpoints_for_new(self) -> None:
        """Pre-prune periodic checkpoints before saving a new one.

        This is useful when the filesystem is near-full: we delete old periodic
        checkpoints first to make room for the new checkpoint.
        """
        keep_last = int(getattr(self.config, "checkpoint_keep_last", 1))
        if keep_last < 1:
            keep_last = 1

        # We want to keep at most (keep_last - 1) existing periodic files,
        # since we are about to write a new one.
        target_existing = max(0, keep_last - 1)

        periodic = []
        for p in self.checkpoint_dir.glob("checkpoint_*.pt"):
            try:
                ep = int(p.stem.split("_")[-1])
            except Exception:
                continue
            periodic.append((ep, p))

        if len(periodic) <= target_existing:
            return

        periodic.sort(key=lambda t: t[0])
        to_delete = periodic[: len(periodic) - target_existing]
        for _, path in to_delete:
            try:
                path.unlink()
            except Exception:
                pass

    def _update_target(self):
        """Update target network."""
        if self.config.use_soft_update:
            # Soft update
            for target_param, policy_param in zip(
                self.target_net.parameters(), self.policy_net.parameters()
            ):
                target_param.data.copy_(
                    self.config.tau * policy_param.data
                    + (1 - self.config.tau) * target_param.data
                )
        else:
            # Hard update
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _get_action(self, state: Dict[str, np.ndarray]) -> int:
        """Get action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randrange(self.policy_net.n_actions)

        with torch.no_grad():
            # Convert state to tensors – NO batch dim; GNN expects unbatched
            state_tensors = {
                k: torch.tensor(v, device=self.device) for k, v in state.items()
            }

            q_values = self.policy_net(
                job_features=state_tensors["job_features"].float(),
                machine_features=state_tensors["machine_features"].float(),
                state_features=state_tensors["state_features"].float(),
                job_to_machine=state_tensors["job_to_machine"].long(),
                static_edge_index=state_tensors["static_edge_index"].long(),
                dynamic_edge_index=state_tensors["dynamic_edge_index"].long(),
                price_features=state_tensors.get("price_per_hour", None),
                machine_exposure=(
                    state_tensors["machine_exposure"].float()
                    if "machine_exposure" in state_tensors
                    else None
                ),
                period_features=(
                    state_tensors["period_features"].float()
                    if "period_features" in state_tensors
                    else None
                ),
                tripartite_edge_index=(
                    state_tensors["tripartite_edge_index"].long()
                    if "tripartite_edge_index" in state_tensors
                    else None
                ),
            )

            return q_values.argmax().item()

    def _compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute Double DQN or IQN loss."""
        # Collate states
        states = collate_states(batch["states"], self.device)
        next_states = collate_states(batch["next_states"], self.device)

        actions = torch.tensor(batch["actions"], device=self.device, dtype=torch.long)
        rewards = torch.tensor(
            batch["rewards"], device=self.device, dtype=torch.float32
        )
        dones = torch.tensor(batch["dones"], device=self.device, dtype=torch.float32)

        batch_size = len(batch["actions"])

        if self.config.use_iqn:
            return self._compute_iqn_loss(
                states, actions, rewards, next_states, dones, batch_size
            )
        else:
            return self._compute_dqn_loss(
                states, actions, rewards, next_states, dones, batch_size
            )

    def _compute_dqn_loss(
        self,
        states: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: Dict[str, torch.Tensor],
        dones: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Compute Double DQN loss."""
        # Current Q-values
        q_values = self._forward_batch(self.policy_net, states)  # (B, A)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        # Target Q-values (Double DQN)
        with torch.no_grad():
            # Select actions with policy network
            next_q_policy = self._forward_batch(self.policy_net, next_states)
            next_actions = next_q_policy.argmax(dim=1)  # (B,)

            # Evaluate with target network
            next_q_target = self._forward_batch(self.target_net, next_states)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # Compute target
            gamma_n = self.config.gamma**self.config.n_step
            target = rewards + gamma_n * next_q * (1 - dones)

        # Huber loss
        loss = nn.functional.smooth_l1_loss(q_values, target)

        return loss

    def _compute_iqn_loss(
        self,
        states: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: Dict[str, torch.Tensor],
        dones: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Compute IQN quantile regression loss."""
        n_samples = self.config.n_quantile_samples
        n_target_samples = self.config.n_target_quantile_samples

        # Sample quantiles for current state
        tau = torch.rand(batch_size, n_samples, device=self.device)

        # Current quantile values
        q_quantiles = self._forward_batch_iqn(self.policy_net, states, tau)  # (B, N, A)

        # Select actions
        q_quantiles = q_quantiles.gather(
            2, actions.unsqueeze(1).unsqueeze(2).expand(-1, n_samples, -1)
        ).squeeze(
            2
        )  # (B, N)

        # Target quantile values
        with torch.no_grad():
            # Sample quantiles for target
            tau_prime = torch.rand(batch_size, n_target_samples, device=self.device)

            # Get target Q-values (for action selection with policy net)
            next_q_policy = self._forward_batch(self.policy_net, next_states)
            next_actions = next_q_policy.argmax(dim=1)  # (B,)

            # Get target quantile values
            next_q_quantiles = self._forward_batch_iqn(
                self.target_net, next_states, tau_prime
            )  # (B, N', A)

            next_q_quantiles = next_q_quantiles.gather(
                2,
                next_actions.unsqueeze(1).unsqueeze(2).expand(-1, n_target_samples, -1),
            ).squeeze(
                2
            )  # (B, N')

            # Compute target
            gamma_n = self.config.gamma**self.config.n_step
            target = rewards.unsqueeze(1) + gamma_n * next_q_quantiles * (
                1 - dones.unsqueeze(1)
            )

        # Quantile Huber loss
        # Reshape for pairwise difference: (B, N, 1) - (B, 1, N') -> (B, N, N')
        td_error = target.unsqueeze(1) - q_quantiles.unsqueeze(2)

        # Huber loss
        huber_loss = torch.where(
            td_error.abs() <= self.config.kappa,
            0.5 * td_error.pow(2),
            self.config.kappa * (td_error.abs() - 0.5 * self.config.kappa),
        )

        # Quantile regression
        # tau: (B, N) -> (B, N, 1)
        tau_expanded = tau.unsqueeze(2)
        quantile_weights = torch.abs(tau_expanded - (td_error < 0).float())

        loss = (quantile_weights * huber_loss).mean()

        return loss

    def _forward_batch(
        self,
        net: nn.Module,
        states: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Batched forward — single super-graph scatter, no loop."""
        has_price = "price_per_hour" in states
        has_period = "period_features" in states
        has_tri = "tripartite_edge_index" in states
        has_exposure = "machine_exposure" in states
        return net.forward_batched(
            job_features=states["job_features"].float(),
            machine_features=states["machine_features"].float(),
            state_features=states["state_features"].float(),
            job_to_machine=states["job_to_machine"].long(),
            static_edge_index=states["static_edge_index"].long(),
            dynamic_edge_index=states["dynamic_edge_index"].long(),
            price_features=(states["price_per_hour"].float() if has_price else None),
            machine_exposure=(
                states["machine_exposure"].float() if has_exposure else None
            ),
            period_features=(states["period_features"].float() if has_period else None),
            tripartite_edge_index=(
                states["tripartite_edge_index"].long() if has_tri else None
            ),
        )  # (B, A)

    def _forward_batch_iqn(
        self,
        net: nn.Module,
        states: Dict[str, torch.Tensor],
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """Batched IQN forward — single super-graph scatter, no loop."""
        has_price = "price_per_hour" in states
        has_period = "period_features" in states
        has_tri = "tripartite_edge_index" in states
        has_exposure = "machine_exposure" in states
        return net.forward_batched(
            job_features=states["job_features"].float(),
            machine_features=states["machine_features"].float(),
            state_features=states["state_features"].float(),
            job_to_machine=states["job_to_machine"].long(),
            static_edge_index=states["static_edge_index"].long(),
            dynamic_edge_index=states["dynamic_edge_index"].long(),
            price_features=(states["price_per_hour"].float() if has_price else None),
            machine_exposure=(
                states["machine_exposure"].float() if has_exposure else None
            ),
            tau=tau,
            period_features=(states["period_features"].float() if has_period else None),
            tripartite_edge_index=(
                states["tripartite_edge_index"].long() if has_tri else None
            ),
        )  # (B, N, A)

    def train_step(self) -> Tuple[float, float]:
        """Perform one training step.

        Returns:
            (loss, grad_norm) — both 0.0 if buffer not ready.
        """
        if len(self.buffer) < self.config.min_buffer_size:
            return 0.0, 0.0

        # Sample batch
        batch = self.buffer.sample(self.config.batch_size)

        # Compute loss
        loss = self._compute_loss(batch)

        # Guard against NaN/inf loss (can happen with corrupted replay data)
        if not torch.isfinite(loss):
            return 0.0, 0.0

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (returns the total norm before clipping)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), 1.0
        ).item()

        self.optimizer.step()

        # Update target network
        if self.config.use_soft_update:
            self._update_target()
        elif self.global_step % self.config.target_update_freq == 0:
            self._update_target()

        self.global_step += 1

        return loss.item(), grad_norm

    def run_episode(self, env, instance, K: int) -> Dict[str, float]:
        """Run one episode and collect experience."""
        ep_start = time.time()
        state = env.reset(instance, K)
        state_features = env.get_state_features()

        episode_reward = 0.0
        episode_length = 0
        initial_cost = state.current_cost
        n_accepted = 0
        n_improved = 0
        n_operator_steps = 0
        losses = []
        grad_norms = []
        q_values_seen = []
        action_counts: Dict[int, int] = {}

        while True:
            # Select action
            action = self._get_action(state_features)
            action_counts[action] = action_counts.get(action, 0) + 1

            # Step environment
            next_state, reward, done, info = env.step(action)
            next_state_features = env.get_state_features()

            # Track acceptance / improvement
            if info.get("accepted", False):
                n_accepted += 1
            if info.get("improved", False):
                n_improved += 1
            if info.get("move_result") is not None or info.get("perturbed", False):
                n_operator_steps += 1

            # Store transition
            self.buffer.push(
                state_features,
                action,
                reward,
                next_state_features,
                done,
            )

            # Train
            loss, grad_norm = self.train_step()
            if loss > 0:
                losses.append(loss)
                grad_norms.append(grad_norm)

            # Update state
            state = next_state
            state_features = next_state_features
            episode_reward += reward
            episode_length += 1

            if done:
                break

        ep_time = time.time() - ep_start
        # Compute improvement
        improvement = (initial_cost - state.best_cost) / initial_cost * 100
        accept_rate = n_accepted / max(1, n_operator_steps)
        improve_rate = n_improved / max(1, n_operator_steps)
        avg_loss = float(np.mean(losses)) if losses else 0.0
        avg_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0

        # Cache stats from env
        cache_stats = getattr(env, "get_cache_stats", lambda: {})()

        return {
            "reward": episode_reward,
            "length": episode_length,
            "improvement": improvement,
            "initial_cost": initial_cost,
            "best_cost": state.best_cost,
            "loss": avg_loss,
            "accept_rate": accept_rate,
            "improve_rate": improve_rate,
            "ep_time": ep_time,
            "n_grad_steps": len(losses),
            "action_counts": action_counts,
            "cache_hits": cache_stats.get("hits", 0),
            "cache_size": cache_stats.get("size", 0),
            "cache_hit_rate": cache_stats.get("hit_rate", 0.0),
            "avg_grad_norm": avg_grad_norm,
        }

    def train(self):
        """Main training loop."""
        from PaST.neurols.env import NeuroLSEnv, EnvConfig
        from PaST.data.sm_benchmark_data import generate_raw_instance
        from PaST.config import DataConfig

        print(f"Starting training: {self.config.exp_name}")
        print(f"Device: {self.device}")
        print(f"Config: {asdict(self.config)}")

        # Create environment
        env_config = EnvConfig(
            max_steps=self.config.max_steps,
            stagnation_limit=self.config.stagnation_limit,
            action_space=self.config.action_space,
            reward_mode=self.config.reward_mode,
            improvement_scale=self.config.improvement_scale,
            reward_eps=self.config.reward_eps,
            best_bonus_lambda=self.config.best_bonus_lambda,
            step_penalty=self.config.step_penalty,
            top_k=self.config.top_k,
            use_proxy=self.config.use_proxy,
            proxy_mode=self.config.proxy_mode,
            job_price_features=getattr(self.config, "job_price_features", "none"),
            exposure_bonus_lambda=self.config.exposure_bonus_lambda,
            exposure_eps=self.config.exposure_eps,
            graph_type=self.config.graph_type,
            price_mode=self.config.price_mode,
        )
        env = NeuroLSEnv(env_config)
        env.seed(self.config.seed)

        # Generate training instances.
        # IMPORTANT: This trainer currently batches by `np.stack`, so tensor shapes
        # must be constant within a run (n_jobs, n_machines, and K/price length).
        data_config = DataConfig(sampling_mode="new_benchmark_grid")
        data_rng = random.Random(self.config.seed)
        hours_per_day = int(getattr(data_config, "hours_per_day", 20))

        fixed_n = int(self.config.n_jobs_train[0])
        fixed_m = int(self.config.n_machines_train[0])
        fixed_K = int(self.config.K_fixed or self.config.K_range[0])
        if fixed_K % hours_per_day != 0:
            fixed_K = (fixed_K // hours_per_day) * hours_per_day
            fixed_K = max(fixed_K, hours_per_day)
        fixed_D_days = int(fixed_K // hours_per_day)

        train_instances = []
        for i in range(self.config.n_instances_per_reset):
            instance = generate_raw_instance(
                config=data_config,
                rng=data_rng,
                instance_id=i,
                n=fixed_n,
                m=fixed_m,
                D_days=fixed_D_days,
            )
            train_instances.append((instance, fixed_K))

        # ── Resolve parallelism level ─────────────────────────
        n_parallel = self.config.n_parallel_envs
        if n_parallel <= 0:
            import multiprocessing as _mp

            n_parallel = max(1, min(16, _mp.cpu_count() // 4))
        use_parallel = n_parallel > 1

        # Training loop
        self.train_start_time = time.time()
        print(
            f"\n{'='*70}\n"
            f"  Instances: {len(train_instances)}  |  n={fixed_n}  m={fixed_m}  K={fixed_K}\n"
            f"  Action space: {self.config.action_space} ({self.policy_net.n_actions} actions)\n"
            f"  Price mode: {self.config.price_mode}\n"
            f"  Buffer: {self.config.min_buffer_size}/{self.config.buffer_size}  "
            f"(training starts after {self.config.min_buffer_size} transitions)\n"
            f"  Parallel workers: {n_parallel}\n"
            f"{'='*70}"
        )

        if use_parallel:
            self._train_parallel(env, env_config, train_instances, n_parallel)
        else:
            self._train_serial(env, train_instances)

        # ── Finish ────────────────────────────────────────────
        total_time = time.time() - self.train_start_time
        self.save_checkpoint("final.pt")
        self.writer.close()

        print(
            f"\n{'='*70}\n"
            f"  Training complete!\n"
            f"  Total time: {total_time/3600:.1f}h\n"
            f"  Best cost ever: {self.best_ever_cost:.2f} "
            f"({self.best_ever_improvement:.2f}% improvement, ep {self.best_ever_episode})\n"
            f"  Final epsilon: {self.epsilon:.4f}\n"
            f"  Global steps: {self.global_step}\n"
            f"{'='*70}"
        )

    # ── Serial training (original loop) ──────────────────────────────────

    def _train_serial(self, env, train_instances):
        """Original serial training loop (1 episode at a time)."""
        start_episode = int(self.episode) + 1 if int(self.episode) >= 0 else 0
        for episode in range(start_episode, self.config.n_episodes):
            instance, K = random.choice(train_instances)
            metrics = self.run_episode(env, instance, K)

            self.epsilon = max(
                self.config.epsilon_end, self.epsilon * self.config.epsilon_decay
            )

            self._track_and_log(episode, metrics, train_instances, env)
            self.episode = episode

    # ── Parallel training ────────────────────────────────────────────────

    def _train_parallel(self, env, env_config, train_instances, n_parallel):
        """Parallel training: N workers collect episodes on CPU, GPU trains."""
        from PaST.neurols.parallel_collector import ParallelCollector

        collector = ParallelCollector(n_workers=n_parallel)
        collector.start()

        env_config_dict = {
            "max_steps": env_config.max_steps,
            "stagnation_limit": env_config.stagnation_limit,
            "action_space": self.config.action_space,
            "reward_mode": env_config.reward_mode,
            "improvement_scale": env_config.improvement_scale,
            "reward_eps": env_config.reward_eps,
            "best_bonus_lambda": env_config.best_bonus_lambda,
            "step_penalty": env_config.step_penalty,
            "top_k": self.config.top_k,
            "use_proxy": self.config.use_proxy,
            "proxy_mode": self.config.proxy_mode,
            "job_price_features": self.config.job_price_features,
            "exposure_bonus_lambda": self.config.exposure_bonus_lambda,
            "exposure_eps": self.config.exposure_eps,
            "graph_type": self.config.graph_type,
            "price_mode": self.config.price_mode,
        }
        model_config = {
            "d_job_in": self.config.d_job_in,
            "d_machine_in": self.config.d_machine_in,
            "d_state_in": self.config.d_state_in,
            "d_emb": self.config.d_emb,
            "n_layers_static": self.config.n_layers_static,
            "n_layers_dynamic": self.config.n_layers_dynamic,
            "use_iqn": self.config.use_iqn,
            "use_dueling": self.config.use_dueling,
            "price_mode": self.config.price_mode,
            "dropout": self.config.dropout,
            "action_space": self.config.action_space,
            "graph_type": self.config.graph_type,
        }

        # If resuming, continue from the next episode after the checkpoint.
        episodes_done = int(self.episode) + 1 if int(self.episode) >= 0 else 0
        round_idx = 0

        try:
            while episodes_done < self.config.n_episodes:
                round_start = time.time()
                n_collect = min(n_parallel, self.config.n_episodes - episodes_done)

                # Select instances for this round
                round_instances = [
                    random.choice(train_instances) for _ in range(n_collect)
                ]

                # ── Phase 1: collect in parallel ──
                results = collector.collect_episodes(
                    instances_and_Ks=round_instances,
                    model_state_dict=self.policy_net.state_dict(),
                    env_config_dict=env_config_dict,
                    model_config=model_config,
                    epsilon=self.epsilon,
                    base_seed=self.config.seed + episodes_done,
                )

                # ── Phase 2: feed transitions into replay buffer ──
                total_transitions = 0
                for result in results:
                    self.buffer.push_episode(result.transitions)
                    total_transitions += len(result.transitions)

                # ── Phase 3: GPU training steps ──
                round_losses = []
                round_grad_norms = []
                if len(self.buffer) >= self.config.min_buffer_size:
                    n_train = max(1, total_transitions // self.config.batch_size)
                    for _ in range(n_train):
                        loss, gn = self.train_step()
                        if loss > 0:
                            round_losses.append(loss)
                            round_grad_norms.append(gn)

                round_time = time.time() - round_start

                # ── Phase 4: update metrics for each collected episode ──
                for i, result in enumerate(results):
                    episode = episodes_done + i
                    m = result.metrics

                    # Epsilon decay per episode
                    self.epsilon = max(
                        self.config.epsilon_end,
                        self.epsilon * self.config.epsilon_decay,
                    )

                    # Inject training loss/grad info into metrics
                    m["loss"] = float(np.mean(round_losses)) if round_losses else 0.0
                    m["avg_grad_norm"] = (
                        float(np.mean(round_grad_norms)) if round_grad_norms else 0.0
                    )
                    m["n_grad_steps"] = len(round_losses)
                    # Time per episode = round_time / n_collect (amortised)
                    m["ep_time"] = round_time / max(1, n_collect)

                    self._track_and_log(episode, m, train_instances, env)
                    self.episode = episode

                episodes_done += n_collect
                round_idx += 1

        finally:
            collector.stop()

    # ── Common helpers for both modes ─────────────────────────────────────

    def _track_and_log(self, episode, metrics, train_instances, env):
        """Track metrics, log, evaluate, and save checkpoints."""
        self.episode_rewards.append(metrics["reward"])
        self.episode_lengths.append(metrics["length"])
        self.episode_improvements.append(metrics["improvement"])
        self.episode_losses.append(metrics.get("loss", 0.0))
        self.episode_accept_rates.append(metrics["accept_rate"])
        self.episode_improve_rates.append(metrics.get("improve_rate", 0.0))
        self.episode_times.append(metrics["ep_time"])
        self.episode_grad_norms.append(metrics.get("avg_grad_norm", 0.0))

        # Best-ever tracking
        if metrics["best_cost"] < self.best_ever_cost:
            self.best_ever_cost = metrics["best_cost"]
            self.best_ever_improvement = metrics["improvement"]
            self.best_ever_episode = episode

            # Save best checkpoint (separate file) if enabled.
            if getattr(self.config, "checkpoint_keep_best", True):
                best_name = getattr(self.config, "checkpoint_best_name", "best.pt")
                self.save_checkpoint(best_name, cleanup_old=False)

        # ── Logging ──────────────────────────────────────────
        if episode % self.config.log_interval == 0:
            elapsed = time.time() - self.train_start_time
            avg_reward = float(np.mean(self.episode_rewards))
            avg_length = float(np.mean(self.episode_lengths))
            avg_improvement = float(np.mean(self.episode_improvements))
            avg_loss = float(np.mean(self.episode_losses))
            avg_accept = float(np.mean(self.episode_accept_rates))
            avg_improve = float(np.mean(self.episode_improve_rates))
            avg_ep_time = float(np.mean(self.episode_times))
            avg_grad_norm = float(np.mean(self.episode_grad_norms))
            lr_now = self.optimizer.param_groups[0]["lr"]

            # ETA
            eps_done = episode + 1
            eps_left = self.config.n_episodes - eps_done
            eta_s = avg_ep_time * eps_left if avg_ep_time > 0 else 0
            eta_str = (
                f"{int(eta_s//3600)}h{int((eta_s%3600)//60):02d}m"
                if eta_s > 3600
                else f"{int(eta_s//60)}m{int(eta_s%60):02d}s"
            )

            # TensorBoard
            self.writer.add_scalar("train/reward", avg_reward, episode)
            self.writer.add_scalar("train/length", avg_length, episode)
            self.writer.add_scalar("train/improvement", avg_improvement, episode)
            self.writer.add_scalar("train/epsilon", self.epsilon, episode)
            self.writer.add_scalar("train/loss", avg_loss, episode)
            self.writer.add_scalar("train/lr", lr_now, episode)
            self.writer.add_scalar("train/buffer_size", len(self.buffer), episode)
            self.writer.add_scalar("train/accept_rate", avg_accept, episode)
            self.writer.add_scalar("train/improve_rate", avg_improve, episode)
            self.writer.add_scalar("train/ep_time", avg_ep_time, episode)
            self.writer.add_scalar("train/grad_norm", avg_grad_norm, episode)
            self.writer.add_scalar(
                "train/grad_steps", metrics.get("n_grad_steps", 0), episode
            )
            self.writer.add_scalar("train/best_ever_cost", self.best_ever_cost, episode)
            if metrics.get("cache_size", 0) > 0:
                self.writer.add_scalar(
                    "train/cache_hit_rate", metrics["cache_hit_rate"], episode
                )
                self.writer.add_scalar(
                    "train/cache_size", metrics["cache_size"], episode
                )

            # GPU memory
            gpu_str = ""
            if self.device != "cpu" and torch.cuda.is_available():
                mem_mb = torch.cuda.max_memory_allocated() / 1e6
                gpu_str = f" | GPU: {mem_mb:.0f}MB"
                self.writer.add_scalar("train/gpu_mem_mb", mem_mb, episode)

            # Buffer progress
            buf_pct = min(
                100, len(self.buffer) * 100 // max(1, self.config.min_buffer_size)
            )
            buf_str = (
                f"Buf: {len(self.buffer)}/{self.config.min_buffer_size} ({buf_pct}%)"
                if len(self.buffer) < self.config.min_buffer_size
                else f"Buf: {len(self.buffer)}"
            )

            print(
                f"Ep {episode:5d}/{self.config.n_episodes} | "
                f"R: {avg_reward:8.1f} | "
                f"Imp: {avg_improvement:5.2f}% | "
                f"Acc: {avg_accept:.0%} | "
                f"L: {avg_loss:.4f} | "
                f"GN: {avg_grad_norm:.3f} | "
                f"Eps: {self.epsilon:.3f} | "
                f"{buf_str} | "
                f"Best*: {self.best_ever_cost:.1f} ({self.best_ever_improvement:.2f}%) | "
                f"LR: {lr_now:.1e} | "
                f"{avg_ep_time:.1f}s/ep | "
                f"ETA: {eta_str}"
                f"{gpu_str}"
            )

        # ── Evaluation ───────────────────────────────────────
        if episode % self.config.eval_interval == 0:
            eval_start = time.time()
            eval_metrics = self.evaluate(env, train_instances)
            eval_time = time.time() - eval_start

            self.writer.add_scalar("eval/reward", eval_metrics["reward"], episode)
            self.writer.add_scalar(
                "eval/improvement", eval_metrics["improvement"], episode
            )
            self.writer.add_scalar("eval/best_cost", eval_metrics["best_cost"], episode)
            self.writer.add_scalar(
                "eval/random_improvement",
                eval_metrics["random_improvement"],
                episode,
            )
            self.writer.add_scalar(
                "eval/advantage_over_random",
                eval_metrics["advantage_over_random"],
                episode,
            )
            self.writer.add_scalar("eval/avg_q", eval_metrics["avg_q"], episode)
            self.writer.add_scalar("eval/time", eval_time, episode)

            adv = eval_metrics["advantage_over_random"]
            adv_str = f"+{adv:.2f}pp" if adv >= 0 else f"{adv:.2f}pp"
            print(
                f"  [EVAL] Greedy: {eval_metrics['improvement']:.2f}% | "
                f"Random: {eval_metrics['random_improvement']:.2f}% | "
                f"Advantage: {adv_str} | "
                f"AvgQ: {eval_metrics['avg_q']:.3f} | "
                f"Best: {eval_metrics['best_cost']:.1f} | "
                f"({eval_time:.1f}s)"
            )

        # Save checkpoint
        if episode % self.config.save_interval == 0:
            self.save_checkpoint(f"checkpoint_{episode}.pt", cleanup_old=True)

        # Update learning rate
        self.scheduler.step()

    def evaluate(self, env, instances: List[Tuple]) -> Dict[str, float]:
        """Evaluate policy on held-out instances vs random baseline."""
        self.policy_net.eval()

        # Make evaluation deterministic: disable stochastic acceptance of worsening moves.
        # This reduces variance and makes greedy vs random comparisons meaningful.
        prev_deterministic = getattr(env.config, "deterministic", False)
        env.config.deterministic = True

        # Use a local RNG so the random baseline is stable and does not depend on
        # global `random` state advanced elsewhere.
        eval_rng = random.Random(self.config.seed + 100_000 + int(self.episode))

        # Save and override epsilon for greedy evaluation
        saved_epsilon = self.epsilon
        self.epsilon = 0.0

        greedy_rewards = []
        greedy_improvements = []
        greedy_best_costs = []
        random_improvements = []
        random_best_costs = []
        avg_q_values = []

        n_eval = min(self.config.n_eval_episodes, len(instances))

        with torch.no_grad():
            for i in range(n_eval):
                instance, K = instances[i]

                # Reseed env per instance for reproducibility.
                try:
                    env.seed(self.config.seed + 200_000 + i)
                except Exception:
                    pass

                # ── Greedy policy (epsilon=0) ──
                state = env.reset(instance, K)
                state_features = env.get_state_features()
                episode_reward = 0.0
                initial_cost = state.current_cost
                q_vals_episode = []

                while True:
                    # Also track Q-values for diagnostics
                    state_tensors = {
                        k: torch.tensor(v, device=self.device)
                        for k, v in state_features.items()
                    }
                    q_values = self.policy_net(
                        job_features=state_tensors["job_features"].float(),
                        machine_features=state_tensors["machine_features"].float(),
                        state_features=state_tensors["state_features"].float(),
                        job_to_machine=state_tensors["job_to_machine"].long(),
                        static_edge_index=state_tensors["static_edge_index"].long(),
                        dynamic_edge_index=state_tensors["dynamic_edge_index"].long(),
                        price_features=state_tensors.get("price_per_hour", None),
                        machine_exposure=(
                            state_tensors["machine_exposure"].float()
                            if "machine_exposure" in state_tensors
                            else None
                        ),
                        period_features=(
                            state_tensors["period_features"].float()
                            if "period_features" in state_tensors
                            else None
                        ),
                        tripartite_edge_index=(
                            state_tensors["tripartite_edge_index"].long()
                            if "tripartite_edge_index" in state_tensors
                            else None
                        ),
                    )
                    q_vals_episode.append(q_values.max().item())
                    action = q_values.argmax().item()

                    state, reward, done, _ = env.step(action)
                    state_features = env.get_state_features()
                    episode_reward += reward
                    if done:
                        break

                improvement = (initial_cost - state.best_cost) / initial_cost * 100
                greedy_rewards.append(episode_reward)
                greedy_improvements.append(improvement)
                greedy_best_costs.append(state.best_cost)
                if q_vals_episode:
                    avg_q_values.append(float(np.mean(q_vals_episode)))

                # ── Random baseline (same instance) ──
                state = env.reset(instance, K)
                initial_cost_r = state.current_cost

                while True:
                    action = eval_rng.randrange(self.policy_net.n_actions)
                    state, reward, done, _ = env.step(action)
                    if done:
                        break

                rand_imp = (initial_cost_r - state.best_cost) / initial_cost_r * 100
                random_improvements.append(rand_imp)
                random_best_costs.append(state.best_cost)

        # Restore epsilon
        self.epsilon = saved_epsilon
        # Restore env setting
        env.config.deterministic = prev_deterministic
        self.policy_net.train()

        return {
            "reward": float(np.mean(greedy_rewards)),
            "improvement": float(np.mean(greedy_improvements)),
            "best_cost": float(np.mean(greedy_best_costs)),
            "random_improvement": float(np.mean(random_improvements)),
            "random_best_cost": float(np.mean(random_best_costs)),
            "advantage_over_random": float(
                np.mean(greedy_improvements) - np.mean(random_improvements)
            ),
            "avg_q": float(np.mean(avg_q_values)) if avg_q_values else 0.0,
        }

    def save_checkpoint(self, filename: str, *, cleanup_old: bool = True):
        """Save checkpoint."""
        path = self.checkpoint_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        is_periodic = filename.startswith("checkpoint_") and filename.endswith(".pt")
        if cleanup_old and is_periodic:
            self._prune_periodic_checkpoints_for_new()

        payload = {
            "episode": self.episode,
            "global_step": self.global_step,
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epsilon": self.epsilon,
            "config": asdict(self.config),
        }

        tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        legacy_tmp_path = path.with_suffix(path.suffix + f".legacy.tmp.{os.getpid()}")
        try:
            # Default: zipfile-based serialization (PyTorch default).
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
            print(f"Saved checkpoint: {path}")

            if cleanup_old and is_periodic:
                self._cleanup_periodic_checkpoints()
            return
        except Exception as exc:
            # Some network/overlay filesystems occasionally fail with
            # "inline_container.cc: unexpected pos ...". Retry with legacy
            # serialization which is often more robust.
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

            try:
                torch.save(
                    payload, legacy_tmp_path, _use_new_zipfile_serialization=False
                )
                os.replace(legacy_tmp_path, path)
                print(f"Saved checkpoint (legacy): {path}")

                if cleanup_old and is_periodic:
                    self._cleanup_periodic_checkpoints()
                return
            except Exception as exc2:
                # Keep tmp files if they exist for post-mortem debugging.
                print(
                    f"[WARN] Failed to save checkpoint to {path}: {exc}\n"
                    f"[WARN] Legacy save also failed: {exc2}\n"
                    f"[WARN] tmp files (if any) left at: {tmp_path} and {legacy_tmp_path}"
                )

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        p = Path(path)
        if not p.exists() and not p.is_absolute():
            # Common usage in curriculum chains: `--resume final.pt`.
            # Resolve relative to this experiment's checkpoint directory.
            alt = self.checkpoint_dir / p
            if alt.exists():
                p = alt

        if not p.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {path}. "
                "If a run crashed during saving, try resuming from the latest "
                "checkpoint_*.pt in the experiment folder instead of final.pt."
            )

        checkpoint = torch.load(str(p), map_location=self.device)

        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.episode = checkpoint["episode"]
        self.global_step = checkpoint["global_step"]
        self.epsilon = checkpoint["epsilon"]

        print(f"Loaded checkpoint from episode {self.episode}")


def main():
    parser = argparse.ArgumentParser(description="Train NeuroLS")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint (.pt) saved by this trainer to resume from.",
    )
    parser.add_argument(
        "--resume-reset-epsilon",
        type=float,
        default=None,
        help=(
            "If set, overrides the loaded checkpoint epsilon with this value after resuming. "
            "Useful to increase exploration when continuing training longer."
        ),
    )
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-episodes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=None,
        help="Override epsilon_start (only affects new runs and decay computation).",
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=None,
        help="Override epsilon_end (only affects decay computation and min epsilon).",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=None,
        help=(
            "Override epsilon_decay directly. If provided, disables auto-computation. "
            "Must be in (0, 1]."
        ),
    )
    parser.add_argument(
        "--use-iqn",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Enable/disable IQN (distributional RL)",
    )
    parser.add_argument(
        "--action-space",
        type=str,
        default=None,
        choices=[
            "AA",
            "AAN",
            "AAN_CLEAN",
            "AANP",
            "AANP_CLEAN",
            "AANP_PRICE",
            "AANPD",
            "AANPD_CLEAN",
        ],
    )
    parser.add_argument(
        "--graph-type",
        type=str,
        default=None,
        choices=["bipartite", "tripartite"],
        help="Graph representation: bipartite (default) or tripartite (adds period nodes)",
    )
    parser.add_argument(
        "--n-parallel",
        type=int,
        default=None,
        help="Parallel env workers (0=auto, 1=serial)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Alias for --n-parallel (number of parallel workers)",
    )
    parser.add_argument(
        "--n-cores",
        type=int,
        default=None,
        help="Alias for --n-parallel (use this many CPU workers)",
    )

    args = parser.parse_args()

    # Load config from file or create default
    if args.config is not None:
        import yaml

        cfg_path = Path(args.config)
        if not cfg_path.exists() and not cfg_path.is_absolute():
            # Convenience: allow `--config neurols_foo.yaml` even if the file lives in PaST/configs
            default_cfg_dir = Path(__file__).resolve().parents[1] / "configs"
            alt = default_cfg_dir / args.config
            if alt.exists():
                cfg_path = alt
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {args.config}. "
                "Pass an explicit path, or place the YAML under PaST/configs/."
            )

        with open(cfg_path) as f:
            config_dict = yaml.safe_load(f)
        config = TrainConfig(**config_dict)
    else:
        config = TrainConfig()

    # Allow CLI to override config file values
    if args.exp_name is not None:
        config.exp_name = args.exp_name
    if args.device is not None:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    if args.n_episodes is not None:
        config.n_episodes = args.n_episodes
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.epsilon_start is not None:
        config.epsilon_start = float(args.epsilon_start)
    if args.epsilon_end is not None:
        config.epsilon_end = float(args.epsilon_end)
    if args.epsilon_decay is not None:
        config.epsilon_decay = float(args.epsilon_decay)
        # Validate now to fail fast.
        if not (0.0 < float(config.epsilon_decay) <= 1.0):
            raise ValueError(
                f"epsilon_decay must be in (0, 1], got {config.epsilon_decay}"
            )
    if args.use_iqn is not None:
        config.use_iqn = args.use_iqn
    if args.action_space is not None:
        config.action_space = args.action_space
    if args.graph_type is not None:
        config.graph_type = args.graph_type
    # Worker count CLI resolution
    n_parallel_cli = args.n_parallel
    if args.n_workers is not None:
        if n_parallel_cli is not None and int(n_parallel_cli) != int(args.n_workers):
            raise ValueError("Conflicting --n-parallel and --n-workers")
        n_parallel_cli = args.n_workers
    if args.n_cores is not None:
        if n_parallel_cli is not None and int(n_parallel_cli) != int(args.n_cores):
            raise ValueError("Conflicting --n-parallel and --n-cores")
        n_parallel_cli = args.n_cores
    if n_parallel_cli is not None:
        config.n_parallel_envs = int(n_parallel_cli)

    # Train
    trainer = NeuroLSTrainer(config)
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
        if args.resume_reset_epsilon is not None:
            trainer.epsilon = float(args.resume_reset_epsilon)
    trainer.train()


if __name__ == "__main__":
    main()

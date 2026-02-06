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

    # Data
    n_jobs_train: List[int] = field(default_factory=lambda: [10, 15, 20])
    n_machines_train: List[int] = field(default_factory=lambda: [2, 3, 4])
    K_range: Tuple[int, int] = (40, 120)  # Range for horizon K
    K_fixed: Optional[int] = 40  # Use a fixed K to keep tensor shapes consistent
    n_instances_per_reset: int = 100  # Train instances to cycle through

    # Environment
    max_steps: int = 500
    stagnation_limit: int = 100
    action_space: str = "AANP"  # AA, AAN, AANP
    reward_mode: str = "improvement"

    # Model — aligned with NeuroLS paper Appendix B
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
    epsilon_decay: float = 0.995

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


def collate_states(states: List[Dict], device: str) -> Dict[str, torch.Tensor]:
    """Collate list of state dicts into batched tensors."""
    keys = states[0].keys()

    batched = {}
    for key in keys:
        arrays = [s[key] for s in states]

        # Stack arrays
        stacked = np.stack(arrays, axis=0)
        batched[key] = torch.tensor(stacked, device=device)

        # Set appropriate dtype
        if key in [
            "job_to_machine",
            "assignment_edges",
            "static_edge_index",
            "dynamic_edge_index",
        ]:
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

        # Logging
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def _init_model(self):
        """Initialize policy and target networks."""
        from PaST.neurols.decoder import NeuroLSPolicy
        from PaST.neurols.action_space import AA_SPACE, AAN_SPACE, AANP_SPACE

        # Get action space size
        space_map = {"AA": AA_SPACE, "AAN": AAN_SPACE, "AANP": AANP_SPACE}
        action_space = space_map.get(self.config.action_space, AANP_SPACE)
        n_actions = action_space.n_actions

        # Create policy network
        self.policy_net = NeuroLSPolicy(
            d_emb=self.config.d_emb,
            n_actions=n_actions,
            n_layers_static=self.config.n_layers_static,
            n_layers_dynamic=self.config.n_layers_dynamic,
            use_iqn=self.config.use_iqn,
            use_dueling=self.config.use_dueling,
            price_mode=self.config.price_mode,
            dropout=self.config.dropout,
        ).to(self.device)

        # Create target network
        self.target_net = NeuroLSPolicy(
            d_emb=self.config.d_emb,
            n_actions=n_actions,
            n_layers_static=self.config.n_layers_static,
            n_layers_dynamic=self.config.n_layers_dynamic,
            use_iqn=self.config.use_iqn,
            use_dueling=self.config.use_dueling,
            price_mode=self.config.price_mode,
            dropout=self.config.dropout,
        ).to(self.device)

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
        """Forward pass for batch of states.

        The GNN encoder operates on single graphs, so we loop over the
        batch and stack the resulting Q-value vectors.
        """
        batch_size = states["job_features"].size(0)
        q_list = []
        has_price = "price_per_hour" in states
        for i in range(batch_size):
            q = net(
                job_features=states["job_features"][i].float(),
                machine_features=states["machine_features"][i].float(),
                state_features=states["state_features"][i].float(),
                job_to_machine=states["job_to_machine"][i].long(),
                static_edge_index=states["static_edge_index"][i].long(),
                dynamic_edge_index=states["dynamic_edge_index"][i].long(),
                price_features=(
                    states["price_per_hour"][i].float() if has_price else None
                ),
            )
            q_list.append(q)  # (A,)
        return torch.stack(q_list, dim=0)  # (B, A)

    def _forward_batch_iqn(
        self,
        net: nn.Module,
        states: Dict[str, torch.Tensor],
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with quantile samples (loops over batch)."""
        batch_size = states["job_features"].size(0)
        q_list = []
        has_price = "price_per_hour" in states
        for i in range(batch_size):
            q = net(
                job_features=states["job_features"][i].float(),
                machine_features=states["machine_features"][i].float(),
                state_features=states["state_features"][i].float(),
                job_to_machine=states["job_to_machine"][i].long(),
                static_edge_index=states["static_edge_index"][i].long(),
                dynamic_edge_index=states["dynamic_edge_index"][i].long(),
                price_features=(
                    states["price_per_hour"][i].float() if has_price else None
                ),
                tau=tau[i : i + 1],  # keep (1, N) shape
            )
            q_list.append(q)  # (N, A) after squeeze
        return torch.stack(q_list, dim=0)  # (B, N, A)

    def train_step(self) -> float:
        """Perform one training step."""
        if len(self.buffer) < self.config.min_buffer_size:
            return 0.0

        # Sample batch
        batch = self.buffer.sample(self.config.batch_size)

        # Compute loss
        loss = self._compute_loss(batch)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)

        self.optimizer.step()

        # Update target network
        if self.config.use_soft_update:
            self._update_target()
        elif self.global_step % self.config.target_update_freq == 0:
            self._update_target()

        self.global_step += 1

        return loss.item()

    def run_episode(self, env, instance, K: int) -> Dict[str, float]:
        """Run one episode and collect experience."""
        state = env.reset(instance, K)
        state_features = env.get_state_features()

        episode_reward = 0.0
        episode_length = 0
        initial_cost = state.current_cost

        while True:
            # Select action
            action = self._get_action(state_features)

            # Step environment
            next_state, reward, done, info = env.step(action)
            next_state_features = env.get_state_features()

            # Store transition
            self.buffer.push(
                state_features,
                action,
                reward,
                next_state_features,
                done,
            )

            # Train
            loss = self.train_step()

            # Update state
            state = next_state
            state_features = next_state_features
            episode_reward += reward
            episode_length += 1

            if done:
                break

        # Compute improvement
        improvement = (initial_cost - state.best_cost) / initial_cost * 100

        return {
            "reward": episode_reward,
            "length": episode_length,
            "improvement": improvement,
            "initial_cost": initial_cost,
            "best_cost": state.best_cost,
            "loss": loss,
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

        # Training loop
        for episode in range(self.config.n_episodes):
            # Select random instance
            instance, K = random.choice(train_instances)

            # Run episode
            metrics = self.run_episode(env, instance, K)

            # Update exploration
            self.epsilon = max(
                self.config.epsilon_end, self.epsilon * self.config.epsilon_decay
            )

            # Track metrics
            self.episode_rewards.append(metrics["reward"])
            self.episode_lengths.append(metrics["length"])
            self.episode_improvements.append(metrics["improvement"])

            # Logging
            if episode % self.config.log_interval == 0:
                avg_reward = np.mean(self.episode_rewards)
                avg_length = np.mean(self.episode_lengths)
                avg_improvement = np.mean(self.episode_improvements)

                self.writer.add_scalar("train/reward", avg_reward, episode)
                self.writer.add_scalar("train/length", avg_length, episode)
                self.writer.add_scalar("train/improvement", avg_improvement, episode)
                self.writer.add_scalar("train/epsilon", self.epsilon, episode)
                self.writer.add_scalar("train/loss", metrics["loss"], episode)
                self.writer.add_scalar("train/buffer_size", len(self.buffer), episode)

                print(
                    f"Episode {episode:5d} | "
                    f"Reward: {avg_reward:8.2f} | "
                    f"Length: {avg_length:6.1f} | "
                    f"Improve: {avg_improvement:5.2f}% | "
                    f"Eps: {self.epsilon:.3f} | "
                    f"Loss: {metrics['loss']:.4f}"
                )

            # Evaluation
            if episode % self.config.eval_interval == 0:
                eval_metrics = self.evaluate(env, train_instances)

                self.writer.add_scalar("eval/reward", eval_metrics["reward"], episode)
                self.writer.add_scalar(
                    "eval/improvement", eval_metrics["improvement"], episode
                )

                print(
                    f"  [EVAL] Reward: {eval_metrics['reward']:.2f} | "
                    f"Improve: {eval_metrics['improvement']:.2f}%"
                )

            # Save checkpoint
            if episode % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_{episode}.pt")

            # Update learning rate
            self.scheduler.step()

            self.episode = episode

        # Final save
        self.save_checkpoint("final.pt")
        self.writer.close()

        print("Training complete!")

    def evaluate(self, env, instances: List[Tuple]) -> Dict[str, float]:
        """Evaluate policy on held-out instances."""
        self.policy_net.eval()

        # Save and override epsilon for greedy evaluation
        saved_epsilon = self.epsilon
        self.epsilon = 0.0

        rewards = []
        improvements = []

        with torch.no_grad():
            for i in range(min(self.config.n_eval_episodes, len(instances))):
                instance, K = instances[i]

                # Run episode with greedy policy (epsilon=0)
                state = env.reset(instance, K)
                state_features = env.get_state_features()

                episode_reward = 0.0
                initial_cost = state.current_cost

                while True:
                    action = self._get_action(state_features)
                    state, reward, done, _ = env.step(action)
                    state_features = env.get_state_features()
                    episode_reward += reward

                    if done:
                        break

                improvement = (initial_cost - state.best_cost) / initial_cost * 100
                rewards.append(episode_reward)
                improvements.append(improvement)

        # Restore epsilon
        self.epsilon = saved_epsilon
        self.policy_net.train()

        return {
            "reward": np.mean(rewards),
            "improvement": np.mean(improvements),
        }

    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        path = self.checkpoint_dir / filename

        torch.save(
            {
                "episode": self.episode,
                "global_step": self.global_step,
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epsilon": self.epsilon,
                "config": asdict(self.config),
            },
            path,
        )

        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

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
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-episodes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--use-iqn",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Enable/disable IQN (distributional RL)",
    )
    parser.add_argument(
        "--action-space", type=str, default=None, choices=["AA", "AAN", "AANP"]
    )

    args = parser.parse_args()

    # Load config from file or create default
    if args.config is not None:
        import yaml

        with open(args.config) as f:
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
    if args.use_iqn is not None:
        config.use_iqn = args.use_iqn
    if args.action_space is not None:
        config.action_space = args.action_space

    # Train
    trainer = NeuroLSTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

"""REINFORCE runner for PaST-SM.

This implements episodic REINFORCE with an optional self-critic baseline.

- Policy gradient uses return-to-go (discounted) per step.
- Self-critic baseline is computed via a greedy rollout on the same instances.

Design goals:
- Keep all compute on-device.
- Be robust to infeasible/all-masked states (treat as terminal for that env).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class ReinforceConfig:
    gamma: float = 0.99
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    use_self_critic: bool = True


class ReinforceRunner:
    def __init__(
        self,
        model: torch.nn.Module,
        env,
        device: torch.device,
        config: ReinforceConfig,
        baseline_env=None,
    ):
        self.model = model
        self.env = env
        self.baseline_env = baseline_env
        self.device = device
        self.config = config

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )

        self.global_step = 0
        self.update_count = 0

    def get_learning_rate(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def set_learning_rate(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "update_count": self.update_count,
        }

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.global_step = int(state.get("global_step", 0))
        self.update_count = int(state.get("update_count", 0))

    @torch.no_grad()
    def _rollout(
        self,
        env,
        batch_data,
        deterministic: bool,
        max_steps: int,
    ) -> Tuple[Dict[str, Tensor], Tensor, Tensor, Tensor]:
        """Run one rollout.

        Returns:
            rewards_t: (T, B)
            dones_t: (T, B)
            log_probs_t: (T, B)  (zeros if deterministic)
            entropy_t: (T, B)
        """
        obs = env.reset(batch_data)
        B = env.batch_size

        rewards_t = []
        dones_t = []
        log_probs_t = []
        entropy_t = []

        done_mask = torch.zeros(B, dtype=torch.bool, device=self.device)

        for _ in range(max_steps):
            if done_mask.all():
                break

            job_mask = obs.get("job_mask")
            period_mask = obs.get("period_mask")
            period_full_mask = obs.get("period_full_mask")

            if job_mask is not None and job_mask.dtype is not torch.bool:
                job_mask = job_mask < 0.5
            if period_mask is not None and period_mask.dtype is not torch.bool:
                period_mask = period_mask < 0.5
            if (
                period_full_mask is not None
                and period_full_mask.dtype is not torch.bool
            ):
                period_full_mask = period_full_mask < 0.5

            periods_full = obs.get("periods_full")
            if (
                getattr(
                    getattr(self.model, "model_config", None),
                    "use_global_horizon",
                    False,
                )
                and periods_full is None
            ):
                periods_full = obs["periods"]
                period_full_mask = period_mask

            logits, _ = self.model(
                jobs=obs["jobs"],
                periods_local=obs["periods"],
                ctx=obs["ctx"],
                job_mask=job_mask,
                period_mask=period_mask,
                periods_full=periods_full,
                period_full_mask=period_full_mask,
            )

            if "action_mask" in obs:
                action_mask = obs["action_mask"]
                logits = logits.masked_fill(action_mask == 0, float("-inf"))

            all_masked = ~torch.isfinite(logits).any(dim=-1)
            newly_infeasible = all_masked & ~done_mask
            if newly_infeasible.any():
                done_mask = done_mask | newly_infeasible
                if hasattr(env, "done_mask"):
                    env.done_mask = env.done_mask | done_mask

            safe_logits = torch.where(
                all_masked.unsqueeze(-1), torch.zeros_like(logits), logits
            )
            dist = torch.distributions.Categorical(logits=safe_logits)

            if deterministic:
                actions = safe_logits.argmax(dim=-1)
                logp = torch.zeros(B, device=self.device)
            else:
                actions = dist.sample()
                logp = dist.log_prob(actions)

            ent = dist.entropy()

            actions = actions.masked_fill(done_mask, 0)
            logp = logp.masked_fill(done_mask, 0.0)
            ent = ent.masked_fill(done_mask, 0.0)

            next_obs, rewards, dones, _ = env.step(actions)

            # record
            rewards_t.append(rewards)
            dones_step = (dones | newly_infeasible).bool()
            dones_t.append(dones_step.float())
            log_probs_t.append(logp)
            entropy_t.append(ent)

            done_mask = done_mask | dones
            if hasattr(env, "done_mask"):
                env.done_mask = env.done_mask | done_mask

            obs = next_obs

        T = len(rewards_t)
        if T == 0:
            zeros = torch.zeros((0, B), device=self.device)
            return obs, zeros, zeros, zeros

        return (
            obs,
            torch.stack(rewards_t, dim=0),
            torch.stack(dones_t, dim=0),
            torch.stack(log_probs_t, dim=0),
            torch.stack(entropy_t, dim=0),
        )

    def _returns_to_go(self, rewards: Tensor, dones: Tensor) -> Tensor:
        """Compute discounted return-to-go per step.

        Args:
            rewards: (T, B)
            dones: (T, B) float 0/1, where 1 indicates termination after that step

        Returns:
            rtg: (T, B)
        """
        T, B = rewards.shape
        rtg = torch.zeros((T, B), device=rewards.device, dtype=rewards.dtype)
        running = torch.zeros((B,), device=rewards.device, dtype=rewards.dtype)

        for t in range(T - 1, -1, -1):
            done = dones[t] > 0.5
            running = rewards[t] + self.config.gamma * running
            running = torch.where(done, rewards[t], running)
            rtg[t] = running

        return rtg

    def update(self, batch_data, max_steps: int) -> Dict[str, float]:
        """Run REINFORCE update on one batch."""
        self.model.train()

        # Stochastic rollout for gradient
        _, rewards, dones, log_probs, entropy = self._rollout(
            env=self.env,
            batch_data=batch_data,
            deterministic=False,
            max_steps=max_steps,
        )

        # Optional self-critic: greedy rollout baseline on same batch
        baseline_rtg = None
        if self.config.use_self_critic:
            if self.baseline_env is None:
                raise ValueError("use_self_critic=True requires baseline_env")
            _, b_rewards, b_dones, _, _ = self._rollout(
                env=self.baseline_env,
                batch_data=batch_data,
                deterministic=True,
                max_steps=max_steps,
            )
            baseline_rtg = self._returns_to_go(b_rewards, b_dones)

        rtg = self._returns_to_go(rewards, dones)

        # Mask out steps beyond termination: when done has occurred earlier in the episode.
        # Build an 'alive' mask: alive[t] indicates the transition at t is part of the episode.
        alive = torch.ones_like(dones, dtype=torch.bool)
        done_cum = torch.zeros((dones.shape[1],), dtype=torch.bool, device=self.device)
        for t in range(dones.shape[0]):
            alive[t] = ~done_cum
            done_cum = done_cum | (dones[t] > 0.5)

        adv = rtg
        if baseline_rtg is not None:
            adv = rtg - baseline_rtg

        adv = adv.detach()

        # Policy gradient loss
        pg = -(adv * log_probs)
        pg = pg.masked_fill(~alive, 0.0)
        policy_loss = pg.sum() / max(1, alive.sum().item())

        ent = entropy.masked_fill(~alive, 0.0)
        ent_mean = ent.sum() / max(1, alive.sum().item())
        loss = policy_loss - self.config.entropy_coef * ent_mean

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()

        # Episode returns (undiscounted) for reporting: sum rewards until done.
        ep_returns = (rewards * alive.float()).sum(dim=0)

        self.update_count += 1
        self.global_step += int(rewards.shape[0]) * int(self.env.batch_size)

        metrics = {
            "train/policy_loss": float(policy_loss.detach().item()),
            "train/entropy": float(ent_mean.detach().item()),
            "train/grad_norm": float(
                grad_norm.item() if isinstance(grad_norm, Tensor) else grad_norm
            ),
            "rollout/rewards_mean": float(ep_returns.mean().item()),
            "rollout/rewards_std": float(ep_returns.std(unbiased=False).item()),
            "rollout/steps": float(rewards.shape[0]),
            "train/self_critic": 1.0 if baseline_rtg is not None else 0.0,
        }

        return metrics

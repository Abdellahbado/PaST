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
        collect_trajectories: bool = False,
    ) -> Tuple[Optional[Dict[str, Tensor]], Tensor, Tensor]:
        """Run one rollout.

        Notes:
            This rollout runs under no-grad. For learning, we collect (obs, actions)
            and later recompute log-probs with gradients in a single batched forward
            pass. This avoids storing a massive autograd graph across time.

        Returns:
            traj: Optional dict of stacked tensors with leading time dim (T, B, ...)
            rewards_t: (T, B)
            dones_t: (T, B) float 0/1, where 1 indicates termination after that step
        """
        obs = env.reset(batch_data)
        B = env.batch_size

        rewards_t = []
        dones_t = []
        actions_t = []

        jobs_t = []
        periods_t = []
        ctx_t = []
        job_mask_t = []
        period_mask_t = []
        periods_full_t = []
        period_full_mask_t = []
        action_mask_t = []

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

            action_mask = obs.get("action_mask")
            if action_mask is not None:
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

            if deterministic:
                actions = safe_logits.argmax(dim=-1)
            else:
                dist = torch.distributions.Categorical(logits=safe_logits)
                actions = dist.sample()

            actions = actions.masked_fill(done_mask, 0)
            next_obs, rewards, dones, _ = env.step(actions)

            if collect_trajectories:
                jobs_t.append(obs["jobs"])
                periods_t.append(obs["periods"])
                ctx_t.append(obs["ctx"])
                job_mask_t.append(job_mask)
                period_mask_t.append(period_mask)
                periods_full_t.append(periods_full)
                period_full_mask_t.append(period_full_mask)
                action_mask_t.append(action_mask)
                actions_t.append(actions)

            rewards_t.append(rewards)
            dones_step = (dones | newly_infeasible).bool()
            dones_t.append(dones_step.float())

            done_mask = done_mask | dones
            if hasattr(env, "done_mask"):
                env.done_mask = env.done_mask | done_mask

            obs = next_obs

        T = len(rewards_t)
        if T == 0:
            zeros = torch.zeros((0, B), device=self.device)
            return None, zeros, zeros

        traj = None
        if collect_trajectories:
            # Helper: stack optional masks. If a key is missing everywhere, return None.
            def _stack_optional(xs):
                if len(xs) == 0 or all(x is None for x in xs):
                    return None
                # Replace None entries with zeros of the right shape/dtype (rare; but makes stacking safe)
                x0 = next(x for x in xs if x is not None)
                filled = [x if x is not None else torch.zeros_like(x0) for x in xs]
                return torch.stack(filled, dim=0)

            traj = {
                "jobs": torch.stack(jobs_t, dim=0),
                "periods": torch.stack(periods_t, dim=0),
                "ctx": torch.stack(ctx_t, dim=0),
                "job_mask": _stack_optional(job_mask_t),
                "period_mask": _stack_optional(period_mask_t),
                "periods_full": _stack_optional(periods_full_t),
                "period_full_mask": _stack_optional(period_full_mask_t),
                "action_mask": _stack_optional(action_mask_t),
                "actions": torch.stack(actions_t, dim=0),
            }

        return traj, torch.stack(rewards_t, dim=0), torch.stack(dones_t, dim=0)

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

        # Rollout under no-grad, but collect (obs, actions) so we can recompute log-probs with grad.
        traj, rewards, dones = self._rollout(
            env=self.env,
            batch_data=batch_data,
            deterministic=False,
            max_steps=max_steps,
            collect_trajectories=True,
        )

        # Optional self-critic: greedy rollout baseline on same batch
        baseline_rtg = None
        if self.config.use_self_critic:
            if self.baseline_env is None:
                raise ValueError("use_self_critic=True requires baseline_env")
            _, b_rewards, b_dones = self._rollout(
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

        if traj is None:
            # Nothing happened; skip update.
            return {
                "train/policy_loss": 0.0,
                "train/entropy": 0.0,
                "train/grad_norm": 0.0,
                "rollout/rewards_mean": 0.0,
                "rollout/rewards_std": 0.0,
                "rollout/steps": 0.0,
                "train/self_critic": 1.0 if baseline_rtg is not None else 0.0,
            }

        # Recompute log-probs and entropy with autograd enabled (single batched forward pass)
        T, B = rewards.shape
        TB = T * B

        def _flat(x: Optional[Tensor]) -> Optional[Tensor]:
            if x is None:
                return None
            return x.reshape((TB,) + x.shape[2:])

        jobs = traj["jobs"].reshape((TB,) + traj["jobs"].shape[2:])
        periods = traj["periods"].reshape((TB,) + traj["periods"].shape[2:])
        ctx = traj["ctx"].reshape((TB,) + traj["ctx"].shape[2:])
        actions = traj["actions"].reshape((TB,))

        job_mask = _flat(traj.get("job_mask"))
        period_mask = _flat(traj.get("period_mask"))
        periods_full = _flat(traj.get("periods_full"))
        period_full_mask = _flat(traj.get("period_full_mask"))

        logits, _ = self.model(
            jobs=jobs,
            periods_local=periods,
            ctx=ctx,
            job_mask=job_mask,
            period_mask=period_mask,
            periods_full=periods_full,
            period_full_mask=period_full_mask,
        )

        action_mask = _flat(traj.get("action_mask"))
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, float("-inf"))

        all_masked = ~torch.isfinite(logits).any(dim=-1)
        safe_logits = torch.where(
            all_masked.unsqueeze(-1), torch.zeros_like(logits), logits
        )
        dist = torch.distributions.Categorical(logits=safe_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        alive_flat = alive.reshape((TB,))
        adv_flat = adv.reshape((TB,)).detach()

        # Policy gradient loss
        pg = -(adv_flat * log_probs)
        pg = pg.masked_fill(~alive_flat, 0.0)
        denom = max(1, int(alive_flat.sum().item()))
        policy_loss = pg.sum() / denom

        ent = entropy.masked_fill(~alive_flat, 0.0)
        ent_mean = ent.sum() / denom
        loss = policy_loss - self.config.entropy_coef * ent_mean

        if not loss.requires_grad:
            raise RuntimeError(
                "REINFORCE loss does not require grad. "
                "This usually means log-probs were computed under no-grad or detached; "
                "ensure log-probs are recomputed from a grad-enabled forward pass."
            )

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

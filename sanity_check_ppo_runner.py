"""Sanity checks for PaST PPO runner + GPU batched env.

Runs on CPU and exercises the previously discussed edge cases:
- all-masked action spaces and repeat-count suppression
- wrapper auto-reset (pulse dones) vs cumulative done_mask detection
- latch clearing across episode boundaries
- minibatch splitting robustness when num_minibatches > batch size
- GAE validity gating (placeholders cannot bootstrap)

Usage:
  /path/to/python PaST/sanity_check_ppo_runner.py

This is intentionally lightweight (no pytest dependency) and prints PASS/FAIL.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional

import os
import sys

# Allow running as a script (python PaST/sanity_check_ppo_runner.py)
_ROOT = os.path.dirname(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as e:
    raise SystemExit(
        "PyTorch is required to run this sanity check.\n"
        "Install a CPU build (example):\n"
        "  pip install torch\n"
        "Or follow the official selector for your OS/Python: https://pytorch.org/get-started/locally/\n"
    ) from e

from PaST.config import DataConfig, EnvConfig
from PaST.config import SlackType
from PaST.sm_benchmark_data import generate_episode_batch
from PaST.sm_env import GPUBatchSingleMachinePeriodEnv
from PaST.ppo_runner import PPORunner, PPOConfig


def _assert(condition: bool, msg: str):
    if not condition:
        raise AssertionError(msg)


def _make_custom_batch(
    *,
    batch_size: int,
    N_job_pad: int,
    K_period_pad: int,
    T_max_pad: int,
    p_lists,
    ct_lists,
    T_limit_list,
    Tk_lists,
    ck_lists,
    period_starts_lists,
    e_single_list,
):
    """Build a minimal batch_data dict compatible with GPUBatchSingleMachinePeriodEnv.reset()."""
    import numpy as np

    batch = {
        "p_subset": np.zeros((batch_size, N_job_pad), dtype=np.int32),
        "n_jobs": np.zeros((batch_size,), dtype=np.int32),
        "job_mask": np.zeros((batch_size, N_job_pad), dtype=np.float32),
        "T_max": np.zeros((batch_size,), dtype=np.int32),
        "T_limit": np.zeros((batch_size,), dtype=np.int32),
        "ct": np.zeros((batch_size, T_max_pad), dtype=np.int32),
        "Tk": np.zeros((batch_size, K_period_pad), dtype=np.int32),
        "ck": np.zeros((batch_size, K_period_pad), dtype=np.int32),
        "period_starts": np.zeros((batch_size, K_period_pad), dtype=np.int32),
        "K": np.zeros((batch_size,), dtype=np.int32),
        "e_single": np.zeros((batch_size,), dtype=np.int32),
    }

    for i in range(batch_size):
        p = list(p_lists[i])
        ct = list(ct_lists[i])
        Tk = list(Tk_lists[i])
        ck = list(ck_lists[i])
        starts = list(period_starts_lists[i])

        n = min(len(p), N_job_pad)
        t_max = min(len(ct), T_max_pad)
        k = min(len(Tk), K_period_pad)

        batch["n_jobs"][i] = n
        batch["T_max"][i] = t_max
        batch["T_limit"][i] = int(T_limit_list[i])
        batch["K"][i] = k
        batch["e_single"][i] = int(e_single_list[i])

        if n:
            batch["p_subset"][i, :n] = np.asarray(p[:n], dtype=np.int32)
            batch["job_mask"][i, :n] = 1.0

        if t_max:
            batch["ct"][i, :t_max] = np.asarray(ct[:t_max], dtype=np.int32)

        if k:
            batch["Tk"][i, :k] = np.asarray(Tk[:k], dtype=np.int32)
            batch["ck"][i, :k] = np.asarray(ck[:k], dtype=np.int32)
            batch["period_starts"][i, :k] = np.asarray(starts[:k], dtype=np.int32)

    return batch


class DummyModel(nn.Module):
    """Minimal model with the same interface as PaSTSMNet.

    It can optionally force env 0 to be all-masked by returning -inf logits.
    """

    def __init__(self, action_dim: int, force_all_mask_env0: bool = True):
        super().__init__()
        self.action_dim = int(action_dim)
        self.force_all_mask_env0 = force_all_mask_env0
        # One learnable parameter so PPO update has gradients.
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        jobs: torch.Tensor,
        periods_local: torch.Tensor,
        ctx: torch.Tensor,
        job_mask: Optional[torch.Tensor] = None,
        period_mask: Optional[torch.Tensor] = None,
        periods_full: Optional[torch.Tensor] = None,
        period_full_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = ctx.shape[0]
        logits = torch.zeros((B, self.action_dim), device=ctx.device) + self.bias
        values = torch.zeros((B, 1), device=ctx.device) + 0.0 * self.bias

        if self.force_all_mask_env0 and B > 0:
            # Make env0 infeasible regardless of action_mask.
            logits[0] = float("-inf")

        return logits, values


class AutoResetAllWrapper:
    """Wrapper that mimics TrainingEnv's full-batch reset behavior.

    - Underlying GPU env returns cumulative done_mask as `dones`.
    - When ANY env becomes newly done, reset the ENTIRE batch and return dones=True for all.
    - Exposes a done_mask attribute like TrainingEnv (which clears on reset).

    This intentionally creates a scenario where `done_mask` can disagree with returned dones.
    """

    def __init__(self, env: GPUBatchSingleMachinePeriodEnv, batch_data_fn):
        self.env = env
        self.batch_size = env.batch_size
        self.device = env.device
        self._batch_data_fn = batch_data_fn
        self.done_mask = torch.zeros(
            self.batch_size, dtype=torch.bool, device=self.device
        )

    def reset(self):
        batch = self._batch_data_fn()
        obs = self.env.reset(batch)
        self.done_mask.zero_()
        return obs

    def step(self, actions: torch.Tensor):
        obs, rewards, dones_cumulative, info = self.env.step(actions)
        newly_done = dones_cumulative & ~self.done_mask

        if newly_done.any():
            obs = self.reset()
            all_done = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
            return obs, rewards, all_done, info

        self.done_mask |= dones_cumulative
        return obs, rewards, newly_done, info


def make_tiny_data_config() -> DataConfig:
    # Make episodes long enough that (without forced all-masked) they won't finish in a short rollout.
    return DataConfig(
        T_max_choices=[200],
        p_min=1,
        p_max=1,
        m_min=1,
        m_max=1,
        n_min=40,
        n_max=40,
        deadline_slack_ratio_min=0.0,
        deadline_slack_ratio_max=0.0,
    )


def make_one_step_done_data_config() -> DataConfig:
    # Each episode has a single job of size 1 and deadline exactly feasible.
    # Any valid action completes the episode in one env.step().
    return DataConfig(
        T_max_choices=[10],
        p_min=1,
        p_max=1,
        m_min=1,
        m_max=1,
        n_min=1,
        n_max=1,
        deadline_slack_ratio_min=0.0,
        deadline_slack_ratio_max=0.0,
    )


def make_env(
    batch_size: int, device: torch.device
) -> Tuple[GPUBatchSingleMachinePeriodEnv, Dict]:
    env_config = EnvConfig()
    # Make the environment deterministic/always-feasible under random action selection:
    # only allow slack=0 so the policy cannot idle itself into a dead-end.
    env_config.slack_type = SlackType.SHORT
    env_config.short_slack_spec.slack_options = [0]
    env = GPUBatchSingleMachinePeriodEnv(
        batch_size=batch_size, env_config=env_config, device=device
    )

    data_config = make_tiny_data_config()
    batch = generate_episode_batch(batch_size=batch_size, config=data_config, seed=123)
    obs = env.reset(batch)
    return env, obs


def test_all_masked_repeat_count_suppressed():
    device = torch.device("cpu")
    batch_size = 4

    env, obs = make_env(batch_size=batch_size, device=device)

    model = DummyModel(action_dim=env.action_dim, force_all_mask_env0=True)
    runner = PPORunner(
        model=model,
        env=env,
        ppo_config=PPOConfig(num_minibatches=64),
        device=device,
        obs_shapes={k: tuple(v.shape[1:]) for k, v in obs.items()},
        rollout_length=8,
    )

    _, metrics = runner.collect_rollout(obs)

    # Only env0 is forced infeasible; with long episodes, others should not finish.
    _assert(metrics["rollout/num_episodes"] == 1, f"Expected 1 episode, got {metrics}")

    # Ensure placeholders did not create non-zero advantages.
    # (Env0 becomes done and stays all-masked; valid rows should be 0 there.)
    env0_valid = runner.buffer.valid[:, 0]
    env0_adv = runner.buffer.advantages[:, 0]
    _assert(
        (env0_valid <= 0.5).any().item(),
        "Expected at least one invalid placeholder for env0",
    )
    _assert(
        (env0_adv[env0_valid <= 0.5].abs() < 1e-6).all().item(),
        "Invalid rows should have ~0 advantage",
    )


def test_wrapper_autoreset_divergence_handling():
    device = torch.device("cpu")
    batch_size = 4

    # Use 1-step episodes so the wrapper resets every step.
    env_config = EnvConfig()
    env_config.slack_type = SlackType.SHORT
    env_config.short_slack_spec.slack_options = [0]
    base_env = GPUBatchSingleMachinePeriodEnv(
        batch_size=batch_size, env_config=env_config, device=device
    )

    data_config = make_one_step_done_data_config()

    def batch_fn():
        return generate_episode_batch(
            batch_size=batch_size, config=data_config, seed=999
        )

    env = AutoResetAllWrapper(base_env, batch_fn)
    obs = env.reset()

    model = DummyModel(action_dim=base_env.action_dim, force_all_mask_env0=False)
    runner = PPORunner(
        model=model,
        env=env,
        ppo_config=PPOConfig(num_minibatches=64),
        device=device,
        obs_shapes={k: tuple(v.shape[1:]) for k, v in obs.items()},
        rollout_length=4,
    )

    _, metrics = runner.collect_rollout(obs)

    # Because env0 forces an episode boundary every step (full reset), we expect one boundary per step,
    # and each boundary applies to ALL envs under this wrapper.
    expected = batch_size * runner.rollout_length
    _assert(
        metrics["rollout/num_episodes"] == expected,
        f"Expected {expected} episodes under full-batch reset, got {metrics}",
    )


def test_minibatch_splitting_and_update_does_not_crash():
    device = torch.device("cpu")
    batch_size = 4

    env, obs = make_env(batch_size=batch_size, device=device)

    # Disable forced infeasible so we get usable rows.
    model = DummyModel(action_dim=env.action_dim, force_all_mask_env0=False)
    runner = PPORunner(
        model=model,
        env=env,
        ppo_config=PPOConfig(ppo_epochs=2, num_minibatches=128),
        device=device,
        obs_shapes={k: tuple(v.shape[1:]) for k, v in obs.items()},
        rollout_length=4,
    )

    obs, _ = runner.collect_rollout(obs)
    metrics = runner.update()

    # Sanity: metrics dict must exist and be finite-ish.
    _assert(
        isinstance(metrics, dict) and len(metrics) > 0, "Expected non-empty metrics"
    )
    _assert(
        torch.isfinite(torch.tensor(list(metrics.values()), dtype=torch.float32))
        .all()
        .item(),
        f"Non-finite metrics: {metrics}",
    )


def test_env_invariants_no_slack_reward_math_and_masks():
    device = torch.device("cpu")

    env_config = EnvConfig()
    env_config.slack_type = SlackType.SHORT
    env_config.short_slack_spec.slack_options = [0]

    env = GPUBatchSingleMachinePeriodEnv(
        batch_size=1, env_config=env_config, device=device
    )

    # Build a tiny deterministic instance with constant price=2 and e_single=3.
    # Jobs are [2, 1, 3]; with slack=0, the schedule is sequential.
    p = [2, 1, 3]
    T_max = 12
    ct = [2] * T_max
    Tk = [T_max]
    ck = [2]
    starts = [0]
    T_limit = sum(p)
    e_single = 3

    batch = _make_custom_batch(
        batch_size=1,
        N_job_pad=env_config.N_job_pad,
        K_period_pad=env.K_pad,
        T_max_pad=env.T_max_pad,
        p_lists=[p],
        ct_lists=[ct],
        T_limit_list=[T_limit],
        Tk_lists=[Tk],
        ck_lists=[ck],
        period_starts_lists=[starts],
        e_single_list=[e_single],
    )

    obs = env.reset(batch)

    total_reward = 0.0
    t_expected = 0

    for step_idx in range(len(p)):
        # action_mask should allow exactly the remaining jobs (slack=0).
        action_mask = obs["action_mask"][0]
        valid_actions = torch.nonzero(action_mask > 0.5).flatten().tolist()
        _assert(
            len(valid_actions) == len(p) - step_idx,
            "Unexpected number of valid actions",
        )

        # Take the first valid action.
        a = torch.tensor([valid_actions[0]], device=device)
        obs2, reward, dones, info = env.step(a)

        # Reward math: reward = - e_single * sum(ct[t:t+p])
        dur = p[step_idx]
        energy_sum = sum(ct[t_expected : t_expected + dur])
        expected_reward = -float(e_single * energy_sum)
        _assert(
            abs(float(reward.item()) - expected_reward) < 1e-6, "Reward/energy mismatch"
        )

        total_reward += float(reward.item())
        t_expected += dur
        obs = obs2

    _assert(env.done_mask[0].item() is True, "Env should be done after all jobs")
    _assert(float(env.job_available.sum().item()) == 0.0, "No jobs should remain")


def test_env_invalid_action_terminates_without_mutation():
    device = torch.device("cpu")

    env_config = EnvConfig()
    env_config.slack_type = SlackType.SHORT
    env_config.short_slack_spec.slack_options = [0]
    env = GPUBatchSingleMachinePeriodEnv(
        batch_size=2, env_config=env_config, device=device
    )

    # Two envs, each has a single job. We'll force env0 to take an invalid padded-job action.
    p_lists = [[1], [1]]
    T_max = 10
    ct_lists = [[1] * T_max, [1] * T_max]
    Tk_lists = [[T_max], [T_max]]
    ck_lists = [[1], [1]]
    starts_lists = [[0], [0]]
    T_limit_list = [1, 1]
    e_single_list = [1, 1]

    batch = _make_custom_batch(
        batch_size=2,
        N_job_pad=env_config.N_job_pad,
        K_period_pad=env.K_pad,
        T_max_pad=env.T_max_pad,
        p_lists=p_lists,
        ct_lists=ct_lists,
        T_limit_list=T_limit_list,
        Tk_lists=Tk_lists,
        ck_lists=ck_lists,
        period_starts_lists=starts_lists,
        e_single_list=e_single_list,
    )

    env.reset(batch)
    t_before = env.t.clone()
    jobs_before = env.job_available.clone()

    invalid_job_id = 5  # padded job (unavailable)
    valid_job_id = 0
    actions = torch.tensor([invalid_job_id, valid_job_id], device=device)

    _, rewards, dones, _ = env.step(actions)

    _assert(env.done_mask[0].item() is True, "Invalid-action env should be marked done")
    _assert(
        float(rewards[0].item()) == 0.0, "Invalid-action env should have zero reward"
    )
    _assert(
        int(env.t[0].item()) == int(t_before[0].item()),
        "Invalid-action env time must not advance",
    )
    _assert(
        float(env.job_available[0].sum().item()) == 0.0,
        "Invalid-action env should have no available jobs",
    )

    # Valid env should progress and complete its only job.
    _assert(env.done_mask[1].item() is True, "Valid env should complete the single job")
    _assert(float(rewards[1].item()) < 0.0, "Valid env should have negative reward")
    _assert(
        int(env.t[1].item()) == int(t_before[1].item()) + 1,
        "Valid env time should advance by 1",
    )
    _assert(
        float(env.job_available[1].sum().item()) == 0.0,
        "Valid env should have no remaining jobs",
    )


def main():
    tests = [
        (
            "all_masked repeat counting suppressed",
            test_all_masked_repeat_count_suppressed,
        ),
        (
            "wrapper auto-reset divergence handling",
            test_wrapper_autoreset_divergence_handling,
        ),
        (
            "minibatch splitting + update",
            test_minibatch_splitting_and_update_does_not_crash,
        ),
        (
            "env invariants (no slack) reward+mask",
            test_env_invariants_no_slack_reward_math_and_masks,
        ),
        (
            "env invalid action terminates",
            test_env_invalid_action_terminates_without_mutation,
        ),
    ]

    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS: {name}")
        except Exception as e:
            failures += 1
            print(f"FAIL: {name}: {e}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

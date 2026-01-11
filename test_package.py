"""
Test script for PaST-SM package.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np

print("=" * 60)
print("PaST-SM Package Test")
print("=" * 60)

# Test 1: Config
print("\n1. Testing config module...")
from PaST.config import PaSTConfig, SlackVariant, ShortSlackSpec, DataConfig, EnvConfig

config = PaSTConfig()
print(f"   Slack variant: {config.env.slack_variant}")
print(f"   K_period_lookahead: {config.env.K_period_lookahead}")
print(f"   Num slack choices: {config.env.get_num_slack_choices()}")

# Test SHORT_SLACK variant
config.env.slack_variant = SlackVariant.SHORT_SLACK
config.env.short_slack_spec = ShortSlackSpec(slack_options=[0, 1, 2, 3, 5, 10])
print(f"   SHORT_SLACK num choices: {config.env.get_num_slack_choices()}")

# Test 2: Data generation
print("\n2. Testing data generation...")
from PaST.sm_benchmark_data import (
    generate_raw_instance,
    generate_single_machine_episode,
    generate_episode_batch,
    DataConfig,
)

data_config = DataConfig()
rng = random.Random(42)

# Raw instance
raw = generate_raw_instance(data_config, rng, instance_id=1)
print(f"   Raw instance: m={raw.m}, n={raw.n}, T_max={raw.T_max}")
print(f"   Periods: K={len(raw.Tk)}")

# Single machine episode
episode = generate_single_machine_episode(data_config, rng)
print(
    f"   Episode: n_jobs={episode.n_jobs}, T_limit={episode.T_limit}, T_min={episode.T_min}"
)

# Batch
batch = generate_episode_batch(batch_size=8, config=data_config, seed=42)
print(f"   Batch p_subset shape: {batch['p_subset'].shape}")
print(f"   Batch n_jobs: {batch['n_jobs']}")

# Test 3: Environment
print("\n3. Testing environment...")
from PaST.sm_env import SingleMachinePeriodEnv

env_config = EnvConfig()  # Default NO_SLACK
env = SingleMachinePeriodEnv(env_config, data_config)

episode = generate_single_machine_episode(data_config, random.Random(123))
obs, info = env.reset(options={"episode": episode})

print(f"   Obs keys: {list(obs.keys())}")
print(f"   Jobs shape: {obs['jobs'].shape}")
print(f"   Periods shape: {obs['periods'].shape}")
print(f"   Action dim: {env.action_dim}")

# Take a step
valid_actions = np.where(obs["action_mask"] > 0)[0]
if len(valid_actions) > 0:
    action = valid_actions[0]
    obs2, reward, done, truncated, info2 = env.step(action)
    print(f"   Step result: reward={reward:.2f}, done={done}")

# Test 4: Model
print("\n4. Testing model...")
import torch
from PaST.past_sm_model import PaSTSMNet
from PaST.config import ModelConfig

model_config = ModelConfig()
env_config = EnvConfig()  # NO_SLACK
model = PaSTSMNet(model_config, env_config)

B = 4
jobs = torch.randn(B, env_config.N_job_pad, env_config.F_job)
periods = torch.randn(B, env_config.K_period_lookahead, env_config.F_period)
ctx = torch.randn(B, env_config.F_ctx)
job_mask = torch.ones(B, env_config.N_job_pad)
action_mask = torch.ones(B, env_config.N_job_pad * env_config.get_num_slack_choices())

probs, logits = model(jobs, periods, ctx, job_mask, action_mask)
print(f"   Model output: probs shape={probs.shape}")
print(f"   Probs sum: {probs.sum(dim=-1)}")

actions, log_probs = model.sample_action(probs)
print(f"   Sampled actions: {actions}")

# Test 5: SHORT_SLACK model
print("\n5. Testing SHORT_SLACK model...")
env_config2 = EnvConfig()
env_config2.slack_variant = SlackVariant.SHORT_SLACK
env_config2.short_slack_spec = ShortSlackSpec(slack_options=[0, 1, 2, 5])
model2 = PaSTSMNet(model_config, env_config2)
A2 = env_config2.N_job_pad * env_config2.get_num_slack_choices()
print(f"   SHORT_SLACK action dim: {A2}")
print(f"   Model parameters: {sum(p.numel() for p in model2.parameters()):,}")

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)

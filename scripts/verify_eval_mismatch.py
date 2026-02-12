
"""Verify if Evaluation Determinism is masking agent performance.

This script compares the agent's performance under two modes:
1. Deterministic (Standard Eval): strict hill-climbing (rejects worsening moves).
2. Stochastic (Training Mode): simulated annealing (accepts worsening moves).

If Mode 2 >> Mode 1, it confirms the agent has learned to use "worsening" moves
(like Balance/Exposure criteria) to escape local optima, but the standard evaluation
is confusing this with failure.
"""
import sys
import os
import torch
import numpy as np
import random
from pathlib import Path
from dataclasses import asdict

# Add repo root to path
sys.path.append(os.getcwd())

from PaST.neurols.train import TrainConfig, NeuroLSTrainer
from PaST.neurols.env import NeuroLSEnv, EnvConfig
from PaST.data.sm_benchmark_data import generate_raw_instance
from PaST.config import DataConfig

def run_eval_mode(trainer, env, instances, deterministic: bool, desc: str):
    print(f"\n--- Running {desc} (deterministic={deterministic}) ---")
    
    # Override env config
    env.config.deterministic = deterministic
    if not deterministic:
        # Reset temperature for SA
        env.temperature = env.config.initial_temp
        
    # Run evaluation
    metrics = trainer.evaluate(env, instances)
    
    print(f"  Improvement: {metrics['improvement']:.2f}%")
    print(f"  Random Imp:  {metrics['random_improvement']:.2f}%")
    print(f"  Advantage:   {metrics['advantage_over_random']:.2f}pp")
    return metrics

def main():
    # Configuration
    checkpoint_path = "checkpoints/neurols/neurols_NMCP_small/checkpoint_4000.pt"
    config_path = "configs/neurols_NMCP.yaml"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        # Try to find any checkpoint
        import glob
        cpts = glob.glob("checkpoints/neurols/neurols_NMCP_small/checkpoint_*.pt")
        if cpts:
            checkpoint_path = sorted(cpts)[-1]
            print(f"Using alternative checkpoint: {checkpoint_path}")
        else:
            print("No checkpoints found. Exiting.")
            return

    # Load Config (roughly, by instantiating default and patching)
    # Ideally we load the yaml, but for this snippet we'll just set the key params 
    # based on the file we read earlier.
    from PaST.neurols.train import TrainConfig
    import yaml
    
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    
    # Hacky config loading
    config = TrainConfig()
    for k, v in cfg_dict.items():
        if hasattr(config, k):
            setattr(config, k, v)
    
    # Ensure correct action space is loaded (str -> class handled in Trainer)
    config.action_space = "NMCP" 
    
    # Initialize Trainer & Load Checkpoint
    trainer = NeuroLSTrainer(config)
    print(f"Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=trainer.device)
    trainer.policy_net.load_state_dict(state["policy_net"])
    trainer.policy_net.eval()
    
    # Setup Env
    env_config = EnvConfig(
        max_steps=config.max_steps,
        action_space=config.action_space,
        reward_mode=config.reward_mode,
        job_price_features=config.job_price_features,
        price_mode=config.price_mode,
        graph_type=config.graph_type,
        # SA params (defaults)
        initial_temp=1.0, 
        cooling_rate=0.995,
        use_acceptance=True
    )
    env = NeuroLSEnv(env_config)
    env.seed(42)
    
    # Generate a few instances
    data_config = DataConfig(sampling_mode="new_benchmark_grid")
    rng = random.Random(42)
    instances = []
    print("Generating validation instances...")
    for i in range(5): # Small number for quick check
        inst = generate_raw_instance(data_config, rng, instance_id=1000+i, n=90, m=9, D_days=10)
        instances.append((inst, 200)) # K=200
        
    # 1. Standard Eval (Strict Greedy)
    # This matches train.py's evaluation
    run_eval_mode(trainer, env, instances, deterministic=True, desc="Standard Eval (Strict Hill-Climbing)")
    
    # 2. SA Eval (Training Logic)
    # This allows the agent to use the strategies it learned
    run_eval_mode(trainer, env, instances, deterministic=False, desc="Training Mode Eval (Simulated Annealing)")

if __name__ == "__main__":
    main()

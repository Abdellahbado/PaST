
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.getcwd())

from PaST.alns_rl.pm_alns_env import PMALNSEnv
from PaST.solvers.alns_parallel import ALNSConfig

def check_feature_scaling():
    print("Initializing Environment...")
    cfg = ALNSConfig(max_iters=10, no_improve_limit=10)
    
    # Use the user's settings: medium scale, energy_aware
    env = PMALNSEnv(
        scale="medium", 
        seed=42, 
        instance_id=0, 
        slack_ratio=0.25, 
        alns_cfg=cfg, 
        action_set="energy_aware", 
        state_variant="energy_aware"
    )
    
    print("Resetting Environment...")
    # This triggers _compute_energy_aware_instance_features
    obs = env.reset()
    
    # Extract features
    # Base: 10
    # Destroy Freqs: 8
    # Repair Freqs: 2
    # Total Base + Freqs = 20
    # Extra Feats start at index 20.
    
    feat_names = [
        "logm", "logn", "logT", 
        "e_min", "e_mean", "e_max", "e_cv",
        "ck_min", "ck_mean", "ck_max", "ck_cv",
        "K_norm", "Tk_mean_norm", "Tk_cv",
        "cheap_frac", "expensive_frac",
        "p_mean_norm", "p_cv"
    ]
    
    # Raw extra features are stored in env._inst_extra_feats
    raw_extra = env._inst_extra_feats
    
    print("\n--- Feature Analysis ---")
    print(f"Number of extra features: {len(raw_extra)}")
    if len(raw_extra) != len(feat_names):
        print(f"WARNING: Name mismatch. Expected {len(feat_names)}, got {len(raw_extra)}")
    
    print(f"{'Feature':<15} | {'Raw Value':<12} | {'Norm (tanh(x/10))':<18} | {'Status':<10}")
    print("-" * 65)
    
    for name, val in zip(feat_names, raw_extra):
        norm_val = np.tanh(val / 10.0)
        status = "OK"
        if abs(norm_val) > 0.99:
            status = "SATURATED"
        elif abs(norm_val) < 0.001:
            status = "VANISHED"
            
        print(f"{name:<15} | {val:12.4f} | {norm_val:18.4f} | {status:<10}")

    # Check Dynamic Feats
    # These are at the very end of the obs.
    # obs length should be 20 + 18 + 2 = 40.
    print(f"\nObservation Shape: {obs.shape}")
    
    # Dynamic feats are the last 2
    dyn_raw = obs[-2:] # Actually obs is already normalized?
    # Wait, env.reset() returns normalized obs if env wrapper does it?
    # PMALNSEnv.reset() returns self._obs().
    # PMALNSEnv._obs() returns raw concatenation? 
    # train_ppo_alns_pm.py applies normalization explicitly.
    # So env.reset() returns RAW observation.
    
    print("\n--- Dynamic Features (Initial) ---")
    dyn_names = ["wmean_norm", "corr"]
    dyn_vals = obs[-2:]
    
    for name, val in zip(dyn_names, dyn_vals):
        norm_val = np.tanh(val / 10.0)
        print(f"{name:<15} | {val:12.4f} | {norm_val:18.4f}")

if __name__ == "__main__":
    check_feature_scaling()

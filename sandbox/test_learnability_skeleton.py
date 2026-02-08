from .neurols.env import NeuroLSEnv, EnvConfig
from .neurols.action_space import get_action_space
from PaST.data.sm_benchmark_data import generate_raw_instance
from PaST.config import DataConfig
import numpy as np
import random
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional, Sequence

# Copy relevant functions from task_learnability_test.py but adapt imports
# to use the sandbox neurols

def _vectorize_features(features: Dict[str, np.ndarray]) -> np.ndarray:
    """Turn the variable-structured observation into a fixed-size vector."""
    vec_parts: List[np.ndarray] = []

    def _mean_std(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return np.zeros((0,), dtype=np.float32)
        if x.ndim == 1:
            return np.concatenate([x, np.zeros_like(x)], axis=0)
        mu = x.mean(axis=0)
        sd = x.std(axis=0)
        return np.concatenate([mu, sd], axis=0)

    vec_parts.append(np.asarray(features["state_features"], dtype=np.float32).ravel())
    vec_parts.append(_mean_std(features["job_features"]))
    vec_parts.append(_mean_std(features["machine_features"]))
    if "price_per_hour" in features:
        vec_parts.append(_mean_std(features["price_per_hour"]))
    if "machine_exposure" in features:
        vec_parts.append(_mean_std(features["machine_exposure"]))
    if "period_features" in features:
        vec_parts.append(_mean_std(features["period_features"]))

    return np.concatenate(vec_parts, axis=0).astype(np.float32)

def _oracle_one_step(env, *, rng: random.Random) -> Tuple[int, Dict[str, float]]:
    # ... (Implementation of oracle using sandbox env) ...
    # Simplified for brevity, assumes env logic is similar
    # We need to reimplement _oracle_one_step because it depends on internal env details
    # which might have changed or need to be consistent.
    
    # For now, let's use a simpler heuristic oracle:
    # 1. Generate all legal actions
    # 2. Evaluate them
    # 3. Pick best
    
    n_actions = env.action_space_size
    best_action = 0
    best_reward = -float('inf')
    action_rewards = []
    
    # Snapshot env
    snap = env._snapshot_env(env) if hasattr(env, "_snapshot_env") else None
    # If _snapshot_env not available in sandbox yet (I didn't copy it), we need to implement it or use what's there.
    # The original script had _snapshot_env as a standalone function.
    
    # To avoid complex env snapshotting in this script, let's just rely on the fact that
    # the env in the original script supported it.
    # Since I copied 'env.py' from original, it DOES NOT have _snapshot_env inside the class, 
    # it was a helper in the script. I need to copy that helper too or import it.
    
    pass

# ... (rest of the script) ...

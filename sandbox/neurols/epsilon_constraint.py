"""Epsilon-constraint K-sweep driver for NeuroLS.

Produces the energy–makespan Pareto frontier by training/evaluating
across a grid of K values (horizon constraints).

For each K_i the DP inner loop enforces makespan ≤ K_i, and the RL
agent optimises energy.  Sweeping K from K_min → K_max traces the
trade-off curve.

Usage:
    # Evaluate a single checkpoint across K values
    python -m PaST.neurols.epsilon_constraint \
        --checkpoint checkpoints/neurols/best.pt \
        --k-values 40 50 60 80 100 120 \
        --n-eval 20

    # Sweep with training at each K
    python -m PaST.neurols.epsilon_constraint \
        --config configs/neurols_AANP_full.yaml \
        --k-values 40 60 80 100 120 \
        --mode train \
        --n-episodes 2000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def evaluate_at_K(
    checkpoint_path: str,
    K: int,
    n_eval: int = 20,
    seed: int = 42,
    device: str = "cpu",
    config_overrides: Optional[Dict] = None,
) -> Dict[str, float]:
    """Evaluate a trained NeuroLS checkpoint at a specific K value.

    Returns:
        Dict with keys:
            energy_mean, energy_std, energy_min,
            makespan_mean, makespan_std, makespan_max,
            improvement_mean, time_per_instance
    """
    import random
    import torch
    from sandbox.neurols.env import NeuroLSEnv, EnvConfig
    from sandbox.neurols.decoder import NeuroLSPolicy
    from sandbox.neurols.train import TrainConfig
    from PaST.data.sm_benchmark_data import generate_raw_instance
    from PaST.config import DataConfig

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    train_cfg = ckpt.get("config", {})
    if config_overrides:
        train_cfg.update(config_overrides)

    # Reconstruct model
    from sandbox.neurols.action_space import get_action_space

    action_space = get_action_space(train_cfg.get("action_space", "AANP"))
    graph_type = train_cfg.get("graph_type", "bipartite")

    model = NeuroLSPolicy(
        d_emb=train_cfg.get("d_emb", 128),
        n_actions=action_space.n_actions,
        n_layers_static=train_cfg.get("n_layers_static", 3),
        n_layers_dynamic=train_cfg.get("n_layers_dynamic", 2),
        use_iqn=train_cfg.get("use_iqn", True),
        use_dueling=train_cfg.get("use_dueling", False),
        price_mode=train_cfg.get("price_mode", "full"),
        dropout=train_cfg.get("dropout", 0.1),
        graph_type=graph_type,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)

    # Create env
    env_config = EnvConfig(
        max_steps=train_cfg.get("max_steps", 500),
        stagnation_limit=train_cfg.get("stagnation_limit", 100),
        action_space=train_cfg.get("action_space", "AANP"),
        graph_type=graph_type,
        price_mode=train_cfg.get("price_mode", "z_price"),
    )
    env = NeuroLSEnv(env_config)

    # Generate evaluation instances
    data_config = DataConfig(sampling_mode="new_benchmark_grid")
    eval_rng = random.Random(seed)
    hours_per_day = int(getattr(data_config, "hours_per_day", 20))

    fixed_n = train_cfg.get("n_jobs_train", [10])[0]
    fixed_m = train_cfg.get("n_machines_train", [3])[0]
    D_days = max(1, K // hours_per_day)

    energies = []
    makespans = []
    improvements = []
    times = []

    for i in range(n_eval):
        instance = generate_raw_instance(
            config=data_config,
            rng=eval_rng,
            instance_id=10000 + i,
            n=fixed_n,
            m=fixed_m,
            D_days=D_days,
        )

        t0 = time.time()
        state = env.reset(instance, K)
        features = env.get_state_features()
        initial_cost = state.current_cost

        while True:
            with torch.no_grad():
                st = {k: torch.tensor(v, device=device) for k, v in features.items()}
                q = model(
                    job_features=st["job_features"].float(),
                    machine_features=st["machine_features"].float(),
                    state_features=st["state_features"].float(),
                    job_to_machine=st["job_to_machine"].long(),
                    static_edge_index=st["static_edge_index"].long(),
                    dynamic_edge_index=st["dynamic_edge_index"].long(),
                    price_features=st.get("price_per_hour", None),
                    machine_exposure=(
                        st["machine_exposure"].float()
                        if "machine_exposure" in st
                        else None
                    ),
                    period_features=(
                        st["period_features"].float()
                        if "period_features" in st
                        else None
                    ),
                    tripartite_edge_index=(
                        st["tripartite_edge_index"].long()
                        if "tripartite_edge_index" in st
                        else None
                    ),
                )
                action = q.argmax().item()

            _, _, done, _ = env.step(action)
            features = env.get_state_features()

            if done:
                break

        elapsed = time.time() - t0
        energies.append(state.best_cost)
        makespans.append(state.makespan)
        improvements.append((initial_cost - state.best_cost) / initial_cost * 100)
        times.append(elapsed)

    return {
        "K": K,
        "energy_mean": float(np.mean(energies)),
        "energy_std": float(np.std(energies)),
        "energy_min": float(np.min(energies)),
        "makespan_mean": float(np.mean(makespans)),
        "makespan_std": float(np.std(makespans)),
        "makespan_max": float(np.max(makespans)),
        "improvement_mean": float(np.mean(improvements)),
        "time_per_instance": float(np.mean(times)),
        "n_eval": n_eval,
    }


def train_at_K(
    base_config_path: str,
    K: int,
    n_episodes: int = 2000,
    output_dir: str = "checkpoints/neurols/epsilon_sweep",
) -> Tuple[str, Dict[str, float]]:
    """Train a NeuroLS agent at a specific K, then evaluate.

    Returns:
        (checkpoint_path, eval_results)
    """
    import yaml
    from sandbox.neurols.train import TrainConfig, NeuroLSTrainer

    with open(base_config_path) as f:
        config_dict = yaml.safe_load(f)

    config_dict["K_fixed"] = K
    config_dict["n_episodes"] = n_episodes
    config_dict["exp_name"] = f"{config_dict.get('exp_name', 'neurols')}_K{K}"
    config_dict["checkpoint_dir"] = output_dir

    config = TrainConfig(**config_dict)
    trainer = NeuroLSTrainer(config)
    trainer.train()

    # Find best checkpoint
    ckpt_dir = Path(output_dir) / config.exp_name
    candidates = list(ckpt_dir.glob("best*.pt")) + list(ckpt_dir.glob("final*.pt"))
    if not candidates:
        candidates = list(ckpt_dir.glob("*.pt"))
    best_ckpt = str(sorted(candidates, key=lambda p: p.stat().st_mtime)[-1])

    # Evaluate
    results = evaluate_at_K(best_ckpt, K, n_eval=20, device=config.device)
    return best_ckpt, results


def run_k_sweep(
    k_values: List[int],
    mode: str = "eval",
    checkpoint: Optional[str] = None,
    config_path: Optional[str] = None,
    n_eval: int = 20,
    n_episodes: int = 2000,
    output_dir: str = "analysis_out/epsilon_constraint",
    device: str = "cpu",
) -> List[Dict[str, float]]:
    """Run epsilon-constraint sweep over K values.

    Args:
        k_values: List of K (horizon) values to sweep
        mode: "eval" (evaluate existing checkpoint) or "train" (train per K)
        checkpoint: Path to pre-trained checkpoint (for eval mode)
        config_path: Path to training config (for train mode)
        n_eval: Number of evaluation instances per K
        n_episodes: Training episodes per K (for train mode)
        output_dir: Output directory for results
        device: "cpu" or "cuda"

    Returns:
        List of result dicts, one per K value
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    print(f"═══ Epsilon-Constraint K-Sweep ═══")
    print(f"  K values: {k_values}")
    print(f"  Mode: {mode}")
    print(f"  Output: {output_dir}")
    print()

    for K in sorted(k_values):
        print(f"── K = {K} ──")
        t0 = time.time()

        if mode == "train":
            if config_path is None:
                raise ValueError("--config required for train mode")
            ckpt_path, result = train_at_K(
                config_path,
                K,
                n_episodes=n_episodes,
                output_dir=os.path.join(output_dir, "checkpoints"),
            )
            result["checkpoint"] = ckpt_path
        else:
            if checkpoint is None:
                raise ValueError("--checkpoint required for eval mode")
            result = evaluate_at_K(
                checkpoint,
                K,
                n_eval=n_eval,
                device=device,
            )

        result["sweep_time"] = time.time() - t0
        results.append(result)

        print(f"  Energy: {result['energy_mean']:.2f} ± {result['energy_std']:.2f}")
        print(
            f"  Makespan: {result['makespan_mean']:.1f} ± {result['makespan_std']:.1f}"
        )
        print(f"  Improvement: {result['improvement_mean']:.2f}%")
        print(f"  Time: {result['sweep_time']:.1f}s")
        print()

    # Save results
    out_path = os.path.join(output_dir, "epsilon_constraint_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

    # Print Pareto frontier summary
    print("\n═══ Pareto Frontier Summary ═══")
    print(f"{'K':>6} {'Energy':>12} {'Makespan':>10} {'Impr%':>8}")
    print("─" * 40)
    for r in results:
        print(
            f"{r['K']:>6} {r['energy_mean']:>12.2f} "
            f"{r['makespan_mean']:>10.1f} {r['improvement_mean']:>8.2f}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Epsilon-constraint K-sweep for NeuroLS"
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        required=True,
        help="List of K (horizon) values to sweep",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="eval",
        choices=["eval", "train"],
        help="'eval' = evaluate existing checkpoint; 'train' = train per K",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Pre-trained checkpoint path (for eval mode)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Training config YAML path (for train mode)",
    )
    parser.add_argument("--n-eval", type=int, default=20)
    parser.add_argument("--n-episodes", type=int, default=2000)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_out/epsilon_constraint",
    )
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    run_k_sweep(
        k_values=args.k_values,
        mode=args.mode,
        checkpoint=args.checkpoint,
        config_path=args.config,
        n_eval=args.n_eval,
        n_episodes=args.n_episodes,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()

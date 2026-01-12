"""Train PaST-SM with REINFORCE.

This module exists so you can run true REINFORCE from the command line:

  python -m PaST.train_reinforce --variant_id reinforce_short_sc --device cuda --num_envs 128 --rollout_length 64 --total_env_steps 300000

It shares configs, evaluation, logging, and checkpointing with train_ppo.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from PaST.config import VariantID, get_variant_config, RLAlgorithm
from PaST.past_sm_model import build_model
from PaST.sm_benchmark_data import generate_episode_batch
from PaST.sm_env import GPUBatchSingleMachinePeriodEnv
from PaST.eval import Evaluator
from PaST.train_ppo import (
    RunConfig,
    MetricsLogger,
    CheckpointManager,
    set_seed,
    get_rng_states,
    set_rng_states,
)
from PaST.reinforce_runner import ReinforceRunner, ReinforceConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="PaST-SM REINFORCE Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--variant_id", type=str, default="reinforce_short_sc")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="runs")

    # Evaluation seeding (for fair cross-variant comparison)
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=None,
        help="Base seed for evaluation instance generation (overrides training-seed-derived eval seeds)",
    )
    parser.add_argument(
        "--eval_seed_mode",
        type=str,
        default=None,
        choices=["fixed", "per_update"],
        help="How to vary evaluation seeds across eval calls when --eval_seed is set",
    )

    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--rollout_length", type=int, default=128)
    parser.add_argument("--total_env_steps", type=int, default=10_000_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)

    return parser.parse_args()


def main():
    args = parse_args()

    # Load or create run config (align behavior with train_ppo)
    if args.config:
        run_config = RunConfig.from_yaml(args.config)
    else:
        run_config = RunConfig()

    # Apply CLI overrides
    run_config.variant_id = args.variant_id
    run_config.seed = args.seed
    run_config.num_envs = args.num_envs
    run_config.rollout_length = args.rollout_length
    run_config.total_env_steps = args.total_env_steps
    run_config.learning_rate = args.learning_rate
    run_config.device = args.device
    run_config.output_dir = args.output_dir

    if args.eval_seed is not None:
        run_config.eval_seed = int(args.eval_seed)
    if args.eval_seed_mode is not None:
        run_config.eval_seed_mode = str(args.eval_seed_mode)

    device = torch.device(run_config.device)
    set_seed(run_config.seed)

    variant_id = VariantID(run_config.variant_id)
    variant_config = get_variant_config(variant_id, seed=run_config.seed)
    assert (
        variant_config.training.algorithm == RLAlgorithm.REINFORCE
    ), f"Variant {variant_id.value} is not REINFORCE"

    run_dir = (
        Path(run_config.output_dir) / run_config.variant_id / f"seed_{run_config.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config.to_yaml(run_dir / "run_config.yaml")
    with open(run_dir / "variant_config.json", "w") as f:
        json.dump(variant_config.to_dict(), f, indent=2)

    logger = MetricsLogger(run_dir, run_dir.name)
    checkpoint_mgr = CheckpointManager(
        run_dir / "checkpoints",
        num_milestones=run_config.num_milestone_checkpoints,
    )

    print("=" * 80)
    print("PaST-SM REINFORCE Training")
    print("=" * 80)
    print(f"Variant: {run_config.variant_id}")
    print(f"Seed: {run_config.seed}")
    print(f"Device: {device}")
    print(f"Num envs: {run_config.num_envs}")
    print(f"Max steps per episode: {run_config.rollout_length}")
    print(f"Total env steps: {run_config.total_env_steps:,}")
    print(f"Output: {run_dir}")
    print("=" * 80)

    # Create envs
    env_config = variant_config.env
    train_env = GPUBatchSingleMachinePeriodEnv(
        batch_size=run_config.num_envs,
        env_config=env_config,
        device=device,
    )
    baseline_env = None
    if variant_config.training.use_self_critic:
        baseline_env = GPUBatchSingleMachinePeriodEnv(
            batch_size=run_config.num_envs,
            env_config=env_config,
            device=device,
        )

    model = build_model(variant_config).to(device)

    runner = ReinforceRunner(
        model=model,
        env=train_env,
        baseline_env=baseline_env,
        device=device,
        config=ReinforceConfig(
            gamma=run_config.gamma,
            entropy_coef=run_config.entropy_coef,
            learning_rate=run_config.learning_rate,
            max_grad_norm=run_config.max_grad_norm,
            use_self_critic=variant_config.training.use_self_critic,
        ),
    )

    eval_env = GPUBatchSingleMachinePeriodEnv(
        batch_size=run_config.num_eval_instances,
        env_config=env_config,
        device=device,
    )
    evaluator = Evaluator(model=model, env=eval_env, device=device)

    start_update = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        runner.load_state_dict(checkpoint["runner"])
        set_rng_states(checkpoint["rng_states"])
        start_update = runner.update_count
        checkpoint_mgr.load_best_energy_from_checkpoint(device)

    update = start_update
    while runner.global_step < run_config.total_env_steps:
        update_start = time.time()

        batch_data = generate_episode_batch(
            batch_size=run_config.num_envs,
            config=variant_config.data,
            seed=run_config.seed + update,
        )

        metrics = runner.update(
            batch_data=batch_data, max_steps=run_config.rollout_length
        )
        metrics["train/lr"] = runner.get_learning_rate()
        # Provide PPO-only keys expected by MetricsLogger output format.
        metrics.setdefault("train/value_loss", 0.0)
        metrics.setdefault("train/approx_kl", 0.0)
        metrics.setdefault("train/clip_frac", 0.0)

        # Optional eval using evaluator (deterministic)
        if update % run_config.eval_every_updates == 0 and update > 0:
            if run_config.eval_seed is None:
                eval_seed = run_config.seed + update + 12345
            else:
                if run_config.eval_seed_mode == "fixed":
                    eval_seed = run_config.eval_seed
                elif run_config.eval_seed_mode == "per_update":
                    eval_seed = run_config.eval_seed + update
                else:
                    raise ValueError(
                        f"Unknown eval_seed_mode: {run_config.eval_seed_mode}"
                    )

            eval_batch = generate_episode_batch(
                batch_size=run_config.num_eval_instances,
                config=variant_config.data,
                seed=eval_seed,
            )
            eval_result, _ = evaluator.evaluate(eval_batch, deterministic=True)
            metrics.update(eval_result.to_dict())
            checkpoint_mgr.save_best(
                runner, get_rng_states(), run_config, variant_config, eval_result
            )

        logger.log(update, runner.global_step, metrics)

        if checkpoint_mgr.should_save_latest(
            update,
            run_config.save_latest_every_updates,
            run_config.save_latest_every_minutes,
        ):
            checkpoint_mgr.save_latest(
                runner, get_rng_states(), run_config, variant_config
            )

        if checkpoint_mgr.should_save_milestone(update, max(1, run_config.num_updates)):
            checkpoint_mgr.save_milestone(
                runner,
                get_rng_states(),
                run_config,
                variant_config,
                update,
                max(1, run_config.num_updates),
            )

        update += 1

    print("Training complete")


if __name__ == "__main__":
    main()

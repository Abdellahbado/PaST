"""Quick diagnostic for price-family variants.

Runs the environment with a simple always-legal policy (argmax(action_mask))
so we can measure whether the env ever produces all-zero action masks (dead-ends)
independent of the model.

Usage:
  /path/to/python -m PaST.temp.diagnose_family_deadends --variant_id ppo_family_q4
"""

from __future__ import annotations

import argparse

import torch

from PaST.config import VariantID, get_variant_config
from PaST.train_ppo import TrainingEnv


def run(variant_id: str, batch_size: int, steps: int, seed: int, device: str) -> None:
    dev = torch.device(device)
    variant_config = get_variant_config(VariantID(variant_id), seed=seed)

    env = TrainingEnv(
        variant_config=variant_config,
        num_envs=batch_size,
        device=dev,
    )

    obs = env.reset(seed=seed)

    def _count_all_zero(mask: torch.Tensor) -> int:
        # mask is float in env contract: 1=valid, 0=invalid
        sums = mask.sum(dim=-1)
        return int((sums <= 0).sum().item())

    zero_at_reset = _count_all_zero(obs["action_mask"])

    zeros_per_step: list[int] = []
    dones_per_step: list[int] = []

    for _t in range(steps):
        action_mask = obs["action_mask"]
        zeros_per_step.append(_count_all_zero(action_mask))

        # Always pick a legal action when possible.
        # If an env is all-zero, argmax returns 0; env may treat that as invalid.
        actions = torch.argmax(action_mask, dim=-1).to(torch.long)

        obs, _rewards, dones, _info = env.step(actions)
        dones_per_step.append(int(dones.sum().item()))

    zeros_tensor = torch.tensor(zeros_per_step, dtype=torch.int64)
    dones_tensor = torch.tensor(dones_per_step, dtype=torch.int64)

    print("=" * 80)
    print(f"variant_id: {variant_id}")
    print(f"device: {device}")
    print(f"batch_size: {batch_size}")
    print(f"steps: {steps}")
    print(f"seed: {seed}")
    print("-")
    print(f"all-zero action_mask @ reset: {zero_at_reset}/{batch_size}")
    print(
        "all-zero action_mask during rollout: "
        f"min={int(zeros_tensor.min().item())}, "
        f"max={int(zeros_tensor.max().item())}, "
        f"mean={float(zeros_tensor.float().mean().item()):.3f}"
    )
    print(
        "dones during rollout (wrapper pulse): "
        f"min={int(dones_tensor.min().item())}, "
        f"max={int(dones_tensor.max().item())}, "
        f"mean={float(dones_tensor.float().mean().item()):.3f}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant_id", type=str, default="ppo_family_q4")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()

    run(
        variant_id=args.variant_id,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()

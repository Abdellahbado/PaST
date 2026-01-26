"""Check env observations for NaNs/Infs during rollout.

If the env ever emits non-finite values in jobs/periods/ctx, the model may produce
all-non-finite logits, which the PPO runner reports as "all actions masked".

Usage:
  /path/to/python -m PaST.temp.diagnose_family_obs_finiteness --variant_id ppo_family_q4_ctx13
"""

from __future__ import annotations

import argparse

import torch

from PaST.config import VariantID, get_variant_config
from PaST.train_ppo import TrainingEnv


def run(variant_id: str, batch_size: int, steps: int, seed: int, device: str) -> None:
    dev = torch.device(device)
    variant_config = get_variant_config(VariantID(variant_id), seed=seed)
    env = TrainingEnv(variant_config=variant_config, num_envs=batch_size, device=dev)

    obs = env.reset(seed=seed)

    def _check(name: str, x: torch.Tensor) -> bool:
        ok = bool(torch.isfinite(x).all().item())
        if not ok:
            bad = (~torch.isfinite(x)).any(dim=tuple(range(1, x.ndim)))
            bad_idx = bad.nonzero(as_tuple=False).flatten().tolist()
            print(f"NON-FINITE in {name} at env indices: {bad_idx[:10]}")
        return ok

    bad_steps = 0
    for t in range(steps):
        ok = True
        ok &= _check("jobs", obs["jobs"])  # (B,N,F)
        ok &= _check("periods", obs["periods"])  # (B,K,F)
        ok &= _check("ctx", obs["ctx"])  # (B,C)
        ok &= _check("action_mask", obs["action_mask"])  # (B,A)

        if not ok:
            bad_steps += 1
            print(f"first bad step: {t}")
            break

        actions = torch.argmax(obs["action_mask"], dim=-1).to(torch.long)
        obs, _rewards, _dones, _info = env.step(actions)

    print("=" * 80)
    print(f"variant_id: {variant_id}")
    print(f"device: {device}")
    print(f"batch_size: {batch_size}")
    print(f"steps checked: {steps}")
    print(f"seed: {seed}")
    print(f"bad_steps: {bad_steps}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant_id", type=str, default="ppo_family_q4")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--steps", type=int, default=256)
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

"""Roll out the env with a legal policy and check for all-masked logits.

This is a tighter reproduction of the PPO runner's all-masked warning:
- compute logits from the model
- apply env action_mask
- check if any env has all logits non-finite

We step the env using a simple always-legal action (argmax(action_mask)) to avoid
invalid-action paths.

Usage:
  /path/to/python -m PaST.temp.diagnose_family_rollout_logits --variant_id ppo_family_q4_ctx13
"""

from __future__ import annotations

import argparse

import torch

from PaST.config import VariantID, get_variant_config
from PaST.train_ppo import TrainingEnv, build_model


def _to_bool_invalid(mask):
    if mask is None:
        return None
    if mask.dtype is torch.bool:
        return mask
    return mask < 0.5


def run(variant_id: str, batch_size: int, steps: int, seed: int, device: str) -> None:
    dev = torch.device(device)
    torch.manual_seed(seed)

    variant_config = get_variant_config(VariantID(variant_id), seed=seed)
    env = TrainingEnv(variant_config=variant_config, num_envs=batch_size, device=dev)
    obs = env.reset(seed=seed)

    model = build_model(variant_config).to(dev)
    model.eval()

    all_masked_counts = []
    zero_action_mask_counts = []

    for _t in range(steps):
        action_mask = obs["action_mask"]
        valid = action_mask > 0.5
        zero_action_mask_counts.append(int((valid.sum(dim=-1) <= 0).sum().item()))

        with torch.no_grad():
            logits, _values = model(
                jobs=obs["jobs"],
                periods_local=obs["periods"],
                ctx=obs["ctx"],
                job_mask=_to_bool_invalid(obs.get("job_mask")),
                period_mask=_to_bool_invalid(obs.get("period_mask")),
                periods_full=obs.get("periods_full"),
                period_full_mask=_to_bool_invalid(obs.get("periods_full_mask")),
            )

        masked_logits = logits.masked_fill(~valid, float("-inf"))
        all_masked = ~torch.isfinite(masked_logits).any(dim=-1)
        all_masked_counts.append(int(all_masked.sum().item()))

        actions = torch.argmax(action_mask, dim=-1).to(torch.long)
        obs, _rewards, _dones, _info = env.step(actions)

    all_masked_t = torch.tensor(all_masked_counts, dtype=torch.int64)
    zero_mask_t = torch.tensor(zero_action_mask_counts, dtype=torch.int64)

    print("=" * 80)
    print(f"variant_id: {variant_id}")
    print(f"device: {device}")
    print(f"batch_size: {batch_size}")
    print(f"steps: {steps}")
    print(f"seed: {seed}")
    print("-")
    print(
        "env all-zero action_mask per step: "
        f"min={int(zero_mask_t.min())}, max={int(zero_mask_t.max())}, mean={float(zero_mask_t.float().mean()):.3f}"
    )
    print(
        "all-masked logits after applying env mask per step: "
        f"min={int(all_masked_t.min())}, max={int(all_masked_t.max())}, mean={float(all_masked_t.float().mean()):.3f}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant_id", type=str, default="ppo_family_q4")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--steps", type=int, default=64)
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

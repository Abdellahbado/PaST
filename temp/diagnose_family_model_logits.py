"""Diagnose whether the model produces finite logits under env masks.

This checks the masked-action PPO contract:
- env action_mask has at least one valid action
- after applying mask to logits, each env still has at least one finite logit

Usage:
  /path/to/python -m PaST.temp.diagnose_family_model_logits --variant_id ppo_family_q4_ctx13
"""

from __future__ import annotations

import argparse

import torch

from PaST.config import VariantID, get_variant_config
from PaST.train_ppo import TrainingEnv, build_model


def run(variant_id: str, batch_size: int, seed: int, device: str) -> None:
    dev = torch.device(device)
    torch.manual_seed(seed)

    variant_config = get_variant_config(VariantID(variant_id), seed=seed)

    env = TrainingEnv(
        variant_config=variant_config,
        num_envs=batch_size,
        device=dev,
    )
    obs = env.reset(seed=seed)

    model = build_model(variant_config).to(dev)
    model.eval()

    with torch.no_grad():
        logits, values = model(
            jobs=obs["jobs"],
            periods_local=obs["periods"],
            ctx=obs["ctx"],
            job_mask=(obs.get("job_mask", None) < 0.5)
            if (obs.get("job_mask", None) is not None and obs.get("job_mask").dtype is not torch.bool)
            else obs.get("job_mask", None),
            period_mask=(obs.get("period_mask", None) < 0.5)
            if (obs.get("period_mask", None) is not None and obs.get("period_mask").dtype is not torch.bool)
            else obs.get("period_mask", None),
            periods_full=obs.get("periods_full", None),
            period_full_mask=(obs.get("periods_full_mask", None) < 0.5)
            if (obs.get("periods_full_mask", None) is not None and obs.get("periods_full_mask").dtype is not torch.bool)
            else obs.get("periods_full_mask", None),
        )

    action_mask = obs.get("action_mask")
    if action_mask is None:
        raise RuntimeError("env did not return action_mask")

    valid = action_mask > 0.5
    env_has_any_valid = valid.any(dim=-1)

    masked_logits = logits.masked_fill(~valid, float("-inf"))
    any_finite_after_mask = torch.isfinite(masked_logits).any(dim=-1)

    print("=" * 80)
    print(f"variant_id: {variant_id}")
    print(f"device: {device}")
    print(f"batch_size: {batch_size}")
    print(f"seed: {seed}")
    print("-")
    print(f"logits finite (raw): {(torch.isfinite(logits).all(dim=-1)).tolist()}")
    print(f"env has any valid action: {env_has_any_valid.tolist()}")
    print(f"any finite logit after mask: {any_finite_after_mask.tolist()}")
    print(f"values finite: {torch.isfinite(values).all().item()}")

    bad_env = (~any_finite_after_mask).nonzero(as_tuple=False).flatten().tolist()
    if bad_env:
        print(f"BAD: all-masked after applying env mask at env indices: {bad_env}")
    else:
        print("OK: no all-masked envs after applying env mask")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant_id", type=str, default="ppo_family_q4")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()

    run(
        variant_id=args.variant_id,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()

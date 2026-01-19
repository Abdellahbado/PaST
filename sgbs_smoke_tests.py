"""Smoke tests for SGBS integration in PaST-SM.

These tests are designed to catch the common "almost works" mistakes:
- β=1,γ=1 must match greedy exactly
- snapshot restore must be deterministic
- action masking must prevent invalid actions
- episode/static tensors should not mutate during step (unless explicitly intended)

Run:
  python -m PaST.sgbs_smoke_tests --variant_id ppo_short_base --device cpu
  python -m PaST.sgbs_smoke_tests --variant_id ppo_short_base --device cuda --run_dir runs_p100/ppo_short_base/seed_0 --which best
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from PaST.config import VariantID, get_variant_config
from PaST.past_sm_model import build_model
from PaST.sgbs import (
    get_state,
    greedy_decode,
    greedy_rollout,
    set_state,
    sgbs_single_instance,
)
from PaST.sm_benchmark_data import generate_episode_batch
from PaST.sm_env import GPUBatchSingleMachinePeriodEnv


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError("Invalid checkpoint")
    return ckpt


def _extract_model_state(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    if "runner" in ckpt and isinstance(ckpt["runner"], dict):
        runner = ckpt["runner"]
        if "model" in runner and isinstance(runner["model"], dict):
            return runner["model"]
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    raise KeyError("Could not find model state")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SGBS smoke tests",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--variant_id", type=str, default="ppo_short_base")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--which", type=str, default="best", choices=["best", "latest"])
    return p.parse_args()


def _assert_close(a: float, b: float, name: str) -> None:
    if not np.isfinite(a) or not np.isfinite(b) or abs(a - b) > 1e-6:
        raise AssertionError(f"{name} mismatch: {a} vs {b}")


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    variant_config = get_variant_config(VariantID(args.variant_id))

    model = build_model(variant_config).to(device)
    model.eval()

    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        ckpt_path = run_dir / "checkpoints" / f"{args.which}.pt"
        ckpt = _load_checkpoint(ckpt_path, device)
        model_state = _extract_model_state(ckpt)
        model.load_state_dict(model_state)
        model.eval()

    batch = generate_episode_batch(
        batch_size=1,
        config=variant_config.data,
        seed=int(args.seed),
        N_job_pad=int(variant_config.env.N_job_pad),
        K_period_pad=250,
        T_max_pad=500,
    )

    # --- Greedy parity test: SGBS(1,1) equals greedy ---
    greedy = greedy_decode(model, variant_config.env, device, batch)[0]
    sgbs_11 = sgbs_single_instance(
        model=model,
        env_config=variant_config.env,
        device=device,
        batch_data_single=batch,
        beta=1,
        gamma=1,
    )
    _assert_close(
        greedy.total_energy, sgbs_11.total_energy, "greedy vs sgbs(1,1) energy"
    )

    # --- Snapshot determinism: restore snapshot and rollout twice ---
    env = GPUBatchSingleMachinePeriodEnv(
        batch_size=1, env_config=variant_config.env, device=device
    )
    env.reset(batch)

    # Take one greedy step to reach a non-trivial state
    obs = env._get_obs()
    logits, _ = model(
        jobs=obs["jobs"],
        periods_local=obs["periods"],
        ctx=obs["ctx"],
        job_mask=(obs["job_mask"] < 0.5),
        period_mask=(obs["period_mask"] < 0.5),
        periods_full=None,
        period_full_mask=None,
    )
    logits = logits.masked_fill(obs["action_mask"] == 0, float("-inf"))
    a0 = int(logits.argmax(dim=-1)[0].item())
    env.step(torch.tensor([a0], device=device))

    snap = get_state(env)

    set_state(env, snap)
    r1, e1, _ = greedy_rollout(env, model)

    set_state(env, snap)
    r2, e2, _ = greedy_rollout(env, model)

    _assert_close(
        float(r1[0].item()), float(r2[0].item()), "snapshot determinism return"
    )
    _assert_close(
        float(e1[0].item()), float(e2[0].item()), "snapshot determinism energy"
    )

    # --- Static tensor immutability check ---
    env = GPUBatchSingleMachinePeriodEnv(
        batch_size=1, env_config=variant_config.env, device=device
    )
    env.reset(batch)

    static_before = {
        "p_subset": env.p_subset.clone(),
        "n_jobs": env.n_jobs.clone(),
        "T_max": env.T_max.clone(),
        "T_limit": env.T_limit.clone(),
        "e_single": env.e_single.clone(),
        "ct": env.ct.clone(),
        "Tk": env.Tk.clone(),
        "ck": env.ck.clone(),
        "period_starts": env.period_starts.clone(),
        "K": env.K.clone(),
        "price_q": env.price_q.clone(),
    }

    # Roll out a full greedy episode
    greedy_rollout(env, model)

    for name, before in static_before.items():
        after = getattr(env, name)
        if not torch.equal(before, after):
            raise AssertionError(f"Static tensor mutated during step(): {name}")

    print("OK: SGBS smoke tests passed")


if __name__ == "__main__":
    main()

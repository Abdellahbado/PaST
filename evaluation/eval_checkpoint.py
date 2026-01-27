"""Evaluate a saved PaST PPO checkpoint on a chosen eval seed.

This is useful when the training script's "final evaluation" seed differs from
what you want to compare (e.g., the fixed eval seed used during training).

Example:
  python -m PaST.eval_checkpoint \
    --run_dir runs_p100/ppo_family_q4/seed_0 \
    --which best \
    --eval_seed 1337
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from PaST.config import VariantID, get_variant_config
from PaST.eval import Evaluator
from PaST.past_sm_model import build_model
from PaST.sm_benchmark_data import generate_episode_batch
from PaST.sm_env import GPUBatchSingleMachinePeriodEnv


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML at {path}")
    return data


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Invalid checkpoint format at {path}")
    return ckpt


def _extract_model_state(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    # Newer checkpoints: {"runner": {"model": ..., "optimizer": ...}, ...}
    if "runner" in ckpt and isinstance(ckpt["runner"], dict):
        runner = ckpt["runner"]
        if "model" in runner and isinstance(runner["model"], dict):
            return runner["model"]

    # Fallback: sometimes checkpoints store model directly.
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]

    raise KeyError("Could not find model state in checkpoint")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a saved PaST checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Run directory like runs_p100/<variant>/seed_<seed>",
    )
    p.add_argument(
        "--which",
        type=str,
        default="best",
        choices=["best", "latest"],
        help="Which checkpoint to evaluate (best.pt or latest.pt)",
    )
    p.add_argument(
        "--eval_seed",
        type=int,
        required=True,
        help="Seed for generating evaluation instances",
    )
    p.add_argument(
        "--num_eval_instances",
        type=int,
        default=None,
        help="Override number of eval instances (defaults to run_config.yaml)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device override (defaults to run_config.yaml if present)",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output JSON path (defaults to <run_dir>/eval_<which>_seed<eval_seed>.json)",
    )
    return p.parse_args()


def _resolve_run_dir(run_dir_arg: str) -> Path:
    """Resolve a run directory path robustly.

    Users often invoke this module from the workspace root (one level above the
    PaST package), while run directories live under PaST/runs_*/.... This helper
    tries a few reasonable interpretations.
    """

    raw = Path(run_dir_arg)

    # 1) As provided (relative to CWD or absolute)
    if raw.exists():
        return raw

    # 2) Relative to the PaST package directory (this file's parent)
    pkg_root = Path(__file__).resolve().parent
    candidate = pkg_root / raw
    if candidate.exists():
        return candidate

    # 3) Relative to CWD but with an implicit PaST/ prefix
    cwd = Path.cwd()
    candidate = cwd / "PaST" / raw
    if candidate.exists():
        return candidate

    # 4) If user passed something starting with "PaST/", try stripping it
    parts = list(raw.parts)
    if parts and parts[0] == "PaST":
        stripped = Path(*parts[1:])
        candidate = pkg_root / stripped
        if candidate.exists():
            return candidate
        candidate = cwd / stripped
        if candidate.exists():
            return candidate

    attempted = [
        str(raw),
        str(pkg_root / raw),
        str(cwd / "PaST" / raw),
    ]
    raise FileNotFoundError(
        "Run directory not found. Tried:\n  " + "\n  ".join(attempted)
    )


def main() -> None:
    args = parse_args()

    run_dir = _resolve_run_dir(args.run_dir)

    run_cfg_path = run_dir / "run_config.yaml"
    run_cfg = _load_yaml(run_cfg_path)

    variant_id_str = str(run_cfg.get("variant_id"))
    if not variant_id_str:
        raise ValueError(f"Missing variant_id in {run_cfg_path}")

    requested_device = str(args.device or run_cfg.get("device") or "cuda")
    # Be robust to CPU-only torch builds (common in some local conda envs).
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(
            "WARNING: CUDA requested but torch.cuda.is_available() is False. "
            "Falling back to CPU. (Pass --device cpu to silence.)"
        )
        requested_device = "cpu"
    device = torch.device(requested_device)

    num_eval = args.num_eval_instances
    if num_eval is None:
        num_eval = int(run_cfg.get("num_eval_instances", 256))

    variant_config = get_variant_config(VariantID(variant_id_str))

    model = build_model(variant_config).to(device)
    ckpt_path = run_dir / "checkpoints" / f"{args.which}.pt"
    ckpt = _load_checkpoint(ckpt_path, device)
    model_state = _extract_model_state(ckpt)
    model.load_state_dict(model_state)
    model.eval()

    env = GPUBatchSingleMachinePeriodEnv(
        batch_size=int(num_eval),
        env_config=variant_config.env,
        device=device,
    )

    evaluator = Evaluator(model=model, env=env, device=device)

    eval_batch = generate_episode_batch(
        batch_size=int(num_eval),
        config=variant_config.data,
        seed=int(args.eval_seed),
    )

    result, _ = evaluator.evaluate(eval_batch, deterministic=True)

    print(
        f"{variant_id_str} | {args.which}.pt | eval_seed={int(args.eval_seed)} | "
        f"energy={result.energy_mean:.4f}±{result.energy_std:.4f} "
        f"infeas={result.infeasible_rate*100:.2f}% | "
        f"makespan={result.makespan_mean:.4f}±{result.makespan_std:.4f}"
    )

    out_path = (
        Path(args.out)
        if args.out
        else (run_dir / f"eval_{args.which}_seed{int(args.eval_seed)}.json")
    )
    payload = result.to_dict()
    payload["eval/seed"] = int(args.eval_seed)
    payload["eval/checkpoint"] = args.which
    payload["eval/variant_id"] = variant_id_str

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

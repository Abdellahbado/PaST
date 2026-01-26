"""Unified long-run training launcher for selected PaST variants.

Runs PPO and Q-sequence variants with tuned hyperparameters for long training.
Supports sequential or multi-GPU parallel scheduling via CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from PaST.config import VariantID, get_variant_config, RLAlgorithm


@dataclass(frozen=True)
class Job:
    variant_id: str
    seed: int


def _training_entrypoint(variant_id: str) -> str:
    # NOTE: Q-sequence variants are trained via supervised regression in
    # `PaST.train_q_sequence`. In `config.py` they keep `training.algorithm` as a
    # PPO placeholder for historical reasons, so we must route by variant_id.
    if variant_id.startswith("q_sequence"):
        return "PaST.train_q_sequence"

    cfg = get_variant_config(VariantID(variant_id))
    if cfg.training.algorithm == RLAlgorithm.PPO:
        return "PaST.train_ppo"
    return "PaST.train_q_sequence"


def _ppo_cmd(
    job: Job, output_dir: str, device: str, overrides: Dict[str, str]
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "PaST.train_ppo",
        "--variant_id",
        job.variant_id,
        "--seed",
        str(job.seed),
        "--device",
        device,
        "--output_dir",
        output_dir,
    ]
    for k, v in overrides.items():
        if v is None:
            continue
        if k == "curriculum":
            cmd += ["--curriculum"]
            continue
        cmd += [f"--{k}", str(v)]
    return cmd


def _q_cmd(
    job: Job, output_dir: str, device: str, overrides: Dict[str, str]
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "PaST.train_q_sequence",
        "--variant_id",
        job.variant_id,
        "--seed",
        str(job.seed),
        "--device",
        device,
        "--output_dir",
        output_dir,
    ]
    for k, v in overrides.items():
        if v is None:
            continue
        if k == "curriculum":
            cmd += ["--curriculum"]
            continue
        cmd += [f"--{k}", str(v)]
    return cmd


def _schedule_parallel(
    jobs: List[Job],
    cmds: List[List[str]],
    gpu_ids: List[int],
    max_parallel: int,
    dry_run: bool,
) -> int:
    running: List[Tuple[subprocess.Popen, Job, int]] = []
    next_job_idx = 0
    exit_code = 0

    def _try_start_more():
        nonlocal next_job_idx
        while next_job_idx < len(jobs) and len(running) < max_parallel:
            job = jobs[next_job_idx]
            cmd = cmds[next_job_idx]
            gpu = gpu_ids[next_job_idx % len(gpu_ids)]

            print("=" * 100, flush=True)
            print(
                f"[Spawn] variant={job.variant_id} seed={job.seed} -> GPU {gpu}",
                flush=True,
            )
            print("[Cmd] " + " ".join(cmd), flush=True)
            print("=" * 100, flush=True)

            if dry_run:
                next_job_idx += 1
                continue

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            proc = subprocess.Popen(cmd, env=env)
            running.append((proc, job, gpu))
            next_job_idx += 1

    while next_job_idx < len(jobs) or running:
        _try_start_more()
        if dry_run:
            if next_job_idx >= len(jobs):
                break
            continue

        # Poll
        for i in range(len(running) - 1, -1, -1):
            proc, job, gpu = running[i]
            ret = proc.poll()
            if ret is None:
                continue
            running.pop(i)
            if ret != 0:
                exit_code = ret
                print(
                    f"[Fail] variant={job.variant_id} seed={job.seed} gpu={gpu} exit={ret}",
                    flush=True,
                )

    return exit_code


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Long-run training launcher for a fixed suite of PaST variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--output_dir", type=str, default="runs_long")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs for parallel scheduling",
    )
    p.add_argument(
        "--max_parallel",
        type=int,
        default=1,
        help="Max concurrent jobs (use >1 only with multiple GPUs)",
    )
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    variants = [
        "ppo_short_base",
        "ppo_family_q4_ctx13_beststart",
        "ppo_family_q4_ctx18_beststart",
        "ppo_duration_aware_family",
        "q_sequence_cnn_ctx13",
        "q_sequence_ctx13",
    ]

    jobs: List[Job] = []
    for v in variants:
        for s in args.seeds:
            jobs.append(Job(variant_id=v, seed=int(s)))

    # PPO long-run defaults (A100-class, stable, slow entropy decay)
    ppo_defaults = {
        "num_envs": 2048,
        "rollout_length": 128,
        "total_env_steps": 400_000_000,
        "learning_rate": 2.5e-4,
        "ppo_epochs": 4,
        "num_minibatches": 32,
        "clip_eps": 0.2,
        "value_coef": 0.5,
        "max_grad_norm": 1.0,
        "target_kl": 0.02,
        "lr_schedule": "cosine",
        "lr_end_factor": 0.1,
        "entropy_schedule": "cosine",
        "entropy_coef_start": 0.02,
        "entropy_coef_end": 0.005,
        "entropy_decay_fraction": 0.95,
        "curriculum": True,
        "curriculum_fraction": 0.4,
        "eval_every_updates": 20,
        "save_latest_every_updates": 20,
    }

    # Variant-specific PPO tweaks (keeps exploration longer for family variants)
    ppo_variant_overrides: Dict[str, Dict[str, object]] = {
        "ppo_short_base": {
            "entropy_coef_start": 0.015,
            "entropy_coef_end": 0.004,
        },
        "ppo_family_q4_ctx13_beststart": {
            "entropy_coef_start": 0.03,
            "entropy_coef_end": 0.008,
        },
        "ppo_family_q4_ctx18_beststart": {
            "entropy_coef_start": 0.03,
            "entropy_coef_end": 0.008,
        },
        "ppo_duration_aware_family": {
            "entropy_coef_start": 0.025,
            "entropy_coef_end": 0.007,
        },
    }

    # Q-sequence long-run defaults (A100-class)
    q_defaults = {
        "episodes_per_round": 8192,
        "num_rounds": 2000,
        "buffer_size": 1000000,
        "collection_batch_size": 256,
        "num_counterfactuals": 32,
        "batch_size": 1024,
        "num_epochs_per_round": 5,
        "learning_rate": 2e-4,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "loss_type": "huber",
        "huber_delta": 1.0,
        "listwise_weight": 0.1,
        "listwise_temperature": 1.0,
        "warmup_rounds": 10,
        "exploration_eps_start": 0.4,
        "exploration_eps_end": 0.05,
        "exploration_eps_decay_rounds": 400,
        "completion_policy": "mix",
        "completion_prob_start": 0.3,
        "completion_prob_end": 0.9,
        "completion_prob_decay_rounds": 300,
        "curriculum": True,
        "curriculum_fraction": 0.4,
        "curriculum_slack_min": 0.2,
        "curriculum_slack_max": 0.7,
        "eval_every_rounds": 20,
        "save_every_rounds": 50,
    }

    q_variant_overrides: Dict[str, Dict[str, object]] = {
        "q_sequence_ctx13": {"batch_size": 512},
        "q_sequence_cnn_ctx13": {"batch_size": 1024},
    }

    cmds: List[List[str]] = []
    for job in jobs:
        entry = _training_entrypoint(job.variant_id)
        if entry == "PaST.train_ppo":
            overrides = dict(ppo_defaults)
            overrides.update(ppo_variant_overrides.get(job.variant_id, {}))
            cmd = _ppo_cmd(job, args.output_dir, args.device, overrides)
        else:
            overrides = dict(q_defaults)
            overrides.update(q_variant_overrides.get(job.variant_id, {}))
            cmd = _q_cmd(job, args.output_dir, args.device, overrides)
        cmds.append(cmd)

    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    else:
        gpu_ids = [0]

    return _schedule_parallel(jobs, cmds, gpu_ids, args.max_parallel, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())

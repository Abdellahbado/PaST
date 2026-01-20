"""Experiment orchestrator for PaST.

Goals:
- Run every PaST variant (PPO + REINFORCE) with consistent settings.
- Support sequential runs (single GPU) or parallel runs (multi-GPU nodes).
- Keep everything self-contained under the PaST folder.

Example:
  python -m PaST.run_experiments --variants all --seeds 0 1 2 --config configs/a100_full.yaml \
    --output_dir runs_a100 --eval_seed 1337 --eval_seed_mode per_update

Parallel (2 GPUs):
  python -m PaST.run_experiments --variants all --seeds 0 1 --config configs/a100_full.yaml \
    --gpus 0,1 --max_parallel 2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from PaST.config import VariantID, get_variant_config, list_variants, RLAlgorithm


@dataclass(frozen=True)
class Job:
    variant_id: str
    seed: int


def _parse_csv_ints(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: List[int] = []
    for p in parts:
        out.append(int(p))
    return out


def _variant_list(arg_variants: Sequence[str]) -> List[str]:
    if len(arg_variants) == 1 and arg_variants[0].lower() == "all":
        return [v.value for v in list_variants()]

    # Validate
    valid = {v.value for v in list_variants()}
    for v in arg_variants:
        if v not in valid:
            raise ValueError(
                f"Unknown variant_id: {v}. Valid: {sorted(valid)} (or 'all')"
            )
    return list(arg_variants)


def _jobs(variants: Sequence[str], seeds: Sequence[int]) -> List[Job]:
    out: List[Job] = []
    for v in variants:
        for s in seeds:
            out.append(Job(variant_id=v, seed=int(s)))
    return out


def _training_entrypoint(variant_id: str) -> str:
    # Q-sequence variants are trained via supervised regression.
    if variant_id.startswith("q_sequence"):
        return "PaST.train_q_sequence"

    cfg = get_variant_config(VariantID(variant_id))
    if cfg.training.algorithm == RLAlgorithm.PPO:
        return "PaST.train_ppo"
    if cfg.training.algorithm == RLAlgorithm.REINFORCE:
        return "PaST.train_reinforce"
    raise ValueError(f"Unsupported algorithm: {cfg.training.algorithm}")


def _build_command(
    job: Job,
    *,
    config: Optional[str],
    output_dir: str,
    device: str,
    eval_seed: Optional[int],
    eval_seed_mode: Optional[str],
    num_envs: Optional[int],
    rollout_length: Optional[int],
    total_env_steps: Optional[int],
    learning_rate: Optional[float],
    ppo_epochs: Optional[int],
) -> List[str]:
    module = _training_entrypoint(job.variant_id)

    cmd = [
        sys.executable,
        "-m",
        module,
        "--variant_id",
        job.variant_id,
        "--seed",
        str(job.seed),
    ]

    if config:
        cmd += ["--config", config]

    cmd += ["--output_dir", output_dir]
    cmd += ["--device", device]

    if eval_seed is not None:
        cmd += ["--eval_seed", str(eval_seed)]
    if eval_seed_mode is not None:
        cmd += ["--eval_seed_mode", str(eval_seed_mode)]

    if num_envs is not None:
        cmd += ["--num_envs", str(num_envs)]
    if rollout_length is not None:
        cmd += ["--rollout_length", str(rollout_length)]
    if total_env_steps is not None:
        cmd += ["--total_env_steps", str(total_env_steps)]
    if learning_rate is not None:
        cmd += ["--learning_rate", str(learning_rate)]

    # PPO-only override (harmless for REINFORCE because train_reinforce doesn't expose it)
    if ppo_epochs is not None and module == "PaST.train_ppo":
        cmd += ["--ppo_epochs", str(ppo_epochs)]

    return cmd


def _run_one(job: Job, cmd: List[str], cuda_visible_devices: Optional[str]) -> int:
    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    print("=" * 100, flush=True)
    print(
        f"[Job] variant={job.variant_id} seed={job.seed} gpu={cuda_visible_devices or 'default'}",
        flush=True,
    )
    print("[Cmd] " + " ".join(cmd), flush=True)
    print("=" * 100, flush=True)

    return subprocess.call(cmd, env=env)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run PaST experiment suites (all variants / multiple seeds), sequentially or in parallel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--variants", nargs="+", default=["all"], help="Variant IDs, or 'all'"
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0], help="Seeds to run")

    p.add_argument(
        "--config", type=str, default=None, help="Optional YAML config (passed through)"
    )
    p.add_argument(
        "--output_dir", type=str, default="runs", help="Base output directory"
    )
    p.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device"
    )

    p.add_argument(
        "--eval_seed",
        type=int,
        default=None,
        help="Eval instance seed (same across variants)",
    )
    p.add_argument(
        "--eval_seed_mode",
        type=str,
        default=None,
        choices=["fixed", "per_update"],
        help="How to vary eval seed across eval calls (when eval_seed is set)",
    )

    # Optional overrides (passed through)
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--rollout_length", type=int, default=None)
    p.add_argument("--total_env_steps", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--ppo_epochs", type=int, default=None)

    # Parallelization controls
    p.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs to schedule jobs onto (e.g. '0,1,2,3'). If omitted, runs on default CUDA device.",
    )
    p.add_argument(
        "--max_parallel",
        type=int,
        default=1,
        help="Max concurrent jobs (use >1 only with multiple GPUs).",
    )

    p.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop the whole suite on first failure.",
    )

    return p.parse_args()


def _schedule_parallel(
    jobs: List[Job],
    *,
    cmds: List[List[str]],
    gpu_ids: List[int],
    max_parallel: int,
    fail_fast: bool,
) -> int:
    # Simple round-robin GPU assignment with a fixed-size process pool.
    running: List[Tuple[subprocess.Popen, Job, int]] = []
    next_job_idx = 0
    exit_code = 0

    def _try_start_more():
        nonlocal next_job_idx
        while next_job_idx < len(jobs) and len(running) < max_parallel:
            job = jobs[next_job_idx]
            cmd = cmds[next_job_idx]
            gpu = gpu_ids[next_job_idx % len(gpu_ids)]

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

            print("=" * 100, flush=True)
            print(
                f"[Spawn] variant={job.variant_id} seed={job.seed} -> GPU {gpu}",
                flush=True,
            )
            print("[Cmd] " + " ".join(cmd), flush=True)
            print("=" * 100, flush=True)

            proc = subprocess.Popen(cmd, env=env)
            running.append((proc, job, gpu))
            next_job_idx += 1

    _try_start_more()

    while running:
        # Poll for any finished process
        still_running: List[Tuple[subprocess.Popen, Job, int]] = []
        for proc, job, gpu in running:
            rc = proc.poll()
            if rc is None:
                still_running.append((proc, job, gpu))
                continue

            print(
                f"[Done] variant={job.variant_id} seed={job.seed} gpu={gpu} exit={rc}",
                flush=True,
            )
            if rc != 0:
                exit_code = rc
                if fail_fast:
                    # Terminate remaining
                    for p2, j2, g2 in still_running:
                        p2.terminate()
                    return exit_code
        running = still_running
        _try_start_more()

    return exit_code


def main() -> None:
    args = parse_args()

    variants = _variant_list(args.variants)
    jobs = _jobs(variants, args.seeds)

    # Ensure output dir exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    cmds = [
        _build_command(
            j,
            config=args.config,
            output_dir=args.output_dir,
            device=args.device,
            eval_seed=args.eval_seed,
            eval_seed_mode=args.eval_seed_mode,
            num_envs=args.num_envs,
            rollout_length=args.rollout_length,
            total_env_steps=args.total_env_steps,
            learning_rate=args.learning_rate,
            ppo_epochs=args.ppo_epochs,
        )
        for j in jobs
    ]

    if args.gpus is None or args.max_parallel <= 1:
        # Sequential (or parallel managed externally)
        for j, cmd in zip(jobs, cmds):
            rc = _run_one(j, cmd, cuda_visible_devices=None)
            if rc != 0:
                if args.fail_fast:
                    raise SystemExit(rc)
        return

    gpu_ids = _parse_csv_ints(args.gpus)
    if not gpu_ids:
        raise ValueError("--gpus must be non-empty when provided")

    if args.max_parallel > len(gpu_ids):
        print(
            f"[Warn] max_parallel={args.max_parallel} > num_gpus={len(gpu_ids)}; clamping to {len(gpu_ids)}",
            flush=True,
        )
        max_parallel = len(gpu_ids)
    else:
        max_parallel = args.max_parallel

    rc = _schedule_parallel(
        jobs,
        cmds=cmds,
        gpu_ids=gpu_ids,
        max_parallel=max_parallel,
        fail_fast=bool(args.fail_fast),
    )
    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()

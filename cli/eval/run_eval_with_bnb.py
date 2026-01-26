"""Evaluate PaST-SM with PPO decoding + classic DP baselines + Branch-and-Bound (BnB).

Compares:
- Greedy model decode (masked argmax in env action space)
- SGBS(beta, gamma)
- SPT + DP scheduling (optimal start times for fixed SPT order)
- LPT + DP scheduling (optimal start times for fixed LPT order)
- BnB (strong baseline): searches over job sequences, uses DP scheduling to evaluate

Instance generation:
- Uses PaST's `generate_episode_batch` (same distribution as training).
- You can select instance size via `--scale` or a specific horizon via `--T_max`.

Note on objective consistency:
- Environment step cost is: energy = e_single * sum(ct[u] for u in processing slots)
- The DP baselines and BnB baseline optimize this same energy objective with a
  hard deadline T_limit (schedule must finish by T_limit).

Example (small instances, CPU):
    python -m PaST.cli.eval.run_eval_with_bnb \
    --checkpoint PaST/runs_p100/ppo_short_base/checkpoints/best.pt \
    --variant_id ppo_short_base \
        --scale small --num_instances 20 --eval_seed 123 \
        --max_machine_jobs 10 \
        --beta 4 --gamma 4 --bnb_time_limit 10 --bnb_quiet
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from PaST.baselines_sequence_dp import spt_lpt_with_dp
from PaST.config import DataConfig, VariantID, get_variant_config
from PaST.past_sm_model import build_model
from PaST.sgbs import (
    compute_masked_logits,
    greedy_decode,
    sgbs,
    _completion_feasible_action_mask,
)
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
    if "runner" in ckpt and isinstance(ckpt["runner"], dict):
        runner = ckpt["runner"]
        if "model" in runner and isinstance(runner["model"], dict):
            return runner["model"]
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    raise KeyError("Could not find model state in checkpoint")


def _resolve_run_dir(run_dir_arg: str) -> Path:
    raw = Path(run_dir_arg)
    if raw.exists():
        return raw

    pkg_root = Path(__file__).resolve().parent
    candidate = pkg_root / raw
    if candidate.exists():
        return candidate

    cwd = Path.cwd()
    candidate = cwd / "PaST" / raw
    if candidate.exists():
        return candidate

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


def _mean(x: List[float]) -> float:
    arr = np.array(x, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return float("nan")
    return float(arr[finite].mean())


def _price_shuffle_batch(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Shuffle prices across periods while keeping durations/period starts fixed."""
    out = {k: v.copy() for k, v in batch.items()}
    B = int(out["K"].shape[0])
    for i in range(B):
        K = int(out["K"][i])
        if K <= 1:
            continue
        perm = np.random.permutation(K)
        ck = out["ck"][i].copy()
        Tk = out["Tk"][i].copy()
        period_starts = out["period_starts"][i].copy()

        ck_shuffled = ck.copy()
        ck_shuffled[:K] = ck[perm]
        out["ck"][i] = ck_shuffled

        T_limit = int(out["T_limit"][i])
        ct = out["ct"][i].copy()
        for k in range(K):
            start = int(period_starts[k])
            end = min(int(start + Tk[k]), T_limit)
            if start < end:
                ct[start:end] = ck_shuffled[k]
        out["ct"][i] = ct
    return out


@torch.no_grad()
def _random_rollout(
    env: GPUBatchSingleMachinePeriodEnv,
    use_completion_mask: bool = False,
    max_steps: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random rollout baseline in the env action space.

    Returns:
        total_energy (B,), scheduled_jobs (B,), infeasible_mask (B,)
    """
    B = env.batch_size
    if max_steps is None:
        max_steps = int(env.N_job_pad) + 5

    scheduled = torch.zeros((B,), dtype=torch.int32, device=env.device)
    done_mask = env.done_mask.clone()
    infeasible = torch.zeros((B,), dtype=torch.bool, device=env.device)

    obs = env._get_obs()
    for _ in range(int(max_steps)):
        if done_mask.all():
            break

        if use_completion_mask:
            mask = _completion_feasible_action_mask(env, obs)
        else:
            mask = obs.get("action_mask", None)
            if mask is None:
                mask = torch.ones((B, env.action_dim), device=env.device)
        mask = mask.float()

        valid_any = mask.sum(dim=1) > 0
        newly_infeasible = (~valid_any) & (~done_mask)
        if newly_infeasible.any():
            done_mask = done_mask | newly_infeasible
            env.done_mask = env.done_mask | newly_infeasible
            infeasible = infeasible | newly_infeasible

        if done_mask.all():
            break

        sum_mask = mask.sum(dim=1, keepdim=True)
        safe = (sum_mask > 0).squeeze(1)
        probs = torch.where(
            sum_mask > 0, mask / (sum_mask + 1e-8), torch.zeros_like(mask)
        )
        actions = torch.zeros((B,), dtype=torch.long, device=env.device)
        if safe.any():
            actions[safe] = torch.multinomial(probs[safe], num_samples=1).squeeze(1)
        actions = actions.masked_fill(done_mask, 0)

        active = ~done_mask
        obs, _rewards, dones, _info = env.step(actions)
        done_mask = done_mask | dones
        scheduled = scheduled + active.to(torch.int32)

    total_energy = env.total_energy.detach().cpu().numpy()
    return (
        total_energy,
        scheduled.detach().cpu().numpy(),
        infeasible.detach().cpu().numpy(),
    )


@torch.no_grad()
def _greedy_entropy_stats(
    env: GPUBatchSingleMachinePeriodEnv,
    model,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    """Collect mean entropy/top1 probability for greedy policy actions."""
    B = env.batch_size
    if max_steps is None:
        max_steps = int(env.N_job_pad) + 5

    entropies: List[float] = []
    top1_probs: List[float] = []
    margins: List[float] = []

    obs = env._get_obs()
    done_mask = env.done_mask.clone()

    for _ in range(int(max_steps)):
        if done_mask.all():
            break

        logits = compute_masked_logits(model, obs)
        logits = logits.masked_fill(
            _completion_feasible_action_mask(env, obs) == 0, float("-inf")
        )
        probs = torch.softmax(logits, dim=-1)

        for i in range(B):
            if bool(done_mask[i]):
                continue
            finite = torch.isfinite(logits[i])
            if not finite.any():
                continue
            p = probs[i][finite]
            p = p / (p.sum() + 1e-12)
            entropy = float(-(p * torch.log(p + 1e-12)).sum().item())
            top1 = float(p.max().item())
            top2 = float(torch.topk(p, k=min(2, p.numel())).values[-1].item())
            margin = top1 - top2 if p.numel() >= 2 else top1
            entropies.append(entropy)
            top1_probs.append(top1)
            margins.append(margin)

        actions = logits.argmax(dim=-1)
        actions = actions.masked_fill(done_mask, 0)
        obs, _rewards, dones, _info = env.step(actions)
        done_mask = done_mask | dones

    def safe_mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "entropy_mean": safe_mean(entropies),
        "top1_prob_mean": safe_mean(top1_probs),
        "margin_mean": safe_mean(margins),
    }


def _load_solver_improved() -> Tuple[type, type]:
    """Dynamically load `solver_improved.py` (path contains spaces)."""
    workspace_root = Path(__file__).resolve().parents[1]
    solver_path = (
        workspace_root
        / "Transformer Implementation"
        / "Data Generation"
        / "solver_improved.py"
    )
    if not solver_path.exists():
        raise FileNotFoundError(f"Could not find solver at: {solver_path}")

    spec = importlib.util.spec_from_file_location("solver_improved", solver_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {solver_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    Instance = getattr(mod, "Instance")
    BranchAndBoundSolver = getattr(mod, "BranchAndBoundSolver")
    return Instance, BranchAndBoundSolver


def _restrict_data_config(
    data: DataConfig,
    scale: Optional[str],
    T_max: Optional[int],
    n_min: Optional[int],
    n_max: Optional[int],
) -> DataConfig:
    cfg = replace(data)

    if T_max is not None:
        cfg.T_max_choices = [int(T_max)]
    elif scale is not None:
        s = scale.lower()
        if s == "small":
            cfg.T_max_choices = [t for t in cfg.T_max_choices if int(t) <= 80]
        elif s in ("mls", "medium"):
            cfg.T_max_choices = [t for t in cfg.T_max_choices if 80 < int(t) <= 300]
        elif s in ("vls", "large"):
            cfg.T_max_choices = [t for t in cfg.T_max_choices if int(t) > 300]
        else:
            raise ValueError("scale must be one of: small, mls/medium, vls/large")

        if not cfg.T_max_choices:
            raise ValueError(
                f"No T_max_choices match scale={scale}. Available: {data.T_max_choices}"
            )

    if n_min is not None:
        cfg.n_min = int(n_min)
    if n_max is not None:
        cfg.n_max = int(n_max)
    if cfg.n_min > cfg.n_max:
        raise ValueError("n_min must be <= n_max")

    return cfg


def _slice_batch(
    batch: Dict[str, np.ndarray], idxs: List[int]
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, v in batch.items():
        if not isinstance(v, np.ndarray):
            raise TypeError(
                f"Expected numpy arrays in batch, got {type(v)} for key={k}"
            )
        out[k] = v[idxs]
    return out


def _concat_batches(batches: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if not batches:
        raise ValueError("No batches to concatenate")
    keys = list(batches[0].keys())
    out: Dict[str, np.ndarray] = {}
    for k in keys:
        out[k] = np.concatenate([b[k] for b in batches], axis=0)
    return out


def _generate_filtered_batch(
    *,
    num_instances: int,
    config: DataConfig,
    seed: int,
    N_job_pad: int,
    max_machine_jobs: Optional[int],
    max_attempts: int = 25,
    oversample_factor: int = 8,
) -> Dict[str, np.ndarray]:
    """Generate a batch and (optionally) filter by per-machine job count.

    BnB becomes impractical when a single-machine episode has too many jobs.
    We filter on `n_jobs` (jobs on the single machine), not the global `n`.
    """
    if num_instances <= 0:
        raise ValueError("num_instances must be > 0")

    if max_machine_jobs is None:
        return generate_episode_batch(
            batch_size=int(num_instances),
            config=config,
            seed=int(seed),
            N_job_pad=int(N_job_pad),
            K_period_pad=250,
            T_max_pad=500,
        )

    selected: List[Dict[str, np.ndarray]] = []
    selected_count = 0
    attempt = 0

    while selected_count < num_instances and attempt < max_attempts:
        attempt += 1
        # Use a different seed each attempt to avoid sampling the same batch.
        batch_seed = int(seed) + attempt * 10_000
        oversample = max(64, int(num_instances) * int(oversample_factor))
        b = generate_episode_batch(
            batch_size=int(oversample),
            config=config,
            seed=int(batch_seed),
            N_job_pad=int(N_job_pad),
            K_period_pad=250,
            T_max_pad=500,
        )

        ok = np.where(b["n_jobs"] <= int(max_machine_jobs))[0].tolist()
        if ok:
            take = ok[: max(0, num_instances - selected_count)]
            selected.append(_slice_batch(b, take))
            selected_count += len(take)

    if selected_count < num_instances:
        raise RuntimeError(
            f"Could not sample enough instances with n_jobs <= {max_machine_jobs}. "
            f"Got {selected_count}/{num_instances}. Try increasing --max_machine_jobs, "
            "decreasing --num_instances, or using a smaller --scale / lower --T_max."
        )

    out = _concat_batches(selected)
    # Truncate in case we overshot.
    return _slice_batch(out, list(range(int(num_instances))))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate checkpoint with Greedy/SGBS + DP baselines + BnB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--variant_id", type=str, default=None)
    p.add_argument("--which", type=str, default="best", choices=["best", "latest"])

    p.add_argument("--eval_seed", type=int, required=True)
    p.add_argument("--num_instances", type=int, default=20)
    p.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])

    p.add_argument(
        "--scale",
        type=str,
        default="small",
        help="small | mls (medium) | vls (large)",
    )
    p.add_argument(
        "--T_max",
        type=int,
        default=None,
        help="Force a specific T_max (overrides --scale)",
    )
    p.add_argument("--n_min", type=int, default=None, help="Override DataConfig.n_min")
    p.add_argument("--n_max", type=int, default=None, help="Override DataConfig.n_max")

    p.add_argument("--beta", type=int, default=4)
    p.add_argument("--gamma", type=int, default=4)

    p.add_argument(
        "--max_machine_jobs",
        type=int,
        default=10,
        help="Filter generated episodes to those with n_jobs <= this (keeps BnB reasonable). Use 0 to disable filtering.",
    )

    p.add_argument(
        "--bnb_time_limit", type=float, default=10.0, help="Seconds per instance"
    )
    p.add_argument(
        "--bnb_quiet", action="store_true", help="Suppress BnB per-instance prints"
    )
    p.add_argument(
        "--skip_bnb",
        action="store_true",
        help="Skip Branch-and-Bound (recommended for mls/vls).",
    )

    p.add_argument("--out_csv", type=str, default=None)
    p.add_argument("--out_json", type=str, default=None)
    p.add_argument(
        "--price_shuffle",
        action="store_true",
        help="Also evaluate on price-shuffled instances.",
    )
    p.add_argument(
        "--random_baselines",
        action="store_true",
        help="Include random-valid and random-feasible baselines.",
    )
    p.add_argument(
        "--greedy_entropy",
        action="store_true",
        help="Report greedy entropy/top1/margin stats.",
    )

    # Inference-time controls to discourage late scheduling
    p.add_argument(
        "--max_wait_slots",
        type=int,
        default=None,
        help="Cap slack-induced waiting: disallow actions that start more than this many slots after the current time. (None = no cap)",
    )
    p.add_argument(
        "--wait_logit_penalty",
        type=float,
        default=0.0,
        help="Decoding-time soft bias: subtract (wait_logit_penalty * wait_slots) from action logits to prefer earlier starts.",
    )
    p.add_argument(
        "--makespan_penalty",
        type=float,
        default=0.0,
        help="SGBS pruning objective regularizer: score = -energy - makespan_penalty * completion_time. (Energy reporting unchanged)",
    )
    p.add_argument(
        "--dp_time_penalty",
        type=float,
        default=0.0,
        help="DP baseline regularizer: add dp_time_penalty * start_time to each job's DP cost to prefer earlier placements.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.run_dir is None and args.checkpoint is None:
        raise ValueError("Provide either --run_dir or --checkpoint")
    if args.run_dir is not None and args.checkpoint is not None:
        raise ValueError("Provide only one of --run_dir or --checkpoint")

    run_dir: Path | None = None
    run_cfg: Dict[str, Any] | None = None

    if args.run_dir is not None:
        run_dir = _resolve_run_dir(args.run_dir)
        run_cfg_path = run_dir / "run_config.yaml"
        if run_cfg_path.exists():
            run_cfg = _load_yaml(run_cfg_path)

    if args.checkpoint is not None:
        if not args.variant_id:
            raise ValueError("--variant_id is required when using --checkpoint")
        variant_id_str = str(args.variant_id)
        ckpt_path = Path(args.checkpoint)
        ckpt_tag = ckpt_path.stem
    else:
        if run_cfg is None:
            raise ValueError(
                "--run_dir must point to a run directory containing run_config.yaml"
            )
        variant_id_str = str(run_cfg.get("variant_id"))
        if not variant_id_str:
            raise ValueError("Missing variant_id in run_config.yaml")
        ckpt_path = run_dir / "checkpoints" / f"{args.which}.pt"  # type: ignore[operator]
        ckpt_tag = args.which

    requested_device = str(
        args.device or (run_cfg.get("device") if run_cfg else None) or "cuda"
    )
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but unavailable; falling back to CPU")
        requested_device = "cpu"
    device = torch.device(requested_device)

    variant_config = get_variant_config(VariantID(variant_id_str))

    data_cfg = _restrict_data_config(
        variant_config.data,
        scale=str(args.scale) if args.scale else None,
        T_max=args.T_max,
        n_min=args.n_min,
        n_max=args.n_max,
    )

    model = build_model(variant_config).to(device)
    ckpt = _load_checkpoint(ckpt_path, device)
    model.load_state_dict(_extract_model_state(ckpt))
    model.eval()

    max_machine_jobs = (
        None if int(args.max_machine_jobs) <= 0 else int(args.max_machine_jobs)
    )
    batch = _generate_filtered_batch(
        num_instances=int(args.num_instances),
        config=data_cfg,
        seed=int(args.eval_seed),
        N_job_pad=int(variant_config.env.N_job_pad),
        max_machine_jobs=max_machine_jobs,
    )

    random_valid = None
    random_feasible = None
    greedy_stats = None
    if bool(args.random_baselines):
        rand_env = GPUBatchSingleMachinePeriodEnv(
            batch_size=int(args.num_instances),
            env_config=variant_config.env,
            device=device,
        )
        rand_env.reset(batch)
        rv_energy, rv_sched, rv_infeas = _random_rollout(
            rand_env, use_completion_mask=False
        )
        random_valid = {
            "energy": rv_energy,
            "scheduled": rv_sched,
            "infeasible": rv_infeas,
        }

        rand_env2 = GPUBatchSingleMachinePeriodEnv(
            batch_size=int(args.num_instances),
            env_config=variant_config.env,
            device=device,
        )
        rand_env2.reset(batch)
        rf_energy, rf_sched, rf_infeas = _random_rollout(
            rand_env2, use_completion_mask=True
        )
        random_feasible = {
            "energy": rf_energy,
            "scheduled": rf_sched,
            "infeasible": rf_infeas,
        }

    if bool(args.greedy_entropy):
        ent_env = GPUBatchSingleMachinePeriodEnv(
            batch_size=int(args.num_instances),
            env_config=variant_config.env,
            device=device,
        )
        ent_env.reset(batch)
        greedy_stats = _greedy_entropy_stats(ent_env, model)

    t0 = time.perf_counter()
    greedy_res = greedy_decode(
        model,
        variant_config.env,
        device,
        batch,
        max_wait_slots=(
            int(args.max_wait_slots) if args.max_wait_slots is not None else None
        ),
        wait_logit_penalty=float(args.wait_logit_penalty),
        makespan_penalty=float(args.makespan_penalty),
    )
    greedy_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    sgbs_res = sgbs(
        model=model,
        env_config=variant_config.env,
        device=device,
        batch_data=batch,
        beta=int(args.beta),
        gamma=int(args.gamma),
        max_wait_slots=(
            int(args.max_wait_slots) if args.max_wait_slots is not None else None
        ),
        wait_logit_penalty=float(args.wait_logit_penalty),
        makespan_penalty=float(args.makespan_penalty),
    )
    sgbs_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    spt_res = spt_lpt_with_dp(
        variant_config.env,
        device,
        batch,
        which="spt",
        dp_time_penalty=float(args.dp_time_penalty),
    )
    spt_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    lpt_res = spt_lpt_with_dp(
        variant_config.env,
        device,
        batch,
        which="lpt",
        dp_time_penalty=float(args.dp_time_penalty),
    )
    lpt_time = time.perf_counter() - t0

    shuffle_means = None
    if bool(args.price_shuffle):
        shuffled = _price_shuffle_batch(batch)
        g_shuf = greedy_decode(
            model,
            variant_config.env,
            device,
            shuffled,
            max_wait_slots=(
                int(args.max_wait_slots) if args.max_wait_slots is not None else None
            ),
            wait_logit_penalty=float(args.wait_logit_penalty),
            makespan_penalty=float(args.makespan_penalty),
        )
        s_shuf = sgbs(
            model=model,
            env_config=variant_config.env,
            device=device,
            batch_data=shuffled,
            beta=int(args.beta),
            gamma=int(args.gamma),
            max_wait_slots=(
                int(args.max_wait_slots) if args.max_wait_slots is not None else None
            ),
            wait_logit_penalty=float(args.wait_logit_penalty),
            makespan_penalty=float(args.makespan_penalty),
        )
        spt_shuf = spt_lpt_with_dp(
            variant_config.env,
            device,
            shuffled,
            which="spt",
            dp_time_penalty=float(args.dp_time_penalty),
        )
        lpt_shuf = spt_lpt_with_dp(
            variant_config.env,
            device,
            shuffled,
            which="lpt",
            dp_time_penalty=float(args.dp_time_penalty),
        )
        shuffle_means = {
            "greedy_energy": _mean([r.total_energy for r in g_shuf]),
            "sgbs_energy": _mean([r.total_energy for r in s_shuf]),
            "spt_dp_energy": _mean([r.total_energy for r in spt_shuf]),
            "lpt_dp_energy": _mean([r.total_energy for r in lpt_shuf]),
        }

    bnb_energies: List[float] = []
    bnb_solve_times: List[float] = []
    bnb_nodes: List[int] = []
    bnb_timed_out: List[bool] = []
    bnb_binpack_attempts: List[int] = []
    bnb_pruned_by_binpack: List[int] = []

    if (not bool(args.skip_bnb)) and float(args.bnb_time_limit) > 0:
        Instance, BranchAndBoundSolver = _load_solver_improved()

        for i in range(int(args.num_instances)):
            if not args.bnb_quiet:
                print(
                    f"BnB {i+1}/{int(args.num_instances)} | n_jobs={int(batch['n_jobs'][i])} | T_limit={int(batch['T_limit'][i])}"
                )
            n_jobs = int(batch["n_jobs"][i])
            T_limit = int(batch["T_limit"][i])
            ct = batch["ct"][i][:T_limit].astype(np.int32)
            e_single = int(batch["e_single"][i])
            processing_times = batch["p_subset"][i][:n_jobs].astype(np.int32)

            energy_costs = (ct.astype(np.int64) * int(e_single)).astype(np.int32)

            inst = Instance(
                n_jobs=int(n_jobs),
                processing_times=processing_times,
                T=int(T_limit),
                energy_costs=energy_costs,
            )

            solver = BranchAndBoundSolver(inst, time_limit=float(args.bnb_time_limit))

            solve_start = time.perf_counter()
            if args.bnb_quiet:
                with redirect_stdout(StringIO()):
                    _seq, best_cost = solver.solve()
            else:
                _seq, best_cost = solver.solve()
            solve_dur = time.perf_counter() - solve_start

            # The solver does not explicitly expose an "optimal proven" flag.
            # Heuristic: if we ran basically up to the time limit, treat as timed out.
            timed_out = solve_dur >= max(0.0, float(args.bnb_time_limit) - 1e-3)

            bnb_energies.append(float(best_cost))
            bnb_solve_times.append(float(solve_dur))
            bnb_nodes.append(int(getattr(solver, "nodes_explored", 0)))
            bnb_timed_out.append(bool(timed_out))
            bnb_binpack_attempts.append(int(getattr(solver, "binpack_attempts", 0)))
            bnb_pruned_by_binpack.append(int(getattr(solver, "pruned_by_binpack", 0)))

    greedy_mean = _mean([r.total_energy for r in greedy_res])
    sgbs_mean = _mean([r.total_energy for r in sgbs_res])
    spt_mean = _mean([r.total_energy for r in spt_res])
    lpt_mean = _mean([r.total_energy for r in lpt_res])
    bnb_mean = _mean(bnb_energies) if bnb_energies else float("nan")

    greedy_infeas = sum(
        int((len((r.actions or [])) != int(batch["n_jobs"][i])))
        for i, r in enumerate(greedy_res)
    )
    sgbs_infeas = sum(
        int((len((r.actions or [])) != int(batch["n_jobs"][i])))
        for i, r in enumerate(sgbs_res)
    )

    print(
        f"{variant_id_str} | {ckpt_tag}.pt | scale={args.scale} | seed={args.eval_seed} | N={args.num_instances}\n"
        f"  greedy:   {greedy_mean:.4f}  (time {greedy_time:.2f}s) | infeasible {greedy_infeas}/{int(args.num_instances)}\n"
        f"  sgbs(b={args.beta},g={args.gamma}): {sgbs_mean:.4f}  (time {sgbs_time:.2f}s) | infeasible {sgbs_infeas}/{int(args.num_instances)}\n"
        f"  spt+dp:   {spt_mean:.4f}  (time {spt_time:.2f}s)\n"
        f"  lpt+dp:   {lpt_mean:.4f}  (time {lpt_time:.2f}s)\n"
        + (
            f"  bnb:      {bnb_mean:.4f}  (sum time {sum(bnb_solve_times):.2f}s)"
            if bnb_energies
            else "  bnb:      (skipped)"
        )
    )

    if random_valid is not None:
        rv_mean = _mean(random_valid["energy"].tolist())
        rv_infeas = int(np.sum(random_valid["infeasible"]))
        print(
            f"  rand-valid: {rv_mean:.4f} | infeasible {rv_infeas}/{int(args.num_instances)}"
        )
    if random_feasible is not None:
        rf_mean = _mean(random_feasible["energy"].tolist())
        rf_infeas = int(np.sum(random_feasible["infeasible"]))
        print(
            f"  rand-feasible: {rf_mean:.4f} | infeasible {rf_infeas}/{int(args.num_instances)}"
        )
    if greedy_stats is not None:
        print(
            "  greedy-stats: "
            f"entropy={greedy_stats['entropy_mean']:.4f} "
            f"top1={greedy_stats['top1_prob_mean']:.4f} "
            f"margin={greedy_stats['margin_mean']:.4f}"
        )
    if shuffle_means is not None:
        print(
            "  price-shuffle: "
            f"greedy={shuffle_means['greedy_energy']:.4f} "
            f"sgbs={shuffle_means['sgbs_energy']:.4f} "
            f"spt+dp={shuffle_means['spt_dp_energy']:.4f} "
            f"lpt+dp={shuffle_means['lpt_dp_energy']:.4f}"
        )

    default_base = run_dir if run_dir is not None else Path.cwd()
    out_csv = (
        Path(args.out_csv)
        if args.out_csv
        else default_base
        / f"eval_all_{ckpt_tag}_seed{args.eval_seed}_{args.scale}_b{args.beta}_g{args.gamma}_bnb{int(args.bnb_time_limit)}s.csv"
    )
    out_json = Path(args.out_json) if args.out_json else out_csv.with_suffix(".json")

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for i in range(int(args.num_instances)):
        n_jobs_i = int(batch["n_jobs"][i])
        greedy_actions = greedy_res[i].actions or []
        sgbs_actions = sgbs_res[i].actions or []
        greedy_scheduled = int(len(greedy_actions))
        sgbs_scheduled = int(len(sgbs_actions))
        rows.append(
            {
                "instance": i,
                "n_jobs": n_jobs_i,
                "T_limit": int(batch["T_limit"][i]),
                "greedy_energy": float(greedy_res[i].total_energy),
                "greedy_scheduled_jobs": greedy_scheduled,
                "greedy_infeasible": bool(greedy_scheduled != n_jobs_i),
                "sgbs_energy": float(sgbs_res[i].total_energy),
                "sgbs_scheduled_jobs": sgbs_scheduled,
                "sgbs_infeasible": bool(sgbs_scheduled != n_jobs_i),
                "spt_dp_energy": float(spt_res[i].total_energy),
                "lpt_dp_energy": float(lpt_res[i].total_energy),
                "bnb_energy": float(bnb_energies[i]) if bnb_energies else float("nan"),
                "bnb_time_sec": (
                    float(bnb_solve_times[i]) if bnb_solve_times else float("nan")
                ),
                "bnb_nodes": int(bnb_nodes[i]) if bnb_nodes else 0,
                "bnb_timed_out": bool(bnb_timed_out[i]) if bnb_timed_out else False,
                "bnb_binpack_attempts": (
                    int(bnb_binpack_attempts[i]) if bnb_binpack_attempts else 0
                ),
                "bnb_pruned_by_binpack": (
                    int(bnb_pruned_by_binpack[i]) if bnb_pruned_by_binpack else 0
                ),
                "rand_valid_energy": (
                    float(random_valid["energy"][i])
                    if random_valid is not None
                    else float("nan")
                ),
                "rand_valid_infeasible": (
                    bool(random_valid["infeasible"][i])
                    if random_valid is not None
                    else False
                ),
                "rand_feasible_energy": (
                    float(random_feasible["energy"][i])
                    if random_feasible is not None
                    else float("nan")
                ),
                "rand_feasible_infeasible": (
                    bool(random_feasible["infeasible"][i])
                    if random_feasible is not None
                    else False
                ),
            }
        )

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)

    summary = {
        "variant_id": variant_id_str,
        "checkpoint": str(ckpt_path),
        "eval_seed": int(args.eval_seed),
        "num_instances": int(args.num_instances),
        "scale": str(args.scale),
        "T_max_choices": list(data_cfg.T_max_choices),
        "beta": int(args.beta),
        "gamma": int(args.gamma),
        "bnb_time_limit": float(args.bnb_time_limit),
        "random_baselines": bool(args.random_baselines),
        "price_shuffle": bool(args.price_shuffle),
        "greedy_entropy": bool(args.greedy_entropy),
        "greedy_stats": greedy_stats,
        "shuffle_means": shuffle_means,
        "means": {
            "greedy_energy": greedy_mean,
            "sgbs_energy": sgbs_mean,
            "spt_dp_energy": spt_mean,
            "lpt_dp_energy": lpt_mean,
            "bnb_energy": bnb_mean,
        },
        "times_sec": {
            "greedy": float(greedy_time),
            "sgbs": float(sgbs_time),
            "spt_dp": float(spt_time),
            "lpt_dp": float(lpt_time),
            "bnb_sum": float(sum(bnb_solve_times)),
            "bnb_mean": float(np.mean(bnb_solve_times) if bnb_solve_times else 0.0),
        },
        "outputs": {
            "csv": str(out_csv),
            "json": str(out_json),
        },
    }

    if random_valid is not None:
        summary["rand_valid_mean"] = _mean(random_valid["energy"].tolist())
    if random_feasible is not None:
        summary["rand_feasible_mean"] = _mean(random_feasible["energy"].tolist())

    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import signal
import json
import time
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from PaST.solvers.alns_parallel import ALNSConfig
from PaST.alns_rl.pm_alns_env import PMALNSEnv


@dataclass
class PPOCfg:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 3e-4
    max_grad_norm: float = 1.0
    ppo_epochs: int = 4
    minibatches: int = 8


class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.body(obs)
        logits = self.pi(h)
        value = self.v(h).squeeze(-1)
        return logits, value


def _normalize_obs_np(obs: np.ndarray) -> np.ndarray:
    """Lightweight observation normalization to improve PPO stability.

    The raw ALNS features include large-magnitude energy values; compress them
    with log1p and scale time/load terms to O(1).
    """

    x = np.asarray(obs, dtype=np.float32)
    y = x.copy()

    # Indices are defined by alns_state_features (10 dims).
    # 0: it_frac, 1: no_improve_frac, 2: tau
    # 3: cur_energy, 4: best_energy, 5: gap
    # 6: makespan, 7: slack, 8: load_mean, 9: load_cv
    for idx in (3, 4):
        y[..., idx] = np.log1p(np.maximum(0.0, y[..., idx])) / 10.0

    gap = y[..., 5]
    y[..., 5] = np.sign(gap) * (np.log1p(np.abs(gap)) / 10.0)

    y[..., 2] = np.clip(y[..., 2], 0.0, 10.0)
    for idx in (6, 7, 8):
        y[..., idx] = y[..., idx] / 300.0

    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _torch_sanitize(x: torch.Tensor) -> torch.Tensor:
    """Replace non-finite values with 0.0 (works across torch versions)."""

    if hasattr(torch, "nan_to_num"):
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    finite = torch.isfinite(x)
    return torch.where(finite, x, torch.zeros_like(x))


def _rollout_worker(
    *,
    worker_id: int,
    policy_state: Dict[str, Any],
    obs_dim: int,
    act_dim: int,
    hidden: int,
    scale: str,
    base_seed: int,
    start_instance_id: int,
    envs_per_worker: int,
    rollout_len: int,
    slack_ratio: float,
    alns_cfg_dict: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    # Limit BLAS threads inside each worker to avoid oversubscription.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    torch.set_num_threads(1)

    net = PolicyValueNet(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden)
    net.load_state_dict(policy_state)
    net.eval()

    alns_cfg = ALNSConfig(**alns_cfg_dict)

    envs: List[PMALNSEnv] = []
    for i in range(int(envs_per_worker)):
        inst_id = int(start_instance_id) + int(worker_id) * int(envs_per_worker) + i
        envs.append(
            PMALNSEnv(
                scale=str(scale),
                seed=int(base_seed) + 17 * int(worker_id) + i,
                instance_id=int(inst_id),
                slack_ratio=float(slack_ratio),
                alns_cfg=alns_cfg,
            )
        )

    obs = np.stack([e.reset() for e in envs], axis=0).astype(np.float32)  # [B,F]
    obs = _normalize_obs_np(obs)
    init_summaries = [e.get_state_summary() for e in envs]

    T = int(rollout_len)
    B = int(envs_per_worker)

    obs_buf = np.zeros((T, B, obs_dim), dtype=np.float32)
    act_buf = np.zeros((T, B), dtype=np.int64)
    logp_buf = np.zeros((T, B), dtype=np.float32)
    val_buf = np.zeros((T, B), dtype=np.float32)
    rew_buf = np.zeros((T, B), dtype=np.float32)
    done_buf = np.zeros((T, B), dtype=np.float32)

    accepted_cnt = 0
    feasible_cnt = 0
    step_cnt = 0

    for t in range(T):
        obs_t = torch.from_numpy(obs)
        with torch.no_grad():
            logits, values = net(obs_t)
            dist = torch.distributions.Categorical(logits=logits, validate_args=False)
            actions = dist.sample()
            logp = dist.log_prob(actions)

        obs_buf[t] = obs
        act_buf[t] = actions.cpu().numpy()
        logp_buf[t] = logp.cpu().numpy()
        val_buf[t] = values.cpu().numpy()

        next_obs = []
        for b, e in enumerate(envs):
            o2, r, d, info = e.step(int(act_buf[t, b]))
            rew_buf[t, b] = float(r)
            done_buf[t, b] = float(d)

            s = info.get("step", {}) if isinstance(info, dict) else {}
            if bool(s.get("accepted", False)):
                accepted_cnt += 1
            if bool(s.get("feasible_cand", False)):
                feasible_cnt += 1
            step_cnt += 1

            if d:
                o2 = e.reset()
            next_obs.append(o2)
        obs = np.stack(next_obs, axis=0).astype(np.float32)
        obs = _normalize_obs_np(obs)

    # Bootstrap value
    with torch.no_grad():
        _, last_v = net(torch.from_numpy(obs))
    last_v = last_v.cpu().numpy().astype(np.float32)

    end_summaries = [e.get_state_summary() for e in envs]

    init_best = np.asarray([s["best_energy"] for s in init_summaries], dtype=np.float64)
    end_best = np.asarray([s["best_energy"] for s in end_summaries], dtype=np.float64)
    best_improve = init_best - end_best

    return {
        "obs": obs_buf,
        "actions": act_buf,
        "logp": logp_buf,
        "values": val_buf,
        "rewards": rew_buf,
        "dones": done_buf,
        "last_values": last_v,
        "accepted_cnt": np.asarray([accepted_cnt], dtype=np.int64),
        "feasible_cnt": np.asarray([feasible_cnt], dtype=np.int64),
        "step_cnt": np.asarray([step_cnt], dtype=np.int64),
        "end_best_energy": end_best.astype(np.float64),
        "best_improve": best_improve.astype(np.float64),
    }


def _compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # rewards/values/dones: [T,B]
    T, B = rewards.shape
    device = rewards.device
    dtype = rewards.dtype
    adv = torch.zeros((T, B), dtype=dtype, device=device)
    last_gae = torch.zeros((B,), dtype=dtype, device=device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_values = last_values
            next_non_terminal = 1.0 - dones[t]
        else:
            next_values = values[t + 1]
            next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_non_terminal * next_values - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        adv[t] = last_gae

    returns = adv + values
    return adv, returns


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--scale", type=str, default="medium")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--slack_ratio", type=float, default=0.25)

    p.add_argument("--out_dir", type=str, default="analysis_out")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for PPO updates (rollouts stay CPU).",
    )
    p.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Save checkpoint every N updates.",
    )
    p.add_argument(
        "--max_seconds",
        type=float,
        default=0.0,
        help="Optional walltime budget; 0 disables.",
    )
    p.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to a checkpoint .pt to resume from.",
    )

    p.add_argument(
        "--metrics_jsonl",
        type=str,
        default="",
        help="Optional JSONL metrics output path (defaults to <out_dir>/<run_name>_metrics.jsonl).",
    )

    p.add_argument("--max_iters", type=int, default=200)
    p.add_argument("--no_improve", type=int, default=80)
    p.add_argument("--destroy_frac", type=float, default=0.12)
    p.add_argument("--destroy_max", type=int, default=80)

    p.add_argument("--hidden", type=int, default=128)

    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--envs_per_worker", type=int, default=2)
    p.add_argument("--rollout_len", type=int, default=32)
    p.add_argument("--updates", type=int, default=30)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--entropy", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=8)

    p.add_argument(
        "--reward_clip",
        type=float,
        default=10.0,
        help="Clip per-step rewards to [-reward_clip, +reward_clip] to improve stability.",
    )
    p.add_argument(
        "--no_obs_norm",
        action="store_true",
        help="Disable observation normalization (not recommended).",
    )

    return p


def _atomic_torch_save(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"Invalid checkpoint file: {path}")
    return ckpt


def main() -> None:
    args = build_parser().parse_args()

    out_dir = str(args.out_dir)
    run_name = str(args.run_name).strip()
    if not run_name:
        run_name = f"ppo_alns_pm_{args.scale}_seed{args.seed}"
    os.makedirs(out_dir, exist_ok=True)
    ckpt_latest = os.path.join(out_dir, f"{run_name}_latest.pt")
    metrics_path = str(args.metrics_jsonl).strip()
    if not metrics_path:
        metrics_path = os.path.join(out_dir, f"{run_name}_metrics.jsonl")
    metrics_path_p = Path(metrics_path)

    if int(args.workers) <= 0:
        workers = max(1, os.cpu_count() or 1)
    else:
        workers = int(args.workers)

    obs_dim = 10
    # action_dim determined by env
    tmp_env = PMALNSEnv(
        scale=str(args.scale),
        seed=int(args.seed),
        instance_id=0,
        slack_ratio=float(args.slack_ratio),
        alns_cfg=ALNSConfig(max_iters=1, no_improve_limit=1),
    )
    act_dim = tmp_env.action_dim

    if str(args.device) == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    random.seed(int(args.seed))

    net = PolicyValueNet(obs_dim=obs_dim, act_dim=act_dim, hidden=int(args.hidden)).to(
        device
    )
    opt = torch.optim.Adam(net.parameters(), lr=float(args.lr))

    ppo = PPOCfg(
        lr=float(args.lr),
        entropy_coef=float(args.entropy),
        clip_eps=float(args.clip),
        ppo_epochs=int(args.ppo_epochs),
        minibatches=int(args.minibatches),
    )

    alns_cfg_dict = {
        "max_iters": int(args.max_iters),
        "no_improve_limit": int(args.no_improve),
        "destroy_frac": float(args.destroy_frac),
        "destroy_max": int(args.destroy_max),
        "keep_iter_log": False,
        "oracle_dataset": False,
        "verify_cost": False,
    }

    envs_per_worker = int(args.envs_per_worker)
    rollout_len = int(args.rollout_len)

    base_instance = 0

    from concurrent.futures import ProcessPoolExecutor

    stop_requested = {"flag": False}

    def _request_stop(signum: int, frame: Any) -> None:  # pragma: no cover
        stop_requested["flag"] = True

    try:
        signal.signal(signal.SIGTERM, _request_stop)
        signal.signal(signal.SIGINT, _request_stop)
    except Exception:
        # Some environments restrict signal handlers.
        pass

    start_time = time.time()
    start_update = 0

    if str(args.resume).strip():
        ckpt_path = str(args.resume).strip()
        ckpt = _load_checkpoint(ckpt_path, device=device)
        net.load_state_dict(ckpt["state_dict"])
        if "opt_state" in ckpt and isinstance(ckpt["opt_state"], dict):
            opt.load_state_dict(ckpt["opt_state"])
        if "update" in ckpt:
            start_update = int(ckpt["update"])
        if "py_random_state" in ckpt:
            try:
                random.setstate(ckpt["py_random_state"])
            except Exception:
                pass
        if "np_random_state" in ckpt:
            try:
                np.random.set_state(ckpt["np_random_state"])
            except Exception:
                pass
        if "torch_rng_state" in ckpt:
            try:
                torch.set_rng_state(ckpt["torch_rng_state"].cpu())
            except Exception:
                pass
        if device.type == "cuda" and "torch_cuda_rng_state" in ckpt:
            try:
                torch.cuda.set_rng_state_all(ckpt["torch_cuda_rng_state"])
            except Exception:
                pass
        print(f"Resumed from {ckpt_path} at update={start_update}")

    print(
        f"Training PPO device={device.type} act_dim={act_dim} workers={workers} envs/worker={envs_per_worker}",
        flush=True,
    )
    print(f"Metrics: {metrics_path_p}", flush=True)

    # Touch metrics file early so monitoring (tail -f) works immediately.
    metrics_path_p.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path_p.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "time": time.time(),
                    "event": "start",
                    "run_name": run_name,
                    "args": vars(args),
                }
            )
            + "\n"
        )

    last_metrics: Dict[str, Any] | None = None

    def _save_checkpoint(update: int) -> str:
        payload: Dict[str, Any] = {
            "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
            "opt_state": opt.state_dict(),
            "act_dim": int(act_dim),
            "obs_dim": int(obs_dim),
            "update": int(update),
            "args": vars(args),
            "last_metrics": last_metrics,
            "py_random_state": random.getstate(),
            "np_random_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            try:
                payload["torch_cuda_rng_state"] = torch.cuda.get_rng_state_all()
            except Exception:
                pass
        _atomic_torch_save(payload, ckpt_latest)
        return ckpt_latest

    total_updates = int(args.updates)
    ex = ProcessPoolExecutor(max_workers=workers)
    try:
        for upd in range(int(start_update), total_updates):
            if stop_requested["flag"]:
                p = _save_checkpoint(update=upd)
                print(f"Stop requested; saved checkpoint: {p}", flush=True)
                break
            if float(args.max_seconds) > 0.0:
                if (time.time() - start_time) >= float(args.max_seconds):
                    p = _save_checkpoint(update=upd)
                    print(f"Walltime reached; saved checkpoint: {p}", flush=True)
                    break

            net.eval()
            state = {k: v.cpu() for k, v in net.state_dict().items()}

            rollout_t0 = time.time()
            futs = []
            for wid in range(workers):
                futs.append(
                    ex.submit(
                        _rollout_worker,
                        worker_id=int(wid),
                        policy_state=state,
                        obs_dim=obs_dim,
                        act_dim=act_dim,
                        hidden=int(args.hidden),
                        scale=str(args.scale),
                        base_seed=int(args.seed) + 1000 * upd,
                        start_instance_id=int(base_instance) + 100000 * upd,
                        envs_per_worker=int(envs_per_worker),
                        rollout_len=int(rollout_len),
                        slack_ratio=float(args.slack_ratio),
                        alns_cfg_dict=alns_cfg_dict,
                    )
                )

            batches = []
            rollout_error = None
            for f in futs:
                try:
                    batches.append(f.result())
                except Exception as e:
                    rollout_error = e
                    break

            if rollout_error is not None:
                p = _save_checkpoint(update=upd)
                print(
                    f"update={upd+1}: rollout worker failed ({type(rollout_error).__name__}); saved checkpoint: {p}",
                    flush=True,
                )
                print(traceback.format_exc(), flush=True)
                break

            rollout_seconds = max(1e-9, float(time.time() - rollout_t0))

            # Concatenate along env dimension
            obs = torch.from_numpy(
                np.concatenate([b["obs"] for b in batches], axis=1)
            ).to(device)
            actions = torch.from_numpy(
                np.concatenate([b["actions"] for b in batches], axis=1)
            ).to(device)
            logp_old = torch.from_numpy(
                np.concatenate([b["logp"] for b in batches], axis=1)
            ).to(device)
            values = torch.from_numpy(
                np.concatenate([b["values"] for b in batches], axis=1)
            ).to(device)
            rewards = torch.from_numpy(
                np.concatenate([b["rewards"] for b in batches], axis=1)
            ).to(device)
            dones = torch.from_numpy(
                np.concatenate([b["dones"] for b in batches], axis=1)
            ).to(device)
            last_values = torch.from_numpy(
                np.concatenate([b["last_values"] for b in batches], axis=0)
            ).to(device)

            accepted_cnt = int(np.sum([int(b["accepted_cnt"][0]) for b in batches]))
            feasible_cnt = int(np.sum([int(b["feasible_cnt"][0]) for b in batches]))
            step_cnt = int(np.sum([int(b["step_cnt"][0]) for b in batches]))

            end_best_energy = np.concatenate(
                [b["end_best_energy"] for b in batches], axis=0
            )
            best_improve = np.concatenate([b["best_improve"] for b in batches], axis=0)

            T, B, _ = obs.shape

            # Safety: sanitize/clip rewards; NaNs in rewards/returns can blow up PPO.
            rewards = _torch_sanitize(rewards)
            rclip = float(args.reward_clip)
            if rclip > 0:
                rewards = torch.clamp(rewards, -rclip, rclip)

            # Optional obs normalization (should already be normalized in worker, but keep robust).
            if bool(getattr(args, "no_obs_norm", False)) is False:
                # obs is [T,B,F] on device; normalize in numpy on CPU once for speed.
                obs_np = obs.detach().cpu().numpy()
                obs = torch.from_numpy(_normalize_obs_np(obs_np)).to(device)

            adv, returns = _compute_gae(
                rewards=rewards,
                values=values,
                dones=dones,
                last_values=last_values,
                gamma=float(ppo.gamma),
                lam=float(ppo.gae_lambda),
            )

            # If anything went non-finite, skip update and keep going.
            if not (
                torch.isfinite(adv).all()
                and torch.isfinite(returns).all()
                and torch.isfinite(values).all()
                and torch.isfinite(logp_old).all()
                and torch.isfinite(obs).all()
            ):
                net.load_state_dict(state)
                print(
                    f"update={upd+1}: non-finite rollout tensors; skipping PPO update",
                    flush=True,
                )
                continue

            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # Flatten
            obs_f = obs.reshape(T * B, obs_dim)
            act_f = actions.reshape(T * B)
            logp_old_f = logp_old.reshape(T * B)
            adv_f = adv.reshape(T * B)
            ret_f = returns.reshape(T * B)

            batch_size = T * B
            mb = max(1, int(batch_size // int(ppo.minibatches)))

            net.train()
            opt_state_before = copy.deepcopy(opt.state_dict())
            bad_update = False
            for _ in range(int(ppo.ppo_epochs)):
                idx = torch.randperm(batch_size)
                for start in range(0, batch_size, mb):
                    j = idx[start : start + mb]
                    logits, v = net(obs_f[j])
                    if not (torch.isfinite(logits).all() and torch.isfinite(v).all()):
                        bad_update = True
                        break
                    dist = torch.distributions.Categorical(
                        logits=logits, validate_args=False
                    )
                    logp = dist.log_prob(act_f[j])
                    log_ratio = torch.clamp(logp - logp_old_f[j], -20.0, 20.0)
                    ratio = torch.exp(log_ratio)
                    surr1 = ratio * adv_f[j]
                    surr2 = (
                        torch.clamp(ratio, 1.0 - ppo.clip_eps, 1.0 + ppo.clip_eps)
                        * adv_f[j]
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = ((v - ret_f[j]) ** 2).mean()
                    entropy = dist.entropy().mean()

                    loss = (
                        policy_loss
                        + ppo.value_coef * value_loss
                        - ppo.entropy_coef * entropy
                    )

                    if not torch.isfinite(loss):
                        bad_update = True
                        break

                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        net.parameters(), float(ppo.max_grad_norm)
                    )
                    opt.step()

                if bad_update:
                    break

            if bad_update:
                # Revert to pre-update weights to avoid poisoning the run.
                net.load_state_dict(state)
                try:
                    opt.load_state_dict(opt_state_before)
                except Exception:
                    pass
                print(
                    f"update={upd+1}: non-finite logits/loss encountered; reverted and skipped",
                    flush=True,
                )
                continue

            avg_r = float(rewards.mean().item())

            accept_rate = float(accepted_cnt) / float(max(1, step_cnt))
            feasible_rate = float(feasible_cnt) / float(max(1, step_cnt))
            steps_per_sec = float(step_cnt) / float(max(1e-9, rollout_seconds))

            eb = np.asarray(end_best_energy, dtype=np.float64)
            bi = np.asarray(best_improve, dtype=np.float64)
            eb_f = eb[np.isfinite(eb)]
            bi_f = bi[np.isfinite(bi)]

            eb_nonfinite = int(eb.size - eb_f.size)
            bi_nonfinite = int(bi.size - bi_f.size)

            if eb_f.size > 0:
                eb_p10, eb_p50, eb_p90 = (
                    float(x) for x in np.quantile(eb_f, [0.1, 0.5, 0.9])
                )
                eb_mean = float(np.mean(eb_f))
            else:
                eb_p10 = eb_p50 = eb_p90 = float("nan")
                eb_mean = float("nan")

            if bi_f.size > 0:
                bi_p10, bi_p50, bi_p90 = (
                    float(x) for x in np.quantile(bi_f, [0.1, 0.5, 0.9])
                )
                bi_mean = float(np.mean(bi_f))
            else:
                bi_p10 = bi_p50 = bi_p90 = float("nan")
                bi_mean = float("nan")

            msg = (
                f"update={upd+1}/{int(args.updates)} "
                f"avg_step_reward={avg_r:+.4f} "
                f"rollout_s={rollout_seconds:.1f} steps/s={steps_per_sec:.1f} "
                f"accept={accept_rate:.3f} feasible={feasible_rate:.3f} "
                f"bestE(p10/p50/p90)={eb_p10:.1f}/{eb_p50:.1f}/{eb_p90:.1f} "
                f"bestImprove(p10/p50/p90)={bi_p10:.2f}/{bi_p50:.2f}/{bi_p90:.2f}"
            )
            print(msg, flush=True)

            # Append JSONL metrics for offline plotting.
            rec = {
                "time": time.time(),
                "update": int(upd + 1),
                "avg_step_reward": float(avg_r),
                "rollout_seconds": float(rollout_seconds),
                "steps_per_sec": float(steps_per_sec),
                "accepted": int(accepted_cnt),
                "feasible": int(feasible_cnt),
                "steps": int(step_cnt),
                "accept_rate": float(accept_rate),
                "feasible_rate": float(feasible_rate),
                "best_energy_mean": float(eb_mean),
                "best_energy_p10": float(eb_p10),
                "best_energy_p50": float(eb_p50),
                "best_energy_p90": float(eb_p90),
                "best_improve_mean": float(bi_mean),
                "best_improve_p50": float(bi_p50),
                "best_energy_nonfinite": int(eb_nonfinite),
                "best_improve_nonfinite": int(bi_nonfinite),
            }
            last_metrics = rec
            metrics_path_p.parent.mkdir(parents=True, exist_ok=True)
            with metrics_path_p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

            if int(args.save_every) > 0 and ((upd + 1) % int(args.save_every) == 0):
                p = _save_checkpoint(update=upd + 1)
                print(f"Saved checkpoint: {p}", flush=True)

    finally:
        try:
            ex.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    # Save final weights (policy only)
    out_final = os.path.join(out_dir, f"{run_name}.pt")
    _atomic_torch_save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
            "act_dim": int(act_dim),
        },
        out_final,
    )
    print("Saved", out_final, flush=True)


if __name__ == "__main__":
    main()

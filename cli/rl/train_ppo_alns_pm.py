from __future__ import annotations

import argparse
import math
import os
import random
import signal
import time
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

    T = int(rollout_len)
    B = int(envs_per_worker)

    obs_buf = np.zeros((T, B, obs_dim), dtype=np.float32)
    act_buf = np.zeros((T, B), dtype=np.int64)
    logp_buf = np.zeros((T, B), dtype=np.float32)
    val_buf = np.zeros((T, B), dtype=np.float32)
    rew_buf = np.zeros((T, B), dtype=np.float32)
    done_buf = np.zeros((T, B), dtype=np.float32)

    for t in range(T):
        obs_t = torch.from_numpy(obs)
        with torch.no_grad():
            logits, values = net(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            logp = dist.log_prob(actions)

        obs_buf[t] = obs
        act_buf[t] = actions.cpu().numpy()
        logp_buf[t] = logp.cpu().numpy()
        val_buf[t] = values.cpu().numpy()

        next_obs = []
        for b, e in enumerate(envs):
            o2, r, d, _ = e.step(int(act_buf[t, b]))
            rew_buf[t, b] = float(r)
            done_buf[t, b] = float(d)
            if d:
                o2 = e.reset()
            next_obs.append(o2)
        obs = np.stack(next_obs, axis=0).astype(np.float32)

    # Bootstrap value
    with torch.no_grad():
        _, last_v = net(torch.from_numpy(obs))
    last_v = last_v.cpu().numpy().astype(np.float32)

    return {
        "obs": obs_buf,
        "actions": act_buf,
        "logp": logp_buf,
        "values": val_buf,
        "rewards": rew_buf,
        "dones": done_buf,
        "last_values": last_v,
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
    adv = torch.zeros((T, B), dtype=torch.float32)
    last_gae = torch.zeros((B,), dtype=torch.float32)

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
        f"Training PPO device={device.type} act_dim={act_dim} workers={workers} envs/worker={envs_per_worker}"
    )

    def _save_checkpoint(update: int) -> str:
        payload: Dict[str, Any] = {
            "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
            "opt_state": opt.state_dict(),
            "act_dim": int(act_dim),
            "obs_dim": int(obs_dim),
            "update": int(update),
            "args": vars(args),
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
    for upd in range(int(start_update), total_updates):
        if stop_requested["flag"]:
            p = _save_checkpoint(update=upd)
            print(f"Stop requested; saved checkpoint: {p}")
            break
        if float(args.max_seconds) > 0.0:
            if (time.time() - start_time) >= float(args.max_seconds):
                p = _save_checkpoint(update=upd)
                print(f"Walltime reached; saved checkpoint: {p}")
                break

        net.eval()
        state = {k: v.cpu() for k, v in net.state_dict().items()}

        with ProcessPoolExecutor(max_workers=workers) as ex:
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
            batches = [f.result() for f in futs]

        # Concatenate along env dimension
        obs = torch.from_numpy(np.concatenate([b["obs"] for b in batches], axis=1)).to(
            device
        )
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

        T, B, _ = obs.shape

        adv, returns = _compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            last_values=last_values,
            gamma=float(ppo.gamma),
            lam=float(ppo.gae_lambda),
        )

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
        for _ in range(int(ppo.ppo_epochs)):
            idx = torch.randperm(batch_size)
            for start in range(0, batch_size, mb):
                j = idx[start : start + mb]
                logits, v = net(obs_f[j])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act_f[j])
                ratio = torch.exp(logp - logp_old_f[j])
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

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net.parameters(), float(ppo.max_grad_norm)
                )
                opt.step()

        avg_r = float(rewards.mean().item())
        print(
            f"update={upd+1}/{int(args.updates)} avg_step_reward={avg_r:+.4f} batch_envs={B}"
        )

        if int(args.save_every) > 0 and ((upd + 1) % int(args.save_every) == 0):
            p = _save_checkpoint(update=upd + 1)
            print(f"Saved checkpoint: {p}")

    # Save final weights (policy only)
    out_final = os.path.join(out_dir, f"{run_name}.pt")
    _atomic_torch_save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
            "act_dim": int(act_dim),
        },
        out_final,
    )
    print("Saved", out_final)


if __name__ == "__main__":
    main()

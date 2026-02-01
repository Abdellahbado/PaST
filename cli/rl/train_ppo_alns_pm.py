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
    # Optional PPO safety valve: stop early when updates get too large.
    # Typical values are ~0.01-0.05. None disables.
    target_kl: float | None = None


class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int, act_dims: List[int], hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.pi_heads = nn.ModuleList([nn.Linear(hidden, int(d)) for d in act_dims])
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        h = self.body(obs)
        logits = [head(h) for head in self.pi_heads]
        value = self.v(h).squeeze(-1)
        return logits, value


def _multi_categorical_logprob(
    logits_list: List[torch.Tensor], actions: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Total log-prob and entropy for a factorized categorical policy.

    Args:
        logits_list: list of [B, A_i] tensors
        actions: [B, H] int tensor, where H == len(logits_list)
    Returns:
        (logp_total [B], entropy_total [B]) summed across heads.
    """

    if actions.ndim != 2 or int(actions.size(1)) != int(len(logits_list)):
        raise ValueError("actions must have shape [B, num_heads]")

    logps: List[torch.Tensor] = []
    ents: List[torch.Tensor] = []
    for i, lg in enumerate(logits_list):
        dist = torch.distributions.Categorical(logits=lg, validate_args=False)
        ai = actions[:, i]
        logps.append(dist.log_prob(ai))
        ents.append(dist.entropy())
    logp_total = torch.stack(logps, dim=0).sum(dim=0)
    ent_total = torch.stack(ents, dim=0).sum(dim=0)
    return logp_total, ent_total


def _flatten_factor_action(act: np.ndarray, act_dims: List[int]) -> np.ndarray:
    """Flatten factorized actions into a joint index (for histograms/debug only)."""

    a = np.asarray(act, dtype=np.int64)
    if a.ndim == 1:
        if int(a.size) != int(len(act_dims)):
            raise ValueError("action vector must have len(act_dims)")
        a = a.reshape(1, -1)
    if a.ndim != 2 or int(a.shape[1]) != int(len(act_dims)):
        raise ValueError("actions must be [N, num_heads]")

    idx = np.zeros((a.shape[0],), dtype=np.int64)
    mult = 1
    for i in reversed(range(int(len(act_dims)))):
        di = int(act_dims[i])
        idx += np.clip(a[:, i], 0, di - 1) * int(mult)
        mult *= di
    return idx


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
    act_dims: List[int],
    hidden: int,
    scales: List[str],
    action_set: str,
    base_seed: int,
    start_instance_id: int,
    envs_per_worker: int,
    rollout_len: int,
    slack_ratio: float,
    reward_scale: float,
    reward_power: float,
    best_improve_bonus: float,
    reward_best_coef: float,
    reward_accept_coef: float,
    sa_worse_accept_penalty_coef: float,
    sa_worse_accept_penalty_power: float,
    reject_penalty: float,
    reject_worse_penalty_coef: float,
    reject_worse_penalty_power: float,
    min_stop_iters: int,
    terminal_best_coef: float,
    terminal_time_coef: float,
    no_obs_norm: bool,
    alns_cfg_dict: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    # Limit BLAS threads inside each worker to avoid oversubscription.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    torch.set_num_threads(1)

    net = PolicyValueNet(obs_dim=obs_dim, act_dims=act_dims, hidden=hidden)
    net.load_state_dict(policy_state)
    net.eval()

    alns_cfg = ALNSConfig(**alns_cfg_dict)

    scales = [str(s).strip().lower() for s in (scales or []) if str(s).strip()]
    if not scales:
        scales = ["medium"]

    envs: List[PMALNSEnv] = []
    for i in range(int(envs_per_worker)):
        inst_id = int(start_instance_id) + int(worker_id) * int(envs_per_worker) + i
        # Deterministic multi-scale assignment so each update has a stable mix.
        scale_i = scales[int(inst_id) % int(len(scales))]
        envs.append(
            PMALNSEnv(
                scale=str(scale_i),
                seed=int(base_seed) + 17 * int(worker_id) + i,
                instance_id=int(inst_id),
                slack_ratio=float(slack_ratio),
                alns_cfg=alns_cfg,
                action_set=str(action_set),
                reward_scale=float(reward_scale),
                reward_power=float(reward_power),
                best_improve_bonus=float(best_improve_bonus),
                reward_best_coef=float(reward_best_coef),
                reward_accept_coef=float(reward_accept_coef),
                sa_worse_accept_penalty_coef=float(sa_worse_accept_penalty_coef),
                sa_worse_accept_penalty_power=float(sa_worse_accept_penalty_power),
                reject_penalty=float(reject_penalty),
                reject_worse_penalty_coef=float(reject_worse_penalty_coef),
                reject_worse_penalty_power=float(reject_worse_penalty_power),
                min_stop_iters=int(min_stop_iters),
                terminal_best_coef=float(terminal_best_coef),
                terminal_time_coef=float(terminal_time_coef),
            )
        )

    obs = np.stack([e.reset() for e in envs], axis=0).astype(np.float32)  # [B,F]
    if bool(no_obs_norm) is False:
        obs = _normalize_obs_np(obs)
    init_summaries = [e.get_state_summary() for e in envs]

    T = int(rollout_len)
    B = int(envs_per_worker)
    H = int(len(act_dims))
    act_dim_total = int(np.prod(np.asarray(act_dims, dtype=np.int64)))

    obs_buf = np.zeros((T, B, obs_dim), dtype=np.float32)
    act_buf = np.zeros((T, B, H), dtype=np.int64)
    logp_buf = np.zeros((T, B), dtype=np.float32)
    val_buf = np.zeros((T, B), dtype=np.float32)
    rew_buf = np.zeros((T, B), dtype=np.float32)
    done_buf = np.zeros((T, B), dtype=np.float32)

    joint_action_counts = np.zeros((int(act_dim_total),), dtype=np.int64)

    accepted_cnt = 0
    feasible_cnt = 0
    step_cnt = 0
    fail_cnt = 0
    infeasible_cnt = 0
    exception_cnt = 0

    for t in range(T):
        obs_t = torch.from_numpy(obs)
        with torch.no_grad():
            logits_list, values = net(obs_t)
            acts: List[torch.Tensor] = []
            logps: List[torch.Tensor] = []
            for lg in logits_list:
                dist = torch.distributions.Categorical(logits=lg, validate_args=False)
                a = dist.sample()
                acts.append(a)
                logps.append(dist.log_prob(a))
            actions = torch.stack(acts, dim=-1)
            logp = torch.stack(logps, dim=0).sum(dim=0)

        obs_buf[t] = obs
        act_buf[t] = actions.cpu().numpy()
        logp_buf[t] = logp.cpu().numpy()
        val_buf[t] = values.cpu().numpy()

        next_obs = []
        for b, e in enumerate(envs):
            o2, r, d, info = e.step(act_buf[t, b].tolist())
            rew_buf[t, b] = float(r)
            done_buf[t, b] = float(d)

            s = info.get("step", {}) if isinstance(info, dict) else {}
            if bool(s.get("accepted", False)):
                accepted_cnt += 1
            if bool(s.get("feasible_cand", False)):
                feasible_cnt += 1

            if isinstance(info, dict):
                if bool(info.get("exception", False)):
                    exception_cnt += 1
                fm = str(info.get("failure_mode", ""))
                if fm == "repair_failed":
                    fail_cnt += 1
                elif fm == "infeasible":
                    infeasible_cnt += 1
            step_cnt += 1

            if d:
                o2 = e.reset()
            next_obs.append(o2)
        obs = np.stack(next_obs, axis=0).astype(np.float32)
        if bool(no_obs_norm) is False:
            obs = _normalize_obs_np(obs)

        # Joint-action histogram for collapse diagnostics.
        joint_idx = _flatten_factor_action(act_buf[t].reshape(B, H), act_dims)
        joint_action_counts += np.bincount(
            joint_idx.astype(np.int64), minlength=int(act_dim_total)
        )

    # Bootstrap value
    with torch.no_grad():
        _, last_v = net(torch.from_numpy(obs))
    last_v = last_v.cpu().numpy().astype(np.float32)

    end_summaries = [e.get_state_summary() for e in envs]

    init_best = np.asarray([s["best_energy"] for s in init_summaries], dtype=np.float64)
    end_best = np.asarray([s["best_energy"] for s in end_summaries], dtype=np.float64)
    best_improve = init_best - end_best

    # Deadline tightness diagnostics
    init_eps = np.asarray([s["epsilon"] for s in init_summaries], dtype=np.float64)
    init_ms = np.asarray([s["cur_makespan"] for s in init_summaries], dtype=np.float64)
    init_slack = np.asarray([s["slack"] for s in init_summaries], dtype=np.float64)
    init_slack_norm = np.asarray(
        [s["slack_norm"] for s in init_summaries], dtype=np.float64
    )

    return {
        "obs": obs_buf,
        "actions": act_buf,
        "logp": logp_buf,
        "values": val_buf,
        "rewards": rew_buf,
        "dones": done_buf,
        "last_values": last_v,
        "rollout_joint_action_counts": joint_action_counts.astype(np.int64),
        "accepted_cnt": np.asarray([accepted_cnt], dtype=np.int64),
        "feasible_cnt": np.asarray([feasible_cnt], dtype=np.int64),
        "step_cnt": np.asarray([step_cnt], dtype=np.int64),
        "fail_cnt": np.asarray([fail_cnt], dtype=np.int64),
        "infeasible_cnt": np.asarray([infeasible_cnt], dtype=np.int64),
        "exception_cnt": np.asarray([exception_cnt], dtype=np.int64),
        "end_best_energy": end_best.astype(np.float64),
        "best_improve": best_improve.astype(np.float64),
        "init_epsilon": init_eps.astype(np.float64),
        "init_makespan": init_ms.astype(np.float64),
        "init_slack": init_slack.astype(np.float64),
        "init_slack_norm": init_slack_norm.astype(np.float64),
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


def _safe_quantiles(x: np.ndarray, qs: List[float]) -> List[float]:
    xf = np.asarray(x, dtype=np.float64)
    xf = xf[np.isfinite(xf)]
    if xf.size == 0:
        return [float("nan") for _ in qs]
    return [float(v) for v in np.quantile(xf, qs)]


def _safe_mean(x: np.ndarray) -> float:
    xf = np.asarray(x, dtype=np.float64)
    xf = xf[np.isfinite(xf)]
    if xf.size == 0:
        return float("nan")
    return float(np.mean(xf))


def _action_count_stats(counts: np.ndarray) -> Dict[str, Any]:
    c = np.asarray(counts, dtype=np.float64)
    c = np.where(np.isfinite(c), c, 0.0)
    c = np.maximum(c, 0.0)
    tot = float(np.sum(c))
    if tot <= 0.0:
        return {
            "counts": [int(0) for _ in range(int(c.size))],
            "max_frac": float("nan"),
            "entropy": float("nan"),
            "top3": [],
        }
    p = c / tot
    # Entropy of empirical action distribution (nats).
    eps = 1e-12
    ent = float(-np.sum(p * np.log(p + eps)))
    max_frac = float(np.max(p))
    top = np.argsort(-p)[: min(3, int(p.size))]
    top3 = [(int(i), float(p[int(i)])) for i in top]
    return {
        "counts": [int(x) for x in c.astype(np.int64).tolist()],
        "max_frac": float(max_frac),
        "entropy": float(ent),
        "top3": top3,
    }


def _eval_policy_vs_random(
    *,
    net: PolicyValueNet,
    act_dims: List[int],
    scales: List[str],
    action_set: str,
    alns_cfg_dict: Dict[str, Any],
    slack_ratio: float,
    reward_scale: float,
    reward_power: float,
    best_improve_bonus: float,
    reward_best_coef: float,
    reward_accept_coef: float,
    sa_worse_accept_penalty_coef: float,
    sa_worse_accept_penalty_power: float,
    reject_penalty: float,
    reject_worse_penalty_coef: float,
    reject_worse_penalty_power: float,
    min_stop_iters: int,
    terminal_best_coef: float,
    terminal_time_coef: float,
    no_obs_norm: bool,
    eval_seed: int,
    eval_num_envs: int,
    eval_rollout_len: int,
    eval_instance_offset: int,
    deterministic: bool,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate current policy against a uniform-random policy on a fixed instance set.

    This is designed as a lightweight sanity-check that learning is beating random,
    and is stable across resume because it depends only on CLI args.
    """

    scales = [str(s).strip().lower() for s in (scales or []) if str(s).strip()]
    if not scales:
        scales = ["medium"]

    alns_cfg = ALNSConfig(**alns_cfg_dict)
    n = int(eval_num_envs)
    T = int(eval_rollout_len)
    H = int(len(act_dims))
    act_dim_total = int(np.prod(np.asarray(act_dims, dtype=np.int64)))

    # Two independent env sets with identical seeds/instance_ids so their initial
    # states match (to the extent the environment is deterministic under seed).
    envs_pol: List[PMALNSEnv] = []
    envs_rnd: List[PMALNSEnv] = []
    for i in range(n):
        inst_id = int(eval_instance_offset) + i
        scale_i = scales[int(inst_id) % int(len(scales))]
        seed_i = int(eval_seed) + 1009 * i
        envs_pol.append(
            PMALNSEnv(
                scale=str(scale_i),
                seed=int(seed_i),
                instance_id=int(inst_id),
                slack_ratio=float(slack_ratio),
                alns_cfg=alns_cfg,
                action_set=str(action_set),
                reward_scale=float(reward_scale),
                reward_power=float(reward_power),
                best_improve_bonus=float(best_improve_bonus),
                reward_best_coef=float(reward_best_coef),
                reward_accept_coef=float(reward_accept_coef),
                sa_worse_accept_penalty_coef=float(sa_worse_accept_penalty_coef),
                sa_worse_accept_penalty_power=float(sa_worse_accept_penalty_power),
                reject_penalty=float(reject_penalty),
                reject_worse_penalty_coef=float(reject_worse_penalty_coef),
                reject_worse_penalty_power=float(reject_worse_penalty_power),
                min_stop_iters=int(min_stop_iters),
                terminal_best_coef=float(terminal_best_coef),
                terminal_time_coef=float(terminal_time_coef),
            )
        )
        envs_rnd.append(
            PMALNSEnv(
                scale=str(scale_i),
                seed=int(seed_i),
                instance_id=int(inst_id),
                slack_ratio=float(slack_ratio),
                alns_cfg=alns_cfg,
                action_set=str(action_set),
                reward_scale=float(reward_scale),
                reward_power=float(reward_power),
                best_improve_bonus=float(best_improve_bonus),
                reward_best_coef=float(reward_best_coef),
                reward_accept_coef=float(reward_accept_coef),
                sa_worse_accept_penalty_coef=float(sa_worse_accept_penalty_coef),
                sa_worse_accept_penalty_power=float(sa_worse_accept_penalty_power),
                reject_penalty=float(reject_penalty),
                reject_worse_penalty_coef=float(reject_worse_penalty_coef),
                reject_worse_penalty_power=float(reject_worse_penalty_power),
                min_stop_iters=int(min_stop_iters),
                terminal_best_coef=float(terminal_best_coef),
                terminal_time_coef=float(terminal_time_coef),
            )
        )

    obs_pol = np.stack([e.reset() for e in envs_pol], axis=0).astype(np.float32)
    obs_rnd = np.stack([e.reset() for e in envs_rnd], axis=0).astype(np.float32)
    # Eval uses the same feature preprocessing as rollouts.
    if bool(no_obs_norm) is False:
        obs_pol = _normalize_obs_np(obs_pol)
        obs_rnd = _normalize_obs_np(obs_rnd)

    init_pol = [e.get_state_summary() for e in envs_pol]
    init_rnd = [e.get_state_summary() for e in envs_rnd]
    init_best_pol = np.asarray([s["best_energy"] for s in init_pol], dtype=np.float64)
    init_best_rnd = np.asarray([s["best_energy"] for s in init_rnd], dtype=np.float64)

    rng = np.random.default_rng(int(eval_seed) + 424242)

    alive_pol = np.ones((n,), dtype=bool)
    alive_rnd = np.ones((n,), dtype=bool)

    accepted_pol = feasible_pol = steps_pol = 0
    accepted_rnd = feasible_rnd = steps_rnd = 0

    # Joint action histograms (counts over attempted steps).
    act_counts_pol = np.zeros((int(act_dim_total),), dtype=np.int64)
    act_counts_rnd = np.zeros((int(act_dim_total),), dtype=np.int64)

    net.eval()
    with torch.no_grad():
        for _t in range(T):
            alive_pol_step = alive_pol.copy()
            alive_rnd_step = alive_rnd.copy()

            # Policy actions (vectorized)
            if bool(alive_pol.any()):
                obs_t = torch.from_numpy(obs_pol).to(device)
                logits_list, _v = net(obs_t)
                if bool(deterministic):
                    act_pol = np.stack(
                        [
                            torch.argmax(lg, dim=-1).detach().cpu().numpy()
                            for lg in logits_list
                        ],
                        axis=-1,
                    )
                else:
                    acts = []
                    for lg in logits_list:
                        dist = torch.distributions.Categorical(
                            logits=lg, validate_args=False
                        )
                        acts.append(dist.sample().detach().cpu().numpy())
                    act_pol = np.stack(acts, axis=-1)
            else:
                act_pol = np.zeros((n, H), dtype=np.int64)

            # Random actions (independent uniform over each factor => uniform over joint).
            act_rnd = np.stack(
                [
                    rng.integers(low=0, high=int(d), size=(n,), dtype=np.int64)
                    for d in act_dims
                ],
                axis=-1,
            )

            # Joint action histograms (only count actions for envs that are alive this step).
            if bool(alive_pol_step.any()):
                ji = _flatten_factor_action(act_pol[alive_pol_step], act_dims)
                act_counts_pol += np.bincount(
                    ji.astype(np.int64), minlength=int(act_dim_total)
                )
            if bool(alive_rnd_step.any()):
                ji = _flatten_factor_action(act_rnd[alive_rnd_step], act_dims)
                act_counts_rnd += np.bincount(
                    ji.astype(np.int64), minlength=int(act_dim_total)
                )

            # Step envs
            for i, e in enumerate(envs_pol):
                if not bool(alive_pol[i]):
                    continue
                o2, _r, d, info = e.step(act_pol[i].tolist())
                s = info.get("step", {}) if isinstance(info, dict) else {}
                if bool(s.get("accepted", False)):
                    accepted_pol += 1
                if bool(s.get("feasible_cand", False)):
                    feasible_pol += 1
                steps_pol += 1
                if bool(d):
                    alive_pol[i] = False
                else:
                    obs_pol[i] = np.asarray(o2, dtype=np.float32)

            for i, e in enumerate(envs_rnd):
                if not bool(alive_rnd[i]):
                    continue
                o2, _r, d, info = e.step(act_rnd[i].tolist())
                s = info.get("step", {}) if isinstance(info, dict) else {}
                if bool(s.get("accepted", False)):
                    accepted_rnd += 1
                if bool(s.get("feasible_cand", False)):
                    feasible_rnd += 1
                steps_rnd += 1
                if bool(d):
                    alive_rnd[i] = False
                else:
                    obs_rnd[i] = np.asarray(o2, dtype=np.float32)

            if bool(no_obs_norm) is False:
                if bool(alive_pol.any()):
                    obs_pol = _normalize_obs_np(obs_pol)
                if bool(alive_rnd.any()):
                    obs_rnd = _normalize_obs_np(obs_rnd)

            if (not bool(alive_pol.any())) and (not bool(alive_rnd.any())):
                break

    end_pol = [e.get_state_summary() for e in envs_pol]
    end_rnd = [e.get_state_summary() for e in envs_rnd]
    end_best_pol = np.asarray([s["best_energy"] for s in end_pol], dtype=np.float64)
    end_best_rnd = np.asarray([s["best_energy"] for s in end_rnd], dtype=np.float64)

    # Lower energy is better; report gaps as (random - policy) so positive => policy better.
    gap = end_best_rnd - end_best_pol

    # One-number "is training healthy?" metric: win rate vs random.
    # win = policy strictly better (lower energy) than random.
    # We treat non-finite energies as losses unless only the opponent is non-finite.
    pol_f = np.isfinite(end_best_pol)
    rnd_f = np.isfinite(end_best_rnd)
    win = np.zeros((n,), dtype=bool)
    tie = np.zeros((n,), dtype=bool)
    eps = 1e-9
    for i in range(int(n)):
        if bool(pol_f[i]) and bool(rnd_f[i]):
            if float(end_best_pol[i]) < float(end_best_rnd[i]) - eps:
                win[i] = True
            elif float(end_best_pol[i]) > float(end_best_rnd[i]) + eps:
                win[i] = False
            else:
                tie[i] = True
        elif bool(pol_f[i]) and (not bool(rnd_f[i])):
            win[i] = True
        elif (not bool(pol_f[i])) and bool(rnd_f[i]):
            win[i] = False
        else:
            tie[i] = True

    win_rate = float(np.mean(win))
    tie_rate = float(np.mean(tie))
    loss_rate = float(max(0.0, 1.0 - win_rate - tie_rate))
    improve_pol = init_best_pol - end_best_pol
    improve_rnd = init_best_rnd - end_best_rnd

    pol_p10, pol_p50, pol_p90 = _safe_quantiles(end_best_pol, [0.1, 0.5, 0.9])
    rnd_p10, rnd_p50, rnd_p90 = _safe_quantiles(end_best_rnd, [0.1, 0.5, 0.9])
    gap_p10, gap_p50, gap_p90 = _safe_quantiles(gap, [0.1, 0.5, 0.9])

    imp_pol_p50 = _safe_quantiles(improve_pol, [0.5])[0]
    imp_rnd_p50 = _safe_quantiles(improve_rnd, [0.5])[0]

    pol_act_stats = _action_count_stats(act_counts_pol)
    rnd_act_stats = _action_count_stats(act_counts_rnd)

    return {
        "eval_seed": int(eval_seed),
        "eval_num_envs": int(n),
        "eval_rollout_len": int(T),
        "eval_instance_offset": int(eval_instance_offset),
        "eval_deterministic": bool(deterministic),
        "eval_policy_best_energy_mean": float(_safe_mean(end_best_pol)),
        "eval_policy_best_energy_p10": float(pol_p10),
        "eval_policy_best_energy_p50": float(pol_p50),
        "eval_policy_best_energy_p90": float(pol_p90),
        "eval_random_best_energy_mean": float(_safe_mean(end_best_rnd)),
        "eval_random_best_energy_p10": float(rnd_p10),
        "eval_random_best_energy_p50": float(rnd_p50),
        "eval_random_best_energy_p90": float(rnd_p90),
        "eval_gap_random_minus_policy_mean": float(_safe_mean(gap)),
        "eval_gap_random_minus_policy_p10": float(gap_p10),
        "eval_gap_random_minus_policy_p50": float(gap_p50),
        "eval_gap_random_minus_policy_p90": float(gap_p90),
        "eval_vs_random_win_rate": float(win_rate),
        "eval_vs_random_tie_rate": float(tie_rate),
        "eval_vs_random_loss_rate": float(loss_rate),
        "eval_policy_improve_p50": float(imp_pol_p50),
        "eval_random_improve_p50": float(imp_rnd_p50),
        "eval_policy_accept_rate": float(accepted_pol) / float(max(1, steps_pol)),
        "eval_policy_feasible_rate": float(feasible_pol) / float(max(1, steps_pol)),
        "eval_random_accept_rate": float(accepted_rnd) / float(max(1, steps_rnd)),
        "eval_random_feasible_rate": float(feasible_rnd) / float(max(1, steps_rnd)),
        # Joint action histograms (debugging policy collapse)
        "eval_policy_joint_action_counts": pol_act_stats["counts"],
        "eval_policy_joint_action_max_frac": float(pol_act_stats["max_frac"]),
        "eval_policy_joint_action_entropy": float(pol_act_stats["entropy"]),
        "eval_policy_joint_action_top3": pol_act_stats["top3"],
        "eval_random_joint_action_counts": rnd_act_stats["counts"],
        "eval_random_joint_action_max_frac": float(rnd_act_stats["max_frac"]),
        "eval_random_joint_action_entropy": float(rnd_act_stats["entropy"]),
        "eval_random_joint_action_top3": rnd_act_stats["top3"],
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--scale", type=str, default="medium")
    p.add_argument(
        "--scales",
        type=str,
        default="",
        help=(
            "Optional comma-separated list of scales to mix in one run, e.g. 'small,medium'. "
            "If set, overrides --scale."
        ),
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--slack_ratio", type=float, default=0.25)

    p.add_argument(
        "--action_set",
        type=str,
        default="balanced",
        choices=["balanced", "safe", "full"],
        help=(
            "Discrete ALNS action set for PPO. "
            "balanced: avoids destroy_all but keeps greedy+random repair; "
            "safe: greedy repair only (highest feasibility); "
            "full: includes destroy_all."
        ),
    )

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
        "--resume_policy_only",
        action="store_true",
        help=(
            "When used with --resume, load model weights only (do NOT load optimizer/RNG/update). "
            "Useful when changing reward shaping or other training dynamics and you want a fresh optimizer."
        ),
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

    p.add_argument(
        "--alns_use_torch_batch_dp_in_repair",
        action="store_true",
        help="Enable torch batched DP in greedy repair (faster when there are many insertion candidates).",
    )
    p.add_argument(
        "--alns_torch_batch_dp_always",
        action="store_true",
        help=(
            "Force torch batched DP in greedy repair regardless of candidate count "
            "(sets --alns_torch_batch_min_total_candidates=0 and implies --alns_use_torch_batch_dp_in_repair)."
        ),
    )
    p.add_argument(
        "--alns_torch_dp_device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda", "auto"),
        help=(
            "Device for torch batched DP inside ALNS repair. "
            "Default cpu (recommended for multi-worker rollouts)."
        ),
    )
    p.add_argument(
        "--alns_torch_batch_min_total_candidates",
        type=int,
        default=512,
        help=(
            "Only use batched DP if total insertion candidates >= this threshold. "
            "Set to 0 to always use batched DP."
        ),
    )

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
        "--target_kl",
        type=float,
        default=None,
        help=(
            "Optional PPO early stopping threshold on |approx_kl|. "
            "Helps prevent unstable updates (reward bouncing / entropy collapse). "
            "Suggested: 0.03."
        ),
    )

    p.add_argument(
        "--reward_clip",
        type=float,
        default=10.0,
        help="Clip per-step rewards to [-reward_clip, +reward_clip] to improve stability.",
    )

    p.add_argument(
        "--reward_scale",
        type=float,
        default=1.0,
        help="Multiply environment rewards by this factor (after shaping).",
    )
    p.add_argument(
        "--reward_power",
        type=float,
        default=1.0,
        help=(
            "Apply a signed power transform to non-zero rewards: sign(r)*|r|^reward_power. "
            "Use <1.0 to make small improvements relatively more salient."
        ),
    )
    p.add_argument(
        "--best_improve_bonus",
        type=float,
        default=0.0,
        help=(
            "Add this constant bonus when an action proposes a feasible candidate that improves the best-so-far energy "
            "(even if the candidate was rejected by SA)."
        ),
    )

    p.add_argument(
        "--reward_best_coef",
        type=float,
        default=1.0,
        help=(
            "Coefficient for best-so-far energy improvement shaping: (best_old - best_new). "
            "This aligns training with eval (final best energy). Default 1.0."
        ),
    )
    p.add_argument(
        "--reward_accept_coef",
        type=float,
        default=0.25,
        help=(
            "Coefficient for accepted-move improvement: (cur_old - cur_new) when accepted. "
            "Can be negative if SA accepts worse. Default 0.25."
        ),
    )

    p.add_argument(
        "--sa_worse_accept_penalty_coef",
        type=float,
        default=0.0,
        help=(
            "Optional extra penalty when SA (accept_idx=0) accepts a worse move (delta>0). "
            "Penalty uses a relative delta: coef * ((delta/|cur_energy|) ** power). Default 0 (off)."
        ),
    )
    p.add_argument(
        "--sa_worse_accept_penalty_power",
        type=float,
        default=1.0,
        help=(
            "Power for sa_worse_accept_penalty: coef * ((delta/|cur_energy|) ** power). Default 1.0."
        ),
    )

    p.add_argument(
        "--reject_penalty",
        type=float,
        default=0.0,
        help=(
            "Optional constant penalty applied when a feasible candidate is rejected by SA. "
            "Default 0 (off). Be careful: too large makes policy overly conservative."
        ),
    )
    p.add_argument(
        "--reject_worse_penalty_coef",
        type=float,
        default=0.0,
        help=(
            "Optional penalty coefficient for proposing a *worse* (delta>0) candidate that gets rejected. "
            "Penalty is coef * (delta ** power). Default 0 (off)."
        ),
    )
    p.add_argument(
        "--reject_worse_penalty_power",
        type=float,
        default=1.0,
        help="Power for reject_worse_penalty: coef * (delta ** power). Default 1.0.",
    )

    # SA schedule (used when accept_idx=0). Defaults match ALNSConfig.
    p.add_argument(
        "--sa_tau0",
        type=float,
        default=1.0,
        help="Initial SA temperature (tau0) used when accept_idx=0. Default 1.0.",
    )
    p.add_argument(
        "--sa_decay",
        type=float,
        default=0.995,
        help="SA temperature decay per iteration when accept_idx=0. Default 0.995.",
    )

    # Paper-style: include termination as an action and (optionally) add a terminal reward.
    p.add_argument(
        "--min_stop_iters",
        type=int,
        default=5,
        help=(
            "Minimum ALNS iterations before the policy can end the episode via stop action. "
            "Default 5."
        ),
    )
    p.add_argument(
        "--terminal_best_coef",
        type=float,
        default=0.0,
        help=(
            "Optional terminal reward coefficient: add terminal_best_coef * (init_best - final_best)/|init_best| "
            "when an episode ends. Default 0 (off)."
        ),
    )
    p.add_argument(
        "--terminal_time_coef",
        type=float,
        default=0.0,
        help=(
            "Optional terminal time penalty coefficient: subtract terminal_time_coef * (iters/max_iters) "
            "when an episode ends. Default 0 (off)."
        ),
    )
    p.add_argument(
        "--no_obs_norm",
        action="store_true",
        help="Disable observation normalization (not recommended).",
    )

    # Periodic evaluation against a uniform-random policy (fixed instance set).
    p.add_argument(
        "--eval_every",
        type=int,
        default=0,
        help="If >0, run evaluation every N updates (logs policy vs random).",
    )
    p.add_argument(
        "--eval_seed",
        type=int,
        default=1337,
        help="Seed used for evaluation instance generation and random baseline.",
    )
    p.add_argument(
        "--eval_num_envs",
        type=int,
        default=32,
        help="Number of fixed evaluation instances per eval step.",
    )
    p.add_argument(
        "--eval_rollout_len",
        type=int,
        default=128,
        help="Number of ALNS steps to run during evaluation.",
    )
    p.add_argument(
        "--eval_instance_offset",
        type=int,
        default=9000000,
        help="Base instance_id for eval instances (instance_id = offset + i).",
    )
    p.add_argument(
        "--eval_stochastic",
        action="store_true",
        help="Use stochastic sampling for the policy during eval (default: deterministic argmax).",
    )

    p.add_argument(
        "--eval_health_threshold",
        type=float,
        default=0.55,
        help=(
            "Threshold for eval health metric vs random. "
            "Health metric is eval_vs_random_win_rate (fraction of eval instances where policy beats random). "
            "Suggested: 0.55 (better than random with margin)."
        ),
    )

    p.add_argument(
        "--save_best",
        action="store_true",
        help=(
            "If set, save a *_best.pt checkpoint whenever the chosen eval metric improves. "
            "Requires --eval_every > 0."
        ),
    )
    p.add_argument(
        "--best_metric",
        type=str,
        default="eval_policy_best_energy_p50",
        choices=(
            "eval_policy_best_energy_p50",
            "eval_policy_best_energy_mean",
            "eval_gap_random_minus_policy_p50",
            "eval_gap_random_minus_policy_mean",
        ),
        help=(
            "Metric used to decide whether to save *_best.pt. "
            "Energy metrics are minimized; gap metrics are maximized."
        ),
    )

    return p


def _atomic_torch_save(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    # PyTorch 2.6 changed torch.load default weights_only from False -> True.
    # Our checkpoints intentionally store optimizer/RNG state (and are produced by this script),
    # so we need a full load. We explicitly request weights_only=False when supported.
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # Older torch versions don't have the weights_only kwarg.
        ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"Invalid checkpoint file: {path}")
    return ckpt


def main() -> None:
    args = build_parser().parse_args()

    scales_raw = str(getattr(args, "scales", "") or "").strip()
    if scales_raw:
        scales_list = [s.strip().lower() for s in scales_raw.split(",") if s.strip()]
    else:
        scales_list = [str(args.scale).strip().lower()]

    action_set = (
        str(getattr(args, "action_set", "balanced") or "balanced").strip().lower()
    )

    out_dir = str(args.out_dir)
    run_name = str(args.run_name).strip()
    if not run_name:
        run_name = f"ppo_alns_pm_{args.scale}_seed{args.seed}"
    os.makedirs(out_dir, exist_ok=True)
    ckpt_latest = os.path.join(out_dir, f"{run_name}_latest.pt")
    ckpt_best = os.path.join(out_dir, f"{run_name}_best.pt")
    metrics_path = str(args.metrics_jsonl).strip()
    if not metrics_path:
        metrics_path = os.path.join(out_dir, f"{run_name}_metrics.jsonl")
    metrics_path_p = Path(metrics_path)

    if int(args.workers) <= 0:
        workers = max(1, os.cpu_count() or 1)
    else:
        workers = int(args.workers)

    # obs_dim / action dimensions are determined by the env.
    tmp_env = PMALNSEnv(
        scale=str(scales_list[0]),
        seed=int(args.seed),
        instance_id=0,
        slack_ratio=float(args.slack_ratio),
        alns_cfg=ALNSConfig(max_iters=1, no_improve_limit=1),
        action_set=str(action_set),
    )
    obs_dim = int(tmp_env.obs_dim)
    act_dims = list(getattr(tmp_env, "action_dims", [int(tmp_env.action_dim)]))
    act_dim_total = int(np.prod(np.asarray(act_dims, dtype=np.int64)))

    if str(args.device) == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    random.seed(int(args.seed))

    net = PolicyValueNet(
        obs_dim=obs_dim, act_dims=act_dims, hidden=int(args.hidden)
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=float(args.lr))

    ppo = PPOCfg(
        lr=float(args.lr),
        entropy_coef=float(args.entropy),
        clip_eps=float(args.clip),
        ppo_epochs=int(args.ppo_epochs),
        minibatches=int(args.minibatches),
        target_kl=(None if args.target_kl is None else float(args.target_kl)),
    )

    alns_cfg_dict = {
        "max_iters": int(args.max_iters),
        "no_improve_limit": int(args.no_improve),
        "destroy_frac": float(args.destroy_frac),
        "destroy_max": int(args.destroy_max),
        "sa_tau0": float(getattr(args, "sa_tau0", 1.0)),
        "sa_decay": float(getattr(args, "sa_decay", 0.995)),
        "keep_iter_log": False,
        "oracle_dataset": False,
        "verify_cost": False,
    }

    use_torch_batch_dp = bool(
        getattr(args, "alns_use_torch_batch_dp_in_repair", False)
    ) or bool(getattr(args, "alns_torch_batch_dp_always", False))
    if use_torch_batch_dp:
        alns_cfg_dict["use_torch_batch_dp_in_repair"] = True
        alns_cfg_dict["torch_dp_device"] = str(args.alns_torch_dp_device)

        if bool(getattr(args, "alns_torch_batch_dp_always", False)):
            min_total_candidates = 0
        else:
            min_total_candidates = int(args.alns_torch_batch_min_total_candidates)

        alns_cfg_dict["torch_batch_min_total_candidates"] = max(
            0, int(min_total_candidates)
        )

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

    best_metric_value: float | None = None

    if str(args.resume).strip():
        ckpt_path = str(args.resume).strip()
        ckpt = _load_checkpoint(ckpt_path, device=device)
        net.load_state_dict(ckpt["state_dict"])
        if not bool(getattr(args, "resume_policy_only", False)):
            if "opt_state" in ckpt and isinstance(ckpt["opt_state"], dict):
                opt.load_state_dict(ckpt["opt_state"])
            if "update" in ckpt:
                start_update = int(ckpt["update"])
            if "best_metric_value" in ckpt:
                try:
                    v = ckpt.get("best_metric_value")
                    best_metric_value = None if v is None else float(v)
                except Exception:
                    best_metric_value = None
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
        else:
            start_update = 0
            best_metric_value = None
            print(f"Loaded weights from {ckpt_path} (policy-only resume)")

    # If we weren't able to restore best metric from the resumed checkpoint,
    # try restoring it from the best checkpoint file.
    if best_metric_value is None and bool(getattr(args, "save_best", False)):
        try:
            if os.path.exists(ckpt_best):
                bckpt = _load_checkpoint(ckpt_best, device=torch.device("cpu"))
                v = bckpt.get("best_metric_value")
                best_metric_value = None if v is None else float(v)
        except Exception:
            pass

    print(
        "Training PPO "
        f"device={device.type} obs_dim={obs_dim} act_dims={act_dims} act_dim_total={act_dim_total} "
        f"workers={workers} envs/worker={envs_per_worker} scales={','.join(scales_list)} action_set={action_set}",
        flush=True,
    )
    if bool(alns_cfg_dict.get("use_torch_batch_dp_in_repair", False)):
        print(
            "ALNS: torch_batched_dp_in_repair=1 "
            f"torch_dp_device={alns_cfg_dict.get('torch_dp_device')} "
            f"min_total_candidates={alns_cfg_dict.get('torch_batch_min_total_candidates')}",
            flush=True,
        )
    else:
        print("ALNS: torch_batched_dp_in_repair=0", flush=True)
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
            "act_dims": [int(x) for x in act_dims],
            "act_dim_total": int(act_dim_total),
            "obs_dim": int(obs_dim),
            "update": int(update),
            "args": vars(args),
            "last_metrics": last_metrics,
            "best_metric": str(
                getattr(args, "best_metric", "eval_policy_best_energy_p50")
            ),
            "best_metric_value": best_metric_value,
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

    def _save_best_checkpoint(update: int, metric_value: float) -> str:
        payload: Dict[str, Any] = {
            "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
            "opt_state": opt.state_dict(),
            "act_dims": [int(x) for x in act_dims],
            "act_dim_total": int(act_dim_total),
            "obs_dim": int(obs_dim),
            "update": int(update),
            "args": vars(args),
            "last_metrics": last_metrics,
            "best_metric": str(
                getattr(args, "best_metric", "eval_policy_best_energy_p50")
            ),
            "best_metric_value": float(metric_value),
            "py_random_state": random.getstate(),
            "np_random_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            try:
                payload["torch_cuda_rng_state"] = torch.cuda.get_rng_state_all()
            except Exception:
                pass
        _atomic_torch_save(payload, ckpt_best)
        return ckpt_best

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
                        act_dims=[int(x) for x in act_dims],
                        hidden=int(args.hidden),
                        scales=scales_list,
                        action_set=str(action_set),
                        base_seed=int(args.seed) + 1000 * upd,
                        start_instance_id=int(base_instance) + 100000 * upd,
                        envs_per_worker=int(envs_per_worker),
                        rollout_len=int(rollout_len),
                        slack_ratio=float(args.slack_ratio),
                        reward_scale=float(args.reward_scale),
                        reward_power=float(args.reward_power),
                        best_improve_bonus=float(args.best_improve_bonus),
                        reward_best_coef=float(getattr(args, "reward_best_coef", 1.0)),
                        reward_accept_coef=float(
                            getattr(args, "reward_accept_coef", 0.25)
                        ),
                        sa_worse_accept_penalty_coef=float(
                            getattr(args, "sa_worse_accept_penalty_coef", 0.0)
                        ),
                        sa_worse_accept_penalty_power=float(
                            getattr(args, "sa_worse_accept_penalty_power", 1.0)
                        ),
                        reject_penalty=float(getattr(args, "reject_penalty", 0.0)),
                        reject_worse_penalty_coef=float(
                            getattr(args, "reject_worse_penalty_coef", 0.0)
                        ),
                        reject_worse_penalty_power=float(
                            getattr(args, "reject_worse_penalty_power", 1.0)
                        ),
                        min_stop_iters=int(getattr(args, "min_stop_iters", 5)),
                        terminal_best_coef=float(
                            getattr(args, "terminal_best_coef", 0.0)
                        ),
                        terminal_time_coef=float(
                            getattr(args, "terminal_time_coef", 0.0)
                        ),
                        no_obs_norm=bool(getattr(args, "no_obs_norm", False)),
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

            rollout_joint_action_counts = np.sum(
                [
                    np.asarray(
                        b.get(
                            "rollout_joint_action_counts",
                            np.zeros((int(act_dim_total),), dtype=np.int64),
                        ),
                        dtype=np.int64,
                    )
                    for b in batches
                ],
                axis=0,
            ).astype(np.int64)

            T, B, _ = obs.shape

            # Safety: sanitize/clip rewards; NaNs in rewards/returns can blow up PPO.
            rewards = _torch_sanitize(rewards)
            rclip = float(args.reward_clip)
            if rclip > 0:
                rewards = torch.clamp(rewards, -rclip, rclip)

            # IMPORTANT: obs and logp_old must be consistent.
            # Rollout workers already apply the same preprocessing used during action sampling.
            # Do NOT renormalize here, or approx_kl/clip_frac will explode and trigger target_kl early-stop.

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
            act_f = actions.reshape(T * B, int(len(act_dims)))
            logp_old_f = logp_old.reshape(T * B)
            adv_f = adv.reshape(T * B)
            ret_f = returns.reshape(T * B)

            batch_size = T * B
            mb = max(1, int(batch_size // int(ppo.minibatches)))

            net.train()
            opt_state_before = copy.deepcopy(opt.state_dict())
            bad_update = False

            # PPO diagnostics (to verify that learning is actually happening).
            diag_policy_loss = 0.0
            diag_value_loss = 0.0
            diag_entropy = 0.0
            diag_approx_kl = 0.0
            diag_clip_frac = 0.0
            diag_mb = 0

            early_stop = False

            for _ in range(int(ppo.ppo_epochs)):
                idx = torch.randperm(batch_size)
                for start in range(0, batch_size, mb):
                    j = idx[start : start + mb]
                    logits_list, v = net(obs_f[j])
                    if not (
                        all(torch.isfinite(lg).all() for lg in logits_list)
                        and torch.isfinite(v).all()
                    ):
                        bad_update = True
                        break
                    logp, ent = _multi_categorical_logprob(logits_list, act_f[j])
                    log_ratio = torch.clamp(logp - logp_old_f[j], -20.0, 20.0)
                    ratio = torch.exp(log_ratio)
                    surr1 = ratio * adv_f[j]
                    surr2 = (
                        torch.clamp(ratio, 1.0 - ppo.clip_eps, 1.0 + ppo.clip_eps)
                        * adv_f[j]
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = ((v - ret_f[j]) ** 2).mean()
                    entropy = ent.mean()

                    # Approx KL and clip fraction are standard PPO stability signals.
                    approx_kl = (logp_old_f[j] - logp).mean()
                    clip_frac = (
                        (torch.abs(ratio - 1.0) > float(ppo.clip_eps)).float().mean()
                    )

                    # Record diagnostics even if we early-stop before stepping.
                    # This avoids NaNs in logs when target_kl triggers on the first minibatch.
                    try:
                        diag_policy_loss += float(policy_loss.detach().cpu().item())
                        diag_value_loss += float(value_loss.detach().cpu().item())
                        diag_entropy += float(entropy.detach().cpu().item())
                        diag_approx_kl += float(approx_kl.detach().cpu().item())
                        diag_clip_frac += float(clip_frac.detach().cpu().item())
                        diag_mb += 1
                    except Exception:
                        pass

                    # Early stop if KL indicates the update is getting too large.
                    # Use absolute value because this approx can be slightly negative due to noise.
                    if ppo.target_kl is not None:
                        try:
                            if float(
                                torch.abs(approx_kl).detach().cpu().item()
                            ) > float(ppo.target_kl):
                                early_stop = True
                                break
                        except Exception:
                            pass

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

                if bad_update or early_stop:
                    break

            # If we early-stopped, we keep the partial update (standard PPO practice)
            # but report it as a stability signal.

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

            if diag_mb > 0:
                diag_policy_loss /= float(diag_mb)
                diag_value_loss /= float(diag_mb)
                diag_entropy /= float(diag_mb)
                diag_approx_kl /= float(diag_mb)
                diag_clip_frac /= float(diag_mb)
            else:
                diag_policy_loss = float("nan")
                diag_value_loss = float("nan")
                diag_entropy = float("nan")
                diag_approx_kl = float("nan")
                diag_clip_frac = float("nan")

            avg_r = float(rewards.mean().item())

            # Optional periodic evaluation (policy vs uniform-random) on a fixed instance set.
            eval_metrics: Dict[str, Any] | None = None
            if int(getattr(args, "eval_every", 0)) > 0 and (
                int(upd + 1) % int(args.eval_every) == 0
            ):
                t_eval0 = time.time()
                try:
                    eval_metrics = _eval_policy_vs_random(
                        net=net,
                        act_dims=[int(x) for x in act_dims],
                        scales=scales_list,
                        action_set=str(action_set),
                        alns_cfg_dict=alns_cfg_dict,
                        slack_ratio=float(args.slack_ratio),
                        reward_scale=float(args.reward_scale),
                        reward_power=float(args.reward_power),
                        best_improve_bonus=float(args.best_improve_bonus),
                        reward_best_coef=float(getattr(args, "reward_best_coef", 1.0)),
                        reward_accept_coef=float(
                            getattr(args, "reward_accept_coef", 0.25)
                        ),
                        sa_worse_accept_penalty_coef=float(
                            getattr(args, "sa_worse_accept_penalty_coef", 0.0)
                        ),
                        sa_worse_accept_penalty_power=float(
                            getattr(args, "sa_worse_accept_penalty_power", 1.0)
                        ),
                        reject_penalty=float(getattr(args, "reject_penalty", 0.0)),
                        reject_worse_penalty_coef=float(
                            getattr(args, "reject_worse_penalty_coef", 0.0)
                        ),
                        reject_worse_penalty_power=float(
                            getattr(args, "reject_worse_penalty_power", 1.0)
                        ),
                        min_stop_iters=int(getattr(args, "min_stop_iters", 5)),
                        terminal_best_coef=float(
                            getattr(args, "terminal_best_coef", 0.0)
                        ),
                        terminal_time_coef=float(
                            getattr(args, "terminal_time_coef", 0.0)
                        ),
                        no_obs_norm=bool(getattr(args, "no_obs_norm", False)),
                        eval_seed=int(args.eval_seed),
                        eval_num_envs=int(args.eval_num_envs),
                        eval_rollout_len=int(args.eval_rollout_len),
                        eval_instance_offset=int(args.eval_instance_offset),
                        deterministic=(
                            not bool(getattr(args, "eval_stochastic", False))
                        ),
                        device=device,
                    )
                    eval_metrics["eval_seconds"] = float(time.time() - t_eval0)

                    # One-number health metric you can watch.
                    try:
                        health = float(eval_metrics.get("eval_vs_random_win_rate"))
                    except Exception:
                        health = float("nan")
                    thr = float(getattr(args, "eval_health_threshold", 0.55))
                    eval_metrics["eval_health"] = float(health)
                    eval_metrics["eval_health_threshold"] = float(thr)
                    eval_metrics["eval_health_pass"] = bool(
                        math.isfinite(health) and (health >= thr)
                    )

                    print(
                        "[Eval] "
                        f"policy_p50={eval_metrics.get('eval_policy_best_energy_p50'):.1f} "
                        f"random_p50={eval_metrics.get('eval_random_best_energy_p50'):.1f} "
                        f"gap_p50(random-policy)={eval_metrics.get('eval_gap_random_minus_policy_p50'):.2f} "
                        f"win_rate={eval_metrics.get('eval_vs_random_win_rate'):.3f} "
                        f"health_pass={int(bool(eval_metrics.get('eval_health_pass', False)))} "
                        f"secs={eval_metrics.get('eval_seconds'):.1f}",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"[Eval] failed: {type(e).__name__}: {e}",
                        flush=True,
                    )

            # Save best checkpoint if requested (based on eval metrics).
            if bool(getattr(args, "save_best", False)) and isinstance(
                eval_metrics, dict
            ):
                metric_key = str(
                    getattr(args, "best_metric", "eval_policy_best_energy_p50")
                )
                try:
                    metric_val = float(eval_metrics.get(metric_key, float("nan")))
                except Exception:
                    metric_val = float("nan")

                if math.isfinite(metric_val):
                    maximize = metric_key.startswith("eval_gap_")
                    improved = False
                    if best_metric_value is None or not math.isfinite(
                        best_metric_value
                    ):
                        improved = True
                    else:
                        if maximize:
                            improved = metric_val > float(best_metric_value)
                        else:
                            improved = metric_val < float(best_metric_value)

                    if improved:
                        best_metric_value = float(metric_val)
                        p_best = _save_best_checkpoint(
                            update=upd + 1, metric_value=metric_val
                        )
                        print(
                            f"[Best] Saved {os.path.basename(p_best)} ({metric_key}={metric_val:.4f})",
                            flush=True,
                        )

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
                f"ent={diag_entropy:.3f} kl={diag_approx_kl:.4f} clip={diag_clip_frac:.3f} "
                f"bestE(p10/p50/p90)={eb_p10:.1f}/{eb_p50:.1f}/{eb_p90:.1f} "
                f"bestImprove(p10/p50/p90)={bi_p10:.2f}/{bi_p50:.2f}/{bi_p90:.2f}"
            )
            print(msg, flush=True)

            # Reward diagnostics: avg_step_reward can look "low" because reward is sparse
            # (0 on reject) and measured in raw energy units.
            r_np = rewards.detach().cpu().numpy().reshape(-1)
            r_f = r_np[np.isfinite(r_np)]
            if r_f.size > 0:
                r_mean = float(np.mean(r_f))
                r_p10, r_p50, r_p90 = (
                    float(x) for x in np.quantile(r_f, [0.1, 0.5, 0.9])
                )
                r_nonzero = r_f[r_f != 0.0]
                r_nonzero_rate = float(r_nonzero.size / float(r_f.size))
                r_nonzero_mean = float(np.mean(r_nonzero)) if r_nonzero.size else 0.0
                r_pos_rate = float(np.mean(r_f > 0.0))
                r_neg_rate = float(np.mean(r_f < 0.0))
            else:
                r_mean = float("nan")
                r_p10 = r_p50 = r_p90 = float("nan")
                r_nonzero_rate = float("nan")
                r_nonzero_mean = float("nan")
                r_pos_rate = float("nan")
                r_neg_rate = float("nan")

            # Append JSONL metrics for offline plotting.
            rec = {
                "time": time.time(),
                "update": int(upd + 1),
                "avg_step_reward": float(avg_r),
                "ppo_policy_loss": float(diag_policy_loss),
                "ppo_value_loss": float(diag_value_loss),
                "ppo_entropy": float(diag_entropy),
                "ppo_approx_kl": float(diag_approx_kl),
                "ppo_clip_frac": float(diag_clip_frac),
                "ppo_target_kl": (
                    None if ppo.target_kl is None else float(ppo.target_kl)
                ),
                "ppo_early_stop": bool(early_stop),
                "reward_mean": float(r_mean),
                "reward_p10": float(r_p10),
                "reward_p50": float(r_p50),
                "reward_p90": float(r_p90),
                "reward_nonzero_rate": float(r_nonzero_rate),
                "reward_nonzero_mean": float(r_nonzero_mean),
                "reward_pos_rate": float(r_pos_rate),
                "reward_neg_rate": float(r_neg_rate),
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

            # Rollout joint-action histogram (debug policy collapse / operator preference).
            try:
                act_stats = _action_count_stats(rollout_joint_action_counts)
                rec["rollout_joint_action_counts"] = act_stats["counts"]
                rec["rollout_joint_action_max_frac"] = float(act_stats["max_frac"])
                rec["rollout_joint_action_entropy"] = float(act_stats["entropy"])
                rec["rollout_joint_action_top3"] = act_stats["top3"]
            except Exception:
                pass

            rec["best_metric"] = str(
                getattr(args, "best_metric", "eval_policy_best_energy_p50")
            )
            rec["best_metric_value"] = (
                None if best_metric_value is None else float(best_metric_value)
            )

            if isinstance(eval_metrics, dict):
                rec.update(eval_metrics)
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
            "act_dims": [int(x) for x in act_dims],
            "act_dim_total": int(act_dim_total),
        },
        out_final,
    )
    print("Saved", out_final, flush=True)


if __name__ == "__main__":
    main()

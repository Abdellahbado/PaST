"""Simulation-guided Beam Search (SGBS) for PaST-SM.

Implements Algorithm 1 from:
  "Simulation-guided Beam Search for Neural Combinatorial Optimization" (NeurIPS 2022)

Paper-faithful behavior:
- Expansion: for each beam node, choose top-γ actions by policy probability (logit order).
- Simulation: single greedy rollout from each child.
- Pruning: keep top-β children by rollout return (NOT cumulative log-probability).

This module is inference-only and does not touch TrainingEnv.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from PaST.past_sm_model import PaSTSMNet
from PaST.sm_env import GPUBatchSingleMachinePeriodEnv


@dataclass(frozen=True)
class EnvState:
    """Minimal dynamic state needed to branch search in GPUBatchSingleMachinePeriodEnv."""

    t: Tensor
    job_available: Tensor
    total_energy: Tensor
    done_mask: Tensor

    def clone_detached(self) -> "EnvState":
        return EnvState(
            t=self.t.clone(),
            job_available=self.job_available.clone(),
            total_energy=self.total_energy.clone(),
            done_mask=self.done_mask.clone(),
        )


@dataclass(frozen=True)
class DecodeResult:
    """Result of decoding a single instance."""

    total_energy: float
    total_return: float
    actions: Optional[List[int]] = None


@dataclass(frozen=True)
class BeamNode:
    state: EnvState
    actions: List[int]


def _slice_single_instance(batch_data: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Take a batched (B, ...) batch_data dict and return a single-instance (1, ...) view."""
    out: Dict[str, Any] = {}
    for k, v in batch_data.items():
        if isinstance(v, np.ndarray):
            out[k] = v[index : index + 1]
        elif torch.is_tensor(v):
            out[k] = v[index : index + 1]
        else:
            raise TypeError(f"Unsupported batch_data type for key={k}: {type(v)}")
    return out


def _repeat_batch_data(
    single_batch_data: Dict[str, Any], repeat: int
) -> Dict[str, Any]:
    """Repeat a (1, ...) batch_data dict into (repeat, ...) without changing values."""
    if repeat <= 0:
        raise ValueError("repeat must be positive")

    out: Dict[str, Any] = {}
    for k, v in single_batch_data.items():
        if isinstance(v, np.ndarray):
            out[k] = np.repeat(v, repeat, axis=0)
        elif torch.is_tensor(v):
            out[k] = v.repeat((repeat,) + (1,) * (v.dim() - 1))
        else:
            raise TypeError(f"Unsupported batch_data type for key={k}: {type(v)}")
    return out


def get_state(env: GPUBatchSingleMachinePeriodEnv) -> EnvState:
    return EnvState(
        t=env.t,
        job_available=env.job_available,
        total_energy=env.total_energy,
        done_mask=env.done_mask,
    ).clone_detached()


def set_state(env: GPUBatchSingleMachinePeriodEnv, state: EnvState) -> None:
    env.t.copy_(state.t)
    env.job_available.copy_(state.job_available)
    env.total_energy.copy_(state.total_energy)
    env.done_mask.copy_(state.done_mask)


def _prepare_model_masks(
    obs: Dict[str, Tensor],
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """Convert env float masks (1=valid) into model bool masks (True=invalid)."""
    job_mask = obs.get("job_mask")
    period_mask = obs.get("period_mask")
    period_full_mask = obs.get("period_full_mask")

    if job_mask is not None and job_mask.dtype is not torch.bool:
        job_mask = job_mask < 0.5
    if period_mask is not None and period_mask.dtype is not torch.bool:
        period_mask = period_mask < 0.5
    if period_full_mask is not None and period_full_mask.dtype is not torch.bool:
        period_full_mask = period_full_mask < 0.5

    return job_mask, period_mask, period_full_mask


@torch.no_grad()
def compute_masked_logits(model: PaSTSMNet, obs: Dict[str, Tensor]) -> Tensor:
    """Compute action logits with env action_mask applied."""
    job_mask, period_mask, period_full_mask = _prepare_model_masks(obs)

    periods_full = obs.get("periods_full")
    if (
        getattr(getattr(model, "model_config", None), "use_global_horizon", False)
        and periods_full is None
    ):
        periods_full = obs["periods"]
        period_full_mask = period_mask

    logits, _ = model(
        jobs=obs["jobs"],
        periods_local=obs["periods"],
        ctx=obs["ctx"],
        job_mask=job_mask,
        period_mask=period_mask,
        periods_full=periods_full,
        period_full_mask=period_full_mask,
    )

    if "action_mask" in obs:
        action_mask = obs["action_mask"]
        logits = logits.masked_fill(action_mask == 0, float("-inf"))

    return logits


def _select_topk_valid_actions(logits_1d: Tensor, k: int) -> List[int]:
    """Select up to k actions with finite logits."""
    if k <= 0:
        return []

    finite = torch.isfinite(logits_1d)
    if not finite.any():
        return []

    k_eff = min(k, int(finite.sum().item()))
    topk = torch.topk(logits_1d, k=k_eff, dim=-1)
    idx = topk.indices.detach().cpu().tolist()
    # torch.topk will only consider finite entries if k_eff was computed from finite.sum()
    return [int(x) for x in idx]


def _completion_feasible_action_mask(
    env: GPUBatchSingleMachinePeriodEnv, obs: Dict[str, Tensor]
) -> Tensor:
    """Extra action mask: keep only actions that still allow scheduling all remaining work.

    The env's built-in `action_mask` only enforces immediate feasibility (chosen job finishes
    before T_limit). It does not prevent choosing slack that leaves insufficient time
    for the remaining unscheduled jobs.

    This mask enforces a necessary-and-sufficient condition for eventual feasibility in
    the single-machine setting: after taking an action, the remaining processing time must
    still fit in the remaining horizon.

    Returns:
        (B, action_dim) float mask (1=valid, 0=invalid)
    """
    if "action_mask" not in obs:
        # If env doesn't provide an action mask, don't restrict further.
        return torch.ones((env.batch_size, env.action_dim), device=env.device)

    base_mask = obs["action_mask"].float()

    # Remaining work in slots (integer).
    p_all = env.p_subset.to(torch.int32)  # (B, N)
    job_avail = env.job_available  # float 0/1
    remaining_work = (p_all.float() * job_avail).sum(dim=1).to(torch.int32)  # (B,)
    T_limit = env.T_limit.to(torch.int32)  # (B,)

    B = env.batch_size
    N = env.N_job_pad

    if getattr(env.env_config, "use_price_families", False):
        # In this variant, slack_id == family_id.
        F = env.num_slack_choices
        T = env.T_max_pad

        slot_families = env._compute_slot_families()  # (B, T)
        slot_indices = torch.arange(T, device=env.device, dtype=torch.int32).unsqueeze(
            0
        )  # (1, T)
        valid_start = slot_indices >= env.t.to(torch.int32).unsqueeze(1)  # (B, T)

        # end_times[b, j, s] = s + p[b, j]
        end_times = slot_indices.unsqueeze(1) + p_all.unsqueeze(2)  # (B, N, T)
        feasible = end_times <= T_limit.unsqueeze(1).unsqueeze(2)
        feasible = feasible & valid_start.unsqueeze(1)

        job_mask_bool = job_avail.unsqueeze(2).bool()  # (B, N, 1)

        large = torch.tensor(T + 1, device=env.device, dtype=torch.int32)
        mask = torch.zeros((B, N, F), dtype=torch.float32, device=env.device)

        # For each family, compute the earliest feasible start time for each job.
        for f in range(int(F)):
            in_family = (slot_families == int(f)).unsqueeze(1)  # (B, 1, T)
            candidates = feasible & in_family
            # start_time = min s where candidates True, else large
            start_times = (
                torch.where(
                    candidates,
                    slot_indices.unsqueeze(1).expand(B, N, T),
                    large,
                )
                .min(dim=2)
                .values
            )  # (B, N)

            end_t = start_times + p_all  # (B, N)
            remaining_after = remaining_work.unsqueeze(1) - p_all  # (B, N)
            completion_ok = (end_t + remaining_after) <= T_limit.unsqueeze(1)
            # If no candidate, start_times==large -> completion_ok False
            valid = completion_ok & job_mask_bool.squeeze(2)
            mask[:, :, f] = valid.float()

        mask = mask.reshape(B, N * F)
        return base_mask * mask

    # Default slack variants: slack_id indexes a discrete start-time mapping.
    S = env.num_slack_choices
    slack_start_times = env._compute_all_slack_start_times().to(torch.int32)  # (B, S)
    start_expanded = slack_start_times.unsqueeze(1)  # (B, 1, S)
    end_times = start_expanded + p_all.unsqueeze(2)  # (B, N, S)

    remaining_after = remaining_work.unsqueeze(1).unsqueeze(2) - p_all.unsqueeze(
        2
    )  # (B, N, 1)
    completion_ok = (end_times + remaining_after) <= T_limit.unsqueeze(1).unsqueeze(2)
    valid = completion_ok & job_avail.unsqueeze(2).bool()
    mask = valid.reshape(B, N * S).float()

    return base_mask * mask


def _max_wait_action_mask(
    env: GPUBatchSingleMachinePeriodEnv,
    max_wait_slots: int,
) -> Tensor:
    """Extra action mask to cap how far the policy may "wait" via slack.

    This is an inference-time constraint. It is useful to prevent the decoder from
    jumping deep into the horizon and implicitly committing to a late schedule.

    Notes:
    - For non-family slack variants, slack choice maps to a concrete start time.
    - For price-family variants, slack_id is not a time; we currently do not
      apply this mask.
    """
    if max_wait_slots < 0:
        raise ValueError("max_wait_slots must be >= 0")

    if getattr(env.env_config, "use_price_families", False):
        return torch.ones((env.batch_size, env.action_dim), device=env.device)

    B = env.batch_size
    N = env.N_job_pad
    S = env.num_slack_choices

    slack_start_times = env._compute_all_slack_start_times().to(torch.int32)  # (B, S)
    wait = (slack_start_times - env.t.to(torch.int32).unsqueeze(1)).clamp_min(
        0
    )  # (B,S)
    slack_ok = (wait <= int(max_wait_slots)).to(torch.float32)  # (B,S)

    # Broadcast (B,S) -> (B,N,S) -> (B,N*S)
    return slack_ok.unsqueeze(1).expand(B, N, S).reshape(B, N * S)


def _apply_wait_logit_penalty(
    logits: Tensor,
    env: GPUBatchSingleMachinePeriodEnv,
    wait_logit_penalty: float,
) -> Tensor:
    """Subtract a penalty from logits proportional to slack-induced waiting.

    This is a decoding-time heuristic (no retraining) that biases toward earlier
    starts while still allowing waiting when the model is confident.
    """
    if wait_logit_penalty <= 0:
        return logits

    if getattr(env.env_config, "use_price_families", False):
        # Family slack doesn't map directly to a start time.
        return logits

    B = env.batch_size
    N = env.N_job_pad
    S = env.num_slack_choices

    slack_start_times = env._compute_all_slack_start_times().to(torch.float32)  # (B,S)
    t_now = env.t.to(torch.float32).unsqueeze(1)  # (B,1)
    wait = (slack_start_times - t_now).clamp_min(0.0)  # (B,S)

    penalty = float(wait_logit_penalty) * wait  # (B,S)
    penalty = penalty.unsqueeze(1).expand(B, N, S).reshape(B, N * S)

    return logits - penalty


@torch.no_grad()
def greedy_rollout(
    env: GPUBatchSingleMachinePeriodEnv,
    model: PaSTSMNet,
    max_steps: Optional[int] = None,
    record_actions: bool = False,
    infeasible_penalty: float = 1e9,
    max_wait_slots: Optional[int] = None,
    wait_logit_penalty: float = 0.0,
    makespan_penalty: float = 0.0,
) -> Tuple[Tensor, Tensor, Optional[List[List[int]]]]:
    """Greedy decode until done for a batch env.

    Returns:
        total_return: (B,) return
        total_energy: (B,) total energy
        actions: optional list (B) of action traces
    """
    model.eval()

    B = env.batch_size
    if max_steps is None:
        max_steps = int(env.N_job_pad) + 5

    actions_trace: Optional[List[List[int]]] = None
    if record_actions:
        actions_trace = [[] for _ in range(B)]

    obs = env._get_obs()
    done_mask = env.done_mask.clone()
    infeasible_mask = torch.zeros((B,), dtype=torch.bool, device=done_mask.device)

    for _ in range(int(max_steps)):
        if done_mask.all():
            break

        logits = compute_masked_logits(model, obs)
        logits = logits.masked_fill(
            _completion_feasible_action_mask(env, obs) == 0, float("-inf")
        )

        if max_wait_slots is not None:
            logits = logits.masked_fill(
                _max_wait_action_mask(env, int(max_wait_slots)) == 0, float("-inf")
            )

        logits = _apply_wait_logit_penalty(logits, env, wait_logit_penalty)

        all_masked = ~torch.isfinite(logits).any(dim=-1)

        newly_infeasible = all_masked & ~done_mask
        if newly_infeasible.any():
            done_mask = done_mask | newly_infeasible
            env.done_mask = env.done_mask | newly_infeasible
            infeasible_mask = infeasible_mask | newly_infeasible

        if done_mask.all():
            break

        actions = logits.argmax(dim=-1)
        actions = actions.masked_fill(done_mask, 0)

        if actions_trace is not None:
            for i in range(B):
                if not bool(done_mask[i]):
                    actions_trace[i].append(int(actions[i].item()))

        obs, rewards, dones, _info = env.step(actions)
        done_mask = done_mask | dones

    total_energy = env.total_energy.clone()
    if infeasible_mask.any():
        penalty = torch.tensor(float(infeasible_penalty), device=total_energy.device)
        total_energy = total_energy + infeasible_mask.to(total_energy.dtype) * penalty
        env.total_energy.add_(infeasible_mask.to(env.total_energy.dtype) * penalty)

    # Keep `total_energy` as the true objective for reporting.
    # `total_return` may optionally include an inference-time makespan regularizer.
    total_return = -total_energy
    if makespan_penalty > 0:
        total_return = total_return - float(makespan_penalty) * env.t.to(
            total_return.dtype
        )

    return total_return, total_energy, actions_trace


@torch.no_grad()
def sgbs(
    model: PaSTSMNet,
    env_config,
    device: torch.device,
    batch_data: Dict[str, Any],
    beta: int,
    gamma: int,
    max_depth_steps: Optional[int] = None,
    max_wait_slots: Optional[int] = None,
    wait_logit_penalty: float = 0.0,
    makespan_penalty: float = 0.0,
) -> List[DecodeResult]:
    """Run SGBS for a batch of instances (solved independently, in a Python loop).

    This keeps the implementation simple and paper-faithful; optimization can be
    added later by doing multi-instance beams.
    """
    if beta <= 0 or gamma <= 0:
        raise ValueError("beta and gamma must be positive")

    # Determine batch size from any array
    some_key = next(iter(batch_data.keys()))
    B = int(batch_data[some_key].shape[0])

    results: List[DecodeResult] = []
    for i in range(B):
        single = _slice_single_instance(batch_data, i)
        res = sgbs_single_instance(
            model=model,
            env_config=env_config,
            device=device,
            batch_data_single=single,
            beta=beta,
            gamma=gamma,
            max_depth_steps=max_depth_steps,
            max_wait_slots=max_wait_slots,
            wait_logit_penalty=wait_logit_penalty,
            makespan_penalty=makespan_penalty,
        )
        results.append(res)

    return results


@torch.no_grad()
def sgbs_single_instance(
    model: PaSTSMNet,
    env_config,
    device: torch.device,
    batch_data_single: Dict[str, Any],
    beta: int,
    gamma: int,
    max_depth_steps: Optional[int] = None,
    max_wait_slots: Optional[int] = None,
    wait_logit_penalty: float = 0.0,
    makespan_penalty: float = 0.0,
) -> DecodeResult:
    """Paper-faithful SGBS for one instance.

    Uses:
      - reference env: batch_size=1 (for expansion steps)
      - rollout env: batch_size=len(E) (for batched greedy rollouts)
    """
    if beta <= 0 or gamma <= 0:
        raise ValueError("beta and gamma must be positive")

    env_ref = GPUBatchSingleMachinePeriodEnv(
        batch_size=1, env_config=env_config, device=device
    )
    env_ref.reset(batch_data_single)

    if max_depth_steps is None:
        max_depth_steps = int(env_ref.N_job_pad) + 5

    root = get_state(env_ref)
    beam: List[BeamNode] = [BeamNode(state=root, actions=[])]

    best_return = torch.tensor([-float("inf")], device=device)
    best_energy = torch.tensor([float("inf")], device=device)

    # Loop level-by-level until all beam nodes are terminal
    best_actions: Optional[List[int]] = None

    while True:
        any_non_terminal = any((~n.state.done_mask).any().item() for n in beam)
        if not any_non_terminal:
            break

        # --- Expansion (pre-pruning) ---
        expanded_children: List[BeamNode] = []
        for node in beam:
            set_state(env_ref, node.state)
            obs = env_ref._get_obs()  # may mark dead_end
            parent_state = get_state(env_ref)

            if bool(parent_state.done_mask[0].item()):
                # Terminal nodes expand to themselves (paper behavior)
                expanded_children.append(
                    BeamNode(state=parent_state, actions=list(node.actions))
                )
                continue

            logits = compute_masked_logits(model, obs)[0]  # (A,)
            feas = _completion_feasible_action_mask(env_ref, obs)[0]
            logits = logits.masked_fill(feas == 0, float("-inf"))

            if max_wait_slots is not None:
                logits = logits.masked_fill(
                    _max_wait_action_mask(env_ref, int(max_wait_slots))[0] == 0,
                    float("-inf"),
                )

            logits = _apply_wait_logit_penalty(
                logits.unsqueeze(0), env_ref, wait_logit_penalty
            ).squeeze(0)

            actions = _select_topk_valid_actions(logits, gamma)
            if len(actions) == 0:
                # No valid actions: treat as terminal
                expanded_children.append(
                    BeamNode(state=parent_state, actions=list(node.actions))
                )
                continue

            for a in actions:
                set_state(env_ref, parent_state)
                _next_obs, _rewards, _dones, _info = env_ref.step(
                    torch.tensor([a], device=device)
                )
                child = get_state(env_ref)
                expanded_children.append(
                    BeamNode(state=child, actions=list(node.actions) + [int(a)])
                )

        # --- Simulation (batched greedy rollout) ---
        E = expanded_children
        rollout_B = len(E)
        rollout_env = GPUBatchSingleMachinePeriodEnv(
            batch_size=int(rollout_B), env_config=env_config, device=device
        )
        rollout_env.reset(_repeat_batch_data(batch_data_single, rollout_B))

        # Restore each candidate snapshot into rollout env rows
        # (dynamic state only)
        t = torch.cat([n.state.t for n in E], dim=0)
        total_energy = torch.cat([n.state.total_energy for n in E], dim=0)
        done_mask = torch.cat([n.state.done_mask for n in E], dim=0)
        job_available = torch.cat([n.state.job_available for n in E], dim=0)

        rollout_env.t.copy_(t)
        rollout_env.total_energy.copy_(total_energy)
        rollout_env.done_mask.copy_(done_mask)
        rollout_env.job_available.copy_(job_available)

        returns, energies, traces = greedy_rollout(
            rollout_env,
            model,
            max_steps=max_depth_steps,
            record_actions=True,
            max_wait_slots=max_wait_slots,
            wait_logit_penalty=wait_logit_penalty,
            makespan_penalty=makespan_penalty,
        )

        # Track global incumbent (best complete solution seen in any rollout)
        # Note: returns are -energy; higher return is better.
        local_best_idx = int(torch.argmax(returns).item())
        if returns[local_best_idx] > best_return[0]:
            best_return = returns[local_best_idx : local_best_idx + 1].clone()
            best_energy = energies[local_best_idx : local_best_idx + 1].clone()
            tail = traces[local_best_idx] if traces is not None else []
            best_actions = list(E[local_best_idx].actions) + [int(a) for a in tail]

        # --- Pruning: keep top-β children by rollout return ---
        topk = torch.topk(returns, k=min(beta, rollout_B), dim=0)
        next_beam = [E[int(j)] for j in topk.indices.detach().cpu().tolist()]
        beam = next_beam

    return DecodeResult(
        total_energy=float(best_energy.item()),
        total_return=float(best_return.item()),
        actions=best_actions,
    )


@torch.no_grad()
def greedy_decode(
    model: PaSTSMNet,
    env_config,
    device: torch.device,
    batch_data: Dict[str, Any],
    max_steps: Optional[int] = None,
    max_wait_slots: Optional[int] = None,
    wait_logit_penalty: float = 0.0,
    makespan_penalty: float = 0.0,
) -> List[DecodeResult]:
    """Greedy decoding baseline for a batch of instances."""
    some_key = next(iter(batch_data.keys()))
    B = int(batch_data[some_key].shape[0])

    results: List[DecodeResult] = []
    for i in range(B):
        single = _slice_single_instance(batch_data, i)
        env = GPUBatchSingleMachinePeriodEnv(
            batch_size=1, env_config=env_config, device=device
        )
        env.reset(single)
        returns, energies, traces = greedy_rollout(
            env,
            model,
            max_steps=max_steps,
            record_actions=True,
            max_wait_slots=max_wait_slots,
            wait_logit_penalty=wait_logit_penalty,
            makespan_penalty=makespan_penalty,
        )
        results.append(
            DecodeResult(
                total_energy=float(energies[0].item()),
                total_return=float(returns[0].item()),
                actions=(traces[0] if traces is not None else None),
            )
        )

    return results

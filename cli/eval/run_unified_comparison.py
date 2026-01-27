"""Unified comparison script for all PPO and Q-Learning model variants.

This is the ultimate comparison script that evaluates all model variants with:
- Two decoding rules: Normal (best-start) and Cmax-aware
- Optional SGBS for all variants
- Gantt-style schedule visualizations
- Command-line variant selection

Example usage:
    # Compare all variants with default settings
    python -m PaST.cli.eval.run_unified_comparison \
        --eval_seed 55 --num_instances 16 --num_viz 4 --scale small

    # Compare specific variants with SGBS
    python -m PaST.cli.eval.run_unified_comparison \
        --variants ppo_family_best ppo_q_seq \
        --use_sgbs --sgbs_beta 8 --sgbs_gamma 4 \
        --eval_seed 55 --num_instances 16

    # Compare with both decoding rules
    python -m PaST.cli.eval.run_unified_comparison \
        --decoding both --slack_ratios "0.2,0.3,0.4" \
        --eval_seed 55 --num_instances 16
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
import random

# Set matplotlib backend before import
os.environ.setdefault("MPLBACKEND", "Agg")

from PaST.baselines_sequence_dp import (
    DPResult,
    spt_lpt_with_dp,
    dp_schedule_for_job_sequence,
)
from PaST.config import DataConfig, VariantID, get_variant_config
from PaST.past_sm_model import build_model
from PaST.q_sequence_model import build_q_model, QSequenceNet
from PaST.sgbs import greedy_decode, sgbs, DecodeResult
from PaST.sm_benchmark_data import (
    generate_raw_instance,
    simulate_metaheuristic_assignment,
    make_single_machine_episode,
    SingleMachineEpisode,
)
from PaST.sequence_env import GPUBatchSequenceEnv
from PaST.batch_dp_solver import BatchSequenceDPSolver
from PaST.sm_env import GPUBatchSingleMachinePeriodEnv


# =============================================================================
# MODEL PATHS - Edit these paths to change which models to compare
# =============================================================================

MODEL_PATHS = {
    "ppo_duration_aware": "runs_p100/ppo_duration_aware/checkpoints/best.pt",
    "ppo_family_best": "runs_p100/ppo_family_best/best.pt",
    "ppo_family_best_cwe": "runs_p100/ppo_family_best_cwe/checkpoints/best.pt",
    "ppo_q_seq": "runs_p100/ppo_q_seq/checkpoints/best.pt",
    "ppo_short_base": "runs_p100/ppo_short_base/checkpoints/best.pt",
    "q_seq_cwe": "runs_p100/q-seq-cwe/checkpoints/best_model.pt",
}

# Mapping from model name to VariantID
VARIANT_IDS = {
    "ppo_duration_aware": VariantID.PPO_DURATION_AWARE_FAMILY_CTX13,
    "ppo_family_best": VariantID.PPO_FAMILY_Q4_CTX13_BESTSTART,
    "ppo_family_best_cwe": VariantID.PPO_FAMILY_Q4_CTX13_BESTSTART_CWE,
    # NOTE: Despite the folder name, this checkpoint uses a dueling Q head (q_head.*)
    # and should be loaded as a Q-sequence model.
    "ppo_q_seq": VariantID.Q_SEQUENCE,
    "ppo_short_base": VariantID.PPO_SHORT_BASE,
    "q_seq_cwe": VariantID.Q_SEQUENCE_CWE_CTX13,
}

# Q-Learning variants (use different model builder)
Q_LEARNING_VARIANTS = {"q_seq_cwe", "ppo_q_seq"}

# PPO variants that are "sequence-only" (action = job index) and must be evaluated
# via GPUBatchSequenceEnv + DP scheduling (NOT via GPUBatchSingleMachinePeriodEnv).
SEQUENCE_POLICY_VARIANTS: set[str] = set()

# All available variants for command-line help
ALL_VARIANTS = list(MODEL_PATHS.keys())

# Architecture variants that are excluded from the default comparison.
# You can still evaluate them explicitly via `--variants ...` or enable them via
# `--include_cwe`.
CWE_VARIANTS = {"ppo_family_best_cwe", "q_seq_cwe"}
DEFAULT_VARIANTS = [v for v in ALL_VARIANTS if v not in CWE_VARIANTS]


# =============================================================================
# Utilities
# =============================================================================


def batch_from_episodes(
    episodes: List[SingleMachineEpisode],
    N_job_pad: int = 50,
    K_period_pad: int = 250,
    T_max_pad: int = 500,
) -> Dict[str, np.ndarray]:
    """Manually batch a list of episodes."""
    batch_size = len(episodes)

    batch = {
        "p_subset": np.zeros((batch_size, N_job_pad), dtype=np.int32),
        "n_jobs": np.zeros((batch_size,), dtype=np.int32),
        "job_mask": np.zeros((batch_size, N_job_pad), dtype=np.float32),
        "T_max": np.zeros((batch_size,), dtype=np.int32),
        "T_limit": np.zeros((batch_size,), dtype=np.int32),
        "T_min": np.zeros((batch_size,), dtype=np.int32),
        "ct": np.zeros((batch_size, T_max_pad), dtype=np.int32),
        "Tk": np.zeros((batch_size, K_period_pad), dtype=np.int32),
        "ck": np.zeros((batch_size, K_period_pad), dtype=np.int32),
        "period_starts": np.zeros((batch_size, K_period_pad), dtype=np.int32),
        "K": np.zeros((batch_size,), dtype=np.int32),
        "period_mask": np.zeros((batch_size, K_period_pad), dtype=np.float32),
        "e_single": np.zeros((batch_size,), dtype=np.int32),
        "price_q": np.zeros((batch_size, 3), dtype=np.float32),
    }

    for i, episode in enumerate(episodes):
        n = min(episode.n_jobs, N_job_pad)
        k = min(episode.K, K_period_pad)
        t_max = min(episode.T_max, T_max_pad)

        batch["n_jobs"][i] = n
        batch["K"][i] = k
        batch["T_max"][i] = t_max
        batch["T_limit"][i] = episode.T_limit
        batch["T_min"][i] = episode.T_min
        batch["e_single"][i] = episode.e_single

        if n > 0:
            batch["p_subset"][i, :n] = episode.p_subset[:n]
            batch["job_mask"][i, :n] = 1.0

        if k > 0:
            batch["Tk"][i, :k] = episode.Tk[:k]
            batch["ck"][i, :k] = episode.ck[:k]
            batch["period_starts"][i, :k] = episode.period_starts[:k]
            batch["period_mask"][i, :k] = 1.0

        if t_max > 0:
            batch["ct"][i, :t_max] = episode.ct[:t_max]
            ct_valid = episode.ct[:t_max]
            if len(ct_valid) > 0:
                q25, q50, q75 = np.quantile(ct_valid, [0.25, 0.5, 0.75])
                batch["price_q"][i] = [q25, q50, q75]

    return batch


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
    """Extract model state dict from PPO checkpoint."""
    # Some checkpoints are saved as a raw state_dict (e.g., OrderedDict of tensors).
    sample_keys = list(ckpt.keys())[:5]
    if any("." in str(k) for k in sample_keys):
        return ckpt

    if "runner" in ckpt and isinstance(ckpt["runner"], dict):
        runner = ckpt["runner"]
        if "model" in runner and isinstance(runner["model"], dict):
            return runner["model"]
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    raise KeyError("Could not find model state in checkpoint")


def _extract_q_model_state(ckpt: Any) -> Dict[str, Any]:
    """Extract model state dict from Q-learning checkpoint."""
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dict")

    # Format 1: checkpoint with 'model_state' key
    if "model_state" in ckpt:
        return ckpt["model_state"]

    # Format 2: Direct state_dict
    sample_keys = list(ckpt.keys())[:5]
    if any("." in k for k in sample_keys):
        return ckpt

    raise KeyError("Could not find model state in checkpoint")


def _restrict_data_config(
    data: DataConfig,
    scale: Optional[str],
) -> DataConfig:
    """Restrict data config to a specific instance scale."""
    cfg = replace(data)

    if scale is None:
        return cfg

    s = scale.lower()
    if s == "small":
        cfg.T_max_choices = [t for t in cfg.T_max_choices if int(t) <= 100]
    elif s in ("medium", "mls"):
        cfg.T_max_choices = [t for t in cfg.T_max_choices if 100 < int(t) <= 350]
    elif s in ("large", "vls"):
        cfg.T_max_choices = [t for t in cfg.T_max_choices if int(t) > 350]
    else:
        raise ValueError(f"scale must be one of: small, medium, large (got {scale})")

    if not cfg.T_max_choices:
        raise ValueError(f"No T_max_choices match scale={scale}.")

    return cfg


def _slice_single_instance(batch_data: Dict[str, Any], index: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch_data.items():
        if isinstance(v, np.ndarray):
            out[k] = v[index : index + 1]
        elif torch.is_tensor(v):
            out[k] = v[index : index + 1]
        else:
            raise TypeError(f"Unsupported batch_data type for key={k}: {type(v)}")
    return out


def _to_invalid_bool_mask(mask: torch.Tensor) -> torch.Tensor:
    """Convert env masks to model masks: True=invalid.

    Env conventions in this repo are usually float masks with 1=valid, 0=invalid.
    Some code paths may already provide bool masks where True=invalid.
    """
    if mask.dtype == torch.bool:
        return mask
    return mask < 0.5


def _sequence_schedule_to_bars(
    job_sequence: List[int],
    start_times: List[int],
    p_subset: np.ndarray,
) -> List[Dict[str, Any]]:
    """Convert sequence schedule to visualization bars."""
    bars = []
    for j, st in zip(job_sequence, start_times):
        job_id = int(j)
        start = int(st)
        p = int(p_subset[job_id])
        bars.append({"job_id": job_id, "start": start, "end": start + p})
    return bars


# =============================================================================
# Decoding Functions
# =============================================================================


def _action_trace_to_bars_normal(
    actions: List[int],
    env_config,
    p_subset: np.ndarray,
    ct: np.ndarray,
    e_single: int,
    T_limit: int,
) -> Tuple[List[Dict[str, Any]], float, int]:
    """Normal best-start decoding for price-family variants.

    Note: For better Cmax behavior, we interpret the predicted family as an
    **end-family** constraint (the job may start earlier in a different family,
    as long as it ends inside the predicted family). Among equal-cost options,
    we tie-break toward earlier finish.

    Returns: (bars, total_energy, cmax)
    """
    num_families = env_config.num_price_families

    # Compute price quantiles
    ct_valid = ct[:T_limit]
    if len(ct_valid) > 0:
        q25, q50, q75 = np.quantile(ct_valid, [0.25, 0.5, 0.75])
    else:
        q25 = q50 = q75 = 0

    # Assign each slot to a family
    slot_families = np.zeros(len(ct), dtype=np.int32)
    for u in range(len(ct)):
        price = ct[u]
        if price > q75:
            slot_families[u] = 3
        elif price > q50:
            slot_families[u] = 2
        elif price > q25:
            slot_families[u] = 1
        else:
            slot_families[u] = 0

    bars = []
    total_energy = 0.0
    t = 0
    cmax = 0

    ct_cumsum = np.zeros(len(ct) + 1, dtype=np.float64)
    ct_cumsum[1:] = np.cumsum(ct)

    def get_interval_cost(start, end):
        s = max(0, min(start, len(ct)))
        e = max(0, min(end, len(ct)))
        return ct_cumsum[e] - ct_cumsum[s]

    n_jobs = len(p_subset)
    remaining_work = int(np.sum(p_subset))
    scheduled_jobs = set()

    for a in actions:
        job_id = int(a) // num_families
        family_id = int(a) % num_families

        if job_id >= n_jobs or job_id in scheduled_jobs:
            continue

        p = int(p_subset[job_id])

        # Find best start such that the job ENDS in the predicted family.
        best_start: Optional[int] = None
        best_cost = float("inf")
        best_end = float("inf")

        max_valid_start = T_limit - remaining_work
        max_valid_start = min(max_valid_start, T_limit - p)
        search_end = max(t, max_valid_start) + 1
        search_end = min(search_end, T_limit - p + 1)

        for u in range(t, search_end):
            end_idx = u + p - 1
            if end_idx < 0 or end_idx >= T_limit:
                continue
            end_family = int(slot_families[end_idx])
            if end_family != family_id:
                continue
            cost = float(get_interval_cost(u, u + p))
            end_time = int(u + p)
            if (cost < best_cost) or (cost == best_cost and end_time < best_end):
                best_cost = cost
                best_end = end_time
                best_start = int(u)

        if best_start is None:
            best_start = int(t)

        start = best_start
        end = start + p
        if end > T_limit:
            end = T_limit
            start = max(t, end - p)

        energy = float(e_single) * float(get_interval_cost(start, end))
        total_energy += energy

        start_family = int(slot_families[start]) if start < T_limit else -1
        end_family = int(slot_families[end - 1]) if 0 <= end - 1 < T_limit else -1

        bars.append(
            {
                "job_id": job_id,
                "start": start,
                "end": end,
                "family_id": family_id,
                "start_family": start_family,
                "end_family": end_family,
            }
        )
        t = end
        cmax = max(cmax, end)
        scheduled_jobs.add(job_id)
        remaining_work -= p

    return bars, total_energy, cmax


def _action_trace_to_bars_cmax_aware(
    actions: List[int],
    env_config,
    p_subset: np.ndarray,
    ct: np.ndarray,
    e_single: int,
    T_limit: int,
) -> Tuple[List[Dict[str, Any]], float, int]:
    """Progressive family-expansion decoding.

    This replaces the older heuristic "if cheap capacity is exhausted, start immediately".

    Rule:
    - Let families be ordered from cheapest (0) to most expensive (3).
    - At each step, compute remaining work (sum of processing times of unscheduled jobs).
    - Find the smallest family threshold f* such that the number of remaining time slots
        whose family <= f* (from current time t to deadline) is >= remaining work.
        This yields the minimal set of families {0..f*} that can still complete the schedule.
    - Schedule the next job by selecting the minimum-energy start among allowed families,
        preferring the model-chosen family when it is allowed.

    Returns: (bars, total_energy, cmax)
    """
    num_families = env_config.num_price_families

    ct_valid = ct[:T_limit]
    if len(ct_valid) > 0:
        q25, q50, q75 = np.quantile(ct_valid, [0.25, 0.5, 0.75])
    else:
        q25 = q50 = q75 = 0

    slot_families = np.zeros(len(ct), dtype=np.int32)
    for u in range(len(ct)):
        price = ct[u]
        if price > q75:
            slot_families[u] = 3
        elif price > q50:
            slot_families[u] = 2
        elif price > q25:
            slot_families[u] = 1
        else:
            slot_families[u] = 0

    bars: List[Dict[str, Any]] = []
    total_energy = 0.0
    t = 0
    cmax = 0

    ct_cumsum = np.zeros(len(ct) + 1, dtype=np.float64)
    ct_cumsum[1:] = np.cumsum(ct)

    def get_interval_cost(start, end):
        s = max(0, min(start, len(ct)))
        e = max(0, min(end, len(ct)))
        return ct_cumsum[e] - ct_cumsum[s]

    n_jobs = len(p_subset)
    remaining_work = int(np.sum(p_subset))
    scheduled_jobs: set[int] = set()

    def allowed_max_family(t_now: int, work_left: int) -> int:
        if work_left <= 0:
            return 0

        end = int(min(T_limit, len(slot_families)))
        if t_now >= end:
            return num_families - 1

        region = slot_families[t_now:end]
        for f in range(num_families):
            cap = int(np.sum(region <= f))
            if cap >= int(work_left):
                return f
        return num_families - 1

    for a in actions:
        job_id = int(a) // num_families
        family_id = int(a) % num_families

        if job_id >= n_jobs or job_id in scheduled_jobs:
            continue

        p = int(p_subset[job_id])

        f_max = allowed_max_family(t, remaining_work)

        # Restrict search window to preserve feasibility for remaining work.
        max_valid_start = T_limit - remaining_work
        max_valid_start = min(max_valid_start, T_limit - p)
        search_end = max(t, max_valid_start) + 1
        search_end = min(search_end, T_limit - p + 1)

        # Candidate starts with window stats.
        # We interpret the model's family as an END-family preference (job ends in that family).
        candidates: List[Tuple[int, float, int, int]] = []
        # (u, cost, end_family, max_family_in_window)
        for u in range(t, search_end):
            end_idx = u + p - 1
            if end_idx < 0 or end_idx >= T_limit:
                continue
            window = slot_families[u : u + p]
            if window.size <= 0:
                continue
            max_f = int(window.max())
            end_f = int(slot_families[end_idx])
            cost = float(get_interval_cost(u, u + p))
            candidates.append((int(u), cost, end_f, max_f))

        def pick_best(
            allowed_max_family: int, prefer_end_family: Optional[int]
        ) -> Optional[Tuple[int, float, int, int]]:
            best: Optional[Tuple[int, float, int, int]] = None
            for u, cost, end_f, max_f in candidates:
                if max_f > allowed_max_family:
                    continue
                if prefer_end_family is not None and end_f != prefer_end_family:
                    continue
                end_time = u + p
                if best is None:
                    best = (u, cost, end_f, max_f)
                    best_end = end_time
                    continue
                best_u, best_cost, _, _ = best
                best_end = best_u + p
                # Minimize cost, then minimize end time (Cmax), then earlier start.
                if (
                    (cost < best_cost)
                    or (cost == best_cost and end_time < best_end)
                    or (cost == best_cost and end_time == best_end and u < best_u)
                ):
                    best = (u, cost, end_f, max_f)
            return best

        chosen: Optional[Tuple[int, float, int, int]] = None
        f_used = int(f_max)

        # Progressive expansion: try minimal allowed max-family first, then expand if needed.
        for f_try in range(int(f_max), num_families):
            # Prefer ending in model-predicted family if possible.
            chosen = pick_best(f_try, prefer_end_family=family_id)
            if chosen is None:
                chosen = pick_best(f_try, prefer_end_family=None)
            if chosen is not None:
                f_used = int(f_try)
                break

        if chosen is None:
            best_start = int(t)
            chosen_end_family = (
                int(slot_families[min(T_limit - 1, t + p - 1)]) if T_limit > 0 else -1
            )
            chosen_max_family = (
                int(slot_families[min(T_limit - 1, t)]) if T_limit > 0 else -1
            )
        else:
            best_start, _, chosen_end_family, chosen_max_family = chosen

        start = best_start
        end = start + p
        if end > T_limit:
            end = T_limit
            start = max(t, end - p)

        energy = float(e_single) * float(get_interval_cost(start, end))
        total_energy += energy

        start_family = int(slot_families[start]) if start < T_limit else -1

        bars.append(
            {
                "job_id": job_id,
                "start": start,
                "end": end,
                "family_id": family_id,
                "allowed_max_family": int(f_used),
                "start_family": start_family,
                "end_family": int(chosen_end_family),
                "max_family_in_window": int(chosen_max_family),
            }
        )
        t = end
        cmax = max(cmax, end)
        scheduled_jobs.add(job_id)
        remaining_work -= p

    return bars, total_energy, cmax


def greedy_decode_q_sequence(
    model: QSequenceNet,
    variant_config,
    batch_data: Dict[str, Any],
    device: torch.device,
) -> List[DPResult]:
    """Greedy decode using Q-values: argmin Q(s, j) for each step."""
    B = int(batch_data["n_jobs"].shape[0])
    env_config = variant_config.env

    results = []

    model.eval()
    with torch.no_grad():
        for i in range(B):
            single = _slice_single_instance(batch_data, i)
            n = int(single["n_jobs"][0])

            env = GPUBatchSequenceEnv(
                batch_size=1,
                env_config=env_config,
                device=device,
            )
            obs = env.reset(single)

            sequence = []

            for step in range(n):
                jobs_t = obs["jobs"]
                periods_t = obs["periods"]
                ctx_t = obs["ctx"]

                job_mask_float = obs.get("job_mask", env.job_available)
                job_mask = _to_invalid_bool_mask(job_mask_float)

                q_values = model(jobs_t, periods_t, ctx_t, job_mask)
                q_values = q_values.masked_fill(job_mask, float("inf"))

                action = q_values.argmin(dim=-1)
                sequence.append(int(action.item()))

                obs, _, done, _ = env.step(action)
                if done.all():
                    break

            result = dp_schedule_for_job_sequence(single, sequence)
            results.append(result)

    return results


def sgbs_q_sequence(
    model: QSequenceNet,
    variant_config,
    batch_data: Dict[str, Any],
    device: torch.device,
    beta: int,
    gamma: int,
) -> List[DPResult]:
    """SGBS decode for Q-sequence model."""
    B = int(batch_data["n_jobs"].shape[0])
    env_config = variant_config.env

    results = []

    model.eval()
    with torch.no_grad():
        for i in range(B):
            single = _slice_single_instance(batch_data, i)
            n = int(single["n_jobs"][0])

            if n <= 2 or (beta == 1 and gamma == 1):
                # Small instance: just do greedy
                env = GPUBatchSequenceEnv(
                    batch_size=1, env_config=env_config, device=device
                )
                obs = env.reset(single)
                sequence = []
                for _ in range(n):
                    jobs_t = obs["jobs"]
                    periods_t = obs["periods"]
                    ctx_t = obs["ctx"]
                    job_mask = obs.get("job_mask", env.job_available)
                    job_mask = _to_invalid_bool_mask(job_mask)
                    q_values = model(jobs_t, periods_t, ctx_t, job_mask)
                    q_values = q_values.masked_fill(job_mask, float("inf"))
                    action = q_values.argmin(dim=-1)
                    sequence.append(int(action.item()))
                    obs, _, done, _ = env.step(action)
                    if done.all():
                        break
                results.append(dp_schedule_for_job_sequence(single, sequence))
                continue

            # Full SGBS implementation
            N_max = int(variant_config.env.N_job_pad)
            beam_sequences = torch.zeros((1, N_max), dtype=torch.long, device=device)
            beam_avail = torch.zeros((1, N_max), dtype=torch.float32, device=device)
            beam_avail[:, :n] = 1.0

            best_energy = float("inf")
            best_sequence = list(range(n))

            def _repeat_single_batch(single_batch, reps):
                out = {}
                for k, v in single_batch.items():
                    if isinstance(v, np.ndarray):
                        out[k] = np.repeat(v, reps, axis=0)
                    elif torch.is_tensor(v):
                        out[k] = v.repeat(reps, *([1] * (v.dim() - 1)))
                return out

            for step in range(n):
                beam_size = int(beam_sequences.shape[0])

                env_beam = GPUBatchSequenceEnv(
                    batch_size=beam_size, env_config=env_config, device=device
                )
                env_beam.reset(_repeat_single_batch(single, beam_size))
                env_beam.job_sequences = beam_sequences.clone()
                env_beam.job_available = beam_avail.clone()
                env_beam.step_indices = torch.full(
                    (beam_size,), step, dtype=torch.long, device=device
                )
                env_beam.done = torch.zeros(beam_size, dtype=torch.bool, device=device)

                obs = env_beam._get_obs()
                invalid = obs.get("job_mask", env_beam.job_available)
                invalid = _to_invalid_bool_mask(invalid)

                q_values = model(obs["jobs"], obs["periods"], obs["ctx"], invalid)
                q_values = q_values.masked_fill(invalid, float("inf"))

                k = min(int(gamma), n)
                neg_q = -q_values[:, :n]
                neg_q = neg_q.masked_fill(invalid[:, :n], float("-inf"))
                top_jobs = torch.topk(neg_q, k=k, dim=-1).indices

                cand_sequences = (
                    beam_sequences.unsqueeze(1).repeat(1, k, 1).reshape(-1, N_max)
                )
                cand_avail = beam_avail.unsqueeze(1).repeat(1, k, 1).reshape(-1, N_max)
                chosen = top_jobs.reshape(-1)

                C = int(chosen.shape[0])
                idx = torch.arange(C, device=device)
                choice_valid = cand_avail[idx, chosen] > 0.5
                if not bool(choice_valid.any()):
                    break
                cand_sequences = cand_sequences[choice_valid].clone()
                cand_avail = cand_avail[choice_valid].clone()
                chosen = chosen[choice_valid]

                C_valid = int(chosen.shape[0])
                idx2 = torch.arange(C_valid, device=device)
                cand_sequences[idx2, step] = chosen
                cand_avail[idx2, chosen] = 0.0

                # Simulate completion
                env_roll = GPUBatchSequenceEnv(
                    batch_size=C_valid, env_config=env_config, device=device
                )
                env_roll.reset(_repeat_single_batch(single, C_valid))
                env_roll.job_sequences = cand_sequences.clone()
                env_roll.job_available = cand_avail.clone()
                env_roll.step_indices = torch.full(
                    (C_valid,), step + 1, dtype=torch.long, device=device
                )
                env_roll.done = torch.zeros(C_valid, dtype=torch.bool, device=device)

                dones = torch.zeros(C_valid, dtype=torch.bool, device=device)
                for _ in range(step + 1, n):
                    if bool(dones.all()):
                        break
                    obs_r = env_roll._get_obs()
                    invalid_r = obs_r.get("job_mask", env_roll.job_available)
                    invalid_r = _to_invalid_bool_mask(invalid_r)
                    q_r = model(
                        obs_r["jobs"], obs_r["periods"], obs_r["ctx"], invalid_r
                    )
                    q_r = q_r.masked_fill(invalid_r, float("inf"))
                    action = q_r.argmin(dim=-1)
                    _, _, dones, _ = env_roll.step(action)

                costs = BatchSequenceDPSolver.solve(
                    job_sequences=env_roll.job_sequences,
                    processing_times=env_roll.p_subset,
                    ct=env_roll.ct,
                    e_single=env_roll.e_single,
                    T_limit=env_roll.T_limit,
                    sequence_lengths=env_roll.n_jobs.long(),
                )

                best_idx = int(torch.argmin(costs).item())
                best_cost = float(costs[best_idx].item())
                if best_cost < best_energy:
                    best_energy = best_cost
                    best_seq_tensor = (
                        env_roll.job_sequences[best_idx, :n].detach().cpu().long()
                    )
                    best_sequence = [int(x) for x in best_seq_tensor.tolist()]

                keep = min(int(beta), C_valid)
                keep_idx = torch.topk(-costs, k=keep).indices
                beam_sequences = cand_sequences[keep_idx].clone()
                beam_avail = cand_avail[keep_idx].clone()

            results.append(dp_schedule_for_job_sequence(single, best_sequence))

    return results


def greedy_decode_ppo_sequence(
    model: Any,
    variant_config,
    batch_data: Dict[str, Any],
    device: torch.device,
) -> List[DPResult]:
    """Greedy decode for PPO_SEQUENCE-like policy (action = job index), then DP schedule."""
    B = int(batch_data["n_jobs"].shape[0])
    env_config = variant_config.env
    results: List[DPResult] = []

    model.eval()
    with torch.no_grad():
        for i in range(B):
            single = _slice_single_instance(batch_data, i)
            n = int(single["n_jobs"][0])

            env = GPUBatchSequenceEnv(
                batch_size=1, env_config=env_config, device=device
            )
            obs = env.reset(single)

            seq: List[int] = []
            for _ in range(n):
                # PPO policy returns logits over actions
                logits, _ = model(
                    jobs=obs["jobs"],
                    periods_local=obs["periods"],
                    ctx=obs["ctx"],
                    job_mask=_to_invalid_bool_mask(
                        obs.get("job_mask", env.job_available)
                    ),
                    period_mask=None,
                    periods_full=None,
                    period_full_mask=None,
                )

                # Apply env action mask if present
                if "action_mask" in obs:
                    logits = logits.masked_fill(obs["action_mask"] < 0.5, float("-inf"))

                action = torch.argmax(logits, dim=-1)
                seq.append(int(action.item()))

                obs, _, done, _ = env.step(action)
                if bool(done.all()):
                    break

            results.append(dp_schedule_for_job_sequence(single, seq))

    return results


def sgbs_ppo_sequence(
    model: Any,
    variant_config,
    batch_data: Dict[str, Any],
    device: torch.device,
    beta: int,
    gamma: int,
) -> List[DPResult]:
    """SGBS-style decode for PPO_SEQUENCE-like policy, using DP cost as rollout score."""
    B = int(batch_data["n_jobs"].shape[0])
    env_config = variant_config.env
    results: List[DPResult] = []

    model.eval()
    with torch.no_grad():
        for i in range(B):
            single = _slice_single_instance(batch_data, i)
            n = int(single["n_jobs"][0])

            if n <= 2 or (beta == 1 and gamma == 1):
                results.extend(
                    greedy_decode_ppo_sequence(
                        model,
                        variant_config,
                        {k: v[i : i + 1] for k, v in batch_data.items()},
                        device,
                    )
                )
                continue

            N_max = int(env_config.N_job_pad)
            beam_sequences = torch.zeros((1, N_max), dtype=torch.long, device=device)
            beam_avail = torch.zeros((1, N_max), dtype=torch.float32, device=device)
            beam_avail[:, :n] = 1.0

            best_energy = float("inf")
            best_sequence = list(range(n))

            def _repeat_single_batch(single_batch, reps):
                out = {}
                for k, v in single_batch.items():
                    if isinstance(v, np.ndarray):
                        out[k] = np.repeat(v, reps, axis=0)
                    elif torch.is_tensor(v):
                        out[k] = v.repeat(reps, *([1] * (v.dim() - 1)))
                return out

            for step in range(n):
                beam_size = int(beam_sequences.shape[0])

                env_beam = GPUBatchSequenceEnv(
                    batch_size=beam_size, env_config=env_config, device=device
                )
                env_beam.reset(_repeat_single_batch(single, beam_size))
                env_beam.job_sequences = beam_sequences.clone()
                env_beam.job_available = beam_avail.clone()
                env_beam.step_indices = torch.full(
                    (beam_size,), step, dtype=torch.long, device=device
                )
                env_beam.done = torch.zeros(beam_size, dtype=torch.bool, device=device)

                obs = env_beam._get_obs()
                invalid = _to_invalid_bool_mask(
                    obs.get("job_mask", env_beam.job_available)
                )

                logits, _ = model(
                    jobs=obs["jobs"],
                    periods_local=obs["periods"],
                    ctx=obs["ctx"],
                    job_mask=invalid,
                    period_mask=None,
                    periods_full=None,
                    period_full_mask=None,
                )
                if "action_mask" in obs:
                    logits = logits.masked_fill(obs["action_mask"] < 0.5, float("-inf"))
                logits = logits.masked_fill(invalid, float("-inf"))

                k = min(int(gamma), n)
                top_jobs = torch.topk(logits[:, :n], k=k, dim=-1).indices

                cand_sequences = (
                    beam_sequences.unsqueeze(1).repeat(1, k, 1).reshape(-1, N_max)
                )
                cand_avail = beam_avail.unsqueeze(1).repeat(1, k, 1).reshape(-1, N_max)
                chosen = top_jobs.reshape(-1)

                C = int(chosen.shape[0])
                idx = torch.arange(C, device=device)
                choice_valid = cand_avail[idx, chosen] > 0.5
                if not bool(choice_valid.any()):
                    break
                cand_sequences = cand_sequences[choice_valid].clone()
                cand_avail = cand_avail[choice_valid].clone()
                chosen = chosen[choice_valid]

                C_valid = int(chosen.shape[0])
                idx2 = torch.arange(C_valid, device=device)
                cand_sequences[idx2, step] = chosen
                cand_avail[idx2, chosen] = 0.0

                # Greedy completion rollout using policy
                env_roll = GPUBatchSequenceEnv(
                    batch_size=C_valid, env_config=env_config, device=device
                )
                env_roll.reset(_repeat_single_batch(single, C_valid))
                env_roll.job_sequences = cand_sequences.clone()
                env_roll.job_available = cand_avail.clone()
                env_roll.step_indices = torch.full(
                    (C_valid,), step + 1, dtype=torch.long, device=device
                )
                env_roll.done = torch.zeros(C_valid, dtype=torch.bool, device=device)

                dones = torch.zeros(C_valid, dtype=torch.bool, device=device)
                for _ in range(step + 1, n):
                    if bool(dones.all()):
                        break
                    obs_r = env_roll._get_obs()
                    invalid_r = _to_invalid_bool_mask(
                        obs_r.get("job_mask", env_roll.job_available)
                    )
                    logits_r, _ = model(
                        jobs=obs_r["jobs"],
                        periods_local=obs_r["periods"],
                        ctx=obs_r["ctx"],
                        job_mask=invalid_r,
                        period_mask=None,
                        periods_full=None,
                        period_full_mask=None,
                    )
                    if "action_mask" in obs_r:
                        logits_r = logits_r.masked_fill(
                            obs_r["action_mask"] < 0.5, float("-inf")
                        )
                    logits_r = logits_r.masked_fill(invalid_r, float("-inf"))
                    action = torch.argmax(logits_r, dim=-1)
                    _, _, dones, _ = env_roll.step(action)

                costs = BatchSequenceDPSolver.solve(
                    job_sequences=env_roll.job_sequences,
                    processing_times=env_roll.p_subset,
                    ct=env_roll.ct,
                    e_single=env_roll.e_single,
                    T_limit=env_roll.T_limit,
                    sequence_lengths=env_roll.n_jobs.long(),
                )

                best_idx = int(torch.argmin(costs).item())
                best_cost = float(costs[best_idx].item())
                if best_cost < best_energy:
                    best_energy = best_cost
                    best_seq_tensor = (
                        env_roll.job_sequences[best_idx, :n].detach().cpu().long()
                    )
                    best_sequence = [int(x) for x in best_seq_tensor.tolist()]

                keep = min(int(beta), C_valid)
                keep_idx = torch.topk(-costs, k=keep).indices
                beam_sequences = cand_sequences[keep_idx].clone()
                beam_avail = cand_avail[keep_idx].clone()

            results.append(dp_schedule_for_job_sequence(single, best_sequence))

    return results


# =============================================================================
# Visualization
# =============================================================================


def visualize_schedule(
    out_path: Path,
    instance_data: Dict[str, Any],
    schedules: Dict[str, Dict[str, Any]],
    title_suffix: str = "",
):
    """Create Gantt-style visualization of schedules."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return

    n_jobs = int(instance_data["n_jobs"])
    T_limit = int(instance_data["T_limit"])
    Tk = instance_data["Tk"]
    ck = instance_data["ck"]
    period_starts = instance_data["period_starts"]
    K = int(instance_data["K"])

    method_order = list(schedules.keys())

    fig_h = 2.0 + 0.8 * len(method_order)
    fig, (ax_top, ax_jobs) = plt.subplots(
        2,
        1,
        figsize=(16, fig_h),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, max(2.0, 0.7 * len(method_order))]},
    )

    # Price period colors
    price_colors = {1: "#d4f1d4", 2: "#fff4b3", 3: "#ffd9b3", 4: "#ffb3b3"}

    # Draw price periods
    for k in range(K):
        start = int(period_starts[k])
        dur = int(Tk[k])
        price = int(ck[k])
        if start >= T_limit:
            continue
        dur = min(dur, T_limit - start)
        rect = mpatches.Rectangle(
            (start, 0),
            dur,
            1.0,
            facecolor=price_colors.get(price, "#cccccc"),
            edgecolor="black",
            linewidth=0.7,
        )
        ax_top.add_patch(rect)
        if dur >= 3:
            ax_top.text(
                start + dur / 2,
                0.5,
                f"p={price}",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
            )

    ax_top.set_ylim(0, 1)
    ax_top.set_yticks([])
    ax_top.set_title(
        f"Schedule Comparison | n_jobs={n_jobs} T_limit={T_limit}{title_suffix}",
        fontsize=11,
        fontweight="bold",
    )

    legend_elements = [
        mpatches.Patch(facecolor=price_colors[c], edgecolor="black", label=f"price={c}")
        for c in sorted(price_colors.keys())
    ]
    ax_top.legend(handles=legend_elements, loc="upper right", fontsize=7)

    # Draw job bars
    job_colors = plt.cm.Set3(np.linspace(0, 1, max(1, n_jobs)))

    method_labels = []
    for m in method_order:
        s = schedules[m]
        label = f"{m} (E={s['energy']:.1f}"
        if "cmax" in s:
            label += f", Cmax={s['cmax']}"
        label += ")"
        if not s.get("complete", True):
            label += " [INCOMPLETE]"
        method_labels.append(label)

    y_positions = list(range(len(method_order)))
    ax_jobs.set_yticks(y_positions)
    ax_jobs.set_yticklabels(method_labels, fontsize=8)

    for row, m in enumerate(method_order):
        bars = schedules[m]["bars"]
        for b in bars:
            job_id = int(b["job_id"])
            start = int(b["start"])
            end = int(b["end"])
            dur = max(0, end - start)
            if dur <= 0:
                continue
            ax_jobs.barh(
                row,
                dur,
                left=start,
                height=0.6,
                color=job_colors[job_id % len(job_colors)],
                edgecolor="black",
                linewidth=0.8,
            )
            if dur >= 4:
                ax_jobs.text(
                    start + dur / 2,
                    row,
                    f"J{job_id}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    fontweight="bold",
                )

    ax_jobs.set_xlim(0, T_limit)
    ax_jobs.set_xlabel("Time")
    ax_jobs.grid(axis="x", alpha=0.25, linestyle="--")
    ax_jobs.axvline(
        T_limit, color="red", linestyle="--", linewidth=1.5, label="Deadline"
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization: {out_path}")


# =============================================================================
# Model Loading
# =============================================================================


def load_models(
    variants: List[str],
    device: torch.device,
    base_path: Path,
) -> Dict[str, Tuple[Any, Any, bool]]:
    """Load models for specified variants.

    Returns: Dict[variant_name -> (model, variant_config, is_q_learning)]
    """
    models = {}

    for variant_name in variants:
        if variant_name not in MODEL_PATHS:
            print(f"WARNING: Unknown variant '{variant_name}', skipping")
            continue

        ckpt_path = base_path / MODEL_PATHS[variant_name]
        if not ckpt_path.exists():
            print(
                f"WARNING: Checkpoint not found for '{variant_name}': {ckpt_path}, skipping"
            )
            continue

        variant_id = VARIANT_IDS[variant_name]
        variant_config = get_variant_config(variant_id)
        is_q_learning = variant_name in Q_LEARNING_VARIANTS

        print(f"Loading {variant_name} from {ckpt_path}...")
        ckpt = _load_checkpoint(ckpt_path, device)

        if is_q_learning:
            model = build_q_model(variant_config).to(device)
            model_state = _extract_q_model_state(ckpt)
        else:
            model = build_model(variant_config).to(device)
            model_state = _extract_model_state(ckpt)

        model.load_state_dict(model_state)
        model.eval()

        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Loaded {variant_name}: {param_count:,} parameters")

        models[variant_name] = (model, variant_config, is_q_learning)

    return models


# =============================================================================
# Main Evaluation
# =============================================================================


def run_evaluation(
    args, models: Dict[str, Tuple[Any, Any, bool]], device: torch.device
):
    """Run evaluation comparing all loaded models."""

    # Parse slack ratios
    if args.slack_ratios:
        ratios = [float(x) for x in args.slack_ratios.split(",")]
    else:
        ratios = [0.2, 0.3, 0.4]

    # Use first variant's data config as reference
    first_variant = list(models.keys())[0]
    _, ref_config, _ = models[first_variant]
    data_cfg = _restrict_data_config(ref_config.data, args.scale)

    py_rng = random.Random(args.eval_seed)

    out_dir = Path(args.out_dir) if args.out_dir else Path("unified_comparison_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for ratio in ratios:
        print(f"\n{'='*70}")
        print(f"EVALUATING SLACK RATIO: {ratio:.2f}")
        print(f"{'='*70}")

        # Generate instances
        episodes = []
        for _ in range(args.num_instances):
            raw = generate_raw_instance(data_cfg, py_rng)
            assignments = simulate_metaheuristic_assignment(raw.n, raw.m, py_rng)
            non_empty = [i for i, a in enumerate(assignments) if len(a) > 0]
            m_idx = py_rng.choice(non_empty) if non_empty else 0
            job_idxs = assignments[m_idx]

            ep = make_single_machine_episode(
                raw,
                m_idx,
                job_idxs,
                py_rng,
                deadline_slack_ratio_min=ratio,
                deadline_slack_ratio_max=ratio,
            )
            episodes.append(ep)

        batch = batch_from_episodes(
            episodes,
            N_job_pad=int(ref_config.env.N_job_pad),
            K_period_pad=250,
            T_max_pad=500,
        )

        # Run DP baselines
        spt_res = spt_lpt_with_dp(ref_config.env, device, batch, which="spt")
        lpt_res = spt_lpt_with_dp(ref_config.env, device, batch, which="lpt")

        # Store per-method results
        method_results = {
            "spt_dp": spt_res,
            "lpt_dp": lpt_res,
        }

        # Evaluate each model
        for variant_name, (model, variant_config, is_q_learning) in models.items():
            print(f"\n  Evaluating {variant_name}...")

            if is_q_learning:
                # Q-Learning model
                if args.use_sgbs:
                    results = sgbs_q_sequence(
                        model,
                        variant_config,
                        batch,
                        device,
                        beta=args.sgbs_beta,
                        gamma=args.sgbs_gamma,
                    )
                else:
                    results = greedy_decode_q_sequence(
                        model, variant_config, batch, device
                    )
                method_results[variant_name] = results
            elif variant_name in SEQUENCE_POLICY_VARIANTS:
                # PPO sequence-only model: decode job order, then DP schedule
                if args.use_sgbs:
                    results = sgbs_ppo_sequence(
                        model,
                        variant_config,
                        batch,
                        device,
                        beta=args.sgbs_beta,
                        gamma=args.sgbs_gamma,
                    )
                else:
                    results = greedy_decode_ppo_sequence(
                        model, variant_config, batch, device
                    )
                method_results[variant_name] = results
            else:
                # PPO model - run with SGBS or greedy
                if args.use_sgbs:
                    results = sgbs(
                        model=model,
                        env_config=variant_config.env,
                        device=device,
                        batch_data=batch,
                        beta=args.sgbs_beta,
                        gamma=args.sgbs_gamma,
                    )
                else:
                    results = greedy_decode(model, variant_config.env, device, batch)
                method_results[variant_name] = results

        # Process results for each instance
        for i in range(len(episodes)):
            single = _slice_single_instance(batch, i)
            n_jobs = int(single["n_jobs"][0])
            T_limit = int(single["T_limit"][0])
            p_subset = single["p_subset"][0][:n_jobs].astype(np.int32)
            ct = single["ct"][0]
            e_single = int(single["e_single"][0])

            row = {
                "instance_idx": i,
                "slack_ratio": ratio,
                "n_jobs": n_jobs,
                "T_limit": T_limit,
                "spt_dp_energy": spt_res[i].total_energy,
                "lpt_dp_energy": lpt_res[i].total_energy,
            }

            # Add results for each variant
            for variant_name, (model, variant_config, is_q_learning) in models.items():
                results = method_results[variant_name]

                if is_q_learning or variant_name in SEQUENCE_POLICY_VARIANTS:
                    # DPResult
                    row[f"{variant_name}_energy"] = results[i].total_energy
                    continue

                # DecodeResult
                if (
                    getattr(variant_config.env, "use_price_families", False)
                    and not getattr(
                        variant_config.env, "use_duration_aware_families", False
                    )
                    and results[i].actions is not None
                ):
                    # Normal decoding energy is already what the env used during decode.
                    if args.decoding in ("normal", "both"):
                        _, _, normal_cmax = _action_trace_to_bars_normal(
                            results[i].actions,
                            variant_config.env,
                            p_subset,
                            ct,
                            e_single,
                            T_limit,
                        )
                        row[f"{variant_name}_normal_energy"] = results[i].total_energy
                        row[f"{variant_name}_normal_cmax"] = normal_cmax

                    # Progressive decoding is a post-processing heuristic on the same action trace.
                    # ('cmax_aware' is kept as a backward-compatible alias.)
                    if args.decoding in ("progressive", "cmax_aware", "both"):
                        _, cmax_energy, cmax_cmax = _action_trace_to_bars_cmax_aware(
                            results[i].actions,
                            variant_config.env,
                            p_subset,
                            ct,
                            e_single,
                            T_limit,
                        )
                        row[f"{variant_name}_progressive_energy"] = cmax_energy
                        row[f"{variant_name}_progressive_cmax"] = cmax_cmax
                else:
                    row[f"{variant_name}_energy"] = results[i].total_energy

            all_results.append(row)

        # Visualizations
        if args.num_viz > 0:
            for viz_idx in range(min(args.num_viz, len(episodes))):
                single = _slice_single_instance(batch, viz_idx)
                n_jobs = int(single["n_jobs"][0])
                T_limit = int(single["T_limit"][0])
                p_subset = single["p_subset"][0][:n_jobs].astype(np.int32)
                ct = single["ct"][0]
                e_single = int(single["e_single"][0])

                schedules = {}

                # SPT+DP
                schedules["SPT+DP"] = {
                    "energy": spt_res[viz_idx].total_energy,
                    "bars": _sequence_schedule_to_bars(
                        spt_res[viz_idx].job_sequence,
                        spt_res[viz_idx].start_times,
                        p_subset,
                    ),
                }

                # LPT+DP
                schedules["LPT+DP"] = {
                    "energy": lpt_res[viz_idx].total_energy,
                    "bars": _sequence_schedule_to_bars(
                        lpt_res[viz_idx].job_sequence,
                        lpt_res[viz_idx].start_times,
                        p_subset,
                    ),
                }

                # Add each variant
                for variant_name, (
                    model,
                    variant_config,
                    is_q_learning,
                ) in models.items():
                    results = method_results[variant_name]

                    if is_q_learning or variant_name in SEQUENCE_POLICY_VARIANTS:
                        # DPResult
                        schedules[variant_name] = {
                            "energy": results[viz_idx].total_energy,
                            "bars": _sequence_schedule_to_bars(
                                results[viz_idx].job_sequence,
                                results[viz_idx].start_times,
                                p_subset,
                            ),
                        }
                        continue

                    # DecodeResult
                    if (
                        getattr(variant_config.env, "use_price_families", False)
                        and not getattr(
                            variant_config.env, "use_duration_aware_families", False
                        )
                        and results[viz_idx].actions is not None
                    ):
                        if args.decoding in ("normal", "both"):
                            bars, _, cmax = _action_trace_to_bars_normal(
                                results[viz_idx].actions,
                                variant_config.env,
                                p_subset,
                                ct,
                                e_single,
                                T_limit,
                            )
                            schedules[f"{variant_name} (normal)"] = {
                                "energy": results[viz_idx].total_energy,
                                "cmax": cmax,
                                "bars": bars,
                            }

                        if args.decoding in ("progressive", "cmax_aware", "both"):
                            bars, energy, cmax = _action_trace_to_bars_cmax_aware(
                                results[viz_idx].actions,
                                variant_config.env,
                                p_subset,
                                ct,
                                e_single,
                                T_limit,
                            )
                            schedules[f"{variant_name} (progressive)"] = {
                                "energy": energy,
                                "cmax": cmax,
                                "bars": bars,
                            }

                fname = f"comparison_slack{ratio:.2f}_idx{viz_idx}.png"
                visualize_schedule(
                    out_dir / fname,
                    {
                        "n_jobs": n_jobs,
                        "T_limit": T_limit,
                        "Tk": single["Tk"][0],
                        "ck": single["ck"][0],
                        "period_starts": single["period_starts"][0],
                        "K": int(single["K"][0]),
                    },
                    schedules,
                    f" | seed={args.eval_seed} slack={ratio:.2f}",
                )

    # Save results to CSV
    import pandas as pd

    df = pd.DataFrame(all_results)
    csv_path = out_dir / f"comparison_results_seed{args.eval_seed}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY (Mean Energy across all instances)")
    print("=" * 80)

    energy_cols = [c for c in df.columns if "energy" in c.lower()]
    summary = df[energy_cols].mean()
    for col, val in summary.items():
        print(f"  {col}: {val:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified comparison of all PPO and Q-Learning model variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available variants: {', '.join(ALL_VARIANTS)}

Default variants (CWE excluded): {', '.join(DEFAULT_VARIANTS)}

Example usage:
  # Compare all variants
  python -m PaST.cli.eval.run_unified_comparison --eval_seed 55 --num_instances 16

  # Compare specific variants with SGBS
  python -m PaST.cli.eval.run_unified_comparison \\
      --variants ppo_family_best q_seq_cwe \\
      --use_sgbs --sgbs_beta 8 --sgbs_gamma 4

  # Compare with both decoding rules
  python -m PaST.cli.eval.run_unified_comparison --decoding both
""",
    )

    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=None,
        help=f"Variants to evaluate (default: {', '.join(DEFAULT_VARIANTS)}). Choices: {', '.join(ALL_VARIANTS)}",
    )
    parser.add_argument(
        "--include_cwe",
        action="store_true",
        help="Include CWE-architecture checkpoints in the default variant list (ignored if --variants is provided)",
    )
    parser.add_argument(
        "--decoding",
        type=str,
        default="both",
        choices=["normal", "progressive", "cmax_aware", "both"],
        help="Decoding rule for PPO price-family variants. 'cmax_aware' is an alias for 'progressive' (default: both)",
    )
    parser.add_argument(
        "--use_sgbs",
        action="store_true",
        help="Enable SGBS decoding for all variants",
    )
    parser.add_argument(
        "--sgbs_beta",
        type=int,
        default=4,
        help="SGBS beam width (default: 4)",
    )
    parser.add_argument(
        "--sgbs_gamma",
        type=int,
        default=4,
        help="SGBS expansion factor (default: 4)",
    )
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=55,
        help="Random seed for instance generation",
    )
    parser.add_argument(
        "--num_instances",
        type=int,
        default=16,
        help="Number of instances to evaluate",
    )
    parser.add_argument(
        "--num_viz",
        type=int,
        default=2,
        help="Number of visualizations to generate per slack ratio",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="small",
        choices=["small", "medium", "large"],
        help="Instance scale filter",
    )
    parser.add_argument(
        "--slack_ratios",
        type=str,
        default=None,
        help="Comma-separated slack ratios (default: 0.2,0.3,0.4)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use",
    )

    args = parser.parse_args()

    # Determine variants to evaluate
    if args.variants:
        variants = args.variants
    else:
        variants = list(DEFAULT_VARIANTS)
        if args.include_cwe:
            variants.extend([v for v in ALL_VARIANTS if v in CWE_VARIANTS])

    # Set device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Find base path for checkpoints (PaST package root contains runs_p100/)
    base_path = Path(__file__).resolve().parent.parent.parent

    # Load models
    print("\n" + "=" * 70)
    print("LOADING MODELS")
    print("=" * 70)
    models = load_models(variants, device, base_path)

    if not models:
        print("ERROR: No models loaded. Check checkpoint paths.")
        return

    # Run evaluation
    run_evaluation(args, models, device)


if __name__ == "__main__":
    main()

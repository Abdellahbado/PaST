"""Quick local sanity test for Q-sequence collection.

Goals (fast on a laptop):
1) Prove DP multiprocessing path runs (dp_eval_workers>1).
2) Check the 'masked +inf is expected' behavior doesn't trigger non-finite handling.

Run:
  conda run -n new-ml-env python -u PaST/dev/quick_test_qseq_parallel_and_inf.py

This script intentionally uses a tiny synthetic distribution (DataConfig overrides)
so it completes in seconds.
"""

from __future__ import annotations

import argparse
import time
from typing import List
from pathlib import Path
import sys

import numpy as np
import torch

# Allow running this file directly (e.g., via `conda run ... python path/to/script.py`).
_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))

from PaST.config import VariantID, get_variant_config
from PaST.q_sequence_model import build_q_model
from PaST.train_q_sequence import (
    QTransitionDataset,
    collate_q_batch,
    collect_round_data,
)


def _override_data_config_for_local_quick(dc):
    dc.sampling_mode = "uniform_range"
    dc.T_max_choices = [50]
    dc.m_min = 3
    dc.m_max = 3
    dc.n_min = 6
    dc.n_max = 8
    dc.p_min = 1
    dc.p_max = 4
    dc.ck_min = 1
    dc.ck_max = 4
    dc.e_min = 1
    dc.e_max = 3
    return dc


def _run_collection_once(*, dp_workers: int, episodes: int, seed: int) -> float:
    variant_config = get_variant_config(VariantID.Q_SEQUENCE, seed=seed)
    data_config = _override_data_config_for_local_quick(variant_config.data)

    device = torch.device("cpu")

    t0 = time.time()
    transitions = collect_round_data(
        env_config=variant_config.env,
        model=None,
        teacher_model=None,
        variant_config=variant_config,
        data_config=data_config,
        num_episodes=episodes,
        num_counterfactuals=2,
        exploration_eps=1.0,
        use_model_completion=False,
        heuristic_policy="spt",
        target_normalization="state_min",
        include_heuristic_candidates=True,
        target_rollouts="spt",
        target_rollout_aggregation="single",
        target_num_random_rollouts=0,
        target_softmin_tau=1.0,
        device=device,
        seed=seed,
        collection_batch_size=episodes,
        num_collection_workers=1,
        allow_gpu_collection_multiprocessing=False,
        num_cpu_threads=4,
        dp_eval_device="cpu",
        dp_eval_workers=int(dp_workers),
        dp_flush_threshold=64,
        dp_eval_async=False,
        collection_log_every_batches=1,
    )
    dt = time.time() - t0

    if len(transitions) == 0:
        raise RuntimeError(
            "No transitions collected; something is wrong with collection."
        )

    # Basic finiteness check on stored obs/targets
    bad = 0
    for t in transitions:
        if not np.isfinite(t.jobs).all():
            bad += 1
        if not np.isfinite(t.periods).all():
            bad += 1
        if not np.isfinite(t.ctx).all():
            bad += 1
        if not np.isfinite(t.q_target):
            bad += 1
    if bad:
        raise RuntimeError(f"Collected {bad} non-finite fields in transitions.")

    print(
        f"Collected {len(transitions)} transitions with dp_workers={dp_workers} in {dt:.3f}s",
        flush=True,
    )
    return dt


def _check_masked_inf_behavior(seed: int) -> None:
    variant_config = get_variant_config(VariantID.Q_SEQUENCE, seed=seed)
    data_config = _override_data_config_for_local_quick(variant_config.data)

    device = torch.device("cpu")
    transitions = collect_round_data(
        env_config=variant_config.env,
        model=None,
        teacher_model=None,
        variant_config=variant_config,
        data_config=data_config,
        num_episodes=2,
        num_counterfactuals=2,
        exploration_eps=1.0,
        use_model_completion=False,
        heuristic_policy="spt",
        target_normalization="state_min",
        include_heuristic_candidates=True,
        target_rollouts="spt",
        target_rollout_aggregation="single",
        target_num_random_rollouts=0,
        target_softmin_tau=1.0,
        device=device,
        seed=seed,
        collection_batch_size=2,
        num_collection_workers=1,
        allow_gpu_collection_multiprocessing=False,
        num_cpu_threads=4,
        dp_eval_device="cpu",
        dp_eval_workers=1,
        dp_flush_threshold=64,
        dp_eval_async=False,
        collection_log_every_batches=0,
    )

    ds = QTransitionDataset(transitions)
    batch = collate_q_batch([ds[i] for i in range(min(8, len(ds)))])

    model = build_q_model(variant_config).to(device)
    model.eval()

    with torch.no_grad():
        q_values = model(
            jobs=batch["jobs"].to(device),
            periods_local=batch["periods"].to(device),
            ctx=batch["ctx"].to(device),
            job_mask=batch["job_mask"].to(device),
            period_mask=batch["period_mask"].to(device),
        )

    job_mask = batch["job_mask"].to(device)
    valid_pos = ~job_mask

    if valid_pos.any() and (not torch.isfinite(q_values[valid_pos]).all().item()):
        raise RuntimeError(
            "Non-finite q_values on valid positions (this is a real bug)."
        )

    masked_has_inf = (
        bool(torch.isinf(q_values[job_mask]).any().item()) if job_mask.any() else False
    )
    print(
        "Masked-inf behavior OK: valid positions finite; "
        f"masked_has_inf={masked_has_inf}",
        flush=True,
    )


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--episodes", type=int, default=2)
    args = p.parse_args(argv)

    print("== Quick parallelization check ==", flush=True)
    t1 = _run_collection_once(dp_workers=1, episodes=args.episodes, seed=args.seed)
    t2 = _run_collection_once(dp_workers=2, episodes=args.episodes, seed=args.seed + 1)
    print(f"dp_workers=1 time: {t1:.3f}s | dp_workers=2 time: {t2:.3f}s", flush=True)

    print("== Quick infinity/masking check ==", flush=True)
    _check_masked_inf_behavior(seed=args.seed + 2)

    print("OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

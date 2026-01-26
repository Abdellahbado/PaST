"""Quick forward-pass sanity checks for Q-sequence model variants.

Runs a single forward pass for each selected variant to catch shape/runtime bugs
without needing to run training.

Usage:
  python -m PaST.sanity_check_q_sequence_model
  python -m PaST.sanity_check_q_sequence_model --device cpu
"""

from __future__ import annotations

import argparse

import torch

from PaST.config import VariantID, get_variant_config
from PaST.q_sequence_model import build_q_model


def _make_dummy_obs(config, batch_size: int, device: torch.device):
    env = config.env

    B = batch_size
    N = env.M_job_bins
    K = env.K_period_local

    jobs = torch.randn(B, N, env.F_job, device=device)
    periods_local = torch.randn(B, K, env.F_period, device=device)
    ctx = torch.randn(B, env.F_ctx, device=device)

    # A couple invalids to exercise masking paths.
    job_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    period_mask = torch.zeros(B, K, dtype=torch.bool, device=device)
    if N >= 2:
        job_mask[:, -1] = True
    if K >= 2:
        period_mask[:, -1] = True

    return jobs, periods_local, ctx, job_mask, period_mask


def _check_variant(variant_id: VariantID, device: torch.device) -> None:
    config = get_variant_config(variant_id)
    model = build_q_model(config).to(device)
    model.eval()

    jobs, periods_local, ctx, job_mask, period_mask = _make_dummy_obs(
        config, batch_size=2, device=device
    )

    with torch.no_grad():
        q = model(
            jobs=jobs,
            periods_local=periods_local,
            ctx=ctx,
            job_mask=job_mask,
            period_mask=period_mask,
            periods_full=None,
            period_full_mask=None,
        )

    assert q.ndim == 2, f"{variant_id.value}: expected q.ndim=2, got {q.shape}"
    assert q.shape[0] == jobs.shape[0], f"{variant_id.value}: bad batch dim"
    assert q.shape[1] == jobs.shape[1], f"{variant_id.value}: bad action dim"

    # Ensure masked jobs are inf (as the head enforces).
    if job_mask.any():
        masked_vals = q[job_mask]
        assert torch.isinf(masked_vals).all(), f"{variant_id.value}: mask not applied"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    variants = [
        VariantID.Q_SEQUENCE,
        VariantID.Q_SEQUENCE_CTX13,
        VariantID.Q_SEQUENCE_CNN,
        VariantID.Q_SEQUENCE_CNN_CTX13,
        VariantID.Q_SEQUENCE_CWE,
        VariantID.Q_SEQUENCE_CWE_CTX13,
    ]

    print("Device:", device)
    for v in variants:
        _check_variant(v, device)
        print("OK:", v.value)


if __name__ == "__main__":
    main()

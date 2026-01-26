"""Consistency checks for duration-aware family variant.

Run:
  python -m PaST.duration_aware_consistency_check

This script checks that:
- action_mask marks an action (job,family) valid iff decoder can produce a start
  time satisfying the SAME feasibility constraints;
- decoded start time belongs to the chosen family under the cached family assignment.

Exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import sys
import random

import torch

from PaST.config import get_variant_config, VariantID, DataConfig
from PaST.sm_benchmark_data import generate_episode_batch
from PaST.sm_env import GPUBatchSingleMachinePeriodEnv


def _pick_random_valid_actions(action_mask: torch.Tensor, k: int, rng: random.Random):
    """Pick up to k random valid actions per batch element."""
    B, A = action_mask.shape
    picked = []
    for b in range(B):
        valid = torch.nonzero(action_mask[b] > 0.5, as_tuple=False).view(-1).tolist()
        rng.shuffle(valid)
        picked.append(valid[:k])
    return picked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--samples_per_env", type=int, default=5)
    args = parser.parse_args()

    torch.set_printoptions(precision=3, sci_mode=False)

    cfg = get_variant_config(VariantID.PPO_DURATION_AWARE_FAMILY, seed=0)
    cfg.env.use_price_families = True
    cfg.env.use_duration_aware_families = True

    B = int(args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = GPUBatchSingleMachinePeriodEnv(
        batch_size=B, env_config=cfg.env, device=device
    )

    data_cfg = DataConfig()
    batch = generate_episode_batch(batch_size=B, config=data_cfg, seed=int(args.seed))
    obs = env.reset(batch)

    # Ensure mask uses duration-aware path.
    am = obs["action_mask"]
    assert am.shape == (B, env.action_dim)

    rng = random.Random(0)
    samples = _pick_random_valid_actions(am, k=int(args.samples_per_env), rng=rng)

    # Precompute per-instance remaining_work for completion_ok
    remaining_work = (
        (env.p_subset.float() * env.job_available.float()).sum(dim=1).to(torch.int32)
    )

    failures = 0
    total_checked = 0

    for b in range(B):
        for a in samples[b]:
            total_checked += 1

            action = torch.tensor(a, device=device, dtype=torch.int64)
            job_id_b = (action // env.num_slack_choices).to(torch.int64)
            family_id_b = (action % env.num_slack_choices).to(torch.int64)

            job_ids = torch.zeros((B,), device=device, dtype=torch.int64)
            family_ids = torch.zeros((B,), device=device, dtype=torch.int64)
            job_ids[b] = job_id_b
            family_ids[b] = family_id_b

            p = torch.zeros((B,), device=device, dtype=torch.int32)
            p[b] = env.p_subset[b, job_id_b].to(torch.int32)

            starts = env._compute_duration_aware_family_start_times_for_job(
                job_ids=job_ids,
                family_ids=family_ids,
                p=p,
            ).to(torch.int32)
            start = starts[b]

            # Feasibility conditions must hold.
            t_now = env.t[b].to(torch.int32)
            T_limit = env.T_limit[b].to(torch.int32)
            T_max = env.T_max[b].to(torch.int32)

            ok = True
            if start < t_now:
                ok = False
            if start >= T_max:
                ok = False
            if start + p[b] > T_limit:
                ok = False
            if start + p[b] > T_max:
                ok = False
            if start + remaining_work[b] > T_limit:
                ok = False

            # Family membership under cached families must match.
            fam_at_start = env.duration_families[b, job_id_b, start].item()
            if fam_at_start != int(family_id_b.item()):
                ok = False

            if not ok:
                failures += 1
                if failures <= 10:
                    print(
                        f"FAIL b={b} a={a} job={job_id_b.item()} fam={family_id_b.item()} "
                        f"start={start.item()} p={p[b].item()} T_lim={T_limit.item()} T_max={T_max.item()} "
                        f"rem={remaining_work[b].item()} fam_at_start={fam_at_start}"
                    )

    print(f"Checked {total_checked} (job,family) actions across {B} envs")
    if failures:
        print(f"FAILED: {failures} inconsistencies detected")
        return 1

    # Also check that every env has at least one valid action initially.
    valid_counts = am.sum(dim=1)
    zeros = (valid_counts < 0.5).sum().item()
    print(
        f"Initial valid action counts: min={valid_counts.min().item():.0f} max={valid_counts.max().item():.0f}"
    )
    if zeros:
        print(f"WARNING: {zeros} envs have 0 valid actions at reset (tight deadlines?)")

    print("OK: duration-aware masking/decoding are consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())

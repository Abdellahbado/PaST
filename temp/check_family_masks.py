import torch

from PaST.config import get_ppo_family_q4, get_ppo_family_q4_ctx13
from PaST.past_sm_model import build_model
from PaST.sm_benchmark_data import generate_episode_batch
from PaST.sm_env import GPUBatchSingleMachinePeriodEnv


def check(factory, batch_size: int = 64, seed: int = 0) -> None:
    cfg = factory()
    device = torch.device("cpu")

    env = GPUBatchSingleMachinePeriodEnv(
        batch_size=batch_size, env_config=cfg.env, device=device
    )
    batch = generate_episode_batch(
        batch_size=batch_size,
        config=cfg.data,
        seed=seed,
        N_job_pad=cfg.env.N_job_pad,
        K_period_pad=cfg.env.K_period_full_max,
        T_max_pad=500,
    )
    obs = env.reset(batch)

    model = build_model(cfg).to(device).eval()

    # Convert env float masks (1=valid) into model bool masks (True=invalid).
    job_mask = obs.get("job_mask")
    period_mask = obs.get("period_mask")
    if job_mask is not None and job_mask.dtype is not torch.bool:
        job_mask = job_mask < 0.5
    if period_mask is not None and period_mask.dtype is not torch.bool:
        period_mask = period_mask < 0.5

    with torch.no_grad():
        logits, _ = model(
            jobs=obs["jobs"],
            periods_local=obs["periods"],
            ctx=obs["ctx"],
            job_mask=job_mask,
            period_mask=period_mask,
            periods_full=None,
            period_full_mask=None,
        )
        logits = logits.masked_fill(obs["action_mask"] == 0, float("-inf"))
        all_masked = ~torch.isfinite(logits).any(dim=-1)

    action_sums = obs["action_mask"].sum(dim=-1)

    print(cfg.variant_id.value)
    print(
        f"  action_dim={obs['action_mask'].shape[1]} all_masked={int(all_masked.sum().item())}/{batch_size} "
        f"action_mask_zeros={int((action_sums == 0).sum().item())}/{batch_size}"
    )


if __name__ == "__main__":
    check(get_ppo_family_q4)
    check(get_ppo_family_q4_ctx13)

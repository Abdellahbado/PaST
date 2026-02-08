#!/usr/bin/env python3
"""Smoke test for NeuroLS: exercises reset → features → forward → loss."""

from __future__ import annotations
import sys, os, traceback

# Ensure PaST is importable
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np, random


def main():
    print("=" * 60)
    print("NeuroLS smoke test")
    print("=" * 60)

    # ── 1. Generate a tiny instance ──────────────────────────────
    print("\n[1] Generating instance …")
    from PaST.data.sm_benchmark_data import generate_raw_instance
    from PaST.config import DataConfig

    cfg = DataConfig(sampling_mode="new_benchmark_grid")
    rng = random.Random(42)
    instance = generate_raw_instance(
        config=cfg, rng=rng, instance_id=0, n=10, m=3, D_days=2
    )
    K = 40
    print(
        f"    n={len(instance.p)}, m={instance.m}, K={K}, " f"ct_len={len(instance.ct)}"
    )

    # ── 2. Reset environment ─────────────────────────────────────
    print("\n[2] Resetting env …")
    from sandbox.neurols.env import NeuroLSEnv, EnvConfig

    env = NeuroLSEnv(EnvConfig(max_steps=10, stagnation_limit=5, action_space="AA"))
    env.seed(42)
    state = env.reset(instance, K)
    print(f"    initial_cost={state.current_cost:.2f}")

    # ── 3. Extract features ──────────────────────────────────────
    print("\n[3] Extracting features …")
    feats = env.get_state_features()
    for k, v in feats.items():
        print(f"    {k:25s} shape={np.asarray(v).shape}  dtype={np.asarray(v).dtype}")

    # ── 4. Build model (price_mode=none first, then full) ────────
    import torch
    from sandbox.neurols.decoder import NeuroLSPolicy
    from sandbox.neurols.action_space import AA_SPACE

    device = "cpu"

    for price_mode in ["none", "full"]:
        print(f"\n[4-{price_mode}] Building NeuroLSPolicy(price_mode={price_mode}) …")
        net = NeuroLSPolicy(
            d_job_in=5,
            d_machine_in=5,
            d_state_in=13,
            d_price_in=64,
            d_emb=64,
            n_actions=AA_SPACE.n_actions,
            n_layers_static=2,
            n_layers_dynamic=1,
            use_iqn=True,
            price_mode=price_mode,
            dropout=0.0,
        ).to(device)
        n_params = sum(p.numel() for p in net.parameters())
        print(f"    params={n_params:,}")

        # ── 5. Single-sample forward (like _get_action) ─────────
        print(f"[5-{price_mode}] Single-sample forward …")
        t = {k: torch.tensor(v, device=device) for k, v in feats.items()}
        with torch.no_grad():
            q = net(
                job_features=t["job_features"].float(),
                machine_features=t["machine_features"].float(),
                state_features=t["state_features"].float(),
                job_to_machine=t["job_to_machine"].long(),
                static_edge_index=t["static_edge_index"].long(),
                dynamic_edge_index=t["dynamic_edge_index"].long(),
                price_features=t.get("price_per_hour", None),
            )
        print(f"    q_values shape={q.shape}  values={q.detach().numpy()}")

        # ── 6. Batched forward (like _forward_batch) ─────────────
        print(f"[6-{price_mode}] Batched forward (B=4) via loop …")
        B = 4
        batch_feats = {
            k: torch.stack([torch.tensor(v, device=device)] * B)
            for k, v in feats.items()
        }
        q_list = []
        has_pp = "price_per_hour" in batch_feats
        for i in range(B):
            qi = net(
                job_features=batch_feats["job_features"][i].float(),
                machine_features=batch_feats["machine_features"][i].float(),
                state_features=batch_feats["state_features"][i].float(),
                job_to_machine=batch_feats["job_to_machine"][i].long(),
                static_edge_index=batch_feats["static_edge_index"][i].long(),
                dynamic_edge_index=batch_feats["dynamic_edge_index"][i].long(),
                price_features=(
                    batch_feats["price_per_hour"][i].float() if has_pp else None
                ),
            )
            q_list.append(qi)
        q_batch = torch.stack(q_list, dim=0)
        print(f"    q_batch shape={q_batch.shape}")

        # ── 7. IQN forward ───────────────────────────────────────
        print(f"[7-{price_mode}] IQN forward (B=4, N=8) via loop …")
        tau = torch.rand(B, 8, device=device)
        q_iqn_list = []
        for i in range(B):
            qi = net(
                job_features=batch_feats["job_features"][i].float(),
                machine_features=batch_feats["machine_features"][i].float(),
                state_features=batch_feats["state_features"][i].float(),
                job_to_machine=batch_feats["job_to_machine"][i].long(),
                static_edge_index=batch_feats["static_edge_index"][i].long(),
                dynamic_edge_index=batch_feats["dynamic_edge_index"][i].long(),
                price_features=(
                    batch_feats["price_per_hour"][i].float() if has_pp else None
                ),
                tau=tau[i : i + 1],
            )
            q_iqn_list.append(qi)
        q_iqn = torch.stack(q_iqn_list, dim=0)
        print(f"    q_iqn shape={q_iqn.shape}")

    # ── 8. Run a tiny env episode ────────────────────────────────
    print("\n[8] Running 5-step episode …")
    net_aa = NeuroLSPolicy(
        d_job_in=5,
        d_machine_in=5,
        d_state_in=13,
        d_price_in=64,
        d_emb=64,
        n_actions=AA_SPACE.n_actions,
        n_layers_static=2,
        n_layers_dynamic=1,
        use_iqn=True,
        price_mode="none",
        dropout=0.0,
    ).to(device)
    state = env.reset(instance, K)
    sf = env.get_state_features()
    for step in range(5):
        t = {k: torch.tensor(v, device=device) for k, v in sf.items()}
        with torch.no_grad():
            q = net_aa(
                job_features=t["job_features"].float(),
                machine_features=t["machine_features"].float(),
                state_features=t["state_features"].float(),
                job_to_machine=t["job_to_machine"].long(),
                static_edge_index=t["static_edge_index"].long(),
                dynamic_edge_index=t["dynamic_edge_index"].long(),
                price_features=None,
            )
        action = q.argmax().item()
        state, reward, done, info = env.step(action)
        sf = env.get_state_features()
        print(f"    step={step}  action={action}  reward={reward:.4f}  done={done}")
        if done:
            break

    # ── 9. Backward pass (gradient check) ────────────────────────
    print("\n[9] Backward pass check …")
    t = {k: torch.tensor(v, device=device) for k, v in sf.items()}
    q = net_aa(
        job_features=t["job_features"].float(),
        machine_features=t["machine_features"].float(),
        state_features=t["state_features"].float(),
        job_to_machine=t["job_to_machine"].long(),
        static_edge_index=t["static_edge_index"].long(),
        dynamic_edge_index=t["dynamic_edge_index"].long(),
        price_features=None,
    )
    loss = q.mean()
    loss.backward()
    grad_norm = sum(
        p.grad.norm().item() for p in net_aa.parameters() if p.grad is not None
    )
    print(f"    loss={loss.item():.4f}  grad_norm={grad_norm:.4f}")

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

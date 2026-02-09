#!/usr/bin/env python3
"""Diagnostics to verify whether the encoder/policy actually uses TOU price inputs.

This is meant to answer: "Is price reachable?" and more importantly "Is price used?"

It runs three families of checks on a batch of collected states:
1) Permutation / ablation tests:
   - shuffle the 20h price profile tokens (phase permutation)
   - zero-out price tokens
   - zero-out machine_exposure (when available)
   and measure how much Q-values / argmax actions change.

2) Gradient sensitivity tests:
   - compute gradient norms of Q(s, a*) w.r.t. price_per_hour and machine_exposure.

3) Counterfactual price shift tests (phase shift only):
   - keep p/e/initial solution identical and rotate ct by a shift,
     then measure action agreement on the initial state.

Usage example:
  conda activate new-ml-env
  python -u -m PaST.scripts.price_usage_diagnostics \
    --checkpoint checkpoints/neurols/neurols_AANP_full/final.pt \
    --n-instances 16 --states-per-instance 10 --device cpu

Notes:
- This script does not change training behavior.
- It requires a checkpoint saved by PaST.neurols.train.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch

    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_checkpoint(path: Path, device: str) -> Dict[str, Any]:
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for this diagnostics script")
    return torch.load(str(path), map_location=device)


def _build_policy_from_checkpoint(ckpt: Dict[str, Any], device: str):
    from PaST.neurols.decoder import NeuroLSPolicy
    from PaST.neurols.action_space import get_action_space

    cfg = ckpt.get("config")
    if not isinstance(cfg, dict):
        raise ValueError("Checkpoint missing 'config' dict")

    action_space = get_action_space(str(cfg.get("action_space", "AANP")))

    model_kwargs = dict(
        d_emb=int(cfg.get("d_emb", 64)),
        n_actions=int(action_space.n_actions),
        n_layers_static=int(cfg.get("n_layers_static", 2)),
        n_layers_dynamic=int(cfg.get("n_layers_dynamic", 2)),
        use_iqn=bool(cfg.get("use_iqn", True)),
        use_dueling=bool(cfg.get("use_dueling", False)),
        price_mode=str(cfg.get("price_mode", "full")),
        dropout=float(cfg.get("dropout", 0.1)),
        graph_type=str(cfg.get("graph_type", "bipartite")),
    )

    net = NeuroLSPolicy(**model_kwargs).to(device)
    net.load_state_dict(ckpt["policy_net"])
    net.eval()
    return net, cfg


def _make_env_from_cfg(cfg: Dict[str, Any]):
    from PaST.neurols.env import NeuroLSEnv, EnvConfig

    env_config = EnvConfig(
        max_steps=int(cfg.get("max_steps", 500)),
        stagnation_limit=int(cfg.get("stagnation_limit", 100)),
        action_space=str(cfg.get("action_space", "AANP")),
        reward_mode=str(cfg.get("reward_mode", "dense_best")),
        improvement_scale=float(cfg.get("improvement_scale", 1.0)),
        reward_eps=float(cfg.get("reward_eps", 1e-8)),
        best_bonus_lambda=float(cfg.get("best_bonus_lambda", 0.3)),
        step_penalty=float(cfg.get("step_penalty", 0.0)),
        top_k=int(cfg.get("top_k", 10)),
        use_proxy=bool(cfg.get("use_proxy", True)),
        proxy_mode=str(cfg.get("proxy_mode", "load")),
        exposure_bonus_lambda=float(cfg.get("exposure_bonus_lambda", 0.0)),
        exposure_eps=float(cfg.get("exposure_eps", 1e-8)),
        graph_type=str(cfg.get("graph_type", "bipartite")),
        price_mode=str(cfg.get("price_mode", "full")),
    )

    env = NeuroLSEnv(env_config)
    env.seed(int(cfg.get("seed", 42)))
    return env


def _generate_instances_from_cfg(
    cfg: Dict[str, Any], n_instances: int, *, seed: int
) -> List[Tuple[Any, int]]:
    from PaST.data.sm_benchmark_data import generate_raw_instance
    from PaST.config import DataConfig

    data_config = DataConfig(sampling_mode="new_benchmark_grid")

    # Training enforces constant shapes; follow that.
    fixed_n = int((cfg.get("n_jobs_train") or [20])[0])
    fixed_m = int((cfg.get("n_machines_train") or [3])[0])
    fixed_K = int(cfg.get("K_fixed") or (cfg.get("K_range") or [40, 120])[0])

    hours_per_day = int(getattr(data_config, "hours_per_day", 20))
    if fixed_K % hours_per_day != 0:
        fixed_K = (fixed_K // hours_per_day) * hours_per_day
        fixed_K = max(fixed_K, hours_per_day)
    fixed_D_days = int(fixed_K // hours_per_day)

    import random as _random

    rng = _random.Random(int(seed))

    out = []
    for i in range(int(n_instances)):
        inst = generate_raw_instance(
            config=data_config,
            rng=rng,
            instance_id=i,
            n=fixed_n,
            m=fixed_m,
            D_days=fixed_D_days,
        )
        out.append((inst, fixed_K))
    return out


def _shift_ct(ct: List[int], shift: int) -> List[int]:
    if not ct:
        return ct
    s = int(shift) % len(ct)
    if s == 0:
        return list(ct)
    return list(ct[s:]) + list(ct[:s])


def _collect_states(
    env,
    instances: List[Tuple[Any, int]],
    *,
    states_per_instance: int,
    rollout_steps: int,
    seed: int,
) -> List[Dict[str, np.ndarray]]:
    rng = np.random.default_rng(int(seed))

    collected: List[Dict[str, np.ndarray]] = []
    for inst, K in instances:
        env.config.deterministic = True
        env.reset(inst, int(K))

        # Grab initial state + optional random transitions.
        for _ in range(int(states_per_instance)):
            collected.append(env.get_state_features())
            if int(rollout_steps) > 0:
                for _ in range(int(rollout_steps)):
                    a = int(rng.integers(0, env.action_space_size))
                    _s, _r, done, _info = env.step(a)
                    if bool(done):
                        break

    return collected


def _to_tensors(state: Dict[str, np.ndarray], device: str) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if v is None:
            continue
        out[k] = torch.tensor(v, device=device)
    return out


@torch.no_grad()
def _forward_q(net, st: Dict[str, torch.Tensor]) -> torch.Tensor:
    return net(
        job_features=st["job_features"].float(),
        machine_features=st["machine_features"].float(),
        state_features=st["state_features"].float(),
        job_to_machine=st["job_to_machine"].long(),
        static_edge_index=st["static_edge_index"].long(),
        dynamic_edge_index=st["dynamic_edge_index"].long(),
        price_features=st.get("price_per_hour", None),
        machine_exposure=(
            st.get("machine_exposure", None).float()
            if "machine_exposure" in st
            else None
        ),
        period_features=(
            st.get("period_features", None).float() if "period_features" in st else None
        ),
        tripartite_edge_index=(
            st.get("tripartite_edge_index", None).long()
            if "tripartite_edge_index" in st
            else None
        ),
    )


def _permute_price_per_hour(pf: torch.Tensor, *, seed: int) -> torch.Tensor:
    # pf: (H, 5)
    gen = torch.Generator(device=pf.device)
    gen.manual_seed(int(seed))
    H = pf.size(0)
    perm = torch.randperm(H, generator=gen, device=pf.device)
    return pf[perm]


def _zero_like(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x)


def _metrics_header() -> str:
    return "metric\tvalue\n" "------\t-----"


def main() -> None:
    if not _TORCH_AVAILABLE:
        raise SystemExit("PyTorch is required")

    ap = argparse.ArgumentParser(description="Diagnostics: does the policy use price?")
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/neurols/neurols_AANP_full/final.pt",
        help="Path to a .pt checkpoint saved by PaST.neurols.train",
    )
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-instances", type=int, default=16)
    ap.add_argument("--states-per-instance", type=int, default=10)
    ap.add_argument(
        "--rollout-steps",
        type=int,
        default=0,
        help="Extra random env.step() transitions per collected state (default: 0 for initial-only).",
    )
    ap.add_argument(
        "--price-shift",
        type=int,
        default=5,
        help="Phase shift (in hours) for counterfactual ct rotation.",
    )

    args = ap.parse_args()

    stage = "start"
    try:
        root = _repo_root()
        ckpt_path = (
            (root / args.checkpoint).resolve()
            if not Path(args.checkpoint).is_absolute()
            else Path(args.checkpoint)
        )
        if not ckpt_path.exists():
            raise SystemExit(f"Missing checkpoint: {ckpt_path}")

        device = str(args.device)
        stage = "load_checkpoint"
        print(f"[stage] {stage}", flush=True)
        ckpt = _load_checkpoint(ckpt_path, device)
        stage = "build_policy"
        print(f"[stage] {stage}", flush=True)
        net, cfg = _build_policy_from_checkpoint(ckpt, device)

        # Build env + instances
        stage = "build_env"
        print(f"[stage] {stage}", flush=True)
        env = _make_env_from_cfg(cfg)
        stage = "generate_instances"
        print(f"[stage] {stage}", flush=True)
        instances = _generate_instances_from_cfg(
            cfg, int(args.n_instances), seed=int(args.seed)
        )

        stage = "collect_states"
        print(f"[stage] {stage}", flush=True)
        states = _collect_states(
            env,
            instances,
            states_per_instance=int(args.states_per_instance),
            rollout_steps=int(args.rollout_steps),
            seed=int(args.seed) + 7,
        )
        if not states:
            raise SystemExit("No states collected")

        stage = "evaluate"
        print(f"[stage] {stage}", flush=True)

        # --- permutation / ablation tests ---
        action_same_pf_perm = 0
        action_same_pf_zero = 0
        action_same_me_zero = 0
        qdiff_pf_perm = []
        qdiff_pf_zero = []
        qdiff_me_zero = []

        baseline_actions: List[int] = []
        baseline_q_absmean: List[float] = []
        baseline_margins: List[float] = []

        # --- gradient tests ---
        grad_pf_norms = []
        grad_me_norms = []

        for i, s_np in enumerate(states):
            st = _to_tensors(s_np, device)
            q = _forward_q(net, st)
            a = int(q.argmax(dim=-1).item())

            baseline_actions.append(a)
            baseline_q_absmean.append(float(q.detach().abs().mean().item()))
            if q.numel() >= 2:
                top2 = torch.topk(q.detach(), k=2, dim=-1).values
                baseline_margins.append(float((top2[0] - top2[1]).abs().item()))

            # Price permute
            if "price_per_hour" in st:
                pf_perm = _permute_price_per_hour(
                    st["price_per_hour"].float(), seed=int(args.seed) + i
                )
                st_perm = dict(st)
                st_perm["price_per_hour"] = pf_perm
                q_perm = _forward_q(net, st_perm)
                a_perm = int(q_perm.argmax(dim=-1).item())
                action_same_pf_perm += int(a_perm == a)
                qdiff_pf_perm.append(float((q_perm - q).abs().mean().item()))

                # Price zero
                st_zero = dict(st)
                st_zero["price_per_hour"] = _zero_like(st["price_per_hour"]).float()
                q_zero = _forward_q(net, st_zero)
                a_zero = int(q_zero.argmax(dim=-1).item())
                action_same_pf_zero += int(a_zero == a)
                qdiff_pf_zero.append(float((q_zero - q).abs().mean().item()))

            # machine_exposure zero (only meaningful in full mode)
            if "machine_exposure" in st:
                st_me0 = dict(st)
                st_me0["machine_exposure"] = _zero_like(st["machine_exposure"]).float()
                q_me0 = _forward_q(net, st_me0)
                a_me0 = int(q_me0.argmax(dim=-1).item())
                action_same_me_zero += int(a_me0 == a)
                qdiff_me_zero.append(float((q_me0 - q).abs().mean().item()))

            # --- gradient sensitivity (single-state) ---
            # Do a tiny grad check on a subset (keep runtime reasonable).
            if i < 32 and "price_per_hour" in st:
                net.zero_grad(set_to_none=True)

                # Create requires_grad copies
                pf = st["price_per_hour"].float().detach().clone().requires_grad_(True)
                me = None
                if "machine_exposure" in st:
                    me = (
                        st["machine_exposure"]
                        .float()
                        .detach()
                        .clone()
                        .requires_grad_(True)
                    )

                qg = net(
                    job_features=st["job_features"].float(),
                    machine_features=st["machine_features"].float(),
                    state_features=st["state_features"].float(),
                    job_to_machine=st["job_to_machine"].long(),
                    static_edge_index=st["static_edge_index"].long(),
                    dynamic_edge_index=st["dynamic_edge_index"].long(),
                    price_features=pf,
                    machine_exposure=me,
                    period_features=(
                        st.get("period_features", None).float()
                        if "period_features" in st
                        else None
                    ),
                    tripartite_edge_index=(
                        st.get("tripartite_edge_index", None).long()
                        if "tripartite_edge_index" in st
                        else None
                    ),
                )

                # backprop from chosen action value
                a_star = int(qg.argmax(dim=-1).item())
                loss = qg[a_star]
                loss.backward()

                grad_pf_norms.append(float(pf.grad.detach().abs().mean().item()))
                if me is not None and me.grad is not None:
                    grad_me_norms.append(float(me.grad.detach().abs().mean().item()))

        n = len(states)

        # --- counterfactual phase shift on initial state ---
        # measure on initial state only per instance
        shift = int(args.price_shift)
        action_same_shift = 0
        qdiff_shift = []

        for _idx, (inst, K) in enumerate(instances):
            env.config.deterministic = True

            # baseline
            env.reset(inst, int(K))
            s0 = env.get_state_features()
            st0 = _to_tensors(s0, device)
            q0 = _forward_q(net, st0)
            a0 = int(q0.argmax(dim=-1).item())

            # shifted ct (env ignores Tk/ck for now, so keep them as-is)
            ct_shifted = _shift_ct(list(inst.ct), shift)
            inst_shift = replace(inst, ct=ct_shifted)

            env.reset(inst_shift, int(K))
            s1 = env.get_state_features()
            st1 = _to_tensors(s1, device)
            q1 = _forward_q(net, st1)
            a1 = int(q1.argmax(dim=-1).item())

            action_same_shift += int(a1 == a0)
            qdiff_shift.append(float((q1 - q0).abs().mean().item()))

        def _mean(xs: List[float]) -> float:
            return float(np.mean(xs)) if xs else 0.0

        print(f"checkpoint\t{ckpt_path}")
        print(f"price_mode\t{cfg.get('price_mode')}")
        print(f"graph_type\t{cfg.get('graph_type')}")
        print(f"states\t{n}")
        if baseline_actions:
            uniq = len(set(baseline_actions))
            mode_action = max(set(baseline_actions), key=baseline_actions.count)
            mode_freq = baseline_actions.count(mode_action) / max(
                1, len(baseline_actions)
            )
            print(f"unique_actions\t{uniq}")
            print(f"mode_action\t{mode_action}")
            print(f"mode_action_freq\t{mode_freq:.3f}")
        if baseline_q_absmean:
            print(f"q_absmean_baseline\t{_mean(baseline_q_absmean):.6g}")
        if baseline_margins:
            print(f"q_margin_top1_top2\t{_mean(baseline_margins):.6g}")
        print()
        print(_metrics_header())

        if qdiff_pf_perm:
            print(f"q_absmean_diff_price_perm\t{_mean(qdiff_pf_perm):.6g}")
            print(f"action_agree_price_perm\t{action_same_pf_perm/max(1,n):.3f}")
        if qdiff_pf_zero:
            print(f"q_absmean_diff_price_zero\t{_mean(qdiff_pf_zero):.6g}")
            print(f"action_agree_price_zero\t{action_same_pf_zero/max(1,n):.3f}")
        if qdiff_me_zero:
            print(f"q_absmean_diff_me_zero\t{_mean(qdiff_me_zero):.6g}")
            print(f"action_agree_me_zero\t{action_same_me_zero/max(1,n):.3f}")

        print(f"q_absmean_diff_ct_shift({shift})\t{_mean(qdiff_shift):.6g}")
        print(
            f"action_agree_ct_shift({shift})\t{action_same_shift/max(1,len(instances)):.3f}"
        )

        if grad_pf_norms:
            print(f"grad_absmean_price_per_hour\t{_mean(grad_pf_norms):.6g}")
        if grad_me_norms:
            print(f"grad_absmean_machine_exposure\t{_mean(grad_me_norms):.6g}")

        print("\nInterpretation:")
        print(
            "- If action_agree_* stays ~1.0 and q diffs are ~0, the policy is price-blind."
        )
        print(
            "- If permuting/zeroing price changes actions/Q materially, the policy uses price."
        )
        print(
            "- If grad_absmean_* is ~0 systematically, gradients are not flowing from Q to price inputs."
        )

    except KeyboardInterrupt:
        raise SystemExit(f"Interrupted (KeyboardInterrupt) during stage: {stage}")


if __name__ == "__main__":
    main()

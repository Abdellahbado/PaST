"""
Unified Comparison Evaluation Script for PaST Variants.

Evaluates 4 specific variants on the same instances:
1. PPO Short Slack (Base)
2. Q Sequence (PPO_Seq) - greedy/SGBS over sequences + DP scoring
3. PPO Family BestStart (Ctx13)
4. PPO Duration Aware (Ctx13)

Compares Greedy, SGBS, and DP Baselines.
"""

import argparse
import os
import sys
import time
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PaST.config import VariantID, get_variant_config, DataConfig
from PaST.past_sm_model import build_model
from PaST.sgbs import greedy_decode, sgbs
from PaST.baselines_sequence_dp import (
    spt_lpt_with_dp,
    _dp_schedule_fixed_order,
    _slice_single_instance,
)
from PaST.sm_benchmark_data import (
    generate_raw_instance,
    simulate_metaheuristic_assignment,
    make_single_machine_episode,
    SingleMachineEpisode,
)
from PaST.run_eval_eas_ppo_short_base import (
    batch_from_episodes,
    _load_checkpoint,
    _extract_model_state,
)
from PaST.q_sequence_model import build_q_model
from PaST.run_eval_q_sequence import greedy_decode_q_sequence, sgbs_q_sequence

# =============================================================================
# Configuration
# =============================================================================

MODELS_TO_EVAL = [
    {
        "name": "PPO_Short",
        "path": "runs_p100/ppo_short_base/checkpoints/best.pt",
        "variant_id": VariantID.PPO_SHORT_BASE,
        "is_sequence": False,
    },
    {
        "name": "Q_Seq",
        "path": "runs_p100/ppo_q_seq/checkpoints/best.pt",
        "variant_id": VariantID.Q_SEQUENCE,  # Identified as Q-Learning/Structure via inspection
        "is_sequence": True,
    },
    {
        "name": "PPO_Family",
        "path": "runs_p100/ppo_family_best/best.pt",
        "variant_id": VariantID.PPO_FAMILY_Q4_CTX13_BESTSTART,
        "is_sequence": False,
    },
    {
        "name": "PPO_DurAware",
        "path": "runs_p100/ppo_duration_aware/checkpoints/best.pt",
        "variant_id": VariantID.PPO_DURATION_AWARE_FAMILY_CTX13,
        "is_sequence": False,
    },
]

# =============================================================================
# Helper: Checkpoint Loading Fix
# =============================================================================


def safe_extract_state_dict(ckpt):
    """Robust extraction of state dict handling nested keys or raw state dicts."""
    if isinstance(ckpt, dict):
        if (
            "runner" in ckpt
            and isinstance(ckpt["runner"], dict)
            and "model" in ckpt["runner"]
        ):
            return ckpt["runner"]["model"]
        if "model" in ckpt:
            return ckpt["model"]

        # Heuristic: if it looks like a flat state dict
        keys = list(ckpt.keys())
        if any(k.startswith("encoder.") for k in keys) or any(
            k.startswith("q_head.") for k in keys
        ):
            return ckpt

    return _extract_model_state(ckpt)  # Fallback to original (which raises error)


# =============================================================================
# Main Evaluation
# =============================================================================


def resolve_path(path_str):
    """Try to resolve path relative to CWD or CWD/PaST"""
    p = Path(path_str)
    if p.exists():
        return p

    # Try adding PaST prefix
    p_past = Path("PaST") / path_str
    if p_past.exists():
        return p_past

    return p


def run_comparison(args):
    print("\n" + "=" * 70)
    print("PAST MODEL COMPARISON EVALUATION")
    print(f"Scale: {args.scale}, Instances: {args.num_instances}")
    print(f"SGBS Params: Beta={args.beta}, Gamma={args.gamma}")
    print("=" * 70)

    device = torch.device(args.device)

    # 1. Data Config
    dummy_cfg = get_variant_config(VariantID.PPO_SHORT_BASE)
    if args.scale == "small":
        T_choices = [t for t in dummy_cfg.data.T_max_choices if int(t) <= 100]
    elif args.scale == "medium":
        T_choices = [t for t in dummy_cfg.data.T_max_choices if 100 < int(t) <= 350]
    else:
        T_choices = dummy_cfg.data.T_max_choices
    
    dummy_cfg.data.T_max_choices = T_choices

    # 2. Generate Instances
    print("\nGenerating instances...")
    py_rng = random.Random(args.eval_seed)
    instances_data = []

    for i in range(args.num_instances):
        raw = generate_raw_instance(dummy_cfg.data, py_rng)
        assignments = simulate_metaheuristic_assignment(raw.n, raw.m, py_rng)
        non_empty = [idx for idx, a in enumerate(assignments) if len(a) > 0]
        m_idx = py_rng.choice(non_empty) if non_empty else 0
        instances_data.append(
            {"raw": raw, "m_idx": m_idx, "job_idxs": assignments[m_idx]}
        )

    # 3. Pre-generate episodes per ratio (reused across models)
    ratios = np.linspace(0.0, 1.0, 5)
    episodes_by_ratio = []
    for r_idx, ratio in enumerate(ratios):
        current_episodes = []
        for inst_idx, inst in enumerate(instances_data):
            inst_seed = args.eval_seed + inst_idx + (r_idx * 1000)
            rng_ep = random.Random(inst_seed)
            ep = make_single_machine_episode(
                inst["raw"], inst["m_idx"], inst["job_idxs"], rng_ep,
                deadline_slack_ratio_min=ratio,
                deadline_slack_ratio_max=ratio,
            )
            current_episodes.append(ep)
        episodes_by_ratio.append(current_episodes)

    results = []

    # 4. Evaluate Models
    for model_info in MODELS_TO_EVAL:
        m_name = model_info["name"]
        m_path_str = model_info["path"]
        m_variant_id = model_info["variant_id"]
        is_seq = model_info["is_sequence"]

        resolved_path = resolve_path(m_path_str)
        if not resolved_path.exists():
            print(
                f"Warning: Checkpoint for {m_name} not found at {resolved_path}. Skipping."
            )
            continue

        print(f"\nEvaluate {m_name} ({m_variant_id.value})...")

        try:
            ckpt = _load_checkpoint(resolved_path, device)
            state_dict = safe_extract_state_dict(ckpt)
            
            var_cfg = get_variant_config(m_variant_id)
            if is_seq:
                model = build_q_model(var_cfg)
            else:
                model = build_model(var_cfg)
            
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            for r_idx, ratio in enumerate(ratios):
                current_episodes = episodes_by_ratio[r_idx]
                batch = batch_from_episodes(
                    current_episodes,
                    N_job_pad=int(var_cfg.env.N_job_pad),
                )

                # --- GREEDY ---
                t0 = time.time()
                if is_seq:
                    greedy_res = greedy_decode_q_sequence(model, var_cfg, batch, device)
                else:
                    greedy_res = greedy_decode(model, var_cfg.env, device, batch)
                t_greedy = time.time() - t0

                energies_greedy = [r.total_energy for r in greedy_res]

                for i, en in enumerate(energies_greedy):
                    results.append(
                        {
                            "model": m_name,
                            "instance_idx": i,
                            "ratio": ratio,
                            "method": "Greedy" if not is_seq else "Greedy+DP",
                            "energy": en,
                            "time": t_greedy / len(greedy_res),
                        }
                    )

                # --- SGBS ---
                # Monkeypatch for duration aware if needed
                if m_variant_id == VariantID.PPO_DURATION_AWARE_FAMILY_CTX13:
                    import PaST.sgbs as sgbs_mod

                    # Ensure SGBS relies on environment-provided action masks.
                    sgbs_mod._completion_feasible_action_mask = lambda env, obs: (
                        obs["action_mask"].float()
                        if isinstance(obs, dict) and "action_mask" in obs
                        else torch.ones((env.batch_size, env.action_dim), device=env.device)
                    )

                t0 = time.time()
                if is_seq:
                    sgbs_res = sgbs_q_sequence(
                        model, var_cfg, batch, device, beta=args.beta, gamma=args.gamma
                    )
                    method_label = f"SGBS+DP(b{args.beta}g{args.gamma})"
                else:
                    sgbs_res = sgbs(
                        model, var_cfg.env, device, batch, beta=args.beta, gamma=args.gamma
                    )
                    method_label = f"SGBS(b{args.beta}g{args.gamma})"
                t_sgbs = time.time() - t0

                energies_sgbs = [r.total_energy for r in sgbs_res]
                for i, en in enumerate(energies_sgbs):
                    results.append(
                        {
                            "model": m_name,
                            "instance_idx": i,
                            "ratio": ratio,
                            "method": method_label,
                            "energy": en,
                            "time": t_sgbs / len(sgbs_res),
                        }
                    )

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error evaluating {m_name}: {e}")
            continue

    # 5. Baselines
    print("\nEvaluate Baselines (SPT+DP, LPT+DP)...")
    base_env_cfg = get_variant_config(VariantID.PPO_SHORT_BASE).env

    for r_idx, ratio in enumerate(ratios):
        current_episodes = episodes_by_ratio[r_idx]
        batch = batch_from_episodes(current_episodes, N_job_pad=int(base_env_cfg.N_job_pad))

        spt_res = spt_lpt_with_dp(base_env_cfg, device, batch, which="spt")
        lpt_res = spt_lpt_with_dp(base_env_cfg, device, batch, which="lpt")

        for i in range(len(spt_res)):
            results.append(
                {
                    "model": "Baseline",
                    "instance_idx": i,
                    "ratio": ratio,
                    "method": "SPT+DP",
                    "energy": spt_res[i].total_energy,
                    "time": 0.0,
                }
            )
            results.append(
                {
                    "model": "Baseline",
                    "instance_idx": i,
                    "ratio": ratio,
                    "method": "LPT+DP",
                    "energy": lpt_res[i].total_energy,
                    "time": 0.0,
                }
            )

    # 6. Save & Summary
    df = pd.DataFrame(results)
    out_csv = f"comparison_results_{args.scale}_seed{args.eval_seed}.csv"
    if args.out_dir:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        out_csv = str(Path(args.out_dir) / out_csv)

    df.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}")

    summary = (
        df.groupby(["model", "method"])["energy"]
        .mean()
        .reset_index()
        .sort_values("energy")
    )
    print("\n" + "=" * 70)
    print("SUMMARY (Mean Energy)")
    print("=" * 70)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scale", type=str, default="small", choices=["small", "medium", "large"]
    )
    parser.add_argument("--num_instances", type=int, default=16)
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--beta", type=int, default=4)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out_dir", type=str, default=None)

    args = parser.parse_args()
    run_comparison(args)
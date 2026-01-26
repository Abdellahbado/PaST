"""Evaluation script with schedule visualization for ppo_short_base.

Mirrors run_eval_family_q4_beststart_viz.py but adapted for the base variant:
- Standard slack action space (job_id, slack_id).
- Uses slack_to_start_time for decoding.
- Gantt-style schedule visualization.
- Epsilon-constraint analysis.

Example:
    python PaST/run_eval_ppo_short_base_viz.py \\
        --checkpoint PaST/runs_p100/ppo_short_base/checkpoints/best.pt \\
        --eval_seed 1337 \\
        --num_instances 16 --num_viz 2 \\
        --scale small --beta 4 --gamma 4 --epsilon_constraint
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import torch
import yaml

# Set matplotlib backend before import
os.environ.setdefault("MPLBACKEND", "Agg")

from PaST.baselines_sequence_dp import DPResult, spt_lpt_with_dp, spt_sequence, lpt_sequence
from PaST.config import DataConfig, VariantID, get_variant_config, EnvConfig
from PaST.past_sm_model import build_model
from PaST.sgbs import greedy_decode, sgbs, DecodeResult
from PaST.sm_benchmark_data import (
    generate_episode_batch,
    generate_raw_instance,
    simulate_metaheuristic_assignment,
    make_single_machine_episode,
    SingleMachineEpisode
)
from PaST.sm_env import slack_to_start_time
import random


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
        n = episode.n_jobs
        k = min(episode.K, K_period_pad)
        t_max = min(episode.T_max, T_max_pad)
        n = min(n, N_job_pad)

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


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML at {path}")
    return data


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
    if "runner" in ckpt and isinstance(ckpt["runner"], dict):
        runner = ckpt["runner"]
        if "model" in runner and isinstance(runner["model"], dict):
            return runner["model"]
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    raise KeyError("Could not find model state in checkpoint")


def _resolve_run_dir(run_dir_arg: str) -> Path:
    raw = Path(run_dir_arg)
    if raw.exists():
        return raw
    pkg_root = Path(__file__).resolve().parent
    candidate = pkg_root / raw
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Run directory not found: {run_dir_arg}")


def _restrict_data_config(
    data: DataConfig,
    scale: Optional[str],
    T_max: Optional[int] = None,
) -> DataConfig:
    cfg = replace(data)
    
    if T_max is not None:
        cfg.T_max_choices = [int(T_max)]
        return cfg
    
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
    
    return cfg


def _slice_single_instance(batch_data: Dict[str, Any], index: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch_data.items():
        if isinstance(v, np.ndarray):
            out[k] = v[index : index + 1]
        elif torch.is_tensor(v):
            out[k] = v[index : index + 1]
    return out


def _sequence_schedule_to_bars(
    job_sequence: List[int],
    start_times: List[int],
    p_subset: np.ndarray,
) -> List[Dict[str, Any]]:
    bars = []
    for j, st in zip(job_sequence, start_times):
        job_id = int(j)
        start = int(st)
        p = int(p_subset[job_id])
        bars.append({"job_id": job_id, "start": start, "end": start + p})
    return bars


def _action_trace_to_bars_base(
    actions: List[int],
    env_config: EnvConfig,
    p_subset: np.ndarray,
    ct: np.ndarray,
    e_single: int,
    T_limit: int,
    period_starts: np.ndarray,
    Tk: np.ndarray,
    K: int,
) -> Tuple[List[Dict[str, Any]], float]:
    """Convert action trace to bars for base ppo_short variant.
    
    Action = job_id * K_slack + slack_id.
    """
    K_slack = env_config.get_num_slack_choices()
    
    bars = []
    total_energy = 0.0
    t = 0
    
    # Pre-calculate cumulative sum for O(1) interval cost
    ct_cumsum = np.zeros(len(ct) + 1, dtype=np.float64)
    ct_cumsum[1:] = np.cumsum(ct)

    def get_interval_cost(start, end):
        s = max(0, min(start, len(ct)))
        e = max(0, min(end, len(ct)))
        return ct_cumsum[e] - ct_cumsum[s]

    for a in actions:
        job_id = int(a) // K_slack
        slack_id = int(a) % K_slack
        
        p = int(p_subset[job_id])
        
        # Use env helper to get start time
        start = slack_to_start_time(
            t_now=t,
            slack_id=slack_id,
            env_config=env_config,
            period_starts=period_starts,
            Tk=Tk,
            K=K,
            T_limit=T_limit
        )
        
        end = start + p
        if end > T_limit:
            # Clip visual but keep logic
            end = T_limit
            start = max(0, end - p)
        
        cost = get_interval_cost(start, end)
        energy = float(e_single) * float(cost)
        total_energy += energy
        
        bars.append({
            "job_id": job_id,
            "start": start,
            "end": end,
            "energy": energy,
        })
        
        # Update time
        t = end
    
    return bars, total_energy


def plot_epsilon_curves(
    results: List[Dict[str, Any]],
    out_path: Path,
    methods: List[str] = ["greedy", "sgbs", "spt_dp", "lpt_dp"],
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing, skipping plot")
        return

    import pandas as pd
    df = pd.DataFrame(results)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel("Slack Ratio (0=Tightest, 1=Loosest)")
    ax1.set_ylabel("Total Energy", color="tab:blue")
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Infeasibility Rate (%)", color="tab:red")
    
    styles = {
        "greedy": {"ls": "--", "marker": "o"},
        "sgbs": {"ls": "-", "marker": "*"},
        "spt_dp": {"ls": ":", "marker": "s"},
        "lpt_dp": {"ls": ":", "marker": "^"},
    }
    
    slacks = sorted(df["slack_ratio"].unique())
    
    for m in methods:
        key_energy = f"{m}_energy"
        if key_energy not in df.columns and m == "sgbs":
            # Find first available sgbs column
            cols = [c for c in df.columns if c.startswith("sgbs_") and c.endswith("_energy")]
            if cols:
                key_energy = cols[0]
        
        if key_energy not in df.columns:
            continue
            
        energies = []
        inf_rates = []
        
        for s in slacks:
            sub = df[df["slack_ratio"] == s]
            vals = sub[key_energy].values
            
            # Simple heuristic: inf means infeasible
            n_inf = np.sum(np.isinf(vals))
            rate = (n_inf / len(vals)) * 100
            
            valid_vals = vals[np.isfinite(vals)]
            mean_e = np.mean(valid_vals) if len(valid_vals) > 0 else np.nan
            
            energies.append(mean_e)
            inf_rates.append(rate)
            
        ax1.plot(slacks, energies, label=m, **styles.get(m, {}), color="tab:blue", alpha=0.7)
        ax2.plot(slacks, inf_rates, linestyle=":", marker=styles.get(m, {})["marker"], color="tab:red", alpha=0.5)

    ax1.set_title("Epsilon Constraint Analysis: Energy & Feasibility")
    lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels() # Skip redundant legend for simplicity
    ax1.legend(lines1, labels1, loc="upper right")
    
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def visualize_schedule(
    out_path: Path,
    instance_data: Dict[str, Any],
    schedules: Dict[str, Dict[str, Any]],
    title_suffix: str = "",
):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return
    
    T_limit = int(instance_data["T_limit"])
    Tk = instance_data["Tk"]
    ck = instance_data["ck"]
    period_starts = instance_data["period_starts"]
    K = int(instance_data["K"])
    
    method_order = list(schedules.keys())
    
    fig_h = 2.0 + 0.8 * len(method_order)
    fig, (ax_top, ax_jobs) = plt.subplots(
        2, 1, figsize=(14, fig_h), sharex=True,
        gridspec_kw={"height_ratios": [1.0, max(2.0, 0.7 * len(method_order))]}
    )
    
    # Price period colors
    price_colors = {1: "#d4f1d4", 2: "#fff4b3", 3: "#ffd9b3", 4: "#ffb3b3"}
    
    # Periods
    for k in range(K):
        start = int(period_starts[k])
        dur = int(Tk[k])
        price = int(ck[k])
        if start >= T_limit:
            continue
        dur = min(dur, T_limit - start)
        rect = mpatches.Rectangle(
            (start, 0), dur, 1.0,
            facecolor=price_colors.get(price, "#cccccc"),
            edgecolor="black", linewidth=0.7
        )
        ax_top.add_patch(rect)
        ax_top.text(start + dur/2, 0.5, str(price), ha="center", va="center", fontsize=8)

    ax_top.set_xlim(0, T_limit)
    ax_top.set_yticks([])
    ax_top.set_title(f"Price Periods (Limit={T_limit})")

    # Job bars
    y_pos = 0
    y_ticks = []
    y_labels = []

    cmap = plt.get_cmap("tab20")
    
    for method in method_order:
        sched = schedules[method]
        bars = sched["bars"]
        energy = sched["energy"]
        
        y_center = y_pos + 0.5
        y_ticks.append(y_center)
        y_labels.append(f"{method}\nE={energy:.0f}")
        
        # Background track
        ax_jobs.add_patch(mpatches.Rectangle((0, y_pos), T_limit, 0.8, color="#f0f0f0"))

        for bar in bars:
            jid = bar["job_id"]
            start = bar["start"]
            end = bar["end"]
            width = end - start
            color = cmap(jid % 20)
            
            rect = mpatches.Rectangle(
                (start, y_pos), width, 0.8,
                facecolor=color, edgecolor="black", alpha=0.9
            )
            ax_jobs.add_patch(rect)
            if width > 5:
                ax_jobs.text(start + width/2, y_center, str(jid), ha="center", va="center", fontsize=7, color="white")
        
        y_pos += 1.2

    ax_jobs.set_ylim(0, y_pos)
    ax_jobs.set_yticks(y_ticks)
    ax_jobs.set_yticklabels(y_labels)
    ax_jobs.set_xlabel("Time Slots")
    ax_jobs.grid(True, axis="x", alpha=0.3)
    
    plt.suptitle(f"Schedule Visualization{title_suffix}", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_batch_schedules(
    batch: Dict[str, Any],
    greedy_res: List[DecodeResult],
    sgbs_res_map: Dict[str, List[DecodeResult]],
    spt_res: List[DPResult],
    lpt_res: List[DPResult],
    variant_config,
    args,
    out_dir: Path,
    scale_str: str,
    title_suffix: str = "",
    filename_suffix: str = "",
):
    if args.num_viz <= 0:
        return
        
    print(f"Generating {args.num_viz} schedule visualizations{title_suffix}...")
    
    for viz_idx in range(min(int(args.num_viz), int(batch["n_jobs"].shape[0]))):
        single = _slice_single_instance(batch, viz_idx)
        n_jobs = int(single["n_jobs"][0])
        T_limit = int(single["T_limit"][0])
        p_subset = single["p_subset"][0][:n_jobs].astype(np.int32)
        
        schedules = {}
        
        # Baselines
        schedules["SPT+DP"] = {
            "energy": spt_res[viz_idx].total_energy,
            "bars": _sequence_schedule_to_bars(
                spt_res[viz_idx].job_sequence,
                spt_res[viz_idx].start_times,
                p_subset
            )
        }
        schedules["LPT+DP"] = {
            "energy": lpt_res[viz_idx].total_energy,
            "bars": _sequence_schedule_to_bars(
                lpt_res[viz_idx].job_sequence,
                lpt_res[viz_idx].start_times,
                p_subset
            )
        }
        
        # Greedy
        if greedy_res[viz_idx].actions is not None:
            g_bars, g_energy = _action_trace_to_bars_base(
                greedy_res[viz_idx].actions,
                variant_config.env,
                p_subset,
                single["ct"][0],
                int(single["e_single"][0]),
                T_limit,
                single["period_starts"][0],
                single["Tk"][0],
                int(single["K"][0]),
            )
            schedules["Greedy"] = {"energy": g_energy, "bars": g_bars}
            
        # SGBS
        for label, res_list in sgbs_res_map.items():
            if res_list[viz_idx].actions is not None:
                s_bars, s_energy = _action_trace_to_bars_base(
                    res_list[viz_idx].actions,
                    variant_config.env,
                    p_subset,
                    single["ct"][0],
                    int(single["e_single"][0]),
                    T_limit,
                    single["period_starts"][0],
                    single["Tk"][0],
                    int(single["K"][0]),
                )
                schedules[f"SGBS({label})"] = {"energy": s_energy, "bars": s_bars}
        
        fname = f"schedule_viz_{scale_str}_seed{args.eval_seed}{filename_suffix}_idx{viz_idx}.png"
        visualize_schedule(
            out_dir / fname,
            {
                "n_jobs": n_jobs, "T_limit": T_limit,
                "Tk": single["Tk"][0], "ck": single["ck"][0],
                "period_starts": single["period_starts"][0],
                "K": int(single["K"][0])
            },
            schedules,
            f" | seed={args.eval_seed} idx={viz_idx}{title_suffix}"
        )


def run_epsilon_constraint_analysis(
    args,
    variant_config,
    data_cfg,
    model,
    device,
    scale_str,
):
    print("\n" + "=" * 70)
    print(f"EPSILON CONSTRAINT ANALYSIS (steps={args.epsilon_steps})")
    print("=" * 70)
    
    if isinstance(args.beta, int):
        betas = [args.beta]
    else:
        betas = [int(x) for x in str(args.beta).split(",")]
        
    if isinstance(args.gamma, int):
        gammas = [args.gamma]
    else:
        gammas = [int(x) for x in str(args.gamma).split(",")]
        
    bg_pairs = []
    if len(betas) == len(gammas) and len(betas) > 1:
        bg_pairs = list(zip(betas, gammas))
    else:
        for b in betas:
            for g in gammas:
                bg_pairs.append((b, g))
    
    print(f"SGBS configurations: {bg_pairs}")
    
    out_dir = Path(args.out_dir) if args.out_dir else (Path(args.checkpoint).parent.parent / "eval_viz")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    py_rng = random.Random(args.eval_seed)
    
    results = []
    ratios = np.linspace(0.0, 1.0, args.epsilon_steps)
    
    # Generate base instances
    base_instances = []
    for _ in range(args.num_instances):
        raw = generate_raw_instance(data_cfg, py_rng)
        assignments = simulate_metaheuristic_assignment(raw.n, raw.m, py_rng)
        non_empty = [i for i, a in enumerate(assignments) if len(a) > 0]
        m_idx = py_rng.choice(non_empty) if non_empty else 0
        base_instances.append({
            "raw": raw, "m_idx": m_idx, "job_idxs": assignments[m_idx]
        })
        
    for r_idx, ratio in enumerate(ratios):
        print(f"Evaluating slack ratio {ratio:.2f} ({r_idx+1}/{len(ratios)})...")
        
        episodes = []
        for b in base_instances:
            ep = make_single_machine_episode(
                b["raw"], b["m_idx"], b["job_idxs"], py_rng, 
                deadline_slack_ratio_min=ratio,
                deadline_slack_ratio_max=ratio,
            )
            episodes.append(ep)
            
        batch = batch_from_episodes(
            episodes,
            N_job_pad=int(variant_config.env.N_job_pad),
            K_period_pad=250,
            T_max_pad=500
        )
        
        g_res = greedy_decode(model, variant_config.env, device, batch)
        
        sgbs_res_map = {}
        for (b_val, g_val) in bg_pairs:
            s_res = sgbs(
                model=model,
                env_config=variant_config.env,
                device=device,
                batch_data=batch,
                beta=int(b_val),
                gamma=int(g_val),
            )
            sgbs_res_map[f"{b_val}-{g_val}"] = s_res
            
        spt_res = spt_lpt_with_dp(variant_config.env, device, batch, which="spt")
        lpt_res = spt_lpt_with_dp(variant_config.env, device, batch, which="lpt")
        
        # Aggregate
        for i in range(len(episodes)):
            row = {
                "instance_idx": i,
                "slack_ratio": float(ratio),
                "T_limit": int(episodes[i].T_limit),
                "greedy_energy": g_res[i].total_energy,
                "spt_dp_energy": spt_res[i].total_energy,
                "lpt_dp_energy": lpt_res[i].total_energy,
            }
            for label, res_list in sgbs_res_map.items():
                row[f"sgbs_{label}_energy"] = res_list[i].total_energy
            results.append(row)
            
        plot_batch_schedules(
            batch, g_res, sgbs_res_map, spt_res, lpt_res,
            variant_config, args, out_dir, scale_str,
            title_suffix=f" (slack={ratio:.2f})",
            filename_suffix=f"_slack{ratio:.2f}"
        )

    out_csv = out_dir / f"epsilon_results_{scale_str}_seed{args.eval_seed}.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")
    
    plot_epsilon_curves(results, out_dir / f"epsilon_plot_{scale_str}_seed{args.eval_seed}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ppo_short_base with viz")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--num_instances", type=int, default=16)
    parser.add_argument("--num_viz", type=int, default=2)
    parser.add_argument("--epsilon_constraint", action="store_true", default=True, help="Run epsilon sweep")
    parser.add_argument("--epsilon_steps", type=int, default=5)
    parser.add_argument("--scale", type=str, default="small", choices=["small", "medium", "large"])
    parser.add_argument("--beta", type=str, default="4", help="SGBS beta (list allowed)")
    parser.add_argument("--gamma", type=str, default="4", help="SGBS gamma (list allowed)")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Load config and model
    ckpt_path = Path(args.checkpoint)
    run_dir = ckpt_path.parent.parent
    config_path = run_dir / "config.yaml"
    
    if config_path.exists():
        raw_config = _load_yaml(config_path)
        # Reconstruct VariantConfig
        # We assume it matches ppo_short_base structure mostly
        # But we use the helper to get fresh defaults, then override
        # Or just manually build it if we know it's ppo_short_base
        variant_config = get_variant_config(VariantID.PPO_SHORT_BASE)
        # Apply overrides from raw_config if needed (dimensions etc)
        # Ideally we trust the checkpoint's implied config, but we need the object.
        # For evaluation, standard defaults + checkpoint weights usually work.
    else:
        print("Config not found, using default PPO_SHORT_BASE config.")
        variant_config = get_variant_config(VariantID.PPO_SHORT_BASE)
        
    device = torch.device(args.device)
    
    # Load model
    ckpt = _load_checkpoint(ckpt_path, device)
    model_state = _extract_model_state(ckpt)
    
    model = build_model(variant_config)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    
    run_epsilon_constraint_analysis(
        args, variant_config, variant_config.data, model, device, args.scale
    )

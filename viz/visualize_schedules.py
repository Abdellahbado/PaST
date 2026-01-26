"""Visualize schedules for multiple methods on one PaST-SM instance.

Produces a Gantt-style figure:
- Top lane: price periods (length Tk, price ck)
- Below: one lane per method showing scheduled jobs as colored bars

Methods visualized:
- Greedy model decode
- SGBS(beta, gamma)
- SPT + DP scheduling
- LPT + DP scheduling
- BnB (solver_improved.py): sequence via BnB, schedule via DP

This script recomputes the methods for a single generated instance so it can
extract full schedules (start/end times), rather than relying on CSV summaries.

Example:
  python -m PaST.visualize_schedules \
    --checkpoint PaST/runs_p100/ppo_short_base/checkpoints/best.pt \
    --variant_id ppo_short_base \
    --scale small --eval_seed 123 \
    --instance_idx 0 \
    --max_machine_jobs 10 \
    --beta 4 --gamma 4 \
    --bnb_time_limit 10 \
    --out_png schedule_viz.png
"""

from __future__ import annotations

import argparse
import os
import importlib.util
import json
import time
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from PaST.baselines_sequence_dp import (
    dp_schedule_for_job_sequence,
    lpt_sequence,
    spt_lpt_with_dp,
    spt_sequence,
)
from PaST.config import DataConfig, VariantID, get_variant_config
from PaST.past_sm_model import build_model
from PaST.sgbs import greedy_decode, sgbs
from PaST.sm_benchmark_data import generate_episode_batch
from PaST.sm_env import slack_to_start_time


# Ensure headless plotting by default (also affects solver_improved imports).
os.environ.setdefault("MPLBACKEND", "Agg")


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


def _restrict_data_config(
    data: DataConfig,
    scale: Optional[str],
    T_max: Optional[int],
    n_min: Optional[int],
    n_max: Optional[int],
) -> DataConfig:
    cfg = replace(data)
    if T_max is not None:
        cfg.T_max_choices = [int(T_max)]
        return cfg

    if scale is None:
        return cfg

    s = scale.lower()
    if s == "small":
        cfg.T_max_choices = [t for t in cfg.T_max_choices if int(t) <= 80]
    elif s in ("mls", "medium"):
        cfg.T_max_choices = [t for t in cfg.T_max_choices if 80 < int(t) <= 300]
    elif s in ("vls", "large"):
        cfg.T_max_choices = [t for t in cfg.T_max_choices if int(t) > 300]
    else:
        raise ValueError("scale must be one of: small, mls/medium, vls/large")

    if not cfg.T_max_choices:
        raise ValueError(f"No T_max_choices match scale={scale}.")

    if n_min is not None:
        cfg.n_min = int(n_min)
    if n_max is not None:
        cfg.n_max = int(n_max)
    if cfg.n_min > cfg.n_max:
        raise ValueError("n_min must be <= n_max")
    return cfg


def _slice_batch(
    batch: Dict[str, np.ndarray], idxs: List[int]
) -> Dict[str, np.ndarray]:
    return {k: v[idxs] for k, v in batch.items()}


def _concat_batches(batches: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = list(batches[0].keys())
    return {k: np.concatenate([b[k] for b in batches], axis=0) for k in keys}


def _generate_filtered_batch(
    *,
    need: int,
    config: DataConfig,
    seed: int,
    N_job_pad: int,
    max_machine_jobs: Optional[int],
    max_attempts: int = 50,
    oversample_factor: int = 16,
) -> Dict[str, np.ndarray]:
    if max_machine_jobs is None:
        return generate_episode_batch(
            batch_size=int(need),
            config=config,
            seed=int(seed),
            N_job_pad=int(N_job_pad),
            K_period_pad=250,
            T_max_pad=500,
        )

    selected: List[Dict[str, np.ndarray]] = []
    selected_count = 0
    attempt = 0

    while selected_count < need and attempt < max_attempts:
        attempt += 1
        batch_seed = int(seed) + attempt * 10_000
        oversample = max(64, int(need) * int(oversample_factor))
        b = generate_episode_batch(
            batch_size=int(oversample),
            config=config,
            seed=int(batch_seed),
            N_job_pad=int(N_job_pad),
            K_period_pad=250,
            T_max_pad=500,
        )
        ok = np.where(b["n_jobs"] <= int(max_machine_jobs))[0].tolist()
        if ok:
            take = ok[: max(0, need - selected_count)]
            selected.append(_slice_batch(b, take))
            selected_count += len(take)

    if selected_count < need:
        raise RuntimeError(
            f"Could not sample {need} instances with n_jobs <= {max_machine_jobs}."
        )

    out = _concat_batches(selected)
    return _slice_batch(out, list(range(int(need))))


def _load_solver_improved() -> Tuple[type, type]:
    workspace_root = Path(__file__).resolve().parents[1]
    solver_path = (
        workspace_root
        / "Transformer Implementation"
        / "Data Generation"
        / "solver_improved.py"
    )
    if not solver_path.exists():
        raise FileNotFoundError(f"Could not find solver at: {solver_path}")

    spec = importlib.util.spec_from_file_location("solver_improved", solver_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {solver_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    return getattr(mod, "Instance"), getattr(mod, "BranchAndBoundSolver")


def _action_trace_to_job_bars(
    *,
    actions: List[int],
    env_config,
    p_subset: np.ndarray,
    ct: np.ndarray,
    e_single: int,
    T_limit: int,
    period_starts: np.ndarray,
    Tk: np.ndarray,
    K: int,
) -> Tuple[List[Dict[str, Any]], float]:
    K_slack = int(env_config.get_num_slack_choices())
    t = 0
    bars: List[Dict[str, Any]] = []
    total_energy = 0.0

    for a in actions:
        job_id = int(a) // K_slack
        slack_id = int(a) % K_slack
        p = int(p_subset[job_id])

        start = int(
            slack_to_start_time(
                int(t),
                int(slack_id),
                env_config,
                period_starts,
                Tk,
                int(K),
                int(T_limit),
            )
        )
        end = start + p
        if end > int(T_limit):
            raise ValueError(
                f"Action trace violates deadline: end={end} > T_limit={T_limit}"
            )

        energy = float(e_single) * float(np.sum(ct[start:end]))
        total_energy += energy
        bars.append(
            {
                "job_id": job_id,
                "start": start,
                "end": end,
                "slack_id": slack_id,
                "energy": energy,
            }
        )
        t = end

    return bars, float(total_energy)


def _is_complete_schedule(bars: List[Dict[str, Any]], n_jobs: int) -> bool:
    if len(bars) != int(n_jobs):
        return False
    job_ids = [int(b["job_id"]) for b in bars]
    return len(set(job_ids)) == int(n_jobs)


def _sequence_schedule_to_job_bars(
    *,
    job_sequence: List[int],
    start_times: List[int],
    p_subset: np.ndarray,
) -> List[Dict[str, Any]]:
    bars: List[Dict[str, Any]] = []
    for j, st in zip(job_sequence, start_times):
        job_id = int(j)
        start = int(st)
        p = int(p_subset[job_id])
        bars.append({"job_id": job_id, "start": start, "end": start + p})
    return bars


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize schedules for one instance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint path, e.g. PaST/runs_p100/ppo_short_base/checkpoints/best.pt",
    )
    p.add_argument("--variant_id", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    p.add_argument("--eval_seed", type=int, required=True)
    p.add_argument("--scale", type=str, default="small")
    p.add_argument("--T_max", type=int, default=None)
    p.add_argument("--n_min", type=int, default=None)
    p.add_argument("--n_max", type=int, default=None)
    p.add_argument("--instance_idx", type=int, default=0)
    p.add_argument(
        "--max_machine_jobs",
        type=int,
        default=0,
        help=(
            "Optional: filter to episodes with n_jobs <= this (helps BnB). "
            "Use 0 to disable. Note: filtering with a large --instance_idx can be slow."
        ),
    )

    p.add_argument("--beta", type=int, default=4)
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--bnb_time_limit", type=float, default=10.0)
    p.add_argument("--bnb_quiet", action="store_true")
    p.add_argument(
        "--skip_bnb",
        action="store_true",
        help="Skip Branch-and-Bound (recommended when instances have large n_jobs).",
    )

    p.add_argument("--out_png", type=str, default=None)

    # Inference-time controls to discourage late scheduling
    p.add_argument(
        "--max_wait_slots",
        type=int,
        default=None,
        help="Cap slack-induced waiting: disallow actions that start more than this many slots after the current time. (None = no cap)",
    )
    p.add_argument(
        "--wait_logit_penalty",
        type=float,
        default=0.0,
        help="Decoding-time soft bias: subtract (wait_logit_penalty * wait_slots) from action logits to prefer earlier starts.",
    )
    p.add_argument(
        "--makespan_penalty",
        type=float,
        default=0.0,
        help="SGBS pruning objective regularizer: score = -energy - makespan_penalty * completion_time. (Energy reporting unchanged)",
    )
    p.add_argument(
        "--dp_time_penalty",
        type=float,
        default=0.0,
        help="DP baseline regularizer: add dp_time_penalty * start_time to each job's DP cost to prefer earlier placements.",
    )

    p.add_argument(
        "--dump_json",
        type=str,
        default=None,
        help="Optional path to dump extracted schedules as JSON",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but unavailable; falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(str(args.device))

    variant_config = get_variant_config(VariantID(str(args.variant_id)))
    data_cfg = _restrict_data_config(
        variant_config.data,
        str(args.scale),
        args.T_max,
        args.n_min,
        args.n_max,
    )

    max_machine_jobs = (
        None if int(args.max_machine_jobs) <= 0 else int(args.max_machine_jobs)
    )

    need = int(args.instance_idx) + 1
    batch = _generate_filtered_batch(
        need=need,
        config=data_cfg,
        seed=int(args.eval_seed),
        N_job_pad=int(variant_config.env.N_job_pad),
        max_machine_jobs=max_machine_jobs,
    )

    i = int(args.instance_idx)
    single = {k: v[i : i + 1] for k, v in batch.items()}

    # Load model
    ckpt_path = Path(args.checkpoint)
    ckpt = _load_checkpoint(ckpt_path, device)
    model = build_model(variant_config).to(device)
    model.load_state_dict(_extract_model_state(ckpt))
    model.eval()

    # Run methods
    greedy_res = greedy_decode(
        model,
        variant_config.env,
        device,
        single,
        max_wait_slots=(
            int(args.max_wait_slots) if args.max_wait_slots is not None else None
        ),
        wait_logit_penalty=float(args.wait_logit_penalty),
        makespan_penalty=float(args.makespan_penalty),
    )[0]
    sgbs_res = sgbs(
        model,
        variant_config.env,
        device,
        single,
        beta=int(args.beta),
        gamma=int(args.gamma),
        max_wait_slots=(
            int(args.max_wait_slots) if args.max_wait_slots is not None else None
        ),
        wait_logit_penalty=float(args.wait_logit_penalty),
        makespan_penalty=float(args.makespan_penalty),
    )[0]
    spt_res = spt_lpt_with_dp(
        variant_config.env,
        device,
        single,
        which="spt",
        dp_time_penalty=float(args.dp_time_penalty),
    )[0]
    lpt_res = spt_lpt_with_dp(
        variant_config.env,
        device,
        single,
        which="lpt",
        dp_time_penalty=float(args.dp_time_penalty),
    )[0]

    n_jobs = int(single["n_jobs"][0])
    T_limit = int(single["T_limit"][0])
    p_subset = single["p_subset"][0][:n_jobs].astype(np.int32)
    ct = single["ct"][0][:T_limit].astype(np.int32)
    e_single = int(single["e_single"][0])

    # BnB (optional)
    bnb_seq: Optional[List[int]] = None
    bnb_cost: Optional[float] = None
    bnb_nodes: Optional[int] = None
    if not bool(args.skip_bnb) and float(args.bnb_time_limit) > 0:
        Instance, BranchAndBoundSolver = _load_solver_improved()
        energy_costs = (ct.astype(np.int64) * int(e_single)).astype(np.int32)

        inst = Instance(
            n_jobs=int(n_jobs),
            processing_times=p_subset.astype(np.int32),
            T=int(T_limit),
            energy_costs=energy_costs,
        )

        solver = BranchAndBoundSolver(inst, time_limit=float(args.bnb_time_limit))
        if args.bnb_quiet:
            with redirect_stdout(StringIO()):
                bnb_seq, bnb_cost = solver.solve()
        else:
            bnb_seq, bnb_cost = solver.solve()
        bnb_nodes = int(getattr(solver, "nodes_explored", 0))

    # Extract schedules for plotting
    Tk = single["Tk"][0]
    ck = single["ck"][0]
    period_starts = single["period_starts"][0]
    K = int(single["K"][0])

    schedules: Dict[str, Any] = {}

    if greedy_res.actions is None:
        raise RuntimeError(
            "greedy_decode did not return actions; update sgbs.py to record actions"
        )
    if sgbs_res.actions is None:
        raise RuntimeError(
            "sgbs did not return actions; update sgbs.py to record actions"
        )

    greedy_bars, greedy_energy = _action_trace_to_job_bars(
        actions=greedy_res.actions,
        env_config=variant_config.env,
        p_subset=p_subset,
        ct=ct,
        e_single=e_single,
        T_limit=T_limit,
        period_starts=period_starts,
        Tk=Tk,
        K=K,
    )

    sgbs_bars, sgbs_energy = _action_trace_to_job_bars(
        actions=sgbs_res.actions,
        env_config=variant_config.env,
        p_subset=p_subset,
        ct=ct,
        e_single=e_single,
        T_limit=T_limit,
        period_starts=period_starts,
        Tk=Tk,
        K=K,
    )

    infeasible_penalty = 1e9
    greedy_ok = _is_complete_schedule(greedy_bars, n_jobs)
    sgbs_ok = _is_complete_schedule(sgbs_bars, n_jobs)

    schedules["greedy"] = {
        "energy": greedy_energy if greedy_ok else float(infeasible_penalty),
        "bars": greedy_bars,
        "complete": bool(greedy_ok),
        "scheduled_jobs": int(len(greedy_bars)),
    }
    schedules["sgbs"] = {
        "energy": sgbs_energy if sgbs_ok else float(infeasible_penalty),
        "bars": sgbs_bars,
        "complete": bool(sgbs_ok),
        "scheduled_jobs": int(len(sgbs_bars)),
    }
    schedules["spt_dp"] = {
        "energy": float(spt_res.total_energy),
        "bars": _sequence_schedule_to_job_bars(
            job_sequence=spt_res.job_sequence,
            start_times=spt_res.start_times,
            p_subset=p_subset,
        ),
        "complete": True,
        "scheduled_jobs": int(n_jobs),
    }
    schedules["lpt_dp"] = {
        "energy": float(lpt_res.total_energy),
        "bars": _sequence_schedule_to_job_bars(
            job_sequence=lpt_res.job_sequence,
            start_times=lpt_res.start_times,
            p_subset=p_subset,
        ),
        "complete": True,
        "scheduled_jobs": int(n_jobs),
    }
    if bnb_seq is not None and bnb_cost is not None:
        bnb_dp = dp_schedule_for_job_sequence(
            single, list(bnb_seq), dp_time_penalty=float(args.dp_time_penalty)
        )
        schedules["bnb"] = {
            "energy": float(bnb_cost),
            "sequence": list(bnb_seq),
            "bars": _sequence_schedule_to_job_bars(
                job_sequence=bnb_dp.job_sequence,
                start_times=bnb_dp.start_times,
                p_subset=p_subset,
            ),
            "nodes": int(bnb_nodes or 0),
            "complete": True,
            "scheduled_jobs": int(n_jobs),
        }

    # Plot
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
        import matplotlib.patches as mpatches  # type: ignore[import-not-found]
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "matplotlib is required for visualization. Install it with: pip install matplotlib"
        ) from e

    method_order = ["greedy", "sgbs", "spt_dp", "lpt_dp"]
    if "bnb" in schedules:
        method_order.append("bnb")
    method_labels = {
        "greedy": (
            f"Greedy ({schedules['greedy']['scheduled_jobs']}/{n_jobs} jobs) "
            f"(E={schedules['greedy']['energy']:.2f})"
            + (" [INFEASIBLE]" if not schedules["greedy"]["complete"] else "")
        ),
        "sgbs": (
            f"SGBS b={int(args.beta)} g={int(args.gamma)} "
            f"({schedules['sgbs']['scheduled_jobs']}/{n_jobs} jobs) "
            f"(E={schedules['sgbs']['energy']:.2f})"
            + (" [INFEASIBLE]" if not schedules["sgbs"]["complete"] else "")
        ),
        "spt_dp": f"SPT+DP (E={schedules['spt_dp']['energy']:.2f})",
        "lpt_dp": f"LPT+DP (E={schedules['lpt_dp']['energy']:.2f})",
        "bnb": (
            f"BnB+DP (E={schedules['bnb']['energy']:.2f})"
            if "bnb" in schedules
            else "BnB+DP"
        ),
    }

    fig_h = 2.2 + 0.7 * len(method_order)
    fig, (ax_top, ax_jobs) = plt.subplots(
        2,
        1,
        figsize=(14, fig_h),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, max(2.0, 0.8 * len(method_order))]},
    )

    # Period lane
    # Use per-slot prices derived from ck/Tk; show each period as a block.
    price_colors = {1: "#d4f1d4", 2: "#fff4b3", 3: "#ffd9b3", 4: "#ffb3b3"}
    for k in range(int(K)):
        start = int(period_starts[k])
        dur = int(Tk[k])
        price = int(ck[k])
        if start >= int(T_limit):
            continue
        dur = min(dur, int(T_limit) - start)
        rect = mpatches.Rectangle(
            (start, 0),
            dur,
            1.0,
            facecolor=price_colors.get(price, "#cccccc"),
            edgecolor="black",
            linewidth=0.7,
        )
        ax_top.add_patch(rect)
        ax_top.text(
            start + dur / 2,
            0.5,
            f"p={price}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax_top.set_ylim(0, 1)
    ax_top.set_yticks([])
    ax_top.set_title(
        f"PaST-SM Schedule Visualization | n_jobs={n_jobs} T_limit={T_limit} | seed={int(args.eval_seed)} idx={i}",
        fontsize=12,
        fontweight="bold",
    )

    legend_elements = [
        mpatches.Patch(facecolor=price_colors[c], edgecolor="black", label=f"price={c}")
        for c in sorted(price_colors.keys())
    ]
    ax_top.legend(handles=legend_elements, loc="upper right", fontsize=9)

    # Job lanes
    job_colors = plt.cm.Set3(np.linspace(0, 1, max(1, n_jobs)))

    y_positions = list(range(len(method_order)))
    ax_jobs.set_yticks(y_positions)
    ax_jobs.set_yticklabels([method_labels[m] for m in method_order], fontsize=9)

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
                linewidth=1.0,
            )
            ax_jobs.text(
                start + dur / 2,
                row,
                f"J{job_id}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

    ax_jobs.set_xlim(0, int(T_limit))
    ax_jobs.set_xlabel("Time")
    ax_jobs.grid(axis="x", alpha=0.25, linestyle="--")

    plt.tight_layout()

    out_png = (
        Path(args.out_png)
        if args.out_png
        else Path.cwd() / f"schedule_viz_seed{int(args.eval_seed)}_idx{i}.png"
    )
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {out_png}")

    if args.dump_json:
        dump_path = Path(args.dump_json)
        dump = {
            "n_jobs": n_jobs,
            "T_limit": T_limit,
            "energies": {
                k: float(v["energy"]) for k, v in schedules.items() if "energy" in v
            },
            "schedules": schedules,
        }
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dump_path, "w") as f:
            json.dump(dump, f, indent=2)
        print(f"Wrote: {dump_path}")


if __name__ == "__main__":
    main()

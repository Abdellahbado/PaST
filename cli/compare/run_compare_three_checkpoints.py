"""Compare 3 checkpoints (2 family architectures + 1 Q-seq) on the same instances.

This is a lightweight, focused variant of `PaST/run_eval_comparison_correct.py`.

It evaluates:
- Family model A (e.g. best_cwe)
- Family model B (e.g. best)
- Q-sequence model (ppo_q_seq)

Methods:
- Greedy (policy decode)
- SGBS (optional)
- Baselines: SPT+DP, LPT+DP

Example:
    python -m PaST.cli.compare.run_compare_three_checkpoints \
    --scale small --num_instances 64 --eval_seed 42 \
    --beta 4 --gamma 4 --device cuda \
    --out_dir analysis_out

You can override checkpoint paths:
    python -m PaST.cli.compare.run_compare_three_checkpoints \
    --family_a PaST/runs_p100/ppo_family_best_cwe/checkpoints/best.pt \
    --family_b PaST/runs_p100/ppo_family_best/best.pt \
    --q_seq   PaST/runs_p100/ppo_q_seq/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from PaST.config import VariantID, get_variant_config
from PaST.past_sm_model import build_model
from PaST.sgbs import greedy_decode, sgbs
from PaST.baselines_sequence_dp import spt_lpt_with_dp
from PaST.sm_benchmark_data import (
    generate_raw_instance,
    simulate_metaheuristic_assignment,
    make_single_machine_episode,
)
from PaST.cli.eval.run_eval_eas_ppo_short_base import (
    batch_from_episodes,
    _load_checkpoint,
    _extract_model_state,
)
from PaST.q_sequence_model import build_q_model
from PaST.cli.eval.run_eval_q_sequence import greedy_decode_q_sequence, sgbs_q_sequence


def _resolve_path(path_str: str) -> Path:
    """Resolve a checkpoint path.

    Accepts:
    - absolute paths
    - workspace-relative paths
    - paths prefixed with "PaST/"
    - paths relative to the package dir
    """
    p = Path(path_str)
    if p.exists():
        return p

    # Try relative to CWD
    p2 = Path.cwd() / path_str
    if p2.exists():
        return p2

    # Try adding PaST prefix if missing
    p3 = Path.cwd() / "PaST" / path_str
    if p3.exists():
        return p3

    # If user provided PaST/..., also try stripping it.
    parts = list(p.parts)
    if parts and parts[0] == "PaST":
        p4 = Path.cwd() / Path(*parts[1:])
        if p4.exists():
            return p4

    return p


def _safe_extract_state_dict(ckpt: Any) -> Dict[str, Any]:
    """Robust extraction of model state dict from various checkpoint formats."""
    if isinstance(ckpt, dict):
        if (
            "runner" in ckpt
            and isinstance(ckpt["runner"], dict)
            and "model" in ckpt["runner"]
        ):
            return ckpt["runner"]["model"]
        if "model" in ckpt:
            return ckpt["model"]

        # Heuristic: flat-ish state dict
        keys = list(ckpt.keys())
        if any(isinstance(k, str) and k.startswith("encoder.") for k in keys):
            return ckpt
        if any(isinstance(k, str) and k.startswith("q_head.") for k in keys):
            return ckpt

    # Fallback to canonical extractor used by PPO scripts
    return _extract_model_state(ckpt)


def _ratios_from_arg(arg: str) -> List[float]:
    items = [x.strip() for x in arg.split(",") if x.strip()]
    out: List[float] = []
    for x in items:
        v = float(x)
        if v < 0.0 or v > 1.0:
            raise ValueError(f"ratio must be in [0,1], got {v}")
        out.append(v)
    if not out:
        raise ValueError("ratios list is empty")
    return out


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to write")

    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def run(args: argparse.Namespace) -> None:
    print("\n" + "=" * 70)
    print("COMPARE 3 CHECKPOINTS")
    print(
        f"Scale={args.scale} | Instances={args.num_instances} | Seed={args.eval_seed}"
    )
    print(
        f"SGBS={'on' if args.sgbs else 'off'}"
        + (f" | beta={args.beta} gamma={args.gamma}" if args.sgbs else "")
    )
    print("=" * 70)

    device = torch.device(args.device)

    # Use a single data config to generate instances; consistent with existing eval scripts.
    dummy_cfg = get_variant_config(VariantID.PPO_SHORT_BASE)
    if args.scale == "small":
        T_choices = [t for t in dummy_cfg.data.T_max_choices if int(t) <= 100]
    elif args.scale == "medium":
        T_choices = [t for t in dummy_cfg.data.T_max_choices if 100 < int(t) <= 350]
    else:
        T_choices = dummy_cfg.data.T_max_choices
    dummy_cfg.data.T_max_choices = T_choices

    # Generate raw instances (then single-machine episodes) once.
    py_rng = random.Random(int(args.eval_seed))
    instances_data = []
    for i in range(int(args.num_instances)):
        raw = generate_raw_instance(dummy_cfg.data, py_rng)
        assignments = simulate_metaheuristic_assignment(raw.n, raw.m, py_rng)
        non_empty = [idx for idx, a in enumerate(assignments) if len(a) > 0]
        m_idx = py_rng.choice(non_empty) if non_empty else 0
        instances_data.append(
            {"raw": raw, "m_idx": m_idx, "job_idxs": assignments[m_idx]}
        )

    ratios = _ratios_from_arg(args.ratios)
    episodes_by_ratio = []
    for r_idx, ratio in enumerate(ratios):
        eps = []
        for inst_idx, inst in enumerate(instances_data):
            inst_seed = int(args.eval_seed) + int(inst_idx) + (int(r_idx) * 1000)
            rng_ep = random.Random(inst_seed)
            ep = make_single_machine_episode(
                inst["raw"],
                inst["m_idx"],
                inst["job_idxs"],
                rng_ep,
                deadline_slack_ratio_min=float(ratio),
                deadline_slack_ratio_max=float(ratio),
            )
            eps.append(ep)
        episodes_by_ratio.append(eps)

    models = [
        {
            "name": args.family_a_name,
            "path": args.family_a,
            # CWE checkpoint uses the Candidate-Window sparse encoder backbone.
            "variant_id": VariantID.PPO_FAMILY_Q4_CTX13_BESTSTART_CWE,
            "is_sequence": False,
        },
        {
            "name": args.family_b_name,
            "path": args.family_b,
            "variant_id": VariantID.PPO_FAMILY_Q4_CTX13_BESTSTART,
            "is_sequence": False,
        },
        {
            "name": args.q_seq_name,
            "path": args.q_seq,
            "variant_id": VariantID.Q_SEQUENCE,
            "is_sequence": True,
        },
    ]

    rows: List[Dict[str, Any]] = []

    # Evaluate each checkpoint.
    for spec in models:
        ckpt_path = _resolve_path(spec["path"])
        if not ckpt_path.exists():
            print(f"[skip] {spec['name']}: checkpoint not found at {ckpt_path}")
            continue

        var_cfg = get_variant_config(spec["variant_id"])
        print(f"\nEvaluate {spec['name']} ({spec['variant_id'].value}) @ {ckpt_path}")

        ckpt = _load_checkpoint(ckpt_path, device)
        state_dict = _safe_extract_state_dict(ckpt)

        if spec["is_sequence"]:
            model = build_q_model(var_cfg)
        else:
            model = build_model(var_cfg)

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        for ratio, eps in zip(ratios, episodes_by_ratio):
            batch = batch_from_episodes(eps, N_job_pad=int(var_cfg.env.N_job_pad))

            # Greedy
            t0 = time.time()
            if spec["is_sequence"]:
                greedy_res = greedy_decode_q_sequence(model, var_cfg, batch, device)
                method = "Greedy+DP"
            else:
                greedy_res = greedy_decode(model, var_cfg.env, device, batch)
                method = "Greedy"
            t_greedy = time.time() - t0

            energies = [float(r.total_energy) for r in greedy_res]
            for i, en in enumerate(energies):
                rows.append(
                    {
                        "model": spec["name"],
                        "instance_idx": int(i),
                        "ratio": float(ratio),
                        "method": method,
                        "energy": float(en),
                        "time": float(t_greedy / max(1, len(energies))),
                    }
                )

            # SGBS
            if args.sgbs:
                t0 = time.time()
                if spec["is_sequence"]:
                    sgbs_res = sgbs_q_sequence(
                        model,
                        var_cfg,
                        batch,
                        device,
                        beta=int(args.beta),
                        gamma=int(args.gamma),
                    )
                    method = f"SGBS+DP(b{int(args.beta)}g{int(args.gamma)})"
                else:
                    sgbs_res = sgbs(
                        model,
                        var_cfg.env,
                        device,
                        batch,
                        beta=int(args.beta),
                        gamma=int(args.gamma),
                    )
                    method = f"SGBS(b{int(args.beta)}g{int(args.gamma)})"
                t_sgbs = time.time() - t0

                energies = [float(r.total_energy) for r in sgbs_res]
                for i, en in enumerate(energies):
                    rows.append(
                        {
                            "model": spec["name"],
                            "instance_idx": int(i),
                            "ratio": float(ratio),
                            "method": method,
                            "energy": float(en),
                            "time": float(t_sgbs / max(1, len(energies))),
                        }
                    )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Baselines (SPT+DP, LPT+DP)
    print("\nEvaluate baselines (SPT+DP, LPT+DP)...")
    base_env_cfg = get_variant_config(VariantID.PPO_SHORT_BASE).env

    for ratio, eps in zip(ratios, episodes_by_ratio):
        batch = batch_from_episodes(eps, N_job_pad=int(base_env_cfg.N_job_pad))
        spt_res = spt_lpt_with_dp(base_env_cfg, device, batch, which="spt")
        lpt_res = spt_lpt_with_dp(base_env_cfg, device, batch, which="lpt")

        for i in range(len(spt_res)):
            rows.append(
                {
                    "model": "Baseline",
                    "instance_idx": int(i),
                    "ratio": float(ratio),
                    "method": "SPT+DP",
                    "energy": float(spt_res[i].total_energy),
                    "time": 0.0,
                }
            )
            rows.append(
                {
                    "model": "Baseline",
                    "instance_idx": int(i),
                    "ratio": float(ratio),
                    "method": "LPT+DP",
                    "energy": float(lpt_res[i].total_energy),
                    "time": 0.0,
                }
            )

    # Save
    out_dir = Path(args.out_dir) if args.out_dir else Path("analysis_out")
    tag = f"b{int(args.beta)}g{int(args.gamma)}" if args.sgbs else "nosgbs"
    out_path = out_dir / f"compare_three_{args.scale}_seed{args.eval_seed}_{tag}.csv"
    _write_csv(out_path, rows)

    print(f"\nSaved: {out_path}")

    # Summary (mean energies)
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
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

        # Extra: directly compare the two family models on the same method names.
        fam_a = args.family_a_name
        fam_b = args.family_b_name
        for method in ["Greedy", f"SGBS(b{int(args.beta)}g{int(args.gamma)})"]:
            if not args.sgbs and method.startswith("SGBS"):
                continue
            a = df[(df["model"] == fam_a) & (df["method"] == method)]["energy"].mean()
            b = df[(df["model"] == fam_b) & (df["method"] == method)]["energy"].mean()
            if np.isfinite(a) and np.isfinite(b):
                diff = float(a - b)
                rel = float(diff / b * 100.0) if b != 0 else float("nan")
                print(f"\nΔ({fam_a} - {fam_b}) on {method}: {diff:.4f} ({rel:+.2f}%)")

    except Exception as e:
        print(f"(pandas summary skipped: {e})")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--family_a",
        type=str,
        default="PaST/runs_p100/ppo_family_best_cwe/checkpoints/best.pt",
        help="Checkpoint path for family model A",
    )
    p.add_argument(
        "--family_b",
        type=str,
        default="PaST/runs_p100/ppo_family_best/best.pt",
        help="Checkpoint path for family model B",
    )
    p.add_argument(
        "--q_seq",
        type=str,
        default="PaST/runs_p100/ppo_q_seq/checkpoints/best.pt",
        help="Checkpoint path for Q-seq model",
    )

    p.add_argument("--family_a_name", type=str, default="PPO_Family_CWE")
    p.add_argument("--family_b_name", type=str, default="PPO_Family")
    p.add_argument("--q_seq_name", type=str, default="Q_Seq")

    p.add_argument(
        "--scale",
        type=str,
        default="small",
        choices=["small", "medium", "large"],
    )
    p.add_argument("--num_instances", type=int, default=32)
    p.add_argument("--eval_seed", type=int, default=42)

    p.add_argument(
        "--ratios",
        type=str,
        default="0.0,0.25,0.5,0.75,1.0",
        help="Comma-separated deadline slack ratios in [0,1]",
    )

    p.add_argument("--sgbs", dest="sgbs", action="store_true", default=True)
    p.add_argument("--no_sgbs", dest="sgbs", action="store_false")
    p.add_argument("--beta", type=int, default=4)
    p.add_argument("--gamma", type=int, default=4)

    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    p.add_argument(
        "--out_dir",
        type=str,
        default="analysis_out",
        help="Output directory for the CSV",
    )

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run(args)

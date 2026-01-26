"""Compare Q-sequence (SGBS) vs Branch-and-Bound (optimal sequence).

This script generates PaST single-machine episodes (same distribution as the other
PaST eval scripts), then compares:
- Q-sequence model decoded with SGBS (beta/gamma) + DP scheduling
- Branch-and-Bound (BnB) that searches over job *sequences*, using DP scheduling
  to compute the cost for any fixed order.

Both methods are timed per instance.

Key compatibility note
----------------------
PaST's objective for a fixed job order is:
  energy = e_single * sum_t ct[t] over processing slots

The provided BnB solver expects per-timestep "energy_costs" and uses:
  cost = sum_t energy_costs[t] over processing slots

So we convert a PaST instance into the BnB format via:
  energy_costs[t] = e_single * ct[t]   for t in [0, T_limit)

Usage
-----
python -m PaST.cli.compare.compare_qseq_sgbs_vs_bnb \
  --checkpoint PaST/runs_p100/ppo_q_seq/checkpoints/best.pt \
  --scale small --num_instances 32 --eval_seed 1 \
  --device cuda --beta 4 --gamma 4 \
  --bnb_time_limit 60 \
  --out_dir analysis_out

Tip: BnB is exponential; use --max_jobs (e.g. 10/12) if needed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from PaST.baselines_sequence_dp import dp_schedule_for_job_sequence
from PaST.config import VariantID, get_variant_config
from PaST.q_sequence_model import QSequenceNet, build_q_model
from PaST.cli.eval.run_eval_q_sequence import (
    _extract_q_model_state,
    _load_checkpoint as _load_q_ckpt,
    sgbs_q_sequence,
)
from PaST.sm_benchmark_data import (
    generate_raw_instance,
    make_single_machine_episode,
    simulate_metaheuristic_assignment,
    SingleMachineEpisode,
)


# =============================================================================
# Local batching utility
# =============================================================================


def batch_from_episodes(
    episodes: List[SingleMachineEpisode],
    N_job_pad: int = 50,
    K_period_pad: int = 250,
    T_max_pad: int = 500,
) -> Dict[str, np.ndarray]:
    """Manually batch a list of single-machine episodes.

    This is intentionally inlined here to keep the compare script self-contained
    (no dependency on other CLI eval scripts).
    """
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
        n = min(int(episode.n_jobs), int(N_job_pad))
        k = min(int(episode.K), int(K_period_pad))
        t_max = min(int(episode.T_max), int(T_max_pad))

        batch["n_jobs"][i] = n
        batch["K"][i] = k
        batch["T_max"][i] = t_max
        batch["T_limit"][i] = int(episode.T_limit)
        batch["T_min"][i] = int(episode.T_min)
        batch["e_single"][i] = int(episode.e_single)

        if n > 0:
            batch["p_subset"][i, :n] = np.asarray(episode.p_subset[:n], dtype=np.int32)
            batch["job_mask"][i, :n] = 1.0

        if k > 0:
            batch["Tk"][i, :k] = np.asarray(episode.Tk[:k], dtype=np.int32)
            batch["ck"][i, :k] = np.asarray(episode.ck[:k], dtype=np.int32)
            batch["period_starts"][i, :k] = np.asarray(
                episode.period_starts[:k], dtype=np.int32
            )
            batch["period_mask"][i, :k] = 1.0

        if t_max > 0:
            batch["ct"][i, :t_max] = np.asarray(episode.ct[:t_max], dtype=np.int32)
            ct_valid = np.asarray(
                episode.ct[: min(int(episode.T_limit), t_max)], dtype=np.int32
            )
            if ct_valid.size > 0:
                q25, q50, q75 = np.quantile(ct_valid, [0.25, 0.5, 0.75])
                batch["price_q"][i] = [q25, q50, q75]

    return batch


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p

    p2 = Path.cwd() / path_str
    if p2.exists():
        return p2

    return p


def _ratios_from_arg(arg: str) -> List[float]:
    items = [x.strip() for x in str(arg).split(",") if x.strip()]
    out: List[float] = []
    for x in items:
        v = float(x)
        if v < 0.0 or v > 1.0:
            raise ValueError(f"ratio must be in [0,1], got {v}")
        out.append(v)
    if not out:
        raise ValueError("ratios list is empty")
    return out


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to write")

    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _load_q_model(
    checkpoint: Path, variant_id: VariantID, device: torch.device
) -> QSequenceNet:
    var_cfg = get_variant_config(variant_id)
    model = build_q_model(var_cfg)
    ckpt = _load_q_ckpt(checkpoint, device)
    state = _extract_q_model_state(ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _generate_episodes_by_ratio(
    *,
    scale: str,
    num_instances: int,
    eval_seed: int,
    ratios: Sequence[float],
    max_jobs: Optional[int],
) -> List[List[Any]]:
    """Generate a fixed set of instances and derive episodes per ratio.

    If max_jobs is provided, only keep instances with n_jobs <= max_jobs.
    """

    dummy_cfg = get_variant_config(VariantID.PPO_SHORT_BASE)
    if scale == "small":
        dummy_cfg.data.T_max_choices = [
            t for t in dummy_cfg.data.T_max_choices if int(t) <= 100
        ]
    elif scale == "medium":
        dummy_cfg.data.T_max_choices = [
            t for t in dummy_cfg.data.T_max_choices if 100 < int(t) <= 350
        ]

    py_rng = random.Random(int(eval_seed))

    base_instances: List[Dict[str, Any]] = []
    tries = 0
    while len(base_instances) < int(num_instances):
        tries += 1
        raw = generate_raw_instance(dummy_cfg.data, py_rng)
        assignments = simulate_metaheuristic_assignment(raw.n, raw.m, py_rng)
        non_empty = [idx for idx, a in enumerate(assignments) if len(a) > 0]
        m_idx = py_rng.choice(non_empty) if non_empty else 0
        job_idxs = assignments[m_idx]

        # Build one episode at ratio=1.0 just to read n_jobs.
        # (We need to filter by max_jobs before creating all ratios.)
        if max_jobs is not None:
            tmp_ep = make_single_machine_episode(
                raw,
                m_idx,
                job_idxs,
                random.Random(int(eval_seed) + tries),
                deadline_slack_ratio_min=1.0,
                deadline_slack_ratio_max=1.0,
            )
            n_jobs = int(tmp_ep.n_jobs)
            if n_jobs > int(max_jobs):
                continue

        base_instances.append({"raw": raw, "m_idx": m_idx, "job_idxs": job_idxs})

        # Safety valve for pathological filters.
        if tries > int(num_instances) * 5000:
            raise RuntimeError(
                f"Could not sample enough instances with max_jobs={max_jobs}. "
                f"Collected {len(base_instances)} / {num_instances}."
            )

    episodes_by_ratio: List[List[Any]] = []
    for r_idx, ratio in enumerate(ratios):
        eps: List[Any] = []
        for inst_idx, b in enumerate(base_instances):
            inst_seed = int(eval_seed) + int(inst_idx) + (int(r_idx) * 1000)
            rng_ep = random.Random(inst_seed)
            ep = make_single_machine_episode(
                b["raw"],
                b["m_idx"],
                b["job_idxs"],
                rng_ep,
                deadline_slack_ratio_min=float(ratio),
                deadline_slack_ratio_max=float(ratio),
            )
            eps.append(ep)
        episodes_by_ratio.append(eps)

    return episodes_by_ratio


def _import_bnb_module(module_path: Path):
    """Import solver_improved.py (path may contain spaces)."""
    module_path = module_path.resolve()
    if not module_path.exists():
        raise FileNotFoundError(str(module_path))

    spec = importlib.util.spec_from_file_location("bnb_solver_module", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {module_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _solve_bnb_one(
    *,
    n_jobs: int,
    p_subset: np.ndarray,
    T: int,
    energy_costs: np.ndarray,
    time_limit: float,
    bnb_path: Optional[str],
) -> Tuple[List[int], float, Dict[str, int]]:
    """Solve a single BnB instance.

    Runs in a worker process when parallelism is enabled.
    """
    InstanceCls = None
    SolverCls = None

    if bnb_path:
        candidate = Path(bnb_path)
        if not candidate.exists():
            # Try relative to CWD in worker.
            candidate = Path.cwd() / bnb_path
        if candidate.exists():
            bnb_mod = _import_bnb_module(candidate)
            if not hasattr(bnb_mod, "BranchAndBoundSolver") or not hasattr(
                bnb_mod, "Instance"
            ):
                raise AttributeError(
                    f"BnB module {candidate} must define Instance and BranchAndBoundSolver"
                )
            InstanceCls = getattr(bnb_mod, "Instance")
            SolverCls = getattr(bnb_mod, "BranchAndBoundSolver")

    if InstanceCls is None or SolverCls is None:
        from PaST.solvers.bnb_solver_custom import Instance as _Instance
        from PaST.solvers.bnb_solver_custom import BranchAndBoundSolver as _Solver

        InstanceCls = _Instance
        SolverCls = _Solver

    inst = InstanceCls(
        n_jobs=int(n_jobs),
        processing_times=np.asarray(p_subset, dtype=np.int32),
        T=int(T),
        energy_costs=np.asarray(energy_costs, dtype=np.int64),
    )
    solver = SolverCls(inst, time_limit=float(time_limit))
    seq, cost = solver.solve()
    stats = {
        "nodes_explored": int(getattr(solver, "nodes_explored", -1)),
        "binpack_attempts": int(getattr(solver, "binpack_attempts", -1)),
        "pruned_by_binpack": int(getattr(solver, "pruned_by_binpack", -1)),
    }
    return [int(x) for x in seq], float(cost), stats


def _slice_single_instance_np(batch: Dict[str, Any], index: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            out[k] = v[index : index + 1]
        elif torch.is_tensor(v):
            out[k] = v[index : index + 1]
        else:
            raise TypeError(f"Unsupported batch type for key={k}: {type(v)}")
    return out


def _past_to_bnb_instance(single_np: Dict[str, Any]):
    """Convert a PaST single-instance dict to a BnB Instance."""
    n_jobs = int(single_np["n_jobs"][0])
    p_subset = np.asarray(single_np["p_subset"][0], dtype=np.int32)[:n_jobs]
    T_limit = int(single_np["T_limit"][0])
    ct = np.asarray(single_np["ct"][0], dtype=np.int32)[:T_limit]
    e_single = int(single_np["e_single"][0])

    energy_costs = (ct.astype(np.int64) * int(e_single)).astype(np.int64)
    return n_jobs, p_subset.astype(np.int32), T_limit, energy_costs


def _finite(x: float) -> bool:
    return bool(np.isfinite(np.asarray([x], dtype=np.float64))[0])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--checkpoint",
        type=str,
        default="PaST/runs_p100/ppo_q_seq/checkpoints/best.pt",
        help="Q_Seq checkpoint path",
    )
    p.add_argument(
        "--variant_id",
        type=str,
        default=VariantID.Q_SEQUENCE.value,
        help="VariantID string for Q_Seq (e.g. 'q_sequence', 'q_sequence_cwe')",
    )

    p.add_argument(
        "--scale", type=str, default="small", choices=["small", "medium", "large"]
    )
    p.add_argument("--num_instances", type=int, default=32)
    p.add_argument("--eval_seed", type=int, default=1)
    p.add_argument("--ratios", type=str, default="1.0")

    p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument("--beta", type=int, default=4)
    p.add_argument("--gamma", type=int, default=4)

    p.add_argument(
        "--bnb_path",
        type=str,
        default=None,
        help=(
            "Optional path to an external solver_improved.py defining Instance and BranchAndBoundSolver. "
            "If omitted or not found, uses the in-repo solver PaST.solvers.bnb_solver_custom."
        ),
    )
    p.add_argument(
        "--bnb_time_limit",
        type=float,
        default=60.0,
        help="BnB time limit in seconds. Use 'inf' for no limit.",
    )

    p.add_argument(
        "--bnb_workers",
        type=int,
        default=1,
        help=(
            "Parallel workers for BnB across instances (CPU only). "
            "Use 1 to disable multiprocessing."
        ),
    )

    p.add_argument(
        "--torch_threads",
        type=int,
        default=0,
        help=(
            "If >0, sets torch.set_num_threads() for CPU runs. "
            "Useful to control SGBS/model inference parallelism."
        ),
    )

    p.add_argument(
        "--max_jobs",
        type=int,
        default=12,
        help="Only evaluate instances with n_jobs <= max_jobs (recommended for BnB).",
    )

    p.add_argument("--out_dir", type=str, default="analysis_out")

    return p


def main() -> None:
    args = build_parser().parse_args()

    device = torch.device(args.device)
    ratios = _ratios_from_arg(args.ratios)

    if int(args.torch_threads) > 0 and device.type == "cpu":
        try:
            torch.set_num_threads(int(args.torch_threads))
            torch.set_num_interop_threads(min(4, int(args.torch_threads)))
            print(
                f"Torch CPU threads: {torch.get_num_threads()} (interop={torch.get_num_interop_threads()})"
            )
        except Exception:
            pass

    ckpt_path = _resolve_path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))

    q_variant = VariantID(args.variant_id)
    q_cfg = get_variant_config(q_variant)

    print(f"Loading Q_Seq checkpoint: {ckpt_path}")
    model = _load_q_model(ckpt_path, q_variant, device)

    # BnB solver is resolved inside the worker when multiprocessing is enabled.
    # For the single-process path we keep the same resolution logic as before.
    InstanceCls = None
    SolverCls = None

    if args.bnb_path:
        candidate = _resolve_path(str(args.bnb_path))
        if candidate.exists():
            bnb_mod = _import_bnb_module(candidate)
            if not hasattr(bnb_mod, "BranchAndBoundSolver") or not hasattr(
                bnb_mod, "Instance"
            ):
                raise AttributeError(
                    f"BnB module {candidate} must define Instance and BranchAndBoundSolver"
                )
            InstanceCls = getattr(bnb_mod, "Instance")
            SolverCls = getattr(bnb_mod, "BranchAndBoundSolver")
            print(f"Using external BnB solver: {candidate}")

    if InstanceCls is None or SolverCls is None:
        from PaST.solvers.bnb_solver_custom import (  # local import to keep optional
            Instance as _Instance,
            BranchAndBoundSolver as _Solver,
        )

        InstanceCls = _Instance
        SolverCls = _Solver
        print("Using in-repo BnB solver: PaST.solvers.bnb_solver_custom")

    episodes_by_ratio = _generate_episodes_by_ratio(
        scale=args.scale,
        num_instances=int(args.num_instances),
        eval_seed=int(args.eval_seed),
        ratios=ratios,
        max_jobs=int(args.max_jobs) if args.max_jobs is not None else None,
    )

    out_dir = Path(args.out_dir)
    if _finite(float(args.bnb_time_limit)):
        bnb_tag = f"bnb{int(args.bnb_time_limit)}s"
    else:
        bnb_tag = "bnbInf"
    tag = f"b{int(args.beta)}g{int(args.gamma)}_{bnb_tag}"
    out_csv = (
        out_dir / f"qseq_sgbs_vs_bnb_{args.scale}_seed{int(args.eval_seed)}_{tag}.csv"
    )
    out_json = (
        out_dir / f"qseq_sgbs_vs_bnb_{args.scale}_seed{int(args.eval_seed)}_{tag}.json"
    )

    rows: List[Dict[str, Any]] = []

    total_qseq_time = 0.0
    total_bnb_time = 0.0

    for ratio, episodes in zip(ratios, episodes_by_ratio):
        print(f"\n=== ratio={ratio:.2f} | instances={len(episodes)} ===")

        # Build one batched input for SGBS (this is usually much faster than
        # calling SGBS 1-by-1 and also lets torch use multiple CPU threads).
        batch = batch_from_episodes(episodes, N_job_pad=int(q_cfg.env.N_job_pad))

        t0 = time.perf_counter()
        q_results = sgbs_q_sequence(
            model=model,
            variant_config=q_cfg,
            batch_data=batch,
            device=device,
            beta=int(args.beta),
            gamma=int(args.gamma),
        )
        t_q_total = time.perf_counter() - t0
        total_qseq_time += float(t_q_total)
        q_time_per_inst = float(t_q_total) / max(1, len(episodes))

        # Prepare BnB inputs.
        bnb_inputs: List[Tuple[int, np.ndarray, int, np.ndarray]] = []
        singles: List[Dict[str, Any]] = []
        for idx in range(len(episodes)):
            single_np = _slice_single_instance_np(batch, idx)
            singles.append(single_np)
            n_jobs, p, T, energy_costs = _past_to_bnb_instance(single_np)
            bnb_inputs.append(
                (
                    int(n_jobs),
                    np.asarray(p, dtype=np.int32),
                    int(T),
                    np.asarray(energy_costs, dtype=np.int64),
                )
            )

        # --- BnB (parallel optional) ---
        bnb_results: List[Tuple[List[int], float, float, Dict[str, int]]] = []

        bnb_workers = int(args.bnb_workers)
        if bnb_workers > 1:
            ctx = mp.get_context("spawn")
            t0 = time.perf_counter()
            with ProcessPoolExecutor(
                max_workers=bnb_workers, mp_context=ctx
            ) as executor:
                futures = [
                    executor.submit(
                        _solve_bnb_one,
                        n_jobs=int(n_jobs),
                        p_subset=p,
                        T=int(T),
                        energy_costs=energy_costs,
                        time_limit=float(args.bnb_time_limit),
                        bnb_path=(
                            str(_resolve_path(args.bnb_path)) if args.bnb_path else None
                        ),
                    )
                    for (n_jobs, p, T, energy_costs) in bnb_inputs
                ]
                for fut in futures:
                    seq, cost, stats = fut.result()
                    bnb_results.append((seq, float(cost), 0.0, stats))
            t_b_total = time.perf_counter() - t0
            total_bnb_time += float(t_b_total)
            bnb_time_per_inst = float(t_b_total) / max(1, len(episodes))
            # Fill in per-instance times (average) for CSV consistency.
            bnb_results = [
                (seq, cost, float(bnb_time_per_inst), stats)
                for (seq, cost, _t, stats) in bnb_results
            ]
        else:
            # Sequential path with per-instance timing and richer stats.
            for n_jobs, p, T, energy_costs in bnb_inputs:
                inst = InstanceCls(
                    n_jobs=int(n_jobs),
                    processing_times=np.asarray(p, dtype=np.int32),
                    T=int(T),
                    energy_costs=np.asarray(energy_costs, dtype=np.int64),
                )
                solver = SolverCls(inst, time_limit=float(args.bnb_time_limit))
                t0 = time.perf_counter()
                bnb_seq, bnb_cost = solver.solve()
                t_b = time.perf_counter() - t0
                total_bnb_time += float(t_b)
                stats = {
                    "nodes_explored": int(getattr(solver, "nodes_explored", -1)),
                    "binpack_attempts": int(getattr(solver, "binpack_attempts", -1)),
                    "pruned_by_binpack": int(getattr(solver, "pruned_by_binpack", -1)),
                }
                bnb_results.append(
                    ([int(x) for x in bnb_seq], float(bnb_cost), float(t_b), stats)
                )

        # Build output rows.
        for idx, (single_np, q_res) in enumerate(zip(singles, q_results)):
            n_jobs, _p, T, _energy_costs = bnb_inputs[idx]
            bnb_seq, bnb_cost, t_b, stats = bnb_results[idx]

            # Cross-check cost with PaST DP (should match if conversion is correct).
            bnb_dp = dp_schedule_for_job_sequence(single_np, [int(x) for x in bnb_seq])

            q_energy = float(q_res.total_energy)
            bnb_energy = float(bnb_cost)
            bnb_energy_dp = float(bnb_dp.total_energy)

            gap = float("nan")
            if _finite(q_energy) and _finite(bnb_energy) and bnb_energy > 0:
                gap = float((q_energy - bnb_energy) / bnb_energy)

            # For batched SGBS we report average per-instance time.
            # For parallel BnB we may not have per-instance timing.
            rows.append(
                {
                    "ratio": float(ratio),
                    "instance_idx": int(idx),
                    "n_jobs": int(n_jobs),
                    "T_limit": int(T),
                    "qseq_sgbs_energy": q_energy,
                    "qseq_sgbs_time_s": float(q_time_per_inst),
                    "bnb_energy": bnb_energy,
                    "bnb_energy_dp_check": bnb_energy_dp,
                    "bnb_time_s": float(t_b),
                    "bnb_nodes_explored": int(stats.get("nodes_explored", -1)),
                    "bnb_binpack_attempts": int(stats.get("binpack_attempts", -1)),
                    "bnb_pruned_by_binpack": int(stats.get("pruned_by_binpack", -1)),
                    "gap_rel_q_minus_opt": gap,
                }
            )

            # Progress print.
            if (idx + 1) % 5 == 0:
                print(
                    f"  {idx+1}/{len(episodes)} | "
                    f"q={q_energy:.3f} (~{q_time_per_inst:.2f}s/inst) | "
                    f"bnb={bnb_energy:.3f} ({t_b:.2f}s) | "
                    f"gap={(100.0*gap):.2f}%"
                )

    # Summary
    gaps = np.array([r["gap_rel_q_minus_opt"] for r in rows], dtype=np.float64)
    finite_mask = np.isfinite(gaps)

    summary: Dict[str, Any] = {
        "config": {
            "checkpoint": str(ckpt_path),
            "variant_id": str(args.variant_id),
            "scale": str(args.scale),
            "eval_seed": int(args.eval_seed),
            "num_instances": int(args.num_instances),
            "actual_instances": int(len(rows)),
            "ratios": [float(x) for x in ratios],
            "beta": int(args.beta),
            "gamma": int(args.gamma),
            "device": str(args.device),
            "bnb_path": str(_resolve_path(args.bnb_path)) if args.bnb_path else None,
            "bnb_time_limit": float(args.bnb_time_limit),
            "bnb_workers": int(args.bnb_workers),
            "torch_threads": int(args.torch_threads),
            "max_jobs": int(args.max_jobs),
        },
        "timing": {
            "total_qseq_sgbs_time_s": float(total_qseq_time),
            "total_bnb_time_s": float(total_bnb_time),
            "avg_qseq_sgbs_time_s": float(total_qseq_time / max(1, len(rows))),
            "avg_bnb_time_s": float(total_bnb_time / max(1, len(rows))),
        },
        "quality": {
            "mean_gap_rel": (
                float(np.mean(gaps[finite_mask])) if finite_mask.any() else float("nan")
            ),
            "median_gap_rel": (
                float(np.median(gaps[finite_mask]))
                if finite_mask.any()
                else float("nan")
            ),
            "max_gap_rel": (
                float(np.max(gaps[finite_mask])) if finite_mask.any() else float("nan")
            ),
            "min_gap_rel": (
                float(np.min(gaps[finite_mask])) if finite_mask.any() else float("nan")
            ),
            "frac_optimal_within_1e_6": (
                float(np.mean(np.abs(gaps[finite_mask]) <= 1e-6))
                if finite_mask.any()
                else float("nan")
            ),
        },
    }

    _write_csv(out_csv, rows)
    _write_json(out_json, summary)

    print(f"\nSaved CSV: {out_csv}")
    print(f"Saved JSON: {out_json}")
    print(
        "Summary: "
        f"avg_q_time={summary['timing']['avg_qseq_sgbs_time_s']:.3f}s | "
        f"avg_bnb_time={summary['timing']['avg_bnb_time_s']:.3f}s | "
        f"median_gap={100.0*summary['quality']['median_gap_rel']:.2f}%"
    )


if __name__ == "__main__":
    main()

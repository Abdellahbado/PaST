"""End-to-end evaluation for learned branching policies in PaST B&B.

This evaluates *solver performance*, not just offline ranking metrics.

It runs the in-repo `PaST.solvers.bnb_solver_custom.BranchAndBoundSolver` on a
reproducible set of synthetic instances (same generator as
`PaST.train_bb_branch_policy.generate`).

Policies compared:
- random: random ordering of duration-classes at each node
- min_w: deterministic heuristic (min window cost)
- model: learned ranker saved by `PaST.train_bb_branch_policy` (LR/LGBM/XGB)

Example:
  python -m PaST.cli.eval.run_eval_bnb_branch_policy \
    --policy all \
    --model artifacts/bb_xgb.pkl \
    --num_instances 100 --seed 1337 \
    --n_jobs 40 --T 200 \
    --duration_vocab "1,2,3,4,6,8,12,16" \
    --time_limit_s 2 \
    --out_csv analysis_out/bnb_branch_policy_end2end.csv
"""

from __future__ import annotations

import argparse
import csv
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from PaST.solvers.bnb_solver_custom import BranchAndBoundSolver, Instance
from PaST.solvers.branching_policies import (
    LightGBMBranchingPolicy,
    MinWindowBranchingPolicy,
    PairwiseLogisticBranchingPolicy,
    XGBoostBranchingPolicy,
)
from PaST.train_bb_branch_policy import generate_instance


def _parse_duration_vocab(spec: str) -> List[int]:
    s = (spec or "").strip()
    if not s:
        return [1, 2, 3, 4, 6, 8, 12, 16]
    if s.startswith("range:"):
        _, rest = s.split(":", 1)
        parts = rest.split(":")
        if len(parts) != 3:
            raise ValueError("range spec must be range:start:stop:step")
        start, stop, step = [int(x) for x in parts]
        return list(range(start, stop + 1, step))
    out = [int(x.strip()) for x in s.split(",") if x.strip()]
    out = [d for d in out if d > 0]
    if not out:
        raise ValueError("Empty duration vocab")
    return sorted(set(out))


@dataclass
class RandomDurationPolicy:
    duration_vocab: Sequence[int]

    def __call__(
        self, partial_sequence: List[int], remaining_jobs: set, solver
    ) -> List[int]:
        # Choose a random ordering over *available* duration classes.
        pts = solver.instance.processing_times
        cand = sorted({int(pts[j]) for j in remaining_jobs})
        # Deterministic per-node random seed (for reproducibility).
        h = 1469598103934665603
        for x in partial_sequence:
            h ^= int(x) + 0x9E3779B97F4A7C15
            h *= 1099511628211
        rng = random.Random(int(h & 0xFFFFFFFF))
        rng.shuffle(cand)
        return cand


def _load_policy(
    *,
    policy_name: str,
    model_path: Optional[str],
    duration_vocab: Sequence[int],
) -> Tuple[str, Optional[Callable]]:
    name = (policy_name or "").strip().lower()
    if name == "none":
        return "none", None
    if name == "random":
        return "random", RandomDurationPolicy(duration_vocab=duration_vocab)
    if name in {"min_w", "minw", "heuristic"}:
        return "min_w", MinWindowBranchingPolicy(duration_vocab=duration_vocab)
    if name == "model":
        if not model_path:
            raise ValueError("--model is required when --policy includes 'model'")
        with Path(model_path).open("rb") as f:
            payload = pickle.load(f)
        mtype = str(payload.get("model_type", "")).strip().lower()

        # Dispatch to the right wrapper.
        if mtype == "pairwise_logistic":
            return "model(pairwise_logistic)", PairwiseLogisticBranchingPolicy(
                model_path=str(model_path), duration_vocab=duration_vocab
            )
        if mtype == "lgbm_ranker":
            return "model(lgbm_ranker)", LightGBMBranchingPolicy(
                model_path=str(model_path), duration_vocab=duration_vocab
            )
        if mtype == "xgb_ranker":
            return "model(xgb_ranker)", XGBoostBranchingPolicy(
                model_path=str(model_path), duration_vocab=duration_vocab
            )
        raise ValueError(f"Unsupported model_type={payload.get('model_type')!r}")

    raise ValueError(f"Unknown policy: {policy_name}")


def _summarize(rows: List[Dict[str, Any]], policy_key: str) -> Dict[str, float]:
    sub = [r for r in rows if r["policy"] == policy_key]
    if not sub:
        return {"n": 0.0}

    times = np.array([float(r["solve_time_sec"]) for r in sub], dtype=np.float64)
    nodes = np.array([int(r["nodes_explored"]) for r in sub], dtype=np.float64)
    costs = np.array([float(r["best_cost"]) for r in sub], dtype=np.float64)
    timed_out = np.array([int(bool(r["timed_out"])) for r in sub], dtype=np.float64)

    def q(x: np.ndarray, p: float) -> float:
        return float(np.quantile(x, p))

    return {
        "n": float(len(sub)),
        "timeouts": float(timed_out.sum()),
        "time_mean": float(times.mean()),
        "time_p50": q(times, 0.50),
        "time_p90": q(times, 0.90),
        "nodes_mean": float(nodes.mean()),
        "nodes_p50": q(nodes, 0.50),
        "cost_mean": float(costs.mean()),
        "cost_p50": q(costs, 0.50),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--policy",
        type=str,
        default="all",
        choices=["all", "random", "min_w", "model"],
        help="Which policy/policies to run. 'all' runs random+min_w+model.",
    )
    p.add_argument("--model", type=str, default="", help="Path to saved model .pkl")

    p.add_argument("--num_instances", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--n_jobs", type=int, default=40)
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--duration_vocab", type=str, default="1,2,3,4,6,8,12,16")
    p.add_argument(
        "--duration_mixture",
        type=str,
        default="mixed",
        choices=["mixed", "uniform", "long", "short"],
    )
    p.add_argument(
        "--price_kind",
        type=str,
        default="mixed",
        choices=["mixed", "flat", "uniform", "sin", "valleys", "spiky"],
    )

    p.add_argument("--time_limit_s", type=float, default=2.0)
    p.add_argument("--out_csv", type=str, default="")
    p.add_argument("--log_every", type=int, default=10)

    return p


def main() -> None:
    args = build_parser().parse_args()

    duration_vocab = _parse_duration_vocab(args.duration_vocab)

    policies_to_run: List[str]
    if args.policy == "all":
        policies_to_run = ["random", "min_w", "model"]
    else:
        policies_to_run = [str(args.policy)]

    policy_specs: List[Tuple[str, Optional[Callable]]] = []
    for pol in policies_to_run:
        name, fn = _load_policy(
            policy_name=pol,
            model_path=str(args.model) if args.model else None,
            duration_vocab=duration_vocab,
        )
        policy_specs.append((name, fn))

    rng = random.Random(int(args.seed))

    out_rows: List[Dict[str, Any]] = []

    for instance_id in range(int(args.num_instances)):
        inst_seed = rng.randint(0, 2**31 - 1)
        inst_rng = random.Random(int(inst_seed))

        inst: Instance = generate_instance(
            n_jobs=int(args.n_jobs),
            T=int(args.T),
            duration_vocab=duration_vocab,
            rng=inst_rng,
            price_kind=str(args.price_kind),
            duration_mixture=str(args.duration_mixture),
        )

        for policy_name, policy_fn in policy_specs:
            solver = BranchAndBoundSolver(
                inst,
                time_limit=float(args.time_limit_s),
                verbose=False,
                branching_policy=policy_fn,
            )
            t0 = time.perf_counter()
            _seq, best_cost = solver.solve()
            wall = time.perf_counter() - t0

            out_rows.append(
                {
                    "instance_id": int(instance_id),
                    "inst_seed": int(inst_seed),
                    "policy": str(policy_name),
                    "n_jobs": int(inst.n_jobs),
                    "T": int(inst.T),
                    "time_limit_s": float(args.time_limit_s),
                    "solve_time_sec": float(getattr(solver, "solve_time_sec", wall)),
                    "wall_time_sec": float(wall),
                    "timed_out": bool(getattr(solver, "timed_out", False))
                    or (wall >= float(args.time_limit_s) * 0.999),
                    "best_cost": float(best_cost),
                    "nodes_explored": int(getattr(solver, "nodes_explored", -1)),
                    "binpack_attempts": int(getattr(solver, "binpack_attempts", -1)),
                    "pruned_by_binpack": int(getattr(solver, "pruned_by_binpack", -1)),
                }
            )

        if int(args.log_every) > 0 and (instance_id + 1) % int(args.log_every) == 0:
            # Print lightweight running summaries.
            keys = [k for k, _ in policy_specs]
            parts = []
            for k in keys:
                s = _summarize(out_rows, k)
                parts.append(
                    f"{k}: n={int(s.get('n',0))} t_p50={s.get('time_p50',float('nan')):.3f}s "
                    f"nodes_p50={s.get('nodes_p50',float('nan')):.0f} to={int(s.get('timeouts',0))}"
                )
            print(f"[{instance_id+1}/{int(args.num_instances)}] " + " | ".join(parts))

    # Final report
    print("\n[final]")
    for policy_name, _ in policy_specs:
        s = _summarize(out_rows, policy_name)
        print(f"  {policy_name}: {s}")

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()

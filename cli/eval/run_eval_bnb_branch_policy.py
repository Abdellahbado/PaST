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
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
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


# NOTE: In multiprocessing mode, passing instantiated policy objects (especially
# model-based ones) to ProcessPool tasks can be extremely slow or hang because
# those objects may be large and expensive to pickle/ship to workers.
#
# Instead, each worker process lazily loads the policy objects the first time it
# needs them, and caches them for subsequent tasks.
_WORKER_POLICY_CACHE: Dict[
    Tuple[str, str, Tuple[int, ...]], Tuple[str, Optional[Callable]]
] = {}


def _load_policy_cached(
    *,
    policy_name: str,
    model_path: Optional[str],
    duration_vocab: Sequence[int],
) -> Tuple[str, Optional[Callable]]:
    key = (
        str(policy_name or ""),
        str(model_path or ""),
        tuple(int(x) for x in duration_vocab),
    )
    hit = _WORKER_POLICY_CACHE.get(key)
    if hit is not None:
        return hit
    loaded = _load_policy(
        policy_name=str(policy_name),
        model_path=str(model_path) if model_path else None,
        duration_vocab=duration_vocab,
    )
    _WORKER_POLICY_CACHE[key] = loaded
    return loaded


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
    # In parallel mode we may not know the exact model policy name upfront
    # (e.g. 'model(xgb_ranker)'). Allow prefix matches for convenience.
    if policy_key.endswith("*"):
        prefix = policy_key[:-1]
        sub = [r for r in rows if str(r.get("policy", "")).startswith(prefix)]
    else:
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
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers over instances (uses multiprocessing).",
    )
    p.add_argument(
        "--compute_root_lb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, compute and log the root lower bound per instance (extra DP work).",
    )
    p.add_argument("--out_csv", type=str, default="")
    p.add_argument("--log_every", type=int, default=10)

    return p


def _solve_one_instance(
    *,
    instance_id: int,
    inst_seed: int,
    policies_to_run: List[str],
    model_path: Optional[str],
    n_jobs: int,
    T: int,
    duration_vocab: Sequence[int],
    price_kind: str,
    duration_mixture: str,
    time_limit_s: float,
    compute_root_lb: bool,
) -> List[Dict[str, Any]]:
    policy_specs: List[Tuple[str, Optional[Callable]]] = []
    for pol in policies_to_run:
        name, fn = _load_policy_cached(
            policy_name=str(pol),
            model_path=str(model_path) if model_path else None,
            duration_vocab=duration_vocab,
        )
        policy_specs.append((name, fn))

    inst_rng = random.Random(int(inst_seed))
    inst: Instance = generate_instance(
        n_jobs=int(n_jobs),
        T=int(T),
        duration_vocab=list(duration_vocab),
        rng=inst_rng,
        price_kind=str(price_kind),
        duration_mixture=str(duration_mixture),
    )

    root_lb = float("nan")
    if bool(compute_root_lb):
        lb_solver = BranchAndBoundSolver(
            inst,
            time_limit=float(time_limit_s),
            verbose=False,
            branching_policy=None,
        )
        try:
            root_lb_val, _blocks, _relaxed = lb_solver._compute_lower_bound_with_blocks(
                [], set(range(int(inst.n_jobs)))
            )
            root_lb = float(root_lb_val)
        except Exception:
            root_lb = float("nan")

    out_rows: List[Dict[str, Any]] = []
    for policy_name, policy_fn in policy_specs:
        solver = BranchAndBoundSolver(
            inst,
            time_limit=float(time_limit_s),
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
                "time_limit_s": float(time_limit_s),
                "solve_time_sec": float(getattr(solver, "solve_time_sec", wall)),
                "wall_time_sec": float(wall),
                "timed_out": bool(getattr(solver, "timed_out", False))
                or (wall >= float(time_limit_s) * 0.999),
                "best_cost": float(best_cost),
                "root_lb": float(root_lb),
                "gap_to_lb": (
                    float(best_cost - root_lb)
                    if (root_lb == root_lb and best_cost == best_cost)
                    else float("nan")
                ),
                "gap_to_lb_rel": (
                    float((best_cost - root_lb) / max(1e-9, abs(root_lb)))
                    if (root_lb == root_lb and best_cost == best_cost)
                    else float("nan")
                ),
                "nodes_explored": int(getattr(solver, "nodes_explored", -1)),
                "binpack_attempts": int(getattr(solver, "binpack_attempts", -1)),
                "pruned_by_binpack": int(getattr(solver, "pruned_by_binpack", -1)),
            }
        )

    return out_rows


def main() -> None:
    args = build_parser().parse_args()

    duration_vocab = _parse_duration_vocab(args.duration_vocab)

    policies_to_run: List[str]
    if args.policy == "all":
        policies_to_run = ["random", "min_w", "model"]
    else:
        policies_to_run = [str(args.policy)]

    workers = int(getattr(args, "workers", 1) or 1)
    if workers <= 1:
        # Safe to resolve exact display names (no multiprocessing fork/spawn).
        policy_keys: List[str] = []
        for pol in policies_to_run:
            name, _fn = _load_policy_cached(
                policy_name=str(pol),
                model_path=str(args.model) if args.model else None,
                duration_vocab=duration_vocab,
            )
            policy_keys.append(str(name))
    else:
        # IMPORTANT: do not unpickle the model in the parent before creating the
        # process pool. On Linux, ProcessPoolExecutor defaults to fork, and
        # unpickling/importing XGBoost/LightGBM/OpenMP in the parent can lead to
        # workers hanging indefinitely.
        #
        # We'll summarize model results via prefix match 'model*'.
        policy_keys = [
            "random",
            "min_w",
            "model*",
        ]

    out_rows: List[Dict[str, Any]] = []

    # Pre-sample instance seeds so parallelism doesn't change which instances are evaluated.
    rng = random.Random(int(args.seed))
    inst_seeds = [rng.randint(0, 2**31 - 1) for _ in range(int(args.num_instances))]

    if workers <= 1:
        for instance_id, inst_seed in enumerate(inst_seeds):
            out_rows.extend(
                _solve_one_instance(
                    instance_id=int(instance_id),
                    inst_seed=int(inst_seed),
                    policies_to_run=policies_to_run,
                    model_path=str(args.model) if args.model else None,
                    n_jobs=int(args.n_jobs),
                    T=int(args.T),
                    duration_vocab=duration_vocab,
                    price_kind=str(args.price_kind),
                    duration_mixture=str(args.duration_mixture),
                    time_limit_s=float(args.time_limit_s),
                    compute_root_lb=bool(args.compute_root_lb),
                )
            )
            if int(args.log_every) > 0 and (instance_id + 1) % int(args.log_every) == 0:
                parts = []
                for k in policy_keys:
                    s = _summarize(out_rows, k)
                    parts.append(
                        f"{k}: n={int(s.get('n',0))} t_p50={s.get('time_p50',float('nan')):.3f}s "
                        f"nodes_p50={s.get('nodes_p50',float('nan')):.0f} to={int(s.get('timeouts',0))}"
                    )
                print(
                    f"[{instance_id+1}/{int(args.num_instances)}] " + " | ".join(parts)
                )
    else:
        # Parallel over instances.
        submitted = 0
        completed = 0
        # Use spawn context for robustness with OpenMP-backed libraries.
        with ProcessPoolExecutor(
            max_workers=int(workers), mp_context=get_context("spawn")
        ) as ex:
            futs = []
            for instance_id, inst_seed in enumerate(inst_seeds):
                futs.append(
                    ex.submit(
                        _solve_one_instance,
                        instance_id=int(instance_id),
                        inst_seed=int(inst_seed),
                        policies_to_run=policies_to_run,
                        model_path=str(args.model) if args.model else None,
                        n_jobs=int(args.n_jobs),
                        T=int(args.T),
                        duration_vocab=list(duration_vocab),
                        price_kind=str(args.price_kind),
                        duration_mixture=str(args.duration_mixture),
                        time_limit_s=float(args.time_limit_s),
                        compute_root_lb=bool(args.compute_root_lb),
                    )
                )
                submitted += 1

            for fut in as_completed(futs):
                try:
                    rows_i = fut.result()
                except Exception as e:
                    completed += 1
                    print(f"[{completed}/{submitted}] worker error: {e!r}")
                    continue
                out_rows.extend(rows_i)
                completed += 1
                if int(args.log_every) > 0 and completed % int(args.log_every) == 0:
                    parts = []
                    for k in policy_keys:
                        s = _summarize(out_rows, k)
                        parts.append(
                            f"{k}: n={int(s.get('n',0))} t_p50={s.get('time_p50',float('nan')):.3f}s "
                            f"nodes_p50={s.get('nodes_p50',float('nan')):.0f} to={int(s.get('timeouts',0))}"
                        )
                    print(f"[{completed}/{submitted}] " + " | ".join(parts))

    # Final report
    print("\n[final]")
    # In parallel mode, 'model*' is a prefix summary; also print exact model
    # policy names observed (e.g. model(xgb_ranker)).
    for policy_name in policy_keys:
        s = _summarize(out_rows, str(policy_name))
        print(f"  {policy_name}: {s}")
    model_names = sorted(
        {
            str(r.get("policy"))
            for r in out_rows
            if str(r.get("policy", "")).startswith("model(")
        }
    )
    for mn in model_names:
        s = _summarize(out_rows, mn)
        print(f"  {mn}: {s}")

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

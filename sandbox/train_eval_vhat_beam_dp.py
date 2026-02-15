"""Sandbox: train a tiny linear Vhat and use it to guide beam-pruned sparse DP.

This is a minimal end-to-end check on small instances:
- generate repeating 20-hour TOU prices
- generate a multiset of jobs (processing times)
- collect training samples by solving exact subproblems
- fit ridge regression for Vhat(t,state)
- run guided beam DP and compare to exact DP

Run:
  python PaST/sandbox/train_eval_vhat_beam_dp.py --seed 0 --D 2 --N 40 --pmax 8 --samples 200 --beam 2000
"""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from PaST.solvers.optimal_benchmark_dp import solve_optimal_benchmark_dp
from PaST.solvers.vhat_linear import (
    FeatureSpec,
    LinearRidgeValueModel,
    fit_ridge,
    phi_for_state,
)
from PaST.solvers.vhat_tou_features import build_tou_feature_context


def daily_tou_20() -> List[float]:
    # Matches New Benchmark/new_data.py daily_tou pattern
    day: List[float] = []
    for h in range(20):
        if 0 <= h < 4:
            day.append(1.0)
        elif 4 <= h < 12:
            day.append(2.0)
        elif 12 <= h < 16:
            day.append(4.0)
        else:
            day.append(2.0)
    return day


def build_instance(
    *, rng: np.random.Generator, D: int, N: int, pmax: int
) -> Tuple[List[int], np.ndarray]:
    prices = np.array(daily_tou_20() * int(D), dtype=np.float64)
    p = rng.integers(1, int(pmax) + 1, size=int(N)).astype(int).tolist()
    # Ensure feasibility: sum(p) <= T (otherwise exact DP returns infeasible)
    T = int(len(prices))
    while sum(p) > T:
        i = int(rng.integers(0, len(p)))
        if p[i] > 1:
            p[i] -= 1
    return p, prices


def encode_setup(p_list: Sequence[int]):
    p_arr = np.asarray(p_list, dtype=np.int32)
    lengths, inv = np.unique(p_arr, return_inverse=True)
    totals = np.bincount(inv, minlength=len(lengths)).astype(np.int32)
    K = int(len(lengths))
    radices = (totals + 1).astype(np.int32)
    mult = np.ones(K, dtype=np.int64)
    for i in range(1, K):
        mult[i] = mult[i - 1] * int(radices[i - 1])
    return lengths.astype(np.int32), totals, radices, mult


def decode_state(state: int, radices: np.ndarray) -> Tuple[int, ...]:
    K = int(len(radices))
    u = [0] * K
    x = int(state)
    for i in range(K):
        r = int(radices[i])
        u[i] = x % r
        x //= r
    return tuple(u)


def remaining_p_list(
    used: Sequence[int], totals: np.ndarray, lengths: np.ndarray
) -> List[int]:
    rem_counts = (totals.astype(np.int32) - np.asarray(used, dtype=np.int32)).astype(
        np.int32
    )
    out: List[int] = []
    for nk, L in zip(rem_counts.tolist(), lengths.tolist()):
        if nk > 0:
            out.extend([int(L)] * int(nk))
    return out


def fit_ridge_with_prior(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float,
    prior_w: np.ndarray,
    prior_strength: float,
) -> np.ndarray:
    """Ridge with quadratic prior around prior_w.

    Minimizes: ||Xw - y||^2 + l2||w||^2 + prior_strength||w - prior_w||^2
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    prior_w = np.asarray(prior_w, dtype=np.float64)

    D = int(X.shape[1])
    if prior_w.shape[0] != D:
        raise ValueError("prior_w dimension mismatch")

    A = X.T @ X + (float(l2) + float(prior_strength)) * np.eye(D, dtype=np.float64)
    b = X.T @ y + float(prior_strength) * prior_w
    return np.linalg.solve(A, b).astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--D", type=int, default=2)
    ap.add_argument("--N", type=int, default=40)
    ap.add_argument("--pmax", type=int, default=8)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--beam", type=int, default=2000)
    ap.add_argument("--prune-factor", type=float, default=2.0)
    ap.add_argument(
        "--skip-exact",
        action="store_true",
        help="Skip solving the full exact DP for the generated instance (useful for large stages).",
    )
    ap.add_argument(
        "--transferable-features",
        action="store_true",
        help="Use fixed-dimension features across different K (recommended for curriculum transfer).",
    )
    ap.add_argument(
        "--load-model",
        type=str,
        default="",
        help="Optional .npz model path to warm-start or reuse.",
    )
    ap.add_argument(
        "--freeze-loaded",
        action="store_true",
        help="Use loaded model directly; skip state-label training.",
    )
    ap.add_argument(
        "--prior-strength",
        type=float,
        default=0.0,
        help="If >0 and a model is loaded, fine-tuning uses a quadratic prior around loaded weights.",
    )
    ap.add_argument(
        "--save-model",
        type=str,
        default="",
        help="Optional output .npz path for trained weights.",
    )
    args = ap.parse_args()

    rng = np.random.default_rng(int(args.seed))

    p, prices = build_instance(
        rng=rng, D=int(args.D), N=int(args.N), pmax=int(args.pmax)
    )
    T = int(len(prices))

    if bool(args.skip_exact):
        exact = None
        t_exact = float("nan")
    else:
        t0 = time.perf_counter()
        exact = solve_optimal_benchmark_dp(p, prices, tie_break="early")
        t_exact = time.perf_counter() - t0

        if not exact.feasible:
            print("Generated infeasible instance unexpectedly.")
            return

    lengths, totals, radices, mult = encode_setup(p)
    ctx = build_tou_feature_context(prices, H=20, validate_repeating=True)

    if bool(args.transferable_features):
        spec = FeatureSpec(
            include_per_class_counts=False,
            include_per_class_now_cost=False,
            include_bins=True,
        )
    else:
        spec = FeatureSpec(
            include_per_class_counts=True,
            include_per_class_now_cost=True,
            include_bins=True,
        )

    loaded_w: np.ndarray | None = None
    loaded_model_path = str(args.load_model).strip()
    if loaded_model_path:
        ckpt = np.load(loaded_model_path)
        loaded_w = np.asarray(ckpt["weights"], dtype=np.float64)
        # If checkpoint has stored spec, prefer it for compatibility.
        if {
            "include_per_class_counts",
            "include_per_class_now_cost",
            "include_bins",
        }.issubset(set(ckpt.files)):
            spec = FeatureSpec(
                include_per_class_counts=bool(int(ckpt["include_per_class_counts"])),
                include_per_class_now_cost=bool(
                    int(ckpt["include_per_class_now_cost"])
                ),
                include_bins=bool(int(ckpt["include_bins"])),
            )

    probe = phi_for_state(
        t=0,
        used=tuple([0] * len(lengths)),
        totals=totals,
        lengths=lengths.tolist(),
        ctx=ctx,
        spec=spec,
    )
    feat_dim = int(probe.shape[0])
    can_use_loaded = loaded_w is not None and int(loaded_w.shape[0]) == feat_dim

    if loaded_w is not None and not can_use_loaded:
        if bool(args.freeze_loaded):
            raise RuntimeError(
                f"Loaded model dim={loaded_w.shape[0]} incompatible with current feat_dim={feat_dim}. "
                f"Use --transferable-features consistently across stages."
            )
        loaded_w = None

    train_mode = "fresh"
    attempts = 0

    if bool(args.freeze_loaded):
        if loaded_w is None:
            raise RuntimeError(
                "--freeze-loaded requires --load-model with compatible feature dimension."
            )
        w = loaded_w
        train_mode = "frozen_loaded"
    else:
        # Collect training samples (robustly: feasible random states can be sparse)
        X_rows: List[np.ndarray] = []
        y_vals: List[float] = []

        target_samples = int(args.samples)
        max_attempts = max(5 * target_samples, 200)
        while len(X_rows) < target_samples and attempts < max_attempts:
            attempts += 1
            t = int(rng.integers(0, T))

            used = tuple(int(x) for x in rng.integers(0, totals + 1))
            rem_work = int(
                np.sum((totals - np.asarray(used, dtype=np.int32)) * lengths)
            )
            if rem_work > (T - t):
                continue

            rem_p = remaining_p_list(used, totals, lengths)
            if not rem_p:
                label_cost = 0.0
            else:
                # Subproblem starting at absolute time t: use suffix prices[t:]
                sub = solve_optimal_benchmark_dp(rem_p, prices[t:], tie_break="cost")
                if not sub.feasible:
                    continue
                label_cost = float(sub.cost)

            x = phi_for_state(
                t=t,
                used=used,
                totals=totals,
                lengths=lengths.tolist(),
                ctx=ctx,
                spec=spec,
            )
            X_rows.append(x)
            y_vals.append(label_cost)

        if len(X_rows) < max(20, target_samples // 2):
            raise RuntimeError(
                f"Too few training samples collected: {len(X_rows)} "
                f"after {attempts} attempts (target={target_samples})."
            )

        X = np.vstack(X_rows)
        y = np.asarray(y_vals, dtype=np.float64)

        if loaded_w is not None and float(args.prior_strength) > 0.0:
            w = fit_ridge_with_prior(
                X,
                y,
                l2=float(args.l2),
                prior_w=loaded_w,
                prior_strength=float(args.prior_strength),
            )
            train_mode = "fine_tune_prior"
        else:
            w = fit_ridge(X, y, l2=float(args.l2))
            train_mode = "fresh" if loaded_w is None else "retrain_from_loaded_spec"

    model = LinearRidgeValueModel(weights=w, spec=spec)

    save_model_path = str(args.save_model).strip()
    if save_model_path:
        out = Path(save_model_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            weights=w,
            include_per_class_counts=int(spec.include_per_class_counts),
            include_per_class_now_cost=int(spec.include_per_class_now_cost),
            include_bins=int(spec.include_bins),
            H=20,
        )

    used_cache: Dict[int, Tuple[int, ...]] = {0: tuple([0] * len(lengths))}

    def vhat(t: int, state: int) -> float:
        s = int(state)
        cached = used_cache.get(s)
        if cached is None:
            cached = decode_state(s, radices)
            used_cache[s] = cached
        return model.predict_from_used(
            t=int(t), used=cached, totals=totals, lengths=lengths.tolist(), ctx=ctx
        )

    t1 = time.perf_counter()
    guided = solve_optimal_benchmark_dp(
        p,
        prices,
        tie_break="early",
        guided=True,
        beam_width=int(args.beam),
        prune_factor=float(args.prune_factor),
        vhat=vhat,
    )
    t_guided = time.perf_counter() - t1

    if exact is None:
        exact_cost = float("nan")
        exact_finish = -1
        gap = float("nan")
    else:
        exact_cost = float(exact.cost)
        exact_finish = int(exact.finish_time)
        gap = (guided.cost - exact.cost) / max(1e-9, abs(exact.cost)) * 100.0

    print(
        " ".join(
            [
                f"T={T}",
                f"N={len(p)}",
                f"K={len(lengths)}",
                f"sum_p={sum(p)}",
                f"exact_cost={exact_cost:.6f}",
                f"exact_finish={exact_finish}",
                f"exact_s={t_exact:.4f}",
                f"guided_cost={guided.cost:.6f}",
                f"guided_finish={guided.finish_time}",
                f"guided_s={t_guided:.4f}",
                f"gap_pct={gap:.3f}",
                f"beam={int(args.beam)}",
                f"feat_dim={feat_dim}",
                f"train_mode={train_mode}",
                f"attempts={attempts}",
            ]
        )
    )


if __name__ == "__main__":
    main()

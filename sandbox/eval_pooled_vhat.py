"""Pooled cross-instance training and evaluation for Vhat-guided beam DP.

Unlike eval_guided_vhat_holdout.py which trains a separate model per instance,
this script:
  1. Collects labeled DP states from multiple TRAINING instances (--train-seeds)
  2. Fits ONE shared model on the pooled data
  3. Evaluates the shared model (frozen) on HELD-OUT instances (--eval-seeds)

Supports multiple model types: linear, poly, mlp, lgbm.
Data collection is parallelized across CPU cores.

Example:
    python PaST/sandbox/eval_pooled_vhat.py \
        --D 6 --N 30 --pmax 3 \
        --train-seeds 0-19  --samples-per-instance 2000 \
        --eval-seeds 100-129 --beams 2,3,5 \
        --transferable-features --normalize --normalize-labels \
        --model-type poly \
        --save-model PaST/models/vhat_pooled_poly.npz \
        --out-csv PaST/logs/pooled_poly.csv
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from functools import partial
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from PaST.solvers.optimal_benchmark_dp import solve_optimal_benchmark_dp
from PaST.solvers.vhat_linear import (
    FeatureSpec,
    LinearRidgeValueModel,
    fit_ridge,
    phi_for_state,
)
from PaST.solvers.vhat_tou_features import build_tou_feature_context
from PaST.sandbox.train_eval_vhat_beam_dp import (
    build_instance,
    decode_state,
    encode_setup,
    fit_ridge_with_prior,
    remaining_p_list,
)
from PaST.solvers.vhat_models import (
    PolyRidgeValueModel,
    MLPValueModel,
    LGBMValueModel,
    fit_poly_ridge,
    fit_mlp,
    fit_lgbm,
    _poly_expand_batch,
)

# Union type for all model types
ValueModel = Union[LinearRidgeValueModel, PolyRidgeValueModel, MLPValueModel, LGBMValueModel]


def _collect_state_labels(
    *,
    rng: np.random.Generator,
    T: int,
    totals: np.ndarray,
    lengths: np.ndarray,
    prices: np.ndarray,
    target_samples: int,
    normalize_labels: bool = False,
    prefix_prices: np.ndarray | None = None,
) -> Tuple[List[Tuple[int, Tuple[int, ...]]], List[float], int]:
    """Sample random (t, used) states and label them with exact DP cost-to-go.

    If normalize_labels is True, each label y is divided by sum(prices[t:])
    so the model predicts cost as a fraction of remaining price budget.
    This makes labels scale-invariant across instance sizes.
    """
    states: List[Tuple[int, Tuple[int, ...]]] = []
    labels: List[float] = []
    attempts = 0
    max_attempts = max(6 * target_samples, 300)

    # Precompute suffix sums of prices for label normalization
    if normalize_labels and prefix_prices is None:
        prefix_prices = np.concatenate([[0.0], np.cumsum(prices, dtype=np.float64)])

    while len(states) < target_samples and attempts < max_attempts:
        attempts += 1
        t = int(rng.integers(0, T))
        used = tuple(int(x) for x in rng.integers(0, totals + 1))

        rem_work = int(
            np.sum(
                (totals - np.asarray(used, dtype=np.int32))
                * np.asarray(lengths, dtype=np.int32)
            )
        )
        if rem_work > (T - t):
            continue

        rem_p = remaining_p_list(used, totals, lengths)
        if not rem_p:
            y = 0.0
        else:
            sub = solve_optimal_benchmark_dp(rem_p, prices[t:], tie_break="cost")
            if not sub.feasible:
                continue
            y = float(sub.cost)

        # Normalize label by remaining price budget
        if normalize_labels and prefix_prices is not None:
            rem_budget = float(prefix_prices[T] - prefix_prices[t])
            if rem_budget > 1e-9:
                y = y / rem_budget

        states.append((t, tuple(int(x) for x in used)))
        labels.append(y)

    return states, labels, attempts


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    var = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if var <= 1e-12:
        return float("nan")
    sse = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - sse / var


def parse_seed_range(s: str) -> List[int]:
    """Parse '0-19' or '0,1,5,10' into list of ints."""
    s = s.strip()
    if "-" in s and "," not in s:
        parts = s.split("-")
        return list(range(int(parts[0]), int(parts[1]) + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def _collect_worker(
    seed: int,
    D: int,
    N: int,
    pmax: int,
    samples_per_instance: int,
    spec: FeatureSpec,
    normalize_labels: bool,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Worker function for parallel data collection. Runs in a subprocess."""
    rng = np.random.default_rng(seed)
    p, prices = build_instance(rng=rng, D=D, N=N, pmax=pmax)
    T = int(len(prices))
    lengths, totals, radices, _mult = encode_setup(p)
    ctx = build_tou_feature_context(prices, H=20, validate_repeating=True)
    prefix_prices = np.concatenate([[0.0], np.cumsum(prices, dtype=np.float64)])

    states, labels, attempts = _collect_state_labels(
        rng=rng,
        T=T,
        totals=totals,
        lengths=lengths,
        prices=prices,
        target_samples=samples_per_instance,
        normalize_labels=normalize_labels,
        prefix_prices=prefix_prices,
    )

    if not states:
        return np.empty((0, 0)), np.empty(0), attempts

    X_inst = np.vstack(
        [
            phi_for_state(
                t=int(t_v),
                used=used_v,
                totals=totals,
                lengths=lengths.tolist(),
                ctx=ctx,
                spec=spec,
            )
            for (t_v, used_v) in states
        ]
    )
    y_inst = np.asarray(labels, dtype=np.float64)
    return X_inst, y_inst, attempts


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pooled cross-instance training: train ONE model on multiple instances, evaluate on held-out instances."
    )
    # Instance parameters
    ap.add_argument("--D", type=int, default=6)
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--pmax", type=int, default=3)

    # Training
    ap.add_argument(
        "--train-seeds",
        type=str,
        default="0-19",
        help="Seeds for training instances, e.g. '0-19' or '0,5,10,15'.",
    )
    ap.add_argument(
        "--samples-per-instance",
        type=int,
        default=2000,
        help="Number of random state samples per training instance.",
    )
    ap.add_argument("--l2", type=float, default=1e-3)

    # Model type
    ap.add_argument(
        "--model-type",
        type=str,
        default="linear",
        choices=["linear", "poly", "mlp", "lgbm"],
        help="Model type: linear (Ridge), poly (degree-2 polynomial Ridge), "
             "mlp (small neural net), lgbm (gradient boosted trees).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers for data collection (0 = auto = cpu_count).",
    )

    # Evaluation
    ap.add_argument(
        "--eval-seeds",
        type=str,
        default="100-129",
        help="Seeds for held-out evaluation instances.",
    )
    ap.add_argument(
        "--beam",
        type=int,
        default=3,
        help="Single beam width (ignored if --beams is provided).",
    )
    ap.add_argument(
        "--beams",
        type=str,
        default="",
        help="Comma-separated beam widths to sweep.",
    )
    ap.add_argument("--prune-factor", type=float, default=2.0)

    # Features
    ap.add_argument(
        "--transferable-features",
        action="store_true",
        help="Drop per-class features for fixed-dimension feature vector.",
    )
    ap.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize features by T for scale-invariant ratios.",
    )
    ap.add_argument(
        "--normalize-labels",
        action="store_true",
        help="Normalize labels by sum(prices[t:]) for scale-invariant cost predictions. "
             "Essential for cross-size transfer: makes weights independent of instance scale.",
    )

    # Model I/O
    ap.add_argument(
        "--load-model",
        type=str,
        default="",
        help="Load pre-trained model instead of training. Skips training phase.",
    )
    ap.add_argument(
        "--save-model",
        type=str,
        default="",
        help="Save the pooled model to this .npz path.",
    )

    # Eval for different sizes (optional overrides for eval phase)
    ap.add_argument("--eval-D", type=int, default=0, help="Override D for eval instances (0 = same as --D).")
    ap.add_argument("--eval-N", type=int, default=0, help="Override N for eval instances (0 = same as --N).")
    ap.add_argument("--eval-pmax", type=int, default=0, help="Override pmax for eval instances (0 = same as --pmax).")

    ap.add_argument(
        "--out-csv",
        type=str,
        default="PaST/logs/eval_pooled_vhat.csv",
    )
    args = ap.parse_args()

    train_seeds = parse_seed_range(args.train_seeds)
    eval_seeds = parse_seed_range(args.eval_seeds)

    if str(args.beams).strip():
        beams = [int(x) for x in str(args.beams).split(",") if x.strip()]
    else:
        beams = [int(args.beam)]

    use_normalize = bool(args.normalize)
    if bool(args.transferable_features):
        spec = FeatureSpec(
            include_per_class_counts=False,
            include_per_class_now_cost=False,
            include_bins=True,
            normalize=use_normalize,
        )
    else:
        spec = FeatureSpec(
            include_per_class_counts=True,
            include_per_class_now_cost=True,
            include_bins=True,
            normalize=use_normalize,
        )

    # Eval instance parameters (possibly different from training)
    eval_D = int(args.eval_D) if int(args.eval_D) > 0 else int(args.D)
    eval_N = int(args.eval_N) if int(args.eval_N) > 0 else int(args.N)
    eval_pmax = int(args.eval_pmax) if int(args.eval_pmax) > 0 else int(args.pmax)

    model_type = str(args.model_type).strip().lower()
    n_workers = int(args.workers) if int(args.workers) > 0 else mp.cpu_count()

    # =========================================================================
    # PHASE 1: TRAINING — pool labeled states from multiple instances
    # =========================================================================
    loaded_model_path = str(args.load_model).strip()
    if loaded_model_path:
        print(f"[pool] Loading pre-trained model from {loaded_model_path}")
        ckpt = np.load(loaded_model_path, allow_pickle=True)
        _mt = str(ckpt["model_type"]) if "model_type" in ckpt.files else "linear"

        # Restore spec from checkpoint
        if {
            "include_per_class_counts",
            "include_per_class_now_cost",
            "include_bins",
        }.issubset(set(ckpt.files)):
            _norm = (
                bool(int(ckpt["normalize"]))
                if "normalize" in ckpt.files
                else use_normalize
            )
            spec = FeatureSpec(
                include_per_class_counts=bool(int(ckpt["include_per_class_counts"])),
                include_per_class_now_cost=bool(
                    int(ckpt["include_per_class_now_cost"])
                ),
                include_bins=bool(int(ckpt["include_bins"])),
                normalize=_norm,
            )

        if _mt == "poly":
            model = PolyRidgeValueModel.load(loaded_model_path)
            model_type = "poly"
        elif _mt == "mlp":
            model = MLPValueModel.load(loaded_model_path)
            model_type = "mlp"
        elif _mt == "lgbm":
            model = LGBMValueModel.load(loaded_model_path)
            model_type = "lgbm"
        else:
            w = np.asarray(ckpt["weights"], dtype=np.float64)
            model = LinearRidgeValueModel(weights=w, spec=spec)
            model_type = "linear"

        train_s = 0.0
        print(f"[pool] Loaded model: type={model_type}, spec={spec}")
    else:
        use_normalize_labels = bool(args.normalize_labels)
        print(
            f"[pool] === TRAINING PHASE ==="
            f"\n[pool] Model type: {model_type}"
            f"\n[pool] Instance params: D={args.D}, N={args.N}, pmax={args.pmax}"
            f"\n[pool] Train seeds: {train_seeds[0]}-{train_seeds[-1]} ({len(train_seeds)} instances)"
            f"\n[pool] Samples per instance: {args.samples_per_instance}"
            f"\n[pool] Features: spec={spec}"
            f"\n[pool] normalize_labels={use_normalize_labels}"
            f"\n[pool] Workers: {n_workers}"
        )

        train_t0 = time.perf_counter()

        # === Parallel data collection ===
        worker_fn = partial(
            _collect_worker,
            D=int(args.D),
            N=int(args.N),
            pmax=int(args.pmax),
            samples_per_instance=int(args.samples_per_instance),
            spec=spec,
            normalize_labels=use_normalize_labels,
        )

        if n_workers > 1 and len(train_seeds) > 1:
            print(f"[pool] Collecting data with {n_workers} parallel workers...")
            with mp.Pool(n_workers) as pool:
                results = []
                for i, result in enumerate(pool.imap_unordered(worker_fn, train_seeds)):
                    results.append(result)
                    print(
                        f"[pool] collected {i+1}/{len(train_seeds)} instances "
                        f"({result[0].shape[0]} samples)"
                    )
        else:
            print(f"[pool] Collecting data sequentially...")
            results = []
            for i, seed in enumerate(train_seeds):
                t_inst = time.perf_counter()
                result = worker_fn(seed)
                results.append(result)
                inst_time = time.perf_counter() - t_inst
                print(
                    f"[pool] train seed={seed} ({i+1}/{len(train_seeds)}) "
                    f"samples={result[0].shape[0]} "
                    f"time={inst_time:.1f}s"
                )

        # Pool results
        all_X = [r[0] for r in results if r[0].size > 0]
        all_y = [r[1] for r in results if r[1].size > 0]
        total_attempts = sum(r[2] for r in results)

        X_pool = np.vstack(all_X)
        y_pool = np.concatenate(all_y)

        collect_time = time.perf_counter() - train_t0
        print(f"[pool] Data collection: {len(y_pool)} samples in {collect_time:.1f}s")

        # Train/test split on pooled data
        idx = np.arange(len(y_pool))
        np.random.default_rng(42).shuffle(idx)
        split = int(0.85 * len(idx))
        train_idx = idx[:split]
        test_idx = idx[split:]

        X_train, y_train = X_pool[train_idx], y_pool[train_idx]
        X_test, y_test = X_pool[test_idx], y_pool[test_idx]

        # ============== Model-specific training ==============
        fit_t0 = time.perf_counter()

        if model_type == "linear":
            w = fit_ridge(X_train, y_train, l2=float(args.l2))
            model = LinearRidgeValueModel(weights=w, spec=spec)
            y_hat_train = X_train @ w
            y_hat_test = X_test @ w
            feat_dim = int(X_pool.shape[1])
            # Print top weights
            top_k = min(10, len(w))
            sorted_idx = np.argsort(np.abs(w))[::-1][:top_k]
            print(f"[pool] Top-{top_k} weights:")
            for rank, fi in enumerate(sorted_idx):
                print(f"    #{rank+1} feat[{fi}] w={w[fi]:.6f}")

        elif model_type == "poly":
            w, powers = fit_poly_ridge(X_train, y_train, l2=float(args.l2), degree=2)
            model = PolyRidgeValueModel(weights=w, spec=spec, powers_=powers)
            X_train_poly = _poly_expand_batch(X_train, powers)
            X_test_poly = _poly_expand_batch(X_test, powers)
            y_hat_train = X_train_poly @ w
            y_hat_test = X_test_poly @ w
            feat_dim = int(X_train_poly.shape[1])
            print(f"[pool] Polynomial: {X_pool.shape[1]} raw → {feat_dim} poly features")

        elif model_type == "mlp":
            mlp_model = fit_mlp(
                X_train, y_train, X_test, y_test,
                hidden1=64, hidden2=32,
                lr=1e-3, batch_size=2048, max_epochs=200, patience=15,
            )
            mlp_model.spec = spec
            model = mlp_model
            # Compute predictions with numpy inference
            h1 = np.maximum(0, X_train @ model.W1 + model.b1)
            h2 = np.maximum(0, h1 @ model.W2 + model.b2)
            y_hat_train = (h2 @ model.W3 + model.b3).ravel()
            h1 = np.maximum(0, X_test @ model.W1 + model.b1)
            h2 = np.maximum(0, h1 @ model.W2 + model.b2)
            y_hat_test = (h2 @ model.W3 + model.b3).ravel()
            feat_dim = int(X_pool.shape[1])

        elif model_type == "lgbm":
            booster = fit_lgbm(
                X_train, y_train, X_test, y_test,
                n_estimators=100, max_depth=5, learning_rate=0.1, n_jobs=n_workers,
            )
            model = LGBMValueModel(booster=booster, spec=spec)
            y_hat_train = booster.predict(X_train)
            y_hat_test = booster.predict(X_test)
            feat_dim = int(X_pool.shape[1])

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        fit_time = time.perf_counter() - fit_t0
        train_s = time.perf_counter() - train_t0

        # Training diagnostics
        r2_train = _r2_score(y_train, y_hat_train)
        mae_train = float(np.mean(np.abs(y_train - y_hat_train)))
        r2_test = _r2_score(y_test, y_hat_test)
        mae_test = float(np.mean(np.abs(y_test - y_hat_test)))

        print(
            f"\n[pool] === TRAINING RESULTS ==="
            f"\n[pool] Model: {model_type}"
            f"\n[pool] Pooled samples: {len(y_pool)} (from {len(train_seeds)} instances)"
            f"\n[pool] feat_dim={feat_dim}"
            f"\n[pool] R2_train={r2_train:.4f}  R2_test={r2_test:.4f}"
            f"\n[pool] MAE_train={mae_train:.4f}  MAE_test={mae_test:.4f}"
            f"\n[pool] Data collection: {collect_time:.1f}s  Model fitting: {fit_time:.1f}s"
            f"\n[pool] Total training time: {train_s:.1f}s"
        )

        # Save model
        save_path = str(args.save_model).strip()
        if save_path:
            save_p = Path(save_path)
            save_p.parent.mkdir(parents=True, exist_ok=True)
            if model_type == "linear":
                np.savez(
                    save_p,
                    weights=w,
                    include_per_class_counts=int(spec.include_per_class_counts),
                    include_per_class_now_cost=int(spec.include_per_class_now_cost),
                    include_bins=int(spec.include_bins),
                    normalize=int(spec.normalize),
                    normalize_labels=int(use_normalize_labels),
                    model_type="linear",
                )
            elif model_type in ("poly", "mlp"):
                model.save(str(save_p))
                # Also save normalize_labels in a sidecar
                np.savez(
                    str(save_p) + ".meta",
                    normalize_labels=int(use_normalize_labels),
                )
            elif model_type == "lgbm":
                model.save(str(save_p))
                np.savez(
                    str(save_p) + ".meta",
                    normalize_labels=int(use_normalize_labels),
                )
            print(f"[pool] Model saved to {save_p} (type={model_type})")

    # =========================================================================
    # PHASE 2: EVALUATION — test shared model on held-out instances
    # =========================================================================
    print(
        f"\n[pool] === EVALUATION PHASE ==="
        f"\n[pool] Eval params: D={eval_D}, N={eval_N}, pmax={eval_pmax}"
        f"\n[pool] Eval seeds: {eval_seeds[0]}-{eval_seeds[-1]} ({len(eval_seeds)} instances)"
        f"\n[pool] Beams: {beams}"
    )

    rows: List[Dict[str, float]] = []
    w = model.weights if hasattr(model, 'weights') else None

    # Determine if labels were normalized during training (once, not per-seed)
    _nlabels = False
    if loaded_model_path:
        ckpt_re = np.load(loaded_model_path, allow_pickle=True)
        if "normalize_labels" in ckpt_re.files:
            _nlabels = bool(int(ckpt_re["normalize_labels"]))
        else:
            # Check sidecar meta file for poly/mlp/lgbm
            meta_path = loaded_model_path + ".meta.npz"
            if Path(meta_path).exists():
                meta = np.load(meta_path)
                _nlabels = bool(int(meta["normalize_labels"])) if "normalize_labels" in meta.files else False
    else:
        _nlabels = use_normalize_labels

    for eval_seed in eval_seeds:
        rng = np.random.default_rng(eval_seed)
        p, prices = build_instance(
            rng=rng, D=eval_D, N=eval_N, pmax=eval_pmax
        )
        T = int(len(prices))

        t0 = time.perf_counter()
        exact = solve_optimal_benchmark_dp(p, prices, tie_break="early")
        exact_s = time.perf_counter() - t0
        if not exact.feasible:
            print(f"[pool] seed={eval_seed} INFEASIBLE, skipping")
            continue

        lengths, totals, radices, _mult = encode_setup(p)
        ctx = build_tou_feature_context(prices, H=20, validate_repeating=True)

        # Build vhat closure for this instance using the SHARED model
        used_cache: Dict[int, Tuple[int, ...]] = {0: tuple([0] * len(lengths))}

        # Precompute prefix prices for label denormalization
        prefix_prices = np.concatenate(
            [[0.0], np.cumsum(prices, dtype=np.float64)]
        )

        def _make_vhat(model_ref, totals_ref, lengths_ref, ctx_ref, radices_ref, cache_ref, T_ref, prefix_ref, nlabels):
            """Create a vhat closure. Needed to capture loop variables correctly."""
            def vhat(t: int, state: int) -> float:
                s = int(state)
                used_cached = cache_ref.get(s)
                if used_cached is None:
                    used_cached = decode_state(s, radices_ref)
                    cache_ref[s] = used_cached
                val = model_ref.predict_from_used(
                    t=int(t),
                    used=used_cached,
                    totals=totals_ref,
                    lengths=lengths_ref.tolist(),
                    ctx=ctx_ref,
                )
                # Denormalize: model predicts cost/remaining_budget, multiply back
                if nlabels:
                    tt = max(0, min(int(t), T_ref))
                    rem_budget = float(prefix_ref[T_ref] - prefix_ref[tt])
                    val = val * rem_budget
                return val
            return vhat

        vhat = _make_vhat(model, totals, lengths, ctx, radices, used_cache, T, prefix_prices, _nlabels)

        # Price heuristic
        def _make_vhat_price(totals_ref, lengths_ref, radices_ref, cache_ref, T_ref, prefix_ref):
            def vhat_price(t: int, state: int) -> float:
                s = int(state)
                used_cached = cache_ref.get(s)
                if used_cached is None:
                    used_cached = decode_state(s, radices_ref)
                    cache_ref[s] = used_cached
                remaining = totals_ref.astype(np.int32) - np.asarray(
                    used_cached, dtype=np.int32
                )
                W = int(np.sum(remaining * lengths_ref))
                if W <= 0:
                    return 0.0
                tt = max(0, min(int(t), T_ref))
                rem_len = max(1, T_ref - tt)
                mean_p = float(
                    (prefix_ref[T_ref] - prefix_ref[tt]) / rem_len
                )
                return float(W) * mean_p
            return vhat_price

        vhat_price = _make_vhat_price(
            totals, lengths, radices, used_cache, T, prefix_prices
        )

        for beam in beams:
            t1 = time.perf_counter()
            guided_learned = solve_optimal_benchmark_dp(
                p, prices, tie_break="early",
                guided=True, beam_width=int(beam),
                prune_factor=float(args.prune_factor), vhat=vhat,
            )
            guided_learned_s = time.perf_counter() - t1

            t2 = time.perf_counter()
            guided_zero = solve_optimal_benchmark_dp(
                p, prices, tie_break="early",
                guided=True, beam_width=int(beam),
                prune_factor=float(args.prune_factor),
                vhat=lambda _t, _s: 0.0,
            )
            guided_zero_s = time.perf_counter() - t2

            t3 = time.perf_counter()
            guided_price = solve_optimal_benchmark_dp(
                p, prices, tie_break="early",
                guided=True, beam_width=int(beam),
                prune_factor=float(args.prune_factor), vhat=vhat_price,
            )
            guided_price_s = time.perf_counter() - t3

            gap_learned = (
                (guided_learned.cost - exact.cost)
                / max(1e-9, abs(exact.cost))
                * 100.0
            )
            gap_zero = (
                (guided_zero.cost - exact.cost)
                / max(1e-9, abs(exact.cost))
                * 100.0
            )
            gap_price = (
                (guided_price.cost - exact.cost)
                / max(1e-9, abs(exact.cost))
                * 100.0
            )

            row = {
                "seed": float(eval_seed),
                "T": float(T),
                "N": float(len(p)),
                "K": float(len(lengths)),
                "exact_cost": float(exact.cost),
                "exact_s": float(exact_s),
                "train_s": float(train_s),
                "guided_learned_cost": float(guided_learned.cost),
                "guided_learned_s": float(guided_learned_s),
                "guided_zero_cost": float(guided_zero.cost),
                "guided_zero_s": float(guided_zero_s),
                "guided_price_cost": float(guided_price.cost),
                "guided_price_s": float(guided_price_s),
                "gap_learned_pct": float(gap_learned),
                "gap_zero_pct": float(gap_zero),
                "gap_price_pct": float(gap_price),
                "speedup_learned": float(
                    exact_s / max(guided_learned_s, 1e-12)
                ),
                "speedup_zero": float(exact_s / max(guided_zero_s, 1e-12)),
                "speedup_learned_incl_train": float(
                    exact_s / max(train_s + guided_learned_s, 1e-12)
                ),
                "beam": float(int(beam)),
            }
            rows.append(row)

            print(
                f"seed={eval_seed} beam={beam} "
                f"exact={row['exact_cost']:.1f} "
                f"L={row['guided_learned_cost']:.1f} "
                f"Z={row['guided_zero_cost']:.1f} "
                f"P={row['guided_price_cost']:.1f} "
                f"gapL={row['gap_learned_pct']:.2f}% "
                f"gapZ={row['gap_zero_pct']:.2f}% "
                f"gapP={row['gap_price_pct']:.2f}% "
                f"t_exact={exact_s:.3f}s "
                f"tL={guided_learned_s:.3f}s"
            )

    if not rows:
        raise RuntimeError("No successful evaluation rows.")

    # Save CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Summary
    print(f"\n[pool] === SUMMARY ===")
    for beam in sorted(set(int(r["beam"]) for r in rows)):
        br = [r for r in rows if int(r["beam"]) == int(beam)]
        gl = [r["gap_learned_pct"] for r in br]
        gz = [r["gap_zero_pct"] for r in br]
        gp = [r["gap_price_pct"] for r in br]
        sl = [r["speedup_learned"] for r in br]
        print(
            f"beam={beam:3d}  n={len(br):3d}  "
            f"gapL={mean(gl):6.2f}%/{median(gl):6.2f}%  "
            f"gapZ={mean(gz):6.2f}%/{median(gz):6.2f}%  "
            f"gapP={mean(gp):6.2f}%/{median(gp):6.2f}%  "
            f"speedL={mean(sl):.2f}x"
        )

    all_gl = [r["gap_learned_pct"] for r in rows]
    all_gz = [r["gap_zero_pct"] for r in rows]
    all_gp = [r["gap_price_pct"] for r in rows]
    print(
        f"\n[pool] Overall: "
        f"gapL={mean(all_gl):.2f}%  gapZ={mean(all_gz):.2f}%  gapP={mean(all_gp):.2f}%"
    )
    print(f"[pool] Train time: {train_s:.1f}s (amortized over {len(eval_seeds)} eval instances)")
    print(f"[pool] CSV: {out_csv}")


if __name__ == "__main__":
    main()

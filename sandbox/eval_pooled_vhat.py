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

_REPO_PARENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_PARENT))

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
ValueModel = Union[
    LinearRidgeValueModel, PolyRidgeValueModel, MLPValueModel, LGBMValueModel
]


def _make_generate_data_daily_prices(
    *,
    seed: int,
    T: int = 20,
    Tk_choices: Sequence[int] = (2, 3, 5),
    ck_low: int = 1,
    ck_high: int = 8,
) -> List[float]:
    """Generate a length-T daily price vector using generate_data.py-style intervals.

    Keeps the overall "20 hours repeating" structure by generating a single
    20-slot day and repeating it for D days.
    """
    import random

    if T <= 0:
        raise ValueError("T must be positive")
    if ck_low > ck_high:
        raise ValueError("ck_low must be <= ck_high")

    rng = random.Random(int(seed))

    while True:
        remaining = int(T)
        Tk: List[int] = []
        while remaining > 0:
            feasible = [int(x) for x in Tk_choices if int(x) <= remaining]
            if not feasible:
                break
            dur = int(rng.choice(feasible))
            Tk.append(dur)
            remaining -= dur
        if remaining == 0 and Tk:
            break

    ck = [int(rng.randint(int(ck_low), int(ck_high))) for _ in range(len(Tk))]
    ct: List[float] = []
    for dur, price in zip(Tk, ck):
        ct.extend([float(price)] * int(dur))
    if len(ct) != int(T):
        raise RuntimeError("Internal error: generated daily profile has wrong length")
    return ct


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
    dp_time_limit: float = -1.0,
    dp_max_states: int = 0,
    require_optimal_labels: bool = False,
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
    cache: Dict[Tuple[int, Tuple[int, ...]], float] = {}

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
            key = (int(t), tuple(int(x) for x in used))
            cached = cache.get(key)
            if cached is not None:
                y = float(cached)
            else:
                sub = solve_optimal_benchmark_dp(
                    rem_p,
                    prices[t:],
                    tie_break="cost",
                    time_limit=(
                        float(dp_time_limit) if float(dp_time_limit) > 0.0 else -1.0
                    ),
                    max_states=int(dp_max_states),
                )
                if not sub.feasible:
                    continue
                if bool(require_optimal_labels) and not bool(sub.is_optimal):
                    continue
                y = float(sub.cost)
                cache[key] = float(y)

        # Normalize label by remaining price budget
        if normalize_labels and prefix_prices is not None:
            rem_budget = float(prefix_prices[T] - prefix_prices[t])
            if rem_budget > 1e-9:
                y = y / rem_budget

        states.append((t, tuple(int(x) for x in used)))
        labels.append(y)

    return states, labels, attempts


def _collect_state_labels_optimal_path(
    *,
    p_list: Sequence[int],
    prices: np.ndarray,
    totals: np.ndarray,
    lengths: np.ndarray,
    target_samples: int,
    normalize_labels: bool = False,
    prefix_prices: np.ndarray | None = None,
    dp_time_limit: float = -1.0,
    dp_max_states: int = 0,
    n_paths: int = 1,
    require_optimal_labels: bool = False,
) -> Tuple[List[Tuple[int, Tuple[int, ...]]], List[float], int]:
    """Collect exact labels using ONE exact DP solve, labeling states on an optimal path.

    This avoids running an exact DP sub-solve per sampled state (which becomes
    extremely expensive for medium/large instances and can OOM/hang).

    We solve the full instance once, then walk the optimal schedule as a shortest
    path on the DAG. Any suffix of an optimal path is optimal from the intermediate
    node, so the remaining cost along that schedule equals the exact cost-to-go
    for those (t, used) states.

    Returns up to O(T) labeled states (decision times), or fewer if T is small.
    """
    prices = np.asarray(prices, dtype=np.float64)
    T = int(len(prices))
    if T <= 0:
        return [], [], 0

    if prefix_prices is None:
        prefix_prices = np.concatenate([[0.0], np.cumsum(prices, dtype=np.float64)])

    n_paths_i = int(n_paths)
    if n_paths_i not in (1, 2):
        raise ValueError("n_paths must be 1 or 2")

    # We can cheaply get more labels by extracting states from multiple
    # equal-cost optimal schedules (different tie-breaks).
    tie_breaks = ["cost"]
    if n_paths_i >= 2:
        tie_breaks.append("early")

    # Map length -> class index
    length_to_idx = {int(L): i for i, L in enumerate(lengths.tolist())}
    K = int(len(lengths))

    # Deduplicate by state; keep first label (they should match anyway).
    state_to_label: Dict[Tuple[int, Tuple[int, ...]], float] = {}
    attempts_total = 0

    for tb in tie_breaks:
        try:
            exact = solve_optimal_benchmark_dp(
                p_list,
                prices,
                tie_break=str(tb),
                time_limit=(
                    float(dp_time_limit) if float(dp_time_limit) > 0.0 else -1.0
                ),
                max_states=int(dp_max_states),
            )
        except MemoryError:
            # Some seeds can trigger extreme DP growth; skip rather than crashing.
            continue
        except Exception:
            continue
        if not exact.feasible:
            continue
        if bool(require_optimal_labels) and not bool(exact.is_optimal):
            continue

        # If we didn't get a full schedule (e.g., reconstruction failed), skip.
        if len(exact.schedule) < len(p_list):
            continue

        # Map job-start time -> length for the optimal schedule.
        start_to_len: Dict[int, int] = {}
        for _jid, s, e in exact.schedule:
            start_to_len[int(s)] = int(e) - int(s)

        used = [0] * K
        t = 0
        cost_so_far = 0.0

        while t < T:
            y = float(exact.cost - cost_so_far)
            if normalize_labels:
                rem_budget = float(prefix_prices[T] - prefix_prices[t])
                if rem_budget > 1e-9:
                    y = y / rem_budget
            key = (int(t), tuple(int(x) for x in used))
            if key not in state_to_label:
                state_to_label[key] = float(y)

            L = start_to_len.get(int(t), 0)
            if int(L) <= 0:
                t += 1
                continue

            end = int(t) + int(L)
            if end > T:
                break
            cost_so_far += float(prefix_prices[end] - prefix_prices[int(t)])
            idx = length_to_idx.get(int(L))
            if idx is not None:
                used[idx] += 1
            t = end

        attempts_total += int(len(state_to_label))

    if not state_to_label:
        return [], [], 0

    # Preserve a stable ordering by time.
    items = sorted(state_to_label.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    states = [k for (k, _v) in items]
    labels = [float(v) for (_k, v) in items]

    # Respect target_samples if it is smaller than what we generated.
    if int(target_samples) > 0 and len(states) > int(target_samples):
        states = states[: int(target_samples)]
        labels = labels[: int(target_samples)]

    return states, labels, int(len(states))


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    var = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if var <= 1e-12:
        return float("nan")
    sse = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - sse / var


def _poly_powers_degree2(d_in: int) -> np.ndarray:
    """Create sklearn-compatible powers_ for degree-2 PolynomialFeatures.

    Duplicated here so we can do streaming poly ridge without materializing
    the full expanded matrix.
    """
    d = int(d_in)
    powers: List[np.ndarray] = []
    powers.append(np.zeros(d, dtype=np.int32))
    for i in range(d):
        e = np.zeros(d, dtype=np.int32)
        e[i] = 1
        powers.append(e)
    for i in range(d):
        e2 = np.zeros(d, dtype=np.int32)
        e2[i] = 2
        powers.append(e2)
        for j in range(i + 1, d):
            e = np.zeros(d, dtype=np.int32)
            e[i] = 1
            e[j] = 1
            powers.append(e)
    return np.stack(powers, axis=0)


def _split_train_mask(
    indices: np.ndarray, *, train_frac: float = 0.85, seed: int = 42
) -> np.ndarray:
    """Deterministic split without storing/shuffling full index arrays."""
    # Hash-like mapping to [0,1000)
    idx = np.asarray(indices, dtype=np.uint64)
    x = (idx * np.uint64(11400714819323198485) + np.uint64(seed)) % np.uint64(1000)
    thr = int(round(float(train_frac) * 1000.0))
    return x < np.uint64(thr)


def _stream_fit_ridge(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float,
    chunk_size: int,
    train_frac: float = 0.85,
    split_seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Closed-form ridge fit using chunked passes over X/y."""
    n = int(y.shape[0])
    d = int(X.shape[1])
    A = np.zeros((d, d), dtype=np.float64)
    b = np.zeros((d,), dtype=np.float64)

    for start in range(0, n, int(chunk_size)):
        end = min(n, start + int(chunk_size))
        idx = np.arange(start, end, dtype=np.int64)
        train_mask = _split_train_mask(
            idx, train_frac=float(train_frac), seed=int(split_seed)
        )
        if not bool(np.any(train_mask)):
            continue
        Xc = np.asarray(X[start:end], dtype=np.float64)
        yc = np.asarray(y[start:end], dtype=np.float64)
        Xt = Xc[train_mask]
        yt = yc[train_mask]
        A += Xt.T @ Xt
        b += Xt.T @ yt

    A += float(l2) * np.eye(d, dtype=np.float64)
    w = np.linalg.solve(A, b).astype(np.float64)

    # Second pass for metrics
    stats = {
        "train_n": 0.0,
        "test_n": 0.0,
        "train_sum": 0.0,
        "train_sum2": 0.0,
        "test_sum": 0.0,
        "test_sum2": 0.0,
        "train_sse": 0.0,
        "test_sse": 0.0,
        "train_mae": 0.0,
        "test_mae": 0.0,
    }

    for start in range(0, n, int(chunk_size)):
        end = min(n, start + int(chunk_size))
        idx = np.arange(start, end, dtype=np.int64)
        train_mask = _split_train_mask(
            idx, train_frac=float(train_frac), seed=int(split_seed)
        )
        Xc = np.asarray(X[start:end], dtype=np.float64)
        yc = np.asarray(y[start:end], dtype=np.float64)
        yhat = Xc @ w

        for is_train, mask in ((True, train_mask), (False, ~train_mask)):
            if not bool(np.any(mask)):
                continue
            yy = yc[mask]
            yh = yhat[mask]
            nmask = float(yy.shape[0])
            if is_train:
                stats["train_n"] += nmask
                stats["train_sum"] += float(np.sum(yy))
                stats["train_sum2"] += float(np.sum(yy * yy))
                stats["train_sse"] += float(np.sum((yy - yh) ** 2))
                stats["train_mae"] += float(np.sum(np.abs(yy - yh)))
            else:
                stats["test_n"] += nmask
                stats["test_sum"] += float(np.sum(yy))
                stats["test_sum2"] += float(np.sum(yy * yy))
                stats["test_sse"] += float(np.sum((yy - yh) ** 2))
                stats["test_mae"] += float(np.sum(np.abs(yy - yh)))

    def _r2_from(sum_y: float, sum_y2: float, sse: float, n_: float) -> float:
        if n_ <= 1.0:
            return float("nan")
        var = float(sum_y2 - (sum_y * sum_y) / max(n_, 1.0))
        if var <= 1e-12:
            return float("nan")
        return 1.0 - float(sse) / var

    metrics = {
        "r2_train": float(
            _r2_from(
                stats["train_sum"],
                stats["train_sum2"],
                stats["train_sse"],
                stats["train_n"],
            )
        ),
        "mae_train": float(stats["train_mae"] / max(stats["train_n"], 1.0)),
        "r2_test": float(
            _r2_from(
                stats["test_sum"],
                stats["test_sum2"],
                stats["test_sse"],
                stats["test_n"],
            )
        ),
        "mae_test": float(stats["test_mae"] / max(stats["test_n"], 1.0)),
        "train_n": float(stats["train_n"]),
        "test_n": float(stats["test_n"]),
    }
    return w, metrics


def _stream_fit_poly_ridge(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float,
    chunk_size: int,
    train_frac: float = 0.85,
    split_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Degree-2 polynomial ridge using chunked passes."""
    n = int(y.shape[0])
    d_in = int(X.shape[1])
    powers = _poly_powers_degree2(d_in)
    d_out = int(powers.shape[0])
    A = np.zeros((d_out, d_out), dtype=np.float64)
    b = np.zeros((d_out,), dtype=np.float64)

    for start in range(0, n, int(chunk_size)):
        end = min(n, start + int(chunk_size))
        idx = np.arange(start, end, dtype=np.int64)
        train_mask = _split_train_mask(
            idx, train_frac=float(train_frac), seed=int(split_seed)
        )
        if not bool(np.any(train_mask)):
            continue
        Xc = np.asarray(X[start:end], dtype=np.float64)
        yc = np.asarray(y[start:end], dtype=np.float64)
        Xt = Xc[train_mask]
        yt = yc[train_mask]
        Xt_poly = _poly_expand_batch(Xt, powers)
        A += Xt_poly.T @ Xt_poly
        b += Xt_poly.T @ yt

    A += float(l2) * np.eye(d_out, dtype=np.float64)
    w = np.linalg.solve(A, b).astype(np.float64)

    # Second pass metrics
    stats = {
        "train_n": 0.0,
        "test_n": 0.0,
        "train_sum": 0.0,
        "train_sum2": 0.0,
        "test_sum": 0.0,
        "test_sum2": 0.0,
        "train_sse": 0.0,
        "test_sse": 0.0,
        "train_mae": 0.0,
        "test_mae": 0.0,
    }

    for start in range(0, n, int(chunk_size)):
        end = min(n, start + int(chunk_size))
        idx = np.arange(start, end, dtype=np.int64)
        train_mask = _split_train_mask(
            idx, train_frac=float(train_frac), seed=int(split_seed)
        )
        Xc = np.asarray(X[start:end], dtype=np.float64)
        yc = np.asarray(y[start:end], dtype=np.float64)
        Xc_poly = _poly_expand_batch(Xc, powers)
        yhat = Xc_poly @ w

        for is_train, mask in ((True, train_mask), (False, ~train_mask)):
            if not bool(np.any(mask)):
                continue
            yy = yc[mask]
            yh = yhat[mask]
            nmask = float(yy.shape[0])
            if is_train:
                stats["train_n"] += nmask
                stats["train_sum"] += float(np.sum(yy))
                stats["train_sum2"] += float(np.sum(yy * yy))
                stats["train_sse"] += float(np.sum((yy - yh) ** 2))
                stats["train_mae"] += float(np.sum(np.abs(yy - yh)))
            else:
                stats["test_n"] += nmask
                stats["test_sum"] += float(np.sum(yy))
                stats["test_sum2"] += float(np.sum(yy * yy))
                stats["test_sse"] += float(np.sum((yy - yh) ** 2))
                stats["test_mae"] += float(np.sum(np.abs(yy - yh)))

    def _r2_from(sum_y: float, sum_y2: float, sse: float, n_: float) -> float:
        if n_ <= 1.0:
            return float("nan")
        var = float(sum_y2 - (sum_y * sum_y) / max(n_, 1.0))
        if var <= 1e-12:
            return float("nan")
        return 1.0 - float(sse) / var

    metrics = {
        "r2_train": float(
            _r2_from(
                stats["train_sum"],
                stats["train_sum2"],
                stats["train_sse"],
                stats["train_n"],
            )
        ),
        "mae_train": float(stats["train_mae"] / max(stats["train_n"], 1.0)),
        "r2_test": float(
            _r2_from(
                stats["test_sum"],
                stats["test_sum2"],
                stats["test_sse"],
                stats["test_n"],
            )
        ),
        "mae_test": float(stats["test_mae"] / max(stats["test_n"], 1.0)),
        "train_n": float(stats["train_n"]),
        "test_n": float(stats["test_n"]),
        "feat_dim_poly": float(d_out),
    }
    return w, powers, metrics


def parse_seed_range(s: str) -> List[int]:
    """Parse '0-19' or '0,1,5,10' into list of ints."""
    s = s.strip()
    if "-" in s and "," not in s:
        parts = s.split("-")
        return list(range(int(parts[0]), int(parts[1]) + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def parse_int_range(s: str) -> Tuple[int, int]:
    """Parse an inclusive integer range 'a-b'."""
    s = str(s).strip()
    if not s or "-" not in s:
        raise ValueError(f"Expected range like 'a-b', got: {s!r}")
    a_s, b_s = s.split("-", 1)
    a = int(a_s)
    b = int(b_s)
    if b < a:
        raise ValueError(f"Invalid range (b<a): {s!r}")
    return a, b


def _sample_D_N_feasible(
    *,
    rng: np.random.Generator,
    base_D: int,
    base_N: int,
    D_range: Tuple[int, int] | None,
    N_range: Tuple[int, int] | None,
    target_util: float,
    H: int = 20,
    max_tries: int = 64,
) -> Tuple[int, int]:
    """Sample (D,N) while ensuring single-machine feasibility.

    For a single machine with 1-hour slots:
    - Need N <= T where T=H*D (since p_j>=1).
    - If target_util>0 and we enforce sum(p)<=floor(target_util*T), also need
      N <= floor(target_util*T).

    If constraints make the requested (D_range,N_range,target_util) impossible
    (e.g. D in [2,4], N in [60,60], target_util=0.8), we fall back by clamping N
    to the largest feasible value for the sampled D.
    """
    D0 = int(base_D)
    N0 = int(base_N)
    tu = float(target_util)
    tu = tu if tu > 0.0 else 0.0

    # Determine N bounds
    if N_range is None:
        N_lo, N_hi = N0, N0
    else:
        N_lo, N_hi = int(N_range[0]), int(N_range[1])
    if N_lo > N_hi:
        N_lo, N_hi = N_hi, N_lo

    # Determine D bounds
    if D_range is None:
        D_lo, D_hi = D0, D0
    else:
        D_lo, D_hi = int(D_range[0]), int(D_range[1])
    if D_lo > D_hi:
        D_lo, D_hi = D_hi, D_lo

    for _ in range(int(max_tries)):
        D_use = int(rng.integers(int(D_lo), int(D_hi) + 1))
        T = int(H) * int(D_use)
        capN = int(T)
        if tu > 0.0:
            capN = min(capN, int(np.floor(tu * float(T))))
        feasible_hi = min(int(N_hi), int(capN))
        if feasible_hi >= int(N_lo):
            N_use = int(rng.integers(int(N_lo), int(feasible_hi) + 1))
            return int(D_use), int(N_use)

    # Fallback: pick D deterministically and clamp N.
    D_use = int(D0 if D_range is None else D_hi)
    T = int(H) * int(D_use)
    capN = int(T)
    if tu > 0.0:
        capN = min(capN, int(np.floor(tu * float(T))))
    N_use = int(min(max(int(N_lo), 1), max(int(capN), 1)))
    return int(D_use), int(N_use)


def _collect_worker(
    seed: int,
    D: int,
    N: int,
    pmax: int,
    D_range: Tuple[int, int] | None,
    N_range: Tuple[int, int] | None,
    samples_per_instance: int,
    spec: FeatureSpec,
    normalize_labels: bool,
    daily_prices_20: List[float] | None,
    target_util: float,
    x_dtype: str,
    label_mode: str,
    dp_time_limit: float,
    dp_max_states: int,
    optimal_path_n_paths: int,
    optimal_path_topup_max: int,
    optimal_path_topup_dp_time_limit: float,
    require_optimal_labels: bool,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Worker function for parallel data collection. Runs in a subprocess."""
    attempts = 0
    try:
        rng = np.random.default_rng(int(seed))

        D_use, N_use = _sample_D_N_feasible(
            rng=rng,
            base_D=int(D),
            base_N=int(N),
            D_range=D_range,
            N_range=N_range,
            target_util=float(target_util),
            H=20,
        )

        tu = float(target_util)
        p, prices = build_instance(
            rng=rng,
            D=int(D_use),
            N=int(N_use),
            pmax=int(pmax),
            daily_prices_20=daily_prices_20,
            target_util=(tu if tu > 0.0 else None),
            M_for_cap=1,
        )

        T = int(len(prices))
        lengths, totals, radices, _mult = encode_setup(p)
        ctx = build_tou_feature_context(prices, H=20, validate_repeating=True)
        prefix_prices = np.concatenate([[0.0], np.cumsum(prices, dtype=np.float64)])

        lm = str(label_mode).strip().lower()
        if lm == "optimal_path":
            states, labels, attempts = _collect_state_labels_optimal_path(
                p_list=p,
                prices=prices,
                totals=totals,
                lengths=lengths,
                target_samples=int(samples_per_instance),
                normalize_labels=bool(normalize_labels),
                prefix_prices=prefix_prices,
                dp_time_limit=float(dp_time_limit),
                dp_max_states=int(dp_max_states),
                n_paths=int(optimal_path_n_paths),
                require_optimal_labels=bool(require_optimal_labels),
            )

            # Optional top-up beyond O(T)
            need = int(samples_per_instance) - int(len(states))
            if need > 0:
                cap = int(optimal_path_topup_max)
                if cap == 0:
                    topup_target = 0
                elif cap < 0:
                    topup_target = int(need)
                else:
                    topup_target = int(min(int(need), int(cap)))

                if topup_target > 0:
                    sp_states, sp_labels, sp_attempts = _collect_state_labels(
                        rng=rng,
                        T=T,
                        totals=totals,
                        lengths=lengths,
                        prices=prices,
                        target_samples=topup_target,
                        normalize_labels=bool(normalize_labels),
                        prefix_prices=prefix_prices,
                        dp_time_limit=float(optimal_path_topup_dp_time_limit),
                        dp_max_states=int(dp_max_states),
                        require_optimal_labels=bool(require_optimal_labels),
                    )
                    attempts += int(sp_attempts)
                    if sp_states:
                        seen = set(states)
                        for s, y in zip(sp_states, sp_labels):
                            if s in seen:
                                continue
                            seen.add(s)
                            states.append(s)
                            labels.append(float(y))

            # Fallback (only if top-up enabled)
            if not states and int(optimal_path_topup_max) != 0:
                sp = min(int(samples_per_instance), 64)
                states, labels, attempts = _collect_state_labels(
                    rng=rng,
                    T=T,
                    totals=totals,
                    lengths=lengths,
                    prices=prices,
                    target_samples=sp,
                    normalize_labels=bool(normalize_labels),
                    prefix_prices=prefix_prices,
                    dp_time_limit=float(optimal_path_topup_dp_time_limit),
                    dp_max_states=int(dp_max_states),
                    require_optimal_labels=bool(require_optimal_labels),
                )

        elif lm == "subproblem":
            states, labels, attempts = _collect_state_labels(
                rng=rng,
                T=T,
                totals=totals,
                lengths=lengths,
                prices=prices,
                target_samples=int(samples_per_instance),
                normalize_labels=bool(normalize_labels),
                prefix_prices=prefix_prices,
                dp_time_limit=float(dp_time_limit),
                dp_max_states=int(dp_max_states),
                require_optimal_labels=bool(require_optimal_labels),
            )
        else:
            raise ValueError(f"Unknown label_mode: {label_mode!r}")

        if not states:
            return np.empty((0, 0)), np.empty(0), int(attempts)

        # Avoid list-of-arrays + vstack (high peak RAM). Preallocate and fill.
        dtype = np.dtype(str(x_dtype))
        t0, used0 = states[0]
        phi0 = phi_for_state(
            t=int(t0),
            used=used0,
            totals=totals,
            lengths=lengths.tolist(),
            ctx=ctx,
            spec=spec,
        )
        feat_dim = int(phi0.shape[0])
        X_inst = np.empty((len(states), feat_dim), dtype=dtype)
        X_inst[0, :] = phi0.astype(dtype, copy=False)
        for i in range(1, len(states)):
            t_v, used_v = states[i]
            phi_v = phi_for_state(
                t=int(t_v),
                used=used_v,
                totals=totals,
                lengths=lengths.tolist(),
                ctx=ctx,
                spec=spec,
            )
            X_inst[i, :] = phi_v.astype(dtype, copy=False)
        y_inst = np.asarray(labels, dtype=np.float64)
        return X_inst, y_inst, int(attempts)

    except MemoryError:
        return np.empty((0, 0)), np.empty(0), int(attempts)
    except Exception:
        return np.empty((0, 0)), np.empty(0), int(attempts)


def _mlp_all_projection_indices(
    *,
    pmax: int,
    target_variant: str,
) -> np.ndarray:
    """Return column indices to project mlp_all pooled features to a target variant.

    This projection is valid for the transferable-features setup used by the
    new MLP variants script (fixed-dimension features: no per-class blocks).
    """
    pmax_h = int(max(1, int(pmax)))
    variant = str(target_variant).strip().lower()

    pre_meta = np.arange(0, 10, dtype=np.int64)
    meta = np.arange(10, 15, dtype=np.int64)
    post_meta = np.arange(15, 23, dtype=np.int64)
    hist = np.arange(23, 23 + pmax_h, dtype=np.int64)
    price = np.arange(23 + pmax_h, 23 + pmax_h + 10, dtype=np.int64)

    if variant == "mlp_all":
        return np.arange(0, 23 + pmax_h + 10, dtype=np.int64)
    if variant == "mlp_hist":
        return np.concatenate([pre_meta, post_meta, hist])
    if variant == "mlp_price":
        return np.concatenate([pre_meta, post_meta, price])
    if variant == "mlp_meta":
        return np.concatenate([pre_meta, meta, post_meta])

    raise ValueError(f"Unsupported target variant for projection: {target_variant}")


def _load_or_project_pooled_data(
    *,
    load_path: str,
    requested_model_type: str,
    pmax: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load pooled X/y and optionally project mlp_all features to another variant."""
    z = np.load(load_path, allow_pickle=False)
    if "X_pool" not in z.files or "y_pool" not in z.files:
        raise ValueError(
            f"Invalid pooled data file (missing X_pool/y_pool): {load_path}"
        )

    X_pool = np.asarray(z["X_pool"])
    y_pool = np.asarray(z["y_pool"], dtype=np.float64)
    src_variant = str(z["model_variant"]) if "model_variant" in z.files else ""

    tgt_variant = str(requested_model_type).strip().lower()
    if src_variant == "mlp_all" and tgt_variant in {
        "mlp_hist",
        "mlp_price",
        "mlp_meta",
        "mlp_all",
    }:
        idx = _mlp_all_projection_indices(pmax=int(pmax), target_variant=tgt_variant)
        if int(np.max(idx)) >= int(X_pool.shape[1]):
            raise ValueError(
                "Pooled mlp_all cache does not match expected feature dimension for projection. "
                f"X columns={X_pool.shape[1]}, max required index={int(np.max(idx))}."
            )
        X_pool = X_pool[:, idx]

    return X_pool, y_pool


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pooled cross-instance training: train ONE model on multiple instances, evaluate on held-out instances."
    )
    # Instance parameters
    ap.add_argument("--D", type=int, default=6)
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--pmax", type=int, default=3)

    ap.add_argument(
        "--daily-price-profile",
        type=str,
        default="daily_tou",
        choices=["daily_tou", "generate_data"],
        help=(
            "Which 20-hour repeating daily profile to use. "
            "daily_tou matches New Benchmark/new_data.py daily_tou; "
            "generate_data samples a 20-slot day using interval prices and repeats it."
        ),
    )
    ap.add_argument(
        "--gd-seed",
        type=int,
        default=20260109,
        help="Seed for --daily-price-profile=generate_data.",
    )
    ap.add_argument("--gd-ck-low", type=int, default=1)
    ap.add_argument("--gd-ck-high", type=int, default=8)

    ap.add_argument(
        "--target-util",
        type=float,
        default=0.0,
        help=(
            "If >0, enforce New Benchmark/new_data.py-style utilization cap when sampling p: "
            "sum(p) <= floor(target_util * 1 * T). If 0, only enforce feasibility sum(p)<=T."
        ),
    )

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

    ap.add_argument(
        "--label-mode",
        type=str,
        default="subproblem",
        choices=["subproblem", "optimal_path"],
        help=(
            "How to generate exact labels for training states. "
            "subproblem: sample random states and solve an exact DP subproblem per state (very expensive for medium/large). "
            "optimal_path: solve the full instance once and label states along an optimal schedule (exact cost-to-go, O(T) labels)."
        ),
    )

    ap.add_argument(
        "--optimal-path-n-paths",
        type=int,
        default=1,
        choices=[1, 2],
        help=(
            "For --label-mode=optimal_path: number of optimal schedules to extract labels from. "
            "Uses different DP tie-breaks (cost vs early). 2 roughly doubles labels per instance with only one extra DP solve."
        ),
    )

    ap.add_argument(
        "--optimal-path-topup-max",
        type=int,
        default=0,
        help=(
            "For --label-mode=optimal_path: maximum number of additional random subproblem labels to add per instance. "
            "This is used to top up beyond the O(T) optimal-path labels. "
            "0 disables top-up entirely (recommended for stability). "
            "-1 means unlimited top-up (will try to fill to --samples-per-instance)."
        ),
    )

    ap.add_argument(
        "--optimal-path-topup-dp-time-limit",
        type=float,
        default=0.2,
        help=(
            "For --label-mode=optimal_path: per-call time limit (seconds) for DP subproblem solves used in top-up labeling. "
            "Kept small so you can safely generate lots of extra labels without stalling."
        ),
    )

    ap.add_argument(
        "--require-optimal-labels",
        action="store_true",
        help=(
            "If set, ONLY accept labels from DP runs proven optimal (DPResult.is_optimal=True). "
            "This rejects any greedy-completed timeout results when --dp-time-limit>0 or "
            "--optimal-path-topup-dp-time-limit>0. To guarantee all labels are optimal, "
            "set the relevant time limits to <=0 (no limit)."
        ),
    )

    ap.add_argument(
        "--dp-time-limit",
        type=float,
        default=-1.0,
        help=(
            "Time limit (seconds) for exact DP calls during label collection. "
            "If >0, DP may return a greedy-completed solution on timeout (still feasible, but labels become approximate). "
            "Use this on HPC to prevent rare seeds from stalling the whole sweep."
        ),
    )

    ap.add_argument(
        "--dp-max-states",
        type=int,
        default=0,
        help=(
            "Memory guardrail for DP: if >0, abort DP (like timeout) when any single DP layer exceeds this number of states. "
            "Use this to prevent OOM during pooled label collection and evaluation."
        ),
    )

    ap.add_argument(
        "--eval-time-limit",
        type=float,
        default=-1.0,
        help=(
            "Time limit (seconds) for DP calls during EVALUATION (exact baseline + guided runs). "
            "If >0, DP may return a greedy-completed solution on timeout (feasible but not proven optimal). "
            "This prevents pathological eval seeds from stalling sweeps."
        ),
    )

    ap.add_argument(
        "--train-D-range",
        type=str,
        default="",
        help="Optional inclusive range 'a-b'. If set, sample D per training seed.",
    )
    ap.add_argument(
        "--train-N-range",
        type=str,
        default="",
        help="Optional inclusive range 'a-b'. If set, sample N per training seed.",
    )
    ap.add_argument("--l2", type=float, default=1e-3)

    # Model training knobs (defaults match previous behavior)
    ap.add_argument("--mlp-lr", type=float, default=1e-3)
    ap.add_argument("--mlp-batch-size", type=int, default=2048)
    ap.add_argument("--mlp-max-epochs", type=int, default=200)
    ap.add_argument("--mlp-patience", type=int, default=15)

    ap.add_argument("--lgbm-n-estimators", type=int, default=100)
    ap.add_argument("--lgbm-max-depth", type=int, default=5)
    ap.add_argument("--lgbm-learning-rate", type=float, default=0.1)

    # Model type
    ap.add_argument(
        "--model-type",
        type=str,
        default="linear",
        choices=[
            "linear",
            "poly",
            "mlp",
            "lgbm",
            # Fast-inference MLP variants (feature-spec toggles)
            "mlp_hist",
            "mlp_price",
            "mlp_meta",
            "mlp_all",
        ],
        help="Model type: linear (Ridge), poly (degree-2 polynomial Ridge), "
        "mlp (small neural net), lgbm (gradient boosted trees).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers for data collection (0 = auto = cpu_count).",
    )

    ap.add_argument(
        "--pool-on-disk",
        action="store_true",
        help=(
            "Store pooled features/labels in on-disk memmaps while collecting to reduce peak RAM. "
            "Useful on HPC with large worker counts."
        ),
    )
    ap.add_argument(
        "--pool-dir",
        type=str,
        default="",
        help=(
            "Directory to place memmap files when using --pool-on-disk. "
            "Default: create a temporary directory."
        ),
    )
    ap.add_argument(
        "--pool-dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Data type for pooled feature matrix X (float32 saves RAM).",
    )
    ap.add_argument(
        "--maxtasksperchild",
        type=int,
        default=0,
        help=(
            "multiprocessing.Pool maxtasksperchild (0 disables). "
            "Can reduce memory growth in long runs at some overhead."
        ),
    )

    ap.add_argument(
        "--stream-fit",
        action="store_true",
        help=(
            "Fit linear/poly models in a streaming (chunked) way to avoid loading the full pooled dataset into RAM. "
            "Recommended when using --pool-on-disk with large numbers of instances/samples."
        ),
    )
    ap.add_argument(
        "--fit-chunk-size",
        type=int,
        default=200_000,
        help="Chunk size (rows) for --stream-fit passes.",
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

    # Optional feature enrichments (still cheap, fixed-dimension under --transferable-features).
    # These allow using the richer MLP-variant feature sets (hist/price/meta) with
    # *any* model type (linear/poly/mlp/lgbm) while keeping one shared pooled cache.
    ap.add_argument(
        "--feat-len-hist",
        action="store_true",
        help="Include fixed-length histogram of remaining job lengths (1..pmax).",
    )
    ap.add_argument(
        "--feat-price-shape",
        action="store_true",
        help="Include daily price-shape features (stats + low-order Fourier).",
    )
    ap.add_argument(
        "--feat-meta",
        action="store_true",
        help="Include extra meta/log-scale features (log1p(T/N/W), util, slack_ratio).",
    )
    ap.add_argument(
        "--feat-pmax-for-hist",
        type=int,
        default=0,
        help="Override pmax_for_hist used by --feat-len-hist (0 = use --pmax).",
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
    ap.add_argument(
        "--load-pooled-data",
        type=str,
        default="",
        help=(
            "Optional .npz file with pooled arrays (X_pool, y_pool). "
            "If provided, skips DP data collection and trains from cached pooled data."
        ),
    )
    ap.add_argument(
        "--save-pooled-data",
        type=str,
        default="",
        help=(
            "Optional output .npz path to save pooled arrays (X_pool, y_pool) after collection."
        ),
    )

    # Eval for different sizes (optional overrides for eval phase)
    ap.add_argument(
        "--eval-D",
        type=int,
        default=0,
        help="Override D for eval instances (0 = same as --D).",
    )
    ap.add_argument(
        "--eval-N",
        type=int,
        default=0,
        help="Override N for eval instances (0 = same as --N).",
    )
    ap.add_argument(
        "--eval-pmax",
        type=int,
        default=0,
        help="Override pmax for eval instances (0 = same as --pmax).",
    )

    ap.add_argument(
        "--eval-D-range",
        type=str,
        default="",
        help="Optional inclusive range 'a-b'. If set, sample D per eval seed.",
    )
    ap.add_argument(
        "--eval-N-range",
        type=str,
        default="",
        help="Optional inclusive range 'a-b'. If set, sample N per eval seed.",
    )

    ap.add_argument(
        "--out-csv",
        type=str,
        default="PaST/logs/eval_pooled_vhat.csv",
    )
    args = ap.parse_args()

    train_seeds = parse_seed_range(args.train_seeds)
    eval_seeds = parse_seed_range(args.eval_seeds)

    daily_prices_20: List[float] | None = None
    if str(args.daily_price_profile).strip().lower() == "generate_data":
        daily_prices_20 = _make_generate_data_daily_prices(
            seed=int(args.gd_seed),
            T=20,
            Tk_choices=(2, 3, 5),
            ck_low=int(args.gd_ck_low),
            ck_high=int(args.gd_ck_high),
        )

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

    # Apply optional feature enrichments for any model type.
    pmax_for_hist = (
        int(args.feat_pmax_for_hist)
        if int(args.feat_pmax_for_hist) > 0
        else int(args.pmax)
    )
    if bool(args.feat_len_hist) or bool(args.feat_price_shape) or bool(args.feat_meta):
        spec = FeatureSpec(
            include_per_class_counts=spec.include_per_class_counts,
            include_per_class_now_cost=spec.include_per_class_now_cost,
            include_bins=spec.include_bins,
            normalize=spec.normalize,
            include_len_hist=(spec.include_len_hist or bool(args.feat_len_hist)),
            pmax_for_hist=pmax_for_hist,
            include_price_shape=(
                spec.include_price_shape or bool(args.feat_price_shape)
            ),
            include_meta=(spec.include_meta or bool(args.feat_meta)),
        )

    # Eval instance parameters (possibly different from training)
    eval_D = int(args.eval_D) if int(args.eval_D) > 0 else int(args.D)
    eval_N = int(args.eval_N) if int(args.eval_N) > 0 else int(args.N)
    eval_pmax = int(args.eval_pmax) if int(args.eval_pmax) > 0 else int(args.pmax)

    train_D_range = (
        parse_int_range(str(args.train_D_range))
        if str(args.train_D_range).strip()
        else None
    )
    train_N_range = (
        parse_int_range(str(args.train_N_range))
        if str(args.train_N_range).strip()
        else None
    )
    eval_D_range = (
        parse_int_range(str(args.eval_D_range))
        if str(args.eval_D_range).strip()
        else None
    )
    eval_N_range = (
        parse_int_range(str(args.eval_N_range))
        if str(args.eval_N_range).strip()
        else None
    )

    requested_model_type = str(args.model_type).strip().lower()
    model_type = requested_model_type
    # Normalize MLP variants to the MLP trainer while toggling FeatureSpec flags.
    if model_type in {"mlp_hist", "mlp_price", "mlp_meta", "mlp_all"}:
        if model_type == "mlp_hist":
            spec = FeatureSpec(
                include_per_class_counts=spec.include_per_class_counts,
                include_per_class_now_cost=spec.include_per_class_now_cost,
                include_bins=spec.include_bins,
                normalize=spec.normalize,
                include_len_hist=True,
                pmax_for_hist=int(args.pmax),
                include_price_shape=False,
                include_meta=False,
            )
        elif model_type == "mlp_price":
            spec = FeatureSpec(
                include_per_class_counts=spec.include_per_class_counts,
                include_per_class_now_cost=spec.include_per_class_now_cost,
                include_bins=spec.include_bins,
                normalize=spec.normalize,
                include_len_hist=False,
                pmax_for_hist=int(args.pmax),
                include_price_shape=True,
                include_meta=False,
            )
        elif model_type == "mlp_meta":
            spec = FeatureSpec(
                include_per_class_counts=spec.include_per_class_counts,
                include_per_class_now_cost=spec.include_per_class_now_cost,
                include_bins=spec.include_bins,
                normalize=spec.normalize,
                include_len_hist=False,
                pmax_for_hist=int(args.pmax),
                include_price_shape=False,
                include_meta=True,
            )
        else:
            spec = FeatureSpec(
                include_per_class_counts=spec.include_per_class_counts,
                include_per_class_now_cost=spec.include_per_class_now_cost,
                include_bins=spec.include_bins,
                normalize=spec.normalize,
                include_len_hist=True,
                pmax_for_hist=int(args.pmax),
                include_price_shape=True,
                include_meta=True,
            )
        model_type = "mlp"
    n_workers_req = int(args.workers) if int(args.workers) > 0 else mp.cpu_count()
    # Spawning more processes than there are instances wastes RAM (each process
    # imports numpy, solver modules, etc.). Cap at number of training seeds.
    n_workers = min(int(n_workers_req), max(1, len(train_seeds)))

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
                include_len_hist=(
                    bool(int(ckpt["include_len_hist"]))
                    if "include_len_hist" in ckpt.files
                    else False
                ),
                pmax_for_hist=(
                    int(ckpt["pmax_for_hist"])
                    if "pmax_for_hist" in ckpt.files
                    else int(args.pmax)
                ),
                include_price_shape=(
                    bool(int(ckpt["include_price_shape"]))
                    if "include_price_shape" in ckpt.files
                    else False
                ),
                include_meta=(
                    bool(int(ckpt["include_meta"]))
                    if "include_meta" in ckpt.files
                    else False
                ),
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
        train_inst_desc = f"D={args.D}, N={args.N}, pmax={args.pmax}"
        if train_D_range is not None or train_N_range is not None:
            train_inst_desc = (
                f"D={args.D} (range={train_D_range}), "
                f"N={args.N} (range={train_N_range}), "
                f"pmax={args.pmax}"
            )
        print(
            f"[pool] === TRAINING PHASE ==="
            f"\n[pool] Model type: {model_type}"
            f"\n[pool] Instance params: {train_inst_desc}"
            f"\n[pool] Train seeds: {train_seeds[0]}-{train_seeds[-1]} ({len(train_seeds)} instances)"
            f"\n[pool] Samples per instance: {args.samples_per_instance}"
            f"\n[pool] Features: spec={spec}"
            f"\n[pool] normalize_labels={use_normalize_labels}"
            f"\n[pool] label_mode={str(args.label_mode).strip()}  dp_time_limit={float(args.dp_time_limit)}  "
            f"dp_max_states={int(args.dp_max_states)}  "
            f"optimal_path_n_paths={int(args.optimal_path_n_paths)}  "
            f"optimal_path_topup_max={int(args.optimal_path_topup_max)}  "
            f"optimal_path_topup_dp_time_limit={float(args.optimal_path_topup_dp_time_limit)}  "
            f"require_optimal_labels={bool(args.require_optimal_labels)}"
            f"\n[pool] Workers: {n_workers}"
        )

        train_t0 = time.perf_counter()
        load_pooled_data_path = str(args.load_pooled_data).strip()
        if load_pooled_data_path:
            print(f"[pool] Loading pooled dataset from {load_pooled_data_path}")
            X_pool, y_pool = _load_or_project_pooled_data(
                load_path=load_pooled_data_path,
                requested_model_type=requested_model_type,
                pmax=int(args.pmax),
            )
            total_attempts = 0
            collect_time = 0.0
            print(
                f"[pool] Loaded pooled data: X={tuple(X_pool.shape)}, y={tuple(y_pool.shape)}"
            )
        else:
            # === Parallel data collection ===
            worker_fn = partial(
                _collect_worker,
                D=int(args.D),
                N=int(args.N),
                pmax=int(args.pmax),
                D_range=train_D_range,
                N_range=train_N_range,
                samples_per_instance=int(args.samples_per_instance),
                spec=spec,
                normalize_labels=use_normalize_labels,
                daily_prices_20=daily_prices_20,
                target_util=float(args.target_util),
                x_dtype=str(args.pool_dtype),
                label_mode=str(args.label_mode),
                dp_time_limit=float(args.dp_time_limit),
                dp_max_states=int(args.dp_max_states),
                optimal_path_n_paths=int(args.optimal_path_n_paths),
                optimal_path_topup_max=int(args.optimal_path_topup_max),
                optimal_path_topup_dp_time_limit=float(
                    args.optimal_path_topup_dp_time_limit
                ),
                require_optimal_labels=bool(args.require_optimal_labels),
            )

            mtpc = int(args.maxtasksperchild)
            mtpc = None if mtpc <= 0 else mtpc

            if bool(args.pool_on_disk):
                import tempfile

                pool_dir = str(args.pool_dir).strip()
                if pool_dir:
                    Path(pool_dir).mkdir(parents=True, exist_ok=True)
                    tmp_dir = tempfile.mkdtemp(prefix="pooled_vhat_", dir=pool_dir)
                else:
                    tmp_dir = tempfile.mkdtemp(prefix="pooled_vhat_")

                expected_per_inst = int(args.samples_per_instance)

                if (
                    str(args.label_mode).strip().lower() == "optimal_path"
                    and int(args.optimal_path_topup_max) == 0
                ):
                    D_hi = (
                        int(train_D_range[1])
                        if train_D_range is not None
                        else int(args.D)
                    )
                    Tmax = int(20 * D_hi)
                    est = int(int(args.optimal_path_n_paths) * Tmax)
                    if est > 0:
                        expected_per_inst = int(min(expected_per_inst, est))

                expected_total = int(len(train_seeds)) * expected_per_inst

                print(f"[pool] memmap: dir={tmp_dir}")
                print("[pool] memmap: probing feature dimension (no DP)...")
                probe_seed = int(train_seeds[0])
                prng = np.random.default_rng(int(probe_seed))
                D_use, N_use = _sample_D_N_feasible(
                    rng=prng,
                    base_D=int(args.D),
                    base_N=int(args.N),
                    D_range=train_D_range,
                    N_range=train_N_range,
                    target_util=float(args.target_util),
                    H=20,
                )
                tu = float(args.target_util)
                p_probe, prices_probe = build_instance(
                    rng=prng,
                    D=int(D_use),
                    N=int(N_use),
                    pmax=int(args.pmax),
                    daily_prices_20=daily_prices_20,
                    target_util=(tu if tu > 0.0 else None),
                    M_for_cap=1,
                )
                lengths_p, totals_p, _rad, _mul = encode_setup(p_probe)
                ctx_p = build_tou_feature_context(
                    prices_probe, H=20, validate_repeating=True
                )
                used0 = tuple([0] * int(len(lengths_p)))
                phi0 = phi_for_state(
                    t=0,
                    used=used0,
                    totals=totals_p,
                    lengths=lengths_p.tolist(),
                    ctx=ctx_p,
                    spec=spec,
                )
                feat_dim = int(phi0.shape[0])
                print(f"[pool] memmap: probe feature_dim={feat_dim}")
                x_dtype = np.dtype(str(args.pool_dtype))
                X_path = os.path.join(tmp_dir, "X_pool.dat")
                y_path = os.path.join(tmp_dir, "y_pool.dat")

                X_mm = np.memmap(
                    X_path, mode="w+", dtype=x_dtype, shape=(expected_total, feat_dim)
                )
                y_mm = np.memmap(
                    y_path, mode="w+", dtype=np.float64, shape=(expected_total,)
                )

                cursor = 0
                total_attempts = 0

                remaining_seeds = train_seeds
                if n_workers > 1 and len(remaining_seeds) > 0:
                    print(
                        f"[pool] Collecting remaining data with {n_workers} parallel workers (memmap)..."
                    )
                    with mp.Pool(processes=n_workers, maxtasksperchild=mtpc) as pool:
                        for i, (X_i, y_i, attempts_i) in enumerate(
                            pool.imap_unordered(worker_fn, remaining_seeds), start=1
                        ):
                            if X_i.size == 0:
                                total_attempts += int(attempts_i)
                                print(
                                    f"[pool] collected {i}/{len(train_seeds)} instances (0 samples)"
                                )
                                continue

                            n_i = int(X_i.shape[0])
                            if cursor + n_i > expected_total:
                                raise RuntimeError(
                                    f"Memmap overflow: cursor={cursor} + n={n_i} > expected_total={expected_total}"
                                )
                            X_mm[cursor : cursor + n_i, :] = X_i
                            y_mm[cursor : cursor + n_i] = y_i
                            cursor += n_i
                            total_attempts += int(attempts_i)
                            print(
                                f"[pool] collected {i}/{len(train_seeds)} instances ({n_i} samples)"
                            )
                else:
                    print("[pool] Collecting remaining data sequentially (memmap)...")
                    for i, seed in enumerate(remaining_seeds, start=1):
                        t_inst = time.perf_counter()
                        X_i, y_i, attempts_i = worker_fn(int(seed))
                        inst_time = time.perf_counter() - t_inst
                        if X_i.size == 0:
                            total_attempts += int(attempts_i)
                            print(
                                f"[pool] train seed={seed} ({i}/{len(train_seeds)}) samples=0 time={inst_time:.1f}s"
                            )
                            continue
                        n_i = int(X_i.shape[0])
                        X_mm[cursor : cursor + n_i, :] = X_i
                        y_mm[cursor : cursor + n_i] = y_i
                        cursor += n_i
                        total_attempts += int(attempts_i)
                        print(
                            f"[pool] train seed={seed} ({i}/{len(train_seeds)}) samples={n_i} time={inst_time:.1f}s"
                        )

                X_pool = X_mm[:cursor]
                y_pool = y_mm[:cursor]
                print(f"[pool] memmap: finished collection; dir={tmp_dir}")

                if int(cursor) <= 0 or int(len(y_pool)) <= 0:
                    raise RuntimeError(
                        "No training samples collected (memmap). "
                        "This usually means the labeler returned 0 states for every seed. "
                        "Try loosening constraints (target_util/ranges), switching label_mode, "
                        "or increasing dp_time_limit."
                    )
            else:
                if n_workers > 1 and len(train_seeds) > 1:
                    print(
                        f"[pool] Collecting data with {n_workers} parallel workers..."
                    )
                    with mp.Pool(processes=n_workers, maxtasksperchild=mtpc) as pool:
                        results = []
                        for i, result in enumerate(
                            pool.imap_unordered(worker_fn, train_seeds)
                        ):
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

                all_X = [r[0] for r in results if r[0].size > 0]
                all_y = [r[1] for r in results if r[1].size > 0]
                total_attempts = sum(r[2] for r in results)

                if not all_X or not all_y:
                    raise RuntimeError(
                        "No training samples collected. "
                        "This usually means the labeler returned 0 states for every seed. "
                        "Try loosening constraints (target_util/ranges), switching label_mode, "
                        "or increasing dp_time_limit."
                    )

                X_pool = np.vstack(all_X)
                y_pool = np.concatenate(all_y)

            collect_time = time.perf_counter() - train_t0
            print(
                f"[pool] Data collection: {len(y_pool)} samples in {collect_time:.1f}s"
            )

            save_pooled_data_path = str(args.save_pooled_data).strip()
            if save_pooled_data_path:
                save_pool_p = Path(save_pooled_data_path)
                save_pool_p.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    save_pool_p,
                    X_pool=np.asarray(X_pool),
                    y_pool=np.asarray(y_pool, dtype=np.float64),
                    model_variant=str(requested_model_type),
                    pmax=np.int64(int(args.pmax)),
                )
                print(f"[pool] Saved pooled dataset: {save_pool_p}")

        # ============== Model-specific training ==============
        fit_t0 = time.perf_counter()

        stream_fit = bool(args.stream_fit)
        chunk_size = int(args.fit_chunk_size)

        if stream_fit and model_type not in ("linear", "poly"):
            raise ValueError(
                "--stream-fit currently supports only model-type linear/poly. "
                "For MLP/LGBM, either reduce pooled sample count or train with linear/poly."
            )

        if model_type == "linear":
            if stream_fit:
                w, m = _stream_fit_ridge(
                    X_pool,
                    y_pool,
                    l2=float(args.l2),
                    chunk_size=chunk_size,
                    train_frac=0.85,
                    split_seed=42,
                )
                model = LinearRidgeValueModel(weights=w, spec=spec)
                feat_dim = int(X_pool.shape[1])
                r2_train = float(m["r2_train"])
                mae_train = float(m["mae_train"])
                r2_test = float(m["r2_test"])
                mae_test = float(m["mae_test"])
            else:
                # Train/test split on pooled data (in-memory indexing)
                idx = np.arange(len(y_pool))
                np.random.default_rng(42).shuffle(idx)
                split = int(0.85 * len(idx))
                train_idx = idx[:split]
                test_idx = idx[split:]

                X_train, y_train = X_pool[train_idx], y_pool[train_idx]
                X_test, y_test = X_pool[test_idx], y_pool[test_idx]
                w = fit_ridge(X_train, y_train, l2=float(args.l2))
                model = LinearRidgeValueModel(weights=w, spec=spec)
                y_hat_train = X_train @ w
                y_hat_test = X_test @ w
                feat_dim = int(X_pool.shape[1])
                r2_train = _r2_score(y_train, y_hat_train)
                mae_train = float(np.mean(np.abs(y_train - y_hat_train)))
                r2_test = _r2_score(y_test, y_hat_test)
                mae_test = float(np.mean(np.abs(y_test - y_hat_test)))

            # Print top weights
            top_k = min(10, len(w))
            sorted_idx = np.argsort(np.abs(w))[::-1][:top_k]
            print(f"[pool] Top-{top_k} weights:")
            for rank, fi in enumerate(sorted_idx):
                print(f"    #{rank+1} feat[{fi}] w={w[fi]:.6f}")

        elif model_type == "poly":
            if stream_fit:
                w, powers, m = _stream_fit_poly_ridge(
                    X_pool,
                    y_pool,
                    l2=float(args.l2),
                    chunk_size=chunk_size,
                    train_frac=0.85,
                    split_seed=42,
                )
                model = PolyRidgeValueModel(weights=w, spec=spec, powers_=powers)
                feat_dim = int(m.get("feat_dim_poly", float(powers.shape[0])))
                r2_train = float(m["r2_train"])
                mae_train = float(m["mae_train"])
                r2_test = float(m["r2_test"])
                mae_test = float(m["mae_test"])
            else:
                idx = np.arange(len(y_pool))
                np.random.default_rng(42).shuffle(idx)
                split = int(0.85 * len(idx))
                train_idx = idx[:split]
                test_idx = idx[split:]

                X_train, y_train = X_pool[train_idx], y_pool[train_idx]
                X_test, y_test = X_pool[test_idx], y_pool[test_idx]
                w, powers = fit_poly_ridge(
                    X_train, y_train, l2=float(args.l2), degree=2
                )
                model = PolyRidgeValueModel(weights=w, spec=spec, powers_=powers)
                X_train_poly = _poly_expand_batch(X_train, powers)
                X_test_poly = _poly_expand_batch(X_test, powers)
                y_hat_train = X_train_poly @ w
                y_hat_test = X_test_poly @ w
                feat_dim = int(X_train_poly.shape[1])
                r2_train = _r2_score(y_train, y_hat_train)
                mae_train = float(np.mean(np.abs(y_train - y_hat_train)))
                r2_test = _r2_score(y_test, y_hat_test)
                mae_test = float(np.mean(np.abs(y_test - y_hat_test)))

            print(
                f"[pool] Polynomial: {X_pool.shape[1]} raw → {int(feat_dim)} poly features"
            )

        elif model_type == "mlp":
            # Train/test split (same seed as linear/poly for consistency)
            idx = np.arange(len(y_pool))
            np.random.default_rng(42).shuffle(idx)
            split = int(0.85 * len(idx))
            X_train, y_train = X_pool[idx[:split]], y_pool[idx[:split]]
            X_test, y_test = X_pool[idx[split:]], y_pool[idx[split:]]

            mlp_model = fit_mlp(
                X_train,
                y_train,
                X_test,
                y_test,
                hidden1=64,
                hidden2=32,
                lr=float(args.mlp_lr),
                batch_size=int(args.mlp_batch_size),
                max_epochs=int(args.mlp_max_epochs),
                patience=int(args.mlp_patience),
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
            # Train/test split (same seed as linear/poly for consistency)
            idx = np.arange(len(y_pool))
            np.random.default_rng(42).shuffle(idx)
            split = int(0.85 * len(idx))
            X_train, y_train = X_pool[idx[:split]], y_pool[idx[:split]]
            X_test, y_test = X_pool[idx[split:]], y_pool[idx[split:]]

            booster = fit_lgbm(
                X_train,
                y_train,
                X_test,
                y_test,
                n_estimators=int(args.lgbm_n_estimators),
                max_depth=int(args.lgbm_max_depth),
                learning_rate=float(args.lgbm_learning_rate),
                n_jobs=n_workers,
            )
            model = LGBMValueModel(booster=booster, spec=spec)
            y_hat_train = booster.predict(X_train)
            y_hat_test = booster.predict(X_test)
            feat_dim = int(X_pool.shape[1])

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        fit_time = time.perf_counter() - fit_t0
        train_s = time.perf_counter() - train_t0

        # Training diagnostics (for streaming linear/poly, computed above)
        if not (stream_fit and model_type in ("linear", "poly")):
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
                    include_len_hist=int(spec.include_len_hist),
                    pmax_for_hist=int(spec.pmax_for_hist),
                    include_price_shape=int(spec.include_price_shape),
                    include_meta=int(spec.include_meta),
                    normalize_labels=int(use_normalize_labels),
                    model_type="linear",
                )
            elif model_type in ("poly", "mlp"):
                model.save(str(save_p))
                # Also save normalize_labels in a sidecar (stable filename)
                np.savez(
                    str(save_p) + ".meta.npz",
                    normalize_labels=int(use_normalize_labels),
                )
            elif model_type == "lgbm":
                model.save(str(save_p))
                np.savez(
                    str(save_p) + ".meta.npz",
                    normalize_labels=int(use_normalize_labels),
                )
            print(f"[pool] Model saved to {save_p} (type={model_type})")

    # =========================================================================
    # PHASE 2: EVALUATION — test shared model on held-out instances
    # =========================================================================
    print(
        f"\n[pool] === EVALUATION PHASE ==="
        f"\n[pool] Eval params: D={eval_D} (range={eval_D_range}), N={eval_N} (range={eval_N_range}), pmax={eval_pmax}"
        f"\n[pool] Eval seeds: {eval_seeds[0]}-{eval_seeds[-1]} ({len(eval_seeds)} instances)"
        f"\n[pool] Beams: {beams}"
    )

    eval_time_limit = float(args.eval_time_limit)
    eval_time_limit = eval_time_limit if eval_time_limit > 0.0 else -1.0

    rows: List[Dict[str, float]] = []
    w = model.weights if hasattr(model, "weights") else None

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
                _nlabels = (
                    bool(int(meta["normalize_labels"]))
                    if "normalize_labels" in meta.files
                    else False
                )
    else:
        _nlabels = use_normalize_labels

    # Make the compared variants explicit in the log.
    model_desc = (
        f"type={model_type}, spec={spec}, normalize_labels={bool(_nlabels)}"
        + (f", ckpt={loaded_model_path}" if loaded_model_path else "")
    )
    print(
        f"[pool] Comparing variants: "
        f"Exact DP vs Guided(Learned-Vhat) vs Guided(Zero-Vhat) vs Guided(Price-Vhat)"
        f"\n[pool] Vhat model: {model_desc}"
        f"\n[pool] Legend per-row: exact=Exact DP cost, L=Learned-Vhat, Z=Zero-Vhat, P=Price-Vhat"
    )

    for eval_seed in eval_seeds:
        print(f"[pool] eval seed={eval_seed} starting...", flush=True)
        rng = np.random.default_rng(eval_seed)
        D_use, N_use = _sample_D_N_feasible(
            rng=rng,
            base_D=int(eval_D),
            base_N=int(eval_N),
            D_range=eval_D_range,
            N_range=eval_N_range,
            target_util=float(args.target_util),
            H=20,
        )
        tu = float(args.target_util)
        p, prices = build_instance(
            rng=rng,
            D=D_use,
            N=N_use,
            pmax=eval_pmax,
            daily_prices_20=daily_prices_20,
            target_util=(tu if tu > 0.0 else None),
            M_for_cap=1,
        )
        T = int(len(prices))

        t0 = time.perf_counter()
        exact = solve_optimal_benchmark_dp(
            p,
            prices,
            tie_break="early",
            time_limit=float(eval_time_limit),
            max_states=int(args.dp_max_states),
        )
        exact_s = time.perf_counter() - t0
        if not exact.feasible:
            print(f"[pool] seed={eval_seed} INFEASIBLE, skipping")
            continue

        if bool(exact.timed_out) or (
            hasattr(exact, "is_optimal") and not exact.is_optimal
        ):
            print(
                f"[pool] seed={eval_seed} exact DP timed out (time_limit={eval_time_limit}); "
                "baseline is feasible but not proven optimal",
                flush=True,
            )

        lengths, totals, radices, _mult = encode_setup(p)
        ctx = build_tou_feature_context(prices, H=20, validate_repeating=True)

        # Build vhat closure for this instance using the SHARED model
        used_cache: Dict[int, Tuple[int, ...]] = {0: tuple([0] * len(lengths))}

        # Precompute prefix prices for label denormalization
        prefix_prices = np.concatenate([[0.0], np.cumsum(prices, dtype=np.float64)])

        def _make_vhat(
            model_ref,
            totals_ref,
            lengths_ref,
            ctx_ref,
            radices_ref,
            cache_ref,
            T_ref,
            prefix_ref,
            nlabels,
        ):
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

        vhat = _make_vhat(
            model, totals, lengths, ctx, radices, used_cache, T, prefix_prices, _nlabels
        )

        # Price heuristic
        def _make_vhat_price(
            totals_ref, lengths_ref, radices_ref, cache_ref, T_ref, prefix_ref
        ):
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
                mean_p = float((prefix_ref[T_ref] - prefix_ref[tt]) / rem_len)
                return float(W) * mean_p

            return vhat_price

        vhat_price = _make_vhat_price(
            totals, lengths, radices, used_cache, T, prefix_prices
        )

        for beam in beams:
            t1 = time.perf_counter()
            guided_learned = solve_optimal_benchmark_dp(
                p,
                prices,
                tie_break="early",
                guided=True,
                beam_width=int(beam),
                prune_factor=float(args.prune_factor),
                vhat=vhat,
                time_limit=float(eval_time_limit),
                max_states=int(args.dp_max_states),
            )
            guided_learned_s = time.perf_counter() - t1

            t2 = time.perf_counter()
            guided_zero = solve_optimal_benchmark_dp(
                p,
                prices,
                tie_break="early",
                guided=True,
                beam_width=int(beam),
                prune_factor=float(args.prune_factor),
                vhat=lambda _t, _s: 0.0,
                time_limit=float(eval_time_limit),
                max_states=int(args.dp_max_states),
            )
            guided_zero_s = time.perf_counter() - t2

            t3 = time.perf_counter()
            guided_price = solve_optimal_benchmark_dp(
                p,
                prices,
                tie_break="early",
                guided=True,
                beam_width=int(beam),
                prune_factor=float(args.prune_factor),
                vhat=vhat_price,
                time_limit=float(eval_time_limit),
                max_states=int(args.dp_max_states),
            )
            guided_price_s = time.perf_counter() - t3

            gap_learned = (
                (guided_learned.cost - exact.cost) / max(1e-9, abs(exact.cost)) * 100.0
            )
            gap_zero = (
                (guided_zero.cost - exact.cost) / max(1e-9, abs(exact.cost)) * 100.0
            )
            gap_price = (
                (guided_price.cost - exact.cost) / max(1e-9, abs(exact.cost)) * 100.0
            )

            row = {
                "seed": float(eval_seed),
                "T": float(T),
                "D": float(D_use),
                "N": float(len(p)),
                "K": float(len(lengths)),
                "exact_cost": float(exact.cost),
                "exact_s": float(exact_s),
                "exact_timed_out": float(int(getattr(exact, "timed_out", False))),
                "exact_is_optimal": float(int(getattr(exact, "is_optimal", True))),
                "train_s": float(train_s),
                "guided_learned_cost": float(guided_learned.cost),
                "guided_learned_s": float(guided_learned_s),
                "guided_learned_timed_out": float(
                    int(getattr(guided_learned, "timed_out", False))
                ),
                "guided_zero_cost": float(guided_zero.cost),
                "guided_zero_s": float(guided_zero_s),
                "guided_zero_timed_out": float(
                    int(getattr(guided_zero, "timed_out", False))
                ),
                "guided_price_cost": float(guided_price.cost),
                "guided_price_s": float(guided_price_s),
                "guided_price_timed_out": float(
                    int(getattr(guided_price, "timed_out", False))
                ),
                "gap_learned_pct": float(gap_learned),
                "gap_zero_pct": float(gap_zero),
                "gap_price_pct": float(gap_price),
                "speedup_learned": float(exact_s / max(guided_learned_s, 1e-12)),
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
    print(
        f"[pool] Train time: {train_s:.1f}s (amortized over {len(eval_seeds)} eval instances)"
    )
    print(f"[pool] CSV: {out_csv}")


if __name__ == "__main__":
    main()

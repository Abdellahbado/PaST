from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from PaST.solvers.vhat_tou_features import TOUFeatureContext, build_tou_feature_context, get_day_window_cost


@dataclass(frozen=True)
class FeatureSpec:
    """Controls which features are included.

    Keep this small and cheap: the feature vector will be computed inside a DP
    loop (beam scoring), potentially millions of times.
    """

    include_per_class_counts: bool = True
    include_per_class_now_cost: bool = True
    include_bins: bool = True
    normalize: bool = False  # divide absolute quantities by T for cross-size transfer


def _remaining_from_used(used: Sequence[int], totals: np.ndarray) -> np.ndarray:
    used_arr = np.asarray(used, dtype=np.int32)
    return (totals.astype(np.int32) - used_arr).astype(np.int32)


def phi_for_state(
    *,
    t: int,
    used: Sequence[int],
    totals: np.ndarray,
    lengths: Sequence[int],
    ctx: TOUFeatureContext,
    spec: FeatureSpec,
) -> np.ndarray:
    """Feature vector for (t, used-counts) state.

    Returns float64 1D array.

    Notes:
    - This is energy-only; tie-break ("early") should remain in the DP.
    - All features are designed to be cheap: O(K) in number of classes.
    - When spec.normalize=True, absolute quantities are divided by T
      so features become scale-invariant ratios suitable for cross-size transfer.
    """
    t = int(t)
    T = int(ctx.T)
    if t < 0:
        t = 0
    if t > T:
        t = T

    # Normalization denominator (T for absolute→ratio conversion)
    norm = float(T) if spec.normalize else 1.0

    lengths_arr = np.asarray(lengths, dtype=np.int32)
    remaining = _remaining_from_used(used, totals)

    N = float(int(np.sum(remaining)))
    W = float(int(np.sum(remaining * lengths_arr)))
    R = float(T - t)
    S = float((T - t) - int(W))
    S_pos = float(max(S, 0.0))

    h = int(t % ctx.H)
    reg = int(ctx.day_regime[h])

    # Regime one-hot (dimensionless, no normalization needed)
    reg_oh = [0.0, 0.0, 0.0]
    if 0 <= reg < 3:
        reg_oh[reg] = 1.0

    # Cheap capacity ahead in remaining horizon
    c_off = float(int(ctx.count_regime[0, t]))
    c_sh = float(int(ctx.count_regime[1, t]))
    c_peak = float(int(ctx.count_regime[2, t]))

    # Pressure ratios (already normalized, no change)
    pressure_off = float(W / (c_off + 1.0))
    pressure_cheap = float(W / (c_off + c_sh + 1.0))

    # Distances to next off/cheap within cycle (bounded by H=20, no normalization)
    d_off = float(int(ctx.dist_to_next_off[h]))
    d_cheap = float(int(ctx.dist_to_next_cheap[h]))

    feats: List[float] = []

    # Bias
    feats.append(1.0)

    # Time-of-day & regime (dimensionless)
    feats.extend(reg_oh)
    feats.extend([d_off, d_cheap])

    # Workload summary (normalized by T when normalize=True)
    feats.extend([N / norm, W / norm, R / norm, S_pos / norm])

    # Slack-regime interactions (normalized)
    feats.extend([(S_pos / norm) * reg_oh[0], (S_pos / norm) * reg_oh[2]])

    # Cheap capacity ahead (normalized) + pressure ratios (already scale-invariant)
    feats.extend([c_off / norm, c_peak / norm, pressure_off, pressure_cheap])

    # Simple bins by length (optional, normalized)
    if spec.include_bins:
        short = float(int(np.sum(remaining[lengths_arr <= 2])))
        median_len = int(np.median(lengths_arr)) if int(lengths_arr.size) > 0 else 0
        long_thr = max(3, median_len)
        long = float(int(np.sum(remaining[lengths_arr >= long_thr])))
        feats.extend([short / norm, long / norm])

    # Per-class counts (normalized)
    if spec.include_per_class_counts:
        feats.extend([float(int(x)) / norm for x in remaining.tolist()])

    # Per-class cost-if-run-now (wrap within day) and aggregate
    # Note: cost values are already price-scaled (bounded by price profile), not by T
    if spec.include_per_class_now_cost:
        agg = 0.0
        for nk, L in zip(remaining.tolist(), lengths_arr.tolist()):
            if nk <= 0:
                feats.append(0.0)
                continue
            # cost of running length L starting at current hour-of-day
            if int(L) <= ctx.H:
                cost_now = float(ctx.day_window_cost[int(L)][h])
            else:
                cost_now = float(get_day_window_cost(ctx, int(L))[h])
            v = float(int(nk)) * cost_now / norm
            feats.append(v)
            agg += v
        feats.append(float(agg))

    return np.asarray(feats, dtype=np.float64)


@dataclass
class LinearRidgeValueModel:
    """Simple linear value function approximator Vhat = w^T phi.

    Trained by ridge regression on (phi, y) pairs.
    """

    weights: np.ndarray  # shape (D,)
    spec: FeatureSpec
    H: int = 20

    def predict_from_used(
        self,
        *,
        t: int,
        used: Sequence[int],
        totals: np.ndarray,
        lengths: Sequence[int],
        ctx: TOUFeatureContext,
    ) -> float:
        x = phi_for_state(t=t, used=used, totals=totals, lengths=lengths, ctx=ctx, spec=self.spec)
        return float(np.dot(self.weights, x))


def fit_ridge(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1e-6,
) -> np.ndarray:
    """Closed-form ridge regression weights."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of rows")

    D = int(X.shape[1])
    A = X.T @ X
    b = X.T @ y
    A = A + float(l2) * np.eye(D, dtype=np.float64)
    w = np.linalg.solve(A, b)
    return w.astype(np.float64)

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from PaST.solvers.vhat_tou_features import (
    TOUFeatureContext,
    build_tou_feature_context,
    get_day_window_cost,
)


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

    # Optional cross-size / cross-profile features (defaults off for compatibility)
    include_len_hist: bool = False  # fixed-length histogram over p in {1..pmax}
    pmax_for_hist: int = 12
    include_price_shape: bool = False  # features derived from ctx.day (length H)
    include_meta: bool = False  # extra metadata/log-scale features
    include_extra: bool = False  # additional generalization features (see §extra block)

    # When > 0, per-class features (counts & now-cost) are padded to this fixed
    # size indexed by processing time 1..per_class_pad, avoiding variable-length
    # feature vectors when the number of unique job lengths varies across instances.
    per_class_pad: int = 0


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

    # Detect NR-honest mode
    _nr = bool(getattr(ctx, "is_nonrepeating", False))

    # Normalization denominator (T for absolute→ratio conversion)
    norm = float(T) if spec.normalize else 1.0

    lengths_arr = np.asarray(lengths, dtype=np.int32)
    remaining = _remaining_from_used(used, totals)

    N = float(int(np.sum(remaining)))
    W = float(int(np.sum(remaining * lengths_arr)))
    R = float(T - t)
    S = float((T - t) - int(W))
    S_pos = float(max(S, 0.0))

    # --- Regime & distances: NR-honest vs repeating-proxy ------------------
    if _nr:
        t_idx = min(t, T - 1) if T > 0 else 0
        reg = int(ctx.regime[t_idx])
        _d_off_full = getattr(ctx, "dist_to_next_off_full", None)
        _d_cheap_full = getattr(ctx, "dist_to_next_cheap_full", None)
        if _d_off_full is not None and _d_cheap_full is not None:
            d_off = float(int(_d_off_full[t_idx]))
            d_cheap = float(int(_d_cheap_full[t_idx]))
        else:
            d_off = 0.0
            d_cheap = 0.0
    else:
        h = int(t % ctx.H)
        reg = int(ctx.day_regime[h])
        d_off = float(int(ctx.dist_to_next_off[h]))
        d_cheap = float(int(ctx.dist_to_next_cheap[h]))

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

    # Distances to next off/cheap (bounded by H in repeating, T in NR)
    # (already computed above in the NR/repeating branch)

    feats: List[float] = []

    # Bias
    feats.append(1.0)

    # Time-of-day & regime (dimensionless)
    feats.extend(reg_oh)
    feats.extend([d_off, d_cheap])

    # Workload summary (normalized by T when normalize=True)
    feats.extend([N / norm, W / norm, R / norm, S_pos / norm])

    # Optional: extra meta features for smoother cross-size generalization.
    if spec.include_meta:
        # Note: use log1p for stability across ranges.
        # Utilization proxies: W/T and slack ratio.
        util = float(W / float(T)) if T > 0 else 0.0
        slack_ratio = float(S_pos / (W + 1.0))
        feats.extend(
            [
                float(np.log1p(float(T))),
                float(np.log1p(float(N))),
                float(np.log1p(float(W))),
                util,
                slack_ratio,
            ]
        )

    # Slack-regime interactions (normalized)
    feats.extend([(S_pos / norm) * reg_oh[0], (S_pos / norm) * reg_oh[2]])

    # Cheap capacity ahead (normalized) + pressure ratios (already scale-invariant)
    feats.extend(
        [c_off / norm, c_sh / norm, c_peak / norm, pressure_off, pressure_cheap]
    )

    # Simple bins by length (optional, normalized)
    if spec.include_bins:
        short = float(int(np.sum(remaining[lengths_arr <= 2])))
        median_len = int(np.median(lengths_arr)) if int(lengths_arr.size) > 0 else 0
        long_thr = max(3, median_len)
        long = float(int(np.sum(remaining[lengths_arr >= long_thr])))
        feats.extend([short / norm, long / norm])

    # Optional: fixed-length histogram of remaining job lengths.
    # This provides a size-agnostic set representation even when K varies.
    if spec.include_len_hist:
        pmax_h = int(max(1, int(spec.pmax_for_hist)))
        # Build counts for p in 1..pmax_h; lengths outside are clipped.
        # Note: remaining is per unique length class, so we accumulate.
        hist = np.zeros(pmax_h, dtype=np.float64)
        for nk, L in zip(remaining.tolist(), lengths_arr.tolist()):
            if nk <= 0:
                continue
            idx = int(L)
            if idx < 1:
                continue
            if idx > pmax_h:
                idx = pmax_h
            hist[idx - 1] += float(int(nk))
        feats.extend((hist / norm).tolist())

    # Optional: price-shape features from the daily pattern.
    # Useful when profiles vary (even if still repeating daily).
    if spec.include_price_shape:
        _lpf = getattr(ctx, "local_price_feats", None)
        if _nr and _lpf is not None:
            # NR-honest: precomputed local stats + Fourier for this time step
            t_idx = min(t, T - 1) if T > 0 else 0
            feats.extend(_lpf[t_idx].tolist())
        else:
            day = np.asarray(ctx.day, dtype=np.float64)
            # Basic stats
            feats.extend(
                [
                    float(np.mean(day)),
                    float(np.std(day)),
                    float(np.min(day)),
                    float(np.max(day)),
                ]
            )
            # Fourier components (k=1..3) to capture within-day shape.
            H = int(ctx.H)
            x = np.arange(H, dtype=np.float64)
            for k in (1, 2, 3):
                ang = 2.0 * np.pi * float(k) * x / float(H)
                feats.append(float(np.mean(day * np.cos(ang))))
                feats.append(float(np.mean(day * np.sin(ang))))

    # Per-class counts (normalized)
    if spec.include_per_class_counts:
        pad = int(spec.per_class_pad)
        if pad > 0:
            # Fixed-length: index by processing time 1..pad
            counts_vec = [0.0] * pad
            for nk, L in zip(remaining.tolist(), lengths_arr.tolist()):
                idx = int(L) - 1  # length 1 → index 0, etc.
                if 0 <= idx < pad:
                    counts_vec[idx] = float(int(nk)) / norm
            feats.extend(counts_vec)
        else:
            feats.extend([float(int(x)) / norm for x in remaining.tolist()])

    # Per-class cost-if-run-now (wrap within day) and aggregate
    # Note: cost values are already price-scaled (bounded by price profile), not by T
    if spec.include_per_class_now_cost:
        _pfx = getattr(ctx, "prefix_prices", None)
        pad = int(spec.per_class_pad)
        agg = 0.0
        if pad > 0:
            # Fixed-length: index by processing time 1..pad
            cost_vec = [0.0] * pad
            for nk, L in zip(remaining.tolist(), lengths_arr.tolist()):
                if nk <= 0:
                    continue
                if _nr and _pfx is not None:
                    # NR-honest: exact cost from prefix sums
                    end = min(t + int(L), T)
                    cost_now = float(_pfx[end] - _pfx[t])
                else:
                    h = int(t % ctx.H)
                    if int(L) <= ctx.H:
                        cost_now = float(ctx.day_window_cost[int(L)][h])
                    else:
                        cost_now = float(get_day_window_cost(ctx, int(L))[h])
                v = float(int(nk)) * cost_now / norm
                idx = int(L) - 1
                if 0 <= idx < pad:
                    cost_vec[idx] = v
                agg += v
            feats.extend(cost_vec)
        else:
            for nk, L in zip(remaining.tolist(), lengths_arr.tolist()):
                if nk <= 0:
                    feats.append(0.0)
                    continue
                if _nr and _pfx is not None:
                    end = min(t + int(L), T)
                    cost_now = float(_pfx[end] - _pfx[t])
                else:
                    h = int(t % ctx.H)
                    if int(L) <= ctx.H:
                        cost_now = float(ctx.day_window_cost[int(L)][h])
                    else:
                        cost_now = float(get_day_window_cost(ctx, int(L))[h])
                v = float(int(nk)) * cost_now / norm
                feats.append(v)
                agg += v
        feats.append(float(agg))

    # Extra generalization features (opt-in for richer signal without breaking
    # compatibility with existing checkpoints).
    if spec.include_extra:
        T_remain = max(1, T - t)
        # 1. fraction of horizon remaining (0..1, normalized by construction)
        feats.append(float(T_remain) / float(max(1, T)))
        # 2. workload-to-remaining-horizon ratio (congestion measure)
        feats.append(float(W) / float(T_remain))
        # 3. mean remaining job length ⇢ captures job-mix complexity
        n_remaining = max(1, int(np.sum(remaining)))
        mean_len = (
            float(np.dot(remaining, lengths_arr)) / float(n_remaining)
            if n_remaining > 0
            else 0.0
        )
        feats.append(mean_len / float(max(1, int(np.max(lengths_arr)))))
        # 4. variance of remaining job lengths (diversity of job sizes)
        if n_remaining > 1:
            # Weighted variance: expand remaining counts for each length
            expanded = np.repeat(lengths_arr.astype(np.float64), remaining.astype(int))
            var_len = float(np.var(expanded)) if expanded.size > 1 else 0.0
        else:
            var_len = 0.0
        feats.append(var_len / float(max(1.0, float(np.max(lengths_arr)) ** 2)))
        # 5. cheap-slot utilization opportunity: fraction of cheap slots
        #    available vs total remaining slots
        feats.append(float(c_off + c_sh) / float(T_remain))

    return np.asarray(feats, dtype=np.float64)


def phi_for_states_batch(
    *,
    t: int,
    states: List[int],
    totals: np.ndarray,
    lengths: List[int],
    ctx: TOUFeatureContext,
    spec: FeatureSpec,
    radices: np.ndarray,
    used_cache: Dict[int, Tuple[int, ...]],
) -> np.ndarray:
    """Batch feature extraction for multiple states at the same time step *t*.

    Returns a (len(states), D) float64 array. Delegates to phi_for_state per
    row — the main saving is avoiding repeated Python call overhead from the
    caller and allowing the downstream model to do a single matrix multiply.
    """
    # Decode states once
    K = len(radices)
    useds: List[Tuple[int, ...]] = []
    for s in states:
        cached = used_cache.get(s)
        if cached is None:
            u = [0] * K
            x = s
            for i in range(K):
                r = int(radices[i])
                u[i] = x % r
                x //= r
            cached = tuple(u)
            used_cache[s] = cached
        useds.append(cached)

    # Compute features for each state
    rows = []
    for used in useds:
        phi = phi_for_state(
            t=t,
            used=used,
            totals=totals,
            lengths=lengths,
            ctx=ctx,
            spec=spec,
        )
        rows.append(phi)
    return np.vstack(rows)


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
        x = phi_for_state(
            t=t, used=used, totals=totals, lengths=lengths, ctx=ctx, spec=self.spec
        )
        return float(np.dot(self.weights, x))

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Batch predict. X: (N, D) -> (N,)."""
        return X @ self.weights


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

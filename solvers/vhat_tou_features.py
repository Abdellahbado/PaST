from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class TOUFeatureContext:
    """Precomputations for fast feature extraction under repeating TOU.

    Assumptions (for now):
    - prices are per-slot, length T
    - daily pattern repeats with cycle length H (default 20)

    This context is intentionally lightweight: it is designed to be used inside
    DP loops (beam scoring), so all operations are O(K) in number of length
    classes and O(1) in time.
    """

    prices: np.ndarray  # shape (T,)
    T: int
    H: int

    # Daily pattern (first H slots)
    day: np.ndarray  # shape (H,)

    # Regime id per time slot: 0=off, 1=shoulder, 2=peak
    regime: np.ndarray  # shape (T,)

    # Suffix counts: count_regime[r][t] = #slots tau in [t,T) with regime==r
    count_regime: np.ndarray  # shape (3, T+1)

    # Per-day regime id and next distances within the day cycle
    day_regime: np.ndarray  # shape (H,)
    dist_to_next_off: np.ndarray  # shape (H,)
    dist_to_next_cheap: np.ndarray  # shape (H,)

    # Window cost if starting at hour-of-day h with length L (wrap within day)
    # Stored as dict: L -> np.ndarray shape (H,)
    day_window_cost: Dict[int, np.ndarray]


def _bucket_3_levels(day_prices: np.ndarray) -> np.ndarray:
    """Map daily prices to 3 regimes by rank (low/mid/high)."""
    uniq = np.unique(day_prices)
    if len(uniq) <= 3:
        # Direct mapping: sorted unique -> 0..2 (pad if <3)
        mapping: Dict[float, int] = {}
        for idx, v in enumerate(sorted(float(x) for x in uniq)):
            mapping[v] = min(idx, 2)
        return np.array([mapping[float(x)] for x in day_prices], dtype=np.int8)

    # More than 3 unique levels: bucket by tertiles of ranks
    ranks = np.argsort(np.argsort(day_prices))
    # ranks in 0..H-1
    out = np.zeros(len(day_prices), dtype=np.int8)
    out[ranks >= (2 * len(day_prices)) // 3] = 2
    out[(ranks >= len(day_prices) // 3) & (ranks < (2 * len(day_prices)) // 3)] = 1
    return out


def _dist_to_next(mask: np.ndarray) -> np.ndarray:
    """For each index i in a cycle, distance to next True in mask (cyclic)."""
    H = int(len(mask))
    if H == 0:
        return np.zeros(0, dtype=np.int32)

    true_idx = np.where(mask)[0]
    if true_idx.size == 0:
        # No such regime exists; set to H (max distance)
        return np.full(H, H, dtype=np.int32)

    dist = np.full(H, H, dtype=np.int32)
    # For each position i, next true is min over j>=i else wrap.
    for i in range(H):
        after = true_idx[true_idx >= i]
        if after.size:
            dist[i] = int(after[0] - i)
        else:
            dist[i] = int((true_idx[0] + H) - i)
    return dist


def build_tou_feature_context(
    prices: np.ndarray,
    *,
    H: int = 20,
    validate_repeating: bool = False,
) -> TOUFeatureContext:
    prices = np.asarray(prices, dtype=np.float64)
    T = int(len(prices))
    if T <= 0:
        raise ValueError("prices must be non-empty")
    if H <= 0:
        raise ValueError("H must be positive")
    if T < H:
        raise ValueError(f"Need T>=H to build repeating context (T={T}, H={H}).")

    day = prices[:H].copy()

    if validate_repeating:
        # Allow last partial day.
        for t in range(T):
            if abs(prices[t] - day[t % H]) > 1e-9:
                raise ValueError(
                    "prices do not repeat with the provided cycle length H"
                )

    day_regime = _bucket_3_levels(day)

    # Define off as regime 0 (lowest bucket). Cheap as bucket in {0,1}.
    dist_to_next_off = _dist_to_next(day_regime == 0)
    dist_to_next_cheap = _dist_to_next(day_regime <= 1)

    # Expand regime over the full horizon
    regime = np.empty(T, dtype=np.int8)
    for t in range(T):
        regime[t] = day_regime[t % H]

    # Suffix regime counts
    count_regime = np.zeros((3, T + 1), dtype=np.int32)
    for t in range(T - 1, -1, -1):
        count_regime[:, t] = count_regime[:, t + 1]
        r = int(regime[t])
        if 0 <= r < 3:
            count_regime[r, t] += 1

    # Precompute day window costs for all L up to H? We fill lazily later too.
    day2 = np.concatenate([day, day])
    pref2 = np.zeros(len(day2) + 1, dtype=np.float64)
    pref2[1:] = np.cumsum(day2)

    day_window_cost: Dict[int, np.ndarray] = {}
    # We only meaningfully support L up to T. For L > 2H, wrap costs repeat; we
    # still compute using repeated day blocks in features rather than exact.
    for L in range(1, H + 1):
        costs = np.zeros(H, dtype=np.float64)
        for h in range(H):
            costs[h] = pref2[h + L] - pref2[h]
        day_window_cost[int(L)] = costs

    return TOUFeatureContext(
        prices=prices,
        T=T,
        H=H,
        day=day,
        regime=regime,
        count_regime=count_regime,
        day_regime=day_regime,
        dist_to_next_off=dist_to_next_off,
        dist_to_next_cheap=dist_to_next_cheap,
        day_window_cost=day_window_cost,
    )


def get_day_window_cost(ctx: TOUFeatureContext, L: int) -> np.ndarray:
    """Return per-hour-of-day cost of running a length-L job starting at hour h.

    For L <= H: exact (wrap within one day).
    For L > H: approximates as full-day blocks + remainder.
    """
    L = int(L)
    H = int(ctx.H)
    if L <= 0:
        raise ValueError("L must be positive")

    cached = ctx.day_window_cost.get(L)
    if cached is not None:
        return cached

    # Approximate: q full days + r remainder
    q, r = divmod(L, H)
    base = float(q) * float(np.sum(ctx.day))
    if r == 0:
        out = np.full(H, base, dtype=np.float64)
    else:
        rem = get_day_window_cost(ctx, r)
        out = base + rem
    # Cache in a new dict (ctx is frozen, so return without storing).
    return out

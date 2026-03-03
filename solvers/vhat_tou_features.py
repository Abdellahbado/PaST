from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

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

    # --- NR-honest fields (populated only by build_tou_feature_context_nonrepeating) ---
    prefix_prices: Optional[np.ndarray] = (
        None  # shape (T+1,) cumsum for exact window costs
    )
    is_nonrepeating: bool = False
    dist_to_next_off_full: Optional[np.ndarray] = (
        None  # shape (T,) forward-looking per slot
    )
    dist_to_next_cheap_full: Optional[np.ndarray] = (
        None  # shape (T,) forward-looking per slot
    )
    local_price_feats: Optional[np.ndarray] = (
        None  # shape (T, 10) local stats + Fourier
    )


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


def _dist_to_next_forward(mask: np.ndarray) -> np.ndarray:
    """For each index t, distance to next True in mask[t:] (non-cyclic).

    If no True exists at or after t, returns len(mask)-t (beyond horizon).
    """
    T = int(len(mask))
    if T == 0:
        return np.zeros(0, dtype=np.int32)

    dist = np.zeros(T, dtype=np.int32)
    last_true = T  # sentinel: beyond horizon
    for t in range(T - 1, -1, -1):
        if mask[t]:
            last_true = t
        dist[t] = int(last_true - t)
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


def build_tou_feature_context_nonrepeating(
    prices: np.ndarray,
    *,
    H: int = 20,
) -> TOUFeatureContext:
    """Build feature context for NON-repeating prices.

    Unlike the repeating version, this uses the ACTUAL prices at every time
    step rather than assuming prices[:H] repeats.

    - Regime buckets are computed from the FULL trajectory (tertile-based).
    - Suffix regime counts reflect actual future prices.
    - Distance-to-next-off/cheap is forward-looking (non-cyclic, per slot).
    - A prefix-sum array enables exact window-cost queries at any (t, L).
    - Local price features (mean/std/min/max + Fourier k=1..3 in a window
      of H slots ahead) are precomputed for each time step.
    """
    prices = np.asarray(prices, dtype=np.float64)
    T = int(len(prices))
    if T <= 0:
        raise ValueError("prices must be non-empty")
    if H <= 0:
        raise ValueError("H must be positive")

    # --- Regime from FULL trajectory (honest bucketing) --------------------
    regime = _bucket_3_levels(prices)  # shape (T,), int8

    # For backward compat: 'day' = first H slots; day_regime from that slice
    day = prices[: min(H, T)].copy()
    if len(day) < H:
        day = np.pad(day, (0, H - len(day)), constant_values=float(day[-1]))
    day_regime = _bucket_3_levels(day)

    # Per-slot forward distance to next off / cheap (non-cyclic)
    dist_off_full = _dist_to_next_forward(regime == 0)
    dist_cheap_full = _dist_to_next_forward(regime <= 1)

    # Cycle-based distances kept for compat (unused in NR phi_for_state)
    dist_to_next_off = _dist_to_next(day_regime == 0)
    dist_to_next_cheap = _dist_to_next(day_regime <= 1)

    # --- Suffix regime counts from ACTUAL regimes --------------------------
    count_regime = np.zeros((3, T + 1), dtype=np.int32)
    for t in range(T - 1, -1, -1):
        count_regime[:, t] = count_regime[:, t + 1]
        r = int(regime[t])
        if 0 <= r < 3:
            count_regime[r, t] += 1

    # --- Prefix prices for exact window-cost queries -----------------------
    prefix_prices = np.zeros(T + 1, dtype=np.float64)
    prefix_prices[1:] = np.cumsum(prices)

    # --- day_window_cost (from day, kept for compat; not used in NR mode) --
    day2 = np.concatenate([day, day])
    pref2 = np.zeros(len(day2) + 1, dtype=np.float64)
    pref2[1:] = np.cumsum(day2)
    day_window_cost: Dict[int, np.ndarray] = {}
    for L in range(1, H + 1):
        costs = np.zeros(H, dtype=np.float64)
        for h_idx in range(H):
            costs[h_idx] = pref2[h_idx + L] - pref2[h_idx]
        day_window_cost[int(L)] = costs

    # --- Local price features per time step (stats + Fourier) --------------
    # Shape (T, 10): [mean, std, min, max, cos1, sin1, cos2, sin2, cos3, sin3]
    local_price_feats = np.zeros((T, 10), dtype=np.float64)
    for t in range(T):
        window = prices[t : min(t + H, T)]
        wH = len(window)
        local_price_feats[t, 0] = float(np.mean(window))
        local_price_feats[t, 1] = float(np.std(window))
        local_price_feats[t, 2] = float(np.min(window))
        local_price_feats[t, 3] = float(np.max(window))
        x = np.arange(wH, dtype=np.float64)
        for ki, k in enumerate((1, 2, 3)):
            ang = 2.0 * np.pi * float(k) * x / float(max(wH, 1))
            local_price_feats[t, 4 + 2 * ki] = float(np.mean(window * np.cos(ang)))
            local_price_feats[t, 5 + 2 * ki] = float(np.mean(window * np.sin(ang)))

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
        prefix_prices=prefix_prices,
        is_nonrepeating=True,
        dist_to_next_off_full=dist_off_full,
        dist_to_next_cheap_full=dist_cheap_full,
        local_price_feats=local_price_feats,
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

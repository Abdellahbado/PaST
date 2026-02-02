"""Shared feature extraction for duration-class branching policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from PaST.solvers.bnb_solver_custom import Instance


@dataclass(frozen=True)
class FeatureSpec:
    duration_vocab: tuple[int, ...]


class WindowFeatureCache:
    """Memoized window-sum statistics for a fixed TOU price curve."""

    def __init__(self, energy_costs: np.ndarray):
        self._c = energy_costs.astype(np.float64)
        self._prefix = np.zeros(len(self._c) + 1, dtype=np.float64)
        self._prefix[1:] = np.cumsum(self._c)
        self._cache: Dict[int, Dict[str, float]] = {}

    def window_stats(self, d: int) -> Dict[str, float]:
        if d <= 0:
            raise ValueError("duration d must be > 0")
        if d in self._cache:
            return self._cache[d]

        T = len(self._c)
        if d > T:
            out = {
                "w_min": float("inf"),
                "w_min2": float("inf"),
                "w_mean": float("inf"),
                "w_std": float("inf"),
                "w_p10": float("inf"),
                "w_p50": float("inf"),
                "w_p90": float("inf"),
                "w_argmin": 1.0,
                "w_gap2": float("inf"),
                "w_min_per_unit": float("inf"),
            }
            self._cache[d] = out
            return out

        w = self._prefix[d:] - self._prefix[:-d]
        w = np.nan_to_num(w, nan=np.inf, posinf=np.inf, neginf=-np.inf)

        if w.size == 0:
            out = {
                "w_min": float("inf"),
                "w_min2": float("inf"),
                "w_mean": float("inf"),
                "w_std": float("inf"),
                "w_p10": float("inf"),
                "w_p50": float("inf"),
                "w_p90": float("inf"),
                "w_argmin": 1.0,
                "w_gap2": float("inf"),
                "w_min_per_unit": float("inf"),
            }
            self._cache[d] = out
            return out

        order = np.argsort(w)
        w_min = float(w[order[0]])
        w_min2 = float(w[order[1]]) if order.size > 1 else w_min
        argmin = float(order[0]) / float(max(1, (T - d)))
        out = {
            "w_min": w_min,
            "w_min2": w_min2,
            "w_mean": float(np.mean(w)),
            "w_std": float(np.std(w)),
            "w_p10": float(np.percentile(w, 10)),
            "w_p50": float(np.percentile(w, 50)),
            "w_p90": float(np.percentile(w, 90)),
            "w_argmin": argmin,
            "w_gap2": float(w_min2 - w_min),
            "w_min_per_unit": float(w_min / max(1, d)),
        }
        self._cache[d] = out
        return out


def price_stats(c: np.ndarray) -> Dict[str, float]:
    c = c.astype(np.float64)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "c_mean": float(np.mean(c)),
        "c_std": float(np.std(c)),
        "c_min": float(np.min(c)),
        "c_max": float(np.max(c)),
        "c_p10": float(np.percentile(c, 10)),
        "c_p50": float(np.percentile(c, 50)),
        "c_p90": float(np.percentile(c, 90)),
    }


def gcd_list(vals: Sequence[int]) -> int:
    if not vals:
        return 0
    g = 0
    for v in vals:
        g = int(np.gcd(g, int(v)))
    return int(g)


def extract_row_features(
    *,
    instance: Instance,
    duration_vocab: Sequence[int],
    window_cache: WindowFeatureCache,
    partial_sequence: Sequence[int],
    remaining_jobs: Sequence[int],
    candidate_d: int,
) -> Dict[str, float]:
    p = instance.processing_times
    c = instance.energy_costs

    prefix_len = int(len(partial_sequence))
    prefix_total_p = int(p[list(partial_sequence)].sum()) if prefix_len > 0 else 0

    rem = list(remaining_jobs)
    rem_pts = p[rem].astype(np.int32) if rem else np.array([], dtype=np.int32)
    remaining_total_p = int(rem_pts.sum()) if rem_pts.size > 0 else 0
    remaining_count = int(rem_pts.size)
    unique_d_count = int(len(set(rem_pts.tolist()))) if rem_pts.size > 0 else 0

    gcd_remaining = gcd_list(rem_pts.tolist()) if rem_pts.size > 0 else 0

    hist = {f"rem_count_d{d}": 0.0 for d in duration_vocab}
    if rem_pts.size > 0:
        for d in rem_pts.tolist():
            if d in hist:
                hist[f"rem_count_d{int(d)}"] += 1.0

    cand_count = float(hist.get(f"rem_count_d{int(candidate_d)}", 0.0))
    cand_frac = float(cand_count / max(1.0, float(remaining_count)))

    feat: Dict[str, float] = {}
    feat.update(price_stats(c))
    feat.update(
        {
            "prefix_len": float(prefix_len),
            "prefix_total_p": float(prefix_total_p),
            "remaining_total_p": float(remaining_total_p),
            "remaining_count": float(remaining_count),
            "unique_d_count": float(unique_d_count),
            "gcd_remaining": float(gcd_remaining),
            "cand_d": float(candidate_d),
            "cand_count": float(cand_count),
            "cand_frac": float(cand_frac),
        }
    )
    feat.update(hist)
    feat.update(window_cache.window_stats(int(candidate_d)))

    feat["ratio_d_to_rem_mean"] = float(
        candidate_d / max(1.0, remaining_total_p / max(1.0, remaining_count))
    )
    feat["ratio_d_to_prefix"] = float(candidate_d / max(1.0, prefix_total_p))
    feat["ratio_rem_to_T"] = float(remaining_total_p / max(1.0, float(instance.T)))

    for k, v in list(feat.items()):
        if not np.isfinite(v):
            feat[k] = 0.0

    return feat

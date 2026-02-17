from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from PaST.data.sm_benchmark_data import RawInstance
from PaST.neurols.price_embedding import PriceFeatureExtractor


def _price_blocks(ct: np.ndarray) -> List[Tuple[int, int, float]]:
    if ct.size == 0:
        return []
    out: List[Tuple[int, int, float]] = []
    start = 0
    cur = float(ct[0])
    for t in range(1, int(ct.size)):
        v = float(ct[t])
        if v != cur:
            out.append((start, t - start, cur))
            start = t
            cur = v
    out.append((start, int(ct.size) - start, cur))
    return out


def _rle_blocks_levels(levels: np.ndarray) -> List[Tuple[int, int, int]]:
    if levels.size == 0:
        return []
    out: List[Tuple[int, int, int]] = []
    start = 0
    cur = int(levels[0])
    for t in range(1, int(levels.size)):
        lv = int(levels[t])
        if lv != cur:
            out.append((start, t - start, cur))
            start = t
            cur = lv
    out.append((start, int(levels.size) - start, cur))
    return out


def _bounded_subset_sum_pick_counts(counts_by_p: np.ndarray, capacity: int) -> np.ndarray:
    """Pick a multiset of p-values (bounded counts) with total sum <= capacity.

    Returns chosen_counts_by_p with same shape as counts_by_p.
    counts_by_p[p] corresponds to processing time p (index 0 unused).
    """
    cap = int(capacity)
    if cap <= 0:
        return np.zeros_like(counts_by_p)

    max_p = int(counts_by_p.size) - 1
    dp = np.full(cap + 1, -1, dtype=np.int32)
    dp[0] = 0

    parent_prev = np.full(cap + 1, -1, dtype=np.int32)
    parent_p = np.full(cap + 1, -1, dtype=np.int32)
    parent_cnt = np.full(cap + 1, -1, dtype=np.int32)

    for p in range(1, max_p + 1):
        avail = int(counts_by_p[p])
        if avail <= 0:
            continue
        k = 1
        remaining = avail
        while remaining > 0:
            take = min(k, remaining)
            weight = take * p
            for s in range(cap, weight - 1, -1):
                if dp[s - weight] >= 0 and dp[s] < dp[s - weight] + 1:
                    dp[s] = dp[s - weight] + 1
                    parent_prev[s] = s - weight
                    parent_p[s] = p
                    parent_cnt[s] = take
            remaining -= take
            k <<= 1

    best_sum = 0
    for s in range(cap, -1, -1):
        if dp[s] >= 0:
            best_sum = int(s)
            break

    chosen = np.zeros_like(counts_by_p)
    s = int(best_sum)
    while s > 0:
        pp = int(parent_p[s])
        if pp <= 0:
            break
        cnt = int(parent_cnt[s])
        chosen[pp] += cnt
        s = int(parent_prev[s])

    return chosen


def _cheap_fit_proxy(
    *,
    job_p_list: np.ndarray,
    slot_levels: np.ndarray,
    cheap_level: int,
    max_p: int,
) -> Tuple[float, float, float]:
    """Compute a cheap-window packing proxy for a machine.

    Returns:
    - fill_ratio: filled_cheap / cheap_capacity
    - leftover_ratio: (cheap_capacity - filled_cheap) / cheap_capacity
    - oversupply_ratio: max(0, load - cheap_capacity) / max(1, load)
    """
    blocks = _rle_blocks_levels(slot_levels)
    cheap_blocks = [int(length) for _s, length, lv in blocks if int(lv) == int(cheap_level)]
    cheap_capacity = float(sum(cheap_blocks))
    load = float(np.sum(job_p_list))
    if cheap_capacity <= 0.0 or load <= 0.0:
        return 0.0, 0.0, 0.0

    # Counts by p for bounded knapsack
    p_clip = np.clip(job_p_list.astype(np.int64), 1, int(max_p))
    counts = np.bincount(p_clip, minlength=int(max_p) + 1).astype(np.int32)

    filled = 0.0
    for L in cheap_blocks:
        if L <= 0:
            continue
        chosen = _bounded_subset_sum_pick_counts(counts, int(L))
        # consume chosen
        counts -= chosen
        filled += float(np.sum(chosen.astype(np.int64) * np.arange(counts.size, dtype=np.int64)))

    fill_ratio = float(filled / max(1e-9, cheap_capacity))
    leftover_ratio = float(max(0.0, cheap_capacity - filled) / max(1e-9, cheap_capacity))
    oversupply_ratio = float(max(0.0, load - cheap_capacity) / max(1.0, load))
    return fill_ratio, leftover_ratio, oversupply_ratio


@dataclass(frozen=True)
class FeatureConfig:
    max_p: int = 20


def extract_assignment_features(
    *,
    instance: RawInstance,
    assignment: Sequence[int],
    K: int,
    config: FeatureConfig,
) -> Dict[str, float]:
    """Rich, size-agnostic features for a full assignment.

    Returns a flat dict of numeric scalars.
    """
    n = int(instance.n)
    m = int(instance.m)
    K = int(K)

    p = np.asarray(instance.p, dtype=np.int64)
    e = np.asarray(instance.e, dtype=np.float64)
    ct = np.asarray(instance.ct[:K], dtype=np.float64)

    if len(assignment) != n:
        raise ValueError(f"Expected assignment length {n}, got {len(assignment)}")

    # Per-machine loads and job counts
    loads = np.zeros(m, dtype=np.int64)
    counts = np.zeros(m, dtype=np.int64)
    for j in range(n):
        mi = int(assignment[j])
        loads[mi] += int(p[j])
        counts[mi] += 1

    load_norm = loads.astype(np.float64) / float(max(1, K))

    # Basic price stats (size-agnostic)
    price_min = float(np.min(ct)) if ct.size else 0.0
    price_max = float(np.max(ct)) if ct.size else 0.0
    price_mean = float(np.mean(ct)) if ct.size else 0.0
    price_std = float(np.std(ct)) if ct.size else 0.0

    extractor = PriceFeatureExtractor(ct, K)
    dist = extractor.get_price_level_distribution().astype(np.float64)
    slot_levels = extractor.slot_levels

    blocks = _price_blocks(ct)
    n_blocks = float(len(blocks))
    if blocks:
        lens = np.array([b[1] for b in blocks], dtype=np.float64)
        avg_block = float(np.mean(lens))
        max_block = float(np.max(lens))
    else:
        avg_block = 0.0
        max_block = 0.0

    # Machine rate stats
    e_norm = e / float(np.max(e) + 1e-9)

    # Correlation-ish signals (helpful when rates differ)
    # Prefer assigning more load to lower e.
    load_share = loads.astype(np.float64) / float(max(1, np.sum(loads)))
    inv_e = 1.0 / (e + 1e-9)
    inv_e_share = inv_e / float(np.sum(inv_e) + 1e-9)
    l1_dist_load_inv_rate = float(np.sum(np.abs(load_share - inv_e_share)))

    # Histogram of processing times (global)
    max_p = int(config.max_p)
    p_clip = np.clip(p, 1, max_p)
    global_hist = np.bincount(p_clip, minlength=max_p + 1).astype(np.float64)
    global_hist = global_hist[1:] / float(max(1, n))

    # Per-machine p-hist aggregated: mean/std across machines per p value
    per_machine_hist = np.zeros((m, max_p), dtype=np.float64)
    for j in range(n):
        mi = int(assignment[j])
        pj = int(p_clip[j])
        per_machine_hist[mi, pj - 1] += 1.0
    per_machine_hist /= np.maximum(1.0, counts[:, None].astype(np.float64))

    hist_mean = per_machine_hist.mean(axis=0)
    hist_std = per_machine_hist.std(axis=0)

    # TOU-specific: cheap-window fit proxy per machine using level-0 blocks.
    # This approximates how well the machine's multiset of job lengths can be
    # packed into the cheap segments (subset-sum style).
    cheap_fill = np.zeros(m, dtype=np.float64)
    cheap_leftover = np.zeros(m, dtype=np.float64)
    cheap_oversupply = np.zeros(m, dtype=np.float64)
    for mi in range(m):
        job_ids = np.nonzero(np.asarray(assignment, dtype=np.int64) == int(mi))[0]
        if job_ids.size == 0:
            continue
        job_p = p[job_ids].astype(np.int64)
        fr, lr, orr = _cheap_fit_proxy(
            job_p_list=job_p,
            slot_levels=slot_levels,
            cheap_level=0,
            max_p=max_p,
        )
        cheap_fill[mi] = fr
        cheap_leftover[mi] = lr
        cheap_oversupply[mi] = orr

    # Cheap-window capacity signal: fraction of horizon at cheapest level
    frac_cheap = float(dist[0]) if dist.size >= 1 else 0.0
    cheap_capacity = frac_cheap * float(K)

    # Load feasibility margin stats
    slack = float(K) - loads.astype(np.float64)
    slack_norm = slack / float(max(1, K))

    feats: Dict[str, float] = {
        "n": float(n),
        "m": float(m),
        "K": float(K),
        "util": float(np.sum(loads)) / float(max(1, m * K)),
        "load_mean": float(np.mean(load_norm)),
        "load_std": float(np.std(load_norm)),
        "load_max": float(np.max(load_norm)),
        "load_min": float(np.min(load_norm)),
        "jobs_mean": float(np.mean(counts)),
        "jobs_std": float(np.std(counts)),
        "rate_mean": float(np.mean(e_norm)),
        "rate_std": float(np.std(e_norm)),
        "rate_max": float(np.max(e_norm)),
        "rate_min": float(np.min(e_norm)),
        "l1_load_vs_inv_rate": float(l1_dist_load_inv_rate),
        "price_min": price_min,
        "price_max": price_max,
        "price_mean": price_mean,
        "price_std": price_std,
        "frac_lvl0": float(dist[0]) if dist.size > 0 else 0.0,
        "frac_lvl1": float(dist[1]) if dist.size > 1 else 0.0,
        "frac_lvl2": float(dist[2]) if dist.size > 2 else 0.0,
        "n_blocks": n_blocks,
        "avg_block": avg_block,
        "max_block": max_block,
        "cheap_capacity_norm": float(cheap_capacity) / float(max(1, np.sum(loads))),
        "slack_mean": float(np.mean(slack_norm)),
        "slack_min": float(np.min(slack_norm)),

        "cheap_fill_mean": float(np.mean(cheap_fill)),
        "cheap_fill_std": float(np.std(cheap_fill)),
        "cheap_fill_min": float(np.min(cheap_fill)),
        "cheap_fill_max": float(np.max(cheap_fill)),

        "cheap_leftover_mean": float(np.mean(cheap_leftover)),
        "cheap_leftover_std": float(np.std(cheap_leftover)),

        "cheap_oversupply_mean": float(np.mean(cheap_oversupply)),
        "cheap_oversupply_std": float(np.std(cheap_oversupply)),
    }

    for i, v in enumerate(global_hist, start=1):
        feats[f"p_hist_global_{i}"] = float(v)
    for i, v in enumerate(hist_mean, start=1):
        feats[f"p_hist_machine_mean_{i}"] = float(v)
    for i, v in enumerate(hist_std, start=1):
        feats[f"p_hist_machine_std_{i}"] = float(v)

    return feats


def dict_to_feature_vector(feats: Dict[str, float], feature_keys: List[str]) -> np.ndarray:
    x = np.zeros(len(feature_keys), dtype=np.float32)
    for i, k in enumerate(feature_keys):
        x[i] = float(feats.get(k, 0.0))
    return x

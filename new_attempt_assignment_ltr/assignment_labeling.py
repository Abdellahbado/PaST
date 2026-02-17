from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from PaST.data.sm_benchmark_data import RawInstance
from PaST.neurols.price_embedding import PriceFeatureExtractor
from PaST.solvers.baselines_sequence_dp import _dp_schedule_fixed_order


@dataclass(frozen=True)
class AssignmentScores:
    l1_total_energy: float
    l2_total_energy: float
    feasible_l1: bool
    feasible_l2: bool
    per_machine_l1: Tuple[float, ...]
    per_machine_l2: Tuple[float, ...]


def _rle_blocks(levels: np.ndarray) -> List[Tuple[int, int, int]]:
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


def _bounded_subset_sum_pick(
    job_ids_by_p: Dict[int, List[int]],
    p_values: Sequence[int],
    capacity: int,
) -> Dict[int, int]:
    cap = int(capacity)
    if cap <= 0:
        return {}

    dp = np.full(cap + 1, -1, dtype=np.int32)
    dp[0] = 0

    parent_prev_sum = np.full(cap + 1, -1, dtype=np.int32)
    parent_p = np.full(cap + 1, -1, dtype=np.int32)
    parent_cnt = np.full(cap + 1, -1, dtype=np.int32)

    for p in p_values:
        p_i = int(p)
        avail = int(len(job_ids_by_p.get(p_i, [])))
        if avail <= 0:
            continue

        k = 1
        remaining = avail
        while remaining > 0:
            take = min(k, remaining)
            weight = take * p_i
            for s in range(cap, weight - 1, -1):
                if dp[s - weight] >= 0 and dp[s] < dp[s - weight] + 1:
                    dp[s] = dp[s - weight] + 1
                    parent_prev_sum[s] = s - weight
                    parent_p[s] = p_i
                    parent_cnt[s] = take
            remaining -= take
            k <<= 1

    best_sum = 0
    for s in range(cap, -1, -1):
        if dp[s] >= 0:
            best_sum = int(s)
            break

    chosen: Dict[int, int] = {}
    s = int(best_sum)
    while s > 0:
        pp = int(parent_p[s])
        if pp < 0:
            break
        cnt = int(parent_cnt[s])
        chosen[pp] = chosen.get(pp, 0) + cnt
        s = int(parent_prev_sum[s])

    return chosen


def _pack_sequence_for_machine(
    job_ids: Sequence[int],
    processing_times: np.ndarray,
    slot_levels: np.ndarray,
    *,
    prefer_levels: Sequence[int],
    seed: Optional[int],
) -> List[int]:
    if not job_ids:
        return []

    rng = np.random.default_rng(seed) if seed is not None else None

    job_ids_by_p: Dict[int, List[int]] = {}
    for j in job_ids:
        pj = int(processing_times[int(j)])
        job_ids_by_p.setdefault(pj, []).append(int(j))

    for pj in job_ids_by_p:
        job_ids_by_p[pj].sort()
        if rng is not None:
            arr = np.array(job_ids_by_p[pj], dtype=np.int64)
            rng.shuffle(arr)
            job_ids_by_p[pj] = [int(x) for x in arr.tolist()]

    p_values = sorted(job_ids_by_p.keys())
    blocks = _rle_blocks(slot_levels)

    packed: List[int] = []

    for level in prefer_levels:
        for _start, length, lv in blocks:
            if int(lv) != int(level):
                continue
            cap = int(length)
            if cap <= 0:
                continue

            pick = _bounded_subset_sum_pick(job_ids_by_p, p_values, cap)
            if not pick:
                continue

            bundle: List[int] = []
            for pj in sorted(pick.keys(), reverse=True):
                cnt = int(pick[pj])
                avail_list = job_ids_by_p.get(int(pj), [])
                take_ids = avail_list[:cnt]
                del avail_list[:cnt]
                if len(avail_list) == 0:
                    job_ids_by_p.pop(int(pj), None)
                    if int(pj) in p_values:
                        p_values = [x for x in p_values if int(x) != int(pj)]
                bundle.extend(take_ids)

            bundle.sort(key=lambda j: (-int(processing_times[int(j)]), int(j)))
            packed.extend(bundle)

    remaining: List[int] = []
    for ids in job_ids_by_p.values():
        remaining.extend(ids)
    remaining.sort(key=lambda j: (-int(processing_times[int(j)]), int(j)))

    packed.extend(remaining)
    return packed


def make_pack_sequence(
    *,
    job_ids: Sequence[int],
    processing_times: np.ndarray,
    ct: np.ndarray,
    K: int,
    mode: str,
    seed: Optional[int] = None,
) -> List[int]:
    extractor = PriceFeatureExtractor(ct, int(K))
    levels = extractor.slot_levels

    m = str(mode).lower()
    if m in ("cheap_first", "cheap", "l0"):
        prefer = (0, 1, 2)
    elif m in ("medium_first", "medium", "l1"):
        prefer = (1, 0, 2)
    else:
        raise ValueError(f"Unknown pack sequence mode: {mode}")

    return _pack_sequence_for_machine(
        job_ids=job_ids,
        processing_times=np.asarray(processing_times, dtype=np.int64),
        slot_levels=levels,
        prefer_levels=prefer,
        seed=seed,
    )


def score_assignment_l1(
    *,
    instance: RawInstance,
    assignment: Sequence[int],
    K: int,
    mode: str = "cheap_first",
) -> Tuple[float, Tuple[float, ...], bool]:
    m = int(instance.m)
    n = int(instance.n)
    K_i = int(K)
    if K_i <= 0 or K_i > int(instance.T_max):
        raise ValueError(f"Invalid K={K_i} for instance T_max={instance.T_max}")

    if len(assignment) != n:
        raise ValueError(f"Expected assignment length {n}, got {len(assignment)}")

    p = np.asarray(instance.p, dtype=np.int64)
    ct = np.asarray(instance.ct[:K_i], dtype=np.int32)

    per_machine: List[float] = []
    total = 0.0
    feasible = True

    jobs_by_machine: List[List[int]] = [[] for _ in range(m)]
    for j, mi in enumerate(assignment):
        mi_i = int(mi)
        if mi_i < 0 or mi_i >= m:
            raise ValueError(f"Invalid machine index in assignment: {mi_i}")
        jobs_by_machine[mi_i].append(int(j))

    for mi in range(m):
        job_ids = jobs_by_machine[mi]
        load = int(sum(int(p[j]) for j in job_ids))
        if load > K_i:
            per_machine.append(float("inf"))
            feasible = False
            continue

        seq = make_pack_sequence(
            job_ids=job_ids,
            processing_times=p,
            ct=ct,
            K=K_i,
            mode=mode,
            seed=None,
        )
        proc_ordered = [int(p[j]) for j in seq]
        energy, _starts = _dp_schedule_fixed_order(
            processing_times=proc_ordered,
            ct=ct,
            e_single=int(instance.e[mi]),
            T_limit=K_i,
            dp_time_penalty=0.0,
        )
        if not np.isfinite(float(energy)):
            per_machine.append(float("inf"))
            feasible = False
            continue
        per_machine.append(float(energy))
        total += float(energy)

    return (float(total) if feasible else float("inf")), tuple(per_machine), bool(feasible)

import random
from dataclasses import dataclass
from typing import List

import numpy as np

from PaST.config import DataConfig
from PaST.data.sm_benchmark_data import generate_raw_instance, RawInstance

from .assignment_labeling import score_assignment_l1
from .feasible_assignment import build_candidate_assignment_pool


@dataclass
class SignalStats:
    n_instances: int
    pool_size: int
    feasible_frac: float
    mean_gap_to_best_pct: float
    median_gap_to_best_pct: float
    p90_gap_to_best_pct: float


def measure_signal(
    *,
    n_instances: int = 30,
    pool_size: int = 50,
    seed: int = 42,
) -> SignalStats:
    cfg = DataConfig()
    rng = random.Random(seed)
    gaps: List[float] = []
    feasible = 0
    total = 0

    for i in range(int(n_instances)):
        raw: RawInstance = generate_raw_instance(cfg, random.Random(seed + i), instance_id=i)

        tmin = int(np.ceil(float(sum(raw.p)) / float(max(1, raw.m))))
        tmin = max(1, min(tmin, int(raw.T_max)))
        u = rng.random()
        K = int(tmin + (raw.T_max - tmin) * (u * u))
        K = max(tmin, min(int(raw.T_max), int(K)))

        pool = build_candidate_assignment_pool(
            processing_times=raw.p,
            n_machines=raw.m,
            K=K,
            pool_size=int(pool_size),
            seed=int(seed + 1_000_000 + i),
        )
        if not pool:
            continue

        scores = []
        for a in pool:
            energy, _pm, feas = score_assignment_l1(instance=raw, assignment=a, K=K, mode="cheap_first")
            total += 1
            if feas and np.isfinite(energy):
                feasible += 1
                scores.append(float(energy))

        if len(scores) < 5:
            continue

        best = float(min(scores))
        med = float(np.median(scores))
        if best > 0 and np.isfinite(best) and np.isfinite(med):
            gaps.append((med - best) / best * 100.0)

    if len(gaps) == 0:
        gaps_arr = np.array([0.0], dtype=np.float64)
    else:
        gaps_arr = np.asarray(gaps, dtype=np.float64)

    return SignalStats(
        n_instances=int(n_instances),
        pool_size=int(pool_size),
        feasible_frac=float(feasible) / float(max(1, total)),
        mean_gap_to_best_pct=float(np.mean(gaps_arr)),
        median_gap_to_best_pct=float(np.median(gaps_arr)),
        p90_gap_to_best_pct=float(np.percentile(gaps_arr, 90)),
    )


def main():
    stats = measure_signal()
    print(stats)


if __name__ == "__main__":
    main()

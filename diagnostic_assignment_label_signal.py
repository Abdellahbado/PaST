import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from PaST.config import DataConfig
from PaST.data.sm_benchmark_data import generate_raw_instance, RawInstance
from PaST.solvers.assignment_labeling import score_assignment_l1_l2


def _balanced_random_assignment(n: int, m: int, rng: random.Random) -> List[int]:
    # Lightweight balanced random assignment: prefer machines with fewer assigned jobs
    assignment: List[int] = []
    counts = [0] * m
    target = float(n) / float(max(1, m))
    for _j in range(n):
        weights = [max(0.1, target - float(c)) for c in counts]
        mi = rng.choices(range(m), weights=weights, k=1)[0]
        assignment.append(int(mi))
        counts[int(mi)] += 1
    return assignment


@dataclass
class SignalStats:
    n_instances: int
    n_assignments: int
    feasible_frac_l1: float
    feasible_frac_l2: float
    mean_gap_pct: float
    median_gap_pct: float
    p90_gap_pct: float
    max_gap_pct: float


def measure_signal(
    *,
    n_instances: int = 20,
    n_assignments: int = 50,
    seed: int = 42,
) -> SignalStats:
    rng0 = random.Random(seed)

    config = DataConfig()
    gaps: List[float] = []
    feas1 = 0
    feas2 = 0
    total = 0

    for i in range(int(n_instances)):
        inst_seed = seed + 10_000 + i
        inst_rng = random.Random(inst_seed)

        raw: RawInstance = generate_raw_instance(config, inst_rng, instance_id=i)

        # Sample a K from the epsilon constraint spectrum.
        # We use K in [T_min_global, T_max] with a bias toward tighter regimes.
        # T_min_global lower bound: ceil(sum(p)/m)
        tmin = int(np.ceil(float(sum(raw.p)) / float(max(1, raw.m))))
        tmin = max(1, min(tmin, int(raw.T_max)))
        # Bias toward tight by sampling u^2
        u = rng0.random()
        K = int(tmin + (raw.T_max - tmin) * (u * u))
        K = max(tmin, min(int(raw.T_max), int(K)))

        arng = random.Random(seed + 20_000 + i)

        for _k in range(int(n_assignments)):
            assignment = _balanced_random_assignment(raw.n, raw.m, arng)
            scores = score_assignment_l1_l2(instance=raw, assignment=assignment, K=K)
            total += 1
            if scores.feasible_l1:
                feas1 += 1
            if scores.feasible_l2:
                feas2 += 1

            if scores.feasible_l1 and scores.feasible_l2 and np.isfinite(scores.l1_total_energy):
                if scores.l1_total_energy > 0 and np.isfinite(scores.l2_total_energy):
                    gap = (scores.l1_total_energy - scores.l2_total_energy) / scores.l1_total_energy
                    gaps.append(float(gap) * 100.0)

    if len(gaps) == 0:
        gaps_arr = np.array([0.0], dtype=np.float64)
    else:
        gaps_arr = np.asarray(gaps, dtype=np.float64)

    return SignalStats(
        n_instances=int(n_instances),
        n_assignments=int(n_assignments),
        feasible_frac_l1=float(feas1) / float(max(1, total)),
        feasible_frac_l2=float(feas2) / float(max(1, total)),
        mean_gap_pct=float(np.mean(gaps_arr)),
        median_gap_pct=float(np.median(gaps_arr)),
        p90_gap_pct=float(np.percentile(gaps_arr, 90)),
        max_gap_pct=float(np.max(gaps_arr)),
    )


def main():
    stats = measure_signal()
    print("=" * 70)
    print("DIAGNOSTIC: L1 vs L2 assignment labeling gap (signal strength)")
    print("=" * 70)
    print(f"Instances:   {stats.n_instances}")
    print(f"Assignments: {stats.n_assignments} per instance")
    print()
    print(f"Feasible L1: {stats.feasible_frac_l1 * 100:.1f}%")
    print(f"Feasible L2: {stats.feasible_frac_l2 * 100:.1f}%")
    print()
    print("Gap = (L1 - L2) / L1  (percent)")
    print(f"Mean:   {stats.mean_gap_pct:.2f}%")
    print(f"Median: {stats.median_gap_pct:.2f}%")
    print(f"P90:    {stats.p90_gap_pct:.2f}%")
    print(f"Max:    {stats.max_gap_pct:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()

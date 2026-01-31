"""Benchmark which DP acceleration ideas actually help.

Runs a local micro-benchmark around the real hot path:
`PaST.solvers.baselines_sequence_dp._dp_schedule_fixed_order`.

We compare:
1) baseline (full DP + backtrack start times)
2) rolling DP (cost-only) with prefix sums computed per call
3) rolling DP with prefix sums precomputed once
4) rolling DP with prefix sums + memoization on the *processing-time sequence*

Why memoize on processing times?
The DP only sees durations, not job IDs. With small duration alphabets
(e.g. p in {1..4}), different job permutations can map to identical duration
sequences, so caching can sometimes be surprisingly effective.

Usage:
  conda run -n new-ml-env python -m PaST.cli.diagnostic.bench_dp_accel --help
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

from PaST.config import DataConfig
from PaST.data.sm_benchmark_data import generate_raw_instance
from PaST.solvers.baselines_sequence_dp import _dp_schedule_fixed_order
from PaST.solvers.dp_fast import (
    dp_cost_fixed_order_rolling,
    dp_cost_fixed_order_rolling_precomputed,
    prefix_costs_from_ct,
)
from PaST.solvers.alns_parallel import _compute_epsilon

# Optional: torch batched DP (already in repo). This can be a major speed lever.
try:
    import torch

    from PaST.solvers.batch_dp_solver import BatchSequenceDPSolver

    _TORCH_OK = True
except Exception:
    torch = None  # type: ignore
    BatchSequenceDPSolver = None  # type: ignore
    _TORCH_OK = False


def _timeit(fn, warmup: int = 3, repeat: int = 1) -> Tuple[float, float]:
    # Returns (mean_s, min_s)
    for _ in range(max(0, int(warmup))):
        fn()
    times: List[float] = []
    for _ in range(max(1, int(repeat))):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.min(times))


def _make_proc_sequences(
    base_proc: Sequence[int],
    n_queries: int,
    rng: random.Random,
) -> List[List[int]]:
    # Permute job order; processing times include duplicates.
    base = list(int(x) for x in base_proc)
    out: List[List[int]] = []
    for _ in range(int(n_queries)):
        v = base.copy()
        rng.shuffle(v)
        out.append(v)
    return out


def _sample_proc_from_instance(
    raw, J: int, rng: random.Random
) -> Tuple[List[int], int, np.ndarray]:
    n = int(raw.n)
    m = int(raw.m)
    J = int(min(max(1, J), n))

    # Pick a random subset of jobs for a pseudo-machine.
    jobs = list(range(n))
    rng.shuffle(jobs)
    jobs = jobs[:J]

    proc = [int(raw.p[j]) for j in jobs]

    # Pick a machine energy rate (use an existing machine).
    machine_idx = int(rng.randrange(max(1, m)))
    e_single = int(raw.e[machine_idx])

    ct = np.asarray(raw.ct, dtype=np.int32)
    return proc, e_single, ct


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--seed", type=int, default=0)

    # Instance shape (used only to sample realistic p, ct, e)
    ap.add_argument("--T_max", type=int, default=300)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--n", type=int, default=100)

    # Machine sequence length to DP-evaluate
    ap.add_argument("--J", type=int, default=25)

    # Epsilon slack ratio used to compute a single deadline (epsilon)
    ap.add_argument("--slack", type=float, default=0.3)

    # Benchmark load
    ap.add_argument("--n_queries", type=int, default=2000)
    ap.add_argument("--repeat", type=int, default=3)

    # Cache sizing
    ap.add_argument("--cache_max", type=int, default=20000)

    args = ap.parse_args()

    rng = random.Random(int(args.seed))

    cfg = DataConfig()
    raw = generate_raw_instance(
        config=cfg,
        rng=rng,
        instance_id=0,
        T_max=int(args.T_max),
        m=int(args.m),
        n=int(args.n),
    )

    epsilon = _compute_epsilon(raw, slack_ratio=float(args.slack))

    proc, e_single, ct = _sample_proc_from_instance(raw, J=int(args.J), rng=rng)

    # Ensure deadline is feasible for this proc set.
    min_feasible = int(sum(proc))
    T_limit = int(max(min_feasible, min(int(epsilon), int(raw.T_max))))
    if T_limit < min_feasible:
        T_limit = int(min_feasible)

    seqs = _make_proc_sequences(proc, n_queries=int(args.n_queries), rng=rng)

    # For memoization we key on processing time tuples.
    unique = len({tuple(s) for s in seqs})
    dup_rate = 1.0 - float(unique) / float(len(seqs))

    print("=== DP Acceleration Benchmark ===")
    print(f"seed={args.seed}  instance(T={raw.T_max}, m={raw.m}, n={raw.n})")
    print(f"DP scenario: J={len(proc)}  T_limit={T_limit}  e_single={e_single}")
    print(f"queries={len(seqs)}  unique_proc_seqs={unique}  dup_rate={dup_rate:.3f}")

    # Precompute prefix costs once.
    prefix = prefix_costs_from_ct(ct, T_limit)

    # 1) baseline full DP
    def run_baseline_full() -> float:
        s = 0.0
        for p in seqs:
            cost, _st = _dp_schedule_fixed_order(p, ct, e_single, T_limit, 0.0)
            s += float(cost)
        return s

    # 2) rolling DP with per-call prefix
    def run_rolling_percall() -> float:
        s = 0.0
        for p in seqs:
            cost, _t = dp_cost_fixed_order_rolling(p, ct, e_single, T_limit)
            s += float(cost)
        return s

    # 3) rolling DP with precomputed prefix
    def run_rolling_precomp() -> float:
        s = 0.0
        for p in seqs:
            cost, _t = dp_cost_fixed_order_rolling_precomputed(
                p, prefix, e_single, T_limit
            )
            s += float(cost)
        return s

    # 4) rolling DP with memoization (processing-time sequence)
    def run_rolling_memo() -> float:
        cache: Dict[Tuple[int, ...], Tuple[float, int]] = {}
        s = 0.0
        hits = 0
        max_sz = int(args.cache_max)
        for p in seqs:
            k = tuple(int(x) for x in p)
            v = cache.get(k)
            if v is not None:
                hits += 1
                cost, _t = v
            else:
                cost, _t = dp_cost_fixed_order_rolling_precomputed(
                    p, prefix, e_single, T_limit
                )
                if len(cache) < max_sz:
                    cache[k] = (float(cost), int(_t))
            s += float(cost)
        # print hit rate once per run
        print(f"memo_hit_rate={hits / float(len(seqs)):.3f}  cache_size={len(cache)}")
        return s

    # 5) torch batched DP
    if _TORCH_OK:
        # Represent each duration sequence as a per-row processing_times tensor,
        # with a fixed identity job_sequence [0..J-1].
        job_seq = (
            torch.arange(len(proc), dtype=torch.long).unsqueeze(0).repeat(len(seqs), 1)
        )
        p_t = torch.tensor(seqs, dtype=torch.long)
        ct_t = torch.tensor(np.asarray(ct[:T_limit], dtype=np.int32), dtype=torch.int32)
        ct_t = ct_t.unsqueeze(0).repeat(len(seqs), 1)
        e_t = torch.full((len(seqs),), int(e_single), dtype=torch.long)
        T_t = torch.full((len(seqs),), int(T_limit), dtype=torch.long)

        def run_torch_batch_eval_only() -> float:
            costs = BatchSequenceDPSolver.solve(job_seq, p_t, ct_t, e_t, T_t)
            return float(costs.sum().item())

        def run_torch_batch_total() -> float:
            job_seq2 = (
                torch.arange(len(proc), dtype=torch.long)
                .unsqueeze(0)
                .repeat(len(seqs), 1)
            )
            p_t2 = torch.tensor(seqs, dtype=torch.long)
            ct_t2 = (
                torch.tensor(
                    np.asarray(ct[:T_limit], dtype=np.int32), dtype=torch.int32
                )
                .unsqueeze(0)
                .repeat(len(seqs), 1)
            )
            e_t2 = torch.full((len(seqs),), int(e_single), dtype=torch.long)
            T_t2 = torch.full((len(seqs),), int(T_limit), dtype=torch.long)
            costs = BatchSequenceDPSolver.solve(job_seq2, p_t2, ct_t2, e_t2, T_t2)
            return float(costs.sum().item())

    # Correctness sanity: compare sums (allow tiny float differences)
    base_sum = run_baseline_full()
    roll_sum = run_rolling_precomp()
    if not (np.isfinite(base_sum) and np.isfinite(roll_sum)):
        print("WARNING: non-finite sums detected; deadline may be too tight.")
    else:
        rel = abs(base_sum - roll_sum) / max(1.0, abs(base_sum))
        print(
            f"sanity: baseline_sum={base_sum:.3e} rolling_sum={roll_sum:.3e} rel_diff={rel:.3e}"
        )

    print("--- Timing (mean / min over repeats) ---")
    mean_s, min_s = _timeit(run_baseline_full, warmup=1, repeat=int(args.repeat))
    print(f"baseline_full:        mean={mean_s:.3f}s  min={min_s:.3f}s")

    mean_s2, min_s2 = _timeit(run_rolling_percall, warmup=1, repeat=int(args.repeat))
    print(
        f"rolling_percall:      mean={mean_s2:.3f}s  min={min_s2:.3f}s  speedup_vs_base={mean_s/mean_s2:.2f}x"
    )

    mean_s3, min_s3 = _timeit(run_rolling_precomp, warmup=1, repeat=int(args.repeat))
    print(
        f"rolling_precomputed:  mean={mean_s3:.3f}s  min={min_s3:.3f}s  speedup_vs_base={mean_s/mean_s3:.2f}x"
    )

    mean_s4, min_s4 = _timeit(run_rolling_memo, warmup=1, repeat=int(args.repeat))
    print(
        f"rolling_memo:         mean={mean_s4:.3f}s  min={min_s4:.3f}s  speedup_vs_base={mean_s/mean_s4:.2f}x"
    )

    if _TORCH_OK:
        mean_s5, min_s5 = _timeit(
            run_torch_batch_eval_only, warmup=1, repeat=int(args.repeat)
        )
        print(
            f"torch_batch_eval:     mean={mean_s5:.3f}s  min={min_s5:.3f}s  speedup_vs_base={mean_s/mean_s5:.2f}x"
        )

        mean_s6, min_s6 = _timeit(
            run_torch_batch_total, warmup=1, repeat=int(args.repeat)
        )
        print(
            f"torch_batch_total:    mean={mean_s6:.3f}s  min={min_s6:.3f}s  speedup_vs_base={mean_s/mean_s6:.2f}x"
        )
    else:
        print("torch not available; skipping BatchSequenceDPSolver benchmark")


if __name__ == "__main__":
    main()

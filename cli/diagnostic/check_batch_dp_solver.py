"""Correctness check for BatchSequenceDPSolver against the numpy DP.

This validates that:
- Costs match `_dp_schedule_fixed_order` for many random sequences.
- The completion time inferred by `solve_with_end_time` matches the completion
  time from the numpy DP backtracked schedule.

Run with:
  /opt/miniconda3/envs/new-ml-env/bin/python -u -m PaST.cli.diagnostic.check_batch_dp_solver
"""

from __future__ import annotations

import argparse
import random
from typing import List

import numpy as np

from PaST.config import DataConfig
from PaST.data.sm_benchmark_data import generate_raw_instance
from PaST.solvers.baselines_sequence_dp import _dp_schedule_fixed_order


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        from PaST.solvers.batch_dp_solver import BatchSequenceDPSolver  # noqa: F401

        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_instances", type=int, default=5)
    ap.add_argument("--n_sequences", type=int, default=200)
    ap.add_argument("--T_max", type=int, default=300)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--J", type=int, default=25)
    ap.add_argument("--eps", type=int, default=300)
    ap.add_argument("--cost_tol", type=float, default=1e-3)
    args = ap.parse_args()

    if not _torch_available():
        raise SystemExit("Torch/BatchSequenceDPSolver not available in this env")

    import torch
    from PaST.solvers.batch_dp_solver import BatchSequenceDPSolver

    rng = random.Random(int(args.seed))
    cfg = DataConfig()

    total = 0
    cost_bad = 0
    ms_bad = 0
    infeas_mismatch = 0

    max_cost_diff = 0.0
    max_ms_diff = 0

    for inst_id in range(int(args.n_instances)):
        raw = generate_raw_instance(
            config=cfg,
            rng=rng,
            instance_id=inst_id,
            T_max=int(args.T_max),
            m=int(args.m),
            n=int(args.n),
        )
        ct = np.asarray(raw.ct, dtype=np.int32)
        epsilon = int(min(int(args.eps), int(raw.T_max)))
        if epsilon <= 0:
            epsilon = int(raw.T_max)

        for _ in range(int(args.n_sequences)):
            # Random subset of J jobs and random permutation.
            jobs = list(range(int(raw.n)))
            rng.shuffle(jobs)
            jobs = jobs[: int(args.J)]

            proc_base = [int(raw.p[j]) for j in jobs]
            perm = list(range(len(proc_base)))
            rng.shuffle(perm)
            proc = [proc_base[i] for i in perm]

            # Numpy DP (cost + schedule)
            cost_np, st = _dp_schedule_fixed_order(
                processing_times=proc,
                ct=ct,
                e_single=int(raw.e[0]),
                T_limit=epsilon,
                dp_time_penalty=0.0,
            )
            if not st:
                ms_np = 0
            else:
                ms_np = int(st[-1]) + int(proc[-1])

            # Torch batch DP (B=1) on duration-sequence
            N = len(proc)
            job_seq = torch.arange(N, dtype=torch.long).unsqueeze(0)
            p_t = torch.tensor([proc], dtype=torch.long)
            ct_t = torch.tensor(ct[:epsilon], dtype=torch.int32).unsqueeze(0)
            e_t = torch.tensor([int(raw.e[0])], dtype=torch.long)
            T_t = torch.tensor([epsilon], dtype=torch.long)

            cost_t, end_t = BatchSequenceDPSolver.solve_with_end_time(
                job_sequences=job_seq,
                processing_times=p_t,
                ct=ct_t,
                e_single=e_t,
                T_limit=T_t,
            )
            cost_th = float(cost_t.item())
            ms_th = int(end_t.item())

            total += 1

            np_finite = np.isfinite(float(cost_np))
            th_finite = np.isfinite(float(cost_th))
            if np_finite != th_finite:
                infeas_mismatch += 1
                continue

            if np_finite:
                diff = abs(float(cost_np) - float(cost_th))
                max_cost_diff = max(max_cost_diff, diff)
                if diff > float(args.cost_tol):
                    cost_bad += 1

                dms = abs(int(ms_np) - int(ms_th))
                max_ms_diff = max(max_ms_diff, dms)
                if dms != 0:
                    ms_bad += 1

    print("=== Batch DP Solver Correctness ===")
    print(f"total_cases={total}")
    print(f"infeasibility_mismatch={infeas_mismatch}")
    print(
        f"cost_mismatch={cost_bad}  max_cost_abs_diff={max_cost_diff:.6g}  tol={args.cost_tol}"
    )
    print(f"makespan_mismatch={ms_bad}  max_ms_abs_diff={max_ms_diff}")

    if infeas_mismatch == 0 and cost_bad == 0 and ms_bad == 0:
        print("OK")
    else:
        print("NOT_OK")


if __name__ == "__main__":
    main()

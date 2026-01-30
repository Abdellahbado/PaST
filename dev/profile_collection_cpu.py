#!/usr/bin/env python3
"""Local CPU utilization profiler for Q-sequence collection.

Purpose
-------
When collection uses CPU-parallel DP (`--dp_eval_device cpu --dp_eval_workers > 1`),
we want to verify that all cores are actually being used (e.g. ~400% CPU on Kaggle).

This script:
- Runs a single collection call in a child process.
- Samples CPU% of the child process + all its descendants using macOS `ps/pgrep`.
- Reports mean/max CPU%, plus simple timing stats.

Example
-------
  conda activate new-ml-env
  python PaST/dev/profile_collection_cpu.py \
    --variant_id q_sequence_ctx13 \
    --num_episodes 256 \
    --collection_batch_size 64 \
    --dp_workers 4 \
    --dp_flush_threshold 2048

Notes
-----
- On macOS, `ps` CPU% is per-core, so totals can exceed 100%.
- Keep `--collection_workers 1` while profiling DP pool saturation.
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import time
from dataclasses import asdict
from multiprocessing import Process, Queue
from typing import Dict, Iterable, List, Set, Tuple


def _run_cmd_lines(cmd: List[str]) -> List[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    txt = out.decode("utf-8", errors="ignore").strip()
    if not txt:
        return []
    return [line.strip() for line in txt.splitlines() if line.strip()]


def _descendant_pids(root_pid: int) -> Set[int]:
    """Return root_pid + all descendants (best-effort) using `pgrep -P`."""
    seen: Set[int] = set()
    stack: List[int] = [int(root_pid)]

    while stack:
        pid = int(stack.pop())
        if pid in seen:
            continue
        seen.add(pid)

        children = _run_cmd_lines(["pgrep", "-P", str(pid)])
        for c in children:
            try:
                stack.append(int(c))
            except Exception:
                pass

    return seen


def _cpu_percent_for_pids(pids: Iterable[int]) -> float:
    """Sum CPU% for a set of PIDs via `ps -o %cpu=`."""
    pids_list = [int(p) for p in pids]
    if not pids_list:
        return 0.0

    # macOS ps supports comma-separated -p list.
    pid_arg = ",".join(str(p) for p in pids_list)
    lines = _run_cmd_lines(["ps", "-o", "%cpu=", "-p", pid_arg])
    total = 0.0
    for ln in lines:
        try:
            total += float(ln.strip())
        except Exception:
            continue
    return float(total)


def _collection_worker(entry_args: Dict[str, object], out_q: Queue) -> None:
    """Child process entrypoint that runs collection."""
    # Import inside child to avoid any multiprocessing pickling surprises.
    from PaST.config import VariantID, get_variant_config
    from PaST.train_q_sequence import collect_round_data

    variant_id = str(entry_args["variant_id"])
    seed = int(entry_args["seed"])

    variant_config = get_variant_config(VariantID(variant_id))
    env_config = variant_config.env
    data_config = copy.deepcopy(variant_config.data)

    # Always use CPU here: we want to measure CPU DP pool saturation.
    import torch

    device = torch.device("cpu")

    t0 = time.time()
    transitions = collect_round_data(
        env_config=env_config,
        model=None,
        teacher_model=None,
        variant_config=variant_config,
        data_config=data_config,
        num_episodes=int(entry_args["num_episodes"]),
        num_counterfactuals=int(entry_args["num_counterfactuals"]),
        exploration_eps=float(entry_args["exploration_eps"]),
        use_model_completion=False,
        heuristic_policy=str(entry_args["heuristic_policy"]),
        target_normalization=str(entry_args["target_normalization"]),
        include_heuristic_candidates=bool(entry_args["include_heuristic_candidates"]),
        target_rollouts=str(entry_args["target_rollouts"]),
        target_rollout_aggregation=str(entry_args["target_rollout_aggregation"]),
        target_num_random_rollouts=int(entry_args["target_num_random_rollouts"]),
        target_softmin_tau=float(entry_args["target_softmin_tau"]),
        device=device,
        seed=seed,
        collection_batch_size=int(entry_args["collection_batch_size"]),
        num_collection_workers=int(entry_args["collection_workers"]),
        allow_gpu_collection_multiprocessing=False,
        num_cpu_threads=int(entry_args["num_cpu_threads"]),
        dp_eval_device=str(entry_args["dp_eval_device"]),
        dp_eval_workers=int(entry_args["dp_workers"]),
        dp_flush_threshold=int(entry_args["dp_flush_threshold"]),
        dp_eval_async=bool(entry_args.get("dp_eval_async", False)),
    )
    dt = float(time.time() - t0)

    out_q.put(
        {
            "ok": True,
            "seconds": dt,
            "num_transitions": int(len(transitions)),
            "transitions_per_s": float(len(transitions) / max(1e-9, dt)),
        }
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--variant_id", type=str, default="q_sequence")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--num_episodes", type=int, default=256)
    p.add_argument("--collection_batch_size", type=int, default=64)
    p.add_argument("--collection_workers", type=int, default=1)

    p.add_argument("--num_counterfactuals", type=int, default=8)
    p.add_argument("--exploration_eps", type=float, default=0.2)

    p.add_argument("--heuristic_policy", type=str, default="mixed")
    p.add_argument("--include_heuristic_candidates", action="store_true")

    p.add_argument("--target_normalization", type=str, default="none")
    p.add_argument("--target_rollouts", type=str, default="auto")
    p.add_argument("--target_rollout_aggregation", type=str, default="min")
    p.add_argument("--target_num_random_rollouts", type=int, default=2)
    p.add_argument("--target_softmin_tau", type=float, default=1.0)

    # Collection/DP tuning
    p.add_argument("--num_cpu_threads", type=int, default=0)
    p.add_argument(
        "--dp_eval_device", type=str, default="cpu", choices=["cpu", "cuda", "auto"]
    )
    p.add_argument("--dp_workers", type=int, default=4)
    p.add_argument("--dp_flush_threshold", type=int, default=2048)
    p.add_argument(
        "--dp_eval_async",
        action="store_true",
        help="Enable async DP (only meaningful when --dp_workers 1).",
    )

    p.add_argument("--sample_interval_s", type=float, default=0.5)

    p.add_argument(
        "--ignore_first_s",
        type=float,
        default=1.0,
        help="Ignore the first N seconds when computing steady-state CPU stats.",
    )
    p.add_argument(
        "--high_cpu_threshold",
        type=float,
        default=350.0,
        help="CPU% threshold used for reporting fraction of time at high utilization.",
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Ensure repo root import works when run as a script.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    entry_args = {
        "variant_id": str(args.variant_id),
        "seed": int(args.seed),
        "num_episodes": int(args.num_episodes),
        "collection_batch_size": int(args.collection_batch_size),
        "collection_workers": int(args.collection_workers),
        "num_counterfactuals": int(args.num_counterfactuals),
        "exploration_eps": float(args.exploration_eps),
        "heuristic_policy": str(args.heuristic_policy),
        "include_heuristic_candidates": bool(args.include_heuristic_candidates),
        "target_normalization": str(args.target_normalization),
        "target_rollouts": str(args.target_rollouts),
        "target_rollout_aggregation": str(args.target_rollout_aggregation),
        "target_num_random_rollouts": int(args.target_num_random_rollouts),
        "target_softmin_tau": float(args.target_softmin_tau),
        "num_cpu_threads": int(args.num_cpu_threads),
        "dp_eval_device": str(args.dp_eval_device),
        "dp_workers": int(args.dp_workers),
        "dp_flush_threshold": int(args.dp_flush_threshold),
        "dp_eval_async": bool(args.dp_eval_async),
    }

    print("Profiling collection CPU utilization")
    print(f"Config: {entry_args}")

    q: Queue = Queue()
    proc = Process(target=_collection_worker, args=(entry_args, q))
    proc.start()

    samples: List[Tuple[float, float, int]] = []  # (t, cpu_total, pid_count)

    t_start = time.time()
    try:
        while proc.is_alive():
            pids = _descendant_pids(proc.pid)
            cpu_total = _cpu_percent_for_pids(pids)
            samples.append((time.time() - t_start, cpu_total, len(pids)))
            time.sleep(float(args.sample_interval_s))
    finally:
        proc.join(timeout=5)

    result = None
    try:
        result = q.get_nowait()
    except Exception:
        result = {"ok": False}

    if samples:
        cpu_vals = [c for _t, c, _n in samples]
        pid_counts = [n for _t, _c, n in samples]
        mean_cpu = sum(cpu_vals) / len(cpu_vals)
        max_cpu = max(cpu_vals)
        mean_pids = sum(pid_counts) / len(pid_counts)

        ignore_s = max(0.0, float(args.ignore_first_s))
        steady = [(t, c, n) for (t, c, n) in samples if float(t) >= ignore_s]
        if steady:
            steady_cpu = [c for _t, c, _n in steady]
            steady_mean = sum(steady_cpu) / len(steady_cpu)
            steady_max = max(steady_cpu)
            thr = float(args.high_cpu_threshold)
            frac_high = sum(1 for v in steady_cpu if v >= thr) / float(len(steady_cpu))
        else:
            steady_mean = mean_cpu
            steady_max = max_cpu
            frac_high = 0.0
    else:
        mean_cpu = 0.0
        max_cpu = 0.0
        mean_pids = 0.0
        steady_mean = 0.0
        steady_max = 0.0
        frac_high = 0.0

    print("\n=== CPU Utilization Summary (child + descendants) ===")
    print(f"Mean CPU%: {mean_cpu:.1f}")
    print(f"Max  CPU%: {max_cpu:.1f}")
    print(
        f"Steady-state (t>={float(args.ignore_first_s):.1f}s) mean/max CPU%: "
        f"{steady_mean:.1f} / {steady_max:.1f}"
    )
    print(
        f"Frac steady samples >= {float(args.high_cpu_threshold):.0f}% CPU: {frac_high*100:.1f}%"
    )
    print(f"Mean PID count (incl. child): {mean_pids:.1f}")

    if isinstance(result, dict) and result.get("ok"):
        print("\n=== Collection Result ===")
        print(f"Seconds: {float(result['seconds']):.2f}")
        print(f"Transitions: {int(result['num_transitions'])}")
        print(f"Transitions/sec: {float(result['transitions_per_s']):.1f}")
    else:
        print("\nCollection process did not return a result (it may have crashed).")

    # Helpful hint on what to expect
    cores = os.cpu_count() or 1
    print("\n=== Interpretation ===")
    print(
        "On macOS, summed CPU% can exceed 100%. Rough rule: ~100% per saturated core.\n"
        f"This machine reports os.cpu_count()={cores}.\n"
        "For Kaggle 4 cores, you want mean CPU% near 350-400% during collection."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

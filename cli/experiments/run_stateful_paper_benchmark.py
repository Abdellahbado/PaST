"""Generate and run the Section 5.1 synthetic stateful benchmark.

Examples:

Generate the standard paper-comparable synthetic dataset:
  PYTHONPATH=/Users/mac/Documents/Study/PFE \
  /Users/mac/Documents/Study/PFE/PaST/.venv/bin/python \
  -m PaST.cli.experiments.run_stateful_paper_benchmark \
  --generate-only \
  --dataset-out artifacts/paper_stateful_benchmark.json

Generate and solve the default dataset with the exact DP:
  PYTHONPATH=/Users/mac/Documents/Study/PFE \
  /Users/mac/Documents/Study/PFE/PaST/.venv/bin/python \
  -m PaST.cli.experiments.run_stateful_paper_benchmark \
  --dataset-out artifacts/paper_stateful_benchmark.json \
  --results-out artifacts/paper_stateful_benchmark_results.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Iterable, List

import numpy as np

from PaST.data.stateful_paper_benchmark import (
    StatefulPaperBenchmarkDataset,
    generate_stateful_paper_dataset,
    get_paper_machine_config,
    load_stateful_paper_dataset,
    save_stateful_paper_dataset,
)
from PaST.solvers.optimal_benchmark_dp import solve_optimal_benchmark_dp


def _parse_int_list(text: str) -> List[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _parse_float_list(text: str) -> List[float]:
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def _ensure_dataset(args: argparse.Namespace) -> StatefulPaperBenchmarkDataset:
    if args.dataset:
        return load_stateful_paper_dataset(args.dataset)

    dataset = generate_stateful_paper_dataset(
        seeds=_parse_int_list(args.seeds),
        n_values=_parse_int_list(args.n_values),
        lambda_values=_parse_float_list(args.lambda_values),
        machine_model=args.machine_model,
    )
    save_stateful_paper_dataset(dataset, args.dataset_out)
    return dataset


def _write_results(path: str | Path, rows: Iterable[dict]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames = [
        "instance_id",
        "machine_model",
        "seed",
        "n_jobs",
        "lambda_scale",
        "horizon",
        "cost",
        "finish_time",
        "feasible",
        "is_optimal",
        "timed_out",
        "runtime_sec",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="")
    ap.add_argument(
        "--dataset-out",
        type=str,
        default="artifacts/paper_stateful_benchmark.json",
    )
    ap.add_argument(
        "--results-out",
        type=str,
        default="artifacts/paper_stateful_benchmark_results.csv",
    )
    ap.add_argument("--generate-only", action="store_true")
    ap.add_argument("--machine-model", type=str, default="paper_nosby")
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--n-values", type=str, default="150,170,190")
    ap.add_argument("--lambda-values", type=str, default="1.3,1.6,1.9,2.2")
    ap.add_argument("--time-limit", type=float, default=3600.0)
    ap.add_argument("--tie-break", choices=["cost", "early"], default="early")
    ap.add_argument("--track-schedule", action="store_true")
    ap.add_argument("--max-states", type=int, default=0)
    ap.add_argument("--max-instances", type=int, default=0)
    args = ap.parse_args()

    dataset = _ensure_dataset(args)
    print(
        f"dataset={args.dataset or args.dataset_out} instances={len(dataset.instances)} machine_model={dataset.machine_model}",
        flush=True,
    )

    if args.generate_only:
        return

    machine_config = get_paper_machine_config(dataset.machine_model)
    rows = []
    instances = dataset.instances
    if args.max_instances > 0:
        instances = instances[: int(args.max_instances)]

    for instance in instances:
        prices = np.asarray(instance.prices, dtype=np.float64)
        t0 = time.perf_counter()
        res = solve_optimal_benchmark_dp(
            processing_times=instance.processing_times,
            prices=prices,
            tie_break=args.tie_break,
            time_limit=float(args.time_limit),
            track_schedule=bool(args.track_schedule),
            max_states=int(args.max_states),
            machine_config=machine_config,
        )
        runtime_sec = time.perf_counter() - t0

        row = {
            "instance_id": instance.instance_id,
            "machine_model": dataset.machine_model,
            "seed": instance.seed,
            "n_jobs": instance.n_jobs,
            "lambda_scale": instance.lambda_scale,
            "horizon": instance.horizon,
            "cost": float(res.cost),
            "finish_time": int(res.finish_time),
            "feasible": bool(res.feasible),
            "is_optimal": bool(res.is_optimal),
            "timed_out": bool(res.timed_out),
            "runtime_sec": float(runtime_sec),
        }
        rows.append(row)
        print(
            " ".join(
                [
                    f"instance={instance.instance_id}",
                    f"n={instance.n_jobs}",
                    f"lambda={instance.lambda_scale:.1f}",
                    f"h={instance.horizon}",
                    f"feasible={res.feasible}",
                    f"optimal={res.is_optimal}",
                    f"timeout={res.timed_out}",
                    f"cost={res.cost:.6f}",
                    f"runtime={runtime_sec:.3f}s",
                ]
            ),
            flush=True,
        )

    _write_results(args.results_out, rows)
    print(f"results={args.results_out}", flush=True)


if __name__ == "__main__":
    main()

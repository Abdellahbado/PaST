"""Synthetic benchmark generation for 1, TOU|states|TEC paper comparisons.

This module implements the publicly reproducible synthetic benchmark recipe
described in Section 5.1 of the attached paper "Green Scheduling with Time-of-
Use Tariffs and Machine States".

Implemented benchmark family:
- Processing times sampled iid from discrete uniform U{1, 2, 3, 4, 5}
- Interval energy prices sampled iid from discrete uniform U{1, ..., 10}
- Horizon computed as
      h = ceil(lambda * (T(off, proc) + sum_j p_j + T(proc, off)))
  with lambda in {1.3, 1.6, 1.9, 2.2}

The attached paper explicitly specifies the Figure 2 machine model for this
synthetic setup. In this codebase that model is exposed as
``MachineStateConfig.paper_nosby()``.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from PaST.solvers.machine_states import MachineStateConfig


@dataclass(frozen=True)
class StatefulPaperBenchmarkInstance:
    instance_id: str
    seed: int
    n_jobs: int
    lambda_scale: float
    horizon: int
    machine_model: str
    processing_times: List[int]
    prices: List[int]


@dataclass(frozen=True)
class StatefulPaperBenchmarkDataset:
    benchmark_name: str
    source: str
    machine_model: str
    seeds: List[int]
    n_values: List[int]
    lambda_values: List[float]
    instances: List[StatefulPaperBenchmarkInstance]


def get_paper_machine_config(machine_model: str) -> MachineStateConfig:
    name = str(machine_model).strip().lower()
    if name in {"paper_nosby", "nosby_paper", "figure2", "shrouf2014"}:
        return MachineStateConfig.paper_nosby()
    raise ValueError(
        "Unsupported paper benchmark machine model: "
        f"{machine_model}. Supported: paper_nosby"
    )


def compute_paper_horizon(
    processing_times: Sequence[int],
    lambda_scale: float,
    machine_config: MachineStateConfig,
) -> int:
    startup = machine_config.get_transition_time(
        machine_config.off_state, machine_config.proc_state
    )
    shutdown = machine_config.get_transition_time(
        machine_config.proc_state, machine_config.off_state
    )
    if startup is None or shutdown is None:
        raise ValueError("Machine model must support off->proc and proc->off")

    lower_bound = (
        int(startup) + int(sum(int(p) for p in processing_times)) + int(shutdown)
    )
    return int(math.ceil(float(lambda_scale) * float(lower_bound)))


def generate_stateful_paper_instance(
    *,
    seed: int,
    n_jobs: int,
    lambda_scale: float,
    machine_model: str = "paper_nosby",
) -> StatefulPaperBenchmarkInstance:
    rng = random.Random(int(seed))
    processing_times = [rng.randint(1, 5) for _ in range(int(n_jobs))]
    machine_config = get_paper_machine_config(machine_model)
    horizon = compute_paper_horizon(processing_times, lambda_scale, machine_config)
    prices = [rng.randint(1, 10) for _ in range(horizon)]
    instance_id = f"{machine_model}_n{n_jobs}_lam{lambda_scale:.1f}_s{seed}"
    return StatefulPaperBenchmarkInstance(
        instance_id=instance_id,
        seed=int(seed),
        n_jobs=int(n_jobs),
        lambda_scale=float(lambda_scale),
        horizon=int(horizon),
        machine_model=str(machine_model),
        processing_times=processing_times,
        prices=prices,
    )


def generate_stateful_paper_dataset(
    *,
    seeds: Iterable[int],
    n_values: Sequence[int] = (150, 170, 190),
    lambda_values: Sequence[float] = (1.3, 1.6, 1.9, 2.2),
    machine_model: str = "paper_nosby",
) -> StatefulPaperBenchmarkDataset:
    instances: List[StatefulPaperBenchmarkInstance] = []
    for seed in seeds:
        seed_i = int(seed)
        for n_jobs in n_values:
            for lambda_scale in lambda_values:
                derived_seed = (
                    seed_i * 1_000_003
                    + int(n_jobs) * 10_007
                    + int(round(float(lambda_scale) * 10)) * 101
                )
                instances.append(
                    generate_stateful_paper_instance(
                        seed=derived_seed,
                        n_jobs=int(n_jobs),
                        lambda_scale=float(lambda_scale),
                        machine_model=machine_model,
                    )
                )

    return StatefulPaperBenchmarkDataset(
        benchmark_name="paper_section_5_1_synthetic",
        source="arXiv:2506.10405 Section 5.1 synthetic benchmark recipe",
        machine_model=str(machine_model),
        seeds=[int(x) for x in seeds],
        n_values=[int(x) for x in n_values],
        lambda_values=[float(x) for x in lambda_values],
        instances=instances,
    )


def save_stateful_paper_dataset(
    dataset: StatefulPaperBenchmarkDataset,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(dataset)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_stateful_paper_dataset(
    input_path: str | Path,
) -> StatefulPaperBenchmarkDataset:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    instances = [
        StatefulPaperBenchmarkInstance(**instance_payload)
        for instance_payload in payload["instances"]
    ]
    return StatefulPaperBenchmarkDataset(
        benchmark_name=payload["benchmark_name"],
        source=payload["source"],
        machine_model=payload["machine_model"],
        seeds=[int(x) for x in payload["seeds"]],
        n_values=[int(x) for x in payload["n_values"]],
        lambda_values=[float(x) for x in payload["lambda_values"]],
        instances=instances,
    )

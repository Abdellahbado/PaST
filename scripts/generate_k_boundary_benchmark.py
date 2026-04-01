#!/usr/bin/env python3
"""
Generate a benchmark focused on many distinct job sizes (K-scaling boundary).

This benchmark is meant to probe a specific structural question:

    How does the method behave when the number of distinct processing times
    grows, while staying in a realistic low-variability / moderate-spread
    regime similar to the paper benchmark?

The families are intentionally small and controlled:

- contiguous low-variability groups, which stay close to the paper narrative;
- moderately spread groups, which create more arithmetic ambiguity;
- K grows from 3 up to 6 while n stays moderate.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from regenerate_instances import (
    IDLE_POWER,
    OFF_ON_POWER,
    OFF_ON_TIME,
    OFF_POWER,
    ON_OFF_POWER,
    ON_OFF_TIME,
    ON_POWER,
    generate_file_costs,
    load_energy_costs,
    write_instance_json,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = (
    ROOT
    / "data"
    / "green-scheduling-bab"
    / "Iirc.EnergyStatesAndCostsScheduling"
    / "data"
    / "datasets"
    / "k_boundary_202604"
)

EC_CONFIGS = [
    {"from_date": "2019-01-21T00:00:00", "repeat_count": 1},
    {"from_date": "2019-04-08T00:00:00", "repeat_count": 1},
    {"from_date": "2019-01-21T00:00:00", "repeat_count": 4},
]


def build_instance(
    *,
    name: str,
    family: str,
    jobs_count: int,
    pt_group: list[int],
    horizon_multiplier: float,
    seed: int,
    ec_config: dict[str, object],
    data_root: Path,
) -> dict:
    rng = random.Random(seed)
    jobs = [rng.choice(pt_group) for _ in range(jobs_count)]

    time_mul = int(ec_config["repeat_count"])
    scaled_jobs = [p * time_mul for p in jobs]
    norm_min_nonproc = 1 + OFF_ON_TIME[0] * time_mul + ON_OFF_TIME[0] * time_mul + 1
    intervals_count = math.ceil(sum(scaled_jobs) * horizon_multiplier) + norm_min_nonproc

    dates, costs = load_energy_costs(data_root)
    prices = generate_file_costs(
        dates,
        costs,
        str(ec_config["from_date"]),
        intervals_count,
        time_mul,
    )
    prices = [max(1, c) for c in prices]

    return {
        "name": name,
        "family": family,
        "n_jobs": jobs_count,
        "horizon": len(prices),
        "jobs": scaled_jobs,
        "prices": prices,
        "off_on_time": [v * time_mul for v in OFF_ON_TIME],
        "on_off_time": [v * time_mul for v in ON_OFF_TIME],
        "off_on_power": OFF_ON_POWER,
        "on_off_power": ON_OFF_POWER,
        "on_power": ON_POWER,
        "idle_power": IDLE_POWER,
        "off_power": OFF_POWER,
        "metadata": {
            "family": family,
            "jobsCount": jobs_count,
            "horizonMultiplier": horizon_multiplier,
            "processing_group": pt_group,
            "K": len(pt_group),
            "seed": seed,
            "ec_from": ec_config["from_date"],
            "ec_repeat": ec_config["repeat_count"],
            "machine": "twosby",
        },
    }


def generate_instances(data_root: Path, seeds_per_case: int) -> list[dict]:
    instances: list[dict] = []

    def ec_for(seed_idx: int) -> dict[str, object]:
        return EC_CONFIGS[seed_idx % len(EC_CONFIGS)]

    families = [
        ("K_contig", [7, 8, 9], [100, 200, 300]),
        ("K_contig", [7, 8, 9, 10], [100, 200, 300]),
        ("K_contig", [7, 8, 9, 10, 11], [100, 200, 300]),
        ("K_contig", [7, 8, 9, 10, 11, 12], [100, 200]),
        ("K_contig", [7, 8, 9, 10, 11, 12, 13], [100, 200]),
        ("K_contig", [7, 8, 9, 10, 11, 12, 13, 14], [100, 200]),
        ("K_moderate_spread", [3, 5, 6, 7], [100, 200, 300]),
        ("K_moderate_spread", [5, 7, 9, 11, 13], [100, 200]),
        ("K_moderate_spread", [4, 5, 6, 8, 9, 11, 12], [100, 200]),
    ]

    for family, group, n_values in families:
        group_tag = "_".join(str(x) for x in group)
        for n in n_values:
            for sidx in range(seeds_per_case):
                seed = 8000 + 97 * sidx + 31 * n + sum(group) + 17 * len(group)
                name = f"{family}_p{group_tag}_n{n}_s{sidx}"
                instances.append(
                    build_instance(
                        name=name,
                        family=family,
                        jobs_count=n,
                        pt_group=group,
                        horizon_multiplier=1.3,
                        seed=seed,
                        ec_config=ec_for(sidx),
                        data_root=data_root,
                    )
                )
    return instances


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate K-boundary benchmark")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds-per-case", type=int, default=3)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=(
            ROOT / "data" / "green-scheduling-bab" / "Iirc.EnergyStatesAndCostsScheduling" / "data"
        ),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for old in args.out_dir.glob("*.json"):
        old.unlink()

    instances = generate_instances(args.data_root, args.seeds_per_case)

    manifest = []
    for idx, inst in enumerate(instances):
        out_path = args.out_dir / f"{idx:04d}_{inst['name']}.json"
        write_instance_json(inst, out_path)
        manifest.append(
            {
                "index": idx,
                "file": out_path.name,
                "family": inst["family"],
                "n_jobs": inst["n_jobs"],
                "horizon": inst["horizon"],
                "metadata": inst["metadata"],
            }
        )

    with open(args.out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(instances)} instances to {args.out_dir}")
    print(f"Manifest: {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()

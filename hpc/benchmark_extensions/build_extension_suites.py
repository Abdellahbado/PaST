#!/usr/bin/env python3
"""
Build the formal benchmark-extension suites used in the paper-facing study.

This script is intentionally self-contained so it can run on HPC without
depending on exploratory generators that may not be present in the branch.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "green-scheduling-bab" / "Iirc.EnergyStatesAndCostsScheduling" / "data"
DATASETS = DATA_ROOT / "datasets"

FORMAL_SUITES = {
    "scalability_large_n": DATASETS / "paperext_scalability_large_n_202604",
    "backup_realistic": DATASETS / "paperext_backup_realistic_202604",
    "k_boundary": DATASETS / "paperext_k_boundary_202604",
    "k_structure_boundary": DATASETS / "paperext_k_structure_boundary_202604",
}

EC_CONFIGS = [
    {"from_date": "2019-01-21T00:00:00", "repeat_count": 1},
    {"from_date": "2019-04-08T00:00:00", "repeat_count": 1},
    {"from_date": "2019-01-21T00:00:00", "repeat_count": 4},
]

FORMAL_EC_CONFIGS = [
    {"from_date": "2019-01-21T00:00:00", "repeat_count": 1},
    {"from_date": "2019-04-08T00:00:00", "repeat_count": 1},
    {"from_date": "2019-01-21T00:00:00", "repeat_count": 1},
]

FORMAL_K_FIXED_N = 100

OFF_ON_TIME = [4, 3, 2]
ON_OFF_TIME = [1, 1, 1]
OFF_ON_POWER = [15, 13, 12]
ON_OFF_POWER = [2, 2, 2]
ON_POWER = 10
IDLE_POWER = 8
OFF_POWER = [0, 2, 4]


def reset_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.json"):
        old.unlink()
    (out_dir / "manifest.json").unlink(missing_ok=True)


def load_energy_costs():
    ec_path = DATA_ROOT / "dataset-generators-prescriptions" / "energy-costs" / "ote2019.json"
    with open(ec_path) as f:
        d = json.load(f)
    return d["dates"], d["costs"]


def generate_file_costs(dates, costs, from_date: str, intervals_count: int, repeat_count: int):
    from_idx = None
    for i, dt in enumerate(dates):
        if dt == from_date:
            from_idx = i
            break
    if from_idx is None:
        raise ValueError(f"Date {from_date} not found in ote2019.json")

    result = []
    curr_idx = from_idx
    while len(result) < intervals_count:
        for _ in range(repeat_count):
            if len(result) == intervals_count:
                break
            if curr_idx >= len(costs):
                curr_idx = from_idx
            result.append(costs[curr_idx])
        curr_idx += 1
    return result


def write_instance_json(inst: dict, out_path: Path):
    jobs = [
        {
            "Index": j_idx,
            "OriginalIndex": j_idx,
            "ReleaseDate": 0,
            "ProcessingTime": pt,
        }
        for j_idx, pt in enumerate(inst["jobs"])
    ]
    data = {
        "MachinesCount": 1,
        "Jobs": jobs,
        "Intervals": [
            {"Index": i, "StartTime": i, "EndTime": i + 1, "EnergyCost": cost}
            for i, cost in enumerate(inst["prices"])
        ],
        "LengthInterval": 1,
        "EnergyCosts": inst["prices"],
        "OffOnTime": inst["off_on_time"],
        "OnOffTime": inst["on_off_time"],
        "OffOnPowerConsumption": inst["off_on_power"],
        "OnOffPowerConsumption": inst["on_off_power"],
        "OffIdleTime": [None, None, None],
        "IdleOffTime": [None, None, None],
        "OffIdlePowerConsumption": [None, None, None],
        "IdleOffPowerConsumption": [None, None, None],
        "OnPowerConsumption": inst["on_power"],
        "IdlePowerConsumption": inst["idle_power"],
        "OffPowerConsumption": inst["off_power"],
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


def build_instance(*, name: str, family: str, jobs_list: list[int], horizon_multiplier: float, ec_config: dict, metadata: dict):
    dates, costs = load_energy_costs()
    time_mul = int(ec_config["repeat_count"])
    scaled_jobs = [p * time_mul for p in jobs_list]
    norm_min_nonproc = 1 + OFF_ON_TIME[0] * time_mul + ON_OFF_TIME[0] * time_mul + 1
    intervals_count = math.ceil(sum(scaled_jobs) * horizon_multiplier) + norm_min_nonproc
    prices = generate_file_costs(dates, costs, str(ec_config["from_date"]), intervals_count, time_mul)
    prices = [max(1, c) for c in prices]
    return {
        "name": name,
        "family": family,
        "n_jobs": len(jobs_list),
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
            **metadata,
            "family": family,
            "jobsCount": len(jobs_list),
            "horizonMultiplier": horizon_multiplier,
            "ec_from": ec_config["from_date"],
            "ec_repeat": ec_config["repeat_count"],
            "machine": "twosby",
        },
    }


def write_suite(out_dir: Path, instances: list[dict], suite_name: str, description: str):
    reset_out_dir(out_dir)
    manifest = []
    for idx, inst in enumerate(instances):
        fname = f"{idx:04d}_{inst['name']}.json"
        write_instance_json(inst, out_dir / fname)
        manifest.append(
            {
                "file": fname,
                "suite": suite_name,
                "description": description,
                "family": inst["family"],
                "metadata": inst["metadata"],
            }
        )
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"{suite_name}: wrote {len(instances)} instances to {out_dir}")


def gen_scalability_suite() -> list[dict]:
    instances = []
    for n in [300, 400, 500, 600, 750, 1000]:
        for sidx in range(3):
            ec = FORMAL_EC_CONFIGS[sidx % len(FORMAL_EC_CONFIGS)]
            seed = 1000 + 37 * sidx + n
            rng = random.Random(seed)
            jobs = [rng.choice([8, 10]) for _ in range(n)]
            instances.append(
                build_instance(
                    name=f"famA_nscale_p8_10_n{n}_s{sidx}",
                    family="A_nscale_8_10",
                    jobs_list=jobs,
                    horizon_multiplier=1.3,
                    ec_config=ec,
                    metadata={"processing_group": [8, 10], "seed": seed},
                )
            )
    return instances


def gen_backup_suite() -> list[dict]:
    instances = []
    configs = [
        ("rescue_focus", [5, 7, 11], 60, {0: 2, 2: 3}, list(range(10))),
        ("control_4610", [4, 6, 10], 60, {0: 1, 2: 3}, list(range(5))),
        ("control_8_10_14", [8, 10, 14], 50, {0: 1, 2: 2}, list(range(5))),
    ]
    for family_tag, group, n_total, scarce, seeds in configs:
        for sidx in seeds:
            ec = FORMAL_EC_CONFIGS[sidx % len(FORMAL_EC_CONFIGS)]
            seed = 4000 + 89 * sidx + 17 * n_total + sum(group)
            rng = random.Random(seed)
            jobs_by_type = [0] * len(group)
            remaining = n_total
            for pos, cnt in scarce.items():
                jobs_by_type[pos] = cnt
                remaining -= cnt
            non_scarce = [i for i in range(len(group)) if i not in scarce]
            if non_scarce:
                per_type = remaining // len(non_scarce)
                extra = remaining % len(non_scarce)
                for i, pos in enumerate(non_scarce):
                    jobs_by_type[pos] = per_type + (1 if i < extra else 0)
            jobs = []
            for i, p in enumerate(group):
                jobs.extend([p] * jobs_by_type[i])
            rng.shuffle(jobs)
            scarce_tag = "_".join(f"{k}c{v}" for k, v in sorted(scarce.items()))
            instances.append(
                build_instance(
                    name=f"famB_backup_p{'_'.join(map(str,group))}_n{n_total}_l1.8_sc{scarce_tag}_s{sidx}",
                    family=f"backup_{family_tag}",
                    jobs_list=jobs,
                    horizon_multiplier=1.8,
                    ec_config=ec,
                    metadata={
                        "processing_group": group,
                        "seed": seed,
                        "scarce_counts": scarce,
                        "category": family_tag,
                    },
                )
            )
    return instances


def gen_k_boundary_suite() -> list[dict]:
    instances = []
    families = [
        ("K_contig", [7, 8, 9]),
        ("K_contig", [7, 8, 9, 10]),
        ("K_contig", [7, 8, 9, 10, 11]),
        ("K_contig", [7, 8, 9, 10, 11, 12]),
        ("K_contig", [7, 8, 9, 10, 11, 12, 13]),
        ("K_contig", [7, 8, 9, 10, 11, 12, 13, 14]),
        ("K_moderate_spread", [3, 5, 6, 7]),
        ("K_moderate_spread", [5, 7, 9, 11, 13]),
        ("K_moderate_spread", [4, 5, 6, 8, 9, 11, 12]),
    ]
    n = FORMAL_K_FIXED_N
    for family, group in families:
        for sidx in range(3):
            ec = FORMAL_EC_CONFIGS[sidx % len(FORMAL_EC_CONFIGS)]
            seed = 8000 + 97 * sidx + 31 * n + sum(group) + 17 * len(group)
            rng = random.Random(seed)
            jobs = [rng.choice(group) for _ in range(n)]
            instances.append(
                build_instance(
                    name=f"{family}_p{'_'.join(map(str,group))}_n{n}_s{sidx}",
                    family=family,
                    jobs_list=jobs,
                    horizon_multiplier=1.3,
                    ec_config=ec,
                    metadata={"processing_group": group, "seed": seed, "K": len(group), "fixed_n": n},
                )
            )
    return instances


def gen_k_structure_boundary_suite() -> list[dict]:
    instances = []
    n = FORMAL_K_FIXED_N
    families = [
        ("K7_contiguous", [7, 8, 9, 10, 11, 12, 13]),
        ("K7_shifted_contiguous", [8, 9, 10, 11, 12, 13, 14]),
        ("K7_even_spread", [4, 6, 8, 10, 12, 14, 16]),
        ("K7_odd_spread", [5, 7, 9, 11, 13, 15, 17]),
        ("K7_irregular", [4, 5, 7, 9, 10, 12, 13]),
    ]
    for family, group in families:
        for sidx in range(5):
            ec = FORMAL_EC_CONFIGS[sidx % len(FORMAL_EC_CONFIGS)]
            seed = 12000 + 131 * sidx + 31 * n + sum(group) + 19 * len(group)
            rng = random.Random(seed)
            jobs = [rng.choice(group) for _ in range(n)]
            instances.append(
                build_instance(
                    name=f"{family}_p{'_'.join(map(str,group))}_n{n}_s{sidx}",
                    family=family,
                    jobs_list=jobs,
                    horizon_multiplier=1.3,
                    ec_config=ec,
                    metadata={
                        "processing_group": group,
                        "seed": seed,
                        "K": len(group),
                        "fixed_n": n,
                        "structure_family": family,
                    },
                )
            )
    return instances


def main() -> None:
    ap = argparse.ArgumentParser(description="Build formal benchmark-extension suites")
    ap.add_argument(
        "--suite",
        choices=["all", "scalability_large_n", "backup_realistic", "k_boundary", "k_structure_boundary"],
        default="all",
    )
    args = ap.parse_args()

    if args.suite in ("all", "scalability_large_n"):
        write_suite(
            FORMAL_SUITES["scalability_large_n"],
            gen_scalability_suite(),
            "scalability_large_n",
            "Large-n extension of the paper-hard {8,10} family; only n is scaled.",
        )
    if args.suite in ("all", "backup_realistic"):
        write_suite(
            FORMAL_SUITES["backup_realistic"],
            gen_backup_suite(),
            "backup_realistic",
            "Realistic bounded-count 3-type suite used to compare semigroup and R_feas.",
        )
    if args.suite in ("all", "k_boundary"):
        write_suite(
            FORMAL_SUITES["k_boundary"],
            gen_k_boundary_suite(),
            "k_boundary",
            "Increasing-K realistic families at fixed n used to probe the structural boundary of the method.",
        )
    if args.suite in ("all", "k_structure_boundary"):
        write_suite(
            FORMAL_SUITES["k_structure_boundary"],
            gen_k_structure_boundary_suite(),
            "k_structure_boundary",
            "Fixed-K=7, fixed-n structure study that varies only the processing-time family.",
        )


if __name__ == "__main__":
    main()

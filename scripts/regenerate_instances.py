#!/usr/bin/env python3
"""
Regenerate benedikt2025b_groups instances using a faithful Python
replication of C# System.Random (Knuth-subtractive / .NET compat mode).

The original prescription file does NOT include hopsCount, which means
the C# generator's `foreach (var hopsCount in hopsCounts)` loop
receives null. The correct behavior is to skip the hops/shuffle entirely,
yielding one instance per (repeatCount) iteration with unmodified energy costs.

Our locally-modified prescription added "hopsCount": [0], which despite
zero hops still calls Shuffle(this.random) and pollutes the Random state,
giving WRONG processing times for all instances after the first.
"""
import json
import math
import os
import sys
from pathlib import Path

# ────────────────────────────────────────────────────────────────────
# C# System.Random replication (.NET 6 compat / Knuth subtractive)
# ────────────────────────────────────────────────────────────────────
MBIG = 2147483647  # Int32.MaxValue
MSEED = 161803398


class CSharpRandom:
    """Faithful Python port of .NET System.Random (seeded, compat mode)."""

    def __init__(self, seed: int):
        seed_array = [0] * 56  # index 0 unused, 1..55
        subtraction = MBIG if seed == -2147483648 else abs(seed)
        mj = MSEED - subtraction
        if mj < 0:
            mj += MBIG + 1  # unsigned wrap-around
        seed_array[55] = mj
        mk = 1
        for i in range(1, 55):
            ii = (21 * i) % 55
            seed_array[ii] = mk
            mk = mj - mk
            if mk < 0:
                mk += MBIG
            mj = seed_array[ii]
        for _ in range(4):
            for i in range(1, 56):
                seed_array[i] -= seed_array[1 + (i + 30) % 55]
                if seed_array[i] < 0:
                    seed_array[i] += MBIG
        self._seed_array = seed_array
        self._inext = 0
        self._inextp = 21

    def _internal_sample(self) -> int:
        loc_inext = self._inext + 1
        if loc_inext >= 56:
            loc_inext = 1
        loc_inextp = self._inextp + 1
        if loc_inextp >= 56:
            loc_inextp = 1
        ret = self._seed_array[loc_inext] - self._seed_array[loc_inextp]
        if ret == MBIG:
            ret -= 1
        if ret < 0:
            ret += MBIG
        self._seed_array[loc_inext] = ret
        self._inext = loc_inext
        self._inextp = loc_inextp
        return ret

    def sample(self) -> float:
        return self._internal_sample() * (1.0 / MBIG)

    def next_int(self, min_val: int, max_val: int) -> int:
        """Equivalent to C# Random.Next(minValue, maxValue)."""
        rng = max_val - min_val
        if rng <= MBIG:
            return int(self.sample() * rng) + min_val
        else:
            raise ValueError("Range too large")

    def next_double(self) -> float:
        return self.sample()


# ────────────────────────────────────────────────────────────────────
# Energy costs from ote2019.json
# ────────────────────────────────────────────────────────────────────
def load_energy_costs(data_dir: Path):
    """Load ote2019.json and return (dates, costs)."""
    ec_path = (
        data_dir / "dataset-generators-prescriptions" / "energy-costs" / "ote2019.json"
    )
    with open(ec_path) as f:
        d = json.load(f)
    return d["dates"], d["costs"]


def generate_file_costs(
    dates, costs, from_date: str, intervals_count: int, repeat_count: int
):
    """Replicate FileEnergyCostsProvider.Generate."""
    from_idx = None
    for i, dt in enumerate(dates):
        if dt == from_date:
            from_idx = i
            break
    if from_idx is None:
        raise ValueError(f"Date {from_date} not found in ote2019.json")

    result = []
    curr_idx = from_idx
    while True:
        for _ in range(repeat_count):
            if len(result) == intervals_count:
                return result
            if curr_idx >= len(costs):
                curr_idx = from_idx
            result.append(costs[curr_idx])
        curr_idx += 1
    return result


# ────────────────────────────────────────────────────────────────────
# Instance Generation
# ────────────────────────────────────────────────────────────────────
# TWOSBY state diagram
OFF_ON_TIME = [4, 3, 2]
ON_OFF_TIME = [1, 1, 1]
OFF_ON_POWER = [15, 13, 12]
ON_OFF_POWER = [2, 2, 2]
ON_POWER = 10
IDLE_POWER = 8
OFF_POWER = [0, 2, 4]

JOBS_COUNTS = [50, 100, 150, 200]
REPETITIONS_COUNT = 5
HORIZON_MULTIPLIERS = [1.3]
LENGTH_INTERVAL = 1

PROCESSING_TIMES_GROUPS = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [1, 2, 3, 5, 7],
    [2, 4, 6, 8, 10],
    [3, 7],
    [2, 4],
    [8, 10],
    [3, 5, 6, 7],
]

ENERGY_COSTS_CONFIGS = [
    {"from_date": "2019-01-21T00:00:00", "repeat_count": 1},
    {"from_date": "2019-04-08T00:00:00", "repeat_count": 1},
    {"from_date": "2019-01-21T00:00:00", "repeat_count": 4},
    {"from_date": "2019-04-08T00:00:00", "repeat_count": 4},
]


def generate_instances(data_dir: Path):
    """
    Generate all 560 benedikt2025b_groups instances.

    CRITICAL: When HopsCount is null (original prescription),
    the hopsCount loop should be SKIPPED entirely.
    No Shuffle call, no random state pollution.
    """
    dates, costs = load_energy_costs(data_dir)
    rnd = CSharpRandom(42)

    instances = []
    instance_idx = 0

    for jobs_count in JOBS_COUNTS:
        for repetition in range(REPETITIONS_COUNT):
            # Only one state diagram: Benedikt2020aTwosby
            for ec_config in ENERGY_COSTS_CONFIGS:
                for pt_group in PROCESSING_TIMES_GROUPS:
                    # --- Inner GenerateInstances ---
                    min_repeat_count = min(
                        cfg["repeat_count"] for cfg in ENERGY_COSTS_CONFIGS
                    )
                    # Actually min_repeat_count is from the CURRENT ec_config's
                    # repeatCount list, not across all configs.
                    # For benedikt2025b_groups, each ec_config has exactly one
                    # repeatCount, so energyCostsRepeatCounts has 1 element.
                    ec_repeat_counts = [ec_config["repeat_count"]]
                    min_rc = min(ec_repeat_counts)

                    # Generate processing times
                    norm_pts = []
                    for _ in range(jobs_count):
                        idx = rnd.next_int(0, len(pt_group))
                        norm_pts.append(pt_group[idx])

                    # Compute horizon
                    max_hm = max(HORIZON_MULTIPLIERS)
                    norm_max_avail = math.ceil(sum(norm_pts) * max_hm)
                    norm_min_nonproc = 1 + OFF_ON_TIME[0] + ON_OFF_TIME[0] + 1
                    norm_all_intervals = norm_max_avail + norm_min_nonproc

                    # Generate energy costs from file (deterministic)
                    all_costs = generate_file_costs(
                        dates, costs, ec_config["from_date"], norm_all_intervals, min_rc
                    )
                    # Clamp to >= 1
                    all_costs = [max(1, c) for c in all_costs]

                    for hm in HORIZON_MULTIPLIERS:
                        norm_avail = math.ceil(sum(norm_pts) * hm)
                        n_intervals = norm_avail + norm_min_nonproc

                        for rc in ec_repeat_counts:
                            time_mul = rc // min_rc

                            # Build intervals
                            intervals_costs = []
                            for ni_idx in range(n_intervals):
                                for _ in range(time_mul):
                                    intervals_costs.append(all_costs[ni_idx])

                            # NO hopsCount loop — original prescription has null
                            # Just yield the instance directly

                            # Build jobs
                            jobs = []
                            for j_idx in range(jobs_count):
                                pt = norm_pts[j_idx] * time_mul
                                jobs.append(pt)

                            inst = {
                                "instance_idx": instance_idx,
                                "n_jobs": jobs_count,
                                "horizon": len(intervals_costs),
                                "jobs": jobs,
                                "prices": intervals_costs,
                                "off_on_time": [v * time_mul for v in OFF_ON_TIME],
                                "on_off_time": [v * time_mul for v in ON_OFF_TIME],
                                "off_on_power": OFF_ON_POWER,
                                "on_off_power": ON_OFF_POWER,
                                "on_power": ON_POWER,
                                "idle_power": IDLE_POWER,
                                "off_power": OFF_POWER,
                                "metadata": {
                                    "repetition": repetition,
                                    "jobsCount": jobs_count,
                                    "horizonMultiplier": hm,
                                    "ec_from": ec_config["from_date"],
                                    "ec_repeat": rc,
                                    "pt_group": pt_group,
                                },
                            }
                            instances.append(inst)
                            instance_idx += 1

    return instances


def write_instance_json(inst: dict, out_path: Path):
    """Write in the same JSON format as the C# JsonInputWriter."""
    intervals = []
    for i, cost in enumerate(inst["prices"]):
        intervals.append(
            {"Index": i, "StartTime": i, "EndTime": i + 1, "EnergyCost": cost}
        )

    jobs = []
    for j_idx, pt in enumerate(inst["jobs"]):
        jobs.append(
            {
                "Index": j_idx,
                "OriginalIndex": j_idx,
                "ReleaseDate": 0,
                "ProcessingTime": pt,
            }
        )

    data = {
        "MachinesCount": 1,
        "Jobs": jobs,
        "Intervals": intervals,
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


def compare_with_existing(instances, existing_dir: Path):
    """Compare generated instances with existing dataset files."""
    mismatches = 0
    matches = 0
    for inst in instances:
        idx = inst["instance_idx"]
        existing_path = existing_dir / f"{idx}.json"
        if not existing_path.exists():
            print(f"  Instance {idx}: MISSING in existing dataset")
            continue
        with open(existing_path) as f:
            existing = json.load(f)
        ex_pts = [j["ProcessingTime"] for j in existing["Jobs"]]
        ex_prices = existing["EnergyCosts"]

        if ex_pts != inst["jobs"]:
            mismatches += 1
            if mismatches <= 10:
                print(f"  Instance {idx}: PTs DIFFER!")
                print(f"    Generated: {inst['jobs'][:10]}...")
                print(f"    Existing:  {ex_pts[:10]}...")
        elif ex_prices != inst["prices"]:
            mismatches += 1
            if mismatches <= 10:
                print(f"  Instance {idx}: Prices DIFFER!")
                print(f"    Generated first 5: {inst['prices'][:5]}")
                print(f"    Existing first 5:  {ex_prices[:5]}")
        else:
            matches += 1

    print(
        f"\n  Summary: {matches} match, {mismatches} mismatch out of {len(instances)}"
    )
    return mismatches


def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = (
        root
        / "data"
        / "green-scheduling-bab"
        / "Iirc.EnergyStatesAndCostsScheduling"
        / "data"
    )
    existing_dir = data_dir / "datasets" / "benedikt2025b_groups"

    print("=== Generating benedikt2025b_groups instances (Python, no-shuffle) ===")
    instances = generate_instances(data_dir)
    print(f"Generated {len(instances)} instances")

    # Quick check: instance 0
    i0 = instances[0]
    print(f"\nInstance 0: n={i0['n_jobs']}, horizon={i0['horizon']}")
    print(f"  PTs: {i0['jobs'][:20]}")
    print(f"  First 10 prices: {i0['prices'][:10]}")

    # Compare
    print(f"\n=== Comparing with existing instances in {existing_dir} ===")
    n_mismatches = compare_with_existing(instances, existing_dir)

    if n_mismatches > 0:
        # Write corrected instances
        out_dir = data_dir / "datasets" / "benedikt2025b_groups_corrected"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Writing corrected instances to {out_dir} ===")
        for inst in instances:
            write_instance_json(inst, out_dir / f"{inst['instance_idx']}.json")
        print(f"Wrote {len(instances)} instances")
    else:
        print("\nAll instances match! No correction needed.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Visualize the benchmark instances (j = 1..90) by combining:
- Data_cj.txt : time-slot energy prices
- Data_ej.txt : machine energy prices
- Data_pj.txt : jobs processing times
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent
INSTANCE_RANGES = {
    "small": range(1, 31),
    "medium": range(31, 61),
    "large": range(61, 91),
}
ALL_INSTANCES = range(1, 91)
PLOT_TYPES = {"split", "combined"}
DEFAULT_RELATIVE_DATA_DIR = (
    Path(
        "A-bi-objective-heuristic-approach-for-green-identical-parallel-machine-scheduling"
    )
    / "Data"
)


def resolve_data_dir(data_dir_arg: str | None) -> Path:
    """Locate the data directory, optionally using a user-supplied path."""

    if data_dir_arg:
        candidate = Path(data_dir_arg).expanduser().resolve()
        if not candidate.is_dir():
            raise FileNotFoundError(
                f"Provided data directory '{candidate}' is not valid."
            )
        return candidate

    candidates = [
        REPO_ROOT / "Data",
        REPO_ROOT / DEFAULT_RELATIVE_DATA_DIR,
        REPO_ROOT.parent / DEFAULT_RELATIVE_DATA_DIR,
    ]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not locate the 'Data' directory. Use --data-dir to point to it explicitly."
    )


def load_instance(instance_id: int, data_dir: Path):
    """Return a dict with the time-slot, machine, and job data for instance j."""
    suffix = str(instance_id)
    time_slot_prices = np.loadtxt(data_dir / f"Data_c{suffix}.txt", dtype=float)
    machine_prices = np.loadtxt(data_dir / f"Data_e{suffix}.txt", dtype=float)
    processing_times = np.loadtxt(data_dir / f"Data_p{suffix}.txt", dtype=float)

    return {
        "instance": instance_id,
        "time_slot_prices": time_slot_prices,
        "machine_prices": machine_prices,
        "processing_times": processing_times,
    }


def plot_instance_split(instance_data):
    """Generate a stacked visualization for a single instance."""
    inst = instance_data["instance"]
    time_slot_prices = instance_data["time_slot_prices"]
    machine_prices = instance_data["machine_prices"]
    processing_times = instance_data["processing_times"]

    num_slots = np.arange(1, len(time_slot_prices) + 1)
    num_jobs = np.arange(1, len(processing_times) + 1)
    num_machines = np.arange(1, len(machine_prices) + 1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
    fig.suptitle(f"Instance {inst} (small-scale)")

    axes[0].step(num_slots, time_slot_prices, where="mid", linewidth=2, color="#0072B2")
    axes[0].set_ylabel("Energy price")
    axes[0].set_xlabel("Time slot")
    axes[0].grid(alpha=0.3)
    axes[0].set_title("Time-slot energy prices (Data_c)")

    axes[1].bar(num_machines, machine_prices, color="#009E73")
    axes[1].set_ylabel("Energy price")
    axes[1].set_xlabel("Machine")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_title("Machine energy prices (Data_e)")

    axes[2].bar(num_jobs, processing_times, color="#D55E00")
    axes[2].set_ylabel("Processing time")
    axes[2].set_xlabel("Job")
    axes[2].grid(axis="y", alpha=0.3)
    axes[2].set_title("Job processing times (Data_p)")

    fig.tight_layout()
    plt.show()


def plot_instance_combined(instance_data):
    """Generate a single-axis visualization combining time horizon and machines."""

    inst = instance_data["instance"]
    time_slot_prices = instance_data["time_slot_prices"]
    machine_prices = instance_data["machine_prices"]
    processing_times = instance_data["processing_times"]

    time_indices = np.arange(1, len(time_slot_prices) + 1)
    max_time = len(time_slot_prices)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f"Instance {inst} summary")

    ax.step(
        time_indices,
        time_slot_prices,
        where="mid",
        linewidth=2,
        color="#0072B2",
        label="Time-slot energy price",
    )

    for idx, price in enumerate(machine_prices, start=1):
        ax.hlines(
            price,
            1,
            max_time,
            colors="#009E73",
            linestyles="dashed",
            linewidth=1.5,
            label="Machine energy price" if idx == 1 else None,
            alpha=0.8,
        )

    ax.set_xlim(1, max_time)
    ax.set_xlabel("Time slot")
    ax.set_ylabel("Energy price")
    ax.grid(alpha=0.3)

    total_processing = processing_times.sum()
    if total_processing > 0:
        scaled_durations = processing_times * (max_time / total_processing)
        starts = np.concatenate(([0.0], np.cumsum(scaled_durations[:-1]))) + 1

        ax_jobs = ax.twinx()
        y_positions = np.arange(1, len(processing_times) + 1)
        ax_jobs.barh(
            y_positions,
            scaled_durations,
            left=starts,
            height=0.8,
            color="#D55E00",
            alpha=0.4,
            label="Job (scaled processing time)",
        )
        ax_jobs.set_ylim(0, len(processing_times) + 1)
        ax_jobs.set_yticks(y_positions)
        ax_jobs.set_ylabel("Job index")
        ax_jobs.grid(alpha=0.2, axis="y")
        ax_jobs.set_xlim(1, max_time)

        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax_jobs.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    else:
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_instance(instance_data, plot_type: str):
    if plot_type == "split":
        plot_instance_split(instance_data)
    elif plot_type == "combined":
        plot_instance_combined(instance_data)
    else:
        raise ValueError(f"Unsupported plot type '{plot_type}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize combined benchmark data for instances j = 1..90."
    )
    parser.add_argument(
        "--instance",
        type=int,
        choices=range(1, 91),
        help="Instance ID to visualize (1..90). If omitted, iterate over the selected subset.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Path to the directory containing the Data_*.txt files (defaults to auto-detected location).",
    )
    parser.add_argument(
        "--subset",
        choices=INSTANCE_RANGES.keys(),
        help="Subset of instances to visualize when --instance is not provided.",
    )
    parser.add_argument(
        "--plot-type",
        choices=PLOT_TYPES,
        default="split",
        help="Visualization style: 'split' (three stacked subplots) or 'combined' (single summary plot).",
    )
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)

    if args.instance:
        instance_ids = [args.instance]
    else:
        if args.subset:
            instance_ids = list(INSTANCE_RANGES[args.subset])
        else:
            instance_ids = list(ALL_INSTANCES)

    for instance_id in instance_ids:
        instance_data = load_instance(instance_id, data_dir)
        plot_instance(instance_data, args.plot_type)


if __name__ == "__main__":
    main()

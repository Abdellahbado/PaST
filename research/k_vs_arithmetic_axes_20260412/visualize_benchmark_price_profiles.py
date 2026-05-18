#!/usr/bin/env python3
"""Generate supervisor-facing price-profile figures for the 560 paper instances."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta


ROOT = Path(__file__).resolve().parents[2]
RESEARCH = Path(__file__).resolve().parent
OUT = RESEARCH / "figures" / "price_profiles"

DATA_ROOT = (
    ROOT
    / "data"
    / "green-scheduling-bab"
    / "Iirc.EnergyStatesAndCostsScheduling"
    / "data"
)
PRESCRIPTION = DATA_ROOT / "dataset-generators-prescriptions" / "benedikt2025b_groups.json"
OTE = DATA_ROOT / "dataset-generators-prescriptions" / "energy-costs" / "ote2019.json"
DATASET = DATA_ROOT / "datasets" / "benedikt2025b_groups"

SCENARIOS = [
    ("Jan 21, repeat=1", 0, "2019-01-21T00:00:00", 1),
    ("Apr 08, repeat=1", 7, "2019-04-08T00:00:00", 1),
    ("Jan 21, repeat=4", 14, "2019-01-21T00:00:00", 4),
    ("Apr 08, repeat=4", 21, "2019-04-08T00:00:00", 4),
]


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def load_instance(idx: int):
    return load_json(DATASET / f"{idx}.json")


def dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def moving_avg(x: list[int], w: int) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if len(arr) < w:
        return arr
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="same")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ote = load_json(OTE)
    prescription = load_json(PRESCRIPTION)

    dates = [dt(x) for x in ote["dates"]]
    costs = ote["costs"]
    date_to_idx = {d: i for i, d in enumerate(dates)}

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 150,
        }
    )

    # Figure 1: the four price scenarios used by the 560 small benchmark instances.
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=False)
    fig.suptitle(
        "Price profiles used in the 560 paper benchmark instances\n"
        "Source: OTE 2019 hourly electricity prices; one model interval per OTE hour for repeat=1",
        y=0.985,
        fontsize=15,
        fontweight="bold",
    )
    colors = ["#2b6f9e", "#3b8f5f", "#a56c23", "#8f4b79"]
    for ax, (label, idx, start, repeat), color in zip(axes, SCENARIOS, colors):
        inst = load_instance(idx)
        prices = inst["EnergyCosts"]
        x = np.arange(len(prices))
        ax.plot(x, prices, color=color, lw=1.05)
        ax.fill_between(x, prices, min(prices), color=color, alpha=0.12)
        ax.set_title(
            f"{label}: representative n=50 instance, horizon={len(prices)} intervals",
            loc="left",
        )
        ax.set_ylabel("price")
        ax.grid(True, axis="y", alpha=0.28)
        ax.text(
            0.99,
            0.82,
            f"min={min(prices)}  mean={np.mean(prices):.0f}  max={max(prices)}",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.9),
        )
        if repeat == 1:
            ax.set_xlabel("model interval = one OTE hourly price")
        else:
            ax.set_xlabel("model interval; each OTE hourly price is repeated 4 times")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(OUT / "paper_560_price_scenarios.png", bbox_inches="tight")
    fig.savefig(OUT / "paper_560_price_scenarios.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: raw OTE hourly profile around the two benchmark start dates.
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.8), sharey=True)
    fig.suptitle(
        "Raw OTE 2019 hourly prices used by the benchmark start dates",
        y=0.98,
        fontsize=15,
        fontweight="bold",
    )
    for ax, start, color in [
        (axes[0], "2019-01-21T00:00:00", "#2b6f9e"),
        (axes[1], "2019-04-08T00:00:00", "#3b8f5f"),
    ]:
        start_dt = dt(start)
        start_idx = date_to_idx[start_dt]
        n = 14 * 24
        xs = dates[start_idx : start_idx + n]
        ys = costs[start_idx : start_idx + n]
        ax.plot(xs, ys, color=color, lw=1.2)
        ax.fill_between(xs, ys, min(ys), color=color, alpha=0.13)
        ax.set_title(f"{start_dt:%Y-%m-%d}: first 14 days from OTE file", loc="left")
        ax.set_ylabel("price")
        ax.grid(True, axis="y", alpha=0.28)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1].set_xlabel("calendar date (hourly samples)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "ote2019_raw_two_start_dates.png", bbox_inches="tight")
    fig.savefig(OUT / "ote2019_raw_two_start_dates.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure 3: small vs large instance horizons.
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.2), sharex=False)
    fig.suptitle(
        "Representative horizon lengths: small n=50 versus larger n=200",
        y=0.98,
        fontsize=15,
        fontweight="bold",
    )
    for ax, idx, label, color in [
        (axes[0], 0, "small instance: n=50, Jan 21 repeat=1", "#2b6f9e"),
        (axes[1], 420, "larger instance: n=200, Jan 21 repeat=1", "#7b4bb3"),
    ]:
        inst = load_instance(idx)
        prices = inst["EnergyCosts"]
        x = np.arange(len(prices))
        ax.plot(x, prices, color=color, lw=0.75, alpha=0.55, label="hourly/profile interval")
        ax.plot(x, moving_avg(prices, 24), color="#111111", lw=1.25, label="24-interval moving average")
        ax.set_title(f"{label}; horizon={len(prices)} intervals", loc="left")
        ax.set_ylabel("price")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="upper right", frameon=True, fontsize=9)
    axes[-1].set_xlabel("model interval")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "small_vs_large_price_horizon.png", bbox_inches="tight")
    fig.savefig(OUT / "small_vs_large_price_horizon.pdf", bbox_inches="tight")
    plt.close(fig)

    # Summary note.
    summary = OUT / "PRICE_PROFILE_VISUALIZATION_NOTE.md"
    summary.write_text(
        f"""# Benchmark Price Profile Visualization

Generated from the actual local benchmark inputs.

## Sources

- Prescription: `{PRESCRIPTION}`
- OTE file: `{OTE}`
- Dataset: `{DATASET}`

## What the paper benchmark uses

The 560-instance set is generated by:

- job counts: `50, 100, 150, 200`
- repetitions: `5`
- processing-time groups: `7`
- price scenarios: `4`

Total: `4 * 5 * 7 * 4 = 560`.

The four price scenarios in the prescription are:

1. `2019-01-21T00:00:00`, `repeatCount=1`
2. `2019-04-08T00:00:00`, `repeatCount=1`
3. `2019-01-21T00:00:00`, `repeatCount=4`
4. `2019-04-08T00:00:00`, `repeatCount=4`

The raw OTE file has `{len(costs)}` timestamped values, starting at `{ote['dates'][0]}` and ending at `{ote['dates'][-1]}`. Consecutive timestamps are hourly in the file.

## Interpretation

For `repeatCount=1`, one model interval takes one hourly OTE price value. The instance JSON field `LengthInterval` is `1`, and each `EnergyCosts[t]` entry is one price interval.

For `repeatCount=4`, each hourly OTE price is repeated four consecutive model intervals, and processing/transition times are scaled by the same factor in the generator. This is not a new volatile price source; it is a repeated/refined version of the same hourly OTE sequence.

## Output figures

- `paper_560_price_scenarios.png`: four benchmark price scenarios on representative small instances (`n=50`).
- `ote2019_raw_two_start_dates.png`: raw OTE hourly prices for the two start dates.
- `small_vs_large_price_horizon.png`: representative small (`n=50`) and larger (`n=200`) price horizons.
""",
        encoding="utf-8",
    )

    print(OUT)


if __name__ == "__main__":
    main()

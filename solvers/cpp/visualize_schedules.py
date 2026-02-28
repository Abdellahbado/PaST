#!/usr/bin/env python3
"""
visualize_schedules.py — Gantt chart + price profile for DP vs BnB schedules.

Usage:
    python3 visualize_schedules.py <path>                 # dir or single .json
    python3 visualize_schedules.py <path> --save-png      # save as PNG
    python3 visualize_schedules.py <path> --ids 0 2 4     # specific instances
    python3 visualize_schedules.py <path> --no-show       # headless

<path> is either:
  - A results directory (e.g. build/results/small_s42)
  - A single instance JSON file
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

matplotlib.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     10,
    "axes.titlesize": 11,
    "figure.dpi":    130,
})

# ─────────────────────────────────────────────────────────────────────────────
#  Colours
# ─────────────────────────────────────────────────────────────────────────────
COL_DP     = "#2980b9"      # solid blue for DP bars
COL_BNB    = "#e67e22"      # solid orange for BnB bars
COL_CHEAP  = "#d5f5e3"      # light green background = cheap slot
COL_MED    = "#fef9e7"      # light yellow  = medium
COL_EXP    = "#fadbd8"      # light red     = expensive slot

BAR_HEIGHT = 0.35
DP_Y       = 1.3            # y position of DP row centre
BNB_Y      = 0.3            # y position of BnB row centre


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
#  Detect price changepoints
# ─────────────────────────────────────────────────────────────────────────────
def changepoints(prices: np.ndarray) -> List[int]:
    """Return indices where the price changes (always includes 0 and T)."""
    cps = [0]
    for t in range(1, len(prices)):
        if prices[t] != prices[t - 1]:
            cps.append(t)
    cps.append(len(prices))
    return cps


def price_bg_colour(price: float, p_min: float, p_max: float) -> str:
    """Map a price level to a background band colour."""
    if p_max == p_min:
        return COL_CHEAP
    t = (price - p_min) / (p_max - p_min)
    if t < 0.33:
        return COL_CHEAP
    elif t < 0.66:
        return COL_MED
    else:
        return COL_EXP


# ─────────────────────────────────────────────────────────────────────────────
#  Main plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_instance(data: dict, show: bool = True, save_path: Optional[str] = None):
    prices     = np.array(data["prices"], dtype=float)
    T          = data["T"]
    n_jobs     = data["n_jobs"]
    proc_times = data["proc_times"]
    iid        = data["instance_id"]
    size       = data.get("size", "?")
    seed       = data.get("seed", "?")

    dp_info    = data.get("dp",  {})
    bnb_info   = data.get("bnb", {})
    dp_cost    = dp_info.get("cost",  float("inf"))
    bnb_cost   = bnb_info.get("cost", float("inf"))
    dp_timed   = dp_info.get("timed_out", False)
    bnb_timed  = bnb_info.get("timed_out", False)
    dp_segs    = dp_info.get("segments", [])    # [[start, length], ...]
    bnb_seq    = bnb_info.get("sequence", [])   # job indices
    bnb_starts = bnb_info.get("starts",   [])   # start times
    match      = data.get("cost_match", False)

    p_min, p_max = float(prices.min()), float(prices.max())
    cps = changepoints(prices)

    # ── Figure: 2 rows, shared x-axis ─────────────────────────────────────
    fig_width = max(14, T / 7)
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(fig_width, 5.5),
        sharex=True,                          # ← KEY: forces alignment
        gridspec_kw={"height_ratios": [1, 2]},
    )
    fig.subplots_adjust(hspace=0.04)          # tight gap between panels

    # ── Title ──────────────────────────────────────────────────────────────
    status = ""
    if dp_timed and bnb_timed:
        status = "both TLE"
    elif dp_timed:
        status = "DP TLE"
    elif bnb_timed:
        status = "BnB TLE (partial)"
    elif match:
        status = "✓ costs match"
    else:
        status = "✗ MISMATCH"

    def fmt_cost(c, tle):
        return ("inf" if c >= 1e299 else f"{c:.4f}") + (" [TLE]" if tle else "")

    fig.suptitle(
        f"Instance {iid}  |  size={size}  seed={seed}  n={n_jobs}  T={T}\n"
        f"DP = {fmt_cost(dp_cost, dp_timed)}    "
        f"BnB = {fmt_cost(bnb_cost, bnb_timed)}    "
        f"[ {status} ]",
        fontsize=10.5, fontweight="bold", y=1.01,
    )

    # ══════════════════════════════════════════════════════════════════════
    #  TOP PANEL — price profile
    # ══════════════════════════════════════════════════════════════════════

    # Shade price intervals as background bands (same colours as Gantt)
    for i in range(len(cps) - 1):
        s, e   = cps[i], cps[i + 1]
        colour = price_bg_colour(prices[s], p_min, p_max)
        ax_top.axvspan(s, e, color=colour, alpha=0.55, zorder=0)

    # Vertical changepoint lines
    for cp in cps[1:-1]:
        ax_top.axvline(cp, color="grey", linewidth=0.6, linestyle=":", zorder=1)

    # Price step line
    t_step = np.arange(T + 1)
    p_step = np.append(prices, prices[-1])
    ax_top.step(t_step, p_step, where="post",
                color="#c0392b", linewidth=1.8, zorder=3, label="price  cₜ")

    ax_top.set_ylim(0, p_max * 1.35)
    ax_top.set_ylabel("price  cₜ", fontsize=9)
    ax_top.set_title("Energy price profile", fontsize=10, pad=3)
    ax_top.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=4))
    ax_top.tick_params(bottom=False)   # ticks hidden (sharex shows them on bot)

    # Price level legend
    cheap_p  = mpatches.Patch(color=COL_CHEAP, label="cheap")
    med_p    = mpatches.Patch(color=COL_MED,   label="medium")
    exp_p    = mpatches.Patch(color=COL_EXP,   label="expensive")
    ax_top.legend(handles=[cheap_p, med_p, exp_p],
                  loc="upper right", fontsize=7.5, framealpha=0.85)

    # ══════════════════════════════════════════════════════════════════════
    #  BOTTOM PANEL — Gantt chart
    # ══════════════════════════════════════════════════════════════════════

    # Background: same price shading so columns align perfectly with top panel
    for i in range(len(cps) - 1):
        s, e   = cps[i], cps[i + 1]
        colour = price_bg_colour(prices[s], p_min, p_max)
        ax_bot.axvspan(s, e, color=colour, alpha=0.40, zorder=0)

    # Vertical changepoint lines
    for cp in cps[1:-1]:
        ax_bot.axvline(cp, color="grey", linewidth=0.6, linestyle=":", zorder=1)

    # Divider between DP and BnB rows
    ax_bot.axhline(1.15, color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=1)

    # ── DP bars ─────────────────────────────────────────────────────────────
    dp_cost_total = 0.0
    for seg in dp_segs:
        s, L = seg[0], seg[1]
        slot_cost = float(np.sum(prices[s:s + L]))
        dp_cost_total += slot_cost
        rect = mpatches.FancyBboxPatch(
            (s + 0.04, DP_Y), L - 0.08, BAR_HEIGHT,
            boxstyle="round,pad=0.03",
            linewidth=1.0, edgecolor="#1a5276",
            facecolor=COL_DP, alpha=0.88, zorder=3,
        )
        ax_bot.add_patch(rect)
        if L >= max(2, T // 60):
            ax_bot.text(
                s + L / 2, DP_Y + BAR_HEIGHT / 2,
                f"L{L}\n{slot_cost:.0f}",
                ha="center", va="center",
                fontsize=max(5.5, 8 - T // 80),
                color="white", fontweight="bold", zorder=4,
                clip_on=True,
            )

    # ── BnB bars ────────────────────────────────────────────────────────────
    for rank, (job_id, start) in enumerate(zip(bnb_seq, bnb_starts)):
        p = proc_times[job_id] if job_id < len(proc_times) else 1
        slot_cost = float(np.sum(prices[start:start + p]))
        rect = mpatches.FancyBboxPatch(
            (start + 0.04, BNB_Y), p - 0.08, BAR_HEIGHT,
            boxstyle="round,pad=0.03",
            linewidth=1.0, edgecolor="#784212",
            facecolor=COL_BNB, alpha=0.88, zorder=3,
        )
        ax_bot.add_patch(rect)
        if p >= max(2, T // 60):
            ax_bot.text(
                start + p / 2, BNB_Y + BAR_HEIGHT / 2,
                f"j{job_id}\n{slot_cost:.0f}",
                ha="center", va="center",
                fontsize=max(5.5, 8 - T // 80),
                color="white", fontweight="bold", zorder=4,
                clip_on=True,
            )

    # Axes setup
    ax_bot.set_xlim(0, T)
    ax_bot.set_ylim(0, 2.1)
    ax_bot.set_yticks([BNB_Y + BAR_HEIGHT / 2, DP_Y + BAR_HEIGHT / 2])
    ax_bot.set_yticklabels(["BnB", "DP"], fontsize=10, fontweight="bold")
    ax_bot.set_xlabel("time  t", fontsize=9)
    ax_bot.set_title("Job schedules (bars show job length L / cost, background = price level)",
                     fontsize=9, pad=3)

    # Legend
    dp_patch  = mpatches.Patch(facecolor=COL_DP,  edgecolor="#1a5276", alpha=0.88,
                                label=f"DP   cost = {fmt_cost(dp_cost, dp_timed)}")
    bnb_patch = mpatches.Patch(facecolor=COL_BNB, edgecolor="#784212", alpha=0.88,
                                label=f"BnB  cost = {fmt_cost(bnb_cost, bnb_timed)}")
    ax_bot.legend(handles=[dp_patch, bnb_patch],
                  loc="upper right", fontsize=8.5, framealpha=0.9)

    # ── Save / show ──────────────────────────────────────────────────────────
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    Saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Visualize DP vs BnB schedules.",
        usage="visualize_schedules.py <path> [--ids 0 1 2] [--save-png] [--no-show]\n"
              "  <path> = results directory  OR  a single instance_XXXX.json file",
    )
    ap.add_argument("path", nargs="?", default=None,
                    help="Results dir or single JSON file")
    ap.add_argument("--ids",  type=int, nargs="*", default=None,
                    help="Show only these instance IDs (dir mode)")
    ap.add_argument("--max",  type=int, default=20, dest="max_instances",
                    help="Max instances in dir mode (default: 20)")
    ap.add_argument("--save-png", action="store_true",
                    help="Save PNG beside each JSON (implies --no-show)")
    ap.add_argument("--no-show",  action="store_true",
                    help="Don't open interactive windows")
    args = ap.parse_args()

    if args.path is None:
        ap.print_help(); sys.exit(0)

    show = not args.no_show and not args.save_png
    p    = Path(args.path)

    json_files: List[str] = []
    if p.is_file() and p.suffix == ".json":
        json_files = [str(p)]
    elif p.is_dir():
        all_j = sorted(p.glob("instance_*.json"))
        if args.ids is not None:
            id_set = set(args.ids)
            all_j  = [f for f in all_j if int(f.stem.split("_")[-1]) in id_set]
        json_files = [str(f) for f in all_j[:args.max_instances]]
    else:
        print(f"Path not found: {args.path}")
        print("Tip: compare writes to build/results/<size>_s<seed>/ if run from build/")
        sys.exit(1)

    if not json_files:
        print(f"No instance_XXXX.json files in: {args.path}")
        print("Run:  ./compare <size> <n_instances> <seed>")
        sys.exit(1)

    print(f"Visualizing {len(json_files)} instance(s) from: {p}")
    for jf in json_files:
        data  = load_json(jf)
        save  = jf.replace(".json", ".png") if args.save_png else None
        dp_c  = data["dp"].get("cost",  float("inf"))
        bnb_c = data["bnb"].get("cost", float("inf"))
        match = data.get("cost_match", False)
        tle_d = data["dp"].get("timed_out", False)
        tle_b = data["bnb"].get("timed_out", False)
        print(f"  [{data['instance_id']:3d}]  size={data.get('size','?')}  "
              f"n={data['n_jobs']}  T={data['T']}  "
              f"dp={'TLE' if tle_d else f'{dp_c:.4f}'}  "
              f"bnb={'TLE' if tle_b else f'{bnb_c:.4f}'}  "
              f"{'✓' if match else ('TLE' if tle_d or tle_b else '✗')}")
        plot_instance(data, show=show, save_path=save)


if __name__ == "__main__":
    main()

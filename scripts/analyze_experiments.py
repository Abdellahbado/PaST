#!/usr/bin/env python3
"""
Analyze experiment results from ADP/logs/ subdirectories.

Reads all CSV outputs and produces summary tables + plots:
  - Gap vs sigma curve (Experiment A noise stress)
  - Profile complexity bar chart (Experiment B)
  - Regularization comparison table (Experiment C)
  - Cross-size generalization, speed ratios
  - Noise × profile heatmap (Experiment D)

Usage:
    python scripts/analyze_experiments.py [--log-root ADP/logs] [--out-dir ADP/logs/analysis]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Optional: matplotlib for plots
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_PLT = True
except ImportError:
    HAS_PLT = False


def collect_csvs(log_root: Path) -> Dict[str, pd.DataFrame]:
    """Recursively collect all experiment CSVs, keyed by relative path."""
    result = {}
    for csv_path in sorted(log_root.rglob("*.csv")):
        rel = csv_path.relative_to(log_root)
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            result[str(rel)] = df
        except Exception as e:
            print(f"[warn] Could not read {csv_path}: {e}", file=sys.stderr)
    return result


def parse_tag(filename: str) -> dict:
    """Extract metadata from filename conventions like 'profile_model_trainX_evalY.csv'."""
    parts = filename.replace(".csv", "").split("_")
    info = {"filename": filename}
    for i, p in enumerate(parts):
        if p.startswith("train"):
            info["train_size"] = p.replace("train", "")
        if p.startswith("eval"):
            info["eval_size"] = p.replace("eval", "")
        if p.startswith("s") and p[1:].replace(".", "").isdigit():
            info["sigma"] = float(p[1:])
        if p.startswith("r") and p[1:].replace(".", "").isdigit():
            info["rho"] = float(p[1:])
        if p.startswith("sp") and p[2:].replace(".", "").isdigit():
            info["spike_prob"] = float(p[2:])
    return info


def analyze_regularization(csvs: Dict[str, pd.DataFrame], out_dir: Path):
    """Experiment C: regularization comparison."""
    exp_dir = "exp_regularization_features"
    dfs = []
    for key, df in csvs.items():
        if not key.startswith(exp_dir):
            continue
        tag_info = parse_tag(Path(key).name)
        for col in tag_info:
            df[col] = tag_info[col]
        dfs.append(df)

    if not dfs:
        print("[analyze] No Experiment C data found.")
        return

    all_df = pd.concat(dfs, ignore_index=True)
    print("\n=== Experiment C — Regularization & Features ===")

    # Summary: mean gap by model + feature config
    gap_col = None
    for c in ["gapL_pct", "gap_L_pct", "gapL", "gap_learned"]:
        if c in all_df.columns:
            gap_col = c
            break

    if gap_col:
        summary = all_df.groupby("filename")[gap_col].agg(["mean", "std", "count"])
        print(summary.to_string())
    else:
        print("[info] No gap column found. Available columns:", list(all_df.columns))
        print(all_df.describe())

    all_df.to_csv(out_dir / "exp_C_combined.csv", index=False)
    print(f"[saved] {out_dir / 'exp_C_combined.csv'}")


def analyze_profile_sweep(csvs: Dict[str, pd.DataFrame], out_dir: Path):
    """Experiment B: profile complexity sweep."""
    exp_dir = "exp_profile_sweep"
    dfs = []
    for key, df in csvs.items():
        if not key.startswith(exp_dir):
            continue
        tag_info = parse_tag(Path(key).name)
        for col in tag_info:
            df[col] = tag_info[col]
        dfs.append(df)

    if not dfs:
        print("[analyze] No Experiment B data found.")
        return

    all_df = pd.concat(dfs, ignore_index=True)
    print("\n=== Experiment B — Profile Sweep ===")
    print(f"Total rows: {len(all_df)}")
    all_df.to_csv(out_dir / "exp_B_combined.csv", index=False)
    print(f"[saved] {out_dir / 'exp_B_combined.csv'}")


def analyze_noise_stress(csvs: Dict[str, pd.DataFrame], out_dir: Path):
    """Experiment A: noise stress test."""
    exp_dir = "exp_noise_stress"
    dfs = []
    for key, df in csvs.items():
        if not key.startswith(exp_dir):
            continue
        tag_info = parse_tag(Path(key).name)
        for col in tag_info:
            df[col] = tag_info[col]
        dfs.append(df)

    if not dfs:
        print("[analyze] No Experiment A data found.")
        return

    all_df = pd.concat(dfs, ignore_index=True)
    print("\n=== Experiment A — Noise Stress ===")
    print(f"Total rows: {len(all_df)}")

    if "sigma" in all_df.columns:
        gap_col = None
        for c in ["gapL_pct", "gap_L_pct", "gapL", "gap_learned"]:
            if c in all_df.columns:
                gap_col = c
                break

        if gap_col:
            pivot = all_df.groupby("sigma")[gap_col].agg(["mean", "std"])
            print("\nGap vs sigma:")
            print(pivot.to_string())

            if HAS_PLT:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.errorbar(
                    pivot.index,
                    pivot["mean"],
                    yerr=pivot["std"],
                    marker="o",
                    capsize=5,
                )
                ax.set_xlabel("Noise sigma")
                ax.set_ylabel(f"Mean {gap_col}")
                ax.set_title("Gap vs Noise Level")
                ax.grid(True, alpha=0.3)
                fig.savefig(out_dir / "noise_gap_vs_sigma.pdf", bbox_inches="tight")
                fig.savefig(
                    out_dir / "noise_gap_vs_sigma.png", dpi=150, bbox_inches="tight"
                )
                plt.close(fig)
                print(f"[saved] {out_dir / 'noise_gap_vs_sigma.pdf'}")

    all_df.to_csv(out_dir / "exp_A_combined.csv", index=False)
    print(f"[saved] {out_dir / 'exp_A_combined.csv'}")


def analyze_combined_stress(csvs: Dict[str, pd.DataFrame], out_dir: Path):
    """Experiment D: combined noise + profile stress."""
    exp_dir = "exp_noise_profile_combined"
    dfs = []
    for key, df in csvs.items():
        if not key.startswith(exp_dir):
            continue
        tag_info = parse_tag(Path(key).name)
        for col in tag_info:
            df[col] = tag_info[col]
        dfs.append(df)

    if not dfs:
        print("[analyze] No Experiment D data found.")
        return

    all_df = pd.concat(dfs, ignore_index=True)
    print("\n=== Experiment D — Combined Stress ===")
    print(f"Total rows: {len(all_df)}")
    all_df.to_csv(out_dir / "exp_D_combined.csv", index=False)
    print(f"[saved] {out_dir / 'exp_D_combined.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument(
        "--log-root", default="ADP/logs", help="Root directory for experiment logs"
    )
    parser.add_argument(
        "--out-dir", default="ADP/logs/analysis", help="Output directory for analysis"
    )
    args = parser.parse_args()

    log_root = Path(args.log_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not log_root.exists():
        print(f"Log root not found: {log_root}")
        sys.exit(1)

    csvs = collect_csvs(log_root)
    print(f"Found {len(csvs)} CSV files in {log_root}")

    analyze_regularization(csvs, out_dir)
    analyze_profile_sweep(csvs, out_dir)
    analyze_noise_stress(csvs, out_dir)
    analyze_combined_stress(csvs, out_dir)

    print(f"\n[done] All analysis output in {out_dir}/")


if __name__ == "__main__":
    main()

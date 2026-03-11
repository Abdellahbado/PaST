#!/usr/bin/env python3
"""
07_analyze_ablation.py — Analyze ablation study results and produce paper tables.

Reads the CSV files produced by 06_ablation_study.py and generates:
  1. LaTeX table: Per-config summary (optimality, runtime, speedup)
  2. LaTeX table: Per-step timing breakdown
  3. LaTeX table: Pairwise speedup matrix
  4. Bar chart:   Speedup decomposition (optional, requires matplotlib)
  5. Console:     Detailed text report

Usage:
  python3 hpc/07_analyze_ablation.py [--input-dir hpc/results_ablation]
                                     [--output-dir hpc/results_ablation/analysis]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent.parent

CONFIG_LABELS = {
    "full":         "Full method (A)",
    "full_spaces":  "Full SPACES (B)",
    "exact_only":   "Exact only + banded (C)",
    "baseline":     "Baseline (D)",
}

CONFIG_ORDER = ["full", "full_spaces", "exact_only", "baseline"]

STEP_KEYS = [
    "t_spaces", "t_fwd_relax", "t_heuristic", "t_local_search",
    "t_bwd_relax", "t_two_class", "t_exact",
]
STEP_LABELS = {
    "t_spaces": "SPACES",
    "t_fwd_relax": "Fwd Relax",
    "t_heuristic": "Heuristics",
    "t_local_search": "Local Search",
    "t_bwd_relax": "Bwd Relax",
    "t_two_class": "Two-class",
    "t_exact": "Exact DP",
}


def load_results(input_dir: Path) -> dict[str, list[dict]]:
    """Load per-config CSV files."""
    results = {}
    combined = input_dir / "ablation_combined.csv"
    if combined.exists():
        with open(combined) as f:
            reader = csv.DictReader(f)
            for row in reader:
                cfg = row["config"]
                if cfg not in results:
                    results[cfg] = []
                # Convert numeric fields
                for key in ["n_jobs", "horizon", "feasible", "is_optimal", "timed_out"]:
                    row[key] = int(row[key])
                for key in ["ub", "lb", "gap_pct", "runtime_sec"] + STEP_KEYS:
                    row[key] = float(row[key])
                results[cfg].append(row)
        return results

    # Fallback: load individual files
    for cfg in CONFIG_ORDER:
        path = input_dir / f"ablation_{cfg}.csv"
        if not path.exists():
            continue
        with open(path) as f:
            reader = csv.DictReader(f)
            results[cfg] = []
            for row in reader:
                for key in ["n_jobs", "horizon", "feasible", "is_optimal", "timed_out"]:
                    row[key] = int(row[key])
                for key in ["ub", "lb", "gap_pct", "runtime_sec"] + STEP_KEYS:
                    row[key] = float(row[key])
                results[cfg].append(row)
    return results


def compute_speedups(results: dict[str, list[dict]]) -> dict:
    """Compute pairwise speedup statistics."""
    runtimes = {}
    for cfg, rows in results.items():
        runtimes[cfg] = {r["instance_id"]: r["runtime_sec"] for r in rows}

    pairs = [
        ("full", "full_spaces",  "Banded SPACES", "A vs B"),
        ("full", "exact_only",   "Bound-and-Refine", "A vs C"),
        ("full", "baseline",     "Both components", "A vs D"),
        ("full_spaces", "baseline", "B&R (full SPACES)", "B vs D"),
    ]

    speedup_data = []
    for fast, slow, label, tag in pairs:
        if fast not in runtimes or slow not in runtimes:
            continue
        common = set(runtimes[fast]) & set(runtimes[slow])
        if not common:
            continue

        su_list = []
        for iid in sorted(common):
            t_fast = max(runtimes[fast][iid], 1e-6)
            t_slow = max(runtimes[slow][iid], 1e-6)
            su_list.append(t_slow / t_fast)

        t_fast_total = sum(runtimes[fast][iid] for iid in common)
        t_slow_total = sum(runtimes[slow][iid] for iid in common)

        speedup_data.append({
            "label": label,
            "tag": tag,
            "fast": fast,
            "slow": slow,
            "n": len(common),
            "agg_speedup": t_slow_total / max(t_fast_total, 1e-9),
            "mean_speedup": mean(su_list),
            "median_speedup": median(su_list),
            "min_speedup": min(su_list),
            "max_speedup": max(su_list),
            "t_fast_total": t_fast_total,
            "t_slow_total": t_slow_total,
        })

    return speedup_data


def generate_summary_table(results: dict[str, list[dict]]) -> str:
    """Generate LaTeX table: per-config summary."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation study: per-configuration performance summary.}")
    lines.append(r"\label{tab:ablation-summary}")
    lines.append(r"\begin{tabular}{l r r r r r}")
    lines.append(r"\toprule")
    lines.append(r"Configuration & Instances & Optimal & Avg (s) & Max (s) & Total (s) \\")
    lines.append(r"\midrule")

    for cfg in CONFIG_ORDER:
        if cfg not in results:
            continue
        rows = results[cfg]
        n = len(rows)
        n_opt = sum(1 for r in rows if r["is_optimal"])
        total = sum(r["runtime_sec"] for r in rows)
        avg = total / max(n, 1)
        mx = max(r["runtime_sec"] for r in rows)
        label = CONFIG_LABELS[cfg]
        lines.append(f"  {label} & {n} & {n_opt} & {avg:.3f} & {mx:.3f} & {total:.1f} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_speedup_table(speedup_data: list[dict]) -> str:
    """Generate LaTeX table: pairwise speedup comparisons."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation study: speedup decomposition.}")
    lines.append(r"\label{tab:ablation-speedup}")
    lines.append(r"\begin{tabular}{l l r r r r}")
    lines.append(r"\toprule")
    lines.append(r"Comparison & Component & Agg.\ $\times$ & Med.\ $\times$ & Mean $\times$ & Max $\times$ \\")
    lines.append(r"\midrule")

    for sd in speedup_data:
        lines.append(
            f"  {sd['tag']} & {sd['label']} & "
            f"{sd['agg_speedup']:.2f} & {sd['median_speedup']:.2f} & "
            f"{sd['mean_speedup']:.2f} & {sd['max_speedup']:.2f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_step_time_table(results: dict[str, list[dict]]) -> str:
    """Generate LaTeX table: per-step average timing breakdown."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation study: per-step average timing breakdown (seconds).}")
    lines.append(r"\label{tab:ablation-steps}")

    cols = "l" + " r" * len(STEP_KEYS) + " r"
    lines.append(r"\begin{tabular}{" + cols + "}")
    lines.append(r"\toprule")

    header = "Config"
    for sk in STEP_KEYS:
        header += f" & {STEP_LABELS[sk]}"
    header += r" & Total \\"
    lines.append(header)
    lines.append(r"\midrule")

    for cfg in CONFIG_ORDER:
        if cfg not in results:
            continue
        rows = results[cfg]
        n = max(len(rows), 1)
        avgs = [sum(r[sk] for r in rows) / n for sk in STEP_KEYS]
        total = sum(avgs)
        label = CONFIG_LABELS[cfg].split("(")[0].strip()
        row = f"  {label}"
        for a in avgs:
            row += f" & {a:.4f}"
        row += f" & {total:.4f} \\\\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_text_report(results: dict[str, list[dict]], speedup_data: list[dict]) -> str:
    """Generate detailed text report."""
    lines = []
    lines.append("=" * 90)
    lines.append("  ABLATION STUDY — ANALYSIS REPORT")
    lines.append("=" * 90)

    # Per-config summary
    lines.append("\n1. PER-CONFIGURATION SUMMARY")
    lines.append("-" * 90)
    lines.append(f"  {'Config':>20} {'N':>6} {'Opt':>6} {'Avg(s)':>10} {'Max(s)':>10} {'Total(s)':>10}")
    lines.append("  " + "-" * 70)

    for cfg in CONFIG_ORDER:
        if cfg not in results:
            continue
        rows = results[cfg]
        n = len(rows)
        n_opt = sum(1 for r in rows if r["is_optimal"])
        total = sum(r["runtime_sec"] for r in rows)
        avg = total / max(n, 1)
        mx = max(r["runtime_sec"] for r in rows)
        lines.append(f"  {CONFIG_LABELS[cfg]:>20} {n:>6} {n_opt:>6} {avg:>10.4f} {mx:>10.3f} {total:>10.1f}")

    # Speedup decomposition
    lines.append("\n2. SPEEDUP DECOMPOSITION")
    lines.append("-" * 90)

    for sd in speedup_data:
        lines.append(f"\n  {sd['tag']}: {sd['label']}")
        lines.append(f"    Instances:          {sd['n']}")
        lines.append(f"    Aggregate speedup:  {sd['agg_speedup']:.2f}x "
                     f"({sd['t_fast_total']:.1f}s vs {sd['t_slow_total']:.1f}s)")
        lines.append(f"    Per-instance:       mean={sd['mean_speedup']:.2f}x  "
                     f"med={sd['median_speedup']:.2f}x  "
                     f"min={sd['min_speedup']:.2f}x  max={sd['max_speedup']:.2f}x")

    # Per-step timing
    lines.append("\n3. PER-STEP AVERAGE TIMING (seconds)")
    lines.append("-" * 90)
    header = f"  {'Config':>20}"
    for sk in STEP_KEYS:
        header += f" {STEP_LABELS[sk]:>10}"
    header += f" {'TOTAL':>10}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for cfg in CONFIG_ORDER:
        if cfg not in results:
            continue
        rows = results[cfg]
        n = max(len(rows), 1)
        avgs = [sum(r[sk] for r in rows) / n for sk in STEP_KEYS]
        total = sum(avgs)
        row = f"  {cfg:>20}"
        for a in avgs:
            row += f" {a:>10.4f}"
        row += f" {total:>10.4f}"
        lines.append(row)

    # Step-reached distribution
    lines.append("\n4. STEP-REACHED DISTRIBUTION")
    lines.append("-" * 90)
    for cfg in CONFIG_ORDER:
        if cfg not in results:
            continue
        rows = results[cfg]
        dist = defaultdict(int)
        for r in rows:
            dist[r["step_reached"]] += 1
        lines.append(f"  {cfg:>20}: " + "  ".join(
            f"{step}={count}" for step, count in sorted(dist.items(), key=lambda x: -x[1])
        ))

    # Key findings
    lines.append("\n5. KEY FINDINGS")
    lines.append("-" * 90)

    if len(speedup_data) >= 3:
        banded = next((s for s in speedup_data if s["tag"] == "A vs B"), None)
        bnr = next((s for s in speedup_data if s["tag"] == "A vs C"), None)
        total = next((s for s in speedup_data if s["tag"] == "A vs D"), None)

        if banded and bnr and total:
            lines.append(f"  - Banded SPACES alone provides {banded['agg_speedup']:.1f}x aggregate speedup")
            lines.append(f"  - Bound-and-Refine alone provides {bnr['agg_speedup']:.1f}x aggregate speedup")
            lines.append(f"  - Combined, both components provide {total['agg_speedup']:.1f}x aggregate speedup")

            # Multiplicative decomposition
            mult = banded['agg_speedup'] * bnr['agg_speedup']
            if total['agg_speedup'] > 0:
                synergy = total['agg_speedup'] / mult
                if synergy > 1.05:
                    lines.append(f"  - Synergy: Combined speedup is {synergy:.2f}x more than "
                                 "the product of individual speedups (positive interaction)")
                elif synergy < 0.95:
                    lines.append(f"  - Overlap: Combined speedup is {synergy:.2f}x of "
                                 "the product (diminishing returns)")
                else:
                    lines.append(f"  - Components are roughly independent "
                                 f"(combined/product ratio: {synergy:.2f})")

    lines.append("")
    return "\n".join(lines)


def try_generate_figure(results: dict[str, list[dict]], speedup_data: list[dict], out_dir: Path):
    """Generate a bar chart if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  (matplotlib not available — skipping figure generation)")
        return

    # Figure 1: Speedup bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [sd["label"] for sd in speedup_data]
    agg_vals = [sd["agg_speedup"] for sd in speedup_data]
    med_vals = [sd["median_speedup"] for sd in speedup_data]

    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax.bar(x - w / 2, agg_vals, w, label="Aggregate", color="#2196F3")
    bars2 = ax.bar(x + w / 2, med_vals, w, label="Median", color="#FF9800")

    ax.set_ylabel("Speedup (×)")
    ax.set_title("Ablation Study: Speedup Decomposition")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend()
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylim(bottom=0)

    for bar in bars1:
        ax.annotate(f"{bar.get_height():.1f}×",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "ablation_speedup.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "ablation_speedup.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'ablation_speedup.pdf'}")

    # Figure 2: Per-step timing stacked bar
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    cfgs = [c for c in CONFIG_ORDER if c in results]
    n_cfgs = len(cfgs)
    bottoms = np.zeros(n_cfgs)

    colors = ["#E53935", "#FB8C00", "#43A047", "#1E88E5", "#8E24AA", "#00ACC1", "#6D4C41"]

    for i, sk in enumerate(STEP_KEYS):
        vals = []
        for cfg in cfgs:
            rows = results[cfg]
            n = max(len(rows), 1)
            vals.append(sum(r[sk] for r in rows) / n)
        vals = np.array(vals)
        ax2.bar(range(n_cfgs), vals, bottom=bottoms, label=STEP_LABELS[sk],
                color=colors[i % len(colors)])
        bottoms += vals

    ax2.set_ylabel("Average time (s)")
    ax2.set_title("Ablation Study: Per-Step Timing Breakdown")
    ax2.set_xticks(range(n_cfgs))
    ax2.set_xticklabels([CONFIG_LABELS[c] for c in cfgs], rotation=15, ha="right")
    ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    fig2.tight_layout()
    fig2.savefig(out_dir / "ablation_steps.pdf", dpi=150, bbox_inches="tight")
    fig2.savefig(out_dir / "ablation_steps.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {out_dir / 'ablation_steps.pdf'}")


def main():
    ap = argparse.ArgumentParser(description="Analyze ablation study results")
    ap.add_argument("--input-dir", default=str(ROOT / "hpc" / "results_ablation"))
    ap.add_argument("--output-dir", default=None,
                    help="Output directory for analysis (default: <input-dir>/analysis)")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        print("  Run 06_ablation_study.py first.")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else input_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    results = load_results(input_dir)
    if not results:
        print("ERROR: No results found. Run 06_ablation_study.py first.")
        sys.exit(1)

    print(f"Loaded configs: {list(results.keys())}")
    for cfg, rows in results.items():
        print(f"  {cfg}: {len(rows)} instances")

    # Compute speedups
    speedup_data = compute_speedups(results)

    # Generate text report
    report = generate_text_report(results, speedup_data)
    print(report)
    with open(out_dir / "ablation_report.txt", "w") as f:
        f.write(report)

    # Generate LaTeX tables
    summary_table = generate_summary_table(results)
    speedup_table = generate_speedup_table(speedup_data)
    step_table = generate_step_time_table(results)

    latex_path = out_dir / "ablation_tables.tex"
    with open(latex_path, "w") as f:
        f.write("% Ablation study tables — auto-generated by 07_analyze_ablation.py\n")
        f.write("% Include in paper with: \\input{ablation_tables}\n\n")
        f.write(summary_table + "\n\n")
        f.write(speedup_table + "\n\n")
        f.write(step_table + "\n")
    print(f"\nLaTeX tables saved: {latex_path}")

    # Try generating figures
    print("\nGenerating figures...")
    try_generate_figure(results, speedup_data, out_dir)

    print(f"\nAll analysis outputs in: {out_dir}/")
    print("  ablation_report.txt   — Detailed text report")
    print("  ablation_tables.tex   — LaTeX tables for paper")
    print("  ablation_speedup.pdf  — Speedup bar chart (if matplotlib available)")
    print("  ablation_steps.pdf    — Per-step timing breakdown (if matplotlib available)")


if __name__ == "__main__":
    main()

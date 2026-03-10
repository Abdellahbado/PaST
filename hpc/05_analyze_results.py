#!/usr/bin/env python3
"""
05_analyze_results.py — Comprehensive analysis & comparison of both solvers.

Reads CSV files from hpc/results_ours/ and hpc/results_paper/,
produces:
  1. Per-section summary tables (Table 1, Table 2, Figure 9, Drops, GCD, Synthetic)
  2. Per-group analysis (processing time groups × n-values)
  3. Head-to-head comparison (our solver vs paper's solver)
  4. Timing breakdown (avg, max, median, p95 per group)
  5. Regression check (any new gaps introduced?)
  6. LaTeX-ready tables for the paper
  7. Optional scatter plot (our time vs paper's time)

Usage:
  python3 hpc/05_analyze_results.py [--ours hpc/results_ours] [--paper hpc/results_paper] \\
                                    [--output hpc/analysis]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Processing time groups for Table 2 (benedikt2025b_groups)
TABLE2_GROUPS_ORDER = [
    "{2,4}",
    "{1,2,3,5,7}",
    "{2,4,6,8,10}",
    "{1,2,3,4,5,6,7,8,9,10}",
    "{3,7}",
    "{3,5,6,7}",
    "{8,10}",
]

# Processing time groups for Figure 9 (benedikt2025_groups)
FIG9_GROUPS_ORDER = [
    "{10}",
    "{1,2,10}",
    "{7,9}",
    "{8,9}",
    "{9,10}",
    "{7,8,9,10}",
    "{8,9,10}",
]

# Instance data dir for extracting processing time groups
INSTANCE_DATA = (
    ROOT
    / "data"
    / "green-scheduling-bab"
    / "Iirc.EnergyStatesAndCostsScheduling"
    / "data"
    / "datasets"
)


def load_csv(path: Path) -> list[dict]:
    """Load results CSV into list of dicts."""
    if not path.exists():
        return []
    results = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for k in [
                "n_jobs",
                "horizon",
                "feasible",
                "is_optimal",
                "timed_out",
                "peak_rss_kb",
            ]:
                if k in row and row[k]:
                    try:
                        row[k] = int(row[k])
                    except (ValueError, TypeError):
                        row[k] = 0
            for k in ["ub", "lb", "gap_pct", "runtime_sec"]:
                if k in row and row[k]:
                    try:
                        row[k] = float(row[k])
                    except (ValueError, TypeError):
                        row[k] = 0.0
            results.append(row)
    return results


def get_instance_group(dataset: str, instance_id: str) -> str:
    """Extract processing time group from instance by reading its JSON."""
    parts = instance_id.split("/")
    if len(parts) != 2:
        return "unknown"
    ds_name, idx_str = parts
    inst_path = INSTANCE_DATA / ds_name / f"{idx_str}.json"
    if not inst_path.exists():
        return "unknown"
    try:
        with open(inst_path) as f:
            d = json.load(f)
        procs = sorted(set(j["ProcessingTime"] for j in d["Jobs"]))
        return "{" + ",".join(str(p) for p in procs) + "}"
    except Exception:
        return "unknown"


def percentile(values: list[float], p: float) -> float:
    """Compute p-th percentile (0-100)."""
    if not values:
        return 0.0
    vs = sorted(values)
    k = (len(vs) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(vs):
        return vs[-1]
    return vs[f] + (k - f) * (vs[c] - vs[f])


def print_section_table(section: str, results: list[dict], out) -> dict:
    """Print and return a summary table for one section."""
    if not results:
        out.write(f"\n{'='*100}\n{section}: NO RESULTS\n{'='*100}\n")
        return {}

    n = len(results)
    n_opt = sum(1 for r in results if r["is_optimal"])
    n_feas = sum(1 for r in results if r["feasible"])
    n_infeas = n - n_feas
    times = [r["runtime_sec"] for r in results]
    total_t = sum(times)
    avg_t = total_t / max(n, 1)
    max_t = max(times, default=0)
    med_t = percentile(times, 50)
    p95_t = percentile(times, 95)

    out.write(f"\n{'='*100}\n")
    out.write(f"  {section}\n")
    out.write(f"{'='*100}\n")
    out.write(f"  Instances:      {n}\n")
    out.write(f"  Feasible:       {n_feas} (+ {n_infeas} infeasible)\n")
    out.write(f"  Optimal:        {n_opt}/{n_feas} ({100*n_opt/max(n_feas,1):.1f}%)\n")
    out.write(f"  Avg time:       {avg_t:.4f}s\n")
    out.write(f"  Median time:    {med_t:.4f}s\n")
    out.write(f"  P95 time:       {p95_t:.4f}s\n")
    out.write(f"  Max time:       {max_t:.4f}s\n")
    out.write(f"  Total time:     {total_t:.2f}s\n")

    # Open gaps
    open_gaps = [r for r in results if r["feasible"] and not r["is_optimal"]]
    if open_gaps:
        out.write(f"\n  *** {len(open_gaps)} OPEN GAPS ***\n")
        for r in open_gaps:
            out.write(
                f"    {r['instance_id']:>50} ub={r['ub']:.1f} lb={r['lb']:.1f} "
                f"gap={r['gap_pct']:.4f}% t={r['runtime_sec']:.3f}s\n"
            )

    return {
        "n": n,
        "n_opt": n_opt,
        "n_feas": n_feas,
        "avg_t": avg_t,
        "max_t": max_t,
        "med_t": med_t,
        "p95_t": p95_t,
        "total_t": total_t,
    }


def print_group_analysis(
    section: str, results: list[dict], groups_order: list[str], out
):
    """Detailed breakdown by processing-time group and n-value."""
    # Build group mapping (cache for speed)
    group_cache = {}
    for r in results:
        iid = r["instance_id"]
        if iid not in group_cache:
            group_cache[iid] = get_instance_group(r.get("dataset", ""), iid)

    # Group results by (group, n)
    by_group_n = defaultdict(list)
    by_group = defaultdict(list)
    by_n = defaultdict(list)

    for r in results:
        g = group_cache.get(r["instance_id"], "unknown")
        n = r["n_jobs"]
        by_group_n[(g, n)].append(r)
        by_group[g].append(r)
        by_n[n].append(r)

    # Table: per group
    out.write(f"\n  --- By processing time group ---\n")
    out.write(
        f"  {'Group':>25} {'K':>3} {'#':>5} {'Opt':>5} {'Avg(s)':>10} "
        f"{'Med(s)':>10} {'P95(s)':>10} {'Max(s)':>10} {'Sum(s)':>10}\n"
    )
    out.write(f"  {'-'*95}\n")

    for g in groups_order:
        gr = by_group.get(g, [])
        if not gr:
            continue
        gn = len(gr)
        g_opt = sum(1 for r in gr if r["is_optimal"])
        g_times = [r["runtime_sec"] for r in gr]
        out.write(
            f"  {g:>25} {len(set(int(x) for x in g[1:-1].split(','))):>3} {gn:>5} {g_opt:>5} "
            f"{sum(g_times)/gn:>10.4f} {percentile(g_times, 50):>10.4f} "
            f"{percentile(g_times, 95):>10.4f} {max(g_times):>10.4f} {sum(g_times):>10.1f}\n"
        )

    # Table: per n
    out.write(f"\n  --- By job count (n) ---\n")
    out.write(
        f"  {'n':>5} {'#':>5} {'Opt':>5} {'Avg(s)':>10} {'Med(s)':>10} "
        f"{'P95(s)':>10} {'Max(s)':>10} {'Sum(s)':>10}\n"
    )
    out.write(f"  {'-'*75}\n")

    for n_val in sorted(by_n.keys()):
        nr = by_n[n_val]
        nn = len(nr)
        n_opt = sum(1 for r in nr if r["is_optimal"])
        n_times = [r["runtime_sec"] for r in nr]
        out.write(
            f"  {n_val:>5} {nn:>5} {n_opt:>5} "
            f"{sum(n_times)/nn:>10.4f} {percentile(n_times, 50):>10.4f} "
            f"{percentile(n_times, 95):>10.4f} {max(n_times):>10.4f} {sum(n_times):>10.1f}\n"
        )

    # Detailed grid: group × n
    out.write(f"\n  --- Group × n grid (avg time / max time) ---\n")
    all_n = sorted(set(r["n_jobs"] for r in results))
    header = f"  {'Group':>25}" + "".join(f" {'n='+str(n):>20}" for n in all_n)
    out.write(header + "\n")
    out.write(f"  {'-'*(25 + 21*len(all_n))}\n")

    for g in groups_order:
        row = f"  {g:>25}"
        for n_val in all_n:
            cell = by_group_n.get((g, n_val), [])
            if cell:
                avg = sum(r["runtime_sec"] for r in cell) / len(cell)
                mx = max(r["runtime_sec"] for r in cell)
                n_opt = sum(1 for r in cell if r["is_optimal"])
                tag = f"{avg:.2f}/{mx:.1f}s"
                if n_opt < len(cell):
                    tag += f" [{n_opt}/{len(cell)}]"
                row += f" {tag:>20}"
            else:
                row += " " * 20 + "-"
        out.write(row + "\n")


def print_comparison(ours: list[dict], paper: list[dict], out):
    """Head-to-head comparison of our solver vs paper's solver."""
    out.write(f"\n{'='*100}\n")
    out.write(f"  HEAD-TO-HEAD COMPARISON: Our Solver vs Paper's B&B-SPACES\n")
    out.write(f"{'='*100}\n")

    if not ours or not paper:
        out.write("  (Comparison requires both result sets)\n")
        return

    # Index paper results by instance_id
    paper_idx = {}
    for r in paper:
        paper_idx[r["instance_id"]] = r

    matched = 0
    ours_faster = 0
    paper_faster = 0
    ours_only_opt = 0
    paper_only_opt = 0
    both_opt = 0
    speedups = []

    for r in ours:
        iid = r["instance_id"]
        if iid not in paper_idx:
            continue
        pr = paper_idx[iid]
        matched += 1

        our_opt = r["is_optimal"]
        paper_opt = pr["is_optimal"]

        if our_opt and paper_opt:
            both_opt += 1
        elif our_opt and not paper_opt:
            ours_only_opt += 1
        elif not our_opt and paper_opt:
            paper_only_opt += 1

        our_t = r["runtime_sec"]
        paper_t = pr["runtime_sec"]

        if our_t > 0 and paper_t > 0:
            speedup = paper_t / our_t
            speedups.append(speedup)
            if our_t < paper_t:
                ours_faster += 1
            elif paper_t < our_t:
                paper_faster += 1

    out.write(f"  Matched instances:      {matched}\n")
    out.write(f"  Both optimal:           {both_opt}\n")
    out.write(f"  Only ours optimal:      {ours_only_opt}\n")
    out.write(f"  Only paper optimal:     {paper_only_opt}\n")
    out.write(f"  Ours faster:            {ours_faster}\n")
    out.write(f"  Paper faster:           {paper_faster}\n")

    if speedups:
        avg_sp = sum(speedups) / len(speedups)
        med_sp = percentile(speedups, 50)
        min_sp = min(speedups)
        max_sp = max(speedups)
        p25_sp = percentile(speedups, 25)
        p75_sp = percentile(speedups, 75)
        out.write(f"\n  Speedup (paper_time / our_time):\n")
        out.write(f"    Mean:   {avg_sp:.2f}x\n")
        out.write(f"    Median: {med_sp:.2f}x\n")
        out.write(f"    Min:    {min_sp:.2f}x\n")
        out.write(f"    Max:    {max_sp:.2f}x\n")
        out.write(f"    P25:    {p25_sp:.2f}x\n")
        out.write(f"    P75:    {p75_sp:.2f}x\n")

    # Comparison by section
    by_section_ours = defaultdict(list)
    by_section_paper = defaultdict(list)
    for r in ours:
        by_section_ours[r.get("section", "")].append(r)
    for r in paper:
        by_section_paper[r.get("section", "")].append(r)

    out.write(f"\n  --- Per-section comparison ---\n")
    out.write(
        f"  {'Section':>15} {'Ours Opt':>10} {'Paper Opt':>10} {'Ours Avg':>10} "
        f"{'Paper Avg':>10} {'Speedup':>10}\n"
    )
    out.write(f"  {'-'*75}\n")

    for sec in ["table2", "fig9", "drops", "gcd"]:
        our_s = by_section_ours.get(sec, [])
        pap_s = by_section_paper.get(sec, [])
        if not our_s or not pap_s:
            continue

        our_n = len(our_s)
        our_opt = sum(1 for r in our_s if r["is_optimal"])
        our_avg = sum(r["runtime_sec"] for r in our_s) / max(our_n, 1)

        pap_n = len(pap_s)
        pap_opt = sum(1 for r in pap_s if r["is_optimal"])
        pap_avg = sum(r["runtime_sec"] for r in pap_s) / max(pap_n, 1)

        sp = pap_avg / our_avg if our_avg > 0 else float("inf")

        out.write(
            f"  {sec:>15} {our_opt:>5}/{our_n:<4} {pap_opt:>5}/{pap_n:<4} "
            f"{our_avg:>10.3f} {pap_avg:>10.3f} {sp:>10.2f}x\n"
        )


def print_latex_tables(ours: list[dict], out):
    """Generate LaTeX-ready tables for the paper."""
    out.write(f"\n{'='*100}\n")
    out.write(f"  LATEX TABLES (copy-paste into paper)\n")
    out.write(f"{'='*100}\n\n")

    # Table 2 by group
    table2 = [r for r in ours if r.get("section") == "table2"]
    if table2:
        group_cache = {}
        for r in table2:
            iid = r["instance_id"]
            if iid not in group_cache:
                group_cache[iid] = get_instance_group(r.get("dataset", ""), iid)

        by_group = defaultdict(list)
        for r in table2:
            g = group_cache.get(r["instance_id"], "unknown")
            by_group[g].append(r)

        out.write("  % Table 2 results by group\n")
        out.write("  \\begin{tabular}{lrrrrr}\n")
        out.write("  \\toprule\n")
        out.write("  Group & $K$ & \\#Opt & Avg (s) & Max (s) & Total (s) \\\\\n")
        out.write("  \\midrule\n")

        for g in TABLE2_GROUPS_ORDER:
            gr = by_group.get(g, [])
            if not gr:
                continue
            K = len(set(int(x) for x in g[1:-1].split(",")))
            n_opt = sum(1 for r in gr if r["is_optimal"])
            times = [r["runtime_sec"] for r in gr]
            out.write(
                f"  {g} & {K} & {n_opt}/{len(gr)} & {sum(times)/len(gr):.2f} "
                f"& {max(times):.2f} & {sum(times):.1f} \\\\\n"
            )

        out.write("  \\bottomrule\n")
        out.write("  \\end{tabular}\n\n")

    # Figure 9 by group
    fig9 = [r for r in ours if r.get("section") == "fig9"]
    if fig9:
        group_cache = {}
        for r in fig9:
            iid = r["instance_id"]
            if iid not in group_cache:
                group_cache[iid] = get_instance_group(r.get("dataset", ""), iid)

        by_group = defaultdict(list)
        for r in fig9:
            g = group_cache.get(r["instance_id"], "unknown")
            by_group[g].append(r)

        out.write("  % Figure 9 results by group\n")
        out.write("  \\begin{tabular}{lrrrrr}\n")
        out.write("  \\toprule\n")
        out.write("  Group & $K$ & \\#Opt & Avg (s) & Max (s) & Total (s) \\\\\n")
        out.write("  \\midrule\n")

        for g in FIG9_GROUPS_ORDER:
            gr = by_group.get(g, [])
            if not gr:
                continue
            K = len(set(int(x) for x in g[1:-1].split(",")))
            n_opt = sum(1 for r in gr if r["is_optimal"])
            times = [r["runtime_sec"] for r in gr]
            out.write(
                f"  {g} & {K} & {n_opt}/{len(gr)} & {sum(times)/len(gr):.2f} "
                f"& {max(times):.2f} & {sum(times):.1f} \\\\\n"
            )

        out.write("  \\bottomrule\n")
        out.write("  \\end{tabular}\n\n")


def generate_scatter_plot(ours: list[dict], paper: list[dict], out_dir: Path):
    """Generate scatter plot comparing runtimes (requires matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    paper_idx = {r["instance_id"]: r for r in paper}

    our_times = []
    paper_times = []
    colors = []
    section_colors = {"table2": "blue", "fig9": "red", "drops": "green", "gcd": "black"}

    for r in ours:
        iid = r["instance_id"]
        if iid not in paper_idx:
            continue
        pr = paper_idx[iid]
        if r["runtime_sec"] > 0 and pr["runtime_sec"] > 0:
            our_times.append(r["runtime_sec"])
            paper_times.append(pr["runtime_sec"])
            colors.append(section_colors.get(r.get("section", ""), "gray"))

    if not our_times:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(paper_times, our_times, c=colors, alpha=0.5, s=10)

    # Diagonal
    max_val = max(max(our_times), max(paper_times)) * 1.1
    ax.plot([0.001, max_val], [0.001, max_val], "k--", alpha=0.3, label="Equal")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Paper's B&B-SPACES time (s)")
    ax.set_ylabel("Our Relaxed DP time (s)")
    ax.set_title("Runtime Comparison: Our Solver vs Paper's B&B-SPACES")
    ax.set_aspect("equal")
    ax.legend()

    plot_path = out_dir / "scatter_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Analyze benchmark results")
    ap.add_argument("--ours", default=str(ROOT / "hpc" / "results_ours"))
    ap.add_argument("--paper", default=str(ROOT / "hpc" / "results_paper"))
    ap.add_argument("--output", default=str(ROOT / "hpc" / "analysis"))
    args = ap.parse_args()

    ours_dir = Path(args.ours)
    paper_dir = Path(args.paper)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    ours_all = load_csv(ours_dir / "all_results.csv")
    paper_all = load_csv(paper_dir / "all_results.csv")

    report_path = out_dir / "analysis_report.txt"
    out = open(report_path, "w")

    out.write("=" * 100 + "\n")
    out.write("  PaST HPC Benchmark — Comprehensive Analysis Report\n")
    out.write("=" * 100 + "\n\n")

    # 1. Our solver results by section
    by_section = defaultdict(list)
    for r in ours_all:
        by_section[r.get("section", "unknown")].append(r)

    section_summaries = {}
    for sec_name, sec_label in [
        ("table1", "TABLE 1 (§5.1) — 72 instances"),
        ("table2", "TABLE 2 (§5.2) — 560 instances"),
        ("fig9", "FIGURE 9 (§5.3) — 560 instances"),
        ("drops", "DROPS ABLATION — 240 instances"),
        ("gcd", "GCD ANALYSIS — 1 instance"),
        ("synthetic", "HARD SYNTHETIC — 84 instances"),
    ]:
        results = by_section.get(sec_name, [])
        if results:
            summary = print_section_table(sec_label, results, out)
            section_summaries[sec_name] = summary

            # Group analysis for Table 2 and Figure 9
            if sec_name == "table2":
                print_group_analysis("Table 2", results, TABLE2_GROUPS_ORDER, out)
            elif sec_name == "fig9":
                print_group_analysis("Figure 9", results, FIG9_GROUPS_ORDER, out)

    # 2. Regression check
    out.write(f"\n{'='*100}\n")
    out.write(f"  REGRESSION CHECK\n")
    out.write(f"{'='*100}\n")

    total = len(ours_all)
    total_opt = sum(1 for r in ours_all if r["is_optimal"])
    total_feas = sum(1 for r in ours_all if r["feasible"])
    total_infeas = total - total_feas
    open_gaps = [r for r in ours_all if r["feasible"] and not r["is_optimal"]]

    out.write(f"  Total instances tested: {total}\n")
    out.write(f"  Feasible + optimal:     {total_opt}\n")
    out.write(f"  Infeasible:             {total_infeas}\n")
    out.write(f"  Open gaps:              {len(open_gaps)}\n")

    expected_optimal = {
        "table1": 70,  # 72 total - 2 infeasible
        "table2": 560,
        "fig9": 560,
        "drops": 240,
        "gcd": 1,
        "synthetic": 84,
    }

    regressions = []
    for sec, expected in expected_optimal.items():
        actual = sum(1 for r in by_section.get(sec, []) if r["is_optimal"])
        if actual < expected:
            regressions.append((sec, expected, actual))
            out.write(
                f"  *** REGRESSION in {sec}: expected {expected} optimal, got {actual} ***\n"
            )

    if not regressions:
        out.write(f"\n  *** NO REGRESSIONS — All sections match expected counts ***\n")
    else:
        out.write(f"\n  *** {len(regressions)} REGRESSIONS DETECTED ***\n")
        for sec, expected, actual in regressions:
            gap_instances = [
                r
                for r in by_section.get(sec, [])
                if r["feasible"] and not r["is_optimal"]
            ]
            for r in gap_instances:
                out.write(f"    {r['instance_id']} gap={r['gap_pct']:.4f}%\n")

    # 3. Comparison
    if paper_all:
        print_comparison(ours_all, paper_all, out)

    # 4. LaTeX tables
    if ours_all:
        print_latex_tables(ours_all, out)

    # 5. Grand summary
    out.write(f"\n{'='*100}\n")
    out.write(f"  GRAND SUMMARY\n")
    out.write(f"{'='*100}\n")
    out.write(
        f"  {'Section':>15} {'Opt':>10} {'Total':>8} {'Avg(s)':>10} {'Max(s)':>10} "
        f"{'P95(s)':>10} {'Sum(s)':>10}\n"
    )
    out.write(f"  {'-'*85}\n")

    for sec in ["table1", "table2", "fig9", "drops", "gcd", "synthetic"]:
        sr = by_section.get(sec, [])
        if not sr:
            continue
        sn = len(sr)
        so = sum(1 for r in sr if r["is_optimal"])
        times = [r["runtime_sec"] for r in sr]
        out.write(
            f"  {sec:>15} {so:>5}/{sn:<4} {sn:>8} {sum(times)/sn:>10.4f} "
            f"{max(times):>10.4f} {percentile(times, 95):>10.4f} {sum(times):>10.1f}\n"
        )

    out.write(
        f"\n  Total: {total_opt}/{total_feas} optimal + {total_infeas} infeasible"
        f" = {total} instances\n"
    )
    out.write(f"  Open gaps: {len(open_gaps)}\n")
    out.write(f"\n  Report saved to: {report_path}\n")

    out.close()

    # 6. Scatter plot
    if paper_all and ours_all:
        generate_scatter_plot(ours_all, paper_all, out_dir)

    # Print summary to console
    print(f"\nAnalysis complete. Report: {report_path}")
    print(f"  Our solver: {total_opt}/{total_feas} optimal, {len(open_gaps)} open gaps")
    if paper_all:
        p_opt = sum(1 for r in paper_all if r["is_optimal"])
        p_total = len(paper_all)
        print(f"  Paper solver: {p_opt}/{p_total} optimal")


if __name__ == "__main__":
    main()

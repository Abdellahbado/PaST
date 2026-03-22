#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

from common import SOLVER, as_float, as_int, load_instances, log, run_relaxation_batch, write_csv


def distinct_signature(inst: dict) -> str:
    return "-".join(str(x) for x in sorted(set(inst["jobs"])))


def gcd_all(vals: list[int]) -> int:
    g = 0
    for v in vals:
        g = math.gcd(g, v)
    return max(g, 1)


def add_metadata(rows: list[dict], inst_map: dict[str, dict]):
    for row in rows:
        inst = inst_map[row["instance_id"]]
        sig = distinct_signature(inst)
        lengths = sorted(set(inst["jobs"]))
        row["length_signature"] = sig
        row["distinct_lengths"] = str(len(lengths))
        row["gcd_lengths"] = str(gcd_all(lengths))
        row["has_unit"] = "1" if 1 in lengths else "0"
        opt = as_float(row, "opt")
        for mode in ("unit", "gcd", "semi"):
            lb = as_float(row, f"lb_{mode}")
            abs_gap = max(opt - lb, 0.0) if opt > 0 else 0.0
            rel_gap = (100.0 * abs_gap / opt) if opt > 0 else 0.0
            row[f"abs_gap_{mode}"] = f"{abs_gap:.6f}"
            row[f"rel_gap_pct_{mode}"] = f"{rel_gap:.6f}"


def summarize_mode(rows: list[dict], mode: str) -> str:
    vals = [as_float(r, f"rel_gap_pct_{mode}") for r in rows if as_int(r, "is_optimal") == 1]
    times = [as_float(r, f"t_{mode}") for r in rows]
    if not vals:
        return f"- `{mode}`: no optimal reference rows"
    return (
        f"- `{mode}`: mean rel gap {sum(vals)/len(vals):.4f}%, "
        f"median {sorted(vals)[len(vals)//2]:.4f}%, "
        f"max {max(vals):.4f}%, avg runtime {sum(times)/len(times):.4f}s"
    )


def build_report(rows: list[dict]) -> str:
    optimal_rows = [r for r in rows if as_int(r, "is_optimal") == 1 and as_float(r, "opt") > 0]
    lines = ["# Relaxation Quality Report", "", f"Instances: {len(rows)}", ""]

    lines.append("## Overall")
    lines.append(summarize_mode(optimal_rows, "unit"))
    lines.append(summarize_mode(optimal_rows, "gcd"))
    lines.append(summarize_mode(optimal_rows, "semi"))

    strict_unit_gcd = sum(
        1 for r in optimal_rows if as_float(r, "lb_gcd") > as_float(r, "lb_unit") + 1e-6
    )
    strict_gcd_semi = sum(
        1 for r in optimal_rows if as_float(r, "lb_semi") > as_float(r, "lb_gcd") + 1e-6
    )
    semi_exact = sum(
        1 for r in optimal_rows if abs(as_float(r, "opt") - as_float(r, "lb_semi")) < 1e-6
    )
    lines.append(
        f"- strict improvements: unit→gcd on {strict_unit_gcd}/{len(optimal_rows)}, "
        f"gcd→semi on {strict_gcd_semi}/{len(optimal_rows)}, "
        f"semi=opt on {semi_exact}/{len(optimal_rows)}."
    )

    lines.extend(["", "## By Length Signature"])
    by_sig: dict[str, list[dict]] = defaultdict(list)
    for row in optimal_rows:
        by_sig[row["length_signature"]].append(row)

    for sig, group in sorted(by_sig.items()):
        strict_u_g = sum(1 for r in group if as_float(r, "lb_gcd") > as_float(r, "lb_unit") + 1e-6)
        strict_g_s = sum(1 for r in group if as_float(r, "lb_semi") > as_float(r, "lb_gcd") + 1e-6)
        semi_eq = sum(1 for r in group if abs(as_float(r, "opt") - as_float(r, "lb_semi")) < 1e-6)
        unit_gap = sum(as_float(r, "rel_gap_pct_unit") for r in group) / len(group)
        gcd_gap = sum(as_float(r, "rel_gap_pct_gcd") for r in group) / len(group)
        semi_gap = sum(as_float(r, "rel_gap_pct_semi") for r in group) / len(group)
        lines.append(
            f"- `{sig}` ({len(group)} inst): "
            f"mean gap unit/gcd/semi = {unit_gap:.4f}% / {gcd_gap:.4f}% / {semi_gap:.4f}%; "
            f"strict unit→gcd = {strict_u_g}, strict gcd→semi = {strict_g_s}, semi=opt = {semi_eq}."
        )

    lines.extend(["", "## Interpretation"])
    lines.append(
        "- `unit→gcd` isolates the benefit of respecting divisibility, while `gcd→semi` isolates the extra tightening from numerical-semigroup reachability."
    )
    lines.append(
        "- `semi=opt` counts the instances where the semigroup relaxation is already exact in lower-bound quality."
    )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Unit vs GCD vs semigroup relaxation-quality study")
    ap.add_argument("--section", default="2", choices=["all", "1", "2", "table1", "table2", "fig9"])
    ap.add_argument("--dataset-substr", default="")
    ap.add_argument("--max-instances", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--solver", default=str(SOLVER))
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    solver_path = Path(args.solver)
    out_dir = Path(args.output_dir or "hpc/results_studies/relaxation_quality")
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = open(out_dir / "run.log", "w")

    log("=== Relaxation Quality Study ===", logfile)
    log(f"Solver: {solver_path}", logfile)
    instances = load_instances(args.section, args.dataset_substr, logfile)
    if args.max_instances > 0:
        instances = instances[: args.max_instances]
    log(f"Instances: {len(instances)}", logfile)

    inst_map = {inst_id: inst for _, _, inst_id, inst in instances}
    rows = []
    for start in range(0, len(instances), args.batch_size):
        batch = instances[start : start + args.batch_size]
        batch_rows = run_relaxation_batch(batch, solver_path, args.time_limit, logfile=logfile)
        rows.extend(batch_rows)
        log(f"  progress {min(start + args.batch_size, len(instances))}/{len(instances)}", logfile)

    add_metadata(rows, inst_map)
    write_csv(rows, out_dir / "combined.csv")

    report = build_report(rows)
    (out_dir / "report.md").write_text(report)
    print(report.strip())
    log(f"Saved results to {out_dir}", logfile)
    logfile.close()


if __name__ == "__main__":
    main()

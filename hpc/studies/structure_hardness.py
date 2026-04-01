#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from common import SOLVER, as_float, load_extension_suite, load_instances, log, run_ablation_batch, write_csv


def signature(inst: dict) -> str:
    return "-".join(str(x) for x in sorted(set(inst["jobs"])))


def load_target(target: str, logfile=None):
    if target == "table2":
        return load_instances("2", logfile=logfile)
    if target == "fig9":
        return load_instances("fig9", logfile=logfile)
    if target == "k_boundary":
        return load_extension_suite("k_boundary", logfile=logfile)
    raise ValueError(f"Unknown target: {target}")


def build_report(rows: list[dict], target: str) -> str:
    by_sig = defaultdict(list)
    for row in rows:
        by_sig[row["length_signature"]].append(row)

    lines = ["# Structure Hardness Report", "", f"Target: `{target}`", "", "## By Length Signature"]
    for sig, rs in sorted(by_sig.items()):
        avg_t = sum(as_float(r, "runtime_sec") for r in rs) / len(rs)
        max_t = max(as_float(r, "runtime_sec") for r in rs)
        phase1 = sum(1 for r in rs if r.get("step_reached") == "fwd_relax")
        avg_gap = sum(as_float(r, "max_gap") for r in rs) / len(rs)
        lines.append(
            f"- `{sig}` ({len(rs)} inst): avg runtime {avg_t:.4f}s, max runtime {max_t:.4f}s, fwd_relax winners {phase1}/{len(rs)}, avg max_gap {avg_gap:.2f}"
        )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Characterize hardness by processing-time structure")
    ap.add_argument("--target", choices=["table2", "fig9", "k_boundary"], default="table2")
    ap.add_argument("--max-instances", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--solver", default=str(SOLVER))
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    solver_path = Path(args.solver)
    out_dir = Path(args.output_dir or f"hpc/results_studies/structure_hardness_{args.target}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = open(out_dir / "run.log", "w")

    log("=== Structure Hardness Study ===", logfile)
    log(f"Target: {args.target}", logfile)
    instances = load_target(args.target, logfile=logfile)
    if args.max_instances > 0:
        instances = instances[: args.max_instances]

    inst_map = {iid: inst for _, _, iid, inst in instances}
    rows = []
    for start in range(0, len(instances), args.batch_size):
        batch = instances[start : start + args.batch_size]
        batch_rows = run_ablation_batch(batch, solver_path, "full", "default", args.time_limit, logfile=logfile)
        for row in batch_rows:
            inst = inst_map[row["instance_id"]]
            row["length_signature"] = signature(inst)
            row["distinct_lengths"] = str(len(set(inst["jobs"])))
            row["target"] = args.target
        rows.extend(batch_rows)
        log(f"  progress {min(start + args.batch_size, len(instances))}/{len(instances)}", logfile)

    write_csv(rows, out_dir / "combined.csv")
    report = build_report(rows, args.target)
    (out_dir / "report.md").write_text(report)
    print(report)
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()

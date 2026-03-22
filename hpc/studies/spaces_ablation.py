#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from common import (
    SOLVER,
    as_float,
    as_int,
    load_instances,
    log,
    run_ablation_batch,
    runtime_summary,
    speedup_report,
    step_counter,
    write_csv,
)

CONFIGS = {
    "banded_auto": {
        "label": "Banded SPACES (auto G)",
        "ab_mode": "full",
    },
    "full_spaces": {
        "label": "Full SPACES",
        "ab_mode": "full_spaces",
    },
}


def build_report(all_rows: list[dict], pack_solver: str) -> str:
    by_config = {}
    for name in CONFIGS:
        by_config[name] = [r for r in all_rows if r["config"] == name]

    lines = ["# SPACES Ablation Report", "", f"Pack solver: `{pack_solver}`", "", "## Config Summary"]
    for name, cfg in CONFIGS.items():
        rows = by_config.get(name, [])
        if not rows:
            continue
        avg_t, max_t = runtime_summary(rows)
        n_opt = sum(as_int(r, "is_optimal") for r in rows)
        avg_spaces = sum(as_float(r, "t_spaces") for r in rows) / max(len(rows), 1)
        avg_fwd = sum(as_float(r, "t_fwd_relax") for r in rows) / max(len(rows), 1)
        lines.append(
            f"- `{name}`: {n_opt}/{len(rows)} optimal, avg {avg_t:.4f}s, max {max_t:.4f}s, avg t_spaces {avg_spaces:.4f}s, avg t_fwd_relax {avg_fwd:.4f}s"
        )
        steps = step_counter(rows)
        lines.append("  stage winners: " + ", ".join(f"{k}={v}" for k, v in steps.items()))

    banded = by_config.get("banded_auto", [])
    full = by_config.get("full_spaces", [])
    if banded and full:
        agg, med, mean_su = speedup_report(banded, full)
        banded_map = {r["instance_id"]: r for r in banded}
        full_map = {r["instance_id"]: r for r in full}
        stage_shift = sum(
            1
            for iid in set(banded_map) & set(full_map)
            if banded_map[iid].get("step_reached") != full_map[iid].get("step_reached")
        )
        lines.extend(
            [
                "",
                "## Main Comparison",
                f"- `banded_auto` vs `full_spaces`: aggregate speedup {agg:.2f}x, median {med:.2f}x, mean {mean_su:.2f}x.",
                f"- instances with different winning stage: {stage_shift}",
            ]
        )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Banded-vs-full SPACES ablation")
    ap.add_argument("--section", default="2", choices=["all", "1", "2", "table1", "table2", "fig9"])
    ap.add_argument("--dataset-substr", default="")
    ap.add_argument("--max-instances", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--pack-solver", choices=["default", "constraint", "ortools", "z3"], default="default")
    ap.add_argument("--solver", default=str(SOLVER))
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    solver_path = Path(args.solver)
    out_dir = Path(args.output_dir or "hpc/results_studies/spaces_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = open(out_dir / "run.log", "w")

    log("=== SPACES Ablation ===", logfile)
    log(f"Solver: {solver_path}", logfile)
    log(f"Pack solver: {args.pack_solver}", logfile)
    instances = load_instances(args.section, args.dataset_substr, logfile)
    if args.max_instances > 0:
        instances = instances[: args.max_instances]
    log(f"Instances: {len(instances)}", logfile)

    all_rows = []
    for name, cfg in CONFIGS.items():
        rows = []
        log(f"Running {name}: {cfg['label']}", logfile)
        for start in range(0, len(instances), args.batch_size):
            batch = instances[start : start + args.batch_size]
            batch_rows = run_ablation_batch(
                batch,
                solver_path,
                cfg["ab_mode"],
                args.pack_solver,
                args.time_limit,
                logfile=logfile,
            )
            for row in batch_rows:
                row["config"] = name
                row["config_label"] = cfg["label"]
                row["pack_backend"] = args.pack_solver
            rows.extend(batch_rows)
            log(f"  progress {min(start + args.batch_size, len(instances))}/{len(instances)}", logfile)
        write_csv(rows, out_dir / f"{name}.csv")
        all_rows.extend(rows)

    write_csv(all_rows, out_dir / "combined.csv")
    report = build_report(all_rows, args.pack_solver)
    (out_dir / "report.md").write_text(report)
    print(report)
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()

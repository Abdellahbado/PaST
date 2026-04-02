#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
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
    write_csv,
)


CONFIGS = {
    "exact_only": {
        "label": "Banded SPACES + exact DP only",
        "ab_mode": "exact_only",
    },
    "exact_guided_only": {
        "label": "Semigroup-guided exact DP",
        "ab_mode": "exact_guided_only",
    },
}


def build_report(rows: list[dict]) -> str:
    by_cfg = defaultdict(list)
    for row in rows:
        by_cfg[row["config"]].append(row)

    lines = ["# Exact Guidance Report", "", "## Config Summary"]
    for name, cfg in CONFIGS.items():
        rs = by_cfg.get(name, [])
        if not rs:
            continue
        avg_t, max_t = runtime_summary(rs)
        n_opt = sum(as_int(r, "is_optimal") for r in rs)
        avg_exact = sum(as_float(r, "t_exact") for r in rs) / max(len(rs), 1)
        avg_fwd = sum(as_float(r, "t_fwd_relax") for r in rs) / max(len(rs), 1)
        lines.append(
            f"- `{name}`: {n_opt}/{len(rs)} optimal, avg runtime {avg_t:.4f}s, "
            f"max runtime {max_t:.4f}s, avg t_fwd_relax {avg_fwd:.4f}s, avg t_exact {avg_exact:.4f}s"
        )

    guided = by_cfg.get("exact_guided_only", [])
    exact = by_cfg.get("exact_only", [])
    if guided and exact:
        agg, med, mean_su = speedup_report(exact, guided)
        exact_map = {r["instance_id"]: r for r in exact}
        mismatches = 0
        guided_better = 0
        for row in guided:
            ref = exact_map.get(row["instance_id"])
            if ref is None:
                continue
            if (
                row.get("ub") != ref.get("ub")
                or row.get("lb") != ref.get("lb")
                or row.get("is_optimal") != ref.get("is_optimal")
            ):
                mismatches += 1
            if as_float(row, "runtime_sec") + 1e-9 < as_float(ref, "runtime_sec"):
                guided_better += 1
        lines.extend(["", "## Guided vs Unguided Exact"])
        lines.append(
            f"- `exact_guided_only` vs `exact_only`: aggregate speedup {agg:.2f}x, "
            f"median {med:.2f}x, mean {mean_su:.2f}x"
        )
        lines.append(f"- runtime wins for guided exact on {guided_better}/{len(guided)} instances")
        lines.append(f"- solution mismatches between the two modes: {mismatches}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Compare exact-only vs semigroup-guided exact DP")
    ap.add_argument("--section", default="2", choices=["all", "1", "2", "table1", "table2", "fig9"])
    ap.add_argument("--dataset-substr", default="")
    ap.add_argument("--max-instances", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--solver", default=str(SOLVER))
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    solver_path = Path(args.solver)
    out_dir = Path(args.output_dir or "hpc/results_studies/exact_guidance")
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = open(out_dir / "run.log", "w")

    log("=== Exact Guidance Study ===", logfile)
    log(f"Solver: {solver_path}", logfile)
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
                "default",
                args.time_limit,
                logfile=logfile,
            )
            for row in batch_rows:
                row["config"] = name
                row["config_label"] = cfg["label"]
            rows.extend(batch_rows)
            log(f"  progress {min(start + args.batch_size, len(instances))}/{len(instances)}", logfile)
        write_csv(rows, out_dir / f"{name}.csv")
        all_rows.extend(rows)

    write_csv(all_rows, out_dir / "combined.csv")
    report = build_report(all_rows)
    (out_dir / "report.md").write_text(report)
    print(report)
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()

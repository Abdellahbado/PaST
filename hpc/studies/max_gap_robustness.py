#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from common import (
    SOLVER,
    as_float,
    as_int,
    load_extension_suite,
    load_instances,
    log,
    run_ablation_batch,
    runtime_summary,
    speedup_report,
    write_csv,
)


CONFIGS = [
    {
        "name": "gap_auto",
        "label": "Automatic sharpened max-gap",
        "ab_mode": "full",
        "extra_env": {"PAST_MAX_GAP_OVERRIDE": "auto"},
    },
    {
        "name": "gap_1p25x",
        "label": "1.25x automatic max-gap",
        "ab_mode": "full",
        "extra_env": {"PAST_MAX_GAP_SCALE": "1.25"},
    },
    {
        "name": "gap_2p00x",
        "label": "2.00x automatic max-gap",
        "ab_mode": "full",
        "extra_env": {"PAST_MAX_GAP_SCALE": "2.00"},
    },
    {
        "name": "gap_full",
        "label": "Full SPACES",
        "ab_mode": "full_spaces",
        "extra_env": {"PAST_MAX_GAP_OVERRIDE": "full"},
    },
]


def load_target(target: str, logfile=None):
    if target == "table2":
        return load_instances("2", logfile=logfile)
    if target == "fig9":
        return load_instances("fig9", logfile=logfile)
    if target == "scalability":
        return load_extension_suite("scalability_large_n", logfile=logfile)
    raise ValueError(f"Unknown target: {target}")


def build_report(rows: list[dict], target: str) -> str:
    by_cfg = defaultdict(list)
    for row in rows:
        by_cfg[row["config"]].append(row)

    lines = ["# Max-Gap Robustness Report", "", f"Target: `{target}`", "", "## Config Summary"]
    for cfg in CONFIGS:
        rs = by_cfg.get(cfg["name"], [])
        if not rs:
            continue
        avg_t, max_t = runtime_summary(rs)
        avg_gap = sum(as_float(r, "max_gap") for r in rs) / max(len(rs), 1)
        n_opt = sum(as_int(r, "is_optimal") for r in rs)
        lines.append(
            f"- `{cfg['name']}`: {n_opt}/{len(rs)} optimal, avg {avg_t:.4f}s, max {max_t:.4f}s, avg max_gap {avg_gap:.2f}"
        )

    full_rows = by_cfg.get("gap_full", [])
    if full_rows:
        full_map = {r["instance_id"]: r for r in full_rows}
        lines.extend(["", "## Against Full SPACES"])
        for cfg in CONFIGS:
            if cfg["name"] == "gap_full":
                continue
            rs = by_cfg.get(cfg["name"], [])
            if not rs:
                continue
            agg, med, mean_su = speedup_report(rs, full_rows)
            mismatches = 0
            for row in rs:
                ref = full_map.get(row["instance_id"])
                if ref is None:
                    continue
                if (
                    row.get("ub") != ref.get("ub")
                    or row.get("lb") != ref.get("lb")
                    or row.get("is_optimal") != ref.get("is_optimal")
                ):
                    mismatches += 1
            lines.append(
                f"- `{cfg['name']}`: aggregate speedup {agg:.2f}x, median {med:.2f}x, mean {mean_su:.2f}x, mismatches vs full={mismatches}"
            )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Study robustness of the sharpened max-gap rule")
    ap.add_argument("--target", choices=["table2", "fig9", "scalability"], default="table2")
    ap.add_argument("--max-instances", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--solver", default=str(SOLVER))
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    solver_path = Path(args.solver)
    out_dir = Path(args.output_dir or f"hpc/results_studies/max_gap_{args.target}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = open(out_dir / "run.log", "w")

    log("=== Max-Gap Robustness Study ===", logfile)
    log(f"Target: {args.target}", logfile)
    instances = load_target(args.target, logfile=logfile)
    if args.max_instances > 0:
        instances = instances[: args.max_instances]
    log(f"Instances: {len(instances)}", logfile)

    all_rows = []
    for cfg in CONFIGS:
        rows = []
        log(f"Running {cfg['name']}: {cfg['label']}", logfile)
        for start in range(0, len(instances), args.batch_size):
            batch = instances[start : start + args.batch_size]
            batch_rows = run_ablation_batch(
                batch,
                solver_path,
                cfg["ab_mode"],
                "default",
                args.time_limit,
                extra_env=cfg.get("extra_env"),
                logfile=logfile,
            )
            for row in batch_rows:
                row["config"] = cfg["name"]
                row["config_label"] = cfg["label"]
                row["target"] = args.target
            rows.extend(batch_rows)
            log(f"  progress {min(start + args.batch_size, len(instances))}/{len(instances)}", logfile)
        write_csv(rows, out_dir / f"{cfg['name']}.csv")
        all_rows.extend(rows)

    write_csv(all_rows, out_dir / "combined.csv")
    report = build_report(all_rows, args.target)
    (out_dir / "report.md").write_text(report)
    print(report)
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()

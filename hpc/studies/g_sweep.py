#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def make_configs(include_unsafe: bool) -> list[dict]:
    configs = [
        {
            "name": "g_auto",
            "label": "G = auto",
            "ab_mode": "full",
            "extra_env": {"PAST_MAX_GAP_OVERRIDE": "auto"},
        },
        {
            "name": "g_1p25x",
            "label": "G = 1.25 * auto",
            "ab_mode": "full",
            "extra_env": {"PAST_MAX_GAP_SCALE": "1.25"},
        },
        {
            "name": "g_1p50x",
            "label": "G = 1.50 * auto",
            "ab_mode": "full",
            "extra_env": {"PAST_MAX_GAP_SCALE": "1.50"},
        },
        {
            "name": "g_2p00x",
            "label": "G = 2.00 * auto",
            "ab_mode": "full",
            "extra_env": {"PAST_MAX_GAP_SCALE": "2.00"},
        },
        {
            "name": "g_full",
            "label": "G = full",
            "ab_mode": "full_spaces",
            "extra_env": {"PAST_MAX_GAP_OVERRIDE": "full"},
        },
    ]
    if include_unsafe:
        configs = [
            {
                "name": "g_0p50x",
                "label": "G = 0.50 * auto (unsafe)",
                "ab_mode": "full",
                "extra_env": {"PAST_MAX_GAP_SCALE": "0.50"},
            },
            {
                "name": "g_0p75x",
                "label": "G = 0.75 * auto (unsafe)",
                "ab_mode": "full",
                "extra_env": {"PAST_MAX_GAP_SCALE": "0.75"},
            },
        ] + configs
    return configs


def build_report(all_rows: list[dict], configs: list[dict], pack_solver: str) -> str:
    by_config = {cfg["name"]: [r for r in all_rows if r["config"] == cfg["name"]] for cfg in configs}
    full_rows = by_config.get("g_full", [])

    lines = ["# G Sweep Report", "", f"Pack solver: `{pack_solver}`", "", "## Config Summary"]
    for cfg in configs:
        rows = by_config.get(cfg["name"], [])
        if not rows:
            continue
        avg_t, max_t = runtime_summary(rows)
        n_opt = sum(as_int(r, "is_optimal") for r in rows)
        avg_g = sum(as_float(r, "max_gap") for r in rows) / max(len(rows), 1)
        lines.append(
            f"- `{cfg['name']}`: {n_opt}/{len(rows)} optimal, avg {avg_t:.4f}s, max {max_t:.4f}s, avg reported G {avg_g:.2f}"
        )

    if full_rows:
        full_map = {r["instance_id"]: r for r in full_rows}
        lines.extend(["", "## Against Full SPACES"])
        for cfg in configs:
            if cfg["name"] == "g_full":
                continue
            rows = by_config.get(cfg["name"], [])
            if not rows:
                continue
            agg, med, mean_su = speedup_report(rows, full_rows)
            mismatches = 0
            for row in rows:
                ref = full_map.get(row["instance_id"])
                if ref is None:
                    continue
                same = (
                    row.get("ub") == ref.get("ub")
                    and row.get("lb") == ref.get("lb")
                    and row.get("is_optimal") == ref.get("is_optimal")
                )
                if not same:
                    mismatches += 1
            lines.append(
                f"- `{cfg['name']}`: aggregate speedup {agg:.2f}x, median {med:.2f}x, mean {mean_su:.2f}x, mismatches vs full={mismatches}"
            )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Sweep the SPACES G parameter")
    ap.add_argument("--section", default="2", choices=["all", "1", "2", "table1", "table2", "fig9"])
    ap.add_argument("--dataset-substr", default="")
    ap.add_argument("--max-instances", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--pack-solver", choices=["default", "constraint", "ortools", "z3"], default="default")
    ap.add_argument("--include-unsafe", action="store_true")
    ap.add_argument("--solver", default=str(SOLVER))
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    solver_path = Path(args.solver)
    out_dir = Path(args.output_dir or "hpc/results_studies/g_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = open(out_dir / "run.log", "w")

    log("=== G Sweep ===", logfile)
    log(f"Solver: {solver_path}", logfile)
    log(f"Pack solver: {args.pack_solver}", logfile)
    instances = load_instances(args.section, args.dataset_substr, logfile)
    if args.max_instances > 0:
        instances = instances[: args.max_instances]
    log(f"Instances: {len(instances)}", logfile)

    configs = make_configs(args.include_unsafe)
    all_rows = []
    for cfg in configs:
        rows = []
        log(f"Running {cfg['name']}: {cfg['label']}", logfile)
        for start in range(0, len(instances), args.batch_size):
            batch = instances[start : start + args.batch_size]
            batch_rows = run_ablation_batch(
                batch,
                solver_path,
                cfg["ab_mode"],
                args.pack_solver,
                args.time_limit,
                extra_env=cfg["extra_env"],
                logfile=logfile,
            )
            for row in batch_rows:
                row["config"] = cfg["name"]
                row["config_label"] = cfg["label"]
                row["pack_backend"] = args.pack_solver
            rows.extend(batch_rows)
            log(f"  progress {min(start + args.batch_size, len(instances))}/{len(instances)}", logfile)
        write_csv(rows, out_dir / f"{cfg['name']}.csv")
        all_rows.extend(rows)

    write_csv(all_rows, out_dir / "combined.csv")
    report = build_report(all_rows, configs, args.pack_solver)
    (out_dir / "report.md").write_text(report)
    print(report)
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()

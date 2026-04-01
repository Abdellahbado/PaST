#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from common import (
    SOLVER,
    as_float,
    as_int,
    load_extension_suite,
    log,
    run_ablation_batch,
    runtime_summary,
    write_csv,
)


CONFIGS = {
    "step1_only": {
        "label": "Front path only",
        "ab_mode": "step1_only",
    },
    "full_default": {
        "label": "Full pipeline",
        "ab_mode": "full",
    },
    "no_smart_recon": {
        "label": "Full pipeline without smart reconstruction",
        "ab_mode": "no_smart_recon",
    },
}


def build_report(rows: list[dict], suite_name: str) -> str:
    by_cfg = defaultdict(list)
    for row in rows:
        by_cfg[row["config"]].append(row)

    lines = ["# Certification Contribution Report", "", f"Suite: `{suite_name}`", "", "## Config Summary"]
    for name, cfg in CONFIGS.items():
        rs = by_cfg.get(name, [])
        if not rs:
            continue
        avg_t, max_t = runtime_summary(rs)
        n_opt = sum(as_int(r, "is_optimal") for r in rs)
        lines.append(f"- `{name}`: {n_opt}/{len(rs)} optimal, avg {avg_t:.4f}s, max {max_t:.4f}s")
        steps = Counter(r.get("step_reached", "unknown") for r in rs)
        lines.append("  stage winners: " + ", ".join(f"{k}={v}" for k, v in sorted(steps.items())))
        sub = Counter(r.get("fwd_pack_method", "none") for r in rs if r.get("step_reached") == "fwd_relax")
        if sub:
            lines.append("  front-path certifier/method counts: " + ", ".join(f"{k}={v}" for k, v in sorted(sub.items())))

    full_rows = by_cfg.get("full_default", [])
    if full_rows:
        full_map = {r["instance_id"]: r for r in full_rows}
        lines.extend(["", "## Practical Interpretation"])
        for name in ("step1_only", "no_smart_recon"):
            rs = by_cfg.get(name, [])
            if not rs:
                continue
            same = 0
            weaker = 0
            for row in rs:
                ref = full_map.get(row["instance_id"])
                if not ref:
                    continue
                if row.get("is_optimal") == ref.get("is_optimal") and row.get("ub") == ref.get("ub"):
                    same += 1
                else:
                    weaker += 1
            lines.append(f"- `{name}` matches the full pipeline on {same}/{len(rs)} rows and is weaker on {weaker}/{len(rs)} rows.")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Study the contribution of the current certification path")
    ap.add_argument("--suite", choices=["scalability_large_n", "k_boundary"], default="scalability_large_n")
    ap.add_argument("--max-instances", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--solver", default=str(SOLVER))
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    solver_path = Path(args.solver)
    out_dir = Path(args.output_dir or f"hpc/results_studies/certification_{args.suite}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = open(out_dir / "run.log", "w")

    log("=== Certification Contribution Study ===", logfile)
    log(f"Suite: {args.suite}", logfile)
    instances = load_extension_suite(args.suite, logfile=logfile)
    if args.max_instances > 0:
        instances = instances[: args.max_instances]
    log(f"Instances: {len(instances)}", logfile)

    if not instances:
        report = (
            "# Certification Contribution Report\n\n"
            f"Suite: `{args.suite}`\n\n"
            "## Config Summary\n\n"
            "No instances were loaded. The extension dataset is missing or empty.\n"
        )
        (out_dir / "report.md").write_text(report)
        print(report)
        print(f"Saved results to {out_dir}")
        return

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
                row["suite"] = args.suite
            rows.extend(batch_rows)
            log(f"  progress {min(start + args.batch_size, len(instances))}/{len(instances)}", logfile)
        write_csv(rows, out_dir / f"{name}.csv")
        all_rows.extend(rows)

    write_csv(all_rows, out_dir / "combined.csv")
    report = build_report(all_rows, args.suite)
    (out_dir / "report.md").write_text(report)
    print(report)
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()

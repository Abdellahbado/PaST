#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
    "full_default": {
        "label": "Full pipeline (default Step 1)",
        "ab_mode": "full",
        "pack_solver": "default",
    },
    "full_exact_pack": {
        "label": "Full pipeline + exact relaxed-block packing",
        "ab_mode": "full",
        "pack_solver": "constraint",
    },
    "no_smart_recon": {
        "label": "Exact pack, no smart reconstruction",
        "ab_mode": "no_smart_recon",
        "pack_solver": "constraint",
    },
    "exact_only": {
        "label": "Exact-only baseline",
        "ab_mode": "exact_only",
        "pack_solver": "default",
    },
}

STEP_ORDER = [
    "fwd_relax",
    "heuristic_ub",
    "local_search",
    "r_feas",
    "r_feas_lagr",
    "smart_recon",
    "exact",
]


def select_exact_only_instances(instances, prior_rows):
    if not prior_rows:
        return instances
    interesting = {
        row["instance_id"]
        for row in prior_rows
        if row.get("config") != "exact_only" and row.get("step_reached") != "fwd_relax"
    }
    if not interesting:
        return []
    return [inst for inst in instances if inst[2] in interesting]


def build_report(all_rows: list[dict]) -> str:
    by_config = defaultdict(list)
    for row in all_rows:
        by_config[row["config"]].append(row)

    lines = ["# Component Ablation Report", "", "## Config Summary"]
    for name in CONFIGS:
        rows = by_config.get(name, [])
        if not rows:
            continue
        avg_t, max_t = runtime_summary(rows)
        n_opt = sum(as_int(r, "is_optimal") for r in rows)
        lines.append(f"- `{name}`: {n_opt}/{len(rows)} optimal, avg {avg_t:.4f}s, max {max_t:.4f}s")
        steps = step_counter(rows)
        lines.append(
            "  stage winners: "
            + ", ".join(f"{step}={steps[step]}" for step in STEP_ORDER if steps.get(step, 0))
        )
        step1 = Counter(
            r.get("fwd_pack_method", "none")
            for r in rows
            if r.get("step_reached") == "fwd_relax"
        )
        if step1:
            lines.append("  Step 1 submethods: " + ", ".join(f"{k}={v}" for k, v in sorted(step1.items())))

    lines.extend(["", "## Main Comparisons"])
    full_default = by_config.get("full_default", [])
    full_exact = by_config.get("full_exact_pack", [])
    no_sr = by_config.get("no_smart_recon", [])
    exact_only = by_config.get("exact_only", [])

    if full_default and full_exact:
        agg, med, mean_su = speedup_report(full_exact, full_default)
        default_map = {r["instance_id"]: r for r in full_default}
        exact_map = {r["instance_id"]: r for r in full_exact}
        new_step1 = [
            iid for iid in sorted(set(default_map) & set(exact_map))
            if default_map[iid].get("step_reached") != "fwd_relax"
            and exact_map[iid].get("step_reached") == "fwd_relax"
        ]
        avoided_exact = [
            iid for iid in sorted(set(default_map) & set(exact_map))
            if default_map[iid].get("step_reached") == "exact"
            and exact_map[iid].get("step_reached") != "exact"
        ]
        lines.append(
            f"- `full_exact_pack` vs `full_default`: aggregate speedup {agg:.2f}x, median {med:.2f}x, mean {mean_su:.2f}x."
        )
        lines.append(
            f"  exact relaxed-block packing newly closes Step 1 on {len(new_step1)} instances and avoids exact DP on {len(avoided_exact)} instances."
        )

    if full_exact and no_sr:
        agg, med, mean_su = speedup_report(full_exact, no_sr)
        exact_map = {r["instance_id"]: r for r in full_exact}
        no_sr_map = {r["instance_id"]: r for r in no_sr}
        smart_only = [
            iid for iid in sorted(set(exact_map) & set(no_sr_map))
            if exact_map[iid].get("step_reached") == "smart_recon"
        ]
        sr_avoids_exact = [
            iid for iid in sorted(set(exact_map) & set(no_sr_map))
            if no_sr_map[iid].get("step_reached") == "exact"
            and exact_map[iid].get("step_reached") != "exact"
        ]
        lines.append(
            f"- `full_exact_pack` vs `no_smart_recon`: aggregate speedup {agg:.2f}x, median {med:.2f}x, mean {mean_su:.2f}x."
        )
        lines.append(
            f"  smart reconstruction is the winner on {len(smart_only)} instances and avoids exact DP on {len(sr_avoids_exact)} instances."
        )

    if full_exact and exact_only:
        agg, med, mean_su = speedup_report(full_exact, exact_only)
        lines.append(
            f"- `full_exact_pack` vs `exact_only`: aggregate speedup {agg:.2f}x, median {med:.2f}x, mean {mean_su:.2f}x."
        )

    lines.extend(["", "## Expected Runtime by Winning Stage"])
    for name, rows in by_config.items():
        lines.append(f"- `{name}`")
        for step in STEP_ORDER:
            vals = [as_float(r, "runtime_sec") for r in rows if r.get("step_reached") == step]
            if vals:
                vals.sort()
                lines.append(
                    f"  {step}: count={len(vals)}, avg={sum(vals)/len(vals):.4f}s, median={vals[len(vals)//2]:.4f}s, max={max(vals):.4f}s"
                )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Focused component ablation")
    ap.add_argument("--section", default="2", choices=["all", "1", "2", "table1", "table2", "fig9"])
    ap.add_argument("--dataset-substr", default="")
    ap.add_argument("--max-instances", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--exact-only-policy", choices=["survivors", "all"], default="survivors")
    ap.add_argument("--config", default="all", choices=["all"] + list(CONFIGS))
    ap.add_argument("--solver", default=str(SOLVER))
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    solver_path = Path(args.solver)
    out_dir = Path(args.output_dir or (Path("hpc/results_studies/component_ablation")))
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = open(out_dir / "run.log", "w")

    log("=== Component Ablation ===", logfile)
    log(f"Solver: {solver_path}", logfile)
    instances = load_instances(args.section, args.dataset_substr, logfile)
    if args.max_instances > 0:
        instances = instances[: args.max_instances]
    log(f"Instances: {len(instances)}", logfile)

    configs = list(CONFIGS) if args.config == "all" else [args.config]
    all_rows = []
    for name in configs:
        cfg = CONFIGS[name]
        config_instances = instances
        if name == "exact_only" and args.exact_only_policy == "survivors":
            config_instances = select_exact_only_instances(instances, all_rows)
            log(f"exact_only survivor subset: {len(config_instances)}/{len(instances)}", logfile)
            if not config_instances:
                continue

        rows = []
        log(f"Running {name}: {cfg['label']}", logfile)
        for start in range(0, len(config_instances), args.batch_size):
            batch = config_instances[start : start + args.batch_size]
            batch_rows = run_ablation_batch(
                batch,
                solver_path,
                cfg["ab_mode"],
                cfg["pack_solver"],
                args.time_limit,
                logfile=logfile,
            )
            for row in batch_rows:
                row["config"] = name
                row["config_label"] = cfg["label"]
                row["pack_backend"] = cfg["pack_solver"]
                row["ablation_mode"] = cfg["ab_mode"]
                row["instance_scope"] = (
                    "survivors"
                    if name == "exact_only" and args.exact_only_policy == "survivors"
                    else "all"
                )
            rows.extend(batch_rows)
            log(f"  progress {min(start + args.batch_size, len(config_instances))}/{len(config_instances)}", logfile)
        write_csv(rows, out_dir / f"{name}.csv")
        all_rows.extend(rows)

    write_csv(all_rows, out_dir / "combined.csv")
    report = build_report(all_rows)
    (out_dir / "report.md").write_text(report)
    print(report)
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()

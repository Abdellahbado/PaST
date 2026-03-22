#!/usr/bin/env python3
"""
08_component_ablation.py — Focused component ablation for PaST.

This script is intentionally narrower than 06_ablation_study.py. It targets the
parts that are both insightful and paper-relevant:

  1. Our full pipeline with the default Step 1 packer.
  2. Our full pipeline with an exact relaxed-block packer.
  3. The same exact-pack pipeline without smart reconstruction.
  4. An exact-only baseline.

The solver output includes detailed Step 1 attribution, so we can see:
  - which stage closed the instance,
  - which Step 1 submethod packed the relaxed blocks,
  - how much time each stage/substage took.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent.parent
SOLVER = ROOT / "solvers" / "cpp" / "build" / "stateful_compare"
PAPER_DATASETS = (
    ROOT
    / "data"
    / "green-scheduling-bab"
    / "Iirc.EnergyStatesAndCostsScheduling"
    / "data"
    / "datasets"
)

TABLE1_DATASETS = [
    "benedikt2025a_large_nosby",
    "benedikt2025a_large_twosby",
    "benedikt2025a_medium_nosby",
    "benedikt2025a_medium_twosby",
    "benedikt2025a_prelim",
    "aghelinejad2017a_1",
]
TABLE2_DATASETS = ["benedikt2025b_groups"]
FIG9_DATASETS = ["benedikt2025_groups"]

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


def log(msg: str, logfile=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if logfile:
        logfile.write(line + "\n")
        logfile.flush()


def load_paper_instance(json_path: Path) -> dict:
    with open(json_path) as f:
        d = json.load(f)
    jobs = [j["ProcessingTime"] for j in d["Jobs"]]
    prices = d["EnergyCosts"]
    noff = len(d.get("OffOnTime", []))
    mtype = "twosby" if noff == 3 else "nosby"
    return {"jobs": jobs, "prices": prices, "machine_type": mtype}


def load_instances(section: str, logfile=None) -> list[tuple[str, str, str, dict]]:
    wanted = []
    if section in ("all", "1", "table1"):
        for ds in TABLE1_DATASETS:
            wanted.append(("table1", ds))
    if section in ("all", "2", "table2"):
        for ds in TABLE2_DATASETS:
            wanted.append(("table2", ds))
    if section in ("all", "fig9"):
        for ds in FIG9_DATASETS:
            wanted.append(("fig9", ds))

    instances = []
    for sec, ds_name in wanted:
        ds_dir = PAPER_DATASETS / ds_name
        if not ds_dir.exists():
            log(f"  WARNING: dataset not found: {ds_dir}", logfile)
            continue
        json_files = sorted(ds_dir.glob("*.json"), key=lambda p: int(p.stem))
        for jf in json_files:
            idx = int(jf.stem)
            inst = load_paper_instance(jf)
            inst_id = f"{ds_name}/{idx}"
            instances.append((sec, ds_name, inst_id, inst))
        log(f"  Loaded {len(json_files)} instances from {ds_name}", logfile)
    return instances


def as_float(row: dict, key: str) -> float:
    try:
        return float(row.get(key, 0) or 0)
    except ValueError:
        return 0.0


def as_int(row: dict, key: str) -> int:
    try:
        return int(float(row.get(key, 0) or 0))
    except ValueError:
        return 0


def solve_batch(
    batch: list[tuple[str, str, str, dict]],
    solver_path: Path,
    ab_mode: str,
    pack_solver: str,
    time_limit: float,
    logfile=None,
) -> list[dict]:
    lines = []
    meta = {}
    for section, dataset, inst_id, inst in batch:
        meta[inst_id] = (section, dataset)
        payload = {
            "instance_id": inst_id,
            "prices": inst["prices"],
            "jobs": inst["jobs"],
            "machine": inst["machine_type"],
        }
        lines.append(json.dumps(payload))

    proc = subprocess.run(
        [str(solver_path), "ablation-stdin", ab_mode],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        timeout=max(int(time_limit * len(batch) + 300), 7200),
        env={
            **os.environ,
            **(
                {"PAST_RELAXED_BINPACK_SOLVER": pack_solver}
                if pack_solver != "default"
                else {}
            ),
        },
    )
    if proc.returncode != 0 and proc.stderr:
        log(f"  Solver stderr ({ab_mode}/{pack_solver}): {proc.stderr[:300]}", logfile)

    rows = []
    header = None
    for line in proc.stdout.strip().splitlines():
        if not line.strip():
            continue
        if header is None:
            header = line.split(",")
            continue
        parts = line.split(",")
        if len(parts) < len(header):
            continue
        row = dict(zip(header, parts))
        section, dataset = meta.get(row["instance_id"], ("", ""))
        row["section"] = section
        row["dataset"] = dataset
        rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_config_summary(name: str, rows: list[dict], logfile=None):
    if not rows:
        return
    total = len(rows)
    n_opt = sum(as_int(r, "is_optimal") for r in rows)
    avg_t = mean(as_float(r, "runtime_sec") for r in rows)
    max_t = max(as_float(r, "runtime_sec") for r in rows)
    log(
        f"  [{name:>16}] {n_opt:>4}/{total:<4} optimal  avg={avg_t:>8.4f}s  max={max_t:>8.3f}s",
        logfile,
    )

    steps = Counter(r.get("step_reached", "unknown") for r in rows)
    log(
        "    winners: "
        + ", ".join(f"{step}={steps.get(step, 0)}" for step in STEP_ORDER if steps.get(step, 0)),
        logfile,
    )
    step1 = Counter(
        r.get("fwd_pack_method", "none")
        for r in rows
        if r.get("step_reached") == "fwd_relax"
    )
    if step1:
        log(
            "    Step 1 packers: "
            + ", ".join(f"{k}={v}" for k, v in sorted(step1.items())),
            logfile,
        )


def speedup_report(fast_rows: list[dict], slow_rows: list[dict]) -> tuple[float, float, float]:
    fast = {r["instance_id"]: as_float(r, "runtime_sec") for r in fast_rows}
    slow = {r["instance_id"]: as_float(r, "runtime_sec") for r in slow_rows}
    common = sorted(set(fast) & set(slow))
    if not common:
        return 0.0, 0.0, 0.0
    ratios = [slow[iid] / max(fast[iid], 1e-9) for iid in common]
    agg = sum(slow[iid] for iid in common) / max(sum(fast[iid] for iid in common), 1e-9)
    return agg, median(ratios), mean(ratios)


def select_exact_only_instances(
    instances: list[tuple[str, str, str, dict]],
    prior_rows: list[dict],
) -> list[tuple[str, str, str, dict]]:
    """Keep only instances that survive beyond Step 1 in any prior non-exact config."""
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
    by_config: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        by_config[row["config"]].append(row)

    lines = []
    lines.append("# Component Ablation Report")
    lines.append("")
    lines.append("## Config Summary")
    for name, cfg in CONFIGS.items():
        rows = by_config.get(name, [])
        if not rows:
            continue
        avg_t = mean(as_float(r, "runtime_sec") for r in rows)
        max_t = max(as_float(r, "runtime_sec") for r in rows)
        n_opt = sum(as_int(r, "is_optimal") for r in rows)
        lines.append(
            f"- `{name}`: {n_opt}/{len(rows)} optimal, avg {avg_t:.4f}s, max {max_t:.4f}s"
        )
        steps = Counter(r.get("step_reached", "unknown") for r in rows)
        step_str = ", ".join(
            f"{step}={steps[step]}" for step in STEP_ORDER if steps.get(step, 0)
        )
        lines.append(f"  stage winners: {step_str}")
        step1 = Counter(
            r.get("fwd_pack_method", "none")
            for r in rows
            if r.get("step_reached") == "fwd_relax"
        )
        if step1:
            lines.append(
                "  Step 1 submethods: "
                + ", ".join(f"{k}={v}" for k, v in sorted(step1.items()))
            )

    lines.append("")
    lines.append("## Main Comparisons")
    full_default = by_config.get("full_default", [])
    full_exact = by_config.get("full_exact_pack", [])
    no_sr = by_config.get("no_smart_recon", [])
    exact_only = by_config.get("exact_only", [])

    if full_default and full_exact:
        agg, med, mean_su = speedup_report(full_exact, full_default)
        default_map = {r["instance_id"]: r for r in full_default}
        exact_map = {r["instance_id"]: r for r in full_exact}
        new_step1 = [
            iid
            for iid in sorted(set(default_map) & set(exact_map))
            if default_map[iid].get("step_reached") != "fwd_relax"
            and exact_map[iid].get("step_reached") == "fwd_relax"
        ]
        avoided_exact = [
            iid
            for iid in sorted(set(default_map) & set(exact_map))
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
            iid
            for iid in sorted(set(exact_map) & set(no_sr_map))
            if exact_map[iid].get("step_reached") == "smart_recon"
        ]
        sr_avoids_exact = [
            iid
            for iid in sorted(set(exact_map) & set(no_sr_map))
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

    lines.append("")
    lines.append("## Expected Runtime by Winning Stage")
    for name, rows in by_config.items():
        lines.append(f"- `{name}`")
        for step in STEP_ORDER:
            vals = [as_float(r, "runtime_sec") for r in rows if r.get("step_reached") == step]
            if vals:
                lines.append(
                    f"  {step}: count={len(vals)}, avg={mean(vals):.4f}s, median={median(vals):.4f}s, max={max(vals):.4f}s"
                )

    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Focused PaST component ablation")
    ap.add_argument("--section", default="2", choices=["all", "1", "2", "table1", "table2", "fig9"])
    ap.add_argument(
        "--config",
        default="all",
        choices=["all"] + list(CONFIGS),
        help="Which configuration to run (default: all)",
    )
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--max-instances", type=int, default=0)
    ap.add_argument(
        "--dataset-substr",
        default="",
        help="Optional substring filter on dataset name for targeted runs.",
    )
    ap.add_argument(
        "--exact-only-policy",
        choices=["survivors", "all"],
        default="survivors",
        help="Run exact_only on all instances or only the nontrivial survivor set.",
    )
    ap.add_argument("--solver", default=str(SOLVER))
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "hpc" / "results_component_ablation"),
    )
    args = ap.parse_args()

    solver_path = Path(args.solver)
    if not solver_path.exists():
        print(f"ERROR: solver not found: {solver_path}")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = open(out_dir / "component_ablation_log.txt", "w")

    log("=" * 90, logfile)
    log("  COMPONENT ABLATION — exact relaxed-binpacking + smart reconstruction", logfile)
    log(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}", logfile)
    log(f"  Solver: {solver_path}", logfile)
    log(f"  Section: {args.section}", logfile)
    log("=" * 90, logfile)

    instances = load_instances(args.section, logfile)
    if args.dataset_substr:
        instances = [inst for inst in instances if args.dataset_substr in inst[1]]
    if args.max_instances > 0:
        instances = instances[: args.max_instances]
    log(f"  Total instances: {len(instances)}", logfile)
    configs = list(CONFIGS) if args.config == "all" else [args.config]
    log(f"  Configs: {configs}", logfile)

    all_rows: list[dict] = []
    for name in configs:
        cfg = CONFIGS[name]
        config_instances = instances
        if name == "exact_only" and args.exact_only_policy == "survivors":
            config_instances = select_exact_only_instances(instances, all_rows)
            log(
                f"  exact_only survivor subset: {len(config_instances)}/{len(instances)} instances",
                logfile,
            )
            if not config_instances:
                log("  exact_only skipped: no survivor instances remained after Step 1", logfile)
                continue
        log("", logfile)
        log(f"### Running {name}: {cfg['label']}", logfile)
        rows = []
        for start in range(0, len(config_instances), args.batch_size):
            batch = config_instances[start : start + args.batch_size]
            batch_rows = solve_batch(
                batch,
                solver_path,
                cfg["ab_mode"],
                cfg["pack_solver"],
                args.time_limit,
                logfile,
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
            log(
                f"    progress {min(start + args.batch_size, len(config_instances))}/{len(config_instances)}",
                logfile,
            )
        print_config_summary(name, rows, logfile)
        write_csv(rows, out_dir / f"{name}.csv")
        all_rows.extend(rows)

    write_csv(all_rows, out_dir / "component_ablation_combined.csv")
    report = build_report(all_rows)
    report_path = out_dir / "component_ablation_report.md"
    report_path.write_text(report)
    log("", logfile)
    log(f"  Combined CSV: {out_dir / 'component_ablation_combined.csv'}", logfile)
    log(f"  Report: {report_path}", logfile)
    logfile.close()

    print(report)
    print(f"Saved combined CSV to {out_dir / 'component_ablation_combined.csv'}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()

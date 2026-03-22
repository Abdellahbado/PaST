#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[2]
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


def load_instances(section: str, dataset_substr: str = "", logfile=None):
    wanted = []
    if section in ("all", "1", "table1"):
        wanted.extend(("table1", ds) for ds in TABLE1_DATASETS)
    if section in ("all", "2", "table2"):
        wanted.extend(("table2", ds) for ds in TABLE2_DATASETS)
    if section in ("all", "fig9"):
        wanted.extend(("fig9", ds) for ds in FIG9_DATASETS)

    out = []
    for sec, ds_name in wanted:
        if dataset_substr and dataset_substr not in ds_name:
            continue
        ds_dir = PAPER_DATASETS / ds_name
        if not ds_dir.exists():
            log(f"  WARNING: dataset not found: {ds_dir}", logfile)
            continue
        json_files = sorted(ds_dir.glob("*.json"), key=lambda p: int(p.stem))
        for jf in json_files:
            idx = int(jf.stem)
            inst = load_paper_instance(jf)
            inst_id = f"{ds_name}/{idx}"
            out.append((sec, ds_name, inst_id, inst))
        log(f"  Loaded {len(json_files)} instances from {ds_name}", logfile)
    return out


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


def write_csv(rows: list[dict], path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_ablation_batch(
    batch: list[tuple[str, str, str, dict]],
    solver_path: Path,
    ab_mode: str,
    pack_solver: str,
    time_limit: float,
    extra_env: dict[str, str] | None = None,
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

    env = os.environ.copy()
    if pack_solver != "default":
        env["PAST_RELAXED_BINPACK_SOLVER"] = pack_solver
    if extra_env:
        env.update(extra_env)

    proc = subprocess.run(
        [str(solver_path), "ablation-stdin", ab_mode],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        timeout=max(int(time_limit * len(batch) + 300), 7200),
        env=env,
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


def speedup_report(fast_rows: list[dict], slow_rows: list[dict]) -> tuple[float, float, float]:
    fast = {r["instance_id"]: as_float(r, "runtime_sec") for r in fast_rows}
    slow = {r["instance_id"]: as_float(r, "runtime_sec") for r in slow_rows}
    common = sorted(set(fast) & set(slow))
    if not common:
        return 0.0, 0.0, 0.0
    ratios = [slow[iid] / max(fast[iid], 1e-9) for iid in common]
    agg = sum(slow[iid] for iid in common) / max(sum(fast[iid] for iid in common), 1e-9)
    return agg, median(ratios), mean(ratios)


def step_counter(rows: list[dict]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for row in rows:
        out[row.get("step_reached", "unknown")] += 1
    return dict(out)


def runtime_summary(rows: list[dict]) -> tuple[float, float]:
    vals = [as_float(r, "runtime_sec") for r in rows]
    if not vals:
        return 0.0, 0.0
    return mean(vals), max(vals)

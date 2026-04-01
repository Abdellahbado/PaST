#!/usr/bin/env python3
"""
Run the K-boundary benchmark and merge:

1. full default-solver behavior (`ablation-stdin full`)
2. relaxation hierarchy (`relax-hierarchy-stdin`)

The goal is to understand whether increasing the number of distinct job sizes
creates a true boundary for:

- the Phase-1 relaxed DP + certification path,
- the backup lower bounds,
- or only the exact fallback.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOLVER = ROOT / "solvers" / "cpp" / "build" / "stateful_compare"
DEFAULT_DATA_DIR = (
    ROOT
    / "data"
    / "green-scheduling-bab"
    / "Iirc.EnergyStatesAndCostsScheduling"
    / "data"
    / "datasets"
    / "k_boundary_202604"
)
DEFAULT_OUT = ROOT / "results" / "k_boundary_202604.csv"


def load_instance(path: Path) -> dict:
    with open(path) as f:
        d = json.load(f)
    jobs = [j["ProcessingTime"] for j in d["Jobs"]]
    prices = d["EnergyCosts"]
    noff = len(d.get("OffOnTime", []))
    machine = "twosby" if noff == 3 else "nosby"
    return {"jobs": jobs, "prices": prices, "machine": machine}


def run_mode(mode: str, extra_arg: str | None, payload: list[str], timeout_sec: int) -> list[dict]:
    cmd = [str(SOLVER), mode]
    if extra_arg is not None:
        cmd.append(extra_arg)
    proc = subprocess.run(
        cmd,
        input="\n".join(payload) + "\n",
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[:1000])

    rows: list[dict] = []
    header: list[str] | None = None
    for line in proc.stdout.strip().splitlines():
        if not line.strip():
            continue
        if header is None:
            header = line.split(",")
            continue
        parts = line.split(",")
        if len(parts) >= len(header):
            rows.append(dict(zip(header, parts)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run K-boundary benchmark")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--solver-timeout", type=int, default=240)
    parser.add_argument("--exact-time-limit", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--name-substr", action="append", default=[])
    parser.add_argument("--name-all-substr", action="append", default=[])
    args = parser.parse_args()

    manifest_path = args.data_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            rows = json.load(f)
        manifest = {row["file"]: row for row in rows}

    json_files = sorted(
        jf for jf in args.data_dir.glob("*.json") if jf.name != "manifest.json"
    )
    if args.name_substr:
        json_files = [
            jf for jf in json_files
            if any(substr in jf.stem for substr in args.name_substr)
        ]
    if args.name_all_substr:
        json_files = [
            jf for jf in json_files
            if all(substr in jf.stem for substr in args.name_all_substr)
        ]

    payload = []
    instance_ids = []
    for jf in json_files:
        inst = load_instance(jf)
        instance_ids.append(jf.stem)
        payload.append(
            json.dumps(
                {
                    "instance_id": jf.stem,
                    "jobs": inst["jobs"],
                    "prices": inst["prices"],
                    "machine": inst["machine"],
                }
            )
        )

    ab_rows: list[dict] = []
    hier_rows: list[dict] = []
    for start in range(0, len(payload), args.batch_size):
        batch = payload[start : start + args.batch_size]
        ab_rows.extend(run_mode("ablation-stdin", "full", batch, args.solver_timeout))
        hier_rows.extend(
            run_mode(
                "relax-hierarchy-stdin",
                f"{args.exact_time_limit:.6f}",
                batch,
                args.solver_timeout,
            )
        )
        print(f"Progress {min(start + args.batch_size, len(payload))}/{len(payload)}")

    by_id: dict[str, dict] = {}
    for row in ab_rows:
        by_id[row["instance_id"]] = dict(row)
    for row in hier_rows:
        tgt = by_id.setdefault(row["instance_id"], {})
        for key, val in row.items():
            if key == "instance_id":
                continue
            tgt[key] = val

    merged: list[dict] = []
    for jf in json_files:
        iid = jf.stem
        row = by_id.get(iid, {"instance_id": iid})
        meta = manifest.get(jf.name, {})
        md = meta.get("metadata", {})
        row["family"] = meta.get("family", "")
        row["K"] = str(md.get("K", len(md.get("processing_group", [])) if md else ""))
        row["processing_group"] = "-".join(str(x) for x in md.get("processing_group", []))
        merged.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in merged:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(merged)

    print(f"CSV: {args.out}")
    print(f"Instances: {len(merged)}")


if __name__ == "__main__":
    main()

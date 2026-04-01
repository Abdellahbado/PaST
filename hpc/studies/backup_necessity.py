#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

from common import PAPER_DATASETS, SOLVER, log


def load_instances(dataset_name: str):
    data_dir = PAPER_DATASETS / dataset_name
    json_files = sorted(jf for jf in data_dir.glob("*.json") if jf.name != "manifest.json")
    payload = []
    manifest = {}
    mp = data_dir / "manifest.json"
    if mp.exists():
        with open(mp) as f:
            rows = json.load(f)
        manifest = {row["file"]: row for row in rows}
    for jf in json_files:
        with open(jf) as f:
            d = json.load(f)
        payload.append(
            json.dumps(
                {
                    "instance_id": jf.stem,
                    "jobs": [j["ProcessingTime"] for j in d["Jobs"]],
                    "prices": d["EnergyCosts"],
                    "machine": "twosby" if len(d.get("OffOnTime", [])) == 3 else "nosby",
                }
            )
        )
    return json_files, payload, manifest


def call_solver(mode: str, payload: list[str], timeout: int, extra_arg: str | None = None, env=None):
    cmd = [str(SOLVER), mode]
    if extra_arg is not None:
        cmd.append(str(extra_arg))
    proc = subprocess.run(
        cmd,
        input="\n".join(payload) + "\n",
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[:1000])
    rows = []
    header = None
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


def call_solver_batched(mode: str, payload: list[str], timeout: int, batch_size: int, extra_arg: str | None = None, env=None):
    rows = []
    for start in range(0, len(payload), batch_size):
        batch = payload[start : start + batch_size]
        rows.extend(call_solver(mode, batch, timeout=timeout, extra_arg=extra_arg, env=env))
        print(f"Progress {min(start + batch_size, len(payload))}/{len(payload)}", flush=True)
    return rows


def build_report(rows: list[dict]) -> str:
    rescue = [r for r in rows if r["semi_packable"] == "0" and r["feas_packable"] == "1"]
    ties = [r for r in rows if r["lb_semi"] == r["lb_feas"]]
    improved = [r for r in rows if r["lb_semi"] != "" and r["lb_feas"] != "" and float(r["lb_feas"]) > float(r["lb_semi"]) + 1e-9]
    cats = Counter(r["category"] for r in rows)

    lines = ["# Backup Necessity Report", "", "## Overall"]
    lines.append(f"- instances: {len(rows)}")
    lines.append(f"- semigroup and R_feas tied on {len(ties)} instances")
    lines.append(f"- R_feas improved the lower bound on {len(improved)} instances")
    lines.append(f"- R_feas rescued packability on {len(rescue)} instances")
    lines.append("")
    lines.append("## By Category")
    for cat, cnt in sorted(cats.items()):
        cat_rows = [r for r in rows if r["category"] == cat]
        cat_rescue = sum(1 for r in cat_rows if r["semi_packable"] == "0" and r["feas_packable"] == "1")
        lines.append(f"- `{cat}`: {cnt} instances, packability rescues = {cat_rescue}")
    if rescue:
        lines.append("")
        lines.append("## Rescue Instances")
        for r in rescue:
            lines.append(
                f"- `{r['instance_id']}`: semi={r['lb_semi']} (unpackable), feas={r['lb_feas']} (packable), opt={r['opt']}"
            )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Study when R_feas is needed beyond semigroup")
    ap.add_argument("--dataset", default="paperext_backup_realistic_202604")
    ap.add_argument("--solver-timeout", type=int, default=240)
    ap.add_argument("--exact-time-limit", type=float, default=60.0)
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    out_dir = Path(args.output_dir or "hpc/results_studies/backup_necessity")
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = open(out_dir / "run.log", "w")

    log("=== Backup Necessity Study ===", logfile)
    json_files, payload, manifest = load_instances(args.dataset)
    log(f"Instances: {len(json_files)}", logfile)

    if not json_files:
        report = (
            "# Backup Necessity Report\n\n"
            "## Overall\n"
            "- instances: 0\n"
            "- dataset missing or empty; no study was run.\n"
        )
        (out_dir / "report.md").write_text(report)
        print(report)
        print(f"Saved results to {out_dir}")
        return

    pack_env = os.environ.copy()
    pack_env["PAST_RELAXED_BINPACK_ALLOW_SMALL_NC"] = "1"
    pack_env["PAST_RELAXED_BINPACK_NATIVE_FIRST"] = "1"

    pack_rows = call_solver_batched("relax-pack-stdin", payload, timeout=args.solver_timeout, batch_size=args.batch_size, env=pack_env)
    hier_rows = call_solver_batched(
        "relax-hierarchy-stdin",
        payload,
        timeout=args.solver_timeout,
        batch_size=args.batch_size,
        extra_arg=args.exact_time_limit,
    )
    pack_by = {r["instance_id"]: r for r in pack_rows}
    hier_by = {r["instance_id"]: r for r in hier_rows}

    merged = []
    for jf in json_files:
        iid = jf.stem
        meta = manifest.get(jf.name, {})
        p = pack_by.get(iid, {})
        h = hier_by.get(iid, {})
        merged.append(
            {
                "instance_id": iid,
                "category": meta.get("metadata", {}).get("category", meta.get("category", "")),
                "lb_semi": p.get("lb_semi", ""),
                "semi_packable": p.get("semi_packable", ""),
                "semi_pack_method": p.get("semi_pack_method", ""),
                "lb_feas": p.get("lb_feas", ""),
                "feas_packable": p.get("feas_packable", ""),
                "feas_pack_method": p.get("feas_pack_method", ""),
                "opt": h.get("opt", ""),
                "is_optimal": h.get("is_optimal", ""),
                "t_semi": h.get("t_semi", ""),
                "t_feas": h.get("t_feas", ""),
            }
        )

    if merged:
        with open(out_dir / "combined.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
            writer.writeheader()
            writer.writerows(merged)

    report = build_report(merged)
    (out_dir / "report.md").write_text(report)
    print(report)
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()

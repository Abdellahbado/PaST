#!/usr/bin/env python3
"""
Run our solver on one formal benchmark-extension suite.

Paper-facing policy:
- scalability_large_n: default production solver only
- backup_realistic: semigroup + R_feas only, with packability + exact check
- k_boundary: default production solver + semigroup/R_feas only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOLVER = ROOT / "solvers" / "cpp" / "build" / "stateful_compare"
DATA = ROOT / "data" / "green-scheduling-bab" / "Iirc.EnergyStatesAndCostsScheduling" / "data" / "datasets"

SUITE_DIRS = {
    "scalability_large_n": DATA / "paperext_scalability_large_n_202604",
    "backup_realistic": DATA / "paperext_backup_realistic_202604",
    "k_boundary": DATA / "paperext_k_boundary_202604",
}


def load_instance(jf: Path) -> dict:
    with open(jf) as f:
        d = json.load(f)
    return {
        "jobs": [j["ProcessingTime"] for j in d["Jobs"]],
        "prices": d["EnergyCosts"],
        "machine": "twosby" if len(d.get("OffOnTime", [])) == 3 else "nosby",
        "n_jobs": len(d["Jobs"]),
        "horizon": len(d["EnergyCosts"]),
    }


def solver_usage_text() -> str:
    proc = subprocess.run(
        [str(SOLVER), "__invalid_mode__"],
        capture_output=True,
        text=True,
    )
    return (proc.stdout or "") + (proc.stderr or "")


def ensure_solver_modes(required_modes: list[str]) -> None:
    usage = solver_usage_text()
    missing = [mode for mode in required_modes if mode not in usage]
    if missing:
        missing_s = ", ".join(missing)
        raise RuntimeError(
            "The built solver does not support the required mode(s): "
            f"{missing_s}. Rebuild `solvers/cpp/build/stateful_compare` on the HPC after pulling the latest branch."
        )


def call_solver(mode: str, payload_lines: list[str], timeout: int, extra_arg: str | None = None, env: dict[str, str] | None = None) -> list[dict]:
    cmd = [str(SOLVER), mode]
    if extra_arg is not None:
        cmd.append(extra_arg)
    try:
        proc = subprocess.run(
            cmd,
            input="\n".join(payload_lines) + "\n",
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Solver timed out after {timeout}s in mode `{mode}` on a batch of {len(payload_lines)} instance(s). "
            "Reduce batch size or increase `--solver-timeout`."
        ) from exc
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


def call_solver_batched(mode: str, payload_lines: list[str], timeout: int, batch_size: int, extra_arg: str | None = None, env: dict[str, str] | None = None) -> list[dict]:
    rows: list[dict] = []
    for start in range(0, len(payload_lines), batch_size):
        batch = payload_lines[start : start + batch_size]
        rows.extend(call_solver(mode, batch, timeout=timeout, extra_arg=extra_arg, env=env))
        print(f"Progress {min(start + batch_size, len(payload_lines))}/{len(payload_lines)}", flush=True)
    return rows


def load_manifest(data_dir: Path) -> dict[str, dict]:
    mp = data_dir / "manifest.json"
    if not mp.exists():
        return {}
    with open(mp) as f:
        rows = json.load(f)
    return {row["file"]: row for row in rows}


def suite_payload(data_dir: Path) -> tuple[list[Path], list[str], dict[str, dict]]:
    manifest = load_manifest(data_dir)
    json_files = sorted(jf for jf in data_dir.glob("*.json") if jf.name != "manifest.json")
    payload_lines = []
    for jf in json_files:
        inst = load_instance(jf)
        payload_lines.append(
            json.dumps(
                {
                    "instance_id": jf.stem,
                    "jobs": inst["jobs"],
                    "prices": inst["prices"],
                    "machine": inst["machine"],
                }
            )
        )
    return json_files, payload_lines, manifest


def run_scalability(data_dir: Path, out: Path, time_limit: float, timeout: int, batch_size: int) -> None:
    json_files, payload, manifest = suite_payload(data_dir)
    rows = call_solver_batched("ablation-stdin", payload, timeout, batch_size, extra_arg="full")
    by_id = {r["instance_id"]: r for r in rows}

    merged = []
    for jf in json_files:
        row = by_id.get(jf.stem, {"instance_id": jf.stem})
        meta = manifest.get(jf.name, {})
        md = meta.get("metadata", {})
        merged.append(
            {
                "suite": "scalability_large_n",
                "instance_id": jf.stem,
                "n_jobs": row.get("n_jobs", ""),
                "horizon": row.get("horizon", ""),
                "runtime_sec": row.get("runtime_sec", ""),
                "ub": row.get("ub", ""),
                "lb": row.get("lb", ""),
                "gap_pct": row.get("gap_pct", ""),
                "feasible": row.get("feasible", ""),
                "is_optimal": row.get("is_optimal", ""),
                "timed_out": row.get("timed_out", ""),
                "step_reached": row.get("step_reached", ""),
                "winner_detail": row.get("winner_detail", ""),
                "max_gap": row.get("max_gap", ""),
                "processing_group": "-".join(str(x) for x in md.get("processing_group", [])),
                "ec_repeat": md.get("ec_repeat", ""),
                "seed": md.get("seed", ""),
            }
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
        writer.writeheader()
        writer.writerows(merged)


def run_backup(data_dir: Path, out: Path, timeout: int, exact_time_limit: float, batch_size: int) -> None:
    json_files, payload, manifest = suite_payload(data_dir)
    pack_env = os.environ.copy()
    pack_env["PAST_RELAXED_BINPACK_ALLOW_SMALL_NC"] = "1"
    pack_env["PAST_RELAXED_BINPACK_NATIVE_FIRST"] = "1"

    pack_rows = call_solver_batched("relax-pack-stdin", payload, timeout=timeout, batch_size=batch_size, env=pack_env)
    hier_rows = call_solver_batched(
        "relax-hierarchy-stdin",
        payload,
        timeout=timeout,
        batch_size=batch_size,
        extra_arg=str(exact_time_limit),
    )
    pack_by = {r["instance_id"]: r for r in pack_rows}
    hier_by = {r["instance_id"]: r for r in hier_rows}

    merged = []
    for jf in json_files:
        iid = jf.stem
        meta = manifest.get(jf.name, {})
        md = meta.get("metadata", {})
        p = pack_by.get(iid, {})
        h = hier_by.get(iid, {})
        merged.append(
            {
                "suite": "backup_realistic",
                "instance_id": iid,
                "category": md.get("category", meta.get("category", "")),
                "n_jobs": p.get("n_jobs", ""),
                "horizon": p.get("horizon", ""),
                "lb_semi": p.get("lb_semi", ""),
                "semi_packable": p.get("semi_packable", ""),
                "semi_pack_outcome": p.get("semi_pack_outcome", ""),
                "semi_pack_method": p.get("semi_pack_method", ""),
                "lb_feas": p.get("lb_feas", ""),
                "feas_packable": p.get("feas_packable", ""),
                "feas_pack_outcome": p.get("feas_pack_outcome", ""),
                "feas_pack_method": p.get("feas_pack_method", ""),
                "opt": h.get("opt", ""),
                "is_optimal": h.get("is_optimal", ""),
                "t_semi": h.get("t_semi", ""),
                "t_feas": h.get("t_feas", ""),
                "t_opt": h.get("t_opt", ""),
            }
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
        writer.writeheader()
        writer.writerows(merged)


def run_k_boundary(data_dir: Path, out: Path, timeout: int, exact_time_limit: float, batch_size: int) -> None:
    json_files, payload, manifest = suite_payload(data_dir)
    ab_rows = call_solver_batched("ablation-stdin", payload, timeout, batch_size, extra_arg="full")
    hier_rows = call_solver_batched("relax-hierarchy-stdin", payload, timeout, batch_size, extra_arg=str(exact_time_limit))
    ab_by = {r["instance_id"]: r for r in ab_rows}
    hier_by = {r["instance_id"]: r for r in hier_rows}

    merged = []
    for jf in json_files:
        iid = jf.stem
        meta = manifest.get(jf.name, {})
        md = meta.get("metadata", {})
        a = ab_by.get(iid, {})
        h = hier_by.get(iid, {})
        merged.append(
            {
                "suite": "k_boundary",
                "instance_id": iid,
                "family": meta.get("source_family", meta.get("family", "")),
                "K": md.get("K", ""),
                "processing_group": "-".join(str(x) for x in md.get("processing_group", [])),
                "n_jobs": a.get("n_jobs", ""),
                "horizon": a.get("horizon", ""),
                "runtime_sec": a.get("runtime_sec", ""),
                "step_reached": a.get("step_reached", ""),
                "winner_detail": a.get("winner_detail", ""),
                "lb_semi": h.get("lb_semi", ""),
                "lb_feas": h.get("lb_feas", ""),
                "opt": h.get("opt", ""),
                "is_optimal": a.get("is_optimal", ""),
                "timed_out": a.get("timed_out", ""),
            }
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
        writer.writeheader()
        writer.writerows(merged)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run our solver on a formal extension suite")
    ap.add_argument("--suite", choices=sorted(SUITE_DIRS.keys()), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--solver-timeout", type=int, default=240)
    ap.add_argument("--exact-time-limit", type=float, default=20.0)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    data_dir = SUITE_DIRS[args.suite]
    if args.suite == "scalability_large_n":
        ensure_solver_modes(["ablation-stdin"])
        run_scalability(data_dir, args.out, args.time_limit, args.solver_timeout, args.batch_size)
    elif args.suite == "backup_realistic":
        ensure_solver_modes(["relax-pack-stdin", "relax-hierarchy-stdin"])
        run_backup(data_dir, args.out, args.solver_timeout, args.exact_time_limit, args.batch_size)
    elif args.suite == "k_boundary":
        ensure_solver_modes(["ablation-stdin", "relax-hierarchy-stdin"])
        run_k_boundary(data_dir, args.out, args.solver_timeout, args.exact_time_limit, args.batch_size)

    print(f"CSV: {args.out}")


if __name__ == "__main__":
    main()

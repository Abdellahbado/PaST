#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hpc.benchmark_extensions.build_extension_suites import build_instance


SOLVER = ROOT / "solvers" / "cpp" / "build" / "stateful_compare"
OUT_DIR = ROOT / "research" / "k_vs_arithmetic_axes_20260412" / "csv" / "plan08"
DEFAULT_CSV = OUT_DIR / "PLAN08_energy_core_campaign.csv"
DEFAULT_JSON = OUT_DIR / "PLAN08_energy_core_campaign.json"

DATASET_DIR = (
    ROOT
    / "data"
    / "green-scheduling-bab"
    / "Iirc.EnergyStatesAndCostsScheduling"
    / "data"
    / "datasets"
    / "paperext_profile_repair_smallk_nscale_plus_20260409"
)

GROUPS: dict[str, list[int]] = {
    "g3567": [3, 5, 6, 7],
    "g12357": [1, 2, 3, 5, 7],
    "g246810": [2, 4, 6, 8, 10],
    "g12345678910": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}

EC_CONFIGS = [
    {"from_date": "2019-01-21T00:00:00", "repeat_count": 1},
    {"from_date": "2019-04-08T00:00:00", "repeat_count": 1},
]


def parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_ablation_row(stdout: str) -> dict[str, str]:
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    if len(lines) < 2:
        return {}
    rows = list(csv.DictReader(lines))
    return rows[0] if rows else {}


def stable_seed(family_id: str, n_jobs: int, lam: float, seed: int) -> int:
    lam_tag = int(round(lam * 100))
    return (
        700000
        + 131 * seed
        + 1009 * n_jobs
        + 17 * lam_tag
        + sum(ord(c) for c in family_id)
    )


def time_limit_for_n(args: argparse.Namespace, n_jobs: int) -> float:
    if n_jobs >= args.large_n_cutoff:
        return args.time_limit_large
    if n_jobs >= args.medium_n_cutoff:
        return args.time_limit_medium
    return args.time_limit_small


def build_group_payload(
    family_id: str,
    n_jobs: int,
    lam: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    lengths = GROUPS[family_id]
    ec = EC_CONFIGS[seed % len(EC_CONFIGS)]
    rng = random.Random(stable_seed(family_id, n_jobs, lam, seed))
    jobs = [rng.choice(lengths) for _ in range(n_jobs)]

    inst = build_instance(
        name=f"plan08/{family_id}_n{n_jobs}_lam{lam:.1f}_s{seed}",
        family=family_id,
        jobs_list=jobs,
        horizon_multiplier=lam,
        ec_config=ec,
        metadata={
            "processing_group": lengths,
            "K": len(lengths),
            "seed": seed,
            "lambda": lam,
            "paper_machine": "twosby",
            "plan": "plan08_energy_core",
        },
    )

    payload = {
        "instance_id": inst["name"],
        "prices": inst["prices"],
        "jobs": inst["jobs"],
        "machine": "twosby",
    }
    return payload, inst


def build_continuity_payload(
    n_jobs: int, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    matches = sorted(
        DATASET_DIR.glob(f"*_profile_smallk_3567_plus_n{n_jobs}_s{seed}.json")
    )
    if not matches:
        raise FileNotFoundError(
            f"Missing historical payload for 3567_plus n={n_jobs} seed={seed} in {DATASET_DIR}"
        )
    src = matches[0]
    raw = json.loads(src.read_text(encoding="utf-8"))

    if "prices" in raw and "jobs" in raw:
        prices = raw["prices"]
        jobs = raw["jobs"]
    else:
        prices = raw.get("EnergyCosts", [])
        jobs = [int(j.get("ProcessingTime", 0)) for j in raw.get("Jobs", [])]
        if not prices or not jobs:
            raise ValueError(f"Unsupported historical payload format in {src}")

    payload = {
        "instance_id": raw.get("instance_id", src.stem),
        "prices": prices,
        "jobs": jobs,
        "machine": "nosby",
    }
    meta = {
        "source_file": str(src),
        "horizon": len(payload["prices"]),
        "n_jobs": len(payload["jobs"]),
    }
    return payload, meta


def load_existing_case_ids(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    out: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            case_id = row.get("case_id", "").strip()
            if case_id:
                out.add(case_id)
    return out


def write_rows_dedup(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    existing: list[dict[str, str]] = []
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))

    all_rows: list[dict[str, Any]] = []
    all_rows.extend(existing)
    all_rows.extend(rows)

    last_idx: dict[str, int] = {}
    for i, row in enumerate(all_rows):
        cid = str(row.get("case_id", "")).strip()
        if cid:
            last_idx[cid] = i

    deduped: list[dict[str, Any]] = []
    for i, row in enumerate(all_rows):
        cid = str(row.get("case_id", "")).strip()
        if cid and last_idx.get(cid) != i:
            continue
        deduped.append(row)

    fieldnames: list[str] = []
    for row in deduped:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(deduped)


def build_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    seeds = parse_ints(args.seeds)
    g3567_ns = parse_ints(args.g3567_ns)
    transfer_ns = parse_ints(args.transfer_ns)
    lam = args.lambda_value

    cases: list[dict[str, Any]] = []

    if not args.skip_g3567:
        for n_jobs in g3567_ns:
            for seed in seeds:
                cases.append(
                    {
                        "case_id": f"g3567_n{n_jobs}_lam{lam:.1f}_s{seed}",
                        "kind": "paper_group",
                        "family_id": "g3567",
                        "n": n_jobs,
                        "lambda": lam,
                        "seed": seed,
                        "machine": "twosby",
                        "time_limit_sec": time_limit_for_n(args, n_jobs),
                    }
                )

    if not args.skip_continuity:
        for n_jobs in (3500, 5000):
            for seed in seeds:
                cases.append(
                    {
                        "case_id": f"continuity_3567plus_n{n_jobs}_s{seed}",
                        "kind": "continuity_3567_plus",
                        "family_id": "3567_plus",
                        "n": n_jobs,
                        "lambda": "historical",
                        "seed": seed,
                        "machine": "nosby",
                        "time_limit_sec": time_limit_for_n(args, n_jobs),
                    }
                )

    if not args.skip_transfer:
        for family_id in ("g12357", "g246810"):
            for n_jobs in transfer_ns:
                for seed in seeds:
                    cases.append(
                        {
                            "case_id": f"transfer_{family_id}_n{n_jobs}_lam{lam:.1f}_s{seed}",
                            "kind": "paper_group_transfer",
                            "family_id": family_id,
                            "n": n_jobs,
                            "lambda": lam,
                            "seed": seed,
                            "machine": "twosby",
                            "time_limit_sec": time_limit_for_n(args, n_jobs),
                        }
                    )

    if args.include_optional_k10:
        for n_jobs in (1000, 1500):
            for seed in seeds:
                cases.append(
                    {
                        "case_id": f"transfer_g12345678910_n{n_jobs}_lam{lam:.1f}_s{seed}",
                        "kind": "paper_group_transfer_optional",
                        "family_id": "g12345678910",
                        "n": n_jobs,
                        "lambda": lam,
                        "seed": seed,
                        "machine": "twosby",
                        "time_limit_sec": time_limit_for_n(args, n_jobs),
                    }
                )

    if args.max_cases > 0:
        return cases[: args.max_cases]
    return cases


def run_case(case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if case["kind"] == "continuity_3567_plus":
        payload, meta = build_continuity_payload(case["n"], case["seed"])
        source_file = meta["source_file"]
        ec = EC_CONFIGS[case["seed"] % len(EC_CONFIGS)]
    else:
        payload, _inst = build_group_payload(
            family_id=case["family_id"],
            n_jobs=case["n"],
            lam=float(case["lambda"]),
            seed=case["seed"],
        )
        source_file = "generated"
        ec = EC_CONFIGS[case["seed"] % len(EC_CONFIGS)]

    cmd = [
        str(SOLVER),
        "ablation-stdin",
        "step1_exact_guided",
        str(case["time_limit_sec"]),
    ]
    env = os.environ.copy()
    env["PAST_RELAXED_BINPACK_SOLVER"] = args.pack_solver
    env["PAST_BLOCK_REPAIR_COMPLETION_MODE"] = args.completion_mode
    env["PAST_BLOCK_REPAIR_TRACE"] = str(args.trace)
    env["PAST_BLOCK_REPAIR_POOL_DIAG"] = str(args.pool_diag)

    ext_timeout = int(max(case["time_limit_sec"] + 240.0, 300.0))
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            input=(json.dumps(payload) + "\n"),
            capture_output=True,
            text=True,
            timeout=ext_timeout,
            check=False,
            env=env,
        )
        status = "ok"
        rc = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        status = "external_timeout"
        rc = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""

    wall = time.monotonic() - t0
    raw = parse_ablation_row(stdout)
    stderr_tail = (stderr or "")[-400:]
    stderr_tail = stderr_tail.replace("\r", "\\r").replace("\n", "\\n")

    row: dict[str, Any] = {
        "case_id": case["case_id"],
        "kind": case["kind"],
        "family_id": case["family_id"],
        "n": case["n"],
        "lambda": case["lambda"],
        "seed": case["seed"],
        "machine": case["machine"],
        "time_limit_sec": case["time_limit_sec"],
        "status": status,
        "solver_returncode": rc,
        "wall_sec": f"{wall:.4f}",
        "source": source_file,
        "ec_from": ec["from_date"],
        "ec_repeat": ec["repeat_count"],
        "pack_solver_env": args.pack_solver,
        "completion_mode_env": args.completion_mode,
        "stderr_tail": stderr_tail,
    }

    for key, value in raw.items():
        row[key] = value

    if not raw:
        row.setdefault("runtime_sec", "")
        row.setdefault("ub", "")
        row.setdefault("lb", "")
        row.setdefault("gap_pct", "")
        row.setdefault("timed_out", "1")
        row.setdefault("feasible", "0")
        row.setdefault("is_optimal", "0")

    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Run PLAN_08 energy-core campaign rows")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--output-csv", default=str(DEFAULT_CSV))
    ap.add_argument("--output-json", default=str(DEFAULT_JSON))
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--lambda-value", type=float, default=1.3)
    ap.add_argument("--g3567-ns", default="1000,1500,2500,3500,5000")
    ap.add_argument("--transfer-ns", default="1000,1500,2500")
    ap.add_argument("--skip-g3567", action="store_true")
    ap.add_argument("--skip-continuity", action="store_true")
    ap.add_argument("--skip-transfer", action="store_true")
    ap.add_argument("--include-optional-k10", action="store_true")
    ap.add_argument("--max-cases", type=int, default=0)

    ap.add_argument("--pack-solver", default="energy_core")
    ap.add_argument("--completion-mode", default="direct")
    ap.add_argument("--trace", type=int, default=0)
    ap.add_argument("--pool-diag", type=int, default=0)

    ap.add_argument("--time-limit-small", type=float, default=900.0)
    ap.add_argument("--time-limit-medium", type=float, default=1800.0)
    ap.add_argument("--time-limit-large", type=float, default=3600.0)
    ap.add_argument("--medium-n-cutoff", type=int, default=2500)
    ap.add_argument("--large-n-cutoff", type=int, default=3500)

    args = ap.parse_args()

    if not SOLVER.exists():
        raise FileNotFoundError(f"Missing solver binary: {SOLVER}")
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Missing dataset directory: {DATASET_DIR}")

    out_csv = Path(args.output_csv)
    out_json = Path(args.output_json)

    cases = build_cases(args)
    print(f"Prepared {len(cases)} cases")

    seen = load_existing_case_ids(out_csv) if args.resume else set()
    rows_new: list[dict[str, Any]] = []

    for idx, case in enumerate(cases, start=1):
        if args.resume and case["case_id"] in seen:
            print(f"[{idx}/{len(cases)}] skip {case['case_id']} (resume)")
            continue

        row = run_case(case, args)
        rows_new.append(row)
        write_rows_dedup(out_csv, [row])

        gap = row.get("gap_pct", "")
        step = "unknown"
        if row.get("diag_step1_decided") == "1":
            step = "step1"
        elif row.get("diag_step2_decided") == "1":
            step = "step2"
        elif row.get("diag_step3_decided") == "1":
            step = "step3"
        elif row.get("diag_step4_decided") == "1":
            step = "step4"
        print(
            f"[{idx}/{len(cases)}] {case['case_id']} status={row.get('status')} "
            f"rc={row.get('solver_returncode')} gap={gap} step={step} "
            f"t={row.get('runtime_sec', row.get('wall_sec', ''))}"
        )

    all_rows: list[dict[str, Any]] = []
    if out_csv.exists():
        with out_csv.open(newline="", encoding="utf-8") as f:
            all_rows = list(csv.DictReader(f))

    payload = {
        "cases_total": len(cases),
        "cases_new": len(rows_new),
        "rows_total": len(all_rows),
        "output_csv": str(out_csv),
        "output_csv_abs": str(out_csv.resolve()),
        "env": {
            "PAST_RELAXED_BINPACK_SOLVER": args.pack_solver,
            "PAST_BLOCK_REPAIR_COMPLETION_MODE": args.completion_mode,
            "PAST_BLOCK_REPAIR_TRACE": args.trace,
            "PAST_BLOCK_REPAIR_POOL_DIAG": args.pool_diag,
        },
        "rows": all_rows,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote JSON: {out_json}")


if __name__ == "__main__":
    main()

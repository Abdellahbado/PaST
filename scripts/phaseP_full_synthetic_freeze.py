#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


CSV_HEADER_PREFIX = "instance_id,epsilon,epsilon_prev,variant,runtime_sec,tec_total"


@dataclass(frozen=True)
class ManifestRow:
    split: str
    instance_uid: str
    instance_id: int
    m: int
    n: int
    k: int
    seed: int
    data_p_path: Path


def _read_int_values(path: Path) -> List[int]:
    vals: List[int] = []
    for tok in path.read_text(encoding="utf-8").split():
        t = tok.strip()
        if not t:
            continue
        vals.append(int(round(float(t))))
    return vals


def _resolve_data_path(path_str: str, manifest_path: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (manifest_path.parent / p).resolve()


def _read_manifest(path: Path, split_expected: str) -> List[ManifestRow]:
    rows: List[ManifestRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = str(row.get("split", "")).strip()
            if split != split_expected:
                raise RuntimeError(
                    f"Manifest split mismatch in {path}: expected {split_expected}, got {split}"
                )
            rows.append(
                ManifestRow(
                    split=split,
                    instance_uid=str(row["instance_uid"]),
                    instance_id=int(row["instance_id"]),
                    m=int(row["M"]),
                    n=int(row["N"]),
                    k=int(row["K"]),
                    seed=int(row["seed"]),
                    data_p_path=_resolve_data_path(str(row["data_p_path"]), path),
                )
            )
    return rows


def _epsilon_from_instance(row: ManifestRow, epsilon_slack: int) -> Tuple[int, int, int, int]:
    p = _read_int_values(row.data_p_path)
    if not p:
        return 0, 0, 0, 0
    p_sum = int(sum(p))
    p_max = int(max(p))
    eps_lb = max(int(math.ceil(float(p_sum) / float(row.m))), p_max)
    eps = min(row.k, eps_lb + epsilon_slack)
    return eps, eps_lb, p_sum, p_max


def _parse_max_rss(stderr_text: str) -> int:
    marker = "maximum resident set size"
    for line in stderr_text.splitlines():
        if marker in line:
            toks = line.strip().split()
            if toks:
                try:
                    return int(toks[0])
                except ValueError:
                    pass
    return -1


def _run_solver(
    solver_bin: Path,
    synthetic_data_dir: Path,
    instance_id: int,
    epsilon: int,
    per_machine_dp_limit_sec: float,
    ls_time_cap_sec: float,
    ls_max_rounds: int,
    ls_max_moves_per_round: int,
    workdir: Path,
) -> Tuple[int, Dict[str, str], int, float, str]:
    cmd = [
        "/usr/bin/time",
        "-l",
        str(solver_bin),
        "paper-instance",
        str(instance_id),
        str(epsilon),
        "stageO_synthetic_dense_logging",
        str(synthetic_data_dir),
        str(per_machine_dp_limit_sec),
        str(ls_time_cap_sec),
        str(ls_max_rounds),
        str(ls_max_moves_per_round),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=workdir, text=True, capture_output=True)
    wall_sec = time.perf_counter() - t0
    max_rss = _parse_max_rss(proc.stderr)

    parsed_row: Dict[str, str] = {}
    stdout_lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if stdout_lines and stdout_lines[0].startswith(CSV_HEADER_PREFIX) and len(stdout_lines) >= 2:
        reader = csv.DictReader(stdout_lines)
        for row in reader:
            parsed_row = row
            break

    return proc.returncode, parsed_row, max_rss, wall_sec, proc.stderr


def _read_labeled_moves(path: Path) -> Tuple[List[Dict[str, str]], int, int]:
    if not path.exists():
        return [], 0, 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    pos = 0
    neg = 0
    for r in rows:
        if str(r.get("label_improving", "0")).strip() == "1":
            pos += 1
        else:
            neg += 1
    return rows, pos, neg


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_existing_progress(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = str(row.get("instance_uid", "")).strip()
            if uid:
                out[uid] = row
    return out


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _safe_int(x: object, default: int = 0) -> int:
    try:
        return int(float(str(x)))
    except (TypeError, ValueError):
        return default


def _rate(num: int, den: int) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def _bucket_id(m: int, n: int, k: int) -> str:
    return f"m{m}_n{n}_k{k}"


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase P full-manifest synthetic dense labeling freeze")
    parser.add_argument(
        "--train-manifest",
        type=str,
        default="temp/phaseM_vls_synthetic_protocol/split_manifest_train.csv",
    )
    parser.add_argument(
        "--val-manifest",
        type=str,
        default="temp/phaseM_vls_synthetic_protocol/split_manifest_val.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="temp/phaseP_full_synthetic_freeze",
    )
    parser.add_argument(
        "--solver-bin",
        type=str,
        default="solvers/cpp/build/parallel_heuristic_compare",
    )
    parser.add_argument("--epsilon-slack", type=int, default=8)
    parser.add_argument("--per-machine-dp-limit-sec", type=float, default=1.0)
    parser.add_argument("--ls-time-cap-sec", type=float, default=2.0)
    parser.add_argument("--ls-max-rounds", type=int, default=2)
    parser.add_argument("--ls-max-moves-per-round", type=int, default=8000)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    train_manifest = (repo_root / args.train_manifest).resolve()
    val_manifest = (repo_root / args.val_manifest).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    solver_bin = (repo_root / args.solver_bin).resolve()

    synthetic_data_dir = train_manifest.parent / "synthetic_instances"
    stageo_raw_dir = repo_root / "temp/phaseO_synthetic_dense_labeling"
    per_instance_dir = output_dir / "per_instance"

    output_dir.mkdir(parents=True, exist_ok=True)
    per_instance_dir.mkdir(parents=True, exist_ok=True)

    train_rows = _read_manifest(train_manifest, split_expected="train")
    val_rows = _read_manifest(val_manifest, split_expected="val")
    all_rows = sorted(train_rows + val_rows, key=lambda r: (r.m, r.n, r.k, r.split, r.instance_id))

    progress_path = output_dir / "batch_progress.csv"
    existing = {} if args.no_resume else _load_existing_progress(progress_path)

    instance_progress: List[Dict[str, object]] = []

    for row in all_rows:
        eps, eps_lb, p_sum, p_max = _epsilon_from_instance(row, args.epsilon_slack)
        bucket = _bucket_id(row.m, row.n, row.k)
        batch_id = bucket

        exact_name = f"moves_exact_labeled_instance_{row.instance_id}_eps_{eps}.csv"
        broad_name = f"moves_broad_instance_{row.instance_id}_eps_{eps}.csv"
        dst_exact = per_instance_dir / exact_name
        dst_broad = per_instance_dir / broad_name
        src_exact = stageo_raw_dir / exact_name
        src_broad = stageo_raw_dir / broad_name

        prev = existing.get(row.instance_uid)
        can_skip = (
            prev is not None
            and str(prev.get("status", "")) == "ok"
            and _safe_int(prev.get("epsilon_used", -1), -1) == eps
            and dst_exact.exists()
        )

        status = "failed"
        attempts = 0
        retries_used = 0
        last_rc = -1
        solver_feasible = 0
        wall_runtime_sec = 0.0
        max_rss = -1
        solver_runtime_sec_reported = ""
        solver_variant_reported = ""
        stderr_tail = ""

        if can_skip:
            status = "ok"
            attempts = _safe_int(prev.get("attempts_total", 1), 1)
            retries_used = _safe_int(prev.get("retries_used", 0), 0)
            last_rc = _safe_int(prev.get("solver_return_code", 0), 0)
            solver_feasible = _safe_int(prev.get("solver_feasible", 1), 1)
            wall_runtime_sec = _safe_float(prev.get("wall_runtime_sec", 0.0), 0.0)
            max_rss = _safe_int(prev.get("max_rss_bytes", -1), -1)
            solver_runtime_sec_reported = str(prev.get("solver_runtime_sec_reported", ""))
            solver_variant_reported = str(prev.get("solver_variant_reported", ""))
            stderr_tail = str(prev.get("stderr_tail", ""))
        else:
            for attempt_idx in range(args.max_retries + 1):
                attempts += 1
                rc, parsed, rss, wall_sec, stderr_text = _run_solver(
                    solver_bin=solver_bin,
                    synthetic_data_dir=synthetic_data_dir,
                    instance_id=row.instance_id,
                    epsilon=eps,
                    per_machine_dp_limit_sec=args.per_machine_dp_limit_sec,
                    ls_time_cap_sec=args.ls_time_cap_sec,
                    ls_max_rounds=args.ls_max_rounds,
                    ls_max_moves_per_round=args.ls_max_moves_per_round,
                    workdir=repo_root,
                )
                last_rc = rc
                wall_runtime_sec += wall_sec
                max_rss = max(max_rss, rss)
                solver_runtime_sec_reported = str(parsed.get("runtime_sec", "")) if parsed else ""
                solver_variant_reported = str(parsed.get("variant", "")) if parsed else ""
                stderr_tail = "\\n".join(stderr_text.splitlines()[-4:])
                if parsed:
                    try:
                        solver_feasible = int(float(parsed.get("tec_total", "-1")) >= 0.0)
                    except ValueError:
                        solver_feasible = 0

                _copy_if_exists(src_exact, dst_exact)
                _copy_if_exists(src_broad, dst_broad)
                exact_rows, _, _ = _read_labeled_moves(dst_exact)
                if rc == 0 and len(exact_rows) > 0:
                    status = "ok"
                    break

                if attempt_idx < args.max_retries:
                    retries_used += 1

        exact_rows, pos_rows, neg_rows = _read_labeled_moves(dst_exact)

        instance_progress.append(
            {
                "batch_id": batch_id,
                "bucket": bucket,
                "split": row.split,
                "instance_uid": row.instance_uid,
                "instance_id": row.instance_id,
                "seed": row.seed,
                "M": row.m,
                "N": row.n,
                "K": row.k,
                "sum_p": p_sum,
                "p_max": p_max,
                "epsilon_lb": eps_lb,
                "epsilon_used": eps,
                "epsilon_stress": round(_rate(eps, row.k), 8),
                "status": status,
                "attempts_total": attempts,
                "retries_used": retries_used,
                "solver_return_code": last_rc,
                "solver_feasible": solver_feasible,
                "solver_runtime_sec_reported": solver_runtime_sec_reported,
                "solver_variant_reported": solver_variant_reported,
                "wall_runtime_sec": round(wall_runtime_sec, 6),
                "max_rss_bytes": max_rss,
                "exact_labeled_moves": len(exact_rows),
                "exact_positive_improving": pos_rows,
                "exact_negative_non_improving": neg_rows,
                "positive_rate_instance": round(_rate(pos_rows, len(exact_rows)), 8),
                "stderr_tail": stderr_tail,
                "exact_file": str(dst_exact),
                "broad_file": str(dst_broad),
            }
        )

    instance_progress.sort(key=lambda r: (str(r["bucket"]), str(r["split"]), int(r["instance_id"])))
    _write_csv(
        progress_path,
        instance_progress,
        [
            "batch_id",
            "bucket",
            "split",
            "instance_uid",
            "instance_id",
            "seed",
            "M",
            "N",
            "K",
            "sum_p",
            "p_max",
            "epsilon_lb",
            "epsilon_used",
            "epsilon_stress",
            "status",
            "attempts_total",
            "retries_used",
            "solver_return_code",
            "solver_feasible",
            "solver_runtime_sec_reported",
            "solver_variant_reported",
            "wall_runtime_sec",
            "max_rss_bytes",
            "exact_labeled_moves",
            "exact_positive_improving",
            "exact_negative_non_improving",
            "positive_rate_instance",
            "stderr_tail",
            "exact_file",
            "broad_file",
        ],
    )

    # Build frozen datasets from successful instances only.
    merged_rows: List[Dict[str, object]] = []
    train_rows_out: List[Dict[str, object]] = []
    val_rows_out: List[Dict[str, object]] = []
    for r in instance_progress:
        if str(r["status"]) != "ok":
            continue
        exact_file = Path(str(r["exact_file"]))
        exact_rows, _, _ = _read_labeled_moves(exact_file)
        for mrow in exact_rows:
            enriched: Dict[str, object] = dict(mrow)
            enriched["manifest_split"] = r["split"]
            enriched["manifest_instance_uid"] = r["instance_uid"]
            enriched["manifest_instance_id"] = r["instance_id"]
            enriched["manifest_seed"] = r["seed"]
            enriched["manifest_M"] = r["M"]
            enriched["manifest_N"] = r["N"]
            enriched["manifest_K"] = r["K"]
            enriched["batch_id"] = r["batch_id"]
            enriched["bucket"] = r["bucket"]
            enriched["epsilon_lb"] = r["epsilon_lb"]
            enriched["epsilon_used"] = r["epsilon_used"]
            enriched["epsilon_stress"] = r["epsilon_stress"]
            merged_rows.append(enriched)
            if str(r["split"]) == "train":
                train_rows_out.append(enriched)
            else:
                val_rows_out.append(enriched)

    def _write_moves(path: Path, rows: List[Dict[str, object]]) -> None:
        if rows:
            fields = list(rows[0].keys())
        else:
            fields = [
                "manifest_split",
                "manifest_instance_uid",
                "manifest_instance_id",
                "manifest_seed",
                "manifest_M",
                "manifest_N",
                "manifest_K",
                "batch_id",
                "bucket",
                "epsilon_lb",
                "epsilon_used",
                "epsilon_stress",
            ]
        _write_csv(path, rows, fields)

    train_frozen_path = output_dir / "synthetic_moves_exact_labeled_train_frozen.csv"
    val_frozen_path = output_dir / "synthetic_moves_exact_labeled_val_frozen.csv"
    merged_frozen_path = output_dir / "synthetic_moves_exact_labeled_frozen_merged.csv"
    _write_moves(train_frozen_path, train_rows_out)
    _write_moves(val_frozen_path, val_rows_out)
    _write_moves(merged_frozen_path, merged_rows)

    schema = {
        "dataset": "synthetic_moves_exact_labeled_frozen_merged",
        "columns": list(merged_rows[0].keys()) if merged_rows else [],
        "label_columns": ["label_improving", "label_accepted", "exact_total_delta"],
        "manifest_context_columns": [
            "manifest_split",
            "manifest_instance_uid",
            "manifest_instance_id",
            "manifest_seed",
            "manifest_M",
            "manifest_N",
            "manifest_K",
            "bucket",
            "epsilon_lb",
            "epsilon_used",
            "epsilon_stress",
        ],
    }
    (output_dir / "feature_schema_frozen.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    # Batch summary.
    batch_keys = sorted({(int(r["M"]), int(r["N"]), int(r["K"])) for r in instance_progress})
    batch_summary: List[Dict[str, object]] = []
    for m, n, k in batch_keys:
        bid = _bucket_id(m, n, k)
        subset = [r for r in instance_progress if int(r["M"]) == m and int(r["N"]) == n and int(r["K"]) == k]
        instances = len(subset)
        ok_subset = [r for r in subset if str(r["status"]) == "ok"]
        rows_total = sum(int(r["exact_labeled_moves"]) for r in ok_subset)
        pos_total = sum(int(r["exact_positive_improving"]) for r in ok_subset)
        neg_total = sum(int(r["exact_negative_non_improving"]) for r in ok_subset)
        retries_total = sum(int(r["retries_used"]) for r in subset)
        failures = sum(1 for r in subset if str(r["status"]) != "ok")
        wall = sum(float(r["wall_runtime_sec"]) for r in subset)
        max_rss = max((_safe_int(r["max_rss_bytes"], -1) for r in subset), default=-1)
        batch_summary.append(
            {
                "batch_id": bid,
                "M": m,
                "N": n,
                "K": k,
                "instances_total": instances,
                "instances_success": len(ok_subset),
                "instances_failed": failures,
                "exact_labeled_rows": rows_total,
                "positives": pos_total,
                "negatives": neg_total,
                "positive_rate": _rate(pos_total, rows_total),
                "negative_rate": _rate(neg_total, rows_total),
                "wall_runtime_sec": round(wall, 6),
                "max_rss_bytes": max_rss,
                "retries_used": retries_total,
            }
        )
    _write_csv(
        output_dir / "batch_summary.csv",
        batch_summary,
        [
            "batch_id",
            "M",
            "N",
            "K",
            "instances_total",
            "instances_success",
            "instances_failed",
            "exact_labeled_rows",
            "positives",
            "negatives",
            "positive_rate",
            "negative_rate",
            "wall_runtime_sec",
            "max_rss_bytes",
            "retries_used",
        ],
    )

    # Split-level summary.
    split_summary: List[Dict[str, object]] = []
    for split in ("train", "val"):
        subset = [r for r in instance_progress if str(r["split"]) == split]
        ok_subset = [r for r in subset if str(r["status"]) == "ok"]
        rows_total = sum(int(r["exact_labeled_moves"]) for r in ok_subset)
        pos_total = sum(int(r["exact_positive_improving"]) for r in ok_subset)
        neg_total = sum(int(r["exact_negative_non_improving"]) for r in ok_subset)
        split_summary.append(
            {
                "split": split,
                "manifest_instances": len(subset),
                "instances_labeled": len(ok_subset),
                "instances_failed": sum(1 for r in subset if str(r["status"]) != "ok"),
                "exact_labeled_rows": rows_total,
                "positives": pos_total,
                "negatives": neg_total,
                "positive_rate": _rate(pos_total, rows_total),
                "negative_rate": _rate(neg_total, rows_total),
                "wall_runtime_sec": round(sum(float(r["wall_runtime_sec"]) for r in subset), 6),
                "max_rss_bytes": max((_safe_int(r["max_rss_bytes"], -1) for r in subset), default=-1),
                "retries_used": sum(int(r["retries_used"]) for r in subset),
            }
        )
    _write_csv(
        output_dir / "dataset_summary_by_split.csv",
        split_summary,
        [
            "split",
            "manifest_instances",
            "instances_labeled",
            "instances_failed",
            "exact_labeled_rows",
            "positives",
            "negatives",
            "positive_rate",
            "negative_rate",
            "wall_runtime_sec",
            "max_rss_bytes",
            "retries_used",
        ],
    )

    # Bucket summary by split.
    bucket_split_summary: List[Dict[str, object]] = []
    for split in ("train", "val"):
        for m, n, k in batch_keys:
            subset = [
                r
                for r in instance_progress
                if str(r["split"]) == split and int(r["M"]) == m and int(r["N"]) == n and int(r["K"]) == k
            ]
            if not subset:
                continue
            ok_subset = [r for r in subset if str(r["status"]) == "ok"]
            rows_total = sum(int(r["exact_labeled_moves"]) for r in ok_subset)
            pos_total = sum(int(r["exact_positive_improving"]) for r in ok_subset)
            neg_total = sum(int(r["exact_negative_non_improving"]) for r in ok_subset)
            bucket_split_summary.append(
                {
                    "split": split,
                    "bucket": _bucket_id(m, n, k),
                    "M": m,
                    "N": n,
                    "K": k,
                    "instances_manifest": len(subset),
                    "instances_labeled": len(ok_subset),
                    "instances_failed": sum(1 for r in subset if str(r["status"]) != "ok"),
                    "exact_labeled_rows": rows_total,
                    "positives": pos_total,
                    "negatives": neg_total,
                    "positive_rate": _rate(pos_total, rows_total),
                    "negative_rate": _rate(neg_total, rows_total),
                    "wall_runtime_sec": round(sum(float(r["wall_runtime_sec"]) for r in subset), 6),
                    "max_rss_bytes": max((_safe_int(r["max_rss_bytes"], -1) for r in subset), default=-1),
                }
            )
    bucket_split_summary.sort(key=lambda r: (str(r["split"]), int(r["M"]), int(r["N"]), int(r["K"])))
    _write_csv(
        output_dir / "dataset_summary_by_bucket.csv",
        bucket_split_summary,
        [
            "split",
            "bucket",
            "M",
            "N",
            "K",
            "instances_manifest",
            "instances_labeled",
            "instances_failed",
            "exact_labeled_rows",
            "positives",
            "negatives",
            "positive_rate",
            "negative_rate",
            "wall_runtime_sec",
            "max_rss_bytes",
        ],
    )

    # Skew diagnosis.
    by_bucket_split: Dict[Tuple[int, int, int, str], Dict[str, float]] = {}
    for row in bucket_split_summary:
        by_bucket_split[(int(row["M"]), int(row["N"]), int(row["K"]), str(row["split"]))] = {
            "rows": float(row["exact_labeled_rows"]),
            "rate": float(row["positive_rate"]),
            "instances": float(row["instances_labeled"]),
        }

    train_rows_total = sum(v["rows"] for k, v in by_bucket_split.items() if k[3] == "train")
    val_rows_total = sum(v["rows"] for k, v in by_bucket_split.items() if k[3] == "val")

    comp_effect = 0.0
    rate_effect = 0.0
    val_with_train_weights = 0.0
    train_with_val_weights = 0.0
    for m, n, k in batch_keys:
        tr = by_bucket_split.get((m, n, k, "train"), {"rows": 0.0, "rate": 0.0})
        vr = by_bucket_split.get((m, n, k, "val"), {"rows": 0.0, "rate": 0.0})
        wt = (tr["rows"] / train_rows_total) if train_rows_total > 0 else 0.0
        wv = (vr["rows"] / val_rows_total) if val_rows_total > 0 else 0.0
        rt = tr["rate"]
        rv = vr["rate"]
        r_pool = 0.5 * (rt + rv)
        w_pool = 0.5 * (wt + wv)
        comp_effect += (wv - wt) * r_pool
        rate_effect += w_pool * (rv - rt)
        val_with_train_weights += wt * rv
        train_with_val_weights += wv * rt

    # Epsilon-policy proxy diagnostics.
    ok_instances = [r for r in instance_progress if str(r["status"]) == "ok" and int(r["exact_labeled_moves"]) > 0]
    eps_stress_all = [float(r["epsilon_stress"]) for r in ok_instances]
    pos_rate_all = [float(r["positive_rate_instance"]) for r in ok_instances]

    def _corr(xs: List[float], ys: List[float]) -> float:
        if len(xs) < 2 or len(ys) < 2:
            return 0.0
        mx = statistics.mean(xs)
        my = statistics.mean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        deny = math.sqrt(sum((y - my) ** 2 for y in ys))
        if denx <= 0.0 or deny <= 0.0:
            return 0.0
        return num / (denx * deny)

    corr_all = _corr(eps_stress_all, pos_rate_all)
    corr_by_split: Dict[str, float] = {}
    for split in ("train", "val"):
        xs = [float(r["epsilon_stress"]) for r in ok_instances if str(r["split"]) == split]
        ys = [float(r["positive_rate_instance"]) for r in ok_instances if str(r["split"]) == split]
        corr_by_split[split] = _corr(xs, ys)

    # Instance/seed variance proxy: val singleton vs train distribution per bucket.
    variance_rows: List[Dict[str, object]] = []
    outside_2std = 0
    comparable = 0
    for m, n, k in batch_keys:
        tr_inst = [
            float(r["positive_rate_instance"])
            for r in ok_instances
            if str(r["split"]) == "train" and int(r["M"]) == m and int(r["N"]) == n and int(r["K"]) == k
        ]
        va_inst = [
            float(r["positive_rate_instance"])
            for r in ok_instances
            if str(r["split"]) == "val" and int(r["M"]) == m and int(r["N"]) == n and int(r["K"]) == k
        ]
        if not tr_inst or not va_inst:
            continue
        comparable += 1
        tr_mean = statistics.mean(tr_inst)
        tr_std = statistics.pstdev(tr_inst) if len(tr_inst) > 1 else 0.0
        va_rate = va_inst[0]
        z = (va_rate - tr_mean) / tr_std if tr_std > 1e-12 else 0.0
        is_out = abs(z) > 2.0 if tr_std > 1e-12 else False
        if is_out:
            outside_2std += 1
        variance_rows.append(
            {
                "bucket": _bucket_id(m, n, k),
                "M": m,
                "N": n,
                "K": k,
                "train_instance_count": len(tr_inst),
                "val_instance_count": len(va_inst),
                "train_mean_positive_rate": tr_mean,
                "train_std_positive_rate": tr_std,
                "val_positive_rate": va_rate,
                "val_minus_train_mean": va_rate - tr_mean,
                "val_vs_train_zscore": z,
                "val_outside_train_2std": int(is_out),
            }
        )
    _write_csv(
        output_dir / "skew_seed_variance_by_bucket.csv",
        variance_rows,
        [
            "bucket",
            "M",
            "N",
            "K",
            "train_instance_count",
            "val_instance_count",
            "train_mean_positive_rate",
            "train_std_positive_rate",
            "val_positive_rate",
            "val_minus_train_mean",
            "val_vs_train_zscore",
            "val_outside_train_2std",
        ],
    )

    # Global summary and freeze metadata.
    global_rows = len(merged_rows)
    global_pos = sum(1 for r in merged_rows if str(r.get("label_improving", "0")).strip() == "1")
    global_neg = global_rows - global_pos

    split_map = {str(r["split"]): r for r in split_summary}
    train_rate = float(split_map.get("train", {}).get("positive_rate", 0.0))
    val_rate = float(split_map.get("val", {}).get("positive_rate", 0.0))

    global_summary = {
        "phase": "phaseP_full_synthetic_freeze",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifests_used": {
            "train": str(train_manifest),
            "val": str(val_manifest),
        },
        "explicitly_not_used": [
            "temp/phaseM_vls_synthetic_protocol/split_manifest_test_primary_vls.csv",
            "temp/phaseM_vls_synthetic_protocol/split_manifest_test_secondary_legacy.csv",
        ],
        "execution": {
            "manifest_instances_train": len(train_rows),
            "manifest_instances_val": len(val_rows),
            "instances_total": len(all_rows),
            "instances_success": sum(1 for r in instance_progress if str(r["status"]) == "ok"),
            "instances_failed": sum(1 for r in instance_progress if str(r["status"]) != "ok"),
            "total_retries_used": sum(int(r["retries_used"]) for r in instance_progress),
            "total_wall_runtime_sec": round(sum(float(r["wall_runtime_sec"]) for r in instance_progress), 6),
            "max_rss_bytes": max((_safe_int(r["max_rss_bytes"], -1) for r in instance_progress), default=-1),
        },
        "dataset": {
            "rows_total": global_rows,
            "positives_total": global_pos,
            "negatives_total": global_neg,
            "positive_rate": _rate(global_pos, global_rows),
            "negative_rate": _rate(global_neg, global_rows),
            "train_rows": _safe_int(split_map.get("train", {}).get("exact_labeled_rows", 0)),
            "val_rows": _safe_int(split_map.get("val", {}).get("exact_labeled_rows", 0)),
            "train_positive_rate": train_rate,
            "val_positive_rate": val_rate,
            "train_val_positive_rate_gap": val_rate - train_rate,
        },
        "skew_diagnosis": {
            "bucket_composition": {
                "row_weight_composition_effect_on_positive_rate_gap": comp_effect,
                "within_bucket_rate_effect_on_positive_rate_gap": rate_effect,
                "val_positive_rate_with_train_bucket_weights": val_with_train_weights,
                "train_positive_rate_with_val_bucket_weights": train_with_val_weights,
            },
            "epsilon_policy": {
                "epsilon_slack": args.epsilon_slack,
                "epsilon_stress_vs_instance_positive_rate_corr_overall": corr_all,
                "epsilon_stress_vs_instance_positive_rate_corr_by_split": corr_by_split,
                "train_mean_epsilon_stress": statistics.mean(
                    [float(r["epsilon_stress"]) for r in ok_instances if str(r["split"]) == "train"]
                )
                if any(str(r["split"]) == "train" for r in ok_instances)
                else 0.0,
                "val_mean_epsilon_stress": statistics.mean(
                    [float(r["epsilon_stress"]) for r in ok_instances if str(r["split"]) == "val"]
                )
                if any(str(r["split"]) == "val" for r in ok_instances)
                else 0.0,
            },
            "instance_seed_variance": {
                "comparable_buckets": comparable,
                "val_outside_train_2std_bucket_count": outside_2std,
                "val_outside_train_2std_bucket_fraction": _rate(outside_2std, comparable),
            },
        },
        "freeze_readiness": {
            "manifest_gated_full_labeling_complete": sum(1 for r in instance_progress if str(r["status"]) == "ok") == len(all_rows),
            "schema_stable": len(schema.get("columns", [])) > 0,
            "class_balance_documented": global_rows > 0,
            "skew_explained_enough": comparable > 0,
        },
    }
    (output_dir / "dataset_summary_global.json").write_text(json.dumps(global_summary, indent=2), encoding="utf-8")

    freeze_manifest = {
        "phase": "phaseP_full_synthetic_freeze",
        "timestamp_utc": global_summary["timestamp_utc"],
        "inputs": {
            "train_manifest": str(train_manifest),
            "val_manifest": str(val_manifest),
            "train_manifest_rows": len(train_rows),
            "val_manifest_rows": len(val_rows),
        },
        "labeling_policy": {
            "solver_variant": "stageO_synthetic_dense_logging",
            "epsilon_policy": "epsilon=min(K, max(ceil(sum(p)/M), max(p)) + epsilon_slack)",
            "epsilon_slack": args.epsilon_slack,
            "per_machine_dp_limit_sec": args.per_machine_dp_limit_sec,
            "ls_time_cap_sec": args.ls_time_cap_sec,
            "ls_max_rounds": args.ls_max_rounds,
            "ls_max_moves_per_round": args.ls_max_moves_per_round,
        },
        "outputs": {
            "batch_progress_csv": str(progress_path),
            "batch_summary_csv": str(output_dir / "batch_summary.csv"),
            "dataset_summary_global_json": str(output_dir / "dataset_summary_global.json"),
            "dataset_summary_by_split_csv": str(output_dir / "dataset_summary_by_split.csv"),
            "dataset_summary_by_bucket_csv": str(output_dir / "dataset_summary_by_bucket.csv"),
            "train_frozen_csv": str(train_frozen_path),
            "val_frozen_csv": str(val_frozen_path),
            "merged_frozen_csv": str(merged_frozen_path),
            "schema_json": str(output_dir / "feature_schema_frozen.json"),
        },
    }
    (output_dir / "freeze_manifest.json").write_text(json.dumps(freeze_manifest, indent=2), encoding="utf-8")

    run_config = {
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "output_dir": str(output_dir),
        "solver_bin": str(solver_bin),
        "synthetic_data_dir": str(synthetic_data_dir),
        "stageo_raw_dir": str(stageo_raw_dir),
        "epsilon_slack": args.epsilon_slack,
        "per_machine_dp_limit_sec": args.per_machine_dp_limit_sec,
        "ls_time_cap_sec": args.ls_time_cap_sec,
        "ls_max_rounds": args.ls_max_rounds,
        "ls_max_moves_per_round": args.ls_max_moves_per_round,
        "max_retries": args.max_retries,
        "resume_enabled": not args.no_resume,
        "batching_policy": "by_(M,N,K)_bucket_within_split_tracking",
    }
    (output_dir / "labeling_run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "instances_total": len(all_rows),
                "instances_success": global_summary["execution"]["instances_success"],
                "rows_total": global_rows,
                "positives": global_pos,
                "negatives": global_neg,
                "train_positive_rate": train_rate,
                "val_positive_rate": val_rate,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

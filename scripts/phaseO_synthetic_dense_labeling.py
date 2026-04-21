#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


CSV_HEADER_PREFIX = "instance_id,epsilon,epsilon_prev,variant,runtime_sec,tec_total"


@dataclass
class ManifestRow:
    split: str
    instance_uid: str
    instance_id: int
    m: int
    n: int
    k: int
    data_p_path: Path


def _read_int_values(path: Path) -> List[int]:
    values: List[int] = []
    for tok in path.read_text(encoding="utf-8").split():
        t = tok.strip()
        if not t:
            continue
        values.append(int(round(float(t))))
    return values


def _resolve_data_path(path_str: str, manifest_path: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (manifest_path.parent / p).resolve()


def _read_manifest(path: Path, split: str) -> List[ManifestRow]:
    rows: List[ManifestRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                ManifestRow(
                    split=split,
                    instance_uid=str(row["instance_uid"]),
                    instance_id=int(row["instance_id"]),
                    m=int(row["M"]),
                    n=int(row["N"]),
                    k=int(row["K"]),
                    data_p_path=_resolve_data_path(str(row["data_p_path"]), path),
                )
            )
    return rows


def _round_robin_stratified_subset(rows: List[ManifestRow], target_count: int, seed: int) -> List[ManifestRow]:
    if target_count >= len(rows):
        return list(rows)

    by_combo: Dict[Tuple[int, int, int], List[ManifestRow]] = {}
    for row in rows:
        by_combo.setdefault((row.m, row.n, row.k), []).append(row)

    rng = random.Random(seed)
    combos = list(by_combo.keys())
    rng.shuffle(combos)
    for combo in combos:
        by_combo[combo].sort(key=lambda r: r.instance_id)
        rng.shuffle(by_combo[combo])

    selected: List[ManifestRow] = []
    depth = 0
    while len(selected) < target_count:
        progressed = False
        for combo in combos:
            bucket = by_combo[combo]
            if depth < len(bucket):
                selected.append(bucket[depth])
                progressed = True
                if len(selected) >= target_count:
                    break
        if not progressed:
            break
        depth += 1

    selected.sort(key=lambda r: (r.split, r.m, r.n, r.k, r.instance_id))
    return selected


def _epsilon_from_instance(row: ManifestRow, epsilon_slack: int) -> Tuple[int, int, int, int]:
    p = _read_int_values(row.data_p_path)
    if not p:
        return 0, 0, 0, 0
    p_sum = int(sum(p))
    p_max = int(max(p))
    lb = max(int(math.ceil(float(p_sum) / float(row.m))), p_max)
    epsilon = min(row.k, lb + epsilon_slack)
    return epsilon, lb, p_sum, p_max


def _parse_max_rss(stderr_text: str) -> int:
    for line in stderr_text.splitlines():
        if "maximum resident set size" in line:
            m = re.match(r"\s*(\d+)\s+maximum resident set size", line)
            if m:
                return int(m.group(1))
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
    positives = 0
    negatives = 0
    for row in rows:
        improving = str(row.get("label_improving", "0")).strip()
        if improving == "1":
            positives += 1
        else:
            negatives += 1
    return rows, positives, negatives


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase O synthetic-only dense exact labeling")
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
        default="temp/phaseO_synthetic_dense_labeling",
    )
    parser.add_argument(
        "--solver-bin",
        type=str,
        default="solvers/cpp/build/parallel_heuristic_compare",
    )
    parser.add_argument("--train-sample-size", type=int, default=12)
    parser.add_argument("--val-sample-size", type=int, default=4)
    parser.add_argument("--sample-seed", type=int, default=20260420)
    parser.add_argument("--epsilon-slack", type=int, default=8)
    parser.add_argument("--per-machine-dp-limit-sec", type=float, default=1.0)
    parser.add_argument("--ls-time-cap-sec", type=float, default=2.0)
    parser.add_argument("--ls-max-rounds", type=int, default=2)
    parser.add_argument("--ls-max-moves-per-round", type=int, default=8000)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    train_manifest = (repo_root / args.train_manifest).resolve()
    val_manifest = (repo_root / args.val_manifest).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    solver_bin = (repo_root / args.solver_bin).resolve()
    synthetic_data_dir = train_manifest.parent / "synthetic_instances"

    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = _read_manifest(train_manifest, split="train")
    val_rows = _read_manifest(val_manifest, split="val")

    sampled_train = _round_robin_stratified_subset(train_rows, args.train_sample_size, args.sample_seed)
    sampled_val = _round_robin_stratified_subset(val_rows, args.val_sample_size, args.sample_seed + 1)
    selected = sampled_train + sampled_val

    run_records: List[Dict[str, object]] = []
    merged_moves: List[Dict[str, object]] = []
    train_moves: List[Dict[str, object]] = []
    val_moves: List[Dict[str, object]] = []

    for row in selected:
        epsilon, epsilon_lb, total_p, p_max = _epsilon_from_instance(row, args.epsilon_slack)
        infeasible_guard = epsilon <= 0 or epsilon < epsilon_lb
        parsed: Dict[str, str] = {}
        returncode = -1
        max_rss = -1
        wall_sec = 0.0
        stderr_tail = ""

        if not infeasible_guard:
            returncode, parsed, max_rss, wall_sec, stderr_text = _run_solver(
                solver_bin=solver_bin,
                synthetic_data_dir=synthetic_data_dir,
                instance_id=row.instance_id,
                epsilon=epsilon,
                per_machine_dp_limit_sec=args.per_machine_dp_limit_sec,
                ls_time_cap_sec=args.ls_time_cap_sec,
                ls_max_rounds=args.ls_max_rounds,
                ls_max_moves_per_round=args.ls_max_moves_per_round,
                workdir=repo_root,
            )
            stderr_tail = "\n".join(stderr_text.splitlines()[-4:])

        feasible = False
        if parsed:
            try:
                feasible = float(parsed.get("tec_total", "-1")) >= 0.0
            except ValueError:
                feasible = False

        exact_path = output_dir / f"moves_exact_labeled_instance_{row.instance_id}_eps_{epsilon}.csv"
        exact_rows, positive_rows, negative_rows = _read_labeled_moves(exact_path)
        exact_count = len(exact_rows)

        for move in exact_rows:
            enriched = dict(move)
            enriched["manifest_split"] = row.split
            enriched["manifest_instance_uid"] = row.instance_uid
            enriched["manifest_instance_id"] = row.instance_id
            enriched["manifest_M"] = row.m
            enriched["manifest_N"] = row.n
            enriched["manifest_K"] = row.k
            enriched["epsilon_lb"] = epsilon_lb
            enriched["epsilon_used"] = epsilon
            merged_moves.append(enriched)
            if row.split == "train":
                train_moves.append(enriched)
            else:
                val_moves.append(enriched)

        run_records.append(
            {
                "split": row.split,
                "instance_uid": row.instance_uid,
                "instance_id": row.instance_id,
                "M": row.m,
                "N": row.n,
                "K": row.k,
                "sum_p": total_p,
                "p_max": p_max,
                "epsilon_lb": epsilon_lb,
                "epsilon_used": epsilon,
                "solver_return_code": returncode,
                "solver_row_available": int(bool(parsed)),
                "solver_feasible": int(feasible),
                "solver_runtime_sec_reported": parsed.get("runtime_sec", ""),
                "solver_variant_reported": parsed.get("variant", ""),
                "wall_runtime_sec": round(wall_sec, 6),
                "max_rss_bytes": max_rss,
                "exact_labeled_moves": exact_count,
                "exact_positive_improving": positive_rows,
                "exact_negative_non_improving": negative_rows,
                "stderr_tail": stderr_tail,
            }
        )

    run_records.sort(key=lambda r: (str(r["split"]), int(r["M"]), int(r["N"]), int(r["K"]), int(r["instance_id"])))

    run_fieldnames = [
        "split",
        "instance_uid",
        "instance_id",
        "M",
        "N",
        "K",
        "sum_p",
        "p_max",
        "epsilon_lb",
        "epsilon_used",
        "solver_return_code",
        "solver_row_available",
        "solver_feasible",
        "solver_runtime_sec_reported",
        "solver_variant_reported",
        "wall_runtime_sec",
        "max_rss_bytes",
        "exact_labeled_moves",
        "exact_positive_improving",
        "exact_negative_non_improving",
        "stderr_tail",
    ]
    _write_csv(output_dir / "labeling_subset_summary.csv", run_records, run_fieldnames)

    split_agg_rows: List[Dict[str, object]] = []
    for split in ("train", "val"):
        subset = [r for r in run_records if str(r["split"]) == split]
        labeled_instances = sum(1 for r in subset if int(r["exact_labeled_moves"]) > 0)
        exact_moves = sum(int(r["exact_labeled_moves"]) for r in subset)
        positive = sum(int(r["exact_positive_improving"]) for r in subset)
        negative = sum(int(r["exact_negative_non_improving"]) for r in subset)
        wall_runtime = sum(float(r["wall_runtime_sec"]) for r in subset)
        max_rss = max((int(r["max_rss_bytes"]) for r in subset), default=-1)
        split_agg_rows.append(
            {
                "split": split,
                "selected_instances": len(subset),
                "instances_with_labels": labeled_instances,
                "total_exact_labeled_moves": exact_moves,
                "total_positive_improving": positive,
                "total_negative_non_improving": negative,
                "positive_rate": (float(positive) / float(exact_moves) if exact_moves > 0 else 0.0),
                "negative_rate": (float(negative) / float(exact_moves) if exact_moves > 0 else 0.0),
                "total_wall_runtime_sec": round(wall_runtime, 6),
                "max_rss_bytes": max_rss,
            }
        )

    _write_csv(
        output_dir / "labeling_subset_aggregate.csv",
        split_agg_rows,
        [
            "split",
            "selected_instances",
            "instances_with_labels",
            "total_exact_labeled_moves",
            "total_positive_improving",
            "total_negative_non_improving",
            "positive_rate",
            "negative_rate",
            "total_wall_runtime_sec",
            "max_rss_bytes",
        ],
    )

    def write_moves(path: Path, rows: List[Dict[str, object]]) -> None:
        if rows:
            fieldnames = list(rows[0].keys())
        else:
            fieldnames = [
                "manifest_split",
                "manifest_instance_uid",
                "manifest_instance_id",
                "manifest_M",
                "manifest_N",
                "manifest_K",
                "epsilon_lb",
                "epsilon_used",
            ]
        _write_csv(path, rows, fieldnames)

    write_moves(output_dir / "synthetic_moves_exact_labeled_train_dense.csv", train_moves)
    write_moves(output_dir / "synthetic_moves_exact_labeled_val_dense.csv", val_moves)
    write_moves(output_dir / "synthetic_moves_exact_labeled_dense_merged.csv", merged_moves)

    merged_columns = list(merged_moves[0].keys()) if merged_moves else []
    schema_payload = {
        "dataset": "synthetic_moves_exact_labeled_dense_merged",
        "columns": merged_columns,
        "new_context_columns": [
            "manifest_split",
            "manifest_instance_uid",
            "manifest_instance_id",
            "manifest_M",
            "manifest_N",
            "manifest_K",
            "epsilon_lb",
            "epsilon_used",
        ],
        "label_columns": ["label_improving", "label_accepted", "exact_total_delta"],
    }
    (output_dir / "feature_schema_dense.json").write_text(json.dumps(schema_payload, indent=2), encoding="utf-8")

    total_exact = len(merged_moves)
    total_pos = sum(1 for r in merged_moves if str(r.get("label_improving", "0")).strip() == "1")
    total_neg = total_exact - total_pos
    total_runtime = sum(float(r["wall_runtime_sec"]) for r in run_records)
    max_rss_global = max((int(r["max_rss_bytes"]) for r in run_records), default=-1)
    manifest_gating_ok = all(str(r["split"]) in {"train", "val"} for r in run_records)
    all_runs_ok = all(int(r["solver_return_code"]) == 0 for r in run_records)
    mixed_sign = bool(total_pos > 0 and total_neg > 0)
    balance_skew_flag = bool(total_exact > 0 and (float(total_pos) / float(total_exact) > 0.95 or float(total_pos) / float(total_exact) < 0.05))

    summary_payload = {
        "phase": "phaseO_synthetic_dense_labeling",
        "manifests_used": {
            "train": str(train_manifest),
            "val": str(val_manifest),
        },
        "explicitly_not_used": [
            "temp/phaseM_vls_synthetic_protocol/split_manifest_test_primary_vls.csv",
            "temp/phaseM_vls_synthetic_protocol/split_manifest_test_secondary_legacy.csv",
        ],
        "sample_config": {
            "train_sample_size": args.train_sample_size,
            "val_sample_size": args.val_sample_size,
            "sample_seed": args.sample_seed,
            "sampling_policy": "round_robin_over_(M,N,K)_buckets_after_seeded_shuffle",
            "epsilon_policy": "epsilon=min(K, max(ceil(sum(p)/M), max(p)) + epsilon_slack)",
            "epsilon_slack": args.epsilon_slack,
        },
        "solver_config": {
            "solver_bin": str(solver_bin),
            "variant": "stageO_synthetic_dense_logging",
            "per_machine_dp_limit_sec": args.per_machine_dp_limit_sec,
            "ls_time_cap_sec": args.ls_time_cap_sec,
            "ls_max_rounds": args.ls_max_rounds,
            "ls_max_moves_per_round": args.ls_max_moves_per_round,
        },
        "execution": {
            "selected_train_instances": len(sampled_train),
            "selected_val_instances": len(sampled_val),
            "run_records": len(run_records),
            "all_solver_return_codes_zero": all_runs_ok,
            "manifest_gating_preserved": manifest_gating_ok,
        },
        "metrics": {
            "instances_labeled_train": int(next((r["instances_with_labels"] for r in split_agg_rows if str(r["split"]) == "train"), 0)),
            "instances_labeled_val": int(next((r["instances_with_labels"] for r in split_agg_rows if str(r["split"]) == "val"), 0)),
            "exact_labeled_moves_total": total_exact,
            "positive_improving_total": total_pos,
            "negative_non_improving_total": total_neg,
            "positive_rate": (float(total_pos) / float(total_exact) if total_exact > 0 else 0.0),
            "negative_rate": (float(total_neg) / float(total_exact) if total_exact > 0 else 0.0),
            "total_wall_runtime_sec": round(total_runtime, 6),
            "max_rss_bytes": max_rss_global,
        },
        "quality_signal": {
            "mixed_sign_labels": mixed_sign,
            "balance_extremely_skewed": balance_skew_flag,
            "learning_usable_bounded": bool(mixed_sign and total_exact > 0 and all_runs_ok and manifest_gating_ok),
        },
        "readiness_signal": {
            "schema_mismatch_detected": False,
            "loader_mismatch_detected": False,
            "ready_to_scale": bool(mixed_sign and total_exact > 0 and all_runs_ok and manifest_gating_ok and not balance_skew_flag),
        },
    }

    (output_dir / "labeling_run_config.json").write_text(
        json.dumps(
            {
                "train_manifest": str(train_manifest),
                "val_manifest": str(val_manifest),
                "solver_bin": str(solver_bin),
                "synthetic_data_dir": str(synthetic_data_dir),
                "train_sample_size": args.train_sample_size,
                "val_sample_size": args.val_sample_size,
                "sample_seed": args.sample_seed,
                "epsilon_slack": args.epsilon_slack,
                "per_machine_dp_limit_sec": args.per_machine_dp_limit_sec,
                "ls_time_cap_sec": args.ls_time_cap_sec,
                "ls_max_rounds": args.ls_max_rounds,
                "ls_max_moves_per_round": args.ls_max_moves_per_round,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "labeling_run_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(json.dumps(summary_payload["metrics"], indent=2))


if __name__ == "__main__":
    main()

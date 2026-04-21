#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pvariance
from typing import Dict, Iterable, List


def _read_int_values(path: Path) -> List[int]:
    return [int(round(float(tok))) for tok in path.read_text(encoding="utf-8").split() if tok.strip()]


def _write_values(path: Path, values: Iterable[int]) -> None:
    path.write_text("\n".join(str(v) for v in values) + "\n", encoding="utf-8")


def _summarize_scalar(values: List[int]) -> Dict[str, float]:
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(mean(values)),
        "variance": float(pvariance(values)) if len(values) > 1 else 0.0,
    }


def _support_counter(values: List[int], lo: int, hi: int) -> Dict[int, int]:
    ctr = Counter(values)
    return {v: int(ctr.get(v, 0)) for v in range(lo, hi + 1)}


def _normalized_support(counter: Dict[int, int]) -> Dict[int, float]:
    total = sum(counter.values())
    if total <= 0:
        return {k: 0.0 for k in sorted(counter)}
    return {k: float(counter[k]) / float(total) for k in sorted(counter)}


def _tv_distance(a: Dict[int, float], b: Dict[int, float]) -> float:
    keys = sorted(set(a).union(b))
    return 0.5 * sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys)


def _catalog_fieldnames() -> List[str]:
    return [
        "instance_uid",
        "instance_id",
        "origin",
        "family",
        "role",
        "split",
        "M",
        "N",
        "K",
        "seed",
        "p_min",
        "p_max",
        "p_mean",
        "p_variance",
        "e_min",
        "e_max",
        "e_mean",
        "e_variance",
        "c_min",
        "c_max",
        "c_mean",
        "c_variance",
        "data_p_path",
        "data_e_path",
        "data_c_path",
    ]


def _synthetic_uid(m: int, n: int, k: int, seed: int) -> str:
    return f"syn_vls_m{m}_n{n}_k{k}_s{seed:03d}"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def generate_synthetic_instances(
    output_root: Path,
    m_values: List[int],
    n_values: List[int],
    k_values: List[int],
    seeds_per_combo: int,
    root_seed: int,
) -> List[Dict[str, object]]:
    synthetic_dir = output_root / "synthetic_instances"
    _ensure_dir(synthetic_dir)

    rows: List[Dict[str, object]] = []
    combo_counter = 0
    for m in sorted(m_values):
        for n in sorted(n_values):
            for k in sorted(k_values):
                combo_counter += 1
                for local_seed in range(seeds_per_combo):
                    deterministic_seed = root_seed + combo_counter * 1000 + local_seed
                    rng = random.Random(deterministic_seed)
                    p = [rng.randint(1, 12) for _ in range(n)]
                    e = [rng.randint(1, 6) for _ in range(m)]
                    c = [rng.randint(1, 8) for _ in range(k)]

                    instance_id = 100000 + len(rows) + 1
                    instance_uid = _synthetic_uid(m=m, n=n, k=k, seed=deterministic_seed)

                    p_name = f"Data_p{instance_id}.txt"
                    e_name = f"Data_e{instance_id}.txt"
                    c_name = f"Data_c{instance_id}.txt"
                    p_path = synthetic_dir / p_name
                    e_path = synthetic_dir / e_name
                    c_path = synthetic_dir / c_name

                    _write_values(p_path, p)
                    _write_values(e_path, e)
                    _write_values(c_path, c)

                    rows.append(
                        {
                            "instance_uid": instance_uid,
                            "instance_id": instance_id,
                            "origin": "synthetic_vls",
                            "family": "generated_vls",
                            "role": "train_or_val",
                            "split": "unassigned",
                            "M": m,
                            "N": n,
                            "K": k,
                            "seed": deterministic_seed,
                            "p_min": min(p),
                            "p_max": max(p),
                            "p_mean": float(mean(p)),
                            "p_variance": float(pvariance(p)) if len(p) > 1 else 0.0,
                            "e_min": min(e),
                            "e_max": max(e),
                            "e_mean": float(mean(e)),
                            "e_variance": float(pvariance(e)) if len(e) > 1 else 0.0,
                            "c_min": min(c),
                            "c_max": max(c),
                            "c_mean": float(mean(c)),
                            "c_variance": float(pvariance(c)) if len(c) > 1 else 0.0,
                            "data_p_path": str(p_path.relative_to(output_root)),
                            "data_e_path": str(e_path.relative_to(output_root)),
                            "data_c_path": str(c_path.relative_to(output_root)),
                        }
                    )
    return rows


def assign_train_val_splits(rows: List[Dict[str, object]], train_fraction: float, split_seed: int) -> None:
    by_combo: Dict[tuple, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_combo[(row["M"], row["N"], row["K"])].append(row)

    rng = random.Random(split_seed)
    for _, group in by_combo.items():
        group.sort(key=lambda r: int(r["seed"]))
        rng.shuffle(group)
        n_train = max(1, int(round(len(group) * train_fraction)))
        n_train = min(n_train, len(group) - 1) if len(group) > 1 else 1
        train_ids = {int(r["instance_id"]) for r in group[:n_train]}
        for row in group:
            row["split"] = "train" if int(row["instance_id"]) in train_ids else "val"


def write_catalog(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_catalog_fieldnames())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_family_summary(rows: List[Dict[str, object]], family_name: str, output_path: Path) -> None:
    m_vals = [int(r["M"]) for r in rows]
    n_vals = [int(r["N"]) for r in rows]
    k_vals = [int(r["K"]) for r in rows]

    p_all: List[int] = []
    e_all: List[int] = []
    c_all: List[int] = []
    for row in rows:
        p_all.extend(_read_int_values(output_path.parent / str(row["data_p_path"])))
        e_all.extend(_read_int_values(output_path.parent / str(row["data_e_path"])))
        c_all.extend(_read_int_values(output_path.parent / str(row["data_c_path"])))

    m_sum = _summarize_scalar(m_vals)
    n_sum = _summarize_scalar(n_vals)
    k_sum = _summarize_scalar(k_vals)
    p_sum = _summarize_scalar(p_all)
    e_sum = _summarize_scalar(e_all)
    c_sum = _summarize_scalar(c_all)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["family", "metric", "min", "max", "mean", "variance", "n_instances"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for metric, stat in [
            ("M", m_sum),
            ("N", n_sum),
            ("K", k_sum),
            ("p", p_sum),
            ("e", e_sum),
            ("c", c_sum),
        ]:
            writer.writerow(
                {
                    "family": family_name,
                    "metric": metric,
                    "min": stat["min"],
                    "max": stat["max"],
                    "mean": stat["mean"],
                    "variance": stat["variance"],
                    "n_instances": len(rows),
                }
            )


def write_synthetic_combo_summary(rows: List[Dict[str, object]], output_path: Path) -> None:
    combo_ctr = Counter((int(r["M"]), int(r["N"]), int(r["K"])) for r in rows)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["M", "N", "K", "instances"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (m, n, k), cnt in sorted(combo_ctr.items()):
            writer.writerow({"M": m, "N": n, "K": k, "instances": cnt})


def load_benchmark_rows(benchmark_dir: Path, ids: Iterable[int], family: str, role: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for iid in ids:
        p = _read_int_values(benchmark_dir / f"Data_p{iid}.txt")
        e = _read_int_values(benchmark_dir / f"Data_e{iid}.txt")
        c = _read_int_values(benchmark_dir / f"Data_c{iid}.txt")
        rows.append(
            {
                "instance_uid": f"benchmark_{iid}",
                "instance_id": iid,
                "origin": "benchmark",
                "family": family,
                "role": role,
                "split": role,
                "M": len(e),
                "N": len(p),
                "K": len(c),
                "seed": "",
                "p_min": min(p),
                "p_max": max(p),
                "p_mean": float(mean(p)),
                "p_variance": float(pvariance(p)) if len(p) > 1 else 0.0,
                "e_min": min(e),
                "e_max": max(e),
                "e_mean": float(mean(e)),
                "e_variance": float(pvariance(e)) if len(e) > 1 else 0.0,
                "c_min": min(c),
                "c_max": max(c),
                "c_mean": float(mean(c)),
                "c_variance": float(pvariance(c)) if len(c) > 1 else 0.0,
                "data_p_path": str((benchmark_dir / f"Data_p{iid}.txt").resolve()),
                "data_e_path": str((benchmark_dir / f"Data_e{iid}.txt").resolve()),
                "data_c_path": str((benchmark_dir / f"Data_c{iid}.txt").resolve()),
            }
        )
    return rows


def write_split_manifest(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = _catalog_fieldnames()
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _collect_raw_values(rows: List[Dict[str, object]], base_dir: Path) -> Dict[str, List[int]]:
    p_all: List[int] = []
    e_all: List[int] = []
    c_all: List[int] = []
    for row in rows:
        p_path = Path(str(row["data_p_path"]))
        e_path = Path(str(row["data_e_path"]))
        c_path = Path(str(row["data_c_path"]))
        if not p_path.is_absolute():
            p_path = (base_dir / p_path).resolve()
        if not e_path.is_absolute():
            e_path = (base_dir / e_path).resolve()
        if not c_path.is_absolute():
            c_path = (base_dir / c_path).resolve()
        p_all.extend(_read_int_values(p_path))
        e_all.extend(_read_int_values(e_path))
        c_all.extend(_read_int_values(c_path))
    return {"p": p_all, "e": e_all, "c": c_all}


def write_generated_vs_benchmark_comparison(
    synthetic_rows: List[Dict[str, object]],
    benchmark_vls_rows: List[Dict[str, object]],
    output_path: Path,
    support_path: Path,
) -> None:
    base_dir = output_path.parent
    syn_vals = _collect_raw_values(synthetic_rows, base_dir)
    bench_vals = _collect_raw_values(benchmark_vls_rows, base_dir)

    syn_m = [int(r["M"]) for r in synthetic_rows]
    syn_n = [int(r["N"]) for r in synthetic_rows]
    syn_k = [int(r["K"]) for r in synthetic_rows]
    ben_m = [int(r["M"]) for r in benchmark_vls_rows]
    ben_n = [int(r["N"]) for r in benchmark_vls_rows]
    ben_k = [int(r["K"]) for r in benchmark_vls_rows]

    dimension_pairs = [
        ("M", syn_m, ben_m, None),
        ("N", syn_n, ben_n, None),
        ("K", syn_k, ben_k, None),
        ("p", syn_vals["p"], bench_vals["p"], (1, 12)),
        ("e", syn_vals["e"], bench_vals["e"], (1, 6)),
        ("c", syn_vals["c"], bench_vals["c"], (1, 8)),
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "dimension",
            "synthetic_min",
            "synthetic_max",
            "synthetic_mean",
            "synthetic_variance",
            "benchmark_min",
            "benchmark_max",
            "benchmark_mean",
            "benchmark_variance",
            "mean_delta_synthetic_minus_benchmark",
            "variance_delta_synthetic_minus_benchmark",
            "total_variation_distance",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        support_rows: List[Dict[str, object]] = []
        for name, syn_list, ben_list, support_range in dimension_pairs:
            syn_sum = _summarize_scalar(syn_list)
            ben_sum = _summarize_scalar(ben_list)
            tv = ""
            if support_range is not None:
                lo, hi = support_range
                syn_ctr = _support_counter(syn_list, lo, hi)
                ben_ctr = _support_counter(ben_list, lo, hi)
                syn_norm = _normalized_support(syn_ctr)
                ben_norm = _normalized_support(ben_ctr)
                tv_val = _tv_distance(syn_norm, ben_norm)
                tv = f"{tv_val:.8f}"
                for v in range(lo, hi + 1):
                    support_rows.append(
                        {
                            "dimension": name,
                            "value": v,
                            "synthetic_count": syn_ctr[v],
                            "benchmark_count": ben_ctr[v],
                            "synthetic_fraction": syn_norm[v],
                            "benchmark_fraction": ben_norm[v],
                        }
                    )
            writer.writerow(
                {
                    "dimension": name,
                    "synthetic_min": syn_sum["min"],
                    "synthetic_max": syn_sum["max"],
                    "synthetic_mean": syn_sum["mean"],
                    "synthetic_variance": syn_sum["variance"],
                    "benchmark_min": ben_sum["min"],
                    "benchmark_max": ben_sum["max"],
                    "benchmark_mean": ben_sum["mean"],
                    "benchmark_variance": ben_sum["variance"],
                    "mean_delta_synthetic_minus_benchmark": syn_sum["mean"] - ben_sum["mean"],
                    "variance_delta_synthetic_minus_benchmark": syn_sum["variance"] - ben_sum["variance"],
                    "total_variation_distance": tv,
                }
            )

    with support_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "dimension",
            "value",
            "synthetic_count",
            "benchmark_count",
            "synthetic_fraction",
            "benchmark_fraction",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in support_rows:
            writer.writerow(row)


def write_generation_config(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase M VLS synthetic protocol generator")
    parser.add_argument("--output-dir", type=str, default="temp/phaseM_vls_synthetic_protocol")
    parser.add_argument("--benchmark-dir", type=str, default="temp/paper_exact_repo/instances")
    parser.add_argument("--root-seed", type=int, default=20260420)
    parser.add_argument("--split-seed", type=int, default=20260421)
    parser.add_argument("--seeds-per-combo", type=int, default=6)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    args = parser.parse_args()

    output_root = Path(args.output_dir).resolve()
    benchmark_dir = Path(args.benchmark_dir).resolve()
    _ensure_dir(output_root)

    m_values = [25, 30, 40]
    n_values = [250, 300, 350, 400, 500]
    k_values = [350, 500]

    synthetic_rows = generate_synthetic_instances(
        output_root=output_root,
        m_values=m_values,
        n_values=n_values,
        k_values=k_values,
        seeds_per_combo=int(args.seeds_per_combo),
        root_seed=int(args.root_seed),
    )

    assign_train_val_splits(
        rows=synthetic_rows,
        train_fraction=float(args.train_fraction),
        split_seed=int(args.split_seed),
    )

    benchmark_primary_rows = load_benchmark_rows(
        benchmark_dir=benchmark_dir,
        ids=range(61, 91),
        family="benchmark_61_90_vls",
        role="test_primary_vls",
    )
    benchmark_secondary_rows = load_benchmark_rows(
        benchmark_dir=benchmark_dir,
        ids=range(1, 61),
        family="benchmark_1_60_legacy",
        role="test_secondary_legacy",
    )

    write_catalog(output_root / "synthetic_instance_catalog.csv", synthetic_rows)
    write_catalog(output_root / "benchmark_instance_catalog.csv", benchmark_primary_rows + benchmark_secondary_rows)

    write_synthetic_combo_summary(synthetic_rows, output_root / "synthetic_family_summary.csv")
    build_family_summary(synthetic_rows, "synthetic_vls", output_root / "synthetic_family_stats.csv")
    build_family_summary(benchmark_primary_rows, "benchmark_61_90_vls", output_root / "benchmark_vls_summary.csv")
    build_family_summary(benchmark_secondary_rows, "benchmark_1_60_legacy", output_root / "benchmark_legacy_summary.csv")
    with (output_root / "benchmark_family_summary.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["family", "instances", "M_min", "M_max", "N_min", "N_max", "K_min", "K_max"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for family_name, fam_rows in [
            ("benchmark_61_90_vls", benchmark_primary_rows),
            ("benchmark_1_60_legacy", benchmark_secondary_rows),
        ]:
            m_vals = [int(r["M"]) for r in fam_rows]
            n_vals = [int(r["N"]) for r in fam_rows]
            k_vals = [int(r["K"]) for r in fam_rows]
            writer.writerow(
                {
                    "family": family_name,
                    "instances": len(fam_rows),
                    "M_min": min(m_vals),
                    "M_max": max(m_vals),
                    "N_min": min(n_vals),
                    "N_max": max(n_vals),
                    "K_min": min(k_vals),
                    "K_max": max(k_vals),
                }
            )

    write_generated_vs_benchmark_comparison(
        synthetic_rows=synthetic_rows,
        benchmark_vls_rows=benchmark_primary_rows,
        output_path=output_root / "generated_vs_benchmark_vls_comparison.csv",
        support_path=output_root / "generated_vs_benchmark_vls_support_counts.csv",
    )

    train_rows = [r for r in synthetic_rows if str(r["split"]) == "train"]
    val_rows = [r for r in synthetic_rows if str(r["split"]) == "val"]
    write_split_manifest(output_root / "split_manifest_train.csv", train_rows)
    write_split_manifest(output_root / "split_manifest_val.csv", val_rows)
    write_split_manifest(output_root / "split_manifest_test_primary_vls.csv", benchmark_primary_rows)
    write_split_manifest(output_root / "split_manifest_test_secondary_legacy.csv", benchmark_secondary_rows)

    write_generation_config(
        output_root / "synthetic_generation_config.json",
        {
            "phase": "phaseM_vls_synthetic_protocol",
            "description": "Synthetic VLS-only corpus for clean train/val, with benchmark-only test manifests.",
            "training_data_policy": "generated_only",
            "validation_data_policy": "generated_only",
            "primary_test_policy": "benchmark_61_90_only",
            "secondary_test_policy": "benchmark_1_60_only_ood_legacy",
            "generator_family": {
                "M_values": m_values,
                "N_values": n_values,
                "K_values": k_values,
                "p_distribution": "discrete_uniform[1,12]",
                "e_distribution": "discrete_uniform[1,6]",
                "c_distribution": "discrete_uniform[1,8]",
            },
            "seeding_scheme": {
                "root_seed": int(args.root_seed),
                "split_seed": int(args.split_seed),
                "instance_seed_formula": "root_seed + combo_index*1000 + local_seed",
                "seeds_per_combo": int(args.seeds_per_combo),
            },
            "pilot_corpus": {
                "n_combinations": len(m_values) * len(n_values) * len(k_values),
                "instances_total": len(synthetic_rows),
                "instances_train": len(train_rows),
                "instances_val": len(val_rows),
            },
            "benchmark_manifests": {
                "primary_vls_count": len(benchmark_primary_rows),
                "secondary_legacy_count": len(benchmark_secondary_rows),
                "benchmark_source_dir": str(benchmark_dir),
            },
        },
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_root),
                "synthetic_total": len(synthetic_rows),
                "train": len(train_rows),
                "val": len(val_rows),
                "primary_test": len(benchmark_primary_rows),
                "secondary_test": len(benchmark_secondary_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

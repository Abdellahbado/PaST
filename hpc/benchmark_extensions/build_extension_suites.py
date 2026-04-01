#!/usr/bin/env python3
"""
Build the formal benchmark-extension suites used in the paper-facing study.

This script creates three explicit suites:

1. scalability_large_n
   The paper-hard {8,10} family with larger n.
2. backup_realistic
   Small realistic bounded-count rescue/control suite for semigroup vs R_feas.
3. k_boundary
   Increasing-K family to probe where certification/exact fallback starts
   becoming the bottleneck.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "green-scheduling-bab" / "Iirc.EnergyStatesAndCostsScheduling" / "data" / "datasets"

FORMAL_SUITES = {
    "scalability_large_n": DATA / "paperext_scalability_large_n_202604",
    "backup_realistic": DATA / "paperext_backup_realistic_202604",
    "k_boundary": DATA / "paperext_k_boundary_202604",
}

STRESS_V3 = DATA / "stress_extended_202603_v3"
BACKUP_SHOWCASE = DATA / "backup_feas_showcase_202604"
K_BOUNDARY = DATA / "k_boundary_202604"


def run_py(script: str, *args: str) -> None:
    cmd = [sys.executable, str(ROOT / script), *args]
    subprocess.run(cmd, check=True)


def reset_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.json"):
        old.unlink()
    (out_dir / "manifest.json").unlink(missing_ok=True)


def copy_suite(src_dir: Path, out_dir: Path, filter_fn, suite_name: str, description: str) -> None:
    manifest_src = {}
    manifest_path = src_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            rows = json.load(f)
        manifest_src = {row["file"]: row for row in rows}

    reset_out_dir(out_dir)
    manifest_rows = []
    for src in sorted(src_dir.glob("*.json")):
        if src.name == "manifest.json":
            continue
        meta = manifest_src.get(src.name, {})
        if not filter_fn(src, meta):
            continue
        shutil.copy2(src, out_dir / src.name)
        manifest_rows.append(
            {
                "file": src.name,
                "source_dataset": src_dir.name,
                "suite": suite_name,
                "description": description,
                "source_family": meta.get("family", ""),
                "metadata": meta.get("metadata", meta),
            }
        )

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest_rows, f, indent=2)

    print(f"{suite_name}: wrote {len(manifest_rows)} instances to {out_dir}")


def ensure_sources() -> None:
    reset_out_dir(STRESS_V3)
    run_py(
        "scripts/generate_extended_stress_benchmark.py",
        "--out-dir",
        str(STRESS_V3),
        "--seeds-per-case",
        "3",
    )
    if not (DATA / "backup_packability_202603").exists():
        run_py("scripts/build_backup_packability_suite.py")
    run_py("scripts/build_backup_feas_showcase_suite.py")
    run_py("scripts/generate_k_boundary_benchmark.py")


def build_scalability_suite() -> None:
    copy_suite(
        STRESS_V3,
        FORMAL_SUITES["scalability_large_n"],
        lambda src, meta: meta.get("family") == "A_nscale_8_10" or "_famA_nscale_" in src.stem,
        "scalability_large_n",
        "Large-n extension of the paper-hard {8,10} family; only n is scaled.",
    )


def build_backup_suite() -> None:
    copy_suite(
        BACKUP_SHOWCASE,
        FORMAL_SUITES["backup_realistic"],
        lambda src, meta: True,
        "backup_realistic",
        "Small realistic bounded-count showcase where semigroup is sometimes enough and sometimes repaired by R_feas.",
    )


def build_k_boundary_suite() -> None:
    copy_suite(
        K_BOUNDARY,
        FORMAL_SUITES["k_boundary"],
        lambda src, meta: True,
        "k_boundary",
        "Increasing-K realistic families used to probe the structural boundary of the method.",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build formal benchmark-extension suites")
    ap.add_argument(
        "--suite",
        choices=["all", "scalability_large_n", "backup_realistic", "k_boundary"],
        default="all",
    )
    args = ap.parse_args()

    ensure_sources()

    if args.suite in ("all", "scalability_large_n"):
        build_scalability_suite()
    if args.suite in ("all", "backup_realistic"):
        build_backup_suite()
    if args.suite in ("all", "k_boundary"):
        build_k_boundary_suite()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class InstanceRow:
    run_dir: str
    size: str
    instance_id: int
    n_jobs: int
    T: int
    seed: int

    dp_cost: float
    dp_ms: float
    dp_tle: bool

    bnb_cost: float
    bnb_ms: float
    bnb_tle: bool
    bnb_nodes: int

    cost_match: bool

    @property
    def comparable(self) -> bool:
        return (not self.dp_tle) and (not self.bnb_tle) and math.isfinite(self.dp_cost) and math.isfinite(self.bnb_cost)

    @property
    def cost_gap(self) -> float:
        # Positive => BnB worse (higher cost). Negative => BnB better.
        if not self.comparable:
            return float("nan")
        return float(self.bnb_cost - self.dp_cost)


def _as_float(x: Any) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, str) and x.lower() == "inf":
        return float("inf")
    try:
        return float(x)
    except Exception:
        return float("nan")


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _as_bool(x: Any) -> bool:
    return bool(x)


def _load_instance(path: Path) -> InstanceRow:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    dp = data.get("dp", {}) or {}
    bnb = data.get("bnb", {}) or {}

    # Run directory is the parent folder name (e.g. vls_s11)
    run_dir = str(path.parent.name)

    return InstanceRow(
        run_dir=run_dir,
        size=str(data.get("size", "?")),
        instance_id=_as_int(data.get("instance_id", -1)),
        n_jobs=_as_int(data.get("n_jobs", -1)),
        T=_as_int(data.get("T", -1)),
        seed=_as_int(data.get("seed", -1)),
        dp_cost=_as_float(dp.get("cost", float("inf"))),
        dp_ms=_as_float(dp.get("time_ms", float("nan"))),
        dp_tle=_as_bool(dp.get("timed_out", False)),
        bnb_cost=_as_float(bnb.get("cost", float("inf"))),
        bnb_ms=_as_float(bnb.get("time_ms", float("nan"))),
        bnb_tle=_as_bool(bnb.get("timed_out", False)),
        bnb_nodes=_as_int(bnb.get("nodes", 0)),
        cost_match=_as_bool(data.get("cost_match", False)),
    )


def _iter_instance_jsons(root: Path) -> Iterable[Path]:
    if root.is_file() and root.suffix == ".json":
        yield root
        return
    if not root.exists():
        return
    if root.is_dir():
        # crawl all runs
        yield from sorted(root.rglob("instance_*.json"))


def _mean(xs: List[float]) -> float:
    vals = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _median(xs: List[float]) -> float:
    vals = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    if not vals:
        return float("nan")
    vals.sort()
    m = len(vals)
    if m % 2 == 1:
        return float(vals[m // 2])
    return float(0.5 * (vals[m // 2 - 1] + vals[m // 2]))


def _write_detailed_csv(path: Path, rows: List[InstanceRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "run_dir",
                "size",
                "instance_id",
                "n_jobs",
                "T",
                "seed",
                "dp_cost",
                "dp_ms",
                "dp_tle",
                "bnb_cost",
                "bnb_ms",
                "bnb_tle",
                "bnb_nodes",
                "cost_match",
                "comparable",
                "cost_gap_bnb_minus_dp",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.run_dir,
                    r.size,
                    r.instance_id,
                    r.n_jobs,
                    r.T,
                    r.seed,
                    r.dp_cost,
                    r.dp_ms,
                    int(r.dp_tle),
                    r.bnb_cost,
                    r.bnb_ms,
                    int(r.bnb_tle),
                    r.bnb_nodes,
                    int(r.cost_match),
                    int(r.comparable),
                    r.cost_gap,
                ]
            )


def _write_group_summary_csv(path: Path, group_key: str, groups: Dict[str, List[InstanceRow]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                group_key,
                "n_instances",
                "dp_tle_count",
                "bnb_tle_count",
                "both_ok_count",
                "match_count",
                "mismatch_count",
                "dp_ms_mean",
                "dp_ms_median",
                "bnb_ms_mean",
                "bnb_ms_median",
                "bnb_nodes_mean",
                "bnb_nodes_median",
                "gap_mean_bnb_minus_dp",
                "gap_median_bnb_minus_dp",
                "gap_min_bnb_minus_dp",
                "gap_max_bnb_minus_dp",
            ]
        )

        for k in sorted(groups.keys()):
            rs = groups[k]
            n = len(rs)
            dp_tle = sum(1 for r in rs if r.dp_tle)
            bnb_tle = sum(1 for r in rs if r.bnb_tle)
            both_ok = [r for r in rs if r.comparable]
            match = sum(1 for r in both_ok if r.cost_match)
            mismatch = sum(1 for r in both_ok if not r.cost_match)

            dp_ms = [r.dp_ms for r in rs]
            bnb_ms = [r.bnb_ms for r in rs]
            nodes = [float(r.bnb_nodes) for r in rs if r.bnb_nodes is not None]
            gaps = [r.cost_gap for r in both_ok]

            gap_min = float("nan")
            gap_max = float("nan")
            gvals = [g for g in gaps if math.isfinite(g)]
            if gvals:
                gap_min = float(min(gvals))
                gap_max = float(max(gvals))

            w.writerow(
                [
                    k,
                    n,
                    dp_tle,
                    bnb_tle,
                    len(both_ok),
                    match,
                    mismatch,
                    _mean(dp_ms),
                    _median(dp_ms),
                    _mean(bnb_ms),
                    _median(bnb_ms),
                    _mean(nodes),
                    _median(nodes),
                    _mean(gaps),
                    _median(gaps),
                    gap_min,
                    gap_max,
                ]
            )


def _print_console_summary(rows: List[InstanceRow]) -> None:
    if not rows:
        print("No instance_*.json found.")
        return

    by_size: Dict[str, List[InstanceRow]] = {}
    for r in rows:
        by_size.setdefault(r.size, []).append(r)

    print("RESULTS SUMMARY")
    print("--------------")
    for size in sorted(by_size.keys()):
        rs = by_size[size]
        n = len(rs)
        dp_tle = sum(1 for r in rs if r.dp_tle)
        bnb_tle = sum(1 for r in rs if r.bnb_tle)
        ok = [r for r in rs if r.comparable]
        match = sum(1 for r in ok if r.cost_match)
        mismatch = sum(1 for r in ok if not r.cost_match)
        gaps = [r.cost_gap for r in ok]
        print(f"size={size}  n={n}  dp_tle={dp_tle}  bnb_tle={bnb_tle}  compared={len(ok)}  match={match}  mismatch={mismatch}")
        if ok:
            print(f"  gap(bnb-dp): mean={_mean(gaps):.4f}  median={_median(gaps):.4f}  min={min(gaps):.4f}  max={max(gaps):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize C++ compare.cpp results folders")
    ap.add_argument(
        "path",
        nargs="?",
        default=str(Path(__file__).resolve().parent / "build" / "results"),
        help="Results root directory (default: solvers/cpp/build/results) or a single instance_*.json",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for summary CSVs (default: <path>/__summary if path is dir)",
    )
    args = ap.parse_args()

    root = Path(args.path).resolve()

    if args.out_dir is None:
        if root.is_dir():
            out_dir = root / "__summary"
        else:
            out_dir = root.parent / "__summary"
    else:
        out_dir = Path(args.out_dir).resolve()

    paths = list(_iter_instance_jsons(root))
    rows: List[InstanceRow] = []
    for p in paths:
        try:
            rows.append(_load_instance(p))
        except Exception as e:
            print(f"SKIP {p}: {e}")

    rows.sort(key=lambda r: (r.size, r.run_dir, r.instance_id))

    _print_console_summary(rows)

    detailed_csv = out_dir / "detailed_instances.csv"
    _write_detailed_csv(detailed_csv, rows)

    by_run: Dict[str, List[InstanceRow]] = {}
    by_size: Dict[str, List[InstanceRow]] = {}
    for r in rows:
        by_run.setdefault(r.run_dir, []).append(r)
        by_size.setdefault(r.size, []).append(r)

    _write_group_summary_csv(out_dir / "summary_by_run.csv", "run_dir", by_run)
    _write_group_summary_csv(out_dir / "summary_by_size.csv", "size", by_size)

    print("")
    print(f"Wrote: {detailed_csv}")
    print(f"Wrote: {out_dir / 'summary_by_run.csv'}")
    print(f"Wrote: {out_dir / 'summary_by_size.csv'}")


if __name__ == "__main__":
    main()

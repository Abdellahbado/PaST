#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_import_path() -> None:
    root = _repo_root_from_here()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _load_instance_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _segments_to_occupied(segments: List[List[int]]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for seg in segments:
        if not isinstance(seg, list) or len(seg) != 2:
            raise ValueError(f"Invalid segment: {seg}")
        s, L = int(seg[0]), int(seg[1])
        out.append((s, s + L))
    return out


def _check_no_overlap(intervals: List[Tuple[int, int]]) -> bool:
    if not intervals:
        return True
    ints = sorted(intervals)
    prev_s, prev_e = ints[0]
    for s, e in ints[1:]:
        if s < prev_e:
            return False
        prev_s, prev_e = s, e
    return True


def _cost_from_intervals(prices: np.ndarray, intervals: List[Tuple[int, int]]) -> float:
    if len(prices) == 0:
        return 0.0
    prefix = np.zeros(len(prices) + 1, dtype=np.float64)
    prefix[1:] = np.cumsum(prices, dtype=np.float64)
    total = 0.0
    for s, e in intervals:
        total += float(prefix[e] - prefix[s])
    return float(total)


def verify_one(path: Path, *, tol: float, tie_break: str) -> Tuple[bool, str]:
    data = _load_instance_json(path)

    T = int(data["T"])
    proc_times = [int(x) for x in data["proc_times"]]
    prices = np.asarray(data["prices"], dtype=np.float64)
    if len(prices) != T:
        return False, f"{path.name}: prices length {len(prices)} != T {T}"

    dp = data.get("dp", {})
    dp_cost_reported = float(dp.get("cost", float("inf")))
    dp_tle = bool(dp.get("timed_out", False))
    dp_segments = dp.get("segments", [])

    if dp_tle:
        return False, f"{path.name}: DP timed_out=true; cannot verify optimality"

    if not isinstance(dp_segments, list):
        return False, f"{path.name}: dp.segments not a list"

    intervals = _segments_to_occupied(dp_segments)

    for s, e in intervals:
        if s < 0 or e < 0 or e < s:
            return False, f"{path.name}: invalid segment [{s},{e})"
        if e > T:
            return False, f"{path.name}: segment [{s},{e}) exceeds T={T}"

    if sum(proc_times) > T:
        return False, f"{path.name}: infeasible by sum(proc_times)={sum(proc_times)} > T={T}"

    total_len = sum(e - s for s, e in intervals)
    if total_len != sum(proc_times):
        return False, (
            f"{path.name}: total scheduled length {total_len} != sum(proc_times) {sum(proc_times)}"
        )

    if not _check_no_overlap(intervals):
        return False, f"{path.name}: DP segments overlap"

    cost_from_segments = _cost_from_intervals(prices, intervals)
    if not np.isfinite(cost_from_segments):
        return False, f"{path.name}: recomputed segment cost is not finite"

    if abs(cost_from_segments - dp_cost_reported) > tol:
        return False, (
            f"{path.name}: recomputed cost {cost_from_segments:.6f} != reported {dp_cost_reported:.6f}"
        )

    _ensure_import_path()
    from PaST.solvers.optimal_benchmark_dp import solve_optimal_benchmark_dp

    res = solve_optimal_benchmark_dp(proc_times, prices, tie_break=tie_break, track_schedule=False)
    if not res.feasible:
        return False, f"{path.name}: python solver says infeasible (unexpected)"

    if abs(float(res.cost) - dp_cost_reported) > tol:
        return False, (
            f"{path.name}: python optimal cost {float(res.cost):.6f} != dp reported {dp_cost_reported:.6f}"
        )

    return True, f"{path.name}: OK (cost={dp_cost_reported:.6f})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=str, help="Results dir or a single instance_XXXX.json")
    ap.add_argument("--ids", type=int, nargs="*", default=None)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--tie-break", type=str, default="early", choices=["cost", "early"])
    args = ap.parse_args()

    p = Path(args.path)
    if p.is_file():
        files = [p]
    else:
        if not p.is_dir():
            raise SystemExit(f"Path not found: {p}")
        files = sorted(p.glob("instance_*.json"))
        if args.ids is not None:
            wanted = {int(x) for x in args.ids}
            files = [f for f in files if int(f.stem.split("_")[-1]) in wanted]

    if not files:
        raise SystemExit(f"No instance_*.json found at: {p}")

    ok_count = 0
    for f in files:
        ok, msg = verify_one(f, tol=float(args.tol), tie_break=str(args.tie_break))
        print(msg)
        if ok:
            ok_count += 1

    total = len(files)
    if ok_count != total:
        raise SystemExit(f"FAILED: {ok_count}/{total} verified")
    print(f"SUCCESS: {ok_count}/{total} verified")


if __name__ == "__main__":
    main()

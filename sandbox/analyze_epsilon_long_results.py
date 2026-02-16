#!/usr/bin/env python3
"""Analyze pooled-Vhat and epsilon-constraint simulation CSV outputs.

Designed to be run after the command chain that writes files into a folder like:
PaST/ADP/logs/epsilon_long/logs/

This script avoids heavy dependencies; it only uses Python stdlib.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics as st
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _to_int(x: str) -> int:
    return int(float(x))


def _to_float(x: str) -> float:
    return float(x)


def _quantile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs_sorted = sorted(xs)
    if len(xs_sorted) == 1:
        return xs_sorted[0]
    idx = int(q * (len(xs_sorted) - 1))
    return xs_sorted[idx]


def _mean(xs: Sequence[float]) -> float:
    return st.fmean(xs) if xs else float("nan")


def _median(xs: Sequence[float]) -> float:
    return st.median(xs) if xs else float("nan")


def _pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx = _mean(xs)
    my = _mean(ys)
    num = 0.0
    vx = 0.0
    vy = 0.0
    for x, y in zip(xs, ys):
        dx = x - mx
        dy = y - my
        num += dx * dy
        vx += dx * dx
        vy += dy * dy
    den = math.sqrt(vx * vy)
    return num / den if den > 0 else float("nan")


def _parse_semicolon_ints(s: str) -> List[int]:
    s = str(s).strip()
    if not s:
        return []
    return [int(x) for x in s.split(";") if x != ""]


@dataclass(frozen=True)
class PooledBeamStats:
    n: int
    gap_mean: float
    gap_median: float
    gap_p90: float
    speed_mean: float
    speed_median: float


def summarize_pooled(path: Path) -> Tuple[Dict[int, PooledBeamStats], PooledBeamStats]:
    rows = _read_csv_rows(path)
    beams = sorted({_to_int(r["beam"]) for r in rows})

    by_beam: Dict[int, PooledBeamStats] = {}
    for b in beams:
        rb = [r for r in rows if _to_int(r["beam"]) == b]
        gaps = [_to_float(r["gap_learned_pct"]) for r in rb]
        speeds = [_to_float(r["speedup_learned"]) for r in rb]
        by_beam[b] = PooledBeamStats(
            n=len(rb),
            gap_mean=_mean(gaps),
            gap_median=_median(gaps),
            gap_p90=_quantile(gaps, 0.90),
            speed_mean=_mean(speeds),
            speed_median=_median(speeds),
        )

    all_gaps = [_to_float(r["gap_learned_pct"]) for r in rows]
    all_speeds = [_to_float(r["speedup_learned"]) for r in rows]
    overall = PooledBeamStats(
        n=len(rows),
        gap_mean=_mean(all_gaps),
        gap_median=_median(all_gaps),
        gap_p90=_quantile(all_gaps, 0.90),
        speed_mean=_mean(all_speeds),
        speed_median=_median(all_speeds),
    )
    return by_beam, overall


@dataclass(frozen=True)
class EpsilonSummary:
    rows: int
    instances: int
    n_range: Tuple[int, int]
    d_range: Tuple[int, int]
    eps_range: Tuple[int, int]
    solve_mean: float
    solve_median: float
    solve_p90: float
    energy_mean: float
    energy_median: float
    energy_p90: float
    makespan_mean: float
    infeasible_rows: int
    nonmono_instances: int
    steps_mean: float
    steps_median: float
    final_slack_mean: float
    final_tight_frac: float
    assign_corr_u_load_mean: float
    assign_corr_u_load_p25: float
    assign_corr_u_load_p75: float
    cheapest_load_share_mean: float


def summarize_epsilon(path: Path) -> EpsilonSummary:
    rows = _read_csv_rows(path)

    Ns = [_to_int(r["N"]) for r in rows]
    Ds = [_to_int(r["D"]) for r in rows]
    eps = [_to_int(r["epsilon"]) for r in rows]
    solve = [_to_float(r["solve_s"]) for r in rows]
    energy = [_to_float(r["total_energy"]) for r in rows]
    makes = [_to_float(r["makespan"]) for r in rows]

    infeasible = sum(
        1 for r in rows if _to_float(r["makespan"]) - _to_float(r["epsilon"]) > 1e-9
    )

    by_inst: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_inst[_to_int(r["instance_seed"])].append(r)

    nonmono = 0
    steps = []
    final_slack = []

    # Assignment-bias sanity metrics
    corr_u_load = []
    cheapest_share = []

    for _, rr in by_inst.items():
        rr_sorted = sorted(rr, key=lambda z: -_to_float(z["epsilon"]))
        ecurve = [_to_float(z["total_energy"]) for z in rr_sorted]
        for i in range(1, len(ecurve)):
            if ecurve[i] + 1e-9 < ecurve[i - 1]:
                nonmono += 1
                break

        last = rr_sorted[-1]
        final_slack.append(_to_float(last["epsilon"]) - _to_float(last["makespan"]))
        steps.append(len(rr_sorted))

        # Pull assignment from any row (same instance)
        loads = _parse_semicolon_ints(rr_sorted[0].get("loads", ""))
        u = _parse_semicolon_ints(rr_sorted[0].get("u", ""))
        if loads and u and len(loads) == len(u) and sum(loads) > 0:
            corr_u_load.append(
                _pearson_corr([float(x) for x in u], [float(x) for x in loads])
            )
            min_u = min(u)
            cheap_load = sum(l for l, ui in zip(loads, u) if ui == min_u)
            cheapest_share.append(cheap_load / sum(loads))

    corr_u_load_sorted = sorted(x for x in corr_u_load if not math.isnan(x))
    corr_mean = _mean(corr_u_load_sorted)
    corr_p25 = (
        _quantile(corr_u_load_sorted, 0.25) if corr_u_load_sorted else float("nan")
    )
    corr_p75 = (
        _quantile(corr_u_load_sorted, 0.75) if corr_u_load_sorted else float("nan")
    )

    return EpsilonSummary(
        rows=len(rows),
        instances=len(by_inst),
        n_range=(min(Ns), max(Ns)) if Ns else (0, 0),
        d_range=(min(Ds), max(Ds)) if Ds else (0, 0),
        eps_range=(min(eps), max(eps)) if eps else (0, 0),
        solve_mean=_mean(solve),
        solve_median=_median(solve),
        solve_p90=_quantile(solve, 0.90),
        energy_mean=_mean(energy),
        energy_median=_median(energy),
        energy_p90=_quantile(energy, 0.90),
        makespan_mean=_mean(makes),
        infeasible_rows=infeasible,
        nonmono_instances=nonmono,
        steps_mean=_mean(steps),
        steps_median=_median(steps),
        final_slack_mean=_mean(final_slack),
        final_tight_frac=(
            (sum(1 for s in final_slack if abs(s) < 1e-9) / len(final_slack))
            if final_slack
            else float("nan")
        ),
        assign_corr_u_load_mean=corr_mean,
        assign_corr_u_load_p25=corr_p25,
        assign_corr_u_load_p75=corr_p75,
        cheapest_load_share_mean=_mean(cheapest_share),
    )


def _fmt(x: float, nd: int = 2) -> str:
    if x is None or math.isnan(x):
        return "nan"
    return f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze pooled + epsilon CSV outputs (epsilon_long run)."
    )
    ap.add_argument(
        "--dir",
        type=str,
        default="PaST/ADP/logs/epsilon_long/logs",
        help="Directory containing the CSV files.",
    )
    args = ap.parse_args()

    folder = Path(str(args.dir)).expanduser().resolve()
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")

    pooled_small = folder / "pooled_nl_LONG_1_same_small.csv"
    pooled_medium = folder / "pooled_nl_LONG_3_same_medium.csv"

    eps_small = folder / "epsilon_nl_LONG_1_small_varND_guided.csv"
    eps_small_to_med = (
        folder / "epsilon_nl_LONG_2_cross_small_to_medium_varND_guided.csv"
    )
    eps_medium = folder / "epsilon_nl_LONG_3_medium_varND_guided.csv"
    eps_med_to_large = (
        folder / "epsilon_nl_LONG_4_cross_medium_to_large_varND_guided.csv"
    )

    print(f"Analysis folder: {folder}")

    print("\n== Pooled DP acceleration (exact vs guided) ==")
    for name, path in [("small", pooled_small), ("medium", pooled_medium)]:
        if not path.exists():
            print(f"- missing pooled {name}: {path.name}")
            continue
        by_beam, overall = summarize_pooled(path)
        print(
            f"\n{name}: rows={overall.n}  gap(mean/med/p90)={_fmt(overall.gap_mean)}%/{_fmt(overall.gap_median)}%/{_fmt(overall.gap_p90)}%  speed(mean/med)={_fmt(overall.speed_mean)}x/{_fmt(overall.speed_median)}x"
        )
        for b in sorted(by_beam):
            s = by_beam[b]
            print(
                f"  beam={b:>2} n={s.n:>3} gap(mean/med/p90)={_fmt(s.gap_mean)}%/{_fmt(s.gap_median)}%/{_fmt(s.gap_p90)}%  speed(mean/med)={_fmt(s.speed_mean)}x/{_fmt(s.speed_median)}x"
            )

    print("\n== Epsilon-constraint simulation (guided, skip exact) ==")
    print(
        "Note: These runs used --skip-exact, so we can assess feasibility/runtime/search behavior, not optimality gaps."
    )

    for label, path in [
        ("eps_small", eps_small),
        ("eps_small_to_med", eps_small_to_med),
        ("eps_medium", eps_medium),
        ("eps_med_to_large", eps_med_to_large),
    ]:
        if not path.exists():
            print(f"\n{label}: missing {path.name}")
            continue
        s = summarize_epsilon(path)
        print(
            f"\n{label}: rows={s.rows} inst={s.instances} N={s.n_range} D={s.d_range} eps={s.eps_range}"
        )
        print(
            f"  solve_s mean/med/p90={_fmt(s.solve_mean,3)}/{_fmt(s.solve_median,3)}/{_fmt(s.solve_p90,3)}"
        )
        print(
            f"  total_energy mean/med/p90={_fmt(s.energy_mean,1)}/{_fmt(s.energy_median,1)}/{_fmt(s.energy_p90,1)}"
        )
        print(
            f"  makespan mean={_fmt(s.makespan_mean,1)}  infeasible rows={s.infeasible_rows}"
        )
        print(
            f"  energy non-monotone instances={s.nonmono_instances} (expect near 0; small >0 can happen due to heuristic noise)"
        )
        print(
            f"  eps steps/instance mean/med={_fmt(s.steps_mean,1)}/{_fmt(s.steps_median,1)}  final tight frac={_fmt(s.final_tight_frac)}"
        )
        print(
            f"  assignment sanity: corr(u,load) mean/p25/p75={_fmt(s.assign_corr_u_load_mean)}/{_fmt(s.assign_corr_u_load_p25)}/{_fmt(s.assign_corr_u_load_p75)} (more negative => cheaper machines more loaded)"
        )
        print(f"  cheapest-machine load share mean={_fmt(s.cheapest_load_share_mean)}")


if __name__ == "__main__":
    main()

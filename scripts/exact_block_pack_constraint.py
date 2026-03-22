#!/usr/bin/env python3
"""Exact fixed-block packing via the free python-constraint solver."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _bootstrap_vendor() -> None:
    root = Path(__file__).resolve().parents[1]
    vendor = root / "vendor" / "constraint_py"
    if vendor.exists():
        sys.path.insert(0, str(vendor))


_bootstrap_vendor()

from constraint import Problem, RecursiveBacktrackingSolver  # noqa: E402


def enumerate_compositions(cap: int, lengths: list[int], totals: list[int]) -> list[tuple[int, ...]]:
    comps: list[tuple[int, ...]] = []
    cur = [0] * len(lengths)

    def rec(i: int, rem: int) -> None:
        if i == len(lengths):
            if rem == 0:
                comps.append(tuple(cur))
            return
        L = lengths[i]
        mx = min(totals[i], rem // L)
        for c in range(mx, -1, -1):
            cur[i] = c
            rec(i + 1, rem - c * L)
        cur[i] = 0

    rec(0, cap)
    return comps


def main() -> int:
    if len(sys.argv) != 3:
        return 2

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    payload = json.loads(input_path.read_text(encoding="utf-8"))

    capacities = [int(x) for x in payload["capacities"]]
    lengths = [int(x) for x in payload["lengths"]]
    totals = [int(x) for x in payload["totals"]]

    block_comps = [enumerate_compositions(cap, lengths, totals) for cap in capacities]
    if any(not comps for comps in block_comps):
        output_path.write_text("infeasible\n", encoding="utf-8")
        return 0

    problem = Problem(RecursiveBacktrackingSolver())
    names = [f"b{b}" for b in range(len(capacities))]
    for name, comps in zip(names, block_comps):
        problem.addVariable(name, comps)

    def inventory_constraint(*values: tuple[int, ...]) -> bool:
        used = [0] * len(lengths)
        for comp in values:
            for i, c in enumerate(comp):
                used[i] += c
                if used[i] > totals[i]:
                    return False
        return used == totals

    problem.addConstraint(inventory_constraint, names)
    solution = problem.getSolution()
    if solution is None:
        output_path.write_text("infeasible\n", encoding="utf-8")
        return 0

    sequence: list[int] = []
    for b in range(len(capacities)):
        comp = solution[f"b{b}"]
        for i, count in enumerate(comp):
            sequence.extend([lengths[i]] * count)

    output_path.write_text(
        "feasible\n" + ",".join(str(v) for v in sequence) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

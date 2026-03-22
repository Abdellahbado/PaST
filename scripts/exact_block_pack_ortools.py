#!/usr/bin/env python3
"""Exact fixed-block packing via OR-Tools CP-SAT.

Input: JSON file with integer arrays:
  {"capacities":[...], "lengths":[...], "totals":[...]}

Output: text file
  feasible
  4,4,6,10,...

or
  infeasible

or
  timeout
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _bootstrap_vendor() -> None:
    root = Path(__file__).resolve().parents[1]
    vendor = root / "vendor" / "ortools_py"
    if vendor.exists():
        sys.path.insert(0, str(vendor))


_bootstrap_vendor()

from ortools.sat.python import cp_model  # noqa: E402


def main() -> int:
    if len(sys.argv) not in (3, 4):
        return 2

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    time_limit_sec = float(sys.argv[3]) if len(sys.argv) == 4 else 20.0
    payload = json.loads(input_path.read_text(encoding="utf-8"))

    capacities = [int(x) for x in payload["capacities"]]
    lengths = [int(x) for x in payload["lengths"]]
    totals = [int(x) for x in payload["totals"]]

    n_blocks = len(capacities)
    n_types = len(lengths)

    model = cp_model.CpModel()
    x: dict[tuple[int, int], cp_model.IntVar] = {}

    for b in range(n_blocks):
        cap = capacities[b]
        for i in range(n_types):
            ub = min(totals[i], cap // lengths[i] if lengths[i] > 0 else 0)
            x[b, i] = model.new_int_var(0, ub, f"x_{b}_{i}")

    for i in range(n_types):
        model.add(sum(x[b, i] for b in range(n_blocks)) == totals[i])

    for b in range(n_blocks):
        model.add(sum(lengths[i] * x[b, i] for i in range(n_types)) == capacities[b])

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = max(time_limit_sec, 0.0)
    status = solver.solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sequence: list[int] = []
        for b in range(n_blocks):
            for i in range(n_types):
                count = int(solver.value(x[b, i]))
                sequence.extend([lengths[i]] * count)

        lines = ["feasible", ",".join(str(v) for v in sequence)]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return 0

    if status == cp_model.INFEASIBLE:
        output_path.write_text("infeasible\n", encoding="utf-8")
        return 0

    output_path.write_text("timeout\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

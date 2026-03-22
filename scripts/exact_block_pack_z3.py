#!/usr/bin/env python3
"""Exact fixed-block packing via the free z3 binary."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        return 2

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    payload = json.loads(input_path.read_text(encoding="utf-8"))

    capacities = [int(x) for x in payload["capacities"]]
    lengths = [int(x) for x in payload["lengths"]]
    totals = [int(x) for x in payload["totals"]]

    n_blocks = len(capacities)
    n_types = len(lengths)

    var_names: list[str] = []
    lines = ["(set-option :produce-models true)"]
    for b in range(n_blocks):
        cap = capacities[b]
        for i in range(n_types):
            name = f"x_{b}_{i}"
            var_names.append(name)
            ub = min(totals[i], cap // lengths[i] if lengths[i] > 0 else 0)
            lines.append(f"(declare-const {name} Int)")
            lines.append(f"(assert (<= 0 {name}))")
            lines.append(f"(assert (<= {name} {ub}))")

    for i in range(n_types):
        expr = " ".join(f"x_{b}_{i}" for b in range(n_blocks))
        lines.append(f"(assert (= (+ {expr}) {totals[i]}))")

    for b in range(n_blocks):
        expr = " ".join(f"(* {lengths[i]} x_{b}_{i})" for i in range(n_types))
        lines.append(f"(assert (= (+ {expr}) {capacities[b]}))")

    lines.append("(check-sat)")
    lines.append("(get-value (" + " ".join(var_names) + "))")

    proc = subprocess.run(
        ["z3", "-in"],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return 1

    out = proc.stdout.strip()
    if out.startswith("unsat"):
        output_path.write_text("infeasible\n", encoding="utf-8")
        return 0
    if not out.startswith("sat"):
        return 1

    values = {name: 0 for name in var_names}
    for name, value in re.findall(r"\(\s*([A-Za-z0-9_]+)\s+(-?[0-9]+)\s*\)", out):
        values[name] = int(value)

    sequence: list[int] = []
    for b in range(n_blocks):
        for i in range(n_types):
            sequence.extend([lengths[i]] * values[f"x_{b}_{i}"])

    output_path.write_text(
        "feasible\n" + ",".join(str(v) for v in sequence) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

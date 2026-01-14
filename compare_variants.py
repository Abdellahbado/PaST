"""Compare PaST training runs by plotting metrics.jsonl curves.

Typical usage (Kaggle / local):
  python -m PaST.compare_variants --runs_dir runs \
    --variant_a ppo_family_q4 --variant_b ppo_family_q4_ctx13 --seed 0

This reads:
  runs/<variant>/seed_<seed>/metrics.jsonl
and writes a PNG next to the runs directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _series(
    rows: List[Dict[str, Any]],
    x_key: str,
    y_key: str,
    *,
    only_if_present: bool = True,
) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for r in rows:
        if only_if_present and y_key not in r:
            continue
        if x_key not in r:
            continue
        xs.append(float(r[x_key]))
        ys.append(float(r.get(y_key, float("nan"))))
    return xs, ys


def _moving_average(y: List[float], window: int) -> List[float]:
    if window <= 1 or len(y) == 0:
        return y
    out: List[float] = []
    acc = 0.0
    q: List[float] = []
    for v in y:
        q.append(v)
        acc += v
        if len(q) > window:
            acc -= q.pop(0)
        out.append(acc / len(q))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot and compare PaST runs (metrics.jsonl).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--variant_a", type=str, required=True)
    p.add_argument("--variant_b", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Moving average window over updates (1 disables).",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output png path. Defaults under runs_dir/compare/.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    runs_dir = Path(args.runs_dir)
    run_a = runs_dir / args.variant_a / f"seed_{args.seed}" / "metrics.jsonl"
    run_b = runs_dir / args.variant_b / f"seed_{args.seed}" / "metrics.jsonl"

    rows_a = _read_jsonl(run_a)
    rows_b = _read_jsonl(run_b)

    x_key = "time/env_steps"

    # Rollout return
    xa, ya = _series(rows_a, x_key, "rollout/rewards_mean", only_if_present=True)
    xb, yb = _series(rows_b, x_key, "rollout/rewards_mean", only_if_present=True)

    # Eval energy (sparser)
    xea, yea = _series(rows_a, x_key, "eval/energy_mean", only_if_present=True)
    xeb, yeb = _series(rows_b, x_key, "eval/energy_mean", only_if_present=True)

    if args.smooth > 1:
        ya_s = _moving_average(ya, args.smooth)
        yb_s = _moving_average(yb, args.smooth)
    else:
        ya_s, yb_s = ya, yb

    out_path: Path
    if args.out is not None:
        out_path = Path(args.out)
    else:
        out_dir = runs_dir / "compare"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.variant_a}_vs_{args.variant_b}_seed{args.seed}.png"

    # Import matplotlib lazily so training deps don't require it.
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    ax.plot(xa, ya_s, label=f"{args.variant_a} (rollout mean)")
    ax.plot(xb, yb_s, label=f"{args.variant_b} (rollout mean)")
    ax.set_ylabel("rollout/rewards_mean")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax2 = axes[1]
    if len(xea) > 0:
        ax2.plot(xea, yea, marker="o", linestyle="-", label=f"{args.variant_a} (eval)")
    if len(xeb) > 0:
        ax2.plot(xeb, yeb, marker="o", linestyle="-", label=f"{args.variant_b} (eval)")
    ax2.set_ylabel("eval/energy_mean")
    ax2.set_xlabel("env steps")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle(f"PaST comparison (seed={args.seed})")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Dict, List


@dataclass
class Record:
    stage: str
    T: int
    N: int
    K: int
    sum_p: int
    exact_cost: float
    guided_cost: float
    exact_finish: int
    guided_finish: int
    exact_s: float
    guided_s: float
    gap_pct: float
    beam: int


def _parse_stage_log(path: Path, stage: str) -> List[Record]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    tokens = text.replace("\n", " ").split()

    records: List[Record] = []
    cur: Dict[str, str] = {}

    for tok in tokens:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        if k == "T":
            cur = {}
        cur[k] = v

        if k == "beam":
            required = {
                "T",
                "N",
                "K",
                "sum_p",
                "exact_cost",
                "guided_cost",
                "exact_finish",
                "guided_finish",
                "exact_s",
                "guided_s",
                "gap_pct",
                "beam",
            }
            if not required.issubset(cur.keys()):
                continue
            try:
                records.append(
                    Record(
                        stage=stage,
                        T=int(cur["T"]),
                        N=int(cur["N"]),
                        K=int(cur["K"]),
                        sum_p=int(cur["sum_p"]),
                        exact_cost=float(cur["exact_cost"]),
                        guided_cost=float(cur["guided_cost"]),
                        exact_finish=int(cur["exact_finish"]),
                        guided_finish=int(cur["guided_finish"]),
                        exact_s=float(cur["exact_s"]),
                        guided_s=float(cur["guided_s"]),
                        gap_pct=float(cur["gap_pct"]),
                        beam=int(cur["beam"]),
                    )
                )
            except ValueError:
                pass
            cur = {}

    return records


def _summarize(records: List[Record]) -> Dict[str, float]:
    if not records:
        return {
            "n": 0,
            "gap_mean": float("nan"),
            "gap_median": float("nan"),
            "gap_max": float("nan"),
            "speedup_mean": float("nan"),
            "speedup_median": float("nan"),
            "ties": 0,
            "guided_worse": 0,
        }

    gaps = [r.gap_pct for r in records]
    speedups = [r.exact_s / max(r.guided_s, 1e-12) for r in records]
    ties = sum(abs(r.guided_cost - r.exact_cost) <= 1e-9 for r in records)
    worse = sum(r.guided_cost > r.exact_cost + 1e-9 for r in records)

    return {
        "n": float(len(records)),
        "gap_mean": mean(gaps),
        "gap_median": median(gaps),
        "gap_max": max(gaps),
        "speedup_mean": mean(speedups),
        "speedup_median": median(speedups),
        "ties": float(ties),
        "guided_worse": float(worse),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate curriculum logs for guided DP vs exact DP."
    )
    ap.add_argument(
        "--logdir",
        type=str,
        required=True,
        help="Directory containing stage*.log files.",
    )
    ap.add_argument(
        "--stages",
        type=str,
        default="stage1_easy,stage2_mid,stage3_hard",
        help="Comma-separated stage names (without .log).",
    )
    ap.add_argument(
        "--out-csv", type=str, default="", help="Optional per-run CSV output path."
    )
    ap.add_argument(
        "--out-json", type=str, default="", help="Optional summary JSON output path."
    )
    args = ap.parse_args()

    logdir = Path(args.logdir)
    stages = [x.strip() for x in args.stages.split(",") if x.strip()]

    all_records: List[Record] = []
    summary: Dict[str, Dict[str, float]] = {}

    for stage in stages:
        path = logdir / f"{stage}.log"
        if not path.exists():
            summary[stage] = {"n": 0.0}
            continue
        recs = _parse_stage_log(path, stage)
        all_records.extend(recs)
        summary[stage] = _summarize(recs)

    summary["overall"] = _summarize(all_records)

    print("=== Guided vs Exact Summary ===")
    for stage in stages + ["overall"]:
        s = summary.get(stage, {"n": 0.0})
        n = int(s.get("n", 0.0))
        if n == 0:
            print(f"[{stage}] n=0")
            continue
        print(
            f"[{stage}] n={n} "
            f"gap(mean/med/max)=({s['gap_mean']:.4f}/{s['gap_median']:.4f}/{s['gap_max']:.4f}) "
            f"speedup(mean/med)=({s['speedup_mean']:.3f}/{s['speedup_median']:.3f}) "
            f"ties={int(s['ties'])} guided_worse={int(s['guided_worse'])}"
        )

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "stage",
                    "T",
                    "N",
                    "K",
                    "sum_p",
                    "exact_cost",
                    "guided_cost",
                    "exact_finish",
                    "guided_finish",
                    "exact_s",
                    "guided_s",
                    "gap_pct",
                    "beam",
                    "speedup_exact_over_guided",
                ]
            )
            for r in all_records:
                writer.writerow(
                    [
                        r.stage,
                        r.T,
                        r.N,
                        r.K,
                        r.sum_p,
                        f"{r.exact_cost:.8f}",
                        f"{r.guided_cost:.8f}",
                        r.exact_finish,
                        r.guided_finish,
                        f"{r.exact_s:.8f}",
                        f"{r.guided_s:.8f}",
                        f"{r.gap_pct:.8f}",
                        r.beam,
                        f"{(r.exact_s / max(r.guided_s, 1e-12)):.8f}",
                    ]
                )
        print(f"Wrote CSV: {out_csv}")

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote JSON: {out_json}")


if __name__ == "__main__":
    main()

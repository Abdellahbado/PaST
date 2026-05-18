#!/usr/bin/env python3

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from run_plan13_two_track_recovery import run_row

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "research" / "k_vs_arithmetic_axes_20260412" / "csv" / "plan16"
OUT_CSV = OUT_DIR / "PLAN16_k_scaling_n1000.csv"

FAMILY_SIZES: dict[str, list[int]] = {
    "g24": [2, 4],
    "g37": [3, 7],
    "g810": [8, 10],
    "g3567": [3, 5, 6, 7],
    "g12357": [1, 2, 3, 5, 7],
    "g246810": [2, 4, 6, 8, 10],
    "g12345678910": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "g1234567891011121314151617181920": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
    ],
}

VARIANTS: list[tuple[str, dict[str, str]]] = [
    ("baseline", {}),
    (
        "dense_step2_fastpath",
        {
            "PAST_DENSE_UNIT_STEP2_FASTPATH": "1",
            "PAST_DENSE_UNIT_FASTPATH_K_MIN": "8",
        },
    ),
]


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def key_of(r: dict[str, Any]) -> tuple[str, str, int]:
    return (str(r.get("variant_label", "")), str(r.get("family_id", "")), int(r.get("seed", "-1")))


def main() -> None:
    rows = [
        r
        for r in load_rows(OUT_CSV)
        if str(r.get("family_id", "")) not in {"g37", "g810"}
    ]
    seen = {key_of(r) for r in rows}

    for fam in FAMILY_SIZES.keys():
        for variant_label, env in VARIANTS:
            for seed in (0, 1):
                key = (variant_label, fam, seed)
                if key in seen:
                    continue
                case_env = dict(env)
                if len(FAMILY_SIZES[fam]) == 2:
                    # K=2 families close through the Step-3 profile-realization path.
                    case_env.update(
                        {
                            "PAST_RELAXED_BINPACK_SOLVER": "profile_repair_beam",
                            "PAST_PROFILE_REALIZATION_SELECTOR_POLICY": "auto_v1",
                        }
                    )
                row = run_row(
                    family_id=fam,
                    n_jobs=1000,
                    seed=seed,
                    time_limit=900.0,
                    variant_label=variant_label,
                    env_overrides=case_env,
                    max_rss_gb=16.0,
                )
                row["k_size"] = str(len(FAMILY_SIZES[fam]))
                row["family_sizes"] = ",".join(str(x) for x in FAMILY_SIZES[fam])
                rows.append(row)
                seen.add(key)
                write_rows(OUT_CSV, rows)
                print(
                    f"{variant_label} fam={fam} K={len(FAMILY_SIZES[fam])} seed={seed} "
                    f"step={row.get('deciding_step')} rt={row.get('runtime_sec')} "
                    f"ub={row.get('ub')} lb={row.get('lb')}"
                )

    print(f"Wrote {OUT_CSV} rows={len(rows)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import csv
import glob
import os
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Tuple


def _safe_float(x: str) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return None
        return float(s)
    except Exception:
        return None


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return [row for row in r]


def _parse_name(path: str) -> Tuple[str, str, str, str]:
    """Return (kind, category, profile, model).

    Expected filenames:
      pooled_{category}_{profile}_{model}.csv
      epsilon_{category}_{profile}_{model}.csv
    """
    base = os.path.basename(path)
    stem = base
    if stem.endswith(".csv"):
        stem = stem[: -len(".csv")]
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected filename (need at least 4 '_' parts): {base}")
    kind = parts[0]
    category = parts[1]
    model = parts[-1]
    profile = "_".join(parts[2:-1])
    return kind, category, profile, model


@dataclass
class SummaryStats:
    n: int
    mean: float
    median: float


def _summary(values: List[float]) -> Optional[SummaryStats]:
    if not values:
        return None
    return SummaryStats(n=len(values), mean=mean(values), median=median(values))


def _fmt_stats(s: Optional[SummaryStats], *, digits: int = 3) -> str:
    if s is None:
        return "n=0"
    return f"n={s.n} mean={s.mean:.{digits}f} med={s.median:.{digits}f}"


def analyze_pooled(pooled_csv: str) -> Dict[str, Optional[SummaryStats]]:
    rows = _read_csv_rows(pooled_csv)

    gaps = []
    speedups = []
    speedups_incl_train = []
    exact_s = []
    guided_s = []
    train_s = []

    for row in rows:
        g = _safe_float(row.get("gap_learned_pct", ""))
        if g is not None:
            gaps.append(g)

        su = _safe_float(row.get("speedup_learned", ""))
        if su is not None:
            speedups.append(su)

        sui = _safe_float(row.get("speedup_learned_incl_train", ""))
        if sui is not None:
            speedups_incl_train.append(sui)

        es = _safe_float(row.get("exact_s", ""))
        if es is not None:
            exact_s.append(es)

        gs = _safe_float(row.get("guided_learned_s", ""))
        if gs is not None:
            guided_s.append(gs)

        ts = _safe_float(row.get("train_s", ""))
        if ts is not None:
            train_s.append(ts)

    return {
        "gap_learned_pct": _summary(gaps),
        "speedup_learned": _summary(speedups),
        "speedup_learned_incl_train": _summary(speedups_incl_train),
        "exact_s": _summary(exact_s),
        "guided_s": _summary(guided_s),
        "train_s": _summary(train_s),
    }


def analyze_epsilon(epsilon_csv: str) -> Dict[str, Optional[SummaryStats]]:
    rows = _read_csv_rows(epsilon_csv)

    # Pair exact vs guided by (instance_seed, epsilon)
    key_to_vals: Dict[Tuple[int, int], Dict[str, Tuple[float, float]]] = {}

    for row in rows:
        inst = _safe_float(row.get("instance_seed", ""))
        eps = _safe_float(row.get("epsilon", ""))
        method = str(row.get("method", "")).strip().lower()
        energy = _safe_float(row.get("total_energy", ""))
        solve_s = _safe_float(row.get("solve_s", ""))
        if inst is None or eps is None or energy is None or solve_s is None:
            continue
        if method not in {"exact", "guided"}:
            continue
        k = (int(inst), int(eps))
        entry = key_to_vals.get(k)
        if entry is None:
            entry = {}
            key_to_vals[k] = entry
        entry[method] = (float(energy), float(solve_s))

    gaps_pct: List[float] = []
    speedups: List[float] = []
    guided_times: List[float] = []
    exact_times: List[float] = []

    for _k, entry in key_to_vals.items():
        if "exact" not in entry or "guided" not in entry:
            continue
        exact_energy, exact_t = entry["exact"]
        guided_energy, guided_t = entry["guided"]

        if exact_energy != 0.0:
            gaps_pct.append(100.0 * (guided_energy - exact_energy) / exact_energy)

        if guided_t > 0.0:
            speedups.append(exact_t / guided_t)

        guided_times.append(guided_t)
        exact_times.append(exact_t)

    return {
        "energy_gap_pct": _summary(gaps_pct),
        "time_speedup_exact_over_guided": _summary(speedups),
        "guided_s": _summary(guided_times),
        "exact_s": _summary(exact_times),
    }


def _rank_models(
    metrics: Dict[str, Dict[str, Optional[SummaryStats]]],
    *,
    effectiveness_key: str,
    efficiency_key: str,
    abs_effectiveness: bool,
    higher_efficiency_better: bool,
) -> List[Tuple[str, float, float]]:
    scored: List[Tuple[str, float, float]] = []
    for model, m in metrics.items():
        eff = m.get(effectiveness_key)
        eff_v = None if eff is None else eff.mean
        if eff_v is None:
            continue
        if abs_effectiveness:
            eff_v = abs(eff_v)

        effi = m.get(efficiency_key)
        effi_v = None if effi is None else effi.mean
        if effi_v is None:
            continue

        scored.append((model, float(eff_v), float(effi_v)))

    # Primary: effectiveness (lower better), secondary: efficiency (higher or lower depending)
    if higher_efficiency_better:
        scored.sort(key=lambda x: (x[1], -x[2], x[0]))
    else:
        scored.sort(key=lambda x: (x[1], x[2], x[0]))
    return scored


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log-dir",
        type=str,
        default="ADP/logs/deploy_epsilon_profiles_all_sizes",
    )
    args = ap.parse_args()

    log_dir = str(args.log_dir)

    pooled_paths = sorted(glob.glob(os.path.join(log_dir, "pooled_*.csv")))
    epsilon_paths = sorted(glob.glob(os.path.join(log_dir, "epsilon_*.csv")))

    pooled_by_group: Dict[Tuple[str, str], Dict[str, Dict[str, Optional[SummaryStats]]]] = defaultdict(dict)
    eps_by_group: Dict[Tuple[str, str], Dict[str, Dict[str, Optional[SummaryStats]]]] = defaultdict(dict)

    for p in pooled_paths:
        kind, category, profile, model = _parse_name(p)
        if kind != "pooled":
            continue
        pooled_by_group[(category, profile)][model] = analyze_pooled(p)

    for p in epsilon_paths:
        kind, category, profile, model = _parse_name(p)
        if kind != "epsilon":
            continue
        eps_by_group[(category, profile)][model] = analyze_epsilon(p)

    groups = sorted(set(pooled_by_group.keys()) | set(eps_by_group.keys()))

    print(f"log_dir={log_dir}")
    print(f"found pooled={len(pooled_paths)} epsilon={len(epsilon_paths)}")
    print("-")

    for (category, profile) in groups:
        print(f"## GROUP category={category} profile={profile}")

        pooled_models = pooled_by_group.get((category, profile), {})
        if pooled_models:
            print("[pooled] per-model summary:")
            for model in sorted(pooled_models.keys()):
                s = pooled_models[model]
                print(
                    f"  - {model:6s} gap_learned_pct({_fmt_stats(s.get('gap_learned_pct'))}) "
                    f"speedup({_fmt_stats(s.get('speedup_learned'))}) "
                    f"guided_s({_fmt_stats(s.get('guided_s'))}) train_s({_fmt_stats(s.get('train_s'))})"
                )

            ranked = _rank_models(
                pooled_models,
                effectiveness_key="gap_learned_pct",
                efficiency_key="speedup_learned",
                abs_effectiveness=True,
                higher_efficiency_better=True,
            )
            if ranked:
                print("[pooled] ranking (best -> worst) by: low |gap| then high speedup:")
                for i, (m, eff, effi) in enumerate(ranked, 1):
                    print(f"  {i}. {m:6s} |gap|={eff:.3f}%  speedup={effi:.3f}x")
        else:
            print("[pooled] no data")

        eps_models = eps_by_group.get((category, profile), {})
        if eps_models:
            print("[epsilon] per-model summary:")
            for model in sorted(eps_models.keys()):
                s = eps_models[model]
                print(
                    f"  - {model:6s} energy_gap_pct({_fmt_stats(s.get('energy_gap_pct'))}) "
                    f"speedup({_fmt_stats(s.get('time_speedup_exact_over_guided'))}) "
                    f"guided_s({_fmt_stats(s.get('guided_s'))})"
                )

            ranked = _rank_models(
                eps_models,
                effectiveness_key="energy_gap_pct",
                efficiency_key="time_speedup_exact_over_guided",
                abs_effectiveness=True,
                higher_efficiency_better=True,
            )
            if ranked:
                print("[epsilon] ranking (best -> worst) by: low |energy_gap| then high speedup:")
                for i, (m, eff, effi) in enumerate(ranked, 1):
                    print(f"  {i}. {m:6s} |gap|={eff:.3f}%  speedup={effi:.3f}x")
        else:
            print("[epsilon] no data")

        print("-")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from PaST.solvers.optimal_benchmark_dp import solve_optimal_benchmark_dp
from PaST.solvers.vhat_linear import (
    FeatureSpec,
    LinearRidgeValueModel,
    fit_ridge,
    phi_for_state,
)
from PaST.solvers.vhat_tou_features import build_tou_feature_context
from PaST.sandbox.train_eval_vhat_beam_dp import (
    build_instance,
    decode_state,
    encode_setup,
    remaining_p_list,
)


def _collect_state_labels(
    *,
    rng: np.random.Generator,
    T: int,
    totals: np.ndarray,
    lengths: np.ndarray,
    prices: np.ndarray,
    target_samples: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    states: List[Tuple[int, Tuple[int, ...]]] = []
    labels: List[float] = []

    attempts = 0
    max_attempts = max(6 * int(target_samples), 300)

    while len(states) < target_samples and attempts < max_attempts:
        attempts += 1
        t = int(rng.integers(0, T))
        used = tuple(int(x) for x in rng.integers(0, totals + 1))

        rem_work = int(np.sum((totals - np.asarray(used, dtype=np.int32)) * lengths))
        if rem_work > (T - t):
            continue

        rem_p = remaining_p_list(used, totals, lengths)
        if not rem_p:
            y = 0.0
        else:
            sub = solve_optimal_benchmark_dp(rem_p, prices[t:], tie_break="cost")
            if not sub.feasible:
                continue
            y = float(sub.cost)

        states.append((t, used))
        labels.append(y)

    if len(states) < max(30, target_samples // 2):
        raise RuntimeError(
            f"Too few feasible state labels: {len(states)} after {attempts} attempts (target={target_samples})."
        )

    return (
        np.asarray(states, dtype=object),
        np.asarray(labels, dtype=np.float64),
        attempts,
    )


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    var = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if var <= 1e-12:
        return float("nan")
    sse = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - sse / var


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Holdout evaluation for learned Vhat-guided beam DP."
    )
    ap.add_argument("--seed-start", type=int, default=30)
    ap.add_argument("--seed-end", type=int, default=59)
    ap.add_argument("--D", type=int, default=3)
    ap.add_argument("--N", type=int, default=18)
    ap.add_argument("--pmax", type=int, default=5)
    ap.add_argument("--samples", type=int, default=4500)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument(
        "--beam",
        type=int,
        default=300,
        help="Single beam width (ignored if --beams is provided).",
    )
    ap.add_argument(
        "--beams",
        type=str,
        default="",
        help="Optional comma-separated beam widths to sweep, e.g. '50,100,200,400'.",
    )
    ap.add_argument("--prune-factor", type=float, default=2.0)
    ap.add_argument("--test-ratio", type=float, default=0.25)
    ap.add_argument(
        "--transferable-features",
        action="store_true",
        help="Use fixed-dimension features (recommended if you want to reuse a model across varying K).",
    )
    ap.add_argument(
        "--random-baseline-scale",
        type=float,
        default=1.0,
        help="Scale of random baseline heuristic values (Normal(0, scale)).",
    )
    ap.add_argument(
        "--out-csv", type=str, default="PaST/logs/eval_guided_vhat_holdout.csv"
    )
    args = ap.parse_args()

    if not (0.05 <= args.test_ratio <= 0.5):
        raise ValueError("--test-ratio should be in [0.05, 0.5]")

    if str(args.beams).strip():
        beams = [int(x) for x in str(args.beams).split(",") if x.strip()]
        if any(b <= 0 for b in beams):
            raise ValueError("All --beams entries must be positive")
    else:
        beams = [int(args.beam)]

    seeds = list(range(int(args.seed_start), int(args.seed_end) + 1))
    rows: List[Dict[str, float]] = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        p, prices = build_instance(
            rng=rng, D=int(args.D), N=int(args.N), pmax=int(args.pmax)
        )
        T = int(len(prices))

        t0 = time.perf_counter()
        exact = solve_optimal_benchmark_dp(p, prices, tie_break="early")
        exact_s = time.perf_counter() - t0
        if not exact.feasible:
            continue

        lengths, totals, radices, _mult = encode_setup(p)
        ctx = build_tou_feature_context(prices, H=20, validate_repeating=True)
        if bool(args.transferable_features):
            spec = FeatureSpec(
                include_per_class_counts=False,
                include_per_class_now_cost=False,
                include_bins=True,
            )
        else:
            spec = FeatureSpec(
                include_per_class_counts=True,
                include_per_class_now_cost=True,
                include_bins=True,
            )

        states_obj, y, attempts = _collect_state_labels(
            rng=rng,
            T=T,
            totals=totals,
            lengths=lengths,
            prices=prices,
            target_samples=int(args.samples),
        )

        X = np.vstack(
            [
                phi_for_state(
                    t=int(t),
                    used=used,
                    totals=totals,
                    lengths=lengths.tolist(),
                    ctx=ctx,
                    spec=spec,
                )
                for (t, used) in states_obj
            ]
        )

        idx = np.arange(len(y))
        rng.shuffle(idx)
        split = int((1.0 - float(args.test_ratio)) * len(idx))
        train_idx = idx[:split]
        test_idx = idx[split:]
        if len(test_idx) < 10:
            test_idx = idx[-10:]
            train_idx = idx[:-10]

        w = fit_ridge(X[train_idx], y[train_idx], l2=float(args.l2))
        model = LinearRidgeValueModel(weights=w, spec=spec)

        y_hat_test = X[test_idx] @ w
        r2 = _r2_score(y[test_idx], y_hat_test)
        mae = float(np.mean(np.abs(y[test_idx] - y_hat_test)))

        used_cache: Dict[int, Tuple[int, ...]] = {0: tuple([0] * len(lengths))}

        def vhat(t: int, state: int) -> float:
            s = int(state)
            used_cached = used_cache.get(s)
            if used_cached is None:
                used_cached = decode_state(s, radices)
                used_cache[s] = used_cached
            return model.predict_from_used(
                t=int(t),
                used=used_cached,
                totals=totals,
                lengths=lengths.tolist(),
                ctx=ctx,
            )

        def vhat_random(t: int, state: int) -> float:
            # Deterministic random baseline per (t,state) so results are reproducible.
            # Mix seed with t/state into a simple hash.
            x = (int(seed) * 1000003 + int(t) * 9176 + int(state) * 1315423911) & 0xFFFFFFFF
            # LCG -> float in (0,1)
            x = (1103515245 * x + 12345) & 0x7FFFFFFF
            u = (x + 1) / (0x7FFFFFFF + 2)
            # Approx Normal(0,1) from inverse error function approximation is overkill;
            # use centered uniform as a cheap stand-in baseline.
            return float((u - 0.5) * 2.0 * float(args.random_baseline_scale))

        prefix_prices = np.concatenate([[0.0], np.cumsum(prices, dtype=np.float64)])

        def vhat_work_mean_price(t: int, state: int) -> float:
            # Cheap price-aware baseline: remaining work * mean future price.
            s = int(state)
            used_cached = used_cache.get(s)
            if used_cached is None:
                used_cached = decode_state(s, radices)
                used_cache[s] = used_cached
            remaining = totals.astype(np.int32) - np.asarray(used_cached, dtype=np.int32)
            W = int(np.sum(remaining * lengths))
            if W <= 0:
                return 0.0
            tt = int(t)
            tt = 0 if tt < 0 else tt
            tt = T if tt > T else tt
            rem_len = max(1, T - tt)
            mean_price = float((prefix_prices[T] - prefix_prices[tt]) / rem_len)
            return float(W) * mean_price

        for beam in beams:
            t1 = time.perf_counter()
            guided_learned = solve_optimal_benchmark_dp(
                p,
                prices,
                tie_break="early",
                guided=True,
                beam_width=int(beam),
                prune_factor=float(args.prune_factor),
                vhat=vhat,
            )
            guided_learned_s = time.perf_counter() - t1

            t2 = time.perf_counter()
            guided_zero = solve_optimal_benchmark_dp(
                p,
                prices,
                tie_break="early",
                guided=True,
                beam_width=int(beam),
                prune_factor=float(args.prune_factor),
                vhat=lambda _t, _s: 0.0,
            )
            guided_zero_s = time.perf_counter() - t2

            t3 = time.perf_counter()
            guided_random = solve_optimal_benchmark_dp(
                p,
                prices,
                tie_break="early",
                guided=True,
                beam_width=int(beam),
                prune_factor=float(args.prune_factor),
                vhat=vhat_random,
            )
            guided_random_s = time.perf_counter() - t3

            t4 = time.perf_counter()
            guided_price = solve_optimal_benchmark_dp(
                p,
                prices,
                tie_break="early",
                guided=True,
                beam_width=int(beam),
                prune_factor=float(args.prune_factor),
                vhat=vhat_work_mean_price,
            )
            guided_price_s = time.perf_counter() - t4

            gap_learned = (
                (guided_learned.cost - exact.cost) / max(1e-9, abs(exact.cost)) * 100.0
            )
            gap_zero = (
                (guided_zero.cost - exact.cost) / max(1e-9, abs(exact.cost)) * 100.0
            )
            gap_random = (
                (guided_random.cost - exact.cost) / max(1e-9, abs(exact.cost)) * 100.0
            )
            gap_price = (
                (guided_price.cost - exact.cost) / max(1e-9, abs(exact.cost)) * 100.0
            )

            row = {
                "seed": float(seed),
                "T": float(T),
                "N": float(len(p)),
                "K": float(len(lengths)),
                "sum_p": float(sum(p)),
                "label_attempts": float(attempts),
                "n_states_labeled": float(len(y)),
                "r2_test": float(r2),
                "mae_test": float(mae),
                "feat_transferable": float(int(bool(args.transferable_features))),
                "exact_cost": float(exact.cost),
                "exact_s": float(exact_s),
                "guided_learned_cost": float(guided_learned.cost),
                "guided_learned_s": float(guided_learned_s),
                "guided_zero_cost": float(guided_zero.cost),
                "guided_zero_s": float(guided_zero_s),
                "guided_random_cost": float(guided_random.cost),
                "guided_random_s": float(guided_random_s),
                "guided_price_cost": float(guided_price.cost),
                "guided_price_s": float(guided_price_s),
                "gap_learned_pct": float(gap_learned),
                "gap_zero_pct": float(gap_zero),
                "gap_random_pct": float(gap_random),
                "gap_price_pct": float(gap_price),
                "speedup_learned": float(exact_s / max(guided_learned_s, 1e-12)),
                "speedup_zero": float(exact_s / max(guided_zero_s, 1e-12)),
                "speedup_random": float(exact_s / max(guided_random_s, 1e-12)),
                "speedup_price": float(exact_s / max(guided_price_s, 1e-12)),
                "beam": float(int(beam)),
            }
            rows.append(row)

            print(
                f"seed={seed} beam={beam} K={int(row['K'])} exact={row['exact_cost']:.4f} "
                f"L={row['guided_learned_cost']:.4f} Z={row['guided_zero_cost']:.4f} "
                f"P={row['guided_price_cost']:.4f} R={row['guided_random_cost']:.4f} "
                f"gapL={row['gap_learned_pct']:.2f}% gapZ={row['gap_zero_pct']:.2f}% "
                f"gapP={row['gap_price_pct']:.2f}% gapR={row['gap_random_pct']:.2f}% "
                f"R2={row['r2_test']:.3f}"
            )

    if not rows:
        raise RuntimeError("No successful rows were produced.")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Print summary grouped by beam.
    print("=== Summary by beam (mean/median gaps; mean speedups) ===")
    for beam in sorted(set(int(r["beam"]) for r in rows)):
        br = [r for r in rows if int(r["beam"]) == int(beam)]
        def _mean(xs: List[float]) -> float:
            return float(mean(xs))
        def _med(xs: List[float]) -> float:
            return float(median(xs))

        print(
            " ".join(
                [
                    f"beam={beam}",
                    f"n={len(br)}",
                    f"gapL(mean/med)={_mean([r['gap_learned_pct'] for r in br]):.3f}/{_med([r['gap_learned_pct'] for r in br]):.3f}",
                    f"gapZ(mean/med)={_mean([r['gap_zero_pct'] for r in br]):.3f}/{_med([r['gap_zero_pct'] for r in br]):.3f}",
                    f"gapP(mean/med)={_mean([r['gap_price_pct'] for r in br]):.3f}/{_med([r['gap_price_pct'] for r in br]):.3f}",
                    f"gapR(mean/med)={_mean([r['gap_random_pct'] for r in br]):.3f}/{_med([r['gap_random_pct'] for r in br]):.3f}",
                    f"speedL(mean)={_mean([r['speedup_learned'] for r in br]):.3f}",
                    f"speedZ(mean)={_mean([r['speedup_zero'] for r in br]):.3f}",
                ]
            )
        )

    gap_learned = [r["gap_learned_pct"] for r in rows]
    gap_zero = [r["gap_zero_pct"] for r in rows]
    r2s = [r["r2_test"] for r in rows if np.isfinite(r["r2_test"])]
    sp_learned = [r["speedup_learned"] for r in rows]
    sp_zero = [r["speedup_zero"] for r in rows]

    print("\n=== Holdout Summary ===")
    print(f"rows={len(rows)} csv={out_csv}")
    print(
        f"gap_learned mean/median/max = {mean(gap_learned):.4f} / {median(gap_learned):.4f} / {max(gap_learned):.4f}"
    )
    print(
        f"gap_zero    mean/median/max = {mean(gap_zero):.4f} / {median(gap_zero):.4f} / {max(gap_zero):.4f}"
    )
    if r2s:
        print(f"r2_test mean/median = {mean(r2s):.4f} / {median(r2s):.4f}")
    print(
        f"speedup_learned mean/median = {mean(sp_learned):.3f} / {median(sp_learned):.3f}"
    )
    print(f"speedup_zero    mean/median = {mean(sp_zero):.3f} / {median(sp_zero):.3f}")
    print("Meaningful learning signal: gap_learned << gap_zero and positive r2_test.")


if __name__ == "__main__":
    main()

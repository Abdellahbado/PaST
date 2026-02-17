#!/usr/bin/env python3
"""
Benchmark instance generator (BPMSTP-style) with 1-hour slots and 20 hours/day.

Outputs JSON instances with:
- N jobs, M machines
- D days, K = 20*D time slots (hours)
- processing times p_j (integers, in slots)
- machine energy rates u_h
- electricity prices c_t per slot
- periods: run-length encoding of equal consecutive prices (optional convenience)

NEW in this version:
- --price-freeze: reuse the exact same generated price profile across all instances
  (useful when you want to change ONLY the price profile and keep everything else
   generated as before).
- --price-freeze-scope: choose how to freeze prices: daily / per_K / master_prefix
- --price-seed: separate RNG seed for the frozen price profile

Usage examples:
  python generate_benchmark_instances.py --out instances --replicates 10 --seed 123

  # Provide a 20-hour daily price vector (will repeat for D days):
  python generate_benchmark_instances.py --out instances \
    --price-json daily_prices_20.json --replicates 10 --seed 123

  # Provide a full K-length price vector:
  python generate_benchmark_instances.py --out instances \
    --price-json prices_fullK.json --replicates 10 --seed 123

  # Random hourly prices (iid) per instance (old behavior):
  python generate_benchmark_instances.py --out instances \
    --price-mode random_uniform --price-low 1 --price-high 8

  # Random hourly prices, BUT frozen and reused across all instances (new):
  python generate_benchmark_instances.py --out instances \
    --price-mode random_uniform --price-low 1 --price-high 8 \
    --price-freeze --price-freeze-scope daily --price-seed 999
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np


# -------------------------
# New benchmark constants
# -------------------------

HOURS_PER_DAY = 20
SLOT_MINUTES = 60

DEFAULT_PMAX = 12  # processing times in slots (hours): 1..12

DEFAULT_BENCHMARK = {
    "small": {
        "Ns": [20, 40, 60],
        "Ms": [3, 5, 7],
        "Ds": [2, 3, 4],          # as requested
        "u_range": [1, 3],        # integer u_h in [1, 3]
        "target_util": 0.80,      # cap sum(p_j) <= target_util*M*K
    },
    "medium": {
        "Ns": [100, 150, 200],
        "Ms": [8, 12, 16],
        "Ds": [5, 10, 15],        # up to 15
        "u_range": [1, 3],
        "target_util": 0.85,
    },
    "large": {
        "Ns": [250, 300, 350, 400, 500],
        "Ms": [25, 30, 40],
        "Ds": [10, 20, 30],       # up to 30
        "u_range": [1, 6],        # integer u_h in [1, 6]
        "target_util": 0.90,
    },
}


# -------------------------
# Helpers
# -------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def compress_periods(prices: List[float]) -> List[Dict[str, Any]]:
    """Return list of {start,end,price} with 1-based inclusive indices."""
    if not prices:
        return []
    periods = []
    start = 0
    for t in range(1, len(prices) + 1):
        if t == len(prices) or prices[t] != prices[start]:
            periods.append({"start": start + 1, "end": t, "price": prices[start]})
            start = t
    return periods


def load_price_vector(path: Path) -> List[float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "prices" in data:
        data = data["prices"]
    if not isinstance(data, list) or not data:
        raise ValueError(f"Invalid price JSON: expected a non-empty list (or {{'prices': [...]}}): {path}")
    return [float(x) for x in data]


def make_prices(
    K: int,
    D: int,
    rng: np.random.Generator,
    mode: str,
    price_json: Optional[List[float]] = None,
    low: float = 1.0,
    high: float = 8.0,
) -> List[float]:
    """
    Create c_t of length K.
    - If price_json is provided:
        * length HOURS_PER_DAY -> repeated D times
        * length K            -> used as-is
    - Else:
        * daily_tou      -> deterministic HOURS_PER_DAY pattern repeated D times
        * random_uniform -> iid integer U[low, high] per hour (uses provided rng)
    """
    if price_json is not None:
        if len(price_json) == HOURS_PER_DAY:
            if K != HOURS_PER_DAY * D:
                raise ValueError("Internal error: K must equal HOURS_PER_DAY*D.")
            return (price_json * D)[:K]
        if len(price_json) == K:
            return price_json
        raise ValueError(f"price_json length must be {HOURS_PER_DAY} or K={K}, got {len(price_json)}.")

    if mode == "daily_tou":
        # Simple, clear 3-level pattern over an "operational day" of 20 hours:
        # 0..3 off-peak, 4..11 shoulder, 12..15 peak, 16..19 shoulder
        daily = []
        for h in range(HOURS_PER_DAY):
            if 0 <= h < 4:
                daily.append(1.0)
            elif 4 <= h < 12:
                daily.append(2.0)
            elif 12 <= h < 16:
                daily.append(4.0)
            else:
                daily.append(2.0)
        return (daily * D)[:K]

    if mode == "random_uniform":
        lo = int(math.floor(low))
        hi = int(math.floor(high))
        if lo > hi:
            raise ValueError("price-low must be <= price-high.")
        return rng.integers(lo, hi + 1, size=K).astype(float).tolist()

    raise ValueError(f"Unknown price mode: {mode}")


def sample_processing_times(
    N: int,
    M: int,
    K: int,
    rng: np.random.Generator,
    pmax: int,
    target_util: float,
) -> List[int]:
    """
    Sample p_j in {1..pmax} and enforce a load cap:
      sum(p_j) <= floor(target_util * M * K)

    If the initial sample exceeds the cap, decrement random jobs with p_j>1
    until it fits.
    """
    if not (0.0 < target_util <= 1.0):
        raise ValueError("target_util must be in (0, 1].")

    cap = int(math.floor(target_util * M * K))
    if cap < N:
        raise ValueError(f"Infeasible cap: cap={cap} < N={N} (even all p_j=1 won't fit).")

    p = rng.integers(1, pmax + 1, size=N).astype(int)
    total = int(p.sum())
    if total <= cap:
        return p.tolist()

    idxs = np.where(p > 1)[0]
    guard = 0
    while total > cap:
        if idxs.size == 0:
            raise ValueError("Cannot reduce processing times enough to satisfy cap.")
        i = int(rng.choice(idxs))
        p[i] -= 1
        total -= 1
        if p[i] == 1:
            idxs = np.where(p > 1)[0]
        guard += 1
        if guard > 10_000_000:
            raise RuntimeError("Guard triggered while reducing processing times.")
    return p.tolist()


def sample_machine_rates(M: int, rng: np.random.Generator, u_low: int, u_high: int) -> List[int]:
    if u_low > u_high:
        raise ValueError("u_range low must be <= high.")
    return rng.integers(int(u_low), int(u_high) + 1, size=M).astype(int).tolist()


def instance_id(category: str, N: int, M: int, D: int, r: int, seed: int) -> str:
    return f"{category}_N{N}_M{M}_D{D}_H{HOURS_PER_DAY}_r{r:02d}_seed{seed}"


def write_instance_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="Output directory.")
    ap.add_argument("--replicates", type=int, default=10, help="Replicates per (N,M,D) combo.")
    ap.add_argument("--seed", type=int, default=123, help="Base RNG seed.")
    ap.add_argument("--pmax", type=int, default=DEFAULT_PMAX, help="Max processing time (slots).")

    ap.add_argument("--config", type=str, default="", help="Optional JSON to override DEFAULT_BENCHMARK.")
    ap.add_argument("--only", type=str, default="", help="Comma list subset: small,medium,large")

    ap.add_argument("--price-mode", type=str, default="daily_tou", choices=["daily_tou", "random_uniform"])
    ap.add_argument("--price-json", type=str, default="", help=f"JSON with either {HOURS_PER_DAY} or K prices.")
    ap.add_argument("--price-low", type=float, default=1.0, help="For random_uniform mode.")
    ap.add_argument("--price-high", type=float, default=8.0, help="For random_uniform mode.")

    # NEW: freeze price profile across instances
    ap.add_argument(
        "--price-freeze",
        action="store_true",
        help="Freeze generated prices so all instances reuse the same price profile (ignored if --price-json is provided).",
    )
    ap.add_argument(
        "--price-freeze-scope",
        type=str,
        default="daily",
        choices=["daily", "per_K", "master_prefix"],
        help=(
            "How to freeze prices when --price-mode=random_uniform and no --price-json is provided: "
            "daily = one 20-hour vector repeated for D days; "
            "per_K = one K-length vector per K; "
            "master_prefix = one maxK-length vector, use prefix of length K."
        ),
    )
    ap.add_argument(
        "--price-seed",
        type=int,
        default=None,
        help="Optional separate seed used only for frozen price generation (defaults to --seed).",
    )

    ap.add_argument("--no-periods", action="store_true", help="Do not store run-length-encoded periods.")
    return ap.parse_args()


def load_benchmark_config(path: str) -> Dict[str, Any]:
    if not path:
        return DEFAULT_BENCHMARK
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    out = json.loads(json.dumps(DEFAULT_BENCHMARK))  # deep-ish copy
    for cat, spec in cfg.items():
        if cat not in out:
            out[cat] = spec
        else:
            out[cat].update(spec)
    return out


def compute_maxK(bench: Dict[str, Any], only: List[str]) -> int:
    maxD = 1
    for category in only:
        Ds = list(bench[category]["Ds"])
        for D in Ds:
            maxD = max(maxD, int(D))
    return HOURS_PER_DAY * maxD


def frozen_prices_factory(
    *,
    mode: str,
    low: float,
    high: float,
    freeze_scope: str,
    price_seed: int,
    maxK: int,
) -> Tuple[Optional[List[float]], Dict[int, List[float]]]:
    """
    Returns (master_prefix_prices, perK_cache).
    Only used when mode == 'random_uniform' and freezing is enabled.
    """
    master = None
    cache: Dict[int, List[float]] = {}

    lo = int(math.floor(low))
    hi = int(math.floor(high))
    if lo > hi:
        raise ValueError("price-low must be <= price-high.")

    if mode != "random_uniform":
        return None, {}

    if freeze_scope == "master_prefix":
        rngp = np.random.default_rng(int(price_seed))
        master = rngp.integers(lo, hi + 1, size=int(maxK)).astype(float).tolist()
        return master, {}

    # daily and per_K will use cache on demand
    return None, cache


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    bench = load_benchmark_config(args.config)
    only = [x.strip() for x in args.only.split(",") if x.strip()] if args.only else list(bench.keys())

    # User-supplied prices (already fixed across all instances by construction)
    price_vec = None
    if args.price_json:
        price_vec = load_price_vector(Path(args.price_json))

    # NEW: frozen price profiles (only if no price_json)
    price_seed = int(args.seed if args.price_seed is None else args.price_seed)
    maxK = compute_maxK(bench, only)
    master_prices, perK_cache = frozen_prices_factory(
        mode=args.price_mode,
        low=args.price_low,
        high=args.price_high,
        freeze_scope=args.price_freeze_scope,
        price_seed=price_seed,
        maxK=maxK,
    )

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "instance_id", "category", "N", "M", "D", "H", "K",
                "seed", "replicate", "path"
            ],
        )
        w.writeheader()

        for category in only:
            if category not in bench:
                raise ValueError(f"Unknown category '{category}'. Available: {list(bench.keys())}")

            spec = bench[category]
            Ns = list(spec["Ns"])
            Ms = list(spec["Ms"])
            Ds = list(spec["Ds"])
            u_low, u_high = spec["u_range"]
            target_util = float(spec["target_util"])

            cat_dir = out_dir / category
            ensure_dir(cat_dir)

            for N in Ns:
                for M in Ms:
                    for D in Ds:
                        K = HOURS_PER_DAY * int(D)

                        # If freezing is on and no --price-json, compute c once per scope
                        frozen_c_for_this_K: Optional[List[float]] = None
                        if args.price_freeze and price_vec is None and args.price_mode == "random_uniform":
                            lo = int(math.floor(args.price_low))
                            hi = int(math.floor(args.price_high))

                            if args.price_freeze_scope == "daily":
                                rngp = np.random.default_rng(price_seed)
                                daily = rngp.integers(lo, hi + 1, size=HOURS_PER_DAY).astype(float).tolist()
                                frozen_c_for_this_K = (daily * int(D))[:K]

                            elif args.price_freeze_scope == "per_K":
                                if K not in perK_cache:
                                    rngp = np.random.default_rng(price_seed + int(K))
                                    perK_cache[K] = rngp.integers(lo, hi + 1, size=K).astype(float).tolist()
                                frozen_c_for_this_K = perK_cache[K]

                            elif args.price_freeze_scope == "master_prefix":
                                if master_prices is None or len(master_prices) < K:
                                    raise RuntimeError("Internal error: master_prefix prices not initialized or too short.")
                                frozen_c_for_this_K = master_prices[:K]

                            else:
                                raise ValueError(f"Unknown price_freeze_scope: {args.price_freeze_scope}")

                        for r in range(1, args.replicates + 1):
                            seed = int(args.seed + (hash((category, N, M, D, r)) % 10_000_000))
                            rng = np.random.default_rng(seed)

                            p = sample_processing_times(
                                N=int(N), M=int(M), K=int(K),
                                rng=rng, pmax=int(args.pmax),
                                target_util=target_util
                            )
                            u = sample_machine_rates(M=int(M), rng=rng, u_low=int(u_low), u_high=int(u_high))

                            if frozen_c_for_this_K is not None:
                                c = frozen_c_for_this_K
                            else:
                                c = make_prices(
                                    K=int(K), D=int(D), rng=rng,
                                    mode=args.price_mode,
                                    price_json=price_vec,
                                    low=args.price_low, high=args.price_high
                                )

                            payload = {
                                "id": instance_id(category, int(N), int(M), int(D), r, seed),
                                "category": category,
                                "N": int(N),
                                "M": int(M),
                                "D_days": int(D),
                                "hours_per_day": HOURS_PER_DAY,
                                "slot_minutes": SLOT_MINUTES,
                                "K": int(K),
                                "p": p,   # length N
                                "u": u,   # length M
                                "c": c,   # length K
                                "meta": {
                                    "seed": seed,
                                    "replicate": r,
                                    "pmax": int(args.pmax),
                                    "target_util": target_util,
                                    "price_mode": args.price_mode,
                                    "price_json": os.path.basename(args.price_json) if args.price_json else "",
                                    "price_freeze": bool(args.price_freeze and price_vec is None),
                                    "price_freeze_scope": args.price_freeze_scope if (args.price_freeze and price_vec is None) else "",
                                    "price_seed": price_seed if (args.price_freeze and price_vec is None) else None,
                                },
                            }

                            if not args.no_periods:
                                payload["periods"] = compress_periods(payload["c"])

                            inst_path = cat_dir / f"{payload['id']}.json"
                            write_instance_json(inst_path, payload)

                            w.writerow({
                                "instance_id": payload["id"],
                                "category": category,
                                "N": payload["N"],
                                "M": payload["M"],
                                "D": payload["D_days"],
                                "H": payload["hours_per_day"],
                                "K": payload["K"],
                                "seed": seed,
                                "replicate": r,
                                "path": str(inst_path.relative_to(out_dir)),
                            })


if __name__ == "__main__":
    main()

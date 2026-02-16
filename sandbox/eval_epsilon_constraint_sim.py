from __future__ import annotations

"""Simulate epsilon-constraint (deadline) deployment with multi-machine decomposition.

This script:
1) Generates parallel-machine instances using the same configuration as
   `New Benchmark/new_data.py` (loaded dynamically by filepath).
2) Assigns jobs to machines in a biased-random way so loads are imbalanced
   (cheaper machines get more jobs).
3) Sweeps a makespan/deadline constraint epsilon from K down to the minimum
   feasible value (for the fixed assignment), using an accelerated update
   epsilon <- Cmax-1.
4) Solves each machine subproblem with the single-machine DP, optionally using
   the learned Vhat-guided beam DP for speed.

Notes
-----
- Here epsilon means a deadline / makespan constraint T_limit (same semantics as
  `PaST/cli/run_full_pipeline.py`). We enforce it by truncating the price vector
  to `prices[:epsilon]`.
- Electricity price c_t is global; machine heterogeneity is modeled via a
  per-machine energy rate u_m. The energy price vector for machine m is
  `u_m * c_t`.
- If you use a learned Vhat model trained on *base* prices c_t, you can reuse it
  across machines by multiplying predictions by u_m (since costs scale linearly).

Example
-------
python PaST/sandbox/eval_epsilon_constraint_sim.py \
  --category small --N 40 --M 5 --D 3 --replicates 5 --seed 123 \
  --price-mode daily_tou --assign-alpha 1.2 \
  --guided --beam 50 --prune-factor 2.0 \
  --load-model checkpoints/vhat_small.npz --transferable-features --normalize \
  --out-csv PaST/logs/epsilon_sim.csv
"""

import argparse
import csv
import importlib.util
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Allow running this script directly: `python PaST/sandbox/...`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Local imports
from PaST.solvers.optimal_benchmark_dp import solve_optimal_benchmark_dp
from PaST.solvers.vhat_linear import FeatureSpec, LinearRidgeValueModel
from PaST.solvers.vhat_models import PolyRidgeValueModel, MLPValueModel, LGBMValueModel
from PaST.solvers.vhat_tou_features import build_tou_feature_context


def encode_setup(p_list: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lengths, totals, radices) for multiset state encoding."""
    p_arr = np.asarray(list(p_list), dtype=np.int32)
    if p_arr.size == 0:
        return (
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
        )
    lengths, inv = np.unique(p_arr, return_inverse=True)
    totals = np.bincount(inv, minlength=len(lengths)).astype(np.int32, copy=False)
    radices = (totals + 1).astype(np.int32, copy=False)
    return lengths.astype(np.int32, copy=False), totals, radices


def decode_state(state: int, radices: np.ndarray) -> Tuple[int, ...]:
    K = int(len(radices))
    u = [0] * K
    x = int(state)
    for i in range(K):
        r = int(radices[i])
        u[i] = x % r
        x //= r
    return tuple(u)


@dataclass(frozen=True)
class InstancePM:
    """Parallel-machine instance."""

    seed: int
    category: str
    N: int
    M: int
    D: int
    K: int
    p: List[int]
    u: List[int]
    c: List[float]


class _ValueModelLike:
    """Minimal protocol-like base for value models used by this script."""

    def predict_from_used(
        self,
        *,
        t: int,
        used: Sequence[int],
        totals: np.ndarray,
        lengths: Sequence[int],
        ctx,
    ) -> float:
        raise NotImplementedError


def _make_generate_data_daily_prices(
    *,
    seed: int,
    T: int = 20,
    Tk_choices: Sequence[int] = (2, 3, 5),
    ck_low: int = 1,
    ck_high: int = 8,
) -> List[float]:
    """Generate a length-T daily price vector using generate_data.py-style intervals.

    This keeps the "20 hours repeating" structure (we generate a 20-slot day and
    repeat it for D days), but changes the within-day profile distribution.
    """
    import random

    if T <= 0:
        raise ValueError("T must be positive")
    if ck_low > ck_high:
        raise ValueError("ck_low must be <= ck_high")

    rng = random.Random(int(seed))

    # Sample interval durations summing exactly to T (simple restart-on-stuck)
    while True:
        remaining = int(T)
        Tk: List[int] = []
        while remaining > 0:
            feasible = [int(x) for x in Tk_choices if int(x) <= remaining]
            if not feasible:
                break
            dur = int(rng.choice(feasible))
            Tk.append(dur)
            remaining -= dur
        if remaining == 0 and Tk:
            break

    ck = [int(rng.randint(int(ck_low), int(ck_high))) for _ in range(len(Tk))]
    ct: List[float] = []
    for dur, price in zip(Tk, ck):
        ct.extend([float(price)] * int(dur))
    if len(ct) != int(T):
        raise RuntimeError("Internal error: generated daily profile has wrong length")
    return ct


def _parse_int_range(s: str) -> Tuple[int, int]:
    """Parse 'a-b' into (a,b), inclusive."""
    s = str(s).strip()
    if "-" not in s:
        raise ValueError(f"Expected range like 'a-b', got: {s}")
    a, b = s.split("-", 1)
    lo = int(a.strip())
    hi = int(b.strip())
    if lo <= 0 or hi <= 0:
        raise ValueError(f"Range bounds must be positive, got: {lo}-{hi}")
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _load_new_benchmark_module() -> object:
    """Load `New Benchmark/new_data.py` by filepath (folder contains a space)."""
    root = Path(__file__).resolve().parents[2]
    mod_path = root / "New Benchmark" / "new_data.py"
    if not mod_path.exists():
        raise FileNotFoundError(f"Cannot find new benchmark generator at: {mod_path}")

    spec = importlib.util.spec_from_file_location(
        "new_benchmark_new_data", str(mod_path)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for: {mod_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - float(np.max(x))
    e = np.exp(x)
    s = float(np.sum(e))
    if s <= 0:
        return np.full_like(e, 1.0 / len(e))
    return e / s


def _is_repeating(prices: np.ndarray, *, H: int = 20, tol: float = 1e-9) -> bool:
    prices = np.asarray(prices, dtype=np.float64)
    if prices.size < H:
        return False
    day = prices[:H]
    for t in range(int(prices.size)):
        if abs(float(prices[t]) - float(day[t % H])) > tol:
            return False
    return True


def biased_random_assignment(
    *,
    p: Sequence[int],
    u: Sequence[int],
    K: int,
    rng: np.random.Generator,
    alpha: float,
    uniform_mix: float,
) -> List[List[int]]:
    """Assign jobs to machines with load imbalance.

    We bias assignment toward machines with lower energy rates u ("cheaper").

    Post-processing rebalances if any machine exceeds horizon K (so instance is
    feasible at epsilon=K).
    """

    M = int(len(u))
    u_arr = np.asarray(u, dtype=np.float64)
    logits = -float(alpha) * (u_arr - float(np.min(u_arr)))
    probs = _softmax(logits)

    mix = float(uniform_mix)
    if mix < 0.0 or mix > 1.0:
        raise ValueError(f"uniform_mix must be in [0,1], got: {mix}")
    if mix > 0.0:
        probs = (1.0 - mix) * probs + mix * (1.0 / float(M))

    assignments: List[List[int]] = [[] for _ in range(M)]
    loads = np.zeros(M, dtype=np.int64)

    # Initial biased random assignment
    for j, pj in enumerate(p):
        m = int(rng.choice(M, p=probs))
        assignments[m].append(j)
        loads[m] += int(pj)

    # Rebalance to ensure loads[m] <= K for all m
    # Keep imbalance: move only from overloaded machines.
    max_iter = 2_000_000
    it = 0
    while int(np.max(loads)) > int(K):
        it += 1
        if it > max_iter:
            raise RuntimeError(
                "Rebalancing guard triggered; try smaller alpha or lower target_util."
            )

        over = int(np.argmax(loads))
        under = int(np.argmin(loads))
        if over == under:
            break

        # Move a random job from overloaded machine to underloaded one.
        if not assignments[over]:
            break
        idx = int(rng.integers(0, len(assignments[over])))
        job = assignments[over].pop(idx)
        pj = int(p[job])
        assignments[under].append(job)
        loads[over] -= pj
        loads[under] += pj

    return assignments


def _machine_job_p(p: Sequence[int], job_indices: Sequence[int]) -> List[int]:
    return [int(p[j]) for j in job_indices]


def _solve_machine(
    *,
    p_m: List[int],
    prices_scaled: np.ndarray,
    guided: bool,
    beam: int,
    prune_factor: float,
    vhat_fn,
) -> Tuple[float, int, float]:
    """Solve a single-machine subproblem; returns (cost, finish_time, wall_s)."""
    t0 = time.perf_counter()
    res = solve_optimal_benchmark_dp(
        p_m,
        prices_scaled,
        tie_break="early",
        guided=guided,
        beam_width=int(beam),
        prune_factor=float(prune_factor),
        vhat=vhat_fn,
    )
    wall = time.perf_counter() - t0
    if not res.feasible:
        return float("inf"), int(prices_scaled.shape[0]), wall
    return float(res.cost), int(res.finish_time), wall


def _load_vhat_checkpoint(path: str, fallback_spec: FeatureSpec) -> _ValueModelLike:
    """Load a pooled Vhat checkpoint.

    Supports linear (legacy), poly, mlp, and lgbm checkpoints.
    """
    ckpt = np.load(path, allow_pickle=True)

    # Newer checkpoints include model_type.
    model_type = None
    if "model_type" in ckpt.files:
        try:
            model_type = str(ckpt["model_type"])  # numpy scalar/array -> str
        except Exception:
            model_type = None

    if model_type is not None:
        mt = model_type.strip().lower()
        if mt == "poly":
            return PolyRidgeValueModel.load(path)
        if mt == "mlp":
            return MLPValueModel.load(path)
        if mt == "lgbm":
            return LGBMValueModel.load(path)
        if mt == "linear":
            # Fall through to legacy linear loader
            pass

    # Heuristics for older checkpoints without model_type.
    if "powers" in ckpt.files and "weights" in ckpt.files:
        return PolyRidgeValueModel.load(path)
    if {"W1", "b1", "W2", "b2", "W3", "b3"}.issubset(set(ckpt.files)):
        return MLPValueModel.load(path)

    # Default: legacy linear ridge
    w = np.asarray(ckpt["weights"], dtype=np.float64)
    spec = fallback_spec
    if {
        "include_per_class_counts",
        "include_per_class_now_cost",
        "include_bins",
    }.issubset(set(ckpt.files)):
        norm = (
            bool(int(ckpt["normalize"]))
            if "normalize" in ckpt.files
            else bool(fallback_spec.normalize)
        )
        spec = FeatureSpec(
            include_per_class_counts=bool(int(ckpt["include_per_class_counts"])),
            include_per_class_now_cost=bool(int(ckpt["include_per_class_now_cost"])),
            include_bins=bool(int(ckpt["include_bins"])),
            normalize=norm,
        )
    return LinearRidgeValueModel(weights=w, spec=spec)


def _load_vhat_checkpoint_meta(path: str) -> Dict[str, bool]:
    """Load auxiliary metadata saved in pooled checkpoints (if present)."""
    def _try_load(p: str) -> Optional[Dict[str, bool]]:
        try:
            ck = np.load(p, allow_pickle=True)
        except Exception:
            return None
        if "normalize_labels" in ck.files:
            try:
                return {"normalize_labels": bool(int(ck["normalize_labels"]))}
            except Exception:
                return {"normalize_labels": False}
        return None

    # 1) Try main checkpoint first
    out = _try_load(path)
    if out is not None:
        return out

    # 2) Try sidecars written by eval_pooled_vhat.py (new and legacy names)
    candidates = [
        str(path) + ".meta.npz",
        str(path) + ".meta",
        str(path) + ".meta.npz.npz",
    ]
    for cand in candidates:
        out = _try_load(cand)
        if out is not None:
            return out

    return {"normalize_labels": False}


def generate_instances(
    *,
    module,
    category: str,
    N: int,
    M: int,
    D: int,
    replicates: int,
    seed: int,
    pmax: int,
    target_util: float,
    price_mode: str,
    price_low: float,
    price_high: float,
    price_freeze: bool,
    price_freeze_scope: str,
    price_seed: Optional[int],
    daily_price_json: Optional[List[float]] = None,
    N_range: Optional[Tuple[int, int]] = None,
    D_range: Optional[Tuple[int, int]] = None,
    M_range: Optional[Tuple[int, int]] = None,
) -> List[InstancePM]:
    """Generate instances in-memory using new benchmark helper functions."""

    HOURS_PER_DAY = int(getattr(module, "HOURS_PER_DAY"))

    # When D varies per replicate, precompute maxK for optional price freezing.
    maxK = HOURS_PER_DAY * (int(D_range[1]) if D_range is not None else int(D))

    # Optional freezing of prices (only relevant for random_uniform AND when no explicit daily profile is provided)
    master_prices = None
    perK_cache: Dict[int, List[float]] = {}
    _price_seed = int(seed if price_seed is None else price_seed)
    if (
        price_freeze
        and price_mode == "random_uniform"
        and daily_price_json is None
        and hasattr(module, "frozen_prices_factory")
    ):
        master_prices, perK_cache = module.frozen_prices_factory(
            mode=price_mode,
            low=price_low,
            high=price_high,
            freeze_scope=price_freeze_scope,
            price_seed=_price_seed,
            maxK=maxK,
        )

    instances: List[InstancePM] = []

    for r in range(1, int(replicates) + 1):
        rng_outer = np.random.default_rng(int(seed) + 1_000_003 * int(r))
        N_r = (
            int(rng_outer.integers(int(N_range[0]), int(N_range[1]) + 1))
            if N_range is not None
            else int(N)
        )
        D_r = (
            int(rng_outer.integers(int(D_range[0]), int(D_range[1]) + 1))
            if D_range is not None
            else int(D)
        )
        M_r = (
            int(rng_outer.integers(int(M_range[0]), int(M_range[1]) + 1))
            if M_range is not None
            else int(M)
        )

        K_r = int(HOURS_PER_DAY * int(D_r))
        inst_seed = (
            int(seed)
            + 10_000 * int(r)
            + 97 * int(N_r)
            + 389 * int(M_r)
            + 7919 * int(D_r)
        )
        rng = np.random.default_rng(int(inst_seed))

        p = module.sample_processing_times(
            N=int(N_r),
            M=int(M_r),
            K=int(K_r),
            rng=rng,
            pmax=int(pmax),
            target_util=float(target_util),
        )

        # Category-based default u_range if available.
        if hasattr(module, "DEFAULT_BENCHMARK") and category in getattr(
            module, "DEFAULT_BENCHMARK"
        ):
            u_low, u_high = module.DEFAULT_BENCHMARK[category]["u_range"]
        else:
            u_low, u_high = 1, 3
        u = module.sample_machine_rates(
            M=int(M_r), rng=rng, u_low=int(u_low), u_high=int(u_high)
        )

        frozen_c_for_this_K: Optional[List[float]] = None
        if price_freeze and price_mode == "random_uniform" and daily_price_json is None:
            lo = int(math.floor(price_low))
            hi = int(math.floor(price_high))
            if price_freeze_scope == "daily":
                rngp = np.random.default_rng(_price_seed)
                daily = (
                    rngp.integers(lo, hi + 1, size=HOURS_PER_DAY).astype(float).tolist()
                )
                frozen_c_for_this_K = (daily * int(D_r))[:K_r]
            elif price_freeze_scope == "per_K":
                if K_r not in perK_cache:
                    rngp = np.random.default_rng(_price_seed + int(K_r))
                    perK_cache[K_r] = (
                        rngp.integers(lo, hi + 1, size=K_r).astype(float).tolist()
                    )
                frozen_c_for_this_K = perK_cache[K_r]
            elif price_freeze_scope == "master_prefix":
                if master_prices is None or len(master_prices) < K_r:
                    raise RuntimeError(
                        "master_prefix prices not initialized or too short"
                    )
                frozen_c_for_this_K = master_prices[:K_r]
            else:
                raise ValueError(f"Unknown price_freeze_scope: {price_freeze_scope}")

        if frozen_c_for_this_K is not None:
            c = frozen_c_for_this_K
        else:
            c = module.make_prices(
                K=int(K_r),
                D=int(D_r),
                rng=rng,
                mode=str(price_mode),
                price_json=daily_price_json,
                low=float(price_low),
                high=float(price_high),
            )

        instances.append(
            InstancePM(
                seed=int(inst_seed),
                category=str(category),
                N=int(N_r),
                M=int(M_r),
                D=int(D_r),
                K=int(K_r),
                p=[int(x) for x in p],
                u=[int(x) for x in u],
                c=[float(x) for x in c],
            )
        )

    return instances


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Epsilon-constraint simulation (multi-machine, biased assignment, epsilon sweep)."
    )

    ap.add_argument(
        "--category",
        type=str,
        default="small",
        choices=["small", "medium", "large"],
        help="Benchmark category (used only for metadata / default u_range).",
    )
    ap.add_argument("--N", type=int, default=40)
    ap.add_argument(
        "--N-range",
        type=str,
        default="",
        help="Optional inclusive range 'a-b'. If set, sample N per replicate.",
    )
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument(
        "--M-range",
        type=str,
        default="",
        help="Optional inclusive range 'a-b'. If set, sample M per replicate.",
    )
    ap.add_argument("--D", type=int, default=3)
    ap.add_argument(
        "--D-range",
        type=str,
        default="",
        help="Optional inclusive range 'a-b'. If set, sample D per replicate.",
    )
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--pmax", type=int, default=12)
    ap.add_argument("--target-util", type=float, default=0.80)

    ap.add_argument(
        "--price-mode",
        type=str,
        default="daily_tou",
        choices=["daily_tou", "random_uniform"],
    )
    ap.add_argument("--price-low", type=float, default=1.0)
    ap.add_argument("--price-high", type=float, default=8.0)
    ap.add_argument("--price-freeze", action="store_true")
    ap.add_argument(
        "--price-freeze-scope",
        type=str,
        default="daily",
        choices=["daily", "per_K", "master_prefix"],
    )
    ap.add_argument("--price-seed", type=int, default=None)

    ap.add_argument(
        "--daily-price-profile",
        type=str,
        default="daily_tou",
        choices=["daily_tou", "generate_data"],
        help=(
            "Which 20-hour repeating daily profile to use. "
            "daily_tou uses New Benchmark/new_data.py built-in deterministic TOU; "
            "generate_data samples a 20-slot day using generate_data.py-style intervals and repeats it."
        ),
    )
    ap.add_argument(
        "--gd-seed",
        type=int,
        default=20260109,
        help="Seed for --daily-price-profile=generate_data.",
    )
    ap.add_argument(
        "--gd-ck-low",
        type=int,
        default=1,
        help="Min interval price for --daily-price-profile=generate_data.",
    )
    ap.add_argument(
        "--gd-ck-high",
        type=int,
        default=8,
        help="Max interval price for --daily-price-profile=generate_data.",
    )

    ap.add_argument(
        "--assign-alpha",
        type=float,
        default=1.0,
        help="Assignment bias strength; higher -> more jobs on cheaper (low-u) machines.",
    )

    ap.add_argument(
        "--assign-uniform-mix",
        type=float,
        default=0.0,
        help=(
            "Add stochasticity by mixing assignment probs with uniform. "
            "0.0 = pure biased-softmax, 1.0 = fully uniform."
        ),
    )

    ap.add_argument(
        "--epsilon-step",
        type=int,
        default=0,
        help="If >0, do a full sweep epsilon=K,K-step,... instead of accelerated epsilon update.",
    )

    ap.add_argument(
        "--guided",
        action="store_true",
        help="Use Vhat-guided beam DP for each machine.",
    )
    ap.add_argument("--beam", type=int, default=50)
    ap.add_argument("--prune-factor", type=float, default=2.0)

    ap.add_argument(
        "--load-model",
        type=str,
        default="",
        help=".npz checkpoint for LinearRidgeValueModel (required if --guided).",
    )
    ap.add_argument("--transferable-features", action="store_true")
    ap.add_argument("--normalize", action="store_true")

    ap.add_argument(
        "--out-csv", type=str, default="PaST/logs/epsilon_constraint_sim.csv"
    )

    ap.add_argument(
        "--skip-exact",
        action="store_true",
        help="If set, do not compute exact DP baseline; only run guided (or exact if not guided).",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if bool(args.skip_exact) and not bool(args.guided):
        raise ValueError("--skip-exact requires --guided")

    module = _load_new_benchmark_module()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Feature spec for Vhat
    if bool(args.transferable_features):
        base_spec = FeatureSpec(
            include_per_class_counts=False,
            include_per_class_now_cost=False,
            include_bins=True,
            normalize=bool(args.normalize),
        )
    else:
        base_spec = FeatureSpec(
            include_per_class_counts=True,
            include_per_class_now_cost=True,
            include_bins=True,
            normalize=bool(args.normalize),
        )

    model: Optional[_ValueModelLike] = None
    model_meta: Dict[str, bool] = {"normalize_labels": False}
    if bool(args.guided):
        if not str(args.load_model).strip():
            raise ValueError("--guided requires --load-model")
        model = _load_vhat_checkpoint(str(args.load_model).strip(), base_spec)
        model_meta = _load_vhat_checkpoint_meta(str(args.load_model).strip())

    N_range = None
    if str(args.N_range).strip():
        N_range = _parse_int_range(str(args.N_range))

    D_range = None
    if str(args.D_range).strip():
        D_range = _parse_int_range(str(args.D_range))

    M_range = None
    if str(args.M_range).strip():
        M_range = _parse_int_range(str(args.M_range))

    daily_price_json = None
    if str(args.daily_price_profile).strip().lower() == "generate_data":
        daily_price_json = _make_generate_data_daily_prices(
            seed=int(args.gd_seed),
            T=20,
            Tk_choices=(2, 3, 5),
            ck_low=int(args.gd_ck_low),
            ck_high=int(args.gd_ck_high),
        )

    instances = generate_instances(
        module=module,
        category=str(args.category),
        N=int(args.N),
        M=int(args.M),
        D=int(args.D),
        replicates=int(args.replicates),
        seed=int(args.seed),
        pmax=int(args.pmax),
        target_util=float(args.target_util),
        price_mode=str(args.price_mode),
        price_low=float(args.price_low),
        price_high=float(args.price_high),
        price_freeze=bool(args.price_freeze),
        price_freeze_scope=str(args.price_freeze_scope),
        price_seed=args.price_seed,
        daily_price_json=daily_price_json,
        N_range=N_range,
        D_range=D_range,
        M_range=M_range,
    )

    fieldnames = [
        "instance_seed",
        "category",
        "N",
        "M",
        "D",
        "K",
        "assign_alpha",
        "assign_uniform_mix",
        "epsilon",
        "loads",  # semicolon separated per-machine sums
        "u",  # semicolon separated
        "total_energy",
        "makespan",
        "solve_s",
        "method",  # exact/guided
        "beam",
        "prune_factor",
    ]

    print(
        "[epsilon] === RUN CONFIG ===\n"
        f"[epsilon] out_csv={out_csv}\n"
        f"[epsilon] category={args.category} replicates={args.replicates} seed={args.seed}\n"
        f"[epsilon] N={args.N} N_range={N_range}  M={args.M}  D={args.D} D_range={D_range}\n"
        f"[epsilon] pmax={args.pmax} target_util={args.target_util}\n"
        f"[epsilon] price_mode={args.price_mode} low={args.price_low} high={args.price_high} "
        f"freeze={bool(args.price_freeze)} freeze_scope={args.price_freeze_scope} price_seed={args.price_seed}\n"
        f"[epsilon] assign_alpha={args.assign_alpha} assign_uniform_mix={args.assign_uniform_mix}\n"
        f"[epsilon] guided={bool(args.guided)} skip_exact={bool(args.skip_exact)} beam={args.beam} prune_factor={args.prune_factor}\n"
        f"[epsilon] load_model={str(args.load_model).strip() if bool(args.guided) else ''} "
        f"normalize_labels={bool(model_meta.get('normalize_labels', False)) if model_meta is not None else False} "
        f"spec={base_spec}"
    )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for inst in instances:
            rng = np.random.default_rng(int(inst.seed) + 999)
            assignments = biased_random_assignment(
                p=inst.p,
                u=inst.u,
                K=int(inst.K),
                rng=rng,
                alpha=float(args.assign_alpha),
                uniform_mix=float(args.assign_uniform_mix),
            )
            loads = [int(sum(inst.p[j] for j in jobs)) for jobs in assignments]
            min_eps = max(1, int(max(loads)) if loads else 1)

            # Epsilon sweep
            eps = int(inst.K)
            step = int(args.epsilon_step)

            while eps >= min_eps:
                prices_base = np.asarray(inst.c[:eps], dtype=np.float64)
                ctx = build_tou_feature_context(
                    prices_base,
                    H=20,
                    validate_repeating=_is_repeating(prices_base, H=20),
                )

                # If the model was trained with label normalization, it predicts
                # (cost_to_go / remaining_budget). Denormalize by remaining budget.
                prefix_prices = np.concatenate(
                    [[0.0], np.cumsum(prices_base, dtype=np.float64)]
                )

                total_energy_exact = 0.0
                total_energy_guided = 0.0
                makespan_exact = 0
                makespan_guided = 0

                # Solve per machine
                exact_solve_s = 0.0
                guided_solve_s = 0.0

                for m, job_indices in enumerate(assignments):
                    p_m = _machine_job_p(inst.p, job_indices)
                    u_m = float(inst.u[m])
                    prices_scaled = (prices_base * u_m).astype(np.float64)

                    # Build Vhat for this machine (scale base model by u_m)
                    vhat_fn = None
                    if model is not None and len(p_m) > 0:
                        lengths, totals, radices = encode_setup(p_m)
                        used_cache: Dict[int, Tuple[int, ...]] = {
                            0: tuple([0] * len(lengths))
                        }

                        def vhat(t: int, state: int) -> float:
                            s = int(state)
                            used = used_cache.get(s)
                            if used is None:
                                used = decode_state(s, radices)
                                used_cache[s] = used
                            # Predict in base-price units and scale by u_m
                            val = model.predict_from_used(
                                t=int(t),
                                used=used,
                                totals=totals,
                                lengths=lengths.tolist(),
                                ctx=ctx,
                            )

                            if bool(model_meta.get("normalize_labels", False)):
                                tt = max(0, min(int(t), int(prices_base.shape[0])))
                                rem_budget = float(
                                    prefix_prices[int(prices_base.shape[0])]
                                    - prefix_prices[tt]
                                )
                                val = float(val) * rem_budget

                            return float(u_m) * float(val)

                        vhat_fn = vhat

                    # guided DP requires a callable vhat; for empty machines, use 0.
                    if bool(args.guided) and vhat_fn is None:
                        vhat_fn = lambda _t, _s: 0.0

                    if not bool(args.guided):
                        # Exact only
                        cost, ft, wall = _solve_machine(
                            p_m=p_m,
                            prices_scaled=prices_scaled,
                            guided=False,
                            beam=int(args.beam),
                            prune_factor=float(args.prune_factor),
                            vhat_fn=None,
                        )
                        total_energy_exact += float(cost)
                        makespan_exact = max(makespan_exact, int(ft))
                        exact_solve_s += float(wall)
                    elif bool(args.skip_exact):
                        # Guided only
                        cost_g, ft_g, wall_g = _solve_machine(
                            p_m=p_m,
                            prices_scaled=prices_scaled,
                            guided=True,
                            beam=int(args.beam),
                            prune_factor=float(args.prune_factor),
                            vhat_fn=vhat_fn,
                        )
                        total_energy_guided += float(cost_g)
                        makespan_guided = max(makespan_guided, int(ft_g))
                        guided_solve_s += float(wall_g)
                    else:
                        # Both methods: exact baseline and guided
                        cost_e, ft_e, wall_e = _solve_machine(
                            p_m=p_m,
                            prices_scaled=prices_scaled,
                            guided=False,
                            beam=int(args.beam),
                            prune_factor=float(args.prune_factor),
                            vhat_fn=None,
                        )
                        total_energy_exact += float(cost_e)
                        makespan_exact = max(makespan_exact, int(ft_e))
                        exact_solve_s += float(wall_e)

                        cost_g, ft_g, wall_g = _solve_machine(
                            p_m=p_m,
                            prices_scaled=prices_scaled,
                            guided=True,
                            beam=int(args.beam),
                            prune_factor=float(args.prune_factor),
                            vhat_fn=vhat_fn,
                        )
                        total_energy_guided += float(cost_g)
                        makespan_guided = max(makespan_guided, int(ft_g))
                        guided_solve_s += float(wall_g)

                loads_str = ";".join(str(x) for x in loads)
                u_str = ";".join(str(x) for x in inst.u)

                if not bool(args.guided):
                    w.writerow(
                        {
                            "instance_seed": float(inst.seed),
                            "category": inst.category,
                            "N": float(inst.N),
                            "M": float(inst.M),
                            "D": float(inst.D),
                            "K": float(inst.K),
                            "assign_alpha": float(args.assign_alpha),
                            "assign_uniform_mix": float(args.assign_uniform_mix),
                            "epsilon": float(eps),
                            "loads": loads_str,
                            "u": u_str,
                            "total_energy": float(total_energy_exact),
                            "makespan": float(makespan_exact),
                            "solve_s": float(exact_solve_s),
                            "method": "exact",
                            "beam": float(args.beam),
                            "prune_factor": float(args.prune_factor),
                        }
                    )
                    cur_mk = int(makespan_exact)
                elif bool(args.skip_exact):
                    w.writerow(
                        {
                            "instance_seed": float(inst.seed),
                            "category": inst.category,
                            "N": float(inst.N),
                            "M": float(inst.M),
                            "D": float(inst.D),
                            "K": float(inst.K),
                            "assign_alpha": float(args.assign_alpha),
                            "assign_uniform_mix": float(args.assign_uniform_mix),
                            "epsilon": float(eps),
                            "loads": loads_str,
                            "u": u_str,
                            "total_energy": float(total_energy_guided),
                            "makespan": float(makespan_guided),
                            "solve_s": float(guided_solve_s),
                            "method": "guided",
                            "beam": float(args.beam),
                            "prune_factor": float(args.prune_factor),
                        }
                    )
                    cur_mk = int(makespan_guided)
                else:
                    # wrote nothing yet; write both
                    w.writerow(
                        {
                            "instance_seed": float(inst.seed),
                            "category": inst.category,
                            "N": float(inst.N),
                            "M": float(inst.M),
                            "D": float(inst.D),
                            "K": float(inst.K),
                            "assign_alpha": float(args.assign_alpha),
                            "assign_uniform_mix": float(args.assign_uniform_mix),
                            "epsilon": float(eps),
                            "loads": loads_str,
                            "u": u_str,
                            "total_energy": float(total_energy_exact),
                            "makespan": float(makespan_exact),
                            "solve_s": float(exact_solve_s),
                            "method": "exact",
                            "beam": float(args.beam),
                            "prune_factor": float(args.prune_factor),
                        }
                    )
                    w.writerow(
                        {
                            "instance_seed": float(inst.seed),
                            "category": inst.category,
                            "N": float(inst.N),
                            "M": float(inst.M),
                            "D": float(inst.D),
                            "K": float(inst.K),
                            "assign_alpha": float(args.assign_alpha),
                            "assign_uniform_mix": float(args.assign_uniform_mix),
                            "epsilon": float(eps),
                            "loads": loads_str,
                            "u": u_str,
                            "total_energy": float(total_energy_guided),
                            "makespan": float(makespan_guided),
                            "solve_s": float(guided_solve_s),
                            "method": "guided",
                            "beam": float(args.beam),
                            "prune_factor": float(args.prune_factor),
                        }
                    )
                    cur_mk = int(makespan_exact)

                print(
                    f"inst_seed={inst.seed} eps={eps} min_eps={min_eps} "
                    f"loads=[{loads_str}] u=[{u_str}] "
                    + (
                        f"exact(E={total_energy_exact:.2f},mk={makespan_exact},t={exact_solve_s:.2f}s) "
                        if (not bool(args.guided)) or (not bool(args.skip_exact))
                        else ""
                    )
                    + (
                        f"guided(E={total_energy_guided:.2f},mk={makespan_guided},t={guided_solve_s:.2f}s)"
                        if bool(args.guided)
                        else ""
                    )
                )

                if step > 0:
                    eps -= step
                else:
                    # Accelerated epsilon update: next epsilon = Cmax-1
                    eps = int(cur_mk) - 1

                if eps <= 0:
                    break

    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()

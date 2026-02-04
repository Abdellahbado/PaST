"""Learnability audit for PaST sequence-cost tabular dataset.

This script answers:
- Is there meaningful within-instance variation in DP costs (ranking even possible)?
- Do the current features make the task learnable for:
  - regression (predict DP cost)
  - ranking (pick best sequence per instance)

It is designed to run quickly on a *subset* of a large dataset.

Example:
  python tools/seq_cost_learnability_audit.py \
    --data artifacts/seq_cost_big/seq_cost_shard0.npz,artifacts/seq_cost_big/seq_cost_shard1.npz \
    --max_instances 5000 \
    --seed 1337

Notes:
- Reads compressed .npz shards produced by PaST.train_sequence_cost_tabular.
- Uses optional dependencies if installed:
  - scikit-learn (Ridge regression)
  - lightgbm (LGBMRegressor and LGBMRanker)
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _parse_paths(arg: str) -> List[Path]:
    parts = [p.strip() for p in (arg or "").split(",") if p.strip()]
    if not parts:
        raise ValueError("--data is empty")
    return [Path(p) for p in parts]


def _try_import_sklearn():
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        return Ridge, StandardScaler
    except Exception:
        return None


def _try_import_lightgbm():
    try:
        import lightgbm as lgb

        return lgb
    except Exception:
        return None


def _infer_feature_layout(
    d: int, meta: Optional[np.ndarray]
) -> Tuple[int, int, int, int, int, bool]:
    """Infer feature layout.

    Returns (n_jobs_pad, window_feats_dim, proxy_cost_dim, ct_bins, price_quant_dim, enhanced).

    Supported layouts:
    - legacy:   seq_block(n_jobs_pad) + ct_bins + scalars(4)
    - enhanced: seq_block(n_jobs_pad) + window_feats(W) + proxy_cost(P) + ct_bins + price_quant(Q) + scalars(4)
    """
    n_jobs_pad = None
    ct_bins = None
    window_dim = 4  # Enhanced default
    proxy_dim = 1  # Enhanced default
    price_quant_dim = 5  # Enhanced default
    enhanced = False

    if meta is not None:
        try:
            # meta is object array of strings like "ct_bins=64"
            meta_s = [str(x) for x in meta.tolist()]
            for s in meta_s:
                if s.startswith("feature_layout=") and "window_feats" in s:
                    enhanced = True
                    m = re.search(r"window_feats\((\d+)\)", s)
                    if m:
                        window_dim = int(m.group(1))
                    m = re.search(r"proxy_cost\((\d+)\)", s)
                    if m:
                        proxy_dim = int(m.group(1))
                    m = re.search(r"price_quant\((\d+)\)", s)
                    if m:
                        price_quant_dim = int(m.group(1))
                if s.startswith("n_jobs_pad="):
                    n_jobs_pad = int(s.split("=", 1)[1])
                elif s.startswith("ct_bins="):
                    ct_bins = int(s.split("=", 1)[1])
        except Exception:
            pass

    if n_jobs_pad is not None and ct_bins is not None:
        n_jobs_pad_i = int(n_jobs_pad)
        ct_bins_i = int(ct_bins)
        d_legacy = n_jobs_pad_i + ct_bins_i + 4
        d_enh = (
            n_jobs_pad_i
            + int(window_dim)
            + int(proxy_dim)
            + ct_bins_i
            + int(price_quant_dim)
            + 4
        )
        if int(d) == int(d_enh):
            return (
                n_jobs_pad_i,
                int(window_dim),
                int(proxy_dim),
                ct_bins_i,
                int(price_quant_dim),
                True,
            )
        if int(d) == int(d_legacy):
            return n_jobs_pad_i, 0, 0, ct_bins_i, 0, False

        # If meta suggests enhanced but dimension doesn't match, still return dims but mark enhanced.
        if enhanced:
            return (
                n_jobs_pad_i,
                int(window_dim),
                int(proxy_dim),
                ct_bins_i,
                int(price_quant_dim),
                True,
            )
        return n_jobs_pad_i, 0, 0, ct_bins_i, 0, False

    # Fallback inference: try enhanced, then legacy.
    common_bins = [32, 48, 64, 96, 128]
    for b in common_bins:
        # Assume enhanced defaults when metadata is missing.
        n_enh = d - b - (int(window_dim) + int(proxy_dim) + int(price_quant_dim) + 4)
        if 8 <= n_enh <= 512:
            return (
                int(n_enh),
                int(window_dim),
                int(proxy_dim),
                int(b),
                int(price_quant_dim),
                True,
            )
        n_leg = d - b - 4
        if 8 <= n_leg <= 512:
            return int(n_leg), 0, 0, int(b), 0, False

    # Last resort: assume ct_bins=64 and enhanced.
    b = 64
    n = max(1, d - b - (int(window_dim) + int(proxy_dim) + int(price_quant_dim) + 4))
    return int(n), int(window_dim), int(proxy_dim), int(b), int(price_quant_dim), True


def _split_by_inst(
    inst_id: np.ndarray, test_frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if test_frac <= 0:
        return np.arange(len(inst_id)), np.array([], dtype=np.int64)
    if test_frac >= 1:
        return np.array([], dtype=np.int64), np.arange(len(inst_id))

    rng = np.random.default_rng(int(seed))
    uniq = np.unique(inst_id)
    rng.shuffle(uniq)
    n_test = int(math.ceil(len(uniq) * float(test_frac)))
    test_set = set(uniq[:n_test].tolist())
    is_test = np.array([i in test_set for i in inst_id], dtype=bool)
    train_idx = np.where(~is_test)[0]
    test_idx = np.where(is_test)[0]
    return train_idx, test_idx


def _within_instance_spread(y: np.ndarray, inst_id: np.ndarray) -> Dict[str, float]:
    deltas: List[float] = []
    rel_deltas: List[float] = []
    for inst in np.unique(inst_id):
        m = inst_id == inst
        if not np.any(m):
            continue
        yy = y[m]
        lo = float(np.min(yy))
        hi = float(np.max(yy))
        d = hi - lo
        deltas.append(d)
        rel_deltas.append(float(d / max(1e-9, abs(lo))))

    if not deltas:
        return {"n_instances": 0.0}

    dd = np.array(deltas, dtype=np.float64)
    rr = np.array(rel_deltas, dtype=np.float64)

    def q(x: np.ndarray, p: float) -> float:
        return float(np.quantile(x, p))

    return {
        "n_instances": float(len(dd)),
        "delta_p50": q(dd, 0.50),
        "delta_p90": q(dd, 0.90),
        "delta_p99": q(dd, 0.99),
        "rel_delta_p50": q(rr, 0.50),
        "rel_delta_p90": q(rr, 0.90),
        "rel_delta_p99": q(rr, 0.99),
    }


def _avg_regret_pick_from_scores(
    y: np.ndarray, inst_id: np.ndarray, scores: np.ndarray, *, seed: int = 0
) -> Dict[str, float]:
    regret_sum = 0.0
    hit1 = 0
    n_inst = 0
    for inst in np.unique(inst_id):
        m = inst_id == inst
        if not np.any(m):
            continue
        yy = y[m]
        ss = scores[m]
        best_true = float(np.min(yy))
        best_idx = int(np.argmin(yy))

        # Remove candidate-order artifact: if scores tie (e.g., constant within-instance
        # features), choose uniformly among argmax ties using a deterministic per-instance RNG.
        mx = float(np.max(ss))
        ties = np.flatnonzero(ss == mx)
        if ties.size <= 1:
            pick = int(np.argmax(ss))
        else:
            mix = int(seed) ^ (int(inst) * 2654435761)
            rng = np.random.default_rng(mix & 0xFFFFFFFF)
            pick = int(rng.choice(ties))

        regret_sum += float(yy[pick] - best_true)
        hit1 += 1 if pick == best_idx else 0
        n_inst += 1

    return {
        "n_instances": float(n_inst),
        "hit1_pick": float(hit1 / max(1, n_inst)),
        "avg_regret_pick": float(regret_sum / max(1, n_inst)),
    }


def _avg_regret_pick_random(
    y: np.ndarray, inst_id: np.ndarray, seed: int
) -> Dict[str, float]:
    rng = np.random.default_rng(int(seed))
    regret_sum = 0.0
    hit1 = 0
    n_inst = 0
    for inst in np.unique(inst_id):
        m = inst_id == inst
        if not np.any(m):
            continue
        yy = y[m]
        best_true = float(np.min(yy))
        best_idx = int(np.argmin(yy))
        pick = int(rng.integers(0, len(yy)))
        regret_sum += float(yy[pick] - best_true)
        hit1 += 1 if pick == best_idx else 0
        n_inst += 1

    return {
        "n_instances": float(n_inst),
        "hit1_pick": float(hit1 / max(1, n_inst)),
        "avg_regret_pick": float(regret_sum / max(1, n_inst)),
    }


def _proxy_sequential_cost_scores(
    X: np.ndarray,
    *,
    n_jobs_pad: int,
    window_dim: int,
    proxy_dim: int,
    ct_bins: int,
    price_quant_dim: int,
) -> np.ndarray:
    """Cheap heuristic score from features only.

    Uses a crude approximation:
    - reconstruct a piecewise-constant price curve from downsampled ct bins
    - schedule jobs sequentially with no gaps
    - cost ~ sum_i p_i * ct[start_i]

    Returns scores where higher=better, so we return negative cost.

    This is *not* DP; it's just a diagnostic baseline.
    """
    p_seq = X[:, : int(n_jobs_pad)].astype(np.float64)

    # Layout offsets:
    # legacy:   [p_seq(n_jobs_pad)] [ct_bins] [scalars(4)]
    # enhanced: [p_seq] [window(window_dim)] [proxy(proxy_dim)] [ct_bins] [price_quant(price_quant_dim)] [scalars(4)]
    d = int(X.shape[1])
    d_enh = (
        int(n_jobs_pad)
        + int(window_dim)
        + int(proxy_dim)
        + int(ct_bins)
        + int(price_quant_dim)
        + 4
    )
    d_leg = int(n_jobs_pad) + int(ct_bins) + 4

    is_enh = (
        d >= d_enh
        and int(window_dim) > 0
        and int(proxy_dim) > 0
        and int(price_quant_dim) > 0
    )

    if is_enh:
        ct_start = int(n_jobs_pad) + int(window_dim) + int(proxy_dim)
        ct_end = ct_start + int(ct_bins)
        scalars_start = ct_end + int(price_quant_dim)
    elif d >= d_leg:
        ct_start = int(n_jobs_pad)
        ct_end = ct_start + int(ct_bins)
        scalars_start = ct_end
    else:
        return np.zeros((X.shape[0],), dtype=np.float32)

    ct_ds = X[:, ct_start:ct_end].astype(np.float64)
    e_single = X[:, scalars_start + 0].astype(np.float64)
    T_limit = X[:, scalars_start + 1].astype(np.float64)
    n_jobs = X[:, scalars_start + 3].astype(np.float64)

    B = X.shape[0]
    out_cost = np.zeros((B,), dtype=np.float64)

    # Vectorizing full reconstruction is messy; keep it simple for audit.
    for i in range(B):
        n = int(max(0, round(float(n_jobs[i]))))
        n = min(n, n_jobs_pad)
        if n <= 0:
            out_cost[i] = 0.0
            continue

        tl = int(max(1, round(float(T_limit[i]))))
        tl = max(tl, 1)
        # Expand ct_ds to length tl by repeating bins.
        reps = int(math.ceil(tl / max(1, ct_bins)))
        ct_up = np.repeat(ct_ds[i], reps)[:tl]

        t = 0
        c = 0.0
        for pj in p_seq[i, :n]:
            p = int(max(0, round(float(pj))))
            if p <= 0:
                continue
            if t >= tl:
                # Past horizon; just clamp to last price.
                price = float(ct_up[-1])
            else:
                price = float(ct_up[t])
            c += float(p) * price
            t += p

        out_cost[i] = c * float(e_single[i])

    return (-out_cost).astype(np.float32)


def _train_eval_ridge(
    *, X: np.ndarray, y: np.ndarray, inst_id: np.ndarray, test_frac: float, seed: int
) -> Dict[str, Dict[str, float]]:
    sk = _try_import_sklearn()
    if sk is None:
        return {"status": {"ok": 0.0, "reason": 1.0}}
    Ridge, StandardScaler = sk

    train_idx, test_idx = _split_by_inst(
        inst_id, test_frac=float(test_frac), seed=int(seed)
    )
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_tr = scaler.fit_transform(X[train_idx])
    X_te = scaler.transform(X[test_idx])

    model = Ridge(alpha=1.0, random_state=int(seed))
    model.fit(X_tr, y[train_idx])

    yhat_te = model.predict(X_te).astype(np.float32)
    # RMSE
    err = yhat_te - y[test_idx]
    rmse = float(np.sqrt(np.mean(err * err)))
    # pick best predicted within instance: lower predicted cost is better
    scores = (-yhat_te).astype(np.float32)
    scores = (-yhat_te).astype(np.float32)
    m_pick = _avg_regret_pick_from_scores(
        y[test_idx], inst_id[test_idx], scores, seed=int(seed)
    )

    out = {"test": {"rmse": rmse, **m_pick}}
    return out


def _train_eval_lgbm_regressor(
    *, X: np.ndarray, y: np.ndarray, inst_id: np.ndarray, test_frac: float, seed: int
) -> Dict[str, Dict[str, float]]:
    lgb = _try_import_lightgbm()
    if lgb is None:
        return {"status": {"ok": 0.0, "reason": 1.0}}

    train_idx, test_idx = _split_by_inst(
        inst_id, test_frac=float(test_frac), seed=int(seed)
    )
    X_tr = X[train_idx]
    y_tr = y[train_idx]
    X_te = X[test_idx]
    y_te = y[test_idx]

    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=127,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=int(seed),
        force_row_wise=True,
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_te, y_te)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    yhat = model.predict(X_te).astype(np.float32)
    err = yhat - y_te
    rmse = float(np.sqrt(np.mean(err * err)))
    scores = (-yhat).astype(np.float32)
    m_pick = _avg_regret_pick_from_scores(
        y_te, inst_id[test_idx], scores, seed=int(seed)
    )

    return {"test": {"rmse": rmse, **m_pick}}


def _build_grouped_rank_data(
    X: np.ndarray, y: np.ndarray, inst_id: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X2, rel, group, y2, inst2) grouped by inst_id."""
    order = np.argsort(inst_id, kind="mergesort")
    X2 = X[order]
    y2 = y[order]
    inst2 = inst_id[order]

    change = np.nonzero(inst2[1:] != inst2[:-1])[0] + 1
    bounds = np.concatenate(([0], change, [len(inst2)]))
    group_sizes = np.diff(bounds).astype(np.int32)

    rel = np.empty((len(y2),), dtype=np.int32)
    for a, b in zip(bounds[:-1], bounds[1:]):
        yy = y2[a:b].astype(np.float64)
        uniq = np.unique(yy)
        uniq.sort()
        pos = np.searchsorted(uniq, yy)
        rel[a:b] = (len(uniq) - 1 - pos).astype(np.int32)

    return X2, rel, group_sizes, y2, inst2


def _train_eval_lgbm_ranker(
    *, X: np.ndarray, y: np.ndarray, inst_id: np.ndarray, test_frac: float, seed: int
) -> Dict[str, Dict[str, float]]:
    lgb = _try_import_lightgbm()
    if lgb is None:
        return {"status": {"ok": 0.0, "reason": 1.0}}

    train_idx, test_idx = _split_by_inst(
        inst_id, test_frac=float(test_frac), seed=int(seed)
    )

    X_tr, rel_tr, group_tr, y_tr, inst_tr = _build_grouped_rank_data(
        X[train_idx], y[train_idx], inst_id[train_idx]
    )
    X_te, rel_te, group_te, y_te, inst_te = _build_grouped_rank_data(
        X[test_idx], y[test_idx], inst_id[test_idx]
    )

    max_rel = int(rel_tr.max()) if rel_tr.size else 0
    label_gain = list(range(max_rel + 1))

    model = lgb.LGBMRanker(
        objective="lambdarank",
        label_gain=label_gain,
        n_estimators=4000,
        learning_rate=0.05,
        num_leaves=127,
        min_data_in_leaf=200,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=5.0,
        random_state=int(seed),
        force_row_wise=True,
        verbose=-1,
    )

    model.fit(
        X_tr,
        rel_tr,
        group=group_tr,
        eval_set=[(X_te, rel_te)],
        eval_group=[group_te],
        eval_at=[1, 3, 5],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    scores = model.predict(X_te).astype(np.float32)
    m_pick = _avg_regret_pick_from_scores(y_te, inst_te, scores, seed=int(seed))
    return {"test": m_pick}


def _sample_by_instance(
    X: np.ndarray,
    y: np.ndarray,
    inst_id: np.ndarray,
    *,
    max_instances: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_instances <= 0:
        return X, y, inst_id

    rng = np.random.default_rng(int(seed))
    uniq = np.unique(inst_id)
    if len(uniq) <= max_instances:
        return X, y, inst_id

    chosen = rng.choice(uniq, size=int(max_instances), replace=False)
    chosen_set = set(chosen.tolist())
    m = np.array([int(i) in chosen_set for i in inst_id], dtype=bool)
    return X[m], y[m], inst_id[m]


def _load_and_sample(
    paths: Sequence[Path], *, max_instances: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    Xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    insts: List[np.ndarray] = []
    meta0: Optional[np.ndarray] = None

    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            X = z["X"].astype(np.float32)
            y = z["y"].astype(np.float32)
            inst_id = z["inst_id"].astype(np.int32)
            if meta0 is None and "meta" in z:
                meta0 = z["meta"]

        X, y, inst_id = _sample_by_instance(
            X, y, inst_id, max_instances=max_instances, seed=seed
        )
        Xs.append(X)
        ys.append(y)
        insts.append(inst_id)

    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    inst_all = np.concatenate(insts, axis=0)
    return X_all, y_all, inst_all, meta0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Comma-separated .npz shards")
    ap.add_argument(
        "--max_instances",
        type=int,
        default=5000,
        help="Sample at most this many instances per shard (0 = all)",
    )
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    paths = _parse_paths(args.data)
    X, y, inst_id, meta = _load_and_sample(
        paths, max_instances=int(args.max_instances), seed=int(args.seed)
    )

    n_jobs_pad, window_dim, proxy_dim, ct_bins, price_quant_dim, enhanced = (
        _infer_feature_layout(int(X.shape[1]), meta)
    )

    # Compute slice boundaries.
    seq_end = int(n_jobs_pad)
    if enhanced:
        window_end = seq_end + int(window_dim)
        proxy_end = window_end + int(proxy_dim)
        ct_end = proxy_end + int(ct_bins)
        price_quant_end = ct_end + int(price_quant_dim)
        scalars_end = price_quant_end + 4
    else:
        window_end = seq_end
        proxy_end = seq_end
        ct_end = seq_end + int(ct_bins)
        price_quant_end = ct_end
        scalars_end = ct_end + 4

    if int(X.shape[1]) < scalars_end:
        raise ValueError(
            f"Feature layout mismatch: d={int(X.shape[1])} < expected_min_d={scalars_end}. "
            "Regenerate shards or update audit layout inference."
        )

    print(
        "[data] n_samples=",
        int(X.shape[0]),
        " n_instances=",
        int(len(np.unique(inst_id))),
        " d=",
        int(X.shape[1]),
    )
    print(
        "[layout] n_jobs_pad=",
        n_jobs_pad,
        " window_dim=",
        window_dim,
        " proxy_dim=",
        proxy_dim,
        " enhanced=",
        int(bool(enhanced)),
        " price_quant_dim=",
        (int(price_quant_dim) if enhanced else 0),
        " ct_bins=",
        ct_bins,
    )

    print("[spread]", _within_instance_spread(y, inst_id))

    print("[baseline:random]", _avg_regret_pick_random(y, inst_id, seed=int(args.seed)))

    proxy_scores = _proxy_sequential_cost_scores(
        X,
        n_jobs_pad=n_jobs_pad,
        window_dim=window_dim,
        proxy_dim=proxy_dim,
        ct_bins=ct_bins,
        price_quant_dim=price_quant_dim,
    )
    print(
        "[baseline:proxy_seq_cost]",
        _avg_regret_pick_from_scores(y, inst_id, proxy_scores, seed=int(args.seed)),
    )

    # Feature ablations:
    # - p_only: sequence features (processing times in order)
    # - seq+window+proxy: sequence + DP-aligned extras (only if enhanced)
    # - inst_only: instance features (ct_bins + [price_quant] + scalars)
    # - all: everything
    p_only = X[:, :seq_end]
    inst_only = X[:, proxy_end:scalars_end]
    all_feats = X[:, :scalars_end]

    # Combine sequence with new features.
    seq_with_window_proxy = X[:, :proxy_end] if enhanced else p_only

    print(
        "\n[regression:ridgetest:all]",
        _train_eval_ridge(
            X=all_feats,
            y=y,
            inst_id=inst_id,
            test_frac=float(args.test_frac),
            seed=int(args.seed),
        ),
    )
    print(
        "[regression:ridgetest:p_only]",
        _train_eval_ridge(
            X=p_only,
            y=y,
            inst_id=inst_id,
            test_frac=float(args.test_frac),
            seed=int(args.seed),
        ),
    )
    print(
        "[regression:ridgetest:seq+window+proxy]",
        _train_eval_ridge(
            X=seq_with_window_proxy,
            y=y,
            inst_id=inst_id,
            test_frac=float(args.test_frac),
            seed=int(args.seed),
        ),
    )
    print(
        "[regression:ridgetest:inst_only]",
        _train_eval_ridge(
            X=inst_only,
            y=y,
            inst_id=inst_id,
            test_frac=float(args.test_frac),
            seed=int(args.seed),
        ),
    )

    print(
        "\n[regression:lgbm:all]",
        _train_eval_lgbm_regressor(
            X=all_feats,
            y=y,
            inst_id=inst_id,
            test_frac=float(args.test_frac),
            seed=int(args.seed),
        ),
    )
    print(
        "[regression:lgbm:p_only]",
        _train_eval_lgbm_regressor(
            X=p_only,
            y=y,
            inst_id=inst_id,
            test_frac=float(args.test_frac),
            seed=int(args.seed),
        ),
    )
    print(
        "[regression:lgbm:seq+window+proxy]",
        _train_eval_lgbm_regressor(
            X=seq_with_window_proxy,
            y=y,
            inst_id=inst_id,
            test_frac=float(args.test_frac),
            seed=int(args.seed),
        ),
    )
    print(
        "[regression:lgbm:inst_only]",
        _train_eval_lgbm_regressor(
            X=inst_only,
            y=y,
            inst_id=inst_id,
            test_frac=float(args.test_frac),
            seed=int(args.seed),
        ),
    )

    print(
        "\n[ranking:lgbm_ranker:all]",
        _train_eval_lgbm_ranker(
            X=all_feats,
            y=y,
            inst_id=inst_id,
            test_frac=float(args.test_frac),
            seed=int(args.seed),
        ),
    )
    print(
        "[ranking:lgbm_ranker:p_only]",
        _train_eval_lgbm_ranker(
            X=p_only,
            y=y,
            inst_id=inst_id,
            test_frac=float(args.test_frac),
            seed=int(args.seed),
        ),
    )
    print(
        "[ranking:lgbm_ranker:seq+window+proxy]",
        _train_eval_lgbm_ranker(
            X=seq_with_window_proxy,
            y=y,
            inst_id=inst_id,
            test_frac=float(args.test_frac),
            seed=int(args.seed),
        ),
    )
    print(
        "[ranking:lgbm_ranker:inst_only]",
        _train_eval_lgbm_ranker(
            X=inst_only,
            y=y,
            inst_id=inst_id,
            test_frac=float(args.test_frac),
            seed=int(args.seed),
        ),
    )


if __name__ == "__main__":
    main()

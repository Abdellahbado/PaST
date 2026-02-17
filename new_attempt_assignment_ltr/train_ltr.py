from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TrainConfig:
    dataset_npz: str
    out_dir: str
    seed: int = 42
    test_frac_qid: float = 0.2

    # XGBoost ranker settings
    n_estimators: int = 600
    learning_rate: float = 0.05
    max_depth: int = 6
    subsample: float = 0.9
    colsample_bytree: float = 0.9


def _split_by_qid(qid: np.ndarray, test_frac: float, seed: int):
    rng = np.random.default_rng(int(seed))
    uniq = np.unique(qid)
    rng.shuffle(uniq)
    n_test = int(max(1, round(float(test_frac) * float(len(uniq)))))
    test_q = set(int(x) for x in uniq[:n_test])
    is_test = np.array([int(q) in test_q for q in qid], dtype=bool)
    return ~is_test, is_test


def _to_integer_relevance_per_group(y: np.ndarray, qid: np.ndarray) -> np.ndarray:
    """Convert continuous per-row utility labels into integer relevance degrees.

    XGBoost's ranking metrics (e.g., ndcg) require labels to be non-negative integers.
    We map within each query group:
        worst -> 0, best -> (group_size - 1)
    with deterministic tie handling.
    """
    y = np.asarray(y, dtype=np.float64)
    qid = np.asarray(qid, dtype=np.int64)
    rel = np.zeros_like(qid, dtype=np.int32)

    # Stable grouping
    order = np.argsort(qid, kind="stable")
    q_sorted = qid[order]
    boundaries = np.flatnonzero(np.r_[True, q_sorted[1:] != q_sorted[:-1], True])
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        idx = order[a:b]
        # ascending by y (worst first) so best gets highest relevance
        rank = np.argsort(y[idx], kind="stable")
        # rank[k] gives position of kth smallest; invert to assign relevance
        # worst -> 0, best -> size-1
        rel[idx[rank]] = np.arange(int(b - a), dtype=np.int32)

    return rel


def train_xgb_ranker(cfg: TrainConfig) -> str:
    try:
        import xgboost as xgb
    except Exception as e:
        raise RuntimeError(
            "xgboost is required for this learning-to-rank training script. "
            "Install it in your conda env: conda install -c conda-forge xgboost"
        ) from e

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = np.load(cfg.dataset_npz)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    qid = data["qid"].astype(np.int32)

    tr_mask, te_mask = _split_by_qid(qid, cfg.test_frac_qid, cfg.seed)

    X_tr, y_tr, q_tr = X[tr_mask], y[tr_mask], qid[tr_mask]
    X_te, y_te, q_te = X[te_mask], y[te_mask], qid[te_mask]

    # Convert continuous labels into integer relevance degrees per group.
    y_tr_rel = _to_integer_relevance_per_group(y_tr, q_tr)
    y_te_rel = _to_integer_relevance_per_group(y_te, q_te)

    # XGBoost requires group sizes in order of rows.
    def _group_sizes(q: np.ndarray) -> np.ndarray:
        uniq, counts = np.unique(q, return_counts=True)
        # ensure order by qid
        order = np.argsort(uniq)
        counts = counts[order]
        return counts.astype(np.int32)

    dtr = xgb.DMatrix(X_tr, label=y_tr_rel)
    dte = xgb.DMatrix(X_te, label=y_te_rel)
    dtr.set_group(_group_sizes(q_tr))
    dte.set_group(_group_sizes(q_te))

    params = {
        "objective": "rank:pairwise",
        "learning_rate": float(cfg.learning_rate),
        "max_depth": int(cfg.max_depth),
        "subsample": float(cfg.subsample),
        "colsample_bytree": float(cfg.colsample_bytree),
        "seed": int(cfg.seed),
        "eval_metric": "ndcg@10",
    }

    evals = [(dtr, "train"), (dte, "test")]
    evals_result = {}
    booster = xgb.train(
        params,
        dtr,
        num_boost_round=int(cfg.n_estimators),
        evals=evals,
        evals_result=evals_result,
        verbose_eval=50,
    )

    model_path = out / "xgb_ranker.json"
    booster.save_model(model_path)

    with open(out / "train_config.json", "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    # Main KPIs: ndcg curves + post-train rank quality on held-out queries.
    train_ndcg = evals_result.get("train", {}).get("ndcg@10", [])
    test_ndcg = evals_result.get("test", {}).get("ndcg@10", [])

    def _safe_last(xs):
        return float(xs[-1]) if xs else float("nan")

    def _safe_best(xs):
        if not xs:
            return float("nan"), -1
        best_i = int(np.argmax(np.asarray(xs, dtype=np.float64)))
        return float(xs[best_i]), best_i

    best_train, best_train_iter = _safe_best(train_ndcg)
    best_test, best_test_iter = _safe_best(test_ndcg)

    # Evaluate on held-out groups using continuous utility labels (y = -energy).
    # This is a more meaningful KPI than integer relevance.
    from .eval_ltr import evaluate_ranker_predictions

    pred_te = booster.predict(xgb.DMatrix(X_te))
    heldout_stats = evaluate_ranker_predictions(qid=q_te, y_true=y_te, y_pred=pred_te)

    metrics = {
        "train_ndcg_last": _safe_last(train_ndcg),
        "test_ndcg_last": _safe_last(test_ndcg),
        "train_ndcg_best": float(best_train),
        "train_ndcg_best_iter": int(best_train_iter),
        "test_ndcg_best": float(best_test),
        "test_ndcg_best_iter": int(best_test_iter),
        "heldout": heldout_stats.__dict__,
    }

    with open(out / "train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(
        json.dumps(
            {
                "train_ndcg_last": metrics["train_ndcg_last"],
                "test_ndcg_last": metrics["test_ndcg_last"],
                "test_ndcg_best": metrics["test_ndcg_best"],
                "test_ndcg_best_iter": metrics["test_ndcg_best_iter"],
                "heldout": metrics["heldout"],
                "model_path": str(model_path),
                "metrics_path": str(out / "train_metrics.json"),
            },
            indent=2,
        )
    )

    return str(model_path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    path = train_xgb_ranker(TrainConfig(dataset_npz=args.dataset, out_dir=args.out))
    print(path)

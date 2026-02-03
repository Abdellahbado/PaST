"""Duration-class branching policy training for PaST B&B.

This is a *targeted* replacement for the old Q-sequence training approach.

Goal
----
Learn a branching policy for branch-and-bound that ranks candidate next choices by
*processing-time class* (duration), not by job ID.

Labels
------
For a B&B node (partial sequence / remaining jobs), we compute for each candidate
unique processing time d among remaining jobs:

    y(d) = LB(child after choosing duration d)

where LB is the solver's relaxation-based lower bound (GCD-split jobs), already
implemented in :class:`PaST.solvers.bnb_solver_custom.BranchAndBoundSolver`.

This gives dense, local supervision aligned with pruning/branching.

Training
--------
- Always-available baseline: deterministic heuristic ordering (min window cost).
- Recommended: pairwise logistic ranker (requires scikit-learn).
- Optional: LightGBM/XGBoost ranking if installed.

Usage
-----
Generate data:
    python -m PaST.train_bb_branch_policy \
        generate --out artifacts/bb_branch_train.csv \
        --num_instances 2000 --walks_per_instance 10 --max_depth 30

Train a model:
    python -m PaST.train_bb_branch_policy \
        train --data artifacts/bb_branch_train.csv \
        --model_out artifacts/bb_branch_lr.pkl

Evaluate:
    python -m PaST.train_bb_branch_policy \
        eval --data artifacts/bb_branch_train.csv \
        --model artifacts/bb_branch_lr.pkl
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from PaST.solvers.bnb_solver_custom import Instance, BranchAndBoundSolver
from PaST.bb_branching_features import WindowFeatureCache, extract_row_features


def _open_csv_text(path: Path, mode: str):
    """Open CSV text stream.

    Supports transparent gzip when path ends with .gz.
    Use newline="" for csv module correctness.
    """
    if "b" in mode:
        raise ValueError("_open_csv_text expects text mode like 'r'/'w'")
    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode + "t", newline="")
    return path.open(mode, newline="")


def _parse_data_paths(data_arg: str) -> List[Path]:
    """Parse --data which can be a single path or comma-separated paths."""
    parts = [p.strip() for p in (data_arg or "").split(",") if p.strip()]
    if not parts:
        raise ValueError("--data is empty")
    return [Path(p) for p in parts]


def _try_import_torch_dp():
    """Lazy import to avoid torch dependency unless requested."""
    try:
        import torch

        from PaST.solvers.batch_dp_solver import BatchSequenceDPSolver

        return torch, BatchSequenceDPSolver
    except Exception:
        return None


# ----------------------------
# Instance generation
# ----------------------------


def _generate_duration_vocab(spec: str) -> List[int]:
    """Parse duration vocab specification.

    Supported forms:
    - "2,3,4,6,8" explicit list
    - "range:2:20:2" => start:stop:step (inclusive stop)
    """
    s = (spec or "").strip()
    if not s:
        return [1, 2, 3, 4, 6, 8, 12, 16]
    if s.startswith("range:"):
        _, rest = s.split(":", 1)
        parts = rest.split(":")
        if len(parts) != 3:
            raise ValueError("range spec must be range:start:stop:step")
        start, stop, step = [int(x) for x in parts]
        if step <= 0:
            raise ValueError("range step must be > 0")
        return list(range(start, stop + 1, step))
    out = [int(x.strip()) for x in s.split(",") if x.strip()]
    out = [d for d in out if d > 0]
    if not out:
        raise ValueError("Empty duration vocab")
    return sorted(set(out))


def _generate_price_curve(T: int, rng: random.Random, kind: str) -> np.ndarray:
    """Generate a TOU price curve with diverse structure."""
    kind = (kind or "mixed").lower()

    def _noise(scale: float = 1.0) -> np.ndarray:
        return np.array([rng.random() for _ in range(T)], dtype=np.float64) * scale

    if kind == "flat":
        base = rng.uniform(0.5, 2.0)
        return np.full(T, base, dtype=np.float64)

    if kind == "uniform":
        lo = rng.uniform(0.0, 2.0)
        hi = lo + rng.uniform(0.5, 5.0)
        return np.array([rng.uniform(lo, hi) for _ in range(T)], dtype=np.float64)

    if kind == "sin":
        base = rng.uniform(0.5, 2.0)
        amp = rng.uniform(0.5, 3.0)
        period = rng.choice([24, 48, 72])
        phase = rng.uniform(0.0, 2 * math.pi)
        t = np.arange(T, dtype=np.float64)
        c = base + amp * np.sin(2 * math.pi * t / period + phase) + 0.3 * _noise()
        return np.maximum(c, -1.0)

    if kind == "valleys":
        # Piecewise constant with a few valleys/peaks.
        c = np.full(T, rng.uniform(1.0, 3.0), dtype=np.float64)
        for _ in range(rng.randint(2, 6)):
            start = rng.randint(0, max(0, T - 1))
            length = rng.randint(1, max(1, T // 6))
            end = min(T, start + length)
            c[start:end] = rng.uniform(0.0, 1.0)
        for _ in range(rng.randint(1, 4)):
            start = rng.randint(0, max(0, T - 1))
            length = rng.randint(1, max(1, T // 8))
            end = min(T, start + length)
            c[start:end] = rng.uniform(3.0, 8.0)
        c += 0.1 * _noise()
        return np.maximum(c, -1.0)

    if kind == "spiky":
        c = np.array([rng.uniform(0.5, 2.0) for _ in range(T)], dtype=np.float64)
        for _ in range(rng.randint(1, max(1, T // 20))):
            idx = rng.randrange(T)
            c[idx] += rng.uniform(5.0, 15.0)
        c += 0.1 * _noise()
        return np.maximum(c, -1.0)

    if kind == "mixed":
        return _generate_price_curve(
            T, rng, rng.choice(["uniform", "sin", "valleys", "spiky"])
        )

    raise ValueError(f"Unknown price curve kind: {kind}")


def generate_instance(
    *,
    n_jobs: int,
    T: int,
    duration_vocab: Sequence[int],
    rng: random.Random,
    price_kind: str = "mixed",
    duration_mixture: str = "mixed",
) -> Instance:
    """Generate an Instance with duplicated processing times (duration classes)."""
    if n_jobs <= 0:
        raise ValueError("n_jobs must be > 0")
    if T <= 0:
        raise ValueError("T must be > 0")

    vocab = list(duration_vocab)
    if not vocab:
        raise ValueError("duration_vocab empty")

    # Encourage duplicates so "duration-class" symmetry is meaningful.
    if (duration_mixture or "mixed").lower() == "mixed":
        weights = np.array(
            [1.0 / (1.0 + i) for i in range(len(vocab))], dtype=np.float64
        )
        weights = weights / weights.sum()
    elif duration_mixture.lower() == "uniform":
        weights = np.ones(len(vocab), dtype=np.float64) / len(vocab)
    elif duration_mixture.lower() == "long":
        weights = np.linspace(0.5, 2.0, num=len(vocab), dtype=np.float64)
        weights = weights / weights.sum()
    elif duration_mixture.lower() == "short":
        weights = np.linspace(2.0, 0.5, num=len(vocab), dtype=np.float64)
        weights = weights / weights.sum()
    else:
        raise ValueError(f"Unknown duration_mixture: {duration_mixture}")

    pts = rng.choices(vocab, weights=weights.tolist(), k=n_jobs)
    processing_times = np.array(pts, dtype=np.int32)

    # Make sure total work fits reasonably into horizon; if not, expand T.
    total_p = int(processing_times.sum())
    if total_p > T:
        # B&B DP assumes jobs must fit within horizon.
        T = max(T, total_p)

    energy_costs = _generate_price_curve(T, rng, price_kind)

    return Instance(
        n_jobs=n_jobs,
        processing_times=processing_times,
        T=int(T),
        energy_costs=energy_costs.astype(np.float64),
    )


# ----------------------------
# Feature extraction (shared)
# ----------------------------


# ----------------------------
# Dataset collection
# ----------------------------


def _unique_duration_candidates(
    p: np.ndarray, remaining_jobs: Sequence[int]
) -> Dict[int, int]:
    """Map duration -> representative job id from remaining set."""
    out: Dict[int, int] = {}
    for j in remaining_jobs:
        d = int(p[j])
        if d not in out:
            out[d] = int(j)
    return out


def _random_walk_collect(
    *,
    instance_id: int,
    instance: Instance,
    duration_vocab: Sequence[int],
    rng: random.Random,
    walks_per_instance: int,
    max_depth: int,
    near_tie_epsilon: float,
    dp_backend: str = "numpy",
    dp_device: str = "auto",
) -> List[Dict[str, float]]:
    """Collect ranking rows via random walks through the search tree."""
    solver = BranchAndBoundSolver(instance, time_limit=1e9, verbose=False)
    window_cache = WindowFeatureCache(instance.energy_costs)

    dp_backend = (dp_backend or "numpy").strip().lower()
    if dp_backend not in {"numpy", "torch"}:
        raise ValueError(f"Unknown dp_backend: {dp_backend}")

    torch_pack = None
    torch_device = None
    ct_t = None
    if dp_backend == "torch":
        torch_pack = _try_import_torch_dp()
        if torch_pack is None:
            raise RuntimeError(
                "dp_backend=torch requested but torch/batch_dp_solver is not available. "
                "Install torch and ensure PaST.solvers.batch_dp_solver is importable."
            )
        torch, BatchSequenceDPSolver = torch_pack
        dev = (dp_device or "auto").strip().lower()
        if dev == "auto":
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif dev in {"cpu", "cuda"}:
            torch_device = torch.device(dev)
        else:
            raise ValueError(f"Unknown dp_device: {dp_device}")

        # Cache ct for this instance once.
        ct_t = (
            torch.from_numpy(instance.energy_costs.astype(np.float32))
            .unsqueeze(0)
            .to(torch_device)
        )

        def _torch_batch_cost(pts_list: List[List[int]]) -> List[float]:
            # Vectorized DP over a batch of fixed sequences, per instance.
            B = len(pts_list)
            if B == 0:
                return []
            N_max = max(len(x) for x in pts_list)
            if N_max <= 0:
                return [0.0 for _ in range(B)]

            # processing_times (B, N_max)
            p_mat = torch.zeros((B, N_max), dtype=torch.long, device=torch_device)
            seq_len = torch.zeros((B,), dtype=torch.long, device=torch_device)
            for i, pts in enumerate(pts_list):
                seq_len[i] = int(len(pts))
                if pts:
                    p_mat[i, : len(pts)] = torch.tensor(
                        pts, dtype=torch.long, device=torch_device
                    )

            # Identity sequence 0..N_max-1
            job_seq = (
                torch.arange(N_max, device=torch_device, dtype=torch.long)
                .unsqueeze(0)
                .expand(B, -1)
            )

            ct = ct_t.expand(B, -1)
            e = torch.ones((B,), dtype=torch.float32, device=torch_device)
            T_limit = torch.full(
                (B,), int(instance.T), dtype=torch.long, device=torch_device
            )
            costs_t = BatchSequenceDPSolver.solve(
                job_sequences=job_seq,
                processing_times=p_mat,
                ct=ct,
                e_single=e,
                T_limit=T_limit,
                sequence_lengths=seq_len,
            )
            return costs_t.detach().cpu().numpy().astype(np.float64).tolist()

    def _build_relaxed_pts(
        partial_sequence: List[int], remaining_jobs: Sequence[int]
    ) -> List[int]:
        if not remaining_jobs:
            if not partial_sequence:
                return []
            return instance.processing_times[partial_sequence].tolist()

        rem_pts = [int(instance.processing_times[j]) for j in remaining_jobs]
        total_rem = int(sum(rem_pts))
        if total_rem <= 0:
            if not partial_sequence:
                return []
            return instance.processing_times[partial_sequence].tolist()

        gcd_val = int(np.gcd.reduce(np.array(rem_pts, dtype=np.int64)))
        if gcd_val <= 0:
            gcd_val = 1
        n_relaxed = total_rem // gcd_val

        fixed = (
            instance.processing_times[partial_sequence].tolist()
            if partial_sequence
            else []
        )
        fixed.extend([int(gcd_val)] * int(n_relaxed))
        return fixed

    rows: List[Dict[str, float]] = []

    all_jobs = list(range(instance.n_jobs))

    for walk in range(int(walks_per_instance)):
        partial: List[int] = []
        remaining = list(all_jobs)

        # Shuffle to diversify ties in representative jobs.
        rng.shuffle(remaining)

        depth = 0
        while remaining and depth < int(max_depth):
            cand = _unique_duration_candidates(instance.processing_times, remaining)
            if len(cand) <= 1:
                break

            if dp_backend == "torch":
                # Batch DP: parent + all candidate children in one call.
                # This is where we win most of the wall-clock time.
                parent_pts = _build_relaxed_pts(partial, remaining)
                child_pts_list: List[List[int]] = []
                meta: List[Tuple[int, int]] = []  # (d, job_id)
                for d, job_id in cand.items():
                    child_partial = partial + [int(job_id)]
                    child_remaining = [j for j in remaining if j != int(job_id)]
                    child_pts_list.append(
                        _build_relaxed_pts(child_partial, child_remaining)
                    )
                    meta.append((int(d), int(job_id)))

                costs = _torch_batch_cost([parent_pts] + child_pts_list)
                if not costs or not np.isfinite(costs[0]):
                    break
                lb_parent = float(costs[0])

                candidates: List[Tuple[int, int, float]] = []
                for (d, job_id), lb_child in zip(meta, costs[1:]):
                    if not np.isfinite(lb_child):
                        continue
                    candidates.append((int(d), int(job_id), float(lb_child)))
            else:
                # Numpy DP path (original, slower): one DP per candidate.
                lb_parent, _, _ = solver._compute_lower_bound_with_blocks(
                    partial, set(remaining)
                )
                if not np.isfinite(lb_parent):
                    break

                candidates = []
                for d, job_id in cand.items():
                    child_partial = partial + [job_id]
                    child_remaining = set(remaining) - {job_id}
                    lb_child, _, _ = solver._compute_lower_bound_with_blocks(
                        child_partial, child_remaining
                    )
                    if not np.isfinite(lb_child):
                        continue
                    candidates.append((int(d), int(job_id), float(lb_child)))

            if len(candidates) <= 1:
                break

            # Filter ultra-ties: if best and 2nd-best are almost identical, skip logging.
            candidates_sorted = sorted(candidates, key=lambda x: x[2])
            best_lb = candidates_sorted[0][2]
            second_lb = candidates_sorted[1][2]
            if float(second_lb - best_lb) < float(near_tie_epsilon):
                # Still continue the walk, but don't add low-signal supervision.
                chosen_d, chosen_job, _ = rng.choice(candidates_sorted)
                partial.append(chosen_job)
                remaining.remove(chosen_job)
                depth += 1
                continue

            # Create a query for this node.
            query_id = instance_id * 10_000_000 + walk * 10_000 + depth

            # Add one row per candidate duration.
            # Use rank 0..k-1 where 0 is best.
            for rank_idx, (d, job_id, lb_child) in enumerate(candidates_sorted):
                feats = extract_row_features(
                    instance=instance,
                    duration_vocab=duration_vocab,
                    window_cache=window_cache,
                    partial_sequence=partial,
                    remaining_jobs=remaining,
                    candidate_d=int(d),
                )
                row: Dict[str, float] = {
                    "instance_id": float(instance_id),
                    "walk_id": float(walk),
                    "depth": float(depth),
                    "query_id": float(query_id),
                    "candidate_d": float(d),
                    "lb_parent": float(lb_parent),
                    "lb_child": float(lb_child),
                    "delta_lb": float(lb_child - lb_parent),
                    "label_rank": float(rank_idx),
                    "is_best": float(1.0 if rank_idx == 0 else 0.0),
                }
                row.update(feats)
                rows.append(row)

            # Continue the walk: biased random toward better candidates (helps cover good branches).
            # This is not training-time leakage because labels are computed for all candidates.
            weights = np.array(
                [1.0 / (1.0 + i) for i in range(len(candidates_sorted))],
                dtype=np.float64,
            )
            weights = weights / weights.sum()
            idx = int(
                np.random.default_rng(rng.randint(0, 2**31 - 1)).choice(
                    len(candidates_sorted), p=weights
                )
            )
            chosen_d, chosen_job, _ = candidates_sorted[idx]

            partial.append(chosen_job)
            remaining.remove(chosen_job)
            depth += 1

    return rows


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write")
    # Stable column order
    fieldnames = sorted(rows[0].keys())
    with _open_csv_text(path, "w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, 0.0) for k in fieldnames})


def read_csv(path: Path) -> List[Dict[str, float]]:
    with _open_csv_text(path, "r") as f:
        r = csv.DictReader(f)
        return [{k: float(v) for k, v in row.items()} for row in r]


def write_csv_stream(path: Path, row_iter: Iterable[Dict[str, float]]) -> int:
    """Stream rows to CSV without keeping everything in RAM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    writer = None
    fieldnames: List[str] = []
    with _open_csv_text(path, "w") as f:
        for row in row_iter:
            if writer is None:
                fieldnames = sorted(row.keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            writer.writerow({k: row.get(k, 0.0) for k in fieldnames})
            n += 1
    return n


# ----------------------------
# Models
# ----------------------------


def _try_import_sklearn():
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        return LogisticRegression, StandardScaler
    except Exception:
        return None


def _try_import_lightgbm():
    try:
        import lightgbm as lgb

        return lgb
    except Exception:
        return None


def _try_import_xgboost():
    try:
        import xgboost as xgb

        return xgb
    except Exception:
        return None


def _split_rows_by_instance(
    rows: List[Dict[str, float]], *, test_frac: float, seed: int
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """Split by instance_id to avoid leakage across nodes of the same instance."""
    if test_frac <= 0:
        return rows, []
    if test_frac >= 1:
        return [], rows

    rng = random.Random(seed)
    instance_ids = sorted({int(r["instance_id"]) for r in rows})
    rng.shuffle(instance_ids)
    n_test = int(round(len(instance_ids) * float(test_frac)))
    test_ids = set(instance_ids[:n_test])

    train_rows: List[Dict[str, float]] = []
    test_rows: List[Dict[str, float]] = []
    for r in rows:
        (test_rows if int(r["instance_id"]) in test_ids else train_rows).append(r)
    return train_rows, test_rows


def _build_pairwise_dataset(
    rows: List[Dict[str, float]],
    feature_cols: List[str],
    *,
    scaler,
    max_pairs_per_query: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build pairwise (z_a - z_b, y) dataset in standardized item space.

    scaler MUST be fit on item-level X (not differences). This makes scoring
    consistent: score(item)=w^T z(item).
    """
    by_q: Dict[int, List[Dict[str, float]]] = {}
    for r in rows:
        q = int(r["query_id"])
        by_q.setdefault(q, []).append(r)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    w_list: List[float] = []

    rng = random.Random(0)

    for _, items in by_q.items():
        if len(items) < 2:
            continue

        items_sorted = sorted(items, key=lambda x: x["lb_child"])
        best = items_sorted[0]
        others = items_sorted[1:]
        if not others:
            continue
        rng.shuffle(others)
        others = others[: int(max_pairs_per_query)]

        x_best = np.array([best[c] for c in feature_cols], dtype=np.float64)
        z_best = scaler.transform(x_best.reshape(1, -1)).reshape(-1)
        for o in others:
            x_o = np.array([o[c] for c in feature_cols], dtype=np.float64)
            z_o = scaler.transform(x_o.reshape(1, -1)).reshape(-1)

            X_list.append(z_best - z_o)
            y_list.append(1)
            w_list.append(float(abs(o["lb_child"] - best["lb_child"]) + 1e-6))

            X_list.append(z_o - z_best)
            y_list.append(0)
            w_list.append(float(abs(o["lb_child"] - best["lb_child"]) + 1e-6))

    if not X_list:
        raise ValueError("No pairwise samples built")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    w = np.array(w_list, dtype=np.float64)
    return X, y, w


def _score_from_pairwise_lr(*, lr, scaler, X_items: np.ndarray) -> np.ndarray:
    z = scaler.transform(X_items)
    w = lr.coef_.reshape(-1)
    return z @ w


def _build_query_grouped_arrays(
    rows: List[Dict[str, float]], feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, group) for learning-to-rank libraries.

    X: item features
    y: relevance (larger = better). We use -lb_child.
    group: group size per query_id
    """
    by_q: Dict[int, List[Dict[str, float]]] = {}
    for r in rows:
        by_q.setdefault(int(r["query_id"]), []).append(r)

    # Keep deterministic ordering by query_id.
    qids = sorted(by_q.keys())
    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    group: List[int] = []
    for qid in qids:
        items = by_q[qid]
        if len(items) < 2:
            continue
        group.append(len(items))
        for it in items:
            X_list.append(np.array([it[c] for c in feature_cols], dtype=np.float64))
            y_list.append(float(-it["lb_child"]))

    if not X_list:
        raise ValueError("No grouped samples built")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float64)
    group_arr = np.array(group, dtype=np.int32)
    return X, y, group_arr


# ----------------------------
# Evaluation
# ----------------------------


def evaluate_hit_regret(
    rows: List[Dict[str, float]],
    feature_cols: List[str],
    *,
    model: Optional[object] = None,
    use_heuristic: bool = False,
) -> Dict[str, float]:
    by_q: Dict[int, List[Dict[str, float]]] = {}
    for r in rows:
        by_q.setdefault(int(r["query_id"]), []).append(r)

    hit1 = 0
    n_q = 0
    regret = 0.0
    rand_hit1 = 0.0
    rand_regret = 0.0

    for qid, items in by_q.items():
        if len(items) < 2:
            continue
        n_q += 1

        # Best by label
        items_sorted = sorted(items, key=lambda x: x["lb_child"])
        best = items_sorted[0]
        best_lb = float(best["lb_child"])

        # Random baseline
        # Deterministic per-query random baseline.
        rng = random.Random(int(qid) ^ 0xC0FFEE)
        rand_choice = rng.choice(items)
        rand_hit1 += 1.0 if rand_choice is best else 0.0
        rand_regret += float(rand_choice["lb_child"] - best_lb)

        # Policy prediction
        if use_heuristic:
            # deterministic: pick smallest window min cost
            pred = min(items, key=lambda x: x.get("w_min", float("inf")))
        elif model is None:
            pred = rng.choice(items)
        else:
            X_items = np.stack(
                [
                    np.array([it[c] for c in feature_cols], dtype=np.float64)
                    for it in items
                ],
                axis=0,
            )
            mtype = model.get("model_type", "pairwise_logistic")
            if mtype == "pairwise_logistic":
                scores = _score_from_pairwise_lr(
                    lr=model["lr"],
                    scaler=model["scaler"],
                    X_items=X_items,
                )
                pred = items[int(np.argmax(scores))]
            elif mtype in {"lgbm_ranker", "xgb_ranker"}:
                scores = model["model"].predict(X_items)
                scores = np.nan_to_num(
                    scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf
                )
                pred = items[int(np.argmax(scores))]
            else:
                pred = rng.choice(items)

        hit1 += 1 if pred is best else 0
        regret += float(pred["lb_child"] - best_lb)

    if n_q == 0:
        return {"n_queries": 0.0}

    return {
        "n_queries": float(n_q),
        "hit1": float(hit1 / n_q),
        "avg_regret": float(regret / n_q),
        "rand_hit1": float(rand_hit1 / n_q),
        "rand_avg_regret": float(rand_regret / n_q),
    }


# ----------------------------
# CLI
# ----------------------------


def cmd_generate(args) -> None:
    out = Path(args.out)
    duration_vocab = _generate_duration_vocab(args.duration_vocab)
    rng = random.Random(args.seed)

    shard_id = int(getattr(args, "shard_id", 0) or 0)
    num_shards = int(getattr(args, "num_shards", 1) or 1)
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("Invalid shard_id/num_shards")

    dp_backend = (args.dp_backend or "numpy").strip().lower()
    dp_device = (args.dp_device or "auto").strip().lower()
    max_rows = int(getattr(args, "max_rows", 0) or 0)

    def _rows_iter() -> Iterable[Dict[str, float]]:
        n_rows = 0
        n_instances_done = 0
        for instance_id in range(int(args.num_instances)):
            if (instance_id % num_shards) != shard_id:
                continue
            if max_rows > 0 and n_rows >= max_rows:
                break
            inst_seed = rng.randint(0, 2**31 - 1)
            inst_rng = random.Random(inst_seed)
            instance = generate_instance(
                n_jobs=int(args.n_jobs),
                T=int(args.T),
                duration_vocab=duration_vocab,
                rng=inst_rng,
                price_kind=args.price_kind,
                duration_mixture=args.duration_mixture,
            )
            rows = _random_walk_collect(
                instance_id=instance_id,
                instance=instance,
                duration_vocab=duration_vocab,
                rng=inst_rng,
                walks_per_instance=int(args.walks_per_instance),
                max_depth=int(args.max_depth),
                near_tie_epsilon=float(args.near_tie_epsilon),
                dp_backend=dp_backend,
                dp_device=dp_device,
            )
            for r in rows:
                yield r
                n_rows += 1
                if max_rows > 0 and n_rows >= max_rows:
                    break
            if max_rows > 0 and n_rows >= max_rows:
                break

            n_instances_done += 1
            if args.log_every > 0 and n_instances_done % int(args.log_every) == 0:
                print(
                    f"[generate] shard={shard_id}/{num_shards} instances={n_instances_done} rows={n_rows}"
                )

    n_written = write_csv_stream(out, _rows_iter())
    print(
        f"[generate] wrote {n_written} rows to {out} (shard={shard_id}/{num_shards}, dp_backend={dp_backend}, dp_device={dp_device})"
    )


def cmd_train(args) -> None:
    model_out = Path(args.model_out)

    data_paths = _parse_data_paths(args.data)
    rows: List[Dict[str, float]] = []
    for p in data_paths:
        rows.extend(read_csv(p))
    if not rows:
        raise ValueError("Empty dataset")

    # Feature columns: everything numeric except identifiers/targets.
    drop = {
        "instance_id",
        "walk_id",
        "depth",
        "query_id",
        "candidate_d",
        "lb_parent",
        "lb_child",
        "delta_lb",
        "label_rank",
        "is_best",
    }
    feature_cols = sorted([k for k in rows[0].keys() if k not in drop])

    sklearn = _try_import_sklearn()
    if sklearn is None:
        raise RuntimeError(
            "scikit-learn is required for the baseline pairwise logistic ranker. "
            "Install with: pip install scikit-learn"
        )

    LogisticRegression, StandardScaler = sklearn

    X_items = np.stack(
        [np.array([r[c] for c in feature_cols], dtype=np.float64) for r in rows], axis=0
    )
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(X_items)

    X, y, w = _build_pairwise_dataset(
        rows,
        feature_cols,
        scaler=scaler,
        max_pairs_per_query=int(args.max_pairs_per_query),
    )

    lr = LogisticRegression(
        penalty="l2",
        C=float(args.C),
        solver="lbfgs",
        max_iter=int(args.max_iter),
        n_jobs=None,
    )
    lr.fit(X, y, sample_weight=w)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": "pairwise_logistic",
        "feature_cols": feature_cols,
        "model": {"scaler": scaler, "lr": lr, "model_type": "pairwise_logistic"},
    }
    with model_out.open("wb") as f:
        pickle.dump(payload, f)

    print(f"[train] saved model to {model_out}")


def cmd_train_lgbm(args) -> None:
    model_out = Path(args.model_out)

    data_paths = _parse_data_paths(args.data)
    rows: List[Dict[str, float]] = []
    for p in data_paths:
        rows.extend(read_csv(p))
    if not rows:
        raise ValueError("Empty dataset")

    train_rows, test_rows = _split_rows_by_instance(
        rows, test_frac=float(args.test_frac), seed=int(args.seed)
    )

    drop = {
        "instance_id",
        "walk_id",
        "depth",
        "query_id",
        "candidate_d",
        "lb_parent",
        "lb_child",
        "delta_lb",
        "label_rank",
        "is_best",
    }
    feature_cols = sorted([k for k in rows[0].keys() if k not in drop])

    lgb = _try_import_lightgbm()
    if lgb is None:
        raise RuntimeError(
            "lightgbm is not installed. Install in your env with: pip install lightgbm"
        )

    X_train, y_train, group_train = _build_query_grouped_arrays(
        train_rows, feature_cols
    )

    # LightGBM ranking expects integer relevance labels (non-negative).
    # Our natural supervision is a *float* lower-bound (lb_child). We convert it
    # into an ordinal relevance per query: smallest lb_child -> highest relevance.
    y_train_int = np.empty_like(y_train, dtype=np.int32)
    offset = 0
    for g in group_train.tolist():
        sl = slice(offset, offset + int(g))
        lbs = (-y_train[sl]).astype(np.float64)
        uniq = np.unique(lbs)
        uniq.sort()  # ascending lb
        pos = np.searchsorted(uniq, lbs)
        # best (smallest lb) gets the highest relevance
        y_train_int[sl] = (len(uniq) - 1 - pos).astype(np.int32)
        offset += int(g)

    max_rel = int(y_train_int.max()) if y_train_int.size else 0
    label_gain = list(range(max_rel + 1))
    model = lgb.LGBMRanker(
        objective="lambdarank",
        label_gain=label_gain,
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        min_data_in_leaf=int(args.min_data_in_leaf),
        max_depth=int(args.max_depth) if int(args.max_depth) > 0 else -1,
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        random_state=int(args.seed),
        verbose=-1,
    )

    model.fit(X_train, y_train_int, group=group_train)

    payload = {
        "model_type": "lgbm_ranker",
        "feature_cols": feature_cols,
        "model": {"model_type": "lgbm_ranker", "model": model},
    }
    model_out.parent.mkdir(parents=True, exist_ok=True)
    with model_out.open("wb") as f:
        pickle.dump(payload, f)

    print(f"[train_lgbm] saved model to {model_out}")
    if test_rows:
        m = evaluate_hit_regret(
            test_rows, feature_cols, model=payload["model"], use_heuristic=False
        )
        print("[train_lgbm] holdout:", m)


def cmd_train_xgb(args) -> None:
    model_out = Path(args.model_out)

    data_paths = _parse_data_paths(args.data)
    rows: List[Dict[str, float]] = []
    for p in data_paths:
        rows.extend(read_csv(p))
    if not rows:
        raise ValueError("Empty dataset")

    train_rows, test_rows = _split_rows_by_instance(
        rows, test_frac=float(args.test_frac), seed=int(args.seed)
    )

    drop = {
        "instance_id",
        "walk_id",
        "depth",
        "query_id",
        "candidate_d",
        "lb_parent",
        "lb_child",
        "delta_lb",
        "label_rank",
        "is_best",
    }
    feature_cols = sorted([k for k in rows[0].keys() if k not in drop])

    xgb = _try_import_xgboost()
    if xgb is None:
        raise RuntimeError(
            "xgboost is not installed. Install in your env with: pip install xgboost"
        )

    X_train, y_train, group_train = _build_query_grouped_arrays(
        train_rows, feature_cols
    )
    model = xgb.XGBRanker(
        objective="rank:pairwise",
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        max_depth=int(args.max_depth),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        reg_lambda=float(args.reg_lambda),
        random_state=int(args.seed),
        tree_method=str(args.tree_method),
    )
    model.fit(X_train, y_train, group=group_train)

    payload = {
        "model_type": "xgb_ranker",
        "feature_cols": feature_cols,
        "model": {"model_type": "xgb_ranker", "model": model},
    }
    model_out.parent.mkdir(parents=True, exist_ok=True)
    with model_out.open("wb") as f:
        pickle.dump(payload, f)

    print(f"[train_xgb] saved model to {model_out}")
    if test_rows:
        m = evaluate_hit_regret(
            test_rows, feature_cols, model=payload["model"], use_heuristic=False
        )
        print("[train_xgb] holdout:", m)


def cmd_eval(args) -> None:
    data_paths = _parse_data_paths(args.data)
    rows: List[Dict[str, float]] = []
    for p in data_paths:
        rows.extend(read_csv(p))

    train_rows, test_rows = _split_rows_by_instance(
        rows, test_frac=float(args.test_frac), seed=int(args.seed)
    )

    payload = None
    if args.model:
        with Path(args.model).open("rb") as f:
            payload = pickle.load(f)

    drop = {
        "instance_id",
        "walk_id",
        "depth",
        "query_id",
        "candidate_d",
        "lb_parent",
        "lb_child",
        "delta_lb",
        "label_rank",
        "is_best",
    }
    feature_cols = sorted([k for k in rows[0].keys() if k not in drop])

    def _print_split(name: str, rr: List[Dict[str, float]]):
        if not rr:
            return
        m_rand = evaluate_hit_regret(rr, feature_cols, model=None, use_heuristic=False)
        m_h = evaluate_hit_regret(rr, feature_cols, model=None, use_heuristic=True)
        print(f"[eval:{name}] random:", m_rand)
        print(f"[eval:{name}] heuristic(min_w):", m_h)
        if payload is not None:
            model = payload["model"]
            fcols = payload["feature_cols"]
            m = evaluate_hit_regret(rr, fcols, model=model, use_heuristic=False)
            print(f"[eval:{name}] model:", m)

    _print_split("train", train_rows)
    _print_split("test", test_rows if test_rows else rows)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate branching training data")
    g.add_argument("--out", required=True)
    g.add_argument("--num_instances", type=int, default=500)
    g.add_argument("--n_jobs", type=int, default=40)
    g.add_argument("--T", type=int, default=200)
    g.add_argument("--duration_vocab", type=str, default="1,2,3,4,6,8,12,16")
    g.add_argument(
        "--duration_mixture",
        type=str,
        default="mixed",
        choices=["mixed", "uniform", "long", "short"],
    )
    g.add_argument(
        "--price_kind",
        type=str,
        default="mixed",
        choices=["mixed", "flat", "uniform", "sin", "valleys", "spiky"],
    )
    g.add_argument("--walks_per_instance", type=int, default=8)
    g.add_argument("--max_depth", type=int, default=30)
    g.add_argument("--near_tie_epsilon", type=float, default=1e-6)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--log_every", type=int, default=50)
    g.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="Optional safety cap: stop after writing this many rows (0 = no cap).",
    )
    g.add_argument(
        "--dp_backend",
        type=str,
        default="numpy",
        choices=["numpy", "torch"],
        help="Label DP backend. torch uses BatchSequenceDPSolver (batched).",
    )
    g.add_argument(
        "--dp_device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for torch DP when dp_backend=torch.",
    )
    g.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="HPC sharding: only generate instances where instance_id %% num_shards == shard_id.",
    )
    g.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="HPC sharding: total number of shards.",
    )

    t = sub.add_parser("train", help="Train a simple ranker")
    t.add_argument("--data", required=True)
    t.add_argument("--model_out", required=True)
    t.add_argument("--max_pairs_per_query", type=int, default=64)
    t.add_argument("--C", type=float, default=1.0)
    t.add_argument("--max_iter", type=int, default=200)

    tl = sub.add_parser("train_lgbm", help="Train a LightGBM ranker (optional)")
    tl.add_argument("--data", required=True)
    tl.add_argument("--model_out", required=True)
    tl.add_argument("--test_frac", type=float, default=0.2)
    tl.add_argument("--seed", type=int, default=0)
    tl.add_argument("--n_estimators", type=int, default=400)
    tl.add_argument("--learning_rate", type=float, default=0.05)
    tl.add_argument("--num_leaves", type=int, default=31)
    tl.add_argument("--min_data_in_leaf", type=int, default=20)
    tl.add_argument("--max_depth", type=int, default=-1)
    tl.add_argument("--subsample", type=float, default=0.9)
    tl.add_argument("--colsample_bytree", type=float, default=0.9)

    tx = sub.add_parser("train_xgb", help="Train an XGBoost ranker (optional)")
    tx.add_argument("--data", required=True)
    tx.add_argument("--model_out", required=True)
    tx.add_argument("--test_frac", type=float, default=0.2)
    tx.add_argument("--seed", type=int, default=0)
    tx.add_argument("--n_estimators", type=int, default=600)
    tx.add_argument("--learning_rate", type=float, default=0.05)
    tx.add_argument("--max_depth", type=int, default=6)
    tx.add_argument("--subsample", type=float, default=0.9)
    tx.add_argument("--colsample_bytree", type=float, default=0.9)
    tx.add_argument("--reg_lambda", type=float, default=1.0)
    tx.add_argument("--tree_method", type=str, default="hist")

    e = sub.add_parser("eval", help="Evaluate vs random/heuristic")
    e.add_argument("--data", required=True)
    e.add_argument("--model", required=False)
    e.add_argument("--test_frac", type=float, default=0.2)
    e.add_argument("--seed", type=int, default=0)

    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.cmd == "generate":
        cmd_generate(args)
        return
    if args.cmd == "train":
        cmd_train(args)
        return
    if args.cmd == "train_lgbm":
        cmd_train_lgbm(args)
        return
    if args.cmd == "train_xgb":
        cmd_train_xgb(args)
        return
    if args.cmd == "eval":
        cmd_eval(args)
        return
    raise SystemExit(2)


if __name__ == "__main__":
    main()

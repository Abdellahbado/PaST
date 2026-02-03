"""Tabular baselines for predicting DP cost of a *fixed job sequence* (PaST-SM).

This is a lightweight alternative to the neural Q-sequence training.

Data generation (matches project distribution)
---------------------------------------------
We reuse PaST's benchmark-style episode generator (segmented price horizons):
`PaST.data.sm_benchmark_data.generate_episode_batch`.

For each instance (episode) we generate multiple sequences within the *same horizon*:
- random permutations
- heuristic sequences (SPT, LPT)
- local perturbations (swap/insert) of heuristic sequences

We label each sequence by the optimal start-time DP cost for that fixed order,
using the batched PyTorch DP (`PaST.solvers.batch_dp_solver.BatchSequenceDPSolver`).

Models
------
- Baseline: Ridge regression (and optionally ElasticNet)
- Second line: LightGBMRegressor / XGBoostRegressor (if installed)

Storage
-------
Writes compressed shards via `.npz` (np.savez_compressed).

Usage (single node)
-------------------
Generate:
  python -m PaST.train_sequence_cost_tabular generate \
    --out artifacts/seq_cost_train_shard0.npz --num_instances 5000 --seqs_per_instance 32 \
    --seed 0 --dp_device cuda

Train Ridge:
  python -m PaST.train_sequence_cost_tabular train_ridge \
    --data artifacts/seq_cost_train_shard0.npz --model_out artifacts/ridge.pkl

Eval:
  python -m PaST.train_sequence_cost_tabular eval \
    --data artifacts/seq_cost_train_shard0.npz --model artifacts/ridge.pkl

Sharding (HPC array)
--------------------
Use --num_shards/--shard_id for generation; then pass comma-separated shards to --data.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from PaST.config import VariantID, get_variant_config
from PaST.data.sm_benchmark_data import generate_episode_batch
from PaST.solvers.batch_dp_solver import BatchSequenceDPSolver


def _parse_paths(arg: str) -> List[Path]:
    parts = [p.strip() for p in (arg or "").split(",") if p.strip()]
    if not parts:
        raise ValueError("--data is empty")
    return [Path(p) for p in parts]


def _downsample_ct(ct: np.ndarray, T: int, bins: int) -> np.ndarray:
    if bins <= 0:
        return np.zeros((0,), dtype=np.float32)
    if T <= 0:
        return np.zeros((bins,), dtype=np.float32)
    x = ct[:T].astype(np.float32)
    # Pad to multiple of bins.
    pad = (-len(x)) % bins
    if pad:
        x = np.pad(x, (0, pad), mode="edge")
    x = x.reshape(bins, -1).mean(axis=1)
    return x.astype(np.float32)


def _seq_from_perm(perm: Sequence[int], n_jobs_pad: int) -> np.ndarray:
    out = np.zeros((n_jobs_pad,), dtype=np.int64)
    k = min(len(perm), n_jobs_pad)
    if k > 0:
        out[:k] = np.asarray(perm[:k], dtype=np.int64)
    return out


def _spt_order(p: np.ndarray, n: int) -> List[int]:
    idx = list(range(n))
    idx.sort(key=lambda j: (int(p[j]), j))
    return idx


def _lpt_order(p: np.ndarray, n: int) -> List[int]:
    idx = list(range(n))
    idx.sort(key=lambda j: (-int(p[j]), j))
    return idx


def _perturb_swap(seq: List[int], rng: random.Random) -> List[int]:
    if len(seq) < 2:
        return list(seq)
    a = rng.randrange(len(seq))
    b = rng.randrange(len(seq))
    out = list(seq)
    out[a], out[b] = out[b], out[a]
    return out


def _perturb_insert(seq: List[int], rng: random.Random) -> List[int]:
    if len(seq) < 2:
        return list(seq)
    i = rng.randrange(len(seq))
    j = rng.randrange(len(seq))
    if i == j:
        return list(seq)
    out = list(seq)
    x = out.pop(i)
    out.insert(j, x)
    return out


def _make_sequences_for_instance(
    *,
    p: np.ndarray,
    n: int,
    rng: random.Random,
    seqs_per_instance: int,
    num_random: int,
    include_heuristics: bool,
    perturbations_per_base: int,
) -> List[List[int]]:
    seqs: List[List[int]] = []

    if include_heuristics:
        seqs.append(_spt_order(p, n))
        seqs.append(_lpt_order(p, n))

    # Random perms
    num_random = max(0, int(num_random))
    for _ in range(num_random):
        perm = list(range(n))
        rng.shuffle(perm)
        seqs.append(perm)

    # Perturbations of heuristic sequences to create nearby alternatives.
    bases = []
    if include_heuristics:
        bases.extend([_spt_order(p, n), _lpt_order(p, n)])
    if not bases and seqs:
        bases.append(seqs[0])

    for base in bases:
        for _ in range(max(0, int(perturbations_per_base))):
            if rng.random() < 0.5:
                seqs.append(_perturb_swap(base, rng))
            else:
                seqs.append(_perturb_insert(base, rng))

    # Trim / pad by adding random perms until reaching seqs_per_instance.
    target = int(seqs_per_instance)
    if target <= 0:
        return []

    # Degenerate cases: if there are <2 jobs, there is at most 1 permutation.
    if n <= 1:
        base = list(range(max(0, n)))
        return [base for _ in range(target)]

    # Deduplicate while preserving order.
    seen = set()
    uniq: List[List[int]] = []
    for s in seqs:
        t = tuple(s)
        if t not in seen:
            uniq.append(s)
            seen.add(t)

    # Add random unique perms until target, but avoid infinite loops when the
    # space of unique permutations is small.
    max_attempts = max(100, 50 * target)
    attempts = 0
    while len(uniq) < target and attempts < max_attempts:
        attempts += 1
        perm = list(range(n))
        rng.shuffle(perm)
        t = tuple(perm)
        if t not in seen:
            uniq.append(perm)
            seen.add(t)

    # If we couldn't find enough unique sequences, backfill with repeats.
    if len(uniq) < target:
        if not uniq:
            uniq.append(list(range(n)))
        while len(uniq) < target:
            uniq.append(list(uniq[rng.randrange(len(uniq))]))

    return uniq[:target]


def _batched_dp_costs(
    *,
    sequences: np.ndarray,  # (B, N_pad) int64
    p_subset: np.ndarray,  # (B, N_pad) int32
    ct: np.ndarray,  # (B, T_pad) int32
    e_single: np.ndarray,  # (B,) int32
    T_limit: np.ndarray,  # (B,) int32
    seq_lens: np.ndarray,  # (B,) int32
    device: str,
    batch_max: int = 0,
) -> np.ndarray:
    import torch

    dev = torch.device(device)

    B = int(sequences.shape[0])
    if B == 0:
        return np.zeros((0,), dtype=np.float32)

    max_b = int(batch_max or 0)
    if max_b <= 0:
        max_b = B

    outs: List[np.ndarray] = []
    for start in range(0, B, max_b):
        end = min(B, start + max_b)
        job_seq = torch.from_numpy(sequences[start:end].astype(np.int64)).to(dev)
        p = torch.from_numpy(p_subset[start:end].astype(np.int64)).to(dev)
        ct_t = torch.from_numpy(ct[start:end].astype(np.float32)).to(dev)
        e = torch.from_numpy(e_single[start:end].astype(np.float32)).to(dev)
        T = torch.from_numpy(T_limit[start:end].astype(np.int64)).to(dev)
        L = torch.from_numpy(seq_lens[start:end].astype(np.int64)).to(dev)

        with torch.no_grad():
            costs = BatchSequenceDPSolver.solve(
                job_sequences=job_seq,
                processing_times=p,
                ct=ct_t,
                e_single=e,
                T_limit=T,
                sequence_lengths=L,
            )
        outs.append(costs.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(outs, axis=0)


def generate_dataset_shard(
    *,
    out: Path,
    variant_id: str,
    seed: int,
    num_instances: int,
    seqs_per_instance: int,
    num_random_per_instance: int,
    include_heuristics: bool,
    perturbations_per_base: int,
    ct_bins: int,
    dp_device: str,
    shard_id: int,
    num_shards: int,
    episodes_batch_size: int,
    dp_batch_max: int,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    # Allow passing either the Enum value (e.g. "q_sequence") or Enum name
    # (e.g. "Q_SEQUENCE") on the CLI.
    try:
        vid = VariantID(str(variant_id))
    except Exception:
        try:
            vid = VariantID[str(variant_id)]
        except Exception as e:
            valid = ", ".join([v.value for v in VariantID])
            raise ValueError(
                f"Unknown --variant_id={variant_id!r}. Expected one of: {valid}"
            ) from e

    cfg = get_variant_config(vid, seed=int(seed))
    data_cfg = cfg.data

    # Generate episodes in batches for efficiency.
    rng = random.Random(int(seed) + 1000 * int(shard_id))

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    inst_list: List[np.ndarray] = []

    n_jobs_pad = int(getattr(data_cfg, "N_job_pad", 50) or 50)
    # Keep aligned with the project's configuration: horizons are <= 500.
    T_pad = int(max(getattr(data_cfg, "T_max_choices", [500]) or [500]))
    if T_pad > 500:
        raise ValueError(
            f"DataConfig implies T_max_pad={T_pad} > 500; this script assumes <=500. "
            "Either lower T_max_choices or extend padding logic safely."
        )

    shard_id = int(shard_id)
    num_shards = int(num_shards)
    if num_shards <= 0:
        raise ValueError("num_shards must be >= 1")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("Invalid shard_id/num_shards")

    # Generate *only* instances belonging to this shard (avoid generate+discard).
    inst_ids_shard = list(range(shard_id, int(num_instances), num_shards))
    inst_off = 0
    while inst_off < len(inst_ids_shard):
        bs = min(int(episodes_batch_size), len(inst_ids_shard) - inst_off)
        batch_seed = rng.randint(0, 2**31 - 1)
        b = generate_episode_batch(
            batch_size=bs,
            config=data_cfg,
            seed=batch_seed,
            N_job_pad=n_jobs_pad,
            T_max_pad=T_pad,
        )

        # Build sequences and compute DP costs in one big DP batch per generated episode-batch.
        seq_rows: List[np.ndarray] = []
        p_rows: List[np.ndarray] = []
        ct_rows: List[np.ndarray] = []
        e_rows: List[int] = []
        T_rows: List[int] = []
        L_rows: List[int] = []
        feat_rows: List[np.ndarray] = []
        inst_ids: List[int] = []

        for i in range(bs):
            inst_id = int(inst_ids_shard[inst_off + i])

            n = int(b["n_jobs"][i])
            if n <= 0:
                continue
            p_i = b["p_subset"][i].copy()  # (N_pad,)
            ct_i = b["ct"][i].copy()  # (T_pad,)
            T_max_i = int(b["T_max"][i])
            T_limit_i = int(b["T_limit"][i])
            e_i = int(b["e_single"][i])

            # Ensure deadline not beyond pad length.
            T_limit_i = min(T_limit_i, T_pad)

            seqs = _make_sequences_for_instance(
                p=p_i,
                n=n,
                rng=random.Random(rng.randint(0, 2**31 - 1)),
                seqs_per_instance=int(seqs_per_instance),
                num_random=int(num_random_per_instance),
                include_heuristics=bool(include_heuristics),
                perturbations_per_base=int(perturbations_per_base),
            )

            # Instance-level (shared) features: downsampled prices + scalars.
            # DP uses T_limit, so prefer deadline-relevant horizon for features.
            T_feat = int(min(T_limit_i, T_max_i))
            ct_ds = _downsample_ct(ct_i, T_feat, int(ct_bins))
            inst_feats = np.concatenate(
                [
                    ct_ds,
                    np.array(
                        [
                            float(e_i),
                            float(T_limit_i),
                            float(T_max_i),
                            float(n),
                        ],
                        dtype=np.float32,
                    ),
                ],
                axis=0,
            ).astype(np.float32)

            for s in seqs:
                seq_rows.append(_seq_from_perm(s, n_jobs_pad))
                p_rows.append(p_i.astype(np.int32, copy=False))
                ct_rows.append(ct_i.astype(np.int32, copy=False))
                e_rows.append(e_i)
                T_rows.append(T_limit_i)
                L_rows.append(n)

                # Sequence-level features: processing times in sequence order.
                p_seq = p_i[np.asarray(s, dtype=np.int64)].astype(np.float32)
                if len(p_seq) < n_jobs_pad:
                    p_seq = np.pad(p_seq, (0, n_jobs_pad - len(p_seq)), mode="constant")
                else:
                    p_seq = p_seq[:n_jobs_pad]
                feat = np.concatenate([p_seq.astype(np.float32), inst_feats], axis=0)
                feat_rows.append(feat)
                inst_ids.append(inst_id)

        if seq_rows:
            sequences = np.stack(seq_rows, axis=0)
            p_mat = np.stack(p_rows, axis=0)
            ct_mat = np.stack(ct_rows, axis=0)
            e_vec = np.asarray(e_rows, dtype=np.int32)
            T_vec = np.asarray(T_rows, dtype=np.int32)
            L_vec = np.asarray(L_rows, dtype=np.int32)

            costs = _batched_dp_costs(
                sequences=sequences,
                p_subset=p_mat,
                ct=ct_mat,
                e_single=e_vec,
                T_limit=T_vec,
                seq_lens=L_vec,
                device=str(dp_device),
                batch_max=int(dp_batch_max or 0),
            )

            X = np.stack(feat_rows, axis=0).astype(np.float32)
            y = costs.astype(np.float32)
            inst_arr = np.asarray(inst_ids, dtype=np.int32)

            X_list.append(X)
            y_list.append(y)
            inst_list.append(inst_arr)

        inst_off += bs

    if not X_list:
        raise ValueError("No data generated (shard may be empty).")

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    inst_all = np.concatenate(inst_list, axis=0)

    np.savez_compressed(
        out,
        X=X_all,
        y=y_all,
        inst_id=inst_all,
        meta=np.array(
            [
                f"variant_id={variant_id}",
                f"seed={seed}",
                f"num_instances={num_instances}",
                f"seqs_per_instance={seqs_per_instance}",
                f"ct_bins={ct_bins}",
                f"n_jobs_pad={n_jobs_pad}",
                f"T_pad={T_pad}",
                f"shard_id={shard_id}",
                f"num_shards={num_shards}",
                f"dp_batch_max={int(dp_batch_max or 0)}",
            ],
            dtype=object,
        ),
    )
    print(f"[generate] wrote {len(y_all)} samples to {out}")


def _split_by_inst(
    inst_id: np.ndarray, test_frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    uniq = np.unique(inst_id)
    rng.shuffle(uniq)
    n_test = int(math.ceil(len(uniq) * float(test_frac)))
    test_set = set(uniq[:n_test].tolist())
    is_test = np.array([i in test_set for i in inst_id], dtype=bool)
    train_idx = np.where(~is_test)[0]
    test_idx = np.where(is_test)[0]
    return train_idx, test_idx


def _eval_metrics(
    X: np.ndarray, y: np.ndarray, inst_id: np.ndarray, yhat: np.ndarray
) -> Dict[str, float]:
    # RMSE overall
    err = yhat - y
    rmse = float(np.sqrt(np.mean(err * err)))

    # Within-instance: pick best predicted sequence among samples of same instance.
    regret_sum = 0.0
    n_inst = 0
    for inst in np.unique(inst_id):
        m = inst_id == inst
        if not np.any(m):
            continue
        yy = y[m]
        yh = yhat[m]
        best_true = float(np.min(yy))
        pick = int(np.argmin(yh))
        regret_sum += float(yy[pick] - best_true)
        n_inst += 1

    return {
        "rmse": rmse,
        "n_samples": float(len(y)),
        "n_instances": float(n_inst),
        "avg_regret_pick": float(regret_sum / max(1, n_inst)),
    }


def _try_import_sklearn_linear():
    try:
        from sklearn.linear_model import Ridge, ElasticNet
        from sklearn.preprocessing import StandardScaler

        return Ridge, ElasticNet, StandardScaler
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


def _load_npz(paths: List[Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xs = []
    ys = []
    insts = []
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            Xs.append(z["X"].astype(np.float32))
            ys.append(z["y"].astype(np.float32))
            insts.append(z["inst_id"].astype(np.int32))
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    inst_id = np.concatenate(insts, axis=0)
    return X, y, inst_id


def cmd_train_ridge(args) -> None:
    paths = _parse_paths(args.data)
    X, y, inst_id = _load_npz(paths)

    train_idx, test_idx = _split_by_inst(inst_id, float(args.test_frac), int(args.seed))

    sk = _try_import_sklearn_linear()
    if sk is None:
        raise RuntimeError("scikit-learn required: pip install scikit-learn")
    Ridge, _, StandardScaler = sk

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])

    model = Ridge(alpha=float(args.alpha), random_state=int(args.seed))
    model.fit(X_train, y[train_idx])

    yhat_train = model.predict(X_train).astype(np.float32)
    yhat_test = model.predict(X_test).astype(np.float32)

    m_train = _eval_metrics(X_train, y[train_idx], inst_id[train_idx], yhat_train)
    m_test = _eval_metrics(X_test, y[test_idx], inst_id[test_idx], yhat_test)

    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump({"model_type": "ridge", "scaler": scaler, "model": model}, f)

    print(f"[train_ridge] saved {out}")
    print("[train_ridge] train:", m_train)
    print("[train_ridge] test:", m_test)


def cmd_train_elasticnet(args) -> None:
    paths = _parse_paths(args.data)
    X, y, inst_id = _load_npz(paths)

    train_idx, test_idx = _split_by_inst(inst_id, float(args.test_frac), int(args.seed))

    sk = _try_import_sklearn_linear()
    if sk is None:
        raise RuntimeError("scikit-learn required: pip install scikit-learn")
    _, ElasticNet, StandardScaler = sk

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])

    model = ElasticNet(
        alpha=float(args.alpha),
        l1_ratio=float(args.l1_ratio),
        max_iter=int(args.max_iter),
        random_state=int(args.seed),
    )
    model.fit(X_train, y[train_idx])

    yhat_train = model.predict(X_train).astype(np.float32)
    yhat_test = model.predict(X_test).astype(np.float32)

    m_train = _eval_metrics(X_train, y[train_idx], inst_id[train_idx], yhat_train)
    m_test = _eval_metrics(X_test, y[test_idx], inst_id[test_idx], yhat_test)

    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump({"model_type": "elasticnet", "scaler": scaler, "model": model}, f)

    print(f"[train_elasticnet] saved {out}")
    print("[train_elasticnet] train:", m_train)
    print("[train_elasticnet] test:", m_test)


def cmd_train_lgbm(args) -> None:
    paths = _parse_paths(args.data)
    X, y, inst_id = _load_npz(paths)

    train_idx, test_idx = _split_by_inst(inst_id, float(args.test_frac), int(args.seed))

    lgb = _try_import_lightgbm()
    if lgb is None:
        raise RuntimeError("lightgbm required: pip install lightgbm")

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    model = lgb.LGBMRegressor(
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        random_state=int(args.seed),
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(int(args.early_stopping_rounds), verbose=False)],
    )

    yhat_train = model.predict(X_train).astype(np.float32)
    yhat_test = model.predict(X_test).astype(np.float32)

    m_train = _eval_metrics(X_train, y_train, inst_id[train_idx], yhat_train)
    m_test = _eval_metrics(X_test, y_test, inst_id[test_idx], yhat_test)

    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump({"model_type": "lgbm", "model": model}, f)

    print(f"[train_lgbm] saved {out}")
    print("[train_lgbm] train:", m_train)
    print("[train_lgbm] test:", m_test)


def cmd_train_xgb(args) -> None:
    paths = _parse_paths(args.data)
    X, y, inst_id = _load_npz(paths)

    train_idx, test_idx = _split_by_inst(inst_id, float(args.test_frac), int(args.seed))

    xgb = _try_import_xgboost()
    if xgb is None:
        raise RuntimeError("xgboost required: pip install xgboost")

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    model = xgb.XGBRegressor(
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        max_depth=int(args.max_depth),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        reg_lambda=float(args.reg_lambda),
        tree_method=str(args.tree_method),
        random_state=int(args.seed),
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
        early_stopping_rounds=int(args.early_stopping_rounds),
    )

    yhat_train = model.predict(X_train).astype(np.float32)
    yhat_test = model.predict(X_test).astype(np.float32)

    m_train = _eval_metrics(X_train, y_train, inst_id[train_idx], yhat_train)
    m_test = _eval_metrics(X_test, y_test, inst_id[test_idx], yhat_test)

    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump({"model_type": "xgb", "model": model}, f)

    print(f"[train_xgb] saved {out}")
    print("[train_xgb] train:", m_train)
    print("[train_xgb] test:", m_test)


def cmd_eval(args) -> None:
    paths = _parse_paths(args.data)
    X, y, inst_id = _load_npz(paths)

    payload = None
    if args.model:
        with Path(args.model).open("rb") as f:
            payload = pickle.load(f)

    if payload is None:
        raise RuntimeError("--model is required")

    model_type = payload.get("model_type")
    if model_type in {"ridge", "elasticnet"}:
        scaler = payload["scaler"]
        model = payload["model"]
        X2 = scaler.transform(X)
        yhat = model.predict(X2).astype(np.float32)
        m = _eval_metrics(X2, y, inst_id, yhat)
    else:
        model = payload["model"]
        yhat = model.predict(X).astype(np.float32)
        m = _eval_metrics(X, y, inst_id, yhat)

    print("[eval]", m)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--out", required=True)
    g.add_argument("--variant_id", type=str, default="q_sequence")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--num_instances", type=int, default=5000)
    g.add_argument("--seqs_per_instance", type=int, default=32)
    g.add_argument("--num_random_per_instance", type=int, default=16)
    g.add_argument(
        "--include_heuristics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include heuristic sequences (SPT/LPT) and perturbations thereof.",
    )
    g.add_argument("--perturbations_per_base", type=int, default=8)
    g.add_argument("--ct_bins", type=int, default=64)
    g.add_argument("--dp_device", type=str, default="cuda", choices=["cpu", "cuda"])
    g.add_argument("--episodes_batch_size", type=int, default=256)
    g.add_argument(
        "--dp_batch_max",
        type=int,
        default=2048,
        help="Max sequences per DP call (chunking to avoid GPU OOM). 0 disables chunking.",
    )
    g.add_argument("--shard_id", type=int, default=0)
    g.add_argument("--num_shards", type=int, default=1)

    tr = sub.add_parser("train_ridge")
    tr.add_argument("--data", required=True)
    tr.add_argument("--model_out", required=True)
    tr.add_argument("--alpha", type=float, default=1.0)
    tr.add_argument("--test_frac", type=float, default=0.2)
    tr.add_argument("--seed", type=int, default=0)

    te = sub.add_parser("train_elasticnet")
    te.add_argument("--data", required=True)
    te.add_argument("--model_out", required=True)
    te.add_argument("--alpha", type=float, default=0.1)
    te.add_argument("--l1_ratio", type=float, default=0.5)
    te.add_argument("--max_iter", type=int, default=5000)
    te.add_argument("--test_frac", type=float, default=0.2)
    te.add_argument("--seed", type=int, default=0)

    tl = sub.add_parser("train_lgbm")
    tl.add_argument("--data", required=True)
    tl.add_argument("--model_out", required=True)
    tl.add_argument("--test_frac", type=float, default=0.2)
    tl.add_argument("--seed", type=int, default=0)
    tl.add_argument("--n_estimators", type=int, default=4000)
    tl.add_argument("--learning_rate", type=float, default=0.05)
    tl.add_argument("--num_leaves", type=int, default=127)
    tl.add_argument("--subsample", type=float, default=0.9)
    tl.add_argument("--colsample_bytree", type=float, default=0.9)
    tl.add_argument("--early_stopping_rounds", type=int, default=100)

    tx = sub.add_parser("train_xgb")
    tx.add_argument("--data", required=True)
    tx.add_argument("--model_out", required=True)
    tx.add_argument("--test_frac", type=float, default=0.2)
    tx.add_argument("--seed", type=int, default=0)
    tx.add_argument("--n_estimators", type=int, default=4000)
    tx.add_argument("--learning_rate", type=float, default=0.05)
    tx.add_argument("--max_depth", type=int, default=8)
    tx.add_argument("--subsample", type=float, default=0.9)
    tx.add_argument("--colsample_bytree", type=float, default=0.9)
    tx.add_argument("--reg_lambda", type=float, default=1.0)
    tx.add_argument("--tree_method", type=str, default="hist")
    tx.add_argument("--early_stopping_rounds", type=int, default=100)

    ev = sub.add_parser("eval")
    ev.add_argument("--data", required=True)
    ev.add_argument("--model", required=True)

    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.cmd == "generate":
        generate_dataset_shard(
            out=Path(args.out),
            variant_id=str(args.variant_id),
            seed=int(args.seed),
            num_instances=int(args.num_instances),
            seqs_per_instance=int(args.seqs_per_instance),
            num_random_per_instance=int(args.num_random_per_instance),
            include_heuristics=bool(args.include_heuristics),
            perturbations_per_base=int(args.perturbations_per_base),
            ct_bins=int(args.ct_bins),
            dp_device=str(args.dp_device),
            shard_id=int(args.shard_id),
            num_shards=int(args.num_shards),
            episodes_batch_size=int(args.episodes_batch_size),
            dp_batch_max=int(getattr(args, "dp_batch_max", 0) or 0),
        )
        return
    if args.cmd == "train_ridge":
        cmd_train_ridge(args)
        return
    if args.cmd == "train_elasticnet":
        cmd_train_elasticnet(args)
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

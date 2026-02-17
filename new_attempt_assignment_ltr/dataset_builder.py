from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from PaST.config import DataConfig
from PaST.data.sm_benchmark_data import RawInstance, generate_raw_instance

from .assignment_labeling import score_assignment_l1
from .feasible_assignment import build_candidate_assignment_pool
from .features import FeatureConfig, dict_to_feature_vector, extract_assignment_features


@dataclass(frozen=True)
class BuildConfig:
    n_instances: int = 200
    pool_size: int = 40
    seed: int = 42
    feature_max_p: int = 20

    # K sampling (epsilon constraint proxy)
    k_bias_tight: float = 2.0  # u^k_bias_tight


def _sample_K(raw: RawInstance, rng: np.random.Generator, bias_tight: float) -> int:
    tmin = int(np.ceil(float(sum(raw.p)) / float(max(1, raw.m))))
    tmin = max(1, min(tmin, int(raw.T_max)))

    u = float(rng.random())
    u = u ** float(max(1e-9, bias_tight))
    K = int(tmin + (int(raw.T_max) - tmin) * u)
    return max(tmin, min(int(raw.T_max), int(K)))


def build_ranking_dataset(
    *,
    out_dir: str,
    cfg: BuildConfig,
) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(cfg.seed))
    data_cfg = DataConfig()

    X_rows: List[np.ndarray] = []
    y_rows: List[float] = []
    qid_rows: List[int] = []

    feature_cfg = FeatureConfig(max_p=int(cfg.feature_max_p))
    feature_keys: Optional[List[str]] = None

    meta: Dict[str, object] = {
        "n_instances": int(cfg.n_instances),
        "pool_size": int(cfg.pool_size),
        "seed": int(cfg.seed),
        "feature_max_p": int(cfg.feature_max_p),
        "k_bias_tight": float(cfg.k_bias_tight),
        "label": "-energy (higher better)",
        "labeler": "score_assignment_l1(mode=cheap_first)",
    }

    qid = 0
    for i in range(int(cfg.n_instances)):
        # generate_raw_instance expects a random.Random-like RNG (uses randrange).
        raw = generate_raw_instance(
            data_cfg,
            random.Random(int(cfg.seed + i)),
            instance_id=i,
        )
        K = _sample_K(raw, rng, cfg.k_bias_tight)

        pool = build_candidate_assignment_pool(
            processing_times=raw.p,
            machine_energy_rates=raw.e,
            n_machines=raw.m,
            K=K,
            pool_size=int(cfg.pool_size),
            seed=int(cfg.seed + 1_000_000 + i),
        )
        if not pool:
            continue

        # Score and feature each assignment (ranking group = same instance + K)
        scored: List[Tuple[List[int], float]] = []
        for a in pool:
            energy, _per_m, feas = score_assignment_l1(instance=raw, assignment=a, K=K, mode="cheap_first")
            if (not feas) or (not np.isfinite(energy)):
                continue
            scored.append((a, float(energy)))

        if len(scored) < 5:
            continue

        # Establish feature keys from first feasible row
        if feature_keys is None:
            feats0 = extract_assignment_features(instance=raw, assignment=scored[0][0], K=K, config=feature_cfg)
            feature_keys = sorted(feats0.keys())
            meta["n_features"] = int(len(feature_keys))
            meta["feature_keys"] = feature_keys

        assert feature_keys is not None

        for a, energy in scored:
            feats = extract_assignment_features(instance=raw, assignment=a, K=K, config=feature_cfg)
            x = dict_to_feature_vector(feats, feature_keys)
            # For ranking: higher label is better
            y = -float(energy)
            X_rows.append(x)
            y_rows.append(y)
            qid_rows.append(int(qid))

        qid += 1

    if feature_keys is None:
        raise RuntimeError("No feasible data generated. Try increasing pool_size or n_instances.")

    X = np.stack(X_rows, axis=0).astype(np.float32)
    y = np.asarray(y_rows, dtype=np.float32)
    qid_arr = np.asarray(qid_rows, dtype=np.int32)

    npz_path = out / "dataset.npz"
    np.savez_compressed(npz_path, X=X, y=y, qid=qid_arr)

    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return str(npz_path)

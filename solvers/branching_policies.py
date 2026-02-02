"""Branching policy helpers for PaST branch-and-bound.

Policies return either:
- a list of job IDs in the order to branch, or
- a list of durations (processing times) in the order to branch.

These are consumed by BranchAndBoundSolver(branching_policy=...).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

from PaST.bb_branching_features import WindowFeatureCache, extract_row_features


def _unique_duration_candidates(
    processing_times: np.ndarray, remaining_jobs: Set[int]
) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for j in remaining_jobs:
        d = int(processing_times[j])
        if d not in out:
            out[d] = int(j)
    return out


@dataclass
class MinWindowBranchingPolicy:
    """Deterministic, safe baseline: sort by min window cost w_min(d)."""

    duration_vocab: Sequence[int]

    def __call__(
        self, partial_sequence: List[int], remaining_jobs: Set[int], solver
    ) -> List[int]:
        instance = solver.instance
        cand = _unique_duration_candidates(instance.processing_times, remaining_jobs)
        cache = WindowFeatureCache(instance.energy_costs)

        scored: List[tuple[float, int]] = []
        for d in cand.keys():
            ws = cache.window_stats(int(d))
            scored.append((float(ws.get("w_min", 0.0)), int(d)))

        scored.sort(key=lambda x: x[0])
        return [d for _, d in scored]


@dataclass
class PairwiseLogisticBranchingPolicy:
    """Learned duration-ranker trained by train_bb_branch_policy.py.

    Safety:
    - If any errors/non-finite predictions occur, falls back to MinWindow policy.
    - If predictions are nearly tied (top-2 margin < tie_eps), falls back.
    """

    model_path: str
    duration_vocab: Sequence[int]
    tie_eps: float = 1e-6

    def __post_init__(self):
        with Path(self.model_path).open("rb") as f:
            payload = pickle.load(f)
        self._feature_cols: List[str] = payload["feature_cols"]
        model = payload["model"]
        self._scaler = model["scaler"]
        self._lr = model["lr"]
        self._fallback = MinWindowBranchingPolicy(duration_vocab=self.duration_vocab)

    def __call__(
        self, partial_sequence: List[int], remaining_jobs: Set[int], solver
    ) -> List[int]:
        instance = solver.instance
        cand = _unique_duration_candidates(instance.processing_times, remaining_jobs)
        if len(cand) <= 1:
            return []

        try:
            cache = WindowFeatureCache(instance.energy_costs)
            items: List[tuple[int, np.ndarray]] = []
            for d in cand.keys():
                feats = extract_row_features(
                    instance=instance,
                    duration_vocab=self.duration_vocab,
                    window_cache=cache,
                    partial_sequence=partial_sequence,
                    remaining_jobs=sorted(list(remaining_jobs)),
                    candidate_d=int(d),
                )
                x = np.array(
                    [feats.get(c, 0.0) for c in self._feature_cols], dtype=np.float64
                )
                items.append((int(d), x))

            X = np.stack([x for _, x in items], axis=0)

            # Score s(x)=w^T z where z is standardized x.
            Xs = self._scaler.transform(X)
            w = self._lr.coef_.reshape(-1)
            scores = Xs @ w
            scores = np.nan_to_num(scores, nan=np.inf, posinf=np.inf, neginf=np.inf)

            # Higher score = better candidate.
            order = np.argsort(-scores)
            if order.size >= 2:
                margin = float(scores[order[0]] - scores[order[1]])
                if margin < float(self.tie_eps):
                    return self._fallback(partial_sequence, remaining_jobs, solver)

            return [items[i][0] for i in order.tolist()]
        except Exception:
            return self._fallback(partial_sequence, remaining_jobs, solver)


@dataclass
class LightGBMBranchingPolicy:
    """Duration-branching policy using a saved LightGBM ranker."""

    model_path: str
    duration_vocab: Sequence[int]
    tie_eps: float = 1e-6

    def __post_init__(self):
        with Path(self.model_path).open("rb") as f:
            payload = pickle.load(f)
        self._feature_cols: List[str] = payload["feature_cols"]
        self._model = payload["model"]["model"]
        self._fallback = MinWindowBranchingPolicy(duration_vocab=self.duration_vocab)

    def __call__(
        self, partial_sequence: List[int], remaining_jobs: Set[int], solver
    ) -> List[int]:
        instance = solver.instance
        cand = _unique_duration_candidates(instance.processing_times, remaining_jobs)
        if len(cand) <= 1:
            return []
        try:
            cache = WindowFeatureCache(instance.energy_costs)
            items: List[tuple[int, np.ndarray]] = []
            for d in cand.keys():
                feats = extract_row_features(
                    instance=instance,
                    duration_vocab=self.duration_vocab,
                    window_cache=cache,
                    partial_sequence=partial_sequence,
                    remaining_jobs=sorted(list(remaining_jobs)),
                    candidate_d=int(d),
                )
                x = np.array(
                    [feats.get(c, 0.0) for c in self._feature_cols], dtype=np.float64
                )
                items.append((int(d), x))
            X = np.stack([x for _, x in items], axis=0)
            scores = self._model.predict(X)
            scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
            order = np.argsort(-scores)
            if order.size >= 2:
                margin = float(scores[order[0]] - scores[order[1]])
                if margin < float(self.tie_eps):
                    return self._fallback(partial_sequence, remaining_jobs, solver)
            return [items[i][0] for i in order.tolist()]
        except Exception:
            return self._fallback(partial_sequence, remaining_jobs, solver)


@dataclass
class XGBoostBranchingPolicy:
    """Duration-branching policy using a saved XGBoost ranker."""

    model_path: str
    duration_vocab: Sequence[int]
    tie_eps: float = 1e-6

    def __post_init__(self):
        with Path(self.model_path).open("rb") as f:
            payload = pickle.load(f)
        self._feature_cols: List[str] = payload["feature_cols"]
        self._model = payload["model"]["model"]
        self._fallback = MinWindowBranchingPolicy(duration_vocab=self.duration_vocab)

    def __call__(
        self, partial_sequence: List[int], remaining_jobs: Set[int], solver
    ) -> List[int]:
        instance = solver.instance
        cand = _unique_duration_candidates(instance.processing_times, remaining_jobs)
        if len(cand) <= 1:
            return []
        try:
            cache = WindowFeatureCache(instance.energy_costs)
            items: List[tuple[int, np.ndarray]] = []
            for d in cand.keys():
                feats = extract_row_features(
                    instance=instance,
                    duration_vocab=self.duration_vocab,
                    window_cache=cache,
                    partial_sequence=partial_sequence,
                    remaining_jobs=sorted(list(remaining_jobs)),
                    candidate_d=int(d),
                )
                x = np.array(
                    [feats.get(c, 0.0) for c in self._feature_cols], dtype=np.float64
                )
                items.append((int(d), x))
            X = np.stack([x for _, x in items], axis=0)
            scores = self._model.predict(X)
            scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
            order = np.argsort(-scores)
            if order.size >= 2:
                margin = float(scores[order[0]] - scores[order[1]])
                if margin < float(self.tie_eps):
                    return self._fallback(partial_sequence, remaining_jobs, solver)
            return [items[i][0] for i in order.tolist()]
        except Exception:
            return self._fallback(partial_sequence, remaining_jobs, solver)

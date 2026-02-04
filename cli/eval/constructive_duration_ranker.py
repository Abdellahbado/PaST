"""Constructive decoding (greedy + SGBS) using a saved duration-class ranker.

This adapts the *branching policy* duration-ranker (trained by
`PaST/train_bb_branch_policy.py`) into a constructive decoder that produces a
job sequence (ordering) and evaluates it with DP scheduling.

Key idea:
- The ranker scores *durations* (processing times) given the current prefix and
  remaining-job multiset.
- We map duration scores to per-job scores by assigning each job the score of
  its duration.
- Since the Q-sequence code expects "lower is better", we use q(job) = -score.

This module is intentionally CPU/simple and does not depend on the GPU env.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from PaST.bb_branching_features import WindowFeatureCache, extract_row_features
from PaST.solvers.baselines_sequence_dp import (
    DPResult,
    dp_schedule_for_job_sequence,
    lpt_sequence,
    spt_sequence,
)
from PaST.solvers.bnb_solver_custom import Instance


def _duration_vocab_from_feature_cols(feature_cols: Sequence[str]) -> List[int]:
    vocab: Set[int] = set()
    prefix = "rem_count_d"
    for c in feature_cols:
        if c.startswith(prefix):
            suf = c[len(prefix) :]
            if suf.isdigit():
                vocab.add(int(suf))
    return sorted(vocab)


def _construct_instance_from_batch_single(single: Dict[str, Any]) -> Instance:
    n_jobs = int(single["n_jobs"][0])
    p_subset = np.asarray(single["p_subset"][0][:n_jobs], dtype=np.int32)

    T = int(single["T_limit"][0])
    ct = np.asarray(single["ct"][0], dtype=np.float64)
    e_single = float(single.get("e_single", np.array([1.0]))[0])

    # Match DP semantics: energy = e_single * sum(ct[t]) over processing time.
    energy_costs = (ct[:T] * e_single).astype(np.float64)

    return Instance(
        n_jobs=int(n_jobs),
        processing_times=p_subset.astype(np.int32),
        T=int(T),
        energy_costs=energy_costs,
    )


@dataclass
class DurationRanker:
    model_path: str
    tie_eps: float = 1e-6

    def __post_init__(self) -> None:
        with Path(self.model_path).open("rb") as f:
            payload = pickle.load(f)

        self.model_type: str = str(payload.get("model_type", ""))
        self.feature_cols: List[str] = list(payload["feature_cols"])
        self.duration_vocab: List[int] = _duration_vocab_from_feature_cols(
            self.feature_cols
        )
        self._payload_model = payload["model"]

        # For fallback ordering when scores are tied or something goes wrong.
        self._fallback_min_window = True

    def _min_window_order(
        self, instance: Instance, remaining_jobs: Set[int]
    ) -> List[int]:
        cache = WindowFeatureCache(instance.energy_costs)
        # Unique durations among remaining.
        cand_ds: Set[int] = set(
            int(instance.processing_times[j]) for j in remaining_jobs
        )
        scored: List[Tuple[float, int]] = []
        for d in cand_ds:
            ws = cache.window_stats(int(d))
            scored.append((float(ws.get("w_min", 0.0)), int(d)))
        scored.sort(key=lambda x: x[0])
        return [d for _, d in scored]

    def score_durations(
        self,
        *,
        instance: Instance,
        partial_sequence: Sequence[int],
        remaining_jobs: Sequence[int],
        candidate_ds: Iterable[int],
    ) -> Dict[int, float]:
        cache = WindowFeatureCache(instance.energy_costs)
        items: List[Tuple[int, np.ndarray]] = []
        rem_sorted = sorted([int(j) for j in remaining_jobs])
        for d in candidate_ds:
            feats = extract_row_features(
                instance=instance,
                duration_vocab=self.duration_vocab,
                window_cache=cache,
                partial_sequence=list(partial_sequence),
                remaining_jobs=rem_sorted,
                candidate_d=int(d),
            )
            x = np.array(
                [feats.get(c, 0.0) for c in self.feature_cols], dtype=np.float64
            )
            items.append((int(d), x))

        if not items:
            return {}

        X = np.stack([x for _, x in items], axis=0)

        # Mirror the scoring conventions used in PaST/solvers/branching_policies.py:
        # - pairwise_logistic: higher score is better (w^T z)
        # - lgbm/xgb rankers: model.predict(X), higher is better
        if self.model_type == "pairwise_logistic":
            scaler = self._payload_model["scaler"]
            lr = self._payload_model["lr"]
            Xs = scaler.transform(X)
            w = lr.coef_.reshape(-1)
            scores = Xs @ w
            scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        elif self.model_type in {"lgbm_ranker", "xgb_ranker"}:
            model = self._payload_model["model"]
            scores = model.predict(X)
            scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        else:
            raise ValueError(
                f"Unsupported model_type={self.model_type!r} in {self.model_path}"
            )

        return {items[i][0]: float(scores[i]) for i in range(len(items))}

    def duration_order(
        self,
        *,
        instance: Instance,
        partial_sequence: List[int],
        remaining_jobs: Set[int],
    ) -> List[int]:
        cand_ds = sorted({int(instance.processing_times[j]) for j in remaining_jobs})
        if len(cand_ds) <= 1:
            return cand_ds

        try:
            scores = self.score_durations(
                instance=instance,
                partial_sequence=partial_sequence,
                remaining_jobs=sorted(list(remaining_jobs)),
                candidate_ds=cand_ds,
            )
            ds = np.array(cand_ds, dtype=np.int32)
            sc = np.array(
                [scores.get(int(d), -np.inf) for d in cand_ds], dtype=np.float64
            )
            order = np.argsort(-sc)
            if order.size >= 2:
                margin = float(sc[order[0]] - sc[order[1]])
                if margin < float(self.tie_eps) and self._fallback_min_window:
                    return self._min_window_order(instance, remaining_jobs)
            return [int(ds[i]) for i in order.tolist()]
        except Exception:
            if self._fallback_min_window:
                return self._min_window_order(instance, remaining_jobs)
            raise

    def job_q_values(
        self,
        *,
        instance: Instance,
        partial_sequence: List[int],
        remaining_jobs: Set[int],
    ) -> np.ndarray:
        """Return per-job Q-values where lower is better.

        For jobs not in remaining_jobs, Q = +inf.
        """
        n = int(instance.n_jobs)
        q = np.full((n,), np.inf, dtype=np.float64)
        if not remaining_jobs:
            return q

        cand_ds = sorted({int(instance.processing_times[j]) for j in remaining_jobs})
        scores = self.score_durations(
            instance=instance,
            partial_sequence=partial_sequence,
            remaining_jobs=sorted(list(remaining_jobs)),
            candidate_ds=cand_ds,
        )
        for j in remaining_jobs:
            d = int(instance.processing_times[j])
            s = float(scores.get(d, -np.inf))
            q[int(j)] = -s
        return q


def greedy_decode_duration_ranker(
    ranker: DurationRanker,
    batch_data_single: Dict[str, Any],
) -> DPResult:
    instance = _construct_instance_from_batch_single(batch_data_single)
    remaining: Set[int] = set(range(int(instance.n_jobs)))
    seq: List[int] = []

    while remaining:
        q = ranker.job_q_values(
            instance=instance, partial_sequence=seq, remaining_jobs=remaining
        )
        # Pick valid job with minimal Q.
        j = int(np.argmin(q))
        if not np.isfinite(q[j]):
            # Fallback: deterministic smallest id.
            j = min(remaining)
        seq.append(j)
        remaining.remove(j)

    return dp_schedule_for_job_sequence(batch_data_single, seq)


def _complete_rollout_sequence(
    *,
    ranker: DurationRanker,
    batch_data_single: Dict[str, Any],
    instance: Instance,
    prefix: List[int],
    remaining: Set[int],
    rollout_policy: str,
    rollout_seed: int,
) -> List[int]:
    rollout_policy = (rollout_policy or "model").strip().lower()

    seq = list(prefix)
    rem = set(remaining)

    if rollout_policy == "model":
        while rem:
            q = ranker.job_q_values(
                instance=instance, partial_sequence=seq, remaining_jobs=rem
            )
            j = int(np.argmin(q))
            if not np.isfinite(q[j]):
                j = min(rem)
            seq.append(j)
            rem.remove(j)
        return seq

    if rollout_policy == "random":
        # Deterministic-ish RNG per rollout call based on prefix.
        h = int(rollout_seed)
        for x in seq:
            h = (h * 1000003) ^ int(x + 1)
        rng = np.random.RandomState(h & 0xFFFFFFFF)
        while rem:
            j = int(rng.choice(sorted(list(rem))))
            seq.append(j)
            rem.remove(j)
        return seq

    if rollout_policy in {"spt", "lpt"}:
        n_jobs = int(batch_data_single["n_jobs"][0])
        p_subset = np.asarray(batch_data_single["p_subset"][0][:n_jobs], dtype=np.int32)
        if rollout_policy == "spt":
            tail = spt_sequence(p_subset, n_jobs)
        else:
            tail = lpt_sequence(p_subset, n_jobs)
        # Respect already-chosen prefix: append remaining in heuristic order.
        tail = [j for j in tail if j in rem]
        return seq + tail

    raise ValueError("rollout_policy must be one of: model, random, spt, lpt")


def sgbs_decode_duration_ranker(
    ranker: DurationRanker,
    batch_data_single: Dict[str, Any],
    *,
    beta: int,
    gamma: int,
    rollout_policy: str = "model",
    rollout_seed: int = 0,
) -> DPResult:
    """SGBS-like beam search over job sequences using the duration ranker.

    Expand by top-gamma jobs (lowest q = best). Simulate completion by rollout,
    evaluate by DP energy, prune to beta.
    """

    instance = _construct_instance_from_batch_single(batch_data_single)
    n = int(instance.n_jobs)

    if n <= 2 or (int(beta) == 1 and int(gamma) == 1):
        return greedy_decode_duration_ranker(ranker, batch_data_single)

    beta = int(beta)
    gamma = int(gamma)

    # Beam stores partial prefixes.
    beam: List[Tuple[List[int], Set[int]]] = [([], set(range(n)))]

    best_res: Optional[DPResult] = None

    for step in range(n):
        expanded: List[Tuple[float, List[int], Set[int]]] = []

        for prefix, remaining in beam:
            if not remaining:
                full = list(prefix)
                res = dp_schedule_for_job_sequence(batch_data_single, full)
                if best_res is None or res.total_energy < best_res.total_energy:
                    best_res = res
                continue

            q = ranker.job_q_values(
                instance=instance, partial_sequence=prefix, remaining_jobs=remaining
            )
            # Candidate jobs = gamma smallest q among remaining.
            rem_list = sorted(list(remaining))
            q_rem = np.array([q[j] for j in rem_list], dtype=np.float64)
            # Handle all-inf edge case.
            if not np.isfinite(q_rem).any():
                cand_jobs = rem_list[: min(gamma, len(rem_list))]
            else:
                order = np.argsort(q_rem)
                cand_jobs = [
                    rem_list[i] for i in order[: min(gamma, len(rem_list))].tolist()
                ]

            for j in cand_jobs:
                new_prefix = prefix + [int(j)]
                new_remaining = set(remaining)
                new_remaining.remove(int(j))

                full_seq = _complete_rollout_sequence(
                    ranker=ranker,
                    batch_data_single=batch_data_single,
                    instance=instance,
                    prefix=new_prefix,
                    remaining=new_remaining,
                    rollout_policy=rollout_policy,
                    rollout_seed=int(rollout_seed) + int(step) * 1009,
                )
                res = dp_schedule_for_job_sequence(batch_data_single, full_seq)
                energy = float(res.total_energy)
                if best_res is None or energy < best_res.total_energy:
                    best_res = res

                expanded.append((energy, new_prefix, new_remaining))

        if not expanded:
            break

        expanded.sort(key=lambda x: x[0])
        beam = [(pfx, rem) for _, pfx, rem in expanded[:beta]]

    if best_res is None:
        return greedy_decode_duration_ranker(ranker, batch_data_single)

    return best_res

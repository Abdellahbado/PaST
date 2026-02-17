from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class EvalStats:
    n_groups: int
    mean_regret: float
    median_regret: float
    top1_acc: float


def _group_indices(qid: np.ndarray):
    order = np.argsort(qid, kind="stable")
    q_sorted = qid[order]
    boundaries = np.flatnonzero(np.r_[True, q_sorted[1:] != q_sorted[:-1], True])
    groups = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        groups.append(order[a:b])
    return groups


def evaluate_ranker_predictions(qid: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> EvalStats:
    groups = _group_indices(qid)

    regrets = []
    correct = 0
    for idx in groups:
        yt = y_true[idx]
        yp = y_pred[idx]

        # best true label (higher is better)
        best_true = float(np.max(yt))
        # chosen by model
        chosen = int(idx[int(np.argmax(yp))])
        chosen_true = float(y_true[chosen])

        regret = best_true - chosen_true
        regrets.append(regret)

        # top1 accuracy: chosen is among best
        if abs(chosen_true - best_true) <= 1e-9:
            correct += 1

    regrets_arr = np.asarray(regrets, dtype=np.float64)
    return EvalStats(
        n_groups=int(len(groups)),
        mean_regret=float(np.mean(regrets_arr)),
        median_regret=float(np.median(regrets_arr)),
        top1_acc=float(correct) / float(max(1, len(groups))),
    )

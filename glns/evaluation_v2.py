"""Assignment-only evaluation engine for decomposed G-LNS.

Replaces the sequence-based evaluation loop with an assignment-based one:
- Operators work on assignment vectors (which machine gets which job)
- Per-machine sequencing is delegated to the optimal DP solver
- Supports parallel evaluation across machines on HPC

This module is the assignment-only counterpart of glns/evaluation.py.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import random
from typing import Dict, List, Optional, Tuple

from glns.config import EvalConfig, SandboxConfig
from glns.pareto import ArchiveEntry, ParetoArchive
from glns.sandbox import run_operator_sandboxed
from glns.schemas import OperatorRecord
from glns.sequencing import (
    evaluate_assignment,
    make_initial_assignment,
    sequences_to_assignment,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core evaluation loop (assignment-only)
# ---------------------------------------------------------------------------

SELECTION_STRATEGIES = ["random", "crowded", "extreme_tec", "extreme_cmax", "stagnated"]


def run_evaluation_phase_v2(
    destroy_pool: List[OperatorRecord],
    repair_pool: List[OperatorRecord],
    instances: List[dict],
    archive: ParetoArchive,
    eval_cfg: EvalConfig,
    sandbox_cfg: SandboxConfig,
    rng: random.Random,
    sequencing_mode: str = "auto",
    n_workers: int = 0,
) -> Tuple[
    Dict[str, float], Dict[str, float], Dict[Tuple[str, str], float], Dict[str, float]
]:
    """Run K multi-episode evaluations using assignment-only operators.

    Operators work on assignment vectors. After each repair, the sequencing
    layer finds optimal job orderings per machine.

    Args:
        sequencing_mode: "optimal", "heuristic", or "auto"
        n_workers: number of parallel workers for batch evaluation (0 = auto)

    Returns:
        (F_destroy, F_repair, synergy, stats)
    """
    n_d = len(destroy_pool)
    n_r = len(repair_pool)
    if n_d == 0 or n_r == 0:
        raise ValueError("Both pools must be non-empty for evaluation.")

    sigma = eval_cfg.sigma

    F_d: Dict[str, float] = {op.id: 0.0 for op in destroy_pool}
    F_r: Dict[str, float] = {op.id: 0.0 for op in repair_pool}
    synergy: Dict[Tuple[str, str], float] = {}

    fail_d: Dict[str, int] = {op.id: 0 for op in destroy_pool}
    fail_r: Dict[str, int] = {op.id: 0 for op in repair_pool}
    FAIL_THRESHOLD = 5

    phase_archive_start = archive.size()
    stats: Dict[str, float] = {
        "sigma1": 0,
        "sigma2": 0,
        "sigma3": 0,
        "sigma4": 0,
        "sa_accept": 0,
        "sa_reject": 0,
        "sa_worse_delta_sum": 0.0,
        "sa_worse_delta_count": 0,
        "sa_worse_delta_max": 0.0,
        "destroy_fail": 0,
        "repair_fail": 0,
        "invalid_candidate": 0,
        "inf_eval": 0,
        "episodes_skipped": 0,
        "archive_entered": 0,
        "archive_dominated_removed": 0,
    }

    for episode in range(eval_cfg.K_episodes):
        inst = rng.choice(instances)
        T_limit = inst["T"]
        inst_id = int(inst.get("instance_id", 0))
        n_jobs = inst["n"]
        m = inst["m"]

        # Pick initial assignment from archive or build new one
        if archive.size(inst_id) > 0:
            strategy = rng.choice(SELECTION_STRATEGIES)
            entry = archive.select(strategy, rng, instance_id=inst_id)
            # Convert archived sequences to assignment
            err0 = _validate_assignment_from_entry(entry, inst)
            if err0 is not None:
                logger.warning(
                    "Dropping invalid archive entry for inst_id=%s: %s", inst_id, err0
                )
                archive.entries = [e for e in archive.entries if e is not entry]
                cur_assign = make_initial_assignment(inst)
            else:
                cur_assign = sequences_to_assignment(entry.sequences, n_jobs)
        else:
            cur_assign = make_initial_assignment(inst)

        # Validate starting assignment
        err_init = _validate_complete_assignment(cur_assign, inst)
        if err_init is not None:
            logger.error("Initial assignment invalid: %s", err_init)
            stats["episodes_skipped"] += 1
            continue

        # Evaluate initial assignment (with optimal sequencing)
        cur_energy, cur_cmax, cur_seqs, cur_starts = evaluate_assignment(
            cur_assign,
            inst,
            sequencing_mode=sequencing_mode,
        )
        if cur_energy == float("inf"):
            stats["episodes_skipped"] += 1
            continue

        # Random scalarisation weight
        w = rng.random()
        ref_cmax = max(float(inst["T"]), 1.0)
        ref_tec = max(float(sum(inst["p"]) * max(inst["e"]) * max(inst["ct"])), 1.0)

        def scalarise(cmax: int, tec: float) -> float:
            return w * cmax / ref_cmax + (1 - w) * tec / ref_tec

        cur_scalar = scalarise(cur_cmax, cur_energy)
        best_scalar = cur_scalar
        tau = eval_cfg.sa_T0

        W_d = {op.id: 1.0 for op in destroy_pool}
        W_r = {op.id: 1.0 for op in repair_pool}

        for it in range(eval_cfg.T_iters):
            # Roulette selection
            d_ids = [op.id for op in destroy_pool]
            d_weights = [
                max(W_d[oid], 1e-6) if fail_d[oid] < FAIL_THRESHOLD else 1e-9
                for oid in d_ids
            ]
            d_total = sum(d_weights)
            d_probs = [ww / d_total for ww in d_weights]
            d_idx = _roulette(d_probs, rng)
            d_op = destroy_pool[d_idx]

            r_ids = [op.id for op in repair_pool]
            r_weights = [
                max(W_r[oid], 1e-6) if fail_r[oid] < FAIL_THRESHOLD else 1e-9
                for oid in r_ids
            ]
            r_total = sum(r_weights)
            r_probs = [ww / r_total for ww in r_weights]
            r_idx = _roulette(r_probs, rng)
            r_op = repair_pool[r_idx]

            # Destroy
            destroy_cnt = max(1, int(eval_cfg.destroy_ratio * n_jobs))
            ok_d, d_result = run_operator_sandboxed(
                fn_code=d_op.code,
                fn_name="destroy",
                args=(
                    list(cur_assign),
                    destroy_cnt,
                    inst,
                    random.Random(rng.randint(0, 2**31)),
                ),
                cfg=sandbox_cfg,
            )
            if not ok_d:
                fail_d[d_op.id] += 1
                stats["destroy_fail"] += 1
                _record_sigma(
                    sigma[3], d_op.id, r_op.id, F_d, F_r, synergy, W_d, W_r, eval_cfg
                )
                stats["sigma4"] += 1
                tau *= eval_cfg.sa_alpha
                continue
            removed_jobs, partial_assign = d_result

            # Repair
            ok_r, r_result = run_operator_sandboxed(
                fn_code=r_op.code,
                fn_name="repair",
                args=(
                    list(partial_assign),
                    list(removed_jobs),
                    inst,
                    random.Random(rng.randint(0, 2**31)),
                ),
                cfg=sandbox_cfg,
            )
            if not ok_r:
                fail_r[r_op.id] += 1
                stats["repair_fail"] += 1
                _record_sigma(
                    sigma[3], d_op.id, r_op.id, F_d, F_r, synergy, W_d, W_r, eval_cfg
                )
                stats["sigma4"] += 1
                tau *= eval_cfg.sa_alpha
                continue

            cand_assign = r_result

            # Validate candidate assignment
            err = _validate_complete_assignment(cand_assign, inst)
            if err is not None:
                fail_r[r_op.id] += 1
                stats["invalid_candidate"] += 1
                _record_sigma(
                    sigma[3], d_op.id, r_op.id, F_d, F_r, synergy, W_d, W_r, eval_cfg
                )
                stats["sigma4"] += 1
                tau *= eval_cfg.sa_alpha
                continue

            # Evaluate candidate (with optimal sequencing)
            cand_energy, cand_cmax, cand_seqs, cand_starts = evaluate_assignment(
                cand_assign,
                inst,
                sequencing_mode=sequencing_mode,
            )
            if cand_energy == float("inf"):
                _record_sigma(
                    sigma[3], d_op.id, r_op.id, F_d, F_r, synergy, W_d, W_r, eval_cfg
                )
                stats["inf_eval"] += 1
                stats["sigma4"] += 1
                tau *= eval_cfg.sa_alpha
                continue

            # Archive entry
            cand_entry = ArchiveEntry(
                instance_id=inst_id,
                makespan=cand_cmax,
                energy=cand_energy,
                sequences=cand_seqs,
                start_times=cand_starts,
            )
            cand_scalar = scalarise(cand_cmax, cand_energy)

            entered, dominated_removed = archive.add(copy.deepcopy(cand_entry))
            if entered:
                stats["archive_entered"] += 1
            stats["archive_dominated_removed"] += int(dominated_removed)

            # SA acceptance hierarchy
            accept_current = False
            if cand_scalar < best_scalar:
                reward = sigma[0]
                best_scalar = cand_scalar
                accept_current = True
                stats["sigma1"] += 1
            elif cand_scalar < cur_scalar:
                reward = sigma[1]
                accept_current = True
                stats["sigma2"] += 1
            else:
                delta = cand_scalar - cur_scalar
                if tau > 1e-12:
                    accept_current = rng.random() < math.exp(-delta / max(tau, 1e-12))
                else:
                    accept_current = False
                reward = sigma[2] if accept_current else sigma[3]
                if accept_current:
                    stats["sigma3"] += 1
                    stats["sa_accept"] += 1
                    stats["sa_worse_delta_sum"] += float(delta)
                    stats["sa_worse_delta_count"] += 1
                    stats["sa_worse_delta_max"] = max(
                        float(stats["sa_worse_delta_max"]), float(delta)
                    )
                else:
                    stats["sigma4"] += 1
                    stats["sa_reject"] += 1

            if accept_current:
                cur_assign = cand_assign
                cur_energy = cand_energy
                cur_cmax = cand_cmax
                cur_scalar = cand_scalar
                cur_seqs = cand_seqs
                cur_starts = cand_starts

            _record_sigma(
                reward, d_op.id, r_op.id, F_d, F_r, synergy, W_d, W_r, eval_cfg
            )
            tau *= eval_cfg.sa_alpha

    # Stagnation tracking
    for e in archive.entries:
        e.stagnation += 1

    # Phase summary
    phase_archive_end_total = archive.size()
    total_sa = stats["sa_accept"] + stats["sa_reject"]
    sa_rate = (stats["sa_accept"] / total_sa) if total_sa else 0.0
    worse_cnt = float(stats["sa_worse_delta_count"])
    worse_delta_mean = (
        float(stats["sa_worse_delta_sum"]) / worse_cnt if worse_cnt > 0 else 0.0
    )
    logger.info(
        "Eval-v2 done | episodes=%d iters=%d (skipped=%d) | σ1=%d σ2=%d σ3=%d σ4=%d | SA accept=%.1f%% (Δworse mean=%.4f max=%.4f) | archive_total %d→%d (entered=%d dom_removed=%d) | fails: d=%d r=%d invalid=%d inf=%d",
        eval_cfg.K_episodes,
        eval_cfg.T_iters,
        int(stats["episodes_skipped"]),
        stats["sigma1"],
        stats["sigma2"],
        stats["sigma3"],
        stats["sigma4"],
        100.0 * sa_rate,
        worse_delta_mean,
        float(stats["sa_worse_delta_max"]),
        phase_archive_start,
        phase_archive_end_total,
        stats["archive_entered"],
        stats["archive_dominated_removed"],
        stats["destroy_fail"],
        stats["repair_fail"],
        stats["invalid_candidate"],
        stats["inf_eval"],
    )

    stats_out = dict(stats)
    stats_out["sa_rate"] = float(sa_rate)
    stats_out["archive_start_total"] = float(phase_archive_start)
    stats_out["archive_end_total"] = float(phase_archive_end_total)
    stats_out["worse_delta_mean"] = float(worse_delta_mean)
    return F_d, F_r, synergy, stats_out


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_complete_assignment(assignment: object, inst: dict) -> Optional[str]:
    """Validate that assignment is a valid complete assignment vector."""
    n = inst["n"]
    m = inst["m"]
    if not isinstance(assignment, (list, tuple)):
        return f"assignment must be a list (got {type(assignment).__name__})"
    if len(assignment) != n:
        return f"assignment length must be {n}, got {len(assignment)}"
    for j, mi in enumerate(assignment):
        if not isinstance(mi, int):
            return f"assignment[{j}] must be int, got {type(mi).__name__}"
        if mi < 0 or mi >= m:
            return f"assignment[{j}] = {mi} out of range [0, {m})"
    return None


def _validate_assignment_from_entry(entry: ArchiveEntry, inst: dict) -> Optional[str]:
    """Check that an archive entry has valid sequences for this instance."""
    m = inst["m"]
    n = inst["n"]
    if not entry.sequences or len(entry.sequences) != m:
        return "sequence length mismatch"
    seen = set()
    for seq in entry.sequences:
        for j in seq:
            if j < 0 or j >= n or j in seen:
                return f"invalid/duplicate job {j}"
            seen.add(j)
    if len(seen) != n:
        return f"missing jobs: {n - len(seen)}"
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _roulette(probs: List[float], rng: random.Random) -> int:
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return i
    return len(probs) - 1


def _record_sigma(
    reward: float,
    d_id: str,
    r_id: str,
    F_d: Dict[str, float],
    F_r: Dict[str, float],
    synergy: Dict[Tuple[str, str], float],
    W_d: Dict[str, float],
    W_r: Dict[str, float],
    cfg: EvalConfig,
) -> None:
    lam = cfg.lambda_smooth
    W_d[d_id] = lam * W_d[d_id] + (1 - lam) * reward
    W_r[r_id] = lam * W_r[r_id] + (1 - lam) * reward
    F_d[d_id] = F_d.get(d_id, 0.0) + reward
    F_r[r_id] = F_r.get(r_id, 0.0) + reward
    key = (d_id, r_id)
    synergy[key] = synergy.get(key, 0.0) + reward

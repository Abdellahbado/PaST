"""Multi-episode LNS evaluation engine.

Scores operator populations by running adaptive LNS on benchmark instances,
updating global fitness F, synergy dict S, and the Pareto archive.

Key adaptation from the G-LNS paper:
- Bi-objective: SA acceptance uses a *random scalarisation* per episode.
- Hypervolume is NEVER computed in the inner loop — only for generation-level
  logging (handled by the runner).
"""

from __future__ import annotations

import copy
import logging
import math
import random
from typing import Dict, List, Optional, Tuple

from glns.config import EvalConfig, SandboxConfig
from glns.pareto import ArchiveEntry, ParetoArchive
from glns.sandbox import run_operator_sandboxed
from glns.schemas import OperatorRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers: convert between internal and operator-facing representations
# ---------------------------------------------------------------------------


def raw_instance_to_dict(raw) -> dict:
    """Convert a RawInstance (or any object with the right attrs) to a plain dict
    that LLM-generated operators can consume.
    """
    # Support both attribute access (RawInstance) and dict (already dict).
    if isinstance(raw, dict):
        return raw
    return {
        "m": int(raw.m),
        "n": int(raw.n),
        "T": int(raw.T_max) if hasattr(raw, "T_max") else int(raw.T),
        "p": list(raw.p),
        "e": list(raw.e),
        "ct": list(raw.ct),
    }


def load_instances_from_json(path: str) -> List[dict]:
    """Load the legacy 90-instance benchmark and convert to operator-facing dicts."""
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    instances = []
    for item in data:
        inst = {
            "m": item["m"],
            "n": item["n"],
            "T": item["T"],
            "p": item["p"],
            "e": item["e"],
            "ct": item["ct"],
            "instance_id": item.get("instance_id", 0),
            "scale": item.get("scale", "unknown"),
        }
        instances.append(inst)
    return instances


# ---------------------------------------------------------------------------
# DP evaluation (lightweight, in-process — used for scoring)
# ---------------------------------------------------------------------------


def dp_schedule_fixed_order(
    processing_times: List[int],
    ct: List[int],
    e_single: int,
    T_limit: int,
) -> Tuple[float, List[int]]:
    """Pure-Python DP for optimal start times given a fixed job order.

    This is a self-contained version of the existing
    ``solvers.baselines_sequence_dp._dp_schedule_fixed_order`` so that
    the G-LNS module can run stand-alone (no numpy dependency required,
    but we use it when available for speed).
    """
    J = len(processing_times)
    if J == 0:
        return 0.0, []
    T = T_limit
    if T <= 0:
        return float("inf"), [0] * J

    if len(ct) < T:
        return float("inf"), [0] * J

    # Prefix sums of ct.
    prefix = [0.0] * (T + 1)
    for t in range(T):
        prefix[t + 1] = prefix[t] + ct[t]

    PT = processing_times

    # Earliest / latest feasible starts.
    ES = [0] * J
    LS = [0] * J
    cum = 0
    for i in range(J):
        ES[i] = cum
        cum += PT[i]
    cum = 0
    for i in range(J - 1, -1, -1):
        cum += PT[i]
        LS[i] = T - cum

    INF = float("inf")

    # TEC[t] for current row; we use rolling arrays.
    prev_row = [INF] * T
    parent = [[-1] * T for _ in range(J + 1)]

    # Row 0 is virtual: TEC[0][0] = 0.
    # Row 1 = first job.
    p0 = PT[0]
    es0, ls0 = ES[0], LS[0]
    cur_row = [INF] * T
    for t in range(es0, min(ls0, T - p0) + 1):
        cost = float(e_single) * (prefix[t + p0] - prefix[t])
        cur_row[t] = cost
        parent[1][t] = 0  # dummy predecessor

    for i in range(2, J + 1):
        job_idx = i - 1
        p_cur = PT[job_idx]
        es_cur, ls_cur = ES[job_idx], LS[job_idx]

        prev_idx = job_idx - 1
        p_prev = PT[prev_idx]
        es_prev, ls_prev = ES[prev_idx], LS[prev_idx]

        prev_row = cur_row
        cur_row = [INF] * T

        # Build prefix-min over prev_row[es_prev..ls_prev].
        pmin_val = [INF] * T
        pmin_arg = [-1] * T
        best_v = INF
        best_a = -1
        for s in range(es_prev, ls_prev + 1):
            v = prev_row[s]
            if v < best_v:
                best_v = v
                best_a = s
            pmin_val[s] = best_v
            pmin_arg[s] = best_a

        latest = min(ls_cur, T - p_cur)
        for t in range(es_cur, latest + 1):
            r = min(ls_prev, t - p_prev)
            if r < es_prev:
                continue
            pb = pmin_val[r]
            if pb == INF:
                continue
            cost = float(e_single) * (prefix[t + p_cur] - prefix[t])
            cur_row[t] = pb + cost
            parent[i][t] = pmin_arg[r]

    # Find best final time.
    best_t = -1
    best_cost = INF
    for t in range(T):
        if cur_row[t] < best_cost:
            best_cost = cur_row[t]
            best_t = t

    if best_cost == INF:
        return INF, [0] * J

    # Backtrack.
    starts = [0] * J
    ct_t = best_t
    for i in range(J, 0, -1):
        starts[i - 1] = ct_t
        ct_t = parent[i][ct_t]

    return best_cost, starts


def evaluate_sequences(
    sequences: List[List[int]],
    inst: dict,
    T_limit: int,
) -> Tuple[float, int, List[List[int]]]:
    """Evaluate a full solution (all machines).

    Returns (total_energy, makespan, per_machine_start_times).
    """
    ct = inst["ct"]
    e = inst["e"]
    p = inst["p"]

    total_energy = 0.0
    makespan = 0
    all_starts: List[List[int]] = []

    # Defensive: never allow a single bad job id to crash the run.
    # (This can happen if an archive entry from a different instance is used,
    # or if an operator produces out-of-range IDs.)
    p_len = len(p)

    for mi, seq in enumerate(sequences):
        if not seq:
            all_starts.append([])
            continue
        # Validate indices to avoid IndexError.
        for j in seq:
            if not isinstance(j, int) or j < 0 or j >= p_len:
                logger.debug(
                    "Invalid job id in sequence: inst_id=%s m=%d n=%d machine=%d job=%r",
                    inst.get("instance_id", "?"),
                    inst.get("m", -1),
                    inst.get("n", -1),
                    mi,
                    j,
                )
                return float("inf"), T_limit + 1, []

        procs = [p[j] for j in seq]
        energy, starts = dp_schedule_fixed_order(procs, ct, e[mi], T_limit)
        if energy == float("inf"):
            return float("inf"), T_limit + 1, []
        total_energy += energy
        ms = starts[-1] + procs[-1] if starts else 0
        makespan = max(makespan, ms)
        all_starts.append(starts)

    return total_energy, makespan, all_starts


def make_initial_solution(inst: dict, rng: random.Random) -> List[List[int]]:
    """Build a simple initial solution (LPT + round-robin assignment)."""
    n, m = inst["n"], inst["m"]
    p = inst["p"]
    jobs = sorted(range(n), key=lambda j: -p[j])
    loads = [0] * m
    seqs: List[List[int]] = [[] for _ in range(m)]
    for j in jobs:
        mi = min(range(m), key=lambda i: loads[i])
        seqs[mi].append(j)
        loads[mi] += p[j]
    return seqs


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

SELECTION_STRATEGIES = ["random", "crowded", "extreme_tec", "extreme_cmax", "stagnated"]


def run_evaluation_phase(
    destroy_pool: List[OperatorRecord],
    repair_pool: List[OperatorRecord],
    instances: List[dict],
    archive: ParetoArchive,
    eval_cfg: EvalConfig,
    sandbox_cfg: SandboxConfig,
    rng: random.Random,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[Tuple[str, str], float]]:
    """Run K multi-episode evaluations. Updates archive in-place.

    Returns:
        (F_destroy, F_repair, synergy)
        F_destroy: {op_id -> cumulative fitness}
        F_repair:  {op_id -> cumulative fitness}
        synergy:   {(destroy_id, repair_id) -> cumulative synergy score}
    """
    n_d = len(destroy_pool)
    n_r = len(repair_pool)
    if n_d == 0 or n_r == 0:
        raise ValueError("Both pools must be non-empty for evaluation.")

    sigma = eval_cfg.sigma  # (σ1, σ2, σ3, σ4)

    # Global fitness accumulators (keyed by stable operator ID).
    F_d: Dict[str, float] = {op.id: 0.0 for op in destroy_pool}
    F_r: Dict[str, float] = {op.id: 0.0 for op in repair_pool}
    synergy: Dict[Tuple[str, str], float] = {}

    # Failure counters (mark broken operators).
    fail_d: Dict[str, int] = {op.id: 0 for op in destroy_pool}
    fail_r: Dict[str, int] = {op.id: 0 for op in repair_pool}
    FAIL_THRESHOLD = 5  # After this many failures in one phase, operator gets 0 weight.

    # Phase-level statistics for logging / debugging.
    phase_archive_start = archive.size()
    stats = {
        "sigma1": 0,
        "sigma2": 0,
        "sigma3": 0,
        "sigma4": 0,
        "sa_accept": 0,
        "sa_reject": 0,
        "destroy_fail": 0,
        "repair_fail": 0,
        "invalid_candidate": 0,
        "inf_eval": 0,
        "archive_entered": 0,
        "archive_dominated_removed": 0,
    }

    for episode in range(eval_cfg.K_episodes):
        # Pick a random instance.
        inst = rng.choice(instances)
        T_limit = inst["T"]
        inst_id = int(inst.get("instance_id", 0))

        # Pick initial solution from archive or build new one.
        if archive.size(inst_id) > 0:
            strategy = rng.choice(SELECTION_STRATEGIES)
            entry = archive.select(strategy, rng, instance_id=inst_id)
            # Validate archive entry against this instance. If invalid, drop it.
            err0 = _validate_complete_solution(entry.sequences, inst)
            if err0 is not None:
                logger.warning(
                    "Dropping invalid archive entry for inst_id=%s (strategy=%s): %s",
                    inst_id,
                    strategy,
                    err0,
                )
                archive.entries = [e for e in archive.entries if e is not entry]
                strategy = "none"
                cur_seqs = make_initial_solution(inst, rng)
            else:
                cur_seqs = copy.deepcopy(entry.sequences)
        else:
            strategy = "none"
            cur_seqs = make_initial_solution(inst, rng)

        # Final guardrail: ensure the starting solution is structurally valid.
        err_init = _validate_complete_solution(cur_seqs, inst)
        if err_init is not None:
            logger.error(
                "Initial solution invalid for inst_id=%s (init_from=%s): %s",
                inst_id,
                strategy,
                err_init,
            )
            # Skip this episode rather than crashing the whole generation.
            continue

        cur_energy, cur_cmax, cur_starts = evaluate_sequences(cur_seqs, inst, T_limit)
        if cur_energy == float("inf"):
            logger.error(
                "Initial solution infeasible for inst_id=%s (init_from=%s); skipping episode",
                inst_id,
                strategy,
            )
            continue

        # Random scalarisation weight for this episode (no hypervolume!).
        w = rng.random()  # w in [0,1]; scalar = w*norm_cmax + (1-w)*norm_tec

        # Normalisation references (from current archive or fallbacks).
        if archive.size(inst_id) >= 2:
            pts = [(e.makespan, e.energy) for e in archive.entries_for(inst_id)]
            cmax_lo = min(c for c, _ in pts)
            cmax_hi = max(c for c, _ in pts)
            tec_lo = min(t for _, t in pts)
            tec_hi = max(t for _, t in pts)
        else:
            cmax_lo, cmax_hi = cur_cmax, cur_cmax + 1
            tec_lo, tec_hi = cur_energy, cur_energy + 1.0

        def scalarise(cmax: int, tec: float) -> float:
            c_range = max(float(cmax_hi - cmax_lo), 1.0)
            t_range = max(tec_hi - tec_lo, 1.0)
            return w * (cmax - cmax_lo) / c_range + (1 - w) * (tec - tec_lo) / t_range

        cur_scalar = scalarise(cur_cmax, cur_energy)
        best_scalar = cur_scalar  # episode-best scalar objective (paper's x*)

        # SA temperature.
        tau = eval_cfg.sa_T0

        # Adaptive weights (reset per episode).
        W_d = {op.id: 1.0 for op in destroy_pool}
        W_r = {op.id: 1.0 for op in repair_pool}

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Episode %d/%d | inst=%s scale=%s | init_from=%s | scalar_w=%.3f | T=%d",
                episode + 1,
                eval_cfg.K_episodes,
                inst.get("instance_id", "?"),
                inst.get("scale", "?"),
                strategy,
                w,
                eval_cfg.T_iters,
            )

        for it in range(eval_cfg.T_iters):
            # ---------- Roulette selection ----------
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

            # ---------- Destroy ----------
            destroy_cnt = max(1, int(eval_cfg.destroy_ratio * inst["n"]))
            ok_d, d_result = run_operator_sandboxed(
                fn_code=d_op.code,
                fn_name="destroy",
                args=(
                    copy.deepcopy(cur_seqs),
                    destroy_cnt,
                    inst,
                    random.Random(rng.randint(0, 2**31)),
                ),
                cfg=sandbox_cfg,
            )
            if not ok_d:
                fail_d[d_op.id] += 1
                stats["destroy_fail"] += 1
                logger.debug("Destroy %s failed: %s", d_op.id, d_result)
                _record_sigma(
                    sigma[3], d_op.id, r_op.id, F_d, F_r, synergy, W_d, W_r, eval_cfg
                )
                stats["sigma4"] += 1
                tau *= eval_cfg.sa_alpha
                continue
            removed_jobs, partial_seqs = d_result

            # ---------- Repair ----------
            ok_r, r_result = run_operator_sandboxed(
                fn_code=r_op.code,
                fn_name="repair",
                args=(
                    copy.deepcopy(partial_seqs),
                    list(removed_jobs),
                    inst,
                    random.Random(rng.randint(0, 2**31)),
                ),
                cfg=sandbox_cfg,
            )
            if not ok_r:
                fail_r[r_op.id] += 1
                stats["repair_fail"] += 1
                logger.debug("Repair %s failed: %s", r_op.id, r_result)
                _record_sigma(
                    sigma[3], d_op.id, r_op.id, F_d, F_r, synergy, W_d, W_r, eval_cfg
                )
                stats["sigma4"] += 1
                tau *= eval_cfg.sa_alpha
                continue

            cand_seqs = r_result

            # ---------- Validate candidate structure (defensive) ----------
            err = _validate_complete_solution(cand_seqs, inst)
            if err is not None:
                fail_r[r_op.id] += 1
                stats["invalid_candidate"] += 1
                logger.debug(
                    "Invalid candidate from (%s,%s): %s", d_op.id, r_op.id, err
                )
                _record_sigma(
                    sigma[3], d_op.id, r_op.id, F_d, F_r, synergy, W_d, W_r, eval_cfg
                )
                stats["sigma4"] += 1
                tau *= eval_cfg.sa_alpha
                continue

            # ---------- Validate candidate ----------
            cand_energy, cand_cmax, cand_starts = evaluate_sequences(
                cand_seqs, inst, T_limit
            )
            if cand_energy == float("inf"):
                _record_sigma(
                    sigma[3], d_op.id, r_op.id, F_d, F_r, synergy, W_d, W_r, eval_cfg
                )
                stats["inf_eval"] += 1
                stats["sigma4"] += 1
                tau *= eval_cfg.sa_alpha
                continue

            # ---------- Scoring ----------
            cand_entry = ArchiveEntry(
                instance_id=inst_id,
                makespan=cand_cmax,
                energy=cand_energy,
                sequences=cand_seqs,
                start_times=cand_starts,
            )
            cand_scalar = scalarise(cand_cmax, cand_energy)

            # Maintain Pareto archive of discovered non-dominated solutions (bi-objective adaptation).
            # This is decoupled from the ALNS scoring tiers, which follow the paper's hierarchy.
            entered, dominated_removed = archive.add(copy.deepcopy(cand_entry))
            if entered:
                stats["archive_entered"] += 1
            stats["archive_dominated_removed"] += int(dominated_removed)

            # Paper hierarchy (adapted): use the episode's scalarised objective as f(·).
            # σ1: improves episode-best (x*)
            # σ2: improves current (xcurr)
            # σ3: accepted by SA despite being worse
            # σ4: rejected
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
                else:
                    stats["sigma4"] += 1
                    stats["sa_reject"] += 1

            if accept_current:
                cur_seqs = cand_seqs
                cur_energy = cand_energy
                cur_cmax = cand_cmax
                cur_scalar = cand_scalar

            _record_sigma(
                reward, d_op.id, r_op.id, F_d, F_r, synergy, W_d, W_r, eval_cfg
            )
            tau *= eval_cfg.sa_alpha

    # Increment stagnation per-instance for all entries that survived this phase.
    # (We touched multiple instances across episodes.)
    for e in archive.entries:
        e.stagnation += 1

    # Phase summary (one line at INFO; deeper detail available via DEBUG logs).
    phase_archive_end = archive.size()
    total_sa = stats["sa_accept"] + stats["sa_reject"]
    sa_rate = (stats["sa_accept"] / total_sa) if total_sa else 0.0
    logger.info(
        "Eval done | episodes=%d iters=%d | σ1=%d σ2=%d σ3=%d σ4=%d | SA accept=%.1f%% | archive %d→%d (entered=%d dom_removed=%d) | fails: d=%d r=%d invalid=%d",
        eval_cfg.K_episodes,
        eval_cfg.T_iters,
        stats["sigma1"],
        stats["sigma2"],
        stats["sigma3"],
        stats["sigma4"],
        100.0 * sa_rate,
        phase_archive_start,
        phase_archive_end,
        stats["archive_entered"],
        stats["archive_dominated_removed"],
        stats["destroy_fail"],
        stats["repair_fail"],
        stats["invalid_candidate"],
    )

    return F_d, F_r, synergy


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _roulette(probs: List[float], rng: random.Random) -> int:
    """Roulette-wheel selection given a probability vector."""
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return i
    return len(probs) - 1


def _validate_complete_solution(seqs: object, inst: dict) -> Optional[str]:
    """Return None if *seqs* is a valid complete solution, else an error string.

    We enforce:
    - seqs is list of length m
    - each element is a list of ints
    - every job in {0..n-1} appears exactly once across all machines
    """
    m = inst["m"]
    n = inst["n"]

    if not isinstance(seqs, list):
        return f"solution must be a list (got {type(seqs).__name__})"
    if len(seqs) != m:
        return f"solution must have length m={m} (got {len(seqs)})"

    seen = [0] * n
    total = 0
    for mi, machine_seq in enumerate(seqs):
        if not isinstance(machine_seq, list):
            return f"machine {mi} sequence must be a list (got {type(machine_seq).__name__})"
        for job in machine_seq:
            if not isinstance(job, int):
                return f"job IDs must be ints (got {type(job).__name__})"
            if job < 0 or job >= n:
                return f"job index out of range: {job} (expected 0..{n-1})"
            seen[job] += 1
            if seen[job] > 1:
                return f"duplicate job detected: {job}"
            total += 1

    if total != n:
        missing = [j for j, c in enumerate(seen) if c == 0]
        return f"solution has {total} jobs, expected {n}; missing={missing[:10]}"

    return None


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
    """Apply Eqs. 6-8 from the G-LNS paper."""
    lam = cfg.lambda_smooth
    # Eq. 6: adaptive weight update.
    W_d[d_id] = lam * W_d[d_id] + (1 - lam) * reward
    W_r[r_id] = lam * W_r[r_id] + (1 - lam) * reward
    # Eq. 7: global fitness accumulation.
    F_d[d_id] = F_d.get(d_id, 0.0) + reward
    F_r[r_id] = F_r.get(r_id, 0.0) + reward
    # Eq. 8: synergy recording (dict keyed by stable IDs).
    key = (d_id, r_id)
    synergy[key] = synergy.get(key, 0.0) + reward

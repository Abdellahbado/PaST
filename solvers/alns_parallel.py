"""ALNS/LNS baseline for the full parallel-machine TOU scheduling problem.

Goal: provide a *strong, careful* non-learning search baseline to decide whether
PPO-style operator selection is worth training.

Problem: Pm,TOU || Cmax, TEC with epsilon-constraint (hard Cmax <= epsilon).

We use:
- CWP (Cheapest-Window Placement) as an initial assignment heuristic.
- ALNS to improve *assignment + sequencing*.
- DP timing layer (_dp_schedule_fixed_order) to evaluate TEC for each machine
  given a fixed sequence.

This module is intentionally dependency-light (numpy + stdlib) so it can run in
HPC settings without extra packages.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from PaST.data.sm_benchmark_data import RawInstance
from PaST.solvers.baselines_sequence_dp import _dp_schedule_fixed_order
from PaST.solvers.cwp_solver import construct_schedule_CWP, get_machine_job_sets

try:
    import torch

    from PaST.solvers.batch_dp_solver import BatchSequenceDPSolver

    _TORCH_DP_OK = True
except Exception:
    torch = None  # type: ignore
    BatchSequenceDPSolver = None  # type: ignore
    _TORCH_DP_OK = False


@dataclass(frozen=True)
class MachineDP:
    energy: float
    makespan: int
    start_times: List[int]


@dataclass(frozen=True)
class FullEval:
    total_energy: float
    makespan: int
    feasible: bool
    per_machine: List[MachineDP]


@dataclass
class Solution:
    """Solution representation: per-machine job sequences (global job ids)."""

    sequences: List[List[int]]  # length m


@dataclass(frozen=True)
class ALNSConfig:
    max_iters: int = 200
    no_improve_limit: int = 80

    # Destruction size control
    destroy_frac: float = 0.15
    destroy_min: int = 2
    destroy_max: int = 40

    # Repair evaluation caps (keep search usable for larger instances)
    max_positions_fullscan: int = 30
    max_positions_sample: int = 24

    # Local search polish (light VND-ish)
    local_search: bool = True
    local_max_pair_trials: int = 64

    # SA acceptance
    sa_tau0: float = 1.0
    sa_decay: float = 0.995

    # Classic ALNS operator adaptation (roulette weights)
    segment_len: int = 25
    reaction_factor: float = 0.2
    sigma_best: float = 5.0
    sigma_improve: float = 2.0
    sigma_accept_worse: float = 0.5

    # Logging / dataset
    keep_iter_log: bool = True
    log_per_machine_every: int = (
        0  # 0 disables; else include per-machine stats every N iters
    )
    verify_cost: bool = False
    oracle_dataset: bool = False
    oracle_max_states: int = 256

    # Optional acceleration: batch DP evaluations in greedy repair.
    use_torch_batch_dp_in_repair: bool = True
    # Use torch device for batched DP: auto|cpu|cuda
    torch_dp_device: str = "auto"
    # Minimum total candidates to batch (across all machines) before using torch DP.
    # Keep this reasonably high because CPU->GPU transfers can dominate when batches are tiny.
    torch_batch_min_total_candidates: int = 64

    # Feasibility helpers for RL training:
    # When an action yields an infeasible candidate, retry with smaller destruction and/or greedy repair.
    feasibility_retries: int = 2
    feasibility_retry_shrink: float = 0.5
    feasibility_force_greedy_on_retry: bool = True
    # Avoid destroy_all when we're close to the deadline (high risk of infeasibility).
    avoid_destroy_all_when_tight: bool = True
    avoid_destroy_all_slack_threshold: float = 0.05


def _torch_choose_device(cfg: ALNSConfig) -> str:
    dev = (cfg.torch_dp_device or "auto").strip().lower()
    if dev not in ("auto", "cpu", "cuda"):
        dev = "auto"
    if dev == "cpu":
        return "cpu"
    if dev == "cuda":
        return "cuda"
    # auto
    if _TORCH_DP_OK and torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _torch_get_ct_tensor(raw: RawInstance, epsilon: int, device: str):
    """Cache `ct[:epsilon]` tensor on the requested device for this instance."""
    if not _TORCH_DP_OK:
        raise RuntimeError("Torch DP not available")

    key = (int(epsilon), str(device))
    cache = getattr(raw, "_torch_ct_cache", None)
    if cache is None:
        cache = {}
        setattr(raw, "_torch_ct_cache", cache)

    v = cache.get(key)
    if v is not None:
        return v

    ct_np = np.asarray(raw.ct[: int(epsilon)], dtype=np.int32)
    ct_t = torch.tensor(
        ct_np, dtype=torch.int32, device=torch.device(device)
    ).unsqueeze(0)
    cache[key] = ct_t
    return ct_t


def _batch_dp_eval_candidates(
    raw: RawInstance,
    epsilon: int,
    candidates: List[Tuple[int, int, List[int]]],
    cfg: ALNSConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batch-evaluate many candidate *job sequences* via Torch DP.

    Args:
        candidates: list of (machine_idx, pos, job_sequence) where job_sequence is
            the full per-machine job order after insertion.
    Returns:
        (energies, makespans) arrays aligned with `candidates`.
    """
    if not _TORCH_DP_OK:
        raise RuntimeError("Torch DP not available")

    B = int(len(candidates))
    if B == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.int64)

    device = _torch_choose_device(cfg)
    dev = torch.device(device)

    lengths = [len(seq) for (_mi, _pos, seq) in candidates]
    maxN = int(max(lengths))

    # Build padded processing time matrix. Padding uses 0-duration jobs,
    # and we pass `sequence_lengths` to ensure only the real prefix counts.
    proc_batch = np.zeros((B, maxN), dtype=np.int64)
    seq_lens = np.asarray(lengths, dtype=np.int64)
    e_single = np.zeros((B,), dtype=np.int64)

    for i, (mi, _pos, seq) in enumerate(candidates):
        for j, job_id in enumerate(seq):
            proc_batch[i, j] = int(raw.p[int(job_id)])
        e_single[i] = int(raw.e[int(mi)])

    # Identity job order [0..maxN-1] for all rows; durations already in sequence order.
    job_sequences = (
        torch.arange(maxN, dtype=torch.long, device=dev).unsqueeze(0).repeat(B, 1)
    )
    processing_times = torch.tensor(proc_batch, dtype=torch.long, device=dev)
    e_t = torch.tensor(e_single, dtype=torch.long, device=dev)
    T_t = torch.full((B,), int(epsilon), dtype=torch.long, device=dev)
    seq_len_t = torch.tensor(seq_lens, dtype=torch.long, device=dev)

    # Reuse cached ct tensor on device.
    ct_t = _torch_get_ct_tensor(raw, epsilon, device).repeat(B, 1)

    costs_t, end_t = BatchSequenceDPSolver.solve_with_end_time(
        job_sequences=job_sequences,
        processing_times=processing_times,
        ct=ct_t,
        e_single=e_t,
        T_limit=T_t,
        sequence_lengths=seq_len_t,
    )

    energies = costs_t.detach().to("cpu").numpy().astype(np.float64)
    makespans = end_t.detach().to("cpu").numpy().astype(np.int64)
    return energies, makespans


def _batch_dp_eval_insertions_for_machine(
    raw: RawInstance,
    epsilon: int,
    machine_idx: int,
    base_seq: Sequence[int],
    job: int,
    positions: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Batch-evaluate candidate insertions on one machine.

    Returns:
        (energies, makespans) arrays of shape (len(positions),)

    Notes:
        This evaluates *processing-time sequences* using `BatchSequenceDPSolver`.
        It does not reconstruct per-job start times; we only need energy and
        completion time (makespan) for ALNS decisions.
    """
    if not _TORCH_DP_OK:
        raise RuntimeError("Torch DP not available")

    base_seq_list = list(int(x) for x in base_seq)
    positions_list = list(int(p) for p in positions)
    B = int(len(positions_list))
    N = int(len(base_seq_list) + 1)
    if B <= 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.int64)

    proc_batch: List[List[int]] = []
    for pos in positions_list:
        seq = base_seq_list.copy()
        seq.insert(int(pos), int(job))
        proc_batch.append([int(raw.p[j]) for j in seq])

    # Identity job order [0..N-1]; durations already in sequence order per row.
    job_sequences = torch.arange(N, dtype=torch.long).unsqueeze(0).repeat(B, 1)
    processing_times = torch.tensor(proc_batch, dtype=torch.long)

    ct_np = np.asarray(raw.ct[: int(epsilon)], dtype=np.int32)
    ct_t = torch.tensor(ct_np, dtype=torch.int32).unsqueeze(0).repeat(B, 1)
    e_t = torch.full((B,), int(raw.e[int(machine_idx)]), dtype=torch.long)
    T_t = torch.full((B,), int(epsilon), dtype=torch.long)

    costs_t, end_t = BatchSequenceDPSolver.solve_with_end_time(
        job_sequences=job_sequences,
        processing_times=processing_times,
        ct=ct_t,
        e_single=e_t,
        T_limit=T_t,
    )

    energies = costs_t.detach().cpu().numpy().astype(np.float64)
    makespans = end_t.detach().cpu().numpy().astype(np.int64)
    return energies, makespans


ALNSAction = Tuple[str, float, str]  # (destroy_name, k_mult, repair_name)


def _assert_solution_valid(sol: Solution, n_jobs: int, m: int) -> None:
    if len(sol.sequences) != int(m):
        raise ValueError(f"Solution has {len(sol.sequences)} machines, expected {m}")
    flat: List[int] = []
    for seq in sol.sequences:
        flat.extend(int(x) for x in seq)
    if len(flat) != int(n_jobs):
        raise ValueError(f"Solution covers {len(flat)} jobs, expected {n_jobs}")
    if len(set(flat)) != len(flat):
        raise ValueError("Solution has duplicate jobs")
    if set(flat) != set(range(int(n_jobs))):
        missing = sorted(set(range(int(n_jobs))) - set(flat))
        extra = sorted(set(flat) - set(range(int(n_jobs))))
        raise ValueError(
            f"Solution job set mismatch; missing={missing[:10]} extra={extra[:10]}"
        )


def _fill_missing_jobs_greedily(
    raw: RawInstance, job_sets: List[List[int]]
) -> List[List[int]]:
    """Ensure every job 0..n-1 is assigned to exactly one machine.

    CWP can return -1 assignments if epsilon is too tight; we must not silently
    drop jobs (would corrupt costs). This function adds missing jobs to the
    currently lightest-load machine by processing time.
    """
    m = int(raw.m)
    n = int(raw.n)
    seen = set(j for js in job_sets for j in js)
    missing = [j for j in range(n) if j not in seen]
    if not missing:
        return job_sets

    loads = [sum(int(raw.p[j]) for j in js) for js in job_sets]
    for j in missing:
        mi = int(min(range(m), key=lambda i: loads[i]))
        job_sets[mi].append(int(j))
        loads[mi] += int(raw.p[j])
    return job_sets


def _compute_epsilon(
    raw: RawInstance,
    slack_ratio: float,
) -> int:
    """Compute a single epsilon deadline for the full instance.

    We use a simple lower bound:
      T_min = max(max(p), ceil(sum(p)/m))
    and interpolate toward T_max.
    """

    p = np.asarray(raw.p, dtype=np.int64)
    if raw.m <= 0:
        return int(raw.T_max)
    t_min = int(
        max(int(p.max(initial=0)), int(math.ceil(float(p.sum()) / float(raw.m))))
    )
    t_max = int(raw.T_max)
    slack_ratio = float(max(0.0, min(1.0, slack_ratio)))
    eps = int(round(t_min + slack_ratio * (t_max - t_min)))
    return int(max(t_min, min(t_max, eps)))


def _eval_machine_sequence(
    raw: RawInstance,
    machine_idx: int,
    job_seq: Sequence[int],
    epsilon: int,
    verify_cost: bool = False,
) -> MachineDP:
    proc = [int(raw.p[j]) for j in job_seq]
    energy, start_times = _dp_schedule_fixed_order(
        processing_times=proc,
        ct=np.asarray(raw.ct, dtype=np.int32),
        e_single=int(raw.e[machine_idx]),
        T_limit=int(epsilon),
        dp_time_penalty=0.0,
    )

    if not np.isfinite(energy):
        return MachineDP(
            energy=float("inf"), makespan=int(epsilon) + 1, start_times=[0] * len(proc)
        )

    if len(proc) == 0:
        return MachineDP(energy=float(energy), makespan=0, start_times=[])

    ms = int(start_times[-1]) + int(proc[-1])
    if verify_cost:
        ct = np.asarray(raw.ct, dtype=np.int64)
        e_single = int(raw.e[machine_idx])
        chk = 0
        for st, p in zip(start_times, proc):
            st_i = int(st)
            p_i = int(p)
            if st_i < 0 or st_i + p_i > len(ct):
                raise ValueError("DP produced out-of-bounds start time")
            chk += int(ct[st_i : st_i + p_i].sum())
        chk_energy = float(e_single) * float(chk)
        if not np.isfinite(chk_energy) or abs(float(chk_energy) - float(energy)) > 1e-6:
            raise ValueError(f"Energy mismatch: DP={energy} recomputed={chk_energy}")
    return MachineDP(
        energy=float(energy), makespan=ms, start_times=[int(t) for t in start_times]
    )


def evaluate_solution(raw: RawInstance, sol: Solution, epsilon: int) -> FullEval:
    per_machine: List[MachineDP] = []
    total = 0.0
    makespan = 0
    feasible = True

    for mi, seq in enumerate(sol.sequences):
        mdp = _eval_machine_sequence(raw, mi, seq, epsilon)
        per_machine.append(mdp)
        if not np.isfinite(mdp.energy):
            feasible = False
        else:
            total += float(mdp.energy)
        makespan = max(makespan, int(mdp.makespan))

    return FullEval(
        total_energy=float(total) if feasible else float("inf"),
        makespan=int(makespan),
        feasible=bool(feasible and makespan <= int(epsilon)),
        per_machine=per_machine,
    )


def _update_eval_for_machines(
    raw: RawInstance,
    epsilon: int,
    sol: Solution,
    prev: FullEval,
    machine_indices: Sequence[int],
) -> FullEval:
    """Incrementally update a FullEval by recomputing DP on selected machines."""
    if not machine_indices:
        return prev

    per_machine = list(prev.per_machine)
    for mi in machine_indices:
        per_machine[int(mi)] = _eval_machine_sequence(
            raw, int(mi), sol.sequences[int(mi)], epsilon
        )

    feasible = True
    total = 0.0
    makespan = 0
    for md in per_machine:
        if not np.isfinite(md.energy):
            feasible = False
        else:
            total += float(md.energy)
        makespan = max(makespan, int(md.makespan))

    return FullEval(
        total_energy=float(total) if feasible else float("inf"),
        makespan=int(makespan),
        feasible=bool(feasible and makespan <= int(epsilon)),
        per_machine=per_machine,
    )


def _randomize_sequences(
    job_sets: List[List[int]], rng: random.Random
) -> List[List[int]]:
    seqs: List[List[int]] = []
    for js in job_sets:
        js2 = list(js)
        rng.shuffle(js2)
        seqs.append(js2)
    return seqs


# -------------------------
# Destroy operators
# -------------------------


def _destroy_random(
    sol: Solution, k: int, rng: random.Random
) -> Tuple[Solution, List[int], List[int]]:
    """Remove k random jobs across all machines.

    Returns: (partial_solution, removed_jobs, touched_machine_indices)
    """
    all_jobs: List[Tuple[int, int]] = []  # (machine_idx, pos)
    for mi, seq in enumerate(sol.sequences):
        for pos in range(len(seq)):
            all_jobs.append((mi, pos))
    if not all_jobs:
        return copy.deepcopy(sol), [], []

    k = int(min(k, len(all_jobs)))
    picked = rng.sample(all_jobs, k=k)
    picked.sort(reverse=True)  # remove from back to keep indices valid

    new_sol = Solution(sequences=[list(s) for s in sol.sequences])
    removed: List[int] = []
    touched: set[int] = set()
    for mi, pos in picked:
        touched.add(mi)
        removed.append(new_sol.sequences[mi].pop(pos))

    return new_sol, removed, sorted(touched)


def _destroy_longest_jobs(
    raw: RawInstance, sol: Solution, k: int, rng: random.Random
) -> Tuple[Solution, List[int], List[int]]:
    jobs: List[int] = [j for seq in sol.sequences for j in seq]
    if not jobs:
        return copy.deepcopy(sol), [], []

    # sort by duration desc, tiebreak random
    rng.shuffle(jobs)
    jobs.sort(key=lambda j: int(raw.p[j]), reverse=True)
    picked = set(jobs[: int(min(k, len(jobs)))])

    new_sol = Solution(sequences=[])
    removed: List[int] = []
    touched: List[int] = []
    for mi, seq in enumerate(sol.sequences):
        kept = []
        rm = []
        for j in seq:
            if j in picked:
                rm.append(j)
            else:
                kept.append(j)
        new_sol.sequences.append(kept)
        if rm:
            removed.extend(rm)
            touched.append(mi)

    return new_sol, removed, touched


def _destroy_worst_machine(
    sol: Solution,
    eval_: FullEval,
    k: int,
    rng: random.Random,
) -> Tuple[Solution, List[int], List[int]]:
    energies = [m.energy for m in eval_.per_machine]
    if not energies:
        return copy.deepcopy(sol), [], []

    # pick the machine with max finite energy; if all inf, fallback random
    worst = None
    worst_val = -1.0
    for i, e in enumerate(energies):
        if np.isfinite(e) and float(e) > worst_val:
            worst_val = float(e)
            worst = i
    if worst is None:
        worst = rng.randrange(len(energies))

    seq = list(sol.sequences[worst])
    if not seq:
        return copy.deepcopy(sol), [], []

    k = int(min(k, len(seq)))
    idxs = list(range(len(seq)))
    picked_pos = rng.sample(idxs, k=k)
    picked_pos.sort(reverse=True)

    new_sol = Solution(sequences=[list(s) for s in sol.sequences])
    removed: List[int] = []
    for pos in picked_pos:
        removed.append(new_sol.sequences[worst].pop(pos))

    return new_sol, removed, [worst]


def _destroy_all(sol: Solution) -> Tuple[Solution, List[int], List[int]]:
    """Remove *all* jobs.

    This operator makes the move set ergodic together with a non-greedy repair:
    in principle, any feasible assignment+sequencing can be reconstructed.
    """
    removed: List[int] = [j for seq in sol.sequences for j in seq]
    m = len(sol.sequences)
    return Solution(sequences=[[] for _ in range(m)]), removed, list(range(m))


# -------------------------
# Repair (greedy insertion)
# -------------------------


def _candidate_positions(n: int, cfg: ALNSConfig, rng: random.Random) -> List[int]:
    """Candidate insertion positions in [0..n]."""
    if n <= cfg.max_positions_fullscan:
        return list(range(n + 1))

    # Always include ends + a few random positions.
    positions = {0, n}
    for _ in range(cfg.max_positions_sample):
        positions.add(rng.randrange(n + 1))
    return sorted(positions)


def _best_insertion_for_job(
    raw: RawInstance,
    epsilon: int,
    sol: Solution,
    eval_: FullEval,
    job: int,
    cfg: ALNSConfig,
    rng: random.Random,
) -> Tuple[Optional[Solution], Optional[FullEval], Optional[int]]:
    """Try inserting `job` in the best (machine, position) by exact DP eval.

    Returns (new_solution, new_eval, touched_machine) or (None,None,None) if infeasible.
    """

    best_sol = None
    best_eval = None
    best_mi = None
    best_total = float("inf")

    # Optional: batch all candidates across all machines into one DP call.
    # This makes GPU usage worthwhile (bigger batch, fewer kernel launches).
    use_batch_all = bool(cfg.use_torch_batch_dp_in_repair) and _TORCH_DP_OK
    if use_batch_all:
        # Build candidate list once.
        candidates: List[Tuple[int, int, List[int]]] = []
        for mi in range(int(raw.m)):
            base_seq = sol.sequences[mi]
            positions = _candidate_positions(len(base_seq), cfg, rng)
            for pos in positions:
                seq = list(int(x) for x in base_seq)
                seq.insert(int(pos), int(job))
                candidates.append((int(mi), int(pos), seq))

        if len(candidates) >= int(cfg.torch_batch_min_total_candidates):
            energies, makespans = _batch_dp_eval_candidates(
                raw=raw,
                epsilon=epsilon,
                candidates=candidates,
                cfg=cfg,
            )

            # Precompute other-machines info for delta totals.
            eval_total_finite = np.isfinite(float(eval_.total_energy))
            per_machine = list(eval_.per_machine)
            other_makespan_by_m: List[int] = []
            for mi in range(int(raw.m)):
                ms_other = 0
                feasible_other = True
                for k, md in enumerate(per_machine):
                    if k == mi:
                        continue
                    if not np.isfinite(float(md.energy)):
                        feasible_other = False
                        break
                    ms_other = max(ms_other, int(md.makespan))
                other_makespan_by_m.append(
                    int(ms_other) if feasible_other else int(epsilon) + 1
                )

            for (mi, pos, seq), new_energy, new_ms in zip(
                candidates, energies, makespans
            ):
                if not np.isfinite(float(new_energy)):
                    continue

                makespan = max(int(other_makespan_by_m[int(mi)]), int(new_ms))
                if makespan > int(epsilon):
                    continue

                old_mdp = per_machine[int(mi)]
                if eval_total_finite and np.isfinite(float(old_mdp.energy)):
                    total = (
                        float(eval_.total_energy)
                        - float(old_mdp.energy)
                        + float(new_energy)
                    )
                    feasible = True
                else:
                    feasible = True
                    total = 0.0
                    for k, md in enumerate(per_machine):
                        if k == int(mi):
                            continue
                        if not np.isfinite(float(md.energy)):
                            feasible = False
                            break
                        total += float(md.energy)
                    if feasible:
                        total += float(new_energy)

                if (not feasible) or (not np.isfinite(float(total))):
                    continue

                if float(total) < best_total:
                    trial_sol = Solution(sequences=[list(s) for s in sol.sequences])
                    trial_sol.sequences[int(mi)].insert(int(pos), int(job))
                    per_machine2 = list(per_machine)
                    per_machine2[int(mi)] = MachineDP(
                        energy=float(new_energy),
                        makespan=int(new_ms),
                        start_times=[],
                    )
                    best_total = float(total)
                    best_sol = trial_sol
                    best_eval = FullEval(
                        total_energy=float(total),
                        makespan=int(makespan),
                        feasible=True,
                        per_machine=per_machine2,
                    )
                    best_mi = int(mi)

            return best_sol, best_eval, best_mi

    # Fallback: try all machines; insertion position candidates per machine.
    for mi in range(raw.m):
        base_seq = sol.sequences[mi]
        positions = _candidate_positions(len(base_seq), cfg, rng)

        for pos in positions:
            trial_sol = Solution(sequences=[list(s) for s in sol.sequences])
            trial_sol.sequences[mi].insert(pos, int(job))

            # Re-evaluate only this machine.
            mdp = _eval_machine_sequence(raw, mi, trial_sol.sequences[mi], epsilon)
            if not np.isfinite(mdp.energy):
                continue

            per_machine = list(eval_.per_machine)
            per_machine[mi] = mdp
            total = 0.0
            feasible = True
            makespan = 0
            for md in per_machine:
                if not np.isfinite(md.energy):
                    feasible = False
                    break
                total += float(md.energy)
                makespan = max(makespan, int(md.makespan))

            if not feasible or makespan > int(epsilon):
                continue

            if float(total) < best_total:
                best_total = float(total)
                best_sol = trial_sol
                best_eval = FullEval(
                    total_energy=float(total),
                    makespan=int(makespan),
                    feasible=True,
                    per_machine=per_machine,
                )
                best_mi = mi

    return best_sol, best_eval, best_mi


def _try_insertion_at(
    raw: RawInstance,
    epsilon: int,
    sol: Solution,
    eval_: FullEval,
    job: int,
    mi: int,
    pos: int,
) -> Tuple[Optional[Solution], Optional[FullEval]]:
    """Try inserting a job at an exact (machine, position).

    This is used by stochastic repairs to allow constructing arbitrary sequences.
    """
    trial_sol = Solution(sequences=[list(s) for s in sol.sequences])
    trial_sol.sequences[int(mi)].insert(int(pos), int(job))

    mdp = _eval_machine_sequence(raw, int(mi), trial_sol.sequences[int(mi)], epsilon)
    if not np.isfinite(mdp.energy):
        return None, None

    per_machine = list(eval_.per_machine)
    per_machine[int(mi)] = mdp

    total = 0.0
    feasible = True
    makespan = 0
    for md in per_machine:
        if not np.isfinite(md.energy):
            feasible = False
            break
        total += float(md.energy)
        makespan = max(makespan, int(md.makespan))
    if (not feasible) or makespan > int(epsilon):
        return None, None

    return (
        trial_sol,
        FullEval(
            total_energy=float(total),
            makespan=int(makespan),
            feasible=True,
            per_machine=per_machine,
        ),
    )


def _repair_greedy(
    raw: RawInstance,
    epsilon: int,
    partial: Solution,
    partial_eval: FullEval,
    removed_jobs: List[int],
    cfg: ALNSConfig,
    rng: random.Random,
) -> Tuple[Optional[Solution], Optional[FullEval], List[int]]:
    touched: set[int] = set()

    # Insert longer jobs first (tends to help feasibility/structure).
    jobs = list(removed_jobs)
    rng.shuffle(jobs)
    jobs.sort(key=lambda j: int(raw.p[j]), reverse=True)

    sol = partial
    ev = partial_eval

    for j in jobs:
        sol2, ev2, mi = _best_insertion_for_job(raw, epsilon, sol, ev, j, cfg, rng)
        if sol2 is None or ev2 is None or mi is None:
            return None, None, sorted(touched)
        sol, ev = sol2, ev2
        touched.add(mi)

    return sol, ev, sorted(touched)


def _repair_random(
    raw: RawInstance,
    epsilon: int,
    partial: Solution,
    partial_eval: FullEval,
    removed_jobs: List[int],
    cfg: ALNSConfig,
    rng: random.Random,
    max_trials_per_job: int = 128,
) -> Tuple[Optional[Solution], Optional[FullEval], List[int]]:
    """Stochastic repair: insert jobs into random machines/positions.

    This is intentionally *non-greedy* so that (destroy_all + repair_random)
    has the capability to reconstruct any feasible assignment+sequence.
    """
    touched: set[int] = set()
    jobs = list(removed_jobs)
    rng.shuffle(jobs)

    sol = partial
    ev = partial_eval

    for j in jobs:
        ok = False
        for _ in range(int(max_trials_per_job)):
            mi = int(rng.randrange(int(raw.m)))
            npos = len(sol.sequences[mi])
            pos = int(rng.randrange(npos + 1))
            sol2, ev2 = _try_insertion_at(raw, epsilon, sol, ev, int(j), mi, pos)
            if sol2 is None or ev2 is None:
                continue
            sol, ev = sol2, ev2
            touched.add(mi)
            ok = True
            break

        if not ok:
            # If pure random failed, fall back to greedy best insertion
            sol2, ev2, mi2 = _best_insertion_for_job(raw, epsilon, sol, ev, j, cfg, rng)
            if sol2 is None or ev2 is None or mi2 is None:
                return None, None, sorted(touched)
            sol, ev = sol2, ev2
            touched.add(int(mi2))

    return sol, ev, sorted(touched)


def default_action_list() -> List[ALNSAction]:
    """Default discrete action set for RL/control experiments."""
    return build_action_list(
        destroy_ops=["random", "worst_machine", "longest", "all"],
        repair_ops=["greedy", "random"],
        k_mults=[0.5, 1.0],
    )


def build_action_list(
    *,
    destroy_ops: List[str],
    repair_ops: List[str],
    k_mults: List[float],
) -> List[ALNSAction]:
    """Build a discrete action list.

    Notes:
        - `destroy_all` ignores `k_mults` (it removes all jobs).
        - This function is used by the RL environment to select safer action sets.
    """

    actions: List[ALNSAction] = []
    for d in destroy_ops:
        for rname in repair_ops:
            if d == "all":
                actions.append(("all", 1.0, str(rname)))
            else:
                for km in k_mults:
                    actions.append((str(d), float(km), str(rname)))
    return actions


def alns_state_features(
    raw: RawInstance,
    sol: Solution,
    ev: FullEval,
    best_ev: FullEval,
    epsilon: int,
    it: int,
    no_improve: int,
    tau: float,
    cfg: ALNSConfig,
) -> List[float]:
    p = np.asarray(raw.p, dtype=np.float32)
    loads = np.array(
        [sum(p[j] for j in seq) for seq in sol.sequences], dtype=np.float32
    )
    if loads.size == 0:
        load_mean = 0.0
        load_cv = 0.0
    else:
        load_mean = float(loads.mean())
        load_std = float(loads.std())
        load_cv = float(load_std / (load_mean + 1e-6))
    return [
        float(it) / float(max(1, cfg.max_iters)),
        float(no_improve) / float(max(1, cfg.no_improve_limit)),
        float(tau),
        float(ev.total_energy) if np.isfinite(ev.total_energy) else 1e9,
        float(best_ev.total_energy) if np.isfinite(best_ev.total_energy) else 1e9,
        (
            float(ev.total_energy - best_ev.total_energy)
            if np.isfinite(ev.total_energy) and np.isfinite(best_ev.total_energy)
            else 0.0
        ),
        float(ev.makespan),
        float(int(epsilon) - int(ev.makespan)),
        load_mean,
        load_cv,
    ]


def alns_apply_action(
    raw: RawInstance,
    epsilon: int,
    sol: Solution,
    ev: FullEval,
    action: ALNSAction,
    cfg: ALNSConfig,
    rng: random.Random,
) -> Tuple[Optional[Solution], Optional[FullEval], List[int], List[int]]:
    """Apply a single destroy+repair action (no acceptance logic here)."""
    dname, km, rname = action

    # Make destruction gentler when we're close to the deadline.
    slack = max(0, int(epsilon) - int(ev.makespan))
    slack_norm = float(slack) / float(max(1, int(epsilon)))
    slack_norm = max(0.0, min(1.0, float(slack_norm)))
    slack_mult = 0.5 + 0.5 * slack_norm

    def _one_attempt(
        *,
        destroy_name: str,
        k_mult: float,
        repair_name: str,
        k_override: Optional[int],
    ) -> Tuple[Optional[Solution], Optional[FullEval], List[int], List[int]]:
        if destroy_name == "all":
            partial, removed, touched = _destroy_all(sol)
        else:
            if k_override is None:
                k_base = (
                    float(cfg.destroy_frac)
                    * float(raw.n)
                    * float(k_mult)
                    * float(slack_mult)
                )
                k = int(max(cfg.destroy_min, min(cfg.destroy_max, int(round(k_base)))))
            else:
                k = int(max(cfg.destroy_min, min(cfg.destroy_max, int(k_override))))

            if destroy_name == "random":
                partial, removed, touched = _destroy_random(sol, k, rng)
            elif destroy_name == "worst_machine":
                partial, removed, touched = _destroy_worst_machine(sol, ev, k, rng)
            elif destroy_name == "longest":
                partial, removed, touched = _destroy_longest_jobs(raw, sol, k, rng)
            else:
                raise ValueError(f"Unknown destroy op: {destroy_name}")

        partial_ev = _update_eval_for_machines(
            raw=raw,
            epsilon=epsilon,
            sol=partial,
            prev=ev,
            machine_indices=touched,
        )

        if repair_name == "greedy":
            repaired, repaired_ev, touched2 = _repair_greedy(
                raw=raw,
                epsilon=epsilon,
                partial=partial,
                partial_eval=partial_ev,
                removed_jobs=removed,
                cfg=cfg,
                rng=rng,
            )
        elif repair_name == "random":
            repaired, repaired_ev, touched2 = _repair_random(
                raw=raw,
                epsilon=epsilon,
                partial=partial,
                partial_eval=partial_ev,
                removed_jobs=removed,
                cfg=cfg,
                rng=rng,
            )
        else:
            raise ValueError(f"Unknown repair op: {repair_name}")

        if repaired is None or repaired_ev is None:
            return None, None, touched, removed
        return repaired, repaired_ev, sorted(set(touched).union(touched2)), removed

    # Reduce the likelihood of catastrophic infeasibility close to the deadline.
    destroy_name0 = str(dname)
    if (
        bool(cfg.avoid_destroy_all_when_tight)
        and destroy_name0 == "all"
        and float(slack_norm) < float(cfg.avoid_destroy_all_slack_threshold)
    ):
        destroy_name0 = "worst_machine"

    # Attempt the requested action; if infeasible, retry with smaller destruction and/or greedy repair.
    max_retries = int(max(0, cfg.feasibility_retries))
    shrink = float(cfg.feasibility_retry_shrink)
    shrink = float(shrink) if np.isfinite(shrink) else 0.5
    shrink = float(max(0.1, min(0.9, shrink)))

    # We only compute a k_override for non-"all" destroys.
    k_base0 = float(cfg.destroy_frac) * float(raw.n) * float(km) * float(slack_mult)
    k0 = int(max(cfg.destroy_min, min(cfg.destroy_max, int(round(k_base0)))))

    last_touched: List[int] = []
    last_removed: List[int] = []
    for attempt in range(max_retries + 1):
        if destroy_name0 == "all":
            k_override = None
        else:
            k_override = int(max(1, int(round(float(k0) * (shrink**attempt)))))

        repair_name = str(rname)
        if attempt > 0 and bool(cfg.feasibility_force_greedy_on_retry):
            repair_name = "greedy"

        repaired, repaired_ev, touched_all, removed = _one_attempt(
            destroy_name=str(destroy_name0),
            k_mult=float(km),
            repair_name=str(repair_name),
            k_override=k_override,
        )
        last_touched, last_removed = list(touched_all), list(removed)

        if repaired is None or repaired_ev is None:
            continue
        if repaired_ev is None or (not bool(repaired_ev.feasible)):
            continue

        _assert_solution_valid(repaired, n_jobs=raw.n, m=raw.m)

        if cfg.verify_cost:
            for mi in touched_all:
                _eval_machine_sequence(
                    raw,
                    int(mi),
                    repaired.sequences[int(mi)],
                    epsilon,
                    verify_cost=True,
                )

        if cfg.local_search:
            repaired, repaired_ev = _try_local_intra_swap(
                raw, epsilon, repaired, repaired_ev, touched_all, cfg, rng
            )

        return repaired, repaired_ev, touched_all, removed

    return None, None, last_touched, last_removed


# -------------------------
# Local search (light VND-ish)
# -------------------------


def _try_local_intra_swap(
    raw: RawInstance,
    epsilon: int,
    sol: Solution,
    ev: FullEval,
    machines: List[int],
    cfg: ALNSConfig,
    rng: random.Random,
) -> Tuple[Solution, FullEval]:
    if not machines:
        return sol, ev

    best_sol = sol
    best_ev = ev

    for mi in machines:
        seq = best_sol.sequences[mi]
        n = len(seq)
        if n < 2:
            continue

        # sample swap pairs
        pairs = set()
        trials = min(cfg.local_max_pair_trials, n * (n - 1) // 2)
        for _ in range(trials):
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            pairs.add((i, j))

        for i, j in pairs:
            trial_sol = Solution(sequences=[list(s) for s in best_sol.sequences])
            trial_seq = trial_sol.sequences[mi]
            trial_seq[i], trial_seq[j] = trial_seq[j], trial_seq[i]

            mdp = _eval_machine_sequence(raw, mi, trial_seq, epsilon)
            if not np.isfinite(mdp.energy):
                continue

            per_machine = list(best_ev.per_machine)
            per_machine[mi] = mdp
            total = sum(float(x.energy) for x in per_machine)
            makespan = max(int(x.makespan) for x in per_machine)
            if makespan > int(epsilon):
                continue

            if float(total) < float(best_ev.total_energy):
                best_sol = trial_sol
                best_ev = FullEval(
                    total_energy=float(total),
                    makespan=int(makespan),
                    feasible=True,
                    per_machine=per_machine,
                )

    return best_sol, best_ev


# -------------------------
# ALNS main
# -------------------------


def build_baseline_solution_cwp_random(
    raw: RawInstance,
    epsilon: int,
    rng: random.Random,
    top_k: int = 80,
) -> Tuple[Solution, FullEval, float]:
    """Baseline requested by user: CWP assignment + random sequencing + DP."""
    t0 = time.perf_counter()
    cwp = construct_schedule_CWP(
        processing_times=raw.p,
        energy_rates=[float(e) for e in raw.e],
        ct=np.asarray(raw.ct, dtype=np.float64),
        epsilon=int(epsilon),
        top_k=int(top_k),
        seed=None,
    )
    job_sets = get_machine_job_sets(cwp, raw.m)
    job_sets = _fill_missing_jobs_greedily(raw, job_sets)
    sequences = _randomize_sequences(job_sets, rng)
    sol = Solution(sequences=sequences)
    _assert_solution_valid(sol, n_jobs=raw.n, m=raw.m)
    ev = evaluate_solution(raw, sol, epsilon)
    t1 = time.perf_counter()
    return sol, ev, (t1 - t0) * 1000.0


def build_initial_solution_cwp_spt(
    raw: RawInstance,
    epsilon: int,
    top_k: int = 80,
) -> Tuple[Solution, FullEval]:
    """Stronger start than baseline: CWP assignment + SPT sequencing."""
    cwp = construct_schedule_CWP(
        processing_times=raw.p,
        energy_rates=[float(e) for e in raw.e],
        ct=np.asarray(raw.ct, dtype=np.float64),
        epsilon=int(epsilon),
        top_k=int(top_k),
        seed=None,
    )
    job_sets = get_machine_job_sets(cwp, raw.m)
    job_sets = _fill_missing_jobs_greedily(raw, job_sets)

    seqs: List[List[int]] = []
    for js in job_sets:
        js2 = list(js)
        js2.sort(key=lambda j: (int(raw.p[j]), int(j)))
        seqs.append(js2)

    sol = Solution(sequences=seqs)
    _assert_solution_valid(sol, n_jobs=raw.n, m=raw.m)
    ev = evaluate_solution(raw, sol, epsilon)
    return sol, ev


def alns_solve(
    raw: RawInstance,
    epsilon: int,
    cfg: ALNSConfig,
    rng: random.Random,
    top_k_cwp: int = 80,
) -> Dict[str, Any]:
    """Run ALNS search on a single instance.

    Returns dict with keys:
      - best_solution, best_eval
      - init_solution, init_eval
      - iter_log (optional)
      - solve_time_ms
      - oracle_dataset (optional)
    """

    t0 = time.perf_counter()

    # Start from a strong deterministic initialization (reduces variance).
    cur_sol, cur_ev = build_initial_solution_cwp_spt(raw, epsilon, top_k=top_k_cwp)
    best_sol, best_ev = copy.deepcopy(cur_sol), cur_ev

    if cfg.verify_cost:
        for mi in range(int(raw.m)):
            _eval_machine_sequence(
                raw,
                int(mi),
                cur_sol.sequences[int(mi)],
                epsilon,
                verify_cost=True,
            )

    tau = float(cfg.sa_tau0)
    no_improve = 0

    iter_log: List[Dict[str, Any]] = []

    action_list = default_action_list()

    oracle_rows: List[Dict[str, Any]] = []

    def state_features(
        sol: Solution, ev: FullEval, it: int, no_imp: int, tau_: float
    ) -> List[float]:
        return alns_state_features(
            raw, sol, ev, best_ev, epsilon, it, no_imp, tau_, cfg
        )

    # Classic ALNS operator adaptation state
    weights = np.ones(len(action_list), dtype=np.float64)
    seg_scores = np.zeros(len(action_list), dtype=np.float64)
    seg_counts = np.zeros(len(action_list), dtype=np.int64)

    # Main loop
    for it in range(int(cfg.max_iters)):
        if no_improve >= int(cfg.no_improve_limit):
            break

        # Optional oracle dataset collection: evaluate all actions from this state.
        if cfg.oracle_dataset and len(oracle_rows) < int(cfg.oracle_max_states):
            feats = state_features(cur_sol, cur_ev, it, no_improve, tau)
            deltas: List[float] = []
            best_a = None
            best_delta = float("inf")
            for a in action_list:
                cand_sol, cand_ev, _, _ = alns_apply_action(
                    raw, epsilon, cur_sol, cur_ev, a, cfg, rng
                )
                if cand_ev is None or not cand_ev.feasible:
                    delta = float("inf")
                else:
                    delta = float(cand_ev.total_energy - cur_ev.total_energy)
                deltas.append(delta)
                if delta < best_delta:
                    best_delta = delta
                    best_a = a
            if best_a is not None:
                oracle_rows.append(
                    {
                        "x": feats,
                        "best_action": action_list.index(best_a),
                        "delta_by_action": deltas,
                    }
                )

        # Roulette selection by adaptive weights
        probs = weights / float(weights.sum())
        a_idx = int(np.searchsorted(np.cumsum(probs), rng.random(), side="right"))
        a_idx = max(0, min(len(action_list) - 1, a_idx))
        action = action_list[a_idx]
        cand_sol, cand_ev, touched_m, removed = alns_apply_action(
            raw, epsilon, cur_sol, cur_ev, action, cfg, rng
        )
        if cand_ev is None or cand_sol is None:
            # Failed repair; continue
            no_improve += 1
            tau *= float(cfg.sa_decay)
            continue

        delta = float(cand_ev.total_energy - cur_ev.total_energy)
        accept = False
        if cand_ev.feasible:
            if delta <= 0:
                accept = True
            else:
                # SA acceptance (guard underflow/overflow)
                if tau > 1e-12:
                    prob = math.exp(-delta / tau)
                else:
                    prob = 0.0
                accept = rng.random() < prob

        if accept:
            cur_sol, cur_ev = cand_sol, cand_ev

        improved_best = False
        if cand_ev.feasible and float(cand_ev.total_energy) < float(
            best_ev.total_energy
        ):
            best_sol, best_ev = copy.deepcopy(cand_sol), cand_ev
            no_improve = 0
            improved_best = True
        else:
            no_improve += 1

        # Update segment scores for operator adaptation
        seg_counts[a_idx] += 1
        if cand_ev.feasible:
            if improved_best:
                seg_scores[a_idx] += float(cfg.sigma_best)
            elif accept and delta < 0:
                seg_scores[a_idx] += float(cfg.sigma_improve)
            elif accept:
                seg_scores[a_idx] += float(cfg.sigma_accept_worse)

        if cfg.keep_iter_log:
            per_machine_stats = None
            if int(cfg.log_per_machine_every) > 0 and (
                (it + 1) % int(cfg.log_per_machine_every) == 0
            ):
                per_machine_stats = {
                    "cur": [
                        (float(x.energy), int(x.makespan)) for x in cur_ev.per_machine
                    ],
                    "best": [
                        (float(x.energy), int(x.makespan)) for x in best_ev.per_machine
                    ],
                }
            iter_log.append(
                {
                    "iter": int(it),
                    "action": {
                        "destroy": action[0],
                        "k_mult": action[1],
                        "repair": action[2],
                    },
                    "k": (
                        int(len(removed))
                        if action[0] == "all"
                        else int(
                            max(
                                cfg.destroy_min,
                                min(
                                    cfg.destroy_max,
                                    int(round(cfg.destroy_frac * raw.n * action[1])),
                                ),
                            )
                        )
                    ),
                    "removed": int(len(removed)),
                    "touched_machines": touched_m,
                    "delta_energy": float(delta) if np.isfinite(delta) else None,
                    "accepted": bool(accept),
                    "cand_energy": (
                        float(cand_ev.total_energy) if cand_ev.feasible else None
                    ),
                    "cur_energy": (
                        float(cur_ev.total_energy) if cur_ev.feasible else None
                    ),
                    "best_energy": (
                        float(best_ev.total_energy) if best_ev.feasible else None
                    ),
                    "cur_makespan": int(cur_ev.makespan),
                    "best_makespan": int(best_ev.makespan),
                    "tau": float(tau),
                    "improved_best": bool(improved_best),
                    "weights": [float(w) for w in weights.tolist()],
                    "probs": [float(p) for p in probs.tolist()],
                    "per_machine": per_machine_stats,
                }
            )

        tau *= float(cfg.sa_decay)

        # End of segment: update roulette weights
        if (it + 1) % int(max(1, cfg.segment_len)) == 0:
            for j in range(len(weights)):
                if seg_counts[j] > 0 and seg_scores[j] > 0:
                    avg = float(seg_scores[j]) / float(seg_counts[j])
                    weights[j] = (1.0 - float(cfg.reaction_factor)) * float(
                        weights[j]
                    ) + float(cfg.reaction_factor) * avg
            # Prevent collapse
            weights = np.maximum(weights, 1e-3)
            seg_scores.fill(0.0)
            seg_counts.fill(0)

    t1 = time.perf_counter()

    out: Dict[str, Any] = {
        "init_solution": cur_sol,
        "init_eval": cur_ev,
        "best_solution": best_sol,
        "best_eval": best_ev,
        "solve_time_ms": (t1 - t0) * 1000.0,
    }

    if cfg.keep_iter_log:
        out["iter_log"] = iter_log

    if cfg.oracle_dataset:
        out["oracle_dataset"] = {
            "action_list": [
                {"destroy": a[0], "k_mult": a[1], "repair": a[2]} for a in action_list
            ],
            "rows": oracle_rows,
        }

    return out


def save_oracle_dataset_npz(oracle_dataset: Dict[str, Any], path: str) -> None:
    rows = oracle_dataset.get("rows", [])
    if not rows:
        raise ValueError("oracle_dataset has no rows")

    X = np.asarray([r["x"] for r in rows], dtype=np.float32)
    y = np.asarray([r["best_action"] for r in rows], dtype=np.int64)
    deltas = np.asarray([r["delta_by_action"] for r in rows], dtype=np.float32)

    meta = {
        "action_list": oracle_dataset.get("action_list", []),
        "num_rows": int(X.shape[0]),
        "num_actions": int(deltas.shape[1]),
        "feature_dim": int(X.shape[1]),
    }

    np.savez_compressed(path, X=X, y=y, deltas=deltas, meta=json.dumps(meta))


def solution_to_jsonable(sol: Solution) -> Any:
    return [[int(j) for j in seq] for seq in sol.sequences]


def eval_to_jsonable(ev: FullEval) -> Any:
    return {
        "total_energy": (
            float(ev.total_energy) if np.isfinite(ev.total_energy) else None
        ),
        "makespan": int(ev.makespan),
        "feasible": bool(ev.feasible),
        "per_machine": [
            {
                "energy": float(m.energy) if np.isfinite(m.energy) else None,
                "makespan": int(m.makespan),
            }
            for m in ev.per_machine
        ],
    }

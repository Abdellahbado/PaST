"""Deterministic candidate move generation for NeuroLS.

This module generates candidate moves in a fixed, reproducible order.
Given the same state and operator, the same candidate move is always produced.

Key principles:
1. Lexicographic enumeration: moves enumerated by (m1, pos1, m2, pos2, ...)
2. Best-improvement: evaluate all candidates, return best
3. Top-K for large instances: use deterministic proxy ranking for N > 50
4. No randomness: fixed tie-breaking by indices
5. Multi-criteria: rank moves by cost, exposure, or balance (for AAN_MC)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from PaST.neurols.solution import PMALNSSolution
from PaST.neurols.operators import Move, Operator, OperatorID, OPERATORS, OPERATOR_BY_ID
from PaST.neurols.move_evaluator import MoveEvaluator, FullEvaluation, MoveEvaluation
from PaST.neurols.price_embedding import PriceFeatureExtractor


class CriterionID(IntEnum):
    """Criterion for selecting the best move.

    COST_BEST: lowest delta_cost (current behavior)
    EXPOSURE_BEST: largest reduction in peak-slot energy exposure
    BALANCE_BEST: largest reduction in load imbalance (std of machine loads)
    """

    COST_BEST = 0
    EXPOSURE_BEST = 1
    BALANCE_BEST = 2


N_CRITERIA = len(CriterionID)
CRITERIA = list(CriterionID)


@dataclass
class CandidateConfig:
    """Configuration for candidate generation."""

    # Maximum moves to enumerate per operator (for large instances)
    max_moves_full: int = 500  # Full enumeration if # moves <= this
    max_moves_topk: int = 100  # Top-K evaluated if # moves > max_moves_full

    # Threshold for using top-K
    n_jobs_topk_threshold: int = 50

    # Proxy scores for top-K selection (when full enumeration too expensive)
    use_proxy_ranking: bool = True

    # Proxy mode:
    # - "load": existing load-balance heuristics (paper-ish)
    # - "price_aware": uses current DP schedule start_times to bias toward
    #   moves that reduce peak/expensive exposure and prefer low-rate machines.
    proxy_mode: str = "load"

    # Weights for price-aware proxy (only used when proxy_mode == "price_aware")
    price_badness_peak_frac_w: float = 0.7
    price_badness_avg_price_w: float = 0.3
    price_rate_w: float = 0.5


class CandidateGenerator:
    """Deterministic generator for candidate LS moves.

    For each operator, generates the best candidate move via:
    1. Enumerate all valid moves in deterministic order
    2. Evaluate each using DP (or proxy for large instances)
    3. Return best-improvement move

    Guarantees:
    - Same solution + operator → same candidate move
    - No random sampling or tie-breaking
    """

    def __init__(
        self,
        evaluator: MoveEvaluator,
        config: Optional[CandidateConfig] = None,
    ):
        """Initialize candidate generator.

        Args:
            evaluator: MoveEvaluator for computing move costs
            config: Configuration for generation
        """
        self.evaluator = evaluator
        self.config = config or CandidateConfig()

        # Local extractor for deterministic, state-dependent proxy signals.
        # (This is cheap; it only holds prefix sums over ct[:K].)
        self._price_extractor = PriceFeatureExtractor(
            self.evaluator.ct, self.evaluator.K
        )

    def generate_best_move(
        self,
        solution: PMALNSSolution,
        current_eval: FullEvaluation,
        operator: Operator,
    ) -> Optional[MoveEvaluation]:
        """Generate the best move for an operator.

        Args:
            solution: Current solution
            current_eval: Current solution evaluation
            operator: Operator to use

        Returns:
            Best MoveEvaluation, or None if no valid moves
        """
        # Enumerate all moves for this operator
        all_moves = operator.enumerate_moves(solution)

        if not all_moves:
            return None

        n_moves = len(all_moves)
        n_jobs = solution.n_jobs

        # Decide strategy based on instance size
        if (
            n_jobs > self.config.n_jobs_topk_threshold
            and n_moves > self.config.max_moves_full
        ):
            # Large neighborhood — limit evaluation
            if self.config.use_proxy_ranking:
                # Use top-K with proxy ranking (heuristic pre-filter)
                moves_to_eval = self._select_topk_by_proxy(
                    solution,
                    current_eval,
                    all_moves,
                    operator,
                    self.config.max_moves_topk,
                )
            else:
                # No proxy: just take first max_moves_topk moves (deterministic order)
                moves_to_eval = all_moves[: self.config.max_moves_topk]
        else:
            # Full enumeration — evaluate all moves (no cap needed)
            moves_to_eval = all_moves

        # Evaluate selected moves
        best_eval: Optional[MoveEvaluation] = None
        best_delta = float("inf")

        for move in moves_to_eval:
            move_eval = self.evaluator.evaluate_move(solution, current_eval, move)

            if not move_eval.feasible:
                continue

            # Best-improvement: lowest delta (ties broken by move order = first wins)
            if move_eval.delta_cost < best_delta:
                best_delta = move_eval.delta_cost
                best_eval = move_eval

        return best_eval

    def generate_multi_criteria_moves(
        self,
        solution: PMALNSSolution,
        current_eval: FullEvaluation,
        operator: Operator,
    ) -> Dict[CriterionID, Optional[MoveEvaluation]]:
        """Generate best move per criterion for an operator.

        Returns up to 3 different moves (one per criterion).
        Criteria:
            COST_BEST: lowest delta_cost (same as generate_best_move)
            EXPOSURE_BEST: largest reduction in peak-slot energy exposure
            BALANCE_BEST: largest reduction in load std deviation

        If two criteria agree on the same move, both return the same result.
        """
        # Enumerate all moves for this operator
        all_moves = operator.enumerate_moves(solution)

        if not all_moves:
            return {c: None for c in CriterionID}

        n_moves = len(all_moves)
        n_jobs = solution.n_jobs

        # Decide strategy based on instance size (same as generate_best_move)
        if (
            n_jobs > self.config.n_jobs_topk_threshold
            and n_moves > self.config.max_moves_full
        ):
            if self.config.use_proxy_ranking:
                moves_to_eval = self._select_topk_by_proxy(
                    solution,
                    current_eval,
                    all_moves,
                    operator,
                    self.config.max_moves_topk,
                )
            else:
                moves_to_eval = all_moves[: self.config.max_moves_topk]
        else:
            moves_to_eval = all_moves

        # Precompute machine loads for balance criterion
        p = self.evaluator.processing_times
        loads = solution.get_machine_loads(p).astype(np.float64)
        current_load_std = float(np.std(loads))

        # Precompute per-machine exposure metric for exposure criterion.
        # We intentionally use a *work-weighted* peak metric (not just peak fraction)
        # so the criterion can prefer moving substantial work off peak, and also
        # prioritize higher-rate machines.
        m = solution.n_machines
        current_peak_metric = np.zeros(m, dtype=np.float64)
        for mi in range(m):
            seq = solution.sequences[mi]
            me = current_eval.per_machine[mi]
            if seq and me.start_times and len(me.start_times) == len(seq):
                proc_times = [int(p[j]) for j in seq]
                exp = self._price_extractor.compute_machine_price_exposure(
                    me.start_times,
                    proc_times,
                    float(self.evaluator.machine_energy_rates[mi]),
                )
                # exp = [work_norm(3), frac(3), avg_price_norm]
                # Use peak workload norm (index 2) weighted by machine rate.
                peak_work_norm = float(exp[2])
                current_peak_metric[mi] = (
                    float(self.evaluator.machine_energy_rates[mi]) * peak_work_norm
                )

        # Track best per criterion
        best_evals: Dict[CriterionID, Optional[MoveEvaluation]] = {
            c: None for c in CriterionID
        }
        best_scores = {
            CriterionID.COST_BEST: float("inf"),  # minimize
            CriterionID.EXPOSURE_BEST: float("inf"),  # minimize
            CriterionID.BALANCE_BEST: float("inf"),  # minimize
        }

        for move in moves_to_eval:
            move_eval = self.evaluator.evaluate_move(solution, current_eval, move)

            if not move_eval.feasible:
                continue

            # --- COST criterion (same as current) ---
            cost_score = move_eval.delta_cost
            if cost_score < best_scores[CriterionID.COST_BEST]:
                best_scores[CriterionID.COST_BEST] = cost_score
                best_evals[CriterionID.COST_BEST] = move_eval

            # --- EXPOSURE criterion ---
            # Compute change in a peak-exposure metric on affected machines.
            # IMPORTANT: Avoid cloning the full solution per move (too slow).
            # We reconstruct only the affected machine sequences from move params.
            exposure_delta = 0.0
            new_seqs = self._get_affected_new_sequences(
                solution, move, move_eval.affected_machines
            )
            for mi in move_eval.affected_machines:
                new_seq = new_seqs.get(mi)
                if new_seq is None:
                    continue
                if len(new_seq) == 0:
                    # Machine became empty → 0 peak exposure
                    exposure_delta += 0.0 - float(current_peak_metric[mi])
                    continue

                new_me = move_eval.affected_machine_evals.get(mi)
                starts = (
                    new_me.start_times
                    if (
                        new_me is not None
                        and new_me.start_times
                        and len(new_me.start_times) == len(new_seq)
                    )
                    else None
                )
                if starts is None:
                    # If the evaluator didn't provide start_times, skip exposure signal.
                    continue

                new_proc = [int(p[j]) for j in new_seq]
                exp = self._price_extractor.compute_machine_price_exposure(
                    starts,
                    new_proc,
                    float(self.evaluator.machine_energy_rates[mi]),
                )
                peak_work_norm = float(exp[2])
                new_metric = (
                    float(self.evaluator.machine_energy_rates[mi]) * peak_work_norm
                )
                exposure_delta += new_metric - float(current_peak_metric[mi])

            if exposure_delta < best_scores[CriterionID.EXPOSURE_BEST]:
                best_scores[CriterionID.EXPOSURE_BEST] = float(exposure_delta)
                best_evals[CriterionID.EXPOSURE_BEST] = move_eval

            # --- BALANCE criterion ---
            new_loads = self._compute_processing_loads_after_move(solution, loads, move)
            new_load_std = float(np.std(new_loads))
            balance_delta = float(new_load_std - current_load_std)

            if balance_delta < best_scores[CriterionID.BALANCE_BEST]:
                best_scores[CriterionID.BALANCE_BEST] = balance_delta
                best_evals[CriterionID.BALANCE_BEST] = move_eval

        return best_evals

    def _compute_processing_loads_after_move(
        self,
        solution: PMALNSSolution,
        current_loads: np.ndarray,
        move: Move,
    ) -> np.ndarray:
        """Compute per-machine processing-time loads after applying a move.

        This is a *structural* criterion (assignment balance), so it should be
        based on processing-time totals, not DP makespan (which includes idling).
        """
        loads = np.asarray(current_loads, dtype=np.float64).copy()
        p = self.evaluator.processing_times

        op = move.operator
        if op == OperatorID.RELOCATE_1:
            from_m, from_pos, to_m, _to_pos = move.params
            job = solution.sequences[from_m][from_pos]
            pj = float(p[job])
            loads[int(from_m)] -= pj
            loads[int(to_m)] += pj
            return loads

        if op == OperatorID.SWAP_1:
            m1, pos1, m2, pos2 = move.params
            if int(m1) == int(m2):
                return loads
            job1 = solution.sequences[m1][pos1]
            job2 = solution.sequences[m2][pos2]
            p1 = float(p[job1])
            p2 = float(p[job2])
            loads[int(m1)] += p2 - p1
            loads[int(m2)] += p1 - p2
            return loads

        if op == OperatorID.BLOCK_RELOCATE:
            from_m, from_start, block_size, to_m, _to_pos = move.params
            if int(from_m) == int(to_m):
                return loads
            block_jobs = solution.sequences[from_m][
                from_start : from_start + block_size
            ]
            block_p = float(sum(float(p[j]) for j in block_jobs))
            loads[int(from_m)] -= block_p
            loads[int(to_m)] += block_p
            return loads

        # Default: no load change (or unsupported operator)
        return loads

    def _get_affected_new_sequences(
        self,
        solution: PMALNSSolution,
        move: Move,
        affected_machines: List[int],
    ) -> Dict[int, List[int]]:
        """Reconstruct new sequences for affected machines only.

        Avoid cloning `PMALNSSolution` for every candidate move.
        Supports the operators used by AAN_MC.
        """
        op = move.operator
        new_seqs: Dict[int, List[int]] = {}

        # Seed with copies of affected sequences
        for mi in affected_machines:
            new_seqs[int(mi)] = list(solution.sequences[int(mi)])

        if op == OperatorID.RELOCATE_1:
            from_m, from_pos, to_m, to_pos = move.params
            from_m = int(from_m)
            to_m = int(to_m)
            from_pos = int(from_pos)
            to_pos = int(to_pos)
            job = new_seqs[from_m].pop(from_pos)
            if from_m == to_m and to_pos > from_pos:
                to_pos -= 1
            new_seqs[to_m].insert(to_pos, job)
            return new_seqs

        if op == OperatorID.SWAP_1:
            m1, pos1, m2, pos2 = move.params
            m1 = int(m1)
            m2 = int(m2)
            pos1 = int(pos1)
            pos2 = int(pos2)
            seq1 = new_seqs[m1]
            seq2 = new_seqs[m2]
            seq1[pos1], seq2[pos2] = seq2[pos2], seq1[pos1]
            return new_seqs

        if op == OperatorID.BLOCK_RELOCATE:
            from_m, from_start, block_size, to_m, to_pos = move.params
            from_m = int(from_m)
            to_m = int(to_m)
            from_start = int(from_start)
            block_size = int(block_size)
            to_pos = int(to_pos)
            block = new_seqs[from_m][from_start : from_start + block_size]
            del new_seqs[from_m][from_start : from_start + block_size]

            if from_m == to_m and to_pos > from_start:
                to_pos -= block_size
            for i, job in enumerate(block):
                new_seqs[to_m].insert(to_pos + i, job)
            return new_seqs

        # Unsupported operator for multi-criteria: just return copies
        return new_seqs

    def generate_all_operator_moves(
        self,
        solution: PMALNSSolution,
        current_eval: FullEvaluation,
    ) -> Dict[OperatorID, Optional[MoveEvaluation]]:
        """Generate best move for each operator.

        Args:
            solution: Current solution
            current_eval: Current solution evaluation

        Returns:
            Dict mapping operator ID to best move (or None)
        """
        result = {}
        for operator in OPERATORS:
            result[operator.id] = self.generate_best_move(
                solution, current_eval, operator
            )
        return result

    def _select_topk_by_proxy(
        self,
        solution: PMALNSSolution,
        current_eval: FullEvaluation,
        moves: List[Move],
        operator: Operator,
        k: int,
    ) -> List[Move]:
        """Select top-K moves using deterministic proxy scores.

        Proxy scoring heuristics by operator:
        - RELOCATE: prefer moving jobs from overloaded machines
        - SWAP: prefer exchanging with different processing times
        - INTRA_INSERT: prefer moves that improve local order
        - BLOCK: prefer moving from overloaded machines
        """
        if len(moves) <= k:
            return moves

        p = self.evaluator.processing_times
        loads = solution.get_machine_loads(p)
        avg_load = float(np.mean(loads))

        # Optional: price-aware badness per machine from the *current* DP schedule.
        # This is intentionally approximate: we use it for pre-filtering only.
        price_badness = None
        if str(self.config.proxy_mode).lower() == "price_aware":
            m = solution.n_machines
            bad = np.zeros(m, dtype=np.float64)
            for mi in range(m):
                seq = solution.sequences[mi]
                me = current_eval.per_machine[mi]
                if (
                    (not seq)
                    or (not me.start_times)
                    or (len(me.start_times) != len(seq))
                ):
                    continue
                proc_times = [int(p[j]) for j in seq]
                exp = self._price_extractor.compute_machine_price_exposure(
                    me.start_times,
                    proc_times,
                    float(self.evaluator.machine_energy_rates[mi]),
                )
                # exp = [work_norm(3), frac(3), avg_price_norm]
                peak_frac = float(exp[3 + 2])
                avg_price_norm = float(exp[6])
                bad[mi] = (
                    float(self.config.price_badness_peak_frac_w) * peak_frac
                    + float(self.config.price_badness_avg_price_w) * avg_price_norm
                )
            price_badness = bad

        scored_moves: List[Tuple[float, int, Move]] = []

        for idx, move in enumerate(moves):
            score = self._compute_proxy_score(
                move,
                solution,
                current_eval,
                loads,
                avg_load,
                p,
                price_badness,
            )
            # Negative score because we want highest scores first
            # Secondary sort by index for determinism
            scored_moves.append((-score, idx, move))

        # Sort by score (highest first), then by index
        scored_moves.sort(key=lambda x: (x[0], x[1]))

        return [move for _, _, move in scored_moves[:k]]

    def _compute_proxy_score(
        self,
        move: Move,
        solution: PMALNSSolution,
        current_eval: FullEvaluation,
        loads: np.ndarray,
        avg_load: float,
        p: np.ndarray,
        price_badness: Optional[np.ndarray],
    ) -> float:
        """Compute proxy score for a move (higher = more promising).

        The score estimates potential improvement without running DP.
        """
        op = move.operator

        proxy_mode = str(self.config.proxy_mode).lower()
        if proxy_mode not in ("load", "price_aware"):
            proxy_mode = "load"

        if proxy_mode == "price_aware" and price_badness is not None:
            rates = self.evaluator.machine_energy_rates

            def _job_start_level(mi: int, pos: int) -> int:
                me = current_eval.per_machine[mi]
                if not me.start_times or pos >= len(me.start_times):
                    return 0
                start = int(me.start_times[pos])
                if start < 0 or start >= len(self._price_extractor.slot_levels):
                    return 0
                return int(self._price_extractor.slot_levels[start])

        if op == OperatorID.RELOCATE_1:
            from_m, from_pos, to_m, to_pos = move.params
            job = solution.sequences[from_m][from_pos]
            job_p = float(p[job])

            if proxy_mode == "price_aware":
                # Prefer moving long work from high price-badness / high-rate
                # machines to lower ones. (Approximate; just for top-K.)
                bad_delta = float(price_badness[from_m]) - float(price_badness[to_m])
                rate_delta = float(rates[from_m]) - float(rates[to_m])
                return (
                    bad_delta + float(self.config.price_rate_w) * rate_delta
                ) * job_p

            # "load" (legacy)
            from_excess = float(loads[from_m]) - avg_load
            to_deficit = avg_load - float(loads[to_m])
            return from_excess + to_deficit + 0.01 * job_p

        elif op == OperatorID.SWAP_1:
            m1, pos1, m2, pos2 = move.params
            job1 = solution.sequences[m1][pos1]
            job2 = solution.sequences[m2][pos2]

            if m1 == m2:
                # Intra-machine swap: score by position change potential
                return abs(float(p[job1]) - float(p[job2]))
            else:
                if proxy_mode == "price_aware":
                    # Prefer swapping to move longer job off worse machine.
                    bad_delta = float(price_badness[m1]) - float(price_badness[m2])
                    rate_delta = float(rates[m1]) - float(rates[m2])
                    p_delta = float(p[job1]) - float(p[job2])
                    return (
                        bad_delta + float(self.config.price_rate_w) * rate_delta
                    ) * p_delta

                # Inter-machine: benefit of balancing loads
                delta1 = float(p[job2]) - float(p[job1])  # Change to machine 1
                load1_new = float(loads[m1]) + delta1
                load2_new = float(loads[m2]) - delta1

                # Score: reduction in max deviation from mean
                old_max_dev = max(
                    abs(float(loads[m1]) - avg_load), abs(float(loads[m2]) - avg_load)
                )
                new_max_dev = max(abs(load1_new - avg_load), abs(load2_new - avg_load))
                return old_max_dev - new_max_dev

        elif op == OperatorID.INTRA_INSERT:
            m, from_pos, to_pos = move.params
            job = solution.sequences[m][from_pos]

            if proxy_mode == "price_aware":
                # If the job currently starts in peak, moving it is more promising.
                level = _job_start_level(int(m), int(from_pos))
                peak = 1.0 if level == 2 else 0.0
                return peak * (abs(to_pos - from_pos) + 0.01 * float(p[job]))

            return abs(to_pos - from_pos) * 0.1 + float(p[job]) * 0.01

        elif op == OperatorID.BLOCK_RELOCATE:
            from_m, from_start, block_size, to_m, to_pos = move.params

            # Block processing time
            block_p = sum(
                float(p[solution.sequences[from_m][from_start + i]])
                for i in range(block_size)
            )

            if from_m == to_m:
                if proxy_mode == "price_aware":
                    level = _job_start_level(int(from_m), int(from_start))
                    peak = 1.0 if level == 2 else 0.0
                    return peak * (abs(to_pos - from_start) + 0.01 * block_p)
                return abs(to_pos - from_start) * 0.1 + block_p * 0.01
            else:
                if proxy_mode == "price_aware":
                    bad_delta = float(price_badness[from_m]) - float(
                        price_badness[to_m]
                    )
                    rate_delta = float(rates[from_m]) - float(rates[to_m])
                    return (
                        bad_delta + float(self.config.price_rate_w) * rate_delta
                    ) * block_p

                from_excess = float(loads[from_m]) - avg_load
                to_deficit = avg_load - float(loads[to_m])
                return from_excess + to_deficit + block_p * 0.01

        return 0.0


class DeterministicMoveSelector:
    """Deterministic selection of moves based on operator action.

    Given an operator selected by the controller, returns exactly one
    deterministic candidate move.
    """

    def __init__(
        self,
        evaluator: MoveEvaluator,
        config: Optional[CandidateConfig] = None,
    ):
        self.generator = CandidateGenerator(evaluator, config)
        self._cached_operator_moves: Optional[
            Dict[OperatorID, Optional[MoveEvaluation]]
        ] = None
        self._cached_solution_hash: Optional[Tuple] = None

    def get_move_for_operator(
        self,
        solution: PMALNSSolution,
        current_eval: FullEvaluation,
        operator_id: OperatorID,
    ) -> Optional[MoveEvaluation]:
        """Get the best move for a specific operator.

        Uses caching to avoid recomputation if solution unchanged.
        """
        solution_hash = solution.hash_key()

        # Check cache validity
        if self._cached_solution_hash != solution_hash:
            self._cached_operator_moves = None
            self._cached_solution_hash = solution_hash

        # Compute if not cached
        if self._cached_operator_moves is None:
            self._cached_operator_moves = self.generator.generate_all_operator_moves(
                solution, current_eval
            )

        return self._cached_operator_moves.get(operator_id)

    def invalidate_cache(self) -> None:
        """Invalidate the move cache."""
        self._cached_operator_moves = None
        self._cached_solution_hash = None

    def precompute_all_moves(
        self,
        solution: PMALNSSolution,
        current_eval: FullEvaluation,
    ) -> Dict[OperatorID, Optional[MoveEvaluation]]:
        """Precompute best moves for all operators.

        Useful when we want to present all options to the controller.
        """
        solution_hash = solution.hash_key()

        if (
            self._cached_solution_hash != solution_hash
            or self._cached_operator_moves is None
        ):
            self._cached_operator_moves = self.generator.generate_all_operator_moves(
                solution, current_eval
            )
            self._cached_solution_hash = solution_hash

        return self._cached_operator_moves


def test_determinism(seed: int = 42, n_tests: int = 10):
    """Test that candidate generation is deterministic."""
    rng = np.random.default_rng(seed)

    for _ in range(n_tests):
        # Random instance
        n_jobs = rng.integers(10, 30)
        n_machines = rng.integers(2, 6)
        K = rng.integers(50, 150)

        p = rng.integers(1, 10, size=n_jobs)
        ct = rng.uniform(0.5, 5.0, size=K)
        e = rng.integers(1, 4, size=n_machines)

        # Random solution
        sol = PMALNSSolution.from_random(n_jobs, n_machines, p, rng)

        # Create evaluator
        evaluator = MoveEvaluator(p, ct, e, K, n_machines)
        generator = CandidateGenerator(evaluator)

        # Evaluate solution
        full_eval = evaluator.evaluate_solution(sol)

        # Generate moves twice
        for operator in OPERATORS:
            move1 = generator.generate_best_move(sol, full_eval, operator)
            move2 = generator.generate_best_move(sol, full_eval, operator)

            if move1 is None and move2 is None:
                continue

            if move1 is None or move2 is None:
                raise AssertionError(f"Determinism failed: {move1} vs {move2}")

            if move1.move != move2.move:
                raise AssertionError(f"Move mismatch: {move1.move} vs {move2.move}")

            if abs(move1.delta_cost - move2.delta_cost) > 1e-9:
                raise AssertionError(
                    f"Cost mismatch: {move1.delta_cost} vs {move2.delta_cost}"
                )

    print(f"[PASS] Determinism test passed ({n_tests} instances)")


if __name__ == "__main__":
    test_determinism()

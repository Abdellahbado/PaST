"""Specialized Branch-and-Bound Solver for Job Scheduling.

Features:
- Relaxation-based Lower Bound (GCD-split jobs)
- Bin Packing Primal Heuristic (First Fit Decreasing into relaxed blocks)
- Alpha-Beta style pruning
- Internal optimized DP evaluation
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set

import numpy as np


@dataclass
class Instance:
    """Problem instance definition"""

    n_jobs: int
    processing_times: np.ndarray
    T: int
    energy_costs: np.ndarray

    def __post_init__(self):
        assert len(self.processing_times) == self.n_jobs
        assert len(self.energy_costs) == self.T


class BranchAndBoundSolver:
    """Branch-and-Bound solver for job scheduling with bin packing heuristic."""

    def __init__(self, instance: Instance, time_limit: float = 300, verbose: bool = False):
        self.instance = instance
        self.time_limit = time_limit
        self.verbose = verbose

        # Precompute prefix sums for O(1) interval cost
        self.prefix_costs = np.zeros(instance.T + 1)
        # energy_costs is (T,), prefix_costs[1:] stores cumsum
        self.prefix_costs[1:] = np.cumsum(instance.energy_costs)

        self.best_sequence: Optional[List[int]] = None
        self.best_cost = float("inf")
        self.nodes_explored = 0
        self.pruned_by_binpack = 0
        self.binpack_attempts = 0

    def _lpt_heuristic(self) -> List[int]:
        """Longest Processing Time first heuristic."""
        jobs = list(range(self.instance.n_jobs))
        jobs.sort(key=lambda j: self.instance.processing_times[j], reverse=True)
        return jobs

    def _spt_heuristic(self) -> List[int]:
        """Shortest Processing Time first heuristic."""
        jobs = list(range(self.instance.n_jobs))
        jobs.sort(key=lambda j: self.instance.processing_times[j])
        return jobs

    def solve(self) -> Tuple[List[int], float]:
        start_time = time.time()

        # Initialize with heuristics
        spt_seq = self._spt_heuristic()
        spt_cost = self._evaluate_sequence(spt_seq)

        lpt_seq = self._lpt_heuristic()
        lpt_cost = self._evaluate_sequence(lpt_seq)

        if spt_cost <= lpt_cost:
            self.best_cost = spt_cost
            self.best_sequence = spt_seq
            if self.verbose:
                print(f"Initial: SPT ({spt_cost:.2f})")
        else:
            self.best_cost = lpt_cost
            self.best_sequence = lpt_seq
            if self.verbose:
                print(f"Initial: LPT ({lpt_cost:.2f})")

        # Run DFS
        self._branch_and_bound_dfs([], set(range(self.instance.n_jobs)), start_time)

        if self.verbose:
            print(f"Details: Nodes={self.nodes_explored}, BP_Pruned={self.pruned_by_binpack}")
        
        return self.best_sequence, self.best_cost

    def _evaluate_sequence(self, sequence: List[int]) -> float:
        """Evaluate exact cost of a fixed sequence using DP."""
        if not sequence:
            return 0.0
        
        # We use a simplified single-sequence DP here or reuse _dp_evaluate_with_schedule
        # The internal _dp_evaluate_with_schedule is more general
        pts = self.instance.processing_times[sequence].tolist()
        cost, _ = self._dp_evaluate_with_schedule(pts)
        return cost

    def _branch_and_bound_dfs(
        self, partial_sequence: List[int], remaining_jobs: set, start_time: float
    ):
        if time.time() - start_time > self.time_limit:
            return

        self.nodes_explored += 1

        if not remaining_jobs:
            cost = self._evaluate_sequence(partial_sequence)
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_sequence = list(partial_sequence)
            return

        # Lower Bound + Blocks
        lb, blocks, _ = self._compute_lower_bound_with_blocks(
            partial_sequence, remaining_jobs
        )

        if lb >= self.best_cost:
            return  # Prune by bound

        # Bin Packing Heuristic
        if blocks and len(blocks) > 0:
            self.binpack_attempts += 1
            packed_jobs = self._try_bin_packing(remaining_jobs, blocks)

            if packed_jobs is not None:
                # Found valid packing that matches LB exactly
                candidate = partial_sequence + packed_jobs
                self.best_cost = lb
                self.best_sequence = candidate
                self.pruned_by_binpack += 1
                return  # Strong prune: we found optimal for this subtree

        # Branching
        # Symmetry breaking: only branch one job per unique processing time
        unique_pts = {}
        for j in remaining_jobs:
            pt = self.instance.processing_times[j]
            if pt not in unique_pts:
                unique_pts[pt] = j
        
        # Heuristic ordering: try largest processing times first? Or just sorted.
        # Standard implementation usually does lexical or based on heuristic.
        # Let's stick to sorted unique processing times for deterministic behavior.
        for pt in sorted(unique_pts.keys()):
            job = unique_pts[pt]
            self._branch_and_bound_dfs(
                partial_sequence + [job], remaining_jobs - {job}, start_time
            )
            
            # Alpha-beta style check
            if self.best_cost <= lb:
                break

    def _compute_lower_bound_with_blocks(
        self, partial_sequence: List[int], remaining_jobs: set
    ) -> Tuple[float, Optional[List[int]], List[int]]:
        """Compute relaxed LB by chopping remaining jobs into GCD pieces."""
        if not remaining_jobs:
            return self._evaluate_sequence(partial_sequence), None, []

        rem_pts = [self.instance.processing_times[j] for j in remaining_jobs]
        total_rem = sum(rem_pts)
        if total_rem == 0:
            return self._evaluate_sequence(partial_sequence), None, []
            
        gcd_val = np.gcd.reduce(rem_pts)
        if gcd_val == 0: gcd_val = 1
        
        n_relaxed = total_rem // gcd_val
        
        # Fixed part
        relaxed_pts = []
        if partial_sequence:
            relaxed_pts = self.instance.processing_times[partial_sequence].tolist()
        
        n_fixed = len(relaxed_pts)
        relaxed_pts.extend([gcd_val] * n_relaxed)
        
        # Solve relaxed
        lb, schedule = self._dp_evaluate_with_schedule(relaxed_pts)
        
        blocks = self._extract_blocks_from_schedule(
            schedule, relaxed_pts, n_fixed, gcd_val
        )
        
        return lb, blocks, relaxed_pts

    def _extract_blocks_from_schedule(
        self, schedule: List[int], processing_times: List[int], n_fixed: int, job_size: int
    ) -> List[int]:
        """Combine consecutive processing intervals of unfixed jobs into blocks."""
        if n_fixed >= len(schedule):
            return []
        
        intervals = []
        for i in range(n_fixed, len(schedule)):
            s = schedule[i]
            e = s + processing_times[i] # should be job_size
            intervals.append((s, e))
            
        if not intervals:
            return []
            
        intervals.sort()
        
        blocks = []
        curr_s, curr_e = intervals[0]
        
        for s, e in intervals[1:]:
            if s <= curr_e:
                curr_e = max(curr_e, e)
            else:
                blocks.append(curr_e - curr_s)
                curr_s, curr_e = s, e
        blocks.append(curr_e - curr_s)
        
        return blocks

    def _try_bin_packing(
        self, remaining_jobs: set, blocks: List[int]
    ) -> Optional[List[int]]:
        """Fit remaining jobs into blocks using First Fit Decreasing."""
        if not blocks or not remaining_jobs:
            return None
            
        jobs = sorted(list(remaining_jobs), 
                      key=lambda j: self.instance.processing_times[j], 
                      reverse=True)
                      
        # Quick check
        if self.instance.processing_times[jobs[0]] > max(blocks):
            return None
            
        bins = np.array(blocks, dtype=np.int32)
        bin_contents = [[] for _ in range(len(blocks))]
        
        for j in jobs:
            pt = self.instance.processing_times[j]
            
            # Vectorized first fit
            fits = (bins >= pt)
            if not fits.any():
                return None # Fail
            
            idx = np.argmax(fits)
            bins[idx] -= pt
            bin_contents[idx].append(j)
            
        # Success - flatten
        packed = []
        for content in bin_contents:
            packed.extend(content)
            
        return packed

    def _dp_evaluate_with_schedule(
        self, processing_times: List[int]
    ) -> Tuple[float, List[int]]:
        """
        DP evaluation returning (cost, schedule).
        
        Tracks start time for each job to reconstruct schedule.
        """
        J = len(processing_times)
        if J == 0:
            return 0.0, []
            
        T = self.instance.T
        PT = np.array(processing_times, dtype=np.int32)
        
        ES = np.zeros(J, dtype=np.int32)
        LS = np.zeros(J, dtype=np.int32)
        for i in range(J):
            ES[i] = np.sum(PT[:i])
            LS[i] = T - np.sum(PT[i:])
            
        # TEC[i, t] = cost of first i jobs with i-th job starting at t
        TEC = np.full((J + 1, T), np.inf)
        parent = np.full((J + 1, T), -1, dtype=np.int32)
        
        # Base case
        TEC[0, 0] = 0.0 # dummy job 0 ends at 0
        
        # DP
        for i in range(1, J + 1):
            job_idx = i - 1
            p = PT[job_idx]
            
            # If first job
            if i == 1:
                # valid starts are [ES, LS]
                # Also must fit in T: s+p <= T -> s <= T-p
                max_s = min(LS[job_idx], T - p)
                starts = np.arange(ES[job_idx], max_s + 1)
                
                if len(starts) > 0:
                    costs = self.prefix_costs[starts + p] - self.prefix_costs[starts]
                    TEC[i, starts] = costs
                    parent[i, starts] = 0 # came from dummy 0 at 0
                continue
            
            # General case
            prev_idx = job_idx - 1
            p_prev = PT[prev_idx]
            
            # Compute prefix min of previous layer
            # We need min_{r <= s - p_prev} TEC[i-1, r]
            # Let's be careful with indices. 
            # Previous job ends at `start_prev + p_prev`.
            # Current job starts at `s`.
            # Constraint: `start_prev + p_prev <= s`  => `start_prev <= s - p_prev`
            
            prev_layer_vals = TEC[i-1]
            
            # To optimize: accum_min[k] = min(TEC[i-1, :k+1])
            # Then best previous is accum_min[s - p_prev]
            # Handling infinity correctly
            
            # Only consider valid range of previous
            # But the array is size T, so we can just accum min
            
            accum_min = np.minimum.accumulate(prev_layer_vals)
            # We also need argmin for backtracking
            # NumPy doesn't have accumulate_argmin easily, but we can iterate
            # Since T is small (~500), iteration is fine or we custom build it
            
            # Let's recreate the loop logic from provided code for safety
            # But vectorized where possible
            
            # The provided code used explicit loops for correctness. 
            # I will trust the provided logic but wrap it slightly cleaner if possible.
            # Keeping the loop structure for safety as requested "full power".
            
            p_min_val = np.inf
            p_min_arg = -1
            
            # Precompute prefix min/argmin for relevant range
            # Range of prev starts: [ES_prev, LS_prev]
            # We can scan 0 to T for simplicity
            
            global_prefix_min = np.full(T + 1, np.inf)
            global_prefix_argmin = np.full(T + 1, -1, dtype=np.int32)
            
            curr = np.inf
            curr_arg = -1
            for t_prev in range(T):
                 if prev_layer_vals[t_prev] < curr:
                     curr = prev_layer_vals[t_prev]
                     curr_arg = t_prev
                 global_prefix_min[t_prev] = curr
                 global_prefix_argmin[t_prev] = curr_arg
                 
            # Compute current layer
            # Current starts s in [ES, LS]
            max_s = min(LS[job_idx], T - p)
            if max_s < ES[job_idx]:
                continue
                
            for s in range(ES[job_idx], max_s + 1):
                # Valid previous must end by s
                # start_prev <= s - p_prev
                limit = s - p_prev
                if limit < 0: continue
                limit = min(limit, T-1) # bounded by T
                
                best_prev_cost = global_prefix_min[limit]
                if best_prev_cost < np.inf:
                    cost_here = self.prefix_costs[s + p] - self.prefix_costs[s]
                    TEC[i, s] = best_prev_cost + cost_here
                    parent[i, s] = global_prefix_argmin[limit]

        # Backtrack
        if np.isinf(np.min(TEC[J])):
             return float("inf"), []
             
        best_end_cost = np.min(TEC[J])
        curr_t = np.argmin(TEC[J])
        
        schedule = [0] * J
        for i in range(J, 0, -1):
            schedule[i-1] = curr_t
            curr_t = parent[i, curr_t]
            
        return best_end_cost, schedule

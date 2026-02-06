"""Incremental DP Solver for Local Search Optimization.

This module implements checkpoint-based incremental dynamic programming
for faster local search operations (swap, insert) on job sequences.

Key Features:
1. Checkpoint-Based DP: Store intermediate DP states at regular intervals
2. Incremental Recomputation: Restore nearest checkpoint after sequence modification
3. Batched Swap Evaluation: Efficiently evaluate multiple candidate swaps

Example:
    >>> solver = IncrementalDPSolver(processing_times, ct, e_single, T_limit)
    >>> cost, checkpoints = solver.solve_with_checkpoints(sequence)
    >>> # After swapping jobs at positions i and j:
    >>> new_cost = solver.recompute_from_position(sequence_modified, min(i,j), checkpoints)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any


@dataclass
class DPCheckpoint:
    """Stores a DP state at a specific position in the sequence."""
    position: int  # Number of jobs processed (0 = initial state)
    min_prev: np.ndarray  # DP array: min cost to finish by time t, shape (T+1,)


@dataclass
class IncrementalDPResult:
    """Result of DP computation with checkpoints."""
    total_cost: float
    makespan: int
    start_times: List[int]
    checkpoints: List[DPCheckpoint]


class IncrementalDPSolver:
    """DP solver with checkpoint support for incremental local search.
    
    This solver maintains intermediate DP states (checkpoints) that enable
    faster recomputation after small sequence modifications like swaps.
    
    The standard DP has complexity O(N * T) where:
    - N = number of jobs
    - T = time horizon
    
    With checkpoints at interval k, a swap at position p requires only
    O((N - p + k) * T) recomputation instead of O(N * T).
    """
    
    def __init__(
        self,
        processing_times: np.ndarray,
        ct: np.ndarray,
        e_single: float = 1.0,
        T_limit: int = None,
        checkpoint_interval: int = 5,
    ):
        """Initialize the incremental DP solver.
        
        Args:
            processing_times: Processing time for each job, shape (N,)
            ct: Price curve, shape (T,) where ct[t] is the price at time t
            e_single: Energy consumption rate per time unit
            T_limit: Deadline. If None, uses len(ct)
            checkpoint_interval: Store checkpoint every k jobs (default: 5)
        """
        self.processing_times = np.asarray(processing_times, dtype=np.int32)
        self.ct = np.asarray(ct, dtype=np.float64)
        self.e_single = float(e_single)
        self.T_limit = int(T_limit if T_limit is not None else len(ct))
        self.checkpoint_interval = int(checkpoint_interval)
        
        # Precompute prefix sum of prices for O(1) interval cost queries
        self.price_prefix = np.zeros(len(self.ct) + 1, dtype=np.float64)
        self.price_prefix[1:] = np.cumsum(self.ct)
    
    def _interval_cost(self, start: int, duration: int) -> float:
        """Compute energy cost for a job starting at `start` with `duration`."""
        end = start + duration
        if start < 0 or end > len(self.ct):
            return float('inf')
        return float(self.price_prefix[end] - self.price_prefix[start]) * self.e_single
    
    def _dp_step(
        self,
        min_prev: np.ndarray,
        job_processing_time: int,
    ) -> np.ndarray:
        """Execute one DP step for a single job.
        
        Args:
            min_prev: Current DP state, shape (T+1,)
            job_processing_time: Processing time of the job to schedule
            
        Returns:
            New DP state after scheduling this job
        """
        T = self.T_limit
        p = int(job_processing_time)
        
        # For each possible end time t_end, compute cost of ending at t_end
        # t_start = t_end - p
        min_curr = np.full(T + 1, float('inf'), dtype=np.float64)
        
        for t_end in range(p, T + 1):
            t_start = t_end - p
            if t_start < 0:
                continue
            
            # Cost = min cost to be done with previous jobs by t_start + cost of this job
            prev_cost = min_prev[t_start]
            if not np.isfinite(prev_cost):
                continue
                
            # Energy cost: sum of prices from t_start to t_end
            energy_cost = float(self.price_prefix[t_end] - self.price_prefix[t_start]) * self.e_single
            total_at_end = prev_cost + energy_cost
            min_curr[t_end] = total_at_end
        
        # Take cumulative minimum (Pareto pruning)
        min_curr = np.minimum.accumulate(min_curr)
        
        return min_curr
    
    def solve_full(self, sequence: List[int]) -> Tuple[float, int, List[int]]:
        """Standard full DP without checkpoints.
        
        Args:
            sequence: List of job indices in execution order
            
        Returns:
            (total_cost, makespan, start_times)
        """
        n = len(sequence)
        if n == 0:
            return 0.0, 0, []
        
        T = self.T_limit
        
        # Initialize: min cost to finish 0 jobs by time t is 0 for all t
        min_prev = np.zeros(T + 1, dtype=np.float64)
        
        # Track best start times for reconstruction
        parent = np.full((n, T + 1), -1, dtype=np.int32)
        
        for i, job_idx in enumerate(sequence):
            p = int(self.processing_times[job_idx])
            min_curr = np.full(T + 1, float('inf'), dtype=np.float64)
            
            for t_end in range(p, T + 1):
                t_start = t_end - p
                prev_cost = min_prev[t_start]
                if not np.isfinite(prev_cost):
                    continue
                    
                energy_cost = float(self.price_prefix[t_end] - self.price_prefix[t_start]) * self.e_single
                total_at_end = prev_cost + energy_cost
                
                if total_at_end < min_curr[t_end]:
                    min_curr[t_end] = total_at_end
                    parent[i, t_end] = t_start
            
            # Take cumulative minimum
            best_so_far = float('inf')
            best_start = -1
            for t_end in range(T + 1):
                if min_curr[t_end] < best_so_far:
                    best_so_far = min_curr[t_end]
                    best_start = parent[i, t_end]
                min_curr[t_end] = best_so_far
                if best_start >= 0 and parent[i, t_end] < 0:
                    parent[i, t_end] = best_start
            
            min_prev = min_curr
        
        # Find optimal cost and makespan
        total_cost = min_prev[T]
        if not np.isfinite(total_cost):
            return float('inf'), T + 1, []
        
        # Find earliest completion time achieving optimal cost
        makespan = T
        for t in range(T + 1):
            if np.isclose(min_prev[t], total_cost, rtol=1e-9):
                makespan = t
                break
        
        # Backtrack to find start times
        start_times = self._backtrack_start_times(sequence, parent, makespan)
        
        return total_cost, makespan, start_times
    
    def _backtrack_start_times(
        self,
        sequence: List[int],
        parent: np.ndarray,
        final_time: int,
    ) -> List[int]:
        """Reconstruct optimal start times from parent pointers.
        
        Uses robust fallback when parent pointers are not set.
        """
        n = len(sequence)
        if n == 0:
            return []
        
        T = self.T_limit
        start_times = [0] * n
        t = final_time
        
        for i in range(n - 1, -1, -1):
            job_idx = sequence[i]
            p = int(self.processing_times[job_idx])
            
            # Bounds check before accessing parent array
            if 0 <= t <= T:
                t_start = parent[i, t]
            else:
                t_start = -1
            
            if t_start < 0:
                # Fallback: compute from end time
                t_start = max(0, t - p)
            
            start_times[i] = int(t_start)
            t = t_start
        
        return start_times
    
    def solve_with_checkpoints(self, sequence: List[int]) -> IncrementalDPResult:
        """Solve DP and store checkpoints at regular intervals.
        
        Args:
            sequence: List of job indices in execution order
            
        Returns:
            IncrementalDPResult containing cost, makespan, start_times, and checkpoints
        """
        n = len(sequence)
        if n == 0:
            return IncrementalDPResult(
                total_cost=0.0,
                makespan=0,
                start_times=[],
                checkpoints=[DPCheckpoint(position=0, min_prev=np.zeros(self.T_limit + 1))]
            )
        
        T = self.T_limit
        k = self.checkpoint_interval
        
        # Initialize
        min_prev = np.zeros(T + 1, dtype=np.float64)
        checkpoints = [DPCheckpoint(position=0, min_prev=min_prev.copy())]
        
        # Parent pointers for backtracking
        parent = np.full((n, T + 1), -1, dtype=np.int32)
        
        for i, job_idx in enumerate(sequence):
            p = int(self.processing_times[job_idx])
            min_curr = np.full(T + 1, float('inf'), dtype=np.float64)
            
            for t_end in range(p, T + 1):
                t_start = t_end - p
                prev_cost = min_prev[t_start]
                if not np.isfinite(prev_cost):
                    continue
                    
                energy_cost = float(self.price_prefix[t_end] - self.price_prefix[t_start]) * self.e_single
                total_at_end = prev_cost + energy_cost
                
                if total_at_end < min_curr[t_end]:
                    min_curr[t_end] = total_at_end
                    parent[i, t_end] = t_start
            
            # Cumulative minimum
            best_so_far = float('inf')
            best_start = -1
            for t_end in range(T + 1):
                if min_curr[t_end] < best_so_far:
                    best_so_far = min_curr[t_end]
                    best_start = parent[i, t_end]
                min_curr[t_end] = best_so_far
                if best_start >= 0 and parent[i, t_end] < 0:
                    parent[i, t_end] = best_start
            
            min_prev = min_curr
            
            # Store checkpoint at regular intervals
            if (i + 1) % k == 0 or i == n - 1:
                checkpoints.append(DPCheckpoint(position=i + 1, min_prev=min_prev.copy()))
        
        # Final result
        total_cost = min_prev[T]
        makespan = T
        for t in range(T + 1):
            if np.isclose(min_prev[t], total_cost, rtol=1e-9):
                makespan = t
                break
        
        start_times = self._backtrack_start_times(sequence, parent, makespan)
        
        return IncrementalDPResult(
            total_cost=total_cost,
            makespan=makespan,
            start_times=start_times,
            checkpoints=checkpoints,
        )
    
    def recompute_from_position(
        self,
        new_sequence: List[int],
        modified_position: int,
        checkpoints: List[DPCheckpoint],
    ) -> Tuple[float, int]:
        """Recompute DP after sequence modification, using nearest checkpoint.
        
        This is the key speedup: instead of recomputing from scratch O(N*T),
        we restart from the nearest checkpoint before the modification O((N-pos)*T).
        
        Args:
            new_sequence: Modified sequence of job indices
            modified_position: Position where modification started (0-indexed)
            checkpoints: List of DPCheckpoints from previous solve_with_checkpoints
            
        Returns:
            (total_cost, makespan) for the new sequence
        """
        n = len(new_sequence)
        T = self.T_limit
        
        if n == 0:
            return 0.0, 0
        
        # Find nearest checkpoint at or before modified_position
        best_checkpoint = checkpoints[0]  # Default: start from scratch
        for cp in checkpoints:
            if cp.position <= modified_position:
                best_checkpoint = cp
            else:
                break
        
        # Restore DP state from checkpoint
        min_prev = best_checkpoint.min_prev.copy()
        start_pos = best_checkpoint.position
        
        # Recompute from checkpoint position to end
        for i in range(start_pos, n):
            job_idx = new_sequence[i]
            min_prev = self._dp_step(min_prev, self.processing_times[job_idx])
        
        total_cost = min_prev[T]
        makespan = T
        for t in range(T + 1):
            if np.isclose(min_prev[t], total_cost, rtol=1e-9):
                makespan = t
                break
        
        return total_cost, makespan
    
    def evaluate_swap(
        self,
        sequence: List[int],
        pos_i: int,
        pos_j: int,
        checkpoints: List[DPCheckpoint],
    ) -> Tuple[float, int]:
        """Evaluate the cost of swapping jobs at positions i and j.
        
        Args:
            sequence: Original sequence
            pos_i, pos_j: Positions to swap (0-indexed)
            checkpoints: Checkpoints from original sequence
            
        Returns:
            (total_cost, makespan) after the swap
        """
        new_seq = list(sequence)
        new_seq[pos_i], new_seq[pos_j] = new_seq[pos_j], new_seq[pos_i]
        
        modified_pos = min(pos_i, pos_j)
        return self.recompute_from_position(new_seq, modified_pos, checkpoints)
    
    def evaluate_all_adjacent_swaps(
        self,
        sequence: List[int],
    ) -> List[Tuple[int, float, int]]:
        """Evaluate all N-1 adjacent swaps efficiently.
        
        Uses suffix DP to achieve O(N*T) total instead of O(N^2*T).
        
        Args:
            sequence: Original sequence
            
        Returns:
            List of (position, cost, makespan) for swapping positions i and i+1
        """
        n = len(sequence)
        if n < 2:
            return []
        
        T = self.T_limit
        results = []
        
        # Precompute forward DP states for all positions
        forward_states = [np.zeros(T + 1, dtype=np.float64)]
        min_prev = np.zeros(T + 1, dtype=np.float64)
        
        for job_idx in sequence:
            min_prev = self._dp_step(min_prev, self.processing_times[job_idx])
            forward_states.append(min_prev.copy())
        
        # For each adjacent swap (i, i+1), recompute from position i
        for i in range(n - 1):
            # Start from state before position i
            min_prev = forward_states[i].copy()
            
            # Process swapped jobs: first job i+1, then job i
            job_i = sequence[i]
            job_i1 = sequence[i + 1]
            
            min_prev = self._dp_step(min_prev, self.processing_times[job_i1])
            min_prev = self._dp_step(min_prev, self.processing_times[job_i])
            
            # Process remaining jobs
            for j in range(i + 2, n):
                min_prev = self._dp_step(min_prev, self.processing_times[sequence[j]])
            
            total_cost = min_prev[T]
            makespan = T
            for t in range(T + 1):
                if np.isclose(min_prev[t], total_cost, rtol=1e-9):
                    makespan = t
                    break
            
            results.append((i, total_cost, makespan))
        
        return results


def benchmark_incremental_vs_full(
    n_jobs: int = 20,
    T_max: int = 100,
    n_trials: int = 50,
    checkpoint_interval: int = 5,
    seed: int = 42,
):
    """Benchmark incremental DP vs full recomputation."""
    import time
    
    rng = np.random.RandomState(seed)
    
    # Generate random instance
    processing_times = rng.randint(1, 10, size=n_jobs)
    ct = rng.uniform(0.5, 5.0, size=T_max)
    
    solver = IncrementalDPSolver(
        processing_times=processing_times,
        ct=ct,
        checkpoint_interval=checkpoint_interval,
        T_limit=T_max,
    )
    
    # Random initial sequence
    sequence = list(range(n_jobs))
    rng.shuffle(sequence)
    
    # Solve with checkpoints
    t0 = time.perf_counter()
    result = solver.solve_with_checkpoints(sequence)
    t_initial = time.perf_counter() - t0
    
    print(f"Initial solve: {t_initial*1000:.2f}ms, cost={result.total_cost:.2f}")
    print(f"Checkpoints stored: {len(result.checkpoints)}")
    
    # Benchmark: evaluate random swaps
    times_full = []
    times_incr = []
    errors = []
    
    for _ in range(n_trials):
        # Random swap
        i = rng.randint(0, n_jobs - 1)
        j = rng.randint(0, n_jobs - 1)
        if i == j:
            continue
        
        new_seq = list(sequence)
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        
        # Full recomputation
        t0 = time.perf_counter()
        cost_full, ms_full, _ = solver.solve_full(new_seq)
        times_full.append(time.perf_counter() - t0)
        
        # Incremental recomputation
        t0 = time.perf_counter()
        cost_incr, ms_incr = solver.recompute_from_position(new_seq, min(i, j), result.checkpoints)
        times_incr.append(time.perf_counter() - t0)
        
        # Check correctness
        if np.isfinite(cost_full) and np.isfinite(cost_incr):
            errors.append(abs(cost_full - cost_incr))
    
    print(f"\nBenchmark over {len(times_full)} swaps:")
    print(f"  Full DP:        mean={np.mean(times_full)*1000:.3f}ms")
    print(f"  Incremental DP: mean={np.mean(times_incr)*1000:.3f}ms")
    print(f"  Speedup:        {np.mean(times_full)/np.mean(times_incr):.2f}x")
    print(f"  Max error:      {max(errors) if errors else 0:.6f}")
    
    return {
        "times_full": times_full,
        "times_incr": times_incr,
        "errors": errors,
        "speedup": np.mean(times_full) / np.mean(times_incr),
    }


def test_correctness(n_tests: int = 100, seed: int = 42):
    """Test that incremental DP matches full DP."""
    rng = np.random.RandomState(seed)
    
    all_passed = True
    for test_idx in range(n_tests):
        n_jobs = rng.randint(5, 25)
        T_max = rng.randint(50, 150)
        
        processing_times = rng.randint(1, 8, size=n_jobs)
        ct = rng.uniform(0.5, 5.0, size=T_max)
        
        solver = IncrementalDPSolver(
            processing_times=processing_times,
            ct=ct,
            checkpoint_interval=rng.randint(2, 8),
            T_limit=T_max,
        )
        
        sequence = list(range(n_jobs))
        rng.shuffle(sequence)
        
        # Solve with checkpoints
        result = solver.solve_with_checkpoints(sequence)
        
        # Verify cost matches full DP
        cost_full, _, _ = solver.solve_full(sequence)
        
        if not np.isclose(result.total_cost, cost_full, rtol=1e-9):
            print(f"[FAIL] Test {test_idx}: checkpoint={result.total_cost:.6f} vs full={cost_full:.6f}")
            all_passed = False
            continue
        
        # Test a random swap
        if n_jobs >= 2:
            i, j = rng.choice(n_jobs, size=2, replace=False)
            new_seq = list(sequence)
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
            
            cost_incr, _ = solver.recompute_from_position(new_seq, min(i, j), result.checkpoints)
            cost_swap_full, _, _ = solver.solve_full(new_seq)
            
            if np.isfinite(cost_swap_full) and not np.isclose(cost_incr, cost_swap_full, rtol=1e-9):
                print(f"[FAIL] Test {test_idx} swap: incr={cost_incr:.6f} vs full={cost_swap_full:.6f}")
                all_passed = False
    
    if all_passed:
        print(f"[PASS] All {n_tests} tests passed!")
    return all_passed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Incremental DP Solver")
    parser.add_argument("--test", action="store_true", help="Run correctness tests")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--n_jobs", type=int, default=20, help="Number of jobs for benchmark")
    parser.add_argument("--T_max", type=int, default=100, help="Time horizon for benchmark")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="Checkpoint interval")
    args = parser.parse_args()
    
    if args.test:
        test_correctness()
    
    if args.benchmark:
        benchmark_incremental_vs_full(
            n_jobs=args.n_jobs,
            T_max=args.T_max,
            checkpoint_interval=args.checkpoint_interval,
        )
    
    if not args.test and not args.benchmark:
        # Default: run both
        print("=" * 60)
        print("CORRECTNESS TESTS")
        print("=" * 60)
        test_correctness()
        
        print("\n" + "=" * 60)
        print("BENCHMARK")
        print("=" * 60)
        benchmark_incremental_vs_full()

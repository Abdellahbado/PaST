"""
Cheapest-Window Placement (CWP) Heuristic for Parallel Machine Scheduling.

This module implements a constructive scheduler that assigns jobs to machines
by placing them in the cheapest available TOU (Time-of-Use) windows.

The algorithm:
1. Precomputes window costs b[d][t] = sum(ct[t:t+d]) for each job duration d
2. Sorts jobs by LPT (longest processing time first)
3. For each job, finds the best (machine, start_time) that minimizes e[i] * b[d][t]
4. Places the job and updates machine availability

Reference: Roberto Ronco's exact and heuristic methods for parallel machine scheduling
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class CWPResult:
    """Result of CWP heuristic."""
    
    # Assignment: job index -> machine index
    assignment: Dict[int, int]
    
    # Start times: job index -> start slot
    start_times: Dict[int, int]
    
    # Total energy cost
    total_cost: float
    
    # Per-machine costs
    machine_costs: List[float]
    
    # Jobs per machine: machine index -> list of (job_idx, start_time)
    machine_schedules: Dict[int, List[Tuple[int, int]]]
    
    # Whether all jobs were successfully placed
    feasible: bool
    
    # Makespan (max completion time)
    makespan: int


class MachineState:
    """Tracks machine availability using a boolean occupancy array."""
    
    def __init__(self, T: int):
        """Initialize machine state.
        
        Args:
            T: Total time horizon (number of slots)
        """
        self.T = T
        self.occupied = np.zeros(T, dtype=bool)
    
    def is_window_free(self, start: int, duration: int) -> bool:
        """Check if window [start, start+duration) is free.
        
        Args:
            start: Start slot
            duration: Job duration
            
        Returns:
            True if all slots in the window are free
        """
        if start < 0 or start + duration > self.T:
            return False
        return not np.any(self.occupied[start:start + duration])
    
    def place(self, start: int, duration: int) -> None:
        """Mark window [start, start+duration) as occupied.
        
        Args:
            start: Start slot
            duration: Job duration
        """
        self.occupied[start:start + duration] = True
    
    def get_free_windows(self, duration: int, max_end: Optional[int] = None) -> List[int]:
        """Get all feasible start positions for a job of given duration.
        
        Args:
            duration: Job duration
            max_end: Maximum end time (for deadline constraint)
            
        Returns:
            List of feasible start positions
        """
        if max_end is None:
            max_end = self.T
        
        max_start = max_end - duration
        if max_start < 0:
            return []
        
        feasible_starts = []
        for t in range(max_start + 1):
            if self.is_window_free(t, duration):
                feasible_starts.append(t)
        
        return feasible_starts


def compute_window_costs(
    ct: np.ndarray,
    durations: List[int],
) -> Dict[int, np.ndarray]:
    """Precompute window costs b[d][t] = sum(ct[t:t+d]) for each duration.
    
    Uses prefix sums for O(1) per-window computation.
    
    Args:
        ct: Per-slot prices, shape (T,)
        durations: List of unique job durations
        
    Returns:
        Dict mapping duration d to array of window costs b[d][t]
    """
    T = len(ct)
    
    # Compute prefix sums: pref[t] = sum(ct[0:t])
    pref = np.zeros(T + 1, dtype=np.float64)
    pref[1:] = np.cumsum(ct)
    
    window_costs = {}
    for d in durations:
        if d <= 0 or d > T:
            continue
        # b[d][t] = pref[t+d] - pref[t] for t in [0, T-d]
        max_start = T - d
        b = np.zeros(max_start + 1, dtype=np.float64)
        for t in range(max_start + 1):
            b[t] = pref[t + d] - pref[t]
        window_costs[d] = b
    
    return window_costs


def compute_sorted_starts_by_cost(
    window_costs: Dict[int, np.ndarray],
) -> Dict[int, np.ndarray]:
    """For each duration, sort start positions by window cost (ascending).
    
    This allows fast search for the cheapest available window.
    
    Args:
        window_costs: Dict from compute_window_costs()
        
    Returns:
        Dict mapping duration d to array of start positions sorted by cost
    """
    sorted_starts = {}
    for d, b in window_costs.items():
        # Get indices that would sort b
        sorted_starts[d] = np.argsort(b)
    return sorted_starts


def find_best_slot(
    duration: int,
    e_i: float,
    machine_state: MachineState,
    window_costs: Dict[int, np.ndarray],
    sorted_starts: Optional[Dict[int, np.ndarray]] = None,
    epsilon: Optional[int] = None,
    top_k: Optional[int] = None,
) -> Tuple[Optional[int], float]:
    """Find the cheapest feasible window on a machine for a job.
    
    Args:
        duration: Job duration
        e_i: Machine energy rate
        machine_state: Current machine availability
        window_costs: Precomputed window costs
        sorted_starts: Optional sorted start positions for fast search
        epsilon: Optional deadline constraint (job must finish by epsilon)
        top_k: Optional limit on how many windows to check
        
    Returns:
        Tuple of (best_start, cost) or (None, inf) if no feasible window
    """
    if duration not in window_costs:
        return None, float('inf')
    
    b = window_costs[duration]
    max_end = epsilon if epsilon is not None else machine_state.T
    max_start = max_end - duration
    
    if max_start < 0:
        return None, float('inf')
    
    best_start = None
    best_cost = float('inf')
    
    if sorted_starts is not None and duration in sorted_starts:
        # Use sorted order for fast search
        starts_to_check = sorted_starts[duration]
        if top_k is not None:
            starts_to_check = starts_to_check[:top_k]
        
        for t in starts_to_check:
            if t > max_start:
                continue
            if machine_state.is_window_free(t, duration):
                cost = e_i * b[t]
                if cost < best_cost:
                    best_cost = cost
                    best_start = t
                    # Since sorted by cost, first feasible is best
                    break
    else:
        # Exhaustive search
        for t in range(max_start + 1):
            if machine_state.is_window_free(t, duration):
                cost = e_i * b[t]
                if cost < best_cost:
                    best_cost = cost
                    best_start = t
    
    return best_start, best_cost


def compute_tou_spread(
    duration: int,
    window_costs: Dict[int, np.ndarray],
) -> float:
    """Compute TOU spread for a duration: max(b[d]) - min(b[d]).
    
    Jobs with higher spread benefit more from careful placement.
    
    Args:
        duration: Job duration
        window_costs: Precomputed window costs
        
    Returns:
        TOU spread value
    """
    if duration not in window_costs:
        return 0.0
    b = window_costs[duration]
    return float(np.max(b) - np.min(b))


def construct_schedule_CWP(
    processing_times: List[int],
    energy_rates: List[float],
    ct: np.ndarray,
    epsilon: Optional[int] = None,
    top_k: Optional[int] = 80,
    seed: Optional[int] = None,
) -> CWPResult:
    """Construct a schedule using Cheapest-Window Placement heuristic.
    
    Args:
        processing_times: List of job processing times, length n
        energy_rates: List of machine energy rates, length m
        ct: Per-slot prices, shape (T,)
        epsilon: Optional makespan limit (deadline constraint)
        top_k: Optional limit on windows to check per placement (speedup)
        seed: Optional random seed for tie-breaking
        
    Returns:
        CWPResult with assignment, start times, and costs
    """
    n = len(processing_times)
    m = len(energy_rates)
    T = len(ct)
    
    if n == 0:
        return CWPResult(
            assignment={},
            start_times={},
            total_cost=0.0,
            machine_costs=[0.0] * m,
            machine_schedules={i: [] for i in range(m)},
            feasible=True,
            makespan=0,
        )
    
    # Set deadline to T if not specified
    if epsilon is None:
        epsilon = T
    
    # Precompute window costs
    unique_durations = list(set(processing_times))
    window_costs = compute_window_costs(ct, unique_durations)
    sorted_starts = compute_sorted_starts_by_cost(window_costs)
    
    # Compute TOU spreads for tie-breaking
    tou_spreads = {d: compute_tou_spread(d, window_costs) for d in unique_durations}
    
    # Sort jobs by LPT, tie-break by TOU spread (higher first)
    job_indices = list(range(n))
    job_indices.sort(
        key=lambda j: (-processing_times[j], -tou_spreads.get(processing_times[j], 0)),
    )
    
    # Initialize machine states
    machine_states = [MachineState(T) for _ in range(m)]
    
    # Assignment results
    assignment = {}
    start_times = {}
    machine_costs = [0.0] * m
    machine_schedules = {i: [] for i in range(m)}
    feasible = True
    makespan = 0
    
    # Place each job
    for j in job_indices:
        d = processing_times[j]
        
        # Find best (machine, start) pair
        best_machine = None
        best_start = None
        best_cost = float('inf')
        
        for i in range(m):
            start, cost = find_best_slot(
                duration=d,
                e_i=energy_rates[i],
                machine_state=machine_states[i],
                window_costs=window_costs,
                sorted_starts=sorted_starts,
                epsilon=epsilon,
                top_k=top_k,
            )
            if start is not None and cost < best_cost:
                best_cost = cost
                best_start = start
                best_machine = i
        
        if best_machine is None:
            # No feasible placement found
            feasible = False
            # Try to place without deadline constraint as fallback
            for i in range(m):
                start, cost = find_best_slot(
                    duration=d,
                    e_i=energy_rates[i],
                    machine_state=machine_states[i],
                    window_costs=window_costs,
                    sorted_starts=sorted_starts,
                    epsilon=None,  # Remove deadline
                    top_k=top_k,
                )
                if start is not None and cost < best_cost:
                    best_cost = cost
                    best_start = start
                    best_machine = i
        
        if best_machine is not None:
            # Place the job
            machine_states[best_machine].place(best_start, d)
            assignment[j] = best_machine
            start_times[j] = best_start
            machine_costs[best_machine] += best_cost
            machine_schedules[best_machine].append((j, best_start))
            makespan = max(makespan, best_start + d)
        else:
            # Failed to place even without deadline
            feasible = False
            assignment[j] = -1
            start_times[j] = -1
    
    total_cost = sum(machine_costs)
    
    return CWPResult(
        assignment=assignment,
        start_times=start_times,
        total_cost=total_cost,
        machine_costs=machine_costs,
        machine_schedules=machine_schedules,
        feasible=feasible,
        makespan=makespan,
    )


def get_machine_job_sets(result: CWPResult, n_machines: int) -> List[List[int]]:
    """Extract job sets per machine from CWP result.
    
    Args:
        result: CWPResult from construct_schedule_CWP
        n_machines: Number of machines
        
    Returns:
        List of job index lists, one per machine
    """
    job_sets = [[] for _ in range(n_machines)]
    for job_idx, machine_idx in result.assignment.items():
        if machine_idx >= 0:
            job_sets[machine_idx].append(job_idx)
    return job_sets


if __name__ == "__main__":
    # Quick sanity test
    print("Testing CWP Solver...")
    
    # Simple test case
    processing_times = [3, 2, 4, 1, 2]
    energy_rates = [1.0, 1.5, 2.0]
    ct = np.array([1, 1, 2, 2, 3, 3, 1, 1, 2, 2], dtype=np.float64)
    
    result = construct_schedule_CWP(
        processing_times=processing_times,
        energy_rates=energy_rates,
        ct=ct,
        epsilon=None,
    )
    
    print(f"Feasible: {result.feasible}")
    print(f"Total cost: {result.total_cost:.2f}")
    print(f"Makespan: {result.makespan}")
    print(f"Assignment: {result.assignment}")
    print(f"Start times: {result.start_times}")
    print(f"Machine costs: {result.machine_costs}")
    
    job_sets = get_machine_job_sets(result, 3)
    for i, jobs in enumerate(job_sets):
        print(f"Machine {i}: jobs {jobs}")
    
    print("\nCWP Solver test passed!")

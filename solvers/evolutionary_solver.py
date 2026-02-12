"""
Evolutionary Algorithm for Parallel Machine TOU Scheduling.

Problem: Pm,TOU || TEC (minimize total energy cost)

Representation:
- Genotype: Assignment vector (job_idx -> machine_idx)
- Phenotype: Schedule derived from sequencing (SGBS) + fixed-seq DP

Evaluation:
- Fast (Inner Loop): Random-Q SGBS + Fixed-Seq DP
- Exact (Refinement): Optimal Sparse DP (Numba)
"""

from __future__ import annotations

import copy
import time
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from PaST.solvers.cwp_solver import (
    construct_schedule_CWP,
    get_machine_job_sets,
)
from PaST.solvers.baselines_sequence_dp import (
    _dp_schedule_fixed_order,
)
from PaST.solvers.dp_fast import (
    dp_cost_fixed_order_rolling_precomputed,
)
from PaST.solvers.optimal_benchmark_dp import (
    solve_optimal_benchmark_dp,
)

# =============================================================================
# Inlined dependencies from VND
# =============================================================================


@dataclass
class MachineState:
    """State of a single machine in the solution."""

    machine_idx: int
    job_indices: List[int]  # Global job indices assigned to this machine
    sequence: List[int]  # Order of global job indices (a permutation of job_indices)
    start_times: List[int]  # Optimal start times from DP
    energy: float  # Total energy cost for this machine
    makespan: int  # Completion time of last job on this machine

    def copy(self) -> "MachineState":
        return MachineState(
            machine_idx=self.machine_idx,
            job_indices=list(self.job_indices),
            sequence=list(self.sequence),
            start_times=list(self.start_times),
            energy=self.energy,
            makespan=self.makespan,
        )


@dataclass
class Solution:
    """Complete solution for the parallel machine scheduling problem."""

    machines: List[MachineState]  # One per machine
    total_energy: float
    makespan: int
    feasible: bool

    def copy(self) -> "Solution":
        return Solution(
            machines=[m.copy() for m in self.machines],
            total_energy=self.total_energy,
            makespan=self.makespan,
            feasible=self.feasible,
        )


def evaluate_machine(
    job_indices: List[int],
    sequence: List[int],
    processing_times: List[int],
    ct: np.ndarray,
    e_rate: int,
    T_limit: int,
) -> Tuple[float, List[int], int]:
    """Evaluate a machine's sequence using fixed-order DP."""
    if not sequence:
        return 0.0, [], 0

    proc = [int(processing_times[j]) for j in sequence]
    energy, start_times = _dp_schedule_fixed_order(
        processing_times=proc,
        ct=ct,
        e_single=e_rate,
        T_limit=T_limit,
    )

    if start_times and np.isfinite(energy):
        last_p = proc[-1]
        makespan = start_times[-1] + last_p
    else:
        makespan = T_limit + 1  # Infeasible marker

    return energy, start_times, makespan


def random_q_sgbs_sequence(
    job_indices: List[int],
    processing_times: List[int],
    ct: np.ndarray,
    e_rate: int,
    T_limit: int,
    beta: int = 4,
    gamma: int = 4,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[int], float, List[int]]:
    """Generate a job sequence using SGBS with random Q-values."""
    if rng is None:
        rng = np.random.default_rng()

    n = len(job_indices)
    if n == 0:
        return [], 0.0, []
    if n == 1:
        energy, st, _ = evaluate_machine(
            job_indices, job_indices, processing_times, ct, e_rate, T_limit
        )
        return list(job_indices), energy, st

    jobs = list(job_indices)

    # Precompute prefix sums once; SGBS evaluates many sequences.
    T = int(T_limit)
    ct_i32 = np.asarray(ct, dtype=np.int32)
    prefix_costs = np.zeros(T + 1, dtype=np.float64)
    if T > 0:
        prefix_costs[1:] = np.cumsum(ct_i32[:T], dtype=np.float64)

    # Beam: list of partial sequences (each is a list of global job indices)
    beam: List[List[int]] = [[]]

    for step in range(n):
        candidates = []

        for partial in beam:
            remaining = [j for j in jobs if j not in set(partial)]
            if not remaining:
                candidates.append((partial, 0.0))
                continue

            # Random Q-values for remaining jobs
            q_vals = rng.random(len(remaining))

            # Top-gamma by random Q (lower = "better")
            indexed = list(zip(remaining, q_vals))
            indexed.sort(key=lambda x: x[1])

            for j, q in indexed[:gamma]:
                candidates.append((partial + [j], q))

        if not candidates:
            break

        # Simulation: complete each candidate greedily with random Q, evaluate with DP
        scored = []
        for cand, _ in candidates:
            remaining = [j for j in jobs if j not in set(cand)]

            # Greedy completion with random Q
            full_seq = list(cand)
            rem_list = list(remaining)
            if rem_list:
                rem_q = rng.random(len(rem_list))
                order = sorted(range(len(rem_list)), key=lambda i: rem_q[i])
                full_seq.extend(rem_list[i] for i in order)

            # Evaluate with faster cost-only DP (no parent pointers, prefix reused)
            proc = [int(processing_times[int(j)]) for j in full_seq]
            energy, _best_last_start = dp_cost_fixed_order_rolling_precomputed(
                processing_times=proc,
                prefix_costs=prefix_costs,
                e_single=int(e_rate),
                T_limit=T,
            )
            scored.append((cand, energy))

        # Pruning: keep top-beta by energy
        scored.sort(key=lambda x: x[1])
        beam = [c for c, _ in scored[:beta]]

    # Final: best beam entry should be complete
    best_seq = beam[0]
    if len(best_seq) < n:
        missing = [j for j in jobs if j not in set(best_seq)]
        best_seq = best_seq + missing

    energy, start_times, makespan = evaluate_machine(
        job_indices, best_seq, processing_times, ct, e_rate, T_limit
    )
    return best_seq, energy, start_times


@dataclass
class EAConfig:
    """Configuration for the Evolutionary Algorithm."""

    pop_size: int = 50
    generations: int = 100
    crossover_prob: float = 0.8
    mutation_prob: float = 0.1
    mutation_reassign_prob: float = 0.05  # Probability per gene
    tournament_size: int = 3

    # Fast Eval (SGBS) settings
    sgbs_beta: int = 2
    sgbs_gamma: int = 2

    # Refinement settings
    refine_top_k: int = 1  # Number of best solutions to refine exactly
    exact_time_limit: float = 5.0  # Time limit per machine for exact DP


@dataclass
class Individual:
    """An individual in the population."""

    assignment: List[int]  # job_idx -> machine_idx
    fitness: float = float("inf")
    makespan: int = 0
    is_exact: bool = False  # True if fitness is from Exact DP
    is_cwp: bool = False  # True if from CWP initialization

    # Cached phenotype (optional, for result reconstruction)
    machine_costs: Optional[List[float]] = None

    def copy(self) -> "Individual":
        ind = Individual(
            assignment=list(self.assignment),
            fitness=self.fitness,
            makespan=self.makespan,
            is_exact=self.is_exact,
            is_cwp=self.is_cwp,
        )
        if self.machine_costs:
            ind.machine_costs = list(self.machine_costs)
        return ind


class GeneticSolver:
    def __init__(
        self,
        processing_times: List[int],
        energy_rates: List[float],
        ct: np.ndarray,
        T_limit: int,
        n_machines: int,
        config: EAConfig,
        seed: int = 42,
    ):
        self.p = processing_times
        self.e_rates = [float(e) for e in energy_rates]
        self.ct = ct
        self.T_limit = T_limit
        self.m = n_machines
        self.n = len(processing_times)
        self.config = config
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # Precompute prefix sums for exact DP
        self.prefix_ct = np.zeros(len(ct) + 1, dtype=np.float64)
        self.prefix_ct[1:] = np.cumsum(ct)

        # Cache exact single-machine DP results by multiset of durations.
        # Key: ((p1,c1), (p2,c2), ...) sorted by p.
        # Value: (feasible, base_cost_for_e_rate_1, finish_time)
        self._exact_dp_cache: Dict[
            Tuple[Tuple[int, int], ...], Tuple[bool, float, int]
        ] = {}

        # Prices vector used by the exact multiset DP (length must be T_limit).
        self._prices = np.asarray(ct[: int(self.T_limit)], dtype=np.float64)

    def initialize_population(self) -> List[Individual]:
        """Initialize population with 1 CWP seed + random assignments."""
        pop = []

        # 1. CWP Seed
        cwp_res = construct_schedule_CWP(
            self.p, self.e_rates, self.ct, epsilon=self.T_limit, top_k=80
        )
        cwp_assign = [0] * self.n
        for j, m_idx in cwp_res.assignment.items():
            if m_idx >= 0:
                cwp_assign[j] = m_idx
            else:
                # Fallback for unassigned jobs: random machine
                cwp_assign[j] = self.rng.randint(0, self.m - 1)

        # Store CWP individual with its pre-calculated cost
        cwp_ind = Individual(
            assignment=cwp_assign,
            fitness=cwp_res.total_cost,
            makespan=cwp_res.makespan,
            is_cwp=True,
        )
        cwp_ind.machine_costs = cwp_res.machine_costs
        pop.append(cwp_ind)

        # 2. Random individuals
        while len(pop) < self.config.pop_size:
            rand_assign = [self.rng.randint(0, self.m - 1) for _ in range(self.n)]
            pop.append(Individual(assignment=rand_assign))

        return pop

    def evaluate_fast(self, ind: Individual) -> None:
        """Evaluate individual using Random-Q SGBS + Fixed-Seq DP."""
        if ind.is_exact or ind.is_cwp:
            return  # Already evaluated exactly or from CWP trusted source

        # Group jobs by machine
        jobs_per_machine = [[] for _ in range(self.m)]
        for j, m_idx in enumerate(ind.assignment):
            jobs_per_machine[m_idx].append(j)

        total_energy = 0.0
        max_makespan = 0
        machine_costs = []

        for m_idx in range(self.m):
            jobs = jobs_per_machine[m_idx]
            if not jobs:
                machine_costs.append(0.0)
                continue

            # Random-Q SGBS
            seq, energy, start_times = random_q_sgbs_sequence(
                job_indices=jobs,
                processing_times=self.p,
                ct=self.ct,
                e_rate=int(self.e_rates[m_idx]),
                T_limit=self.T_limit,
                beta=self.config.sgbs_beta,
                gamma=self.config.sgbs_gamma,
                rng=self.np_rng,
            )

            if np.isfinite(energy):
                total_energy += energy
                if start_times:
                    mk = start_times[-1] + self.p[seq[-1]]
                    max_makespan = max(max_makespan, mk)
            else:
                total_energy = float("inf")

            machine_costs.append(energy)

        ind.fitness = total_energy
        ind.makespan = max_makespan
        ind.machine_costs = machine_costs
        ind.is_exact = False

    def evaluate_exact(self, ind: Individual) -> None:
        """Refine evaluation using exact multiset DP with caching.

        The objective is linear in the machine energy rate `e_rate`, so the
        optimal start times and finish time are invariant to scaling by `e_rate`.
        We therefore cache the DP result for `e_rate=1` and scale the cost.
        """
        # Group jobs by machine
        jobs_per_machine = [[] for _ in range(self.m)]
        for j, m_idx in enumerate(ind.assignment):
            jobs_per_machine[m_idx].append(j)

        total_energy = 0.0
        max_makespan = 0
        machine_costs = []

        for m_idx in range(self.m):
            jobs = jobs_per_machine[m_idx]
            if not jobs:
                machine_costs.append(0.0)
                continue

            p_jobs = [int(self.p[j]) for j in jobs]
            unique_p, counts = np.unique(
                np.asarray(p_jobs, dtype=np.int32), return_counts=True
            )
            signature: Tuple[Tuple[int, int], ...] = tuple(
                (int(p), int(c)) for p, c in zip(unique_p.tolist(), counts.tolist())
            )

            cached = self._exact_dp_cache.get(signature)
            if cached is None:
                res = solve_optimal_benchmark_dp(
                    processing_times=p_jobs,
                    prices=self._prices,
                    tie_break="early",
                    time_limit=float(self.config.exact_time_limit),
                )
                cached = (bool(res.feasible), float(res.cost), int(res.finish_time))
                self._exact_dp_cache[signature] = cached

            feasible_1, base_cost, finish_time = cached
            if feasible_1 and np.isfinite(base_cost):
                e_rate = float(self.e_rates[m_idx])
                cost_m = float(base_cost) * e_rate
                total_energy += cost_m
                max_makespan = max(max_makespan, int(finish_time))
                machine_costs.append(cost_m)
            else:
                total_energy = float("inf")
                machine_costs.append(float("inf"))

        ind.fitness = total_energy
        ind.makespan = max_makespan
        ind.machine_costs = machine_costs
        ind.is_exact = True

    def select_parent(self, pop: List[Individual]) -> Individual:
        """Tournament selection."""
        candidates = self.rng.sample(pop, self.config.tournament_size)
        return min(candidates, key=lambda i: i.fitness)

    def crossover(self, p1: Individual, p2: Individual) -> Individual:
        """Uniform crossover."""
        child_assign = []
        for j in range(self.n):
            if self.rng.random() < 0.5:
                child_assign.append(p1.assignment[j])
            else:
                child_assign.append(p2.assignment[j])
        return Individual(assignment=child_assign)

    def mutate(self, ind: Individual) -> None:
        """Mutation: Swap and Reassign."""
        # 1. Reassign mutation
        for j in range(self.n):
            if self.rng.random() < self.config.mutation_reassign_prob:
                ind.assignment[j] = self.rng.randint(0, self.m - 1)

        # 2. Swap mutation (if selected)
        if self.rng.random() < self.config.mutation_prob:
            j1, j2 = self.rng.sample(range(self.n), 2)
            ind.assignment[j1], ind.assignment[j2] = (
                ind.assignment[j2],
                ind.assignment[j1],
            )

        ind.is_exact = False  # Invalidate fitness
        ind.is_cwp = False  # Invalidate CWP status if mutated

    def solve(self, verbose: bool = False) -> Tuple[Solution, List[float]]:
        """Run the Genetic Algorithm."""
        start_time = time.perf_counter()

        # Init
        pop = self.initialize_population()
        for ind in pop:
            self.evaluate_fast(ind)

        best_log = []
        best_ever = min(pop, key=lambda i: i.fitness)
        best_log.append(best_ever.fitness)

        if verbose:
            print(f"Gen 0: Best Fast Fitness = {best_ever.fitness:.2f}")

        # Main Loop
        for gen in range(self.config.generations):
            next_pop = []

            # Elitism: keep best
            best_curr = min(pop, key=lambda i: i.fitness)
            if best_curr.fitness < best_ever.fitness:
                best_ever = best_curr.copy()
            next_pop.append(best_curr.copy())

            # Offspring
            while len(next_pop) < self.config.pop_size:
                p1 = self.select_parent(pop)
                p2 = self.select_parent(pop)

                if self.rng.random() < self.config.crossover_prob:
                    child = self.crossover(p1, p2)
                else:
                    child = p1.copy()

                self.mutate(child)
                self.evaluate_fast(child)
                next_pop.append(child)

            pop = next_pop
            best_curr = min(pop, key=lambda i: i.fitness)
            best_log.append(best_curr.fitness)

            if verbose and (gen + 1) % 10 == 0:
                print(f"Gen {gen+1}: Best Fast Fitness = {best_curr.fitness:.2f}")

        # Final Refinement
        if verbose:
            print("Refining top individuals with Exact DP...")

        # Sort by fast fitness
        pop.sort(key=lambda i: i.fitness)

        # Refine top-k
        to_refine = pop[: self.config.refine_top_k]
        best_refined_sol = to_refine[0] if to_refine else None
        min_exact_cost = float("inf")

        for ind in to_refine:
            original_fast_fitness = ind.fitness
            self.evaluate_exact(ind)

            # Sanity check: If Exact DP is worse than Fast Eval (which is a valid upper bound),
            # trust Fast Eval. Exact DP might have timed out or hit state limits.
            if ind.fitness > original_fast_fitness and np.isfinite(
                original_fast_fitness
            ):
                if verbose:
                    print(
                        f"Warning: Exact DP cost {ind.fitness} > Fast cost {original_fast_fitness}. Reverting."
                    )
                ind.fitness = original_fast_fitness
                ind.is_exact = False

            if ind.fitness < min_exact_cost:
                min_exact_cost = ind.fitness
                best_refined_sol = ind

        if verbose:
            print(f"Final Best Fitness = {min_exact_cost:.2f}")

        # Reconstruct Solution object (using sequence from SGBS for final schedule)
        if best_refined_sol is None:
            # Should not happen if pop_size > 0
            return Solution([], float("inf"), 0, False), best_log

        final_machines = []
        jobs_per_machine = [[] for _ in range(self.m)]
        for j, m_idx in enumerate(best_refined_sol.assignment):
            jobs_per_machine[m_idx].append(j)

        for m_idx in range(self.m):
            jobs = jobs_per_machine[m_idx]
            if not jobs:
                final_machines.append(MachineState(m_idx, [], [], [], 0.0, 0))
                continue

            # Re-run SGBS with higher effort for final sequence
            seq, _, start_times = random_q_sgbs_sequence(
                job_indices=jobs,
                processing_times=self.p,
                ct=self.ct,
                e_rate=int(self.e_rates[m_idx]),
                T_limit=self.T_limit,
                beta=16,  # Higher effort for final
                gamma=8,
                rng=self.np_rng,
            )

            # Note: We use the EXACT DP cost if available (stored in machine_costs)
            # best_refined_sol.machine_costs contains the exact costs
            exact_cost = best_refined_sol.machine_costs[m_idx]

            # Makespan is approx from SGBS or exact from DP?
            # We don't have exact makespan per machine easily accessible in `ind`
            # (only max makespan). So we'll use SGBS makespan for the schedule object.

            ms_makespan = start_times[-1] + self.p[seq[-1]] if start_times else 0

            final_machines.append(
                MachineState(
                    machine_idx=m_idx,
                    job_indices=jobs,
                    sequence=seq,
                    start_times=start_times,
                    energy=exact_cost,
                    makespan=ms_makespan,
                )
            )

        sol = Solution(
            machines=final_machines,
            total_energy=best_refined_sol.fitness,
            makespan=best_refined_sol.makespan,  # From exact DP
            feasible=np.isfinite(best_refined_sol.fitness),
        )

        return sol, best_log

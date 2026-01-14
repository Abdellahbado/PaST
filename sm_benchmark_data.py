"""
Data generation module for PaST-SM (Period-aware Scheduler Transformer - Single Machine).

Generates benchmark-style instances with segmented price horizons, NOT the paper ECSP
repeating 20-slot TOU pattern.

Key functions:
- generate_raw_instance: Creates a full benchmark-style instance
- make_single_machine_episode: Extracts single-machine episode from raw instance
- generate_episode_batch: Generates batches for training
"""

import random
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict

from .config import DataConfig


def discrete_uniform(a: int, b: int, rng: random.Random) -> int:
    """Inclusive discrete uniform U[a,b]."""
    return rng.randint(a, b)


def sample_intervals_sum_to_T(
    T: int, choices: Tuple[int, ...], rng: random.Random
) -> List[int]:
    """
    Sample interval durations Tk from 'choices' until they sum exactly to T.
    Uses backtracking to ensure exact sum.

    Args:
        T: Target sum (total horizon length)
        choices: Allowed interval durations (e.g., (2, 3, 5))
        rng: Random number generator

    Returns:
        List of interval durations that sum to T
    """
    Tk = []
    remaining = T
    max_attempts = 1000
    attempts = 0

    while remaining > 0:
        feasible = [x for x in choices if x <= remaining]
        if not feasible:
            # Restart if stuck (rare with choices {2,3,5})
            Tk = []
            remaining = T
            attempts += 1
            if attempts > max_attempts:
                raise ValueError(
                    f"Could not sample intervals summing to {T} from {choices}"
                )
            continue
        x = rng.choice(feasible)
        Tk.append(x)
        remaining -= x

    return Tk


def expand_ck_to_ct(Tk: List[int], ck: List[int]) -> List[int]:
    """
    Expand interval (period) prices ck into per-slot prices ct.

    Args:
        Tk: List of interval durations
        ck: List of interval prices (same length as Tk)

    Returns:
        Per-slot price vector of length sum(Tk)
    """
    ct = []
    for dur, price in zip(Tk, ck):
        ct.extend([price] * dur)
    return ct


def compute_period_start_slots(Tk: List[int]) -> List[int]:
    """
    Compute the starting slot index for each period.

    Args:
        Tk: List of period durations

    Returns:
        List of starting slot indices for each period
    """
    starts = []
    current = 0
    for dur in Tk:
        starts.append(current)
        current += dur
    return starts


@dataclass
class RawInstance:
    """
    A raw benchmark-style instance (parallel machine, full problem).

    This represents the full problem before extracting single-machine episodes.
    """

    instance_id: int
    scale: str  # "small", "mls", "vls"
    m: int  # Number of machines
    n: int  # Number of jobs
    T_max: int  # Total horizon length
    p: List[int]  # Job processing times, length n
    e: List[int]  # Machine energy rates, length m
    Tk: List[int]  # Period durations, sum(Tk) = T_max
    ck: List[int]  # Period prices, length K = len(Tk)
    ct: List[int]  # Per-slot prices, length T_max
    period_starts: List[int]  # Starting slot for each period


@dataclass
class SingleMachineEpisode:
    """
    A single-machine episode extracted from a raw instance.

    This is what the RL agent sees and acts on.
    """

    # Job information
    p_subset: (
        np.ndarray
    )  # Processing times of jobs assigned to this machine (1D int array)
    n_jobs: int  # Number of jobs = len(p_subset)

    # Time/price information
    T_max: int  # Total horizon length
    T_limit: int  # Deadline constraint (schedule must finish by this time)
    T_min: int  # Minimum feasible deadline = sum(p_subset)
    ct: np.ndarray  # Per-slot prices, length T_max (int array)

    # Period information
    Tk: np.ndarray  # Period durations (int array)
    ck: np.ndarray  # Period prices (int array)
    period_starts: np.ndarray  # Starting slot for each period (int array)
    K: int  # Number of periods = len(Tk)

    # Machine information
    e_single: int  # Energy rate of this machine


def generate_raw_instance(
    config: DataConfig,
    rng: random.Random,
    instance_id: int = 0,
    T_max: Optional[int] = None,
    m: Optional[int] = None,
    n: Optional[int] = None,
) -> RawInstance:
    """
    Generate a raw benchmark-style instance.

    Args:
        config: Data configuration
        rng: Random number generator
        instance_id: Instance identifier
        T_max: Horizon length (sampled from config if None)
        m: Number of machines (sampled from config if None)
        n: Number of jobs (sampled from config if None)

    Returns:
        RawInstance with all fields populated
    """
    # Sample parameters if not provided
    if T_max is None:
        T_max = rng.choice(config.T_max_choices)
    if m is None:
        m = discrete_uniform(config.m_min, config.m_max, rng)
    if n is None:
        n = discrete_uniform(config.n_min, config.n_max, rng)

    # Determine scale based on T_max
    if T_max <= 80:
        scale = "small"
    elif T_max <= 300:
        scale = "mls"
    else:
        scale = "vls"

    # Generate period structure
    Tk = sample_intervals_sum_to_T(T_max, config.Tk_choices, rng)
    K = len(Tk)
    ck = [discrete_uniform(config.ck_min, config.ck_max, rng) for _ in range(K)]
    ct = expand_ck_to_ct(Tk, ck)
    period_starts = compute_period_start_slots(Tk)

    # Generate job processing times
    p = [discrete_uniform(config.p_min, config.p_max, rng) for _ in range(n)]

    # Generate machine energy rates
    e = [discrete_uniform(config.e_min, config.e_max, rng) for _ in range(m)]

    return RawInstance(
        instance_id=instance_id,
        scale=scale,
        m=m,
        n=n,
        T_max=T_max,
        p=p,
        e=e,
        Tk=Tk,
        ck=ck,
        ct=ct,
        period_starts=period_starts,
    )


def simulate_metaheuristic_assignment(
    n: int,
    m: int,
    rng: random.Random,
    concentration: float = 1.0,
) -> List[List[int]]:
    """
    Simulate a meta-heuristic assigning n jobs to m machines.

    Uses Dirichlet distribution to create realistic unbalanced assignments.

    Args:
        n: Number of jobs
        m: Number of machines
        rng: Random number generator
        concentration: Dirichlet concentration parameter (lower = more unbalanced)

    Returns:
        List of m lists, each containing job indices assigned to that machine
    """
    # Sample machine weights from Dirichlet
    # Lower concentration = more unbalanced assignments
    alpha = [concentration] * m
    weights = np.random.default_rng(rng.randint(0, 2**31)).dirichlet(alpha)

    # Assign jobs to machines based on weights
    assignments = [[] for _ in range(m)]
    for job_idx in range(n):
        machine_idx = np.random.default_rng(rng.randint(0, 2**31)).choice(m, p=weights)
        assignments[machine_idx].append(job_idx)

    return assignments


def make_single_machine_episode(
    raw_instance: RawInstance,
    machine_index: int,
    job_indices: List[int],
    rng: random.Random,
    deadline_slack_ratio_min: float = 0.0,
    deadline_slack_ratio_max: float = 0.5,
) -> SingleMachineEpisode:
    """
    Extract a single-machine episode from a raw instance.

    Args:
        raw_instance: The full benchmark instance
        machine_index: Which machine this episode is for
        job_indices: Indices of jobs assigned to this machine
        rng: Random number generator
        deadline_slack_ratio_min: Minimum ratio for deadline slack
        deadline_slack_ratio_max: Maximum ratio for deadline slack

    Returns:
        SingleMachineEpisode ready for the RL environment
    """
    # Extract job processing times for this machine
    p_subset = np.array([raw_instance.p[j] for j in job_indices], dtype=np.int32)
    n_jobs = len(p_subset)

    # Compute minimum feasible deadline (no idle time)
    T_min = int(np.sum(p_subset)) if n_jobs > 0 else 0

    # Sample deadline constraint using epsilon-constraint approach
    # T_limit in [T_min, T_min + slack_ratio * (T_max - T_min)]
    T_max = raw_instance.T_max

    if n_jobs == 0:
        T_limit = T_max
    else:
        max_slack = T_max - T_min
        slack_ratio = rng.uniform(deadline_slack_ratio_min, deadline_slack_ratio_max)
        actual_slack = int(slack_ratio * max_slack)
        T_limit = min(T_min + actual_slack, T_max)
        # Ensure T_limit >= T_min
        T_limit = max(T_limit, T_min)

    # Get machine energy rate
    e_single = raw_instance.e[machine_index]

    return SingleMachineEpisode(
        p_subset=p_subset,
        n_jobs=n_jobs,
        T_max=T_max,
        T_limit=T_limit,
        T_min=T_min,
        ct=np.array(raw_instance.ct, dtype=np.int32),
        Tk=np.array(raw_instance.Tk, dtype=np.int32),
        ck=np.array(raw_instance.ck, dtype=np.int32),
        period_starts=np.array(raw_instance.period_starts, dtype=np.int32),
        K=len(raw_instance.Tk),
        e_single=e_single,
    )


def generate_single_machine_episode(
    config: DataConfig,
    rng: random.Random,
) -> SingleMachineEpisode:
    """
    Generate a single-machine episode directly.

    This is a convenience function that:
    1. Generates a raw instance
    2. Simulates meta-heuristic assignment
    3. Picks a random machine's job set

    Args:
        config: Data configuration
        rng: Random number generator

    Returns:
        SingleMachineEpisode ready for the RL environment
    """
    # Generate raw instance
    raw = generate_raw_instance(config, rng)

    # Simulate job assignment
    assignments = simulate_metaheuristic_assignment(raw.n, raw.m, rng)

    # Pick a random machine (prefer non-empty assignments)
    non_empty = [i for i, a in enumerate(assignments) if len(a) > 0]
    if non_empty:
        machine_idx = rng.choice(non_empty)
    else:
        # All empty - just pick machine 0
        machine_idx = 0

    job_indices = assignments[machine_idx]

    return make_single_machine_episode(
        raw,
        machine_idx,
        job_indices,
        rng,
        config.deadline_slack_ratio_min,
        config.deadline_slack_ratio_max,
    )


def episode_to_dict(episode: SingleMachineEpisode) -> Dict[str, Any]:
    """
    Convert episode to dictionary format for batching.

    Args:
        episode: SingleMachineEpisode

    Returns:
        Dictionary with numpy arrays
    """
    return {
        "p_subset": episode.p_subset,
        "n_jobs": episode.n_jobs,
        "T_max": episode.T_max,
        "T_limit": episode.T_limit,
        "T_min": episode.T_min,
        "ct": episode.ct,
        "Tk": episode.Tk,
        "ck": episode.ck,
        "period_starts": episode.period_starts,
        "K": episode.K,
        "e_single": episode.e_single,
    }


def generate_episode_batch(
    batch_size: int,
    config: DataConfig,
    seed: Optional[int] = None,
    N_job_pad: int = 50,
    K_period_pad: int = 250,  # Max periods: T_max=500 / min_period=2 = 250
    T_max_pad: int = 500,
) -> Dict[str, np.ndarray]:
    """
    Generate a batch of episodes for training.

    All arrays are padded to fixed sizes for batching.

    Args:
        batch_size: Number of episodes in batch
        config: Data configuration
        seed: Random seed
        N_job_pad: Maximum number of jobs (for padding)
        K_period_pad: Maximum number of periods (for padding)
        T_max_pad: Maximum horizon length (for padding)

    Returns:
        Dictionary with batched numpy arrays:
        - p_subset: (B, N_job_pad) int32 - job processing times
        - n_jobs: (B,) int32 - number of jobs per episode
        - job_mask: (B, N_job_pad) float32 - 1 for valid jobs, 0 for padding
        - T_max: (B,) int32 - horizon lengths
        - T_limit: (B,) int32 - deadline constraints
        - T_min: (B,) int32 - minimum feasible deadlines
        - ct: (B, T_max_pad) int32 - per-slot prices
        - Tk: (B, K_period_pad) int32 - period durations
        - ck: (B, K_period_pad) int32 - period prices
        - period_starts: (B, K_period_pad) int32 - period start slots
        - K: (B,) int32 - number of periods per episode
        - period_mask: (B, K_period_pad) float32 - 1 for valid periods
        - e_single: (B,) int32 - machine energy rates
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    # Initialize output arrays
    batch = {
        "p_subset": np.zeros((batch_size, N_job_pad), dtype=np.int32),
        "n_jobs": np.zeros((batch_size,), dtype=np.int32),
        "job_mask": np.zeros((batch_size, N_job_pad), dtype=np.float32),
        "T_max": np.zeros((batch_size,), dtype=np.int32),
        "T_limit": np.zeros((batch_size,), dtype=np.int32),
        "T_min": np.zeros((batch_size,), dtype=np.int32),
        "ct": np.zeros((batch_size, T_max_pad), dtype=np.int32),
        "Tk": np.zeros((batch_size, K_period_pad), dtype=np.int32),
        "ck": np.zeros((batch_size, K_period_pad), dtype=np.int32),
        "period_starts": np.zeros((batch_size, K_period_pad), dtype=np.int32),
        "K": np.zeros((batch_size,), dtype=np.int32),
        "period_mask": np.zeros((batch_size, K_period_pad), dtype=np.float32),
        "e_single": np.zeros((batch_size,), dtype=np.int32),
        # Price quantiles for price-family variant: [q25, q50, q75]
        "price_q": np.zeros((batch_size, 3), dtype=np.float32),
    }

    for i in range(batch_size):
        episode = generate_single_machine_episode(config, rng)

        # Fill in the batch arrays
        n = episode.n_jobs
        k = min(episode.K, K_period_pad)  # Clip to padding limit
        t_max = min(episode.T_max, T_max_pad)  # Clip to padding limit
        n = min(n, N_job_pad)  # Clip to padding limit

        batch["n_jobs"][i] = n
        batch["K"][i] = k
        batch["T_max"][i] = t_max
        batch["T_limit"][i] = episode.T_limit
        batch["T_min"][i] = episode.T_min
        batch["e_single"][i] = episode.e_single

        # Job data (padded)
        if n > 0:
            batch["p_subset"][i, :n] = episode.p_subset[:n]
            batch["job_mask"][i, :n] = 1.0

        # Period data (padded)
        if k > 0:
            batch["Tk"][i, :k] = episode.Tk[:k]
            batch["ck"][i, :k] = episode.ck[:k]
            batch["period_starts"][i, :k] = episode.period_starts[:k]
            batch["period_mask"][i, :k] = 1.0

        # Per-slot prices (padded)
        if t_max > 0:
            batch["ct"][i, :t_max] = episode.ct[:t_max]
            # Compute price quantiles for this episode (for price-family variant)
            ct_valid = episode.ct[:t_max]
            q25, q50, q75 = np.quantile(ct_valid, [0.25, 0.5, 0.75])
            batch["price_q"][i] = [q25, q50, q75]

    return batch


def generate_episode_batch_variable(
    batch_size: int,
    config: DataConfig,
    seed: Optional[int] = None,
) -> List[SingleMachineEpisode]:
    """
    Generate a batch of episodes without padding (variable sizes).

    Useful for inference or when padding overhead is undesirable.

    Args:
        batch_size: Number of episodes
        config: Data configuration
        seed: Random seed

    Returns:
        List of SingleMachineEpisode objects
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    episodes = []
    for _ in range(batch_size):
        episode = generate_single_machine_episode(config, rng)
        episodes.append(episode)

    return episodes


if __name__ == "__main__":
    # Test data generation
    print("Testing PaST-SM Data Generation...")
    print("=" * 60)

    # Create config
    config = DataConfig()
    rng = random.Random(42)

    # Test raw instance generation
    print("\n1. Testing raw instance generation...")
    raw = generate_raw_instance(config, rng, instance_id=1)
    print(f"   Instance ID: {raw.instance_id}")
    print(f"   Scale: {raw.scale}")
    print(f"   m={raw.m}, n={raw.n}, T_max={raw.T_max}")
    print(f"   K={len(raw.Tk)} periods")
    print(f"   First 5 job times: {raw.p[:5]}")
    print(f"   First 5 period durations: {raw.Tk[:5]}")
    print(f"   First 5 period prices: {raw.ck[:5]}")
    print(f"   Machine energy rates: {raw.e}")

    # Test meta-heuristic simulation
    print("\n2. Testing meta-heuristic assignment simulation...")
    assignments = simulate_metaheuristic_assignment(raw.n, raw.m, rng)
    for i, a in enumerate(assignments):
        print(f"   Machine {i}: {len(a)} jobs")

    # Test single machine episode extraction
    print("\n3. Testing single machine episode extraction...")
    machine_idx = 0
    episode = make_single_machine_episode(
        raw,
        machine_idx,
        assignments[machine_idx],
        rng,
        config.deadline_slack_ratio_min,
        config.deadline_slack_ratio_max,
    )
    print(f"   Machine {machine_idx}:")
    print(f"   n_jobs={episode.n_jobs}")
    print(f"   Job times: {episode.p_subset}")
    print(f"   T_min={episode.T_min}, T_limit={episode.T_limit}, T_max={episode.T_max}")
    print(f"   e_single={episode.e_single}")
    print(f"   K={episode.K} periods")

    # Test direct episode generation
    print("\n4. Testing direct episode generation...")
    episode = generate_single_machine_episode(config, rng)
    print(f"   n_jobs={episode.n_jobs}")
    print(f"   T_min={episode.T_min}, T_limit={episode.T_limit}, T_max={episode.T_max}")

    # Test batch generation
    print("\n5. Testing batch generation...")
    batch = generate_episode_batch(batch_size=8, config=config, seed=42)
    print(f"   Batch keys: {list(batch.keys())}")
    print(f"   p_subset shape: {batch['p_subset'].shape}")
    print(f"   n_jobs: {batch['n_jobs']}")
    print(f"   T_limit: {batch['T_limit']}")
    print(f"   e_single: {batch['e_single']}")

    # Test variable batch generation
    print("\n6. Testing variable batch generation...")
    episodes = generate_episode_batch_variable(batch_size=4, config=config, seed=42)
    for i, ep in enumerate(episodes):
        print(f"   Episode {i}: n_jobs={ep.n_jobs}, T_limit={ep.T_limit}, K={ep.K}")

    print("\n" + "=" * 60)
    print("All data generation tests passed!")

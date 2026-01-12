"""
Single-Machine Period-aware Environment for PaST-SM.

Event-based scheduler environment with:
- Segmented price horizon (not repeating TOU)
- Hard deadline constraint (epsilon-constraint)
- Configurable slack action variants
- Dense energy-only reward

The agent decides:
1. Which job to schedule next
2. How much slack (idle time) to insert before the job
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, List, Union, Any
from dataclasses import dataclass

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from .config import (
    EnvConfig,
    SlackVariant,
    ShortSlackSpec,
    PeriodAlignedSlackSpec,
)
from .sm_benchmark_data import (
    SingleMachineEpisode,
    generate_single_machine_episode,
    DataConfig,
)

# Version identifier
ENV_VERSION = "1.0-SM"
print(
    f"[SingleMachinePeriodEnv v{ENV_VERSION}] Loading single-machine period-aware environment..."
)


def slack_to_start_time(
    t_now: int,
    slack_id: int,
    env_config: EnvConfig,
    period_starts: np.ndarray,
    Tk: np.ndarray,
    K: int,
    T_limit: int,
) -> int:
    """
    Convert slack_id to actual start time in slots.

    Args:
        t_now: Current time slot
        slack_id: Slack action identifier
        env_config: Environment configuration (contains slack variant and specs)
        period_starts: Array of period start times
        Tk: Array of period durations
        K: Number of periods
        T_limit: Deadline constraint

    Returns:
        Start time (slot index) >= t_now
    """
    variant = env_config.slack_variant

    if variant == SlackVariant.NO_SLACK:
        # Only option is start now
        return t_now

    elif variant == SlackVariant.SHORT_SLACK:
        if env_config.short_slack_spec is None:
            raise ValueError(
                "SHORT_SLACK variant requires 'short_slack_spec' to be configured."
            )
        # SHORT_SLACK uses period offsets, not slot offsets
        # slack_id_to_period_offset returns the period offset from current period
        period_offset = env_config.short_slack_spec.slack_id_to_period_offset(slack_id)

        if period_offset == 0:
            # Start now
            return t_now

        # Find the current period
        current_period = 0
        for k in range(K):
            if period_starts[k] <= t_now < period_starts[k] + Tk[k]:
                current_period = k
                break
            elif period_starts[k] > t_now:
                current_period = max(0, k - 1)
                break

        # Target period is current + offset
        target_period = current_period + period_offset
        if target_period >= K:
            return T_limit

        target_start = period_starts[target_period]
        return min(max(target_start, t_now), T_limit)

    elif variant == SlackVariant.PERIOD_ALIGNED:
        if env_config.period_aligned_spec is None:
            env_config.period_aligned_spec = PeriodAlignedSlackSpec()

        if slack_id == 0:
            # Start now
            return t_now

        # Find the current period
        current_period = 0
        for k in range(K):
            if period_starts[k] <= t_now < period_starts[k] + Tk[k]:
                current_period = k
                break
            elif period_starts[k] > t_now:
                current_period = max(0, k - 1)
                break

        # slack_id corresponds to starting at the (current_period + slack_id)-th period
        target_period = current_period + slack_id
        if target_period >= K:
            # Beyond last period - start at T_limit
            return T_limit

        target_start = period_starts[target_period]
        # Ensure we don't go back in time
        return max(target_start, t_now)

    elif variant == SlackVariant.COARSE_TO_FINE:
        if env_config.c2f_slack_spec is None:
            raise ValueError(
                "COARSE_TO_FINE variant requires 'c2f_slack_spec' to be configured."
            )

        # Get period offset from the coarse-to-fine spec
        period_offset = env_config.c2f_slack_spec.slack_id_to_period_offset(slack_id)

        if period_offset == 0:
            # Start now
            return t_now

        # Find the current period
        current_period = 0
        for k in range(K):
            if period_starts[k] <= t_now < period_starts[k] + Tk[k]:
                current_period = k
                break
            elif period_starts[k] > t_now:
                current_period = max(0, k - 1)
                break

        # Target period is current + offset
        target_period = current_period + period_offset
        if target_period >= K:
            # Beyond last period - start at T_limit
            return T_limit

        target_start = period_starts[target_period]
        # Ensure we don't go back in time and don't exceed deadline
        return min(max(target_start, t_now), T_limit)

    elif variant == SlackVariant.FULL_SLACK:
        # FULL_SLACK: slack_id is the period offset from current period
        # (matching config.FullSlackSpec which defines K_full_max as period count)
        period_offset = slack_id

        if period_offset == 0:
            return t_now

        # Find the current period
        current_period = 0
        for k in range(K):
            if period_starts[k] <= t_now < period_starts[k] + Tk[k]:
                current_period = k
                break
            elif period_starts[k] > t_now:
                current_period = max(0, k - 1)
                break

        target_period = current_period + period_offset
        if target_period >= K:
            return T_limit

        target_start = period_starts[target_period]
        return min(max(target_start, t_now), T_limit)

    elif variant == SlackVariant.LEARNED_SLACK:
        raise NotImplementedError("LEARNED_SLACK variant is not yet implemented.")

    else:
        raise ValueError(f"Unknown slack variant: {variant}")


def find_period_at_time(
    t: int, period_starts: np.ndarray, Tk: np.ndarray, K: int
) -> int:
    """
    Find which period contains time slot t.

    Args:
        t: Time slot
        period_starts: Array of period start times
        Tk: Array of period durations
        K: Number of periods

    Returns:
        Period index (0 to K-1), or K if beyond all periods
    """
    for k in range(K):
        if period_starts[k] <= t < period_starts[k] + Tk[k]:
            return k
    return K  # Beyond all periods


class SingleMachinePeriodEnv(gym.Env):
    """
    Single-Machine Period-aware Environment.

    State:
    - Current time t (slot index)
    - Remaining jobs (list of processing times with mask)
    - Period/price structure
    - Deadline T_limit
    - Machine energy rate e_single

    Action:
    - Single integer encoding (job_id, slack_id) pair
    - action = job_id * num_slack_choices + slack_id

    Reward:
    - Dense, energy-only: reward = -energy_consumed_this_step

    Termination:
    - When all jobs are scheduled
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        env_config: EnvConfig = None,
        data_config: DataConfig = None,
    ):
        """
        Initialize the environment.

        Args:
            env_config: Environment configuration
            data_config: Data generation configuration (for auto-generating episodes)
        """
        super().__init__()

        self.env_config = env_config if env_config is not None else EnvConfig()
        self.data_config = data_config if data_config is not None else DataConfig()

        # Dimensions from config
        self.N_job_pad = self.env_config.N_job_pad
        self.K_period_lookahead = self.env_config.K_period_lookahead
        self.F_job = self.env_config.F_job
        self.F_period = self.env_config.F_period
        self.F_ctx = self.env_config.F_ctx

        # Slack configuration
        self.num_slack_choices = self.env_config.get_num_slack_choices()

        # Action space: job_id * num_slack_choices + slack_id
        self.action_dim = self.N_job_pad * self.num_slack_choices
        self.action_space = spaces.Discrete(self.action_dim)

        # Observation space (fixed-size tensors)
        self.observation_space = spaces.Dict(
            {
                "jobs": spaces.Box(
                    low=0,
                    high=1000,
                    shape=(self.N_job_pad, self.F_job),
                    dtype=np.float32,
                ),
                "periods": spaces.Box(
                    low=-1000,
                    high=1000,
                    shape=(self.K_period_lookahead, self.F_period),
                    dtype=np.float32,
                ),
                "ctx": spaces.Box(
                    low=-1000, high=1000, shape=(self.F_ctx,), dtype=np.float32
                ),
                "job_mask": spaces.Box(
                    low=0, high=1, shape=(self.N_job_pad,), dtype=np.float32
                ),
                "action_mask": spaces.Box(
                    low=0, high=1, shape=(self.action_dim,), dtype=np.float32
                ),
            }
        )

        # Episode state (initialized in reset)
        self._reset_state()

    def _reset_state(self):
        """Reset internal state variables."""
        self.t = 0  # Current time
        self.remaining_jobs = None  # List of remaining job processing times
        self.job_mask_internal = None  # Which jobs are still available
        self.episode = None  # Current episode data
        self.done = False
        self.total_energy = 0.0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Reset the environment to start a new episode.

        Args:
            seed: Random seed
            options: Optional dict with:
                - "episode": SingleMachineEpisode to use (if not provided, auto-generate)

        Returns:
            observation: Initial observation dict
            info: Additional information
        """
        super().reset(seed=seed)

        # Get or generate episode
        if options is not None and "episode" in options:
            self.episode = options["episode"]
        else:
            # Auto-generate episode
            import random

            rng = random.Random(seed)
            self.episode = generate_single_machine_episode(self.data_config, rng)

        # Initialize state
        self.t = 0
        self.done = False
        self.total_energy = 0.0

        # Initialize remaining jobs
        self.remaining_jobs = list(self.episode.p_subset.copy())
        self.job_mask_internal = np.ones(len(self.remaining_jobs), dtype=np.float32)

        # Build observation
        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Build observation dictionary."""
        ep = self.episode

        # Jobs tensor: (N_job_pad, F_job)
        # F_job = 1 (just processing time for now)
        jobs = np.zeros((self.N_job_pad, self.F_job), dtype=np.float32)
        n_remaining = len(self.remaining_jobs)
        if n_remaining > 0:
            jobs[:n_remaining, 0] = self.remaining_jobs

        # Job mask: (N_job_pad,)
        job_mask = np.zeros(self.N_job_pad, dtype=np.float32)
        job_mask[:n_remaining] = self.job_mask_internal[:n_remaining]

        # Periods tensor: (K_period_lookahead, F_period)
        # F_period = 4: [duration, price, start_slot, is_current]
        periods = np.zeros((self.K_period_lookahead, self.F_period), dtype=np.float32)

        # Find current period
        current_period_idx = find_period_at_time(self.t, ep.period_starts, ep.Tk, ep.K)

        # Fill period lookahead window
        for i in range(self.K_period_lookahead):
            period_idx = current_period_idx + i
            if period_idx >= ep.K:
                break

            # Check if this period is before T_limit
            period_end = ep.period_starts[period_idx] + ep.Tk[period_idx]
            if ep.period_starts[period_idx] >= ep.T_limit:
                break

            periods[i, 0] = ep.Tk[period_idx]  # Duration
            periods[i, 1] = ep.ck[period_idx]  # Price
            periods[i, 2] = ep.period_starts[period_idx]  # Start slot
            periods[i, 3] = 1.0 if i == 0 else 0.0  # Is current period

        # Context tensor: (F_ctx,)
        # F_ctx = 6: [t, T_limit, remaining_work, e_single, avg_price_after_window, min_price_after_window]
        ctx = np.zeros(self.F_ctx, dtype=np.float32)
        ctx[0] = self.t
        ctx[1] = ep.T_limit
        ctx[2] = sum(self.remaining_jobs)  # Remaining work
        ctx[3] = ep.e_single

        # Compute price statistics beyond lookahead window
        window_end_period = min(current_period_idx + self.K_period_lookahead, ep.K)
        if window_end_period < ep.K:
            # There are periods beyond the window
            remaining_periods = range(window_end_period, ep.K)
            remaining_durations = [ep.Tk[k] for k in remaining_periods]
            remaining_prices = [ep.ck[k] for k in remaining_periods]

            total_duration = sum(remaining_durations)
            if total_duration > 0:
                ctx[4] = (
                    sum(d * p for d, p in zip(remaining_durations, remaining_prices))
                    / total_duration
                )
                ctx[5] = min(remaining_prices)
            else:
                ctx[4] = 0.0
                ctx[5] = 0.0
        else:
            ctx[4] = 0.0
            ctx[5] = 0.0

        # Action mask: (action_dim,)
        action_mask = self.get_action_mask()

        return {
            "jobs": jobs,
            "periods": periods,
            "ctx": ctx,
            "job_mask": job_mask,
            "action_mask": action_mask,
        }

    def _get_info(self) -> Dict:
        """Get additional info dictionary."""
        return {
            "t": self.t,
            "n_remaining": len(self.remaining_jobs),
            "total_energy": self.total_energy,
            "T_limit": self.episode.T_limit if self.episode else 0,
            "T_max": self.episode.T_max if self.episode else 0,
        }

    def get_action_mask(self) -> np.ndarray:
        """
        Compute action mask indicating valid (job, slack) pairs.

        An action (job_id, slack_id) is valid if:
        1. job_id refers to a remaining job
        2. The job can finish by T_limit given the slack

        Returns:
            Action mask of shape (action_dim,) where 1 = valid, 0 = invalid
        """
        mask = np.zeros(self.action_dim, dtype=np.float32)

        if self.done or self.episode is None:
            return mask

        ep = self.episode
        n_remaining = len(self.remaining_jobs)

        for job_id in range(n_remaining):
            if self.job_mask_internal[job_id] == 0:
                continue

            p = self.remaining_jobs[job_id]

            for slack_id in range(self.num_slack_choices):
                # Compute start time for this slack
                start_time = slack_to_start_time(
                    self.t,
                    slack_id,
                    self.env_config,
                    ep.period_starts,
                    ep.Tk,
                    ep.K,
                    ep.T_limit,
                )

                # Check if job can finish by deadline
                end_time = start_time + p
                if end_time <= ep.T_limit:
                    action_idx = job_id * self.num_slack_choices + slack_id
                    mask[action_idx] = 1.0

        return mask

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """
        Take a step in the environment.

        Args:
            action: Integer action encoding (job_id, slack_id)

        Returns:
            observation: New observation
            reward: Step reward (negative energy)
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        if self.done:
            return self._get_obs(), 0.0, True, False, self._get_info()

        ep = self.episode

        # Decode action
        job_id = action // self.num_slack_choices
        slack_id = action % self.num_slack_choices

        # Validate action
        n_remaining = len(self.remaining_jobs)
        if job_id >= n_remaining or self.job_mask_internal[job_id] == 0:
            raise ValueError(f"Invalid job_id {job_id}: job not available")

        # Get job processing time
        p = self.remaining_jobs[job_id]

        # Compute start time
        start_time = slack_to_start_time(
            self.t,
            slack_id,
            self.env_config,
            ep.period_starts,
            ep.Tk,
            ep.K,
            ep.T_limit,
        )

        end_time = start_time + p

        # Check deadline constraint
        if end_time > ep.T_limit:
            raise ValueError(
                f"Action violates deadline: end_time={end_time} > T_limit={ep.T_limit}. "
                "This action should have been masked out."
            )

        # Compute energy cost
        # Energy = e_single * sum(ct[u] for u in [start_time, end_time))
        energy = ep.e_single * np.sum(ep.ct[start_time:end_time])
        self.total_energy += energy

        # Update state
        self.t = end_time

        # Remove job from remaining list
        # We mark it as unavailable in the mask, then compact later
        self.job_mask_internal[job_id] = 0.0

        # Actually remove the job (shift remaining jobs)
        self.remaining_jobs.pop(job_id)
        self.job_mask_internal = np.ones(len(self.remaining_jobs), dtype=np.float32)

        # Check if done
        if len(self.remaining_jobs) == 0:
            self.done = True

        # Reward is negative energy (we want to minimize energy)
        reward = -float(energy)

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, self.done, False, info

    def get_final_metrics(self) -> Tuple[float, float]:
        """
        Get final episode metrics.

        Returns:
            total_energy: Total energy consumed
            makespan: Final time (completion time of last job)
        """
        return self.total_energy, float(self.t)


class GPUBatchSingleMachinePeriodEnv:
    """
    GPU-native batched single-machine environment using PyTorch tensors.
    All operations stay on GPU to eliminate CPU-GPU transfer bottleneck.

    Mirrors the design of GPUBatchECSPEnv from the paper implementation.
    """

    def __init__(
        self,
        batch_size: int = 2048,
        env_config: EnvConfig = None,
        device: torch.device = None,
    ):
        """
        Initialize batched GPU environment.

        Args:
            batch_size: Number of parallel environments
            env_config: Environment configuration
            device: PyTorch device
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for GPUBatchSingleMachinePeriodEnv")

        self.batch_size = batch_size
        self.env_config = env_config if env_config is not None else EnvConfig()
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Dimensions
        self.N_job_pad = self.env_config.N_job_pad
        self.K_period_lookahead = self.env_config.K_period_lookahead
        self.F_job = self.env_config.F_job
        self.F_period = self.env_config.F_period
        self.F_ctx = self.env_config.F_ctx

        self.num_slack_choices = self.env_config.get_num_slack_choices()
        self.action_dim = self.N_job_pad * self.num_slack_choices

        # Allocate persistent tensors
        self._allocate_tensors()

        # State tracking
        self.done_mask = None

    def _allocate_tensors(self):
        """Allocate persistent GPU tensors."""
        B = self.batch_size

        # Episode data (set during reset)
        self.p_subset = torch.zeros(
            (B, self.N_job_pad), dtype=torch.int32, device=self.device
        )
        self.n_jobs = torch.zeros((B,), dtype=torch.int32, device=self.device)
        self.T_max = torch.zeros((B,), dtype=torch.int32, device=self.device)
        self.T_limit = torch.zeros((B,), dtype=torch.int32, device=self.device)
        self.e_single = torch.zeros((B,), dtype=torch.int32, device=self.device)

        # Price data - use larger padding for ct
        self.K_pad = 250  # Maximum periods: T_max=500 / min_period=2 = 250
        self.T_max_pad = 500  # Maximum slots
        self.ct = torch.zeros(
            (B, self.T_max_pad), dtype=torch.int32, device=self.device
        )
        self.Tk = torch.zeros((B, self.K_pad), dtype=torch.int32, device=self.device)
        self.ck = torch.zeros((B, self.K_pad), dtype=torch.int32, device=self.device)
        self.period_starts = torch.zeros(
            (B, self.K_pad), dtype=torch.int32, device=self.device
        )
        self.K = torch.zeros((B,), dtype=torch.int32, device=self.device)

        # Dynamic state
        self.t = torch.zeros((B,), dtype=torch.int32, device=self.device)
        self.job_available = torch.zeros(
            (B, self.N_job_pad), dtype=torch.float32, device=self.device
        )
        self.total_energy = torch.zeros((B,), dtype=torch.float32, device=self.device)

    def reset(
        self,
        batch_data: Dict[str, Union[np.ndarray, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Reset environments with batch data.

        Args:
            batch_data: Dictionary from generate_episode_batch, containing:
                - p_subset, n_jobs, T_max, T_limit, T_min, ct, Tk, ck,
                  period_starts, K, e_single, job_mask, period_mask

        Returns:
            Initial observation dictionary with GPU tensors
        """
        B = self.batch_size

        # Helper to convert to tensor
        def to_tensor(x, dtype=torch.float32):
            if isinstance(x, torch.Tensor):
                return x.to(device=self.device, dtype=dtype)
            return torch.tensor(x, dtype=dtype, device=self.device)

        # Load episode data
        self.p_subset[:] = to_tensor(batch_data["p_subset"], torch.int32)
        self.n_jobs[:] = to_tensor(batch_data["n_jobs"], torch.int32)
        self.T_max[:] = to_tensor(batch_data["T_max"], torch.int32)
        self.T_limit[:] = to_tensor(batch_data["T_limit"], torch.int32)
        self.e_single[:] = to_tensor(batch_data["e_single"], torch.int32)

        # Price/period data
        ct = batch_data["ct"]
        if ct.shape[1] <= self.T_max_pad:
            self.ct[:, : ct.shape[1]] = to_tensor(ct, torch.int32)
        else:
            self.ct[:] = to_tensor(ct[:, : self.T_max_pad], torch.int32)

        Tk = batch_data["Tk"]
        if Tk.shape[1] <= self.K_pad:
            self.Tk[:, : Tk.shape[1]] = to_tensor(Tk, torch.int32)
            self.ck[:, : Tk.shape[1]] = to_tensor(batch_data["ck"], torch.int32)
            self.period_starts[:, : Tk.shape[1]] = to_tensor(
                batch_data["period_starts"], torch.int32
            )

        self.K[:] = to_tensor(batch_data["K"], torch.int32)

        # Initialize dynamic state
        self.t.zero_()
        self.total_energy.zero_()

        # Job availability (1 where job exists, 0 otherwise)
        self.job_available[:] = to_tensor(batch_data["job_mask"], torch.float32)

        # Done mask
        self.done_mask = torch.zeros((B,), dtype=torch.bool, device=self.device)

        return self._get_obs()

    def _compute_current_period(self) -> torch.Tensor:
        """
        Compute the current period index for each instance.

        Returns:
            (B,) tensor of current period indices
        """
        # For each instance, find which period contains current time t:
        # period_starts[k] <= t < period_starts[k] + Tk[k]
        B = self.batch_size
        t_expanded = self.t.unsqueeze(1)  # (B, 1)
        period_ends = self.period_starts + self.Tk  # (B, K_pad)

        # Mask out padding periods beyond each instance's K
        idx = torch.arange(self.K_pad, device=self.device).unsqueeze(0)  # (1, K_pad)
        valid = idx < self.K.unsqueeze(1)  # (B, K_pad)

        in_period = (
            valid & (self.period_starts <= t_expanded) & (t_expanded < period_ends)
        )

        any_in = in_period.any(dim=1)  # (B,)
        # argmax on float gives the first occurrence of max (i.e., first True)
        first_idx = in_period.float().argmax(dim=1).to(torch.int32)  # (B,)

        # If t is beyond all periods, default to last valid period (K-1)
        fallback = torch.clamp(self.K - 1, min=0).to(torch.int32)
        current_period = torch.where(any_in, first_idx, fallback)

        return current_period

    def _compute_c2f_period_offsets(self, slack_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute period offsets from coarse-to-fine slack_ids.

        Uses the same per-bucket direct-vs-strided logic as _compute_c2f_period_offsets_batch()
        to ensure consistency between action mask and step dynamics.

        Args:
            slack_ids: (B,) tensor of slack action indices

        Returns:
            (B,) tensor of period offsets
        """
        # Delegate to batch version with shape (B, 1) then squeeze
        # This ensures mask and step use identical decoding logic
        slack_ids_expanded = slack_ids.unsqueeze(1)  # (B, 1)
        period_offsets = self._compute_c2f_period_offsets_batch(
            slack_ids_expanded
        )  # (B, 1)
        return period_offsets.squeeze(1)  # (B,)

    def _compute_period_aligned_starts(
        self, period_offsets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute start times from period offsets.

        Args:
            period_offsets: (B,) tensor of target period offsets from current period

        Returns:
            (B,) tensor of start times
        """
        B = self.batch_size
        current_period = self._compute_current_period()  # (B,)

        # Target period = current + offset
        target_period = current_period + period_offsets.int()

        # Clamp to valid range [0, K-1] (avoid torch.clamp mixed scalar/tensor args)
        zero = torch.zeros_like(target_period)
        k_minus_1 = torch.clamp(self.K - 1, min=0).to(target_period.dtype)
        target_period = torch.maximum(target_period, zero)
        target_period = torch.minimum(target_period, k_minus_1)

        # Gather start time for target period
        start_times = torch.gather(
            self.period_starts, 1, target_period.unsqueeze(1).long()
        ).squeeze(1)

        # Ensure start >= current time and <= T_limit
        start_times = torch.maximum(start_times, self.t)
        start_times = torch.minimum(start_times, self.T_limit)

        return start_times

    def _compute_action_mask(self, job_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute deadline-aware action mask.

        An action (job_id, slack_id) is valid if:
        1. job_id refers to an available job (job_mask[job_id] == 1)
        2. The job can finish by T_limit given the slack choice

        Args:
            job_mask: (B, N_job_pad) tensor of job availability

        Returns:
            (B, action_dim) tensor where 1 = valid, 0 = invalid
        """
        B = self.batch_size
        N = self.N_job_pad
        S = self.num_slack_choices

        # Compute start times for each slack choice
        # Shape: (B, S)
        slack_start_times = self._compute_all_slack_start_times()

        # Processing times for all jobs: (B, N)
        p_all = self.p_subset.int()

        # Expand for broadcasting: start_times (B, 1, S), p (B, N, 1)
        start_expanded = slack_start_times.unsqueeze(1)  # (B, 1, S)
        p_expanded = p_all.unsqueeze(2)  # (B, N, 1)

        # End times for all (job, slack) combinations: (B, N, S)
        end_times = start_expanded + p_expanded

        # Check deadline feasibility: end_time <= T_limit
        T_limit_expanded = self.T_limit.unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
        feasible = end_times <= T_limit_expanded  # (B, N, S)

        # Combine with job availability
        job_available = job_mask.unsqueeze(2).bool()  # (B, N, 1)
        valid = feasible & job_available  # (B, N, S)

        # Reshape to action_dim: action = job_id * S + slack_id
        # So action_mask[b, j*S + s] = valid[b, j, s]
        action_mask = valid.reshape(B, N * S).float()

        return action_mask

    def _compute_all_slack_start_times(self) -> torch.Tensor:
        """
        Compute start times for all slack choices.

        Returns:
            (B, num_slack_choices) tensor of start times
        """
        B = self.batch_size
        S = self.num_slack_choices

        if self.env_config.slack_variant == SlackVariant.NO_SLACK:
            # Only one choice: start now
            return self.t.unsqueeze(1)  # (B, 1)

        elif self.env_config.slack_variant == SlackVariant.SHORT_SLACK:
            if self.env_config.short_slack_spec is not None:
                # SHORT_SLACK uses period offsets, not slot offsets
                # slack_options are period offsets from current period
                slack_options = torch.tensor(
                    self.env_config.short_slack_spec.slack_options,
                    dtype=torch.int32,
                    device=self.device,
                )  # (S,)

                # Compute period-aligned start times
                # period_offsets for all slack choices: (S,)
                # Expand to (B, S) for batch processing
                period_offsets = slack_options.unsqueeze(0).expand(B, S)  # (B, S)

                # Use the same period-aligned computation as C2F
                start_times = self._compute_period_aligned_starts_batch(period_offsets)
                return start_times
            else:
                return self.t.unsqueeze(1).expand(B, S)

        elif self.env_config.slack_variant == SlackVariant.COARSE_TO_FINE:
            if self.env_config.c2f_slack_spec is not None:
                # For each slack_id in [0, S), compute the start time
                slack_ids = torch.arange(S, device=self.device)  # (S,)
                slack_ids_expanded = slack_ids.unsqueeze(0).expand(B, S)  # (B, S)

                # Compute period offsets for all slack choices
                period_offsets = self._compute_c2f_period_offsets_batch(
                    slack_ids_expanded
                )

                # Convert to start times
                start_times = self._compute_period_aligned_starts_batch(period_offsets)
                return start_times
            else:
                return self.t.unsqueeze(1).expand(B, S)

        elif self.env_config.slack_variant == SlackVariant.FULL_SLACK:
            # FULL_SLACK: slack_id is period offset from current period
            # (matching config.FullSlackSpec which defines K_full_max as period count)
            slack_ids = torch.arange(S, dtype=torch.int32, device=self.device)  # (S,)
            period_offsets = slack_ids.unsqueeze(0).expand(B, S)  # (B, S)

            # Use period-aligned computation
            start_times = self._compute_period_aligned_starts_batch(period_offsets)
            return start_times

        else:
            # Default: all start now
            return self.t.unsqueeze(1).expand(B, S)

    def _compute_c2f_period_offsets_batch(
        self, slack_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute period offsets for a batch of slack_ids.

        Matches the logic in CoarseToFineSlackSpec.slack_id_to_period_offset():
        - If fine_resolution >= bucket_size: direct mapping (fine_idx clamped to bucket)
        - Else: strided mapping

        Args:
            slack_ids: (B, S) tensor

        Returns:
            (B, S) tensor of period offsets
        """
        spec = self.env_config.c2f_slack_spec
        fine_resolution = spec.fine_resolution

        coarse_idx = slack_ids // fine_resolution
        fine_idx = slack_ids % fine_resolution

        buckets = torch.tensor(
            spec.coarse_buckets, dtype=torch.int32, device=self.device
        )

        coarse_idx = torch.clamp(coarse_idx.long(), 0, len(spec.coarse_buckets) - 1)

        bucket_starts = buckets[coarse_idx, 0]
        bucket_ends = buckets[coarse_idx, 1]
        bucket_sizes = bucket_ends - bucket_starts + 1

        # Match spec logic: direct mapping when fine_resolution >= bucket_size
        # For each (b, s), choose between direct and strided mapping
        use_direct = fine_resolution >= bucket_sizes  # (B, S) bool

        # Direct mapping: period_in_bucket = min(fine_idx, bucket_size - 1)
        direct_result = torch.minimum(fine_idx.int(), bucket_sizes - 1)

        # Strided mapping: period_in_bucket = int(fine_idx * stride)
        stride = bucket_sizes.float() / fine_resolution
        strided_result = (fine_idx.float() * stride).int()
        strided_result = torch.minimum(strided_result, bucket_sizes - 1)

        period_in_bucket = torch.where(use_direct, direct_result, strided_result)

        return bucket_starts + period_in_bucket

    def _compute_period_aligned_starts_batch(
        self, period_offsets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute start times from period offsets (batched over slack choices).

        Args:
            period_offsets: (B, S) tensor

        Returns:
            (B, S) tensor of start times
        """
        B, S = period_offsets.shape
        current_period = self._compute_current_period()  # (B,)

        target_period = current_period.unsqueeze(1) + period_offsets.int()  # (B, S)

        # Clamp to valid range [0, K-1] (avoid torch.clamp mixed scalar/tensor args)
        zero = torch.zeros_like(target_period)
        k_minus_1 = torch.clamp(self.K - 1, min=0).to(target_period.dtype).unsqueeze(1)
        target_period = torch.maximum(target_period, zero)
        target_period = torch.minimum(target_period, k_minus_1)

        # Gather start times - need to handle (B, S) indices into (B, K_pad)
        # Flatten, gather, reshape
        target_flat = target_period.reshape(B * S)  # (B*S,)
        period_starts_expanded = self.period_starts.unsqueeze(1).expand(
            B, S, -1
        )  # (B, S, K_pad)
        period_starts_flat = period_starts_expanded.reshape(B * S, -1)  # (B*S, K_pad)

        start_times_flat = torch.gather(
            period_starts_flat, 1, target_flat.unsqueeze(1).long()
        ).squeeze(
            1
        )  # (B*S,)

        start_times = start_times_flat.reshape(B, S)  # (B, S)

        # Clamp to [t, T_limit]
        start_times = torch.maximum(start_times, self.t.unsqueeze(1))
        start_times = torch.minimum(start_times, self.T_limit.unsqueeze(1))

        return start_times

    def _get_obs(self) -> Dict[str, torch.Tensor]:
        """Build observation dictionary (GPU tensors)."""
        B = self.batch_size

        # Jobs tensor
        jobs = self.p_subset.float().unsqueeze(-1)  # (B, N_job_pad, 1)

        # Job availability/mask (float 0/1 where 1 means valid/available)
        # Note: PPO runner/eval convert this to the model's bool-invalid convention.
        job_mask = self.job_available.clone()
        job_available = job_mask

        # Periods tensor: current-period lookahead window
        # periods[..., :] = [duration, price, start_slot, is_current]
        K_local = self.K_period_lookahead
        periods = torch.zeros(
            (B, K_local, self.F_period),
            dtype=torch.float32,
            device=self.device,
        )

        # Compute per-instance current period and gather a lookahead window
        current_period = self._compute_current_period().long()  # (B,)
        offsets = torch.arange(K_local, device=self.device).long()  # (K_local,)
        target_period = current_period.unsqueeze(1) + offsets.unsqueeze(
            0
        )  # (B, K_local)

        # Valid token if it points to an actual period for that instance
        # K is per-instance number of periods.
        valid_period = target_period < self.K.unsqueeze(1)  # (B, K_local) bool

        # Clamp gather indices to avoid OOB even for invalid tokens
        target_clamped = torch.clamp(target_period, 0, self.K_pad - 1)

        durations = torch.gather(self.Tk, 1, target_clamped).float()  # (B, K_local)
        prices = torch.gather(self.ck, 1, target_clamped).float()  # (B, K_local)
        starts = torch.gather(
            self.period_starts, 1, target_clamped
        ).float()  # (B, K_local)

        valid_f = valid_period.float()
        periods[:, :, 0] = durations * valid_f
        periods[:, :, 1] = prices * valid_f
        periods[:, :, 2] = starts * valid_f
        # is_current for the first token (only if valid)
        periods[:, 0, 3] = valid_period[:, 0].float()

        # Period mask in env convention: float 0/1 where 1 means valid token
        # Runner/eval convert to bool-invalid convention for the model.
        period_mask = valid_period.float()

        # Context
        remaining_work = (self.p_subset.float() * job_available).sum(dim=1)

        # Match CPU semantics for ctx[4]/ctx[5]: price stats beyond lookahead window
        # window_end = min(current_period + K_period_lookahead, K)
        window_end = torch.minimum(
            (current_period + K_local).to(torch.int32), self.K
        ).to(
            torch.int32
        )  # (B,)
        idx = torch.arange(self.K_pad, device=self.device).unsqueeze(0)  # (1, K_pad)
        beyond = (idx >= window_end.unsqueeze(1)) & (idx < self.K.unsqueeze(1))

        Tk_f = self.Tk.float()
        ck_f = self.ck.float()
        dur_beyond = Tk_f * beyond.float()
        total_dur = dur_beyond.sum(dim=1)
        weighted = (dur_beyond * ck_f).sum(dim=1)
        avg_price_after = torch.where(
            total_dur > 0,
            weighted / (total_dur + 1e-8),
            torch.zeros_like(total_dur),
        )

        any_beyond = beyond.any(dim=1)
        masked_ck = ck_f.masked_fill(~beyond, float("inf"))
        min_price_after = masked_ck.min(dim=1).values
        min_price_after = torch.where(
            any_beyond,
            min_price_after,
            torch.zeros_like(min_price_after),
        )

        ctx = torch.stack(
            [
                self.t.float(),
                self.T_limit.float(),
                remaining_work,
                self.e_single.float(),
                avg_price_after,
                min_price_after,
            ],
            dim=1,
        )

        # Action mask: check deadline feasibility for each (job, slack) pair
        # Env's action feasibility uses availability (1 = available)
        action_mask = self._compute_action_mask(job_available)
        # Done envs must have no valid actions
        action_mask = action_mask * (~self.done_mask).float().unsqueeze(1)

        return {
            "jobs": jobs,
            "periods": periods,
            "ctx": ctx,
            "job_mask": job_mask,
            "period_mask": period_mask,
            "action_mask": action_mask,
        }

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """
        Take a batched step.

        Args:
            actions: (B,) tensor of actions

        Returns:
            obs: Observation dictionary
            rewards: (B,) tensor of rewards
            dones: (B,) tensor of done flags
            info: Additional info dict
        """
        B = self.batch_size

        # Decode actions
        job_ids = actions // self.num_slack_choices
        slack_ids = actions % self.num_slack_choices

        # Validate actions: check job_ids refer to available jobs
        # Gather job availability for selected jobs
        job_available_selected = torch.gather(
            self.job_available, 1, job_ids.unsqueeze(1).long()
        ).squeeze(1)

        # For active (non-done) envs, selected job must be available
        active = ~self.done_mask
        invalid_action = active & (job_available_selected < 0.5)

        if invalid_action.any():
            # Invalid actions indicate a policy/mask bug.
            # Safer than clamping (which changes the MDP) is to terminate those envs
            # without applying any state transition.
            import warnings

            warnings.warn(
                f"Invalid actions detected: {invalid_action.sum().item()} envs selected unavailable jobs. "
                "Marking them done with zero reward and no state update."
            )
            self.done_mask = self.done_mask | invalid_action
            # Make them terminal with no available jobs/actions
            self.job_available[invalid_action] = 0.0
            active = ~self.done_mask

        # Get processing times for selected jobs
        # Use gather to get p[job_id] for each instance
        p = torch.gather(self.p_subset, 1, job_ids.unsqueeze(1).long()).squeeze(1)

        # Compute start times based on slack variant
        if self.env_config.slack_variant == SlackVariant.NO_SLACK:
            start_times = self.t.clone()
        elif self.env_config.slack_variant == SlackVariant.SHORT_SLACK:
            if self.env_config.short_slack_spec is not None:
                # SHORT_SLACK uses period offsets, not slot offsets
                # Get the period offset for each instance's slack_id
                slack_options = torch.tensor(
                    self.env_config.short_slack_spec.slack_options,
                    dtype=torch.int32,
                    device=self.device,
                )
                period_offsets = slack_options[slack_ids.long()]  # (B,)
                # Use period-aligned computation
                start_times = self._compute_period_aligned_starts(period_offsets)
            else:
                start_times = self.t.clone()
        elif self.env_config.slack_variant == SlackVariant.COARSE_TO_FINE:
            # Coarse-to-fine: decode slack_id to period offset, then to start time
            if self.env_config.c2f_slack_spec is not None:
                # Vectorized computation of period offsets
                period_offsets = self._compute_c2f_period_offsets(slack_ids)
                # Find current period for each instance and compute target start
                start_times = self._compute_period_aligned_starts(period_offsets)
            else:
                start_times = self.t.clone()
        elif self.env_config.slack_variant == SlackVariant.FULL_SLACK:
            # FULL_SLACK: slack_id is period offset from current period
            # (matching config.FullSlackSpec which defines K_full_max as period count)
            period_offsets = slack_ids.int()  # (B,)
            start_times = self._compute_period_aligned_starts(period_offsets)
        else:
            # Default: no slack
            start_times = self.t.clone()

        end_times = start_times + p
        # Clamp end_times to T_limit (actions should be masked, but safety clamp)
        end_times = torch.minimum(end_times, self.T_limit)

        # Compute active mask (envs that haven't finished yet)
        active = ~self.done_mask  # (B,)

        # Compute energy for each instance using cumulative sum for vectorized range sums
        # Build cumsum: cumsum[i] = sum(ct[0:i]), so sum(ct[s:e]) = cumsum[e] - cumsum[s]
        ct_float = self.ct.float()  # (B, T_max_pad)
        ct_cumsum = torch.zeros(
            (B, self.T_max_pad + 1), dtype=torch.float32, device=self.device
        )
        ct_cumsum[:, 1:] = torch.cumsum(ct_float, dim=1)

        # Clamp indices to valid range
        s_clamped = torch.clamp(start_times.long(), 0, self.T_max_pad)
        e_clamped = torch.clamp(end_times.long(), 0, self.T_max_pad)

        # Gather cumsum values at start and end indices
        cumsum_at_end = torch.gather(ct_cumsum, 1, e_clamped.unsqueeze(1)).squeeze(1)
        cumsum_at_start = torch.gather(ct_cumsum, 1, s_clamped.unsqueeze(1)).squeeze(1)
        energy_sums = cumsum_at_end - cumsum_at_start  # (B,)

        # Compute energies (only for active envs)
        energies = self.e_single.float() * energy_sums
        energies = energies * active.float()  # Zero out done envs

        self.total_energy += energies

        # Update time ONLY for active environments (done envs keep their final time)
        self.t = torch.where(active, end_times.int(), self.t)

        # Mark jobs as done ONLY for active environments
        # Create a one-hot mask for the selected jobs and subtract from availability
        job_one_hot = torch.zeros(
            (B, self.N_job_pad), dtype=torch.float32, device=self.device
        )
        job_one_hot.scatter_(1, job_ids.unsqueeze(1).long(), 1.0)
        # Only update job_available for active envs
        job_one_hot = job_one_hot * active.float().unsqueeze(1)
        self.job_available = self.job_available - job_one_hot
        self.job_available = torch.clamp(self.job_available, 0, 1)

        # Check if done (no more jobs available)
        n_remaining = self.job_available.sum(dim=1)
        newly_done = (
            n_remaining == 0
        ) & active  # Only active envs can become newly done
        self.done_mask = self.done_mask | newly_done

        # Rewards
        rewards = -energies

        obs = self._get_obs()
        info = {"total_energy": self.total_energy.clone()}

        return obs, rewards, self.done_mask, info

    def get_final_metrics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get final metrics for all instances."""
        return self.total_energy, self.t.float()


if __name__ == "__main__":
    # Test environment
    print("Testing SingleMachinePeriodEnv...")
    print("=" * 60)

    import random
    from .sm_benchmark_data import generate_single_machine_episode

    # Create configs
    env_config = EnvConfig()
    data_config = DataConfig()

    # Test NO_SLACK variant
    print("\n1. Testing NO_SLACK variant...")
    env_config.slack_variant = SlackVariant.NO_SLACK
    env = SingleMachinePeriodEnv(env_config, data_config)

    # Generate an episode
    rng = random.Random(42)
    episode = generate_single_machine_episode(data_config, rng)
    print(f"   Episode: n_jobs={episode.n_jobs}, T_limit={episode.T_limit}")

    # Reset environment
    obs, info = env.reset(options={"episode": episode})
    print(f"   Initial obs shapes:")
    for k, v in obs.items():
        print(f"      {k}: {v.shape}")
    print(f"   Info: {info}")

    # Take steps until done
    total_reward = 0
    step_count = 0
    while True:
        # Get valid actions
        valid_actions = np.where(obs["action_mask"] > 0)[0]
        if len(valid_actions) == 0:
            print("   No valid actions!")
            break

        # Take first valid action
        action = valid_actions[0]
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        if done:
            break

    print(f"   Episode finished in {step_count} steps")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Total energy: {info['total_energy']:.2f}")

    # Test SHORT_SLACK variant
    print("\n2. Testing SHORT_SLACK variant...")
    env_config.slack_variant = SlackVariant.SHORT_SLACK
    env_config.short_slack_spec = ShortSlackSpec(slack_options=[0, 1, 2, 3, 5])

    env = SingleMachinePeriodEnv(env_config, data_config)
    obs, info = env.reset(options={"episode": episode})

    print(f"   Num slack choices: {env.num_slack_choices}")
    print(f"   Action dim: {env.action_dim}")

    # Test PERIOD_ALIGNED variant
    print("\n3. Testing PERIOD_ALIGNED variant...")
    env_config.slack_variant = SlackVariant.PERIOD_ALIGNED
    env_config.period_aligned_spec = PeriodAlignedSlackSpec(max_periods_lookahead=5)

    env = SingleMachinePeriodEnv(env_config, data_config)
    obs, info = env.reset(options={"episode": episode})

    print(f"   Num slack choices: {env.num_slack_choices}")
    print(f"   Action dim: {env.action_dim}")

    # Test GPU batch environment if torch available
    if TORCH_AVAILABLE:
        print("\n4. Testing GPUBatchSingleMachinePeriodEnv...")
        from .sm_benchmark_data import generate_episode_batch

        env_config.slack_variant = SlackVariant.NO_SLACK
        batch_size = 8
        batch_env = GPUBatchSingleMachinePeriodEnv(
            batch_size=batch_size,
            env_config=env_config,
        )

        # Generate batch
        batch_data = generate_episode_batch(batch_size, data_config, seed=42)
        print(f"   Batch data keys: {list(batch_data.keys())}")

        # Reset
        obs = batch_env.reset(batch_data)
        print(f"   Batch obs shapes:")
        for k, v in obs.items():
            print(f"      {k}: {v.shape}")

        # Take a step
        actions = torch.zeros(batch_size, dtype=torch.int64, device=batch_env.device)
        obs, rewards, dones, info = batch_env.step(actions)
        print(f"   After step: rewards={rewards}, dones={dones}")

    print("\n" + "=" * 60)
    print("All environment tests passed!")

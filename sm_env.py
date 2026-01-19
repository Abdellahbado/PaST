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

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, List, Union, Any, TYPE_CHECKING
from dataclasses import dataclass

# NOTE: We intentionally import torch in a TYPE_CHECKING-safe way.
# Pyright/Pylance treats a name as a "variable" (not a module) if it can be
# assigned (e.g., torch = None), which then breaks type annotations like
# `torch.Tensor` with "Variable not allowed in type expression".
if TYPE_CHECKING:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
else:
    try:
        import torch  # type: ignore

        TORCH_AVAILABLE = True
    except ImportError:
        torch = None  # type: ignore[assignment]
        TORCH_AVAILABLE = False

from .config import (
    EnvConfig,
    SlackType,
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
ENV_VERSION = "1.1-SM"  # Updated for price-family variant

# Only print loading message once (not in worker subprocesses)
_ENV_LOADING_PRINTED = False
if os.environ.get("PAST_QUIET_ENV", "0") != "1":
    import __main__

    if hasattr(__main__, "__file__") or not hasattr(__main__, "__spec__"):
        # Main process or interactive - print once
        if not _ENV_LOADING_PRINTED:
            print(
                f"[SingleMachinePeriodEnv v{ENV_VERSION}] Loading single-machine period-aware environment..."
            )
            _ENV_LOADING_PRINTED = True


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
        # Base (6): [t, T_limit, remaining_work, e_single, avg_price_after_window, min_price_after_window]
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

        # Optional: price-family identification features for enhanced variants.
        # If enabled, append:
        #  - q25,q50,q75 (3)
        #  - delta_to_next_slot_in_family[0..3] (4)
        # Total F_ctx = 13.
        if self.env_config.use_price_families and self.F_ctx >= 13:
            # Compute per-episode quantiles over valid horizon.
            ct_valid = np.asarray(ep.ct[: int(ep.T_max)], dtype=np.float32)
            if ct_valid.size > 0:
                q25, q50, q75 = np.quantile(ct_valid, [0.25, 0.5, 0.75]).astype(
                    np.float32
                )
            else:
                q25 = q50 = q75 = np.float32(0.0)

            ctx[6] = q25
            ctx[7] = q50
            ctx[8] = q75

            # Compute family assignment per slot and delta to next slot of each family.
            # Valid search region: [t, min(T_limit, T_max)).
            t_now = int(self.t)
            t_end = int(min(ep.T_limit, ep.T_max))
            slot_idx = np.arange(int(ep.T_max), dtype=np.int32)
            valid = (slot_idx >= t_now) & (slot_idx < t_end)

            family = np.zeros_like(ct_valid, dtype=np.int32)
            family = np.where(ct_valid > q75, 3, family)
            family = np.where((ct_valid > q50) & (ct_valid <= q75), 2, family)
            family = np.where((ct_valid > q25) & (ct_valid <= q50), 1, family)

            large = np.float32(ep.T_max) if int(ep.T_max) > 0 else np.float32(0.0)
            for f in range(4):
                cand = np.where(valid & (family == f))[0]
                if cand.size == 0:
                    ctx[9 + f] = large
                else:
                    ctx[9 + f] = np.float32(cand[0] - t_now)

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

        # Price quantiles for price-family variant: [q25, q50, q75]
        self.price_q = torch.zeros((B, 3), dtype=torch.float32, device=self.device)

        # Duration-aware family caches (computed on reset/reset_indices when enabled)
        # - duration_q: (B, 3) quantiles over avg window costs [q25, q50, q75]
        # - duration_families: (B, N_job_pad, T_max_pad) family id per (job,start)
        self.duration_q = torch.zeros((B, 3), dtype=torch.float32, device=self.device)
        self.duration_families = torch.zeros(
            (B, self.N_job_pad, self.T_max_pad), dtype=torch.int32, device=self.device
        )

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

        # Price quantiles for price-family variant
        if "price_q" in batch_data:
            self.price_q[:] = to_tensor(batch_data["price_q"], torch.float32)
        else:
            # Compute quantiles on the fly if not provided
            ct_np = (
                batch_data["ct"]
                if isinstance(batch_data["ct"], np.ndarray)
                else batch_data["ct"].cpu().numpy()
            )
            t_max_np = (
                batch_data["T_max"]
                if isinstance(batch_data["T_max"], np.ndarray)
                else batch_data["T_max"].cpu().numpy()
            )
            price_q = np.zeros((B, 3), dtype=np.float32)
            for i in range(B):
                t_max_i = int(t_max_np[i])
                if t_max_i > 0:
                    ct_valid = ct_np[i, :t_max_i]
                    price_q[i] = np.quantile(ct_valid, [0.25, 0.5, 0.75])
            self.price_q[:] = to_tensor(price_q, torch.float32)

        # Initialize dynamic state
        self.t.zero_()
        self.total_energy.zero_()

        # Job availability (1 where job exists, 0 otherwise)
        self.job_available[:] = to_tensor(batch_data["job_mask"], torch.float32)

        # Precompute duration-aware family cache if enabled.
        # This makes family semantics stable within an episode and avoids recomputing
        # per-step quantiles (which is expensive and makes family IDs non-stationary).
        if getattr(self.env_config, "use_duration_aware_families", False):
            self._compute_duration_aware_family_cache(
                ct=self.ct,
                p_all=self.p_subset,
                job_mask=self.job_available,
                T_limit=self.T_limit,
                T_max=self.T_max,
                out_quantiles=self.duration_q,
                out_families=self.duration_families,
            )

        # Done mask
        self.done_mask = torch.zeros((B,), dtype=torch.bool, device=self.device)

        return self._get_obs()

    def reset_indices(
        self,
        batch_data: Dict[str, Union[np.ndarray, torch.Tensor]],
        indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Reset only a subset of environments (vectorized env semantics).

        This replaces episode data and clears dynamic state for the specified
        batch indices, leaving other indices untouched.

        Args:
            batch_data: Batch dictionary (same schema as reset()) with batch_size == len(indices)
            indices: 1D int tensor of batch indices to reset (on any device)

        Returns:
            Current observation dictionary after applying the partial reset.
        """
        if indices.numel() == 0:
            return self._get_obs()

        # Ensure indices are on-device and long
        idx = indices.to(device=self.device, dtype=torch.long)
        sub_B = int(idx.numel())

        # Helper to convert to tensor on device
        def to_tensor(x, dtype=torch.float32):
            if isinstance(x, torch.Tensor):
                return x.to(device=self.device, dtype=dtype)
            return torch.tensor(x, dtype=dtype, device=self.device)

        # --- Episode/static data ---
        self.p_subset.index_copy_(
            0, idx, to_tensor(batch_data["p_subset"], torch.int32)
        )
        self.n_jobs.index_copy_(0, idx, to_tensor(batch_data["n_jobs"], torch.int32))
        self.T_max.index_copy_(0, idx, to_tensor(batch_data["T_max"], torch.int32))
        self.T_limit.index_copy_(0, idx, to_tensor(batch_data["T_limit"], torch.int32))
        self.e_single.index_copy_(
            0, idx, to_tensor(batch_data["e_single"], torch.int32)
        )

        # Price/period data
        ct = batch_data["ct"]
        ct_t = to_tensor(ct, torch.int32)
        L_ct = int(min(ct_t.shape[1], self.T_max_pad))
        # Clear then copy (avoid leaving stale tail when ct shorter)
        self.ct.index_fill_(0, idx, 0)
        self.ct[idx, :L_ct] = ct_t[:, :L_ct]

        Tk_t = to_tensor(batch_data["Tk"], torch.int32)
        ck_t = to_tensor(batch_data["ck"], torch.int32)
        ps_t = to_tensor(batch_data["period_starts"], torch.int32)
        L_k = int(min(Tk_t.shape[1], self.K_pad))
        self.Tk.index_fill_(0, idx, 0)
        self.ck.index_fill_(0, idx, 0)
        self.period_starts.index_fill_(0, idx, 0)
        self.Tk[idx, :L_k] = Tk_t[:, :L_k]
        self.ck[idx, :L_k] = ck_t[:, :L_k]
        self.period_starts[idx, :L_k] = ps_t[:, :L_k]
        self.K.index_copy_(0, idx, to_tensor(batch_data["K"], torch.int32))

        # Price quantiles for price-family variant
        if "price_q" in batch_data:
            pq_t = to_tensor(batch_data["price_q"], torch.float32)
            if pq_t.shape[0] != sub_B:
                raise ValueError(
                    f"batch_data['price_q'] batch dim {pq_t.shape[0]} != len(indices) {sub_B}"
                )
            self.price_q.index_copy_(0, idx, pq_t)
        else:
            # Keep existing reset() fallback behavior (CPU numpy quantiles), but do it only
            # for the subset. This path should be avoided in training.
            ct_np = ct if isinstance(ct, np.ndarray) else ct.cpu().numpy()
            t_max_np = (
                batch_data["T_max"]
                if isinstance(batch_data["T_max"], np.ndarray)
                else batch_data["T_max"].cpu().numpy()
            )
            price_q = np.zeros((sub_B, 3), dtype=np.float32)
            for i in range(sub_B):
                t_max_i = int(t_max_np[i])
                if t_max_i > 0:
                    ct_valid = ct_np[i, :t_max_i]
                    price_q[i] = np.quantile(ct_valid, [0.25, 0.5, 0.75])
            self.price_q.index_copy_(0, idx, to_tensor(price_q, torch.float32))

        # --- Dynamic state ---
        # Reset time, energy, availability, and done flag for those indices.
        self.t.index_fill_(0, idx, 0)
        self.total_energy.index_fill_(0, idx, 0)
        self.job_available.index_fill_(0, idx, 0.0)
        self.job_available.index_copy_(
            0, idx, to_tensor(batch_data["job_mask"], torch.float32)
        )

        # Refresh duration-aware family cache for the reset subset.
        if getattr(self.env_config, "use_duration_aware_families", False):
            ct_sub = self.ct.index_select(0, idx)
            p_sub = self.p_subset.index_select(0, idx)
            jm_sub = self.job_available.index_select(0, idx)
            tl_sub = self.T_limit.index_select(0, idx)
            tm_sub = self.T_max.index_select(0, idx)
            q_sub = torch.zeros((sub_B, 3), dtype=torch.float32, device=self.device)
            fam_sub = torch.zeros(
                (sub_B, self.N_job_pad, self.T_max_pad),
                dtype=torch.int32,
                device=self.device,
            )
            self._compute_duration_aware_family_cache(
                ct=ct_sub,
                p_all=p_sub,
                job_mask=jm_sub,
                T_limit=tl_sub,
                T_max=tm_sub,
                out_quantiles=q_sub,
                out_families=fam_sub,
            )
            self.duration_q.index_copy_(0, idx, q_sub)
            self.duration_families.index_copy_(0, idx, fam_sub)
        if self.done_mask is None:
            self.done_mask = torch.zeros(
                (self.batch_size,), dtype=torch.bool, device=self.device
            )
        self.done_mask.index_fill_(0, idx, False)

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

    # =========================================================================
    # Price-Family Variant Helpers
    # =========================================================================

    def _compute_slot_families(self) -> torch.Tensor:
        """
        Compute price family for each time slot based on per-episode quantiles.

        Family assignment using quartiles [q25, q50, q75]:
        - Family 0: price <= q25 (cheapest)
        - Family 1: q25 < price <= q50
        - Family 2: q50 < price <= q75
        - Family 3: price > q75 (most expensive)

        Returns:
            (B, T_max_pad) tensor of family indices (0-3)
        """
        ct_float = self.ct.float()  # (B, T_max_pad)
        q25 = self.price_q[:, 0:1]  # (B, 1)
        q50 = self.price_q[:, 1:2]  # (B, 1)
        q75 = self.price_q[:, 2:3]  # (B, 1)

        # Assign families based on quantile thresholds
        # Note: we use <= for lower thresholds to handle ties properly
        family = torch.zeros_like(self.ct, dtype=torch.int32)
        family = torch.where(
            ct_float > q75, torch.tensor(3, device=self.device), family
        )
        family = torch.where(
            (ct_float > q50) & (ct_float <= q75),
            torch.tensor(2, device=self.device),
            family,
        )
        family = torch.where(
            (ct_float > q25) & (ct_float <= q50),
            torch.tensor(1, device=self.device),
            family,
        )
        # Family 0 is default (price <= q25)

        return family

    def _compute_family_action_mask(self, job_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute action mask for price-family variant.

        An action (job_id, family_id) is valid if:
        1. job_id refers to an available job
        2. There exists at least one feasible start time whose slot is in family_id

        Args:
            job_mask: (B, N_job_pad) tensor of job availability

        Returns:
            (B, action_dim) tensor where 1 = valid, 0 = invalid
        """
        B = self.batch_size
        N = self.N_job_pad
        F = self.num_slack_choices  # num_price_families

        # Compute slot families
        slot_families = self._compute_slot_families()  # (B, T_max_pad)

        # Processing times for all jobs: (B, N)
        p_all = self.p_subset.int()

        # For each (job, family), check if there's any feasible start time in that family.
        #
        # Training-critical: also enforce a completion-feasibility bound so we do not allow
        # actions that "wait too long" and make it impossible to schedule remaining work.
        #
        # If we start a job at time s >= t_now, the earliest possible completion time
        # (with no additional idle) is s + remaining_work. Therefore we require:
        #   s + remaining_work <= T_limit
        # This eliminates dead-ends where the env would later have no valid actions.

        # Create mask for valid slots: (B, T_max_pad)
        slot_indices = torch.arange(self.T_max_pad, device=self.device).unsqueeze(
            0
        )  # (1, T_max_pad)
        valid_start_slots = slot_indices >= self.t.unsqueeze(
            1
        )  # (B, T_max_pad) - slot >= t_now

        # Remaining work in slots for each instance
        remaining_work = (
            (p_all.float() * job_mask.float()).sum(dim=1).to(torch.int32)
        )  # (B,)

        # Completion bound: slot + remaining_work <= T_limit
        completion_ok = slot_indices.to(torch.int32) + remaining_work.unsqueeze(
            1
        ) <= self.T_limit.unsqueeze(1).to(
            torch.int32
        )  # (B, T_max_pad)

        # For each job and slot, check if job can finish by deadline
        # end_time[b, j, s] = s + p[b, j]
        # (B, N, T_max_pad)
        slot_indices_expanded = slot_indices.unsqueeze(1)  # (1, 1, T_max_pad)
        p_expanded = p_all.unsqueeze(2)  # (B, N, 1)
        end_times = slot_indices_expanded + p_expanded  # (B, N, T_max_pad)

        # Feasibility mask: end_time <= T_limit AND start >= t_now
        T_limit_expanded = self.T_limit.unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
        feasible = (end_times <= T_limit_expanded) & valid_start_slots.unsqueeze(1)
        feasible = feasible & completion_ok.unsqueeze(1)  # (B, N, T_max_pad)

        # For each family, check if there's a feasible slot in that family
        # Shape: (B, N, F)
        action_mask = torch.zeros((B, N, F), dtype=torch.float32, device=self.device)

        for f in range(F):
            # Slots in this family: (B, T_max_pad)
            in_family = slot_families == f
            # Feasible slots in this family for each job: (B, N, T_max_pad)
            feasible_in_family = feasible & in_family.unsqueeze(1)
            # Any feasible slot in this family: (B, N)
            has_feasible = feasible_in_family.any(dim=2)
            action_mask[:, :, f] = has_feasible.float()

        # Combine with job availability
        job_available = job_mask.unsqueeze(2)  # (B, N, 1)
        action_mask = action_mask * job_available

        # Reshape to action_dim: action = job_id * F + family_id
        action_mask = action_mask.reshape(B, N * F)

        return action_mask

    def _compute_family_start_times(self, family_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute earliest feasible start time for each instance given family choice.

        Args:
            family_ids: (B,) tensor of chosen family indices

        Returns:
            (B,) tensor of start times (earliest feasible in chosen family)
        """
        B = self.batch_size

        # Compute slot families
        slot_families = self._compute_slot_families()  # (B, T_max_pad)

        # Slot indices
        slot_indices = torch.arange(self.T_max_pad, device=self.device).unsqueeze(
            0
        )  # (1, T_max_pad)

        # Valid start slots (>= t_now)
        valid_slots = slot_indices >= self.t.unsqueeze(1)  # (B, T_max_pad)

        # Slots in chosen family
        family_ids_expanded = family_ids.unsqueeze(1)  # (B, 1)
        in_family = slot_families == family_ids_expanded  # (B, T_max_pad)

        # Candidate slots: valid AND in family
        candidates = valid_slots & in_family  # (B, T_max_pad)

        # Find earliest slot (argmax on bool finds first True)
        # Use masked_fill to set non-candidates to a large value, then argmin
        large_val = self.T_max_pad + 1
        slot_indices_masked = torch.where(
            candidates,
            slot_indices.expand(B, -1),
            torch.full((B, self.T_max_pad), large_val, device=self.device),
        )
        start_times = slot_indices_masked.min(dim=1).values  # (B,)

        # Clamp to T_limit (safety, shouldn't happen if mask is correct)
        start_times = torch.minimum(start_times, self.T_limit)

        return start_times

    def _compute_family_start_times_for_job(
        self, job_ids: torch.Tensor, family_ids: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute earliest feasible start time for specific jobs and families.

        Args:
            job_ids: (B,) tensor of chosen job indices
            family_ids: (B,) tensor of chosen family indices
            p: (B,) tensor of job processing times

        Returns:
            (B,) tensor of start times (earliest feasible in chosen family for that job)
        """
        B = self.batch_size

        # Compute slot families
        slot_families = self._compute_slot_families()  # (B, T_max_pad)

        # Slot indices
        slot_indices = torch.arange(self.T_max_pad, device=self.device).unsqueeze(
            0
        )  # (1, T_max_pad)

        # Valid start slots (>= t_now)
        valid_slots = slot_indices >= self.t.unsqueeze(1)  # (B, T_max_pad)

        # Slots in chosen family
        family_ids_expanded = family_ids.unsqueeze(1)  # (B, 1)
        in_family = slot_families == family_ids_expanded  # (B, T_max_pad)

        # Deadline feasibility: slot + p <= T_limit
        end_times = slot_indices + p.unsqueeze(1)  # (B, T_max_pad)
        deadline_ok = end_times <= self.T_limit.unsqueeze(1)  # (B, T_max_pad)

        # Candidate slots: valid AND in family AND meets deadline
        candidates = valid_slots & in_family & deadline_ok  # (B, T_max_pad)

        # Find earliest slot
        large_val = self.T_max_pad + 1
        slot_indices_masked = torch.where(
            candidates,
            slot_indices.expand(B, -1),
            torch.full((B, self.T_max_pad), large_val, device=self.device),
        )
        start_times = slot_indices_masked.min(dim=1).values  # (B,)

        # Clamp to T_limit (safety)
        start_times = torch.minimum(start_times, self.T_limit)

        return start_times

    def _compute_family_best_start_times_for_job(
        self, job_ids: torch.Tensor, family_ids: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute best (minimum energy) feasible start time for chosen job and family.

        Uses the same feasibility criteria as _compute_family_action_mask:
        start >= t, end <= T_limit, and completion_ok (start + remaining_work <= T_limit).

        Args:
            job_ids: (B,) tensor of chosen job indices
            family_ids: (B,) tensor of chosen family indices
            p: (B,) tensor of job processing times

        Returns:
            (B,) tensor of start times (min-energy feasible in chosen family)
        """
        B = self.batch_size

        # Compute slot families
        slot_families = self._compute_slot_families()  # (B, T_max_pad)

        # Slot indices
        slot_indices = torch.arange(self.T_max_pad, device=self.device).unsqueeze(0)

        # Valid start slots (>= t_now)
        valid_slots = slot_indices >= self.t.unsqueeze(1)  # (B, T_max_pad)

        # Slots in chosen family
        family_ids_expanded = family_ids.unsqueeze(1)  # (B, 1)
        in_family = slot_families == family_ids_expanded  # (B, T_max_pad)

        # Deadline feasibility: slot + p <= T_limit
        end_times = slot_indices + p.unsqueeze(1)  # (B, T_max_pad)
        deadline_ok = end_times <= self.T_limit.unsqueeze(1)  # (B, T_max_pad)

        # Completion bound (prevents waiting into infeasible region)
        remaining_work = (
            (self.p_subset.float() * self.job_available.float())
            .sum(dim=1)
            .to(torch.int32)
        )  # (B,)
        completion_ok = slot_indices.to(torch.int32) + remaining_work.unsqueeze(
            1
        ) <= self.T_limit.unsqueeze(1).to(torch.int32)

        # Candidate slots: valid AND in family AND meets deadline AND completion_ok
        candidates = (
            valid_slots & in_family & deadline_ok & completion_ok
        )  # (B, T_max_pad)

        # Compute window energy cost for each candidate start
        # cost(s) = sum(ct[s : s+p]) (e_single is constant, so omitted)
        ct_float = self.ct.float()  # (B, T_max_pad)
        ct_cumsum = torch.zeros(
            (B, self.T_max_pad + 1), dtype=torch.float32, device=self.device
        )
        ct_cumsum[:, 1:] = torch.cumsum(ct_float, dim=1)

        end_idx = torch.clamp(end_times.long(), 0, self.T_max_pad)
        start_idx = torch.clamp(slot_indices.long(), 0, self.T_max_pad)

        cumsum_at_end = torch.gather(ct_cumsum, 1, end_idx)
        cumsum_at_start = torch.gather(ct_cumsum, 1, start_idx)
        window_cost = cumsum_at_end - cumsum_at_start  # (B, T_max_pad)

        # Mask non-candidates with +inf, then take argmin
        inf = torch.tensor(float("inf"), device=self.device)
        masked_cost = torch.where(candidates, window_cost, inf)
        best_idx = masked_cost.argmin(dim=1)  # (B,)

        has_any = candidates.any(dim=1)
        best_start = torch.where(has_any, best_idx, self.T_limit)

        # Clamp to T_limit (safety)
        best_start = torch.minimum(best_start, self.T_limit)

        return best_start

    # =========================================================================
    # Duration-Aware Family Variant Helpers
    # =========================================================================

    def _compute_window_costs_from(
        self, ct: torch.Tensor, p_all: torch.Tensor
    ) -> torch.Tensor:
        """Compute avg window cost w(s,p)/p for all (job,start), for an arbitrary ct tensor.

        Args:
            ct: (B, T_max_pad) int/float tensor of per-slot prices
            p_all: (B, N) int tensor of processing times

        Returns:
            (B, N, T_max_pad) float32 avg window costs. Invalid (s+p > T_max_pad) -> inf.
        """
        B = int(ct.shape[0])
        N = int(p_all.shape[1])
        T = int(ct.shape[1])

        ct_float = ct.float()
        cumsum = torch.zeros((B, T + 1), dtype=torch.float32, device=ct.device)
        cumsum[:, 1:] = torch.cumsum(ct_float, dim=1)

        slot_idx = torch.arange(T, device=ct.device).view(1, 1, T)  # (1,1,T)
        p_expanded = p_all.to(torch.int32).unsqueeze(2)  # (B,N,1)
        end_idx = slot_idx + p_expanded  # (B,N,T)

        end_idx_clamped = torch.clamp(end_idx, max=T).long()
        cumsum_expanded = cumsum.unsqueeze(1).expand(B, N, T + 1)  # (B,N,T+1)

        start_cumsum = cumsum[:, :T].unsqueeze(1).expand(B, N, T)
        end_cumsum = torch.gather(cumsum_expanded, 2, end_idx_clamped)

        window_cost = end_cumsum - start_cumsum
        p_safe = torch.clamp(p_expanded.float(), min=1.0)
        avg_window_cost = window_cost / p_safe

        invalid = end_idx > T
        avg_window_cost = torch.where(
            invalid,
            torch.full_like(avg_window_cost, float("inf")),
            avg_window_cost,
        )
        return avg_window_cost

    def _compute_window_costs(self, p_all: torch.Tensor) -> torch.Tensor:
        """
        Compute average window cost w(s,p)/p for all (job, start) pairs.

        Uses prefix sums for efficient O(N*T) computation instead of O(N*T*p).

        For each job j with processing time p[j], and each potential start s:
            avg_window_cost[j, s] = sum(ct[s : s+p[j]]) / p[j]

        Args:
            p_all: (B, N) tensor of processing times

        Returns:
            (B, N, T_max_pad) tensor of average window costs (float32).
            Invalid positions (s + p > T_max_pad) are set to inf.
        """
        return self._compute_window_costs_from(self.ct, p_all)

    def _compute_duration_aware_family_quantiles(
        self, avg_window_costs: torch.Tensor, feasible_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-episode quantile thresholds for duration-aware families.

        Quantiles are computed over ALL feasible (job, start) pairs in each episode,
        not per-job, so that family boundaries are consistent across jobs.

        Args:
            avg_window_costs: (B, N, T) tensor of average window costs
            feasible_mask: (B, N, T) boolean tensor of feasible (job, start) pairs

        Returns:
            (B, 3) tensor of quantile thresholds [q25, q50, q75]
        """
        B = int(avg_window_costs.shape[0])

        # Flatten to (B, N*T)
        flat_costs = avg_window_costs.view(B, -1)  # (B, N*T)
        flat_mask = feasible_mask.view(B, -1)  # (B, N*T)

        # Compute quantiles per episode
        quantiles = torch.zeros((B, 3), dtype=torch.float32, device=self.device)

        for b in range(B):
            valid_costs = flat_costs[b][flat_mask[b]]
            if valid_costs.numel() > 0:
                quantiles[b, 0] = torch.quantile(valid_costs, 0.25)
                quantiles[b, 1] = torch.quantile(valid_costs, 0.50)
                quantiles[b, 2] = torch.quantile(valid_costs, 0.75)
            else:
                # No feasible actions - use zeros (will be masked out anyway)
                quantiles[b] = 0.0

        return quantiles

    def _compute_duration_aware_family_cache(
        self,
        *,
        ct: torch.Tensor,
        p_all: torch.Tensor,
        job_mask: torch.Tensor,
        T_limit: torch.Tensor,
        T_max: torch.Tensor,
        out_quantiles: torch.Tensor,
        out_families: torch.Tensor,
    ) -> None:
        """Compute and store stable duration-aware families for an episode batch.

        This is intended to be called at reset/reset_indices only.

        Families are computed once per episode over all (job,start) windows that are
        feasible by deadline/horizon (no dynamic t or remaining-work constraint).
        This keeps family IDs stable across steps in an episode.
        """
        B = int(ct.shape[0])
        N = int(p_all.shape[1])
        T = int(ct.shape[1])

        p_int = p_all.to(torch.int32)
        jm = job_mask.float()

        avg_window_costs = self._compute_window_costs_from(ct, p_int)  # (B,N,T)

        slot = torch.arange(T, device=ct.device).view(1, 1, T)  # (1,1,T)
        p_exp = p_int.unsqueeze(2)  # (B,N,1)
        end = slot + p_exp  # (B,N,T)

        tl = T_limit.to(torch.int32).view(B, 1, 1)
        tm = T_max.to(torch.int32).view(B, 1, 1)

        # Static feasibility: within true horizon/deadline and for real (unpadded) jobs.
        start_ok = slot < tm
        end_ok = (end <= tl) & (end <= tm)
        job_ok = (jm.view(B, N, 1) > 0.5) & (p_exp > 0)
        feasible_static = start_ok.expand(B, N, T) & end_ok & job_ok

        q = self._compute_duration_aware_family_quantiles(
            avg_window_costs, feasible_static
        )
        fam = self._assign_duration_aware_families(avg_window_costs, q)

        out_quantiles.copy_(q)
        out_families.copy_(fam)

    def _assign_duration_aware_families(
        self, avg_window_costs: torch.Tensor, quantiles: torch.Tensor
    ) -> torch.Tensor:
        """
        Assign family indices to each (job, start) pair based on avg window cost.

        Family assignment using quartiles [q25, q50, q75]:
        - Family 0: cost <= q25 (cheapest average window cost)
        - Family 1: q25 < cost <= q50
        - Family 2: q50 < cost <= q75
        - Family 3: cost > q75 (most expensive)

        Args:
            avg_window_costs: (B, N, T) tensor of average window costs
            quantiles: (B, 3) tensor of [q25, q50, q75] thresholds

        Returns:
            (B, N, T) tensor of family indices (0-3)
        """
        q25 = quantiles[:, 0:1].unsqueeze(2)  # (B, 1, 1)
        q50 = quantiles[:, 1:2].unsqueeze(2)  # (B, 1, 1)
        q75 = quantiles[:, 2:3].unsqueeze(2)  # (B, 1, 1)

        # Default to family 0 (cheapest)
        families = torch.zeros_like(avg_window_costs, dtype=torch.int32)

        # Assign families based on quantile thresholds
        families = torch.where(
            avg_window_costs > q75, torch.tensor(3, device=self.device), families
        )
        families = torch.where(
            (avg_window_costs > q50) & (avg_window_costs <= q75),
            torch.tensor(2, device=self.device),
            families,
        )
        families = torch.where(
            (avg_window_costs > q25) & (avg_window_costs <= q50),
            torch.tensor(1, device=self.device),
            families,
        )
        # Family 0 is default (cost <= q25)

        return families

    def _compute_duration_aware_family_action_mask(
        self, job_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute action mask for duration-aware family variant.

        An action (job_id, family_id) is valid if:
        1. job_id refers to an available job
        2. There exists at least one feasible start time whose avg window cost
           falls into family_id

        This is more accurate than price-family because it accounts for the
        full energy cost over the job's duration, not just the start slot price.

        Args:
            job_mask: (B, N_job_pad) tensor of job availability

        Returns:
            (B, action_dim) tensor where 1 = valid, 0 = invalid
        """
        B = self.batch_size
        N = self.N_job_pad
        F = self.num_slack_choices  # num_price_families (4)
        T = self.T_max_pad

        # Processing times for all jobs: (B, N)
        p_all = self.p_subset.int()

        # --- Compute feasibility mask for all (job, start) pairs ---
        slot_indices = (
            torch.arange(T, device=self.device).unsqueeze(0).unsqueeze(0)
        )  # (1, 1, T)

        # Valid start slots (>= t_now): (B, 1, T)
        valid_start = slot_indices >= self.t.unsqueeze(1).unsqueeze(2)

        # End times: (B, N, T)
        p_expanded = p_all.unsqueeze(2)  # (B, N, 1)
        end_times = slot_indices + p_expanded  # (B, N, T)

        # Deadline feasibility: end_time <= T_limit
        T_limit_exp = self.T_limit.unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
        deadline_ok = end_times <= T_limit_exp  # (B, N, T)

        # Horizon feasibility: start and end must be within true T_max
        T_max_exp = self.T_max.unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
        start_ok = slot_indices < T_max_exp
        horizon_ok = end_times <= T_max_exp

        # Remaining work constraint: slot + remaining_work <= T_limit
        remaining_work = (p_all.float() * job_mask.float()).sum(dim=1)  # (B,)
        completion_ok = slot_indices.squeeze(1) + remaining_work.unsqueeze(
            1
        ).int() <= self.T_limit.unsqueeze(
            1
        )  # (B, T)

        # Combined feasibility: (B, N, T)
        feasible = (
            valid_start.expand(B, N, T)
            & start_ok.expand(B, N, T)
            & deadline_ok
            & horizon_ok
            & completion_ok.unsqueeze(1)
        )

        # Use precomputed stable family assignment (computed at reset/reset_indices).
        families = self.duration_families  # (B, N, T)

        # --- Build action mask ---
        action_mask = torch.zeros((B, N, F), dtype=torch.float32, device=self.device)

        for f in range(F):
            # (job, start) pairs in this family that are feasible: (B, N, T)
            in_family = families == f
            feasible_in_family = feasible & in_family
            # Any feasible start in this family for each job: (B, N)
            has_feasible = feasible_in_family.any(dim=2)
            action_mask[:, :, f] = has_feasible.float()

        # Combine with job availability
        job_available = job_mask.unsqueeze(2)  # (B, N, 1)
        action_mask = action_mask * job_available

        # Reshape to action_dim: action = job_id * F + family_id
        action_mask = action_mask.reshape(B, N * F)

        return action_mask

    def _compute_duration_aware_family_start_times_for_job(
        self, job_ids: torch.Tensor, family_ids: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute earliest feasible start time for specific jobs and families
        using duration-aware family assignment.

        CRITICAL: Must use the same logic as _compute_duration_aware_family_action_mask
        to ensure action mask and decoding are consistent.

        Args:
            job_ids: (B,) tensor of chosen job indices
            family_ids: (B,) tensor of chosen family indices
            p: (B,) tensor of job processing times

        Returns:
            (B,) tensor of start times (earliest feasible in chosen family)
        """
        B = self.batch_size
        N = self.N_job_pad
        T = self.T_max_pad

        # Use precomputed stable family assignment.
        families = self.duration_families  # (B, N, T)

        # --- Compute feasibility mask (MUST match _compute_duration_aware_family_action_mask) ---
        slot_indices = (
            torch.arange(T, device=self.device).unsqueeze(0).unsqueeze(0)
        )  # (1, 1, T)
        valid_start = slot_indices >= self.t.unsqueeze(1).unsqueeze(2)  # (B, 1, T)

        # Horizon bounds
        start_ok = slot_indices < self.T_max.unsqueeze(1).unsqueeze(2)

        # Deadline / horizon feasibility for the chosen job duration p
        end_times = slot_indices + p.unsqueeze(1).unsqueeze(2)  # (B, 1, T)
        deadline_ok = end_times <= self.T_limit.unsqueeze(1).unsqueeze(2)
        horizon_ok = end_times <= self.T_max.unsqueeze(1).unsqueeze(2)

        # Remaining-work bound (prevents waiting into infeasible region)
        remaining_work = (
            (self.p_subset.float() * self.job_available.float())
            .sum(dim=1)
            .to(torch.int32)
        )  # (B,)
        completion_ok = slot_indices.squeeze(1).to(
            torch.int32
        ) + remaining_work.unsqueeze(1) <= self.T_limit.unsqueeze(1).to(
            torch.int32
        )  # (B, T)

        # --- Extract info for selected jobs ---
        job_ids_long = job_ids.unsqueeze(1).unsqueeze(2).expand(B, 1, T).long()
        families_for_job = torch.gather(families, 1, job_ids_long).squeeze(1)  # (B, T)

        # Feasibility for the chosen job (B,T)
        feasible_for_job = (
            valid_start.squeeze(1)
            & start_ok.squeeze(1)
            & deadline_ok.squeeze(1)
            & horizon_ok.squeeze(1)
            & completion_ok
        )

        # Slots in chosen family that are feasible
        family_ids_expanded = family_ids.unsqueeze(1)  # (B, 1)
        in_family = families_for_job == family_ids_expanded  # (B, T)
        candidates = feasible_for_job & in_family  # (B, T)

        # Find earliest slot
        slot_1d = torch.arange(T, device=self.device).unsqueeze(0)  # (1, T)
        large_val = T + 1
        slot_masked = torch.where(
            candidates,
            slot_1d.expand(B, -1),
            torch.full((B, T), large_val, device=self.device),
        )
        start_times = slot_masked.min(dim=1).values  # (B,)

        # Clamp to T_limit (safety)
        start_times = torch.minimum(start_times, self.T_limit)

        return start_times

    def _compute_action_mask(self, job_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute deadline-aware action mask.

        An action (job_id, slack_id) is valid if:
        1. job_id refers to an available job (job_mask[job_id] == 1)
        2. The job can finish by T_limit given the slack choice

        For price-family variant: slack_id is family_id, and validity depends
        on whether there's any feasible start time in that family.

        For duration-aware family variant: same as price-family, but families
        are based on average window cost (w(s,p)/p) instead of start slot price.

        Args:
            job_mask: (B, N_job_pad) tensor of job availability

        Returns:
            (B, action_dim) tensor where 1 = valid, 0 = invalid
        """
        # Use duration-aware family mask if enabled (takes precedence)
        if (
            self.env_config.use_price_families
            and self.env_config.use_duration_aware_families
        ):
            return self._compute_duration_aware_family_action_mask(job_mask)

        # Use slot-price-based family mask if price-family variant is enabled
        if self.env_config.use_price_families:
            return self._compute_family_action_mask(job_mask)

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

        # Training-critical: completion-feasibility bound (prevents dead-ends).
        # If we choose to start a job at time s, earliest completion with no extra idle is
        # s + remaining_work. Require s + remaining_work <= T_limit.
        remaining_work = (
            (p_all.float() * job_mask.float()).sum(dim=1).to(torch.int32)
        )  # (B,)
        completion_ok = slack_start_times.to(torch.int32) + remaining_work.unsqueeze(
            1
        ) <= self.T_limit.to(torch.int32).unsqueeze(
            1
        )  # (B, S)
        feasible = feasible & completion_ok.unsqueeze(1)  # (B, N, S)

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

        # New configs use `slack_type` (SHORT / COARSE_TO_FINE / FULL).
        # Older codepaths use `slack_variant`. We prefer slack_type when present
        # to avoid mismatches when only slack_type is set.
        slack_type = getattr(self.env_config, "slack_type", None)
        if slack_type is not None:
            if slack_type == SlackType.SHORT:
                if self.env_config.short_slack_spec is not None:
                    slack_options = torch.tensor(
                        self.env_config.short_slack_spec.slack_options,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    # period_offsets: (B, S)
                    period_offsets = slack_options.unsqueeze(0).expand(B, S)
                    return self._compute_period_aligned_starts_batch(period_offsets)
                return self.t.unsqueeze(1).expand(B, S)

            if slack_type == SlackType.COARSE_TO_FINE:
                if self.env_config.c2f_slack_spec is not None:
                    slack_ids = torch.arange(S, device=self.device)  # (S,)
                    slack_ids_expanded = slack_ids.unsqueeze(0).expand(B, S)  # (B, S)
                    period_offsets = self._compute_c2f_period_offsets_batch(
                        slack_ids_expanded
                    )
                    return self._compute_period_aligned_starts_batch(period_offsets)
                return self.t.unsqueeze(1).expand(B, S)

            if slack_type == SlackType.FULL:
                slack_ids = torch.arange(
                    S, dtype=torch.int32, device=self.device
                )  # (S,)
                period_offsets = slack_ids.unsqueeze(0).expand(B, S)  # (B, S)
                return self._compute_period_aligned_starts_batch(period_offsets)

            # Unknown slack_type: default to start now
            return self.t.unsqueeze(1).expand(B, S)

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
        # Model expects jobs[..., F_job] where F_job comes from EnvConfig.
        # Backward-compatible behavior: feature 0 is processing time; feature 1 (if present)
        # is availability (0/1), which is a useful signal and matches the model's 2D job inputs.
        jobs = torch.zeros(
            (B, self.N_job_pad, self.F_job), dtype=torch.float32, device=self.device
        )
        jobs[..., 0] = self.p_subset.float()

        # Job availability/mask (float 0/1 where 1 means valid/available)
        # Note: PPO runner/eval convert this to the model's bool-invalid convention.
        job_mask = self.job_available.clone()
        job_available = job_mask

        if self.F_job >= 2:
            jobs[..., 1] = job_available

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

        ctx6 = torch.stack(
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

        # Optional: price-family identification features for enhanced variants.
        # Append:
        #  - q25,q50,q75 (3)
        #  - delta_to_next_slot_in_family[0..3] (4)
        # Total F_ctx = 13.
        if self.env_config.use_price_families and self.F_ctx >= 13:
            if getattr(self.env_config, "use_duration_aware_families", False):
                # Duration-aware ctx13:
                #  - duration_q: [q25,q50,q75] over avg window costs
                #  - delta_to_next_feasible_start_in_family[0..3]
                #
                # "Feasible" is defined consistently with the duration-aware masking/decoding:
                # start >= t, start < T_max, end <= T_limit, end <= T_max,
                # and completion_ok: start + remaining_work <= T_limit.

                # Shape helpers
                slot_idx_1d = torch.arange(
                    self.T_max_pad, device=self.device, dtype=torch.int32
                )
                slot_idx = slot_idx_1d.view(1, 1, -1)  # (1,1,T)

                t_now = self.t.view(B, 1, 1)
                T_limit = self.T_limit.view(B, 1, 1)
                T_max = self.T_max.view(B, 1, 1)

                p_all = self.p_subset.to(torch.int32).view(B, self.N_job_pad, 1)
                job_ok = (job_available > 0.5).view(B, self.N_job_pad, 1)

                # Remaining work for completion_ok (same semantics as masking/decoding)
                remaining_work_i32 = remaining_work.to(torch.int32).view(B, 1, 1)
                completion_ok = (slot_idx + remaining_work_i32) <= T_limit

                end = slot_idx + p_all  # (B,N,T)
                feasible = slot_idx >= t_now
                feasible = feasible & (slot_idx < T_max)
                feasible = feasible & (end <= T_limit)
                feasible = feasible & (end <= T_max)
                feasible = feasible & completion_ok
                feasible = feasible & job_ok

                large_val = torch.tensor(
                    self.T_max_pad + 1, device=self.device, dtype=torch.int32
                )
                deltas = torch.full(
                    (B, self.num_slack_choices),
                    float(self.T_max_pad),
                    dtype=torch.float32,
                    device=self.device,
                )

                fam = self.duration_families.to(torch.int32)
                slot_indices_bt = slot_idx_1d.view(1, -1).expand(B, -1)  # (B,T)

                for f in range(self.num_slack_choices):
                    candidates_slot = (feasible & (fam == f)).any(dim=1)  # (B,T)
                    masked = torch.where(
                        candidates_slot,
                        slot_indices_bt,
                        large_val.expand_as(slot_indices_bt),
                    )
                    next_slot = masked.min(dim=1).values  # (B,)
                    delta_f = (
                        (next_slot - self.t.to(torch.int32))
                        .clamp(min=0)
                        .to(torch.float32)
                    )
                    delta_f = torch.where(
                        next_slot > self.T_max_pad,
                        torch.full_like(delta_f, float(self.T_max_pad)),
                        delta_f,
                    )
                    deltas[:, f] = delta_f

                ctx = torch.cat([ctx6, self.duration_q.float(), deltas], dim=1)
            else:
                # Price-family ctx13 (slot-price based)
                slot_families = self._compute_slot_families()  # (B, T_max_pad)
                slot_indices = torch.arange(
                    self.T_max_pad, device=self.device
                ).unsqueeze(0)

                # Valid region: [t, min(T_limit, T_max))
                valid = slot_indices >= self.t.unsqueeze(1)
                valid = valid & (slot_indices < self.T_limit.unsqueeze(1))
                valid = valid & (slot_indices < self.T_max.unsqueeze(1))

                large_val = self.T_max_pad + 1
                deltas = torch.full(
                    (B, self.num_slack_choices),
                    float(self.T_max_pad),
                    dtype=torch.float32,
                    device=self.device,
                )

                for f in range(self.num_slack_choices):
                    candidates = valid & (slot_families == f)
                    masked = torch.where(
                        candidates,
                        slot_indices.expand(B, -1),
                        torch.full((B, self.T_max_pad), large_val, device=self.device),
                    )
                    next_slot = masked.min(dim=1).values  # (B,)
                    delta_f = (next_slot - self.t).clamp(min=0).float()
                    delta_f = torch.where(
                        next_slot > self.T_max_pad,
                        torch.full_like(delta_f, float(self.T_max_pad)),
                        delta_f,
                    )
                    deltas[:, f] = delta_f

                ctx = torch.cat([ctx6, self.price_q.float(), deltas], dim=1)
        else:
            ctx = ctx6

        # Defensive: keep observations finite to prevent NaNs/Infs in model logits.
        # (This should rarely trigger; it's here to harden training on edge cases.)
        jobs = torch.nan_to_num(jobs, nan=0.0, posinf=0.0, neginf=0.0)
        periods = torch.nan_to_num(periods, nan=0.0, posinf=0.0, neginf=0.0)
        ctx = torch.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)

        # Action mask: check deadline feasibility for each (job, slack) pair
        # Env's action feasibility uses availability (1 = available)
        action_mask = self._compute_action_mask(job_available)

        # Done envs must have no valid actions.
        # NOTE: `_get_obs()` is intentionally side-effect free; termination logic
        # (including dead-ends/infeasibility) must be handled in `step()` and/or
        # the training runner.
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

        # Track infeasible terminations so we can apply a penalty and log it.
        # This is critical: without a failure penalty, the agent can benefit from
        # "ending early" (fewer negative energy rewards) which misaligns training
        # returns vs. true scheduling performance.
        infeasible_done = torch.zeros((B,), dtype=torch.bool, device=self.device)

        # Penalty baseline: remaining work before any mutation/termination logic.
        # Using the pre-step remaining work makes the penalty robust even if a terminal
        # path clears `job_available` for dead-ends.
        remaining_work_penalty_base = (
            self.p_subset.float() * self.job_available.float()
        ).sum(dim=1)

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
            # Invalid actions indicate a policy/mask bug (sampling without respecting masks)
            # or a wrapper/state sync issue. Terminating those envs is extremely harmful for PPO
            # because it injects artificial terminals and discards learning signal.
            #
            # Training-stable behavior: repair by selecting a valid fallback action for those envs.
            # If no valid action exists, then the state is a true dead-end/infeasible and we mark done.
            import warnings

            warnings.warn(
                f"Invalid actions detected: {invalid_action.sum().item()} envs selected unavailable jobs. "
                "Replacing with a valid fallback action (if available)."
            )

            # Compute a fresh valid action mask from the CURRENT state.
            # (Uses job_available and completion/deadline constraints.)
            am = self._compute_action_mask(self.job_available)  # (B, A)
            has_any = am.sum(dim=1) > 0.5
            fallback = torch.argmax(am, dim=1)  # (B,)

            repairable = invalid_action & has_any
            if repairable.any():
                actions = torch.where(repairable, fallback, actions)
                job_ids = actions // self.num_slack_choices
                slack_ids = actions % self.num_slack_choices
                # Recompute availability selection for repaired rows
                job_available_selected = torch.gather(
                    self.job_available, 1, job_ids.unsqueeze(1).long()
                ).squeeze(1)
                invalid_action = active & (job_available_selected < 0.5)

            # If still invalid or truly no valid action exists, treat as dead-end/infeasible.
            dead = invalid_action
            if dead.any():
                self.done_mask = self.done_mask | dead
                self.job_available[dead] = 0.0
                infeasible_done = infeasible_done | dead
            active = ~self.done_mask

        # Get processing times for selected jobs
        # Use gather to get p[job_id] for each instance
        p = torch.gather(self.p_subset, 1, job_ids.unsqueeze(1).long()).squeeze(1)

        # Compute start times based on variant
        if (
            self.env_config.use_price_families
            and self.env_config.use_duration_aware_families
        ):
            # Duration-aware family variant: slack_id is family_id
            # Find earliest feasible start time whose avg window cost is in that family
            family_ids = slack_ids  # (B,)
            start_times = self._compute_duration_aware_family_start_times_for_job(
                job_ids, family_ids, p
            )
        elif self.env_config.use_price_families:
            # Price-family variant: slack_id is family_id
            # Find best feasible start time in that family (optionally min-cost)
            family_ids = slack_ids  # (B,)
            if getattr(self.env_config, "use_best_family_start", False):
                start_times = self._compute_family_best_start_times_for_job(
                    job_ids, family_ids, p
                )
            else:
                start_times = self._compute_family_start_times_for_job(
                    job_ids, family_ids, p
                )
        else:
            # Prefer slack_type (new config). Fall back to slack_variant (legacy).
            slack_type = getattr(self.env_config, "slack_type", None)

            if slack_type == SlackType.SHORT:
                if self.env_config.short_slack_spec is not None:
                    slack_options = torch.tensor(
                        self.env_config.short_slack_spec.slack_options,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    period_offsets = slack_options[slack_ids.long()]  # (B,)
                    start_times = self._compute_period_aligned_starts(period_offsets)
                else:
                    start_times = self.t.clone()

            elif slack_type == SlackType.COARSE_TO_FINE:
                if self.env_config.c2f_slack_spec is not None:
                    period_offsets = self._compute_c2f_period_offsets(slack_ids)
                    start_times = self._compute_period_aligned_starts(period_offsets)
                else:
                    start_times = self.t.clone()

            elif slack_type == SlackType.FULL:
                period_offsets = slack_ids.int()  # (B,)
                start_times = self._compute_period_aligned_starts(period_offsets)

            else:
                # Legacy fallback
                if self.env_config.slack_variant == SlackVariant.NO_SLACK:
                    start_times = self.t.clone()
                elif self.env_config.slack_variant == SlackVariant.SHORT_SLACK:
                    if self.env_config.short_slack_spec is not None:
                        slack_options = torch.tensor(
                            self.env_config.short_slack_spec.slack_options,
                            dtype=torch.int32,
                            device=self.device,
                        )
                        period_offsets = slack_options[slack_ids.long()]  # (B,)
                        start_times = self._compute_period_aligned_starts(
                            period_offsets
                        )
                    else:
                        start_times = self.t.clone()
                elif self.env_config.slack_variant == SlackVariant.COARSE_TO_FINE:
                    if self.env_config.c2f_slack_spec is not None:
                        period_offsets = self._compute_c2f_period_offsets(slack_ids)
                        start_times = self._compute_period_aligned_starts(
                            period_offsets
                        )
                    else:
                        start_times = self.t.clone()
                elif self.env_config.slack_variant == SlackVariant.FULL_SLACK:
                    period_offsets = slack_ids.int()  # (B,)
                    start_times = self._compute_period_aligned_starts(period_offsets)
                else:
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

        # Detect post-action infeasibility: even starting immediately, remaining work cannot fit.
        # This is a correct necessary-and-sufficient condition for feasibility on a single machine
        # when idling is allowed and there are no release dates.
        remaining_work_now = (self.p_subset.float() * self.job_available.float()).sum(
            dim=1
        )
        dead_end = (
            (~self.done_mask)
            & (remaining_work_now > 0)
            & ((self.t.float() + remaining_work_now) > self.T_limit.float())
        )
        if dead_end.any():
            self.done_mask = self.done_mask | dead_end
            self.job_available[dead_end] = 0.0
            infeasible_done = infeasible_done | dead_end

        if infeasible_done.any():
            # Penalty upper bound: remaining_work * max_slot_price * e_single.
            ct_max = self.ct.float().max(dim=1).values
            penalty = self.e_single.float() * remaining_work_penalty_base * ct_max + 1.0
            rewards = rewards - penalty * infeasible_done.float()

        # Final guard: if an env is still marked active but has 0 valid actions, it is a
        # true dead-end/infeasible state. Mark it as infeasible terminal here so PPO
        # never sees ACTIVE envs with action_mask all-zero.
        obs = self._get_obs()
        mask_zero_next = (obs["action_mask"].sum(dim=1) < 0.5) & (~self.done_mask)
        # Only treat as infeasible if there are still jobs remaining.
        mask_zero_next = mask_zero_next & (self.job_available.sum(dim=1) > 0.5)

        if mask_zero_next.any():
            self.done_mask = self.done_mask | mask_zero_next
            self.job_available[mask_zero_next] = 0.0
            infeasible_done = infeasible_done | mask_zero_next

            # Apply the same infeasibility penalty.
            ct_max = self.ct.float().max(dim=1).values
            penalty = self.e_single.float() * remaining_work_penalty_base * ct_max + 1.0
            rewards = rewards - penalty * mask_zero_next.float()

            # Rebuild obs so runner sees done_mask reflected.
            obs = self._get_obs()

        info = {
            "total_energy": self.total_energy.clone(),
            "infeasible_done": infeasible_done.clone(),
            "mask_zero_done": mask_zero_next.clone(),
        }

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

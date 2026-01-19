"""Batch Sequence Environment for PaST-SM.

This environment wrapper adapts the Single Machine Scheduling problem into a
pure sequencing task. The agent selects the order of jobs, and the optimal timing
is computed ex-post using a vectorized Batch DP solver.
"""

from typing import Dict, Tuple, Any

import torch
import numpy as np

from PaST.sm_env import GPUBatchSingleMachinePeriodEnv
from PaST.batch_dp_solver import BatchSequenceDPSolver


class GPUBatchSequenceEnv(GPUBatchSingleMachinePeriodEnv):
    """
    Environment where the agent learns to sequence jobs.

    Differences from base env:
    - Step: Agent picks a job index (0..N-1). Time does NOT advance.
    - Reward: 0 for all steps except the last.
    - Final Step: Batch DP computes optimal schedule cost for the sequence.
      Reward is -total_energy.
    - Action Mask: Only masks out already-selected jobs.
    """

    def __init__(self, batch_size: int, env_config, device: torch.device):
        super().__init__(batch_size, env_config, device)

        # State tracking for sequencing
        # (B, N_max) tensor to store the sequence of job indices
        self.job_sequences = None
        self.step_indices = None
        self.N_max = 0

    def reset(self, batch_data: Dict[str, Any] = None) -> Dict[str, torch.Tensor]:
        """Reset the environment."""
        # Standard data loading
        obs = super().reset(batch_data)

        # N_job_pad is set in super().reset() via _set_batch_data
        self.N_max = self.N_job_pad

        # Initialize sequence buffer
        self.job_sequences = torch.zeros(
            (self.batch_size, self.N_max), dtype=torch.long, device=self.device
        )

        # Vectorized step index
        self.step_indices = torch.zeros(
            self.batch_size, dtype=torch.long, device=self.device
        )

        # Done flag (local mask)
        self.done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

        # We enforce T=0 for the "planning" perspective
        # super().reset() sets self.t = 0.

        return self._get_obs()

    def reset_indices(
        self,
        batch_data: Dict[str, Any],
        indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Reset subset of environments."""
        obs = super().reset_indices(batch_data, indices)

        # Reset local state for indices
        idx = indices.to(self.device, dtype=torch.long)
        self.step_indices[idx] = 0
        self.job_sequences[idx] = 0  # Zero out buffer
        self.done[idx] = False

        return obs

    def step(
        self, action: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """
        Take a sequence step.

        Args:
            action: (B,) job indices to append to sequence.
        """
        # Store action in sequence using gathered write
        # job_sequences[b, step_indices[b]] = action[b]
        self.job_sequences.scatter_(
            1, self.step_indices.unsqueeze(1), action.unsqueeze(1)
        )

        # Mark job as used (avail=0)
        self.job_available.scatter_(1, action.unsqueeze(1), 0.0)

        # Increment step
        self.step_indices += 1

        # Check termination per instance
        # terminated if step_indices >= n_jobs
        # (Note: self.n_jobs is (B,) tensor from parent)
        dones = self.step_indices >= self.n_jobs

        rewards = torch.zeros(self.batch_size, device=self.device)

        if dones.any():
            # Extract subset of done instances
            done_idx = torch.nonzero(dones, as_tuple=True)[0]

            # Prepare inputs for solver (subset)
            # We must pass the FULL sequence buffer for these instances
            # But the solver needs to know valid length?
            # sequence_lengths = n_jobs[done_idx]
            # Since we pass sequence_lengths to solver, it will ignore garbage after n_jobs.

            subset_seqs = self.job_sequences[done_idx]
            subset_p = self.p_subset[done_idx]
            subset_ct = self.ct[done_idx]
            subset_e = self.e_single[done_idx]
            subset_T = self.T_limit[done_idx]
            subset_lens = self.n_jobs[done_idx].long()

            # Call solver
            costs = BatchSequenceDPSolver.solve(
                job_sequences=subset_seqs,
                processing_times=subset_p,
                ct=subset_ct,
                e_single=subset_e,
                T_limit=subset_T,
                sequence_lengths=subset_lens,
            )

            rewards[done_idx] = -costs

            # Update internal done flag (optional, mainly for safety)
            self.done[dones] = True

        return self._get_obs(), rewards, dones, self._get_info()

    def _compute_action_mask(self, job_available: torch.Tensor = None) -> torch.Tensor:
        """
        Compute action mask.
        """
        if job_available is not None:
            return job_available.float()
        return self.job_available.clone()

    def _get_info(self) -> Dict[str, Any]:
        return {}

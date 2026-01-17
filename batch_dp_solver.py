"""Batch Sequence DP Solver for PaST-SM.

This module implements a fully vectorized PyTorch DP solver that computes the
optimal start times (and total cost) for a fixed sequence of jobs.
This serves as the reward calculator for the `ppo_sequence` variant.
"""

import torch


class BatchSequenceDPSolver:
    """Computes optimal scheduling cost for fixed job sequences using DP."""

    @staticmethod
    def solve(
        job_sequences: torch.Tensor,
        processing_times: torch.Tensor,
        ct: torch.Tensor,
        e_single: torch.Tensor,
        T_limit: torch.Tensor,
        sequence_lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Computes optimal energy cost for a batch of fixed job sequences.

        Args:
            job_sequences: (B, N) Tensor of job indices.
            processing_times: (B, N) Tensor of job durations.
            ct: (B, T_max_pad) Tensor of price curve.
            e_single: (B,) Tensor of energy per unit time.
            T_limit: (B,) Tensor of deadlines.
            sequence_lengths: (B,) Tensor of valid jobs per sequence. 
                              If None, assumes all N jobs are valid.

        Returns:
            final_cost: (B,) Tensor of minimum energy costs.
        """
        B, N = job_sequences.shape
        device = job_sequences.device
        T_max = ct.shape[1]

        # Check inputs
        assert processing_times.shape == (B, N)
        assert T_limit.shape == (B,)
        
        if sequence_lengths is None:
            sequence_lengths = torch.full((B,), N, device=device, dtype=torch.long)

        # Precompute prefix sum of prices for O(1) interval cost
        # P[b, t] = sum(ct[b, :t])
        price_prefix = torch.zeros(
            (B, T_max + 1), dtype=torch.float32, device=device
        )
        price_prefix[:, 1:] = torch.cumsum(ct.float(), dim=1)

        # Initialize DP state: min cost to have finished 0 jobs by time t.
        # Cost is 0 for all t >= 0.
        min_prev = torch.zeros((B, T_max + 1), dtype=torch.float32, device=device)

        # Create time indices for broadcasting
        t_indices = torch.arange(T_max + 1, device=device).unsqueeze(0)  # (1, T+1)

        # Process jobs in sequence (sequential loop over N, vectorized over B and T)
        for i in range(N):
            # Check which sequences are still active (i < length)
            active = (i < sequence_lengths)  # (B,)
            
            # Gather processing time for the current job in the sequence
            job_idx = job_sequences[:, i]  # (B,)
            
            # Map sequence job index to original job properties
            # processing_times is (B, N), where N is number of total jobs
            # job_idx are indices 0..N-1
            p = torch.gather(processing_times, 1, job_idx.unsqueeze(1)).squeeze(1)  # (B,)

            # We want to compute cost for job starting at t_start.
            # t_end = t_start + p  =>  t_start = t_end - p

            # 1. Compute t_start for all possible t_end
            p_expanded = p.unsqueeze(1)  # (B, 1)
            t_start = t_indices - p_expanded  # (B, T+1)

            # 2. Identify valid start times (t_start >= 0)
            valid_mask = t_start >= 0

            # Clamp negative indices to 0 to avoid gather error
            safe_t_start = t_start.clamp(min=0).long()

            # 3. Retrieve min_prev cost (cost to finish previous job by t_start)
            cost_prev = torch.gather(min_prev, 1, safe_t_start)

            # 4. Compute Job Energy Cost
            # Energy = (P[b, t_end] - P[b, t_start]) * e
            # P_end: gather at t_indices
            # P_start: gather at safe_t_start
            price_end = torch.gather(price_prefix, 1, t_indices)
            price_start = torch.gather(price_prefix, 1, safe_t_start)
            
            # e_single might be (B,) or (B,1)
            e = e_single
            if e.ndim == 1:
                e = e.unsqueeze(1)
            
            energy_cost = (price_end - price_start) * e

            # 5. Total cost to finish at exactly t_end
            total_cost_at_end = cost_prev + energy_cost

            # Apply validity mask (t_start >= 0)
            # Use float('inf') for invalid transitions
            total_cost_at_end = torch.where(
                valid_mask,
                total_cost_at_end,
                torch.tensor(float('inf'), device=device)
            )

            # 6. Compute new min_prev (Prefix Min)
            # min_prev[t] = min(total_cost_at_end[row, 0...t])
            # Use torch.cummin if available (PyTorch 1.11+), else scan.
            
            min_curr, _ = torch.cummin(total_cost_at_end, dim=1)
            
            # UPDATE STEP:
            # If active, min_prev = min_curr
            # If inactive, min_prev remains same (identity for cost)
            min_prev = torch.where(
                active.unsqueeze(1),
                min_curr,
                min_prev
            )

        # Final result: min cost to finish all N jobs by T_limit
        # Clamp T_limit to be within bounds
        safe_T_limit = T_limit.clamp(max=T_max).long().unsqueeze(1)
        final_cost = torch.gather(min_prev, 1, safe_T_limit).squeeze(1)

        return final_cost

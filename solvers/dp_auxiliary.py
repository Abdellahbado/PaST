"""
DP Auxiliary Loss Module for PaST-SM.

Provides mechanisms to strengthen the learning signal for sequence-based RL
by evaluating multiple candidate actions per state using DP completion costs
and computing an auxiliary ranking loss.

Key Components:
- DPCompletionEvaluator: Evaluates Q(s, a) for multiple candidate actions.
- compute_ranking_loss: Computes a listwise ranking loss from Q-values.
"""

from typing import Tuple, Optional
import torch
import torch.nn.functional as F
from torch import Tensor

from PaST.batch_dp_solver import BatchSequenceDPSolver


class DPCompletionEvaluator:
    """
    Evaluates DP completion costs for multiple candidate actions.
    
    Given a partial sequence (jobs already chosen) and a set of candidate
    next-actions, computes the optimal DP cost for completing the schedule
    with each candidate.
    """

    def __init__(self, device: torch.device):
        self.device = device

    def evaluate_candidates(
        self,
        partial_sequences: Tensor,  # (B, step_idx) jobs already in sequence
        step_idx: Tensor,           # (B,) current step index per instance
        candidate_actions: Tensor,  # (B, K) candidate job indices to evaluate
        job_available: Tensor,      # (B, N) current availability mask
        processing_times: Tensor,   # (B, N) job processing times
        ct: Tensor,                 # (B, T_max) price curve
        e_single: Tensor,           # (B,) energy per unit time
        T_limit: Tensor,            # (B,) deadline
        n_jobs: Tensor,             # (B,) total jobs per instance
    ) -> Tensor:
        """
        Evaluate DP completion costs for K candidate actions per instance.
        
        For each candidate action, this:
        1. Appends the candidate to the partial sequence.
        2. Greedily completes the sequence with remaining jobs (in index order).
        3. Runs DP to get optimal timing cost for the full sequence.
        
        Args:
            partial_sequences: (B, max_steps) current partial sequences (padded).
            step_idx: (B,) number of jobs already in sequence.
            candidate_actions: (B, K) candidate job indices.
            job_available: (B, N) mask of available jobs (1=available).
            processing_times: (B, N) job processing times.
            ct: (B, T_max) per-slot prices.
            e_single: (B,) energy rates.
            T_limit: (B,) deadlines.
            n_jobs: (B,) total number of jobs.
            
        Returns:
            q_values: (B, K) DP completion costs (lower is better).
        """
        B, K = candidate_actions.shape
        N = processing_times.shape[1]
        max_steps = partial_sequences.shape[1]
        
        # We need to evaluate K candidates per instance.
        # Strategy: Expand batch to B*K, evaluate, reshape back.
        
        # Expand all tensors: (B, ...) -> (B*K, ...)
        partial_exp = partial_sequences.unsqueeze(1).expand(B, K, max_steps).reshape(B * K, max_steps)
        step_idx_exp = step_idx.unsqueeze(1).expand(B, K).reshape(B * K)
        job_avail_exp = job_available.unsqueeze(1).expand(B, K, N).reshape(B * K, N).clone()
        p_exp = processing_times.unsqueeze(1).expand(B, K, N).reshape(B * K, N)
        ct_exp = ct.unsqueeze(1).expand(B, K, ct.shape[1]).reshape(B * K, ct.shape[1])
        e_exp = e_single.unsqueeze(1).expand(B, K).reshape(B * K)
        T_exp = T_limit.unsqueeze(1).expand(B, K).reshape(B * K)
        n_jobs_exp = n_jobs.unsqueeze(1).expand(B, K).reshape(B * K)
        
        # Flatten candidate actions: (B, K) -> (B*K,)
        candidates_flat = candidate_actions.reshape(B * K)
        
        # Build full sequences by:
        # 1. Copy partial sequence
        # 2. Append candidate action
        # 3. Greedily append remaining jobs in index order
        
        full_sequences = torch.zeros(B * K, N, dtype=torch.long, device=self.device)
        
        # Copy partial
        for i in range(max_steps):
            mask = i < step_idx_exp
            full_sequences[:, i] = torch.where(mask, partial_exp[:, i], full_sequences[:, i])
        
        # Append candidate at position step_idx
        seq_write_idx = step_idx_exp.clone()
        full_sequences.scatter_(1, seq_write_idx.unsqueeze(1), candidates_flat.unsqueeze(1))
        
        # Mark candidate as used in availability
        job_avail_exp.scatter_(1, candidates_flat.unsqueeze(1), 0.0)
        seq_write_idx += 1
        
        # Greedily append remaining jobs
        # For simplicity, iterate through job indices and append if available
        for job_idx in range(N):
            # Check if this job is still available
            is_available = job_avail_exp[:, job_idx] > 0.5  # (B*K,)
            # Check if we still need more jobs
            need_more = seq_write_idx < n_jobs_exp
            should_add = is_available & need_more
            
            # Append job_idx to sequence at seq_write_idx
            full_sequences.scatter_(
                1, 
                seq_write_idx.unsqueeze(1), 
                torch.where(should_add, torch.tensor(job_idx, device=self.device), full_sequences.gather(1, seq_write_idx.unsqueeze(1)).squeeze(1)).unsqueeze(1)
            )
            
            # Update write index and mark used
            seq_write_idx = torch.where(should_add, seq_write_idx + 1, seq_write_idx)
            job_avail_exp[:, job_idx] = torch.where(should_add, 0.0, job_avail_exp[:, job_idx])
        
        # Now full_sequences contains complete sequences for each (instance, candidate) pair
        # Run DP solver
        costs = BatchSequenceDPSolver.solve(
            job_sequences=full_sequences,
            processing_times=p_exp,
            ct=ct_exp,
            e_single=e_exp,
            T_limit=T_exp,
            sequence_lengths=n_jobs_exp,
        )
        
        # Reshape back to (B, K)
        q_values = costs.reshape(B, K)
        
        return q_values


def compute_ranking_loss(
    logits: Tensor,           # (B, N) policy logits for all actions
    candidate_actions: Tensor, # (B, K) indices of evaluated candidates
    q_values: Tensor,         # (B, K) DP costs for candidates (lower is better)
    action_mask: Tensor,      # (B, N) valid actions mask
    temperature: float = 1.0,
) -> Tensor:
    """
    Compute a listwise ranking loss to encourage the policy to rank
    lower-cost actions higher.
    
    Uses ListMLE: probability of the observed ranking under the model.
    Target ranking: sorted by ascending Q-value (lowest cost first).
    
    Args:
        logits: (B, N) raw policy logits.
        candidate_actions: (B, K) indices of candidates.
        q_values: (B, K) DP costs (lower = better).
        action_mask: (B, N) mask for valid actions.
        temperature: Temperature for softmax.
        
    Returns:
        loss: Scalar ranking loss.
    """
    B, K = candidate_actions.shape
    
    # Gather logits for candidate actions
    candidate_logits = torch.gather(logits, 1, candidate_actions)  # (B, K)
    
    # Sort candidates by Q-value (ascending = best first)
    sorted_indices = torch.argsort(q_values, dim=1)  # (B, K) indices into K dimension
    
    # Reorder logits according to target ranking
    sorted_logits = torch.gather(candidate_logits, 1, sorted_indices)  # (B, K)
    
    # ListMLE loss: -sum_i log(softmax(remaining)[0])
    # For each position, probability that the correct item is ranked first among remaining
    loss = torch.zeros(B, device=logits.device)
    
    for i in range(K):
        remaining_logits = sorted_logits[:, i:] / temperature  # (B, K-i)
        log_probs = F.log_softmax(remaining_logits, dim=1)  # (B, K-i)
        loss -= log_probs[:, 0]  # Log prob of first element being ranked first
    
    return loss.mean()


def sample_candidates(
    logits: Tensor,      # (B, N) policy logits
    action_mask: Tensor, # (B, N) valid actions (1=valid)
    k: int,
    temperature: float = 1.0,
    include_top: bool = True,
) -> Tensor:
    """
    Sample k candidate actions from the policy.
    
    Args:
        logits: (B, N) raw policy logits.
        action_mask: (B, N) valid action mask.
        k: Number of candidates to sample.
        temperature: Sampling temperature.
        include_top: If True, always include the greedy action.
        
    Returns:
        candidates: (B, K) sampled action indices.
    """
    B, N = logits.shape
    
    # Mask invalid actions
    masked_logits = logits.masked_fill(action_mask < 0.5, float('-inf'))
    
    # Temperature scaling
    scaled_logits = masked_logits / temperature
    
    # Sample k actions without replacement
    probs = F.softmax(scaled_logits, dim=1)
    
    # Use multinomial sampling (with replacement, then dedupe)
    # For simplicity, sample more than k and take unique
    candidates_list = []
    
    if include_top:
        # Always include greedy action
        top_action = masked_logits.argmax(dim=1, keepdim=True)  # (B, 1)
        candidates_list.append(top_action)
        remaining_k = k - 1
    else:
        remaining_k = k
    
    if remaining_k > 0:
        # Sample remaining candidates
        # Handle edge case where fewer than k actions are valid
        valid_counts = action_mask.sum(dim=1)  # (B,)
        sample_k = min(remaining_k, int(valid_counts.min().item()))
        
        if sample_k > 0:
            sampled = torch.multinomial(probs, sample_k, replacement=False)  # (B, sample_k)
            candidates_list.append(sampled)
    
    if candidates_list:
        candidates = torch.cat(candidates_list, dim=1)  # (B, <=k)
    else:
        # Fallback: just use greedy
        candidates = masked_logits.argmax(dim=1, keepdim=True)
    
    # Pad to exactly k if needed
    actual_k = candidates.shape[1]
    if actual_k < k:
        # Pad with the greedy action
        padding = candidates[:, :1].expand(B, k - actual_k)
        candidates = torch.cat([candidates, padding], dim=1)
    
    return candidates[:, :k]

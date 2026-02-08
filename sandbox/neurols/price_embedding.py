"""Price embedding for TOU-aware scheduling.

This module implements price-aware embeddings following the plan:
1. z_price: Global price profile embedding via CNN over daily pattern
2. Per-machine price-exposure statistics
3. Ablation support: no price features / z_price only / full features

The 20-hour daily TOU pattern (off-peak/shoulder/peak) is encoded per hour,
then processed by a 1D-CNN to capture temporal patterns.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# Constants from the benchmark
HOURS_PER_DAY = 20
N_PRICE_LEVELS = 3  # off-peak (1), shoulder (2), peak (4)


class PriceFeatureExtractor:
    """Extract price-related features for the NeuroLS controller.

    Features:
    1. z_price: Global embedding of price profile (requires torch)
    2. price_level_distribution: Fraction of time in each price level
    3. Per-machine price exposure stats
    """

    def __init__(
        self,
        ct: np.ndarray,
        K: int,
        hours_per_day: int = HOURS_PER_DAY,
    ):
        """Initialize price feature extractor.

        Args:
            ct: Per-slot prices, shape (T_max,)
            K: Horizon constraint
            hours_per_day: Hours per operational day (default 20)
        """
        self.ct = np.asarray(ct[:K], dtype=np.float64)
        self.K = K
        self.hours_per_day = hours_per_day
        self.n_days = max(1, K // hours_per_day)

        # Precompute price prefix sum for O(1) interval cost
        self.price_prefix = np.zeros(len(self.ct) + 1, dtype=np.float64)
        self.price_prefix[1:] = np.cumsum(self.ct)

        # Categorize prices into levels
        self._compute_price_levels()

        # Precompute level-specific prefix sums for fast exposure queries
        # level_count_prefix[l, t] = number of slots < t in level l
        # level_price_prefix[l, t] = sum of prices for slots < t in level l
        n = len(self.ct)
        self.level_count_prefix = np.zeros((N_PRICE_LEVELS, n + 1), dtype=np.int32)
        self.level_price_prefix = np.zeros((N_PRICE_LEVELS, n + 1), dtype=np.float64)
        for l in range(N_PRICE_LEVELS):
            mask = (self.slot_levels == l).astype(np.int32)
            self.level_count_prefix[l, 1:] = np.cumsum(mask)
            self.level_price_prefix[l, 1:] = np.cumsum(self.ct * mask)

    def get_interval_level_counts(self, start: int, duration: int) -> np.ndarray:
        """Get counts of slots per price level over [start, start+duration).

        Returns:
            (3,) int array with counts per level.
        """
        end = min(start + duration, len(self.ct))
        if start < 0 or start >= end:
            return np.zeros(N_PRICE_LEVELS, dtype=np.int32)
        counts = self.level_count_prefix[:, end] - self.level_count_prefix[:, start]
        return counts.astype(np.int32)

    def get_interval_level_price_sums(self, start: int, duration: int) -> np.ndarray:
        """Get sum of prices per level over [start, start+duration).

        Returns:
            (3,) float array with sum(ct[t]) per level.
        """
        end = min(start + duration, len(self.ct))
        if start < 0 or start >= end:
            return np.zeros(N_PRICE_LEVELS, dtype=np.float64)
        sums = self.level_price_prefix[:, end] - self.level_price_prefix[:, start]
        return sums.astype(np.float64)

    def _compute_price_levels(self) -> None:
        """Categorize prices into off-peak/shoulder/peak."""
        unique_prices = sorted(set(self.ct))

        if len(unique_prices) <= 1:
            self.price_thresholds = [float("inf"), float("inf")]
        elif len(unique_prices) == 2:
            mid = (unique_prices[0] + unique_prices[1]) / 2
            self.price_thresholds = [mid, float("inf")]
        else:
            # Use tertiles or known levels
            p33 = np.percentile(self.ct, 33)
            p66 = np.percentile(self.ct, 66)
            self.price_thresholds = [p33, p66]

        # Assign level to each slot
        self.slot_levels = np.zeros(len(self.ct), dtype=np.int32)
        for t, price in enumerate(self.ct):
            if price <= self.price_thresholds[0]:
                self.slot_levels[t] = 0  # off-peak
            elif price <= self.price_thresholds[1]:
                self.slot_levels[t] = 1  # shoulder
            else:
                self.slot_levels[t] = 2  # peak

    def get_interval_cost(self, start: int, duration: int) -> float:
        """Get energy cost for interval [start, start+duration)."""
        end = min(start + duration, len(self.ct))
        if start < 0 or start >= end:
            return float("inf")
        return float(self.price_prefix[end] - self.price_prefix[start])

    def get_price_level_distribution(self) -> np.ndarray:
        """Get fraction of time in each price level.

        Returns:
            (3,) array: [frac_offpeak, frac_shoulder, frac_peak]
        """
        counts = np.bincount(self.slot_levels, minlength=N_PRICE_LEVELS)
        return counts.astype(np.float32) / max(1, len(self.ct))

    def get_daily_pattern(self) -> np.ndarray:
        """Get daily price pattern (averaged if multiple days).

        Returns:
            (hours_per_day,) array of average prices per hour
        """
        if len(self.ct) < self.hours_per_day:
            pattern = np.zeros(self.hours_per_day, dtype=np.float64)
            pattern[: len(self.ct)] = self.ct
            return pattern.astype(np.float32)

        # Reshape to (n_days, hours_per_day) and average
        valid_slots = self.n_days * self.hours_per_day
        if valid_slots > len(self.ct):
            valid_slots = len(self.ct)
            self.n_days = valid_slots // self.hours_per_day

        reshaped = self.ct[: self.n_days * self.hours_per_day].reshape(
            self.n_days, self.hours_per_day
        )
        return reshaped.mean(axis=0).astype(np.float32)

    def get_per_hour_features(self) -> np.ndarray:
        """Get per-hour token features for CNN input.

        Returns:
            (hours_per_day, 5) array:
                - one-hot price level (3)
                - normalized price (1)
                - position encoding (1)
        """
        daily = self.get_daily_pattern()

        # Normalize prices
        price_min = float(np.min(self.ct))
        price_max = float(np.max(self.ct))
        price_range = max(1e-9, price_max - price_min)
        prices_norm = (daily - price_min) / price_range

        features = []
        for h in range(self.hours_per_day):
            # One-hot price level
            level = 0
            if daily[h] <= self.price_thresholds[0]:
                level = 0
            elif daily[h] <= self.price_thresholds[1]:
                level = 1
            else:
                level = 2

            one_hot = [0.0, 0.0, 0.0]
            one_hot[level] = 1.0

            # Normalized price
            norm_price = float(prices_norm[h])

            # Position encoding (sinusoidal)
            pos_enc = math.sin(math.pi * h / self.hours_per_day)

            features.append(one_hot + [norm_price, pos_enc])

        return np.array(features, dtype=np.float32)

    def compute_machine_price_exposure(
        self,
        start_times: List[int],
        processing_times: List[int],
        machine_energy_rate: float,
    ) -> np.ndarray:
        """Compute price exposure statistics for a machine's schedule.

        Args:
            start_times: Start time for each job on this machine
            processing_times: Processing time for each job
            machine_energy_rate: Energy rate for this machine

        Returns:
            (7,) feature vector:
                - workload in each price level (3)
                - fraction in each level (3)
                - average paid price per slot (1)
        """
        workload_by_level = np.zeros(N_PRICE_LEVELS, dtype=np.float64)
        total_slots = 0
        total_price = 0.0

        for start, duration in zip(start_times, processing_times):
            duration_i = int(duration)
            if duration_i <= 0:
                continue

            counts = self.get_interval_level_counts(int(start), duration_i)
            workload_by_level += counts.astype(np.float64)
            total_slots += int(np.sum(counts))
            total_price += float(self.get_interval_cost(int(start), duration_i))

        # Normalize
        total_work = float(np.sum(workload_by_level))
        if total_work > 0:
            fraction_by_level = workload_by_level / total_work
        else:
            fraction_by_level = np.zeros(N_PRICE_LEVELS, dtype=np.float64)

        avg_price = total_price / max(1, total_slots)
        # Normalize average price
        price_mean = float(np.mean(self.ct))
        avg_price_norm = avg_price / max(1e-9, price_mean)

        return np.concatenate(
            [
                workload_by_level / max(1.0, float(self.K)),  # Normalized workload
                fraction_by_level,  # Fractions
                [avg_price_norm],  # Normalized average price
            ]
        ).astype(np.float32)


if _TORCH_AVAILABLE:

    class PriceCNNEncoder(nn.Module):
        """1D-CNN encoder for daily price pattern.

        Input: (batch, hours_per_day, 5) per-hour features
        Output: (batch, d_emb) price embedding
        """

        def __init__(
            self,
            hours_per_day: int = HOURS_PER_DAY,
            input_dim: int = 5,
            hidden_dim: int = 32,
            output_dim: int = 64,
            kernel_size: int = 3,
            n_layers: int = 2,
        ):
            super().__init__()
            self.hours_per_day = hours_per_day
            self.input_dim = input_dim
            self.output_dim = output_dim

            # CNN layers
            layers = []
            in_ch = input_dim
            for i in range(n_layers):
                out_ch = hidden_dim if i < n_layers - 1 else hidden_dim
                layers.append(
                    nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
                )
                layers.append(nn.GELU())
                in_ch = out_ch

            self.conv = nn.Sequential(*layers)

            # Pool and project
            self.pool_proj = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(hidden_dim, output_dim),
                nn.GELU(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (batch, hours_per_day, input_dim)

            Returns:
                (batch, output_dim) price embedding
            """
            # (B, H, D) -> (B, D, H) for Conv1d
            x = x.transpose(1, 2)
            x = self.conv(x)
            x = self.pool_proj(x)
            return x

    class PriceEmbedding(nn.Module):
        """Full price embedding module with ablation support.

        Modes:
        - "none": No price features (returns zeros)
        - "z_price": Only global price embedding
        - "full": z_price + per-machine exposure stats
        """

        def __init__(
            self,
            d_emb: int = 64,
            hours_per_day: int = HOURS_PER_DAY,
            mode: str = "full",
            n_machines: int = 1,
        ):
            super().__init__()
            self.d_emb = d_emb
            self.hours_per_day = hours_per_day
            self.mode = mode.lower()
            self.n_machines = n_machines

            if self.mode in ("z_price", "full"):
                self.cnn = PriceCNNEncoder(
                    hours_per_day=hours_per_day,
                    output_dim=d_emb,
                )
            else:
                self.cnn = None

            if self.mode == "full":
                # Project per-machine exposure stats
                self.exposure_proj = nn.Sequential(
                    nn.Linear(7, d_emb // 2),
                    nn.GELU(),
                    nn.Linear(d_emb // 2, d_emb // 4),
                )
                # Combine z_price with exposure
                self.combine = nn.Linear(d_emb + d_emb // 4, d_emb)
            else:
                self.exposure_proj = None
                self.combine = None

        def forward(
            self,
            per_hour_features: torch.Tensor,
            machine_exposure: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Args:
                per_hour_features: (batch, hours_per_day, 5)
                machine_exposure: (batch, n_machines, 7) per-machine stats

            Returns:
                (batch, d_emb) price embedding
            """
            batch_size = per_hour_features.size(0)

            if self.mode == "none":
                return torch.zeros(
                    batch_size, self.d_emb, device=per_hour_features.device
                )

            # Global price embedding
            z_price = self.cnn(per_hour_features)  # (B, d_emb)

            if self.mode == "z_price" or machine_exposure is None:
                return z_price

            # Add per-machine exposure (aggregate)
            # (B, M, 7) -> (B, 7) via mean
            exposure_agg = machine_exposure.mean(dim=1)
            exposure_emb = self.exposure_proj(exposure_agg)  # (B, d_emb//4)

            # Combine
            combined = torch.cat([z_price, exposure_emb], dim=-1)
            return self.combine(combined)

        def get_output_dim(self) -> int:
            return self.d_emb


def extract_price_features_numpy(
    ct: np.ndarray,
    K: int,
    per_machine_schedules: Optional[List[Tuple[List[int], List[int], float]]] = None,
) -> dict:
    """Extract all price features as numpy arrays.

    Args:
        ct: Per-slot prices
        K: Horizon
        per_machine_schedules: List of (start_times, processing_times, e_rate) per machine

    Returns:
        Dict with:
        - per_hour_features: (hours_per_day, 5)
        - price_level_dist: (3,)
        - machine_exposure: (n_machines, 7) if schedules provided
    """
    extractor = PriceFeatureExtractor(ct, K)

    result = {
        "per_hour_features": extractor.get_per_hour_features(),
        "price_level_dist": extractor.get_price_level_distribution(),
        "daily_pattern": extractor.get_daily_pattern(),
    }

    if per_machine_schedules:
        exposures = []
        for start_times, proc_times, e_rate in per_machine_schedules:
            exp = extractor.compute_machine_price_exposure(
                start_times, proc_times, e_rate
            )
            exposures.append(exp)
        result["machine_exposure"] = np.stack(exposures, axis=0)

    return result

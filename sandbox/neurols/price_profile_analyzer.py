from typing import List, Tuple, Dict
import numpy as np

class PriceProfileAnalyzer:
    """Analyzes ToU price profiles to identify Peak and Valley windows."""

    def __init__(self, prices: List[float], hours_per_day: int = 20):
        self.prices = np.array(prices, dtype=np.float32)
        self.hours_per_day = hours_per_day
        self.K = len(prices)
        self.valleys = []  # List of (start, end) tuples
        self.peaks = []    # List of (start, end) tuples
        self._analyze()

    def _analyze(self):
        """Segment prices into tiers."""
        # Simple clustering: < median = valley, >= median = peak
        # Or better: use unique values. 
        # The user said: "maximize utility of cheap periods".
        # Typically there are 3 levels: Off-Peak (Valley), Standard, Peak.
        
        unique_prices = sorted(np.unique(self.prices))
        if len(unique_prices) <= 1:
            # Flat profile
            self.valleys = [(0, self.K)]
            return

        # Heuristic: Lowest price tier is "Valley". Highest is "Peak".
        min_p = unique_prices[0]
        max_p = unique_prices[-1]
        
        # Thresholds
        valley_thresh = min_p + 1e-6
        peak_thresh = max_p - 1e-6
        
        # Identify windows
        is_valley = self.prices <= valley_thresh
        is_peak = self.prices >= peak_thresh
        
        self.valleys = self._get_intervals(is_valley)
        self.peaks = self._get_intervals(is_peak)

    def _get_intervals(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Convert boolean mask to list of (start, end) intervals."""
        # Pad with False to handle boundaries
        padded = np.concatenate(([False], mask, [False]))
        # Find transitions
        starts = np.flatnonzero(padded[1:] & ~padded[:-1])
        ends = np.flatnonzero(~padded[1:] & padded[:-1])
        return list(zip(starts.tolist(), ends.tolist()))

    def get_tier_at(self, t: int) -> str:
        """Return 'VALLEY', 'PEAK', or 'STANDARD' for time t."""
        if self.K == 0: return "STANDARD"
        t = t % self.K # subtle: if t >= K, wrap? No, usually t < K.
        # But wait, K is the horizon.
        
        p = self.prices[t]
        unique_prices = sorted(np.unique(self.prices))
        if p <= unique_prices[0] + 1e-6:
            return "VALLEY"
        if p >= unique_prices[-1] - 1e-6:
            return "PEAK"
        return "STANDARD"

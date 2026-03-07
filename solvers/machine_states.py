"""Machine state modelling and SPACES preprocessing for energy-aware scheduling.

This module implements the SPACES (Shortest Path Algorithm for Cost Efficient
Switchings) technique from Benedikt et al. (2020), as used in the B&B-SPACES
algorithm of Benedikt et al. (2025, arXiv:2506.10405).

The key idea: precompute the *optimal switching cost* c*(i, j) between any
pair of time intervals (i, j) where the machine transitions from proc state
at end-of-interval i to proc state at start-of-interval j.  This encapsulates
all possible intermediate state sequences (off, idle, standby, ...) into a
single scalar cost, allowing the main DP to work without tracking machine
power states explicitly.

Usage::

    config = MachineStateConfig.shrouf2014()
    spaces = compute_spaces(prices, config)
    # spaces.c_star[i, j] = optimal switching cost from proc@i to proc@j
    # spaces.c_start[t]   = optimal startup cost off@0 → proc@t
    # spaces.c_end[t]     = optimal shutdown cost proc@t → off@h
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

_INF = float("inf")


# ─────────────────────────────────────────────────────────────────────
# Machine State Configuration
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MachineStateConfig:
    """Configuration for a stateful machine with energy-consuming states.

    The machine has a set of named states (at least 'off' and 'proc').
    Transitions between states have durations and power consumptions.

    Attributes:
        states: List of state names.  Must include 'off' and 'proc'.
        T_trans: Dict mapping (s, s') -> transition duration in intervals.
                 T(s, s) = 1 for all s (staying in same state for 1 interval).
                 Missing keys or None values mean the transition is impossible.
        P_trans: Dict mapping (s, s') -> power consumption per interval during
                 the transition.  Missing keys or None values mean impossible.
        off_state: Name of the off state (default 'off').
        proc_state: Name of the processing state (default 'proc').
    """

    states: Tuple[str, ...]
    T_trans: Dict[Tuple[str, str], Optional[int]]
    P_trans: Dict[Tuple[str, str], Optional[float]]
    off_state: str = "off"
    proc_state: str = "proc"

    def __post_init__(self):
        if self.off_state not in self.states:
            raise ValueError(
                f"off_state '{self.off_state}' not in states {self.states}"
            )
        if self.proc_state not in self.states:
            raise ValueError(
                f"proc_state '{self.proc_state}' not in states {self.states}"
            )
        # Validate T(s,s) == 1 for all states
        for s in self.states:
            t_ss = self.T_trans.get((s, s))
            if t_ss is not None and t_ss != 1:
                raise ValueError(
                    f"T_trans[({s}, {s})] must be 1 (staying in same state), got {t_ss}"
                )

    def get_transition_time(self, s: str, s_prime: str) -> Optional[int]:
        """Return transition time T(s, s'), or None if impossible."""
        v = self.T_trans.get((s, s_prime))
        return v if v is not None else None

    def get_transition_power(self, s: str, s_prime: str) -> Optional[float]:
        """Return transition power P(s, s'), or None if impossible."""
        v = self.P_trans.get((s, s_prime))
        return v if v is not None else None

    @property
    def P_proc(self) -> float:
        """Power consumption while processing (P(proc, proc))."""
        return float(self.P_trans.get((self.proc_state, self.proc_state), 0.0) or 0.0)

    @property
    def n_states(self) -> int:
        return len(self.states)

    def is_trivial(self) -> bool:
        """True if this is a no-states config (stateless machine)."""
        return len(self.states) <= 1

    # ── Factory methods ──────────────────────────────────────────────

    @staticmethod
    def no_states() -> MachineStateConfig:
        """Trivial single-state config: machine is always 'on', idle is free.

        This reproduces the behavior of the original (stateless) DP solver.
        """
        return MachineStateConfig(
            states=("proc",),
            T_trans={("proc", "proc"): 1},
            P_trans={("proc", "proc"): 1.0},
            off_state="proc",
            proc_state="proc",
        )

    @staticmethod
    def nosby(
        turn_on_time: int = 2,
        turn_on_power: float = 5.0,
        turn_off_time: int = 1,
        turn_off_power: float = 0.0,
        proc_power: float = 4.0,
        off_power: float = 0.0,
    ) -> MachineStateConfig:
        """NOSBY model: 2 states (off, proc), no standby/idle.

        Default parameters match the Shrouf (2014) transition graph (Fig. 2
        of arXiv:2506.10405):
            T(off, proc) = 2,  P(off, proc) = 5
            T(proc, off) = 1,  P(proc, off) = 0
            T(proc, proc) = 1, P(proc, proc) = 4
            T(off, off) = 1,   P(off, off) = 0
        """
        return MachineStateConfig(
            states=("off", "proc"),
            T_trans={
                ("off", "off"): 1,
                ("off", "proc"): turn_on_time,
                ("proc", "off"): turn_off_time,
                ("proc", "proc"): 1,
            },
            P_trans={
                ("off", "off"): off_power,
                ("off", "proc"): turn_on_power,
                ("proc", "off"): turn_off_power,
                ("proc", "proc"): proc_power,
            },
        )

    @staticmethod
    def shrouf2014() -> MachineStateConfig:
        """The 3-state model from Shrouf et al. (2014), Fig. 2 of the paper.

        States: off, proc, idle
        Transition times and power:
            T(proc, proc)=1, P(proc, proc)=4
            T(idle, idle)=1, P(idle, idle)=2
            T(off, off)=1,   P(off, off)=0
            T(off, proc)=2,  P(off, proc)=5
            T(proc, off)=1,  P(proc, off)=0
            T(proc, idle)=1, P(proc, idle)=0  (instant, free)
            T(idle, proc)=1, P(idle, proc)=0  (instant, free)
            off→idle: impossible
            idle→off: impossible
        """
        return MachineStateConfig(
            states=("off", "proc", "idle"),
            T_trans={
                ("off", "off"): 1,
                ("off", "proc"): 2,
                ("proc", "proc"): 1,
                ("proc", "off"): 1,
                ("proc", "idle"): 1,
                ("idle", "idle"): 1,
                ("idle", "proc"): 1,
                # off→idle and idle→off are impossible (not listed)
            },
            P_trans={
                ("off", "off"): 0.0,
                ("off", "proc"): 5.0,
                ("proc", "proc"): 4.0,
                ("proc", "off"): 0.0,
                ("proc", "idle"): 0.0,
                ("idle", "idle"): 2.0,
                ("idle", "proc"): 0.0,
                # off→idle and idle→off are impossible (not listed)
            },
        )

    @staticmethod
    def paper_nosby() -> MachineStateConfig:
        """Section 5.1 synthetic benchmark model from the attached paper.

        The paper refers to the Figure 2 benchmark family as ``NOSBY``. In the
        attached manuscript, that benchmark uses the Figure 2 transition graph
        with zero-time, zero-energy transitions between ``proc`` and ``idle``.
        This differs from ``shrouf2014()`` above, which intentionally keeps the
        older one-interval helper used in the initial prototype tests.

        Key parameters (from arXiv:2506.10405 Figure 2):
            T(off→proc)=2, P(off→proc)=5   (startup: 2 intervals at power 5)
            T(proc→off)=1, P(proc→off)=1   (shutdown: 1 interval at power 1)
            T(proc↔idle)=0, P(proc↔idle)=0  (zero-time free transitions)
            T(idle→idle)=1, P(idle→idle)=2   (idle power consumption)
        """
        return MachineStateConfig(
            states=("off", "proc", "idle"),
            T_trans={
                ("off", "off"): 1,
                ("off", "proc"): 2,
                ("proc", "proc"): 1,
                ("proc", "off"): 1,
                ("proc", "idle"): 0,
                ("idle", "idle"): 1,
                ("idle", "proc"): 0,
            },
            P_trans={
                ("off", "off"): 0.0,
                ("off", "proc"): 5.0,
                ("proc", "proc"): 4.0,
                ("proc", "off"): 1.0,  # Fig 2: proc→off edge labeled 1/1 (T=1,P=1)
                ("proc", "idle"): 0.0,
                ("idle", "idle"): 2.0,
                ("idle", "proc"): 0.0,
            },
        )

    @staticmethod
    def custom(
        states: List[str],
        transitions: List[Tuple[str, str, int, float]],
        off_state: str = "off",
        proc_state: str = "proc",
    ) -> MachineStateConfig:
        """Build from a list of (from, to, duration, power) tuples.

        Self-loops T(s,s)=1 are added automatically if not specified.
        """
        T_trans: Dict[Tuple[str, str], Optional[int]] = {}
        P_trans: Dict[Tuple[str, str], Optional[float]] = {}

        for s, s_prime, dur, power in transitions:
            T_trans[(s, s_prime)] = dur
            P_trans[(s, s_prime)] = power

        # Ensure self-loops
        for s in states:
            if (s, s) not in T_trans:
                T_trans[(s, s)] = 1
                P_trans[(s, s)] = 0.0

        return MachineStateConfig(
            states=tuple(states),
            T_trans=T_trans,
            P_trans=P_trans,
            off_state=off_state,
            proc_state=proc_state,
        )


# ─────────────────────────────────────────────────────────────────────
# SPACES Result
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SPACESResult:
    """Result of SPACES preprocessing.

    Attributes:
        c_star: 2D array of shape (h+1, h+1).  c_star[i, j] = optimal
            switching cost from proc@end-of-interval-i to proc@start-of-
            interval-j.  Only valid for j >= i; entries with j < i are inf.
            c_star[i, i] = 0 (no gap, consecutive processing).
        c_start: 1D array of shape (h+1,).  c_start[t] = optimal cost from
            off@0 to proc@t (startup cost).  Includes the cost of the off
            state during interval 0.
        c_end: 1D array of shape (h+1,).  c_end[t] = optimal cost from
            proc@t to off@(h-1) (shutdown cost).  Includes the cost of the
            off state during the last interval.
        P_proc: Power consumption while processing.
        early: Earliest interval where proc is possible.
        late: Latest interval where proc can start and still shut down.
        h: Number of intervals (horizon).
        max_gap: The maximum gap used in banded computation (-1 = full).
    """

    c_star: np.ndarray
    c_start: np.ndarray
    c_end: np.ndarray
    P_proc: float
    early: int
    late: int
    h: int
    max_gap: int = -1

    def gap_cost(self, t_end: int, t_start: int) -> float:
        """Cost of optimal switching from proc@t_end to proc@t_start.

        If t_start == t_end, returns 0.0 (consecutive processing).
        If t_start < t_end, returns inf.
        If banded and gap exceeds max_gap, decomposes as c_end + c_start.
        """
        if t_start < t_end:
            return _INF
        if t_start == t_end:
            return 0.0
        if self.max_gap > 0 and (t_start - t_end) > self.max_gap:
            # Beyond banded range — decompose
            return float(self.c_end[t_end]) + float(self.c_start[t_start])
        if self.max_gap > 0:
            return float(self.c_star[t_end, t_start - t_end])
        return float(self.c_star[t_end, t_start])


# ─────────────────────────────────────────────────────────────────────
# SPACES Computation
# ─────────────────────────────────────────────────────────────────────


def compute_spaces(
    prices: np.ndarray,
    config: MachineStateConfig,
    *,
    max_gap: int = -1,
) -> SPACESResult:
    """Compute optimal switching costs via the SPACES algorithm.

    Builds the interval-state graph and finds shortest paths from each
    (i, proc) node to all future (j, proc) nodes.

    The interval-state graph is a layered DAG: vertex (i, s) represents
    "at the beginning of interval i, machine is in state s".  An edge from
    (i, s) to (i + T(s, s'), s') has weight Σ_{j=i}^{i+T(s,s')-1} c_j · P(s, s').

    Since the graph is a DAG (time only increases), we compute shortest
    paths by forward relaxation in topological (time) order — O(h·|S|²)
    per source.

    Args:
        prices: Per-interval energy costs, shape (h,).
        config: Machine state configuration.
        max_gap: If > 0, only compute c_star[i,j] for j-i <= max_gap.
            For larger gaps, the caller should use c_end[i] + c_start[j].
            Set to -1 (default) for full computation.

    Returns:
        SPACESResult with precomputed switching costs.
    """
    prices = np.asarray(prices, dtype=np.float64)
    h = len(prices)

    if config.is_trivial():
        # Stateless machine: no switching costs, idle is free
        return SPACESResult(
            c_star=np.zeros((h + 1, h + 1), dtype=np.float64),
            c_start=np.zeros(h + 1, dtype=np.float64),
            c_end=np.zeros(h + 1, dtype=np.float64),
            P_proc=config.P_proc if config.P_proc > 0 else 1.0,
            early=0,
            late=h,
            h=h,
            max_gap=-1,
        )

    n_s = config.n_states
    state_idx = {s: i for i, s in enumerate(config.states)}
    off_idx = state_idx[config.off_state]
    proc_idx = state_idx[config.proc_state]
    P_proc = config.P_proc

    # Precompute edge costs for each transition type: (s, s') -> (duration, power)
    # Edge from (i, s) to (i + dur, s') has weight = Σ prices[i:i+dur] * power
    edges: List[Tuple[int, int, int, float]] = []  # (s_from, s_to, duration, power)
    zero_closure = np.full((n_s, n_s), _INF, dtype=np.float64)
    for s_idx in range(n_s):
        zero_closure[s_idx, s_idx] = 0.0
    for s in config.states:
        for s_prime in config.states:
            dur = config.get_transition_time(s, s_prime)
            power = config.get_transition_power(s, s_prime)
            if dur is not None and power is not None:
                s_from = state_idx[s]
                s_to = state_idx[s_prime]
                edges.append((s_from, s_to, dur, power))
                if dur == 0:
                    zero_closure[s_from, s_to] = min(
                        zero_closure[s_from, s_to],
                        0.0,
                    )

    # Same-time shortest-path closure induced by zero-duration transitions.
    # This is required for the Figure 2 benchmark where proc<->idle has T=0.
    for k in range(n_s):
        for i in range(n_s):
            if zero_closure[i, k] >= _INF:
                continue
            for j in range(n_s):
                via = zero_closure[i, k] + zero_closure[k, j]
                if via < zero_closure[i, j]:
                    zero_closure[i, j] = via

    # Precompute prefix sums of prices for fast interval-cost queries
    prefix = np.zeros(h + 1, dtype=np.float64)
    prefix[1:] = np.cumsum(prices)

    def _edge_cost(start_interval: int, duration: int, power: float) -> float:
        """Cost of a transition: Σ prices[start:start+dur] * power."""
        end = start_interval + duration
        if end > h:
            return _INF
        return (prefix[end] - prefix[start_interval]) * power

    def _close_same_time(dist_vec: np.ndarray) -> None:
        """Apply zero-duration transitive closure to a state-distance vector."""
        out = dist_vec.copy()
        for s_to in range(n_s):
            best = out[s_to]
            for s_from in range(n_s):
                if dist_vec[s_from] >= _INF or zero_closure[s_from, s_to] >= _INF:
                    continue
                cand = dist_vec[s_from] + zero_closure[s_from, s_to]
                if cand < best:
                    best = cand
            out[s_to] = best
        dist_vec[:] = out

    # ── Compute early and late ────────────────────────────────────
    # early = earliest interval where proc state is reachable from off@0
    # We need to traverse from off at interval 0 forward.
    T_off_proc = config.get_transition_time(config.off_state, config.proc_state)
    T_proc_off = config.get_transition_time(config.proc_state, config.off_state)

    if T_off_proc is None or T_proc_off is None:
        raise ValueError(
            "Machine must have a path from off→proc and proc→off. "
            f"T(off, proc)={T_off_proc}, T(proc, off)={T_proc_off}"
        )

    # Simplistic early/late: direct transition only.
    # A more thorough approach would BFS through intermediate states,
    # but for typical machines (off→proc path is direct or via 1 hop),
    # this is correct.  We compute the minimum number of intervals to
    # reach proc from off@0.
    early = _compute_earliest_proc(config, state_idx, n_s, h)
    late = _compute_latest_proc(config, state_idx, n_s, h)

    if early > late:
        # Cannot even process a single slot
        inf_arr = np.full(h + 1, _INF, dtype=np.float64)
        return SPACESResult(
            c_star=np.full((h + 1, h + 1), _INF, dtype=np.float64),
            c_start=inf_arr.copy(),
            c_end=inf_arr.copy(),
            P_proc=P_proc,
            early=early,
            late=late,
            h=h,
            max_gap=max_gap,
        )

    # ── Determine effective max_gap ───────────────────────────────
    if max_gap <= 0:
        effective_max_gap = h  # Full computation
    else:
        effective_max_gap = max_gap

    # ── Build c_star via forward relaxation ───────────────────────
    # For each source time t_src (representing proc@t_src after a job ends),
    # compute shortest path to all (t, s) vertices with t > t_src.
    # We only need the cost to reach (t, proc) vertices.

    # Allocate c_star: banded storage
    if effective_max_gap < h:
        # Banded: store only c_star[i, i:i+max_gap+1]
        c_star = np.full((h + 1, effective_max_gap + 1), _INF, dtype=np.float64)
        banded = True
    else:
        c_star = np.full((h + 1, h + 1), _INF, dtype=np.float64)
        banded = False

    # Fill diagonal: c_star[i, i] = 0 (no gap)
    if banded:
        for i in range(h + 1):
            c_star[i, 0] = 0.0
    else:
        np.fill_diagonal(c_star, 0.0)

    # Run forward relaxation from each source
    for t_src in range(h + 1):
        t_max = min(t_src + effective_max_gap, h)
        if t_src >= t_max:
            continue

        # dist[t, s] = shortest distance from (t_src, proc) to (t, s)
        n_layers = t_max - t_src + 1
        dist = np.full((n_layers, n_s), _INF, dtype=np.float64)
        dist[0, proc_idx] = 0.0

        # Forward sweep through time layers
        for dt in range(n_layers):
            t = t_src + dt
            _close_same_time(dist[dt])
            if t >= h:
                break
            for s_from, s_to, dur, power in edges:
                if dur == 0:
                    continue
                if dist[dt, s_from] == _INF:
                    continue
                t_next = t + dur
                if t_next > t_max or t_next > h:
                    continue
                cost = dist[dt, s_from] + _edge_cost(t, dur, power)
                dt_next = t_next - t_src
                if cost < dist[dt_next, s_to]:
                    dist[dt_next, s_to] = cost

        # Extract proc-state costs into c_star
        for dt in range(1, n_layers):
            if dist[dt, proc_idx] < _INF:
                if banded:
                    c_star[t_src, dt] = dist[dt, proc_idx]
                else:
                    c_star[t_src, t_src + dt] = dist[dt, proc_idx]

    # ── Compute c_start: off during interval 0, then proc@t ──────
    c_start = np.full(h + 1, _INF, dtype=np.float64)
    dist_start = np.full((h + 1, n_s), _INF, dtype=np.float64)
    if h > 0:
        off_hold = config.get_transition_power(config.off_state, config.off_state)
        if off_hold is None:
            off_hold = 0.0
        dist_start[1, off_idx] = _edge_cost(0, 1, float(off_hold))

    for t in range(1, h):
        _close_same_time(dist_start[t])
        for s_from, s_to, dur, power in edges:
            if dur == 0:
                continue
            if dist_start[t, s_from] == _INF:
                continue
            t_next = t + dur
            if t_next > h:
                continue
            cost = dist_start[t, s_from] + _edge_cost(t, dur, power)
            if cost < dist_start[t_next, s_to]:
                dist_start[t_next, s_to] = cost

    for t in range(h + 1):
        c_start[t] = dist_start[t, proc_idx]

    # ── Compute c_end: proc@t → off during the last interval ─────
    # We do a *backward* relaxation from off@(h-1) and seed the
    # last off interval cost there so the machine is already off
    # when the last interval begins.
    # Reverse edges: from (t_next, s_to) back to (t_next - dur, s_from),
    # cost = Σ prices[t_next-dur : t_next] * power.
    c_end = np.full(h + 1, _INF, dtype=np.float64)

    # dist_end[t, s] = shortest distance from (t, s) to being off at the
    # start of the last interval.  This matches the paper statement that the
    # machine is off during the first and the last interval.
    # We process time layers in reverse.
    dist_end = np.full((h + 1, n_s), _INF, dtype=np.float64)
    if h > 0:
        off_hold = config.get_transition_power(config.off_state, config.off_state)
        if off_hold is None:
            off_hold = 0.0
        dist_end[h - 1, off_idx] = _edge_cost(h - 1, 1, float(off_hold))

    # Backward sweep
    for t in range(h - 1, 0, -1):
        _close_same_time(dist_end[t])
        for s_from, s_to, dur, power in edges:
            if dur == 0:
                continue
            # Edge goes from (t - dur, s_from) to (t_from + dur, s_to) = (t', s_to)
            # In reverse: if we know dist_end[t', s_to], we can update
            # dist_end[t' - dur, s_from].
            # But we need to be careful: the edge costs depend on the
            # *source* time of the forward edge.
            t_from = t - dur
            if t_from < 0:
                continue
            # Forward edge: (t_from, s_from) → (t, s_to), cost = Σ prices[t_from:t] * power
            if dist_end[t, s_to] == _INF:
                continue
            cost = dist_end[t, s_to] + _edge_cost(t_from, dur, power)
            if cost < dist_end[t_from, s_from]:
                dist_end[t_from, s_from] = cost

    for t in range(h + 1):
        c_end[t] = dist_end[t, proc_idx]

    # Convert banded c_star to full if needed for SPACESResult
    if banded:
        c_star_result = c_star  # Keep banded
    else:
        c_star_result = c_star

    return SPACESResult(
        c_star=c_star_result,
        c_start=c_start,
        c_end=c_end,
        P_proc=P_proc,
        early=early,
        late=late,
        h=h,
        max_gap=effective_max_gap if banded else -1,
    )


# ─────────────────────────────────────────────────────────────────────
# Helper: compute earliest / latest proc intervals
# ─────────────────────────────────────────────────────────────────────


def _compute_earliest_proc(
    config: MachineStateConfig,
    state_idx: Dict[str, int],
    n_s: int,
    h: int,
) -> int:
    """Compute earliest interval where proc state is reachable from off@0.

    Uses BFS on the transition graph to find the minimum number of intervals
    to reach proc from off.
    """
    off_idx = state_idx[config.off_state]
    proc_idx = state_idx[config.proc_state]

    # BFS: dist[s] = minimum intervals to reach state s from off
    dist = [_INF] * n_s
    dist[off_idx] = 0
    changed = True

    while changed:
        changed = False
        for s in config.states:
            s_i = state_idx[s]
            if dist[s_i] == _INF:
                continue
            for s_prime in config.states:
                s_p_i = state_idx[s_prime]
                dur = config.get_transition_time(s, s_prime)
                if dur is None:
                    continue
                if s == s_prime:
                    continue  # Self-loop doesn't advance
                new_dist = dist[s_i] + dur
                if new_dist < dist[s_p_i]:
                    dist[s_p_i] = new_dist
                    changed = True

    earliest = int(dist[proc_idx]) if dist[proc_idx] < _INF else h + 1
    # Interval 0 must be spent in the off state, so startup can only begin
    # afterwards. A duration-d path to proc therefore yields the first
    # feasible processing interval 1 + d.
    return min(1 + earliest, h)


def _compute_latest_proc(
    config: MachineStateConfig,
    state_idx: Dict[str, int],
    n_s: int,
    h: int,
) -> int:
    """Compute latest interval where proc can end and still reach off@h.

    The machine must reach off by interval h. We find the minimum number
    of intervals needed to go from proc to off.
    """
    off_idx = state_idx[config.off_state]
    proc_idx = state_idx[config.proc_state]

    # BFS: dist[s] = minimum intervals to reach off from state s
    # Run BFS backward from off.
    dist = [_INF] * n_s
    dist[off_idx] = 0
    changed = True

    while changed:
        changed = False
        for s in config.states:
            s_i = state_idx[s]
            for s_prime in config.states:
                s_p_i = state_idx[s_prime]
                if dist[s_p_i] == _INF:
                    continue
                if s == s_prime:
                    continue
                dur = config.get_transition_time(s, s_prime)
                if dur is None:
                    continue
                new_dist = dur + dist[s_p_i]
                if new_dist < dist[s_i]:
                    dist[s_i] = new_dist
                    changed = True

    shutdown_time = int(dist[proc_idx]) if dist[proc_idx] < _INF else h + 1
    # The last interval itself must be off, so processing must finish one
    # interval earlier than the off@start(last-interval) reachability bound.
    return max(0, (h - 2) - shutdown_time)


# ─────────────────────────────────────────────────────────────────────
# Utility: build processing-cost prefix array
# ─────────────────────────────────────────────────────────────────────


def build_proc_prefix(prices: np.ndarray, P_proc: float) -> np.ndarray:
    """Build prefix sums of prices * P_proc for fast processing-cost queries.

    prefix_proc[t+L] - prefix_proc[t] = Σ_{i=t}^{t+L-1} prices[i] * P_proc

    Args:
        prices: Per-interval energy costs, shape (h,).
        P_proc: Processing power rate P(proc, proc).

    Returns:
        Prefix sum array of shape (h+1,).
    """
    prices = np.asarray(prices, dtype=np.float64)
    prefix = np.zeros(len(prices) + 1, dtype=np.float64)
    if len(prices) > 0:
        prefix[1:] = np.cumsum(prices * P_proc)
    return prefix

"""G-LNS configuration — all hyperparameters in one place.

Uses Pydantic BaseModel for validation and serialization.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------


class LLMConfig(BaseModel):
    """Settings for the Groq / Kimi-K2 backend."""

    # Pinned model ID for reproducibility (Groq naming).
    model: str = "moonshotai/Kimi-K2-Instruct-0905"
    temperature: float = 0.8
    # Enough headroom for search-context reasoning + full operator code.
    max_tokens: int = 4_500
    # Built-in Groq SDK retry (covers 429/5xx). No extra tenacity layer.
    max_retries: int = 3
    # Client-side throttle: minimum seconds between consecutive API calls.
    min_call_interval_sec: float = 1.2
    # Timeout per request (seconds).
    timeout_sec: float = 120.0

    # Prompt budget controls (best-effort token burn reduction).
    n_reference_ops: int = Field(
        2, description="Number of reference operators to include (top performers)"
    )
    prompt_max_code_lines: int = Field(
        120, description="Max lines of operator code to include in prompt blocks"
    )
    prompt_max_code_chars: int = Field(
        3_500, description="Max characters of operator code to include in prompt blocks"
    )

    include_search_context: bool = Field(
        True,
        description=(
            "If true, include a concise summary of search state (stagnation/front/SA/inf) "
            "in evolution prompts to help the LLM diversify."
        ),
    )
    search_context_max_chars: int = Field(
        900, description="Max characters of the search context block"
    )


# ---------------------------------------------------------------------------
# Population & evolution
# ---------------------------------------------------------------------------


class PopulationConfig(BaseModel):
    """Operator pool sizes and pruning."""

    N: int = Field(5, description="Capacity per pool (destroy / repair)")
    M: int = Field(2, description="Number of operators pruned per pool each generation")


class EvolutionConfig(BaseModel):
    """Evolutionary loop settings."""

    G_max: int = Field(200, description="Maximum generations")
    # Strategy sampling weights (m1, m2, c1, c2).  Uniform by default.
    strategy_weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    max_retries_per_operator: int = 3


# ---------------------------------------------------------------------------
# Evaluation (inner ALNS loop)
# ---------------------------------------------------------------------------


class EvalConfig(BaseModel):
    """Multi-episode LNS evaluation settings."""

    K_episodes: int = Field(20, description="Independent eval episodes per generation")
    T_iters: int = Field(100, description="LNS iterations per episode")
    T_test: int = Field(500, description="LNS iterations for final testing phase")

    # Simulated Annealing
    sa_T0: float = Field(
        0.5, description="Initial SA temperature (scaled for [0,1] scalarised deltas)"
    )
    sa_alpha: float = Field(
        0.98, description="SA cooling rate (per iteration within each episode)"
    )
    sa_T0_gen_decay: float = Field(
        0.97,
        description=(
            "Per-generation multiplicative decay on effective T0.  "
            "Overridden by the adaptive mechanism when SA acceptance is too high/low."
        ),
    )
    sa_adaptive: bool = Field(
        True,
        description=(
            "Enable adaptive SA T0 adjustment: halve T0 when accept > 70%%, "
            "restore slowly when accept < 15%%."
        ),
    )

    # Destruction
    destroy_ratio: float = Field(0.2, description="Fraction of jobs to remove")

    # Adaptive weight smoothing (Eq. 6 in G-LNS paper)
    lambda_smooth: float = Field(0.5, description="Weight update smoothing factor")

    # Scoring vector σ = (σ1, σ2, σ3, σ4)
    sigma: Tuple[float, float, float, float] = (1.5, 1.2, 0.8, 0.1)

    # Periodic benchmark evaluation (every N generations; 0 = disabled).
    benchmark_eval_every: int = Field(
        10,
        description=(
            "Run a dual-benchmark snapshot evaluation every N generations. "
            "0 disables periodic benchmark probes (still runs at end)."
        ),
    )
    benchmark_eval_episodes: int = Field(
        5,
        description="Number of ALNS episodes per benchmark probe (lighter than normal eval)",
    )
    benchmark_eval_iters: int = Field(
        50,
        description="ALNS iterations per episode during benchmark probes",
    )


# ---------------------------------------------------------------------------
# Pareto archive
# ---------------------------------------------------------------------------


class ArchiveConfig(BaseModel):
    """Bi-objective Pareto archive settings."""

    max_size: int = Field(100, description="Maximum archive capacity")


# ---------------------------------------------------------------------------
# Sandbox (operator execution safety)
# ---------------------------------------------------------------------------


class SandboxConfig(BaseModel):
    """Safety limits for LLM-generated operator execution."""

    start_method: str = Field(
        "auto",
        description=(
            "Multiprocessing start method for sandbox workers. "
            "'auto' picks 'fork' on macOS and 'spawn' on Linux/Windows. "
            "Use 'spawn' for maximum isolation; use 'fork' for speed on macOS."
        ),
    )

    timeout_sec: float = Field(
        5.0, description="Hard wall-clock timeout per operator call"
    )
    max_memory_mb: int = Field(512, description="RSS memory limit for worker process")


# ---------------------------------------------------------------------------
# Instance selection
# ---------------------------------------------------------------------------


class InstanceConfig(BaseModel):
    """Which benchmark instances to use."""

    instances_json: Path = Field(
        Path("New Benchmark/instances_90.json"),
        description="Path to the normal-config 90-instance JSON (train & internal test)",
    )
    benchmark_data_dir: Optional[Path] = Field(
        Path("Benchmark/Data"),
        description=(
            "Path to the original Wang2018 benchmark Data folder "
            "(Data_c*.txt / Data_e*.txt / Data_p*.txt). "
            "None = skip external benchmark testing."
        ),
    )
    # Evolution uses small + medium instances for cheap fitness evaluation.
    evolution_ids: Optional[List[int]] = Field(
        None,
        description=(
            "Instance IDs to use during evolution. "
            "None = auto-select small + all medium."
        ),
    )
    # Testing uses the rest.
    test_ids: Optional[List[int]] = Field(
        None,
        description="Instance IDs for final testing. None = complement of evolution_ids.",
    )

    # ---- Benchmark adaptation mode ----------------------------------------
    # When True, the Wang2018 external benchmark is used DIRECTLY for evolution
    # (instead of instances_90.json).  The benchmark is split stratified-by-scale
    # into an evolution set and a held-out evaluation set.
    benchmark_adaptation: bool = Field(
        False,
        description=(
            "If True, evolve operators directly on the external Wang2018 benchmark "
            "(test-time adaptation). The benchmark is split stratified-by-scale into "
            "evolution and evaluation subsets."
        ),
    )
    benchmark_adaptation_evo_frac: float = Field(
        2 / 3,
        description=(
            "Fraction of each scale group (small/mls/large) assigned to the evolution "
            "set in benchmark_adaptation mode. Default 2/3 → 20 evo + 10 eval per scale."
        ),
    )
    benchmark_adaptation_split_seed: int = Field(
        42,
        description="Random seed for the stratified train/eval split in benchmark_adaptation mode.",
    )


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class GLNSConfig(BaseModel):
    """Master configuration for G-LNS."""

    seed: int = 42
    output_dir: Path = Path("results/glns")

    llm: LLMConfig = LLMConfig()
    population: PopulationConfig = PopulationConfig()
    evolution: EvolutionConfig = EvolutionConfig()
    eval: EvalConfig = EvalConfig()
    archive: ArchiveConfig = ArchiveConfig()
    sandbox: SandboxConfig = SandboxConfig()
    instances: InstanceConfig = InstanceConfig()

    class Config:
        # Allow Path objects to be serialized nicely.
        json_encoders = {Path: str}

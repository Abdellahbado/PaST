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
    # Keep this conservative to reduce TPD burn; raise via config if needed.
    max_tokens: int = 3_000
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

    K_episodes: int = Field(10, description="Independent eval episodes per generation")
    T_iters: int = Field(100, description="LNS iterations per episode")
    T_test: int = Field(500, description="LNS iterations for final testing phase")

    # Simulated Annealing
    sa_T0: float = Field(10.0, description="Initial SA temperature")
    sa_alpha: float = Field(0.95, description="SA cooling rate")

    # Destruction
    destroy_ratio: float = Field(0.2, description="Fraction of jobs to remove")

    # Adaptive weight smoothing (Eq. 6 in G-LNS paper)
    lambda_smooth: float = Field(0.5, description="Weight update smoothing factor")

    # Scoring vector σ = (σ1, σ2, σ3, σ4)
    sigma: Tuple[float, float, float, float] = (1.5, 1.2, 0.8, 0.1)


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
        description="Path to the legacy 90-instance JSON",
    )
    # Evolution uses small + some medium instances for cheap fitness evaluation.
    evolution_ids: Optional[List[int]] = Field(
        None,
        description=(
            "Instance IDs to use during evolution. "
            "None = auto-select small + first 10 medium."
        ),
    )
    # Testing uses the rest (medium-large, VLS).
    test_ids: Optional[List[int]] = Field(
        None,
        description="Instance IDs for final testing. None = complement of evolution_ids.",
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

"""
PaST-SM: Period-aware Scheduler Transformer - Single Machine

This package implements an RL-based single-machine scheduler for the
parallel machine scheduling problem with energy consideration under
time-of-use (TOU) electricity pricing.

The agent decides:
1. The ORDER of jobs to schedule
2. WHEN to insert idle time (slack) before running a chosen job

The agent does NOT assign jobs to machines (done by meta-heuristic outside).

Optimization mode: Energy minimization with hard deadline constraint (epsilon-constraint).

Supports 6 ablation variants:
- ppo_short_base: PPO + Short slack + Local periods (K=48) + Base model
- ppo_short_large: PPO + Short slack + Local periods (K=150) + Large model
- ppo_c2f: PPO + Coarse-to-fine slack + Local periods (K=48) + Base model
- ppo_full_tokens: PPO + Full slack + Full periods (K=250) + Base model (no pooling)
- ppo_full_global: PPO + Full slack + Full periods (K=250) + Base model + Global pooling
- reinforce_short_sc: REINFORCE + Short slack + Local periods (K=48) + Base model + Self-critic
"""

# ---------------------------------------------------------------------------
# Global warning suppression
#
# Kaggle's environment can emit noisy Gym deprecation banners during import.
# Because `python -m PaST.train_q_sequence` imports the `PaST` package first,
# suppression must happen here (before importing env modules).
# ---------------------------------------------------------------------------

import os
import sys
import io
import warnings

os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")
warnings.filterwarnings("ignore")


class _FilteredTextIO(io.TextIOBase):
    def __init__(self, base: io.TextIOBase, *, drop_substrings: list[str]):
        self._base = base
        self._drop = drop_substrings
        self._buf = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buf += s
        out: list[str] = []
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line_nl = line + "\n"
            if line.startswith("Traceback") or line.startswith("Exception"):
                out.append(line_nl)
                continue
            if any(sub in line for sub in self._drop):
                continue
            if (
                "Warning:" in line
                or "DeprecationWarning" in line
                or "UserWarning" in line
            ):
                continue
            out.append(line_nl)
        if out:
            return self._base.write("".join(out))
        return len(s)

    def flush(self) -> None:
        try:
            self._base.flush()
        except Exception:
            pass


_DROP_SUBSTRINGS = [
    "Gym has been unmaintained since 2022",
    "Please upgrade to Gymnasium",
    "See the migration guide at https://gymnasium",
]

try:
    sys.stderr = _FilteredTextIO(sys.stderr, drop_substrings=_DROP_SUBSTRINGS)
    sys.stdout = _FilteredTextIO(sys.stdout, drop_substrings=_DROP_SUBSTRINGS)
except Exception:
    pass

__version__ = "2.0.0"

from .config import (
    SlackType,
    RLAlgorithm,
    ModelSize,
    VariantID,
    ShortSlackSpec,
    CoarseToFineSlackSpec,
    FullSlackSpec,
    ModelConfig,
    EnvConfig,
    DataConfig,
    TrainingConfig,
    VariantConfig,
    get_variant_config,
    list_variants,
    VARIANT_FACTORIES,
)

from .sm_benchmark_data import (
    generate_raw_instance,
    make_single_machine_episode,
    generate_episode_batch,
    generate_episode_batch_variable,
    generate_single_machine_episode,
    sample_intervals_sum_to_T,
    expand_ck_to_ct,
    compute_period_start_slots,
    RawInstance,
    SingleMachineEpisode,
)

from .sm_env import (
    SingleMachinePeriodEnv,
    GPUBatchSingleMachinePeriodEnv,
    slack_to_start_time,
    find_period_at_time,
    ENV_VERSION,
)

from .past_sm_model import (
    PaSTSMNet,
    build_model,
)

from .q_sequence_model import (
    DuelingQHead,
    QSequenceNet,
    QModelWrapper,
    build_q_model,
)

# NOTE: We intentionally do NOT import `PaST.train_ppo` here.
# Running `python -m PaST.train_ppo` first imports the `PaST` package; if we
# import `train_ppo` inside __init__, runpy detects it in sys.modules and emits
# a RuntimeWarning. Import training entrypoints from `PaST.train_ppo` directly.

__all__ = [
    "SlackType",
    "RLAlgorithm",
    "ModelSize",
    "VariantID",
    "ShortSlackSpec",
    "CoarseToFineSlackSpec",
    "FullSlackSpec",
    "ModelConfig",
    "EnvConfig",
    "DataConfig",
    "TrainingConfig",
    "VariantConfig",
    "get_variant_config",
    "list_variants",
    "VARIANT_FACTORIES",
    "generate_raw_instance",
    "make_single_machine_episode",
    "generate_episode_batch",
    "generate_episode_batch_variable",
    "generate_single_machine_episode",
    "sample_intervals_sum_to_T",
    "expand_ck_to_ct",
    "compute_period_start_slots",
    "RawInstance",
    "SingleMachineEpisode",
    "SingleMachinePeriodEnv",
    "GPUBatchSingleMachinePeriodEnv",
    "slack_to_start_time",
    "find_period_at_time",
    "ENV_VERSION",
    "PaSTSMNet",
    "build_model",
    "DuelingQHead",
    "QSequenceNet",
    "QModelWrapper",
    "build_q_model",
]

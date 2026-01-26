"""Backward-compatible wrapper for :pymod:`PaST.train.ppo_runner`.

The implementation lives in `PaST/train/ppo_runner.py`. This wrapper preserves
existing imports and CLI usage like `python -m PaST.ppo_runner`.
"""

from PaST.train.ppo_runner import *  # noqa: F401,F403


if __name__ == "__main__":
    import runpy

    runpy.run_module("PaST.train.ppo_runner", run_name="__main__")

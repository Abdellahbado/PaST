"""Backward-compatible wrapper for :pymod:`PaST.algorithms.sgbs`.

The implementation lives in `PaST/algorithms/sgbs.py`. This wrapper preserves
existing imports and CLI usage like `python -m PaST.sgbs`.
"""

from PaST.algorithms.sgbs import *  # noqa: F401,F403


if __name__ == "__main__":
    import runpy

    runpy.run_module("PaST.algorithms.sgbs", run_name="__main__")

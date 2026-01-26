"""Backward-compatible wrapper for :pymod:`PaST.models.q_sequence_model`.

The implementation lives in `PaST/models/q_sequence_model.py`. This wrapper
preserves existing imports like `from PaST.q_sequence_model import build_q_model`.
"""

from PaST.models.q_sequence_model import *  # noqa: F401,F403


if __name__ == "__main__":
    import runpy

    runpy.run_module("PaST.models.q_sequence_model", run_name="__main__")

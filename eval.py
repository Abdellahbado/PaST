"""Backward-compatible wrapper for :pymod:`PaST.evaluation.eval`."""

from PaST.evaluation.eval import *  # noqa: F401,F403


if __name__ == "__main__":
    import runpy

    runpy.run_module("PaST.evaluation.eval", run_name="__main__")

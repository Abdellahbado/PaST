"""Backward-compatible wrapper for :pymod:`PaST.evaluation.eval_checkpoint`."""

from PaST.evaluation.eval_checkpoint import *  # noqa: F401,F403


if __name__ == "__main__":
    import runpy

    runpy.run_module("PaST.evaluation.eval_checkpoint", run_name="__main__")

"""Backward-compatible wrapper for :pymod:`PaST.train.reinforce_runner`."""

from PaST.train.reinforce_runner import *  # noqa: F401,F403


if __name__ == "__main__":
    import runpy

    runpy.run_module("PaST.train.reinforce_runner", run_name="__main__")

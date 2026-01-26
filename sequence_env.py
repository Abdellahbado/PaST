"""Backward-compatible wrapper for :pymod:`PaST.env.sequence_env`."""

from PaST.env.sequence_env import *  # noqa: F401,F403


if __name__ == "__main__":
    import runpy

    runpy.run_module("PaST.env.sequence_env", run_name="__main__")

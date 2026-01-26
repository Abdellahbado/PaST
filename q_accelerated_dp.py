"""Backward-compatible wrapper for :pymod:`PaST.solvers.q_accelerated_dp`."""

from PaST.solvers.q_accelerated_dp import *  # noqa: F401,F403


if __name__ == "__main__":
    import runpy

    runpy.run_module("PaST.solvers.q_accelerated_dp", run_name="__main__")

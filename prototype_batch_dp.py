"""Backward-compatible wrapper for :pymod:`PaST.solvers.prototype_batch_dp`."""

from PaST.solvers.prototype_batch_dp import *  # noqa: F401,F403


if __name__ == "__main__":
    import runpy

    runpy.run_module("PaST.solvers.prototype_batch_dp", run_name="__main__")

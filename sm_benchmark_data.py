"""Backward-compatible wrapper for :pymod:`PaST.data.sm_benchmark_data`."""

from PaST.data.sm_benchmark_data import *  # noqa: F401,F403


if __name__ == "__main__":
    import runpy

    runpy.run_module("PaST.data.sm_benchmark_data", run_name="__main__")

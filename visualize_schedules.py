"""Backward-compatible wrapper for :pymod:`PaST.viz.visualize_schedules`."""

from PaST.viz.visualize_schedules import *  # noqa: F401,F403


if __name__ == "__main__":
    import runpy

    runpy.run_module("PaST.viz.visualize_schedules", run_name="__main__")

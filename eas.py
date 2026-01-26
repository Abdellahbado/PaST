"""Backward-compatible wrapper for :pymod:`PaST.algorithms.eas`."""

from PaST.algorithms.eas import *  # noqa: F401,F403


if __name__ == "__main__":
    import runpy

    runpy.run_module("PaST.algorithms.eas", run_name="__main__")

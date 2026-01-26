"""Backward-compatible wrapper for :pymod:`PaST.algorithms.sgbs_eas`."""

from PaST.algorithms.sgbs_eas import *  # noqa: F401,F403


if __name__ == "__main__":
    import runpy

    runpy.run_module("PaST.algorithms.sgbs_eas", run_name="__main__")

"""Backward-compatible wrapper for the SGBS implementation.

The codebase historically moved the implementation between module paths.
This wrapper keeps external scripts working by importing from whichever
location exists in the current checkout.
"""

try:
    # Older layout
    from PaST.algorithms.sgbs import *  # type: ignore  # noqa: F401,F403
except ModuleNotFoundError:
    # Current layout
    from PaST.artifacts.algorithms.sgbs import *  # type: ignore  # noqa: F401,F403


if __name__ == "__main__":
    import runpy

    try:
        runpy.run_module("PaST.algorithms.sgbs", run_name="__main__")
    except ModuleNotFoundError:
        runpy.run_module("PaST.artifacts.algorithms.sgbs", run_name="__main__")

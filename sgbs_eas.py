"""Backward-compatible wrapper for the SGBS+EAS implementation.

The implementation module path moved in some checkouts.
Import from whichever location exists.
"""

try:
    from PaST.algorithms.sgbs_eas import *  # type: ignore  # noqa: F401,F403
except ModuleNotFoundError:
    from PaST.artifacts.algorithms.sgbs_eas import *  # type: ignore  # noqa: F401,F403


if __name__ == "__main__":
    import runpy

    try:
        runpy.run_module("PaST.algorithms.sgbs_eas", run_name="__main__")
    except ModuleNotFoundError:
        runpy.run_module("PaST.artifacts.algorithms.sgbs_eas", run_name="__main__")

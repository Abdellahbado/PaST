"""Sandboxed execution of LLM-generated operator code.

Runs each operator call in a **separate process** with hard wall-clock timeout
and memory limits, so a single infinite loop or memory bomb cannot stall the
entire evaluation phase.

On macOS the memory limit uses ``resource.RLIMIT_AS`` (best-effort; not as
strict as Linux cgroups, but sufficient for catching runaway allocations).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import resource
import sys
import traceback
from typing import Any, Callable, Dict, Optional, Tuple

from glns.config import SandboxConfig

logger = logging.getLogger(__name__)


def _resolve_start_method(cfg: SandboxConfig) -> str:
    method = str(getattr(cfg, "start_method", "spawn")).strip().lower()
    if method == "auto":
        # macOS: fork is dramatically faster than spawn for short-lived workers.
        # Linux/HPC: prefer spawn to avoid fork-related virtual-memory accounting
        # surprises in cgroup/ulimit environments.
        # Windows: only spawn is supported.
        if sys.platform == "darwin":
            return "fork"
        return "spawn"
    if method in {"spawn", "fork", "forkserver"}:
        return method
    logger.warning("Unknown sandbox start_method=%r; falling back to 'spawn'", method)
    return "spawn"


# ---------------------------------------------------------------------------
# Worker executed in a child process
# ---------------------------------------------------------------------------


def _worker(
    fn_code: str,
    fn_name: str,
    args: tuple,
    mem_limit_bytes: int,
    apply_mem_limit: bool,
    result_queue: mp.Queue,
) -> None:
    """Execute *fn_name* (compiled from *fn_code*) with *args* in isolation."""
    try:
        # Memory limit: RLIMIT_AS caps *virtual memory*.
        # This is generally OK with 'spawn'. With 'fork' on Linux containers,
        # RLIMIT_AS can trip due to inherited address space even with COW.
        if apply_mem_limit and mem_limit_bytes > 0:
            try:
                soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                resource.setrlimit(resource.RLIMIT_AS, (mem_limit_bytes, hard))
            except (ValueError, OSError):
                pass  # not fatal — proceed without memory cap

        # Compile & extract the function.
        ns: Dict[str, Any] = {"__builtins__": __builtins__}
        exec(compile(fn_code, "<operator>", "exec"), ns)  # noqa: S102
        fn: Callable = ns[fn_name]
        result = fn(*args)
        result_queue.put(("ok", result))
    except Exception:
        result_queue.put(("error", traceback.format_exc()))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_operator_sandboxed(
    fn_code: str,
    fn_name: str,
    args: tuple,
    cfg: SandboxConfig,
) -> Tuple[bool, Any]:
    """Run a compiled operator in a sandboxed child process.

    Returns:
        (success: bool, result_or_error: Any)
        On success: (True, <return value of the operator>)
        On failure: (False, <error message string>)
    """
    mem_bytes = cfg.max_memory_mb * 1024 * 1024

    start_method = _resolve_start_method(cfg)
    try:
        ctx = mp.get_context(start_method)
    except ValueError:
        logger.warning(
            "Sandbox start_method=%r not available; falling back to 'spawn'",
            start_method,
        )
        ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue(maxsize=1)

    # See comment in _worker: RLIMIT_AS is safest with spawn.
    apply_mem_limit = (start_method == "spawn") or (sys.platform == "darwin")
    proc = ctx.Process(
        target=_worker,
        args=(fn_code, fn_name, args, mem_bytes, apply_mem_limit, q),
        daemon=True,
    )
    proc.start()
    proc.join(timeout=cfg.timeout_sec)

    if proc.is_alive():
        proc.kill()
        proc.join(timeout=2)
        return False, f"Operator timed out after {cfg.timeout_sec}s"

    if proc.exitcode != 0 and q.empty():
        return False, f"Operator process crashed (exit code {proc.exitcode})"

    if q.empty():
        return False, "Operator produced no result (possible segfault or OOM kill)"

    status, payload = q.get_nowait()
    if status == "ok":
        return True, payload
    else:
        return False, f"Operator raised an exception:\n{payload}"


def compile_operator(code: str, fn_name: str) -> Callable:
    """Compile operator code and return the callable (in-process, no sandbox).

    Used for seed operators and quick local checks where sandboxing overhead is
    unnecessary.
    """
    ns: Dict[str, Any] = {"__builtins__": __builtins__}
    exec(compile(code, "<operator>", "exec"), ns)  # noqa: S102
    if fn_name not in ns:
        raise KeyError(
            f"Compiled code does not define a function named '{fn_name}'. "
            f"Available names: {[k for k in ns if not k.startswith('_')]}"
        )
    return ns[fn_name]
